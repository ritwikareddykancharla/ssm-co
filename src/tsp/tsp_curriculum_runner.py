# --------------------------------------------------
# Curriculum TSP Trainer (DDP, STABLE, AMP-SAFE)
# WITH SELF-DISTILLATION (MODERN, PAPER-STYLE)
# --------------------------------------------------
# - SINGLE fixed-size model (same for all N)
# - Curriculum over N (10 → 20 → 50 → 100)
# - POMO-lite sampling (same logic for ALL N)
# - Self-distillation on BEST sampled trajectory
# - Optional REINFORCE (kept minimal, low weight)
# - Entropy regularization + floor (prevents collapse)
# - Greedy eval only (rank 0)
# --------------------------------------------------

import argparse
import csv
import os
import time
import torch
import torch.distributed as dist

from model import TSPModel
from utils.device import sync


# --------------------------------------------------
# Entropy schedule (linear)
# --------------------------------------------------
def entropy_schedule(step, total_steps, start, end):
    frac = min(step / total_steps, 1.0)
    return start * (1 - frac) + end * frac


# --------------------------------------------------
# Train ONE curriculum stage (SELF-DISTILLATION)
# --------------------------------------------------
def train_stage(
    model,
    opt,
    device,
    n_nodes,
    steps,
    batch,
    entropy_start,
    entropy_end,
    log_every,
    log_path,
    is_main,
    pomo_k=8,
    entropy_floor=0.01,
):
    model.train()
    net = model.module if hasattr(model, "module") else model

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if is_main:
        f = open(log_path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "n_nodes",
            "loss",
            "avg_sampled",
            "avg_pomo_best",
            "entropy_coef",
        ])

    for step in range(steps):
        coords = torch.rand(batch, n_nodes, 2, device=device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            # ==================================================
            # 1. MULTI-SAMPLE (POMO-lite, NO GRAD)
            # ==================================================
            with torch.no_grad():
                all_logp = []
                all_len = []
                for _ in range(pomo_k):
                    logp_k, _, len_k = net.rollout(coords, greedy=False)
                    all_logp.append(logp_k)
                    all_len.append(len_k)

                all_len = torch.stack(all_len)        # [K, B]
                all_logp = torch.stack(all_logp)      # [K, B]
                best_idx = all_len.argmin(dim=0)      # [B]
                pomo_best = all_len.min(dim=0).values # [B]

            # ==================================================
            # 2. SELF-DISTILLATION LOSS (MAIN SIGNAL)
            # ==================================================
            B = batch
            batch_idx = torch.arange(B, device=device)
            best_logp = all_logp[best_idx, batch_idx]
            distill_loss = -best_logp.mean()

            # ==================================================
            # 3. ENTROPY (KEEP EXPLORATION ALIVE)
            # ==================================================
            _, ent, sampled_len = net.rollout(coords, greedy=False)

            entropy_coef = entropy_schedule(step, steps, entropy_start, entropy_end)
            entropy_coef = max(entropy_coef, entropy_floor)

            # ==================================================
            # 4. FINAL LOSS (NO ALPHA HACKING)
            # ==================================================
            loss = distill_loss - entropy_coef * ent.mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % log_every == 0:
            sync("cuda")
            if is_main:
                writer.writerow([
                    step,
                    n_nodes,
                    round(loss.item(), 6),
                    round(sampled_len.mean().item(), 6),
                    round(pomo_best.mean().item(), 6),
                    round(entropy_coef, 6),
                ])
                f.flush()

                print(
                    f"[TRAIN][N={n_nodes}] "
                    f"step {step:05d} | "
                    f"sampled {sampled_len.mean():.3f} | "
                    f"pomo-best {pomo_best.mean():.3f} | "
                    f"entropy {entropy_coef:.4f}"
                )

    if is_main:
        f.close()


# --------------------------------------------------
# Evaluation (GREEDY ONLY)
# --------------------------------------------------
@torch.no_grad()
def evaluate(model, device, n_nodes, batch, eval_batches):
    model.eval()
    net = model.module if hasattr(model, "module") else model

    tours, times = [], []

    for _ in range(eval_batches):
        coords = torch.rand(batch, n_nodes, 2, device=device)
        start = time.perf_counter()
        _, _, length = net.rollout(coords, greedy=True)
        sync("cuda")
        end = time.perf_counter()
        tours.append(length.mean().item())
        times.append((end - start) * 1000)

    return sum(tours) / len(tours), sum(times) / len(times)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--pomo_k", type=int, default=8)
    args = parser.parse_args()

    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main = dist.get_rank() == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    if is_main:
        print(f"\n Device: {device}")
        os.makedirs("logs", exist_ok=True)

    model = TSPModel(dim=args.dim, layers=args.layers).to(device)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    curriculum = [
        dict(N=10,  steps=1000,  ent=(0.04, 0.02)),
        dict(N=20,  steps=3000,  ent=(0.05, 0.025)),
        dict(N=50,  steps=5000,  ent=(0.06, 0.03)),
        dict(N=100, steps=10000, ent=(0.07, 0.035)),
    ]

    for stage in curriculum:
        N = stage["N"]
        steps = stage["steps"]
        ent_start, ent_end = stage["ent"]

        if is_main:
            print(f"\n Training TSP{N} for {steps} steps")

        train_log = f"logs/tsp_self_distill_N{N}.csv"

        train_stage(
            model=model,
            opt=opt,
            device=device,
            n_nodes=N,
            steps=steps,
            batch=args.batch,
            entropy_start=ent_start,
            entropy_end=ent_end,
            log_every=args.log_every,
            log_path=train_log,
            is_main=is_main,
            pomo_k=args.pomo_k,
        )

        if is_main:
            avg_tour, avg_time = evaluate(
                model,
                device,
                n_nodes=N,
                batch=args.batch,
                eval_batches=20,
            )
            print(
                f"📊 Eval TSP{N} | tour={avg_tour:.3f} | time={avg_time:.1f} ms"
            )

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
