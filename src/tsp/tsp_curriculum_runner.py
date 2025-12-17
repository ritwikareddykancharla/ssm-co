# --------------------------------------------------
# Curriculum TSP Trainer (DDP, STABLE, AMP-SAFE)
# - SINGLE fixed-size model (same for all N)
# - Curriculum over N (10 → 20 → 50 → 100)
# - POMO-lite baseline (same logic for ALL N, no hardcoding)
# - CORRECT REINFORCE (ADVANTAGE * LOGP)
# - Advantage normalization
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
# Train ONE curriculum stage (POMO-lite)
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
    pomo_k=4,
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
            "avg_sampled_tour",
            "avg_pomo_best",
            "entropy_coef",
        ])

    for step in range(steps):
        coords = torch.rand(batch, n_nodes, 2, device=device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            # -------- SAMPLE POLICY --------
            logp, ent, sampled_len = net.rollout(coords, greedy=False)

            # -------- POMO-LITE BASELINE (NO GRAD, SAME FOR ALL N) --------
            with torch.no_grad():
                best_len = None
                for _ in range(pomo_k):
                    _, _, l = net.rollout(coords, greedy=False)
                    best_len = l if best_len is None else torch.minimum(best_len, l)

            advantage = sampled_len - best_len

            # -------- ADVANTAGE NORMALIZATION --------
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            advantage = advantage.detach()

            # -------- ENTROPY --------
            entropy_coef = entropy_schedule(step, steps, entropy_start, entropy_end)
            entropy_coef = max(entropy_coef, entropy_floor)

            # -------- REINFORCE LOSS --------
            loss = (advantage * logp).mean() - entropy_coef * ent.mean()

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
                    round(best_len.mean().item(), 6),
                    round(entropy_coef, 6),
                ])
                f.flush()

                print(
                    f"[TRAIN][N={n_nodes}] "
                    f"step {step:05d} | "
                    f"sampled {sampled_len.mean():.3f} | "
                    f"pomo-best {best_len.mean():.3f} | "
                    f"entropy {entropy_coef:.4f}"
                )

    if is_main:
        f.close()


# --------------------------------------------------
# Evaluation (GREEDY ONLY, rank 0)
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
    parser.add_argument("--pomo_k", type=int, default=4)

    args = parser.parse_args()

    # ---------------- DDP ----------------
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

    # ---------------- MODEL ----------------
    model = TSPModel(dim=args.dim, layers=args.layers).to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------------- CURRICULUM ----------------
    curriculum = [
        dict(N=10,  steps=1000,  ent=(0.04, 0.015)),
        dict(N=20,  steps=3000,  ent=(0.05, 0.02)),
        dict(N=50,  steps=5000,  ent=(0.06, 0.025)),
        dict(N=100, steps=10000, ent=(0.07, 0.03)),
    ]

    for stage in curriculum:
        N = stage["N"]
        steps = stage["steps"]
        ent_start, ent_end = stage["ent"]

        if is_main:
            print(f"\n Training TSP{N} for {steps} steps")

        train_log = f"logs/tsp_curriculum_N{N}_train.csv"

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
            entropy_floor=0.01,
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
                f"📊 Eval TSP{N} | "
                f"tour={avg_tour:.3f} | "
                f"time={avg_time:.1f} ms"
            )

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
