# --------------------------------------------------
# TWO-PHASE CURRICULUM TSP TRAINER (DDP, AMP-SAFE)
# --------------------------------------------------
# PHASE 1: COST-ONLY REINFORCE (learn geometry)
# PHASE 2: SELF-DISTILLATION FROM POMO-BEST (learn search)
# --------------------------------------------------
# - Single fixed-size model (same for all N)
# - Curriculum over N (10 → 20 → 50 → 100)
# - No hardcoded alphas, clean separation of phases
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
# Entropy schedule
# --------------------------------------------------
def entropy_schedule(step, total_steps, start, end):
    frac = min(step / total_steps, 1.0)
    return start * (1 - frac) + end * frac


# --------------------------------------------------
# PHASE 1 — COST-ONLY REINFORCE
# --------------------------------------------------
def train_phase1(
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
):
    model.train()
    net = model.module if hasattr(model, "module") else model

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if is_main:
        f = open(log_path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["step", "n_nodes", "loss", "avg_tour", "entropy"])

    for step in range(steps):
        coords = torch.rand(batch, n_nodes, 2, device=device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logp, ent, length = net.rollout(coords, greedy=False)
            baseline = length.detach().mean()
            advantage = (length - baseline).detach()

            entropy_coef = entropy_schedule(step, steps, entropy_start, entropy_end)
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
                    round(length.mean().item(), 6),
                    round(entropy_coef, 6),
                ])
                f.flush()
                print(
                    f"[PHASE1][N={n_nodes}] step {step:05d} | tour {length.mean():.3f} | entropy {entropy_coef:.4f}"
                )

    if is_main:
        f.close()


# --------------------------------------------------
# PHASE 2 — SELF-DISTILLATION FROM POMO-BEST
# --------------------------------------------------
def train_phase2(
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
            "entropy",
        ])

    for step in range(steps):
        coords = torch.rand(batch, n_nodes, 2, device=device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            # ----- POMO sampling (teacher signal) -----
            with torch.no_grad():
                all_logp, all_len = [], []
                for _ in range(pomo_k):
                    lp, _, ln = net.rollout(coords, greedy=False)
                    all_logp.append(lp)
                    all_len.append(ln)

                all_logp = torch.stack(all_logp)  # [K,B]
                all_len = torch.stack(all_len)    # [K,B]
                best_idx = all_len.argmin(dim=0)
                pomo_best = all_len.min(dim=0).values

            B = batch
            best_logp = all_logp[best_idx, torch.arange(B, device=device)]
            distill_loss = -best_logp.mean()

            # ----- entropy -----
            _, ent, sampled_len = net.rollout(coords, greedy=False)
            entropy_coef = entropy_schedule(step, steps, entropy_start, entropy_end)

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
                    f"[PHASE2][N={n_nodes}] step {step:05d} | sampled {sampled_len.mean():.3f} | pomo {pomo_best.mean():.3f}"
                )

    if is_main:
        f.close()


# --------------------------------------------------
# Evaluation (greedy)
# --------------------------------------------------
@torch.no_grad()
def evaluate(model, device, n_nodes, batch, eval_batches):
    model.eval()
    net = model.module if hasattr(model, "module") else model

    vals = []
    for _ in range(eval_batches):
        coords = torch.rand(batch, n_nodes, 2, device=device)
        _, _, length = net.rollout(coords, greedy=True)
        vals.append(length.mean().item())
    return sum(vals) / len(vals)


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
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main = dist.get_rank() == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    if is_main:
        print(f"\nDevice: {device}")
        os.makedirs("logs", exist_ok=True)

    model = TSPModel(dim=args.dim, layers=args.layers).to(device)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    curriculum = [10, 20, 50, 100]

    for N in curriculum:
        if is_main:
            print(f"\n=== TSP{N}: PHASE 1 ===")
        train_phase1(
            model, opt, device, N, 1000, args.batch, 0.05, 0.02,
            args.log_every, f"logs/tsp{N}_phase1.csv", is_main
        )

        if is_main:
            print(f"Eval after phase1: {evaluate(model, device, N, args.batch, 20):.3f}")
            print(f"\n=== TSP{N}: PHASE 2 ===")

        train_phase2(
            model, opt, device, N, 2000, args.batch, 0.03, 0.01,
            args.log_every, f"logs/tsp{N}_phase2.csv", is_main, args.pomo_k
        )

        if is_main:
            print(f"Eval after phase2: {evaluate(model, device, N, args.batch, 20):.3f}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
