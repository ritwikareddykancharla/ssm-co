# --------------------------------------------------
# Curriculum TSP Trainer (DDP, STABLE, AMP-SAFE)
# TWO-PHASE TRAINING WITH GAP-WEIGHTED REINFORCE
# --------------------------------------------------
# Phase 1: Pure tour-cost REINFORCE (GREEDY baseline, FAST)
# Phase 2: Gap-weighted REINFORCE + late self-distillation (POMO-lite)
# --------------------------------------------------
# - SINGLE fixed-size model (same for all N)
# - Curriculum over N (10 → 20 → 50 → 100)
# - NO POMO in Phase 1 (speed + stability)
# - POMO-lite + distillation only in Phase 2
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
# Train ONE curriculum stage (2 phases)
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
    phase1_frac=0.6,        # first 60% = pure REINFORCE
    distill_start_frac=0.8, # distill only in last 20%
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
            "phase",
            "n_nodes",
            "loss",
            "avg_sampled",
            "baseline",
            "entropy",
        ])

    for step in range(steps):
        coords = torch.rand(batch, n_nodes, 2, device=device)
        phase1 = step < int(phase1_frac * steps)
        distill_on = step >= int(distill_start_frac * steps)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            # --------------------------------------------------
            # Sample policy
            # --------------------------------------------------
            logp, ent, sampled_len = net.rollout(coords, greedy=False)

            # ==================================================
            # PHASE 1: GREEDY BASELINE REINFORCE (FAST)
            # ==================================================
            if phase1:
                with torch.no_grad():
                    _, _, greedy_len = net.rollout(coords, greedy=True)

                advantage = sampled_len - greedy_len
                loss = (advantage.detach() * logp).mean()
                baseline_val = greedy_len.mean()
                phase_name = "P1"

            # ==================================================
            # PHASE 2: GAP-WEIGHTED + OPTIONAL DISTILLATION
            # ==================================================
            else:
                with torch.no_grad():
                    best_len = None
                    best_logp = None
                    for _ in range(pomo_k):
                        lp, _, l = net.rollout(coords, greedy=False)
                        if best_len is None:
                            best_len, best_logp = l, lp
                        else:
                            mask = l < best_len
                            best_len = torch.where(mask, l, best_len)
                            best_logp = torch.where(mask, lp, best_logp)

                gap = sampled_len - best_len
                loss = (gap.detach() * logp).mean()
                baseline_val = best_len.mean()
                phase_name = "P2"

                if distill_on:
                    loss = loss - best_logp.mean()
                    phase_name = "P2+DISTILL"

            # --------------------------------------------------
            # Entropy (anneal to zero)
            # --------------------------------------------------
            entropy_coef = entropy_schedule(step, steps, entropy_start, entropy_end)
            loss = loss - entropy_coef * ent.mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % log_every == 0:
            sync("cuda")
            if is_main:
                writer.writerow([
                    step,
                    phase_name,
                    n_nodes,
                    round(loss.item(), 6),
                    round(sampled_len.mean().item(), 6),
                    round(baseline_val.item(), 6),
                    round(entropy_coef, 6),
                ])
                f.flush()

                print(
                    f"[{phase_name}][N={n_nodes}] step {step:05d} | "
                    f"sampled {sampled_len.mean():.3f} | "
                    f"baseline {baseline_val:.3f} | "
                    f"entropy {entropy_coef:.4f}"
                )

    if is_main:
        f.close()


# --------------------------------------------------
# Evaluation (GREEDY)
# --------------------------------------------------
@torch.no_grad()
def evaluate(model, device, n_nodes, batch, eval_batches):
    model.eval()
    net = model.module if hasattr(model, "module") else model

    tours = []
    for _ in range(eval_batches):
        coords = torch.rand(batch, n_nodes, 2, device=device)
        _, _, length = net.rollout(coords, greedy=True)
        tours.append(length.mean().item())

    return sum(tours) / len(tours)


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
        print(f"\nDevice: {device}")
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
        dict(N=10,  steps=1500, ent=(0.05, 0.0)),
        dict(N=20,  steps=3000, ent=(0.05, 0.0)),
        dict(N=50,  steps=5000, ent=(0.06, 0.0)),
        dict(N=100, steps=8000, ent=(0.06, 0.0)),
    ]

    for stage in curriculum:
        N = stage["N"]
        steps = stage["steps"]
        ent_start, ent_end = stage["ent"]

        if is_main:
            print(f"\nTraining TSP{N}")

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
            log_path=f"logs/tsp_gap_weighted_N{N}.csv",
            is_main=is_main,
            pomo_k=args.pomo_k,
        )

        if is_main:
            avg = evaluate(model, device, N, args.batch, eval_batches=20)
            print(f"Eval TSP{N}: {avg:.3f}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
