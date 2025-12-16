# --------------------------------------------------
# Curriculum TSP Trainer (PAPER-READY)
# - SINGLE fixed-size model
# - Curriculum over N (10 → 20 → 50 → 100)
# - EMA baseline (strong advantage signal)
# - Entropy regularization (training only)
# - Sampling-only evaluation
# - CLEAN CSV logs
# --------------------------------------------------

import argparse
import csv
import os
import time
import torch

from model import TSPModel
from utils.device import sync


# --------------------------------------------------
# Entropy schedule (per stage)
# --------------------------------------------------

def entropy_schedule(step, total_steps, start, end):
    frac = step / total_steps
    return start * (1 - frac) + end * frac


# --------------------------------------------------
# Train ONE stage
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
    ema_beta,
    log_every,
    log_path,
):
    model.train()
    net = model

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    ema_baseline = None

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "n_nodes",
            "loss",
            "avg_tour",
            "entropy_coef",
            "ema_baseline",
        ])

        for step in range(steps):
            coords = torch.rand(batch, n_nodes, 2, device=device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logp, ent, length = net.rollout(coords, greedy=False)

                # EMA baseline
                with torch.no_grad():
                    mean_len = length.mean().item()
                    ema_baseline = (
                        mean_len if ema_baseline is None
                        else ema_beta * ema_baseline + (1 - ema_beta) * mean_len
                    )

                entropy_coef = entropy_schedule(
                    step, steps, entropy_start, entropy_end
                )

                advantage = length - ema_baseline
                loss = (advantage * logp).mean() - entropy_coef * ent.mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % log_every == 0:
                sync("cuda")

                writer.writerow([
                    step,
                    n_nodes,
                    round(loss.item(), 6),
                    round(length.mean().item(), 6),
                    round(entropy_coef, 6),
                    round(ema_baseline, 6),
                ])
                f.flush()

                print(
                    f"[TRAIN][N={n_nodes}] "
                    f"step {step:05d} | "
                    f"tour {length.mean():.3f} | "
                    f"ema {ema_baseline:.3f} | "
                    f"entropy {entropy_coef:.3f}"
                )


# --------------------------------------------------
# Evaluation (sampling-only)
# --------------------------------------------------

@torch.no_grad()
def evaluate(model, device, n_nodes, batch, eval_k, eval_batches):
    model.eval()
    net = model

    tours, times = [], []

    for _ in range(eval_batches):
        coords = torch.rand(batch, n_nodes, 2, device=device)

        best = None
        start = time.perf_counter()

        for _ in range(eval_k):
            _, _, length = net.rollout(coords, greedy=False)
            val = length.mean().item()
            best = val if best is None else min(best, val)

        sync("cuda")
        end = time.perf_counter()

        tours.append(best)
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

    parser.add_argument("--entropy_start", type=float, default=0.05)
    parser.add_argument("--entropy_end", type=float, default=0.02)
    parser.add_argument("--ema_beta", type=float, default=0.99)

    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Device: {device}")

    os.makedirs("logs", exist_ok=True)

    # FIXED BIG MODEL
    model = TSPModel(dim=args.dim, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # CURRICULUM (ONLY N CHANGES)
    curriculum = [
        dict(N=10,  steps=3000),
        dict(N=20,  steps=10000),
        dict(N=50,  steps=20000),
        dict(N=100, steps=30000),
    ]

    for stage in curriculum:
        N = stage["N"]
        steps = stage["steps"]

        print(f"\n Training TSP{N} for {steps} steps")

        train_log = f"logs/tsp_curriculum_N{N}_train.csv"

        train_stage(
            model=model,
            opt=opt,
            device=device,
            n_nodes=N,
            steps=steps,
            batch=args.batch,
            entropy_start=args.entropy_start,
            entropy_end=args.entropy_end,
            ema_beta=args.ema_beta,
            log_every=args.log_every,
            log_path=train_log,
        )

        avg_tour, avg_time = evaluate(
            model,
            device,
            n_nodes=N,
            batch=args.batch,
            eval_k=32,
            eval_batches=20,
        )

        print(
            f"📊 Eval TSP{N} | "
            f"tour={avg_tour:.3f} | "
            f"time={avg_time:.1f} ms"
        )


if __name__ == "__main__":
    main()
