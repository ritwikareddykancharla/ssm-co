# --------------------------------------------------
# Curriculum Training Runner for TSP (PAPER-READY)
# - SINGLE model trained across increasing N
# - Hybrid baseline: greedy + EMA
# - Entropy schedule per stage
# - Sampling-only multi-start evaluation
# --------------------------------------------------

import csv
import os
import time
import torch
import torch.nn as nn

from model import TSPModel
from utils.device import sync


# ---------------- Curriculum ----------------

CURRICULUM = [
    dict(n=10,  steps=2000,  layers=4, entropy_end=0.02),
    dict(n=20,  steps=10000, layers=4, entropy_end=0.01),
    dict(n=50,  steps=20000, layers=5, entropy_end=0.005),
    dict(n=100, steps=30000, layers=6, entropy_end=0.002),
]

DIM = 256
BATCH = 512
LR = 1e-4
EMA_BETA = 0.99
ENTROPY_START = 0.05
EVAL_K = 32
LOG_EVERY = 50


# ---------------- Utils ----------------

def entropy_schedule(step, total, start, end):
    frac = step / total
    return start * (1 - frac) + end * frac


def expand_model(old_model, new_layers):
    """Safely increase depth while preserving weights."""
    if old_model.layers == new_layers:
        return old_model

    print(f" Expanding model: {old_model.layers} → {new_layers} layers")

    new_model = TSPModel(dim=DIM, layers=new_layers)
    new_model.load_state_dict(old_model.state_dict(), strict=False)
    return new_model


# ---------------- Training ----------------

def train_stage(model, optimizer, stage, device, ema_baseline):
    n = stage["n"]
    steps = stage["steps"]
    entropy_end = stage["entropy_end"]

    model.train()
    net = model

    os.makedirs("logs", exist_ok=True)
    train_log = f"logs/tsp_N{n}_train.csv"

    with open(train_log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "avg_tour", "entropy", "ema"])

        for step in range(steps):
            coords = torch.rand(BATCH, n, 2, device=device)

            logp, ent, length = net.rollout(coords, greedy=False)
            with torch.no_grad():
                _, _, greedy_len = net.rollout(coords, greedy=True)

                batch_mean = length.mean().item()
                if ema_baseline is None:
                    ema_baseline = batch_mean
                else:
                    ema_baseline = EMA_BETA * ema_baseline + (1 - EMA_BETA) * batch_mean

            baseline = torch.minimum(
                greedy_len,
                torch.tensor(ema_baseline, device=device)
            )

            entropy_coef = entropy_schedule(
                step, steps, ENTROPY_START, entropy_end
            )

            loss = ((length - baseline) * logp).mean() - entropy_coef * ent.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % LOG_EVERY == 0:
                sync("cuda")
                print(
                    f"[N={n}] step {step:05d} | "
                    f"tour {length.mean():.3f} | "
                    f"entropy {entropy_coef:.4f} | "
                    f"ema {ema_baseline:.3f}"
                )

                writer.writerow([
                    step,
                    round(loss.item(), 6),
                    round(length.mean().item(), 6),
                    round(entropy_coef, 6),
                    round(ema_baseline, 6),
                ])
                f.flush()

    return ema_baseline


# ---------------- Evaluation ----------------

@torch.no_grad()
def evaluate(model, n, device):
    model.eval()
    net = model

    tours = []
    times = []

    for _ in range(20):
        coords = torch.rand(BATCH, n, 2, device=device)
        best = None
        start = time.perf_counter()

        for _ in range(EVAL_K):
            _, _, length = net.rollout(coords, greedy=False)
            m = length.mean().item()
            best = m if best is None else min(best, m)

        sync("cuda")
        times.append((time.perf_counter() - start) * 1000)
        tours.append(best)

    return sum(tours) / len(tours), sum(times) / len(times)


# ---------------- Main ----------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Device: {device}\n")

    model = TSPModel(dim=DIM, layers=CURRICULUM[0]["layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ema_baseline = None

    for stage in CURRICULUM:
        model = expand_model(model, stage["layers"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        print(f"\n Curriculum stage: N={stage['n']} ({stage['steps']} steps)\n")

        ema_baseline = train_stage(
            model, optimizer, stage, device, ema_baseline
        )

        avg_tour, avg_time = evaluate(model, stage["n"], device)

        print(
            f"\n📊 Eval N={stage['n']} | "
            f"tour={avg_tour:.3f} | time={avg_time:.1f} ms\n"
        )


if __name__ == "__main__":
    main()
