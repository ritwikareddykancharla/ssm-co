# --------------------------------------------------
# Clean experiment runner for TSP (PAPER-READY)
# - entropy schedule (training)
# - multi-start evaluation (k rollouts)
# - CLEAN LOGGING + ROUNDED VALUES
#   * train CSV: step, loss, avg_tour, entropy
#   * eval  CSV: eval_avg_tour, inference_ms
# --------------------------------------------------

import argparse
import csv
import os
import time
import torch

from model import TSPModel
from utils.device import get_device, optimizer_step, sync


# --------------------------------------------------
# Auto device selection
# --------------------------------------------------
def auto_device():
    try:
        import torch_xla.core.xla_model as xm
        _ = xm.xla_device()
        return "tpu"
    except Exception:
        pass

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# --------------------------------------------------
# Entropy schedule
# --------------------------------------------------
def entropy_schedule(step, total_steps, start, end):
    frac = step / total_steps
    return start * (1 - frac) + end * frac


# --------------------------------------------------
# Training
# --------------------------------------------------
def train(model, opt, device, args, device_name, train_writer):
    model.train()

    for step in range(args.steps):
        coords = torch.rand(args.batch, args.n_nodes, 2, device=device)

        logp, ent, length = model.rollout(coords, greedy=False)

        with torch.no_grad():
            _, _, greedy_len = model.rollout(coords, greedy=True)

        entropy_coef = entropy_schedule(
            step,
            args.steps,
            args.entropy_start,
            args.entropy_end,
        )

        loss = ((length - greedy_len) * logp).mean() - entropy_coef * ent.mean()

        opt.zero_grad()
        loss.backward()
        optimizer_step(opt, device_name)

        if step % args.log_every == 0:
            sync(device_name)

            avg_tour = length.mean().item()
            loss_val = loss.item()

            print(
                f"[TRAIN][N={args.n_nodes}] "
                f"step {step:04d} | "
                f"loss {loss_val:.4f} | "
                f"tour {avg_tour:.3f} | "
                f"entropy {entropy_coef:.4f}"
            )

            # ---- TRAIN CSV (ROUNDED, NO REDUNDANT COLS) ----
            train_writer.writerow([
                step,
                round(loss_val, 4),
                round(avg_tour, 3),
                round(entropy_coef, 4),
            ])


# --------------------------------------------------
# Evaluation (multi-start)
# --------------------------------------------------
@torch.no_grad()
def evaluate(model, device, args, device_name):
    model.eval()

    tours = []
    times = []

    for _ in range(args.eval_batches):
        coords = torch.rand(args.batch, args.n_nodes, 2, device=device)

        best_len = None
        start = time.perf_counter()

        for _ in range(args.eval_k):
            _, _, length = model.rollout(
                coords, greedy=not args.sample_eval
            )
            mean_len = length.mean().item()
            best_len = mean_len if best_len is None else min(best_len, mean_len)

        sync(device_name)
        end = time.perf_counter()

        tours.append(best_len)
        times.append((end - start) * 1000.0)

    return sum(tours) / len(tours), sum(times) / len(times)


# --------------------------------------------------
# Main
# --------------------------------------------------
def run(args):
    device_name = auto_device()
    device = get_device(device_name)

    print(f"\n🚀 Device: {device_name.upper()}")
    print(
        f"📦 Config: N={args.n_nodes}, dim={args.dim}, "
        f"layers={args.layers}, batch={args.batch}"
    )
    print(
        f"🔁 Eval: k={args.eval_k}, "
        f"{'sampling' if args.sample_eval else 'greedy'}\n"
    )

    model = TSPModel(dim=args.dim, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("logs", exist_ok=True)

    run_tag = f"tsp_N{args.n_nodes}_D{args.dim}_L{args.layers}_B{args.batch}_S{args.steps}"
    train_log = f"logs/{run_tag}_train.csv"
    eval_log = f"logs/{run_tag}_eval.csv"

    # -------- TRAIN LOG --------
    with open(train_log, "w", newline="") as f_train:
        train_writer = csv.writer(f_train)
        train_writer.writerow([
            "step",
            "loss",
            "avg_tour",
            "entropy",
        ])

        train(model, opt, device, args, device_name, train_writer)

    # -------- EVAL LOG --------
    avg_tour, avg_time = evaluate(model, device, args, device_name)

    with open(eval_log, "w", newline="") as f_eval:
        eval_writer = csv.writer(f_eval)
        eval_writer.writerow([
            "eval_avg_tour",
            "inference_ms",
        ])

        eval_writer.writerow([
            round(avg_tour, 3),
            round(avg_time, 2),
        ])

    print("\n📊 Evaluation Results")
    print(f"Avg tour length   : {avg_tour:.3f}")
    print(f"Inference time    : {avg_time:.2f} ms")
    print(f"Saved train log   : {train_log}")
    print(f"Saved eval log    : {eval_log}\n")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_nodes", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)

    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=20)

    parser.add_argument("--eval_k", type=int, default=8)
    parser.add_argument("--sample_eval", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy_start", type=float, default=0.05)
    parser.add_argument("--entropy_end", type=float, default=0.005)
    parser.add_argument("--log_every", type=int, default=50)

    args = parser.parse_args()
    run(args)
