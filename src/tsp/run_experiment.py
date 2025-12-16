# --------------------------------------------------
# Clean experiment runner for TSP (PAPER-READY, DDP)
# - entropy schedule (training)
# - multi-start evaluation (k rollouts)
# - CLEAN LOGGING + ROUNDED VALUES
#   * train CSV: step, loss, avg_tour, entropy
#   * eval  CSV: eval_avg_tour, inference_ms
# - Kaggle GPU–safe
# - DDP-enabled (torchrun)
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
    frac = step / total_steps
    return start * (1 - frac) + end * frac


# --------------------------------------------------
# Training
# --------------------------------------------------

def train(model, opt, device, args, is_main):
    model.train()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for step in range(args.steps):
        coords = torch.rand(args.batch, args.n_nodes, 2, device=device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
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

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0 and is_main:
            sync("cuda")
            print(
                f"[TRAIN][N={args.n_nodes}] "
                f"step {step:04d} | "
                f"loss {loss.item():.4f} | "
                f"tour {length.mean().item():.3f} | "
                f"entropy {entropy_coef:.4f}"
            )


# --------------------------------------------------
# Evaluation (single-GPU, rank 0 only)
# --------------------------------------------------
@torch.no_grad()

def evaluate(model, device, args):
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

        sync("cuda")
        end = time.perf_counter()

        tours.append(best_len)
        times.append((end - start) * 1000.0)

    return sum(tours) / len(tours), sum(times) / len(times)


# --------------------------------------------------
# Main (DDP-aware)
# --------------------------------------------------

def run(args):
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
        print(f"\n🚀 Device: {device}")
        print(
            f"📦 Config: N={args.n_nodes}, dim={args.dim}, "
            f"layers={args.layers}, batch={args.batch}"
        )
        print(
            f"🔁 Eval: k={args.eval_k}, "
            f"{'sampling' if args.sample_eval else 'greedy'}\n"
        )

    model = TSPModel(dim=args.dim, layers=args.layers).to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if is_main:
        os.makedirs("logs", exist_ok=True)

    run_tag = (
        f"tsp_N{args.n_nodes}_D{args.dim}_L{args.layers}_"
        f"B{args.batch}_S{args.steps}"
    )

    train_log = f"logs/{run_tag}_train.csv"
    eval_log = f"logs/{run_tag}_eval.csv"

    if is_main:
        with open(train_log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "avg_tour", "entropy"])

    train(model, opt, device, args, is_main)

    if is_main:
        avg_tour, avg_time = evaluate(model, device, args)

        with open(eval_log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["eval_avg_tour", "inference_ms"])
            writer.writerow([
                round(avg_tour, 3),
                round(avg_time, 2),
            ])

        print("\n📊 Evaluation Results")
        print(f"Avg tour length   : {avg_tour:.3f}")
        print(f"Inference time    : {avg_time:.2f} ms")
        print(f"Saved train log   : {train_log}")
        print(f"Saved eval log    : {eval_log}\n")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


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
