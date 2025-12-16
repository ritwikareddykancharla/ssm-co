#!/bin/bash
# ==================================================
# Sequential TSP Experiments (DDP, Kaggle 2×GPU)
# - Sampling-only evaluation (best-of-k)
# - Entropy regularization ONLY during training
# - EMA baseline inside runner
# - Steps scaled by problem difficulty
# ==================================================
#
# 📊 Training Budget (Tours-based, not vibes-based)
#
# | N   | Tours Target | Steps   | Why                |
# | --- | ------------ | ------- | ------------------ |
# | 10  | 2–3M         | 2–3k    | sanity only        |
# | 20  | 10M          | 10k     | proper convergence |
# | 50  | 15–20M       | 15–20k  | strong neural CO   |
# | 100 | 20–30M       | 20–30k  | scaling flex       |
#
# Notes:
# - tours/step = batch_per_gpu × num_gpus = 512 × 2 = 1024
# - steps ≈ target_tours / 1024
# ==================================================

# ---------------- GLOBAL CONFIG ----------------
BATCH_PER_GPU=512        # effective batch = 1024
LR=1e-4
EVAL_K=32
LOG_EVERY=20
NUM_GPUS=2

# Entropy schedule (slow decay, avoid collapse)
ENTROPY_START=0.05
ENTROPY_END=0.02

RUNNER=tsp_runner_sampling_eval.py

echo "======================================"
echo "Running Sequential TSP Experiments (DDP)"
echo "batch/gpu=$BATCH_PER_GPU gpus=$NUM_GPUS"
echo "eval: sampling-only (k=$EVAL_K)"
echo "entropy: start=$ENTROPY_START end=$ENTROPY_END"
echo "======================================"

# ==================================================
# TSP10 — sanity check (≈3M tours)
# ==================================================
DIM=256
LAYERS=4
STEPS=3000

echo -e "\n TSP N=10 | steps=$STEPS"
torchrun --nproc_per_node=$NUM_GPUS $RUNNER \
  --n_nodes 10 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH_PER_GPU \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --entropy_start $ENTROPY_START \
  --entropy_end $ENTROPY_END \
  --log_every $LOG_EVERY \
  --lr $LR

# ==================================================
# TSP20 — proper convergence (≈10M tours)
# ==================================================
DIM=256
LAYERS=4
STEPS=10000

echo -e "\n TSP N=20 | steps=$STEPS"
torchrun --nproc_per_node=$NUM_GPUS $RUNNER \
  --n_nodes 20 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH_PER_GPU \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --entropy_start $ENTROPY_START \
  --entropy_end $ENTROPY_END \
  --log_every $LOG_EVERY \
  --lr $LR

# ==================================================
# TSP50 ≈20M tours
# ==================================================
DIM=256
LAYERS=5
STEPS=20000

echo -e "\n TSP N=50 | steps=$STEPS"
torchrun --nproc_per_node=$NUM_GPUS $RUNNER \
  --n_nodes 50 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH_PER_GPU \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --entropy_start $ENTROPY_START \
  --entropy_end $ENTROPY_END \
  --log_every $LOG_EVERY \
  --lr $LR

# ==================================================
# TSP100 ≈30M tours
# ==================================================
DIM=256
LAYERS=6
STEPS=30000

echo -e "\n TSP N=100 | steps=$STEPS"
torchrun --nproc_per_node=$NUM_GPUS $RUNNER \
  --n_nodes 100 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH_PER_GPU \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --entropy_start $ENTROPY_START \
  --entropy_end $ENTROPY_END \
  --log_every $LOG_EVERY \
  --lr $LR

echo "======================================"
echo "All experiments finished 💅"
echo "Logs saved in logs/"
echo "======================================"
