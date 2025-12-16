#!/bin/bash
# ==================================================
# Sequential TSP Experiments (DDP, Kaggle 2×GPU)
# - Sampling-only evaluation (multi-start)
# - Entropy regularization ONLY during training
# - Paper-ready, clean baselines
# ==================================================

# ---------------- CONFIG ----------------
DIM=256
LAYERS=4
STEPS=1000

# IMPORTANT:
# Batch is PER GPU in DDP
# Effective batch = BATCH_PER_GPU × NUM_GPUS
BATCH_PER_GPU=512

LR=1e-4

# Sampling runs at evaluation (best-of-k)
EVAL_K=32

# 🔥 ENTROPY (THIS IS WHAT YOU ASKED FOR)
ENTROPY_START=0.05
ENTROPY_END=0.02

LOG_EVERY=10
NUM_GPUS=2

RUNNER=tsp_runner_sampling_eval.py

echo "======================================"
echo "Running Sequential TSP Experiments (DDP)"
echo "dim=$DIM layers=$LAYERS steps=$STEPS batch/gpu=$BATCH_PER_GPU gpus=$NUM_GPUS"
echo "eval: sampling-only (k=$EVAL_K)"
echo "entropy: start=$ENTROPY_START end=$ENTROPY_END"
echo "======================================"

# -------- TSP 10 --------
echo -e "\n TSP N=10"
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

# -------- TSP 20 --------
echo -e "\n TSP N=20"
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

# -------- TSP 50 --------
echo -e "\n TSP N=50"
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

# -------- TSP 100 --------
echo -e "\n TSP N=100"
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
