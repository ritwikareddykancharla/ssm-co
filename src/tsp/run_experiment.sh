#!/bin/bash
# ==================================================
# Sequential TSP Experiments (DDP, Kaggle 2×GPU)
# Uses BOTH GPUs via torchrun
# Same model, different N (sequential)
# ==================================================

# ---------------- CONFIG ----------------
DIM=256
LAYERS=4
STEPS=1000

# IMPORTANT:
# Batch is PER GPU in DDP
# Effective batch = BATCH_PER_GPU × 2
BATCH_PER_GPU=256

LR=1e-4
EVAL_K=16
LOG_EVERY=10

NUM_GPUS=2

echo "======================================"
echo "Running Sequential TSP Experiments (DDP)"
echo "dim=$DIM layers=$LAYERS steps=$STEPS batch/gpu=$BATCH_PER_GPU gpus=$NUM_GPUS"
echo "======================================"

# -------- TSP 10 --------
echo "\n TSP N=10"
torchrun --nproc_per_node=$NUM_GPUS run_experiment.py \
  --n_nodes 10 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH_PER_GPU \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --log_every $LOG_EVERY

# -------- TSP 20 --------
echo "\n TSP N=20"
torchrun --nproc_per_node=$NUM_GPUS run_experiment.py \
  --n_nodes 20 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH_PER_GPU \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --log_every $LOG_EVERY

# -------- TSP 50 --------
echo "\n TSP N=50"
torchrun --nproc_per_node=$NUM_GPUS run_experiment.py \
  --n_nodes 50 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH_PER_GPU \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --log_every $LOG_EVERY

# -------- TSP 100 --------
echo "\n TSP N=100"
torchrun --nproc_per_node=$NUM_GPUS run_experiment.py \
  --n_nodes 100 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH_PER_GPU \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --log_every $LOG_EVERY

echo "======================================"
echo "All experiments finished 💅"
echo "Logs saved in logs/"
echo "======================================"
