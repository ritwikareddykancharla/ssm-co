#!/bin/bash
# ==================================================
# Sequential TSP Experiments (TPU / GPU / CPU)
# Same model, different N
# ==================================================

DIM=256
LAYERS=4
STEPS=10000
BATCH=512        # TPU-friendly (use 512 if OOM)
LR=1e-4

EVAL_K=16
LOG_EVERY=10

echo "======================================"
echo "Running Sequential TSP Experiments"
echo "dim=$DIM layers=$LAYERS steps=$STEPS batch=$BATCH"
echo "======================================"

# -------- TSP 10 --------
python run_experiment.py \
  --n_nodes 10 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --log_every $LOG_EVERY

# -------- TSP 20 --------
python run_experiment.py \
  --n_nodes 20 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --log_every $LOG_EVERY

# -------- TSP 50 --------
python run_experiment.py \
  --n_nodes 50 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --log_every $LOG_EVERY

# -------- TSP 100 --------
python run_experiment.py \
  --n_nodes 100 \
  --dim $DIM \
  --layers $LAYERS \
  --batch $BATCH \
  --steps $STEPS \
  --eval_k $EVAL_K \
  --log_every $LOG_EVERY

echo "======================================"
echo "All experiments finished 💅"
echo "Logs saved in logs/"
echo "======================================"
