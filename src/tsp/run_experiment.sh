#!/bin/bash
set -e

mkdir -p logs

echo "Running TSP experiments on TPU..."

# -------- SMALL PROBLEMS --------
python tsp_toy.py \
  --n_nodes 10 \
  --dim 128 \
  --layers 2 \
  --batch 16 \
  --steps 500 \
  --log_every 5 \
  > logs/tsp10_dim128.log

python tsp_toy.py \
  --n_nodes 20 \
  --dim 128 \
  --layers 2 \
  --batch 16 \
  --steps 500 \
  --log_every 5 \
  > logs/tsp20_dim128.log

# -------- MEDIUM PROBLEM --------
python tsp_toy.py \
  --n_nodes 50 \
  --dim 256 \
  --layers 4 \
  --batch 16 \
  --steps 500 \
  --log_every 10 \
  > logs/tsp50_dim256.log

# -------- LARGE PROBLEM --------
python tsp_toy.py \
  --n_nodes 100 \
  --dim 256 \
  --layers 4 \
  --batch 16 \
  --steps 500 \
  --log_every 10 \
  > logs/tsp100_dim256.log

echo "All experiments completed."
