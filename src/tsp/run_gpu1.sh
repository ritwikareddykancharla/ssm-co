#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

echo "Running on GPU 1: N=20 and N=50"

python run_experiment.py \
  --n_nodes 20 \
  --dim 256 \
  --layers 4 \
  --batch 512 \
  --steps 10000 \
  --eval_k 16

python run_experiment.py \
  --n_nodes 50 \
  --dim 256 \
  --layers 4 \
  --batch 256 \
  --steps 10000 \
  --eval_k 16
