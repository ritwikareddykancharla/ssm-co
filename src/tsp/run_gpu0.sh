#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "Running on GPU 0: N=10 and N=100"

python run_experiment.py \
  --n_nodes 10 \
  --dim 256 \
  --layers 4 \
  --batch 512 \
  --steps 1000 \
  --eval_k 16

python run_experiment.py \
  --n_nodes 100 \
  --dim 256 \
  --layers 4 \
  --batch 256 \
  --steps 1000 \
  --eval_k 16
