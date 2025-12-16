# State-Space Autoregressive Neural Combinatorial Optimization

This repository contains a minimal reference implementation for the
AAAI-Bridge submission *State-Space Autoregressive Decoding for Neural
Combinatorial Optimization*.

## Setup

### CPU / GPU
pip install -r requirements.txt

### TPU
pip install torch==2.1.0
pip install torch_xla[tpu]==2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html


## Running Experiments

Run a single experiment:

```bash
./run_experiment.sh 100 256 4
