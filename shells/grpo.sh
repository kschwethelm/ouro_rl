#!/bin/bash
set -e

cd ~/ouro_rl
uv sync
source .venv/bin/activate

source shells/_machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from machine_config.sh, defaults to 1)
NPROC_PER_NODE=${NUM_GPUS:-1}

# Run GRPO training
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE scripts/grpo_train.py \
    "$@"
