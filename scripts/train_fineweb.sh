#!/bin/bash
set -e
[ -f .env ] && source .env

export TORCH_HOME=/workspace/torch_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_PATH="/workspace/data/fineweb_2b5.npy"
NUM_TOKENS=2500000000

if [ ! -f "$DATA_PATH" ]; then
    echo "Token file not found, pretokenizing..."
    python scripts/pretokenize.py \
        --num_tokens $NUM_TOKENS \
        --output $DATA_PATH
    echo "Pretokenization complete"
else
    echo "Token file found at $DATA_PATH, skipping pretokenization"
fi

python -m gpt2.train \
    --run_name flash-bs64 \
    --batch_size 32 \
    --num_tokens $NUM_TOKENS \
    --dataset fineweb_file \
    --lr 4e-4
