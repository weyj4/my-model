#!/bin/bash
set -e
source .env

python -m gpt2.train \
    --run_name fineweb-baseline \
    --batch_size 32 \
    --num_tokens 2_500_000_000 \
    --lr 4e-4
