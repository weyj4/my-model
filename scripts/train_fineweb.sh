#!/bin/bash
set -e
[ -f .env ] && source .env

python -m gpt2.train \
    --run_name fineweb-baseline \
    --batch_size 32 \
    --num_tokens 2500000000 \
    --lr 4e-4
