#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m gpt2.train \
    --run_name fineweb-baseline \
    --batch_size 32 \
    --num_tokens 2500000000 \
    --lr 4e-4
