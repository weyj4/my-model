#!/usr/bin/env python3
"""
Run once to tokenize Fineweb and save to disk.
Usage: python scripts/pretokenize.py --num_tokens 2500000000 --output /workspace/data/fineweb_2b5.npy
"""
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
import os

DATASETS = {
    "fineweb": "HuggingFaceFW/fineweb",
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
}

def pretokenize(hf_dataset: str, config: str, num_tokens: int, output_path: str):
    hf_name = DATASETS[hf_dataset]
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = load_dataset(
        # "HuggingFaceFW/fineweb",
        # "HuggingFaceFW/fineweb-edu"
        hf_name,
        name=config,
        split="train",
        streaming=True
    )

    # pre-allocate — no Python list accumulation
    tokens = np.zeros(num_tokens, dtype=np.uint16)
    idx = 0

    for example in dataset:
        chunk = tokenizer.encode_ordinary(example["text"])
        chunk.append(tokenizer.eot_token)
        end = min(idx + len(chunk), num_tokens)
        tokens[idx:end] = chunk[:end - idx]
        idx = end
        if idx % 1_000_000 == 0:
            print(f"  {idx:,} / {num_tokens:,} tokens")
        if idx >= num_tokens:
            break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, tokens[:idx])
    print(f"Saved {idx:,} tokens to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e9:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fineweb")
    parser.add_argument("--config", type=str, default="sample-10BT")
    parser.add_argument("--num_tokens", type=int, default=2_500_000_000)
    parser.add_argument("--output", type=str, default="/workspace/data/fineweb_2b5.npy")
    args = parser.parse_args()
    pretokenize(args.dataset, args.config, args.num_tokens, args.output)
