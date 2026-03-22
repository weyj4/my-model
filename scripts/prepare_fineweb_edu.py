"""
Download and tokenize FineWeb-Edu sample-10BT.
Saves to /workspace/data/fineweb_edu_3b.npy (~6.6GB, uint16).
Runtime: ~45 min on CPU, ~15 min on GPU pod.

Usage:
    python scripts/prepare_fineweb_edu.py
"""
import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path

SAVE_PATH = "/workspace/data/fineweb_edu_3b.npy"
TARGET_TOKENS = 3_300_000_000

Path("/workspace/data").mkdir(parents=True, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256

all_tokens = []
total = 0

print("Streaming HuggingFaceFW/fineweb-edu sample-10BT...")
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

for doc in ds:
    tokens = enc.encode_ordinary(doc["text"])
    tokens.append(EOT)
    all_tokens.extend(tokens)
    total += len(tokens)
    if total % 50_000_000 == 0:
        print(f"  {total/1e9:.2f}B / {TARGET_TOKENS/1e9:.1f}B tokens...")
    if total >= TARGET_TOKENS:
        break

print(f"Done. {total:,} tokens. Saving...")
arr = np.array(all_tokens[:TARGET_TOKENS], dtype=np.uint16)
np.save(SAVE_PATH, arr)
print(f"Saved: {SAVE_PATH}  ({arr.nbytes/1e9:.1f} GB)")
