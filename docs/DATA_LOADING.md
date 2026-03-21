# Data Loading Strategies for LLM Pretraining

## The Problem: Python List Tokenization OOMs at Scale

The naive approach streams tokens into a Python list:

```python
tokens = []
for example in dataset:
    tokens.extend(tokenizer.encode(example["text"]))
```

**Why this fails:** Each Python integer takes ~28 bytes of memory due to object overhead, plus list pointer overhead. For 10M tokens this consumed ~46GB RAM and OOM'd the pod. For 2.5B tokens it would require ~6TB RAM.

## Option 1: Eager Python List (naive, avoid)
- **Memory:** ~2500 bytes/token (Python object overhead)
- **10M tokens:** ~25GB RAM
- **2.5B tokens:** ~6TB RAM (impossible)
- **Use case:** Never for pretraining scale

## Option 2: Pre-tokenize to numpy file + mmap (current approach)
Tokenize once, write to disk as `np.int32` array, load with `mmap_mode='r'`:

```python
# Pretokenize (run once)
tokens = np.zeros(num_tokens, dtype=np.int32)
# ... fill from streaming dataset ...
np.save("/workspace/data/fineweb_2b5.npy", tokens)

# Load (every training run)
tokens = np.load(path, mmap_mode='r')  # doesn't load into RAM
```

**Memory:** 4 bytes/token (int32)
**10M tokens:** 40MB on disk, ~0MB RAM (mmap)
**2.5B tokens:** 10GB on disk, ~0MB RAM (mmap)
**Use case:** Standard practice for fixed datasets

`mmap_mode='r'` means numpy reads only the pages needed from disk on demand. The OS handles caching. RAM usage stays near zero regardless of dataset size.

## Option 3: HuggingFace datasets with Arrow format
HuggingFace's `datasets` library does mmap automatically via Apache Arrow:

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb", split="train")
# Automatically memory-mapped, column-oriented storage
```

**Memory:** Near zero (Arrow mmap)
**Flexibility:** Easy filtering, shuffling, multi-shard
**Use case:** When you want HF ecosystem integration

## Option 4: Streaming (no pre-tokenization)
```python
dataset = load_dataset(..., streaming=True)
for example in dataset:
    # tokenize and yield on the fly
```

**Memory:** Near zero (one document at a time)
**Tradeoff:** Can't shuffle globally, DataLoader prefetch is harder
**Use case:** Datasets too large to pre-tokenize, or when storage is constrained

## Current Repo Approach

```
scripts/pretokenize.py    # run once per dataset size
↓
/workspace/data/fineweb_2b5.npy    # persisted on RunPod network volume
↓
create_fineweb_loaders_from_file()  # mmap load, slices in __getitem__
```

The startup script (`scripts/train_fineweb.sh`) checks if the tokenized file exists before training and runs pretokenization if not.

## Future: Larger Dataset Slices

`sample-10BT` has 10B tokens. Full Fineweb has ~15T tokens across CC dumps.

To scale up:
1. Change `name="sample-10BT"` to `name="default"` in pretokenize.py
2. Increase `NUM_TOKENS` in the startup script
3. Use a new output filename (`fineweb_10b.npy`) — old file stays on volume

File naming convention: encode what's in it — `fineweb_2b5.npy` = 2.5B tokens from sample-10BT.

## References
- [nanoGPT data preparation](https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py) — same mmap pattern
- [numpy mmap_mode docs](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
- [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
