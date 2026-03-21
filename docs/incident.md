# Incident Report: mmap Data Loading & dtype Bug

**Date:** 2026-03-21  
**Severity:** Medium — blocked training run, required multiple pod relaunches  
**Status:** Resolved (workaround applied, proper fix deferred)

---

## Summary

Two related issues prevented the first Fineweb training run from completing:
1. The `TokenDataset` numpy/mmap branch was written but never properly wired up
2. numpy's default `int32` dtype is incompatible with PyTorch's `cross_entropy` which requires `int64` targets

---

## Timeline

1. OOM on first Fineweb run — Python list tokenization consumed all 46GB RAM for 10M tokens
2. Designed mmap solution: pretokenize to `np.int32` file, load with `mmap_mode='r'`, slice in `__getitem__`
3. Implemented `TokenDataset` numpy branch in `__getitem__` but **never updated `__init__`** to set `self.use_numpy` or `self.tokens`
4. `create_fineweb_loaders_from_file` was implemented and called correctly
5. But `TokenDataset.__init__` still ran the eager list-building path, consuming memory as before
6. Additionally, `torch.tensor()` on numpy `int32` data produces `int32` tensors
7. `F.cross_entropy` requires `int64` targets → `RuntimeError: not implemented for 'Int'`

---

## Root Cause

### Issue 1: Incomplete mmap implementation
`TokenDataset.__init__` was never updated to handle numpy arrays:

```python
# What was written (broken — __init__ still builds lists)
class TokenDataset(Dataset):
    def __init__(self, tokens: list[int], context_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(tokens) - context_length, stride):
            self.input_ids.append(torch.tensor(tokens[i:i+context_length]))  # no dtype specified
            ...

    def __getitem__(self, idx):
        if self.use_numpy:  # AttributeError — self.use_numpy never set
            ...
```

### Issue 2: dtype mismatch
- `np.save()` saves as `int32` by default
- `torch.tensor(np_array)` preserves `int32` → produces `torch.int32` tensor  
- `F.cross_entropy(logits, targets)` requires targets to be `torch.int64` (Long)
- CUDA kernel `nll_loss_forward_reduce_cuda_kernel_2d_index` not implemented for `Int`

---

## Changes Made

### Workaround applied (immediate fix)
Added `.long()` cast in `calc_loss_batch` in `utils.py`:

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device).long()  # force int64
    ...
```

Added `dtype=torch.long` in `TokenDataset.__init__`:

```python
self.input_ids.append(
    torch.tensor(tokens[i:i+context_length], dtype=torch.long)
)
```

### What was NOT fixed
The mmap branch is still dead code. `TokenDataset` still eagerly loads all tokens into lists at init time. This means:
- RAM usage still scales linearly with dataset size
- 2.5B token run will likely OOM (estimated ~600GB RAM needed with Python list overhead)
- The pretokenized `.npy` file on the network volume is being loaded but then immediately converted to a Python list

---

## Clarification: int vs float dtypes

These are completely separate concerns:

| Tensor | dtype | Why |
|--------|-------|-----|
| Token IDs (input/target) | `int64` | Vocabulary indices, must be integers |
| Model weights | `float32` | Learnable parameters |
| Forward pass (bfloat16) | `bfloat16` | Mixed precision for speed |
| Embeddings, activations | `float32`/`bfloat16` | Floating point computations |

Changing token ID tensors from `int32` to `int64` has zero effect on model weights or training quality.

---

## Current State

- ✅ Pretokenization script works — `fineweb_2b5.npy` (10GB) saved to network volume
- ✅ dtype fix applied — `cross_entropy` error resolved  
- ✅ DataLoader wired up correctly via `create_fineweb_loaders_from_file`
- ❌ mmap never activated — still eager loading into RAM
- ⚠️ 2.5B token run will OOM — needs proper mmap fix before full run

---

## Future Fix: Proper mmap Implementation

The `TokenDataset` needs to be split into two classes or have proper numpy branch:

```python
class TokenDataset(Dataset):
    def __init__(self, tokens, context_length: int, stride: int):
        import numpy as np
        self.context_length = context_length
        self.stride = stride
        
        if isinstance(tokens, np.ndarray):
            # mmap path — store reference, slice on demand
            self.tokens = tokens  # numpy memmap, stays on disk
            self.use_numpy = True
            self.length = (len(tokens) - context_length) // stride
        else:
            # eager path — for small datasets (verdict)
            self.use_numpy = False
            self.input_ids = []
            self.target_ids = []
            for i in range(0, len(tokens) - context_length, stride):
                self.input_ids.append(
                    torch.tensor(tokens[i:i+context_length], dtype=torch.long)
                )
                self.target_ids.append(
                    torch.tensor(tokens[i+1:i+context_length+1], dtype=torch.long)
                )

    def __len__(self):
        if self.use_numpy:
            return self.length
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.use_numpy:
            start = idx * self.stride
            x = torch.from_numpy(
                self.tokens[start:start + self.context_length].astype(np.int64)
            )
            y = torch.from_numpy(
                self.tokens[start+1:start + self.context_length + 1].astype(np.int64)
            )
            return x, y
        return self.input_ids[idx], self.target_ids[idx]
```

With this fix, `create_fineweb_loaders_from_file` passes the mmap array directly and RAM usage drops to near zero.

---

## Memory Impact Comparison

| Approach | 10M tokens RAM | 2.5B tokens RAM |
|----------|---------------|-----------------|
| Python list (current) | ~25GB | ~6TB (impossible) |
| numpy eager load | ~40MB | ~10GB |
| numpy mmap (target) | ~0MB | ~0MB |

---

## Action Items

- [ ] Implement proper mmap `TokenDataset` before 2.5B token run
- [ ] Validate on 10M token run with mmap active (RAM should stay <1GB)
- [ ] Consider using HuggingFace Arrow format as alternative to custom mmap
- [ ] Add RAM monitoring to W&B logging to catch this earlier next time
