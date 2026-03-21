# mmap: What It Is, How It Works, and Why We Use It

## The problem it solves

After pretokenization, our training corpus is a flat array of 2.5 billion int32 tokens stored in a single `.npy` file (~10GB on the RunPod network volume). The naive approach is:

```python
tokens = np.load(path)  # loads entire 10GB into DRAM
```

This works but forces the OS to copy the entire file into CPU DRAM before training can start. For 10GB that's annoying. For 100GB it's a hard constraint — you need 100GB of free DRAM just to hold the dataset, before model weights, optimizer state, or anything else.

mmap solves this by making the file *addressable* without making it *resident*.

---

## What mmap actually is

`mmap()` is a POSIX system call that has existed since Unix in the 1980s. NumPy's `mmap_mode='r'` is a thin wrapper around it. When you call:

```python
tokens = np.load(path, mmap_mode='r')
```

What actually happens:

1. The OS reserves a region of the process's **virtual address space** the same size as the file
2. It records a mapping: "virtual addresses X through Y correspond to file at path Z"
3. It returns immediately — **no data has moved**
4. The returned numpy array looks and behaves like a normal array, but its backing memory is the file, not DRAM

When your code later accesses `tokens[42000]`, the CPU tries to read that virtual address and finds no physical RAM page backing it. This triggers a **page fault** — a hardware interrupt handled by the OS. The OS then:

1. Allocates a physical RAM page
2. Reads the corresponding 4KB chunk from disk into that page
3. Updates the page table to map the virtual address to the physical page
4. Resumes your program, which sees the data as if it had always been there

Your code never explicitly initiated a disk read. It just accessed an array index.

---

## The hardware picture

It helps to have a concrete mental model of where data lives at each stage.

```
NVMe SSD / Network Volume / GCS
    ↓  (page fault → OS reads 4KB page)
CPU DRAM  (~tens of GB, ~50 GB/s bandwidth)
    ↓  (CPU L3 cache, automatic, ~tens of MB)
CPU L2/L1 cache  (SRAM, ~MB, ~TB/s bandwidth, managed by hardware)
    ↓  (DataLoader → pinned memory → PCIe transfer)
GPU HBM  (~48GB on A40, ~2 TB/s bandwidth)
    ↓  (CUDA kernel reads from HBM into shared memory)
GPU shared memory / registers  (SRAM on die, ~100s KB per SM, very fast)
```

DRAM is what people usually mean by "program memory" or "RAM". It's where your numpy arrays, Python objects, and model weights live on the CPU side. SRAM is what CPU caches are made of — much faster, much smaller (~tens of MB total), managed automatically by the CPU hardware. You don't explicitly control it in Python, and you don't need to.

On the GPU side, HBM (High Bandwidth Memory) is the GPU's equivalent of DRAM — this is where PyTorch tensors live when you `.to(device)`. GPU shared memory is the GPU's equivalent of L1 cache — when writing Triton or CUDA C kernels, you explicitly load tiles from HBM into shared memory to reuse them, because HBM bandwidth (~2 TB/s) is still the bottleneck for many operations.

For our dataset loading, the relevant path is: **disk → DRAM via mmap page faults → GPU HBM via DataLoader's pinned memory transfer**.

---

## Page size and access granularity

The OS does not fetch one token at a time. It fetches one **page** at a time. On Linux the default page size is **4KB**.

At 4 bytes per int32 token:

```
4096 bytes / 4 bytes per token = 1024 tokens per page
```

This happens to match our context length exactly. Each new training window of 1024 tokens triggers roughly one page fault and one 4KB disk read — in the common case. In practice the OS also performs **readahead**: when it detects sequential access patterns, it speculatively prefetches the next several pages before they're needed. This amortizes the page fault latency across many accesses and means sequential training through the dataset rarely stalls waiting for disk.

With shuffled DataLoader access (random `__getitem__` indices), readahead helps less because access is non-sequential. Each random access into a cold page triggers a fresh page fault. This is the main cost of shuffle with mmap — random I/O instead of sequential I/O.

---

## OS page cache and memory management

Pages that have been loaded from disk stay in DRAM in the **page cache** until the OS decides to evict them. The eviction policy is LRU (Least Recently Used) — pages that haven't been accessed recently get evicted first when memory pressure increases.

This means:

- You don't explicitly manage which pages are in RAM — the OS does
- If your training is sequential and your DRAM is large enough, pages accumulate and stay resident — effectively caching the whole dataset in RAM over time
- If your DRAM is smaller than the dataset and access is random, the OS continuously evicts old pages and loads new ones — you're doing genuine just-in-time loading throughout training
- Multiple processes (DataLoader workers) that mmap the same file share the same physical pages in the page cache — no duplication

You don't have "one address for the current chunk" because `TokenDataset.__getitem__` needs to support arbitrary random access. PyTorch's DataLoader with `shuffle=True` calls `__getitem__(idx)` with random indices. The large virtual address space gives you an index into any position in the corpus instantly, and the OS handles what's actually resident in physical RAM.

---

## Why not have one pointer and free as you go?

Two reasons:

**Random access.** With `shuffle=True`, the DataLoader requests arbitrary indices in arbitrary order. A single-pointer streaming approach only supports sequential access. If you want the model to see tokens in shuffled order (which you do, to avoid learning spurious sequential correlations in the corpus), you need addressable random access across the full dataset.

**The OS already does the freeing.** You don't need to explicitly free pages because the OS's page cache management handles this automatically. Implementing your own free-on-use logic would just be reinventing LRU eviction at the application layer, worse than what the OS already does.

---

## The tradeoff in plain terms

mmap is a **memory-for-latency tradeoff** — but it's a favorable one for training workloads.

| Approach | DRAM usage | Startup time | Access latency |
|---|---|---|---|
| `np.load()` (eager) | Full dataset size | Slow (full load) | Zero (all in RAM) |
| `mmap_mode='r'` | Only resident pages | Instant | Page fault on cold access |
| Streaming from HuggingFace | Near zero | Instant | Network latency per document |

For our training loop on an A40, the GPU is almost certainly **compute-bound** — the matrix multiplications inside the transformer dominate wall clock time. The DataLoader runs in parallel background workers prefetching the next batch while the GPU processes the current one. As long as DataLoader throughput exceeds GPU throughput (which it usually does for this model size), page fault latency is completely hidden.

The case where mmap becomes a bottleneck is:
- Random access pattern (shuffle) on a very large dataset that doesn't fit in DRAM
- Slow storage (high-latency network volume or GCS)
- Very small model where GPU steps are fast and DataLoader becomes the bottleneck

In those cases the mitigation is increasing `num_workers` in DataLoader (more parallel prefetch), or switching to sequential access (no shuffle, or shuffle only within large chunks).

---

## mmap vs HuggingFace streaming — the key distinction

These solve different problems and should not be conflated:

**HuggingFace `streaming=True`**: The dataset is **remote** (on HuggingFace servers). Streaming means "fetch documents one at a time over HTTP rather than downloading everything first." Solves the problem of datasets too large to download. Data moves from HuggingFace → your machine on each iteration.

**mmap**: The file is **local** (on your disk or network volume). mmap means "map this local file into virtual address space and let the OS page it in as needed." Solves the problem of local files too large to hold entirely in DRAM. Data moves from your local disk → DRAM on each page fault.

Both defer data movement until access time, but at completely different layers: HuggingFace streaming is application-level lazy fetching over a network; mmap is OS-level virtual memory management against local storage.

In our architecture, streaming is used **once** (during pretokenization, to avoid downloading the full 40GB FineWeb sample). After that, we never touch HuggingFace again. mmap is used **every training run**, to make the pretokenized `.npy` file accessible without loading it entirely into DRAM.

---

## Relevance to PagedAttention / vLLM

PagedAttention (Kwon et al., 2023) explicitly borrows the virtual memory paging metaphor — the paper cites OS virtual memory as the direct inspiration. But it applies the idea to a different resource:

| OS virtual memory / mmap | PagedAttention |
|---|---|
| Physical DRAM pages | GPU HBM blocks |
| Virtual address space | Logical KV cache sequence |
| Page table | Block table |
| Page fault → load from disk | Block request → allocate from pool |
| LRU eviction to disk | Eviction to CPU RAM (swap) |

The problem PagedAttention solves is KV cache fragmentation during inference serving. Naive KV cache allocation reserves a contiguous HBM region for the maximum sequence length upfront, wasting most of it for short sequences. PagedAttention allocates fixed-size non-contiguous blocks and maps logical sequence positions to physical blocks via a block table — exactly like a hardware page table.

Same mental model. Different resource (HBM instead of DRAM), different trigger (sequence generation instead of array access), different backing store (CPU RAM instead of disk).
