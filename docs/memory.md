# GPU Memory: Distribution, Napkin Math, and OOM Debugging

## The problem we hit

On an A40 (44.43 GiB HBM), training crashed immediately with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.07 GiB.
Of the allocated memory 40.72 GiB is allocated by PyTorch
```

The model has 163M parameters. In bfloat16 that's ~326MB. So how is 40GB allocated
before a single training step completes? This document breaks it down.

---

## The five buckets of GPU memory during training

### 1. Model weights

The parameters of the model itself.

**Formula:** `num_params × bytes_per_param`

For our GPT-2 124M style model:
- 163M parameters
- bfloat16 = 2 bytes per parameter
- **326 MB**

In float32 (if you weren't using mixed precision) this would be 652MB. Still not
the culprit.

### 2. Optimizer state

AdamW maintains two moment tensors per parameter: the first moment (mean of
gradients) and the second moment (uncentered variance of gradients). These are
kept in **float32 regardless of model precision** — downcast optimizer state
causes instability.

**Formula:** `num_params × 4 bytes × 2 moments`

- 163M × 4 × 2 = **1.3 GB**

This is significant but not catastrophic. SGD has no optimizer state; AdamW
always costs this 2× parameter overhead.

### 3. Gradients

One gradient tensor per parameter, same dtype as the weights (bfloat16 in our
case, though PyTorch sometimes keeps them in float32 depending on autocast
configuration).

**Formula:** `num_params × bytes_per_param`

- 163M × 2 bytes = **326 MB** (bfloat16)
- 163M × 4 bytes = **652 MB** (float32)

### 4. Activations (the main culprit)

This is where the real memory goes. During the forward pass, PyTorch must retain
the intermediate activations at every layer so it can compute gradients during
the backward pass. For a transformer, the dominant term is the **attention
matrix**.

**Attention matrix per layer:**

```
batch_size × num_heads × seq_len × seq_len × bytes_per_element
```

With our config (batch=32, heads=12, seq=1024, bfloat16):
```
32 × 12 × 1024 × 1024 × 2 bytes = 805 MB per layer
```

With 12 layers:
```
805 MB × 12 = 9.7 GB just for attention matrices
```

Plus the residual stream, FFN activations, layer norm intermediates, etc. —
the full activation memory for a transformer forward pass is roughly:

```
~12 × seq_len × emb_dim × num_layers × batch_size × bytes
```

For our config:
```
12 × 1024 × 768 × 12 × 32 × 2 bytes ≈ 7.2 GB
```

Combined with attention matrices, **activations alone account for ~17 GB** at
batch size 32.

### 5. torch.compile / TorchInductor working memory

`torch.compile` fuses operations into optimized CUDA kernels via TorchInductor.
During the **first forward pass**, it:

1. Traces the computation graph
2. Compiles CUDA kernels (this is what makes the first step slow)
3. Allocates working buffers for fused operations

This can consume **10-15 GB of additional HBM** during compilation on the first
step, which is released afterward. But on an A40 with 44GB, the first-step spike
can tip you over into OOM before compilation finishes.

---

## Putting it together: our OOM napkin math

| Component | Memory |
|---|---|
| Model weights (bf16) | ~326 MB |
| Optimizer state (fp32) | ~1.3 GB |
| Gradients (bf16) | ~326 MB |
| Activations at batch=32 | ~17 GB |
| torch.compile working memory | ~10-15 GB (first step only) |
| CUDA runtime, fragmentation | ~2-3 GB |
| **Total (first step)** | **~32-37 GB + spike** |

The A40 has 44.43 GB. At batch=32 with torch.compile on the first step, we're
right at the limit — the 3.07 GiB allocation that failed was the system trying
to allocate one more activation buffer after everything else was in place.

The log confirms this:
```
40.72 GiB allocated by PyTorch
2.30 GiB reserved but unallocated
```
Reserved-but-unallocated is PyTorch's memory pool — it grabbed it from CUDA but
hasn't used it yet. The 3.07 GiB request couldn't fit in the remaining 1.08 GiB
of truly free HBM.

---

## Why activations scale with batch size

Weights and optimizer state are **fixed** regardless of batch size — you have the
same number of parameters whether you process 1 or 128 sequences at once.

Activations scale **linearly with batch size**. Every sequence in the batch needs
its own activation tensors at every layer. So:

- batch=4: activations ≈ 17 GB / 8 ≈ 2.1 GB
- batch=8: activations ≈ 17 GB / 4 ≈ 4.3 GB
- batch=16: activations ≈ 17 GB / 2 ≈ 8.5 GB
- batch=32: activations ≈ 17 GB

The attention matrix scales as `batch × seq²`, so it's particularly sensitive
to both batch size and sequence length simultaneously.

---

## The fix: reduce batch size

The immediate fix is reducing `--batch_size` in `scripts/train_fineweb.sh`.

**Rough safe values for A40 with torch.compile:**

| Batch size | Estimated activation memory | Likely outcome |
|---|---|---|
| 4 | ~2 GB | Comfortably fits |
| 8 | ~4 GB | Should fit |
| 16 | ~8 GB | Probably fits after compile cache warms |
| 32 | ~17 GB | OOM on first step |

Start at 4, verify it runs, then increase.

Also add to `train_fineweb.sh` before the python call:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This tells PyTorch's memory allocator to use expandable memory segments rather
than fixed-size blocks, reducing fragmentation. It won't conjure more memory but
it prevents the situation where you have 5 GB free in total but can't allocate
3 GB contiguously.

---

## Flash Attention: the architectural fix

Reducing batch size is a workaround. The real fix is **Flash Attention**
(`F.scaled_dot_product_attention`), which is one line in `model.py`.

The standard attention implementation stores the full `(batch, heads, seq, seq)`
attention matrix in HBM to use during the backward pass. For seq=1024, that's
the 805MB per layer we calculated above.

Flash Attention uses a **tiled, recomputation-based** approach: it computes
attention in small blocks that fit in GPU SRAM (shared memory), and during the
backward pass it recomputes the attention values rather than reading them from
HBM. This reduces attention memory from `O(seq²)` to `O(seq)`.

**Before (current implementation in model.py):**
```python
attn_scores = queries @ keys.transpose(2, 3)
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
context_vec = (attn_weights @ values).transpose(1, 2)
```

**After (Flash Attention):**
```python
context_vec = F.scaled_dot_product_attention(
    queries, keys, values, is_causal=True
).transpose(1, 2)
```

One line. Removes the need to store the attention matrix in HBM entirely. At
seq=1024 this saves ~9.7 GB at batch=32. With Flash Attention you can likely
run batch=32 comfortably on the A40.

Also: remove `self.mask` registration from `__init__` — `F.scaled_dot_product_attention`
handles the causal mask internally via `is_causal=True`.

---

## How memory scales with architecture changes

This is useful for ablation studies. Each architectural change has a predictable
memory impact:

**Sequence length** — attention matrix scales as `seq²`. Doubling seq from 1024
to 2048 quadruples attention memory. Flash Attention changes this to linear.

**Embedding dimension** — FFN memory scales as `batch × seq × emb_dim`. Doubling
emb_dim from 768 to 1536 doubles FFN activation memory.

**Number of layers** — activation memory scales linearly with layers. 24 layers
uses 2× the activation memory of 12 layers.

**Number of heads** — doesn't change total attention memory significantly (same
total computation, just split differently). GQA (Grouped Query Attention) reduces
KV cache memory during inference but has minimal training memory impact.

**Batch size** — all activation memory scales linearly. Weights and optimizer
state are unaffected.

**dtype** — switching from float32 to bfloat16 for weights and activations halves
their memory. Optimizer state stays float32 (the standard mixed precision recipe).

---

## Gradient checkpointing: trading compute for memory

If you want large batch sizes without Flash Attention, **gradient checkpointing**
(`torch.utils.checkpoint`) is another option. Instead of storing all activations
during the forward pass, it only stores activations at certain "checkpoint" layers
and recomputes intermediate activations during the backward pass.

This reduces activation memory by roughly `sqrt(num_layers)` at the cost of ~33%
more compute (one extra forward pass worth of computation per backward pass).

For our purposes Flash Attention is strictly better — it reduces the specific
memory that's killing us (attention matrices) without the compute overhead. But
gradient checkpointing is worth knowing for when you scale to larger models where
even Flash Attention doesn't fully solve the memory budget.

---

## Quick reference: memory budget for common configs on A40 (44GB)

All estimates assume bfloat16 weights/activations, float32 optimizer state,
**without** Flash Attention.

| Config | Params | Batch | Est. total memory | Fits? |
|---|---|---|---|---|
| GPT-2 124M, batch=4 | 124M | 4 | ~8 GB | ✅ |
| GPT-2 124M, batch=16 | 124M | 16 | ~22 GB | ✅ |
| GPT-2 124M, batch=32 | 124M | 32 | ~38 GB | ⚠️ tight |
| GPT-2 1.5B, batch=4 | 1.5B | 4 | ~28 GB | ✅ |
| GPT-2 1.5B, batch=8 | 1.5B | 8 | ~44 GB | ❌ |
| Llama 3 8B, batch=1 | 8B | 1 | ~40 GB | ⚠️ tight |

With Flash Attention, add roughly +2× headroom on batch size for the same model.
