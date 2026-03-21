# GPU Memory & Training: Napkin Math Reference

A reference for quick back-of-envelope calculations when configuring LLM training
runs. All formulas are approximate but accurate enough for planning and debugging.

---

## Part 1: Memory Buckets

Every training run has five memory buckets. Know these cold.

### 1. Model Weights

```
memory = num_params × bytes_per_dtype
```

| dtype | bytes |
|-------|-------|
| float32 | 4 |
| bfloat16 | 2 |
| float16 | 2 |
| int8 | 1 |

**Examples:**
- 163M params × bf16 = 163M × 2 = **326 MB**
- 163M params × fp32 = 163M × 4 = **652 MB**
- 7B params × bf16 = 7B × 2 = **14 GB**
- 70B params × bf16 = 70B × 2 = **140 GB** (needs multiple GPUs)

### 2. Optimizer State (AdamW)

AdamW stores two moment tensors per parameter, both in **float32 regardless of
model dtype**.

```
memory = num_params × 4 bytes × 2 moments = num_params × 8 bytes
```

**Examples:**
- 163M params → 163M × 8 = **1.3 GB**
- 7B params → 7B × 8 = **56 GB**

**Note:** SGD has no optimizer state. AdamW always costs 2× parameter memory in fp32.

### 3. Gradients

Same shape as weights, usually same dtype as weights.

```
memory = num_params × bytes_per_dtype
```

- 163M params × bf16 = **326 MB**

### 4. Activations (the dominant term)

This is what kills you. Scales linearly with batch size.

**Residual stream (all layers):**
```
memory ≈ batch_size × seq_len × emb_dim × num_layers × bytes × 2
```
(the ×2 is for both forward and backward pass storage)

**Attention matrix (without Flash Attention):**
```
memory = batch_size × num_heads × seq_len × seq_len × bytes_per_element
         per layer
```

For GPT-2 124M style (batch=32, seq=1024, 12 heads, 12 layers, bf16):
```
32 × 12 × 1024 × 1024 × 2 bytes = 805 MB per layer
805 MB × 12 layers = 9.7 GB  ← just attention matrices
```

**With Flash Attention:** attention matrices never materialize in HBM.
Memory drops from O(seq²) to O(seq). This is why FA is not optional at scale.

**Total activation estimate (rough):**
```
~1.5 GB per sample in the batch (GPT-2 124M scale, without Flash Attention)
~0.3 GB per sample in the batch (GPT-2 124M scale, with Flash Attention)
```

### 5. Framework Overhead

torch.compile, CUDA runtime, memory fragmentation: budget **2-3 GB** fixed.

---

## Part 2: The Full Budget Formula

```
total_VRAM = weights + optimizer + gradients + (activation_per_sample × batch_size) + overhead
```

For our model (163M params, bf16, AdamW, with Flash Attention):
```
= 0.326 + 1.3 + 0.326 + (0.3 × batch_size) + 2.5  GB
= 4.45 + (0.3 × batch_size)  GB
```

| batch_size | estimated VRAM | % of A40 (45GB) |
|------------|---------------|-----------------|
| 4 | ~5.7 GB | 13% |
| 8 | ~6.9 GB | 15% |
| 16 | ~9.3 GB | 21% |
| 32 | ~14.1 GB | 31% |
| 64 | ~23.7 GB | 53% |
| 128 | ~42.9 GB | 95% ⚠️ |

Without Flash Attention, add ~9.7 GB to every row.

---

## Part 3: Scaling Laws

### Chinchilla Optimal Token Count

For a given model size, the optimal number of training tokens is approximately:

```
optimal_tokens ≈ 20 × num_params
```

| Model size | Chinchilla-optimal tokens |
|-----------|--------------------------|
| 163M | ~3.3B tokens |
| 1B | ~20B tokens |
| 7B | ~140B tokens |
| 70B | ~1.4T tokens |

Our setup: 163M model, 2.5B token dataset → slightly under Chinchilla-optimal.
Close enough for learning purposes.

### Sequence Length and Quadratic Scaling

Attention memory (without Flash Attention) scales as seq²:

```
halve seq_len → attention memory drops 4×
double seq_len → attention memory increases 4×
```

Compare to batch size, which scales linearly:
```
halve batch_size → activation memory drops ~2×
double batch_size → activation memory increases ~2×
```

**Sequence length is a much more powerful lever on memory than batch size.**
This is why Flash Attention was urgent: it changes seq² to seq.

---

## Part 4: Throughput

### Tokens per Step

```
tokens_per_step = batch_size × context_length
```

| batch_size | context_length | tokens/step |
|-----------|---------------|-------------|
| 4 | 1024 | 4,096 |
| 16 | 1024 | 16,384 |
| 32 | 1024 | 32,768 |
| 64 | 1024 | 65,536 |

### Total Training Steps

```
train_tokens = total_tokens × (1 - val_ratio)
total_steps = train_tokens / tokens_per_step
```

For our setup (2.5B tokens, val_ratio=0.05):
```
train_tokens = 2.5B × 0.95 = 2.375B
```

| batch_size | total_steps |
|-----------|-------------|
| 4 | ~579,833 |
| 16 | ~144,958 |
| 32 | ~72,479 |
| 64 | ~36,240 |

### Time to Complete an Epoch

```
time = total_tokens / tokens_per_sec
```

At observed throughputs on A40:

| config | tokens/sec | epoch time | cost @ $0.40/hr |
|--------|-----------|------------|-----------------|
| bs=4, no FA | ~30,000 | ~23 hours | ~$9.20 |
| bs=4, FA | ~38,000 | ~18 hours | ~$7.20 |
| bs=32, FA (est.) | ~150,000 | ~4.5 hours | ~$1.80 |
| bs=64, FA (est.) | ~250,000 | ~2.5 hours | ~$1.00 |

### Learning Rate Scaling with Batch Size

When you increase batch size, you need to scale LR to maintain equivalent training:

**Linear scaling rule (aggressive):**
```
new_lr = base_lr × (new_batch / base_batch)
```

**Square root scaling rule (conservative, more stable):**
```
new_lr = base_lr × sqrt(new_batch / base_batch)
```

Example: base_lr=4e-4 at batch=4, scaling to batch=64:
```
linear: 4e-4 × (64/4) = 4e-4 × 16 = 6.4e-3  (aggressive)
sqrt:   4e-4 × sqrt(16) = 4e-4 × 4 = 1.6e-3  (conservative)
```

In practice: for short learning runs, keeping lr=4e-4 is fine. Watch grad_norm
in W&B — if it's consistently hitting the clip threshold (1.0), LR may be too high.

---

## Part 5: Checkpoint Sizes

```
checkpoint_size = weights + optimizer_state
               = (num_params × weight_dtype) + (num_params × 8)
```

| Model | weight dtype | checkpoint size |
|-------|-------------|----------------|
| 163M | fp32 | 652MB + 1.3GB = ~1.95 GB |
| 163M | bf16 | 326MB + 1.3GB = ~1.63 GB |
| 7B | bf16 | 14GB + 56GB = ~70 GB |

**Checkpoint rotation math:**
If you save every N steps and keep K checkpoints:
```
max_disk_usage = K × checkpoint_size
```

Our setup: K=2, checkpoint_size=1.63GB → **~3.3 GB max** for checkpoints.

---

## Part 6: Dataset Storage

```
dataset_size_bytes = num_tokens × bytes_per_token
```

| dtype | bytes/token | 2.5B tokens | 10B tokens |
|-------|------------|-------------|------------|
| int32 (standard) | 4 | 10 GB | 40 GB |
| int16 (small vocab) | 2 | 5 GB | 20 GB |
| int64 | 8 | 20 GB | 80 GB |

Note: GPT-2 vocab is 50,257 → needs at least 17 bits → int32 is correct.
int16 would overflow (max 32,767 < 50,257).

---

## Fill-in-the-Blank Exercises

Test yourself. Answers at the bottom.

**Q1.** Your model has 500M parameters trained in bfloat16 with AdamW.
How much VRAM does it need just for weights + optimizer state?

**Q2.** You're training GPT-2 124M (163M params) at batch=32 with context_length=1024
**without** Flash Attention. The A40 has 45GB. Will you OOM? Show your work.

**Q3.** You double your sequence length from 1024 to 2048 without Flash Attention.
How does attention memory change?

**Q4.** You have 7B tokens of training data and want to train a Chinchilla-optimal
model. Approximately how many parameters should it have?

**Q5.** Your training run is doing 50,000 tokens/sec. Your dataset has 5B tokens
with val_ratio=0.1. How long will a full epoch take?

**Q6.** You want to save checkpoints in bf16 for a 163M param model. How much
disk space does each checkpoint take? How much for 3 checkpoints?

**Q7.** You're scaling from batch=8 to batch=64 (8× increase) with base_lr=2e-4.
What's your new LR using the square root rule?

**Q8.** You have a 1B param model in bf16. A40 has 45GB. What's the maximum
batch size you can use, assuming Flash Attention and ~0.5 GB per sample activation cost?

**Q9.** Your model has 163M params. You're training on 500M tokens. Are you
over or under Chinchilla-optimal, and by roughly how much?

**Q10.** At batch=64 and context_length=1024, how many training steps will you
take over 2.375B training tokens?

---

## Answers

**A1.**
- Weights: 500M × 2 bytes = 1.0 GB
- Optimizer: 500M × 8 bytes = 4.0 GB
- Total: **5.0 GB**

**A2.**
- Fixed: weights (326MB) + optimizer (1.3GB) + gradients (326MB) + overhead (2.5GB) ≈ 4.5GB
- Attention matrices: 32 × 12 × 1024 × 1024 × 2 bytes × 12 layers = 9.7GB
- Other activations: ~5GB
- Total: ~19GB
- **No OOM** — fits in 45GB, but tight enough that torch.compile's first-step spike
  can push you over. (This is exactly what happened in our run.)

**A3.**
- Attention scales as seq². Doubling seq: 2² = 4×
- **Attention memory increases 4×** (from ~9.7GB to ~38.8GB for our model at batch=32)

**A4.**
- optimal_params = total_tokens / 20 = 7B / 20 = **350M parameters**

**A5.**
- train_tokens = 5B × 0.9 = 4.5B
- time = 4,500,000,000 / 50,000 = 90,000 seconds = **25 hours**

**A6.**
- Weights (bf16): 163M × 2 = 326 MB
- Optimizer (fp32): 163M × 8 = 1,304 MB
- Per checkpoint: **~1.63 GB**
- 3 checkpoints: **~4.9 GB**

**A7.**
- sqrt(64/8) = sqrt(8) = 2.83
- new_lr = 2e-4 × 2.83 = **~5.7e-4**

**A8.**
- Fixed overhead: 2B × 2 (bf16) + 2B × 8 (optimizer) + 2GB overhead ≈ 12 GB
- Remaining for activations: 45 - 12 = 33 GB
- max_batch = 33 / 0.5 = **66** → batch=64 is safe

**A9.**
- Chinchilla-optimal for 163M params: 163M × 20 = 3.26B tokens
- You have 500M tokens
- **Under by ~6.5×** — significantly undertrained, model has more capacity than
  the data can use

**A10.**
- tokens_per_step = 64 × 1024 = 65,536
- total_steps = 2,375,000,000 / 65,536 = **~36,240 steps**
