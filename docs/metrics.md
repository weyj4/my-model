# Training Metrics & GPU Performance Analysis

Reference for reasoning about model training health, learning progress,
and hardware utilization. Based on the flash-bs64 W&B run (163M GPT-2,
FineWeb, A40, batch=64, step 41800).

---

## Part 1 — Reading the Loss Curves

### What loss actually measures

Cross-entropy loss is the negative log probability the model assigns to the
correct next token:

```
loss = -log p(correct_token)
```

A loss of 3.73 means the model assigns probability `e^(-3.73) ≈ 2.4%` to
the correct token on average. Two useful reference points:

```
Uniform random over 50,257 tokens:  ln(50257) ≈ 10.8   (where you start)
English lower bound (Shannon):      ~1.0-1.2 nats/token (theoretical floor)
GPT-2 124M on WebText:             ~3.1 nats
Your model at step 41800:          ~3.73 nats
```

The steep early drop (steps 0-50) is the model learning basic token
statistics — frequency distributions, common bigrams, obvious syntax. The
long flat tail is where the model is learning higher-order structure, and
where LR scheduling matters most.

### Perplexity

`perplexity = e^loss`

At loss 3.73 → perplexity ≈ 41.6. Interpretation: the model is "as
uncertain as if choosing uniformly among ~42 plausible options" at each
token position. After SFT this number will drop substantially because the
model learns to stay on-topic.

### Bits per byte (BPB)

`BPB = loss / ln(2)`

Information-theoretic framing: how many bits to encode each byte of text.
Lower = better compression = better language model. Shannon entropy of
English is ~1.0-1.2 bits/byte. Your BPB ≈ 1.41 at step 41800, converging
toward the theoretical floor for this model size.

### Train/val gap

Your run: train loss 3.728, val loss 3.772 — gap of 0.044 nats. This is
extremely tight. It means:

- No meaningful overfitting
- You are firmly in the underfitting regime at 1.4B tokens
- Training longer would continue to improve val loss
- The model has spare capacity it hasn't used yet

At Chinchilla optimal (3.26B tokens for 163M params) you'd expect the gap
to widen slightly as the model begins to memorize training examples.

### What the loss curve shape tells you about the optimizer

```
Steep drop (steps 0-50):     learning basic statistics, large gradient signal
Gradual descent (50-300):    learning syntax, common patterns
Flat tail (300+):            diminishing returns from fixed LR

If you see: sudden spike    → bad batch, LR too high, or numerical instability
If you see: plateau then rise → overfitting (not your case)
If you see: monotonic flat  → fixed LR is the ceiling (your case)
```

The flat tail in your run is the direct signature of no cosine decay. With
warmup + cosine decay the curve would continue bending downward in the
final third of training, typically buying 0.05-0.15 nats at this scale.

---

## Part 2 — Gradient Norm

### What it measures

`grad_norm = ||∇θ||₂` — the L2 norm of the concatenated gradient vector
across all parameters. Clipped at 1.0 in your config.

### Reading your chart

```
Steps 0-10:    ~1.2-1.3, hitting clip threshold   → compile warmup + no LR warmup
Steps 10-400:  ~0.35-0.45 stable                  → optimizer in smooth loss bowl
Occasional spikes to ~0.8:                         → normal batch variance
```

The early clipping is exactly what LR warmup prevents — starting with a
large LR before the second moment estimate `v` has accumulated causes large
initial steps that destabilize the loss surface.

### Diagnostic rules

| grad_norm pattern | Diagnosis |
|---|---|
| Stable below clip | Healthy — optimizer making steady progress |
| Consistently at clip | LR too high, or warmup needed |
| Rising over training | Possible instability — reduce LR |
| Near zero | LR too low or vanishing gradients |
| Single large spike then recovery | Bad batch — normal, ignore |
| Multiple large spikes | Data quality issue or LR too high |

---

## Part 3 — Throughput Metrics

### tokens/sec: 62,199

This is your primary GPU efficiency metric. It measures end-to-end
throughput: forward pass + backward pass + optimizer step + logging,
averaged over eval intervals.

The shape of your tokens/sec curve (rapid ramp from ~5k to ~60k in the
first 10 steps) is `torch.compile` warming up — TorchInductor is compiling
CUDA kernels for your specific model shape and caching them.

### Estimating Model FLOP Utilization (MFU)

MFU = actual FLOP rate / peak hardware FLOP rate

```
FLOPs per forward pass ≈ 2 × context_tokens × num_parameters
= 2 × 1024 × 163,000,000
= 333.8 GFLOPs per sample

FLOPs per step (batch=64):
= 333.8 × 64 = 21.4 TFLOPs (forward)
× 3 for forward+backward+optimizer ≈ 64 TFLOPs per step

Steps per second = 62,199 / (64 × 1024) = 0.946 steps/sec

Actual FLOP rate = 64T × 0.946 ≈ 60.5 TFLOPS

A40 peak (bf16) = 149.7 TFLOPS
MFU = 60.5 / 149.7 ≈ 40%
```

40% MFU is genuinely good for single-GPU training. 15-50% is the typical
range; hitting 50%+ usually requires multi-GPU with careful overlap of
compute and communication.

### tokens_seen: 1.37B

Progress toward token budget. Key reference points:

```
Chinchilla optimal for 163M model: 20 × 163M = 3.26B tokens
Your current run at termination:   1.4B tokens = 43% of optimal
FineWeb-Edu run target:            3.26B tokens = 100% of optimal
```

---

## Part 4 — GPU Hardware Metrics

### Power: 305W sustained, ~95% power utilization

The A40 has a 300W TDP. Running at 305W means the GPU is drawing slightly
above rated power — RunPod pods sometimes allow brief headroom. Sustained
high power means the GPU is doing real compute, not sitting idle waiting
for data.

Power drops (visible at ~step 50 and ~step 200) correspond to evaluation
periods where you call `evaluate_model()` — inference-only, no backward
pass, lower power draw.

### GPU Utilization: ~99%

The SM (Streaming Multiprocessor) utilization metric. At 99%, the CUDA
cores are almost never idle. This confirms you are **compute-bound**, not
data-loading-bound or memory-bandwidth-bound.

If this were below 80%, the first thing to check would be DataLoader
bottlenecks (`num_workers`, prefetching) or excessive Python overhead in
the training loop.

### GPU Memory Allocated: ~25GB (60% of 48GB)

```
Memory breakdown:
  Model weights (bf16):     163M × 2 bytes = 326 MB
  Optimizer state (fp32):   163M × 8 bytes = 1.3 GB
  Activations (batch=64):   ~8-12 GB (Flash Attention reduces this significantly)
  Gradient buffers:         ~326 MB
  torch.compile cache:      ~2-3 GB
  Total observed:           ~25 GB / 48 GB = 52%
```

The 60% memory utilization with headroom available means batch=64 is not
your memory ceiling. Batch=128 would fit. However since you're already
compute-bound and achieving good MFU, larger batch primarily helps
gradient variance (smoother updates) not throughput.

### SM Clock: 1500-1700 MHz oscillating

The A40 boosts clock speed when thermal headroom allows. The oscillation
is the GPU's boost algorithm dynamically adjusting. Stable oscillation in
a narrow band = healthy thermal management, no throttling.

### GPU Time Spent Accessing Memory: ~80%

This metric is often misread. It does NOT mean you're memory-bandwidth-
bound. It means that during the fraction of time the SM is active, ~80%
of cycles involve memory accesses. This is normal for transformer training
where attention and FFN layers alternate between compute-heavy matmuls and
memory-heavy activation reads/writes.

Flash Attention specifically addresses this by keeping the attention
softmax computation in L2/SRAM rather than writing intermediate results
back to HBM, which is why it shows up as a speedup even when you're not
OOM.

---

## Part 5 — The Roofline Model

### The framework

Every GPU operation is bounded by one of two ceilings:

```
Compute ceiling:  peak TFLOPS (how fast can we multiply?)
Memory ceiling:   peak GB/s bandwidth (how fast can we read weights?)

Arithmetic intensity = FLOPs performed / bytes read from HBM
```

The roofline plots arithmetic intensity on the x-axis. Below the ridge
point → memory-bandwidth-bound. Above → compute-bound.

```
A40 ridge point:
= peak TFLOPS / peak bandwidth
= 149.7 TFLOPS / 0.696 TB/s
= 215 FLOPs/byte

Operations above 215 FLOPs/byte → compute-bound
Operations below 215 FLOPs/byte → memory-bound
```

### Where your model's operations land

**Matrix multiplications (attention projections, FFN):**
For a [768, 768] matmul with batch=64, seq=1024:
```
FLOPs = 2 × 64 × 1024 × 768 × 768 ≈ 78 GFLOPs
Bytes = (input + weight + output) × dtype_size
      = (64×1024×768 + 768×768 + 64×1024×768) × 2 bytes
      ≈ 400 MB
Arithmetic intensity ≈ 78G / 0.4G = ~195 FLOPs/byte
```
Just below the ridge point — near compute-bound. This is why matmuls
dominate training time and why GPU vendors optimize tensor cores for them.

**Attention score computation (without Flash Attention):**
```
FLOPs = 2 × batch × heads × seq × seq × head_dim
      = 2 × 64 × 12 × 1024 × 1024 × 64 ≈ 103 GFLOPs

Bytes = attention matrix HBM writes/reads
      = batch × heads × seq × seq × 2 bytes
      = 64 × 12 × 1024 × 1024 × 2 = 1.6 GB

Arithmetic intensity = 103G / 1.6G ≈ 64 FLOPs/byte
```
Well below the ridge point → **memory-bandwidth-bound**. This is exactly
why Flash Attention exists: by computing attention in tiles that fit in
SRAM, it eliminates the 1.6 GB HBM round-trip entirely. The FLOPs are
the same but the bytes accessed drops to ~400 MB → intensity ~257 → now
compute-bound.

**Elementwise ops (GELU, LayerNorm, residuals):**
Low arithmetic intensity (~1-5 FLOPs/byte) → heavily memory-bound.
These are fast because they're small, but they can't be made faster by
adding more compute — they're waiting for memory.

### What this means for architecture decisions

| Architecture choice | Roofline impact |
|---|---|
| Larger batch size | Better matmul utilization, amortizes weight reads |
| Larger model (more params) | Higher arithmetic intensity per layer |
| Longer sequences | Attention memory cost scales O(T²) → pushes toward memory-bound |
| Flash Attention | Moves attention from memory-bound to compute-bound |
| GQA (grouped query attention) | Reduces KV cache size, reduces memory pressure |
| torch.compile | Fuses elementwise ops, reduces kernel launch overhead |

### Practical rule of thumb

For transformer training at your scale:
- **If GPU utilization < 80%**: data pipeline bottleneck or Python overhead
- **If memory utilization > 90%**: reduce batch, add gradient checkpointing
- **If tokens/sec low despite high utilization**: memory-bandwidth-bound, check sequence length and attention implementation
- **If MFU < 20%**: something is wrong — usually optimizer overhead or too many small ops

---

## Part 6 — Architecture → Loss Floor Relationship

### Capacity and the loss floor

A model has a **loss floor** — the minimum achievable loss given infinite
data and perfect optimization. This floor is set by the architecture's
capacity to represent language patterns, which scales roughly with:

```
capacity ∝ num_parameters × effective_depth
```

For your 163M model:
- Loss floor on FineWeb: approximately 3.1-3.3 nats (estimated)
- To get below ~3.0: need more parameters or better architecture
- GPT-2 124M on WebText achieves ~3.0: comparable size, curated English data

The key architectural parameters and what they control:

**`emb_dim = 768`** — the width of the model. Controls how much information
each token representation can hold. Scaling this up is the most direct way
to increase capacity per layer.

**`n_layers = 12`** — the depth. Each layer can refine the representation.
More layers = model can compose more abstract patterns, but diminishing
returns past a point. Residual connections make this tractable.

**`n_heads = 12, head_dim = 64`** — each head attends to different aspects
of context. Head dimension 64 is small — each head has limited expressivity
individually. The diversity across 12 heads compensates.

**`context_length = 1024`** — how far back the model can look. Longer
context = more information available = lower loss on long-range dependencies.
Also increases attention memory cost O(T²).

**`ffn_dim = 3072 (4× emb_dim)`** — the FFN stores factual associations.
The "memory" of the transformer. Scaling this relative to emb_dim is an
active research direction (MoE models take this to an extreme).

### What SFT/DPO will do to these metrics

After SFT on OpenHermes:
- **val/loss will drop** significantly (3.77 → probably ~2.0-2.5) because
  the model is now evaluated on instruction-following patterns it was
  trained on
- **Perplexity will drop** from ~43 to ~8-15
- **Grad norm will be lower** (fine-tuning with small LR)
- **tokens/sec will be similar** (same architecture, LoRA adds minimal overhead)

The loss numbers are no longer comparable to pretraining loss after SFT —
they're measuring different distributions. The right eval post-SFT is
qualitative generation quality and benchmark scores (HellaSwag, MMLU),
not raw loss.

---

## Summary: The Three Questions to Ask About Any Training Run

**1. Is the model learning?**
→ Is val/loss decreasing? Is the train/val gap reasonable (not growing)?

**2. Is the GPU being used efficiently?**
→ Is utilization >90%? Is MFU reasonable for single-GPU (>20%)?
→ Is memory utilization leaving headroom or hitting the ceiling?

**3. Is the optimization healthy?**
→ Is grad_norm stable and below the clip threshold?
→ Is the LR schedule appropriate for the token budget?
→ Is the train/val gap consistent with the token budget vs. Chinchilla optimal?
