# Training Optimizations Reference

Theory, implications, and implementation for the key hyperparameters and
scheduling decisions in GPT-style pretraining.

---

## 1. Learning Rate Warmup

### Theory

At the start of training, the model weights are random. The gradient signal is
noisy and high-variance — the model doesn't yet know which directions are
meaningful. If you start with a large learning rate immediately, early gradient
steps can push weights into bad regions of the loss landscape that are hard to
escape. The loss spikes and instability follows.

Warmup solves this by starting with a very small LR (near zero) and linearly
increasing it to the target LR over a fixed number of steps. By the time the
LR is at full magnitude, the model has seen enough data that the gradient
estimates are more reliable.

There's also a second-order effect: AdamW's second moment estimate (the `v`
term) takes time to accumulate. Early in training, `v` is near zero, which
makes the effective step size unpredictable. Warmup gives `v` time to build up
before the LR is large enough for that to matter.

### Implications

- Without warmup: frequent loss spikes early in training, possible divergence
  at high LRs
- With warmup: smoother early loss curve, can train with higher peak LR
- Typical warmup length: 1-2% of total steps (nanoGPT uses 2000/600000 = 0.33%)
- Grad norm at step 0 in your runs was ~1.3, hitting the clip threshold —
  this is exactly the warmup instability signal

### Implementation

Your `GPTConfig` has `warmup_iters` defined but it was never wired into the
optimizer. Here's the fix:

```python
# In train.py, replace fixed LR with a scheduler function

def get_lr(step: int, config) -> float:
    """
    Linear warmup then cosine decay to min_lr.
    Follows nanoGPT / GPT-3 paper schedule.
    """
    # 1) Linear warmup
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters

    # 2) After decay period, return min LR
    if step > config.lr_decay_iters:
        return config.min_lr

    # 3) Cosine decay between warmup and lr_decay_iters
    decay_ratio = (step - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ranges 1 → 0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# In the training loop:
lr = get_lr(step, config)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

Add to `GPTConfig`:
```python
warmup_iters: int = 2000
lr_decay_iters: int = 72479   # = total_steps for your run
min_lr: float = 6e-5          # = learning_rate / 10 is the standard rule
learning_rate: float = 6e-4
```

---

## 2. Cosine Learning Rate Decay

### Theory

After warmup, you want the LR to decrease over training. The intuition: early
in training you need large steps to explore the loss landscape. Late in
training, you're fine-tuning around a good basin and large steps will
overshoot. A lower LR at the end lets the optimizer settle into a sharper
minimum.

Why cosine specifically? A linear decay works, but cosine has a nice property:
it decays slowly at first (when you still want some exploration), then faster
in the middle, then slowly again at the end (gentle landing). The shape
matches the typical loss curve — steep improvement early, diminishing returns
late.

The standard recommendation (GPT-3 paper, Chinchilla) is to decay the LR over
the full training run, reaching `min_lr = learning_rate / 10` at the end.

### Implications

- Fixed LR (what your current runs use): loss plateaus earlier because the
  optimizer keeps overshooting the minimum
- Cosine decay: typically buys 0.05-0.15 loss reduction in the final third
  of training — meaningful at your scale
- The final loss improvement is most visible in the last 20-30% of training;
  your current curves flatten because there's no decay pulling them down

### Visualization of the schedule

```
LR
^
|    /‾‾‾‾‾‾‾‾‾‾\
|   /             \
|  /               \_____________ (min_lr)
| /
|/
+---------------------------------> step
  warmup  |                      |
          lr_decay_iters (= max_iters ideally)
```

The combined warmup + cosine is the standard schedule for every major LLM
(GPT-3, Llama, Mistral, FineWeb baseline). It's not optional at serious scale.

---

## 3. Weight Decay

### Theory

Weight decay adds an L2 penalty to the loss: `loss_total = loss_ce + λ * ||θ||²`.
This penalizes large weights, which acts as regularization — it prevents the
model from memorizing noise by keeping weights small and distributed.

In AdamW specifically, weight decay is applied directly to the weights rather
than through the gradient (the original "decoupled weight decay" insight from
the AdamW paper). This matters because Adam's gradient scaling would otherwise
make weight decay ineffective for parameters with small gradients.

**Critical nuance:** weight decay should NOT be applied to all parameters
equally. Applying it to bias terms and LayerNorm scale/shift parameters is
counterproductive — these parameters work best unrestricted. The standard
practice is to only decay 2D+ weight matrices (the actual projection weights),
not 1D parameters.

nanoGPT's `configure_optimizers` does this correctly:
```python
# Decay: weight matrices (2D tensors)
# No decay: biases, layernorm params (1D tensors)
decay_params = [p for p in params if p.dim() >= 2]
nodecay_params = [p for p in params if p.dim() < 2]
```

### Implications

- Too little weight decay: can overfit, especially on smaller datasets
- Too much: underfits, model can't express complex functions
- Standard value for LLMs: `1e-1` (0.1) — nanoGPT, GPT-3, Llama all use this
- Your current config: check what `weight_decay` is set to in your AdamW call

---

## 4. AdamW Beta2: 0.95 vs 0.999

### Theory

Beta2 controls the exponential moving average of the squared gradient (the
second moment `v`). It determines how quickly the optimizer "forgets" old
gradient information.

- `beta2=0.999` (PyTorch default): very long memory, ~1000 step window
- `beta2=0.95` (GPT-3 / nanoGPT): shorter memory, ~20 step window

For LLM pretraining, `0.95` is preferred because:
1. The gradient distribution shifts significantly over training as the model
   learns different things
2. A long memory means `v` is stale — it's still remembering gradient
   magnitudes from early training when the model was doing very different things
3. Shorter memory = `v` adapts faster = more accurate per-parameter step sizing

The practical effect is more stable late-stage training and slightly better
final loss. The difference is small but consistent.

### Implementation

```python
optimizer = torch.optim.AdamW(
    optim_groups,
    lr=config.learning_rate,
    betas=(0.9, 0.95),   # NOT (0.9, 0.999)
    weight_decay=config.weight_decay,
    fused=True,           # ~10% speedup on CUDA if available
)
```

---

## 5. Gradient Clipping

### Theory

Gradient clipping caps the global L2 norm of all gradients before the
optimizer step. If `||∇θ|| > clip_value`, all gradients are scaled down
proportionally so the norm equals `clip_value`.

This prevents "gradient explosions" — rare but catastrophic events where a
single bad batch produces enormous gradients that blow up the weights. They
appear as sudden loss spikes and are especially common early in training (when
the loss surface is steep) and on long sequences.

The key insight: clipping is not "aggressive regularization," it's a safety
net. You don't want it firing constantly — if your `grad_norm` is consistently
near or above the clip threshold, your LR is probably too high.

### Reading your grad_norm charts

Your run shows grad_norm starting at ~1.3 (hitting the clip=1.0 threshold),
then settling to ~0.33-0.4. This is a healthy pattern:
- Early spikes: normal, this is where warmup would help
- Stable 0.3-0.4: optimizer is in a smooth region, clipping not firing
- Occasional spikes to ~0.8-1.0: fine, normal batch variance

If grad_norm were consistently above 1.0: lower LR or increase warmup
If grad_norm were consistently below 0.1: LR might be too low, training slowly

### Implementation

```python
# After loss.backward(), before optimizer.step()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Log it (you're already doing this)
wandb.log({"train/grad_norm": grad_norm})
```

---

## 6. Batch Size and Gradient Accumulation

### Theory

Large batch sizes stabilize gradient estimates (lower variance per step) but
require more memory and give fewer gradient updates per token seen. Small
batches give noisier but more frequent updates.

The standard finding (from large-scale training experiments): there's a
"critical batch size" beyond which increasing batch size gives diminishing
returns in sample efficiency. For GPT-2 scale, this is roughly 0.5M tokens
per step.

Gradient accumulation lets you simulate large batches on limited VRAM by
accumulating gradients over multiple micro-batches before calling
`optimizer.step()`. The effective batch size is:
```
effective_batch = micro_batch × seq_len × accum_steps × n_gpus
```

nanoGPT targets ~0.5M tokens per step:
`12 × 1024 × 5 × 8 GPUs = 491,520 ≈ 0.5M`

Your current run: `64 × 1024 × 1 × 1 GPU = 65,536 ≈ 0.064M`

This is ~8× smaller than nanoGPT's effective batch. You could add gradient
accumulation to close some of that gap without changing VRAM usage:

```python
# To get ~0.5M tokens/step on one A40:
# 64 (batch) × 1024 (seq) × 8 (accum) = 524,288
gradient_accumulation_steps = 8

# In the training loop:
for micro_step in range(gradient_accumulation_steps):
    with ctx:
        logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps
    loss.backward()

# Only step after accumulating all micro-batches
optimizer.step()
optimizer.zero_grad()
```

---

## 7. `bias=False` in Linear and LayerNorm

### Theory

Karpathy's nanoGPT (following GPT-2 more faithfully) sets `bias=False` in all
Linear layers and removes the `shift` parameter from LayerNorm. This is a
minor architecture choice, not a critical one.

The argument for removing biases: in a deep network with pre-LN, biases are
largely redundant — the LayerNorm can absorb any constant offset. Removing
them reduces parameters slightly (124M vs your 163M) and marginally reduces
overfitting risk.

In practice the difference is negligible. Your model is not wrong for using
`bias=True`, it's just a different (and equally valid) default.

---

## What to actually implement next

In rough priority order for your next run:

1. **LR warmup + cosine decay** — highest impact, easy to add, directly
   addresses the early instability you can see in your grad_norm chart

2. **`beta2=0.95`** — one line change, worth doing alongside #1

3. **Gradient accumulation to ~0.5M effective batch** — brings you closer
   to nanoGPT's training regime, marginal improvement but principled

4. **Verify weight decay is applied selectively** — check that your AdamW
   call separates 2D from 1D params

The expected improvement from #1+#2 on a full 3.26B token run (Chinchilla
optimal for 163M): roughly 3.4-3.5 val loss instead of ~3.6-3.7. Not dramatic
but measurable, and more importantly, you'll have a correctly configured
training loop going forward.
