# OpenAI Parameter Golf — Architecture Notes

Reference: [openai/parameter-golf](https://github.com/openai/parameter-golf)

The parameter golf competition asks participants to achieve the lowest bits-per-byte
(BPB) on a FineWeb-Edu validation set subject to a hard parameter count cap. The
baseline `train_gpt.py` script is a useful reference point for modern small-model
training practice — but some of its choices are competition-specific and shouldn't
be blindly copied.

---

## What the competition baseline does

**Model shape (default config):**
- 9 transformer blocks, width 512, 8 attention heads
- 4 KV heads (GQA), 2× MLP expansion
- Vocab size 1024 (custom SentencePiece BPE)
- Sequence length 1024, tied embeddings

**Architecture choices:**
- RoPE positional encoding (no learned position embeddings)
- RMSNorm (Pre-LN)
- No biases on any linear layers
- Tied input/output embeddings
- Flash Attention
- `logit_softcap = 30.0` (Gemma 2 technique, see below)
- `qk_gain_init = 1.5` (higher-than-default Q/K initialization gain)
- GQA (4 KV heads vs 8 Q heads)
- bfloat16 throughout

**Optimizer:**
- Muon for 2D matrix parameters (W_query, W_key, W_value, FFN weights, etc.)
- Adam for 1D/scalar parameters (RMSNorm scales, embeddings, output head)
- Separate learning rates per parameter class: `embed_lr`, `head_lr`,
  `matrix_lr`, `scalar_lr`
- Very short warmup (20 steps), linear warmdown over last 1200 steps

**Data:**
- FineWeb-Edu `sample-10BT` (same as our setup)
- Custom 1024-token vocabulary (competition artifact)
- Pre-sharded binary files, multi-GPU DDP

---

## Comparison to our model

| Feature | Ours | Parameter Golf |
|---|---|---|
| Positional encoding | Learned absolute | RoPE |
| Normalization | RMSNorm (Pre-LN) | RMSNorm (Pre-LN) |
| Activation | SwiGLU | SwiGLU (implied by 2× MLP) |
| Attention | Flash Attention | Flash Attention |
| Tied embeddings | ✅ | ✅ |
| Biases | ✅ removed | ✅ removed |
| Vocab size | 50,257 (GPT-2 BPE) | 1,024 (custom) |
| Optimizer | AdamW | Muon + Adam |
| Logit softcap | ❌ | ✅ (30.0) |
| GQA | ❌ | ✅ (4 KV heads) |
| Distributed | Single GPU | Multi-GPU DDP |

---

## Competition-specific choices to ignore

**Tiny vocabulary (1024 tokens).** Fewer vocab tokens means fewer embedding
parameters, which means more of the parameter budget goes to transformer weights.
This is a direct optimization for the BPB scoring metric. For any real use case
(SFT, inference, HF Hub) you want the standard GPT-2 BPE 50,257 vocab.

**Custom tokenizer.** Related to the above — their scoring is tokenizer-agnostic
(BPB rather than loss), which lets them freely shrink the vocabulary. Not relevant
outside the competition context.

---

## Things worth adopting

### Logit softcap

A technique from Gemma 2. Instead of passing raw logits directly to the loss:

```python
# Standard
loss = F.cross_entropy(logits, targets)

# With softcap
cap = 30.0
logits = cap * torch.tanh(logits / cap)
loss = F.cross_entropy(logits, targets)
```

This prevents logits from growing to extreme values during training. The `tanh`
squashes anything beyond ±cap smoothly rather than letting individual logits
dominate the softmax. Useful for training stability, especially later in runs when
the model has started to form confident predictions. Simple to add, no parameter
cost.

### Muon optimizer

The most significant algorithmic difference. Muon applies Newton-Schulz
orthogonalization to 2D gradient matrices before the weight update:

```python
# Conceptually: instead of param -= lr * grad
# Muon does:    param -= lr * orthogonalize(grad)
```

The Newton-Schulz iteration approximates the matrix square root of `G @ G.T`,
effectively normalizing the gradient to have orthonormal rows. The motivation is
that weight matrices live on a Riemannian manifold and gradient descent should
respect that geometry — the standard Euclidean update is suboptimal.

In practice Muon converges faster than AdamW at the same compute budget. The
tradeoff is complexity: you need separate optimizer instances for matrix vs scalar
parameters, and the orthogonalization adds compute (though it's fast — typically
5 Newton-Schulz steps).

Reference: [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)

Muon is a strong candidate for a future ablation — swap AdamW for Muon on matrix
params, keep Adam on scalars/embeddings, compare loss curves at matched compute.

### QK gain initialization

They initialize Q/K projection weights with `gain=1.5` rather than the PyTorch
default (Kaiming uniform). This affects how quickly attention patterns sharpen
early in training. Low cost to try, but requires measuring the effect carefully —
it interacts with learning rate and warmup length.

---

## Things not worth adopting at our scale

**GQA.** Grouped Query Attention reduces the KV cache size during inference by
sharing K/V projections across groups of Q heads. At 125M parameters you will
never be KV cache-constrained at inference. The training loss improvement is
marginal and it complicates the attention implementation.

**Multi-GPU DDP.** Not relevant for single-GPU RunPod runs. Worth revisiting if
scaling to >1B params or running parallel ablations.

---

## Planned ablations

In rough priority order:

1. **Logit softcap** — add to current model, measure stability improvement
2. **RoPE** — replace learned absolute position embeddings, measure loss delta
   vs baseline at matched token count
3. **Muon** — replace AdamW for matrix params, compare convergence curve
4. **QK gain init** — low priority, interacts with other hyperparameters
