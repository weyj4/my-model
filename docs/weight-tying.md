# Weight Tying, Initialization, and Why They're Entangled

## Weight Tying

Weight tying means making the output projection head (`out_head`) share the same weight matrix as the token embedding (`tok_emb`):

```python
self.out_head.weight = self.tok_emb.weight  # one tensor, two roles
```

This is a well-established technique from [Press & Wolf 2017](https://arxiv.org/abs/1608.05859). The intuition: the embedding matrix maps token IDs → vectors, and the output head maps vectors → logits over token IDs. These are conceptually inverse operations over the same vocabulary, so it makes sense to share parameters. It reduces total parameter count (for GPT-2 small, the embedding matrix is 50257 × 768 ≈ 38.6M params — not trivial) and acts as a regularizer.

## The Default Init Problem

PyTorch initializes `nn.Embedding` with **N(0, 1)** — standard normal, std = 1.0.

PyTorch initializes `nn.Linear` with **Kaiming uniform**, which for a 768-dim layer gives values roughly in [-0.036, +0.036].

Without weight tying, these live in separate worlds:
- The embedding produces vectors with std ≈ 1.0, but they pass through 12 transformer blocks whose Linear layers have small weights (std ≈ 0.036), so activations get scaled down naturally.
- The output head has its *own* small Kaiming-init weights, so the final logits end up with a reasonable std, and cross-entropy loss lands near ln(50257) ≈ 10.8 — the expected loss for a uniform distribution over the vocabulary.

**With weight tying, the output head inherits the embedding's N(0, 1) weights.** Now the final logits are computed as:

```
logits = x @ W_emb.T    where W_emb ~ N(0, 1)
```

Even if `x` has std ≈ 1 after the final RMSNorm, the matmul across 768 dimensions produces logits with std ≈ √768 ≈ 27.7. Cross-entropy on logits that large is catastrophic — that's where the ~530 nats comes from. The softmax is so saturated that the correct token gets almost zero probability.

## What `_init_weights` Does

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

`self.apply(self._init_weights)` walks every submodule and re-initializes all Linear and Embedding layers to **N(0, 0.02)**. This is the GPT-2 init scheme from the original OpenAI code.

For the weight-tying case, the critical effect is: the shared embedding/output-head matrix goes from std=1.0 to std=0.02. That 50× reduction means initial logit std drops from ~27.7 to ~0.55, putting you right at the expected ~10.8 nats.

## Why You Don't Need It Without Weight Tying

Without tying, the output head is a regular `nn.Linear` with Kaiming uniform init. For a (768, 50257) matrix, Kaiming uniform gives values in roughly ±1/√768 ≈ ±0.036. This produces logits with std ≈ 1.0 — perfectly fine for cross-entropy. The embedding's N(0, 1) init is also fine because it only feeds *into* the network, not directly into the loss function.

The whole issue is specifically that weight tying forces N(0, 1) embedding weights into the role of an output projection, where they produce logits that are way too large.

## Summary

| Scenario | Embedding Init | Output Head Init | Initial Logit Std | Init Loss |
|---|---|---|---|---|
| No tying, no `_init_weights` | N(0, 1) | Kaiming ≈ ±0.036 | ~1.0 | ~10.8 ✓ |
| Tying, no `_init_weights` | N(0, 1) | = embedding = N(0, 1) | ~27.7 | ~530 ✗ |
| Tying + `_init_weights` | N(0, 0.02) | = embedding = N(0, 0.02) | ~0.55 | ~10.8 ✓ |

The `_init_weights` method is **not** about fixing Linear layers (Kaiming is fine for those). It's about taming the embedding weights that, through tying, now also serve as the output projection. You could achieve the same effect by *only* re-initializing `tok_emb`:

```python
nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
# out_head.weight is the same tensor, so this fixes both
```

The blanket `apply` approach is just defensive — it makes init consistent everywhere and insulates you from whatever PyTorch might change its defaults to in the future.
