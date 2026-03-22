# Model Architecture Reference

**my-gpt2-163m** — GPT-2 style, 163M parameters  
Pretrained on FineWeb (1.4B tokens), step 42000, val_loss=3.77

---

## Config

```python
vocab_size     = 50257   # GPT-2 BPE tokenizer
context_length = 1024    # max sequence length
emb_dim        = 768     # d_model
n_heads        = 12      # attention heads
n_layers       = 12      # transformer blocks
head_dim       = 64      # emb_dim / n_heads
ffn_dim        = 3072    # 4 × emb_dim
drop_rate      = 0.0     # no dropout during pretraining
use_flash      = True    # F.scaled_dot_product_attention
bias           = True    # nn.Linear bias=True (163M vs 124M)
```

---

## Parameter count breakdown

| Component | Shape | Params |
|---|---|---|
| `tok_emb.weight` | [50257, 768] | 38,597,376 |
| `pos_emb.weight` | [1024, 768] | 786,432 |
| Per block × 12: | | |
| `att.W_query` | [768, 768] + bias [768] | 590,592 |
| `att.W_key` | [768, 768] + bias [768] | 590,592 |
| `att.W_value` | [768, 768] + bias [768] | 590,592 |
| `att.out_proj` | [768, 768] + bias [768] | 590,592 |
| `ff.layers.0` (expand) | [3072, 768] + bias [3072] | 2,362,368 |
| `ff.layers.2` (contract) | [768, 3072] + bias [768] | 2,360,064 |
| `norm1.scale/shift` | [768] × 2 | 1,536 |
| `norm2.scale/shift` | [768] × 2 | 1,536 |
| `final_norm.scale/shift` | [768] × 2 | 1,536 |
| `out_head.weight` | [50257, 768] | 38,597,376 |
| `out_head.bias` | [50257] | 50,257 |
| **Total** | | **~163M** |

---

## Full weight map with shapes

All weights as stored in checkpoint (`ckpt_042000.pt`).  
Note: keys have `_orig_mod.` prefix due to `torch.compile` — stripped on load.

### Embeddings

```
tok_emb.weight          [50257, 768]   bf16   token embeddings (vocab → d_model)
pos_emb.weight          [1024,  768]   bf16   position embeddings (pos → d_model)
```

### Transformer block (repeated × 12, index i = 0..11)

```
# Pre-attention LayerNorm (Pre-LN architecture)
trf_blocks.{i}.norm1.scale             [768]         bf16   LN gain (γ)
trf_blocks.{i}.norm1.shift             [768]         bf16   LN bias (β)

# Multi-head self-attention
trf_blocks.{i}.att.W_query.weight      [768, 768]    bf16   Q projection (out, in)
trf_blocks.{i}.att.W_query.bias        [768]         bf16
trf_blocks.{i}.att.W_key.weight        [768, 768]    bf16   K projection
trf_blocks.{i}.att.W_key.bias          [768]         bf16
trf_blocks.{i}.att.W_value.weight      [768, 768]    bf16   V projection
trf_blocks.{i}.att.W_value.bias        [768]         bf16
trf_blocks.{i}.att.out_proj.weight     [768, 768]    bf16   output projection
trf_blocks.{i}.att.out_proj.bias       [768]         bf16

# Pre-FFN LayerNorm
trf_blocks.{i}.norm2.scale             [768]         bf16
trf_blocks.{i}.norm2.shift             [768]         bf16

# Feed-forward network (GELU, 4× expansion)
trf_blocks.{i}.ff.layers.0.weight     [3072, 768]   bf16   expand  (out=3072, in=768)
trf_blocks.{i}.ff.layers.0.bias       [3072]        bf16
trf_blocks.{i}.ff.layers.2.weight     [768, 3072]   bf16   contract (out=768, in=3072)
trf_blocks.{i}.ff.layers.2.bias       [768]         bf16
```

*(layers.1 is the GELU activation — no parameters)*

### Output

```
final_norm.scale                       [768]         bf16   final LayerNorm γ
final_norm.shift                       [768]         bf16   final LayerNorm β
out_head.weight                        [50257, 768]  bf16   unembedding (vocab logits)
out_head.bias                          [50257]       bf16   logit bias
```

---

## HuggingFace key mapping

When converting to HF format, each key is remapped and **weight matrices transposed**
because HF GPT-2 uses `Conv1D` (a TensorFlow legacy artifact) which stores weights
as `[in_features, out_features]` instead of PyTorch `nn.Linear`'s `[out, in]`.

| Custom key | HF key | Transpose? | Notes |
|---|---|---|---|
| `tok_emb.weight` | `transformer.wte.weight` | No | embedding lookup |
| `pos_emb.weight` | `transformer.wpe.weight` | No | |
| `trf_blocks.{i}.norm1.scale` | `transformer.h.{i}.ln_1.weight` | No | 1D |
| `trf_blocks.{i}.norm1.shift` | `transformer.h.{i}.ln_1.bias` | No | 1D |
| `trf_blocks.{i}.norm2.scale` | `transformer.h.{i}.ln_2.weight` | No | 1D |
| `trf_blocks.{i}.norm2.shift` | `transformer.h.{i}.ln_2.bias` | No | 1D |
| `W_query + W_key + W_value` (concat) | `transformer.h.{i}.attn.c_attn.weight` | **Yes** | cat dim=0 then .T → [768, 2304] |
| Q+K+V biases (concat) | `transformer.h.{i}.attn.c_attn.bias` | No | cat → [2304] |
| `att.out_proj.weight` | `transformer.h.{i}.attn.c_proj.weight` | **Yes** | [768,768]→[768,768] |
| `att.out_proj.bias` | `transformer.h.{i}.attn.c_proj.bias` | No | |
| `ff.layers.0.weight` | `transformer.h.{i}.mlp.c_fc.weight` | **Yes** | [3072,768]→[768,3072] |
| `ff.layers.0.bias` | `transformer.h.{i}.mlp.c_fc.bias` | No | |
| `ff.layers.2.weight` | `transformer.h.{i}.mlp.c_proj.weight` | **Yes** | [768,3072]→[3072,768] |
| `ff.layers.2.bias` | `transformer.h.{i}.mlp.c_proj.bias` | No | |
| `final_norm.scale` | `transformer.ln_f.weight` | No | |
| `final_norm.shift` | `transformer.ln_f.bias` | No | |
| `out_head.weight` | `lm_head.weight` | No | not Conv1D in HF |
| `out_head.bias` | `lm_head.bias` | No | injected manually post-init |

---

## Forward pass data flow

```
tokens [B, T]
  ↓  tok_emb                         lookup → [B, T, 768]
  +  pos_emb                         add positions
  ↓
  for i in 0..11:
    x = x + Attention(LayerNorm(x))  pre-LN, residual
    x = x + FFN(LayerNorm(x))        pre-LN, residual
  ↓
  LayerNorm(x)                        final_norm
  ↓
  out_head                            [B, T, 50257] logits
```

### Attention internals (Flash Attention path)

```
x_ln = norm1(x)                       [B, T, 768]
Q = x_ln @ W_query.T + b_query        [B, T, 768]  → reshape → [B, 12, T, 64]
K = x_ln @ W_key.T   + b_key          [B, T, 768]  → reshape → [B, 12, T, 64]
V = x_ln @ W_value.T + b_value        [B, T, 768]  → reshape → [B, 12, T, 64]

# F.scaled_dot_product_attention (Flash Attention kernel)
# Never materializes [B, 12, T, T] attention matrix — O(T) memory
attn_out = SDPA(Q, K, V, is_causal=True)   [B, 12, T, 64]
         → reshape → [B, T, 768]
         @ out_proj.T + out_proj.bias       [B, T, 768]
```

### FFN internals

```
x_ln = norm2(x)                       [B, T, 768]
h    = GELU(x_ln @ W0.T + b0)        [B, T, 3072]   expand 4×
out  = h @ W2.T + b2                  [B, T, 768]    contract
```

---

## Checkpoint structure

```python
{
  "step":                int,          # training step (42000 for ckpt_042000.pt)
  "model_state_dict":    dict,         # weights (bf16, _orig_mod. prefixed)
  "optimizer_state_dict": dict,        # AdamW state (fp32, large — ~1.3GB)
  "config":              GPTConfig,    # dataclass with all hyperparameters
}
```

Total checkpoint size: ~1.6GB (weights in bf16 + optimizer state in fp32)  
Weights only (bf16): ~326MB  
HF model (float32): ~652MB
