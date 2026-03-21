# CLAUDE.md — Project Context for LLM Assistants

This file provides context for LLM assistants helping with this project.
Read this before making suggestions or writing code.

---

## What This Project Is

A from-scratch GPT-2 style LLM pretraining implementation in PyTorch, built for learning and experimentation. The primary goals are:

1. **Deep understanding of transformer architecture** — implementing every component from scratch (MHA, LayerNorm, GELU, FFN, positional embeddings) rather than using HuggingFace abstractions
2. **Learning PyTorch internals** — `nn.Module` registry system, autograd, optimizer state, `torch.compile`, bfloat16 mixed precision
3. **Observability and tooling** — W&B for training metrics and system telemetry, torch.profiler for compute profiling, Mosaic for GPU memory profiling
4. **Understanding compute/memory profiles** — what does training actually cost? Where are the bottlenecks? How does memory scale with sequence length, batch size, model size?
5. **Ablation studies** — systematically varying architecture choices and measuring their effect on loss curves and throughput

This is explicitly a **learning project**, not a production system. The owner has a strong systems engineering background (distributed systems, Kafka, Redis, Postgres, GCP).

This project was strongly based on https://github.com/rasbt/LLMs-from-scratch and the owner has been following along with that repo's associated print book.

---

## Repository Structure

```
my-model/
├── gpt2/                    # main package
│   ├── config.py            # GPTConfig and TrainingConfig dataclasses
│   ├── model.py             # GPTModel, MultiHeadAttention, TransformerBlock,
│   │                        # FeedForward, LayerNorm, GELU
│   ├── data.py              # GPTDatasetV1, TokenDataset, DataLoader factories,
│   │                        # create_verdict_loaders, create_fineweb_loaders,
│   │                        # create_fineweb_loaders_from_file
│   ├── train.py             # training loop, evaluate_model, main() with argparse
│   ├── generate.py          # generate_text_simple, text/token helpers
│   └── utils.py             # calc_loss_batch, save/load checkpoint, bits_per_byte
├── scripts/
│   ├── pretokenize.py       # tokenize Fineweb to .npy file (run once on RunPod)
│   ├── train_fineweb.sh     # main training entrypoint — checks for token file,
│   │                        # runs pretokenize if missing, then launches training
│   └── train_verdict.sh     # quick sanity check on local short story dataset
├── docs/
│   ├── data_loading.md      # mmap vs eager loading strategies and tradeoffs
│   └── incident_mmap_dtype.md  # postmortem on mmap implementation bug
├── tests/                   # pytest shape/logic tests (partially implemented)
├── Dockerfile               # runpod/pytorch base + pip install deps
├── .github/workflows/
│   └── docker.yml           # builds and pushes to Docker Hub on every push to main
├── pyproject.toml           # uv-managed dependencies
└── data/                    # gitignored — local text files for smoke tests
    └── the-verdict.txt
```

---

## Current Architecture: GPT-2 124M Style

**Config (`GPT_CONFIG_124M`):**
```python
vocab_size = 50257       # GPT-2 BPE tokenizer (tiktoken)
context_length = 1024    # sequence length
emb_dim = 768            # embedding dimension
n_heads = 12             # attention heads
n_layers = 12            # transformer blocks
drop_rate = 0.0          # dropout disabled (modern convention)
qkv_bias = False         # no bias on QKV projections (modern convention)
```

**Architecture notes:**
- Pre-LN (pre-normalization) — LayerNorm before attention and FFN, not after
- Learned absolute positional embeddings (not RoPE)
- GELU activation (tanh approximation)
- 4x FFN expansion (768 → 3072 → 768)
- Custom LayerNorm implementation (not nn.LayerNorm)
- Causal mask registered as buffer (context_length × context_length upper triangular)
- Output head: nn.Linear(768, 50257), no weight tying with token embeddings

**Current parameter count:** ~163M (slightly over 124M due to bias=True defaults on Linear layers)

---

## Training Setup

**Dataset:** FineWeb (`HuggingFaceFW/fineweb`, `sample-10BT` shard)
**Tokenizer:** tiktoken GPT-2 BPE (50257 vocab)
**Optimizer:** AdamW, lr=4e-4, weight_decay=0.1
**Gradient clipping:** 1.0
**Batch size:** 32 (on A40)
**Mixed precision:** bfloat16 autocast on CUDA
**Compiled:** torch.compile (TorchInductor backend)

**Data pipeline:**
1. `scripts/pretokenize.py` — streams Fineweb, tokenizes with tiktoken, saves to `np.int32` array at `/workspace/data/fineweb_2b5.npy` (~10GB for 2.5B tokens)
2. `create_fineweb_loaders_from_file` — loads with `np.load(mmap_mode='r')`, slices in `__getitem__`
3. `TokenDataset` — **NOTE: mmap branch is not yet properly implemented** (see `docs/incident_mmap_dtype.md`)

**Metrics logged to W&B:**
- `train/loss` — cross-entropy loss (nats)
- `train/perplexity` — e^loss
- `train/bpb` — bits per byte (loss / 0.693 / 4.0)
- `train/tokens_seen` — cumulative token count
- `train/step` — global step
- System metrics auto-logged: GPU utilization, VRAM, CPU, RAM, power usage

---

## Infrastructure / Ops Stack

### Compute: RunPod
- GPU: A40 (48GB VRAM) for validation runs, H100 for full runs
- Container disk: 20GB (temporary)
- Network Volume: 20GB persistent at `/workspace` — survives pod termination
- Secrets injected as env vars: `WANDB_API_KEY`, `HF_TOKEN`
- Startup: `train_fineweb.sh` checks for pretokenized file, skips if present

### CI/CD: GitHub Actions → Docker Hub
- Every push to `main` triggers `.github/workflows/docker.yml`
- Builds `weyj4/my-model:latest` using `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` base
- GHA cache (`type=gha,scope=my-model`) — only rebuilds changed layers
- ~3 min rebuild for code-only changes, ~25 min for dependency changes

### Experiment Tracking: Weights & Biases
- Project: `gpt2-pretraining` at `wandb.ai/weylandjoyner4-self`
- Each architectural variant gets its own named run
- Config dict (model + training hyperparams) logged at run init via `asdict()`
- System metrics auto-instrumented (no extra code needed)
- Parallel coordinates view for ablation comparison

### Dataset: HuggingFace FineWeb
- `HuggingFaceFW/fineweb`, `sample-10BT` shard (10B tokens available)
- Streamed via `datasets` library, tokenized, saved to numpy binary
- Cached on RunPod network volume — only downloaded/tokenized once
- Future: scale to full FineWeb corpus by changing shard name and token count

### Local Dev
- macOS, M2 Pro (16GB), Ghostty terminal, Neovim, tmux
- `uv` for dependency management
- Smoke tests run locally on CPU with tiny config (2 layers, 64 dim)
- `WANDB_MODE=offline uv run python -m gpt2.train --smoke`

---

## Immediate Issues to Fix

1. **mmap not wired up** — `TokenDataset.__init__` still builds eager Python lists even when passed a numpy array. Full fix is in `docs/incident_mmap_dtype.md`. Must fix before 2.5B token run or it will OOM.

2. **LR warmup not implemented** — `warmup_steps` exists in `TrainingConfig` but the scheduler isn't implemented in the training loop. Flat LR works for short runs but cosine decay with warmup is needed for stable full training.

3. **No val loss in training loop** — `evaluate_model` exists but the training loop only logs train loss (uses current batch, not full val set). Should call `evaluate_model` at eval checkpoints.

---

## Planned Ablation Studies

The goal is to run systematic experiments varying one component at a time, compare loss curves and throughput on W&B, and build empirical intuition for why modern architectures made the choices they did.

### Normalization
| Variant | Change | Expected effect |
|---------|--------|-----------------|
| Baseline | LayerNorm (current) | — |
| RMSNorm | Drop mean subtraction and shift parameter | Marginal speedup, similar loss |

**RMSNorm implementation:**
```python
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)
```

### Positional Encoding
| Variant | Change | Expected effect |
|---------|--------|-----------------|
| Baseline | Learned absolute (current) | — |
| RoPE | Rotate Q/K in attention | Better length generalization |

RoPE applies a rotation matrix to Q and K inside attention based on position, rather than adding positional vectors to embeddings. Used in Llama, Mistral, and most modern models.

### Activation Function
| Variant | Change | Expected effect |
|---------|--------|-----------------|
| Baseline | GELU (current) | — |
| SwiGLU | Gated linear unit in FFN | Empirically better, used in Llama |

SwiGLU changes the FFN from 2 linear layers to 3, adding a gating mechanism:
```python
# GELU FFN (current): x → Linear(768, 3072) → GELU → Linear(3072, 768)
# SwiGLU FFN: x → (Linear(768, 3072) * SiLU(Linear(768, 3072))) → Linear(3072, 768)
```

### Attention Mechanism
| Variant | Change | Expected effect |
|---------|--------|-----------------|
| Baseline | MHA (current) | — |
| GQA | Share K/V heads across Q head groups | Faster inference, smaller KV cache |
| Flash Attention | Use F.scaled_dot_product_attention | Significant memory/speed win |

Flash Attention is the easiest win — just replace the manual attention computation:
```python
# Replace manual Q@K.T + softmax + @V with:
context_vec = F.scaled_dot_product_attention(
    queries, keys, values, is_causal=True
)
```

### Config Ablations (no code changes)
- `n_heads`: 1 vs 4 vs 12
- `n_layers`: 6 vs 12 vs 24
- `emb_dim`: 256 vs 512 vs 768
- `batch_size`: 8 vs 32 vs 128 (throughput vs stability)
- `lr`: 1e-4 vs 4e-4 vs 1e-3

---

## Observability Targets (not yet implemented)

- **Gradient norm logging** — `wandb.log({"train/grad_norm": grad_norm})` after `clip_grad_norm_`
- **Tokens per second** — measure and log throughput to detect DataLoader bottlenecks
- **torch.profiler** — profile first N steps to get op-level timing breakdown
- **Mosaic memory snapshot** — `torch.cuda.memory._record_memory_history()` for GPU memory timeline

---

## Key Design Decisions

**Why dataclasses over dicts for config?**
Type safety, IDE autocomplete, `asdict()` for W&B logging. All config access uses `cfg.emb_dim` not `cfg["emb_dim"]`.

**Why bfloat16 over float16?**
BF16 has the same exponent range as FP32 (less overflow risk) with half the memory. Better for training stability than FP16. Native on A100/H100/A40.

**Why torch.compile?**
Fuses ops into single CUDA kernels via TorchInductor/Triton. ~15-40% speedup for free. First forward pass is slow (compilation), subsequent passes use cached kernels.

**Why Pre-LN over Post-LN?**
Better gradient flow — gradients pass through the residual stream directly without going through a norm layer. Makes deep networks more stable to train. GPT-2 switched to Pre-LN; original "Attention is All You Need" used Post-LN.

**Why no dropout?**
Modern architectures at scale have largely dropped dropout. Regularization comes from data diversity at large token counts. Dropout was more important at smaller scales with less data.

---

## Reference Architectures

For comparison and future implementation targets:

| Model | Params | Layers | Heads | emb_dim | Norm | Pos | Act | Notes |
|-------|--------|--------|-------|---------|------|-----|-----|-------|
| GPT-2 124M | 124M | 12 | 12 | 768 | LayerNorm | Learned | GELU | This implementation |
| GPT-2 1.5B | 1.5B | 48 | 25 | 1600 | LayerNorm | Learned | GELU | Same arch, scaled |
| Llama 3 8B | 8B | 32 | 32 | 4096 | RMSNorm | RoPE | SwiGLU | GQA (8 KV heads) |
| Mistral 7B | 7B | 32 | 32 | 4096 | RMSNorm | RoPE | SwiGLU | Sliding window attn |

---

## Next Steps (suggested priority order)

1. Fix mmap `TokenDataset` — prerequisite for 2.5B token run
2. Implement LR warmup + cosine decay scheduler
3. Add val loss to training loop eval checkpoints
4. Add gradient norm logging to W&B
5. Add tokens/sec throughput metric
6. Run 2.5B token baseline on A40, get W&B loss curves
7. Implement Flash Attention (`F.scaled_dot_product_attention`) — easiest win
8. Implement RMSNorm and run ablation vs LayerNorm baseline
9. Implement RoPE
10. Implement SwiGLU
11. Upload trained checkpoint to HuggingFace Hub
