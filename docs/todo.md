# Training TODO

## Active bugs

- [ ] **Fix mmap in `TokenDataset`** — `TokenDataset.__init__` currently eager-loads all tokens into Python lists even when passed a numpy memmap array, defeating the entire point of `mmap_mode='r'` in `create_fineweb_loaders_from_file`. Fix is documented in `docs/incident.md`. Must fix before 2.5B token run or it will OOM.

---

## Performance / training quality investigations

- [ ] **Consider turning off `shuffle=True` for FineWeb runs** — FineWeb is already shuffled at construction time and the pretokenizer concatenates across document boundaries, so consecutive training windows are already from different web pages. The theoretical benefit of shuffle (breaking batch correlations) is weak for this dataset. The practical cost on a network volume is real: random page faults vs sequential reads with OS readahead. Worth trying sequential access and comparing tokens/sec. Probably no meaningful difference in final loss.

---

## Observability / logging

- [ ] Add gradient norm logging to W&B — `wandb.log({"train/grad_norm": grad_norm})` after `clip_grad_norm_`
- [ ] Add tokens/sec throughput metric — measures DataLoader vs GPU bottleneck, reveals if storage is a constraint
- [ ] Add val loss to training loop eval checkpoints — `evaluate_model` exists in `utils.py` but is not called from the training loop
- [ ] Set up `torch.profiler` for first N steps — op-level timing breakdown to understand compute profile

---

## Architecture / training improvements

- [ ] Implement LR warmup + cosine decay scheduler — `warmup_steps` exists in `TrainingConfig` but scheduler is not implemented, flat LR is unstable for long runs
- [ ] Implement Flash Attention — replace manual `Q@K.T + softmax + @V` in `model.py` with `F.scaled_dot_product_attention(queries, keys, values, is_causal=True)`, one-line change, meaningful memory and speed improvement

---

## Ablations (after baseline run)

- [ ] RMSNorm vs LayerNorm
- [ ] RoPE vs learned absolute positional embeddings
- [ ] SwiGLU vs GELU
- [ ] Flash Attention vs manual attention
- [ ] Config sweeps: n_heads, n_layers, emb_dim, batch_size, lr
