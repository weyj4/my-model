---
language:
- en
license: apache-2.0
tags:
- gpt2
- causal-lm
- pretraining
- from-scratch
datasets:
- HuggingFaceFW/fineweb
pipeline_tag: text-generation
---

# my-gpt2-163m

GPT-2 style autoregressive language model (163M parameters) trained from scratch
on FineWeb. Built as a learning project following Raschka's *Build an LLM from
Scratch*, with a custom Flash Attention implementation and mmap data pipeline.

**This is a base (pretrained) model — not instruction tuned.** It will continue
text in a web-document style but will not follow instructions or produce coherent
responses without further fine-tuning. See the generation notes below.

## Model Details

| | |
|---|---|
| Architecture | GPT-2 (Pre-LN, learned absolute positional embeddings) |
| Parameters | 163M |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dim | 768 |
| Context length | 1024 tokens |
| Vocab size | 50,257 (GPT-2 BPE) |
| Attention | Flash Attention (`F.scaled_dot_product_attention`) |

## Training

| | |
|---|---|
| Dataset | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| Tokens seen | 1.4B (~43% of Chinchilla optimal) |
| Steps | 42,000 |
| Batch size | 64 × 1024 tokens = 65,536 tokens/step |
| Learning rate | 4e-4 (fixed, no warmup/decay) |
| Optimizer | AdamW (β1=0.9, β2=0.999, weight_decay=0.1) |
| Hardware | RunPod A40 (48GB) |
| Precision | bfloat16 (stored), float32 (training) |
| val_loss | 3.77 |
| Perplexity | ~43 |

## Usage

```python
import torch
import tiktoken
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("weyj4/my-gpt2-163m")
model.eval()

enc = tiktoken.get_encoding("gpt2")

prompt = "The study found that"
input_ids = torch.tensor([enc.encode(prompt)])

with torch.no_grad():
    out = model.generate(
        input_ids,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        repetition_penalty=1.3,
        pad_token_id=50256,
    )

print(enc.decode(out[0].tolist()))
```

**Note on tokenizer:** The HuggingFace GPT2Tokenizer saved with this model has
a known issue in some environments. Use `tiktoken` directly as shown above, or
load the tokenizer with `GPT2Tokenizer.from_pretrained("gpt2", use_fast=False)`.

## Generation Notes

As an undertrained base model, generation quality is limited:

- Works best with **web-document style prompts**: "According to researchers...",
  "The study found that...", "In a recent paper..."
- Use `repetition_penalty=1.3` and `top_k=50` to reduce degenerate loops
- Will occasionally produce non-English tokens (FineWeb contains multilingual content)
- Does not follow instructions — this is expected for a base model

## Limitations

- Trained on only 1.4B tokens vs. ~3.3B Chinchilla optimal for this size
- No learning rate warmup or cosine decay (fixed LR throughout)
- FineWeb contains multilingual content; model has no language filtering
- Not evaluated on standard benchmarks (HellaSwag, PIQA, WinoGrande)
- Not suitable for production use

## Planned

- [ ] `weyj4/my-gpt2-163m-v2`: retrain on FineWeb-Edu with warmup + cosine decay
- [ ] `weyj4/my-gpt2-163m-sft`: SFT on OpenHermes 2.5 using `trl`
- [ ] `weyj4/my-gpt2-163m-dpo`: DPO alignment using `trl`

## Repo

Training code: [github.com/weyj4/my-model](https://github.com/weyj4/my-model)
