# HuggingFace Post-Training Project

**Goal:** Convert a custom pretrained GPT-2 163M to HuggingFace format, then run a complete SFT → DPO post-training pipeline using `trl`. Publish all checkpoints to the Hub with proper model cards.

**Estimated timeline:** 3 weeks  
**Compute:** RunPod A40 ($0.40/hr), same volume as pretraining  
**Prerequisites:** Pretrained checkpoint at step 42900 (val_loss=3.77, 1.4B tokens on FineWeb)

---

## Phase 1 — HF Conversion (Day 1-2)

Convert the custom checkpoint to standard HuggingFace format so it can be loaded with `AutoModelForCausalLM.from_pretrained()`.

**Key challenge:** The custom architecture uses `nn.Linear` for attention projections, but HF GPT-2 uses `Conv1D` (a historical TensorFlow artifact). `Conv1D` stores weights transposed relative to `nn.Linear`, so all weight matrices must be `.T`'d during conversion. Bias vectors are unaffected.

**Weight key mapping:**

| Custom key | HF key | Needs transpose? |
|---|---|---|
| `tok_emb.weight` | `transformer.wte.weight` | No |
| `pos_emb.weight` | `transformer.wpe.weight` | No |
| `trf_blocks.{i}.att.W_query/key/value.weight` | `transformer.h.{i}.attn.c_attn.weight` | Yes (concat Q,K,V then `.T`) |
| `trf_blocks.{i}.att.out_proj.weight` | `transformer.h.{i}.attn.c_proj.weight` | Yes |
| `trf_blocks.{i}.norm1.scale/shift` | `transformer.h.{i}.ln_1.weight/bias` | No |
| `trf_blocks.{i}.norm2.scale/shift` | `transformer.h.{i}.ln_2.weight/bias` | No |
| `trf_blocks.{i}.ff.layers.0.weight/bias` | `transformer.h.{i}.mlp.c_fc.weight/bias` | weight yes, bias no |
| `trf_blocks.{i}.ff.layers.2.weight/bias` | `transformer.h.{i}.mlp.c_proj.weight/bias` | weight yes, bias no |
| `final_norm.scale/shift` | `transformer.ln_f.weight/bias` | No |
| `out_head.weight` | `lm_head.weight` | No |

**Script:** `scripts/convert_to_hf.py`

```bash
python scripts/convert_to_hf.py \
    --checkpoint /workspace/checkpoints/model_step_042900.pt \
    --output_dir /workspace/hf_model \
    --push_to_hub
```

**Sanity check:** After conversion, run inference and verify the model produces coherent (if incoherent-content) text. A pretrained base model with no instruction tuning will ramble — that's expected.

**Deliverable:** `weyj4/my-gpt2-163m` on HuggingFace Hub

---

## Phase 2 — Supervised Fine-Tuning (Week 1)

Fine-tune the pretrained base on instruction-following data using `trl.SFTTrainer`. This teaches the model to respond to prompts rather than just continue text.

**Dataset:** [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) — 1M high-quality instruction pairs in ChatML format. Alternatively Alpaca for a smaller run.

**Key concepts learned:**
- `datasets` library: streaming, `.map()`, `.filter()`, batching, the Arrow format
- Chat templates: `tokenizer.apply_chat_template()`, the ChatML format (`<|im_start|>user\n...<|im_end|>`)
- `SFTTrainer`: how it wraps HF `Trainer`, the `DataCollatorForLanguageModeling` it uses internally
- PEFT/LoRA: `peft.LoraConfig`, rank/alpha tradeoffs, which modules to target (`c_attn`, `c_proj`), adapter merging with `merge_and_unload()`
- `TrainingArguments`: gradient accumulation, mixed precision, eval strategy, logging

**Why LoRA:** At 163M parameters, full fine-tuning fits in VRAM easily. But LoRA is worth learning because at 7B+ it becomes necessary, and the `peft` library is used everywhere in the HF ecosystem.

```python
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="/workspace/sft_checkpoints",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=500,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
)
trainer.train()
```

**Deliverable:** `weyj4/my-gpt2-163m-sft` on Hub

---

## Phase 3 — DPO Alignment (Week 2)

Run Direct Preference Optimization on top of the SFT model. DPO is the current industry-standard alternative to PPO-based RLHF — it's simpler (no reward model), more stable, and used in Llama 3, Mistral, Tülu 3, and most modern aligned models.

**Dataset:** [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) or [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback). Both provide `chosen`/`rejected` response pairs.

**Key concepts learned:**
- What preference data looks like and how it's collected
- The DPO objective: maximizes log-ratio of `chosen` vs `rejected` under the policy relative to a frozen reference model
- How `trl.DPOTrainer` manages the reference model internally (it freezes a copy)
- The β hyperparameter: controls KL divergence from reference (higher β = stays closer to SFT)
- Difference between DPO, IPO, and ORPO (all supported by `trl`)

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,
    output_dir="/workspace/dpo_checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    bf16=True,
    report_to="wandb",
)

dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=ref_model,   # frozen SFT model copy
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

**Reference codebase to study:** [allenai/open-instruct](https://github.com/allenai/open-instruct) — production implementation of exactly this pipeline for OLMo/Tülu. Reading their DPO training script alongside building your own is the fastest way to understand how serious labs structure post-training.

**Deliverable:** `weyj4/my-gpt2-163m-dpo` on Hub

---

## Phase 4 — Evals & Model Cards (Week 3)

Close the loop with benchmarks comparing all three checkpoints.

**Eval pipeline:** `lm_eval` against the existing BigQuery tracking table (`eval_tracking.benchmark_results` in `cuda-489615`).

```bash
# Run evals on all three checkpoints
for MODEL in weyj4/my-gpt2-163m weyj4/my-gpt2-163m-sft weyj4/my-gpt2-163m-dpo; do
    lm_eval --model hf \
        --model_args pretrained=$MODEL \
        --tasks hellaswag,piqa,winogrande \
        --batch_size 16 \
        --output_path /workspace/evals/${MODEL##*/}.json
done
```

**What to expect:** The pretrained base will perform near chance on instruction-following tasks. SFT will jump significantly on benchmarks that reward coherent responses. DPO will show more subtle improvements — better refusals, less sycophancy, improved preference alignment — which may not show up cleanly in automatic benchmarks but is visible in qualitative testing.

**Model card template (for each Hub repo):**

```markdown
---
language: en
license: apache-2.0
tags:
- gpt2
- pretraining
- causal-lm
datasets:
- HuggingFaceFW/fineweb
---

# my-gpt2-163m[-sft/-dpo]

GPT-2 style 163M parameter model pretrained from scratch on FineWeb.
Built following Raschka's "Build an LLM from Scratch" with custom
Flash Attention implementation and mmap data pipeline.

## Training details
- Architecture: GPT-2 (12 layers, 12 heads, 768 emb_dim)
- Pretraining data: FineWeb 2.5B tokens
- Tokens seen: 1.4B (step 42900)
- Hardware: RunPod A40
- val_loss: 3.77 (pretrained) / X.XX (SFT) / X.XX (DPO)

## Evals
| Task | Score |
|------|-------|
| HellaSwag | X.X% |
| PIQA | X.X% |
| WinoGrande | X.X% |
```

---

## Libraries Covered

| Library | What you learn |
|---|---|
| `transformers` | Model classes, tokenizers, `AutoModel`, `PreTrainedModel`, generation configs |
| `datasets` | Arrow format, streaming, `.map()`/`.filter()`, batching, Hub integration |
| `trl` | `SFTTrainer`, `DPOTrainer`, reward modeling utilities, `PPOTrainer` (optional) |
| `peft` | LoRA, `LoraConfig`, adapter merging, `get_peft_model` |
| `accelerate` | Distributed training abstraction (encountered when scaling to multi-GPU) |
| `huggingface_hub` | Programmatic Hub interaction, `HfApi`, `upload_folder` |

---

## Stretch Goals

- **PPO run:** Use `trl.PPOTrainer` with a reward model trained on the same preference data. Compare to DPO — slower and more complex, but teaches you the full RLHF pipeline.
- **GRPO:** The DeepSeek-R1 training method, now supported in `trl`. Apply to a reasoning dataset.
- **Multi-GPU SFT:** Scale to 2× A40 with `accelerate` and DDP. Write the `accelerate config` and launch script.
- **Contribute to `trl`:** Fix a bug or add a feature. The codebase is approachable and the maintainers are active.
