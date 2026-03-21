from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.0
    qkv_bias: bool = False

@dataclass
class TrainingConfig:
    lr: float = 4e-4
    weight_decay: float = 0.1
    num_epochs: int = 1
    batch_size: int = 32
    eval_freq: int = 100
    eval_iter: int = 10
    grad_clip: float = 1.0
    warmup_steps: int = 100
    dataset: str = "fineweb_file"
    num_tokens: int = 10_000_000
    wandb_project: str = "gpt2-pretraining"
    wandb_run_name: str = "baseline"
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 500
    data_path: str = "/workspace/data/fineweb_2b5.npy"


