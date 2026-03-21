import argparse
import os
import time
from dataclasses import asdict
import tiktoken
import torch
import wandb
from gpt2.config import GPTConfig, TrainingConfig
from gpt2.model import GPTModel
from gpt2.data import create_verdict_loaders, create_fineweb_loaders_from_file
from gpt2.utils import calc_loss_batch, save_checkpoint, bits_per_byte, evaluate_model
from gpt2.generate import token_ids_to_text, text_to_token_ids, generate_text_simple

SMOKE_CONFIG = GPTConfig(
    vocab_size=50257,
    context_length=64,   # tiny
    emb_dim=64,          # tiny
    n_heads=2,
    n_layers=2,
)

def train(model, train_loader, val_loader, optimizer, device, train_cfg, model_cfg, tokenizer):
    tokens_seen, global_step = 0, -1
    t0 = time.time()

    for epoch in range(train_cfg.num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % train_cfg.eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, train_cfg.eval_iter
                )
                tokens_per_sec = tokens_seen / (time.time() - t0)
                wandb.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/perplexity": torch.exp(torch.tensor(train_loss)).item(),
                    "train/bpb": bits_per_byte(train_loss),
                    "train/tokens_seen": tokens_seen,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": global_step,
                    "train/grad_norm": grad_norm.item()
                })
                print(
                    f"Epoch {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"BPB {bits_per_byte(train_loss):.3f}"
                )

            if global_step % train_cfg.checkpoint_freq == 0 and global_step > 0:
                os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
                save_checkpoint(
                    model, optimizer, model_cfg,
                    step=global_step,
                    path=f"{train_cfg.checkpoint_dir}/ckpt_{global_step:06d}.pt"
                )

        # end of epoch generation sample
        encoded = text_to_token_ids("Every effort moves you", tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=50,
                context_size=model_cfg.context_length
            )
        print(token_ids_to_text(token_ids, tokenizer))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_tokens", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", type=str, default="fineweb_file")
    args = parser.parse_args()

    if args.smoke:
        model_cfg = SMOKE_CONFIG
        train_cfg = TrainingConfig(
            num_tokens=10_000,
            batch_size=2,
            eval_freq=5,
            num_epochs=2,
            dataset=args.dataset
        )
    else:
        model_cfg = GPTConfig()
        train_cfg = TrainingConfig(
            batch_size=args.batch_size,
            num_tokens=args.num_tokens,
            lr=args.lr,
            wandb_run_name=args.run_name
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=train_cfg.wandb_project,
        name=train_cfg.wandb_run_name,
        config={**asdict(model_cfg), **asdict(train_cfg)}
    )

    model = GPTModel(model_cfg)
    model = torch.compile(model)
    model.to(device)

    tokenizer = tiktoken.get_encoding("gpt2")

    if train_cfg.dataset == "verdict":
        train_loader, val_loader = create_verdict_loaders(
            batch_size=train_cfg.batch_size,
            context_length=model_cfg.context_length
        )
    else:
        train_loader, val_loader = create_fineweb_loaders_from_file(
            path=train_cfg.data_path,
            batch_size=train_cfg.batch_size,
            context_length=model_cfg.context_length,
        )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay
    )

    train(model, train_loader, val_loader, optimizer, device, train_cfg, model_cfg, tokenizer)

    wandb.finish()

if __name__ == "__main__":
    main()
