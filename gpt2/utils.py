from dataclasses import asdict
import torch
import torch.nn.functional as F

def save_checkpoint(model, optimizer, config, step, path):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
    }, path)

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"]

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_batch)
    else:
        logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(loader), len(loader))
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            total_loss += calc_loss_batch(x, y, model, device).item()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def bits_per_byte(loss_nats, bytes_per_token=4.0):
    return (loss_nats / 0.693) / bytes_per_token
