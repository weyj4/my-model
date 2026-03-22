import torch, glob

ckpts = sorted(glob.glob("/workspace/checkpoints/*.pt"))
print("Checkpoints found:")
for c in ckpts:
    import os
    print(f"  {c}  ({os.path.getsize(c)/1e6:.1f} MB)")

if not ckpts:
    print("No checkpoints found — checking broader workspace:")
    import subprocess
    subprocess.run(["find", "/workspace", "-name", "*.pt", "-ls"])
else:
    latest = ckpts[-1]
    print(f"\nInspecting: {latest}")
    ckpt = torch.load(latest, map_location="cpu")
    print(f"Type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
        state = ckpt.get("model_state_dict", ckpt)
    else:
        state = ckpt
    print(f"\nAll weight keys and shapes:")
    for k, v in state.items():
        print(f"  {k}: {v.shape} {v.dtype}")
