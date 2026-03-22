# graybase — Server Reference

> Last surveyed: 2026-03-22. Host: `weyland@graybase` (192.168.1.1)

---

## Hardware

### GPUs — 2× NVIDIA RTX PRO 6000 Blackwell
| | |
|---|---|
| Architecture | Blackwell (GB202) |
| VRAM per card | ~95.6 GB (97,887 MiB) |
| **Combined VRAM** | **~191 GB** |
| TDP | 450W each |
| Driver | 580.126.09 |
| CUDA (driver) | 13.0 |
| CUDA (compiler) | 12.4 (nvcc 12.4.131) |
| PCIe slots | `95:00.0` (GPU 0), `B8:00.0` (GPU 1) |
| ECC | Disabled |
| MIG | Disabled |

These are newer than A100 80GB in VRAM terms, and more recent architecture than the A40 (48GB). Combined, they exceed what Runpod A40 pairs offer. `nvidia-smi topo -m` will show whether they share NVLink.

### CPU — 2× Intel Xeon 6747P (Granite Rapids-SP, 2024)
| | |
|---|---|
| Cores / threads per socket | 48C / 96T |
| Sockets | 2 |
| **Total logical CPUs** | **192** |
| Max clock | 3.9 GHz |
| NUMA nodes | 4 (nodes 0–3) |
| L2 cache | 192 MB (96 instances) |
| L3 cache | 576 MB (2 instances) |
| Notable ISA extensions | AVX-512, AVX-512-BF16, AMX (tile/int8/bf16), AVX-VNNI |

AMX (Advanced Matrix Extensions) means the CPU itself can accelerate matrix ops — useful if you're doing CPU-side inference or pre/post-processing at scale.

### RAM
| | |
|---|---|
| Total | 1.0 TiB (~1,052 GB) |
| Available (at survey) | ~618 GB |
| Swap | None configured |

### Storage — 4× NVMe drives (~28 TB each, ~112 TB raw)

| Mount | Size | Used | Free | Use% | Notes |
|---|---|---|---|---|---|
| `/` (nvme0n1) | 365 GB | 58 GB | 289 GB | 17% | OS drive |
| `/mnt/big1` (nvme2n1) | 28 TB | 23 TB | 4.3 TB | **84%** | GIS data — getting full |
| `/mnt/big2` (nvme1n1) | 28 TB | 10 TB | 17 TB | 38% | Your home dir lives here |
| `/mnt/big3` (nvme3n1) | 28 TB | 7.5 TB | 19 TB | 29% | |
| `/mnt/big4` (nvme4n1) | 28 TB | 26 TB | 1.4 TB | **95%** | Nearly full — do not write here |

`/mnt/big4` is effectively full. `/mnt/big1` is heading that way. Default to `/mnt/big2` or `/mnt/big3` for your own workloads.

---

## Software Stack

### CUDA / GPU
- CUDA Toolkit: **12.4** (`nvcc --version`)
- Driver CUDA cap: **13.0**
- Ollama: `/usr/local/bin/ollama` (systemd service, currently **inactive**)

### Ollama Models (as of 2026-03-22)
Stored locally, available via `ollama run <name>`. Notable models:

| Model | Size | Notes |
|---|---|---|
| `qwen3-coder:480b` | 290 GB | Requires both GPUs — the whole rig |
| `qwen3:235b-a22b-thinking-2507-q4_K_M` | 142 GB | Full thinking mode |
| `qwen3-vl:235b-a22b` | 143 GB | Vision + language, full size |
| `nemotron-3-super:120b` | 86 GB | |
| `qwen3.5:122b` | 81 GB | |
| `qwen3-coder-next:q8_0` | 84 GB | |
| `gpt-oss:120b` | 65 GB | Meta's GPT-4 open weights |
| `llama4:16x17b` | 67 GB | |
| `deepseek-r1:70b` | 42 GB | Fits on one GPU |
| `qwen3:30b` | 18 GB | Fast, single GPU |
| `gemma3:27b-it-fp16` | 54 GB | |
| `gemma3:4b` | 3.3 GB | Tiny, instant load |

Ollama API endpoint: `http://localhost:11434`

### GIS / Postgres
Owner runs a parcel data processing business. Postgres is the primary workload. Data includes NYC PLUTO, statewide parcel GDBs, PMTiles tilesets, and GeoJSON. Heavy I/O on `/mnt/big1` and `/mnt/big2`.

---

## SSH & Terminal Setup

### Mac `~/.ssh/config` entry
```
Host graybase
    HostName 192.168.1.1
    User weyland
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 10
    SetEnv TERM=xterm-256color
```

### Ghostty TERM fix
Ghostty sets `$TERM=xterm-ghostty` which the server doesn't have in its terminfo database. This breaks `htop`, `systemctl`, `less`, and other ncurses apps. Workarounds:

**Quick fix (per command):**
```bash
TERM=xterm-256color htop
systemctl status ollama --no-pager
```

**Permanent fix (install terminfo on server):**
```bash
# Run once from your Mac
infocmp xterm-ghostty | ssh weyland@graybase -- tic -x -
```

**After installing terminfo**, the `SetEnv` line in `.ssh/config` is optional but still good practice.

---

## Key Commands

### GPU Status
```bash
nvidia-smi                          # snapshot: model, VRAM, utilization, processes
watch -n 1 nvidia-smi               # live refresh every second (Ctrl+C to quit)
nvidia-smi dmon -s u                # streaming utilization + memory bandwidth
nvidia-smi -q | head -80            # verbose specs
nvidia-smi topo -m                  # GPU topology / NVLink
```

### CPU & Memory
```bash
TERM=xterm-256color htop            # interactive CPU/RAM monitor
top                                 # fallback if htop unavailable
free -h                             # RAM summary
vmstat 1 5                          # memory + CPU activity, 5 samples
```

### Disk
```bash
df -h                               # all mounts — check big1/big4 before writing
du -sh /mnt/big2/weyland/*          # how much space you're using
ncdu /mnt/big2/weyland              # interactive disk usage browser (if installed)
```

### Ollama
```bash
ollama list                         # all downloaded models
ollama ps                           # currently loaded models (VRAM in use)
ollama run qwen3:30b                # interactive chat
curl http://localhost:11434/api/tags   # REST API — list models
systemctl status ollama --no-pager  # service status
sudo systemctl stop ollama          # free GPUs for your own work — check with owner first
sudo systemctl start ollama         # restart it after you're done
```

### "Am I impacting his work?" — Checklist
```bash
# 1. Is Postgres busy?
sudo systemctl status postgresql --no-pager
# Watch query activity (requires pg access):
# sudo -u postgres psql -c "SELECT pid, state, query_start, query FROM pg_stat_activity WHERE state != 'idle';"

# 2. Is Ollama running / loaded?
ollama ps                           # shows what's in VRAM right now
systemctl is-active ollama

# 3. What's using the GPUs?
nvidia-smi | grep -E "Processes|MiB"

# 4. What's eating CPU/RAM?
ps aux --sort=-%cpu | head -15      # top CPU consumers
ps aux --sort=-%mem | head -15      # top RAM consumers

# 5. Is the disk I'm writing to under pressure?
iostat -x 2 3                       # disk I/O stats (install: sudo apt install sysstat)
# Check you're not on big1 (84%) or big4 (95% — almost full):
df -h /mnt/big1 /mnt/big4
```

### Long-Running Work (use tmux!)
```bash
tmux new -s train                   # start a named session
tmux attach -t train                # reattach after disconnect
tmux ls                             # list sessions
# Inside tmux: Ctrl+B, D to detach without killing the session
```

### Jupyter (when ready)
```bash
# On graybase — start in a tmux session
jupyter lab --no-browser --port=8888 --ip=127.0.0.1

# On your Mac — SSH tunnel (no firewall changes needed)
ssh -L 8888:localhost:8888 graybase

# Then open: http://localhost:8888
```

---

## Notes & Etiquette

- **Coordinate on Ollama.** It's a systemd service (`enabled`, auto-starts on boot). If you need the full 190GB for a model run, ask before `systemctl stop ollama`.
- **Avoid `/mnt/big4`** — 95% full. Don't write here.
- **Check `/mnt/big1`** before large GIS-adjacent writes — 84% full and it's his primary data volume.
- **Use tmux** for anything that takes more than a few minutes. SSH drops kill unprotected processes.
- **No swap configured.** OOM kills are hard kills. Be conscious of RAM usage on large model loads.
- **4 NUMA nodes.** For performance-sensitive work, pin processes to a NUMA node: `numactl --cpunodebind=0 --membind=0 python train.py`
