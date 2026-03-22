FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV HF_HOME=/workspace/hf_cache
ENV WANDB_DIR=/workspace/wandb

RUN apt-get update && apt-get install -y neovim && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install tiktoken wandb datasets huggingface_hub

COPY . .

CMD ["bash", "scripts/train_fineweb.sh"]
