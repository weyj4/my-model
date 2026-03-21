FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1

WORKDIR /workspace

ENV HF_HOME=/workspace/hf_cache
ENV WANDB_DIR=/workspace/wandb

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

CMD ["bash", "scripts/train_fineweb.sh"]
