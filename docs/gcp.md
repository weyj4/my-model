# GCP Migration Guide: my-model on Vertex AI

## The big picture

Current architecture:
```
GitHub push → Docker Hub → RunPod pulls image → train_fineweb.sh runs
```

Target architecture:
```
GitHub push → Artifact Registry → Vertex AI Custom Job runs container → 
  reads data from GCS → writes checkpoints to GCS → logs to Cloud Logging
```

Your training code needs **zero changes**. Everything is infrastructure.

---

## Part 1: Critical Path (get it running)

### Step 1 — Enable APIs (10 min)

Every GCP service requires its API enabled. Do this first or everything downstream fails.

```bash
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com
```

**GCP products:** Service Usage API, Cloud Resource Manager

---

### Step 2 — Create a GCS bucket for data and artifacts (15 min)

Replace the RunPod network volume with Cloud Storage.

```bash
# training data and checkpoints
gcloud storage buckets create gs://YOUR_PROJECT_ID-ml-data \
  --location=us-central1 \
  --uniform-bucket-level-access

# upload your pretokenized dataset
# (from RunPod: install gcloud SDK, then run)
gsutil -m cp /workspace/data/fineweb_2b5.npy \
  gs://YOUR_PROJECT_ID-ml-data/data/fineweb_2b5.npy

# verify upload
gsutil ls -lh gs://YOUR_PROJECT_ID-ml-data/data/
```

**GCP products:** **Cloud Storage (GCS)**

Key concepts to know:
- `uniform-bucket-level-access` disables per-object ACLs — use IAM instead. This is the recommended modern approach
- Storage classes: Standard (hot data), Nearline (access < 1/month), Coldline, Archive — choose based on access frequency
- Object lifecycle rules: auto-delete files after N days — useful for auto-pruning old checkpoints at bucket level
- No egress charges for GCS → Vertex AI in the same region

---

### Step 3 — Create Artifact Registry repository (10 min)

Replace Docker Hub with GCP's managed container registry.

```bash
gcloud artifacts repositories create my-model \
  --repository-format=docker \
  --location=us-central1 \
  --description="Training container images"

# configure local docker to authenticate
gcloud auth configure-docker us-central1-docker.pkg.dev
```

**GCP products:** **Artifact Registry**

Key concepts:
- AR supports Docker, Python packages, Maven, npm — not just containers
- Images pulled from AR to Vertex in the same region have zero egress cost
- Artifact Registry replaced the older Container Registry (gcr.io) — always use AR for new projects
- Vulnerability scanning is available on images at rest

---

### Step 4 — Store secrets in Secret Manager (15 min)

Replace RunPod env var secrets with proper secret management.

```bash
# store secrets (no plaintext files)
echo -n "YOUR_WANDB_KEY" | \
  gcloud secrets create wandb-api-key --data-file=- --replication-policy=automatic

echo -n "YOUR_HF_TOKEN" | \
  gcloud secrets create hf-token --data-file=- --replication-policy=automatic

# verify
gcloud secrets list
```

**GCP products:** **Secret Manager**

Key concepts:
- Secrets are versioned — you can rotate without changing references in code
- Access is fully audited in Cloud Audit Logs
- Can be injected as env vars in Vertex training jobs at runtime
- Alternative to Kubernetes secrets (which are just base64-encoded, not encrypted)
- CMEK (Customer-Managed Encryption Keys) available for compliance requirements

---

### Step 5 — Create a service account with least-privilege IAM (20 min)

Your training container needs an identity with only the permissions it requires.

```bash
# create the service account
gcloud iam service-accounts create training-sa \
  --display-name="Vertex Training SA" \
  --description="Service account for custom training jobs"

SA_EMAIL="training-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com"

# grant only what it needs
# read/write GCS
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.objectAdmin"

# submit and manage Vertex jobs
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.user"

# read secrets
gcloud secrets add-iam-policy-binding wandb-api-key \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding hf-token \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/secretmanager.secretAccessor"

# write logs
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/logging.logWriter"
```

**GCP products:** **IAM** (Identity and Access Management)

Key concepts:
- Principle of least privilege: grant only what's needed, nothing more
- `roles/storage.objectAdmin` ≠ `roles/storage.admin` — the latter includes bucket management
- Project-level vs resource-level bindings — where possible, grant at resource level (e.g., secret) not project level
- Predefined roles vs custom roles — custom roles let you combine specific permissions

---

### Step 6 — Set up Workload Identity Federation for GitHub Actions (30 min)

This lets GitHub Actions push to Artifact Registry without storing service account key files.
Key files can be leaked, forgotten, or over-scoped. WIF uses short-lived OIDC tokens instead.

```bash
# create a WIF pool
gcloud iam workload-identity-pools create github-pool \
  --location=global \
  --display-name="GitHub Actions Pool"

# create a provider in the pool
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --display-name="GitHub provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# allow the GitHub repo to impersonate the build service account
# (create a separate SA for CI/CD)
gcloud iam service-accounts create cicd-sa \
  --display-name="CI/CD Service Account"

CICD_SA="cicd-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com"
POOL_ID=$(gcloud iam workload-identity-pools describe github-pool \
  --location=global --format="value(name)")

gcloud iam service-accounts add-iam-policy-binding $CICD_SA \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/weyj4/my-model"

# grant cicd-sa permission to push to Artifact Registry
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:$CICD_SA" \
  --role="roles/artifactregistry.writer"
```

Store these values as GitHub Actions secrets:
- `WIF_PROVIDER`: the full provider resource name
- `GCP_SA_EMAIL`: `cicd-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com`
- `GCP_PROJECT_ID`: your project ID

**GCP products:** **Workload Identity Federation**

---

### Step 7 — Update `.github/workflows/docker.yml`

```yaml
name: Build and Push to Artifact Registry

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write  # required for WIF

    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_SA_EMAIL }}

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker us-central1-docker.pkg.dev

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/my-model/my-model:latest
          cache-from: type=gha,scope=my-model
          cache-to: type=gha,mode=max,scope=my-model
```

---

### Step 8 — Update Dockerfile

Replace the RunPod base image with Google's Deep Learning Container (DLC).
DLCs come pre-installed with CUDA, cuDNN, PyTorch, and are tested on GCP hardware.

```dockerfile
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-4.py311:latest

WORKDIR /app

# Vertex mounts GCS buckets automatically at /gcs/BUCKET_NAME
ENV HF_HOME=/gcs/YOUR_PROJECT_ID-ml-data/hf_cache
ENV WANDB_DIR=/gcs/YOUR_PROJECT_ID-ml-data/wandb
ENV TORCH_HOME=/gcs/YOUR_PROJECT_ID-ml-data/torch_cache
ENV CHECKPOINT_DIR=/gcs/YOUR_PROJECT_ID-ml-data/checkpoints

COPY pyproject.toml ./
RUN pip install uv && uv sync --frozen

COPY . .

CMD ["bash", "scripts/train_fineweb.sh"]
```

Note: Vertex AI automatically mounts GCS buckets at `/gcs/BUCKET_NAME` inside your container.
No GCSFuse setup needed. Your existing `mmap_mode='r'` code works unchanged.

---

### Step 9 — Update `scripts/train_fineweb.sh`

```bash
#!/bin/bash
set -e
[ -f .env ] && source .env

export TORCH_HOME=${TORCH_HOME:-/gcs/YOUR_PROJECT_ID-ml-data/torch_cache}

# Vertex mounts GCS at /gcs/
DATA_PATH="/gcs/YOUR_PROJECT_ID-ml-data/data/fineweb_2b5.npy"
NUM_TOKENS=2500000000

# RUN_NAME, BATCH_SIZE, LR injected as env vars from Vertex job spec
RUN_NAME=${RUN_NAME:-"baseline"}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-4e-4}

if [ ! -f "$DATA_PATH" ]; then
    echo "Token file not found, pretokenizing..."
    python scripts/pretokenize.py \
        --num_tokens $NUM_TOKENS \
        --output $DATA_PATH
else
    echo "Token file found at $DATA_PATH, skipping pretokenization"
fi

python -m gpt2.train \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --num_tokens $NUM_TOKENS \
    --dataset fineweb_file \
    --lr $LR
```

---

### Step 10 — Update `gpt2/config.py`

```python
import os

@dataclass
class TrainingConfig:
    # ...existing fields...
    checkpoint_dir: str = os.environ.get(
        "CHECKPOINT_DIR", 
        "/gcs/YOUR_PROJECT_ID-ml-data/checkpoints"
    )
    data_path: str = os.environ.get(
        "DATA_PATH",
        "/gcs/YOUR_PROJECT_ID-ml-data/data/fineweb_2b5.npy"
    )
```

---

### Step 11 — Submit your first Vertex AI Custom Training Job

```bash
# via gcloud CLI
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=flash-bs32 \
  --worker-pool-spec=\
machine-type=a2-highgpu-1g,\
accelerator-type=NVIDIA_TESLA_A100,\
accelerator-count=1,\
replica-count=1,\
container-image-uri=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/my-model/my-model:latest \
  --env-vars=RUN_NAME=flash-bs32,BATCH_SIZE=32,LR=4e-4 \
  --service-account=training-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --enable-web-access

# monitor logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

Or via Python SDK (better for programmatic experiment launching):

```python
from google.cloud import aiplatform

aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name="flash-bs32",
    container_uri="us-central1-docker.pkg.dev/YOUR_PROJECT_ID/my-model/my-model:latest",
)

model = job.run(
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    environment_variables={
        "RUN_NAME": "flash-bs32",
        "BATCH_SIZE": "32",
        "LR": "4e-4",
    },
    service_account="training-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com",
)
```

**GCP products:** **Vertex AI Custom Training**

Key concepts:
- Jobs are fire-and-forget — submit and come back when done
- Logs stream to Cloud Logging automatically, no setup needed
- `FLEX_START` scheduling strategy: job waits for GPU availability rather than failing immediately — great for non-urgent experiment runs
- Spot VMs: ~60-80% cheaper, can be preempted — only viable with checkpoint restarts
- Hyperparameter Tuning Jobs: Vertex runs N jobs in parallel with different configs, finds best

---

## Part 2: Observability — Do you still need W&B?

**Short answer:** Keep W&B for now, but Vertex Experiments is worth adding alongside it.

### Cloud Logging (automatic, free)

All stdout/stderr from your training container streams to Cloud Logging automatically.
No code changes needed. Query logs:

```bash
# tail live logs
gcloud logging read \
  'resource.type="aiplatform.googleapis.com/CustomJob"' \
  --format="value(textPayload)" \
  --limit=50 \
  --freshness=1h

# filter for your loss output
gcloud logging read \
  'resource.type="aiplatform.googleapis.com/CustomJob" AND
   textPayload:"Train loss"' \
  --format="value(textPayload)"
```

**GCP products:** **Cloud Logging** (formerly Stackdriver Logging)

### Cloud Monitoring (automatic, free)

GPU utilization, memory, temperature, and power draw stream to Cloud Monitoring dashboards
automatically when running on Vertex. You can create custom dashboards in the console,
set alert policies (e.g., alert if GPU utilization drops below 80% for 10 minutes),
and create log-based metrics (extract loss values from logs and plot them as a metric).

**GCP products:** **Cloud Monitoring** (formerly Stackdriver Monitoring)

### Vertex AI Experiments (native ML tracking)

The GCP-native alternative to W&B. Log metrics, parameters, and artifacts directly to Vertex.
Add this to your training loop:

```python
from google.cloud import aiplatform

aiplatform.init(
    project="YOUR_PROJECT_ID",
    location="us-central1",
    experiment="gpt2-pretraining"
)

with aiplatform.start_run(run_name=train_cfg.wandb_run_name):
    # in your eval loop:
    aiplatform.log_metrics({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "train/tokens_per_sec": tokens_per_sec,
    })
    aiplatform.log_params({
        "batch_size": train_cfg.batch_size,
        "lr": train_cfg.lr,
        "use_flash": model_cfg.use_flash,
    })
```

Vertex Experiments gives you run comparison tables, parallel coordinates plots, and
artifact lineage — similar to W&B but native in GCP and integrated with Vertex Model Registry.

For your purposes: **run both W&B and Vertex Experiments simultaneously** for a while.
You'll get to compare the two UIs and know both systems, which is good for interviews.

**GCP products:** **Vertex AI Experiments**

### Looker / Looker Studio

Looker (enterprise BI) requires a license and is overkill here. **Looker Studio** (formerly
Data Studio) is free and can visualize BigQuery tables — useful if you're writing training
metrics to BigQuery (see bonus section). Not a W&B replacement but complementary for
batch analytics over many runs.

---

## Part 3: GPU Selection — A100 vs H100 vs L4

### Why GCP doesn't have A40s

GCP uses NVIDIA's datacenter A100 (Ampere architecture) rather than the A40.
The A40 is a workstation/rendering GPU that RunPod runs because it's cheaper.
The A100 is the datacenter-class Ampere GPU with NVLink and higher HBM bandwidth.
Architecturally they're similar but the A100 is better for training.

### GCP GPU options for your workload

| GPU | Machine type | VRAM | On-demand $/hr | Notes |
|-----|-------------|------|----------------|-------|
| L4 | g2-standard-* | 24GB | ~$0.70 | Cheap, good for inference/small training |
| A100 40GB | a2-highgpu-1g | 40GB | ~$3.67 | **Best starting point for your use case** |
| A100 80GB | a2-ultragpu-1g | 80GB | ~$5.50 | Double VRAM, worth it for larger models |
| H100 80GB | a3-highgpu-1g | 80GB | ~$11.00 | ~3x faster than A100 for training |
| H200 141GB | a3-ultragpu-1g | 141GB | ~$15.00+ | Newest, massive VRAM |

### A100 vs H100 — is H100 worth the price?

For your 163M GPT-2 model: **probably not**. Here's the math.

The H100's advantages over A100:
- ~3x higher tensor core throughput (FP16/BF16 matmuls)
- NVLink 4.0 (critical for multi-GPU, less relevant for single GPU)
- Flash Attention 3 support

But your workload at 163M params with Flash Attention at batch=32 is likely already
**compute-bound near the A100's ceiling**. The H100's extra throughput would show up
as faster steps, but not necessarily 3x faster given memory bandwidth and overhead costs.

Realistic estimate for your specific workload:
- A100 40GB: baseline
- H100 80GB: ~2x faster in practice (not 3x), costs 3x more
- Net result: H100 costs ~50% more per token trained

For your current learning stage, A100 is the right call. H100 starts making sense when:
- Your model is large enough to be compute-bound on the A100's tensor cores
- You're doing multi-GPU runs where NVLink matters
- You're iteration-rate constrained and time is genuinely the bottleneck

### GCP A100 availability reality

GCP quota starts at 0 for new projects — you need to request it. This is the primary
friction point:

```bash
# check your current GPU quotas
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -i nvidia

# request quota increase via Cloud Console:
# IAM & Admin → Quotas → filter for "NVIDIA_TESLA_A100" → request increase
```

Recommended regions for A100 availability: `us-central1`, `us-east4`, `europe-west4`.
us-central1 is the safest starting point — most quota available.

For T4 (your Workbench issue): T4s are heavily oversubscribed on Workbench/Colab.
For Vertex Custom Training they're much more available since it's a different quota pool.

### FLEX_START: the availability problem solver

Instead of failing when GPUs aren't immediately available, use FLEX_START scheduling:

```python
job.run(
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    scheduling_strategy="FLEX_START",  # wait up to...
    max_wait_duration=7200,  # ...2 hours for capacity
)
```

The job queues and starts when capacity is available. Good for non-urgent training runs.

---

## Part 4: Bonus GCP Surface Area (for interview depth)

### BigQuery: training metrics as a queryable data warehouse

Write your training metrics to BigQuery in addition to W&B:

```python
from google.cloud import bigquery

client = bigquery.Client()
table_id = "YOUR_PROJECT_ID.ml_metrics.training_runs"

rows = [{
    "run_name": train_cfg.wandb_run_name,
    "step": global_step,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "tokens_per_sec": tokens_per_sec,
    "batch_size": train_cfg.batch_size,
    "use_flash": model_cfg.use_flash,
    "timestamp": datetime.utcnow().isoformat(),
}]
client.insert_rows_json(table_id, rows)
```

Then query across all your runs:
```sql
SELECT run_name, MAX(tokens_per_sec) as peak_tps, MIN(val_loss) as best_val_loss
FROM ml_metrics.training_runs
GROUP BY run_name
ORDER BY best_val_loss ASC
```

This is exactly what you already did with your lm_eval eval tracking pipeline.
**GCP products:** **BigQuery**

### Vertex AI Pipelines: orchestrate multi-step workflows

Codify the full pipeline: pretokenize → train → eval → register as a versioned pipeline.

```python
from kfp import dsl
from google.cloud.aiplatform import pipeline_jobs

@dsl.pipeline(name="gpt2-training-pipeline")
def training_pipeline(
    batch_size: int = 32,
    num_tokens: int = 2_500_000_000,
):
    pretokenize_op = pretokenize_component(num_tokens=num_tokens)
    
    train_op = train_component(
        data_path=pretokenize_op.outputs["data_path"],
        batch_size=batch_size,
    ).after(pretokenize_op)
    
    eval_op = eval_component(
        checkpoint=train_op.outputs["checkpoint"],
    ).after(train_op)
```

**GCP products:** **Vertex AI Pipelines** (backed by Kubeflow Pipelines)

### Vertex AI Model Registry: version your trained artifacts

After training, register your checkpoint as a versioned model:

```python
model = aiplatform.Model.upload(
    display_name="gpt2-163m-flash-bs32",
    artifact_uri="gs://YOUR_PROJECT_ID-ml-data/checkpoints/",
    serving_container_image_uri="us-docker.pkg.dev/.../serving:latest",
)
```

This creates a lineage from training job → model artifact → deployment.
**GCP products:** **Vertex AI Model Registry**

### Vertex AI Endpoints: serve your trained model

Deploy a registered model to a managed endpoint:

```python
endpoint = model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)

# send a prediction
response = endpoint.predict(instances=[{"prompt": "Every effort moves you"}])
```

**GCP products:** **Vertex AI Endpoints**

### Cloud Build: GCP-native CI/CD alternative to GitHub Actions

GCP has its own CI/CD system that can trigger on pushes to Cloud Source Repositories
or GitHub. Worth knowing exists even if you keep GitHub Actions.

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'us-central1-docker.pkg.dev/$PROJECT_ID/my-model/my-model:$COMMIT_SHA', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'us-central1-docker.pkg.dev/$PROJECT_ID/my-model/my-model:$COMMIT_SHA']
```

**GCP products:** **Cloud Build**

### Pub/Sub + Cloud Functions: event-driven training triggers

When new data lands in GCS, automatically trigger a training run:

```
GCS object finalize event → Pub/Sub topic → Cloud Functions → Vertex AI job submission
```

This is the production pattern for continuous training pipelines.
**GCP products:** **Pub/Sub**, **Cloud Functions**

### GKE: when Vertex feels too managed

For multi-GPU distributed training where you want more control than Vertex provides,
Google Kubernetes Engine with GPU node pools is the alternative. You'd run DDP or FSDP
training with `torchrun` across pods. This is how large-scale pretraining is typically done.

```bash
# create a GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=ml-cluster \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --num-nodes=2
```

**GCP products:** **Google Kubernetes Engine (GKE)**

### Cloud Billing: budget alerts (do this immediately)

```bash
# create a budget alert so you don't get surprised
# (do this in Cloud Console: Billing → Budgets & Alerts)
# Set: $50/month alert at 50%, 90%, 100% of budget
```

GPU instances are expensive. Budget alerts are the first thing to set up on any GCP project.

---

## GCP Products Touched Summary

| Product | What you use it for |
|---------|-------------------|
| Cloud Storage (GCS) | Dataset storage, checkpoints, wandb artifacts |
| Artifact Registry | Container image storage |
| Vertex AI Custom Training | Running training jobs |
| Vertex AI Experiments | Native ML experiment tracking |
| Vertex AI Model Registry | Versioning trained models |
| Vertex AI Endpoints | Serving trained models |
| Vertex AI Pipelines | Orchestrating multi-step workflows |
| Secret Manager | WANDB_API_KEY, HF_TOKEN |
| IAM | Service accounts, roles, WIF |
| Workload Identity Federation | Keyless GHA → GCP auth |
| Cloud Logging | Training logs, debugging |
| Cloud Monitoring | GPU metrics, dashboards, alerts |
| BigQuery | Training metrics warehouse, SQL queries across runs |
| Cloud Build | GCP-native CI/CD |
| Pub/Sub | Event-driven triggers |
| Cloud Functions | Serverless trigger handlers |
| GKE | Multi-GPU distributed training |
| Cloud Billing | Budget alerts |
