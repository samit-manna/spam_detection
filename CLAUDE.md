# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade email spam detection system demonstrating end-to-end MLOps on Azure Kubernetes Service (AKS). The model is an XGBoost classifier with 524 engineered features trained on the SpamAssassin Public Corpus (~6,000 emails), exported to ONNX for serving via Triton/KServe.

## Build & Test Commands

Each subsystem has its own Makefile. Run `make help` in any directory to see available targets.

### Training Pipeline
```bash
cd training
make compile-pipeline      # Recompile Kubeflow pipeline DSL → YAML
make build-images          # Build Docker images for pipeline components
```

### Model Serving
```bash
cd model-serving
make build-images          # Build api-gateway, feature-transformer, etc.
make deploy-staging        # Deploy to staging KServe
make test                  # Run integration tests (requires K8s)
make test-local            # Run tests via docker-compose
```

### Data Pipeline
```bash
cd data-pipeline
make run-pipeline          # Execute all 3 Ray job stages sequentially
make verify-data           # Verify outputs in Azure Blob Storage
```

### Monitoring
```bash
cd monitoring
make test                  # Run unit tests
make test-cov              # Tests with coverage report
make generate-baseline     # Generate baseline from training data
make run-drift-detection   # Run drift detection locally
```

## Architecture

The platform follows a linear MLOps flow:

```
Data Pipeline → Training Pipeline → Model Registry → Model Serving → Monitoring
  (Ray Jobs)    (Kubeflow + Ray)      (MLflow)       (KServe/Triton)  (Evidently)
```

### Kubernetes Namespaces
| Namespace | Purpose |
|-----------|---------|
| `kubeflow` | Training pipelines |
| `mlflow` | Experiment tracking & model registry |
| `ray` | Distributed training & batch inference |
| `kserve` | Model serving (staging + production) |
| `serving` | API gateway & feature transformer |
| `feast` | Feature store & materialization jobs |
| `monitoring` | Drift detection CronJobs |

### Key Components

**Data Pipeline** (`data-pipeline/`): Three sequential Ray jobs — download SpamAssassin corpus → parse emails to Parquet → extract 524 features. Outputs land in Azure Blob Storage and are materialized to Feast (Redis).

**Training Pipeline** (`training/pipeline/`): 8-step Kubeflow pipeline defined in `spam_detection_pipeline.py`. Compiled to `spam_detection_pipeline.yaml`. Steps: data prep → feature retrieval → baseline training → HPO (Ray Tune, 20 trials) → evaluation → validation → comparison vs staging → MLflow promotion. Quality gates: F1≥0.85, AUC≥0.90, latency<10ms, max 5% regression vs current staging model.

**Model Serving** (`model-serving/`):
- `api-gateway/` — FastAPI, single `/predict` endpoint, routes staging/prod via `X-Environment` header, scales 2–10 pods
- `feature-transformer/` — Real-time feature extraction, interfaces with Feast/Redis
- `inference-service/` — KServe manifests; staging (min 1, max 3 replicas), production (min 2, max 10)
- `batch-inference/` — Ray Jobs for bulk predictions from Azure Blob

**Monitoring** (`monitoring/`): Evidently-based drift detector runs as an hourly Kubernetes CronJob. Compares live inference data against baselines stored in Azure Blob. PSI thresholds: <0.1 = no drift, 0.1–0.2 = minor, >0.2 = significant drift.

### CI/CD Workflows (`.github/workflows/`)
- `model-serving.yaml` — Triggered on `model-serving/**` changes: build images → staging deploy → integration tests → production deploy (manual approval required)
- `model-deploy.yaml` — Manual trigger: verify MLflow staging → export ONNX → deploy staging → smoke tests → promote to production
- `rollback.yaml` — Manual rollback to previous model version

### Infrastructure (`terraform/`)
Three Terraform modules applied in order:
1. `base-infra/` — AKS, ACR, Azure Blob Storage, Redis
2. `ml-platform/` — Kubeflow, MLflow, Ray, KServe (via Helm)
3. `serving-infra/` — Istio, Knative, cert-manager

## Key File Locations

| What | Where |
|------|-------|
| Pipeline DSL (source of truth) | `training/pipeline/spam_detection_pipeline.py` |
| Compiled pipeline YAML | `training/pipeline/spam_detection_pipeline.yaml` |
| Pipeline components | `training/pipeline/components/` (8 components) |
| API gateway entry point | `model-serving/api-gateway/app/main.py` |
| Feature transformer entry | `model-serving/feature-transformer/app/main.py` |
| Drift detector | `monitoring/drift_detector/detector.py` |
| Postman collection | `model-serving/postman/` |
| Integration tests | `model-serving/tests/` |
| Monitoring tests | `monitoring/tests/` |
| End-to-end demo | `scripts/demo.sh` |

## Important Conventions

- The compiled `spam_detection_pipeline.yaml` is checked into the repo. After editing `spam_detection_pipeline.py`, always run `make compile-pipeline` to regenerate it.
- Multi-environment routing uses the `X-Environment` HTTP header (`staging` or `production`).
- Model promotion path: local training → MLflow Staging → MLflow Production → KServe deployment.
- ONNX export (`model-serving/model-export/`) is required before deploying a new model version to KServe/Triton.
