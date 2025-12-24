# Training Pipeline

Kubeflow-based ML training pipeline with Ray distributed training and MLflow experiment tracking.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │    Data      │───▶│   Feature    │───▶│   Baseline   │                   │
│  │ Preparation  │    │  Retrieval   │    │   Training   │                   │
│  │              │    │   (Feast)    │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                 │                            │
│                                                 ▼                            │
│                                         ┌──────────────┐                    │
│                                         │  HPO Tuning  │                    │
│                                         │  (Ray Tune)  │                    │
│                                         │  20 trials   │                    │
│                                         └──────┬───────┘                    │
│                                                │                            │
│                                                ▼                            │
│                                         ┌──────────────┐                    │
│                                         │    Model     │                    │
│                                         │ Evaluation & │                    │
│                                         │ Registration │───▶ MLflow        │
│                                         └──────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Build training images
make build-images IMAGE_TAG=v1.0

# 2. Compile pipeline
make compile-pipeline IMAGE_TAG=v1.0

# 3. Run pipeline (via Kubeflow UI)
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
# Open http://localhost:8080, upload spam_detection_pipeline.yaml
```

## Directory Structure

```
training/
├── docker/
│   ├── data-prep/       # Data preparation container
│   ├── ray-train/       # Ray HPO training container
│   └── mlflow-ops/      # Model evaluation container
├── pipeline/
│   ├── components/      # Kubeflow components
│   └── spam_detection_pipeline.py
├── scripts/
│   ├── build_images.sh
│   └── setup.sh
├── Makefile
└── README.md
```

## Features Retrieved from Feast

| Feature View | Features | Description |
|--------------|----------|-------------|
| `email_text_features` | 8 | url_count, word_count, uppercase_ratio, etc. |
| `email_structural_features` | 8 | has_html, subject_length, etc. |
| `email_temporal_features` | 4 | hour_of_day, is_weekend, etc. |
| `email_tfidf_features` | 500 | TF-IDF vectors |
| `sender_domain_features` | 4 | spam_ratio, email_count, etc. |
| **Total** | **524** | |

## Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entity_data_path` | `datasets/processed/emails/all_emails.parquet` | Input data |
| `test_split_ratio` | `0.2` | Test set fraction |
| `num_hpo_trials` | `20` | HPO trial count |
| `f1_threshold` | `0.85` | Min F1 to register model |
| `model_name` | `spam-detector` | MLflow model name |

## Monitoring

```bash
# Kubeflow Pipelines UI
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80

# MLflow UI
kubectl port-forward svc/mlflow-service -n mlflow 5000:5000

# Ray Dashboard (during HPO)
kubectl port-forward <ray-head-pod> -n kubeflow 8265:8265
```

## Makefile Targets

```bash
make build-images    # Build and push Docker images
make compile-pipeline # Compile Kubeflow pipeline
make run-pipeline    # Submit pipeline run
make clean           # Clean generated files
```
