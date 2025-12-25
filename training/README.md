# Training Pipeline

Kubeflow-based ML training pipeline with Ray distributed training and MLflow experiment tracking. Includes automated validation, comparison against staging, and promotion gates.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE (8 Steps)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  1. Data     │───▶│ 2. Feature   │───▶│ 3. Baseline  │                   │
│  │ Preparation  │    │  Retrieval   │    │   Training   │                   │
│  │              │    │   (Feast)    │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                 │                            │
│                                                 ▼                            │
│                                         ┌──────────────┐                    │
│                                         │ 4. HPO       │                    │
│                                         │  (Ray Tune)  │                    │
│                                         │  20 trials   │                    │
│                                         └──────┬───────┘                    │
│                                                │                            │
│                                                ▼                            │
│                                         ┌──────────────┐                    │
│                                         │ 5. Model     │                    │
│                                         │ Evaluation   │───▶ MLflow        │
│                                         │ (visualize)  │    (artifacts)    │
│                                         └──────┬───────┘                    │
│                                                │                            │
│                                                ▼                            │
│                                         ┌──────────────┐                    │
│                                         │ 6. Model     │                    │
│                                         │ Validation   │                    │
│                                         │ (quality)    │                    │
│                                         └──────┬───────┘                    │
│                                                │ if passed                   │
│                                                ▼                            │
│                                         ┌──────────────┐                    │
│                                         │ 7. Model     │                    │
│                                         │ Comparison   │                    │
│                                         │ (vs staging) │                    │
│                                         └──────┬───────┘                    │
│                                                │ if passed                   │
│                                                ▼                            │
│                                         ┌──────────────┐                    │
│                                         │ 8. Model     │                    │
│                                         │ Promotion    │───▶ MLflow        │
│                                         │ (to Staging) │    (registry)     │
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
│   └── mlflow-ops/      # Model evaluation/promotion container
├── pipeline/
│   ├── components/      # Kubeflow components
│   │   ├── data_preparation.py
│   │   ├── feature_retrieval.py
│   │   ├── baseline_training.py
│   │   ├── hpo_tuning.py
│   │   ├── model_evaluation.py    # Metrics & visualizations
│   │   ├── model_validation.py    # Quality gates
│   │   ├── model_comparison.py    # Compare vs staging
│   │   └── model_promotion.py     # Promote to Staging
│   └── spam_detection_pipeline.py
├── scripts/
│   ├── build_images.sh
│   └── setup.sh
├── Makefile
├── README.md
└── PIPELINE_OVERVIEW.md
```

## Pipeline Steps

| Step | Component | Description |
|------|-----------|-------------|
| 1 | `data_preparation` | Load and split data into train/test |
| 2 | `feature_retrieval` | Get 524 features from Feast |
| 3 | `baseline_training` | Quick validation with simple model |
| 4 | `hpo_tuning` | Ray Tune hyperparameter optimization |
| 5 | `model_evaluation` | Test metrics + visualizations (confusion matrix, ROC, PR) |
| 6 | `model_validation` | Quality gates (F1, AUC, latency, sanity) |
| 7 | `model_comparison` | Compare against current staging model |
| 8 | `model_promotion` | Register to MLflow Staging if all gates pass |

## Features Retrieved from Feast

| Feature View | Features | Description |
|--------------|----------|-------------|
| `email_text_features` | 8 | url_count, word_count, uppercase_ratio, etc. |
| `email_structural_features` | 8 | has_html, subject_length, etc. |
| `email_temporal_features` | 4 | hour_of_day, is_weekend, etc. |
| `email_tfidf_features` | 500 | TF-IDF vectors |
| `sender_domain_features` | 4 | spam_ratio, email_count, etc. |
| **Total** | **524** | |

## Validation & Promotion Thresholds

| Threshold | Default | Description |
|-----------|---------|-------------|
| `f1_threshold` | 0.85 | Minimum F1 score for registration |
| `min_auc_roc` | 0.90 | Minimum AUC-ROC |
| `min_precision` | 0.80 | Minimum precision |
| `min_recall` | 0.80 | Minimum recall |
| `max_inference_time_ms` | 10.0 | Max inference latency (ms) |
| `max_f1_regression` | 0.05 | Max F1 drop vs staging (5%) |
| `max_auc_regression` | 0.05 | Max AUC drop vs staging (5%) |

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
make build-images     # Build and push Docker images
make compile-pipeline # Compile Kubeflow pipeline
make run-pipeline     # Submit pipeline run
make clean            # Clean generated files
```

## Model Lifecycle

1. **Training Pipeline** → Model registered in MLflow **Staging**
2. **Manual Review** → Inspect metrics in MLflow UI
3. **Deployment** → Use `model-serving/` to deploy to KServe (separate process)
