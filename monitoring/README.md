# Model Monitoring Module

Automated drift detection for ML models in production. Compares live inference data against training baselines to detect distribution shifts.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   API Gateway   │────▶│  Inference Logs  │────▶│   Drift Detector    │
│   (predictions) │     │  (Azure Blob)    │     │   (K8s CronJob)     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                                          │
                        ┌──────────────────┐              │
                        │    Baselines     │◀─────────────┤
                        │  (Azure Blob)    │              ▼
                        └──────────────────┘     ┌─────────────────────┐
                                                 │   Drift Reports     │
                                                 │   + Alerts          │
                                                 └─────────────────────┘
```

## Quick Start

```bash
# 1. Build Docker images
make build-images IMAGE_TAG=v0.1

# 2. Generate baseline from training data
make generate-baseline MODEL_NAME=spam-detector MODEL_VERSION=v1

# 3. Deploy drift detection CronJob (runs hourly)
make deploy IMAGE_TAG=v0.1

# 4. Check status
make status

# 5. View latest drift report
make view-metrics
```

## Key Components

| Component | Description |
|-----------|-------------|
| **Baseline Generator** | Computes statistical profiles (mean, std, histograms) from training data |
| **Drift Detector** | Compares production data against baseline using PSI and KS tests |
| **Inference Logger** | Buffers and writes prediction logs to Azure Blob (in api-gateway) |

## Drift Metrics

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| **PSI** | < 0.1 | No drift |
| | 0.1 - 0.2 | Minor drift, monitor |
| | > 0.2 | Significant drift, investigate |
| **KS p-value** | > 0.05 | No significant difference |
| | < 0.05 | Distribution shift detected |

## Makefile Targets

```bash
make help              # Show all targets

# Development
make test              # Run unit tests
make test-cov          # Tests with coverage

# Local Operations
make generate-baseline # Create baseline from training data
make run-drift-detection # Run drift detection locally

# Kubernetes
make deploy            # Deploy CronJob
make status            # Check deployment status
make logs              # View latest job logs
make trigger-job       # Manually trigger detection

# Utilities
make list-baselines    # List stored baselines
make list-reports      # List drift reports
make view-metrics      # Quick drift status
```

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DRIFT_PSI_THRESHOLD` | 0.2 | PSI threshold for drift |
| `DRIFT_ANALYSIS_WINDOW_HOURS` | 24 | Hours of logs to analyze |
| `DRIFT_MIN_SAMPLES` | 100 | Minimum samples required |
| `DRIFT_SCORE_CRITICAL_THRESHOLD` | 0.5 | Critical alert threshold |

## Storage Layout

```
Azure Blob Storage
├── baselines/spam-detector/v1/baseline.json
├── inference-logs/year=2025/month=12/day=24/hour=17/*.parquet
└── drift-reports/spam-detector/latest.json
```

## Example Output

```
============================================================
DRIFT DETECTION SUMMARY
============================================================
Model: spam-detector v1
Samples analyzed: 112

Overall drift detected: YES
Drift score: 85.0%
Features drifted: 8/50

Drifted features: url_count, word_count, uppercase_ratio...
Alerts: 3 (1 CRITICAL, 2 WARNING)
============================================================
```

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_baseline.py -v

# Run synthetic drift scenarios
make test-synthetic
```
