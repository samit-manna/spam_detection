# ML Model Lifecycle Management Platform

A production-grade MLOps platform for spam detection, demonstrating end-to-end machine learning lifecycle management on Azure Kubernetes Service.

## 🎯 Overview

This platform provides automated model training, multi-environment deployment, real-time and batch inference, and continuous monitoring with drift detection.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              ML LIFECYCLE MANAGEMENT PLATFORM                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Kubeflow  │───▶│    Ray      │───▶│   MLflow    │───▶│   KServe    │              │
│  │  Pipelines  │    │  Training   │    │  Registry   │    │  Serving    │              │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘              │
│        │                  │                  │                  │                       │
│        │                  │                  │                  ▼                       │
│        │                  │                  │          ┌─────────────┐                 │
│        │                  │                  │          │ API Gateway │◀── Requests    │
│        │                  │                  │          │  (FastAPI)  │                 │
│        │                  │                  │          └──────┬──────┘                 │
│        │                  │                  │                 │                        │
│        ▼                  ▼                  ▼                 ▼                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           Azure Blob Storage                                      │  │
│  │   training-data/  │  models/  │  baselines/  │  inference-logs/  │  drift-reports/│  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                              │
│                                          ▼                                              │
│                                 ┌─────────────────┐                                     │
│                                 │ Drift Detection │                                     │
│                                 │    (CronJob)    │                                     │
│                                 └─────────────────┘                                     │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## ✨ Features

| Category | Feature | Implementation |
|----------|---------|----------------|
| **Training** | Automated pipelines | Kubeflow Pipelines |
| | Distributed training | Ray |
| | Experiment tracking | MLflow |
| **Deployment** | Multi-environment | KServe (staging/production) |
| | Feature serving | Feast + Redis |
| | Real-time inference | FastAPI API Gateway |
| | Batch inference | Ray Jobs |
| **Monitoring** | Drift detection | PSI/KS tests + Evidently |
| | Inference logging | Azure Blob (Parquet) |
| | Alerting | Webhook + structured logs |
| **Infrastructure** | Container orchestration | AKS (Kubernetes) |
| | Service mesh | Istio |
| | IaC | Terraform |

## 🏗️ Architecture

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| `training/` | Kubeflow + Ray + MLflow | Model training pipeline |
| `model-serving/` | KServe + FastAPI | Inference services |
| `monitoring/` | Custom + Evidently | Drift detection |
| `data-pipeline/` | Ray Jobs | Data preprocessing |
| `terraform/` | Terraform | Infrastructure as Code |

### Kubernetes Namespaces

```
kubeflow    - Training pipelines
mlflow      - Experiment tracking & model registry
ray         - Distributed computing
kserve      - Model serving (staging + production)
serving     - API gateway & feature transformer
monitoring  - Drift detection jobs
```

## 🚀 Quick Start

### Prerequisites

- Azure subscription
- Azure CLI (`az`)
- Terraform
- kubectl
- Docker

### 1. Deploy Infrastructure

```bash
# Base infrastructure (AKS, ACR, Storage)
cd terraform/base-infra
terraform init && terraform apply

# ML platform (Kubeflow, MLflow, Ray, KServe)
cd ../ml-platform
terraform init && terraform apply
```

### 2. Build & Deploy Services

```bash
cd model-serving
make build-images IMAGE_TAG=v1.0
make deploy-all IMAGE_TAG=v1.0
```

### 3. Run Training Pipeline

```bash
cd training
make run-pipeline
```

### 4. Test Inference

```bash
# Port-forward API gateway
kubectl port-forward svc/api-gateway -n serving 8000:80

# Test prediction (staging)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-operator-key" \
  -d '{"email_id": "1", "subject": "Win $1000!", "body": "Click here now", "sender": "promo@test.com"}'

# Test prediction (production)
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: test-operator-key" \
  -H "X-Environment: production" \
  -d '{"email_id": "1", "subject": "Meeting tomorrow", "body": "Hi, can we meet?", "sender": "colleague@company.com"}'
```

### 5. Monitor Drift

```bash
cd monitoring
make trigger-job    # Manual drift check
make view-metrics   # View results
```

## 📁 Project Structure

```
spam_detection/
├── README.md                 # This file
├── terraform/
│   ├── base-infra/          # AKS, ACR, Storage, Redis
│   └── ml-platform/         # Kubeflow, MLflow, Ray, KServe
├── training/
│   ├── pipeline/            # Kubeflow pipeline definition
│   └── docker/              # Training container images
├── model-serving/
│   ├── api-gateway/         # FastAPI inference API
│   ├── feature-transformer/ # Feature extraction service
│   ├── inference-service/   # KServe model deployments
│   ├── batch-inference/     # Ray batch processing
│   └── feast/               # Feature store config
├── monitoring/
│   ├── baseline/            # Baseline generation
│   ├── drift_detector/      # Drift detection logic
│   └── tests/               # Unit tests
├── data-pipeline/           # Data preprocessing jobs
└── scripts/                 # Utility scripts
```

## 🔑 Key Design Decisions

### 1. Multi-Environment Deployment
- **Staging**: Lower resources, scale-to-zero, for testing
- **Production**: Higher resources, min 2 replicas, HA
- Switch via `X-Environment` header

### 2. Scalability
- HPA on all components (2-10 replicas)
- Ray for distributed training/batch
- Async inference logging

### 3. Monitoring Strategy
- **Baseline**: Generated from training data (PSI histograms)
- **Drift Detection**: Hourly CronJob comparing production vs baseline
- **Metrics**: PSI, KS-test, aggregate drift score

### 4. Security
- API key authentication
- Kubernetes RBAC
- Azure managed identities (where possible)

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single email prediction |
| `/predict/batch-sync` | POST | Batch prediction (≤100) |
| `/batch/submit` | POST | Async batch job (Ray) |
| `/batch/{job_id}` | GET | Batch job status |
| `/metrics/drift` | GET | Drift summary |
| `/health` | GET | Service health check |

## 🧪 Testing

```bash
# API Gateway tests
cd model-serving && make test

# Monitoring tests
cd monitoring && make test

# Run demo
./scripts/demo.sh
```

## 📈 Demo Script

Run the end-to-end demo:

```bash
./scripts/demo.sh
```

This demonstrates:
1. ✅ Service health checks
2. ✅ Staging vs Production inference
3. ✅ Batch predictions
4. ✅ Drift detection
5. ✅ Scalability (HPA status)

## 📚 Documentation

| Module | README |
|--------|--------|
| Training | [training/README.md](training/README.md) |
| Model Serving | [model-serving/README.md](model-serving/README.md) |
| Monitoring | [monitoring/README.md](monitoring/README.md) |
| Infrastructure | [terraform/base-infra/README.md](terraform/base-infra/README.md) |

## 🛠️ Technologies

- **Cloud**: Azure (AKS, ACR, Blob Storage, Redis)
- **ML Framework**: scikit-learn, ONNX
- **Orchestration**: Kubernetes, Kubeflow, Ray
- **Serving**: KServe, Triton, FastAPI
- **Monitoring**: Custom drift detection, Evidently
- **IaC**: Terraform
- **Service Mesh**: Istio

## 📝 License

MIT
