# Design Decisions

This document explains the technology choices and architectural decisions made for the ML Model Lifecycle Management Platform.

## Overview

The platform follows a **microservices architecture** with clear separation of concerns:
- **Training** → **Registry** → **Serving** → **Monitoring**

Each component is independently deployable, scalable, and replaceable.

---

## Model Choice

### XGBoost - Spam Classification Model

**Why XGBoost?**
- **High accuracy**: Consistently top performer on tabular/structured data
- **Fast inference**: Sub-millisecond predictions (critical for real-time spam filtering)
- **Interpretable**: Feature importance built-in for debugging
- **ONNX export**: Easy conversion for Triton deployment
- **Proven for text**: Excellent with TF-IDF features for text classification

**Model Architecture:**
```
Raw Email → Feature Extraction (524 features) → XGBoost Classifier → Spam/Ham
                    │
                    ├── TF-IDF features (text content)
                    ├── Metadata features (sender, subject length, etc.)
                    └── Structural features (HTML ratio, link count, etc.)
```

**Hyperparameter Optimization:**
- **Ray Tune**: Distributed hyperparameter search
- **Search space**: learning_rate, max_depth, n_estimators, subsample
- **Objective**: Maximize F1-score (balances precision/recall for spam)

**Why Not Deep Learning?**
| Consideration | XGBoost Advantage |
|---------------|-------------------|
| Dataset size | ~6,000 emails - too small for transformers to shine |
| Feature engineering | TF-IDF + handcrafted features work extremely well |
| Inference latency | <1ms vs 10-100ms for BERT/transformers |
| Resource cost | CPU-only vs GPU required |
| Interpretability | Clear feature importance vs black box |

**Production Performance:**
- **F1 Score**: >0.95 on test set
- **Inference latency**: <1ms (ONNX on Triton)
- **Model size**: ~5MB (vs 400MB+ for BERT)

---

## Technology Choices

### 1. Kubeflow Pipelines - Training Orchestration

**Why Kubeflow Pipelines?**
- **Native Kubernetes**: Runs as K8s CRDs, no separate cluster needed
- **DAG-based workflows**: Clear visualization of training steps
- **Caching**: Automatic caching of pipeline steps for faster iteration
- **Reproducibility**: Each run is versioned with full lineage

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Airflow | More general-purpose, heavier setup for ML-specific workflows |
| Argo Workflows | Good but lacks ML-specific features (experiment tracking UI) |
| Prefect | Excellent but adds external dependency outside K8s |

---

### 2. Ray - Distributed Computing

**Why Ray?**
- **Unified API**: Same code works locally and distributed
- **Dynamic scaling**: Automatic resource management
- **Ray Train**: Native distributed training support
- **Ray Jobs**: Serverless batch processing

**Use Cases in Platform:**
```
Ray Train  → Distributed model training
Ray Jobs   → Batch inference processing
```

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Spark MLlib | JVM overhead, less Python-native |
| Dask | Good for data, less mature for ML training |
| Horovod | Training only, no batch inference support |

---

### 3. MLflow - Experiment Tracking & Model Registry

**Why MLflow?**
- **De-facto standard**: Wide industry adoption
- **Lightweight**: Simple setup, minimal overhead
- **Model Registry**: Built-in staging/production transitions
- **Framework agnostic**: Works with any ML library

**Key Features Used:**
```python
# Experiment Tracking
mlflow.log_params({"learning_rate": 0.01})
mlflow.log_metrics({"accuracy": 0.95})

# Model Registry
mlflow.register_model(model_uri, "spam-detector")
client.transition_model_version_stage("spam-detector", version, "Production")
```

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Weights & Biases | Excellent but SaaS-based (data residency concerns) |
| Neptune | Similar to W&B, external dependency |

---

### 4. KServe - Model Serving

**Why KServe?**
- **Kubernetes-native**: InferenceService CRD
- **Auto-scaling**: Scale-to-zero and HPA support
- **Multi-framework**: Supports sklearn, TensorFlow, PyTorch, ONNX
- **Canary deployments**: Built-in traffic splitting

**Key Features Used:**
```yaml
# Multi-environment deployment
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: spam-detector-staging  # vs spam-detector (production)
spec:
  predictor:
    minReplicas: 1  # staging
    # minReplicas: 2  # production
```

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Seldon Core | More complex, enterprise-focused |
| TensorFlow Serving | TensorFlow-only |
| TorchServe | PyTorch-only |
| BentoML | Good but less K8s-native |

---

### 5. Triton Inference Server - Model Runtime

**Why Triton?**
- **High performance**: Optimized C++ inference
- **Dynamic batching**: Automatic request batching
- **Multi-model**: Run multiple models in single container
- **ONNX support**: Framework-agnostic model format

**Why ONNX format?**
```
scikit-learn model → ONNX → Triton
```
- Portable across frameworks
- Optimized inference runtime
- Production-proven performance

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Native sklearn | 10x slower than ONNX/Triton |
| ONNX Runtime | Good but Triton adds batching/metrics |

---

### 6. FastAPI - API Gateway

**Why FastAPI?**
- **Async**: High concurrency with asyncio
- **Auto-docs**: OpenAPI/Swagger built-in
- **Type hints**: Pydantic validation
- **Performance**: One of fastest Python frameworks

**Key Design Pattern:**
```python
# Environment routing via headers
@router.post("/predict")
async def predict(request: PredictRequest, env: str = Header("staging")):
    endpoint = staging_url if env == "staging" else production_url
    return await kserve_client.predict(endpoint, features)
```

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Flask | Sync-only, slower |
| Django | Overkill for API gateway |

---

### 7. Terraform - Infrastructure as Code

**Why Terraform?**
- **Multi-cloud**: Works with any provider
- **Declarative**: Desired state configuration
- **State management**: Track infrastructure changes
- **Module ecosystem**: Reusable components

**Structure:**
```
terraform/
├── base-infra/     # AKS, ACR, Storage, Redis
└── ml-platform/    # Kubeflow, MLflow, Ray, KServe
```

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Pulumi | Easy to addopt |

---

### 8. Azure Kubernetes Service (AKS)

**Why AKS?**
- **Managed K8s**: No control plane management
- **Azure integration**: AAD, ACR, Blob Storage
- **Cost optimization**: Spot instances, scale-to-zero

**Key Integrations:**
```
AKS ←→ ACR (Container images)
AKS ←→ Azure Blob (Model artifacts, logs)
AKS ←→ Azure Redis (Feature cache)
```

---

### 9. Istio - Service Mesh

**Why Istio?**
- **Traffic management**: Routing, load balancing
- **Security**: mTLS between services
- **Observability**: Distributed tracing
- **Gateway**: Unified ingress

**Key Use Case:**
```yaml
# Route to staging vs production based on header
VirtualService:
  match:
    - headers:
        X-Environment:
          exact: production
  route:
    - destination: spam-detector
```

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Linkerd | Simpler but less features |
| Nginx Ingress | No service mesh capabilities |

---

### 10. Feast - Feature Store

**Why Feast?**
- **Online/Offline**: Unified feature serving
- **Redis backend**: Low-latency online serving
- **Time-travel**: Point-in-time feature retrieval

**Architecture:**
```
Training:  Feast offline store (Parquet) → Ray
Inference: Feast online store (Redis) → API Gateway
```

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| Tecton | SaaS-based, expensive |
| Hopsworks | More complex setup |
| Custom | Maintenance burden |

---

### 11. Evidently - Drift Detection

**Why Evidently?**
- **Comprehensive metrics**: PSI, KS-test, data quality
- **Visual reports**: HTML dashboards
- **Lightweight**: No server required

**Drift Detection Strategy:**
```python
# Population Stability Index (PSI) for feature drift
# Kolmogorov-Smirnov test for distribution shift
# Alert if drift_score > 0.1
```

**Alternatives Considered:**
| Alternative | Why Not Chosen |
|-------------|----------------|
| WhyLabs | SaaS-based |
| Great Expectations | More data validation than drift |
| Custom | Evidently covers most cases |

---

## Architectural Patterns

### 1. Multi-Environment Deployment
```
Staging (safe experiments) ←→ Production (high availability)
                    ↓
            Header-based routing (X-Environment)
```

### 2. GitOps-Ready
- All infrastructure as code
- Kubernetes manifests in Git
- GitHub Actions for CI/CD

### 3. Observability Stack
```
Metrics:  Prometheus (scrape) → Custom dashboards
Logs:     Structured JSON → Azure Blob (Parquet)
Tracing:  Istio → Jaeger (optional)
Alerts:   Drift webhook → Slack/PagerDuty
```

### 4. Scalability Design
```
Component         Min    Max    Trigger
─────────────────────────────────────────
API Gateway       2      10     CPU > 70%
Feature Trans.    2      10     CPU > 70%
Staging Model     1      3      Requests
Production Model  2      10     Requests
```

---

## Trade-offs & Future Improvements

| Decision | Trade-off | Future Improvement |
|----------|-----------|-------------------|
| ONNX conversion | Extra build step | Auto-conversion in CI |
| Single cluster | No geo-redundancy | Multi-region with Federation |
| Azure-specific | Vendor lock-in | Abstract storage layer |
| Manual approval | Slower deployments | Full auto-promotion with tests |

---

## Summary

This platform demonstrates enterprise-grade MLOps with:
- ✅ **Automation**: End-to-end pipeline from data to production
- ✅ **Scalability**: Kubernetes-native horizontal scaling
- ✅ **Reliability**: Multi-environment, health checks, retries
- ✅ **Observability**: Metrics, logging, drift detection
- ✅ **Security**: API auth, RBAC, namespaced resources
- ✅ **Maintainability**: IaC, GitOps, modular design
