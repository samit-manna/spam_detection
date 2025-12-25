# Model Serving

Production-grade model serving with KServe, featuring multi-environment support (staging/production), auto-scaling, and real-time monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL SERVING PLATFORM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         API Gateway                                  │    │
│  │                  X-Environment: staging | production                 │    │
│  └──────────────────────────┬──────────────────────────────────────────┘    │
│                             │                                                │
│            ┌────────────────┴────────────────┐                              │
│            │                                 │                              │
│            ▼                                 ▼                              │
│  ┌─────────────────────┐          ┌─────────────────────┐                   │
│  │  spam-detector-     │          │  spam-detector      │                   │
│  │    staging          │          │   (production)      │                   │
│  │  (min:1, max:3)     │          │  (min:2, max:10)    │                   │
│  └─────────────────────┘          └─────────────────────┘                   │
│                                                                              │
│  ┌─────────────────────┐          ┌─────────────────────┐                   │
│  │  feature-           │          │  drift-detector     │                   │
│  │   transformer       │          │   (CronJob)         │                   │
│  │  (min:2, max:10)    │          │                     │                   │
│  └─────────────────────┘          └─────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

All operations are available via Make targets. Run from the `model-serving/` directory:

### One Command Setup

```bash
# Run full setup: build → deploy gateway → deploy transformer → deploy feast → export model → deploy staging → test → deploy prod
make all
```

### Step-by-Step Setup

```bash
# 1. Check prerequisites and build Docker images
make check-env
make build-images

# 2. Deploy supporting services
make deploy-api-gateway    # FastAPI gateway
make deploy-transformer    # Feature transformer
make deploy-feast          # Feast materialization cronjob

# 3. Export model from MLflow and deploy
make export-model          # Export to ONNX format
make deploy-staging        # Deploy to staging

# 4. Run integration tests
make test

# 5. Deploy to production (after staging validation)
make deploy-prod
```

## Make Targets Reference

| Target | Description |
|--------|-------------|
| **Quick Start** | |
| `make all` | Full setup: build → deploy all → export → test → production |
| `make setup` | Same as `make all` |
| `make deploy-all` | Deploy all components (gateway, transformer, feast, staging, prod) |
| `make help` | Show all available targets |
| **Infrastructure** | |
| `make infra` | Deploy serving infrastructure (namespaces, secrets) |
| `make check-env` | Verify Terraform outputs and prerequisites |
| **Build** | |
| `make build-images` | Build and push all Docker images to ACR |
| **Deployment** | |
| `make deploy-api-gateway` | Deploy API gateway |
| `make deploy-transformer` | Deploy feature transformer |
| `make deploy-feast` | Deploy Feast materialization cronjob |
| `make deploy-staging` | Deploy staging InferenceService |
| `make deploy-prod` | Deploy production InferenceService |
| **Model Lifecycle** | |
| `make export-model` | Export model from MLflow to blob storage |
| `make verify-onnx-conversion` | Validate ONNX conversion quality before deployment |
| `make model-deploy` | Full deployment: export + staging + test + prod |
| **Testing** | |
| `make test` | Run integration tests via public ingress |
| `make test-local-cluster` | Run tests with port-forwarding (debugging) |
| `make materialize` | Trigger Feast feature materialization |
| `make batch-predict` | Submit Ray batch prediction job |
| `make clean` | Remove all deployed resources |
| `make status` | Show status of all components |

## Multi-Environment Deployment

### Using X-Environment Header

```bash
# Get the Istio ingress gateway IP
export GATEWAY_URL=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Default: Staging environment
curl -X POST http://$GATEWAY_URL/predict \
  -H "Host: api.ml-platform.example.com" \
  -H "Content-Type: application/json" \
  -d '{"email_content": "Hello world"}'

# Explicit: Staging environment  
curl -X POST http://$GATEWAY_URL/predict \
  -H "Host: api.ml-platform.example.com" \
  -H "Content-Type: application/json" \
  -H "X-Environment: staging" \
  -d '{"email_content": "Hello world"}'

# Production environment
curl -X POST http://$GATEWAY_URL/predict \
  -H "Host: api.ml-platform.example.com" \
  -H "Content-Type: application/json" \
  -H "X-Environment: production" \
  -d '{"email_content": "Hello world"}'
```

### Batch Prediction

```bash
# Submit batch prediction job via Ray
make batch-predict

# Or via API:
curl -X POST http://$GATEWAY_URL/predict/batch \
  -H "Host: api.ml-platform.example.com" \
  -H "Content-Type: application/json" \
  -H "X-Environment: production" \
  -d '{"emails": ["Email 1 content", "Email 2 content", "Email 3 content"]}'
```

## Components

| Component | Description | Scaling |
|-----------|-------------|---------|
| api-gateway | FastAPI gateway with env routing | HPA 2-10 pods |
| spam-detector-staging | Staging model (safe experiments) | HPA 1-3 pods |
| spam-detector | Production model (high availability) | HPA 2-10 pods |
| feature-transformer | Real-time feature extraction | HPA 2-10 pods |
| drift-detector | Hourly drift monitoring | CronJob |

## Directory Structure

```
model-serving/
├── api-gateway/           # FastAPI gateway with env routing
├── feature-transformer/   # Real-time feature extraction
├── drift-detector/        # Drift monitoring (Evidently)
├── inference-service/     # KServe InferenceService manifests
│   ├── staging-isvc.yaml
│   └── production-isvc.yaml
├── scripts/               # Lifecycle management scripts
│   └── model_lifecycle.py # Export/deploy/promote models
├── tests/                 # Integration tests
└── Makefile               # All deployment targets
```

## API Endpoints

| Endpoint | Method | Headers | Description |
|----------|--------|---------|-------------|
| `/predict` | POST | X-Environment (optional) | Single prediction |
| `/predict/batch` | POST | X-Environment (optional) | Batch inference (Ray) |
| `/health` | GET | - | Health check |
| `/ready` | GET | - | Readiness check |

## Testing

```bash
# Run full integration tests (uses public ingress)
make test

# Run tests with port-forwarding (for debugging)
make test-local-cluster

# Run local tests with docker-compose
make test-local
```

## Monitoring

```bash
# Check component status
make status

# View pod scaling
kubectl get hpa -n serving -n kserve

# View inference logs
kubectl logs -l app=api-gateway -n serving --tail=50

# Check drift detection results
kubectl logs -l app=drift-detector -n monitoring --tail=50
```

## Model Lifecycle Management

The `scripts/model_lifecycle.py` script provides programmatic model lifecycle management:

```bash
# Export model from MLflow to blob storage
python scripts/model_lifecycle.py export \
  --model-name spam-detection-model \
  --model-stage Production

# Deploy to staging
python scripts/model_lifecycle.py deploy \
  --environment staging

# Run smoke tests
python scripts/model_lifecycle.py smoke-test \
  --environment staging

# Promote to production (after successful smoke tests)
python scripts/model_lifecycle.py promote
```

This script is used by the GitHub Actions workflow for automated deployments.

## CI/CD Integration

Model deployments are automated via GitHub Actions:

1. **Trigger**: Push to `model-serving/**` or manual workflow dispatch
2. **Build**: Docker images built and pushed to ACR
3. **Deploy Staging**: InferenceService deployed to staging
4. **Smoke Tests**: Automated validation of staging deployment
5. **Deploy Production**: Auto-promotion after successful smoke tests

See `.github/workflows/model-deploy.yaml` for the full workflow.
