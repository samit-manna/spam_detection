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

## Multi-Environment Deployment

### Using X-Environment Header

```bash
# Default: Staging environment
curl -X POST http://$GATEWAY_URL/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"email_content": "Hello world"}'

# Explicit: Staging environment  
curl -X POST http://$GATEWAY_URL/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-Environment: staging" \
  -d '{"email_content": "Hello world"}'

# Production environment
curl -X POST http://$GATEWAY_URL/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-Environment: production" \
  -d '{"email_content": "Hello world"}'
```

### Batch Prediction

```bash
# Batch prediction to production
curl -X POST http://$GATEWAY_URL/api/v1/batch-predict \
  -H "Content-Type: application/json" \
  -H "X-Environment: production" \
  -d '{"emails": ["Email 1 content", "Email 2 content", "Email 3 content"]}'
```

## Quick Start

```bash
# 1. Deploy KServe InferenceServices
kubectl apply -f kubernetes/serving/

# 2. Deploy API Gateway
kubectl apply -f api-gateway/deployment.yaml
kubectl apply -f api-gateway/service.yaml

# 3. Port-forward for testing
kubectl port-forward svc/api-gateway -n serving 8080:8080

# 4. Test inference
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"email_content": "Win a free prize! Click here now!"}'
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
├── api-gateway/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Environment config
│   │   ├── routers/predict.py   # Prediction endpoints
│   │   └── services/kserve_client.py
│   ├── deployment.yaml
│   └── Dockerfile
├── feature-transformer/         # Real-time feature extraction
├── drift-detector/              # Drift monitoring
├── baseline-generator/          # Baseline data generation
└── kubernetes/
    └── serving/                 # KServe manifests
```

## API Endpoints

| Endpoint | Method | Headers | Description |
|----------|--------|---------|-------------|
| `/api/v1/predict` | POST | X-Environment (optional) | Single prediction |
| `/api/v1/batch-predict` | POST | X-Environment (optional) | Batch predictions |
| `/health` | GET | - | Health check |
| `/ready` | GET | - | Readiness check |

## Scaling Configuration

```yaml
# HPA scales based on CPU utilization
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      targetAverageUtilization: 70
```

## Monitoring

```bash
# Check pod scaling
kubectl get hpa -n serving

# View inference logs
kubectl logs -l app=api-gateway -n serving --tail=50

# Check drift detection results
kubectl logs -l app=drift-detector -n monitoring --tail=50
```

## Verification

```bash
# Check all components running
kubectl get pods -n serving
kubectl get pods -n kserve

# Verify InferenceServices
kubectl get inferenceservice -n kserve

# Test environment routing
export GATEWAY_URL=localhost:8080
curl -s http://$GATEWAY_URL/health
curl -X POST http://$GATEWAY_URL/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-Environment: production" \
  -d '{"email_content": "Test email"}'
```
