# API Gateway - Model Serving API

A FastAPI-based serving API for the Spam Detection ML platform. Provides unified access to real-time and batch inference, deployed model information, and serving metrics.

> **Note**: This is a **serving-only** API. Training workflows, model registration, and promotion are internal operations that can be accessed via separate Istio gateways to MLflow, Kubeflow, etc.

## Features

- **Real-time Inference**: Single and small batch predictions via KServe/Triton
- **Batch Inference**: Large-scale async predictions via Ray
- **Model Info**: View currently deployed model versions
- **Monitoring**: Drift detection and performance metrics
- **Authentication**: API key-based access control
- **Caching**: Redis-based feature and prediction caching

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       API Gateway (FastAPI) - Serving Only                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   /health   │    │   /models   │    │  /predict   │    │  /metrics   │  │
│  │   /info     │    │  /deployed  │    │ /batch-sync │    │   /drift    │  │
│  │  /live      │    │  /staging   │    │   /batch    │    │ /performance│  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Service Clients                              │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │   │
│  │  │ MLflow  │    │ KServe  │    │   Ray   │    │  Redis  │          │   │
│  │  │(read)   │    │(infer)  │    │ (batch) │    │ (cache) │          │   │
│  │  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘          │   │
│  └───────┼──────────────┼──────────────┼──────────────┼───────────────┘   │
│          │              │              │              │                    │
└──────────┼──────────────┼──────────────┼──────────────┼────────────────────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ MLflow  │    │ KServe  │    │   Ray   │    │  Redis  │
      │ Server  │    │ Triton  │    │ Cluster │    │  Azure  │
      └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

## API Endpoints

### Health & Info
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Comprehensive health check (all services) |
| GET | `/health/live` | Kubernetes liveness probe |
| GET | `/health/ready` | Kubernetes readiness probe |
| GET | `/info` | API version and deployed model info |

### Model Information (Read-Only)
| Method | Endpoint | Description | Role |
|--------|----------|-------------|------|
| GET | `/models` | Summary of deployed models | viewer |
| GET | `/models/deployed` | Production model details | viewer |
| GET | `/models/staging` | Staging model details | viewer |

### Real-time Inference
| Method | Endpoint | Description | Role |
|--------|----------|-------------|------|
| POST | `/predict` | Single email prediction | operator |
| POST | `/predict/batch-sync` | Small batch (<100 items) | operator |

### Batch Inference
| Method | Endpoint | Description | Role |
|--------|----------|-------------|------|
| POST | `/predict/batch` | Submit async batch job | operator |
| GET | `/predict/batch/{job_id}` | Get job status | viewer |
| GET | `/predict/batch/{job_id}/results` | Get job results | viewer |

### Monitoring
| Method | Endpoint | Description | Role |
|--------|----------|-------------|------|
| GET | `/metrics` | Prometheus metrics | - |
| GET | `/metrics/drift` | Model drift scores | viewer |
| GET | `/metrics/performance` | Latency & throughput | viewer |

## Authentication

### API Key Authentication

Include `X-API-Key` header in all requests:

```bash
curl -H "X-API-Key: your-api-key" http://api-gateway/predict \
  -H "Content-Type: application/json" \
  -d '{"email_id": "123", "subject": "Hello", "body": "Test email", "sender": "user@example.com"}'
```

### Roles

| Role | Permissions |
|------|-------------|
| `viewer` | GET endpoints (models, metrics, job status) |
| `operator` | GET + predict endpoints |

Configure API keys via environment variables:
```bash
OPERATOR_API_KEYS=key1,key2
VIEWER_API_KEYS=key3,key4
```

## Request/Response Examples

### Single Prediction

```bash
# Request
curl -X POST http://api-gateway/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "email_id": "email-001",
    "subject": "URGENT: You have won $1,000,000!",
    "body": "Click here to claim your prize immediately...",
    "sender": "winner@lottery-prize.com"
  }'

# Response
{
  "email_id": "email-001",
  "prediction": "spam",
  "spam_probability": 0.94,
  "confidence": "high",
  "model_version": "3",
  "model_stage": "Production",
  "latency_ms": 23,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Sync Batch Prediction

```bash
# Request (up to 100 emails)
curl -X POST http://api-gateway/predict/batch-sync \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"email_id": "e1", "subject": "Meeting", "body": "See you tomorrow", "sender": "boss@company.com"},
      {"email_id": "e2", "subject": "FREE!", "body": "Click now!", "sender": "spam@fake.com"}
    ],
    "model_stage": "Production"
  }'

# Response
{
  "predictions": [...],
  "total_count": 2,
  "success_count": 2,
  "error_count": 0,
  "total_latency_ms": 45,
  "avg_latency_ms": 22.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Async Batch Job

```bash
# Submit job
curl -X POST http://api-gateway/predict/batch \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "abfss://data@storage.dfs.core.windows.net/input/emails.parquet",
    "output_path": "abfss://data@storage.dfs.core.windows.net/output/predictions.parquet"
  }'

# Response
{
  "job_id": "batch-abc123",
  "status": "submitted",
  "estimated_duration_minutes": 15,
  "created_at": "2024-01-15T10:30:00Z"
}

# Check status
curl http://api-gateway/predict/batch/batch-abc123 \
  -H "X-API-Key: your-key"

# Get results
curl http://api-gateway/predict/batch/batch-abc123/results \
  -H "X-API-Key: your-key"
```

### View Deployed Models

```bash
curl http://api-gateway/models \
  -H "X-API-Key: your-key"

# Response
{
  "production": {
    "name": "spam-detector",
    "version": "3",
    "endpoint": "spam-detector-production",
    "status": "ready"
  },
  "staging": {
    "name": "spam-detector",
    "version": "4",
    "endpoint": "spam-detector-staging",
    "status": "ready"
  }
}
```

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "PREDICTION_FAILED",
    "message": "Failed to extract features from email",
    "details": {}
  },
  "correlation_id": "abc-123-def-456",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Common error codes:
- `AUTHENTICATION_REQUIRED` - Missing API key
- `INVALID_API_KEY` - Invalid API key
- `INSUFFICIENT_PERMISSIONS` - Role not authorized
- `VALIDATION_FAILED` - Request validation error
- `PREDICTION_FAILED` - Inference error
- `MODEL_NOT_AVAILABLE` - Model not deployed
- `BATCH_JOB_NOT_FOUND` - Invalid job ID
- `INTERNAL_ERROR` - Unexpected server error

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --port 8000
```

### Environment Variables

Create `.env` file:
```bash
# API
DEBUG=true
LOG_LEVEL=DEBUG

# Services (use port-forwarded URLs for local dev)
MLFLOW_TRACKING_URI=http://localhost:5000
FEATURE_TRANSFORMER_URL=http://localhost:8080
RAY_ADDRESS=ray://localhost:10001

# KServe
KSERVE_NAMESPACE=kserve
INFERENCE_SERVICE_NAME=spam-detector

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_SSL=false

# API Keys
OPERATOR_API_KEYS=dev-operator-key
VIEWER_API_KEYS=dev-viewer-key
```

### Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## Deployment

### Build Docker Image

```bash
# Build
docker build -t api-gateway:latest .

# Push to ACR
az acr login --name $ACR_NAME
docker tag api-gateway:latest $ACR_NAME.azurecr.io/serving/api-gateway:latest
docker push $ACR_NAME.azurecr.io/serving/api-gateway:latest
```

### Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f deployment.yaml

# Apply Istio VirtualService
kubectl apply -f istio.yaml

# Check status
kubectl get pods -n serving -l app=api-gateway
```

### Verify Deployment

```bash
# Port forward
kubectl port-forward svc/api-gateway 8000:80 -n serving

# Test health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"email_id":"test","subject":"Test","body":"Test body","sender":"test@test.com"}'
```

## Monitoring

### Prometheus Metrics

The `/metrics` endpoint exposes:
- `api_requests_total` - Request count by method, endpoint, status
- `api_request_latency_seconds` - Request latency histogram
- `predictions_total` - Prediction count by label
- `batch_jobs_total` - Batch job count by status

### Grafana Dashboards

Import dashboards for:
- API Gateway Overview
- Prediction Latency Analysis
- Error Rate Tracking
- Batch Job Monitoring

### Logging

Structured JSON logs include:
- Correlation ID (via `X-Correlation-ID` header)
- Request method, path, status
- Response time
- Prediction results (anonymized)

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `API_VERSION` | API version string | `1.0.0` |
| `DEBUG` | Enable debug mode | `false` |
| `WORKERS` | Gunicorn workers | `4` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | required |
| `KSERVE_NAMESPACE` | KServe namespace | `kserve` |
| `INFERENCE_SERVICE_NAME` | KServe inference service | `spam-detector` |
| `RAY_ADDRESS` | Ray cluster address | required |
| `REDIS_HOST` | Redis host | required |
| `REDIS_PORT` | Redis port | `6380` |
| `REDIS_SSL` | Enable Redis TLS | `true` |
| `FEATURE_TRANSFORMER_URL` | Feature transformer URL | required |
| `RATE_LIMIT_REQUESTS` | Max requests per window | `100` |
| `RATE_LIMIT_PERIOD` | Rate limit window (sec) | `60` |

## Internal Services Access

Training-related services (MLflow UI, Kubeflow Pipelines, Ray Dashboard) are exposed internally via Istio gateways:

```yaml
# Example: Access MLflow UI internally
# Gateway: mlflow.internal.example.com
# Kubeflow: kubeflow.internal.example.com
# Ray: ray-dashboard.internal.example.com
```

These are separate from the serving API and managed by platform administrators.
