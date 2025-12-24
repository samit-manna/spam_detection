# Serving Infrastructure Terraform

This Terraform configuration deploys the model serving infrastructure on AKS:

- **cert-manager** - Certificate management for Kubernetes
- **Istio** - Service mesh for traffic management and observability
- **Knative Serving** - Serverless platform for auto-scaling
- **KServe** - ML model serving with standardized inference protocol

## Prerequisites

1. Base infrastructure deployed (`../base-infra`)
2. Azure CLI logged in
3. kubectl configured with AKS cluster access

## Usage

```bash
# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Apply changes
terraform apply

# View outputs
terraform output
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AKS Cluster                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  cert-manager   │  │  istio-system   │                   │
│  │  (certificates) │  │ (service mesh)  │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ knative-serving │  │     kserve      │                   │
│  │  (serverless)   │  │  (ML serving)   │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    serving                           │    │
│  │  (InferenceServices, Feature Transformer, etc.)      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Order

Terraform handles dependencies automatically:

1. cert-manager (required by Istio/KServe)
2. Istio Base → Istiod → Ingress Gateway
3. Knative Serving CRDs → Core → Net-Istio
4. KServe → ClusterServingRuntimes → Config
5. Serving namespace + secrets

## Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `cert_manager_enabled` | Deploy cert-manager | `true` |
| `cert_manager_version` | cert-manager version | `v1.13.3` |
| `istio_enabled` | Deploy Istio | `true` |
| `istio_version` | Istio version | `1.20.2` |
| `knative_enabled` | Deploy Knative | `true` |
| `knative_version` | Knative version | `1.12.3` |
| `kserve_enabled` | Deploy KServe | `true` |
| `kserve_version` | KServe version | `0.12.0` |

## Outputs

```bash
# Get all outputs
terraform output

# Get specific outputs
terraform output kserve_namespace
terraform output inference_endpoint_info
terraform output component_versions
```

## Accessing Inference Endpoints

After deployment, get the Istio Ingress Gateway IP:

```bash
kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

InferenceServices will be accessible at:
```
http://<service-name>-<namespace>.<ingress-ip>.sslip.io/v1/models/<model-name>:predict
```

## Clean Up

```bash
# Destroy all serving infrastructure
terraform destroy
```

**Note:** Destroying will remove all InferenceServices deployed in the `serving` and `kserve` namespaces.

## Integration with model-serving

After deploying this infrastructure, you can deploy models using the `model-serving/` directory:

```bash
cd ../model-serving

# Build and push images
make build && make push

# Deploy staging inference service
make deploy-staging

# Deploy feature transformer
make deploy-transformer
```

## Troubleshooting

### KServe pods not starting
```bash
kubectl get pods -n kserve
kubectl logs -n kserve deployment/kserve-controller-manager
```

### Istio ingress not getting external IP
```bash
kubectl get svc -n istio-system
kubectl describe svc istio-ingressgateway -n istio-system
```

### Knative services stuck in "Unknown"
```bash
kubectl get ksvc -A
kubectl describe ksvc <name> -n <namespace>
```
