# Azure ML Training Infrastructure - Base Infrastructure

This Terraform project sets up the foundational infrastructure for ML training on Azure AKS, including networking, AKS cluster with GPU support, container registry, Redis cache, and supporting resources.

## Architecture Overview

This infrastructure includes:

- **Virtual Network (VNet)** with multiple subnets for isolation
- **Azure Kubernetes Service (AKS)** with:
  - System node pool for cluster management
  - GPU node pool (NVIDIA GPUs) for ML training workloads
  - Spot instance pool for cost-effective training
- **Azure Container Registry (ACR)** for container images
- **Azure Cache for Redis** for caching and session management
- **Azure Storage Account** with containers for datasets, models, and artifacts
- **Azure Key Vault** for secrets management
- **Azure PostgreSQL** for MLFlow backend database
- **Log Analytics Workspace** for monitoring

## Prerequisites

1. **Azure CLI**: Install and authenticate
   ```bash
   az login
   az account set --subscription <subscription-id>
   ```

2. **Terraform**: Version >= 1.5.0
   ```bash
   terraform version
   ```

3. **Azure Permissions**: Contributor access to the subscription

## Quick Start

### 1. Configure Variables

Copy the example variables file:
```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your values:
```hcl
environment  = "dev"
location     = "eastus"
project_name = "mltraining"

# Adjust node counts and VM sizes based on your needs
aks_gpu_node_count = 1
aks_gpu_node_vm_size = "Standard_NC6s_v3"
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Plan Infrastructure

```bash
terraform plan -out=tfplan
```

Review the planned resources carefully.

### 4. Apply Infrastructure

```bash
terraform apply tfplan
```

This will take approximately 15-20 minutes to complete.

### 5. Get AKS Credentials

```bash
# Get the cluster name from outputs
CLUSTER_NAME=$(terraform output -raw aks_cluster_name)
RESOURCE_GROUP=$(terraform output -raw resource_group_name)

# Configure kubectl
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME
```

### 6. Verify Installation

```bash
# Check cluster nodes
kubectl get nodes

# Verify GPU nodes
kubectl get nodes -l accelerator=nvidia-gpu

# Check spot instances
kubectl get nodes -l kubernetes.azure.com/scalesetpriority=spot
```

## GPU Node Configuration

### Available GPU VM Sizes

The infrastructure supports various GPU VM sizes:

| VM Size | GPU | GPU Memory | vCPUs | RAM | Use Case |
|---------|-----|------------|-------|-----|----------|
| Standard_NC6s_v3 | 1x V100 | 16GB | 6 | 112GB | Development, small models |
| Standard_NC12s_v3 | 2x V100 | 32GB | 12 | 224GB | Medium workloads |
| Standard_NC24s_v3 | 4x V100 | 64GB | 24 | 448GB | Large-scale training |
| Standard_NC4as_T4_v3 | 1x T4 | 16GB | 4 | 28GB | Inference, light training |

### NVIDIA GPU Driver Installation

The AKS cluster automatically installs GPU drivers. Verify with:

```bash
kubectl get pods -n kube-system | grep nvidia
```

## Cost Optimization

### Spot Instances

Spot instances can reduce costs by up to 90% but may be evicted:

```hcl
aks_spot_node_count     = 0  # Start with 0
aks_spot_node_max_count = 10 # Allow scaling up to 10
```

To use spot instances in your workloads:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
spec:
  nodeSelector:
    kubernetes.azure.com/scalesetpriority: spot
  tolerations:
  - key: kubernetes.azure.com/scalesetpriority
    operator: Equal
    value: spot
    effect: NoSchedule
```

### Auto-scaling

AKS node pools are configured with auto-scaling to optimize costs:

- System pool: 2-5 nodes
- GPU pool: min to max (configurable)
- Spot pool: 0 to max (scales to zero when idle)

## Storage Configuration

### Storage Containers

The following containers are created:

- `datasets`: Training and validation datasets
- `models`: Trained model artifacts
- `mlflow`: MLFlow experiment tracking artifacts
- `feast`: Feast feature store data

### Accessing Storage

```bash
# Get storage account name
STORAGE_ACCOUNT=$(terraform output -raw storage_account_name)

# Upload dataset
az storage blob upload \
  --account-name $STORAGE_ACCOUNT \
  --container-name datasets \
  --name my-dataset.csv \
  --file ./data/my-dataset.csv
```

## Security

### Key Vault Integration

Secrets are stored in Azure Key Vault:

```bash
# Get Key Vault name
KEY_VAULT=$(terraform output -raw key_vault_uri | cut -d'/' -f3 | cut -d'.' -f1)

# Retrieve a secret
az keyvault secret show --vault-name $KEY_VAULT --name redis-connection-string
```

### Network Security

- AKS uses Azure CNI networking with network policies
- Subnets are isolated with service endpoints
- NSGs control traffic flow

## Monitoring

### Log Analytics

All AKS logs are sent to Log Analytics:

```bash
# View logs in Azure Portal
LOG_WORKSPACE=$(terraform output -raw log_analytics_workspace_id)
echo "View logs at: https://portal.azure.com/#blade/Microsoft_Azure_Monitoring_Logs/LogsBlade/resourceId/$LOG_WORKSPACE"
```

### Prometheus & Grafana (Optional)

Install monitoring stack:

```bash
# Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Access Grafana
kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80
```

## Outputs

After deployment, important outputs are available:

```bash
# View all outputs
terraform output

# Get specific outputs
terraform output aks_cluster_name
terraform output acr_login_server
terraform output redis_hostname
terraform output storage_account_name
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will permanently delete all resources including data in storage accounts.

## Troubleshooting

### Issue: GPU nodes not starting

Check GPU node pool status:
```bash
az aks nodepool show \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpu
```

### Issue: ACR authentication fails

Verify AKS has AcrPull role:
```bash
az role assignment list --scope $(terraform output -raw acr_id)
```

### Issue: Insufficient quota for GPU VMs

Request quota increase:
```bash
az vm list-usage --location eastus -o table | grep Standard_NC
```

Request increase in Azure Portal under "Quotas".

## Next Steps

After deploying the base infrastructure:

1. Deploy ML Platform components (in `../ml-platform/`)
2. Configure MLFlow for experiment tracking
3. Set up Kubeflow for ML pipelines
4. Deploy Ray for distributed training
5. Configure Feast for feature management

## Support

For issues or questions:
- Check [Azure AKS Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- Review [Terraform Azure Provider Docs](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- Open an issue in the project repository

## License

[Your License Here]
