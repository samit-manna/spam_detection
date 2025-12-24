# Azure ML Training Platform

This Terraform project deploys ML training platform components on AKS, including MLFlow, Kubeflow, Ray, and Feast.

## Components

### 1. MLFlow
- **Purpose**: Experiment tracking, model registry, and deployment
- **Features**:
  - PostgreSQL backend for metadata storage
  - Azure Blob Storage for artifacts
  - Web UI for experiment visualization
  - REST API for programmatic access

### 2. Kubeflow
- **Purpose**: End-to-end ML platform
- **Features**:
  - Jupyter Notebooks for development
  - Kubeflow Pipelines for ML workflows
  - Katib for hyperparameter tuning
  - KServe for model serving
  - Training operators (TensorFlow, PyTorch, etc.)

### 3. Ray
- **Purpose**: Distributed computing framework for ML
- **Features**:
  - Ray Train for distributed training
  - Ray Tune for hyperparameter optimization
  - Ray Serve for model serving
  - Autoscaling cluster with CPU and GPU workers

### 4. Feast
- **Purpose**: Feature store for ML
- **Features**:
  - Online features via Redis
  - Offline features via Azure Storage
  - Feature versioning and lineage
  - Real-time and batch feature serving

## Prerequisites

1. **Base Infrastructure Deployed**: Complete the `base-infra` deployment first
2. **kubectl Access**: Configured for your AKS cluster
3. **Terraform**: Version >= 1.5.0
4. **Base Infrastructure Outputs**: Required values from base-infra

## Quick Start

### 1. Get Base Infrastructure Outputs

From the `base-infra` directory:

```bash
cd ../base-infra

# Get required outputs
terraform output -raw aks_cluster_name
terraform output -raw resource_group_name
terraform output -raw storage_account_primary_connection_string
terraform output -raw redis_hostname
terraform output -raw redis_primary_access_key
```

### 2. Configure Variables

Copy and edit the variables file:

```bash
cd ../ml-platform
cp terraform.tfvars.example terraform.tfvars
```

Update `terraform.tfvars` with outputs from base infrastructure:

```hcl
resource_group_name = "mltraining-dev-rg"
aks_cluster_name    = "mltraining-dev-aks"

# From Key Vault or terraform outputs
mlflow_storage_connection_string = "DefaultEndpointsProtocol=https;..."
mlflow_postgres_host             = "mltraining-dev-psql-xxxxx.postgres.database.azure.com"
mlflow_postgres_password         = "your-password"

redis_host                      = "mltraining-dev-redis-xxxxx.redis.cache.windows.net"
redis_password                  = "your-redis-key"
feast_storage_connection_string = "DefaultEndpointsProtocol=https;..."
```

### 3. Deploy ML Platform

```bash
# Initialize
terraform init

# Plan
terraform plan -out=tfplan

# Apply
terraform apply tfplan
```

## Component Details

### MLFlow Setup

#### Access MLFlow UI

```bash
# Get MLFlow service IP
kubectl get svc mlflow-service -n mlflow

# Or use port-forward
kubectl port-forward svc/mlflow-service -n mlflow 5000:5000
```

Open browser: http://localhost:5000

#### Using MLFlow in Training Code

```python
import mlflow
import mlflow.pytorch

# Set tracking URI
mlflow.set_tracking_uri("http://mlflow-service.mlflow:5000")

# Start experiment
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 10)
    
    # Train model
    model = train_model()
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### Kubeflow Setup

Kubeflow requires manual installation using kustomize:

```bash
# Clone Kubeflow manifests
git clone https://github.com/kubeflow/manifests.git
cd manifests

# Install Kubeflow
kustomize build example | kubectl apply -f -

# Wait for all pods to be ready
kubectl get pods -n kubeflow -w

# Access Kubeflow dashboard
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

Open browser: http://localhost:8080

#### Creating a Notebook Server

1. Navigate to Kubeflow UI
2. Go to "Notebook Servers"
3. Click "New Server"
4. Select GPU image if needed
5. Configure resources and volumes

#### Running a Pipeline

```python
import kfp
from kfp import dsl

@dsl.pipeline(
    name='Training Pipeline',
    description='ML training pipeline'
)
def training_pipeline(
    learning_rate: float = 0.001,
    epochs: int = 10
):
    # Define pipeline steps
    prep_op = dsl.ContainerOp(
        name='data-prep',
        image='your-acr.azurecr.io/data-prep:latest'
    )
    
    train_op = dsl.ContainerOp(
        name='train',
        image='your-acr.azurecr.io/trainer:latest',
        arguments=[
            '--learning-rate', learning_rate,
            '--epochs', epochs
        ]
    ).after(prep_op).set_gpu_limit(1)

# Compile pipeline
kfp.compiler.Compiler().compile(training_pipeline, 'pipeline.yaml')

# Upload to Kubeflow
client = kfp.Client(host='http://localhost:8080')
client.create_run_from_pipeline_func(training_pipeline)
```

### Ray Setup

#### Access Ray Dashboard

```bash
kubectl port-forward svc/ray-dashboard -n ray 8265:8265
```

Open browser: http://localhost:8265

#### Using Ray for Distributed Training

```python
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

# Connect to Ray cluster
ray.init(address="ray://ray-cluster-ray-head.ray.svc.cluster.local:10001")

def train_func(config):
    import torch
    import torch.nn as nn
    
    # Your training code
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        # Training loop
        loss = train_epoch(model, optimizer)
        train.report({"loss": loss})

# Create distributed trainer
trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )
)

result = trainer.fit()
```

#### Ray Tune for Hyperparameter Tuning

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def objective(config):
    # Training function
    accuracy = train_model(
        lr=config["lr"],
        batch_size=config["batch_size"]
    )
    return {"accuracy": accuracy}

# Run hyperparameter search
analysis = tune.run(
    objective,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128])
    },
    num_samples=20,
    scheduler=ASHAScheduler(metric="accuracy", mode="max"),
    resources_per_trial={"cpu": 2, "gpu": 1}
)

print("Best config:", analysis.best_config)
```

### Feast Setup

#### Initialize Feast Repository

```bash
# Create Feast repository locally
feast init my_feature_repo
cd my_feature_repo
```

#### Define Features

Create `features.py`:

```python
from feast import Entity, Feature, FeatureView, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.file_source import FileSource
from datetime import timedelta

# Define entity
user = Entity(
    name="user_id",
    description="User identifier",
)

# Define feature view
user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
        Field(name="credit_score", dtype=Float32),
    ],
    source=FileSource(
        path="data/user_features.parquet",
        timestamp_field="event_timestamp",
    ),
    ttl=timedelta(days=1),
)
```

#### Apply Features

```bash
# Apply feature definitions
feast apply

# Materialize features to online store
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

#### Get Online Features

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

features = store.get_online_features(
    features=[
        "user_features:age",
        "user_features:income",
        "user_features:credit_score",
    ],
    entity_rows=[
        {"user_id": 1001},
        {"user_id": 1002},
    ],
).to_dict()

print(features)
```

## GPU Workload Scheduling

### Scheduling on GPU Nodes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training-job
spec:
  nodeSelector:
    accelerator: nvidia-gpu
  tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "true"
    effect: NoSchedule
  containers:
  - name: trainer
    image: your-acr.azurecr.io/trainer:gpu
    resources:
      limits:
        nvidia.com/gpu: 1
```

### Using Spot Instances

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job-spot
spec:
  template:
    spec:
      nodeSelector:
        kubernetes.azure.com/scalesetpriority: spot
      tolerations:
      - key: kubernetes.azure.com/scalesetpriority
        operator: Equal
        value: spot
        effect: NoSchedule
      restartPolicy: OnFailure
      containers:
      - name: trainer
        image: your-acr.azurecr.io/trainer:latest
```

## Monitoring

### Check Component Status

```bash
# MLFlow
kubectl get pods -n mlflow
kubectl logs -n mlflow -l app=mlflow

# Ray
kubectl get pods -n ray
kubectl logs -n ray -l ray.io/node-type=head

# Feast
kubectl get pods -n feast
kubectl logs -n feast -l app=feast
```

### Resource Usage

```bash
# GPU utilization
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu

# Node resource usage
kubectl top nodes

# Pod resource usage
kubectl top pods -n ray
```

## Troubleshooting

### MLFlow Not Connecting to Database

```bash
# Check PostgreSQL connectivity
kubectl run -it --rm psql-test --image=postgres:14 --restart=Never -- \
  psql -h YOUR_POSTGRES_HOST -U mlflowadmin -d mlflow
```

### Ray Workers Not Joining Cluster

```bash
# Check Ray head logs
kubectl logs -n ray -l ray.io/node-type=head

# Check Ray worker logs
kubectl logs -n ray -l ray.io/node-type=worker
```

### GPU Not Available in Pods

```bash
# Verify NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Check GPU availability
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu
```

## Cost Optimization Tips

1. **Use Spot Instances**: For interruptible training workloads
2. **Auto-scale Ray Cluster**: Configure min replicas to 0 when idle
3. **Right-size GPU VMs**: Use T4 instances for smaller models
4. **Scheduled Scaling**: Scale down during non-business hours
5. **Use Cheaper Storage Tiers**: For archived datasets and models

## Backup and Disaster Recovery

### Backup MLFlow Data

```bash
# Backup PostgreSQL database
az postgres flexible-server backup create \
  --resource-group $RESOURCE_GROUP \
  --name $POSTGRES_SERVER \
  --backup-name mlflow-backup-$(date +%Y%m%d)
```

### Backup Feast Registry

```bash
# Registry is stored in Azure Blob Storage
# Configure lifecycle management for versioning
```

## Next Steps

1. Train your first model with MLFlow tracking
2. Create a Kubeflow pipeline for your ML workflow
3. Experiment with Ray distributed training
4. Set up Feast feature store for your features
5. Configure CI/CD for automated training

## Support

For issues:
- MLFlow: https://github.com/mlflow/mlflow
- Kubeflow: https://github.com/kubeflow/kubeflow
- Ray: https://github.com/ray-project/ray
- Feast: https://github.com/feast-dev/feast

## License

[Your License Here]
