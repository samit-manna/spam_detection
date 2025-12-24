# Namespace for Kubeflow
resource "kubernetes_namespace" "kubeflow" {
  count = var.kubeflow_enabled ? 1 : 0

  metadata {
    name = var.kubeflow_namespace
    labels = merge(
      local.common_labels,
      {
        component = "kubeflow"
      }
    )
  }
}

resource "kubernetes_secret" "kubeflow_storage" {
  count = var.kubeflow_enabled ? 1 : 0

  metadata {
    name      = "azure-storage-secret"
    namespace = kubernetes_namespace.kubeflow[0].metadata[0].name
  }

  data = {
    AZURE_STORAGE_CONNECTION_STRING = data.terraform_remote_state.aks.outputs.storage_account_primary_connection_string
    AZURE_STORAGE_ACCOUNT_NAME      = data.terraform_remote_state.aks.outputs.storage_account_name
    AZURE_STORAGE_ACCOUNT_KEY       = data.terraform_remote_state.aks.outputs.storage_account_primary_access_key
  }
}

# Install Kubeflow Training Operator using null_resource with kubectl
resource "null_resource" "training_operator" {
  count = var.kubeflow_enabled ? 1 : 0

  triggers = {
    version        = "v1.8.0"
    namespace      = kubernetes_namespace.kubeflow[0].metadata[0].name
    cluster_name   = data.terraform_remote_state.aks.outputs.aks_cluster_name
    resource_group = data.terraform_remote_state.aks.outputs.resource_group_name
  }

  provisioner "local-exec" {
    command = <<-EOT
      # Get AKS credentials to ensure kubectl is pointing to correct cluster
      az aks get-credentials --resource-group ${self.triggers.resource_group} --name ${self.triggers.cluster_name} --overwrite-existing && \
      kubectl apply --server-side -k 'github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=${self.triggers.version}' && \
      kubectl wait --for=condition=available --timeout=300s deployment/training-operator -n ${self.triggers.namespace}
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      kubectl delete --ignore-not-found=true -k 'github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=${self.triggers.version}' || true
    EOT
  }

  depends_on = [kubernetes_namespace.kubeflow]
}

# Install Kubeflow Pipelines
resource "null_resource" "kubeflow_pipelines" {
  count = var.kubeflow_enabled ? 1 : 0

  triggers = {
    version        = "2.14.0"
    namespace      = kubernetes_namespace.kubeflow[0].metadata[0].name
    cluster_name   = data.terraform_remote_state.aks.outputs.aks_cluster_name
    resource_group = data.terraform_remote_state.aks.outputs.resource_group_name
  }

  provisioner "local-exec" {
    command = <<-EOT
      # Get AKS credentials to ensure kubectl is pointing to correct cluster
      az aks get-credentials --resource-group ${self.triggers.resource_group} --name ${self.triggers.cluster_name} --overwrite-existing && \
      
      # Install Kubeflow Pipelines standalone (v2.14.0 - latest stable)
      kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${self.triggers.version}" && \
      kubectl wait --for condition=established --timeout=60s crd/workflows.argoproj.io && \
      kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${self.triggers.version}" && \
      
      # Fix MinIO image (GCR image is missing)
      kubectl set image deployment/minio -n ${self.triggers.namespace} minio=minio/minio:RELEASE.2019-08-14T20-37-41Z && \
      
      # Wait for core components
      kubectl wait --for=condition=available --timeout=600s deployment/ml-pipeline -n ${self.triggers.namespace} || true && \
      kubectl wait --for=condition=available --timeout=300s deployment/minio -n ${self.triggers.namespace} || true
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      kubectl delete --ignore-not-found=true -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=${self.triggers.version}" || true && \
      kubectl delete --ignore-not-found=true -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${self.triggers.version}" || true
    EOT
  }

  depends_on = [kubernetes_namespace.kubeflow]
}

# Service to expose Kubeflow Pipelines UI
resource "kubernetes_service" "kubeflow_pipelines_ui" {
  count = var.kubeflow_enabled ? 1 : 0

  metadata {
    name      = "ml-pipeline-ui-external"
    namespace = kubernetes_namespace.kubeflow[0].metadata[0].name
    labels = merge(
      local.common_labels,
      {
        component = "kubeflow-pipelines-ui"
      }
    )
  }

  spec {
    type = "LoadBalancer"
    
    selector = {
      app = "ml-pipeline-ui"
    }

    port {
      name        = "http"
      port        = 80
      target_port = 3000
      protocol    = "TCP"
    }
  }

  depends_on = [null_resource.kubeflow_pipelines]
}

# ConfigMap with Training Operator usage examples
resource "kubernetes_config_map" "kubeflow_install" {
  count = var.kubeflow_enabled ? 1 : 0

  metadata {
    name      = "training-operator-examples"
    namespace = kubernetes_namespace.kubeflow[0].metadata[0].name
  }

  data = {
    "pytorch-job.yaml" = <<-YAML
      apiVersion: kubeflow.org/v1
      kind: PyTorchJob
      metadata:
        name: pytorch-simple
        namespace: kubeflow
      spec:
        pytorchReplicaSpecs:
          Master:
            replicas: 1
            restartPolicy: OnFailure
            template:
              spec:
                containers:
                - name: pytorch
                  image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
                  command:
                  - python
                  - -c
                  - |
                    import torch
                    import torch.nn as nn
                    import torch.optim as optim
                    
                    # Simple training example
                    model = nn.Linear(10, 1)
                    criterion = nn.MSELoss()
                    optimizer = optim.SGD(model.parameters(), lr=0.01)
                    
                    for epoch in range(100):
                        inputs = torch.randn(32, 10)
                        targets = torch.randn(32, 1)
                        
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        
                        if epoch % 10 == 0:
                            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
                    
                    print('Training completed!')
          Worker:
            replicas: 2
            restartPolicy: OnFailure
            template:
              spec:
                containers:
                - name: pytorch
                  image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
                  command:
                  - python
                  - -c
                  - |
                    import torch
                    print('PyTorch worker ready')
                    # Worker training code here
    YAML
    
    "tensorflow-job.yaml" = <<-YAML
      apiVersion: kubeflow.org/v1
      kind: TFJob
      metadata:
        name: tensorflow-simple
        namespace: kubeflow
      spec:
        tfReplicaSpecs:
          Chief:
            replicas: 1
            restartPolicy: OnFailure
            template:
              spec:
                containers:
                - name: tensorflow
                  image: tensorflow/tensorflow:2.13.0
                  command:
                  - python
                  - -c
                  - |
                    import tensorflow as tf
                    
                    # Simple training example
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
                        tf.keras.layers.Dense(1)
                    ])
                    
                    model.compile(optimizer='adam', loss='mse')
                    
                    # Generate dummy data
                    import numpy as np
                    x_train = np.random.randn(1000, 10)
                    y_train = np.random.randn(1000, 1)
                    
                    model.fit(x_train, y_train, epochs=10, batch_size=32)
                    print('Training completed!')
          Worker:
            replicas: 2
            restartPolicy: OnFailure
            template:
              spec:
                containers:
                - name: tensorflow
                  image: tensorflow/tensorflow:2.13.0
                  command:
                  - python
                  - -c
                  - |
                    import tensorflow as tf
                    print('TensorFlow worker ready')
                    # Worker training code here
    YAML
    
    "xgboost-job.yaml" = <<-YAML
      apiVersion: kubeflow.org/v1
      kind: XGBoostJob
      metadata:
        name: xgboost-simple
        namespace: kubeflow
      spec:
        xgbReplicaSpecs:
          Master:
            replicas: 1
            restartPolicy: OnFailure
            template:
              spec:
                containers:
                - name: xgboost
                  image: python:3.9
                  command:
                  - sh
                  - -c
                  - |
                    pip install xgboost scikit-learn &&
                    python -c "
                    import xgboost as xgb
                    from sklearn.datasets import load_breast_cancer
                    from sklearn.model_selection import train_test_split
                    
                    # Load data
                    data = load_breast_cancer()
                    X_train, X_test, y_train, y_test = train_test_split(
                        data.data, data.target, test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dtest = xgb.DMatrix(X_test, label=y_test)
                    
                    params = {
                        'objective': 'binary:logistic',
                        'max_depth': 3,
                        'eta': 0.1,
                        'eval_metric': 'auc'
                    }
                    
                    model = xgb.train(params, dtrain, num_boost_round=100)
                    print('Training completed!')
                    "
          Worker:
            replicas: 2
            restartPolicy: OnFailure
            template:
              spec:
                containers:
                - name: xgboost
                  image: python:3.9
                  command:
                  - python
                  - -c
                  - |
                    print('XGBoost worker ready')
    YAML
    
    "README.md" = <<-EOT
      # Kubeflow Components
      
      ## Training Operator
      
      The Training Operator supports distributed training for:
      - PyTorch (PyTorchJob)
      - TensorFlow (TFJob)
      - XGBoost (XGBoostJob)
      - MPI (MPIJob)
      - PaddlePaddle (PaddleJob)
      - MXNet (MXNetJob)
      
      ## Kubeflow Pipelines
      
      KFP v2.2.0 is installed with bundled components:
      - **Database**: MySQL (in-cluster)
      - **Artifact Storage**: MinIO (in-cluster) - patched to use minio/minio official image
      - **Cache**: Cache server (in-cluster)
      - **UI**: Disabled (GCR images unavailable) - use API port-forward or KFP SDK
      
      ### What's Installed
      
      **Working components:**
      - ✅ ML Pipeline API Server (core API) - RUNNING
      - ✅ Argo Workflows (pipeline orchestration) - RUNNING
      - ✅ MySQL (metadata storage) - RUNNING
      - ✅ MinIO (artifact storage) - RUNNING with fixed image
      - ✅ Cache Server (execution caching) - RUNNING
      - ✅ Metadata services - RUNNING
      - ❌ ML Pipeline UI (frontend) - DISABLED due to missing GCR images
      
      ## Usage Examples
      
      ### Training Jobs
      
      Apply any of the example jobs:
      ```bash
      kubectl apply -f pytorch-job.yaml
      kubectl apply -f tensorflow-job.yaml
      kubectl apply -f xgboost-job.yaml
      ```
      
      Monitor job status:
      ```bash
      kubectl get pytorchjobs -n kubeflow
      kubectl get tfjobs -n kubeflow
      kubectl get xgboostjobs -n kubeflow
      ```
      
      Check logs:
      ```bash
      kubectl logs -n kubeflow <job-name>-master-0
      kubectl logs -n kubeflow <job-name>-worker-0
      ```
      
      ### Pipelines
      
      **Access the API (UI unavailable due to image issues):**
      ```bash
      # Port-forward to API server
      kubectl port-forward svc/ml-pipeline 8888:8888 -n kubeflow
      
      # Or use KFP SDK directly
      import kfp
      client = kfp.Client(host='http://localhost:8888')
      client.list_pipelines()
      ```
      
      **Submit pipelines programmatically:**
      ```python
      import kfp
      from kfp import dsl
      
      client = kfp.Client(host='http://ml-pipeline.kubeflow.svc.cluster.local:8888')
      
      # Upload and run pipeline
      pipeline_file = 'pipeline.yaml'
      client.create_run_from_pipeline_package(pipeline_file)
      ```
      
      ## Integration with MLflow
      
      Track experiments in MLflow by setting environment variables:
      ```yaml
      env:
      - name: MLFLOW_TRACKING_URI
        value: "http://mlflow-service.mlflow.svc.cluster.local:5000"
      ```
      
      ## Storage Access
      
      Pipelines can access Azure Blob Storage using the kubeflow-storage-secret:
      ```yaml
      env:
      - name: AZURE_STORAGE_CONNECTION_STRING
        valueFrom:
          secretKeyRef:
            name: kubeflow-storage-secret
            key: azure-storage-connection-string
      ```
    EOT
    
    "pipeline-example.py" = <<-EOT
      # Kubeflow Pipelines example
      from kfp import dsl
      from kfp import compiler
      
      @dsl.component(base_image='python:3.9')
      def preprocess_data(input_path: str, output_path: str):
          import pandas as pd
          # Data preprocessing logic
          print(f"Processing data from {input_path} to {output_path}")
      
      @dsl.component(base_image='python:3.9')
      def train_model(data_path: str, model_path: str):
          import pickle
          # Model training logic
          print(f"Training model with data from {data_path}")
          print(f"Saving model to {model_path}")
      
      @dsl.component(base_image='python:3.9')
      def evaluate_model(model_path: str, test_data_path: str) -> float:
          # Model evaluation logic
          print(f"Evaluating model from {model_path}")
          accuracy = 0.95
          return accuracy
      
      @dsl.pipeline(
          name='ml-training-pipeline',
          description='End-to-end ML training pipeline'
      )
      def ml_pipeline(
          input_data: str = 'gs://bucket/data.csv',
          model_output: str = 'gs://bucket/model.pkl'
      ):
          preprocess_task = preprocess_data(
              input_path=input_data,
              output_path='/tmp/processed_data.csv'
          )
          
          train_task = train_model(
              data_path=preprocess_task.outputs['output_path'],
              model_path=model_output
          )
          
          evaluate_task = evaluate_model(
              model_path=train_task.outputs['model_path'],
              test_data_path='/tmp/test_data.csv'
          )
      
      if __name__ == '__main__':
          compiler.Compiler().compile(ml_pipeline, 'pipeline.yaml')
    EOT
  }

  depends_on = [kubernetes_namespace.kubeflow]
}

# RBAC: ClusterRole for Ray Job management
resource "kubernetes_cluster_role" "ray_job_manager" {
  count = var.kubeflow_enabled ? 1 : 0

  metadata {
    name = "ray-job-manager"
    labels = merge(
      local.common_labels,
      {
        component = "kubeflow-ray-rbac"
      }
    )
  }

  rule {
    api_groups = ["ray.io"]
    resources  = ["rayjobs", "rayclusters"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  rule {
    api_groups = ["ray.io"]
    resources  = ["rayjobs/status", "rayclusters/status"]
    verbs      = ["get"]
  }
}

# RBAC: Bind Ray Job manager role to pipeline-runner service account
resource "kubernetes_cluster_role_binding" "pipeline_runner_ray_job_manager" {
  count = var.kubeflow_enabled ? 1 : 0

  metadata {
    name = "pipeline-runner-ray-job-manager"
    labels = merge(
      local.common_labels,
      {
        component = "kubeflow-ray-rbac"
      }
    )
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.ray_job_manager[0].metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = "pipeline-runner"
    namespace = kubernetes_namespace.kubeflow[0].metadata[0].name
  }

  depends_on = [
    null_resource.kubeflow_pipelines,
    kubernetes_cluster_role.ray_job_manager
  ]
}

# Output Kubeflow Pipelines UI URL
output "kubeflow_pipelines_ui_url" {
  description = "URL to access Kubeflow Pipelines UI"
  value       = var.kubeflow_enabled ? "http://${kubernetes_service.kubeflow_pipelines_ui[0].status[0].load_balancer[0].ingress[0].ip}" : null
}

output "kubeflow_namespace" {
  description = "Namespace where Kubeflow components are installed"
  value       = var.kubeflow_enabled ? kubernetes_namespace.kubeflow[0].metadata[0].name : null
}
