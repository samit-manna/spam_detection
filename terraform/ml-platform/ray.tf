# Namespace for Ray
resource "kubernetes_namespace" "ray" {
  count = var.ray_enabled ? 1 : 0

  metadata {
    name = var.ray_namespace
    labels = merge(
      local.common_labels,
      {
        component = "ray"
      }
    )
  }
}

# Secret for Azure Blob Storage access
resource "kubernetes_secret" "ray_storage" {
  count = var.ray_enabled ? 1 : 0

  metadata {
    name      = "ray-azure-secret"
    namespace = kubernetes_namespace.ray[0].metadata[0].name
  }

  data = {
    azure-storage-account-name = data.terraform_remote_state.aks.outputs.storage_account_name
    azure-storage-account-key  = data.terraform_remote_state.aks.outputs.storage_account_primary_access_key
  }
}

# Install Ray Operator using Helm
resource "helm_release" "ray_operator" {
  count = var.ray_enabled ? 1 : 0

  name       = "ray-operator"
  repository = "https://ray-project.github.io/kuberay-helm/"
  chart      = "kuberay-operator"
  version    = "1.0.0"
  namespace  = kubernetes_namespace.ray[0].metadata[0].name

  set {
    name  = "image.tag"
    value = "v1.0.0"
  }
}

# Ray Cluster Custom Resource
resource "kubectl_manifest" "ray_cluster" {
  count = var.ray_enabled ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: ray.io/v1
    kind: RayCluster
    metadata:
      name: ray-cluster
      namespace: ${var.ray_namespace}
      labels:
        environment: ${var.environment}
        project: ${var.project_name}
    spec:
      rayVersion: '2.9.0'
      enableInTreeAutoscaling: true
      headGroupSpec:
        rayStartParams:
          dashboard-host: '0.0.0.0'
          num-cpus: '0'
        template:
          spec:
            containers:
            - name: ray-head
              image: rayproject/ray-ml:2.9.0-py310
              ports:
              - containerPort: 6379
                name: gcs
              - containerPort: 8265
                name: dashboard
              - containerPort: 10001
                name: client
              env:
              - name: AZURE_STORAGE_ACCOUNT_NAME
                valueFrom:
                  secretKeyRef:
                    name: ray-azure-secret
                    key: azure-storage-account-name
              - name: AZURE_STORAGE_ACCOUNT_KEY
                valueFrom:
                  secretKeyRef:
                    name: ray-azure-secret
                    key: azure-storage-account-key
              lifecycle:
                postStart:
                  exec:
                    command: ["/bin/bash", "-c", "pip install adlfs azure-identity fsspec"]
              resources:
                requests:
                  cpu: "250m"
                  memory: "1Gi"
                limits:
                  cpu: "1"
                  memory: "4Gi"
              volumeMounts:
              - name: ray-logs
                mountPath: /tmp/ray
            volumes:
            - name: ray-logs
              emptyDir: {}
      workerGroupSpecs:
      - replicas: 1
        minReplicas: 1
        maxReplicas: 10
        groupName: worker-group
        rayStartParams:
          num-cpus: '4'
        template:
          spec:
            containers:
            - name: ray-worker
              image: rayproject/ray-ml:2.9.0-py310
              env:
              - name: AZURE_STORAGE_ACCOUNT_NAME
                valueFrom:
                  secretKeyRef:
                    name: ray-azure-secret
                    key: azure-storage-account-name
              - name: AZURE_STORAGE_ACCOUNT_KEY
                valueFrom:
                  secretKeyRef:
                    name: ray-azure-secret
                    key: azure-storage-account-key
              lifecycle:
                postStart:
                  exec:
                    command: ["/bin/bash", "-c", "pip install adlfs azure-identity fsspec"]
              resources:
                requests:
                  cpu: "2"
                  memory: "8Gi"
                limits:
                  cpu: "4"
                  memory: "16Gi"
              volumeMounts:
              - name: ray-logs
                mountPath: /tmp/ray
            volumes:
            - name: ray-logs
              emptyDir: {}
            nodeSelector:
              kubernetes.azure.com/scalesetpriority: spot
            tolerations:
            - key: kubernetes.azure.com/scalesetpriority
              operator: Equal
              value: spot
              effect: NoSchedule
      - replicas: 0
        minReplicas: 0
        maxReplicas: 5
        groupName: gpu-worker-group
        rayStartParams:
          num-cpus: '6'
          num-gpus: '1'
        template:
          spec:
            containers:
            - name: ray-worker
              image: rayproject/ray-ml:2.9.0-py310-gpu
              env:
              - name: AZURE_STORAGE_ACCOUNT_NAME
                valueFrom:
                  secretKeyRef:
                    name: ray-azure-secret
                    key: azure-storage-account-name
              - name: AZURE_STORAGE_ACCOUNT_KEY
                valueFrom:
                  secretKeyRef:
                    name: ray-azure-secret
                    key: azure-storage-account-key
              lifecycle:
                postStart:
                  exec:
                    command: ["/bin/bash", "-c", "pip install adlfs azure-identity fsspec"]
              resources:
                requests:
                  cpu: "4"
                  memory: "16Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "6"
                  memory: "32Gi"
                  nvidia.com/gpu: "1"
              volumeMounts:
              - name: ray-logs
                mountPath: /tmp/ray
            volumes:
            - name: ray-logs
              emptyDir: {}
            nodeSelector:
              gpu-type: nvidia
            tolerations:
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
  YAML

  depends_on = [helm_release.ray_operator]
}

# Service for Ray Dashboard
resource "kubernetes_service" "ray_dashboard" {
  count = var.ray_enabled ? 1 : 0

  metadata {
    name      = "ray-dashboard"
    namespace = kubernetes_namespace.ray[0].metadata[0].name
    labels = {
      app = "ray"
      component = "dashboard"
    }
  }

  spec {
    type = "ClusterIP"

    selector = {
      "ray.io/node-type" = "head"
    }

    port {
      name        = "dashboard"
      port        = 8265
      target_port = 8265
      protocol    = "TCP"
    }

    port {
      name        = "client"
      port        = 10001
      target_port = 10001
      protocol    = "TCP"
    }
  }

  depends_on = [kubectl_manifest.ray_cluster]
}

# ConfigMap with Ray usage examples
resource "kubernetes_config_map" "ray_examples" {
  count = var.ray_enabled ? 1 : 0

  metadata {
    name      = "ray-examples"
    namespace = kubernetes_namespace.ray[0].metadata[0].name
  }

  data = {
    "train_example.py" = <<-PYTHON
      import ray
      from ray import train
      from ray.train import ScalingConfig
      from ray.train.torch import TorchTrainer
      import pandas as pd
      import fsspec
      import os
      
      # Connect to Ray cluster
      ray.init(address="ray://ray-cluster-ray-head:10001")
      
      def train_func(config):
          # Load data from Azure Blob Storage
          account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')
          blob_path = f"abfss://feast@{account_name}.dfs.core.windows.net/training_data.parquet"
          
          # Read data using fsspec (credentials from environment)
          with fsspec.open(blob_path, 'rb') as f:
              df = pd.read_parquet(f)
          
          # Your training code here
          import torch
          import torch.nn as nn
          
          model = nn.Linear(10, 1)
          optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
          
          for epoch in range(10):
              # Training loop
              loss = model(torch.randn(32, 10)).sum()
              loss.backward()
              optimizer.step()
              
              train.report({"loss": loss.item()})
      
      # Create trainer
      trainer = TorchTrainer(
          train_func,
          scaling_config=ScalingConfig(
              num_workers=4,
              use_gpu=True
          )
      )
      
      result = trainer.fit()
      print(result)
    PYTHON
    
    "distributed_training.py" = <<-PYTHON
      import ray
      from ray import tune
      from ray.tune.schedulers import ASHAScheduler
      
      # Connect to Ray cluster
      ray.init(address="ray://ray-cluster-ray-head:10001")
      
      def training_function(config):
          # Hyperparameter tuning with Ray Tune
          import time
          import numpy as np
          
          for step in range(100):
              # Simulate training
              score = config["alpha"] * step + config["beta"]
              score += np.random.randn() * 0.1
              
              tune.report(score=score)
              time.sleep(0.1)
      
      # Run hyperparameter tuning
      analysis = tune.run(
          training_function,
          config={
              "alpha": tune.uniform(0, 1),
              "beta": tune.uniform(0, 10)
          },
          num_samples=20,
          scheduler=ASHAScheduler(metric="score", mode="max")
      )
      
      print("Best config:", analysis.best_config)
    PYTHON
  }
}
