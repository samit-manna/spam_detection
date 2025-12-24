#!/bin/bash
set -e

echo "========================================"
echo "ML Training Pipeline - Setup"
echo "========================================"

# Check prerequisites
echo "Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required"; exit 1; }
kubectl cluster-info >/dev/null 2>&1 || { echo "Cannot connect to cluster"; exit 1; }
echo "✓ Connected to cluster"

# Check namespaces
echo ""
echo "Checking namespaces..."
kubectl get namespace kubeflow >/dev/null 2>&1 || kubectl create namespace kubeflow
kubectl get namespace mlflow >/dev/null 2>&1 || echo "Warning: mlflow namespace not found"
kubectl get namespace feast >/dev/null 2>&1 || echo "Warning: feast namespace not found"

# Install Kubeflow Pipelines
echo ""
echo "Checking Kubeflow Pipelines..."
if kubectl get deployment ml-pipeline -n kubeflow >/dev/null 2>&1; then
    echo "✓ Kubeflow Pipelines already installed"
else
    echo "Installing Kubeflow Pipelines..."
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.5"
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io || true
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.0.5" -n kubeflow
    echo "Waiting for KFP to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/ml-pipeline -n kubeflow || true
    echo "✓ Kubeflow Pipelines installed"
fi

# Check Azure Storage Secret
echo ""
echo "Checking Azure Storage Secret..."
if kubectl get secret azure-storage-secret -n kubeflow >/dev/null 2>&1; then
    echo "✓ azure-storage-secret exists"
else
    echo "Creating azure-storage-secret..."
    echo "Enter your Azure Storage connection string:"
    read -s AZURE_CONN_STRING
    kubectl create secret generic azure-storage-secret \
        --namespace=kubeflow \
        --from-literal=AZURE_STORAGE_CONNECTION_STRING="${AZURE_CONN_STRING}"
    echo "✓ azure-storage-secret created"
fi

# Check Ray Operator
echo ""
echo "Checking Ray Operator..."
if kubectl get crd rayjobs.ray.io >/dev/null 2>&1; then
    echo "✓ Ray Operator installed"
else
    echo "Warning: Ray Operator CRDs not found. Please install Ray Operator."
fi

# Check Feast
echo ""
echo "Checking Feast..."
if kubectl get svc feast-service -n feast >/dev/null 2>&1; then
    echo "✓ Feast service found"
else
    echo "Warning: Feast service not found in feast namespace"
fi

# Check MLflow
echo ""
echo "Checking MLflow..."
if kubectl get svc mlflow-service -n mlflow >/dev/null 2>&1; then
    echo "✓ MLflow service found"
else
    echo "Warning: MLflow service not found in mlflow namespace"
fi

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Build and push container images:"
echo "   ./scripts/build_images.sh"
echo ""
echo "2. Compile the pipeline:"
echo "   cd pipeline"
echo "   pip install kfp==2.5.0"
echo "   python spam_detection_pipeline.py"
echo ""
echo "3. Access Kubeflow Pipelines UI:"
echo "   kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80"
echo "   Open http://localhost:8080"
echo ""
echo "4. Upload and run spam_detection_pipeline.yaml"
