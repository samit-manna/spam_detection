#!/bin/bash
# apply_to_aks.sh - Apply Feast features to AKS (serverless mode)
#
# This script registers features in Feast without running a Feast server.
# Features are registered to Azure Blob Storage and accessed directly by
# training/inference jobs using the Feast SDK.

set -e

NAMESPACE="feast"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "Applying Feast Features to AKS (Serverless Mode)"
echo "======================================================================"

# Load Azure credentials
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
elif [ -f .env ]; then
    source .env
fi

if [ -z "$AZURE_STORAGE_ACCOUNT_NAME" ] || [ -z "$AZURE_STORAGE_ACCOUNT_KEY" ]; then
    echo "⚠️  Azure credentials not in .env, fetching from Kubernetes secret..."
    export AZURE_STORAGE_ACCOUNT_NAME=$(kubectl get secret feast-storage-secret -n $NAMESPACE -o jsonpath='{.data.azure-storage-account-name}' | base64 -d)
    export AZURE_STORAGE_ACCOUNT_KEY=$(kubectl get secret feast-storage-secret -n $NAMESPACE -o jsonpath='{.data.azure-storage-account-key}' | base64 -d)
fi

if [ -z "$AZURE_STORAGE_ACCOUNT_NAME" ] || [ -z "$AZURE_STORAGE_ACCOUNT_KEY" ]; then
    echo "❌ Error: Could not obtain Azure credentials"
    exit 1
fi

echo "✅ Using storage account: $AZURE_STORAGE_ACCOUNT_NAME"

# Create/update ConfigMap with feature definitions
echo ""
echo "📦 Creating ConfigMap with feature definitions..."
kubectl create configmap feast-feature-defs \
    --from-file="$SCRIPT_DIR/entities.py" \
    --from-file="$SCRIPT_DIR/features.py" \
    --from-file="$SCRIPT_DIR/feature_store.yaml" \
    -n $NAMESPACE \
    --dry-run=client -o yaml | kubectl apply -f -

echo "✅ ConfigMap created"

# Clean up old feast-apply jobs
echo ""
echo "🧹 Cleaning up old jobs..."
kubectl delete jobs -n $NAMESPACE -l app=feast-apply --ignore-not-found

# Run feast apply as a Job (no server needed)
echo ""
echo "🚀 Running feast apply..."
JOB_NAME="feast-apply-$(date +%s)"

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: $NAMESPACE
  labels:
    app: feast-apply
spec:
  ttlSecondsAfterFinished: 3600
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: feast-apply
        image: python:3.10-slim
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -e
          pip install -q feast adlfs azure-identity pyarrow pandas
          
          # Copy feature definitions
          cp /feast-defs/* /feast-repo/
          cd /feast-repo
          
          echo "📋 Feature store configuration:"
          cat feature_store.yaml
          echo ""
          echo "🔄 Running feast apply..."
          feast apply
          
          echo ""
          echo "📊 Registered feature views:"
          feast feature-views list
          
          echo ""
          echo "📊 Registered entities:"
          feast entities list
        env:
        - name: AZURE_STORAGE_ACCOUNT_NAME
          valueFrom:
            secretKeyRef:
              name: feast-storage-secret
              key: azure-storage-account-name
        - name: AZURE_STORAGE_ACCOUNT_KEY
          valueFrom:
            secretKeyRef:
              name: feast-storage-secret
              key: azure-storage-account-key
        volumeMounts:
        - name: feature-definitions
          mountPath: /feast-defs
        - name: feast-repo
          mountPath: /feast-repo
      volumes:
      - name: feature-definitions
        configMap:
          name: feast-feature-defs
      - name: feast-repo
        emptyDir: {}
EOF

# Wait for job to complete
echo "⏳ Waiting for job $JOB_NAME to complete..."
kubectl wait --for=condition=complete job/$JOB_NAME -n $NAMESPACE --timeout=300s

# Show job logs
echo ""
echo "📋 Job output:"
kubectl logs job/$JOB_NAME -n $NAMESPACE

echo ""
echo "======================================================================"
echo "✅ Feast features applied successfully (serverless mode)"
echo "======================================================================"
echo ""
echo "Features are registered in Azure Blob Storage. No Feast server needed."
echo "Training/inference jobs access features directly via Feast SDK."
echo ""
echo "Usage in Python:"
echo "  from feast import FeatureStore"
echo "  store = FeatureStore(repo_path='path/to/feature_repo')"
echo "  features = store.get_historical_features(...).to_df()"
