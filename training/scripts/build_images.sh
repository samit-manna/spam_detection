#!/bin/bash
set -e

# Function to display usage
usage() {
    echo "Usage: $0 <ACR_NAME> [IMAGE_TAG] [PLATFORM]"
    echo ""
    echo "Arguments:"
    echo "  ACR_NAME      Azure Container Registry name (required)"
    echo "  IMAGE_TAG     Docker image tag (optional, default: latest)"
    echo "  PLATFORM      Docker build platform (optional, default: linux/amd64)"
    echo ""
    echo "Example:"
    echo "  $0 myacr.azurecr.io v1.0.0"
    echo "  $0 myacr.azurecr.io latest linux/arm64"
    exit 1
}

# Check if required argument is provided
if [ -z "$1" ]; then
    echo "Error: ACR_NAME is required"
    echo ""
    usage
fi

# Configuration
ACR_NAME="$1"
IMAGE_TAG="${2:-latest}"
PLATFORM="${3:-linux/amd64}"

# Extract ACR short name (without .azurecr.io)
ACR_SHORT_NAME=$(echo "$ACR_NAME" | sed 's/\.azurecr\.io//')

echo "========================================"
echo "Building ML Training Pipeline Images"
echo "ACR: ${ACR_NAME}"
echo "Tag: ${IMAGE_TAG}"
echo "Platform: ${PLATFORM}"
echo "========================================"

# Login to ACR
echo "Logging into ACR..."
az acr login --name "${ACR_SHORT_NAME}"

# Build and push data-prep image
echo ""
echo "Building data-prep image..."
cd "$(dirname "$0")/../docker/data-prep"
docker build --platform "${PLATFORM}" -t ${ACR_NAME}/ml-data-prep:${IMAGE_TAG} .
docker push ${ACR_NAME}/ml-data-prep:${IMAGE_TAG}
echo "✓ data-prep image pushed"

# Build and push ray-train image
echo ""
echo "Building ray-train image..."
cd ../ray-train
docker build --platform "${PLATFORM}" -t ${ACR_NAME}/ml-ray-train:${IMAGE_TAG} .
docker push ${ACR_NAME}/ml-ray-train:${IMAGE_TAG}
echo "✓ ray-train image pushed"

# Build and push mlflow-ops image
echo ""
echo "Building mlflow-ops image..."
cd ../mlflow-ops
docker build --platform "${PLATFORM}" -t ${ACR_NAME}/ml-mlflow-ops:${IMAGE_TAG} .
docker push ${ACR_NAME}/ml-mlflow-ops:${IMAGE_TAG}
echo "✓ mlflow-ops image pushed"

echo ""
echo "========================================"
echo "All images built and pushed!"
echo "========================================"
echo ""
echo "Images:"
echo "  - ${ACR_NAME}/ml-data-prep:${IMAGE_TAG}"
echo "  - ${ACR_NAME}/ml-ray-train:${IMAGE_TAG}"
echo "  - ${ACR_NAME}/ml-mlflow-ops:${IMAGE_TAG}"
