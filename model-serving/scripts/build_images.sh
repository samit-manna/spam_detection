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

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract ACR short name (without .azurecr.io)
ACR_SHORT_NAME=$(echo "$ACR_NAME" | sed 's/\.azurecr\.io//')

echo "========================================"
echo "Building Model Serving Images"
echo "ACR: ${ACR_NAME}"
echo "Tag: ${IMAGE_TAG}"
echo "Platform: ${PLATFORM}"
echo "Project Root: ${PROJECT_ROOT}"
echo "========================================"

# Login to ACR
echo "Logging into ACR..."
az acr login --name "${ACR_SHORT_NAME}"

# Build and push api-gateway image
echo ""
echo "Building api-gateway image..."
cd "${PROJECT_ROOT}/api-gateway"
docker build --platform "${PLATFORM}" -t ${ACR_NAME}/api-gateway:${IMAGE_TAG} .
docker push ${ACR_NAME}/api-gateway:${IMAGE_TAG}
echo "✓ api-gateway image pushed"

# Build and push feature-transformer image
echo ""
echo "Building feature-transformer image..."
cd "${PROJECT_ROOT}/feature-transformer"
docker build --platform "${PLATFORM}" -t ${ACR_NAME}/feature-transformer:${IMAGE_TAG} .
docker push ${ACR_NAME}/feature-transformer:${IMAGE_TAG}
echo "✓ feature-transformer image pushed"

# Build and push model-export image
echo ""
echo "Building model-export image..."
cd "${PROJECT_ROOT}/model-export"
docker build --platform "${PLATFORM}" -t ${ACR_NAME}/model-export:${IMAGE_TAG} .
docker push ${ACR_NAME}/model-export:${IMAGE_TAG}
echo "✓ model-export image pushed"

# Build and push batch-inference image
echo ""
echo "Building batch-inference image..."
cd "${PROJECT_ROOT}/batch-inference"
docker build --platform "${PLATFORM}" -t ${ACR_NAME}/batch-inference:${IMAGE_TAG} .
docker push ${ACR_NAME}/batch-inference:${IMAGE_TAG}
echo "✓ batch-inference image pushed"

echo ""
echo "========================================"
echo "All images built and pushed!"
echo "========================================"
echo ""
echo "Images:"
echo "  - ${ACR_NAME}/api-gateway:${IMAGE_TAG}"
echo "  - ${ACR_NAME}/feature-transformer:${IMAGE_TAG}"
echo "  - ${ACR_NAME}/model-export:${IMAGE_TAG}"
echo "  - ${ACR_NAME}/batch-inference:${IMAGE_TAG}"
