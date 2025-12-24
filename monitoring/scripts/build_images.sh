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
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

# Extract ACR short name (without .azurecr.io)
ACR_SHORT_NAME=$(echo "$ACR_NAME" | sed 's/\.azurecr\.io//')

echo "========================================"
echo "Building Monitoring Images"
echo "ACR: ${ACR_NAME}"
echo "Tag: ${IMAGE_TAG}"
echo "Platform: ${PLATFORM}"
echo "Project Root: ${PROJECT_ROOT}"
echo "========================================"

# Login to ACR
echo "Logging into ACR..."
az acr login --name "${ACR_SHORT_NAME}"

# Build and push baseline-generator image
echo ""
echo "Building baseline-generator image..."
cd "${PROJECT_ROOT}"
docker build --platform "${PLATFORM}" \
    -t ${ACR_NAME}/baseline-generator:${IMAGE_TAG} \
    -f baseline/Dockerfile \
    "${WORKSPACE_ROOT}"
docker push ${ACR_NAME}/baseline-generator:${IMAGE_TAG}
echo "✓ baseline-generator image pushed"

# Build and push drift-detector image
echo ""
echo "Building drift-detector image..."
docker build --platform "${PLATFORM}" \
    -t ${ACR_NAME}/drift-detector:${IMAGE_TAG} \
    -f drift_detector/Dockerfile \
    "${WORKSPACE_ROOT}"
docker push ${ACR_NAME}/drift-detector:${IMAGE_TAG}
echo "✓ drift-detector image pushed"

echo ""
echo "========================================"
echo "All images built and pushed!"
echo "========================================"
echo ""
echo "Images:"
echo "  - ${ACR_NAME}/baseline-generator:${IMAGE_TAG}"
echo "  - ${ACR_NAME}/drift-detector:${IMAGE_TAG}"
