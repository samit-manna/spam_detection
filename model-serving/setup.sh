#!/bin/bash
#
# Model Serving Deployment Script
# 
# This script deploys model serving components (NOT infrastructure):
# 1. Builds and pushes Docker images
# 2. Exports model from MLflow to ONNX
# 3. Deploys InferenceService and feature transformer
#
# Prerequisites:
#   - Infrastructure deployed via Terraform (../terraform/serving-infra)
#   - Base infrastructure deployed (../terraform/base-infra)
#
# Usage:
#   ./setup.sh [--skip-build] [--staging-only]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_BASE_DIR="$SCRIPT_DIR/../terraform/base-infra"
TERRAFORM_SERVING_DIR="$SCRIPT_DIR/../terraform/serving-infra"
cd "$SCRIPT_DIR"

# Parse arguments
SKIP_BUILD=false
STAGING_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --staging-only)
            STAGING_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./setup.sh [--skip-build] [--staging-only]"
            echo ""
            echo "Options:"
            echo "  --skip-build    Skip Docker image building"
            echo "  --staging-only  Deploy only to staging"
            echo ""
            echo "Note: Infrastructure must be deployed first via Terraform:"
            echo "  cd ../terraform/serving-infra && terraform apply"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check required tools
    for cmd in kubectl docker az envsubst; do
        if ! command -v "$cmd" &> /dev/null; then
            echo -e "${RED}Error: $cmd is not installed${NC}"
            exit 1
        fi
    done
    
    # Check Terraform state exists
    if [[ ! -f "$TERRAFORM_BASE_DIR/terraform.tfstate" ]]; then
        echo -e "${RED}Error: base-infra Terraform state not found${NC}"
        echo "Run: cd ../terraform/base-infra && terraform apply"
        exit 1
    fi
    
    # Check serving-infra deployed
    if [[ ! -f "$TERRAFORM_SERVING_DIR/terraform.tfstate" ]]; then
        echo -e "${RED}Error: serving-infra Terraform state not found${NC}"
        echo "Run: cd ../terraform/serving-infra && terraform apply"
        exit 1
    fi
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
        exit 1
    fi
    
    # Check KServe is installed
    if ! kubectl get crd inferenceservices.serving.kserve.io &> /dev/null; then
        echo -e "${RED}Error: KServe CRDs not found${NC}"
        echo "Run: cd ../terraform/serving-infra && terraform apply"
        exit 1
    fi
    
    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Fetch Terraform outputs
fetch_terraform_outputs() {
    echo -e "${YELLOW}Fetching configuration from Terraform outputs...${NC}"
    
    ACR_NAME=$(cd "$TERRAFORM_BASE_DIR" && terraform output -raw acr_login_server)
    AZURE_STORAGE_ACCOUNT_NAME=$(cd "$TERRAFORM_BASE_DIR" && terraform output -raw storage_account_name)
    AZURE_STORAGE_ACCOUNT_KEY=$(cd "$TERRAFORM_BASE_DIR" && terraform output -raw storage_account_primary_access_key)
    REDIS_HOST=$(cd "$TERRAFORM_BASE_DIR" && terraform output -raw redis_hostname)
    REDIS_KEY=$(cd "$TERRAFORM_BASE_DIR" && terraform output -raw redis_primary_access_key)
    
    export ACR_NAME AZURE_STORAGE_ACCOUNT_NAME AZURE_STORAGE_ACCOUNT_KEY REDIS_HOST REDIS_KEY
    
    echo "  ACR: $ACR_NAME"
    echo "  Storage: $AZURE_STORAGE_ACCOUNT_NAME"
    echo "  Redis: $REDIS_HOST"
}

# Build and push images
build_and_push_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        echo -e "${YELLOW}Skipping image build${NC}"
        return
    fi
    
    echo -e "${YELLOW}Building Docker images...${NC}"
    
    # Login to ACR
    az acr login --name "${ACR_NAME%.azurecr.io}"
    
    # Build images
    docker build -t "${ACR_NAME}/feature-transformer:latest" feature-transformer/
    docker build -t "${ACR_NAME}/model-export:latest" model-export/
    docker build -t "${ACR_NAME}/batch-inference:latest" batch-inference/
    
    # Push images
    docker push "${ACR_NAME}/feature-transformer:latest"
    docker push "${ACR_NAME}/model-export:latest"
    docker push "${ACR_NAME}/batch-inference:latest"
    
    echo -e "${GREEN}Images built and pushed${NC}"
}

# Export model
export_model() {
    echo -e "${YELLOW}Exporting model from MLflow...${NC}"
    
    MLFLOW_URL="http://mlflow-service.mlflow.svc.cluster.local:5000"
    
    kubectl run model-export-$(date +%s) \
        --image="${ACR_NAME}/model-export:latest" \
        --env="MLFLOW_TRACKING_URI=$MLFLOW_URL" \
        --env="AZURE_STORAGE_ACCOUNT_NAME=$AZURE_STORAGE_ACCOUNT_NAME" \
        --env="AZURE_STORAGE_ACCOUNT_KEY=$AZURE_STORAGE_ACCOUNT_KEY" \
        --restart=Never \
        --rm \
        -it \
        -- --model-name spam-detector --model-stage Staging
    
    echo -e "${GREEN}Model exported${NC}"
}

# Deploy services
deploy_services() {
    echo -e "${YELLOW}Deploying services...${NC}"
    
    # Deploy feature transformer
    envsubst < feature-transformer/deployment.yaml | kubectl apply -f -
    
    # Wait for feature transformer
    kubectl wait --for=condition=Available --timeout=300s \
        deployment/feature-transformer -n serving || true
    
    # Deploy staging inference service
    envsubst < inference-service/staging-isvc.yaml | kubectl apply -f -
    
    # Wait for staging
    echo "Waiting for staging InferenceService..."
    kubectl wait --for=condition=Ready --timeout=600s \
        inferenceservice/spam-detector-staging -n kserve || true
    
    echo -e "${GREEN}Staging deployment complete${NC}"
    
    if [[ "$STAGING_ONLY" != "true" ]]; then
        echo -e "${YELLOW}Deploying production...${NC}"
        
        envsubst < inference-service/production-isvc.yaml | kubectl apply -f -
        
        kubectl wait --for=condition=Ready --timeout=600s \
            inferenceservice/spam-detector -n kserve || true
        
        echo -e "${GREEN}Production deployment complete${NC}"
    fi
}

# Verify deployment
verify_deployment() {
    echo -e "${YELLOW}Verifying deployment...${NC}"
    
    echo ""
    echo -e "${GREEN}=== Inference Services ===${NC}"
    kubectl get inferenceservice -n kserve
    
    echo ""
    echo -e "${GREEN}=== Feature Transformer ===${NC}"
    kubectl get deployment,pod -l app=feature-transformer -n serving
    
    # Get service URLs
    echo ""
    echo -e "${GREEN}=== Service URLs ===${NC}"
    
    STAGING_URL=$(kubectl get inferenceservice spam-detector-staging -n kserve -o jsonpath='{.status.url}' 2>/dev/null || echo "Not available")
    echo "Staging URL: $STAGING_URL"
    
    if [[ "$STAGING_ONLY" != "true" ]]; then
        PROD_URL=$(kubectl get inferenceservice spam-detector -n kserve -o jsonpath='{.status.url}' 2>/dev/null || echo "Not available")
        echo "Production URL: $PROD_URL"
    fi
    
    echo ""
    echo -e "${GREEN}Deployment verification complete${NC}"
}

# Main execution
main() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Model Serving Deployment${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    check_prerequisites
    fetch_terraform_outputs
    
    echo ""
    echo "Configuration:"
    echo "  STAGING_ONLY: $STAGING_ONLY"
    echo "  SKIP_BUILD: $SKIP_BUILD"
    echo ""
    
    build_and_push_images
    export_model
    deploy_services
    verify_deployment
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run integration tests: make test"
    echo "  2. Monitor services: make status"
    echo "  3. View logs: kubectl logs -f -l app=feature-transformer -n serving"
    echo ""
}

# Run main
main "$@"
