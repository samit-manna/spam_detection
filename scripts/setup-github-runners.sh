#!/bin/bash
# =============================================================================
# Setup GitHub Actions Runner Controller (ARC) on AKS
# =============================================================================
# This script helps set up self-hosted GitHub Actions runners in your AKS cluster
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GitHub Actions Runner Controller Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# Prerequisites Check
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

if ! command -v terraform &> /dev/null; then
    echo -e "${RED}terraform not found. Please install terraform first.${NC}"
    exit 1
fi

if ! command -v az &> /dev/null; then
    echo -e "${RED}az CLI not found. Please install Azure CLI first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites found${NC}"
echo ""

# -----------------------------------------------------------------------------
# Instructions
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Before proceeding, you need:${NC}"
echo ""
echo "1. A GitHub Personal Access Token (PAT) with these scopes:"
echo "   - repo (Full control of private repositories)"
echo "   - workflow (Update GitHub Action workflows)"
echo "   - admin:org (if using organization runners)"
echo ""
echo "2. Your GitHub username or organization name"
echo ""
echo "To create a PAT:"
echo "   1. Go to https://github.com/settings/tokens"
echo "   2. Click 'Generate new token (classic)'"
echo "   3. Select the scopes mentioned above"
echo "   4. Copy the token"
echo ""

# -----------------------------------------------------------------------------
# Collect Information
# -----------------------------------------------------------------------------
read -p "Enter your GitHub username or organization: " GITHUB_ORG
read -p "Enter your repository name [spam_detection]: " GITHUB_REPO
GITHUB_REPO=${GITHUB_REPO:-spam_detection}
read -sp "Enter your GitHub PAT: " GITHUB_PAT
echo ""

if [ -z "$GITHUB_PAT" ]; then
    echo -e "${RED}GitHub PAT is required${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  GitHub Org/User: ${GITHUB_ORG}"
echo "  Repository: ${GITHUB_REPO}"
echo "  PAT: ****${GITHUB_PAT: -4}"
echo ""

read -p "Continue with these settings? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# -----------------------------------------------------------------------------
# Update Terraform Variables
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}Updating Terraform configuration...${NC}"

cd "$(dirname "$0")/../terraform/ml-platform"

# Check if terraform.tfvars exists
if [ ! -f terraform.tfvars ]; then
    cp terraform.tfvars.example terraform.tfvars
    echo -e "${GREEN}Created terraform.tfvars from example${NC}"
fi

# Update or add GitHub runner variables
if grep -q "github_runners_enabled" terraform.tfvars; then
    # Update existing values
    sed -i.bak "s/github_runners_enabled.*/github_runners_enabled = true/" terraform.tfvars
    sed -i.bak "s/github_org.*/github_org = \"${GITHUB_ORG}\"/" terraform.tfvars
    sed -i.bak "s/github_repo.*/github_repo = \"${GITHUB_REPO}\"/" terraform.tfvars
    sed -i.bak "s/github_pat.*/github_pat = \"${GITHUB_PAT}\"/" terraform.tfvars
    rm -f terraform.tfvars.bak
else
    # Append new values
    cat >> terraform.tfvars <<EOF

# GitHub Actions Runner Configuration
github_runners_enabled = true
github_org             = "${GITHUB_ORG}"
github_repo            = "${GITHUB_REPO}"
github_pat             = "${GITHUB_PAT}"
runner_min_count       = 0
runner_max_count       = 3
EOF
fi

echo -e "${GREEN}✓ Updated terraform.tfvars${NC}"

# -----------------------------------------------------------------------------
# Apply Terraform
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}Applying Terraform configuration...${NC}"

# First update base-infra to add the new output
echo "Updating base-infra outputs..."
cd ../base-infra
terraform init -upgrade
terraform apply -auto-approve -target=output.kubelet_identity_client_id -target=output.kubelet_identity_object_id || true

# Now apply ml-platform
echo ""
echo "Deploying GitHub Actions Runner Controller..."
cd ../ml-platform
terraform init -upgrade
terraform apply -auto-approve

# -----------------------------------------------------------------------------
# Verify Installation
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"

# Wait for controller to be ready
echo "Waiting for ARC controller to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/arc-gha-runner-scale-set-controller -n github-runners || true

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Your self-hosted runners are now configured."
echo ""
echo "To use them in your GitHub Actions workflow:"
echo ""
echo "  jobs:"
echo "    build:"
echo "      runs-on: ml-platform-runners"
echo ""
echo "Check runner status:"
echo "  kubectl get pods -n github-runners"
echo ""
echo "View runner logs:"
echo "  kubectl logs -n github-runners -l app.kubernetes.io/component=runner"
echo ""
