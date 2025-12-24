#!/bin/bash
# setup_env.sh - Helper script to set up environment variables for Feast

set -e

echo "====================================================================="
echo "Setting up environment for Feast Feature Store"
echo "====================================================================="

# Get Terraform outputs
cd ../../../terraform/base-infra

echo ""
echo "📦 Getting Azure Storage Account details from Terraform..."

# Check if terraform state exists
if [ ! -f "terraform.tfstate" ]; then
    echo "❌ ERROR: terraform.tfstate not found"
    echo "Please run 'terraform apply' in terraform/base-infra first"
    exit 1
fi

STORAGE_ACCOUNT_NAME=$(terraform output -raw storage_account_name 2>/dev/null || echo "")
RESOURCE_GROUP=$(terraform output -raw resource_group_name 2>/dev/null || echo "")

if [ -z "$STORAGE_ACCOUNT_NAME" ] || [ -z "$RESOURCE_GROUP" ]; then
    echo "❌ ERROR: Could not get Terraform outputs"
    echo "Please ensure Terraform has been applied successfully"
    exit 1
fi

echo "✅ Storage Account: $STORAGE_ACCOUNT_NAME"
echo "✅ Resource Group: $RESOURCE_GROUP"

# Get storage account key
echo ""
echo "🔑 Fetching storage account key from Azure..."

STORAGE_ACCOUNT_KEY=$(az storage account keys list \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query '[0].value' -o tsv 2>/dev/null || echo "")

if [ -z "$STORAGE_ACCOUNT_KEY" ]; then
    echo "❌ ERROR: Could not retrieve storage account key"
    echo "Please ensure you're logged in to Azure CLI:"
    echo "   az login"
    exit 1
fi

echo "✅ Retrieved storage account key"

# Get Redis connection string if available
cd ../ml-platform

REDIS_CONNECTION_STRING=""

if [ -f "terraform.tfstate" ]; then
    echo ""
    echo "🔍 Checking for Redis configuration..."
    
    REDIS_HOST=$(terraform output -raw redis_host 2>/dev/null || echo "")
    REDIS_PASSWORD=$(terraform output -raw redis_primary_access_key 2>/dev/null || echo "")
    
    if [ -n "$REDIS_HOST" ] && [ -n "$REDIS_PASSWORD" ]; then
        REDIS_CONNECTION_STRING="rediss://:${REDIS_PASSWORD}@${REDIS_HOST}:6380/0"
        echo "✅ Redis connection string configured"
    else
        echo "⚠️  Redis not found in Terraform state"
    fi
else
    echo ""
    echo "⚠️  ml-platform Terraform state not found - skipping Redis"
fi

# Create .env file
cd ../../phase1/feast/feature_repo

echo ""
echo "📝 Creating .env file..."

cat > .env << EOF
# Azure Storage Configuration
export AZURE_STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME
export AZURE_STORAGE_ACCOUNT_KEY=$STORAGE_ACCOUNT_KEY

# Redis Configuration (optional - for online store)
EOF

if [ -n "$REDIS_CONNECTION_STRING" ]; then
    echo "export FEAST_REDIS_CONNECTION_STRING=$REDIS_CONNECTION_STRING" >> .env
else
    echo "# export FEAST_REDIS_CONNECTION_STRING=rediss://:password@host:6380/0" >> .env
fi

echo ""
echo "✅ Created .env file in phase1/feast/feature_repo/"

# Also export for current shell
export AZURE_STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME
export AZURE_STORAGE_ACCOUNT_KEY=$STORAGE_ACCOUNT_KEY

if [ -n "$REDIS_CONNECTION_STRING" ]; then
    export FEAST_REDIS_CONNECTION_STRING=$REDIS_CONNECTION_STRING
fi

echo ""
echo "====================================================================="
echo "✅ Environment setup complete!"
echo "====================================================================="
echo ""
echo "To use these variables in your current shell, run:"
echo "   source .env"
echo ""
echo "Or export them directly:"
echo "   export AZURE_STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME"
echo "   export AZURE_STORAGE_ACCOUNT_KEY=<your-key>"

if [ -n "$REDIS_CONNECTION_STRING" ]; then
    echo "   export FEAST_REDIS_CONNECTION_STRING=<your-redis-connection-string>"
fi

echo ""
echo "Next steps:"
echo "   1. source .env"
echo "   2. python verify_setup.py"
echo "   3. python apply_feast.py"
echo "   4. python test_feature.py"
echo ""
