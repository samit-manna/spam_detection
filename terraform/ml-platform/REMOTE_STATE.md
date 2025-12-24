# ML Platform - Remote State Integration

## Overview

The ML platform now automatically fetches configuration values from the base-infra Terraform state, eliminating the need to manually copy sensitive values like connection strings and passwords.

## What Changed

### ✅ Automated via Remote State

The following values are now automatically retrieved from `base-infra/terraform.tfstate`:

1. **Kubernetes Configuration**
   - Cluster host
   - Client certificate
   - Client key
   - CA certificate

2. **PostgreSQL (MLFlow Backend)**
   - Server FQDN
   - Admin username
   - Admin password
   - Database name

3. **Azure Storage**
   - Connection string (for MLFlow artifacts and Feast offline store)

4. **Redis** (when enabled in base-infra)
   - Hostname
   - Primary access key

### ❌ Removed Variables

These variables are no longer needed in `terraform.tfvars`:

```hcl
# Removed - now from remote state
# resource_group_name
# aks_cluster_name
# mlflow_storage_connection_string
# mlflow_postgres_host
# mlflow_postgres_password
# redis_host
# redis_password
# feast_storage_connection_string
```

## Configuration

### Minimal terraform.tfvars

```hcl
# Only these variables are needed now:
environment  = "dev"
project_name = "mltraining"

mlflow_enabled   = true
kubeflow_enabled = true
ray_enabled      = true
feast_enabled    = true
```

## How It Works

### Remote State Data Source

In `providers.tf`:

```hcl
data "terraform_remote_state" "aks" {
  backend = "local"
  config = {
    path = "../base-infra/terraform.tfstate"
  }
}
```

### Usage in Resources

Access values using:

```hcl
data.terraform_remote_state.aks.outputs.postgres_server_fqdn
data.terraform_remote_state.aks.outputs.storage_account_primary_connection_string
```

## Benefits

✅ **No manual copying** of sensitive values  
✅ **Single source of truth** from base-infra  
✅ **Automatic updates** when base-infra changes  
✅ **Fewer errors** from typos or stale values  
✅ **Better security** - secrets stay in state file  

## Deployment

### Step 1: Deploy Base Infrastructure

```bash
cd terraform/base-infra
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

### Step 2: Deploy ML Platform

```bash
cd ../ml-platform

# Minimal configuration needed
cat > terraform.tfvars <<EOF
environment  = "dev"
project_name = "mltraining"

mlflow_enabled   = true
ray_enabled      = true
feast_enabled    = true
kubeflow_enabled = true
EOF

terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

## Added Outputs in Base-Infra

New outputs in `base-infra/outputs.tf`:

```hcl
# PostgreSQL
output "postgres_server_fqdn" { ... }
output "postgres_admin_username" { ... }
output "postgres_admin_password" { ... }
output "postgres_database_name" { ... }

# Kubernetes (for provider configuration)
output "aks_host" { ... }
output "aks_client_certificate" { ... }
output "aks_client_key" { ... }
output "aks_cluster_ca_certificate" { ... }
```

## Remote Backend (Optional)

For production, use Azure Storage backend:

### base-infra/backend.tf

```hcl
terraform {
  backend "azurerm" {
    resource_group_name  = "tfstate-rg"
    storage_account_name = "tfstatestorage"
    container_name       = "tfstate"
    key                  = "base-infra.terraform.tfstate"
  }
}
```

### ml-platform/providers.tf

```hcl
data "terraform_remote_state" "aks" {
  backend = "azurerm"
  config = {
    resource_group_name  = "tfstate-rg"
    storage_account_name = "tfstatestorage"
    container_name       = "tfstate"
    key                  = "base-infra.terraform.tfstate"
  }
}
```

## Notes

### Redis Status

Redis is currently commented out in base-infra. If you enable it:

1. Uncomment Redis resources in `base-infra/redis.tf`
2. Uncomment Redis outputs in `base-infra/outputs.tf`
3. Uncomment Redis secret in `ml-platform/feast.tf`
4. Run `terraform apply` in both projects

### Security

Sensitive outputs are marked with `sensitive = true`:
- PostgreSQL password
- Storage connection strings
- Redis keys
- Kubernetes certificates

These values are encrypted in the state file and won't be displayed in console output.

## Troubleshooting

### Error: No outputs found

**Problem**: ML platform can't find base-infra outputs

**Solution**: Ensure base-infra is deployed first:
```bash
cd terraform/base-infra
terraform apply
```

### Error: Failed to load state

**Problem**: Can't read base-infra state file

**Solution**: Check the path in `providers.tf`:
```hcl
path = "../base-infra/terraform.tfstate"
```

Make sure this path is correct relative to ml-platform directory.

### Error: Output not found

**Problem**: Trying to access an output that doesn't exist

**Solution**: Check available outputs:
```bash
cd terraform/base-infra
terraform output
```

## Migration from Old Configuration

If you have an existing deployment:

1. **Keep your old terraform.tfvars as backup**
2. **Update to new minimal configuration**
3. **Run terraform plan** to verify changes
4. **Apply if no resource changes** (should be in-place updates only)

The state migration should be seamless as we're only changing variable sources, not resources.

---

**Updated**: November 28, 2025  
**Version**: 2.0 - Remote State Integration
