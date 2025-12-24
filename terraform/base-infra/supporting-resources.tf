# Get current client configuration
data "azurerm_client_config" "current" {}

# Storage Account for ML artifacts, datasets, and model storage
resource "azurerm_storage_account" "main" {
  name                     = "${var.project_name}${var.environment}sa${random_string.unique.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = var.storage_account_tier
  account_replication_type = var.storage_account_replication_type
  account_kind             = "StorageV2"
  is_hns_enabled           = true # Enable hierarchical namespace for Data Lake
  
  # Disable anonymous blob access for security
  allow_nested_items_to_be_public = false

  blob_properties {
    # versioning_enabled = true
    
    delete_retention_policy {
      days = 7
    }
    
    container_delete_retention_policy {
      days = 7
    }
  }

  network_rules {
    default_action             = "Allow"
    virtual_network_subnet_ids = [azurerm_subnet.services.id]
    bypass                     = ["AzureServices"]
  }
  # public_network_access_enabled = false

  tags = local.common_tags
}

# Storage Containers
resource "azurerm_storage_container" "datasets" {
  name                  = "datasets"
  storage_account_name = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "mlflow" {
  name                  = "mlflow"
  storage_account_name = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "feast" {
  name                  = "feast"
  storage_account_name = azurerm_storage_account.main.name
  container_access_type = "private"
}

# Key Vault for secrets management
resource "azurerm_key_vault" "main" {
  name                       = "${var.project_name}-${var.environment}-kv"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false

  network_acls {
    default_action             = "Allow"
    bypass                     = "AzureServices"
    virtual_network_subnet_ids = [azurerm_subnet.services.id]
  }

  tags = local.common_tags
}

# Key Vault Access Policy for current user/service principal
resource "azurerm_key_vault_access_policy" "current" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  secret_permissions = [
    "Get",
    "List",
    "Set",
    "Delete",
    "Purge",
    "Recover"
  ]

  key_permissions = [
    "Get",
    "List",
    "Create",
    "Delete",
    "Purge",
    "Recover"
  ]
}

# Key Vault Access Policy for AKS
resource "azurerm_key_vault_access_policy" "aks" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id

  secret_permissions = [
    "Get",
    "List"
  ]
}

# Store Redis connection string in Key Vault
# resource "azurerm_key_vault_secret" "redis_connection_string" {
#   name         = "redis-connection-string"
#   value        = azurerm_redis_cache.main.primary_connection_string
#   key_vault_id = azurerm_key_vault.main.id

#   depends_on = [azurerm_key_vault_access_policy.current]
# }

# Store Storage Account connection string in Key Vault
resource "azurerm_key_vault_secret" "storage_connection_string" {
  name         = "storage-connection-string"
  value        = azurerm_storage_account.main.primary_connection_string
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.current]
}

# PostgreSQL Flexible Server for MLFlow backend
resource "azurerm_postgresql_flexible_server" "mlflow" {
  name                   = "${var.project_name}-${var.environment}-psql-${random_string.unique.result}"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "14"
  administrator_login    = "mlflowadmin"
  administrator_password = random_string.psql_password.result
  storage_mb             = 32768
  sku_name               = "B_Standard_B1ms"
  zone                   = "1"

  tags = local.common_tags
}

resource "azurerm_postgresql_flexible_server_database" "mlflow" {
  name      = "mlflow"
  server_id = azurerm_postgresql_flexible_server.mlflow.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# Firewall rule to allow Azure services to access PostgreSQL
resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_azure_services" {
  name             = "AllowAllAzureServicesAndResourcesWithinAzureIps"
  server_id        = azurerm_postgresql_flexible_server.mlflow.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# Random password for PostgreSQL
resource "random_string" "psql_password" {
  length  = 24
  special = true
}

# Store PostgreSQL password in Key Vault
resource "azurerm_key_vault_secret" "psql_password" {
  name         = "mlflow-psql-password"
  value        = random_string.psql_password.result
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.current]
}

# Role assignment for Storage Blob Data Contributor to AKS
resource "azurerm_role_assignment" "aks_storage" {
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  role_definition_name             = "Storage Blob Data Contributor"
  scope                            = azurerm_storage_account.main.id
  skip_service_principal_aad_check = true
}
