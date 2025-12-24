# Log Analytics Workspace for AKS monitoring
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.project_name}-${var.environment}-law"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.common_tags
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.project_name}-${var.environment}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${var.project_name}-${var.environment}"
  kubernetes_version  = var.aks_kubernetes_version

  default_node_pool {
    name            = "system"
    node_count      = var.aks_system_node_count
    vm_size         = var.aks_system_node_vm_size
    vnet_subnet_id  = azurerm_subnet.aks.id
    os_disk_size_gb = 128
    os_disk_type    = "Managed"
    type            = "VirtualMachineScaleSets"
    
    upgrade_settings {
      max_surge = "10%"
    }
    
    node_labels = {
      "nodepool-type" = "system"
      "workload"      = "system"
    }

    tags = local.common_tags
  }

  identity {
    type = "SystemAssigned"
  }

    # Enable OIDC Issuer and Workload Identity
  oidc_issuer_enabled       = true
  workload_identity_enabled = true

  network_profile {
    network_plugin     = "azure"
    network_policy     = "azure"
    dns_service_ip     = "10.0.4.10"
    service_cidr       = "10.0.4.0/24"
    load_balancer_sku  = "standard"
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  azure_policy_enabled = true

  auto_scaler_profile {
    balance_similar_node_groups      = true
    expander                         = "random"
    max_graceful_termination_sec     = 600
    max_node_provisioning_time       = "15m"
    max_unready_nodes                = 3
    new_pod_scale_up_delay           = "10s"
    scale_down_delay_after_add       = "10m"
    scale_down_delay_after_delete    = "10s"
    scale_down_delay_after_failure   = "3m"
    scan_interval                    = "10s"
    scale_down_unneeded              = "10m"
    scale_down_unready               = "20m"
    scale_down_utilization_threshold = "0.5"
  }

  tags = local.common_tags
}

# GPU Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.aks_gpu_node_vm_size
  vnet_subnet_id        = azurerm_subnet.aks_gpu.id
  
  # Autoscaling configuration
  enable_auto_scaling   = true
  min_count             = var.aks_gpu_node_count
  max_count             = var.aks_gpu_node_max_count
  
  os_disk_size_gb       = 256
  os_disk_type          = "Managed"
  os_type               = "Linux"
  
  node_labels = {
    "nodepool-type" = "gpu"
    "workload"      = "ml-training"
    "gpu-type"      = "nvidia"
  }

  node_taints = [
    "nvidia.com/gpu=true:NoSchedule"
  ]

  tags = merge(
    local.common_tags,
    {
      "nodepool" = "gpu"
    }
  )
}

# Spot Instance Node Pool for cost-effective training
resource "azurerm_kubernetes_cluster_node_pool" "spot" {
  name                  = "spot"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.aks_spot_node_vm_size
  vnet_subnet_id        = azurerm_subnet.aks.id
  
  # Autoscaling configuration
  enable_auto_scaling   = true
  min_count             = var.aks_spot_node_count
  max_count             = var.aks_spot_node_max_count
  
  os_disk_size_gb       = 128
  os_disk_type          = "Managed"
  os_type               = "Linux"
  
  # Spot configuration
  priority              = "Spot"
  eviction_policy       = "Delete"
  spot_max_price        = -1 # Pay up to regular price
  
  node_labels = {
    "nodepool-type" = "spot"
    "workload"      = "ml-training"
  }

  node_taints = [
    "kubernetes.azure.com/scalesetpriority=spot:NoSchedule"
  ]

  tags = merge(
    local.common_tags,
    {
      "nodepool" = "spot"
    }
  )
}

# Role assignment for AKS to pull from ACR
resource "azurerm_role_assignment" "aks_acr" {
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.main.id
}

resource "azurerm_role_assignment" "aks_identity_acr" {
  principal_id                     = azurerm_kubernetes_cluster.main.identity[0].principal_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.main.id
}

# Role assignment for AKS to manage network
resource "azurerm_role_assignment" "aks_network" {
  principal_id                     = azurerm_kubernetes_cluster.main.identity[0].principal_id
  role_definition_name             = "Network Contributor"
  scope                            = azurerm_virtual_network.main.id
}

# Explicit ACR attachment to AKS for better integration
# This ensures kubelet credentials are properly configured for ACR access
resource "null_resource" "attach_acr_to_aks" {
  depends_on = [
    azurerm_kubernetes_cluster.main,
    azurerm_container_registry.main,
    azurerm_role_assignment.aks_acr,
    azurerm_kubernetes_cluster_node_pool.gpu,
    azurerm_kubernetes_cluster_node_pool.spot
  ]

  provisioner "local-exec" {
    command = "az aks update --name ${azurerm_kubernetes_cluster.main.name} --resource-group ${azurerm_resource_group.main.name} --attach-acr ${azurerm_container_registry.main.name}"
  }

  # Trigger re-run if ACR or AKS changes
  triggers = {
    aks_id = azurerm_kubernetes_cluster.main.id
    acr_id = azurerm_container_registry.main.id
  }
}

