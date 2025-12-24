variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "mltraining"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = ""
}

# Network Variables
variable "vnet_address_space" {
  description = "Address space for VNet"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "aks_subnet_address_prefix" {
  description = "Address prefix for AKS subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "services_subnet_address_prefix" {
  description = "Address prefix for services subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "gpu_subnet_address_prefix" {
  description = "Address prefix for GPU node pool subnet"
  type        = string
  default     = "10.0.3.0/24"
}

# AKS Variables
variable "aks_kubernetes_version" {
  description = "Kubernetes version for AKS cluster"
  type        = string
  default     = "1.28.3"
}

variable "aks_system_node_count" {
  description = "Number of nodes in system node pool"
  type        = number
  default     = 3
}

variable "aks_system_node_vm_size" {
  description = "VM size for system node pool"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "aks_gpu_node_count" {
  description = "Number of nodes in GPU node pool"
  type        = number
  default     = 1
}

variable "aks_gpu_node_max_count" {
  description = "Maximum number of nodes in GPU node pool (for autoscaling)"
  type        = number
  default     = 5
}

variable "aks_gpu_node_vm_size" {
  description = "VM size for GPU node pool"
  type        = string
  default     = "Standard_NC6s_v3" # NVIDIA V100
}

variable "aks_spot_node_count" {
  description = "Number of nodes in spot instance pool"
  type        = number
  default     = 0
}

variable "aks_spot_node_max_count" {
  description = "Maximum number of nodes in spot instance pool"
  type        = number
  default     = 10
}

variable "aks_spot_node_vm_size" {
  description = "VM size for spot instance pool"
  type        = string
  default     = "Standard_D8s_v3"
}

# ACR Variables
variable "acr_sku" {
  description = "SKU for Azure Container Registry"
  type        = string
  default     = "Premium"
}

# Redis Variables
variable "deploy_redis" {
  description = "Whether to deploy Azure Cache for Redis"
  type        = bool
  default     = false
}

variable "redis_capacity" {
  description = "Capacity for Redis cache"
  type        = number
  default     = 1
}

variable "redis_family" {
  description = "Family for Redis cache"
  type        = string
  default     = "C"
}

variable "redis_sku_name" {
  description = "SKU name for Redis cache"
  type        = string
  default     = "Standard"
}

# Storage Variables
variable "storage_account_tier" {
  description = "Storage account tier"
  type        = string
  default     = "Standard"
}

variable "storage_account_replication_type" {
  description = "Storage account replication type"
  type        = string
  default     = "LRS"
}

# Tags
variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default = {
    ManagedBy = "Terraform"
    Purpose   = "ML-Training"
  }
}
