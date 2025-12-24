terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "~> 1.14"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }

#   backend "azurerm" {
    # Configure backend in backend.tf or via CLI
    # resource_group_name  = "tfstate-rg"
    # storage_account_name = "tfstate<unique>"
    # container_name       = "tfstate"
    # key                  = "ml-platform.terraform.tfstate"
#   }
}

# Read AKS cluster information from the AKS Terraform state
data "terraform_remote_state" "aks" {
  backend = "local"
  config = {
    path = "../base-infra/terraform.tfstate"
  }
}

# Get AKS credentials from base infrastructure
# data "azurerm_kubernetes_cluster" "main" {
#   name                = var.aks_cluster_name
#   resource_group_name = var.resource_group_name
# }

provider "azurerm" {
  features {}
}

# provider "kubernetes" {
#   host                   = data.azurerm_kubernetes_cluster.main.kube_config.0.host
#   client_certificate     = base64decode(data.azurerm_kubernetes_cluster.main.kube_config.0.client_certificate)
#   client_key             = base64decode(data.azurerm_kubernetes_cluster.main.kube_config.0.client_key)
#   cluster_ca_certificate = base64decode(data.azurerm_kubernetes_cluster.main.kube_config.0.cluster_ca_certificate)
# }

provider "kubernetes" {
  host                   = data.terraform_remote_state.aks.outputs.aks_host
  client_certificate     = base64decode(data.terraform_remote_state.aks.outputs.aks_client_certificate)
  client_key             = base64decode(data.terraform_remote_state.aks.outputs.aks_client_key)
  cluster_ca_certificate = base64decode(data.terraform_remote_state.aks.outputs.aks_cluster_ca_certificate)
}

# provider "helm" {
#   kubernetes {
#     host                   = data.azurerm_kubernetes_cluster.main.kube_config.0.host
#     client_certificate     = base64decode(data.azurerm_kubernetes_cluster.main.kube_config.0.client_certificate)
#     client_key             = base64decode(data.azurerm_kubernetes_cluster.main.kube_config.0.client_key)
#     cluster_ca_certificate = base64decode(data.azurerm_kubernetes_cluster.main.kube_config.0.cluster_ca_certificate)
#   }
# }

provider "helm" {
  kubernetes {
    host                   = data.terraform_remote_state.aks.outputs.aks_host
    client_certificate     = base64decode(data.terraform_remote_state.aks.outputs.aks_client_certificate)
    client_key             = base64decode(data.terraform_remote_state.aks.outputs.aks_client_key)
    cluster_ca_certificate = base64decode(data.terraform_remote_state.aks.outputs.aks_cluster_ca_certificate)
  }
}

provider "kubectl" {
  host                   = data.terraform_remote_state.aks.outputs.aks_host
  client_certificate     = base64decode(data.terraform_remote_state.aks.outputs.aks_client_certificate)
  client_key             = base64decode(data.terraform_remote_state.aks.outputs.aks_client_key)
  cluster_ca_certificate = base64decode(data.terraform_remote_state.aks.outputs.aks_cluster_ca_certificate)
  load_config_file       = false
}
