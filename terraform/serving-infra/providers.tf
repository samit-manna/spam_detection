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
    time = {
      source  = "hashicorp/time"
      version = "~> 0.10"
    }
  }

  # Uncomment for remote state
  # backend "azurerm" {
  #   resource_group_name  = "tfstate-rg"
  #   storage_account_name = "tfstate<unique>"
  #   container_name       = "tfstate"
  #   key                  = "serving-infra.terraform.tfstate"
  # }
}

# Read AKS cluster information from base-infra state
data "terraform_remote_state" "aks" {
  backend = "local"
  config = {
    path = "../base-infra/terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
}

provider "kubernetes" {
  host                   = data.terraform_remote_state.aks.outputs.aks_host
  client_certificate     = base64decode(data.terraform_remote_state.aks.outputs.aks_client_certificate)
  client_key             = base64decode(data.terraform_remote_state.aks.outputs.aks_client_key)
  cluster_ca_certificate = base64decode(data.terraform_remote_state.aks.outputs.aks_cluster_ca_certificate)
}

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
