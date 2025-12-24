locals {
  common_labels = {
    environment = var.environment
    project     = var.project_name
    managed_by  = "terraform"
    component   = "serving-infra"
  }

  # AKS cluster info from remote state
  aks_cluster_name   = data.terraform_remote_state.aks.outputs.aks_cluster_name
  resource_group     = data.terraform_remote_state.aks.outputs.resource_group_name
  storage_account    = data.terraform_remote_state.aks.outputs.storage_account_name
  storage_key        = data.terraform_remote_state.aks.outputs.storage_account_primary_access_key
  redis_hostname     = data.terraform_remote_state.aks.outputs.redis_hostname
  redis_key          = data.terraform_remote_state.aks.outputs.redis_primary_access_key
}
