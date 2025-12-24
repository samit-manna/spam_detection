output "mlflow_url" {
  description = "MLFlow tracking server URL"
  value       = var.mlflow_enabled ? "http://${kubernetes_service.mlflow[0].status[0].load_balancer[0].ingress[0].ip}:5000" : null
}

output "kubeflow_url" {
  description = "Kubeflow central dashboard URL"
  value       = var.kubeflow_enabled ? "Access via port-forward: kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80" : null
}

output "ray_dashboard_url" {
  description = "Ray dashboard URL"
  value       = var.ray_enabled ? "Access via port-forward: kubectl port-forward svc/ray-cluster-head-svc -n ${var.ray_namespace} 8265:8265" : null
}

output "feast_registry_path" {
  description = "Feast feature registry path"
  value       = var.feast_enabled ? "Azure Blob Storage: ${data.terraform_remote_state.aks.outputs.storage_account_primary_connection_string}" : null
  sensitive   = true
}
