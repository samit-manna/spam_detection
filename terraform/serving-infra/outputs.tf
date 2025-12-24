# =============================================================================
# Outputs
# =============================================================================

# cert-manager
output "cert_manager_namespace" {
  description = "cert-manager namespace"
  value       = var.cert_manager_enabled ? kubernetes_namespace.cert_manager[0].metadata[0].name : null
}

# Istio
output "istio_namespace" {
  description = "Istio namespace"
  value       = var.istio_enabled ? kubernetes_namespace.istio_system[0].metadata[0].name : null
}

output "istio_ingress_gateway_service" {
  description = "Istio Ingress Gateway service name"
  value       = var.istio_enabled ? "istio-ingressgateway.${var.istio_namespace}.svc.cluster.local" : null
}

# Knative
output "knative_namespace" {
  description = "Knative Serving namespace"
  value       = var.knative_enabled ? kubernetes_namespace.knative_serving[0].metadata[0].name : null
}

# KServe
output "kserve_namespace" {
  description = "KServe namespace"
  value       = var.kserve_enabled ? kubernetes_namespace.kserve[0].metadata[0].name : null
}

output "kserve_version" {
  description = "KServe version installed"
  value       = var.kserve_enabled ? var.kserve_version : null
}

# Serving
output "serving_namespace" {
  description = "Serving namespace for InferenceServices"
  value       = kubernetes_namespace.serving.metadata[0].name
}

output "serving_service_account" {
  description = "Service account for model serving"
  value       = kubernetes_service_account.serving.metadata[0].name
}

# Component versions
output "component_versions" {
  description = "Installed component versions"
  value = {
    cert_manager = var.cert_manager_enabled ? var.cert_manager_version : "not installed"
    istio        = var.istio_enabled ? var.istio_version : "not installed"
    knative      = var.knative_enabled ? var.knative_version : "not installed"
    kserve       = var.kserve_enabled ? var.kserve_version : "not installed"
  }
}

# Connection info
output "inference_endpoint_info" {
  description = "Information for accessing inference endpoints"
  value = {
    ingress_gateway = var.istio_enabled ? "istio-ingressgateway.${var.istio_namespace}.svc.cluster.local" : null
    domain          = "sslip.io"
    url_pattern     = "<service-name>-<namespace>.sslip.io"
    note            = "Get external IP with: kubectl get svc istio-ingressgateway -n ${var.istio_namespace}"
  }
}
