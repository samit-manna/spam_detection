variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "mltraining"
}

# =============================================================================
# cert-manager Variables
# =============================================================================

variable "cert_manager_enabled" {
  description = "Enable cert-manager deployment"
  type        = bool
  default     = true
}

variable "cert_manager_version" {
  description = "cert-manager version"
  type        = string
  default     = "v1.13.3"
}

variable "cert_manager_namespace" {
  description = "Kubernetes namespace for cert-manager"
  type        = string
  default     = "cert-manager"
}

# =============================================================================
# Istio Variables
# =============================================================================

variable "istio_enabled" {
  description = "Enable Istio deployment"
  type        = bool
  default     = true
}

variable "istio_version" {
  description = "Istio version"
  type        = string
  default     = "1.20.2"
}

variable "istio_namespace" {
  description = "Kubernetes namespace for Istio"
  type        = string
  default     = "istio-system"
}

# =============================================================================
# Knative Variables
# =============================================================================

variable "knative_enabled" {
  description = "Enable Knative Serving deployment"
  type        = bool
  default     = true
}

variable "knative_version" {
  description = "Knative version"
  type        = string
  default     = "1.12.3"
}

variable "knative_namespace" {
  description = "Kubernetes namespace for Knative Serving"
  type        = string
  default     = "knative-serving"
}

# =============================================================================
# KServe Variables
# =============================================================================

variable "kserve_enabled" {
  description = "Enable KServe deployment"
  type        = bool
  default     = true
}

variable "kserve_version" {
  description = "KServe version"
  type        = string
  default     = "0.12.0"
}

variable "kserve_namespace" {
  description = "Kubernetes namespace for KServe"
  type        = string
  default     = "kserve"
}

# =============================================================================
# Serving Namespace Variables
# =============================================================================

variable "serving_namespace" {
  description = "Kubernetes namespace for model serving workloads"
  type        = string
  default     = "serving"
}

# =============================================================================
# Tags
# =============================================================================

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default = {
    ManagedBy = "Terraform"
    Component = "Serving-Infrastructure"
  }
}
