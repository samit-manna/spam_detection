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

# MLFlow Variables
variable "mlflow_enabled" {
  description = "Enable MLFlow deployment"
  type        = bool
  default     = true
}

variable "mlflow_namespace" {
  description = "Kubernetes namespace for MLFlow"
  type        = string
  default     = "mlflow"
}

# Kubeflow Variables
variable "kubeflow_enabled" {
  description = "Enable Kubeflow deployment"
  type        = bool
  default     = true
}

variable "kubeflow_namespace" {
  description = "Kubernetes namespace for Kubeflow"
  type        = string
  default     = "kubeflow"
}

# Ray Variables
variable "ray_enabled" {
  description = "Enable Ray deployment"
  type        = bool
  default     = true
}

variable "ray_namespace" {
  description = "Kubernetes namespace for Ray"
  type        = string
  default     = "ray"
}

variable "ray_head_cpu" {
  description = "CPU request for Ray head node"
  type        = string
  default     = "2"
}

variable "ray_head_memory" {
  description = "Memory request for Ray head node"
  type        = string
  default     = "8Gi"
}

variable "ray_worker_replicas" {
  description = "Number of Ray worker replicas"
  type        = number
  default     = 1
}

variable "ray_gpu_worker_replicas" {
  description = "Number of Ray GPU worker replicas"
  type        = number
  default     = 0
}

# Feast Variables
variable "feast_enabled" {
  description = "Enable Feast deployment"
  type        = bool
  default     = true
}

variable "feast_namespace" {
  description = "Kubernetes namespace for Feast"
  type        = string
  default     = "feast"
}

# Tags
variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default = {
    ManagedBy = "Terraform"
    Component = "ML-Platform"
  }
}
