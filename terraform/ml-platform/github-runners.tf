# =============================================================================
# GitHub Actions Runner Controller (ARC) for Self-Hosted Runners
# =============================================================================
# This deploys GitHub's official Actions Runner Controller on AKS
# Runners have direct access to internal services (MLflow, Redis, KServe)
# =============================================================================

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

variable "github_runners_enabled" {
  description = "Enable GitHub Actions Runner Controller"
  type        = bool
  default     = true
}

variable "github_runners_namespace" {
  description = "Kubernetes namespace for GitHub runners"
  type        = string
  default     = "github-runners"
}

variable "github_org" {
  description = "GitHub organization or username"
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository name (for repo-level runners)"
  type        = string
  default     = "spam_detection"
}

variable "github_pat" {
  description = "GitHub Personal Access Token with repo and admin:org scope"
  type        = string
  sensitive   = true
  default     = ""
}

variable "runner_min_count" {
  description = "Minimum number of runners"
  type        = number
  default     = 0  # Scale to zero when idle
}

variable "runner_max_count" {
  description = "Maximum number of runners"
  type        = number
  default     = 3
}

# -----------------------------------------------------------------------------
# Namespace
# -----------------------------------------------------------------------------

resource "kubernetes_namespace" "github_runners" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name = var.github_runners_namespace
    labels = {
      "app.kubernetes.io/name"       = "github-runners"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }
}

# -----------------------------------------------------------------------------
# GitHub PAT Secret
# -----------------------------------------------------------------------------

resource "kubernetes_secret" "github_pat" {
  count = var.github_runners_enabled && var.github_pat != "" ? 1 : 0

  metadata {
    name      = "github-pat"
    namespace = kubernetes_namespace.github_runners[0].metadata[0].name
  }

  data = {
    github_token = var.github_pat
  }

  type = "Opaque"
}

# -----------------------------------------------------------------------------
# Actions Runner Controller (ARC) - Controller
# -----------------------------------------------------------------------------

resource "helm_release" "arc_controller" {
  count = var.github_runners_enabled ? 1 : 0

  name             = "arc"
  repository       = "oci://ghcr.io/actions/actions-runner-controller-charts"
  chart            = "gha-runner-scale-set-controller"
  version          = "0.9.3"
  namespace        = kubernetes_namespace.github_runners[0].metadata[0].name
  create_namespace = false

  values = [
    yamlencode({
      replicaCount = 1
      
      # Resource limits for controller
      resources = {
        limits = {
          cpu    = "500m"
          memory = "512Mi"
        }
        requests = {
          cpu    = "100m"
          memory = "128Mi"
        }
      }
    })
  ]

  depends_on = [kubernetes_namespace.github_runners]
}

# -----------------------------------------------------------------------------
# Actions Runner Scale Set - Runners with DinD and Azure CLI
# -----------------------------------------------------------------------------

resource "helm_release" "arc_runner_set" {
  count = var.github_runners_enabled && var.github_pat != "" ? 1 : 0

  name             = "ml-platform-runners"
  repository       = "oci://ghcr.io/actions/actions-runner-controller-charts"
  chart            = "gha-runner-scale-set"
  version          = "0.9.3"
  namespace        = kubernetes_namespace.github_runners[0].metadata[0].name
  create_namespace = false

  values = [
    yamlencode({
      # GitHub configuration
      githubConfigUrl = var.github_org != "" ? "https://github.com/${var.github_org}/${var.github_repo}" : "https://github.com/${var.github_repo}"
      
      githubConfigSecret = {
        github_token = var.github_pat
      }

      # Runner configuration
      runnerScaleSetName = "ml-platform-runners"
      
      # Scaling configuration
      minRunners = var.runner_min_count
      maxRunners = var.runner_max_count

      # Use ARC's built-in DinD mode
      containerMode = {
        type = "dind"
      }

      # Runner pod template - minimal config, ARC handles DinD
      template = {
        spec = {
          serviceAccountName = "github-runner"
          
          containers = [{
            name  = "runner"
            image = "ghcr.io/actions/actions-runner:latest"
            command = ["/home/runner/run.sh"]
            
            env = [
              # ACR name for convenience in workflows
              {
                name  = "ACR_NAME"
                value = data.terraform_remote_state.aks.outputs.acr_name
              }
            ]
            
            resources = {
              limits = {
                cpu    = "4"
                memory = "8Gi"
              }
              requests = {
                cpu    = "1"
                memory = "2Gi"
              }
            }
          }]
        }
      }
    })
  ]

  depends_on = [
    helm_release.arc_controller,
    kubernetes_secret.github_pat
  ]
}

# -----------------------------------------------------------------------------
# Service Account for Runners
# -----------------------------------------------------------------------------

resource "kubernetes_service_account" "github_runner" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name      = "github-runner"
    namespace = kubernetes_namespace.github_runners[0].metadata[0].name
    # Workload identity annotation - optional, works without it using node identity
    # annotations = {
    #   "azure.workload.identity/client-id" = data.terraform_remote_state.aks.outputs.kubelet_identity_client_id
    # }
  }
}

# -----------------------------------------------------------------------------
# RBAC - Allow runners to deploy to cluster
# -----------------------------------------------------------------------------

resource "kubernetes_cluster_role" "github_runner" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name = "github-runner"
  }

  # Allow managing deployments, services, etc.
  rule {
    api_groups = [""]
    resources  = ["pods", "services", "configmaps", "secrets", "serviceaccounts"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  # Allow accessing pod logs and exec/attach (needed for smoke tests)
  rule {
    api_groups = [""]
    resources  = ["pods/log", "pods/attach", "pods/exec"]
    verbs      = ["get", "create"]
  }

  rule {
    api_groups = ["apps"]
    resources  = ["deployments", "replicasets", "statefulsets"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  rule {
    api_groups = ["batch"]
    resources  = ["jobs"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  # Allow managing KServe inference services
  rule {
    api_groups = ["serving.kserve.io"]
    resources  = ["inferenceservices"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  # Allow managing Istio resources
  rule {
    api_groups = ["networking.istio.io"]
    resources  = ["virtualservices", "destinationrules", "gateways"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  # Allow managing HPA
  rule {
    api_groups = ["autoscaling"]
    resources  = ["horizontalpodautoscalers"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  # Allow managing PDB
  rule {
    api_groups = ["policy"]
    resources  = ["poddisruptionbudgets"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  # Allow managing RBAC
  rule {
    api_groups = ["rbac.authorization.k8s.io"]
    resources  = ["clusterroles", "clusterrolebindings", "roles", "rolebindings"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }

  # Allow managing namespaces (read-only)
  rule {
    api_groups = [""]
    resources  = ["namespaces"]
    verbs      = ["get", "list", "watch"]
  }
}

resource "kubernetes_cluster_role_binding" "github_runner" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name = "github-runner"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.github_runner[0].metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.github_runner[0].metadata[0].name
    namespace = kubernetes_namespace.github_runners[0].metadata[0].name
  }

  # Grant access to all service accounts in github-runners namespace
  # This covers ARC controller and dynamically created runner pods
  subject {
    kind      = "Group"
    name      = "system:serviceaccounts:${kubernetes_namespace.github_runners[0].metadata[0].name}"
    api_group = "rbac.authorization.k8s.io"
  }
}

# -----------------------------------------------------------------------------
# Role for serving namespace (additional permissions)
# -----------------------------------------------------------------------------

resource "kubernetes_role" "github_runner_serving" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name      = "github-runner"
    namespace = "serving"
  }

  rule {
    api_groups = ["*"]
    resources  = ["*"]
    verbs      = ["*"]
  }
}

resource "kubernetes_role_binding" "github_runner_serving" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name      = "github-runner"
    namespace = "serving"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role.github_runner_serving[0].metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.github_runner[0].metadata[0].name
    namespace = kubernetes_namespace.github_runners[0].metadata[0].name
  }

  # Grant access to all service accounts in github-runners namespace
  subject {
    kind      = "Group"
    name      = "system:serviceaccounts:${kubernetes_namespace.github_runners[0].metadata[0].name}"
    api_group = "rbac.authorization.k8s.io"
  }
}

# -----------------------------------------------------------------------------
# Role for kserve namespace
# -----------------------------------------------------------------------------

resource "kubernetes_role" "github_runner_kserve" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name      = "github-runner"
    namespace = "kserve"
  }

  rule {
    api_groups = ["*"]
    resources  = ["*"]
    verbs      = ["*"]
  }
}

resource "kubernetes_role_binding" "github_runner_kserve" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name      = "github-runner"
    namespace = "kserve"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role.github_runner_kserve[0].metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.github_runner[0].metadata[0].name
    namespace = kubernetes_namespace.github_runners[0].metadata[0].name
  }

  # Grant access to all service accounts in github-runners namespace
  # This covers ARC controller and dynamically created runner pods
  subject {
    kind      = "Group"
    name      = "system:serviceaccounts:${kubernetes_namespace.github_runners[0].metadata[0].name}"
    api_group = "rbac.authorization.k8s.io"
  }
}

# -----------------------------------------------------------------------------
# Platform Configuration ConfigMap
# Available to runners for accessing platform resources
# -----------------------------------------------------------------------------

resource "kubernetes_config_map" "platform_config" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name      = "platform-config"
    namespace = kubernetes_namespace.github_runners[0].metadata[0].name
    labels = {
      "app.kubernetes.io/name"       = "platform-config"
      "app.kubernetes.io/component"  = "configuration"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  data = {
    ACR_NAME                   = data.terraform_remote_state.aks.outputs.acr_login_server
    STORAGE_ACCOUNT_NAME       = data.terraform_remote_state.aks.outputs.storage_account_name
    MLFLOW_TRACKING_URI        = "http://mlflow-service.mlflow.svc.cluster.local:5000"
    KSERVE_NAMESPACE           = "kserve"
    MLFLOW_NAMESPACE           = "mlflow"
    SERVING_NAMESPACE          = "serving"
  }
}

# Also create in kserve namespace for convenience
resource "kubernetes_config_map" "platform_config_kserve" {
  count = var.github_runners_enabled ? 1 : 0

  metadata {
    name      = "platform-config"
    namespace = "kserve"
    labels = {
      "app.kubernetes.io/name"       = "platform-config"
      "app.kubernetes.io/component"  = "configuration"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  data = {
    ACR_NAME                   = data.terraform_remote_state.aks.outputs.acr_login_server
    STORAGE_ACCOUNT_NAME       = data.terraform_remote_state.aks.outputs.storage_account_name
    MLFLOW_TRACKING_URI        = "http://mlflow-service.mlflow.svc.cluster.local:5000"
  }
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "github_runners_namespace" {
  description = "Namespace for GitHub runners"
  value       = var.github_runners_enabled ? kubernetes_namespace.github_runners[0].metadata[0].name : null
}

output "github_runner_scale_set_name" {
  description = "Name of the runner scale set (use this in workflow 'runs-on')"
  value       = var.github_runners_enabled ? "ml-platform-runners" : null
}
