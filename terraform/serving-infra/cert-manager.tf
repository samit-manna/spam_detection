# =============================================================================
# cert-manager Installation
# =============================================================================
# cert-manager is required for KServe and provides certificate management
# for Kubernetes. It's installed via Helm for better lifecycle management.

resource "kubernetes_namespace" "cert_manager" {
  count = var.cert_manager_enabled ? 1 : 0

  metadata {
    name = var.cert_manager_namespace
    labels = merge(local.common_labels, {
      component = "cert-manager"
    })
  }
}

resource "helm_release" "cert_manager" {
  count = var.cert_manager_enabled ? 1 : 0

  name             = "cert-manager"
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  version          = var.cert_manager_version
  namespace        = kubernetes_namespace.cert_manager[0].metadata[0].name
  create_namespace = false

  # Install CRDs
  set {
    name  = "installCRDs"
    value = "true"
  }

  # Resource limits
  set {
    name  = "resources.requests.cpu"
    value = "50m"
  }

  set {
    name  = "resources.requests.memory"
    value = "64Mi"
  }

  set {
    name  = "resources.limits.cpu"
    value = "200m"
  }

  set {
    name  = "resources.limits.memory"
    value = "256Mi"
  }

  # Webhook settings
  set {
    name  = "webhook.timeoutSeconds"
    value = "30"
  }

  timeout = 600

  depends_on = [kubernetes_namespace.cert_manager]
}

# Wait for cert-manager to be fully ready before proceeding
resource "time_sleep" "wait_for_cert_manager" {
  count = var.cert_manager_enabled ? 1 : 0

  depends_on = [helm_release.cert_manager]

  create_duration = "30s"
}
