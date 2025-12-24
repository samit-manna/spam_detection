# =============================================================================
# Istio Service Mesh Installation
# =============================================================================
# Istio provides traffic management, security, and observability for KServe.
# We use the Istio Helm charts for installation.

resource "kubernetes_namespace" "istio_system" {
  count = var.istio_enabled ? 1 : 0

  metadata {
    name = var.istio_namespace
    labels = merge(local.common_labels, {
      component = "istio"
    })
  }
}

# Install Istio Base (CRDs)
resource "helm_release" "istio_base" {
  count = var.istio_enabled ? 1 : 0

  name             = "istio-base"
  repository       = "https://istio-release.storage.googleapis.com/charts"
  chart            = "base"
  version          = var.istio_version
  namespace        = kubernetes_namespace.istio_system[0].metadata[0].name
  create_namespace = false

  timeout = 300

  depends_on = [
    kubernetes_namespace.istio_system,
    time_sleep.wait_for_cert_manager
  ]
}

# Install Istiod (control plane)
resource "helm_release" "istiod" {
  count = var.istio_enabled ? 1 : 0

  name             = "istiod"
  repository       = "https://istio-release.storage.googleapis.com/charts"
  chart            = "istiod"
  version          = var.istio_version
  namespace        = kubernetes_namespace.istio_system[0].metadata[0].name
  create_namespace = false

  # Pilot settings
  set {
    name  = "pilot.resources.requests.cpu"
    value = "100m"
  }

  set {
    name  = "pilot.resources.requests.memory"
    value = "256Mi"
  }

  set {
    name  = "pilot.resources.limits.cpu"
    value = "500m"
  }

  set {
    name  = "pilot.resources.limits.memory"
    value = "1Gi"
  }

  # Auto-injection settings
  set {
    name  = "global.proxy.autoInject"
    value = "disabled"
  }

  # Mesh config
  set {
    name  = "meshConfig.enablePrometheusMerge"
    value = "true"
  }

  timeout = 600

  depends_on = [helm_release.istio_base]
}

# Install Istio Ingress Gateway
resource "helm_release" "istio_ingress" {
  count = var.istio_enabled ? 1 : 0

  name             = "istio-ingressgateway"
  repository       = "https://istio-release.storage.googleapis.com/charts"
  chart            = "gateway"
  version          = var.istio_version
  namespace        = kubernetes_namespace.istio_system[0].metadata[0].name
  create_namespace = false

  # Service type
  set {
    name  = "service.type"
    value = "LoadBalancer"
  }

  # Resources
  set {
    name  = "resources.requests.cpu"
    value = "100m"
  }

  set {
    name  = "resources.requests.memory"
    value = "128Mi"
  }

  set {
    name  = "resources.limits.cpu"
    value = "500m"
  }

  set {
    name  = "resources.limits.memory"
    value = "512Mi"
  }

  timeout = 300

  depends_on = [helm_release.istiod]
}

# Wait for Istio to be fully ready
resource "time_sleep" "wait_for_istio" {
  count = var.istio_enabled ? 1 : 0

  depends_on = [helm_release.istio_ingress]

  create_duration = "30s"
}
