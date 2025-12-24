# =============================================================================
# Knative Serving Installation
# =============================================================================
# Knative Serving provides serverless capabilities for KServe.
# Installed via kubectl since Knative doesn't have official Helm charts.

resource "kubernetes_namespace" "knative_serving" {
  count = var.knative_enabled ? 1 : 0

  metadata {
    name = var.knative_namespace
    labels = merge(local.common_labels, {
      component                       = "knative"
      "app.kubernetes.io/name"        = "knative-serving"
      "app.kubernetes.io/version"     = var.knative_version
    })
  }

  depends_on = [time_sleep.wait_for_istio]
}

# Install Knative Serving CRDs
resource "null_resource" "knative_serving_crds" {
  count = var.knative_enabled ? 1 : 0

  triggers = {
    version        = var.knative_version
    cluster_name   = local.aks_cluster_name
    resource_group = local.resource_group
  }

  provisioner "local-exec" {
    command = <<-EOT
      az aks get-credentials --resource-group ${self.triggers.resource_group} --name ${self.triggers.cluster_name} --overwrite-existing && \
      kubectl apply -f "https://github.com/knative/serving/releases/download/knative-v${self.triggers.version}/serving-crds.yaml" && \
      kubectl wait --for condition=established --timeout=60s crd/services.serving.knative.dev
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      kubectl delete --ignore-not-found=true -f "https://github.com/knative/serving/releases/download/knative-v${self.triggers.version}/serving-crds.yaml" || true
    EOT
  }

  depends_on = [kubernetes_namespace.knative_serving]
}

# Install Knative Serving Core
resource "null_resource" "knative_serving_core" {
  count = var.knative_enabled ? 1 : 0

  triggers = {
    version        = var.knative_version
    namespace      = var.knative_namespace
    cluster_name   = local.aks_cluster_name
    resource_group = local.resource_group
  }

  provisioner "local-exec" {
    command = <<-EOT
      az aks get-credentials --resource-group ${self.triggers.resource_group} --name ${self.triggers.cluster_name} --overwrite-existing && \
      kubectl apply -f "https://github.com/knative/serving/releases/download/knative-v${self.triggers.version}/serving-core.yaml" && \
      kubectl wait --for=condition=available --timeout=300s deployment/activator -n ${self.triggers.namespace} && \
      kubectl wait --for=condition=available --timeout=300s deployment/controller -n ${self.triggers.namespace}
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      kubectl delete --ignore-not-found=true -f "https://github.com/knative/serving/releases/download/knative-v${self.triggers.version}/serving-core.yaml" || true
    EOT
  }

  depends_on = [null_resource.knative_serving_crds]
}

# Install Knative Istio Controller (networking layer)
resource "null_resource" "knative_net_istio" {
  count = var.knative_enabled && var.istio_enabled ? 1 : 0

  triggers = {
    version        = var.knative_version
    namespace      = var.knative_namespace
    cluster_name   = local.aks_cluster_name
    resource_group = local.resource_group
  }

  provisioner "local-exec" {
    command = <<-EOT
      az aks get-credentials --resource-group ${self.triggers.resource_group} --name ${self.triggers.cluster_name} --overwrite-existing && \
      kubectl apply -f "https://github.com/knative/net-istio/releases/download/knative-v${self.triggers.version}/net-istio.yaml" && \
      kubectl wait --for=condition=available --timeout=300s deployment/net-istio-controller -n ${self.triggers.namespace} || true
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      kubectl delete --ignore-not-found=true -f "https://github.com/knative/net-istio/releases/download/knative-v${self.triggers.version}/net-istio.yaml" || true
    EOT
  }

  depends_on = [
    null_resource.knative_serving_core,
    helm_release.istio_ingress
  ]
}

# Configure Knative for Azure (patch config-network)
resource "kubectl_manifest" "knative_config_network" {
  count = var.knative_enabled ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: config-network
      namespace: ${var.knative_namespace}
      labels:
        app.kubernetes.io/name: knative-serving
        app.kubernetes.io/component: controller
    data:
      ingress-class: "istio.ingress.networking.knative.dev"
      autocreate-cluster-domain-claims: "true"
      enable-auto-tls: "false"
  YAML

  depends_on = [null_resource.knative_net_istio]
}

# Configure Knative domain
resource "kubectl_manifest" "knative_config_domain" {
  count = var.knative_enabled ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: config-domain
      namespace: ${var.knative_namespace}
      labels:
        app.kubernetes.io/name: knative-serving
        app.kubernetes.io/component: controller
    data:
      # Use sslip.io for development (auto-resolving wildcard DNS)
      sslip.io: ""
  YAML

  depends_on = [null_resource.knative_net_istio]
}

# Configure Knative features - enable init containers for KServe storage-initializer
resource "kubectl_manifest" "knative_config_features" {
  count = var.knative_enabled ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: config-features
      namespace: ${var.knative_namespace}
      labels:
        app.kubernetes.io/name: knative-serving
        app.kubernetes.io/component: controller
    data:
      # Enable init containers support - REQUIRED for KServe storage-initializer
      # Without this, KServe cannot inject init containers to download models from storage
      kubernetes.podspec-init-containers: "enabled"
      # Enable emptyDir volumes for model storage
      kubernetes.podspec-volumes-emptydir: "enabled"
      # Enable multi-container pods
      multi-container: "enabled"
  YAML

  depends_on = [null_resource.knative_net_istio]
}

# Wait for Knative to be fully ready
resource "time_sleep" "wait_for_knative" {
  count = var.knative_enabled ? 1 : 0

  depends_on = [
    kubectl_manifest.knative_config_network,
    kubectl_manifest.knative_config_domain,
    kubectl_manifest.knative_config_features
  ]

  create_duration = "30s"
}
