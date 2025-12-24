# =============================================================================
# KServe Installation
# =============================================================================
# KServe provides standardized serverless ML inference on Kubernetes.
# It requires cert-manager, Istio, and Knative Serving.

resource "kubernetes_namespace" "kserve" {
  count = var.kserve_enabled ? 1 : 0

  metadata {
    name = var.kserve_namespace
    labels = merge(local.common_labels, {
      component                      = "kserve"
      "control-plane"                = "kserve-controller-manager"
      "app.kubernetes.io/name"       = "kserve"
      "app.kubernetes.io/version"    = var.kserve_version
    })
  }

  depends_on = [time_sleep.wait_for_knative]
}

# Install KServe CRDs and Controller
resource "null_resource" "kserve_install" {
  count = var.kserve_enabled ? 1 : 0

  triggers = {
    version        = var.kserve_version
    namespace      = var.kserve_namespace
    cluster_name   = local.aks_cluster_name
    resource_group = local.resource_group
  }

  provisioner "local-exec" {
    command = <<-EOT
      az aks get-credentials --resource-group ${self.triggers.resource_group} --name ${self.triggers.cluster_name} --overwrite-existing && \
      
      # Install KServe
      kubectl apply -f "https://github.com/kserve/kserve/releases/download/v${self.triggers.version}/kserve.yaml" && \
      
      # Wait for CRDs
      kubectl wait --for condition=established --timeout=60s crd/inferenceservices.serving.kserve.io && \
      
      # Wait for controller
      kubectl wait --for=condition=available --timeout=300s deployment/kserve-controller-manager -n ${self.triggers.namespace}
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      kubectl delete --ignore-not-found=true -f "https://github.com/kserve/kserve/releases/download/v${self.triggers.version}/kserve.yaml" || true
    EOT
  }

  depends_on = [kubernetes_namespace.kserve]
}

# Install KServe Built-in ClusterServingRuntimes
resource "null_resource" "kserve_runtimes" {
  count = var.kserve_enabled ? 1 : 0

  triggers = {
    version        = var.kserve_version
    cluster_name   = local.aks_cluster_name
    resource_group = local.resource_group
  }

  provisioner "local-exec" {
    command = <<-EOT
      az aks get-credentials --resource-group ${self.triggers.resource_group} --name ${self.triggers.cluster_name} --overwrite-existing && \
      kubectl apply -f "https://github.com/kserve/kserve/releases/download/v${self.triggers.version}/kserve-cluster-resources.yaml"
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      kubectl delete --ignore-not-found=true -f "https://github.com/kserve/kserve/releases/download/v${self.triggers.version}/kserve-cluster-resources.yaml" || true
    EOT
  }

  depends_on = [null_resource.kserve_install]
}

# Configure KServe InferenceService defaults
resource "kubectl_manifest" "kserve_config" {
  count = var.kserve_enabled ? 1 : 0

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: inferenceservice-config
      namespace: ${var.kserve_namespace}
    data:
      # Deploy mode: Serverless (Knative) or RawDeployment
      deploy: |
        {
          "defaultDeploymentMode": "Serverless"
        }
      # Ingress configuration
      ingress: |
        {
          "ingressGateway": "knative-serving/knative-ingress-gateway",
          "ingressService": "istio-ingressgateway.istio-system.svc.cluster.local",
          "localGateway": "knative-serving/knative-local-gateway",
          "localGatewayService": "knative-local-gateway.istio-system.svc.cluster.local",
          "ingressDomain": "sslip.io",
          "ingressClassName": "istio",
          "domainTemplate": "{{ .Name }}-{{ .Namespace }}.{{ .IngressDomain }}",
          "urlScheme": "http"
        }
      # Logger configuration
      logger: |
        {
          "image": "kserve/agent:v${var.kserve_version}",
          "memoryRequest": "100Mi",
          "memoryLimit": "1Gi",
          "cpuRequest": "100m",
          "cpuLimit": "1"
        }
      # Batcher configuration
      batcher: |
        {
          "image": "kserve/agent:v${var.kserve_version}",
          "memoryRequest": "1Gi",
          "memoryLimit": "1Gi",
          "cpuRequest": "1",
          "cpuLimit": "1"
        }
      # Agent configuration
      agent: |
        {
          "image": "kserve/agent:v${var.kserve_version}",
          "memoryRequest": "100Mi",
          "memoryLimit": "1Gi",
          "cpuRequest": "100m",
          "cpuLimit": "1"
        }
      # Storage initializer
      storageInitializer: |
        {
          "image": "kserve/storage-initializer:v${var.kserve_version}",
          "memoryRequest": "100Mi",
          "memoryLimit": "1Gi",
          "cpuRequest": "100m",
          "cpuLimit": "1",
          "storageSpecSecretName": "storage-config"
        }
      # Credentials for model storage
      credentials: |
        {
          "storageSpecSecretName": "storage-config",
          "storageSecretNameAnnotation": "serving.kserve.io/storageSecretName",
          "s3": {
            "s3AccessKeyIDName": "AWS_ACCESS_KEY_ID",
            "s3SecretAccessKeyName": "AWS_SECRET_ACCESS_KEY"
          },
          "azure": {
            "azureSubscriptionIdName": "AZURE_SUBSCRIPTION_ID",
            "azureClientIdName": "AZURE_CLIENT_ID",
            "azureTenantIdName": "AZURE_TENANT_ID",
            "azureClientSecretName": "AZURE_CLIENT_SECRET",
            "azureStorageAccountName": "AZURE_STORAGE_ACCOUNT",
            "azureStorageAccessKeyName": "AZURE_STORAGE_ACCESS_KEY"
          }
        }
  YAML

  depends_on = [null_resource.kserve_install]
}

# Wait for KServe to be fully ready
resource "time_sleep" "wait_for_kserve" {
  count = var.kserve_enabled ? 1 : 0

  depends_on = [
    null_resource.kserve_runtimes,
    kubectl_manifest.kserve_config
  ]

  create_duration = "30s"
}
