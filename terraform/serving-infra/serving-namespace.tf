# =============================================================================
# Serving Namespace and Secrets
# =============================================================================
# Creates the serving namespace for model deployments and configures
# necessary secrets for Azure Storage and Redis access.

# Serving namespace for InferenceServices
resource "kubernetes_namespace" "serving" {
  metadata {
    name = var.serving_namespace
    labels = merge(local.common_labels, {
      component           = "serving"
      istio-injection     = "enabled"
    })
  }

  depends_on = [time_sleep.wait_for_kserve]
}

# Azure Storage Secret for KServe namespace
resource "kubernetes_secret" "kserve_storage" {
  count = var.kserve_enabled ? 1 : 0

  metadata {
    name      = "azure-storage-secret"
    namespace = kubernetes_namespace.kserve[0].metadata[0].name
  }

  data = {
    AZURE_STORAGE_ACCOUNT_NAME      = local.storage_account
    AZURE_STORAGE_ACCOUNT_KEY       = local.storage_key
    AZURE_STORAGE_ACCESS_KEY        = local.storage_key
    AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=${local.storage_account};AccountKey=${local.storage_key};EndpointSuffix=core.windows.net"
  }

  depends_on = [kubernetes_namespace.kserve]
}

# Azure Storage Secret for Serving namespace
resource "kubernetes_secret" "serving_storage" {
  metadata {
    name      = "azure-storage-secret"
    namespace = kubernetes_namespace.serving.metadata[0].name
  }

  data = {
    AZURE_STORAGE_ACCOUNT_NAME      = local.storage_account
    AZURE_STORAGE_ACCOUNT_KEY       = local.storage_key
    AZURE_STORAGE_ACCESS_KEY        = local.storage_key
    AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=${local.storage_account};AccountKey=${local.storage_key};EndpointSuffix=core.windows.net"
  }
}

# Azure Redis Secret for KServe namespace
resource "kubernetes_secret" "kserve_redis" {
  count = var.kserve_enabled ? 1 : 0

  metadata {
    name      = "azure-redis-secret"
    namespace = kubernetes_namespace.kserve[0].metadata[0].name
  }

  data = {
    REDIS_HOST           = local.redis_hostname
    REDIS_KEY            = local.redis_key
    REDIS_CONNECTION_STRING = "rediss://:${local.redis_key}@${local.redis_hostname}:6380/0"
  }

  depends_on = [kubernetes_namespace.kserve]
}

# Azure Redis Secret for Serving namespace
resource "kubernetes_secret" "serving_redis" {
  metadata {
    name      = "azure-redis-secret"
    namespace = kubernetes_namespace.serving.metadata[0].name
  }

  data = {
    REDIS_HOST           = local.redis_hostname
    REDIS_KEY            = local.redis_key
    REDIS_CONNECTION_STRING = "rediss://:${local.redis_key}@${local.redis_hostname}:6380/0"
  }
}

# KServe Storage Config Secret (for model downloading)
resource "kubernetes_secret" "kserve_storage_config" {
  count = var.kserve_enabled ? 1 : 0

  metadata {
    name      = "storage-config"
    namespace = kubernetes_namespace.kserve[0].metadata[0].name
    annotations = {
      "serving.kserve.io/s3-endpoint"    = ""
      "serving.kserve.io/s3-usehttps"    = "1"
      "serving.kserve.io/s3-verifyssl"   = "1"
      "serving.kserve.io/s3-region"      = ""
    }
  }

  data = {
    # Azure Blob Storage credentials in format KServe expects
    "azure" = jsonencode({
      type                    = "azure"
      storage_account         = local.storage_account
      storage_access_key      = local.storage_key
      container               = "models"
    })
  }

  depends_on = [kubernetes_namespace.kserve]
}

# Storage Config for Serving namespace
resource "kubernetes_secret" "serving_storage_config" {
  metadata {
    name      = "storage-config"
    namespace = kubernetes_namespace.serving.metadata[0].name
    annotations = {
      "serving.kserve.io/s3-endpoint"    = ""
      "serving.kserve.io/s3-usehttps"    = "1"
      "serving.kserve.io/s3-verifyssl"   = "1"
      "serving.kserve.io/s3-region"      = ""
    }
  }

  data = {
    "azure" = jsonencode({
      type                    = "azure"
      storage_account         = local.storage_account
      storage_access_key      = local.storage_key
      container               = "models"
    })
  }
}

# Service Account for model serving
resource "kubernetes_service_account" "serving" {
  metadata {
    name      = "model-serving-sa"
    namespace = kubernetes_namespace.serving.metadata[0].name
    labels    = local.common_labels
  }
}
