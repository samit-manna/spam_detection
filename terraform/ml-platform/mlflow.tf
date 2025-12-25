# Namespace for MLFlow
resource "kubernetes_namespace" "mlflow" {
  count = var.mlflow_enabled ? 1 : 0

  metadata {
    name = var.mlflow_namespace
    labels = merge(
      local.common_labels,
      {
        component = "mlflow"
      }
    )
  }
}

# Secret for MLFlow database connection
resource "kubernetes_secret" "mlflow_db" {
  count = var.mlflow_enabled ? 1 : 0

  metadata {
    name      = "mlflow-db-secret"
    namespace = kubernetes_namespace.mlflow[0].metadata[0].name
  }

  data = {
    postgres-host     = data.terraform_remote_state.aks.outputs.postgres_server_fqdn
    postgres-password = data.terraform_remote_state.aks.outputs.postgres_admin_password
    postgres-db       = data.terraform_remote_state.aks.outputs.postgres_database_name
    postgres-user     = data.terraform_remote_state.aks.outputs.postgres_admin_username
    backend-store-uri = "postgresql+psycopg2://${data.terraform_remote_state.aks.outputs.postgres_admin_username}:${urlencode(data.terraform_remote_state.aks.outputs.postgres_admin_password)}@${data.terraform_remote_state.aks.outputs.postgres_server_fqdn}:5432/${data.terraform_remote_state.aks.outputs.postgres_database_name}"
  }
}

# Secret for MLFlow artifact storage
resource "kubernetes_secret" "mlflow_storage" {
  count = var.mlflow_enabled ? 1 : 0

  metadata {
    name      = "mlflow-storage-secret"
    namespace = kubernetes_namespace.mlflow[0].metadata[0].name
  }

  data = {
    azure-storage-connection-string = data.terraform_remote_state.aks.outputs.storage_account_primary_connection_string
    azure-storage-account-name      = data.terraform_remote_state.aks.outputs.storage_account_name
    azure-storage-access-key        = data.terraform_remote_state.aks.outputs.storage_account_primary_access_key
  }
}

# MLFlow Deployment
resource "kubernetes_deployment" "mlflow" {
  count = var.mlflow_enabled ? 1 : 0

  metadata {
    name      = "mlflow-server"
    namespace = kubernetes_namespace.mlflow[0].metadata[0].name
    labels = merge(
      local.common_labels,
      {
        app       = "mlflow"
        component = "tracking-server"
      }
    )
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "mlflow"
      }
    }

    template {
      metadata {
        labels = {
          app       = "mlflow"
          component = "tracking-server"
        }
      }

      spec {
        volume {
          name = "mlflow-data"
          empty_dir {}
        }

        container {
          name  = "mlflow"
          image = "ghcr.io/mlflow/mlflow:v2.9.2"
          
          command = [
            "sh",
            "-c",
            "pip install --quiet psycopg2-binary azure-storage-blob && echo 'Starting MLflow server with PostgreSQL backend...' && mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --default-artifact-root \"$MLFLOW_DEFAULT_ARTIFACT_ROOT\" 2>&1"
          ]

          port {
            container_port = 5000
            name          = "http"
          }

          env {
            name = "MLFLOW_BACKEND_STORE_URI"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.mlflow_db[0].metadata[0].name
                key  = "backend-store-uri"
              }
            }
          }

          env {
            name = "MLFLOW_DEFAULT_ARTIFACT_ROOT"
            value = "wasbs://mlflow@${data.terraform_remote_state.aks.outputs.storage_account_name}.blob.core.windows.net/artifacts"
          }

          env {
            name = "AZURE_STORAGE_CONNECTION_STRING"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.mlflow_storage[0].metadata[0].name
                key  = "azure-storage-connection-string"
              }
            }
          }

          env {
            name = "AZURE_STORAGE_ACCOUNT_NAME"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.mlflow_storage[0].metadata[0].name
                key  = "azure-storage-account-name"
              }
            }
          }

          env {
            name = "AZURE_STORAGE_ACCESS_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.mlflow_storage[0].metadata[0].name
                key  = "azure-storage-access-key"
              }
            }
          }

          volume_mount {
            name       = "mlflow-data"
            mount_path = "/mlflow"
          }

          resources {
            requests = {
              cpu    = "1"
              memory = "2Gi"
            }
            limits = {
              cpu    = "2"
              memory = "4Gi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 5000
            }
            initial_delay_seconds = 120
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 5
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 5000
            }
            initial_delay_seconds = 90
            period_seconds        = 10
            timeout_seconds       = 10
            failure_threshold     = 5
          }
        }
      }
    }
  }
}

# MLFlow Service
resource "kubernetes_service" "mlflow" {
  count = var.mlflow_enabled ? 1 : 0

  metadata {
    name      = "mlflow-service"
    namespace = kubernetes_namespace.mlflow[0].metadata[0].name
    labels = {
      app = "mlflow"
    }
  }

  spec {
    type = "LoadBalancer"

    selector = {
      app = "mlflow"
    }

    port {
      port        = 5000
      target_port = 5000
      protocol    = "TCP"
      name        = "http"
    }
  }
}
