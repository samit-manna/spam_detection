# Namespace for Feast
resource "kubernetes_namespace" "feast" {
  count = var.feast_enabled ? 1 : 0

  metadata {
    name = var.feast_namespace
    labels = merge(
      local.common_labels,
      {
        component = "feast"
      }
    )
  }
}

# Secret for Redis connection (commented out - Redis not deployed in base-infra)
# Uncomment when Redis is added to base-infra
resource "kubernetes_secret" "feast_redis" {
  count = var.feast_enabled ? 1 : 0

  metadata {
    name      = "feast-redis-secret"
    namespace = kubernetes_namespace.feast[0].metadata[0].name
  }

  data = {
    redis-host     = data.terraform_remote_state.aks.outputs.redis_hostname
    redis-password = data.terraform_remote_state.aks.outputs.redis_primary_access_key
  }
}

# Secret for Azure Storage (offline store)
resource "kubernetes_secret" "feast_storage" {
  count = var.feast_enabled ? 1 : 0

  metadata {
    name      = "feast-storage-secret"
    namespace = kubernetes_namespace.feast[0].metadata[0].name
  }

  data = {
    azure-storage-connection-string = data.terraform_remote_state.aks.outputs.storage_account_primary_connection_string
    azure-storage-account-name      = data.terraform_remote_state.aks.outputs.storage_account_name
    azure-storage-account-key       = data.terraform_remote_state.aks.outputs.storage_account_primary_access_key
  }
}

# ConfigMap for Feast feature store configuration
resource "kubernetes_config_map" "feast_config" {
  count = var.feast_enabled ? 1 : 0

  metadata {
    name      = "feast-config"
    namespace = kubernetes_namespace.feast[0].metadata[0].name
  }

  data = {
    "feature_store.yaml" = <<-YAML
      project: spam_detection
      provider: local
      registry: /data/registry.db
      offline_store:
        type: file
      entity_key_serialization_version: 3
    YAML
  }
}

# PersistentVolumeClaim for Feast registry
resource "kubernetes_persistent_volume_claim" "feast_registry" {
  count = var.feast_enabled ? 1 : 0

  metadata {
    name      = "feast-registry-pvc"
    namespace = kubernetes_namespace.feast[0].metadata[0].name
  }

  spec {
    access_modes = ["ReadWriteMany"]
    resources {
      requests = {
        storage = "1Gi"
      }
    }
    storage_class_name = "azurefile"
  }
}

# Feast Deployment (serverless mode - 0 replicas, PVC still available for feast apply jobs)
resource "kubernetes_deployment" "feast" {
  count = var.feast_enabled ? 1 : 0

  metadata {
    name      = "feast-server"
    namespace = kubernetes_namespace.feast[0].metadata[0].name
    labels = merge(
      local.common_labels,
      {
        app       = "feast"
        component = "feature-server"
      }
    )
  }

  spec {
    replicas = 0  # Serverless mode - no server running, use feast apply jobs instead

    selector {
      match_labels = {
        app = "feast"
      }
    }

    template {
      metadata {
        labels = {
          app       = "feast"
          component = "feature-server"
        }
      }

      spec {
        security_context {
          fs_group = 1001
        }

        init_container {
          name  = "feast-init"
          image = "feastdev/feature-server:latest"
          
          command = ["sh", "-c", <<-EOT
            set -e
            echo "Installing Azure dependencies..."
            pip install --quiet adlfs azure-identity fsspec

            echo "Setting up Feast..."
            mkdir -p /data
            cp /config/feature_store.yaml /data/feature_store.yaml
            
            # Create empty registry if it doesn't exist on the PVC
            if [ ! -f /data/registry.db ]; then
               touch /data/registry.db
            fi

            cd /data
            echo "Feast initialized."
          EOT
          ]

          security_context {
            run_as_user  = 1001
            run_as_group = 1001
          }

          env {
            name = "REDIS_HOST"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_redis[0].metadata[0].name
                key  = "redis-host"
              }
            }
          }

          env {
            name = "REDIS_PASSWORD"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_redis[0].metadata[0].name
                key  = "redis-password"
              }
            }
          }

          env {
            name = "AZURE_STORAGE_CONNECTION_STRING"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_storage[0].metadata[0].name
                key  = "azure-storage-connection-string"
              }
            }
          }

          env {
            name = "AZURE_STORAGE_ACCOUNT_NAME"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_storage[0].metadata[0].name
                key  = "azure-storage-account-name"
              }
            }
          }

          env {
            name = "AZURE_STORAGE_ACCOUNT_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_storage[0].metadata[0].name
                key  = "azure-storage-account-key"
              }
            }
          }

          volume_mount {
            name       = "feast-config"
            mount_path = "/config"
          }

          volume_mount {
            name       = "feast-data"
            mount_path = "/data"
          }
        }

        container {
          name  = "feast"
          image = "feastdev/feature-server:latest"
          
          command = ["sh", "-c", "pip install --quiet adlfs azure-identity fsspec && cd /data && feast serve --host 0.0.0.0"]

          security_context {
            run_as_user  = 1001
            run_as_group = 1001
          }

          port {
            container_port = 6566
            name          = "grpc"
          }

          env {
            name = "FEAST_REPO_PATH"
            value = "/data"
          }

          env {
            name = "REDIS_HOST"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_redis[0].metadata[0].name
                key  = "redis-host"
              }
            }
          }

          env {
            name = "REDIS_PASSWORD"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_redis[0].metadata[0].name
                key  = "redis-password"
              }
            }
          }

          env {
            name = "AZURE_STORAGE_CONNECTION_STRING"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_storage[0].metadata[0].name
                key  = "azure-storage-connection-string"
              }
            }
          }

          env {
            name = "AZURE_STORAGE_ACCOUNT_NAME"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_storage[0].metadata[0].name
                key  = "azure-storage-account-name"
              }
            }
          }

          env {
            name = "AZURE_STORAGE_ACCOUNT_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.feast_storage[0].metadata[0].name
                key  = "azure-storage-account-key"
              }
            }
          }

          volume_mount {
            name       = "feast-data"
            mount_path = "/data"
          }

          resources {
            requests = {
              cpu    = "500m"
              memory = "1Gi"
            }
            limits = {
              cpu    = "2"
              memory = "4Gi"
            }
          }

          liveness_probe {
            tcp_socket {
              port = 6566
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }

          readiness_probe {
            tcp_socket {
              port = 6566
            }
            initial_delay_seconds = 10
            period_seconds        = 5
          }
        }

        volume {
          name = "feast-config"
          config_map {
            name = kubernetes_config_map.feast_config[0].metadata[0].name
          }
        }

        volume {
          name = "feast-data"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.feast_registry[0].metadata[0].name
          }
        }
      }
    }
  }
}

# Feast Service
resource "kubernetes_service" "feast" {
  count = var.feast_enabled ? 1 : 0

  metadata {
    name      = "feast-service"
    namespace = kubernetes_namespace.feast[0].metadata[0].name
    labels = {
      app = "feast"
    }
  }

  spec {
    type = "ClusterIP"

    selector = {
      app = "feast"
    }

    port {
      port        = 6566
      target_port = 6566
      protocol    = "TCP"
      name        = "grpc"
    }
  }
}

# ConfigMap with Feast usage examples
resource "kubernetes_config_map" "feast_examples" {
  count = var.feast_enabled ? 1 : 0

  metadata {
    name      = "feast-examples"
    namespace = kubernetes_namespace.feast[0].metadata[0].name
  }

  data = {
    "define_features.py" = <<-PYTHON
      import os
      from feast import Entity, FeatureView, Field
      from feast.types import Float32, Int64
      from feast.infra.offline_stores.file_source import FileSource
      from datetime import timedelta
      
      # Get Account Name from Env to construct path dynamically
      account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "mltrainingsdevsaj69hpg")
      
      # Define the path to your parquet file in Azure Blob
      # Format: abfss://<container>@<account>.dfs.core.windows.net/<path>
      # The registry is stored locally on PVC, but data files can be on Blob
      f_source = FileSource(
          path=f"abfss://feast@{account_name}.dfs.core.windows.net/user_features.parquet",
          timestamp_field="event_timestamp",
      )
      
      user = Entity(name="user_id", description="User identifier")
      
      user_features = FeatureView(
          name="user_features",
          entities=[user],
          schema=[
              Field(name="age", dtype=Int64),
              Field(name="income", dtype=Float32),
              Field(name="credit_score", dtype=Float32),
          ],
          source=f_source,
          ttl=timedelta(days=7),
      )
      
      # To apply this definition:
      # 1. Save this file in the Feast pod
      # 2. Run: feast apply
      # 3. The registry.db will be updated on the PVC
      # 4. Data will be read from Azure Blob Storage when needed
    PYTHON
    
    "materialize_features.py" = <<-PYTHON
      from feast import FeatureStore
      from datetime import datetime, timedelta
      
      # Initialize the feature store
      store = FeatureStore(repo_path=".")
      
      # Materialize features to online store
      store.materialize(
          start_date=datetime.now() - timedelta(days=7),
          end_date=datetime.now()
      )
      
      print("Features materialized successfully!")
    PYTHON
    
    "get_online_features.py" = <<-PYTHON
      from feast import FeatureStore
      
      # Initialize the feature store
      store = FeatureStore(repo_path=".")
      
      # Get online features for inference
      feature_vector = store.get_online_features(
          features=[
              "user_features:age",
              "user_features:income",
              "user_features:credit_score",
          ],
          entity_rows=[
              {"user_id": 1001},
              {"user_id": 1002},
          ],
      ).to_dict()
      
      print("Online features:", feature_vector)
    PYTHON
  }
}
