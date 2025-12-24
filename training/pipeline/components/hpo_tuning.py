"""
Hyperparameter Optimization Component

Submits a RayJob to run HPO using Ray Tune for XGBoost model.
"""
import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Output, Metrics, InputPath, OutputPath


@dsl.component(
    base_image="placeholder",  # Will be set dynamically
    packages_to_install=["kubernetes==29.0.0"],
)
def hpo_tuning(
    acr_name: str,
    image_tag: str,
    train_features_path: str,
    storage_account: str,
    container_name: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    num_trials: int,
    max_concurrent_trials: int,
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("best_f1", float), ("best_params", str), ("best_model_path", str)]):
    """
    Run hyperparameter optimization using Ray Tune via RayJob.
    
    Args:
        acr_name: Azure Container Registry name
        image_tag: Docker image tag
        train_features_path: Blob path to training features parquet
        storage_account: Azure storage account name
        container_name: Azure blob container name
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: Name of MLflow experiment
        num_trials: Number of HPO trials to run
        max_concurrent_trials: Maximum concurrent trials
        best_model_path: Output path for best model
        metrics: Output metrics object
        
    Returns:
        best_f1: Best F1 score achieved
        best_params: JSON string of best hyperparameters
    """
    import os
    import json
    import time
    from kubernetes import client, config
    
    # Load in-cluster config
    config.load_incluster_config()
    
    print(f"Training features path: {train_features_path}")
    
    # RayJob specification for HPO
    job_name = f"hpo-tuning-{int(time.time())}"
    
    ray_job_spec = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": job_name,
            "namespace": "kubeflow",
            "finalizers": [],  # Disable finalizers to allow automatic cleanup
        },
        "spec": {
            "entrypoint": "python /app/scripts/hpo_train.py",
            "runtimeEnvYAML": f"""
env_vars:
  TRAINING_FEATURES_PATH: "{train_features_path}"
  STORAGE_ACCOUNT: "{storage_account}"
  CONTAINER_NAME: "{container_name}"
  MLFLOW_TRACKING_URI: "{mlflow_tracking_uri}"
  MLFLOW_EXPERIMENT_NAME: "{mlflow_experiment_name}"
  NUM_TRIALS: "{num_trials}"
  MAX_CONCURRENT_TRIALS: "{max_concurrent_trials}"
working_dir: "/app"
""",
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 600,
            "rayClusterSpec": {
                "rayVersion": "2.9.0",
                "headGroupSpec": {
                    "rayStartParams": {
                        "dashboard-host": "0.0.0.0",
                    },
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "ray-head",
                                    "image": f"{acr_name}/ml-ray-train:{image_tag}",
                                    "resources": {
                                        "requests": {"cpu": "1", "memory": "2Gi"},  # Reduced from 2 CPU / 4Gi
                                        "limits": {"cpu": "2", "memory": "4Gi"},  # Keep higher limit
                                    },
                                    "envFrom": [
                                        {"secretRef": {"name": "azure-storage-secret"}}
                                    ],
                                }
                            ],
                        }
                    },
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "workers",
                        "replicas": 2,  # Reduced from 4 to 2 workers
                        "minReplicas": 1,  # Reduced from 2 to 1
                        "maxReplicas": 4,  # Reduced from 6 to 4
                        "rayStartParams": {},
                        "template": {
                            "spec": {
                                "tolerations": [
                                    {
                                        "key": "kubernetes.azure.com/scalesetpriority",
                                        "operator": "Equal",
                                        "value": "spot",
                                        "effect": "NoSchedule",
                                    }
                                ],
                                "containers": [
                                    {
                                        "name": "ray-worker",
                                        "image": f"{acr_name}/ml-ray-train:{image_tag}",
                                        "resources": {
                                            "requests": {"cpu": "1", "memory": "2Gi"},  # Reduced from 4 CPU / 8Gi
                                            "limits": {"cpu": "2", "memory": "4Gi"},  # Reduced from 4 CPU / 8Gi
                                        },
                                        "envFrom": [
                                            {"secretRef": {"name": "azure-storage-secret"}}
                                        ],
                                    }
                                ],
                            }
                        },
                    }
                ],
            },
        },
    }
    
    # Create RayJob
    api = client.CustomObjectsApi()
    
    print(f"Creating RayJob: {job_name}")
    api.create_namespaced_custom_object(
        group="ray.io",
        version="v1",
        namespace="kubeflow",
        plural="rayjobs",
        body=ray_job_spec,
    )
    
    # Wait for job completion
    print("Waiting for HPO RayJob to complete...")
    max_wait_time = 7200  # 2 hours for HPO
    poll_interval = 60
    elapsed = 0
    
    while elapsed < max_wait_time:
        time.sleep(poll_interval)
        elapsed += poll_interval
        
        try:
            job = api.get_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace="kubeflow",
                plural="rayjobs",
                name=job_name,
            )
            
            status = job.get("status", {})
            job_status = status.get("jobStatus", "PENDING")
            job_deployment_status = status.get("jobDeploymentStatus", "Unknown")
            
            print(f"RayJob status: {job_status}, deployment: {job_deployment_status} (elapsed: {elapsed}s)")
            
            if job_status == "SUCCEEDED":
                print("HPO RayJob completed successfully!")
                break
            elif job_status in ["FAILED", "STOPPED"]:
                # Get more details
                message = status.get("message", "No message")
                raise RuntimeError(f"HPO RayJob failed: {job_status} - {message}")
                
        except client.ApiException as e:
            if e.status == 404:
                print(f"RayJob {job_name} not found, may have been cleaned up")
                break
            raise
    else:
        raise TimeoutError(f"HPO RayJob did not complete within {max_wait_time} seconds")
    
    # Cleanup: Delete the RayJob to avoid accumulation
    try:
        print(f"Cleaning up RayJob: {job_name}")
        api.delete_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace="kubeflow",
            plural="rayjobs",
            name=job_name,
        )
        print(f"Successfully deleted RayJob: {job_name}")
    except Exception as e:
        print(f"Warning: Failed to delete RayJob (may have been auto-deleted): {e}")
    
    # Read HPO results from blob storage
    from azure.storage.blob import BlobServiceClient
    
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    
    results_blob_path = "hpo/results.json"
    print(f"Reading HPO results from: {container_name}/{results_blob_path}")
    
    try:
        blob_client = container_client.get_blob_client(results_blob_path)
        results_data = blob_client.download_blob().readall()
        hpo_results = json.loads(results_data)
    except Exception as e:
        raise RuntimeError(f"HPO results not found. Job may have failed. Error: {e}")
    
    best_f1 = hpo_results.get("best_f1", 0.0)
    best_params = json.dumps(hpo_results.get("best_params", {}))
    
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best params: {best_params}")
    
    # Define output path
    model_path = f"{container_name}/hpo/best_model.pkl"
    
    # Log metrics
    metrics.log_metric("best_f1", best_f1)
    metrics.log_metric("best_precision", hpo_results.get("best_precision", 0.0))
    metrics.log_metric("best_recall", hpo_results.get("best_recall", 0.0))
    metrics.log_metric("best_auc_roc", hpo_results.get("best_auc_roc", 0.0))
    metrics.log_metric("trials_completed", hpo_results.get("trials_completed", 0))
    metrics.log_metric("total_time_seconds", hpo_results.get("total_time_seconds", 0))
    
    from collections import namedtuple
    outputs = namedtuple("Outputs", ["best_f1", "best_params", "best_model_path"])
    return outputs(float(best_f1), best_params, model_path)
