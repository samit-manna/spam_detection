"""
Spam Detection Training Pipeline

End-to-end ML pipeline for training spam detection model using:
- Kubeflow Pipelines for orchestration
- Feast for feature retrieval
- Ray Tune for hyperparameter optimization
- MLflow for experiment tracking and model registry
"""
from kfp import dsl
from kfp.dsl import pipeline
from kfp import kubernetes
import os

# Import components
from components.data_preparation import data_preparation
from components.feature_retrieval import feature_retrieval
from components.baseline_training import baseline_training
from components.hpo_tuning import hpo_tuning
from components.model_evaluation import model_evaluation


@pipeline(
    name="spam-detection-training",
    description="End-to-end spam detection model training pipeline with Feast feature retrieval, baseline training, HPO, and model registration",
)
def spam_detection_pipeline(
    # Infrastructure parameters
    acr_name: str,  # Azure Container Registry name (e.g., myacr.azurecr.io)
    image_tag: str = "latest",
    
    # Data parameters
    entity_data_path: str = "processed/emails/all_emails.parquet",
    storage_account: str = "mltrainingsdevsal6xriy",
    container_name: str = "datasets",
    feast_container_name: str = "feast",
    model_container_name: str = "models",
    test_split_ratio: float = 0.2,
    
    # Feast parameters
    feast_server_url: str = "feast-service.feast.svc.cluster.local:6566",
    include_sender_features: bool = True,
    
    # MLflow parameters
    mlflow_tracking_uri: str = "http://mlflow-service.mlflow.svc.cluster.local:5000",
    mlflow_experiment_name: str = "spam-detection",
    
    # Training parameters
    baseline_model_type: str = "logistic_regression",
    
    # HPO parameters
    num_hpo_trials: int = 20,
    max_concurrent_trials: int = 4,
    
    # Registration parameters
    model_name: str = "spam-detector",
    f1_threshold: float = 0.85,
):
    """
    Spam Detection Training Pipeline.
    
    This pipeline:
    1. Prepares entity dataframes from processed email data
    2. Retrieves features from Feast (524 features across 5 feature views)
    3. Trains a baseline model to validate the pipeline
    4. Runs HPO with Ray Tune to find optimal XGBoost hyperparameters
    5. Evaluates the best model on holdout test set
    6. Registers the model to MLflow if it meets the F1 threshold
    
    Args:
        acr_name: Azure Container Registry name (used for setting component images)
        image_tag: Docker image tag for pipeline components
        entity_data_path: Path to all_emails.parquet in blob storage
        storage_account: Azure storage account name
        container_name: Azure blob container name
        feast_container_name: Azure blob container for Feast features
        model_container_name: Azure blob container for models
        test_split_ratio: Fraction of data for test set
        feast_server_url: Feast server gRPC endpoint
        include_sender_features: Whether to include sender_domain_features
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: Name of MLflow experiment
        baseline_model_type: Type of baseline model (logistic_regression or random_forest)
        num_hpo_trials: Number of HPO trials
        max_concurrent_trials: Max concurrent HPO trials
        model_name: Name for model registry
        f1_threshold: Minimum F1 score to register model
    """
    
    # Step 1: Data Preparation
    # Creates train/test entity dataframes with email_id, sender_domain, label, event_timestamp
    prep_task = data_preparation(
        acr_name=acr_name,
        image_tag=image_tag,
        entity_data_path=entity_data_path,
        storage_account=storage_account,
        container_name=container_name,
        test_split_ratio=test_split_ratio,
    )
    prep_task.set_display_name("1. Data Preparation")
    prep_task.set_caching_options(False)
    
    # Add Azure Storage secret as environment variable
    kubernetes.use_secret_as_env(
        task=prep_task,
        secret_name='azure-storage-secret',
        secret_key_to_env={'AZURE_STORAGE_CONNECTION_STRING': 'AZURE_STORAGE_CONNECTION_STRING'}
    )
    
    # Step 2: Feature Retrieval from Feast
    # Retrieves 524 features: email_text(8) + email_structural(8) + email_temporal(4) + email_tfidf(500) + sender_domain(4)
    retrieval_task = feature_retrieval(
        acr_name=acr_name,
        image_tag=image_tag,
        train_entity_path=prep_task.outputs["train_entity_path"],
        test_entity_path=prep_task.outputs["test_entity_path"],
        feast_server_url=feast_server_url,
        storage_account=storage_account,
        container_name=feast_container_name,
        include_sender_features=include_sender_features,
    )
    retrieval_task.set_display_name("2. Feature Retrieval (Feast)")
    retrieval_task.after(prep_task)
    
    # Add Azure Storage secret as environment variable
    kubernetes.use_secret_as_env(
        task=retrieval_task,
        secret_name='azure-storage-secret',
        secret_key_to_env={'AZURE_STORAGE_CONNECTION_STRING': 'AZURE_STORAGE_CONNECTION_STRING'}
    )
    
    # Step 3: Baseline Training
    # Validates pipeline with simple model before expensive HPO
    baseline_task = baseline_training(
        acr_name=acr_name,
        image_tag=image_tag,
        train_features_path=retrieval_task.outputs["train_features_path"],
        storage_account=storage_account,
        container_name=model_container_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        model_type=baseline_model_type,
    )
    baseline_task.set_display_name(f"3. Baseline Training ({baseline_model_type})")
    baseline_task.after(retrieval_task)
    
    # Add Azure Storage secret as environment variable
    kubernetes.use_secret_as_env(
        task=baseline_task,
        secret_name='azure-storage-secret',
        secret_key_to_env={'AZURE_STORAGE_CONNECTION_STRING': 'AZURE_STORAGE_CONNECTION_STRING'}
    )
    
    # Step 4: Hyperparameter Optimization with Ray Tune
    # Finds optimal XGBoost hyperparameters
    hpo_task = hpo_tuning(
        acr_name=acr_name,
        image_tag=image_tag,
        train_features_path=retrieval_task.outputs["train_features_path"],
        storage_account=storage_account,
        container_name=model_container_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        num_trials=num_hpo_trials,
        max_concurrent_trials=max_concurrent_trials,
    )
    hpo_task.set_display_name("4. HPO (Ray Tune)")
    hpo_task.after(baseline_task)
    
    # Add Azure Storage secret as environment variable
    kubernetes.use_secret_as_env(
        task=hpo_task,
        secret_name='azure-storage-secret',
        secret_key_to_env={'AZURE_STORAGE_CONNECTION_STRING': 'AZURE_STORAGE_CONNECTION_STRING'}
    )
    
    # Step 5: Model Evaluation & Registration
    # Evaluates on test set and registers to MLflow if threshold met
    eval_task = model_evaluation(
        acr_name=acr_name,
        image_tag=image_tag,
        best_model_path=hpo_task.outputs["best_model_path"],
        test_features_path=retrieval_task.outputs["test_features_path"],
        storage_account=storage_account,
        container_name=model_container_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        model_name=model_name,
        f1_threshold=f1_threshold,
    )
    eval_task.set_display_name("5. Model Evaluation & Registration")
    eval_task.after(hpo_task)
    
    # Add Azure Storage secret as environment variable
    kubernetes.use_secret_as_env(
        task=eval_task,
        secret_name='azure-storage-secret',
        secret_key_to_env={'AZURE_STORAGE_CONNECTION_STRING': 'AZURE_STORAGE_CONNECTION_STRING'}
    )


if __name__ == "__main__":
    import sys
    from kfp import compiler
    import re
    
    # Get ACR name, image tag, and storage account from command line or environment
    acr_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ACR_NAME")
    image_tag = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("IMAGE_TAG", "latest")
    storage_account = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("STORAGE_ACCOUNT")
    
    if not acr_name:
        print("Error: ACR name is required")
        print("Usage: python spam_detection_pipeline.py <ACR_NAME> [IMAGE_TAG] [STORAGE_ACCOUNT]")
        print("   or: ACR_NAME=myacr.azurecr.io python spam_detection_pipeline.py")
        sys.exit(1)
    
    print(f"Compiling pipeline with:")
    print(f"  ACR: {acr_name}")
    print(f"  Image Tag: {image_tag}")
    if storage_account:
        print(f"  Storage Account: {storage_account}")
    
    # Define actual images
    data_prep_image = f"{acr_name}/ml-data-prep:{image_tag}"
    mlflow_ops_image = f"{acr_name}/ml-mlflow-ops:{image_tag}"
    
    # Compile pipeline to YAML
    compiler.Compiler().compile(
        pipeline_func=spam_detection_pipeline,
        package_path="spam_detection_pipeline.yaml",
    )
    
    # Post-process YAML to fix placeholder images and remove unsupported fields
    with open("spam_detection_pipeline.yaml", "r") as f:
        yaml_content = f.read()
    
    # Replace placeholder images with actual images
    # exec-data-preparation and exec-feature-retrieval use ml-data-prep image
    # exec-baseline-training and exec-model-evaluation use ml-mlflow-ops image
    # exec-hpo-tuning uses ml-data-prep image
    
    # Split into sections and replace appropriately
    yaml_content = re.sub(
        r'(exec-data-preparation:.*?image:\s*)placeholder',
        rf'\g<1>{data_prep_image}',
        yaml_content,
        flags=re.DOTALL
    )
    yaml_content = re.sub(
        r'(exec-feature-retrieval:.*?image:\s*)placeholder',
        rf'\g<1>{data_prep_image}',
        yaml_content,
        flags=re.DOTALL
    )
    yaml_content = re.sub(
        r'(exec-baseline-training:.*?image:\s*)placeholder',
        rf'\g<1>{mlflow_ops_image}',
        yaml_content,
        flags=re.DOTALL
    )
    yaml_content = re.sub(
        r'(exec-hpo-tuning:.*?image:\s*)placeholder',
        rf'\g<1>{data_prep_image}',
        yaml_content,
        flags=re.DOTALL
    )
    yaml_content = re.sub(
        r'(exec-model-evaluation:.*?image:\s*)placeholder',
        rf'\g<1>{mlflow_ops_image}',
        yaml_content,
        flags=re.DOTALL
    )
    
    # Remove 'optional: false' lines
    yaml_content = re.sub(r'\n\s*optional:\s*false\s*\n', '\n', yaml_content)
    
    # Remove secretNameParameter blocks
    yaml_content = re.sub(
        r'\n\s*secretNameParameter:\s*\n\s*runtimeValue:\s*\n\s*constant:\s*[^\n]+\n',
        '\n',
        yaml_content
    )
    
    # If storage account was provided, update the default value in the YAML
    if storage_account:
        # Update the default value in the root parameters section
        yaml_content = re.sub(
            r"(storage_account:\s*\n\s*defaultValue:\s*)mltrainingsdevsal6xriy",
            rf"\g<1>{storage_account}",
            yaml_content
        )
        # Also update the comment at the top
        yaml_content = re.sub(
            r"(#\s*storage_account: str \[Default: ')mltrainingsdevsal6xriy(')",
            rf"\g<1>{storage_account}\g<2>",
            yaml_content
        )
    
    with open("spam_detection_pipeline.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"\n✓ Pipeline compiled to spam_detection_pipeline.yaml")
    print(f"  Images:")
    print(f"    - data-prep, feature-retrieval, hpo-tuning: {data_prep_image}")
    print(f"    - baseline-training, model-evaluation: {mlflow_ops_image}")
    if storage_account:
        print(f"  Default Storage Account: {storage_account}")
