"""
Model Evaluation and Registration Component

Evaluates the best model on holdout test set and registers to MLflow if threshold is met.
"""
import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Output, Metrics, InputPath


@dsl.component(
    base_image="placeholder",  # Will be set dynamically
)
def model_evaluation(
    acr_name: str,
    image_tag: str,
    best_model_path: str,
    test_features_path: str,
    storage_account: str,
    container_name: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    model_name: str,
    f1_threshold: float,
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("test_f1", float), ("registered", bool), ("model_version", str)]):
    """
    Evaluate best model on test set and register to MLflow if threshold is met.
    
    Args:
        acr_name: Azure Container Registry name
        image_tag: Docker image tag
        best_model_path: Blob path to best model pickle
        test_features_path: Blob path to test features parquet
        storage_account: Azure storage account name
        container_name: Azure blob container name
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: Name of MLflow experiment
        model_name: Name for model registry
        f1_threshold: Minimum F1 score to register model
        metrics: Output metrics object
        
    Returns:
        test_f1: F1 score on test set
        registered: Whether model was registered
        model_version: Version number if registered, else empty string
    """
    import os
    import pickle
    import io
    import mlflow
    import mlflow.xgboost
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score as sklearn_f1_score,
        roc_auc_score,
        classification_report,
        confusion_matrix,
        roc_curve,
        precision_recall_curve,
    )
    from sklearn.preprocessing import StandardScaler
    from azure.storage.blob import BlobServiceClient
    
    # Initialize Azure storage
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    storage_options = {"connection_string": connection_string}
    
    print(f"Loading model from: {best_model_path}")
    print(f"Loading test data from: {test_features_path}")
    
    # Extract blob path (remove container prefix if present)
    model_blob_path = best_model_path.split('/', 1)[1] if '/' in best_model_path else best_model_path
    
    # Load model and scaler from blob storage
    model_blob_client = container_client.get_blob_client(model_blob_path)
    model_data = model_blob_client.download_blob().readall()
    model = pickle.loads(model_data)
    
    scaler_path = model_blob_path.replace("best_model.pkl", "scaler.pkl")
    scaler_blob_client = container_client.get_blob_client(scaler_path)
    scaler_data = scaler_blob_client.download_blob().readall()
    scaler = pickle.loads(scaler_data)
    
    # Load test data using abfs://
    test_read_path = f"abfs://{test_features_path}"
    test_df = pd.read_parquet(test_read_path, storage_options=storage_options)
    
    print(f"Test set: {len(test_df)} samples")
    
    # Prepare features - filter to numeric columns only
    exclude_cols = ["email_id", "sender_domain", "event_timestamp", "label"]
    numeric_cols = test_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    
    # Encode labels if they are strings
    if y_test.dtype == 'object':
        print("Encoding string labels to numeric...")
        label_map = {"ham": 0, "spam": 1}
        y_test = np.array([label_map.get(label.lower() if isinstance(label, str) else label, label) for label in y_test])
        print(f"Label encoding: {label_map}")
    
    print(f"Features shape: {X_test.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Predict
    print("Running predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = sklearn_f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"{'='*50}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    print(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Suppress git warnings
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    
    # Set up MLflow (with error handling for connection issues)
    registered = False
    model_version = ""
    
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        print("Successfully connected to MLflow")
    except Exception as e:
        print(f"Warning: Failed to connect to MLflow: {e}")
        print("Continuing without MLflow tracking...")
    
    try:
        with mlflow.start_run(run_name="final_evaluation"):
            # Log parameters
            mlflow.log_param("test_samples", len(test_df))
            mlflow.log_param("f1_threshold", f1_threshold)
            mlflow.log_param("num_features", len(feature_cols))
            
            # Log metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_metric("test_auc_roc", auc_roc)
            
            # Log classification report
            try:
                mlflow.log_text(report, "test_classification_report.txt")
            except Exception as e:
                print(f"Warning: Failed to log classification report: {e}")
            
            # Create and log confusion matrix plot
            try:
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Ham', 'Spam'],
                            yticklabels=['Ham', 'Spam'])
                plt.title('Confusion Matrix - Test Set')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                cm_path = "/tmp/confusion_matrix.png"
                plt.savefig(cm_path, dpi=100)
                mlflow.log_artifact(cm_path)
                plt.close()
                print("Successfully logged confusion matrix")
            except Exception as e:
                print(f"Warning: Failed to log confusion matrix: {e}")
            
            # Create and log ROC curve
            try:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve - Test Set')
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                roc_path = "/tmp/roc_curve.png"
                plt.savefig(roc_path, dpi=100)
                mlflow.log_artifact(roc_path)
                plt.close()
                print("Successfully logged ROC curve")
            except Exception as e:
                print(f"Warning: Failed to log ROC curve: {e}")
            
            # Create and log precision-recall curve
            try:
                prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(rec_curve, prec_curve, 'b-', linewidth=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve - Test Set')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                pr_path = "/tmp/pr_curve.png"
                plt.savefig(pr_path, dpi=100)
                mlflow.log_artifact(pr_path)
                plt.close()
                print("Successfully logged PR curve")
            except Exception as e:
                print(f"Warning: Failed to log PR curve: {e}")
            
            # Check if model meets threshold
            if f1 >= f1_threshold:
                print(f"\n✓ Model F1 ({f1:.4f}) >= threshold ({f1_threshold})")
                print(f"Registering model to MLflow as '{model_name}'...")
                
                # Log and register model
                try:
                    mlflow.xgboost.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=model_name,
                    )
                    print("Successfully registered XGBoost model")
                except Exception as e:
                    print(f"Warning: Failed to register model: {e}")
                
                # Also log scaler
                try:
                    mlflow.sklearn.log_model(scaler, artifact_path="scaler")
                    print("Successfully logged scaler")
                except Exception as e:
                    print(f"Warning: Failed to log scaler: {e}")
                
                # Get registered model version and add tags
                try:
                    client = mlflow.tracking.MlflowClient()
                    
                    # Wait a moment for registration to complete
                    import time
                    time.sleep(2)
                    
                    registered_model = client.get_registered_model(model_name)
                    latest_versions = registered_model.latest_versions
                    if latest_versions:
                        model_version = latest_versions[0].version
                        
                        # Add tags to model version
                        client.set_model_version_tag(
                            name=model_name,
                            version=model_version,
                            key="test_f1",
                            value=str(round(f1, 4)),
                        )
                        client.set_model_version_tag(
                            name=model_name,
                            version=model_version,
                            key="test_auc_roc",
                            value=str(round(auc_roc, 4)),
                        )
                        client.set_model_version_tag(
                            name=model_name,
                            version=model_version,
                            key="test_precision",
                            value=str(round(precision, 4)),
                        )
                        client.set_model_version_tag(
                            name=model_name,
                            version=model_version,
                            key="test_recall",
                            value=str(round(recall, 4)),
                        )
                        
                        # Transition to Staging
                        client.transition_model_version_stage(
                            name=model_name,
                            version=model_version,
                            stage="Staging",
                        )
                        
                        registered = True
                        print(f"✓ Model registered as version {model_version} in Staging")
                    else:
                        print("Warning: No model versions found after registration")
                except Exception as e:
                    print(f"Warning: Failed to get model version or add tags: {e}")
                    # Still consider it registered if the model logging succeeded
                    registered = True
                    model_version = "unknown"
            else:
                print(f"\n✗ Model F1 ({f1:.4f}) < threshold ({f1_threshold})")
                print("Model NOT registered.")
    except Exception as e:
        print(f"Warning: MLflow run failed: {e}")
        print("Continuing without MLflow...")
        
        mlflow.log_param("registered", registered)
        mlflow.log_param("model_version", model_version)
    
    # Log output metrics
    metrics.log_metric("test_f1", f1)
    metrics.log_metric("test_precision", precision)
    metrics.log_metric("test_recall", recall)
    metrics.log_metric("test_auc_roc", auc_roc)
    metrics.log_metric("test_accuracy", accuracy)
    metrics.log_metric("registered", int(registered))
    
    from collections import namedtuple
    outputs = namedtuple("Outputs", ["test_f1", "registered", "model_version"])
    return outputs(float(f1), registered, model_version)
