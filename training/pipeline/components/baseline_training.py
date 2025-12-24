"""
Baseline Model Training Component

Trains a baseline model (Logistic Regression or Random Forest) to validate 
the pipeline before running more expensive HPO.
"""
import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Output, Metrics, InputPath, OutputPath


@dsl.component(
    base_image="placeholder",  # Will be set dynamically
)
def baseline_training(
    acr_name: str,
    image_tag: str,
    train_features_path: str,
    storage_account: str,
    container_name: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    model_type: str,  # "logistic_regression" or "random_forest"
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("f1_score", float), ("precision", float), ("recall", float), ("auc_roc", float), ("baseline_model_path", str)]):
    """
    Train a baseline model for spam detection.
    
    Args:
        acr_name: Azure Container Registry name
        image_tag: Docker image tag
        train_features_path: Blob path to training features parquet
        storage_account: Azure storage account name
        container_name: Azure blob container name
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: Name of MLflow experiment
        model_type: Type of baseline model
        baseline_model_path: Output path for model artifact
        metrics: Output metrics object
        
    Returns:
        f1_score: F1 score on validation set
        precision: Precision on validation set
        recall: Recall on validation set
        auc_roc: AUC-ROC on validation set
    """
    import os
    import pickle
    import io
    
    # Suppress git warning from MLflow
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    
    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score as sklearn_f1_score,
        roc_auc_score,
        classification_report,
    )
    from sklearn.preprocessing import StandardScaler
    from azure.storage.blob import BlobServiceClient
    
    # Initialize Azure Blob client
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    
    storage_options = {"connection_string": connection_string}
    
    # Use abfs:// protocol for reading
    read_path = f"abfs://{train_features_path}"
    print(f"Reading training features from: {read_path}")
    
    storage_options = {"connection_string": connection_string}
    df = pd.read_parquet(read_path, storage_options=storage_options)
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()[:20]}...")
    
    # Prepare features and labels
    # Exclude entity columns, label, and any timestamp/metadata columns
    exclude_cols = ["email_id", "sender_domain", "event_timestamp", "label", 
                    "event_timestamp_feat", "created_timestamp"]
    
    # Filter to only numeric columns and exclude the above
    numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} numeric feature columns")
    
    # Verify label column exists and extract it
    if "label" not in df.columns:
        raise ValueError("Label column not found in dataframe")
    
    X = df[feature_cols].values
    y = df["label"].values
    
    # Encode labels if they are strings
    if y.dtype == 'object' or y.dtype.name == 'string':
        print(f"Converting string labels to binary. Unique values: {np.unique(y)}")
        # Map spam/ham to 1/0
        label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1}
        y = np.array([label_map.get(label, label) for label in y])
        print(f"After encoding: {np.unique(y)}")
    
    print(f"Features shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Train/validation split (from training set)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training subset: {len(X_train)} samples")
    print(f"Validation subset: {len(X_val)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Handle NaN/Inf values
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Select model
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            solver="saga",  # Better for high-dimensional data
        )
        model_params = {
            "model_type": "logistic_regression",
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "saga",
        }
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model_params = {
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 15,
            "class_weight": "balanced",
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Train model with MLflow tracking
    with mlflow.start_run(run_name=f"baseline_{model_type}"):
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        
        # Train
        print(f"Training {model_type} model...")
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = sklearn_f1_score(y_val, y_pred)
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"\nValidation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc_roc)
        
        # Log classification report
        report = classification_report(y_val, y_pred)
        print(f"\nClassification Report:\n{report}")
        
        # Save report to file and log as artifact (avoid mlflow.log_text issues)
        try:
            report_path = "/tmp/classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)
        except Exception as e:
            print(f"Warning: Could not log classification report to MLflow: {e}")
        
        # Log feature importance (for tree-based models)
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            
            print(f"\nTop 20 Important Features:")
            print(importance_df.head(20))
            
            try:
                importance_path = "/tmp/feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
            except Exception as e:
                print(f"Warning: Could not log feature importance to MLflow: {e}")
        
        # Log model to MLflow
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.sklearn.log_model(scaler, artifact_path="scaler")
            print("Successfully logged model and scaler to MLflow")
        except Exception as e:
            print(f"Warning: Could not log model to MLflow: {e}")
            print("Continuing with blob storage upload...")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
    
    # Save model to blob storage using BlobServiceClient
    container_client = blob_service.get_container_client(container_name)
    
    model_blob_path = f"baseline/{model_type}/model.pkl"
    scaler_blob_path = f"baseline/{model_type}/scaler.pkl"
    
    # Upload model
    model_bytes = pickle.dumps(model)
    container_client.upload_blob(model_blob_path, model_bytes, overwrite=True)
    
    # Upload scaler  
    scaler_bytes = pickle.dumps(scaler)
    container_client.upload_blob(scaler_blob_path, scaler_bytes, overwrite=True)
    
    model_output_path = f"{container_name}/{model_blob_path}"
    print(f"\nSaved model to: {model_output_path}")
    
    # Log output metrics
    metrics.log_metric("f1_score", f1)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)
    metrics.log_metric("auc_roc", auc_roc)
    metrics.log_metric("accuracy", accuracy)
    
    from collections import namedtuple
    outputs = namedtuple("Outputs", ["f1_score", "precision", "recall", "auc_roc", "baseline_model_path"])
    return outputs(float(f1), float(precision), float(recall), float(auc_roc), model_output_path)
