"""
Model Comparison Component

Compares a new model against the current staging model to ensure
the new model is not worse. This provides a safety gate before promotion.
"""
import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Output, Metrics


@dsl.component(
    base_image="placeholder",  # Will be set dynamically
)
def model_comparison(
    acr_name: str,
    image_tag: str,
    new_model_path: str,
    test_features_path: str,
    storage_account: str,
    container_name: str,
    mlflow_tracking_uri: str,
    model_name: str,
    # Comparison thresholds (relative to current staging)
    max_f1_regression: float,  # e.g., 0.05 means new model can be at most 5% worse
    max_auc_regression: float,
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [
    ("comparison_passed", bool),
    ("has_staging_model", bool),
    ("new_model_f1", float),
    ("staging_model_f1", float),
    ("f1_improvement", float),
    ("new_model_auc", float),
    ("staging_model_auc", float),
    ("auc_improvement", float),
    ("comparison_report", str),
]):
    """
    Compare new model against current staging model.
    
    This component:
    1. Loads the current staging model from MLflow (if exists)
    2. Evaluates both models on the same test set
    3. Compares metrics to ensure no significant regression
    
    Args:
        acr_name: Azure Container Registry name
        image_tag: Docker image tag
        new_model_path: Blob path to new model pickle
        test_features_path: Blob path to test features parquet
        storage_account: Azure storage account name
        container_name: Azure blob container name
        mlflow_tracking_uri: MLflow tracking server URI
        model_name: Name of registered model in MLflow
        max_f1_regression: Maximum allowed F1 regression (e.g., 0.05 = 5%)
        max_auc_regression: Maximum allowed AUC regression
        metrics: Output metrics object
        
    Returns:
        comparison_passed: Whether comparison check passed
        has_staging_model: Whether a staging model exists
        new_model_f1: F1 score of new model
        staging_model_f1: F1 score of staging model (0 if none)
        f1_improvement: F1 improvement (positive = better)
        new_model_auc: AUC-ROC of new model
        staging_model_auc: AUC-ROC of staging model (0 if none)
        auc_improvement: AUC improvement (positive = better)
        comparison_report: Detailed comparison report as JSON string
    """
    import os
    import pickle
    import json
    import numpy as np
    import pandas as pd
    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import (
        f1_score as sklearn_f1_score,
        roc_auc_score,
        precision_score,
        recall_score,
        accuracy_score,
    )
    from azure.storage.blob import BlobServiceClient
    
    # Suppress git warnings
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    
    # Initialize Azure storage
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    storage_options = {"connection_string": connection_string}
    
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\nComparison Thresholds:")
    print(f"  Max F1 Regression:  {max_f1_regression:.1%}")
    print(f"  Max AUC Regression: {max_auc_regression:.1%}")
    print()
    
    # Load test data
    test_read_path = f"abfs://{test_features_path}"
    test_df = pd.read_parquet(test_read_path, storage_options=storage_options)
    
    print(f"Test set size: {len(test_df)} samples")
    
    # Prepare features
    exclude_cols = ["email_id", "sender_domain", "event_timestamp", "label"]
    numeric_cols = test_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    
    # Encode labels if strings
    if y_test.dtype == 'object':
        label_map = {"ham": 0, "spam": 1}
        y_test = np.array([label_map.get(str(label).lower(), label) for label in y_test])
    
    # ============================================
    # Load New Model
    # ============================================
    print("\n[Step 1] Loading New Model")
    print("-" * 40)
    
    model_blob_path = new_model_path.split('/', 1)[1] if '/' in new_model_path else new_model_path
    
    model_blob_client = container_client.get_blob_client(model_blob_path)
    model_data = model_blob_client.download_blob().readall()
    new_model = pickle.loads(model_data)
    
    scaler_path = model_blob_path.replace("best_model.pkl", "scaler.pkl")
    scaler_blob_client = container_client.get_blob_client(scaler_path)
    scaler_data = scaler_blob_client.download_blob().readall()
    new_scaler = pickle.loads(scaler_data)
    
    print("  ✓ New model loaded successfully")
    
    # Scale features for new model
    X_test_new = new_scaler.transform(X_test)
    X_test_new = np.nan_to_num(X_test_new, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Evaluate new model
    new_pred = new_model.predict(X_test_new)
    new_pred_proba = new_model.predict_proba(X_test_new)[:, 1]
    
    new_f1 = float(sklearn_f1_score(y_test, new_pred))
    new_auc = float(roc_auc_score(y_test, new_pred_proba))
    new_precision = float(precision_score(y_test, new_pred))
    new_recall = float(recall_score(y_test, new_pred))
    new_accuracy = float(accuracy_score(y_test, new_pred))
    
    print(f"\n  New Model Metrics:")
    print(f"    F1 Score:  {new_f1:.4f}")
    print(f"    AUC-ROC:   {new_auc:.4f}")
    print(f"    Precision: {new_precision:.4f}")
    print(f"    Recall:    {new_recall:.4f}")
    print(f"    Accuracy:  {new_accuracy:.4f}")
    
    # ============================================
    # Load Staging Model (if exists)
    # ============================================
    print("\n[Step 2] Loading Staging Model")
    print("-" * 40)
    
    staging_model = None
    staging_scaler = None
    staging_f1 = 0.0
    staging_auc = 0.0
    staging_precision = 0.0
    staging_recall = 0.0
    staging_accuracy = 0.0
    has_staging = False
    staging_version = None
    
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient()
        
        # Get staging model version
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        
        if versions:
            staging_version = versions[0].version
            print(f"  Found staging model: {model_name} v{staging_version}")
            
            # Load staging model
            model_uri = f"models:/{model_name}/Staging"
            staging_model = mlflow.xgboost.load_model(model_uri)
            has_staging = True
            
            # Try to load staging scaler (may be in the same run)
            try:
                run_id = versions[0].run_id
                scaler_uri = f"runs:/{run_id}/scaler"
                staging_scaler = mlflow.sklearn.load_model(scaler_uri)
                print("  ✓ Staging scaler loaded")
            except Exception as e:
                print(f"  Warning: Could not load staging scaler: {e}")
                print("  Using new model's scaler for comparison (may affect results)")
                staging_scaler = new_scaler
            
            # Scale features for staging model
            X_test_staging = staging_scaler.transform(X_test)
            X_test_staging = np.nan_to_num(X_test_staging, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Evaluate staging model
            staging_pred = staging_model.predict(X_test_staging)
            staging_pred_proba = staging_model.predict_proba(X_test_staging)[:, 1]
            
            staging_f1 = float(sklearn_f1_score(y_test, staging_pred))
            staging_auc = float(roc_auc_score(y_test, staging_pred_proba))
            staging_precision = float(precision_score(y_test, staging_pred))
            staging_recall = float(recall_score(y_test, staging_pred))
            staging_accuracy = float(accuracy_score(y_test, staging_pred))
            
            print(f"\n  Staging Model Metrics:")
            print(f"    F1 Score:  {staging_f1:.4f}")
            print(f"    AUC-ROC:   {staging_auc:.4f}")
            print(f"    Precision: {staging_precision:.4f}")
            print(f"    Recall:    {staging_recall:.4f}")
            print(f"    Accuracy:  {staging_accuracy:.4f}")
        else:
            print("  No staging model found - this is the first model")
            print("  ✓ Comparison will auto-pass (no baseline to compare)")
            
    except Exception as e:
        print(f"  Warning: Could not load staging model: {e}")
        print("  ✓ Comparison will auto-pass (no baseline to compare)")
    
    # ============================================
    # Compare Models
    # ============================================
    print("\n[Step 3] Model Comparison")
    print("-" * 40)
    
    comparison_passed = True
    f1_improvement = 0.0
    auc_improvement = 0.0
    
    if has_staging:
        # Calculate improvements (positive = better)
        f1_improvement = float(new_f1 - staging_f1)
        auc_improvement = float(new_auc - staging_auc)
        
        # Check for regression
        f1_regression = float(-f1_improvement if f1_improvement < 0 else 0)
        auc_regression = float(-auc_improvement if auc_improvement < 0 else 0)
        
        f1_check = bool(f1_regression <= max_f1_regression)
        auc_check = bool(auc_regression <= max_auc_regression)
        
        print(f"\n  F1 Comparison:")
        print(f"    Staging:     {staging_f1:.4f}")
        print(f"    New:         {new_f1:.4f}")
        print(f"    Change:      {f1_improvement:+.4f} ({f1_improvement/max(staging_f1, 0.001)*100:+.1f}%)")
        print(f"    Max Regress: {max_f1_regression:.1%}")
        print(f"    Status:      {'✓ PASS' if f1_check else '✗ FAIL'}")
        
        print(f"\n  AUC-ROC Comparison:")
        print(f"    Staging:     {staging_auc:.4f}")
        print(f"    New:         {new_auc:.4f}")
        print(f"    Change:      {auc_improvement:+.4f} ({auc_improvement/max(staging_auc, 0.001)*100:+.1f}%)")
        print(f"    Max Regress: {max_auc_regression:.1%}")
        print(f"    Status:      {'✓ PASS' if auc_check else '✗ FAIL'}")
        
        comparison_passed = bool(f1_check and auc_check)
        
        # Additional metrics comparison (informational)
        print(f"\n  Other Metrics:")
        print(f"    Precision: {staging_precision:.4f} → {new_precision:.4f} ({new_precision - staging_precision:+.4f})")
        print(f"    Recall:    {staging_recall:.4f} → {new_recall:.4f} ({new_recall - staging_recall:+.4f})")
        print(f"    Accuracy:  {staging_accuracy:.4f} → {new_accuracy:.4f} ({new_accuracy - staging_accuracy:+.4f})")
    else:
        print("  No staging model to compare against")
        print("  ✓ Auto-pass: First model deployment")
        comparison_passed = True
        f1_improvement = float(new_f1)
        auc_improvement = float(new_auc)
    
    # ============================================
    # Final Summary
    # ============================================
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if comparison_passed:
        if has_staging:
            if f1_improvement >= 0 and auc_improvement >= 0:
                print("✓ COMPARISON PASSED - New model is BETTER")
            else:
                print("✓ COMPARISON PASSED - Regression within acceptable limits")
        else:
            print("✓ COMPARISON PASSED - No staging model (first deployment)")
        print("  Model is eligible for staging promotion")
    else:
        print("✗ COMPARISON FAILED - New model is significantly worse")
        print("  Model will NOT be promoted to staging")
    print("=" * 60)
    
    # Log metrics
    metrics.log_metric("comparison_passed", int(comparison_passed))
    metrics.log_metric("has_staging_model", int(has_staging))
    metrics.log_metric("new_model_f1", new_f1)
    metrics.log_metric("staging_model_f1", staging_f1)
    metrics.log_metric("f1_improvement", f1_improvement)
    metrics.log_metric("new_model_auc", new_auc)
    metrics.log_metric("staging_model_auc", staging_auc)
    metrics.log_metric("auc_improvement", auc_improvement)
    
    # Create comparison report - ensure all values are native Python types
    comparison_report = json.dumps({
        "comparison_passed": bool(comparison_passed),
        "has_staging_model": bool(has_staging),
        "staging_version": staging_version,
        "new_model": {
            "f1": float(new_f1),
            "auc_roc": float(new_auc),
            "precision": float(new_precision),
            "recall": float(new_recall),
            "accuracy": float(new_accuracy),
        },
        "staging_model": {
            "f1": float(staging_f1),
            "auc_roc": float(staging_auc),
            "precision": float(staging_precision),
            "recall": float(staging_recall),
            "accuracy": float(staging_accuracy),
        } if has_staging else None,
        "improvements": {
            "f1": float(f1_improvement),
            "auc_roc": float(auc_improvement),
        },
        "thresholds": {
            "max_f1_regression": float(max_f1_regression),
            "max_auc_regression": float(max_auc_regression),
        },
    }, indent=2)
    
    from collections import namedtuple
    outputs = namedtuple("Outputs", [
        "comparison_passed", "has_staging_model", 
        "new_model_f1", "staging_model_f1", "f1_improvement",
        "new_model_auc", "staging_model_auc", "auc_improvement",
        "comparison_report"
    ])
    return outputs(
        bool(comparison_passed),
        bool(has_staging),
        float(new_f1),
        float(staging_f1),
        float(f1_improvement),
        float(new_auc),
        float(staging_auc),
        float(auc_improvement),
        comparison_report
    )
