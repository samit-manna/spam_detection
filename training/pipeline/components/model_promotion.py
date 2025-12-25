"""
Model Promotion Component

Promotes a validated model to the Staging stage in MLflow.
This component should only be called after validation and comparison pass.
"""
import os
from typing import NamedTuple
from datetime import datetime, timezone

from kfp import dsl
from kfp.dsl import Output, Metrics


@dsl.component(
    base_image="placeholder",  # Will be set dynamically
)
def model_promotion(
    acr_name: str,
    image_tag: str,
    best_model_path: str,
    test_features_path: str,
    storage_account: str,
    container_name: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    model_name: str,
    validation_passed: bool,
    comparison_passed: bool,
    validation_report: str,
    comparison_report: str,
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [
    ("promoted", bool),
    ("model_version", str),
    ("promotion_report", str),
]):
    """
    Promote validated model to Staging in MLflow.
    
    This component:
    1. Checks that validation and comparison both passed
    2. Registers the model to MLflow
    3. Transitions the model to Staging stage
    4. Archives previous staging model
    5. Logs all promotion metadata
    
    Args:
        acr_name: Azure Container Registry name
        image_tag: Docker image tag
        best_model_path: Blob path to best model pickle
        test_features_path: Blob path to test features parquet (for evaluation)
        storage_account: Azure storage account name
        container_name: Azure blob container name
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        model_name: Name for model registry
        validation_passed: Whether validation checks passed
        comparison_passed: Whether comparison checks passed
        validation_report: JSON report from validation step
        comparison_report: JSON report from comparison step
        metrics: Output metrics object
        
    Returns:
        promoted: Whether model was promoted
        model_version: Version number if promoted, else empty string
        promotion_report: Detailed promotion report as JSON string
    """
    import os
    import pickle
    import json
    from datetime import datetime, timezone
    import numpy as np
    import pandas as pd
    import mlflow
    import mlflow.xgboost
    import mlflow.sklearn
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
    
    print("=" * 60)
    print("MODEL PROMOTION TO STAGING")
    print("=" * 60)
    
    # ============================================
    # Check Gates
    # ============================================
    print("\n[Step 1] Checking Promotion Gates")
    print("-" * 40)
    
    print(f"  Validation Passed: {'✓ YES' if validation_passed else '✗ NO'}")
    print(f"  Comparison Passed: {'✓ YES' if comparison_passed else '✗ NO'}")
    
    if not validation_passed or not comparison_passed:
        print("\n" + "=" * 60)
        print("✗ PROMOTION BLOCKED - Prerequisites not met")
        print("=" * 60)
        
        if not validation_passed:
            print("  - Model failed validation checks")
        if not comparison_passed:
            print("  - Model failed comparison against staging")
        
        # Log metrics
        metrics.log_metric("promoted", 0)
        metrics.log_metric("blocked_by_validation", int(not validation_passed))
        metrics.log_metric("blocked_by_comparison", int(not comparison_passed))
        
        promotion_report = json.dumps({
            "promoted": False,
            "reason": "Prerequisites not met",
            "validation_passed": validation_passed,
            "comparison_passed": comparison_passed,
        }, indent=2)
        
        from collections import namedtuple
        outputs = namedtuple("Outputs", ["promoted", "model_version", "promotion_report"])
        return outputs(False, "", promotion_report)
    
    print("\n  ✓ All gates passed - proceeding with promotion")
    
    # Initialize Azure storage
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    storage_options = {"connection_string": connection_string}
    
    # ============================================
    # Load Model and Evaluate
    # ============================================
    print("\n[Step 2] Loading Model")
    print("-" * 40)
    
    model_blob_path = best_model_path.split('/', 1)[1] if '/' in best_model_path else best_model_path
    
    model_blob_client = container_client.get_blob_client(model_blob_path)
    model_data = model_blob_client.download_blob().readall()
    model = pickle.loads(model_data)
    
    scaler_path = model_blob_path.replace("best_model.pkl", "scaler.pkl")
    scaler_blob_client = container_client.get_blob_client(scaler_path)
    scaler_data = scaler_blob_client.download_blob().readall()
    scaler = pickle.loads(scaler_data)
    
    print("  ✓ Model and scaler loaded")
    
    # Load test data for final metrics
    test_read_path = f"abfs://{test_features_path}"
    test_df = pd.read_parquet(test_read_path, storage_options=storage_options)
    
    exclude_cols = ["email_id", "sender_domain", "event_timestamp", "label"]
    numeric_cols = test_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    
    if y_test.dtype == 'object':
        label_map = {"ham": 0, "spam": 1}
        y_test = np.array([label_map.get(str(label).lower(), label) for label in y_test])
    
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    final_f1 = sklearn_f1_score(y_test, y_pred)
    final_auc = roc_auc_score(y_test, y_pred_proba)
    final_precision = precision_score(y_test, y_pred)
    final_recall = recall_score(y_test, y_pred)
    final_accuracy = accuracy_score(y_test, y_pred)
    
    # ============================================
    # Register to MLflow
    # ============================================
    print("\n[Step 3] Registering to MLflow")
    print("-" * 40)
    
    promoted = False
    model_version = ""
    previous_staging_version = None
    
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        client = MlflowClient()
        
        # Check for existing staging model
        try:
            versions = client.get_latest_versions(model_name, stages=["Staging"])
            if versions:
                previous_staging_version = versions[0].version
                print(f"  Current staging: v{previous_staging_version} (will be archived)")
        except Exception:
            pass
        
        # Start MLflow run for registration
        with mlflow.start_run(run_name="model_promotion"):
            # Log final metrics
            mlflow.log_metric("test_f1", final_f1)
            mlflow.log_metric("test_auc_roc", final_auc)
            mlflow.log_metric("test_precision", final_precision)
            mlflow.log_metric("test_recall", final_recall)
            mlflow.log_metric("test_accuracy", final_accuracy)
            mlflow.log_metric("num_features", len(feature_cols))
            
            # Log validation and comparison reports
            mlflow.log_text(validation_report, "validation_report.json")
            mlflow.log_text(comparison_report, "comparison_report.json")
            
            # Register model
            print("  Registering XGBoost model...")
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name,
            )
            
            # Register scaler
            print("  Registering scaler...")
            mlflow.sklearn.log_model(scaler, artifact_path="scaler")
            
            print("  ✓ Model registered")
        
        # Wait for registration to complete
        import time
        time.sleep(2)
        
        # Get the new version
        registered_model = client.get_registered_model(model_name)
        latest_versions = registered_model.latest_versions
        
        if latest_versions:
            # Find the newest version (highest version number)
            model_version = max([v.version for v in latest_versions])
            print(f"  New model version: v{model_version}")
            
            # Add metadata tags
            timestamp = datetime.now(timezone.utc).isoformat()
            
            client.set_model_version_tag(model_name, model_version, "test_f1", str(round(final_f1, 4)))
            client.set_model_version_tag(model_name, model_version, "test_auc_roc", str(round(final_auc, 4)))
            client.set_model_version_tag(model_name, model_version, "test_precision", str(round(final_precision, 4)))
            client.set_model_version_tag(model_name, model_version, "test_recall", str(round(final_recall, 4)))
            client.set_model_version_tag(model_name, model_version, "promotion_timestamp", timestamp)
            client.set_model_version_tag(model_name, model_version, "promoted_by", "kubeflow_pipeline")
            client.set_model_version_tag(model_name, model_version, "validation_passed", "true")
            client.set_model_version_tag(model_name, model_version, "comparison_passed", "true")
            
            if previous_staging_version:
                client.set_model_version_tag(model_name, model_version, "replaced_version", str(previous_staging_version))
            
            # Transition to Staging (archives previous staging automatically)
            print(f"\n[Step 4] Transitioning to Staging")
            print("-" * 40)
            
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging",
                archive_existing_versions=True,
            )
            
            promoted = True
            print(f"  ✓ Model v{model_version} promoted to Staging")
            
            if previous_staging_version:
                print(f"  ✓ Previous staging (v{previous_staging_version}) archived")
        else:
            print("  ✗ Failed to get model version after registration")
            
    except Exception as e:
        print(f"  ✗ MLflow registration failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================
    # Final Summary
    # ============================================
    print("\n" + "=" * 60)
    print("PROMOTION SUMMARY")
    print("=" * 60)
    
    if promoted:
        print(f"✓ MODEL PROMOTED TO STAGING")
        print(f"  Model:   {model_name}")
        print(f"  Version: v{model_version}")
        print(f"  F1:      {final_f1:.4f}")
        print(f"  AUC-ROC: {final_auc:.4f}")
        if previous_staging_version:
            print(f"  Replaced: v{previous_staging_version} (now Archived)")
        print()
        print("NEXT STEPS:")
        print("  1. Review model in MLflow UI")
        print("  2. Run deployment pipeline to deploy to KServe")
        print("  3. Test staging endpoint before production promotion")
    else:
        print("✗ PROMOTION FAILED")
        print("  See logs above for details")
    print("=" * 60)
    
    # Log metrics
    metrics.log_metric("promoted", int(promoted))
    if model_version:
        metrics.log_metric("model_version", int(model_version))
    metrics.log_metric("final_f1", final_f1)
    metrics.log_metric("final_auc_roc", final_auc)
    
    # Create promotion report
    promotion_report = json.dumps({
        "promoted": promoted,
        "model_name": model_name,
        "model_version": model_version,
        "previous_staging_version": previous_staging_version,
        "final_metrics": {
            "f1": final_f1,
            "auc_roc": final_auc,
            "precision": final_precision,
            "recall": final_recall,
            "accuracy": final_accuracy,
        },
        "validation_passed": validation_passed,
        "comparison_passed": comparison_passed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2)
    
    from collections import namedtuple
    outputs = namedtuple("Outputs", ["promoted", "model_version", "promotion_report"])
    return outputs(promoted, str(model_version) if model_version else "", promotion_report)
