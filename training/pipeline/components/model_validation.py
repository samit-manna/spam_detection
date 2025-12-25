"""
Model Validation Component

Validates that a trained model meets quality thresholds before promotion.
This provides a gate between training and staging promotion.
"""
import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Output, Metrics


@dsl.component(
    base_image="placeholder",  # Will be set dynamically
)
def model_validation(
    acr_name: str,
    image_tag: str,
    best_model_path: str,
    test_features_path: str,
    storage_account: str,
    container_name: str,
    # Validation thresholds
    min_f1_score: float,
    min_auc_roc: float,
    min_precision: float,
    min_recall: float,
    max_inference_time_ms: float,
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [
    ("validation_passed", bool),
    ("f1_score", float),
    ("auc_roc", float),
    ("precision", float),
    ("recall", float),
    ("inference_time_ms", float),
    ("validation_report", str),
]):
    """
    Validate model against quality thresholds.
    
    This component runs comprehensive validation checks:
    1. Metric thresholds (F1, AUC-ROC, Precision, Recall)
    2. Inference latency check
    3. Prediction sanity check (known examples)
    4. Feature importance analysis
    
    Args:
        acr_name: Azure Container Registry name
        image_tag: Docker image tag
        best_model_path: Blob path to best model pickle
        test_features_path: Blob path to test features parquet
        storage_account: Azure storage account name
        container_name: Azure blob container name
        min_f1_score: Minimum required F1 score
        min_auc_roc: Minimum required AUC-ROC
        min_precision: Minimum required precision
        min_recall: Minimum required recall
        max_inference_time_ms: Maximum allowed inference time in milliseconds
        metrics: Output metrics object
        
    Returns:
        validation_passed: Whether all validation checks passed
        f1_score: Actual F1 score
        auc_roc: Actual AUC-ROC score
        precision: Actual precision
        recall: Actual recall
        inference_time_ms: Actual inference time in milliseconds
        validation_report: Detailed validation report as JSON string
    """
    import os
    import pickle
    import json
    import time
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score as sklearn_f1_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    from azure.storage.blob import BlobServiceClient
    
    # Initialize Azure storage
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    storage_options = {"connection_string": connection_string}
    
    print("=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)
    print(f"\nValidation Thresholds:")
    print(f"  Min F1 Score:        {min_f1_score}")
    print(f"  Min AUC-ROC:         {min_auc_roc}")
    print(f"  Min Precision:       {min_precision}")
    print(f"  Min Recall:          {min_recall}")
    print(f"  Max Inference Time:  {max_inference_time_ms}ms")
    print()
    
    # Load model and scaler
    model_blob_path = best_model_path.split('/', 1)[1] if '/' in best_model_path else best_model_path
    
    model_blob_client = container_client.get_blob_client(model_blob_path)
    model_data = model_blob_client.download_blob().readall()
    model = pickle.loads(model_data)
    
    scaler_path = model_blob_path.replace("best_model.pkl", "scaler.pkl")
    scaler_blob_client = container_client.get_blob_client(scaler_path)
    scaler_data = scaler_blob_client.download_blob().readall()
    scaler = pickle.loads(scaler_data)
    
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
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Initialize validation results
    validation_checks = {}
    all_passed = True
    
    # ============================================
    # Check 1: Metric Thresholds
    # ============================================
    print("\n[Check 1] Metric Thresholds")
    print("-" * 40)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    f1 = sklearn_f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    f1_passed = bool(f1 >= min_f1_score)
    auc_passed = bool(auc >= min_auc_roc)
    prec_passed = bool(prec >= min_precision)
    rec_passed = bool(rec >= min_recall)
    
    print(f"  F1 Score:   {f1:.4f} >= {min_f1_score} : {'✓ PASS' if f1_passed else '✗ FAIL'}")
    print(f"  AUC-ROC:    {auc:.4f} >= {min_auc_roc} : {'✓ PASS' if auc_passed else '✗ FAIL'}")
    print(f"  Precision:  {prec:.4f} >= {min_precision} : {'✓ PASS' if prec_passed else '✗ FAIL'}")
    print(f"  Recall:     {rec:.4f} >= {min_recall} : {'✓ PASS' if rec_passed else '✗ FAIL'}")
    print(f"  Accuracy:   {acc:.4f}")
    
    metrics_passed = bool(f1_passed and auc_passed and prec_passed and rec_passed)
    validation_checks["metrics"] = {
        "passed": metrics_passed,
        "f1_score": {"value": float(f1), "threshold": min_f1_score, "passed": f1_passed},
        "auc_roc": {"value": float(auc), "threshold": min_auc_roc, "passed": auc_passed},
        "precision": {"value": float(prec), "threshold": min_precision, "passed": prec_passed},
        "recall": {"value": float(rec), "threshold": min_recall, "passed": rec_passed},
        "accuracy": float(acc),
    }
    all_passed = all_passed and metrics_passed
    
    # ============================================
    # Check 2: Inference Latency
    # ============================================
    print("\n[Check 2] Inference Latency")
    print("-" * 40)
    
    # Warm up
    _ = model.predict(X_test_scaled[:10])
    
    # Measure inference time (batch of 100 samples)
    batch_size = min(100, len(X_test_scaled))
    n_runs = 10
    times = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test_scaled[:batch_size])
        end = time.perf_counter()
        times.append((end - start) * 1000 / batch_size)  # ms per sample
    
    avg_inference_time = float(np.mean(times))
    p95_inference_time = float(np.percentile(times, 95))
    
    latency_passed = bool(p95_inference_time <= max_inference_time_ms)
    print(f"  Avg per sample:  {avg_inference_time:.3f}ms")
    print(f"  P95 per sample:  {p95_inference_time:.3f}ms <= {max_inference_time_ms}ms : {'✓ PASS' if latency_passed else '✗ FAIL'}")
    
    validation_checks["latency"] = {
        "passed": latency_passed,
        "avg_ms": avg_inference_time,
        "p95_ms": p95_inference_time,
        "threshold_ms": max_inference_time_ms,
    }
    all_passed = all_passed and latency_passed
    
    # ============================================
    # Check 3: Prediction Sanity
    # ============================================
    print("\n[Check 3] Prediction Sanity")
    print("-" * 40)
    
    # Check class distribution in predictions
    pred_spam_ratio = float(np.mean(y_pred))
    actual_spam_ratio = float(np.mean(y_test))
    
    # Sanity check: predicted ratio should be within 50% of actual ratio
    ratio_diff = float(abs(pred_spam_ratio - actual_spam_ratio) / max(actual_spam_ratio, 0.01))
    sanity_passed = bool(ratio_diff < 0.5)  # Within 50%
    
    print(f"  Actual spam ratio:    {actual_spam_ratio:.3f}")
    print(f"  Predicted spam ratio: {pred_spam_ratio:.3f}")
    print(f"  Ratio difference:     {ratio_diff:.1%} < 50% : {'✓ PASS' if sanity_passed else '✗ FAIL'}")
    
    # Check that model isn't predicting all same class
    unique_preds = int(len(np.unique(y_pred)))
    diversity_passed = bool(unique_preds > 1)
    print(f"  Prediction diversity: {unique_preds} classes : {'✓ PASS' if diversity_passed else '✗ FAIL'}")
    
    sanity_overall = bool(sanity_passed and diversity_passed)
    validation_checks["sanity"] = {
        "passed": sanity_overall,
        "actual_spam_ratio": actual_spam_ratio,
        "predicted_spam_ratio": pred_spam_ratio,
        "ratio_difference": ratio_diff,
        "unique_predictions": unique_preds,
    }
    all_passed = all_passed and sanity_overall
    
    # ============================================
    # Check 4: Feature Importance (Informational)
    # ============================================
    print("\n[Check 4] Feature Importance (Informational)")
    print("-" * 40)
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            
            print("  Top 10 features:")
            top_features = []
            for i, idx in enumerate(top_indices):
                feat_name = feature_cols[idx] if idx < len(feature_cols) else f"feature_{idx}"
                print(f"    {i+1}. {feat_name}: {importances[idx]:.4f}")
                top_features.append({"name": feat_name, "importance": float(importances[idx])})
            
            validation_checks["feature_importance"] = {
                "top_features": top_features,
                "total_features": len(feature_cols),
            }
    except Exception as e:
        print(f"  Could not extract feature importance: {e}")
        validation_checks["feature_importance"] = {"error": str(e)}
    
    # ============================================
    # Final Summary
    # ============================================
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for check_name, check_result in validation_checks.items():
        if isinstance(check_result, dict) and "passed" in check_result:
            status = "✓ PASS" if check_result["passed"] else "✗ FAIL"
            print(f"  {check_name}: {status}")
    
    print()
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("  Model is eligible for staging promotion")
    else:
        print("✗ VALIDATION FAILED")
        print("  Model will NOT be promoted to staging")
    print("=" * 60)
    
    # Log metrics to Kubeflow
    metrics.log_metric("validation_passed", int(all_passed))
    metrics.log_metric("f1_score", f1)
    metrics.log_metric("auc_roc", auc)
    metrics.log_metric("precision", prec)
    metrics.log_metric("recall", rec)
    metrics.log_metric("inference_time_ms", p95_inference_time)
    
    # Create validation report - ensure all values are native Python types
    validation_report = json.dumps({
        "validation_passed": bool(all_passed),
        "checks": validation_checks,
        "thresholds": {
            "min_f1_score": float(min_f1_score),
            "min_auc_roc": float(min_auc_roc),
            "min_precision": float(min_precision),
            "min_recall": float(min_recall),
            "max_inference_time_ms": float(max_inference_time_ms),
        },
    }, indent=2)
    
    from collections import namedtuple
    outputs = namedtuple("Outputs", [
        "validation_passed", "f1_score", "auc_roc", "precision", 
        "recall", "inference_time_ms", "validation_report"
    ])
    return outputs(
        bool(all_passed), 
        float(f1), 
        float(auc), 
        float(prec), 
        float(rec), 
        float(p95_inference_time),
        validation_report
    )
