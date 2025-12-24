"""
HPO Training Script - Runs inside Ray cluster

Uses Ray Tune to optimize XGBoost hyperparameters for spam detection.
"""
import json
import os
import pickle
import time

import mlflow
import numpy as np
import pandas as pd
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_xgboost(config: dict, training_data: pd.DataFrame, feature_cols: list):
    """
    Train XGBoost with given hyperparameters.
    
    This function is called by Ray Tune for each trial.
    """
    # Prepare data
    X = training_data[feature_cols].values
    y = training_data["label"].values
    
    # Encode labels if they are strings
    if y.dtype == 'object' or y.dtype.name == 'string':
        # Map spam/ham to 1/0
        label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1}
        y = np.array([label_map.get(label, label) for label in y])
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Handle NaN/Inf
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train model
    model = XGBClassifier(
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        gamma=config["gamma"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        scale_pos_weight=config.get("scale_pos_weight", 1.0),
        random_state=42,
        n_jobs=2,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=10,
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    
    # Report metrics to Ray Train (replaces deprecated tune.report)
    train.report({
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
    })


def main():
    start_time = time.time()
    
    # Read environment variables
    features_path = os.environ.get("TRAINING_FEATURES_PATH")
    storage_account = os.environ.get("STORAGE_ACCOUNT")
    container_name = os.environ.get("CONTAINER_NAME", "feast")
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    num_trials = int(os.environ.get("NUM_TRIALS", "20"))
    max_concurrent_trials = int(os.environ.get("MAX_CONCURRENT_TRIALS", "4"))
    
    print(f"Training features path: {features_path}")
    print(f"Num trials: {num_trials}")
    print(f"Max concurrent trials: {max_concurrent_trials}")
    
    # Initialize Ray
    ray.init()
    print(f"Ray initialized. Resources: {ray.cluster_resources()}")
    
    # Initialize Azure filesystem
    import adlfs
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    fs = adlfs.AzureBlobFileSystem(connection_string=connection_string)
    
    # Read training data
    print(f"Reading training features from: {features_path}")
    with fs.open(features_path, "rb") as f:
        training_df = pd.read_parquet(f)
    
    print(f"Loaded {len(training_df)} samples with {len(training_df.columns)} columns")
    
    # Identify feature columns
    exclude_cols = ["email_id", "sender_domain", "event_timestamp", "label", 
                    "event_timestamp_feat", "created_timestamp"]
    
    # Filter to only numeric columns
    numeric_cols = training_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    print(f"Feature columns: {len(feature_cols)}")
    
    # Calculate class weight for imbalanced data
    spam_ratio = training_df["label"].mean()
    scale_pos_weight = (1 - spam_ratio) / spam_ratio if spam_ratio > 0 else 1.0
    print(f"Spam ratio: {spam_ratio:.2%}, scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Put training data in Ray object store
    training_data_ref = ray.put(training_df)
    feature_cols_ref = ray.put(feature_cols)
    
    # Define search space
    search_space = {
        "max_depth": tune.randint(3, 12),
        "learning_rate": tune.loguniform(0.01, 0.3),
        "n_estimators": tune.randint(50, 300),
        "min_child_weight": tune.randint(1, 10),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "gamma": tune.uniform(0, 1),
        "reg_alpha": tune.loguniform(1e-8, 10),
        "reg_lambda": tune.loguniform(1e-8, 10),
        "scale_pos_weight": scale_pos_weight,
    }
    
    # Define scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="f1_score",
        mode="max",
        max_t=1,
        grace_period=1,
        reduction_factor=2,
    )
    
    # Define trainable function
    def trainable(config):
        training_data = ray.get(training_data_ref)
        feature_cols = ray.get(feature_cols_ref)
        train_xgboost(config, training_data, feature_cols)
    
    # Run HPO
    print(f"\nStarting HPO with {num_trials} trials...")
    
    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=num_trials,
        scheduler=scheduler,
        resources_per_trial={"cpu": 2},
        max_concurrent_trials=max_concurrent_trials,
        verbose=1,
        raise_on_failed_trial=False,
    )
    
    # Get best trial
    best_trial = analysis.get_best_trial("f1_score", "max", "last")
    best_config = best_trial.config
    best_metrics = best_trial.last_result
    
    print(f"\n{'='*50}")
    print(f"Best trial config: {best_config}")
    print(f"Best trial F1: {best_metrics['f1_score']:.4f}")
    print(f"Best trial Precision: {best_metrics['precision']:.4f}")
    print(f"Best trial Recall: {best_metrics['recall']:.4f}")
    print(f"Best trial AUC-ROC: {best_metrics['auc_roc']:.4f}")
    print(f"{'='*50}")
    
    # Retrain best model on full training data
    print("\nRetraining best model on full training data...")
    
    X = training_df[feature_cols].values
    y = training_df["label"].values
    
    # Encode labels if they are strings
    if y.dtype == 'object' or y.dtype.name == 'string':
        print(f"Converting string labels to binary. Unique values: {np.unique(y)}")
        label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1}
        y = np.array([label_map.get(label, label) for label in y])
        print(f"After encoding: {np.unique(y)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    best_model = XGBClassifier(
        max_depth=best_config["max_depth"],
        learning_rate=best_config["learning_rate"],
        n_estimators=best_config["n_estimators"],
        min_child_weight=best_config["min_child_weight"],
        subsample=best_config["subsample"],
        colsample_bytree=best_config["colsample_bytree"],
        gamma=best_config["gamma"],
        reg_alpha=best_config["reg_alpha"],
        reg_lambda=best_config["reg_lambda"],
        scale_pos_weight=best_config.get("scale_pos_weight", 1.0),
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    
    best_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    
    # Final evaluation
    y_pred = best_model.predict(X_val_scaled)
    y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    
    final_f1 = f1_score(y_val, y_pred)
    final_precision = precision_score(y_val, y_pred)
    final_recall = recall_score(y_val, y_pred)
    final_auc_roc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\nFinal model metrics:")
    print(f"  F1: {final_f1:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall: {final_recall:.4f}")
    print(f"  AUC-ROC: {final_auc_roc:.4f}")
    
    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    
    with mlflow.start_run(run_name="hpo_best_model"):
        # Log best hyperparameters
        mlflow.log_params({k: v for k, v in best_config.items() if not callable(v)})
        
        # Log metrics
        mlflow.log_metric("f1_score", final_f1)
        mlflow.log_metric("precision", final_precision)
        mlflow.log_metric("recall", final_recall)
        mlflow.log_metric("auc_roc", final_auc_roc)
        mlflow.log_metric("num_trials", num_trials)
        mlflow.log_metric("trials_completed", len(analysis.trial_dataframes))
        
        # Log model (with error handling for WASBS URI issues)
        try:
            mlflow.xgboost.log_model(best_model, artifact_path="model")
            print("Successfully logged XGBoost model to MLflow")
        except Exception as e:
            print(f"Warning: Failed to log XGBoost model to MLflow: {e}")
        
        try:
            mlflow.sklearn.log_model(scaler, artifact_path="scaler")
            print("Successfully logged scaler to MLflow")
        except Exception as e:
            print(f"Warning: Failed to log scaler to MLflow: {e}")
        
        # Log trial results summary
        trial_results = []
        for trial in analysis.trials:
            if trial.last_result:
                trial_results.append({
                    "trial_id": trial.trial_id,
                    "f1_score": trial.last_result.get("f1_score"),
                    "precision": trial.last_result.get("precision"),
                    "recall": trial.last_result.get("recall"),
                })
        try:
            mlflow.log_dict(trial_results, "trial_results.json")
            print("Successfully logged trial results to MLflow")
        except Exception as e:
            print(f"Warning: Failed to log trial results to MLflow: {e}")
    
    # Save model and results to blob storage
    model_path = f"{container_name}/hpo/best_model.pkl"
    scaler_path = f"{container_name}/hpo/scaler.pkl"
    results_path = f"{container_name}/hpo/results.json"
    
    with fs.open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    
    with fs.open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    # Convert config values to serializable types
    serializable_config = {}
    for k, v in best_config.items():
        if isinstance(v, (int, float, str, bool)):
            serializable_config[k] = v
        elif isinstance(v, np.integer):
            serializable_config[k] = int(v)
        elif isinstance(v, np.floating):
            serializable_config[k] = float(v)
        else:
            serializable_config[k] = str(v)
    
    results = {
        "best_f1": float(final_f1),
        "best_precision": float(final_precision),
        "best_recall": float(final_recall),
        "best_auc_roc": float(final_auc_roc),
        "best_params": serializable_config,
        "trials_completed": len(analysis.trial_dataframes),
        "total_time_seconds": time.time() - start_time,
    }
    
    with fs.open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved best model to: {model_path}")
    print(f"Saved results to: {results_path}")
    print(f"Total HPO time: {results['total_time_seconds']:.2f} seconds")
    
    ray.shutdown()


if __name__ == "__main__":
    main()
