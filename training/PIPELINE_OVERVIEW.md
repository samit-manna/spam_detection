# Spam Detection Training Pipeline - Architecture Overview

## Pipeline Summary

This Kubeflow pipeline orchestrates an end-to-end machine learning workflow for spam email detection, integrating Feast for feature management, Ray Tune for hyperparameter optimization, and MLflow for experiment tracking and model registry.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Spam Detection Training Pipeline                     │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  1. Data Preparation │
    │  (data_preparation)  │
    └──────────┬───────────┘
               │ Outputs: train_entity_path, test_entity_path
               │ (email_id, sender_domain, label, event_timestamp)
               ▼
    ┌──────────────────────┐
    │ 2. Feature Retrieval │
    │  (feature_retrieval) │
    │   Feast Server       │
    └──────────┬───────────┘
               │ Outputs: train_features_path, test_features_path
               │ (524 features per email)
               │
               ├─────────────────────┐
               ▼                     ▼
    ┌──────────────────────┐   ┌────────────────────────┐
    │ 3. Baseline Training │   │ (Used for final eval)  │
    │  (baseline_training) │   │                        │
    │  Logistic Regression │   │                        │
    │  or Random Forest    │   │                        │
    └──────────┬───────────┘   │                        │
               │                │                        │
               ▼                │                        │
    ┌──────────────────────┐   │                        │
    │   4. HPO Tuning      │   │                        │
    │   (hpo_tuning)       │   │                        │
    │   Ray Tune + XGBoost │   │                        │
    └──────────┬───────────┘   │                        │
               │ Output: best_model_path                 │
               │                                         │
               └────────────┬────────────────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │ 5. Model Evaluation  │
                 │  (model_evaluation)  │
                 │  If F1 >= threshold: │
                 │  → Register to MLflow│
                 └──────────────────────┘
```

---

## Component Details

### 1. Data Preparation
**Component:** `data_preparation`  
**Image:** `ml-data-prep:v0.5`

**Purpose:** Prepares entity dataframes for feature retrieval

**Inputs:**
- `entity_data_path`: Path to processed email parquet file in Azure Blob Storage
- `storage_account`: Azure storage account name
- `container_name`: Blob container name (e.g., "datasets")
- `test_split_ratio`: Train/test split ratio (default: 0.2)

**Processing:**
- Reads `all_emails.parquet` from Azure Blob Storage
- Extracts entity columns: `email_id`, `sender_domain`, `event_timestamp`, `label`
- Encodes labels: `ham` → 0, `spam` → 1
- Performs stratified train/test split (preserves spam ratio)
- Validates data quality (checks for nulls, ensures binary labels)

**Outputs:**
- `train_entity_path`: Path to training entities parquet
- `test_entity_path`: Path to test entities parquet
- `train_count`: Number of training samples
- `test_count`: Number of test samples
- `spam_ratio`: Spam percentage in dataset

**Metrics Logged:**
- Train/test sample counts
- Spam ratio
- Presence of sender_domain column

---

### 2. Feature Retrieval
**Component:** `feature_retrieval`  
**Image:** `ml-data-prep:v0.5`

**Purpose:** Retrieves 524 features from Feast feature store

**Inputs:**
- `train_entity_path`: Path to training entities (from Step 1)
- `test_entity_path`: Path to test entities (from Step 1)
- `feast_server_url`: Feast server gRPC endpoint
- `include_sender_features`: Include sender domain features

**Feature Views Retrieved:**
1. **email_text_features** (8 features)
   - URL count, uppercase ratio, spam keywords, exclamation/question marks
   - Word count, char count, has_unsubscribe flag

2. **email_structural_features** (8 features)
   - HTML indicators, subject line analysis, sender domain length
   - Received hop count (email routing hops)

3. **email_temporal_features** (4 features)
   - Hour of day, day of week, is_weekend, is_night_hour

4. **email_tfidf_features** (500 features)
   - TF-IDF vectors for email content (top 500 terms)

5. **sender_domain_features** (4 features) *[optional]*
   - Email count, spam/ham counts, spam ratio per domain

**Processing:**
- Connects to Feast server via gRPC
- Calls `get_historical_features()` for each feature view
- Handles missing values (fills with 0)
- Joins features with entity dataframes

**Outputs:**
- `train_features_path`: Training features with 524 columns
- `test_features_path`: Test features with 524 columns
- `feature_count`: Total features retrieved
- `train_samples`, `test_samples`: Sample counts

**Metrics Logged:**
- Feature count breakdown
- Sample counts for train/test

---

### 3. Baseline Training
**Component:** `baseline_training`  
**Image:** `ml-mlflow-ops:v0.1`

**Purpose:** Quick validation with simple model before expensive HPO

**Inputs:**
- `train_features_path`: Training features (from Step 2)
- `model_type`: "logistic_regression" or "random_forest"
- `mlflow_tracking_uri`: MLflow server URL
- `mlflow_experiment_name`: Experiment name

**Processing:**
- Loads training features from Azure Blob
- Splits into train/validation (80/20)
- Applies StandardScaler for feature normalization
- Trains either:
  - **Logistic Regression**: max_iter=1000, solver='saga', class_weight='balanced'
  - **Random Forest**: n_estimators=100, max_depth=15, class_weight='balanced'
- Logs experiment to MLflow (parameters, metrics, artifacts)

**Outputs:**
- `baseline_model_path`: Path to saved model pickle
- `f1_score`, `precision`, `recall`, `auc_roc`: Validation metrics

**Artifacts Logged to MLflow:**
- Trained model (sklearn)
- Scaler (StandardScaler)
- Classification report
- Feature importance (for tree-based models)

**Why Baseline?**
- Validates pipeline works end-to-end
- Establishes performance floor
- Faster than HPO (completes in minutes vs hours)

---

### 4. HPO Tuning
**Component:** `hpo_tuning`  
**Image:** `ml-data-prep:v0.5`

**Purpose:** Distributed hyperparameter optimization using Ray Tune

**Inputs:**
- `train_features_path`: Training features (from Step 2)
- `num_trials`: Number of hyperparameter combinations to try (default: 20)
- `max_concurrent_trials`: Parallel trials (default: 4)
- `mlflow_tracking_uri`, `mlflow_experiment_name`: MLflow config

**Processing:**
- Creates a RayJob CRD (Custom Resource Definition) on Kubernetes
- Deploys Ray cluster with:
  - **Head node**: 2 CPU, 4Gi memory
  - **Worker nodes**: 4 replicas (2-6 autoscaling), 4 CPU, 8Gi memory each
- Executes `/app/scripts/hpo_train.py` on Ray cluster
- Ray Tune explores hyperparameter space for XGBoost:
  - Learning rate: [0.01, 0.3]
  - Max depth: [3, 10]
  - Min child weight: [1, 10]
  - Subsample: [0.6, 1.0]
  - Colsample_bytree: [0.6, 1.0]
  - Gamma: [0, 5]
- Uses ASHA scheduler for early stopping
- Logs all trials to MLflow

**Outputs:**
- `best_model_path`: Path to best XGBoost model pickle
- `best_f1`: Best F1 score achieved
- `best_params`: JSON string of optimal hyperparameters

**Results Written to Blob:**
- `models/hpo/results.json`: Summary of best trial
- `models/hpo/best_model.pkl`: Trained XGBoost model
- `models/hpo/scaler.pkl`: StandardScaler

**Wait Time:** Up to 2 hours (polls every 60s)

---

### 5. Model Evaluation & Registration
**Component:** `model_evaluation`  
**Image:** `ml-mlflow-ops:v0.1`

**Purpose:** Evaluate best model on holdout test set and register if threshold met

**Inputs:**
- `best_model_path`: Best model from HPO (from Step 4)
- `test_features_path`: Test features (from Step 2)
- `f1_threshold`: Minimum F1 score for registration (default: 0.85)
- `model_name`: Name in MLflow Model Registry (default: "spam-detector")

**Processing:**
- Loads best model and scaler from Azure Blob
- Loads test features (held out from entire pipeline)
- Scales test features with same scaler
- Generates predictions and probabilities
- Calculates comprehensive metrics:
  - Accuracy, Precision, Recall, F1, AUC-ROC
  - Confusion matrix
  - Classification report

**Conditional Registration:**
```python
if test_f1 >= f1_threshold:
    # Register model to MLflow Model Registry
    # Set stage to "Staging"
    # Tag with test metrics
else:
    # Skip registration (model not good enough)
```

**Outputs:**
- `test_f1`: F1 score on test set
- `registered`: Boolean (True if registered)
- `model_version`: MLflow model version (e.g., "1", "2")

**Artifacts Logged to MLflow:**
- Confusion matrix plot
- ROC curve plot
- Precision-Recall curve plot
- Classification report (text)
- Registered model (if threshold met)

**Model Tags (if registered):**
- `test_f1`, `test_auc_roc`, `test_precision`, `test_recall`
- Stage: "Staging" (ready for further validation)

---

## Data Flow Summary

| Step | Input Data | Processing | Output Data |
|------|-----------|------------|-------------|
| 1. Data Prep | `all_emails.parquet` | Train/test split | `train_entities.parquet`, `test_entities.parquet` |
| 2. Feature Retrieval | Entity parquets | Feast feature lookup | `train_features.parquet` (524 cols), `test_features.parquet` |
| 3. Baseline | Train features | Simple model training | `baseline_model.pkl`, validation metrics |
| 4. HPO | Train features | Ray Tune XGBoost optimization | `best_model.pkl`, best hyperparameters |
| 5. Evaluation | Best model + test features | Test set evaluation | Test metrics, MLflow registration |

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | Kubeflow Pipelines 2.14.0 | Workflow management, DAG execution |
| **Feature Store** | Feast 0.36.0+ | Feature management, versioning, retrieval |
| **HPO Framework** | Ray Tune 2.9.0 | Distributed hyperparameter search |
| **Experiment Tracking** | MLflow 2.x | Metrics logging, model registry |
| **Storage** | Azure Blob Storage | Parquet files, model artifacts |
| **Secrets Management** | Kubernetes Secrets | Azure connection string injection |
| **Container Registry** | Azure Container Registry | Docker images (ml-data-prep, ml-mlflow-ops, ml-ray-train) |

---

## Pipeline Execution

### How to Run
```bash
# Submit pipeline to Kubeflow
kfp run create \
  --experiment-name spam-detection \
  --pipeline-file spam_detection_pipeline.yaml
```

### Expected Runtime
- **Data Preparation**: 1-2 minutes
- **Feature Retrieval**: 3-5 minutes (depends on Feast server)
- **Baseline Training**: 5-10 minutes
- **HPO Tuning**: 30-120 minutes (depends on num_trials, cluster resources)
- **Model Evaluation**: 2-5 minutes
- **Total**: ~45-140 minutes

### Resource Requirements
- **Data Prep/Feature Retrieval**: 2 CPU, 4Gi memory
- **Baseline Training**: 2 CPU, 4Gi memory
- **HPO (Ray Cluster)**: 1 head + 4 workers = 18 CPU, 36Gi memory total
- **Model Evaluation**: 2 CPU, 4Gi memory

---

## Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entity_data_path` | `processed/emails/all_emails.parquet` | Input email data path |
| `storage_account` | `mltrainingsdevsa8y3dy2` | Azure storage account |
| `container_name` | `datasets` | Blob container |
| `test_split_ratio` | `0.2` | Test set size (20%) |
| `feast_server_url` | `feast-service.feast.svc.cluster.local:6566` | Feast gRPC endpoint |
| `include_sender_features` | `True` | Include sender domain features |
| `mlflow_tracking_uri` | `http://mlflow-service.mlflow.svc.cluster.local:5000` | MLflow server |
| `mlflow_experiment_name` | `spam-detection` | Experiment name |
| `baseline_model_type` | `logistic_regression` | Baseline model type |
| `num_hpo_trials` | `20` | Number of HPO trials |
| `max_concurrent_trials` | `4` | Concurrent HPO trials |
| `model_name` | `spam-detector` | Model registry name |
| `f1_threshold` | `0.85` | Min F1 for registration |

---

## Output Artifacts

### Azure Blob Storage (`datasets` container)
```
datasets/
├── training/
│   ├── train_entities.parquet      # Step 1 output
│   ├── test_entities.parquet       # Step 1 output
│   ├── train_features.parquet      # Step 2 output
│   └── test_features.parquet       # Step 2 output
└── models/
    ├── baseline/
    │   ├── logistic_regression/
    │   │   ├── model.pkl
    │   │   └── scaler.pkl
    │   └── random_forest/
    │       ├── model.pkl
    │       └── scaler.pkl
    └── hpo/
        ├── results.json            # Step 4 output
        ├── best_model.pkl          # Step 4 output
        └── scaler.pkl              # Step 4 output
```

### MLflow Artifacts
- **Experiments**: All runs logged under `spam-detection` experiment
- **Model Registry**: `spam-detector` model (if F1 >= threshold)
- **Artifacts per run**:
  - Trained models (sklearn, xgboost)
  - Scalers (StandardScaler)
  - Plots (confusion matrix, ROC curve, PR curve)
  - Reports (classification_report.txt)

---

## Key Design Decisions

### 1. Why Separate Baseline and HPO Steps?
- **Fast Validation**: Baseline completes in minutes, catches data/config issues early
- **Cost Efficiency**: Avoid expensive Ray cluster deployment if baseline fails
- **Performance Floor**: Establishes minimum expected performance

### 2. Why Use Azure Blob for Intermediate Data?
- **Persistence**: Artifacts survive pod restarts
- **Debugging**: Inspect intermediate outputs (parquet files) manually
- **Reproducibility**: Same data accessible across pipeline runs
- **Scalability**: Better than ephemeral volumes for large datasets

### 3. Why Ray Tune for HPO?
- **Distributed**: Trials run in parallel across worker nodes
- **Efficient**: ASHA scheduler stops unpromising trials early
- **Scalable**: Autoscaling workers (2-6 replicas) based on demand
- **Kubernetes-native**: RayJob CRD integrates seamlessly with Kubeflow

### 4. Why Feast for Features?
- **Consistency**: Same features in training and production
- **Versioning**: Feature definitions tracked as code
- **Reusability**: Features computed once, shared across experiments
- **Point-in-time correctness**: Avoids data leakage

### 5. Why Conditional Registration?
- **Quality Control**: Only production-ready models reach registry
- **Manual Override**: Threshold can be adjusted per use case
- **Staging Environment**: Registered models start in "Staging" stage

---

## Troubleshooting

### Common Issues

**1. AZURE_STORAGE_CONNECTION_STRING not set**
- **Cause**: `azure-storage-secret` not created or not injected
- **Fix**: Ensure secret exists in `kubeflow` namespace and pipeline uses `kubernetes.use_secret_as_env()`

**2. Feast server connection timeout**
- **Cause**: Feast service not running or wrong URL
- **Fix**: Verify Feast deployment: `kubectl get svc -n feast`

**3. Pandas version conflict**
- **Cause**: Docker image has pandas < 2.0 (incompatible with Dask)
- **Fix**: Rebuild images with `pandas>=2.0.1`

**4. HPO RayJob timeout**
- **Cause**: Insufficient resources or too many trials
- **Fix**: Reduce `num_trials`, increase `max_wait_time`, or add more worker nodes

**5. Model not registered**
- **Cause**: F1 score below threshold
- **Fix**: Check MLflow logs, adjust `f1_threshold`, or retrain with more data/trials

---

## Next Steps

After successful pipeline execution:

1. **Review Results in MLflow**
   ```bash
   kubectl port-forward svc/mlflow-service -n mlflow 5000:5000
   # Open http://localhost:5000
   ```

2. **Promote Model to Production**
   ```python
   from mlflow.tracking import MlflowClient
   client = MlflowClient("http://mlflow-service.mlflow.svc.cluster.local:5000")
   client.transition_model_version_stage(
       name="spam-detector",
       version="1",
       stage="Production"
   )
   ```

3. **Deploy Model for Inference**
   - Create inference service using MLflow model URI
   - Example: Seldon Core, KServe, or custom Flask API

4. **Monitor Model Performance**
   - Track production metrics (precision, recall, latency)
   - Set up alerts for data drift
   - Retrain periodically with new data

---

## References

- **Pipeline Code**: `ml-training-pipeline-2/pipeline/spam_detection_pipeline.py`
- **Component Definitions**: `ml-training-pipeline-2/pipeline/components/`
- **Docker Images**: `ml-training-pipeline-2/docker/`
- **Feast Features**: `phase1/feast/feature_repo/`
