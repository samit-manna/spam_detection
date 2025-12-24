"""
Batch Prediction Job using Ray

This script performs batch inference on email data stored in Azure Blob Storage.
It uses Ray for distributed processing and MLflow for model loading.

Features:
- Loads model from MLflow registry
- Reads input data from Azure Blob (parquet format)
- Uses ray.data for parallel processing
- Applies same feature transformation as real-time inference (including StandardScaler)
- Writes predictions back to Azure Blob

Usage:
    python batch_predict.py \
        --input-path datasets/batch/emails.parquet \
        --output-path predictions/batch_output.parquet \
        --model-name spam-detector \
        --model-stage Production

Environment variables:
    MLFLOW_TRACKING_URI: MLflow tracking server URL
    AZURE_STORAGE_ACCOUNT_NAME: Azure storage account name
    AZURE_STORAGE_ACCOUNT_KEY: Azure storage account key
"""

import os
import sys
import re
import json
import logging
import argparse
import pickle
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import ray
from ray import data as ray_data
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from adlfs import AzureBlobFileSystem
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-service.mlflow.svc.cluster.local:5000"
)
AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "datasets")
FEAST_CONTAINER = os.environ.get("FEAST_CONTAINER", "feast")
MODELS_CONTAINER = os.environ.get("MODELS_CONTAINER", "models")

# Batch processing settings
DEFAULT_BATCH_SIZE = 1000
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

# Classification threshold - must match real-time inference
SPAM_THRESHOLD = 0.7

# Spam keywords for feature extraction
SPAM_KEYWORDS = [
    "free", "winner", "click here", "act now", "limited time",
    "congratulations", "urgent", "money", "cash", "prize",
    "viagra", "pharmacy", "buy now", "order now", "subscribe",
    "unsubscribe", "remove", "opt-out", "credit card", "wire transfer"
]


# =============================================================================
# Azure Storage Helpers
# =============================================================================

def get_azure_fs() -> AzureBlobFileSystem:
    """Create Azure Blob FileSystem client."""
    if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
        raise ValueError("Azure storage credentials not set")
    
    return AzureBlobFileSystem(
        account_name=AZURE_STORAGE_ACCOUNT_NAME,
        account_key=AZURE_STORAGE_ACCOUNT_KEY
    )


def load_tfidf_vectorizer(fs: AzureBlobFileSystem):
    """Load TF-IDF vectorizer from Azure Blob Storage."""
    vectorizer_path = f"{FEAST_CONTAINER}/features/artifacts/tfidf_vectorizer.pkl"
    logger.info(f"Loading TF-IDF vectorizer from {vectorizer_path}")
    
    with fs.open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    logger.info(f"Loaded TF-IDF vectorizer with {len(vectorizer.vocabulary_)} features")
    return vectorizer


def load_scaler(fs: AzureBlobFileSystem):
    """Load StandardScaler from Azure Blob Storage - CRITICAL for correct predictions."""
    scaler_path = f"{MODELS_CONTAINER}/hpo/scaler.pkl"
    logger.info(f"Loading StandardScaler from {scaler_path}")
    
    with fs.open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    logger.info(f"Loaded StandardScaler expecting {scaler.n_features_in_} features")
    return scaler


# =============================================================================
# Feature Extraction Functions (must match real-time inference exactly)
# =============================================================================

def count_urls(text: str) -> int:
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return len(re.findall(url_pattern, text or ""))


def count_emails_in_text(text: str) -> int:
    """Count email addresses in text body."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return len(re.findall(email_pattern, text or ""))


def uppercase_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)


def count_exclamations(text: str) -> int:
    return (text or "").count("!")


def count_question_marks(text: str) -> int:
    return (text or "").count("?")


def avg_word_length(text: str) -> float:
    """Calculate average word length."""
    words = (text or "").split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def word_count(text: str) -> int:
    return len((text or "").split())


def char_count(text: str) -> int:
    return len(text or "")


def spam_keyword_count(text: str) -> int:
    text_lower = (text or "").lower()
    return sum(1 for kw in SPAM_KEYWORDS if kw in text_lower)


def has_unsubscribe(text: str) -> int:
    return 1 if "unsubscribe" in (text or "").lower() else 0


def has_html(html_body: Optional[str]) -> int:
    return 1 if html_body else 0


def html_to_text_ratio(text_body: str, html_body: Optional[str]) -> float:
    text_len = len(text_body or "")
    html_len = len(html_body or "")
    if text_len == 0:
        return float(html_len) if html_len > 0 else 0.0
    return html_len / text_len


def subject_length(subject: str) -> int:
    return len(subject or "")


def subject_has_re(subject: str) -> int:
    return 1 if (subject or "").lower().startswith("re:") else 0


def subject_has_fwd(subject: str) -> int:
    s = (subject or "").lower()
    return 1 if s.startswith("fwd:") or s.startswith("fw:") else 0


def subject_all_caps(subject: str) -> int:
    """Check if subject is all caps."""
    if not subject:
        return 0
    alpha_chars = [c for c in subject if c.isalpha()]
    if not alpha_chars:
        return 0
    return 1 if all(c.isupper() for c in alpha_chars) else 0


def has_x_mailer_header(x_mailer: Optional[str]) -> int:
    return 1 if x_mailer else 0


def sender_domain_length(sender_email: str) -> int:
    if "@" in (sender_email or ""):
        return len(sender_email.split("@")[-1])
    return 0


def sender_has_numbers(sender_email: str) -> int:
    """Check if sender email has numbers."""
    return 1 if any(c.isdigit() for c in (sender_email or "")) else 0


def extract_hour(date_str: Optional[str]) -> int:
    if not date_str:
        return 12
    try:
        dt = pd.to_datetime(date_str)
        return dt.hour
    except Exception:
        return 12


def extract_day_of_week(date_str: Optional[str]) -> int:
    if not date_str:
        return 2
    try:
        dt = pd.to_datetime(date_str)
        return dt.weekday()
    except Exception:
        return 2


def is_weekend(date_str: Optional[str]) -> int:
    if not date_str:
        return 0
    try:
        dt = pd.to_datetime(date_str)
        return 1 if dt.weekday() >= 5 else 0
    except Exception:
        return 0


def is_night_hour(date_str: Optional[str]) -> int:
    if not date_str:
        return 0
    try:
        dt = pd.to_datetime(date_str)
        hour = dt.hour
        return 1 if hour >= 22 or hour < 6 else 0
    except Exception:
        return 0


# =============================================================================
# Batch Feature Extraction
# =============================================================================

@dataclass
class BatchPredictor:
    """Ray Actor for batch prediction."""
    
    model: xgb.Booster
    vectorizer: Any
    scaler: Any
    model_version: str
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract all features for a batch of emails.
        Feature order MUST match training pipeline (extract_features_batch in data-pipeline).
        
        Order: Text(8) -> Structural(10) -> Temporal(4) -> SpamIndicators(2) -> TF-IDF(500) -> Sender(4)
        Total: 528 features
        """
        
        # === Text features (8) ===
        text_features = pd.DataFrame()
        text_features["url_count"] = df["body_text"].apply(count_urls)
        text_features["email_count_in_body"] = df["body_text"].apply(count_emails_in_text)
        text_features["uppercase_ratio"] = df["body_text"].apply(uppercase_ratio)
        text_features["exclamation_count"] = df["body_text"].apply(count_exclamations)
        text_features["question_mark_count"] = df["body_text"].apply(count_question_marks)
        text_features["avg_word_length"] = df["body_text"].apply(avg_word_length)
        text_features["word_count"] = df["body_text"].apply(word_count)
        text_features["char_count"] = df["body_text"].apply(char_count)
        
        # === Structural features (10) ===
        structural_features = pd.DataFrame()
        structural_features["has_html"] = df["body_html"].apply(has_html) if "body_html" in df.columns else 0
        structural_features["html_to_text_ratio"] = df.apply(
            lambda r: html_to_text_ratio(r.get("body_text", ""), r.get("body_html")), axis=1
        ) if "body_html" in df.columns else 0
        structural_features["subject_length"] = df["subject"].apply(subject_length) if "subject" in df.columns else 0
        structural_features["subject_has_re"] = df["subject"].apply(subject_has_re) if "subject" in df.columns else 0
        structural_features["subject_has_fwd"] = df["subject"].apply(subject_has_fwd) if "subject" in df.columns else 0
        structural_features["subject_all_caps"] = df["subject"].apply(subject_all_caps) if "subject" in df.columns else 0
        structural_features["has_x_mailer"] = df["x_mailer"].apply(has_x_mailer_header) if "x_mailer" in df.columns else 0
        structural_features["sender_domain_length"] = df["sender_email"].apply(sender_domain_length) if "sender_email" in df.columns else 0
        structural_features["sender_has_numbers"] = df["sender_email"].apply(sender_has_numbers) if "sender_email" in df.columns else 0
        structural_features["received_hop_count"] = df["received_hop_count"] if "received_hop_count" in df.columns else 0
        
        # === Temporal features (4) ===
        date_col = "date" if "date" in df.columns else None
        temporal_features = pd.DataFrame()
        temporal_features["hour_of_day"] = df[date_col].apply(extract_hour) if date_col else 12
        temporal_features["day_of_week"] = df[date_col].apply(extract_day_of_week) if date_col else 2
        temporal_features["is_weekend"] = df[date_col].apply(is_weekend) if date_col else 0
        temporal_features["is_night_hour"] = df[date_col].apply(is_night_hour) if date_col else 0
        
        # === Spam indicator features (2) ===
        spam_indicator_features = pd.DataFrame()
        spam_indicator_features["spam_keyword_count"] = df["body_text"].apply(spam_keyword_count)
        spam_indicator_features["has_unsubscribe"] = df["body_text"].apply(has_unsubscribe)
        
        # === TF-IDF features (500) ===
        texts = df["body_text"].fillna("").tolist()
        tfidf_matrix = self.vectorizer.transform(texts).toarray()
        
        # === Sender domain features (4) ===
        n = len(df)
        sender_features = pd.DataFrame(index=range(n))
        sender_features["email_count"] = df["email_count"].values if "email_count" in df.columns else np.zeros(n)
        sender_features["spam_count"] = df["spam_count"].values if "spam_count" in df.columns else np.zeros(n)
        sender_features["ham_count"] = df["ham_count"].values if "ham_count" in df.columns else np.zeros(n)
        sender_features["spam_ratio"] = df["spam_ratio"].values if "spam_ratio" in df.columns else np.full(n, 0.5)
        
        # Combine all features in EXACT order: Text(8) + Structural(10) + Temporal(4) + SpamIndicators(2) + TF-IDF(500) + Sender(4)
        full_features = np.hstack([
            text_features.values,           # 8
            structural_features.values,     # 10
            temporal_features.values,       # 4
            spam_indicator_features.values, # 2
            tfidf_matrix,                   # 500
            sender_features.values          # 4
        ])  # Total: 528
        
        # Apply StandardScaler (CRITICAL - model was trained on scaled features)
        full_features = np.nan_to_num(full_features, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_features = self.scaler.transform(full_features)
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return scaled_features.astype(np.float32)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on a batch of emails."""
        
        # Extract and scale features
        features = self.extract_features(df)
        
        # Create DMatrix for XGBoost
        dmatrix = xgb.DMatrix(features)
        
        # Get predictions (probabilities)
        probabilities = self.model.predict(dmatrix)
        
        # Apply threshold (must match real-time inference)
        predictions = (probabilities >= SPAM_THRESHOLD).astype(int)
        
        # Add prediction columns
        df = df.copy()
        df["spam_probability"] = probabilities
        df["spam_label"] = predictions
        df["prediction_label"] = df["spam_label"].apply(lambda x: "spam" if x == 1 else "ham")
        df["prediction_timestamp"] = datetime.utcnow().isoformat()
        df["model_version"] = self.model_version
        
        return df


class BatchPredictorActor:
    """Ray Actor wrapper for BatchPredictor."""
    
    def __init__(
        self,
        model_bytes: bytes,
        vectorizer_bytes: bytes,
        scaler_bytes: bytes,
        model_version: str
    ):
        # Load model from bytes (avoiding MLflow's DefaultAzureCredential issue)
        self.model = pickle.loads(model_bytes)
        
        # Load vectorizer
        self.vectorizer = pickle.loads(vectorizer_bytes)
        
        # Load scaler (CRITICAL for correct predictions)
        self.scaler = pickle.loads(scaler_bytes)
        
        self.model_version = model_version
        
        self.predictor = BatchPredictor(
            model=self.model,
            vectorizer=self.vectorizer,
            scaler=self.scaler,
            model_version=model_version
        )
    
    def predict_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.predict(batch)


# =============================================================================
# Main Batch Processing
# =============================================================================

def load_model_from_mlflow(
    model_name: str,
    model_stage: str,
    fs: AzureBlobFileSystem
) -> tuple[bytes, str, Dict[str, Any]]:
    """
    Load model from MLflow and return as bytes.
    
    This downloads the model directly from Azure Blob Storage using
    the storage account key to avoid DefaultAzureCredential issues.
    """
    
    logger.info(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Get model version
    versions = client.get_latest_versions(model_name, stages=[model_stage])
    if not versions:
        raise ValueError(f"No model found for {model_name} with stage {model_stage}")
    
    mv = versions[0]
    
    # Get run info
    run = client.get_run(mv.run_id)
    
    metadata = {
        "model_name": model_name,
        "model_version": mv.version,
        "model_stage": mv.current_stage,
        "run_id": mv.run_id,
        "metrics": run.data.metrics,
    }
    
    logger.info(f"Found model version {mv.version} from run {mv.run_id}")
    
    # Get the artifact URI and extract the blob path
    # MLflow artifact URI format: wasbs://container@account.blob.core.windows.net/path
    artifact_uri = run.info.artifact_uri
    logger.info(f"Artifact URI: {artifact_uri}")
    
    # Parse the artifact URI to get blob path
    # Format: wasbs://mlflow@account.blob.core.windows.net/experiment_id/run_id/artifacts
    if artifact_uri.startswith("wasbs://"):
        # Extract path from wasbs://container@account.blob.core.windows.net/path
        parts = artifact_uri.replace("wasbs://", "").split("/", 1)
        container_and_account = parts[0]  # container@account.blob.core.windows.net
        container = container_and_account.split("@")[0]
        blob_path = parts[1] if len(parts) > 1 else ""
    else:
        raise ValueError(f"Unsupported artifact URI format: {artifact_uri}")
    
    # The model is stored under artifacts/model/
    model_blob_path = f"{container}/{blob_path}/model/model.xgb"
    logger.info(f"Loading model from blob: {model_blob_path}")
    
    # Download model file directly using our authenticated filesystem
    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_path = os.path.join(tmpdir, "model.xgb")
        
        with fs.open(model_blob_path, "rb") as f_remote:
            model_data = f_remote.read()
            with open(local_model_path, "wb") as f_local:
                f_local.write(model_data)
        
        logger.info(f"Downloaded model file: {len(model_data)} bytes")
        
        # Load as XGBoost Booster
        model = xgb.Booster()
        model.load_model(local_model_path)
        
        # Serialize to bytes for distribution to workers
        model_bytes = pickle.dumps(model)
        logger.info(f"Model serialized: {len(model_bytes)} bytes")
    
    return model_bytes, str(mv.version), metadata


def run_batch_prediction(
    input_path: str,
    output_path: str,
    model_name: str,
    model_stage: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Dict[str, Any]:
    """
    Run batch prediction on input data.
    
    Args:
        input_path: Azure Blob path to input parquet file
        output_path: Azure Blob path for output parquet file
        model_name: MLflow model name
        model_stage: Model stage (Staging, Production)
        batch_size: Batch size for processing
        num_workers: Number of Ray workers
        
    Returns:
        Dictionary with job statistics
    """
    logger.info("=" * 60)
    logger.info("Starting Batch Prediction Job")
    logger.info("=" * 60)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address="auto")
    logger.info(f"Connected to Ray cluster: {ray.cluster_resources()}")
    
    # Get Azure filesystem
    fs = get_azure_fs()
    
    # Load model from MLflow (returns bytes to avoid Azure auth issues in workers)
    model_bytes, model_version, model_metadata = load_model_from_mlflow(
        model_name, model_stage, fs
    )
    
    # Load TF-IDF vectorizer
    vectorizer = load_tfidf_vectorizer(fs)
    vectorizer_bytes = pickle.dumps(vectorizer)
    
    # Load StandardScaler (CRITICAL - model was trained on scaled features)
    scaler = load_scaler(fs)
    scaler_bytes = pickle.dumps(scaler)
    
    # Read input data
    full_input_path = f"{AZURE_STORAGE_CONTAINER}/{input_path}"
    logger.info(f"Reading input data from {full_input_path}")
    
    with fs.open(full_input_path, "rb") as f:
        input_df = pd.read_parquet(f)
    
    total_records = len(input_df)
    logger.info(f"Loaded {total_records} records for prediction")
    
    # Create Ray dataset
    ray_dataset = ray_data.from_pandas(input_df)
    
    # Create prediction actor with serialized model (avoids Azure auth issues)
    BatchPredictorActorRemote = ray.remote(BatchPredictorActor)
    predictor = BatchPredictorActorRemote.remote(
        model_bytes=model_bytes,
        vectorizer_bytes=vectorizer_bytes,
        scaler_bytes=scaler_bytes,
        model_version=model_version
    )
    
    # Process in batches
    logger.info(f"Processing {total_records} records in batches of {batch_size}")
    
    results = []
    num_batches = (total_records + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_records)
        batch_df = input_df.iloc[start_idx:end_idx]
        
        # Submit batch for prediction
        result_ref = predictor.predict_batch.remote(batch_df)
        results.append(result_ref)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Submitted batch {i + 1}/{num_batches}")
    
    # Collect results
    logger.info("Collecting prediction results...")
    predicted_batches = ray.get(results)
    output_df = pd.concat(predicted_batches, ignore_index=True)
    
    # Calculate statistics
    spam_count = (output_df["spam_label"] == 1).sum()
    ham_count = (output_df["spam_label"] == 0).sum()
    avg_probability = output_df["spam_probability"].mean()
    
    logger.info(f"Predictions complete: {spam_count} spam, {ham_count} ham")
    logger.info(f"Average spam probability: {avg_probability:.4f}")
    
    # Write output
    full_output_path = f"{AZURE_STORAGE_CONTAINER}/{output_path}"
    logger.info(f"Writing output to {full_output_path}")
    
    table = pa.Table.from_pandas(output_df, preserve_index=False)
    with fs.open(full_output_path, "wb") as f:
        pq.write_table(table, f)
    
    # Job statistics
    stats = {
        "status": "success",
        "total_records": total_records,
        "spam_count": int(spam_count),
        "ham_count": int(ham_count),
        "avg_spam_probability": float(avg_probability),
        "model_name": model_name,
        "model_version": model_version,
        "model_stage": model_stage,
        "input_path": input_path,
        "output_path": output_path,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info("=" * 60)
    logger.info("Batch Prediction Job Complete")
    logger.info(json.dumps(stats, indent=2))
    logger.info("=" * 60)
    
    return stats


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch prediction on email data"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Azure Blob path to input parquet file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Azure Blob path for output parquet file"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="spam-detector",
        help="MLflow model name"
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default="Production",
        choices=["Staging", "Production"],
        help="Model stage"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of Ray workers"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    result = run_batch_prediction(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
        model_stage=args.model_stage,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    sys.exit(0 if result["status"] == "success" else 1)
