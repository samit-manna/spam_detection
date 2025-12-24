"""
Training Baseline Generator

Generates baseline statistics and distributions from training data
for use in drift detection. Run this after model training to create
a reference baseline.

Usage:
    python -m monitoring.baseline.generator \
        --training-data "abfss://data@store.dfs.core.windows.net/training/train.parquet" \
        --model-name "spam-detector" \
        --model-version "3" \
        --output-path "abfss://data@store.dfs.core.windows.net/baselines/"
"""

import os
import sys
import json
import logging
import argparse
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import tempfile

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.identity import DefaultAzureCredential

from .schema import (
    BaselineData,
    BaselineMetadata,
    FeatureStatistics,
    DistributionBin,
    HistogramBin,
    TargetDistribution,
    PredictionDistribution,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "data")

# Default number of histogram bins
DEFAULT_NUM_BINS = 20

# Feature columns (528 total for spam-detector)
# Order: Text(8) -> Structural(10) -> Temporal(4) -> SpamIndicators(2) -> TF-IDF(500) -> Sender(4)
FEATURE_COLUMNS = [
    # Text features (8)
    "url_count", "email_count_in_body", "uppercase_ratio", "exclamation_count", 
    "question_mark_count", "avg_word_length", "word_count", "char_count",
    # Structural features (10)
    "has_html", "html_to_text_ratio", "subject_length", "subject_has_re",
    "subject_has_fwd", "subject_all_caps", "has_x_mailer", "sender_domain_length", 
    "sender_has_numbers", "received_hop_count",
    # Temporal features (4)
    "hour_of_day", "day_of_week", "is_weekend", "is_night_hour",
    # Spam indicator features (2)
    "spam_keyword_count", "has_unsubscribe",
] + [f"tfidf_{i}" for i in range(500)] + [
    # Sender domain features (4)
    "email_count", "spam_count", "ham_count", "spam_ratio"
]


# =============================================================================
# Azure Storage Client
# =============================================================================

class AzureBlobClient:
    """Client for Azure Blob Storage operations."""
    
    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        self.account_name = account_name or AZURE_STORAGE_ACCOUNT_NAME
        self.account_key = account_key or AZURE_STORAGE_ACCOUNT_KEY
        self.container_name = container_name or AZURE_STORAGE_CONTAINER
        
        if not self.account_name:
            raise ValueError("Azure storage account name is required")
        
        # Build connection
        if self.account_key:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.account_name};"
                f"AccountKey={self.account_key};"
                f"EndpointSuffix=core.windows.net"
            )
            self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        else:
            # Use managed identity
            credential = DefaultAzureCredential()
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            self.blob_service = BlobServiceClient(account_url, credential=credential)
        
        self.container_client = self.blob_service.get_container_client(self.container_name)
    
    def download_parquet(self, blob_path: str) -> pd.DataFrame:
        """Download a parquet file from blob storage."""
        logger.info(f"Downloading parquet from {blob_path}")
        
        blob_client = self.container_client.get_blob_client(blob_path)
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
            data = blob_client.download_blob().readall()
            tmp.write(data)
        
        try:
            df = pd.read_parquet(tmp_path)
            return df
        finally:
            os.unlink(tmp_path)
    
    def upload_json(self, data: Dict[str, Any], blob_path: str) -> str:
        """Upload JSON data to blob storage."""
        logger.info(f"Uploading JSON to {blob_path}")
        
        blob_client = self.container_client.get_blob_client(blob_path)
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        blob_client.upload_blob(json_bytes, overwrite=True)
        
        return f"abfss://{self.container_name}@{self.account_name}.dfs.core.windows.net/{blob_path}"
    
    def upload_pickle(self, obj: Any, blob_path: str) -> str:
        """Upload a pickled object to blob storage."""
        logger.info(f"Uploading pickle to {blob_path}")
        
        blob_client = self.container_client.get_blob_client(blob_path)
        pickle_bytes = pickle.dumps(obj)
        blob_client.upload_blob(pickle_bytes, overwrite=True)
        
        return f"abfss://{self.container_name}@{self.account_name}.dfs.core.windows.net/{blob_path}"
    
    def check_exists(self, blob_path: str) -> bool:
        """Check if a blob exists."""
        blob_client = self.container_client.get_blob_client(blob_path)
        return blob_client.exists()


# =============================================================================
# Statistics Calculation
# =============================================================================

def compute_histogram(
    values: np.ndarray,
    num_bins: int = DEFAULT_NUM_BINS,
) -> Tuple[List[HistogramBin], List[float], List[float]]:
    """
    Compute histogram for a feature.
    
    Returns:
        Tuple of (histogram_bins, bin_edges, proportions)
    """
    # Remove NaN values
    clean_values = values[~np.isnan(values)]
    
    if len(clean_values) == 0:
        return [], [], []
    
    # Compute histogram
    counts, bin_edges = np.histogram(clean_values, bins=num_bins)
    total = len(clean_values)
    
    histogram_bins = []
    proportions = []
    
    for i in range(len(counts)):
        density = counts[i] / total if total > 0 else 0
        proportions.append(density)
        
        histogram_bins.append(HistogramBin(
            bin_start=float(bin_edges[i]),
            bin_end=float(bin_edges[i + 1]),
            count=int(counts[i]),
            density=density
        ))
    
    return histogram_bins, bin_edges.tolist(), proportions


def compute_feature_statistics(
    values: np.ndarray,
    feature_name: str,
    num_bins: int = DEFAULT_NUM_BINS,
) -> Tuple[FeatureStatistics, DistributionBin]:
    """
    Compute comprehensive statistics for a single feature.
    
    Returns:
        Tuple of (FeatureStatistics, DistributionBin)
    """
    # Handle missing values
    total_count = len(values)
    missing_count = int(np.isnan(values).sum())
    clean_values = values[~np.isnan(values)]
    
    if len(clean_values) == 0:
        # All values are missing
        stats = FeatureStatistics(
            mean=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            median=0.0,
            q25=0.0,
            q75=0.0,
            missing_count=missing_count,
            missing_ratio=1.0,
            histogram=[],
            unique_count=0,
        )
        dist = DistributionBin(bin_edges=[], counts=[], proportions=[])
        return stats, dist
    
    # Compute statistics
    histogram_bins, bin_edges, proportions = compute_histogram(clean_values, num_bins)
    
    stats = FeatureStatistics(
        mean=float(np.mean(clean_values)),
        std=float(np.std(clean_values)),
        min=float(np.min(clean_values)),
        max=float(np.max(clean_values)),
        median=float(np.median(clean_values)),
        q25=float(np.percentile(clean_values, 25)),
        q75=float(np.percentile(clean_values, 75)),
        missing_count=missing_count,
        missing_ratio=missing_count / total_count if total_count > 0 else 0.0,
        histogram=histogram_bins,
        unique_count=len(np.unique(clean_values)),
    )
    
    # Distribution for PSI calculation
    counts, _ = np.histogram(clean_values, bins=bin_edges if bin_edges else num_bins)
    dist = DistributionBin(
        bin_edges=bin_edges,
        counts=counts.tolist(),
        proportions=proportions,
    )
    
    return stats, dist


def compute_target_distribution(
    labels: np.ndarray,
) -> TargetDistribution:
    """Compute target label distribution."""
    total = len(labels)
    
    if total == 0:
        return TargetDistribution(spam=0.0, ham=0.0, total_samples=0)
    
    # Handle both numeric (0/1) and string labels
    if labels.dtype == np.object_ or labels.dtype.kind == 'U':
        spam_count = np.sum(labels == "spam")
        ham_count = np.sum(labels == "ham")
    else:
        spam_count = np.sum(labels == 1)
        ham_count = np.sum(labels == 0)
    
    return TargetDistribution(
        spam=float(spam_count / total),
        ham=float(ham_count / total),
        total_samples=total,
    )


def compute_prediction_distribution(
    probabilities: np.ndarray,
    num_bins: int = DEFAULT_NUM_BINS,
) -> PredictionDistribution:
    """Compute prediction probability distribution."""
    clean_probs = probabilities[~np.isnan(probabilities)]
    
    if len(clean_probs) == 0:
        return PredictionDistribution(
            mean_probability=0.5,
            std_probability=0.0,
            median_probability=0.5,
            histogram=[],
        )
    
    histogram_bins, _, _ = compute_histogram(clean_probs, num_bins)
    
    return PredictionDistribution(
        mean_probability=float(np.mean(clean_probs)),
        std_probability=float(np.std(clean_probs)),
        median_probability=float(np.median(clean_probs)),
        histogram=histogram_bins,
    )


# =============================================================================
# Baseline Generator
# =============================================================================

class BaselineGenerator:
    """
    Generates baseline statistics from training data.
    
    Usage:
        generator = BaselineGenerator(
            model_name="spam-detector",
            model_version="3"
        )
        baseline = generator.generate(training_df)
        generator.save(baseline, output_path)
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str,
        feature_columns: Optional[List[str]] = None,
        num_bins: int = DEFAULT_NUM_BINS,
        azure_client: Optional[AzureBlobClient] = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.num_bins = num_bins
        self.azure_client = azure_client or AzureBlobClient()
    
    def generate(
        self,
        df: pd.DataFrame,
        label_column: str = "label",
        prediction_column: Optional[str] = None,
        training_data_path: str = "unknown",
    ) -> BaselineData:
        """
        Generate baseline from training data DataFrame.
        
        Args:
            df: Training data with features and labels
            label_column: Name of the label column
            prediction_column: Name of prediction probability column (if available)
            training_data_path: Path to training data for metadata
            
        Returns:
            BaselineData object containing all baseline statistics
        """
        logger.info(f"Generating baseline for {self.model_name} v{self.model_version}")
        logger.info(f"Dataset shape: {df.shape}")
        
        # Identify available feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        logger.info(f"Found {len(available_features)} of {len(self.feature_columns)} features")
        
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing features: {list(missing)[:10]}...")
        
        # Compute feature statistics
        feature_stats: Dict[str, FeatureStatistics] = {}
        feature_distributions: Dict[str, DistributionBin] = {}
        
        for i, feature in enumerate(available_features):
            if i % 50 == 0:
                logger.info(f"Processing feature {i+1}/{len(available_features)}: {feature}")
            
            values = df[feature].values.astype(float)
            stats, dist = compute_feature_statistics(values, feature, self.num_bins)
            feature_stats[feature] = stats
            feature_distributions[feature] = dist
        
        # Compute target distribution
        if label_column in df.columns:
            target_dist = compute_target_distribution(df[label_column].values)
        else:
            logger.warning(f"Label column '{label_column}' not found")
            target_dist = TargetDistribution(spam=0.5, ham=0.5, total_samples=len(df))
        
        # Compute prediction distribution if available
        pred_dist = None
        if prediction_column and prediction_column in df.columns:
            pred_dist = compute_prediction_distribution(df[prediction_column].values)
        
        # Create metadata
        metadata = BaselineMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            created_at=datetime.now(timezone.utc),
            training_data_path=training_data_path,
            num_samples=len(df),
            feature_count=len(available_features),
        )
        
        # Build baseline
        baseline = BaselineData(
            metadata=metadata,
            feature_stats=feature_stats,
            feature_distributions=feature_distributions,
            target_distribution=target_dist,
            prediction_distribution=pred_dist,
            feature_names=available_features,
        )
        
        logger.info(f"Baseline generated: {len(feature_stats)} features, {target_dist.total_samples} samples")
        return baseline
    
    def save(
        self,
        baseline: BaselineData,
        output_path: str,
    ) -> Dict[str, str]:
        """
        Save baseline to Azure Blob Storage.
        
        Args:
            baseline: BaselineData object to save
            output_path: Base path in blob storage (e.g., "baselines/spam-detector/v3/")
            
        Returns:
            Dictionary with paths to saved files
        """
        # Ensure path ends with /
        if not output_path.endswith("/"):
            output_path += "/"
        
        # Build full paths
        baseline_json_path = f"{output_path}baseline.json"
        latest_json_path = f"baselines/{self.model_name}/latest/baseline.json"
        
        # Convert to dict for JSON serialization
        baseline_dict = baseline.model_dump()
        
        # Save main baseline file
        json_url = self.azure_client.upload_json(baseline_dict, baseline_json_path)
        logger.info(f"Saved baseline JSON: {json_url}")
        
        # Save as "latest" for easy access
        latest_url = self.azure_client.upload_json(baseline_dict, latest_json_path)
        logger.info(f"Updated latest baseline: {latest_url}")
        
        # Save raw distributions as pickle for faster loading
        distributions_path = f"{output_path}distributions.pkl"
        dist_url = self.azure_client.upload_pickle(
            {
                "feature_distributions": {k: v.model_dump() for k, v in baseline.feature_distributions.items()},
                "feature_names": baseline.feature_names,
            },
            distributions_path
        )
        logger.info(f"Saved distributions pickle: {dist_url}")
        
        return {
            "baseline_json": json_url,
            "latest_json": latest_url,
            "distributions_pkl": dist_url,
        }
    
    def generate_from_blob(
        self,
        training_data_path: str,
        label_column: str = "label",
        prediction_column: Optional[str] = None,
    ) -> BaselineData:
        """
        Generate baseline from training data stored in Azure Blob.
        
        Args:
            training_data_path: Blob path to training parquet file
            label_column: Name of label column
            prediction_column: Name of prediction probability column
            
        Returns:
            BaselineData object
        """
        # Download training data
        df = self.azure_client.download_parquet(training_data_path)
        
        # Generate baseline
        return self.generate(
            df,
            label_column=label_column,
            prediction_column=prediction_column,
            training_data_path=training_data_path,
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate baseline statistics from training data"
    )
    
    parser.add_argument(
        "--training-data",
        required=True,
        help="Path to training data parquet file (Azure Blob path)"
    )
    parser.add_argument(
        "--model-name",
        default="spam-detector",
        help="Name of the model"
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="Version of the model"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path in Azure Blob storage"
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column"
    )
    parser.add_argument(
        "--prediction-column",
        default=None,
        help="Name of prediction probability column (optional)"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Number of histogram bins"
    )
    parser.add_argument(
        "--container",
        default="data",
        help="Azure Blob container name"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for baseline generation."""
    args = parse_args()
    
    logger.info(f"Starting baseline generation")
    logger.info(f"Model: {args.model_name} v{args.model_version}")
    logger.info(f"Training data: {args.training_data}")
    
    try:
        # Initialize Azure client
        azure_client = AzureBlobClient(container_name=args.container)
        
        # Create generator
        generator = BaselineGenerator(
            model_name=args.model_name,
            model_version=args.model_version,
            num_bins=args.num_bins,
            azure_client=azure_client,
        )
        
        # Parse training data path (handle abfss:// URIs)
        training_path = args.training_data
        if training_path.startswith("abfss://"):
            # Extract blob path from abfss URI
            # Format: abfss://container@account.dfs.core.windows.net/path
            parts = training_path.replace("abfss://", "").split("/", 1)
            if len(parts) > 1:
                training_path = parts[1]
        
        # Generate baseline
        baseline = generator.generate_from_blob(
            training_data_path=training_path,
            label_column=args.label_column,
            prediction_column=args.prediction_column,
        )
        
        # Determine output path
        output_path = args.output_path
        if output_path.startswith("abfss://"):
            parts = output_path.replace("abfss://", "").split("/", 1)
            if len(parts) > 1:
                output_path = parts[1]
        
        # Add model/version to path if not present
        if args.model_name not in output_path:
            output_path = f"{output_path.rstrip('/')}/{args.model_name}/v{args.model_version}/"
        
        # Save baseline
        paths = generator.save(baseline, output_path)
        
        logger.info("Baseline generation complete!")
        logger.info(f"Output paths: {json.dumps(paths, indent=2)}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("BASELINE GENERATION SUMMARY")
        print("=" * 60)
        print(f"Model: {args.model_name} v{args.model_version}")
        print(f"Samples: {baseline.metadata.num_samples:,}")
        print(f"Features: {baseline.metadata.feature_count}")
        print(f"Target distribution: {baseline.target_distribution.spam:.1%} spam, {baseline.target_distribution.ham:.1%} ham")
        print(f"\nSaved to: {paths['baseline_json']}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Baseline generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
