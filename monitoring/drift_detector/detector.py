"""
Drift Detector Job

Main drift detection logic that compares inference data against baseline
and generates comprehensive drift reports.

Usage:
    python -m monitoring.drift_detector.detector \
        --baseline-path "baselines/spam-detector/v3/baseline.json" \
        --inference-logs-path "inference-logs/" \
        --window-hours 24 \
        --output-path "drift-reports/"
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.identity import DefaultAzureCredential
import pyarrow.parquet as pq

from monitoring.baseline.schema import (
    BaselineData,
    DriftReport,
    DriftAlert,
    DriftSeverity,
    FeatureDriftResult,
    PredictionDriftResult,
    DataQualityResult,
    AnalysisWindow,
)
from .config import DriftConfig, KEY_FEATURES
from .metrics import (
    calculate_psi,
    calculate_ks_test,
    calculate_mean_shift,
    calculate_data_quality_metrics,
    compute_aggregate_drift_score,
)
from .evidently_runner import EvidentlyRunner, create_reference_dataframe
from .alerts import AlertGenerator, AlertNotifier, AlertConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        self.account_name = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = container_name or os.environ.get("AZURE_STORAGE_CONTAINER", "data")
        
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
            credential = DefaultAzureCredential()
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            self.blob_service = BlobServiceClient(account_url, credential=credential)
        
        self.container_client = self.blob_service.get_container_client(self.container_name)
    
    def download_json(self, blob_path: str) -> Dict[str, Any]:
        """Download and parse a JSON file from blob storage."""
        logger.info(f"Downloading JSON from {blob_path}")
        blob_client = self.container_client.get_blob_client(blob_path)
        data = blob_client.download_blob().readall()
        return json.loads(data.decode("utf-8"))
    
    def list_blobs(self, prefix: str, suffix: str = ".parquet") -> List[str]:
        """List blobs with given prefix and suffix."""
        blobs = []
        for blob in self.container_client.list_blobs(name_starts_with=prefix):
            if blob.name.endswith(suffix):
                blobs.append(blob.name)
        return blobs
    
    def download_parquet_files(
        self,
        prefix: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Download and concatenate parquet files within time range.
        
        The files are expected to be partitioned by date:
        prefix/year=YYYY/month=MM/day=DD/hour=HH/*.parquet
        """
        logger.info(f"Downloading parquet files from {prefix}")
        logger.info(f"Time range: {start_time} to {end_time}")
        
        all_dfs = []
        
        # Generate date partitions to check
        current = start_time
        while current <= end_time:
            partition_prefix = (
                f"{prefix.rstrip('/')}/"
                f"year={current.year}/"
                f"month={current.month:02d}/"
                f"day={current.day:02d}/"
            )
            
            # List blobs in this partition
            try:
                blobs = self.list_blobs(partition_prefix)
                logger.info(f"Found {len(blobs)} files in {partition_prefix}")
                
                for blob_path in blobs:
                    try:
                        blob_client = self.container_client.get_blob_client(blob_path)
                        
                        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                            tmp_path = tmp.name
                            data = blob_client.download_blob().readall()
                            tmp.write(data)
                        
                        df = pd.read_parquet(tmp_path)
                        os.unlink(tmp_path)
                        
                        # Filter by timestamp if column exists
                        if "timestamp" in df.columns:
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                            df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
                        
                        if len(df) > 0:
                            all_dfs.append(df)
                            
                    except Exception as e:
                        logger.warning(f"Failed to read {blob_path}: {e}")
                        
            except Exception as e:
                logger.warning(f"Failed to list blobs in {partition_prefix}: {e}")
            
            # Move to next day
            current += timedelta(days=1)
        
        if not all_dfs:
            logger.warning("No inference logs found in time range")
            return pd.DataFrame()
        
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} inference records")
        return combined
    
    def upload_json(self, data: Dict[str, Any], blob_path: str) -> str:
        """Upload JSON data to blob storage."""
        logger.info(f"Uploading JSON to {blob_path}")
        blob_client = self.container_client.get_blob_client(blob_path)
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        blob_client.upload_blob(json_bytes, overwrite=True)
        return f"abfss://{self.container_name}@{self.account_name}.dfs.core.windows.net/{blob_path}"
    
    def upload_file(self, local_path: str, blob_path: str) -> str:
        """Upload a local file to blob storage."""
        logger.info(f"Uploading file to {blob_path}")
        blob_client = self.container_client.get_blob_client(blob_path)
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
        return f"abfss://{self.container_name}@{self.account_name}.dfs.core.windows.net/{blob_path}"


# =============================================================================
# Drift Detector
# =============================================================================

class DriftDetector:
    """
    Main drift detection class.
    
    Compares inference data against baseline and generates drift reports.
    """
    
    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        azure_client: Optional[AzureBlobClient] = None,
    ):
        self.config = config or DriftConfig()
        self.azure_client = azure_client or AzureBlobClient(
            account_name=self.config.azure_storage_account_name,
            account_key=self.config.azure_storage_account_key,
            container_name=self.config.azure_storage_container,
        )
        
        self.alert_generator = AlertGenerator()
        self.alert_notifier = AlertNotifier(AlertConfig(
            enabled=self.config.alert_enabled,
            webhook_url=self.config.alert_webhook_url,
        ))
    
    def load_baseline(self, baseline_path: str) -> Dict[str, Any]:
        """Load baseline data from blob storage."""
        return self.azure_client.download_json(baseline_path)
    
    def load_inference_logs(
        self,
        logs_path: str,
        window_hours: int,
    ) -> pd.DataFrame:
        """Load inference logs for analysis window."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=window_hours)
        
        return self.azure_client.download_parquet_files(
            prefix=logs_path,
            start_time=start_time,
            end_time=end_time,
        )
    
    def extract_features_from_logs(
        self,
        logs_df: pd.DataFrame,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """
        Extract feature values from inference logs.
        
        Inference logs store features as JSON string in features_json column.
        """
        if "features_json" not in logs_df.columns:
            logger.warning("No features_json column in logs")
            return pd.DataFrame()
        
        # Parse features from JSON
        feature_data = []
        for _, row in logs_df.iterrows():
            try:
                features = json.loads(row["features_json"])
                feature_data.append(features)
            except Exception as e:
                logger.warning(f"Failed to parse features: {e}")
        
        if not feature_data:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(feature_data)
        
        # Add prediction columns if available
        if "prediction" in logs_df.columns:
            features_df["prediction"] = logs_df["prediction"].values
        if "spam_probability" in logs_df.columns:
            features_df["spam_probability"] = logs_df["spam_probability"].values
        
        return features_df
    
    def detect_feature_drift(
        self,
        baseline: Dict[str, Any],
        current_df: pd.DataFrame,
        feature_names: List[str],
    ) -> Dict[str, FeatureDriftResult]:
        """
        Detect drift for each feature.
        
        Returns dictionary of feature name -> FeatureDriftResult
        """
        results = {}
        feature_stats = baseline.get("feature_stats", {})
        feature_distributions = baseline.get("feature_distributions", {})
        
        for feature in feature_names:
            if feature not in current_df.columns:
                continue
            
            baseline_stats = feature_stats.get(feature, {})
            baseline_dist = feature_distributions.get(feature, {})
            
            if not baseline_stats:
                continue
            
            current_values = current_df[feature].values.astype(float)
            
            # Get baseline values from distribution
            baseline_props = baseline_dist.get("proportions", [])
            bin_edges = baseline_dist.get("bin_edges", [])
            
            # Calculate PSI
            if baseline_props and bin_edges and len(bin_edges) > 1:
                # Bin current values using baseline bin edges
                current_counts, _ = np.histogram(
                    current_values[~np.isnan(current_values)],
                    bins=bin_edges
                )
                total_current = current_counts.sum()
                current_props = current_counts / total_current if total_current > 0 else current_counts
                
                # Calculate PSI from proportions
                from .metrics import calculate_psi_from_proportions
                psi = calculate_psi_from_proportions(baseline_props, current_props.tolist())
            else:
                # Fall back to generating baseline samples
                baseline_mean = baseline_stats.get("mean", 0)
                baseline_std = baseline_stats.get("std", 1)
                baseline_samples = np.random.normal(baseline_mean, max(baseline_std, 1e-6), 1000)
                psi = calculate_psi(baseline_samples, current_values)
            
            # Calculate KS test
            baseline_mean = baseline_stats.get("mean", 0)
            baseline_std = baseline_stats.get("std", 1)
            baseline_samples = np.random.normal(baseline_mean, max(baseline_std, 1e-6), 1000)
            ks_stat, ks_pval = calculate_ks_test(baseline_samples, current_values)
            
            # Current statistics
            current_mean = float(np.nanmean(current_values))
            mean_shift = current_mean - baseline_mean
            
            # Determine if drift detected
            drift_detected = psi > self.config.psi_threshold or ks_pval < self.config.ks_pvalue_threshold
            
            results[feature] = FeatureDriftResult(
                feature_name=feature,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                drift_detected=drift_detected,
                baseline_mean=baseline_mean,
                current_mean=current_mean,
                mean_shift=mean_shift,
            )
        
        return results
    
    def detect_prediction_drift(
        self,
        baseline: Dict[str, Any],
        current_df: pd.DataFrame,
    ) -> Optional[PredictionDriftResult]:
        """Detect drift in prediction probabilities."""
        if "spam_probability" not in current_df.columns:
            return None
        
        baseline_pred = baseline.get("prediction_distribution", {})
        if not baseline_pred:
            return None
        
        baseline_mean = baseline_pred.get("mean_probability", 0.5)
        baseline_std = baseline_pred.get("std_probability", 0.25)
        
        current_probs = current_df["spam_probability"].values
        current_mean = float(np.nanmean(current_probs))
        
        # Generate baseline samples for PSI calculation
        baseline_samples = np.random.normal(baseline_mean, max(baseline_std, 1e-6), 1000)
        baseline_samples = np.clip(baseline_samples, 0, 1)
        
        psi = calculate_psi(baseline_samples, current_probs)
        mean_shift = current_mean - baseline_mean
        
        drift_detected = psi > self.config.prediction_drift_psi_threshold
        
        return PredictionDriftResult(
            psi=psi,
            mean_shift=mean_shift,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            drift_detected=drift_detected,
        )
    
    def check_data_quality(
        self,
        baseline: Dict[str, Any],
        current_df: pd.DataFrame,
        feature_names: List[str],
    ) -> DataQualityResult:
        """Check data quality metrics."""
        total_values = 0
        missing_values = 0
        out_of_range = 0
        null_features = 0
        
        feature_stats = baseline.get("feature_stats", {})
        
        for feature in feature_names:
            if feature not in current_df.columns:
                null_features += 1
                continue
            
            values = current_df[feature].values
            total_values += len(values)
            missing_values += int(pd.isna(values).sum())
            
            # Check out of range
            baseline_stat = feature_stats.get(feature, {})
            if baseline_stat:
                min_val = baseline_stat.get("min", float("-inf"))
                max_val = baseline_stat.get("max", float("inf"))
                clean_values = values[~pd.isna(values)]
                out_of_range += int(((clean_values < min_val) | (clean_values > max_val)).sum())
        
        missing_pct = (missing_values / total_values * 100) if total_values > 0 else 0
        out_of_range_pct = (out_of_range / total_values * 100) if total_values > 0 else 0
        
        issues_detected = (
            missing_pct > self.config.missing_values_threshold_pct or
            out_of_range_pct > self.config.out_of_range_threshold_pct
        )
        
        return DataQualityResult(
            missing_values_pct=missing_pct,
            out_of_range_pct=out_of_range_pct,
            null_features_count=null_features,
            issues_detected=issues_detected,
        )
    
    def run_detection(
        self,
        baseline_path: Optional[str] = None,
        inference_logs_path: Optional[str] = None,
        window_hours: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> DriftReport:
        """
        Run complete drift detection.
        
        Args:
            baseline_path: Path to baseline JSON file
            inference_logs_path: Path prefix for inference logs
            window_hours: Analysis window in hours
            output_path: Path to save drift report
            
        Returns:
            DriftReport with all drift metrics
        """
        # Use config defaults if not provided
        baseline_path = baseline_path or self.config.baseline_path
        inference_logs_path = inference_logs_path or self.config.inference_logs_path
        window_hours = window_hours or self.config.analysis_window_hours
        output_path = output_path or self.config.output_path
        
        logger.info(f"Starting drift detection for {self.config.model_name}")
        logger.info(f"Baseline: {baseline_path}")
        logger.info(f"Logs: {inference_logs_path}")
        logger.info(f"Window: {window_hours} hours")
        
        # Load baseline
        baseline = self.load_baseline(baseline_path)
        baseline_metadata = baseline.get("metadata", {})
        feature_names = baseline.get("feature_names", KEY_FEATURES)
        
        # Filter to key features if configured
        if self.config.top_features_count > 0:
            # Prioritize KEY_FEATURES
            monitored = [f for f in KEY_FEATURES if f in feature_names]
            remaining = [f for f in feature_names if f not in KEY_FEATURES]
            feature_names = monitored + remaining[:self.config.top_features_count - len(monitored)]
        
        # Exclude configured features
        if self.config.excluded_features:
            feature_names = [f for f in feature_names if f not in self.config.excluded_features]
        
        logger.info(f"Monitoring {len(feature_names)} features")
        
        # Load inference logs
        logs_df = self.load_inference_logs(inference_logs_path, window_hours)
        
        if len(logs_df) < self.config.min_samples_required:
            logger.warning(
                f"Insufficient samples: {len(logs_df)} < {self.config.min_samples_required}"
            )
        
        # Extract features from logs
        current_df = self.extract_features_from_logs(logs_df, feature_names)
        
        # Define analysis window
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=window_hours)
        analysis_window = AnalysisWindow(
            start=start_time,
            end=end_time,
            num_samples=len(current_df),
        )
        
        # Detect feature drift
        feature_drift = self.detect_feature_drift(baseline, current_df, feature_names)
        
        # Detect prediction drift
        prediction_drift = self.detect_prediction_drift(baseline, current_df)
        
        # Check data quality
        data_quality = self.check_data_quality(baseline, current_df, feature_names)
        
        # Calculate aggregate drift score
        feature_psi_values = {k: v.psi for k, v in feature_drift.items()}
        drift_score = compute_aggregate_drift_score(
            feature_psi_values,
            prediction_psi=prediction_drift.psi if prediction_drift else None,
        )
        
        # Identify drifted features
        features_drifted = [k for k, v in feature_drift.items() if v.drift_detected]
        
        # Determine overall drift
        overall_drift = (
            drift_score >= self.config.drift_score_warning_threshold or
            len(features_drifted) >= self.config.features_drifted_warning_count or
            (prediction_drift and prediction_drift.drift_detected)
        )
        
        # Generate report ID
        report_id = f"drift-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        
        # Build drift report
        report = DriftReport(
            report_id=report_id,
            generated_at=datetime.now(timezone.utc),
            model_name=self.config.model_name,
            model_version=baseline_metadata.get("model_version", "unknown"),
            baseline_version=baseline_metadata.get("model_version", "unknown"),
            analysis_window=analysis_window,
            overall_drift_detected=overall_drift,
            drift_score=drift_score,
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            data_quality=data_quality,
            total_features=len(feature_names),
            features_drifted_count=len(features_drifted),
            features_drifted_names=features_drifted,
            baseline_path=baseline_path,
            inference_logs_path=inference_logs_path,
        )
        
        # Generate alerts
        alerts = self.alert_generator.generate_alerts(report, self.config)
        report.alerts = alerts
        
        logger.info(f"Drift detection complete: drift_score={drift_score:.3f}, drifted_features={len(features_drifted)}")
        
        # Save report
        if output_path:
            self._save_report(report, output_path, current_df, baseline, feature_names)
        
        return report
    
    def _save_report(
        self,
        report: DriftReport,
        output_path: str,
        current_df: pd.DataFrame,
        baseline: Dict[str, Any],
        feature_names: List[str],
    ):
        """Save drift report and optionally generate Evidently HTML."""
        timestamp = report.generated_at
        
        # Build paths
        dated_path = (
            f"{output_path.rstrip('/')}/"
            f"{self.config.model_name}/"
            f"{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}/"
        )
        
        # Save JSON report
        json_path = f"{dated_path}report_{timestamp.strftime('%H%M%S')}.json"
        json_url = self.azure_client.upload_json(report.model_dump(), json_path)
        logger.info(f"Saved JSON report: {json_url}")
        
        # Update latest report
        latest_path = f"{output_path.rstrip('/')}/{self.config.model_name}/latest.json"
        self.azure_client.upload_json(report.model_dump(), latest_path)
        
        # Generate Evidently HTML report
        if self.config.generate_evidently_html and len(current_df) > 0:
            try:
                # Create reference DataFrame from baseline
                reference_df = create_reference_dataframe(
                    baseline,
                    feature_names,
                    num_samples=min(len(current_df) * 2, 10000),
                )
                
                # Initialize Evidently runner
                evidently_runner = EvidentlyRunner(
                    model_name=self.config.model_name,
                    feature_columns=feature_names,
                )
                
                # Generate HTML report
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                    html_path = tmp.name
                
                evidently_result = evidently_runner.generate_drift_report(
                    reference_data=reference_df[feature_names],
                    current_data=current_df[feature_names] if all(f in current_df.columns for f in feature_names) else current_df,
                    output_path=html_path,
                )
                
                # Upload HTML report
                html_blob_path = f"{dated_path}report_{timestamp.strftime('%H%M%S')}.html"
                html_url = self.azure_client.upload_file(html_path, html_blob_path)
                report.evidently_report_path = html_url
                
                # Cleanup temp file
                os.unlink(html_path)
                
                logger.info(f"Saved Evidently HTML report: {html_url}")
                
            except Exception as e:
                logger.error(f"Failed to generate Evidently report: {e}")
    
    async def run_with_alerts(
        self,
        **kwargs,
    ) -> DriftReport:
        """Run drift detection and send alerts."""
        report = self.run_detection(**kwargs)
        
        if report.alerts:
            await self.alert_notifier.send_alerts(report.alerts, report)
        
        return report


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run drift detection on inference logs"
    )
    
    parser.add_argument(
        "--baseline-path",
        required=True,
        help="Path to baseline JSON file in blob storage"
    )
    parser.add_argument(
        "--inference-logs-path",
        required=True,
        help="Path prefix for inference logs in blob storage"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to save drift reports in blob storage"
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Analysis window in hours"
    )
    parser.add_argument(
        "--model-name",
        default="spam-detector",
        help="Name of the model"
    )
    parser.add_argument(
        "--container",
        default=None,
        help="Azure Blob container name (defaults to AZURE_STORAGE_CONTAINER env var)"
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.2,
        help="PSI threshold for drift detection"
    )
    parser.add_argument(
        "--generate-html",
        action="store_true",
        default=True,
        help="Generate Evidently HTML report"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for drift detection."""
    args = parse_args()
    
    logger.info("Starting drift detection job")
    
    try:
        # Create config (only override azure_storage_container if explicitly provided)
        config_kwargs = {
            "model_name": args.model_name,
        }
        if args.container:
            config_kwargs["azure_storage_container"] = args.container
        
        config = DriftConfig(
            **config_kwargs,
            baseline_path=args.baseline_path,
            inference_logs_path=args.inference_logs_path,
            output_path=args.output_path,
            analysis_window_hours=args.window_hours,
            psi_threshold=args.psi_threshold,
            generate_evidently_html=args.generate_html,
        )
        
        # Create detector
        detector = DriftDetector(config=config)
        
        # Run detection
        report = detector.run_detection()
        
        # Print summary
        print("\n" + "=" * 60)
        print("DRIFT DETECTION SUMMARY")
        print("=" * 60)
        print(f"Report ID: {report.report_id}")
        print(f"Model: {report.model_name} v{report.model_version}")
        print(f"Samples analyzed: {report.analysis_window.num_samples}")
        print(f"")
        print(f"Overall drift detected: {'YES' if report.overall_drift_detected else 'NO'}")
        print(f"Drift score: {report.drift_score:.1%}")
        print(f"Features drifted: {report.features_drifted_count}/{report.total_features}")
        
        if report.features_drifted_names:
            print(f"Drifted features: {', '.join(report.features_drifted_names[:10])}")
        
        if report.prediction_drift:
            print(f"")
            print(f"Prediction drift PSI: {report.prediction_drift.psi:.3f}")
            print(f"Prediction mean shift: {report.prediction_drift.mean_shift:.3f}")
        
        if report.alerts:
            print(f"")
            print(f"Alerts: {len(report.alerts)}")
            for alert in report.alerts[:5]:
                print(f"  [{alert.severity.value.upper()}] {alert.message}")
        
        print("=" * 60)
        
        # Exit with code based on drift status
        if any(a.severity == DriftSeverity.CRITICAL for a in report.alerts):
            return 2  # Critical drift
        elif report.overall_drift_detected:
            return 1  # Warning level drift
        return 0  # No significant drift
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}", exc_info=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
