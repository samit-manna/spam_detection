"""
Evidently AI Integration for Drift Detection

Generates rich HTML drift reports using Evidently AI library.
"""

import logging
import tempfile
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
        DatasetMissingValuesMetric,
        DatasetCorrelationsMetric,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not installed. HTML reports will not be generated.")

logger = logging.getLogger(__name__)


class EvidentlyRunner:
    """
    Generates drift reports using Evidently AI.
    
    Creates comprehensive HTML reports with visualizations for:
    - Feature drift detection
    - Data quality analysis
    - Distribution comparisons
    """
    
    def __init__(
        self,
        model_name: str,
        feature_columns: List[str],
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        prediction_column: Optional[str] = None,
    ):
        """
        Initialize Evidently runner.
        
        Args:
            model_name: Name of the model
            feature_columns: List of feature column names
            numerical_features: List of numerical feature names (optional)
            categorical_features: List of categorical feature names (optional)
            target_column: Name of target column (optional)
            prediction_column: Name of prediction column (optional)
        """
        self.model_name = model_name
        self.feature_columns = feature_columns
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features or []
        self.target_column = target_column
        self.prediction_column = prediction_column
        
        # Set up column mapping
        self._column_mapping = None
        if EVIDENTLY_AVAILABLE:
            self._column_mapping = ColumnMapping(
                numerical_features=numerical_features or feature_columns,
                categorical_features=categorical_features,
                target=target_column,
                prediction=prediction_column,
            )
    
    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report.
        
        Args:
            reference_data: Baseline/reference DataFrame
            current_data: Current/production DataFrame
            output_path: Path to save HTML report (optional)
            
        Returns:
            Dictionary containing drift metrics and report path
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, returning basic metrics only")
            return self._generate_basic_metrics(reference_data, current_data)
        
        logger.info(f"Generating Evidently drift report for {self.model_name}")
        
        # Ensure column names match
        common_cols = list(set(reference_data.columns) & set(current_data.columns))
        reference_data = reference_data[common_cols]
        current_data = current_data[common_cols]
        
        # Create report with drift metrics
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
            DatasetMissingValuesMetric(),
        ])
        
        try:
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self._column_mapping,
            )
        except Exception as e:
            logger.error(f"Failed to run Evidently report: {e}")
            return self._generate_basic_metrics(reference_data, current_data)
        
        # Extract metrics from report
        result = self._extract_metrics(report)
        
        # Save HTML report if path provided
        if output_path:
            try:
                report.save_html(output_path)
                result["html_report_path"] = output_path
                logger.info(f"Saved HTML report to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save HTML report: {e}")
        
        return result
    
    def generate_data_quality_report(
        self,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a data quality report.
        
        Args:
            current_data: Current data to analyze
            reference_data: Reference data for comparison (optional)
            output_path: Path to save HTML report (optional)
            
        Returns:
            Dictionary containing data quality metrics
        """
        if not EVIDENTLY_AVAILABLE:
            return {"error": "Evidently not available"}
        
        report = Report(metrics=[
            DatasetMissingValuesMetric(),
            DatasetCorrelationsMetric(),
        ])
        
        try:
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self._column_mapping,
            )
        except Exception as e:
            logger.error(f"Failed to run data quality report: {e}")
            return {"error": str(e)}
        
        result = {
            "generated_at": datetime.utcnow().isoformat(),
            "samples_analyzed": len(current_data),
        }
        
        # Extract quality metrics
        try:
            report_dict = report.as_dict()
            for metric in report_dict.get("metrics", []):
                metric_id = metric.get("metric", "")
                if "MissingValues" in metric_id:
                    result["missing_values"] = metric.get("result", {})
                elif "Correlations" in metric_id:
                    result["correlations"] = metric.get("result", {})
        except Exception as e:
            logger.warning(f"Failed to extract quality metrics: {e}")
        
        if output_path:
            try:
                report.save_html(output_path)
                result["html_report_path"] = output_path
            except Exception as e:
                logger.error(f"Failed to save HTML report: {e}")
        
        return result
    
    def _extract_metrics(self, report: "Report") -> Dict[str, Any]:
        """Extract metrics from Evidently report."""
        result = {
            "generated_at": datetime.utcnow().isoformat(),
            "dataset_drift": {},
            "feature_drift": {},
        }
        
        try:
            report_dict = report.as_dict()
            
            for metric in report_dict.get("metrics", []):
                metric_id = metric.get("metric", "")
                metric_result = metric.get("result", {})
                
                if "DatasetDriftMetric" in metric_id:
                    result["dataset_drift"] = {
                        "drift_detected": metric_result.get("dataset_drift", False),
                        "drift_share": metric_result.get("drift_share", 0.0),
                        "number_of_columns": metric_result.get("number_of_columns", 0),
                        "number_of_drifted_columns": metric_result.get("number_of_drifted_columns", 0),
                    }
                
                elif "DataDriftTable" in metric_id:
                    drift_by_columns = metric_result.get("drift_by_columns", {})
                    for col_name, col_data in drift_by_columns.items():
                        result["feature_drift"][col_name] = {
                            "drift_detected": col_data.get("drift_detected", False),
                            "drift_score": col_data.get("drift_score", 0.0),
                            "stattest_name": col_data.get("stattest_name", ""),
                            "stattest_threshold": col_data.get("stattest_threshold", 0.0),
                        }
                        
        except Exception as e:
            logger.error(f"Failed to extract metrics from report: {e}")
        
        return result
    
    def _generate_basic_metrics(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Generate basic metrics without Evidently."""
        from .metrics import calculate_psi, calculate_ks_test
        
        result = {
            "generated_at": datetime.utcnow().isoformat(),
            "evidently_available": False,
            "feature_drift": {},
        }
        
        drifted_count = 0
        for col in reference_data.columns:
            if col not in current_data.columns:
                continue
            
            try:
                ref_values = reference_data[col].values.astype(float)
                cur_values = current_data[col].values.astype(float)
                
                psi = calculate_psi(ref_values, cur_values)
                ks_stat, ks_pval = calculate_ks_test(ref_values, cur_values)
                
                drift_detected = psi > 0.2 or ks_pval < 0.05
                if drift_detected:
                    drifted_count += 1
                
                result["feature_drift"][col] = {
                    "psi": psi,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pval,
                    "drift_detected": drift_detected,
                }
            except Exception as e:
                logger.warning(f"Failed to compute drift for {col}: {e}")
        
        result["dataset_drift"] = {
            "drift_detected": drifted_count > 0,
            "drift_share": drifted_count / len(reference_data.columns) if len(reference_data.columns) > 0 else 0,
            "number_of_columns": len(reference_data.columns),
            "number_of_drifted_columns": drifted_count,
        }
        
        return result


def create_reference_dataframe(
    baseline_data: Dict[str, Any],
    feature_names: List[str],
    num_samples: int = 1000,
) -> pd.DataFrame:
    """
    Create a reference DataFrame from baseline statistics.
    
    Synthesizes data based on baseline distribution statistics
    for use with Evidently reporting.
    
    Args:
        baseline_data: Baseline data containing feature_stats
        feature_names: List of feature names
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthesized reference data
    """
    data = {}
    feature_stats = baseline_data.get("feature_stats", {})
    
    for feature in feature_names:
        if feature not in feature_stats:
            # Generate random normal data for missing features
            data[feature] = np.random.randn(num_samples)
            continue
        
        stats = feature_stats[feature]
        mean = stats.get("mean", 0)
        std = stats.get("std", 1)
        
        # Handle zero std
        if std == 0:
            std = 1e-6
        
        # Generate samples from normal distribution
        # This is an approximation of the original distribution
        samples = np.random.normal(mean, std, num_samples)
        
        # Clip to min/max from baseline
        min_val = stats.get("min", samples.min())
        max_val = stats.get("max", samples.max())
        samples = np.clip(samples, min_val, max_val)
        
        data[feature] = samples
    
    return pd.DataFrame(data)


def create_inference_dataframe(
    inference_records: List[Dict[str, Any]],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Create a DataFrame from inference log records.
    
    Args:
        inference_records: List of inference record dictionaries
        feature_names: List of feature names to extract
        
    Returns:
        DataFrame with inference features
    """
    data = {feature: [] for feature in feature_names}
    data["prediction"] = []
    data["spam_probability"] = []
    
    for record in inference_records:
        features = record.get("features", {})
        
        for feature in feature_names:
            value = features.get(feature, np.nan)
            data[feature].append(value)
        
        data["prediction"].append(record.get("prediction", ""))
        data["spam_probability"].append(record.get("spam_probability", 0.5))
    
    return pd.DataFrame(data)
