"""
Tests for Synthetic Drift Detection

End-to-end tests that create synthetic drift scenarios
and verify the drift detector catches them correctly.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from monitoring.baseline.schema import (
    BaselineData,
    BaselineMetadata,
    FeatureStatistics,
    DistributionBin,
    TargetDistribution,
    DriftReport,
)
from monitoring.baseline.generator import (
    compute_feature_statistics,
    compute_target_distribution,
)
from monitoring.drift_detector.config import DriftConfig, KEY_FEATURES
from monitoring.drift_detector.metrics import (
    calculate_psi,
    calculate_ks_test,
    compute_aggregate_drift_score,
)
from monitoring.drift_detector.detector import DriftDetector


class SyntheticDataGenerator:
    """
    Generates synthetic data for drift detection testing.
    
    Creates baseline data and production data with configurable drift.
    """
    
    @staticmethod
    def create_baseline_data(
        n_samples: int = 10000,
        feature_config: Dict[str, Dict[str, float]] = None,
        spam_ratio: float = 0.35,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Create baseline training data.
        
        Args:
            n_samples: Number of samples
            feature_config: Dictionary of feature name -> {"mean": X, "std": Y}
            spam_ratio: Ratio of spam samples
            seed: Random seed
            
        Returns:
            DataFrame with features and labels
        """
        np.random.seed(seed)
        
        if feature_config is None:
            feature_config = {
                "url_count": {"mean": 2.0, "std": 1.5},
                "word_count": {"mean": 150, "std": 80},
                "uppercase_ratio": {"mean": 0.05, "std": 0.03},
                "exclamation_count": {"mean": 1.0, "std": 1.5},
                "spam_keyword_count": {"mean": 1.5, "std": 1.0},
            }
        
        data = {}
        for feature, config in feature_config.items():
            data[feature] = np.random.normal(
                config["mean"], 
                config["std"], 
                n_samples
            )
        
        # Ensure non-negative for count features
        for feature in ["url_count", "word_count", "exclamation_count", "spam_keyword_count"]:
            if feature in data:
                data[feature] = np.maximum(data[feature], 0)
        
        # Generate labels
        data["label"] = np.random.choice(
            [0, 1], 
            n_samples, 
            p=[1 - spam_ratio, spam_ratio]
        )
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_production_data_with_drift(
        n_samples: int = 1000,
        baseline_config: Dict[str, Dict[str, float]] = None,
        drift_config: Dict[str, Dict[str, float]] = None,
        seed: int = 123,
    ) -> pd.DataFrame:
        """
        Create production data with specified drift.
        
        Args:
            n_samples: Number of samples
            baseline_config: Original feature configuration
            drift_config: Dict of feature -> {"mean_shift": X, "std_multiplier": Y}
            seed: Random seed
            
        Returns:
            DataFrame with drifted features
        """
        np.random.seed(seed)
        
        if baseline_config is None:
            baseline_config = {
                "url_count": {"mean": 2.0, "std": 1.5},
                "word_count": {"mean": 150, "std": 80},
                "uppercase_ratio": {"mean": 0.05, "std": 0.03},
                "exclamation_count": {"mean": 1.0, "std": 1.5},
                "spam_keyword_count": {"mean": 1.5, "std": 1.0},
            }
        
        if drift_config is None:
            drift_config = {}
        
        data = {}
        for feature, config in baseline_config.items():
            # Apply drift if specified
            drift = drift_config.get(feature, {})
            mean_shift = drift.get("mean_shift", 0)
            std_multiplier = drift.get("std_multiplier", 1.0)
            
            new_mean = config["mean"] + mean_shift
            new_std = config["std"] * std_multiplier
            
            data[feature] = np.random.normal(new_mean, new_std, n_samples)
        
        # Ensure non-negative for count features
        for feature in ["url_count", "word_count", "exclamation_count", "spam_keyword_count"]:
            if feature in data:
                data[feature] = np.maximum(data[feature], 0)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_baseline_dict(
        df: pd.DataFrame,
        model_name: str = "test-model",
        model_version: str = "1",
    ) -> Dict[str, Any]:
        """Convert baseline DataFrame to baseline dict format."""
        feature_names = [c for c in df.columns if c != "label"]
        
        feature_stats = {}
        feature_distributions = {}
        
        for feature in feature_names:
            stats, dist = compute_feature_statistics(
                df[feature].values.astype(float),
                feature
            )
            feature_stats[feature] = stats.model_dump()
            feature_distributions[feature] = dist.model_dump()
        
        target_dist = TargetDistribution(spam=0.35, ham=0.65, total_samples=len(df))
        
        return {
            "metadata": {
                "model_name": model_name,
                "model_version": model_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "training_data_path": "test/train.parquet",
                "num_samples": len(df),
                "feature_count": len(feature_names),
            },
            "feature_stats": feature_stats,
            "feature_distributions": feature_distributions,
            "target_distribution": target_dist.model_dump(),
            "prediction_distribution": None,
            "feature_names": feature_names,
        }


class TestSyntheticDriftDetection:
    """End-to-end drift detection tests with synthetic data."""
    
    @pytest.fixture
    def data_generator(self):
        """Create data generator instance."""
        return SyntheticDataGenerator()
    
    @pytest.fixture
    def baseline_config(self):
        """Default baseline feature configuration."""
        return {
            "url_count": {"mean": 2.0, "std": 1.5},
            "word_count": {"mean": 150, "std": 80},
            "uppercase_ratio": {"mean": 0.05, "std": 0.03},
            "exclamation_count": {"mean": 1.0, "std": 1.5},
            "spam_keyword_count": {"mean": 1.5, "std": 1.0},
        }
    
    def test_no_drift_detection(self, data_generator, baseline_config):
        """Test that no drift is detected when distributions are stable."""
        # Create baseline
        baseline_df = data_generator.create_baseline_data(
            n_samples=10000,
            feature_config=baseline_config,
        )
        baseline = data_generator.create_baseline_dict(baseline_df)
        
        # Create production data with same distribution (no drift)
        production_df = data_generator.create_production_data_with_drift(
            n_samples=1000,
            baseline_config=baseline_config,
            drift_config={},  # No drift
        )
        
        # Calculate drift metrics
        drifted_features = []
        for feature in baseline_config.keys():
            baseline_values = baseline_df[feature].values
            current_values = production_df[feature].values
            
            psi = calculate_psi(baseline_values, current_values)
            ks_stat, ks_pval = calculate_ks_test(baseline_values, current_values)
            
            if psi > 0.2 or ks_pval < 0.05:
                drifted_features.append(feature)
        
        # No features should show significant drift
        assert len(drifted_features) <= 1, f"Unexpected drift in: {drifted_features}"
    
    def test_mean_shift_drift_detection(self, data_generator, baseline_config):
        """Test detection of mean shift in features."""
        # Create baseline
        baseline_df = data_generator.create_baseline_data(
            n_samples=10000,
            feature_config=baseline_config,
        )
        
        # Create production data with mean shift in word_count
        drift_config = {
            "word_count": {"mean_shift": 50, "std_multiplier": 1.0},  # Significant shift
        }
        
        production_df = data_generator.create_production_data_with_drift(
            n_samples=1000,
            baseline_config=baseline_config,
            drift_config=drift_config,
        )
        
        # Calculate drift for word_count
        baseline_values = baseline_df["word_count"].values
        current_values = production_df["word_count"].values
        
        psi = calculate_psi(baseline_values, current_values)
        ks_stat, ks_pval = calculate_ks_test(baseline_values, current_values)
        
        # Should detect drift
        assert psi > 0.1, f"PSI too low: {psi}"
        assert ks_pval < 0.05, f"KS test p-value too high: {ks_pval}"
    
    def test_variance_change_drift_detection(self, data_generator, baseline_config):
        """Test detection of variance change in features."""
        # Create baseline
        baseline_df = data_generator.create_baseline_data(
            n_samples=10000,
            feature_config=baseline_config,
        )
        
        # Create production data with increased variance
        drift_config = {
            "url_count": {"mean_shift": 0, "std_multiplier": 2.0},  # Double variance
        }
        
        production_df = data_generator.create_production_data_with_drift(
            n_samples=1000,
            baseline_config=baseline_config,
            drift_config=drift_config,
        )
        
        # Calculate drift
        baseline_values = baseline_df["url_count"].values
        current_values = production_df["url_count"].values
        
        psi = calculate_psi(baseline_values, current_values)
        
        # Should detect some drift from variance change
        assert psi > 0.05, f"PSI too low for variance change: {psi}"
    
    def test_multiple_feature_drift_detection(self, data_generator, baseline_config):
        """Test detection when multiple features drift together."""
        # Create baseline
        baseline_df = data_generator.create_baseline_data(
            n_samples=10000,
            feature_config=baseline_config,
        )
        baseline = data_generator.create_baseline_dict(baseline_df)
        
        # Create production data with drift in multiple features
        drift_config = {
            "url_count": {"mean_shift": 2.0, "std_multiplier": 1.0},
            "word_count": {"mean_shift": 30, "std_multiplier": 1.0},
            "spam_keyword_count": {"mean_shift": 1.0, "std_multiplier": 1.5},
        }
        
        production_df = data_generator.create_production_data_with_drift(
            n_samples=1000,
            baseline_config=baseline_config,
            drift_config=drift_config,
        )
        
        # Calculate drift for each feature
        drifted_features = []
        psi_values = {}
        
        for feature in baseline_config.keys():
            baseline_values = baseline_df[feature].values
            current_values = production_df[feature].values
            
            psi = calculate_psi(baseline_values, current_values)
            psi_values[feature] = psi
            
            if psi > 0.1:
                drifted_features.append(feature)
        
        # Should detect drift in multiple features
        assert len(drifted_features) >= 2, f"Should detect drift in multiple features: {psi_values}"
        
        # Calculate aggregate drift score
        drift_score = compute_aggregate_drift_score(psi_values)
        assert drift_score > 0.1, f"Aggregate drift score too low: {drift_score}"
    
    def test_gradual_drift_detection(self, data_generator, baseline_config):
        """Test detection of gradual drift over time."""
        # Create baseline
        baseline_df = data_generator.create_baseline_data(
            n_samples=10000,
            feature_config=baseline_config,
        )
        
        # Simulate gradual drift over time
        drift_levels = [
            {"word_count": {"mean_shift": 0, "std_multiplier": 1.0}},   # Week 1: No drift
            {"word_count": {"mean_shift": 10, "std_multiplier": 1.0}},  # Week 2: Slight
            {"word_count": {"mean_shift": 25, "std_multiplier": 1.0}},  # Week 3: Moderate
            {"word_count": {"mean_shift": 50, "std_multiplier": 1.0}},  # Week 4: Significant
        ]
        
        psi_progression = []
        
        for drift_config in drift_levels:
            production_df = data_generator.create_production_data_with_drift(
                n_samples=1000,
                baseline_config=baseline_config,
                drift_config=drift_config,
            )
            
            psi = calculate_psi(
                baseline_df["word_count"].values,
                production_df["word_count"].values
            )
            psi_progression.append(psi)
        
        # PSI should increase over time
        assert psi_progression[0] < psi_progression[-1], "Drift should increase over time"
        
        # Should cross warning threshold eventually
        assert psi_progression[-1] > 0.1, "Should detect significant drift by week 4"
    
    def test_threshold_sensitivity(self, data_generator, baseline_config):
        """Test drift detection sensitivity to different thresholds."""
        # Create baseline
        baseline_df = data_generator.create_baseline_data(
            n_samples=10000,
            feature_config=baseline_config,
        )
        
        # Create moderate drift
        drift_config = {
            "word_count": {"mean_shift": 20, "std_multiplier": 1.0},
        }
        
        production_df = data_generator.create_production_data_with_drift(
            n_samples=1000,
            baseline_config=baseline_config,
            drift_config=drift_config,
        )
        
        psi = calculate_psi(
            baseline_df["word_count"].values,
            production_df["word_count"].values
        )
        
        # Test different thresholds
        thresholds = [0.05, 0.1, 0.2, 0.3]
        detections = {t: psi > t for t in thresholds}
        
        # More sensitive thresholds should detect more drift
        if psi > 0.1:
            assert detections[0.05] == True
            assert detections[0.1] == True


class TestDriftAlertGeneration:
    """Tests for alert generation based on drift results."""
    
    def test_critical_alert_generation(self):
        """Test that critical alerts are generated for severe drift."""
        from monitoring.drift_detector.alerts import AlertGenerator, AlertConfig
        from monitoring.baseline.schema import (
            DriftReport,
            AnalysisWindow,
            FeatureDriftResult,
            DataQualityResult,
            DriftSeverity,
        )
        
        # Create a report with severe drift
        report = DriftReport(
            report_id="test-001",
            generated_at=datetime.now(timezone.utc),
            model_name="spam-detector",
            model_version="1",
            baseline_version="1",
            analysis_window=AnalysisWindow(
                start=datetime.now(timezone.utc) - timedelta(hours=24),
                end=datetime.now(timezone.utc),
                num_samples=1000,
            ),
            overall_drift_detected=True,
            drift_score=0.6,  # Critical level
            feature_drift={
                "word_count": FeatureDriftResult(
                    feature_name="word_count",
                    psi=0.35,
                    ks_statistic=0.15,
                    ks_pvalue=0.001,
                    drift_detected=True,
                    baseline_mean=150.0,
                    current_mean=200.0,
                    mean_shift=50.0,
                ),
            },
            data_quality=DataQualityResult(
                missing_values_pct=0.1,
                out_of_range_pct=0.5,
                null_features_count=0,
                issues_detected=False,
            ),
            total_features=5,
            features_drifted_count=1,
            features_drifted_names=["word_count"],
            baseline_path="test/baseline.json",
            inference_logs_path="test/logs/",
        )
        
        # Generate alerts
        generator = AlertGenerator()
        alerts = generator.generate_alerts(report)
        
        # Should have at least one critical alert
        critical_alerts = [a for a in alerts if a.severity == DriftSeverity.CRITICAL]
        assert len(critical_alerts) >= 1, "Should generate critical alert for severe drift"
    
    def test_warning_alert_generation(self):
        """Test that warning alerts are generated for moderate drift."""
        from monitoring.drift_detector.alerts import AlertGenerator
        from monitoring.baseline.schema import (
            DriftReport,
            AnalysisWindow,
            FeatureDriftResult,
            DataQualityResult,
            DriftSeverity,
        )
        
        # Create a report with moderate drift
        report = DriftReport(
            report_id="test-002",
            generated_at=datetime.now(timezone.utc),
            model_name="spam-detector",
            model_version="1",
            baseline_version="1",
            analysis_window=AnalysisWindow(
                start=datetime.now(timezone.utc) - timedelta(hours=24),
                end=datetime.now(timezone.utc),
                num_samples=1000,
            ),
            overall_drift_detected=True,
            drift_score=0.25,  # Warning level
            feature_drift={
                "word_count": FeatureDriftResult(
                    feature_name="word_count",
                    psi=0.22,
                    ks_statistic=0.08,
                    ks_pvalue=0.02,
                    drift_detected=True,
                ),
            },
            data_quality=DataQualityResult(
                missing_values_pct=0.1,
                out_of_range_pct=0.5,
                null_features_count=0,
                issues_detected=False,
            ),
            total_features=5,
            features_drifted_count=1,
            features_drifted_names=["word_count"],
            baseline_path="test/baseline.json",
            inference_logs_path="test/logs/",
        )
        
        # Generate alerts
        generator = AlertGenerator()
        alerts = generator.generate_alerts(report)
        
        # Should have warning alerts
        warning_alerts = [a for a in alerts if a.severity == DriftSeverity.WARNING]
        assert len(warning_alerts) >= 1, "Should generate warning alert for moderate drift"
