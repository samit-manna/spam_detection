"""
Tests for Drift Detector

Tests drift detection metrics and detection logic.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from monitoring.drift_detector.metrics import (
    calculate_psi,
    calculate_psi_from_proportions,
    calculate_ks_test,
    calculate_chi_square_test,
    calculate_jensen_shannon_divergence,
    calculate_wasserstein_distance,
    calculate_mean_shift,
    detect_outliers,
    calculate_data_quality_metrics,
    compute_aggregate_drift_score,
)


class TestPSICalculation:
    """Tests for PSI (Population Stability Index) calculation."""
    
    def test_psi_identical_distributions(self):
        """PSI should be 0 for identical distributions."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(100, 15, 1000)
        
        psi = calculate_psi(baseline, current)
        
        assert psi < 0.1  # Should be very small
    
    def test_psi_slight_shift(self):
        """PSI should be small for slight distribution shift."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(105, 15, 1000)  # Mean shifted by 5
        
        psi = calculate_psi(baseline, current)
        
        assert 0.01 < psi < 0.2  # Should show some drift but not critical
    
    def test_psi_significant_shift(self):
        """PSI should be large for significant distribution shift."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(130, 15, 1000)  # Mean shifted by 30
        
        psi = calculate_psi(baseline, current)
        
        assert psi > 0.2  # Should indicate significant drift
    
    def test_psi_variance_change(self):
        """PSI should detect variance changes."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(100, 30, 1000)  # Same mean, double std
        
        psi = calculate_psi(baseline, current)
        
        assert psi > 0.1  # Should show drift from variance change
    
    def test_psi_with_nan(self):
        """PSI should handle NaN values."""
        baseline = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10])
        current = np.array([2, 3, 4, 5, np.nan, 7, 8, 9, 10, 11])
        
        psi = calculate_psi(baseline, current)
        
        assert psi >= 0  # Should not fail
    
    def test_psi_empty_arrays(self):
        """PSI should return 0 for empty arrays."""
        baseline = np.array([])
        current = np.array([1, 2, 3])
        
        psi = calculate_psi(baseline, current)
        
        assert psi == 0.0
    
    def test_psi_from_proportions(self):
        """Test PSI calculation from pre-computed proportions."""
        baseline_props = [0.1, 0.2, 0.4, 0.2, 0.1]
        current_props = [0.05, 0.15, 0.5, 0.2, 0.1]
        
        psi = calculate_psi_from_proportions(baseline_props, current_props)
        
        assert psi >= 0


class TestKSTest:
    """Tests for Kolmogorov-Smirnov test."""
    
    def test_ks_identical_distributions(self):
        """KS test should show no difference for identical distributions."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(100, 15, 1000)
        
        ks_stat, p_value = calculate_ks_test(baseline, current)
        
        assert p_value > 0.05  # Should NOT reject null hypothesis
    
    def test_ks_different_distributions(self):
        """KS test should detect different distributions."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(120, 15, 1000)  # Different mean
        
        ks_stat, p_value = calculate_ks_test(baseline, current)
        
        assert p_value < 0.05  # Should reject null hypothesis
        assert ks_stat > 0.1   # Should show significant difference
    
    def test_ks_with_nan(self):
        """KS test should handle NaN values."""
        baseline = np.array([1, 2, 3, np.nan, 5])
        current = np.array([2, 3, 4, np.nan, 6])
        
        ks_stat, p_value = calculate_ks_test(baseline, current)
        
        assert 0 <= ks_stat <= 1
        assert 0 <= p_value <= 1


class TestChiSquareTest:
    """Tests for Chi-square test."""
    
    def test_chi_square_similar(self):
        """Chi-square should show no difference for similar counts."""
        baseline_counts = np.array([100, 200, 150, 100])
        current_counts = np.array([105, 195, 155, 95])
        
        chi2, p_value = calculate_chi_square_test(baseline_counts, current_counts)
        
        assert p_value > 0.05  # Should NOT reject null hypothesis
    
    def test_chi_square_different(self):
        """Chi-square should detect different distributions."""
        baseline_counts = np.array([100, 200, 150, 100])
        current_counts = np.array([50, 300, 100, 100])  # Very different
        
        chi2, p_value = calculate_chi_square_test(baseline_counts, current_counts)
        
        assert p_value < 0.05  # Should reject null hypothesis


class TestOtherMetrics:
    """Tests for other drift metrics."""
    
    def test_jensen_shannon_identical(self):
        """JSD should be 0 for identical distributions."""
        props = [0.1, 0.2, 0.4, 0.2, 0.1]
        
        jsd = calculate_jensen_shannon_divergence(np.array(props), np.array(props))
        
        assert jsd < 0.01
    
    def test_wasserstein_distance(self):
        """Test Wasserstein distance calculation."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(110, 15, 1000)  # Mean shifted by 10
        
        distance = calculate_wasserstein_distance(baseline, current)
        
        assert 8 < distance < 12  # Should be approximately 10
    
    def test_mean_shift_calculation(self):
        """Test mean shift calculation."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(115, 15, 1000)  # Mean shifted by 15
        
        shift = calculate_mean_shift(baseline, current, normalize=False)
        
        assert 13 < shift < 17  # Should be approximately 15
    
    def test_mean_shift_normalized(self):
        """Test normalized mean shift."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(115, 15, 1000)  # Mean shifted by 15 (1 std)
        
        shift = calculate_mean_shift(baseline, current, normalize=True)
        
        assert 0.8 < shift < 1.2  # Should be approximately 1 (15/15)


class TestOutlierDetection:
    """Tests for outlier detection."""
    
    def test_detect_outliers_none(self):
        """Test when no outliers present."""
        np.random.seed(42)
        values = np.random.normal(100, 15, 1000)
        
        result = detect_outliers(values, baseline_mean=100, baseline_std=15, n_sigmas=3)
        
        assert result["outlier_pct"] < 1  # Should have very few outliers
    
    def test_detect_outliers_present(self):
        """Test when outliers are present."""
        # Values with some extreme outliers
        values = np.concatenate([
            np.random.normal(100, 15, 900),
            np.array([200, 250, 300, 0, -50])  # Outliers
        ])
        
        result = detect_outliers(values, baseline_mean=100, baseline_std=15, n_sigmas=3)
        
        assert result["outlier_count"] >= 5


class TestDataQualityMetrics:
    """Tests for data quality metrics."""
    
    def test_data_quality_no_issues(self):
        """Test when no data quality issues."""
        values = np.array([1, 2, 3, 4, 5])
        
        result = calculate_data_quality_metrics(values, baseline_min=0, baseline_max=10)
        
        assert result["missing_pct"] == 0
        assert result["out_of_range_pct"] == 0
    
    def test_data_quality_missing_values(self):
        """Test missing value detection."""
        values = np.array([1, 2, np.nan, 4, np.nan])
        
        result = calculate_data_quality_metrics(values, baseline_min=0, baseline_max=10)
        
        assert result["missing_count"] == 2
        assert result["missing_pct"] == 40.0
    
    def test_data_quality_out_of_range(self):
        """Test out of range detection."""
        values = np.array([1, 2, 15, 4, -5])  # 15 and -5 out of range
        
        result = calculate_data_quality_metrics(values, baseline_min=0, baseline_max=10)
        
        assert result["out_of_range_count"] == 2


class TestAggregateScore:
    """Tests for aggregate drift score."""
    
    def test_aggregate_score_no_drift(self):
        """Test aggregate score with no drift."""
        feature_psi = {"f1": 0.01, "f2": 0.02, "f3": 0.01}
        
        score = compute_aggregate_drift_score(feature_psi)
        
        assert score < 0.1  # Should be very low
    
    def test_aggregate_score_some_drift(self):
        """Test aggregate score with some drift."""
        feature_psi = {"f1": 0.15, "f2": 0.25, "f3": 0.10}
        
        score = compute_aggregate_drift_score(feature_psi)
        
        assert 0.1 < score < 0.5
    
    def test_aggregate_score_with_prediction(self):
        """Test aggregate score with prediction drift."""
        feature_psi = {"f1": 0.05, "f2": 0.05}
        prediction_psi = 0.3  # Significant prediction drift
        
        score = compute_aggregate_drift_score(feature_psi, prediction_psi=prediction_psi)
        
        # Should be higher due to prediction drift
        score_without = compute_aggregate_drift_score(feature_psi)
        assert score > score_without
