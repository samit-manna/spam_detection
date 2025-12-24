"""
Tests for Baseline Generator

Tests baseline statistics computation and storage.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from monitoring.baseline.schema import (
    BaselineData,
    FeatureStatistics,
    DistributionBin,
    HistogramBin,
    TargetDistribution,
    PredictionDistribution,
)
from monitoring.baseline.generator import (
    compute_feature_statistics,
    compute_target_distribution,
    compute_prediction_distribution,
    compute_histogram,
    BaselineGenerator,
)


class TestHistogramComputation:
    """Tests for histogram computation."""
    
    def test_compute_histogram_normal_distribution(self):
        """Test histogram for normal distribution."""
        np.random.seed(42)
        values = np.random.normal(100, 15, 1000)
        
        histogram, bin_edges, proportions = compute_histogram(values, num_bins=10)
        
        assert len(histogram) == 10
        assert len(bin_edges) == 11
        assert len(proportions) == 10
        assert abs(sum(proportions) - 1.0) < 0.01
        
        # Check that middle bins have more samples (normal distribution)
        assert histogram[4].count > histogram[0].count
        assert histogram[5].count > histogram[9].count
    
    def test_compute_histogram_with_nan(self):
        """Test histogram handles NaN values."""
        values = np.array([1, 2, 3, np.nan, 5, np.nan, 7, 8, 9, 10])
        
        histogram, bin_edges, proportions = compute_histogram(values, num_bins=5)
        
        # Should only count non-NaN values (8 values)
        total_count = sum(h.count for h in histogram)
        assert total_count == 8
    
    def test_compute_histogram_empty(self):
        """Test histogram with empty array."""
        values = np.array([])
        
        histogram, bin_edges, proportions = compute_histogram(values)
        
        assert histogram == []
        assert bin_edges == []
        assert proportions == []
    
    def test_compute_histogram_single_value(self):
        """Test histogram with all same values."""
        values = np.array([5.0] * 100)
        
        histogram, bin_edges, proportions = compute_histogram(values, num_bins=10)
        
        # All values should be in one or two bins
        non_zero_bins = [h for h in histogram if h.count > 0]
        assert len(non_zero_bins) >= 1


class TestFeatureStatistics:
    """Tests for feature statistics computation."""
    
    def test_compute_feature_statistics_normal(self):
        """Test statistics for normal distribution."""
        np.random.seed(42)
        values = np.random.normal(100, 15, 1000)
        
        stats, dist = compute_feature_statistics(values, "test_feature")
        
        assert abs(stats.mean - 100) < 2
        assert abs(stats.std - 15) < 2
        assert stats.min < 70
        assert stats.max > 130
        assert stats.missing_count == 0
        assert stats.missing_ratio == 0.0
    
    def test_compute_feature_statistics_with_missing(self):
        """Test statistics with missing values."""
        values = np.array([1, 2, 3, np.nan, 5, np.nan, 7, 8, 9, 10])
        
        stats, dist = compute_feature_statistics(values, "test_feature")
        
        assert stats.missing_count == 2
        assert stats.missing_ratio == 0.2
        assert stats.mean == pytest.approx(5.625, rel=0.01)  # Mean of non-NaN
    
    def test_compute_feature_statistics_all_missing(self):
        """Test statistics when all values are missing."""
        values = np.array([np.nan, np.nan, np.nan])
        
        stats, dist = compute_feature_statistics(values, "test_feature")
        
        assert stats.missing_count == 3
        assert stats.missing_ratio == 1.0
        assert stats.mean == 0.0
        assert stats.std == 0.0


class TestTargetDistribution:
    """Tests for target distribution computation."""
    
    def test_target_distribution_numeric(self):
        """Test target distribution with numeric labels."""
        labels = np.array([1, 1, 1, 0, 0])
        
        dist = compute_target_distribution(labels)
        
        assert dist.spam == 0.6  # 3/5
        assert dist.ham == 0.4   # 2/5
        assert dist.total_samples == 5
    
    def test_target_distribution_string(self):
        """Test target distribution with string labels."""
        labels = np.array(["spam", "spam", "ham", "ham", "ham"])
        
        dist = compute_target_distribution(labels)
        
        assert dist.spam == 0.4  # 2/5
        assert dist.ham == 0.6   # 3/5
        assert dist.total_samples == 5
    
    def test_target_distribution_empty(self):
        """Test target distribution with empty array."""
        labels = np.array([])
        
        dist = compute_target_distribution(labels)
        
        assert dist.spam == 0.0
        assert dist.ham == 0.0
        assert dist.total_samples == 0


class TestPredictionDistribution:
    """Tests for prediction distribution computation."""
    
    def test_prediction_distribution_uniform(self):
        """Test prediction distribution with uniform values."""
        np.random.seed(42)
        probabilities = np.random.uniform(0, 1, 1000)
        
        dist = compute_prediction_distribution(probabilities)
        
        assert abs(dist.mean_probability - 0.5) < 0.05
        assert abs(dist.std_probability - 0.29) < 0.05  # std of uniform(0,1)
        assert len(dist.histogram) > 0
    
    def test_prediction_distribution_skewed(self):
        """Test prediction distribution with skewed values."""
        # Most values near 0.8 (spam-biased model)
        np.random.seed(42)
        probabilities = np.random.beta(4, 1, 1000)
        
        dist = compute_prediction_distribution(probabilities)
        
        assert dist.mean_probability > 0.7
        assert dist.median_probability > 0.7


class TestBaselineGenerator:
    """Tests for baseline generator."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            "url_count": np.random.poisson(2, n_samples),
            "word_count": np.random.normal(150, 50, n_samples).astype(int),
            "uppercase_ratio": np.random.beta(2, 10, n_samples),
            "label": np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        })
        
        return df
    
    def test_generate_baseline(self, sample_training_data):
        """Test baseline generation from DataFrame."""
        generator = BaselineGenerator(
            model_name="test-model",
            model_version="1",
            feature_columns=["url_count", "word_count", "uppercase_ratio"],
        )
        
        baseline = generator.generate(
            sample_training_data,
            label_column="label",
            training_data_path="test/path.parquet",
        )
        
        assert baseline.metadata.model_name == "test-model"
        assert baseline.metadata.model_version == "1"
        assert baseline.metadata.num_samples == 1000
        assert baseline.metadata.feature_count == 3
        
        # Check feature stats
        assert "url_count" in baseline.feature_stats
        assert "word_count" in baseline.feature_stats
        assert "uppercase_ratio" in baseline.feature_stats
        
        # Check target distribution
        assert abs(baseline.target_distribution.spam - 0.35) < 0.05
        assert abs(baseline.target_distribution.ham - 0.65) < 0.05
    
    def test_generate_baseline_missing_features(self, sample_training_data):
        """Test baseline generation with missing features."""
        generator = BaselineGenerator(
            model_name="test-model",
            model_version="1",
            feature_columns=["url_count", "word_count", "missing_feature"],
        )
        
        baseline = generator.generate(sample_training_data, label_column="label")
        
        # Should only include available features
        assert baseline.metadata.feature_count == 2
        assert "url_count" in baseline.feature_stats
        assert "missing_feature" not in baseline.feature_stats
