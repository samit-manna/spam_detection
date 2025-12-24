"""
Statistical Metrics for Drift Detection

Implements PSI (Population Stability Index), KS Test, and other
statistical metrics for detecting data drift.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures how much a distribution has shifted compared to a baseline.
    
    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Slight change, monitor
    - PSI >= 0.2: Significant change, action required
    
    Args:
        baseline: Reference distribution values
        current: Current distribution values  
        num_bins: Number of bins for histogram
        eps: Small value to avoid division by zero
        
    Returns:
        PSI value (float)
    """
    # Clean data
    baseline_clean = baseline[~np.isnan(baseline)]
    current_clean = current[~np.isnan(current)]
    
    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return 0.0
    
    # Create bins based on baseline distribution
    min_val = min(baseline_clean.min(), current_clean.min())
    max_val = max(baseline_clean.max(), current_clean.max())
    
    # Handle edge case where all values are the same
    if min_val == max_val:
        return 0.0
    
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    # Calculate proportions
    baseline_counts, _ = np.histogram(baseline_clean, bins=bins)
    current_counts, _ = np.histogram(current_clean, bins=bins)
    
    baseline_props = baseline_counts / len(baseline_clean)
    current_props = current_counts / len(current_clean)
    
    # Add epsilon to avoid log(0) and division by zero
    baseline_props = np.clip(baseline_props, eps, 1.0)
    current_props = np.clip(current_props, eps, 1.0)
    
    # Calculate PSI
    psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))
    
    return float(psi)


def calculate_psi_from_proportions(
    baseline_props: List[float],
    current_props: List[float],
    eps: float = 1e-6,
) -> float:
    """
    Calculate PSI from pre-computed proportions.
    
    Args:
        baseline_props: Baseline bin proportions
        current_props: Current bin proportions
        eps: Small value to avoid division by zero
        
    Returns:
        PSI value
    """
    baseline = np.array(baseline_props)
    current = np.array(current_props)
    
    # Ensure same length
    if len(baseline) != len(current):
        logger.warning(f"Proportion arrays have different lengths: {len(baseline)} vs {len(current)}")
        return 0.0
    
    # Add epsilon
    baseline = np.clip(baseline, eps, 1.0)
    current = np.clip(current, eps, 1.0)
    
    # Normalize to ensure they sum to 1
    baseline = baseline / baseline.sum()
    current = current / current.sum()
    
    # Calculate PSI
    psi = np.sum((current - baseline) * np.log(current / baseline))
    
    return float(psi)


def calculate_ks_test(
    baseline: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test between two distributions.
    
    The KS test measures the maximum difference between two cumulative
    distribution functions.
    
    Args:
        baseline: Reference distribution values
        current: Current distribution values
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    # Clean data
    baseline_clean = baseline[~np.isnan(baseline)]
    current_clean = current[~np.isnan(current)]
    
    if len(baseline_clean) < 2 or len(current_clean) < 2:
        return 0.0, 1.0
    
    try:
        ks_stat, p_value = stats.ks_2samp(baseline_clean, current_clean)
        return float(ks_stat), float(p_value)
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return 0.0, 1.0


def calculate_chi_square_test(
    baseline_counts: np.ndarray,
    current_counts: np.ndarray,
) -> Tuple[float, float]:
    """
    Perform Chi-square test for categorical distributions.
    
    Args:
        baseline_counts: Expected counts from baseline
        current_counts: Observed counts from current data
        
    Returns:
        Tuple of (Chi-square statistic, p-value)
    """
    try:
        # Scale current counts to match baseline total
        baseline_total = baseline_counts.sum()
        current_total = current_counts.sum()
        
        if baseline_total == 0 or current_total == 0:
            return 0.0, 1.0
        
        expected = baseline_counts * (current_total / baseline_total)
        
        # Filter out zero expected values
        mask = expected > 0
        if not mask.any():
            return 0.0, 1.0
        
        chi2_stat, p_value = stats.chisquare(
            current_counts[mask], 
            f_exp=expected[mask]
        )
        
        return float(chi2_stat), float(p_value)
        
    except Exception as e:
        logger.warning(f"Chi-square test failed: {e}")
        return 0.0, 1.0


def calculate_jensen_shannon_divergence(
    baseline_props: np.ndarray,
    current_props: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.
    
    JSD is a symmetric and bounded measure of similarity between distributions.
    Range: [0, 1] where 0 means identical distributions.
    
    Args:
        baseline_props: Baseline distribution proportions
        current_props: Current distribution proportions
        eps: Small value to avoid log(0)
        
    Returns:
        JSD value in range [0, 1]
    """
    # Normalize
    p = np.array(baseline_props) + eps
    q = np.array(current_props) + eps
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate mixture distribution
    m = 0.5 * (p + q)
    
    # Calculate JSD
    jsd = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
    
    return float(jsd)


def calculate_wasserstein_distance(
    baseline: np.ndarray,
    current: np.ndarray,
) -> float:
    """
    Calculate Wasserstein distance (Earth Mover's Distance).
    
    Measures the minimum "work" required to transform one distribution
    into another.
    
    Args:
        baseline: Reference distribution values
        current: Current distribution values
        
    Returns:
        Wasserstein distance
    """
    baseline_clean = baseline[~np.isnan(baseline)]
    current_clean = current[~np.isnan(current)]
    
    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return 0.0
    
    try:
        distance = stats.wasserstein_distance(baseline_clean, current_clean)
        return float(distance)
    except Exception as e:
        logger.warning(f"Wasserstein distance calculation failed: {e}")
        return 0.0


def calculate_mean_shift(
    baseline: np.ndarray,
    current: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Calculate mean shift between distributions.
    
    Args:
        baseline: Reference distribution values
        current: Current distribution values
        normalize: If True, normalize by baseline std
        
    Returns:
        Mean shift (optionally normalized)
    """
    baseline_clean = baseline[~np.isnan(baseline)]
    current_clean = current[~np.isnan(current)]
    
    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return 0.0
    
    baseline_mean = np.mean(baseline_clean)
    current_mean = np.mean(current_clean)
    
    shift = current_mean - baseline_mean
    
    if normalize:
        baseline_std = np.std(baseline_clean)
        if baseline_std > 0:
            shift = shift / baseline_std
    
    return float(shift)


def detect_outliers(
    values: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    n_sigmas: float = 3.0,
) -> Dict[str, float]:
    """
    Detect outliers based on baseline statistics.
    
    Args:
        values: Current values to check
        baseline_mean: Mean from baseline
        baseline_std: Standard deviation from baseline
        n_sigmas: Number of standard deviations for outlier threshold
        
    Returns:
        Dictionary with outlier statistics
    """
    clean_values = values[~np.isnan(values)]
    
    if len(clean_values) == 0 or baseline_std == 0:
        return {
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "lower_bound": baseline_mean,
            "upper_bound": baseline_mean,
        }
    
    lower_bound = baseline_mean - n_sigmas * baseline_std
    upper_bound = baseline_mean + n_sigmas * baseline_std
    
    outliers = (clean_values < lower_bound) | (clean_values > upper_bound)
    outlier_count = int(outliers.sum())
    
    return {
        "outlier_count": outlier_count,
        "outlier_pct": float(outlier_count / len(clean_values) * 100),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def calculate_data_quality_metrics(
    values: np.ndarray,
    baseline_min: float,
    baseline_max: float,
) -> Dict[str, float]:
    """
    Calculate data quality metrics.
    
    Args:
        values: Current values to check
        baseline_min: Minimum value from baseline
        baseline_max: Maximum value from baseline
        
    Returns:
        Dictionary with data quality metrics
    """
    total = len(values)
    
    if total == 0:
        return {
            "missing_count": 0,
            "missing_pct": 0.0,
            "out_of_range_count": 0,
            "out_of_range_pct": 0.0,
        }
    
    # Missing values
    missing_count = int(np.isnan(values).sum())
    
    # Out of range values
    clean_values = values[~np.isnan(values)]
    out_of_range = (clean_values < baseline_min) | (clean_values > baseline_max)
    out_of_range_count = int(out_of_range.sum())
    
    return {
        "missing_count": missing_count,
        "missing_pct": float(missing_count / total * 100),
        "out_of_range_count": out_of_range_count,
        "out_of_range_pct": float(out_of_range_count / len(clean_values) * 100) if len(clean_values) > 0 else 0.0,
    }


def compute_aggregate_drift_score(
    feature_psi_values: Dict[str, float],
    prediction_psi: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute aggregate drift score from individual feature PSI values.
    
    The aggregate score is a weighted average of individual drift scores,
    normalized to [0, 1] range.
    
    Args:
        feature_psi_values: Dictionary of feature name -> PSI value
        prediction_psi: PSI for prediction distribution (optional)
        weights: Optional weights for features
        
    Returns:
        Aggregate drift score in [0, 1]
    """
    if not feature_psi_values:
        return 0.0
    
    psi_values = list(feature_psi_values.values())
    
    # Add prediction PSI with higher weight
    if prediction_psi is not None:
        psi_values.append(prediction_psi * 2)  # Double weight for prediction drift
    
    # Calculate weighted average
    if weights:
        weighted_sum = sum(
            psi * weights.get(feat, 1.0) 
            for feat, psi in feature_psi_values.items()
        )
        total_weight = sum(weights.get(feat, 1.0) for feat in feature_psi_values)
        avg_psi = weighted_sum / total_weight if total_weight > 0 else 0.0
    else:
        avg_psi = np.mean(psi_values)
    
    # Normalize to [0, 1] using sigmoid-like transformation
    # PSI of 0.5 maps to ~0.5, PSI of 1.0 maps to ~0.73
    drift_score = 1 - np.exp(-avg_psi)
    
    return float(np.clip(drift_score, 0, 1))
