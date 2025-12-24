"""
Drift Detector Module

Scheduled drift detection comparing inference data against baseline.
"""

from .config import DriftConfig, DEFAULT_FEATURE_COLUMNS, KEY_FEATURES
from .metrics import (
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
from .evidently_runner import (
    EvidentlyRunner,
    create_reference_dataframe,
    create_inference_dataframe,
    EVIDENTLY_AVAILABLE,
)
from .alerts import (
    AlertGenerator,
    AlertNotifier,
    AlertConfig,
    create_slack_payload,
)
from .detector import DriftDetector, AzureBlobClient

__all__ = [
    # Config
    "DriftConfig",
    "DEFAULT_FEATURE_COLUMNS",
    "KEY_FEATURES",
    # Metrics
    "calculate_psi",
    "calculate_psi_from_proportions",
    "calculate_ks_test",
    "calculate_chi_square_test",
    "calculate_jensen_shannon_divergence",
    "calculate_wasserstein_distance",
    "calculate_mean_shift",
    "detect_outliers",
    "calculate_data_quality_metrics",
    "compute_aggregate_drift_score",
    # Evidently
    "EvidentlyRunner",
    "create_reference_dataframe",
    "create_inference_dataframe",
    "EVIDENTLY_AVAILABLE",
    # Alerts
    "AlertGenerator",
    "AlertNotifier",
    "AlertConfig",
    "create_slack_payload",
    # Detector
    "DriftDetector",
    "AzureBlobClient",
]
