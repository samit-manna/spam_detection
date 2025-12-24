"""
Baseline module for model monitoring.

Provides functionality to generate and store baseline statistics
from training data for drift detection.
"""

from .schema import (
    BaselineData,
    BaselineMetadata,
    FeatureStatistics,
    DistributionBin,
    HistogramBin,
    TargetDistribution,
    PredictionDistribution,
    InferenceRecord,
    InferenceBatch,
    DriftReport,
    DriftAlert,
    DriftSeverity,
    FeatureDriftResult,
    PredictionDriftResult,
    DataQualityResult,
    AnalysisWindow,
    DriftSummaryResponse,
    DriftHistoryPoint,
    DriftHistoryResponse,
)
from .generator import (
    BaselineGenerator,
    AzureBlobClient,
    FEATURE_COLUMNS,
    compute_feature_statistics,
    compute_target_distribution,
    compute_prediction_distribution,
)

__all__ = [
    # Schema
    "BaselineData",
    "BaselineMetadata", 
    "FeatureStatistics",
    "DistributionBin",
    "HistogramBin",
    "TargetDistribution",
    "PredictionDistribution",
    "InferenceRecord",
    "InferenceBatch",
    "DriftReport",
    "DriftAlert",
    "DriftSeverity",
    "FeatureDriftResult",
    "PredictionDriftResult",
    "DataQualityResult",
    "AnalysisWindow",
    "DriftSummaryResponse",
    "DriftHistoryPoint",
    "DriftHistoryResponse",
    # Generator
    "BaselineGenerator",
    "AzureBlobClient",
    "FEATURE_COLUMNS",
    "compute_feature_statistics",
    "compute_target_distribution",
    "compute_prediction_distribution",
]
