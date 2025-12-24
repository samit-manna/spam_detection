"""
Baseline data models and schemas for model monitoring.

Defines the structure of baseline data saved after training,
used for drift detection comparisons.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class HistogramBin(BaseModel):
    """A single histogram bin."""
    bin_start: float
    bin_end: float
    count: int
    density: float


class FeatureStatistics(BaseModel):
    """Statistics for a single feature."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    missing_count: int = 0
    missing_ratio: float = 0.0
    histogram: List[HistogramBin] = []
    unique_count: Optional[int] = None


class DistributionBin(BaseModel):
    """Distribution bin for PSI calculation."""
    bin_edges: List[float]
    counts: List[int]
    proportions: List[float]


class TargetDistribution(BaseModel):
    """Distribution of target labels."""
    spam: float
    ham: float
    total_samples: int


class PredictionDistribution(BaseModel):
    """Distribution of prediction probabilities."""
    mean_probability: float
    std_probability: float
    median_probability: float
    histogram: List[HistogramBin] = []


class BaselineMetadata(BaseModel):
    """Metadata about the baseline."""
    model_name: str
    model_version: str
    created_at: datetime
    created_by: str = "training-pipeline"
    training_data_path: str
    num_samples: int
    feature_count: int


class BaselineData(BaseModel):
    """
    Complete baseline data structure.
    
    Contains all information needed to perform drift detection:
    - Feature statistics for each feature
    - Feature distributions for PSI calculation
    - Target distribution
    - Prediction distribution
    """
    
    metadata: BaselineMetadata
    feature_stats: Dict[str, FeatureStatistics] = Field(
        default_factory=dict,
        description="Statistics for each feature"
    )
    feature_distributions: Dict[str, DistributionBin] = Field(
        default_factory=dict,
        description="Binned distributions for PSI calculation"
    )
    target_distribution: TargetDistribution
    prediction_distribution: Optional[PredictionDistribution] = None
    feature_names: List[str] = Field(
        default_factory=list,
        description="Ordered list of feature names"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "model_name": "spam-detector",
                    "model_version": "3",
                    "created_at": "2024-01-15T10:30:00Z",
                    "training_data_path": "datasets/train.parquet",
                    "num_samples": 50000,
                    "feature_count": 528
                },
                "feature_stats": {
                    "url_count": {
                        "mean": 2.3,
                        "std": 1.5,
                        "min": 0,
                        "max": 15,
                        "median": 2.0,
                        "q25": 1.0,
                        "q75": 3.0
                    }
                },
                "target_distribution": {
                    "spam": 0.35,
                    "ham": 0.65,
                    "total_samples": 50000
                }
            }
        }


# =============================================================================
# Inference Log Schema
# =============================================================================

class InferenceRecord(BaseModel):
    """Schema for a single inference log record."""
    inference_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    
    # Input features (flattened for parquet storage)
    features: Dict[str, float]
    
    # Prediction output
    prediction: str  # "spam" or "ham"
    spam_probability: float
    
    # Metadata
    latency_ms: float
    api_version: str = "1.0.0"
    environment: str = "production"
    
    # Optional context
    email_id: Optional[str] = None
    sender_domain: Optional[str] = None


class InferenceBatch(BaseModel):
    """A batch of inference records for writing."""
    records: List[InferenceRecord]
    batch_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    record_count: int


# =============================================================================
# Drift Report Schema
# =============================================================================

class DriftSeverity(str, Enum):
    """Drift alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FeatureDriftResult(BaseModel):
    """Drift detection result for a single feature."""
    feature_name: str
    psi: float  # Population Stability Index
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float  # KS test p-value
    drift_detected: bool
    baseline_mean: Optional[float] = None
    current_mean: Optional[float] = None
    mean_shift: Optional[float] = None


class PredictionDriftResult(BaseModel):
    """Drift detection result for predictions."""
    psi: float
    mean_shift: float
    baseline_mean: float
    current_mean: float
    drift_detected: bool


class DataQualityResult(BaseModel):
    """Data quality metrics."""
    missing_values_pct: float
    out_of_range_pct: float
    null_features_count: int
    issues_detected: bool


class DriftAlert(BaseModel):
    """A single drift alert."""
    severity: DriftSeverity
    feature: Optional[str] = None
    message: str
    recommendation: str
    metric_name: str
    metric_value: float
    threshold: float


class AnalysisWindow(BaseModel):
    """Time window for drift analysis."""
    start: datetime
    end: datetime
    num_samples: int


class DriftReport(BaseModel):
    """
    Complete drift detection report.
    
    Contains all drift metrics, alerts, and recommendations.
    """
    report_id: str
    generated_at: datetime
    model_name: str
    model_version: str
    baseline_version: str
    
    analysis_window: AnalysisWindow
    
    overall_drift_detected: bool
    drift_score: float  # Aggregate score 0-1
    
    feature_drift: Dict[str, FeatureDriftResult] = Field(default_factory=dict)
    prediction_drift: Optional[PredictionDriftResult] = None
    data_quality: DataQualityResult
    
    alerts: List[DriftAlert] = Field(default_factory=list)
    
    # Counts
    total_features: int
    features_drifted_count: int
    features_drifted_names: List[str] = Field(default_factory=list)
    
    # Report paths
    evidently_report_path: Optional[str] = None
    baseline_path: str
    inference_logs_path: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "drift-20240115-103000",
                "generated_at": "2024-01-15T10:30:00Z",
                "model_name": "spam-detector",
                "model_version": "3",
                "overall_drift_detected": True,
                "drift_score": 0.35
            }
        }


# =============================================================================
# API Response Models
# =============================================================================

class DriftSummaryResponse(BaseModel):
    """Summary drift response for API."""
    model_name: str
    model_version: str
    last_checked: datetime
    drift_detected: bool
    drift_score: float
    features_drifted: List[str]
    alerts_count: int
    details_url: str
    report_url: Optional[str] = None


class DriftHistoryPoint(BaseModel):
    """Single point in drift history."""
    timestamp: datetime
    drift_score: float
    drift_detected: bool
    features_drifted_count: int


class DriftHistoryResponse(BaseModel):
    """Drift history response."""
    model_name: str
    history: List[DriftHistoryPoint]
    period_days: int
