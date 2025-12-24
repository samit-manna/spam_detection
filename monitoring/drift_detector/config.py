"""
Drift Detector Configuration

Centralized configuration for drift detection thresholds and parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    
    # ==========================================================================
    # Azure Storage
    # ==========================================================================
    azure_storage_account_name: str = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
    )
    azure_storage_account_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
    )
    azure_storage_container: str = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_CONTAINER", "data")
    )
    
    # ==========================================================================
    # Model Info
    # ==========================================================================
    model_name: str = field(
        default_factory=lambda: os.environ.get("DRIFT_MODEL_NAME", "spam-detector")
    )
    model_version: str = field(
        default_factory=lambda: os.environ.get("DRIFT_MODEL_VERSION", "latest")
    )
    
    # ==========================================================================
    # Paths
    # ==========================================================================
    baseline_path: str = field(
        default_factory=lambda: os.environ.get(
            "DRIFT_BASELINE_PATH", 
            "baselines/spam-detector/latest/baseline.json"
        )
    )
    inference_logs_path: str = field(
        default_factory=lambda: os.environ.get("DRIFT_INFERENCE_LOGS_PATH", "inference-logs")
    )
    output_path: str = field(
        default_factory=lambda: os.environ.get("DRIFT_OUTPUT_PATH", "drift-reports")
    )
    
    # ==========================================================================
    # Analysis Window
    # ==========================================================================
    analysis_window_hours: int = field(
        default_factory=lambda: int(os.environ.get("DRIFT_ANALYSIS_WINDOW_HOURS", "24"))
    )
    min_samples_required: int = field(
        default_factory=lambda: int(os.environ.get("DRIFT_MIN_SAMPLES", "100"))
    )
    
    # ==========================================================================
    # Drift Detection Thresholds
    # ==========================================================================
    
    # PSI (Population Stability Index)
    # < 0.1: No significant change
    # 0.1 - 0.2: Slight change, monitor
    # > 0.2: Significant change, action needed
    psi_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.2"))
    )
    
    # KS Test p-value threshold
    # p < 0.05: Statistically significant difference
    ks_pvalue_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_KS_PVALUE_THRESHOLD", "0.05"))
    )
    
    # Aggregate drift score thresholds
    drift_score_warning_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_SCORE_WARNING_THRESHOLD", "0.2"))
    )
    drift_score_critical_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_SCORE_CRITICAL_THRESHOLD", "0.5"))
    )
    
    # Feature drift count threshold
    features_drifted_warning_count: int = field(
        default_factory=lambda: int(os.environ.get("DRIFT_FEATURES_WARNING_COUNT", "3"))
    )
    
    # Prediction drift
    prediction_drift_psi_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_PREDICTION_PSI_THRESHOLD", "0.2"))
    )
    prediction_drift_critical_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_PREDICTION_CRITICAL_THRESHOLD", "0.3"))
    )
    
    # ==========================================================================
    # Data Quality Thresholds
    # ==========================================================================
    missing_values_threshold_pct: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_MISSING_VALUES_THRESHOLD", "5.0"))
    )
    out_of_range_threshold_pct: float = field(
        default_factory=lambda: float(os.environ.get("DRIFT_OUT_OF_RANGE_THRESHOLD", "1.0"))
    )
    
    # ==========================================================================
    # Alerting
    # ==========================================================================
    alert_enabled: bool = field(
        default_factory=lambda: os.environ.get("ALERT_ENABLED", "true").lower() == "true"
    )
    alert_webhook_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("ALERT_WEBHOOK_URL")
    )
    
    # ==========================================================================
    # Feature Selection
    # ==========================================================================
    # Features to monitor (empty = all features)
    monitored_features: List[str] = field(default_factory=list)
    
    # Features to skip (useful for high-cardinality or noisy features)
    excluded_features: List[str] = field(
        default_factory=lambda: os.environ.get(
            "DRIFT_EXCLUDED_FEATURES", ""
        ).split(",") if os.environ.get("DRIFT_EXCLUDED_FEATURES") else []
    )
    
    # Number of top features to include in report (for efficiency)
    top_features_count: int = field(
        default_factory=lambda: int(os.environ.get("DRIFT_TOP_FEATURES_COUNT", "50"))
    )
    
    # ==========================================================================
    # Report Generation
    # ==========================================================================
    generate_evidently_html: bool = field(
        default_factory=lambda: os.environ.get("DRIFT_GENERATE_HTML", "true").lower() == "true"
    )
    save_latest_report: bool = True
    
    # ==========================================================================
    # MLflow Integration (optional)
    # ==========================================================================
    mlflow_tracking_uri: Optional[str] = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI")
    )
    log_metrics_to_mlflow: bool = field(
        default_factory=lambda: os.environ.get("DRIFT_LOG_TO_MLFLOW", "false").lower() == "true"
    )


# Default feature columns for spam-detector (528 total)
DEFAULT_FEATURE_COLUMNS = [
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

# Key features to always monitor (non-TF-IDF features)
KEY_FEATURES = [
    "url_count", "email_count_in_body", "uppercase_ratio", "exclamation_count", 
    "question_mark_count", "avg_word_length", "word_count", "char_count",
    "has_html", "html_to_text_ratio", "subject_length", "subject_has_re",
    "subject_has_fwd", "subject_all_caps", "has_x_mailer", "sender_domain_length", 
    "sender_has_numbers", "received_hop_count",
    "hour_of_day", "day_of_week", "is_weekend", "is_night_hour",
    "spam_keyword_count", "has_unsubscribe",
    "email_count", "spam_count", "ham_count", "spam_ratio"
]
