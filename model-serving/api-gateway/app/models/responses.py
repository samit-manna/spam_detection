"""
Pydantic models for API responses (serving-only).
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Prediction confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionLabel(str, Enum):
    """Prediction labels."""
    SPAM = "spam"
    HAM = "ham"


class JobStatus(str, Enum):
    """Batch job status."""
    SUBMITTED = "submitted"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# Error Responses
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail structure."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: ErrorDetail
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: str = Field(..., description="Error timestamp in ISO format")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "PREDICTION_FAILED",
                    "message": "Failed to extract features from email",
                    "details": {}
                },
                "correlation_id": "abc-123-def-456",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


# ============================================================================
# Health Responses
# ============================================================================

class ServiceHealth(BaseModel):
    """Health status for a service."""
    name: str
    status: str  # healthy, unhealthy, degraded
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Check timestamp")
    services: List[ServiceHealth] = Field(default_factory=list)


# ============================================================================
# Prediction Responses
# ============================================================================

class PredictResponse(BaseModel):
    """Single prediction response."""
    email_id: str = Field(..., description="Email identifier from request")
    prediction: PredictionLabel = Field(..., description="Prediction label")
    spam_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being spam")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    model_version: str = Field(..., description="Model version used")
    model_stage: str = Field(..., description="Model stage (Staging/Production)")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "email_id": "abc123",
                "prediction": "spam",
                "spam_probability": 0.94,
                "confidence": "high",
                "model_version": "3",
                "model_stage": "Production",
                "latency_ms": 23,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class BatchSyncResponse(BaseModel):
    """Synchronous batch prediction response."""
    predictions: List[PredictResponse]
    total_count: int
    success_count: int
    error_count: int
    total_latency_ms: float
    avg_latency_ms: float
    timestamp: str


class BatchJobResponse(BaseModel):
    """Async batch job submission response."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated completion time")
    created_at: str = Field(..., description="Job creation timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "batch-abc123",
                "status": "submitted",
                "estimated_duration_minutes": 15,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }
    }


class BatchJobStatusResponse(BaseModel):
    """Batch job status response."""
    job_id: str
    status: JobStatus
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Progress percentage")
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    records_processed: Optional[int] = None
    records_total: Optional[int] = None
    output_path: Optional[str] = None


class BatchJobResultsResponse(BaseModel):
    """Batch job results response."""
    job_id: str
    status: JobStatus
    output_path: str
    records_processed: int
    spam_count: int
    ham_count: int
    completed_at: str


# ============================================================================
# Metrics Responses
# ============================================================================

class DriftMetrics(BaseModel):
    """Model drift detection metrics."""
    feature_drift_score: float = Field(..., ge=0.0, le=1.0)
    prediction_drift_score: float = Field(..., ge=0.0, le=1.0)
    data_quality_score: float = Field(..., ge=0.0, le=1.0)
    drift_detected: bool
    drifted_features: List[str] = Field(default_factory=list)
    last_checked: str
    reference_window: str
    current_window: str


class PerformanceMetrics(BaseModel):
    """Model serving performance metrics."""
    model_version: str
    model_stage: str
    
    # Latency metrics
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    
    # Throughput
    requests_per_second: float
    total_requests_24h: int
    
    # Error rates
    error_rate: float
    timeout_rate: float
    
    # Time window
    window_start: str
    window_end: str


class MetricsSummary(BaseModel):
    """Combined metrics summary."""
    drift: DriftMetrics
    performance: PerformanceMetrics
    timestamp: str
