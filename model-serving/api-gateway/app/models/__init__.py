"""
Pydantic models for API requests and responses.
Serving-only API - no training-related models.
"""

from app.models.requests import (
    ModelStage,
    PredictRequest,
    BatchSyncRequest,
    BatchAsyncRequest,
)

from app.models.responses import (
    # Enums
    ConfidenceLevel,
    PredictionLabel,
    JobStatus,
    
    # Error
    ErrorDetail,
    ErrorResponse,
    
    # Health
    ServiceHealth,
    HealthResponse,
    
    # Prediction
    PredictResponse,
    BatchSyncResponse,
    BatchJobResponse,
    BatchJobStatusResponse,
    BatchJobResultsResponse,
    
    # Metrics
    DriftMetrics,
    PerformanceMetrics,
    MetricsSummary,
)

__all__ = [
    # Request models
    "ModelStage",
    "PredictRequest",
    "BatchSyncRequest",
    "BatchAsyncRequest",
    
    # Response enums
    "ConfidenceLevel",
    "PredictionLabel",
    "JobStatus",
    
    # Error responses
    "ErrorDetail",
    "ErrorResponse",
    
    # Health responses
    "ServiceHealth",
    "HealthResponse",
    
    # Prediction responses
    "PredictResponse",
    "BatchSyncResponse",
    "BatchJobResponse",
    "BatchJobStatusResponse",
    "BatchJobResultsResponse",
    
    # Metrics responses
    "DriftMetrics",
    "PerformanceMetrics",
    "MetricsSummary",
]
