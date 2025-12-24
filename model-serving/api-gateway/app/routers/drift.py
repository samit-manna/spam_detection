"""
Drift Metrics API Router

FastAPI router for drift detection endpoints.
Self-contained module with all models, service, and router.

Usage:
    from app.routers.drift import router as drift_router
    app.include_router(drift_router, prefix="/metrics/drift", tags=["Drift Monitoring"])
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential


logger = logging.getLogger(__name__)


# =============================================================================
# Schema Models
# =============================================================================

class DriftSeverity(str, Enum):
    """Drift alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FeatureDriftResult(BaseModel):
    """Drift detection result for a single feature."""
    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
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
    missing_values_pct: float = 0.0
    out_of_range_pct: float = 0.0
    null_features_count: int = 0
    issues_detected: bool = False


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
    """Complete drift detection report."""
    report_id: str
    generated_at: datetime
    model_name: str
    model_version: str
    baseline_version: str
    
    analysis_window: AnalysisWindow
    
    overall_drift_detected: bool
    drift_score: float
    
    feature_drift: Dict[str, FeatureDriftResult] = Field(default_factory=dict)
    prediction_drift: Optional[PredictionDriftResult] = None
    data_quality: DataQualityResult = Field(default_factory=DataQualityResult)
    
    alerts: List[DriftAlert] = Field(default_factory=list)
    
    total_features: int = 0
    features_drifted_count: int = 0
    features_drifted_names: List[str] = Field(default_factory=list)
    
    evidently_report_path: Optional[str] = None
    baseline_path: str = ""
    inference_logs_path: str = ""


# =============================================================================
# API Response Models
# =============================================================================

class DriftSummaryResponse(BaseModel):
    """Summary drift response for quick health check."""
    
    model_name: str
    model_version: str
    last_checked: datetime
    drift_detected: bool
    drift_score: float = Field(ge=0.0, le=1.0)
    features_drifted: List[str] = Field(default_factory=list)
    alerts_count: int
    samples_analyzed: int
    details_url: str = "/metrics/drift/details"
    report_url: Optional[str] = None


class DriftDetailsResponse(BaseModel):
    """Detailed drift response with full report."""
    
    report_id: str
    generated_at: datetime
    model_name: str
    model_version: str
    baseline_version: str
    
    analysis_start: datetime
    analysis_end: datetime
    num_samples: int
    
    overall_drift_detected: bool
    drift_score: float
    
    feature_drift: Dict[str, FeatureDriftResult] = Field(default_factory=dict)
    total_features: int
    features_drifted_count: int
    features_drifted_names: List[str] = Field(default_factory=list)
    
    prediction_drift: Optional[PredictionDriftResult] = None
    data_quality: DataQualityResult
    
    alerts: List[DriftAlert] = Field(default_factory=list)
    evidently_report_path: Optional[str] = None


class DriftHistoryPoint(BaseModel):
    """Single point in drift history."""
    timestamp: datetime
    drift_score: float
    drift_detected: bool
    features_drifted_count: int
    samples_analyzed: int


class DriftHistoryResponse(BaseModel):
    """Historical drift data response."""
    
    model_name: str
    period_days: int
    history: List[DriftHistoryPoint] = Field(default_factory=list)
    
    avg_drift_score: float
    max_drift_score: float
    drift_detected_count: int
    total_reports: int


class FeatureDriftDetailsResponse(BaseModel):
    """Drift details for a specific feature."""
    
    feature_name: str
    drift_detected: bool
    psi: float
    ks_statistic: float
    ks_pvalue: float
    baseline_mean: Optional[float]
    baseline_std: Optional[float]
    current_mean: Optional[float]
    current_std: Optional[float]
    mean_shift: Optional[float]
    
    history: List[dict] = Field(default_factory=list)


class DriftStatusResponse(BaseModel):
    """Simple drift status for health checks."""
    
    status: str  # "healthy", "warning", "critical", "unknown"
    drift_detected: bool
    drift_score: float
    last_checked: Optional[datetime]
    message: str


# =============================================================================
# Drift Service
# =============================================================================

class DriftService:
    """
    Service for fetching drift reports and metrics from Azure Blob Storage.
    """
    
    def __init__(
        self,
        storage_account_name: Optional[str] = None,
        storage_account_key: Optional[str] = None,
        container_name: Optional[str] = None,
        model_name: str = "spam-detector",
        reports_path: str = "drift-reports",
    ):
        self.storage_account_name = storage_account_name or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        self.storage_account_key = storage_account_key or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = container_name or os.environ.get("AZURE_STORAGE_CONTAINER", "data")
        self.model_name = model_name
        self.reports_path = reports_path
        
        self._blob_client: Optional[BlobServiceClient] = None
        self._latest_report: Optional[DriftReport] = None
        self._latest_report_time: Optional[datetime] = None
        
        # Cache TTL in seconds
        self.cache_ttl = int(os.environ.get("DRIFT_CACHE_TTL_SECONDS", "300"))
    
    def _get_blob_client(self) -> BlobServiceClient:
        """Get or create blob service client."""
        if self._blob_client is None:
            if not self.storage_account_name:
                raise ValueError("Azure storage account name is required")
            
            if self.storage_account_key:
                connection_string = (
                    f"DefaultEndpointsProtocol=https;"
                    f"AccountName={self.storage_account_name};"
                    f"AccountKey={self.storage_account_key};"
                    f"EndpointSuffix=core.windows.net"
                )
                self._blob_client = BlobServiceClient.from_connection_string(connection_string)
            else:
                credential = DefaultAzureCredential()
                account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
                self._blob_client = BlobServiceClient(account_url, credential=credential)
        
        return self._blob_client
    
    def _download_json(self, blob_path: str) -> Optional[Dict[str, Any]]:
        """Download and parse JSON from blob storage."""
        try:
            client = self._get_blob_client()
            container = client.get_container_client(self.container_name)
            blob = container.get_blob_client(blob_path)
            
            if not blob.exists():
                return None
            
            data = blob.download_blob().readall()
            return json.loads(data.decode("utf-8"))
            
        except Exception as e:
            logger.error(f"Failed to download {blob_path}: {e}")
            return None
    
    def _list_reports(self, days: int = 7) -> List[str]:
        """List drift report paths for recent days."""
        try:
            client = self._get_blob_client()
            container = client.get_container_client(self.container_name)
            
            reports = []
            now = datetime.now(timezone.utc)
            
            for i in range(days):
                date = now - timedelta(days=i)
                prefix = (
                    f"{self.reports_path}/{self.model_name}/"
                    f"{date.year}/{date.month:02d}/{date.day:02d}/"
                )
                
                for blob in container.list_blobs(name_starts_with=prefix):
                    if blob.name.endswith(".json") and "latest" not in blob.name:
                        reports.append(blob.name)
            
            return sorted(reports, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list reports: {e}")
            return []
    
    def get_latest_report(self, force_refresh: bool = False) -> Optional[DriftReport]:
        """Get the latest drift report with caching."""
        now = datetime.now(timezone.utc)
        
        # Check cache
        if not force_refresh and self._latest_report and self._latest_report_time:
            age = (now - self._latest_report_time).total_seconds()
            if age < self.cache_ttl:
                return self._latest_report
        
        # Fetch latest report
        latest_path = f"{self.reports_path}/{self.model_name}/latest.json"
        data = self._download_json(latest_path)
        
        if data:
            try:
                self._latest_report = DriftReport(**data)
                self._latest_report_time = now
                return self._latest_report
            except Exception as e:
                logger.error(f"Failed to parse drift report: {e}")
        
        return None
    
    async def get_drift_summary(self) -> DriftSummaryResponse:
        """Get drift summary for API response."""
        report = self.get_latest_report()
        
        if not report:
            return DriftSummaryResponse(
                model_name=self.model_name,
                model_version="unknown",
                last_checked=datetime.now(timezone.utc),
                drift_detected=False,
                drift_score=0.0,
                features_drifted=[],
                alerts_count=0,
                samples_analyzed=0,
                details_url="/metrics/drift/details",
                report_url=None,
            )
        
        return DriftSummaryResponse(
            model_name=report.model_name,
            model_version=report.model_version,
            last_checked=report.generated_at,
            drift_detected=report.overall_drift_detected,
            drift_score=report.drift_score,
            features_drifted=report.features_drifted_names[:10],
            alerts_count=len(report.alerts),
            samples_analyzed=report.analysis_window.num_samples,
            details_url="/metrics/drift/details",
            report_url=report.evidently_report_path,
        )
    
    async def get_drift_details(self) -> Optional[DriftDetailsResponse]:
        """Get full drift report details."""
        report = self.get_latest_report()
        
        if not report:
            return None
        
        return DriftDetailsResponse(
            report_id=report.report_id,
            generated_at=report.generated_at,
            model_name=report.model_name,
            model_version=report.model_version,
            baseline_version=report.baseline_version,
            analysis_start=report.analysis_window.start,
            analysis_end=report.analysis_window.end,
            num_samples=report.analysis_window.num_samples,
            overall_drift_detected=report.overall_drift_detected,
            drift_score=report.drift_score,
            feature_drift=report.feature_drift,
            total_features=report.total_features,
            features_drifted_count=report.features_drifted_count,
            features_drifted_names=report.features_drifted_names,
            prediction_drift=report.prediction_drift,
            data_quality=report.data_quality,
            alerts=report.alerts,
            evidently_report_path=report.evidently_report_path,
        )
    
    async def get_drift_history(self, days: int = 7) -> DriftHistoryResponse:
        """Get drift history for specified number of days."""
        report_paths = self._list_reports(days=days)
        
        history = []
        for path in report_paths[:100]:
            data = self._download_json(path)
            if data:
                try:
                    history.append(DriftHistoryPoint(
                        timestamp=datetime.fromisoformat(data.get("generated_at", "").replace("Z", "+00:00")),
                        drift_score=data.get("drift_score", 0.0),
                        drift_detected=data.get("overall_drift_detected", False),
                        features_drifted_count=data.get("features_drifted_count", 0),
                        samples_analyzed=data.get("analysis_window", {}).get("num_samples", 0),
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse report {path}: {e}")
        
        drift_scores = [h.drift_score for h in history]
        drift_detected_count = sum(1 for h in history if h.drift_detected)
        
        return DriftHistoryResponse(
            model_name=self.model_name,
            period_days=days,
            history=history,
            avg_drift_score=sum(drift_scores) / len(drift_scores) if drift_scores else 0.0,
            max_drift_score=max(drift_scores) if drift_scores else 0.0,
            drift_detected_count=drift_detected_count,
            total_reports=len(history),
        )
    
    async def get_drift_status(self) -> DriftStatusResponse:
        """Get simple drift status for health checks."""
        report = self.get_latest_report()
        
        if not report:
            return DriftStatusResponse(
                status="unknown",
                drift_detected=False,
                drift_score=0.0,
                last_checked=None,
                message="No drift reports available",
            )
        
        if report.drift_score >= 0.5:
            status = "critical"
            message = f"Critical drift detected (score: {report.drift_score:.1%})"
        elif report.overall_drift_detected:
            status = "warning"
            message = f"Drift detected (score: {report.drift_score:.1%}, {report.features_drifted_count} features)"
        else:
            status = "healthy"
            message = "No significant drift detected"
        
        return DriftStatusResponse(
            status=status,
            drift_detected=report.overall_drift_detected,
            drift_score=report.drift_score,
            last_checked=report.generated_at,
            message=message,
        )
    
    async def get_feature_drift_details(self, feature_name: str) -> Optional[FeatureDriftDetailsResponse]:
        """Get drift details for a specific feature."""
        report = self.get_latest_report()
        
        if not report or feature_name not in report.feature_drift:
            return None
        
        feature_result = report.feature_drift[feature_name]
        
        # Get historical data for this feature
        history = []
        report_paths = self._list_reports(days=7)
        
        for path in report_paths[:20]:
            data = self._download_json(path)
            if data:
                feature_drift = data.get("feature_drift", {}).get(feature_name)
                if feature_drift:
                    history.append({
                        "timestamp": data.get("generated_at"),
                        "psi": feature_drift.get("psi", 0),
                        "drift_detected": feature_drift.get("drift_detected", False),
                    })
        
        return FeatureDriftDetailsResponse(
            feature_name=feature_name,
            drift_detected=feature_result.drift_detected,
            psi=feature_result.psi,
            ks_statistic=feature_result.ks_statistic,
            ks_pvalue=feature_result.ks_pvalue,
            baseline_mean=feature_result.baseline_mean,
            baseline_std=None,
            current_mean=feature_result.current_mean,
            current_std=None,
            mean_shift=feature_result.mean_shift,
            history=history,
        )


# =============================================================================
# Singleton Service Instance
# =============================================================================

_drift_service: Optional[DriftService] = None


def get_drift_service() -> DriftService:
    """Get or create the drift service singleton."""
    global _drift_service
    if _drift_service is None:
        _drift_service = DriftService()
    return _drift_service


def init_drift_service(
    storage_account_name: Optional[str] = None,
    storage_account_key: Optional[str] = None,
    model_name: str = "spam-detector",
) -> DriftService:
    """Initialize the drift service with custom configuration."""
    global _drift_service
    _drift_service = DriftService(
        storage_account_name=storage_account_name,
        storage_account_key=storage_account_key,
        model_name=model_name,
    )
    return _drift_service


# =============================================================================
# FastAPI Router
# =============================================================================

router = APIRouter()


@router.get(
    "",
    response_model=DriftSummaryResponse,
    summary="Get drift summary",
    description="Returns a summary of the latest drift detection results.",
)
async def get_drift_summary():
    """
    Get drift summary.
    
    Returns the latest drift detection summary including:
    - Overall drift score
    - List of drifted features
    - Number of alerts
    - Link to detailed report
    """
    try:
        service = get_drift_service()
        return await service.get_drift_summary()
    except Exception as e:
        logger.error(f"Failed to get drift summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/details",
    response_model=DriftDetailsResponse,
    summary="Get detailed drift report",
    description="Returns the full drift detection report with all metrics.",
)
async def get_drift_details():
    """
    Get detailed drift report.
    
    Returns the complete drift detection report including:
    - Individual feature drift metrics (PSI, KS test)
    - Prediction drift metrics
    - Data quality metrics
    - All alerts with recommendations
    """
    try:
        service = get_drift_service()
        result = await service.get_drift_details()
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="No drift report available"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get drift details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history",
    response_model=DriftHistoryResponse,
    summary="Get drift history",
    description="Returns historical drift scores over a specified period.",
)
async def get_drift_history(
    days: int = Query(
        default=7,
        ge=1,
        le=90,
        description="Number of days of history to retrieve"
    ),
):
    """
    Get drift history.
    
    Returns historical drift detection results for the specified period.
    Useful for tracking drift trends over time.
    """
    try:
        service = get_drift_service()
        return await service.get_drift_history(days=days)
    except Exception as e:
        logger.error(f"Failed to get drift history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/status",
    response_model=DriftStatusResponse,
    summary="Get drift status",
    description="Returns simple drift status for health checks.",
)
async def get_drift_status():
    """
    Get drift status.
    
    Returns a simple status indicator for health monitoring:
    - "healthy": No significant drift
    - "warning": Drift detected, monitoring recommended
    - "critical": Significant drift, action required
    """
    try:
        service = get_drift_service()
        return await service.get_drift_status()
    except Exception as e:
        logger.error(f"Failed to get drift status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/feature/{feature_name}",
    response_model=FeatureDriftDetailsResponse,
    summary="Get feature drift details",
    description="Returns drift details for a specific feature.",
)
async def get_feature_drift(feature_name: str):
    """
    Get drift details for a specific feature.
    
    Returns detailed drift metrics and history for the specified feature.
    """
    try:
        service = get_drift_service()
        result = await service.get_feature_drift_details(feature_name)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Feature '{feature_name}' not found in drift report"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature drift for {feature_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/refresh",
    summary="Refresh drift data",
    description="Force refresh of cached drift data.",
)
async def refresh_drift_data():
    """
    Force refresh drift data.
    
    Clears the cache and fetches the latest drift report from storage.
    """
    try:
        service = get_drift_service()
        report = service.get_latest_report(force_refresh=True)
        
        return {
            "status": "refreshed",
            "report_available": report is not None,
            "report_id": report.report_id if report else None,
        }
    except Exception as e:
        logger.error(f"Failed to refresh drift data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
