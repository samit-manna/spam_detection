"""
Monitoring and metrics endpoints.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import settings
from app.models.responses import (
    DriftMetrics,
    PerformanceMetrics,
    MetricsSummary,
)
from app.dependencies import ServiceClients, get_clients

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/drift", response_model=DriftMetrics)
async def get_drift_metrics(
    clients: ServiceClients = Depends(get_clients),
    window_hours: int = Query(24, ge=1, le=168, description="Analysis window in hours")
):
    """
    Get latest drift detection scores.
    
    Returns feature drift, prediction drift, and data quality metrics.
    """
    try:
        # Get drift metrics from Redis cache
        drift_data = await clients.redis.hgetall("metrics:drift:latest")
        
        if not drift_data:
            # Return default values if no drift data available
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(hours=window_hours)
            
            return DriftMetrics(
                feature_drift_score=0.0,
                prediction_drift_score=0.0,
                data_quality_score=1.0,
                drift_detected=False,
                drifted_features=[],
                last_checked=now.isoformat(),
                reference_window=f"{window_start.date()} to {(window_start + timedelta(hours=window_hours//2)).date()}",
                current_window=f"{(now - timedelta(hours=window_hours//2)).date()} to {now.date()}"
            )
        
        # Parse stored drift data
        drifted_features = drift_data.get("drifted_features", "").split(",")
        drifted_features = [f for f in drifted_features if f]
        
        return DriftMetrics(
            feature_drift_score=float(drift_data.get("feature_drift_score", 0.0)),
            prediction_drift_score=float(drift_data.get("prediction_drift_score", 0.0)),
            data_quality_score=float(drift_data.get("data_quality_score", 1.0)),
            drift_detected=drift_data.get("drift_detected", "false").lower() == "true",
            drifted_features=drifted_features,
            last_checked=drift_data.get("last_checked", datetime.now(timezone.utc).isoformat()),
            reference_window=drift_data.get("reference_window", "N/A"),
            current_window=drift_data.get("current_window", "N/A")
        )
        
    except Exception as e:
        logger.error(f"Failed to get drift metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"DRIFT_METRICS_FAILED: {str(e)}"
        )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    clients: ServiceClients = Depends(get_clients),
    window_hours: int = Query(24, ge=1, le=168, description="Metrics window in hours")
):
    """
    Get model performance metrics.
    
    Returns latency percentiles, throughput, and error rates.
    """
    try:
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=window_hours)
        
        # Get performance metrics from Redis
        perf_data = await clients.redis.hgetall("metrics:performance:latest")
        
        # Get model version
        model_version = await clients.kserve.get_model_version("Production")
        
        if not perf_data:
            # Return default values
            return PerformanceMetrics(
                model_version=model_version or "unknown",
                model_stage="Production",
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                avg_latency_ms=0.0,
                requests_per_second=0.0,
                total_requests_24h=0,
                error_rate=0.0,
                timeout_rate=0.0,
                window_start=window_start.isoformat(),
                window_end=now.isoformat()
            )
        
        return PerformanceMetrics(
            model_version=model_version or perf_data.get("model_version", "unknown"),
            model_stage="Production",
            p50_latency_ms=float(perf_data.get("p50_latency_ms", 0.0)),
            p95_latency_ms=float(perf_data.get("p95_latency_ms", 0.0)),
            p99_latency_ms=float(perf_data.get("p99_latency_ms", 0.0)),
            avg_latency_ms=float(perf_data.get("avg_latency_ms", 0.0)),
            requests_per_second=float(perf_data.get("requests_per_second", 0.0)),
            total_requests_24h=int(perf_data.get("total_requests_24h", 0)),
            error_rate=float(perf_data.get("error_rate", 0.0)),
            timeout_rate=float(perf_data.get("timeout_rate", 0.0)),
            window_start=perf_data.get("window_start", window_start.isoformat()),
            window_end=perf_data.get("window_end", now.isoformat())
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PERFORMANCE_METRICS_FAILED: {str(e)}"
        )


@router.get("/summary", response_model=MetricsSummary)
async def get_metrics_summary(
    clients: ServiceClients = Depends(get_clients)
):
    """
    Get combined drift and performance metrics summary.
    """
    try:
        drift = await get_drift_metrics(clients=clients)
        performance = await get_performance_metrics(clients=clients)
        
        return MetricsSummary(
            drift=drift,
            performance=performance,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"METRICS_SUMMARY_FAILED: {str(e)}"
        )


@router.get("/latency/history")
async def get_latency_history(
    clients: ServiceClients = Depends(get_clients),
    hours: int = Query(24, ge=1, le=168, description="History in hours"),
    resolution: str = Query("1h", description="Resolution: 1m, 5m, 1h")
):
    """
    Get historical latency metrics for charting.
    """
    try:
        # Get latency history from Redis
        metrics = await clients.redis.get_metrics("latency")
        
        return {
            "metric": "latency",
            "unit": "ms",
            "resolution": resolution,
            "data": metrics[:hours * 60] if resolution == "1m" else metrics[:hours]
        }
        
    except Exception as e:
        logger.error(f"Failed to get latency history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LATENCY_HISTORY_FAILED: {str(e)}"
        )


@router.get("/predictions/distribution")
async def get_prediction_distribution(
    clients: ServiceClients = Depends(get_clients),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours")
):
    """
    Get distribution of predictions (spam vs ham) over time.
    """
    try:
        # Get prediction counts from Redis
        spam_count = await clients.redis.get_counter(f"predictions:spam:{hours}h")
        ham_count = await clients.redis.get_counter(f"predictions:ham:{hours}h")
        total = spam_count + ham_count
        
        return {
            "window_hours": hours,
            "total_predictions": total,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "spam_percentage": round(spam_count / total * 100, 2) if total > 0 else 0,
            "ham_percentage": round(ham_count / total * 100, 2) if total > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get prediction distribution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"DISTRIBUTION_FAILED: {str(e)}"
        )


@router.post("/refresh")
async def refresh_metrics(
    clients: ServiceClients = Depends(get_clients)
):
    """
    Trigger a metrics refresh from all sources.
    
    Requires `operator` role.
    """
    try:
        # This would typically trigger a background job to:
        # 1. Calculate fresh drift scores
        # 2. Aggregate performance metrics
        # 3. Update Redis cache
        
        logger.info("Metrics refresh triggered")
        
        return {
            "status": "refresh_initiated",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"REFRESH_FAILED: {str(e)}"
        )
