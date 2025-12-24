"""
Health check endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends

from app.config import settings
from app.models.responses import HealthResponse, ServiceHealth
from app.dependencies import ServiceClients, get_clients

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(clients: ServiceClients = Depends(get_clients)):
    """
    Comprehensive health check for all serving services.
    
    Checks connectivity to:
    - MLflow (model registry)
    - KServe (model inference)
    - Ray (batch processing)
    - Redis (caching)
    """
    services: List[ServiceHealth] = []
    overall_status = "healthy"
    
    # Check MLflow
    mlflow_health = await clients.mlflow.health_check()
    services.append(ServiceHealth(
        name="mlflow",
        status=mlflow_health.get("status", "unknown"),
        latency_ms=mlflow_health.get("latency_ms"),
        message=mlflow_health.get("error")
    ))
    
    # Check KServe
    kserve_health = await clients.kserve.health_check()
    services.append(ServiceHealth(
        name="kserve",
        status=kserve_health.get("status", "unknown"),
        message=kserve_health.get("error")
    ))
    
    # Check Ray
    ray_health = await clients.ray.health_check()
    services.append(ServiceHealth(
        name="ray",
        status=ray_health.get("status", "unknown"),
        message=ray_health.get("error")
    ))
    
    # Check Redis
    redis_health = await clients.redis.health_check()
    services.append(ServiceHealth(
        name="redis",
        status=redis_health.get("status", "unknown"),
        latency_ms=redis_health.get("latency_ms"),
        message=redis_health.get("error")
    ))
    
    # Determine overall status
    unhealthy_services = [s for s in services if s.status != "healthy"]
    if len(unhealthy_services) > 0:
        if len(unhealthy_services) == len(services):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services
    )


@router.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe.
    
    Simple check that the service is running.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe(clients: ServiceClients = Depends(get_clients)):
    """
    Kubernetes readiness probe.
    
    Checks that essential services are available.
    """
    # Check if we're shutting down
    if clients.is_shutting_down():
        return {"status": "not_ready", "reason": "shutting_down"}
    
    # Check essential services (MLflow and KServe)
    mlflow_ok = (await clients.mlflow.health_check()).get("status") == "healthy"
    kserve_ok = (await clients.kserve.health_check()).get("status") == "healthy"
    
    if mlflow_ok and kserve_ok:
        return {"status": "ready"}
    
    return {
        "status": "not_ready",
        "mlflow": "ok" if mlflow_ok else "unavailable",
        "kserve": "ok" if kserve_ok else "unavailable"
    }
