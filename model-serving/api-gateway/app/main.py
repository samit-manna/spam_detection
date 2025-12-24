"""
FastAPI Management API - Model Serving Gateway

This service acts as the gateway for model serving operations including:
- Model information (MLflow read-only)
- Real-time inference (KServe)
- Batch inference (Ray)
- Monitoring and metrics

Note: Training operations (Kubeflow, MLflow management) are accessible
via separate internal Istio gateways.
"""

import asyncio
import logging
import os
import signal
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app, Counter, Histogram

from app.config import settings
from app.routers import health, models, predict, batch, metrics
from app.middleware.auth import AuthMiddleware
from app.middleware.logging import LoggingMiddleware, setup_logging
from app.dependencies import ServiceClients, clients, get_clients
from app.models.responses import ErrorResponse, ErrorDetail

# Import drift router (self-contained in api-gateway)
try:
    from app.routers.drift import router as drift_router, init_drift_service
    DRIFT_ROUTER_AVAILABLE = True
except ImportError:
    DRIFT_ROUTER_AVAILABLE = False
    drift_router = None
    init_drift_service = None

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting ML Platform API Gateway...")
    await clients.initialize()
    
    # Initialize drift service if available
    if DRIFT_ROUTER_AVAILABLE and init_drift_service:
        init_drift_service(
            storage_account_name=settings.AZURE_STORAGE_ACCOUNT_NAME,
            storage_account_key=settings.AZURE_STORAGE_ACCOUNT_KEY,
            model_name=settings.MODEL_NAME,
        )
        logger.info("Drift monitoring service initialized")
    
    # Setup graceful shutdown handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(graceful_shutdown(s))
        )
    
    yield
    
    # Shutdown
    await clients.shutdown()
    logger.info("ML Platform API Gateway stopped")


async def graceful_shutdown(sig: signal.Signals):
    """Handle graceful shutdown on signal."""
    logger.info(f"Received {sig.name}, initiating graceful shutdown...")
    await clients.shutdown()


# Create FastAPI application
app = FastAPI(
    title="ML Serving API",
    description="""
    Model Serving API for the Spam Detection ML Platform.
    
    ## Features
    
    - **Real-time Inference**: Single and small batch predictions via KServe/Triton
    - **Batch Inference**: Large-scale async predictions via Ray
    - **Model Info**: Current deployed model information
    - **Monitoring**: Latency, throughput, and drift metrics
    
    ## Authentication
    
    All endpoints require API key authentication via `X-API-Key` header.
    
    Roles:
    - `viewer`: GET endpoints only
    - `operator`: GET + predict endpoints
    """,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(predict.router, prefix="/predict", tags=["Inference"])
app.include_router(batch.router, prefix="/predict/batch", tags=["Batch Inference"])
app.include_router(metrics.router, prefix="/metrics", tags=["Monitoring"])

# Include drift monitoring router (if available)
if DRIFT_ROUTER_AVAILABLE and drift_router:
    app.include_router(drift_router, prefix="/metrics/drift", tags=["Drift Monitoring"])


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=exc.detail if isinstance(exc.detail, str) else "HTTP_ERROR",
            message=str(exc.detail),
            details={"status_code": exc.status_code}
        ),
        correlation_id=correlation_id,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    logger.exception(f"Unhandled exception [correlation_id={correlation_id}]: {exc}")
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__} if settings.DEBUG else {}
        ),
        correlation_id=correlation_id,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


@app.get("/info")
async def get_info(clients: ServiceClients = Depends(get_clients)):
    """Get API version and model information."""
    model_info = await clients.mlflow.get_latest_model_info(settings.MODEL_NAME)
    
    return {
        "api_version": settings.API_VERSION,
        "model_name": settings.MODEL_NAME,
        "model_info": model_info,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )
