"""
Model information endpoints (serving context only).

Provides information about deployed models - NOT a full model registry interface.
For model management (promote, rollback), use MLflow directly or via Istio gateway.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import settings
from app.dependencies import ServiceClients, get_clients

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/deployed")
async def get_deployed_model(
    clients: ServiceClients = Depends(get_clients)
):
    """
    Get information about the currently deployed production model.
    
    Returns the model version currently serving production traffic.
    """
    try:
        # Get production model info from MLflow
        model_info = await clients.mlflow.get_latest_model_info(
            settings.MODEL_NAME,
            stage="Production"
        )
        
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail="NO_PRODUCTION_MODEL: No production model is currently deployed"
            )
        
        # Get model metadata from KServe
        kserve_version = await clients.kserve.get_model_version("Production")
        
        return {
            "model_name": settings.MODEL_NAME,
            "stage": "Production",
            "version": model_info.get("version"),
            "run_id": model_info.get("run_id"),
            "created_at": model_info.get("creation_timestamp"),
            "kserve_version": kserve_version,
            "status": "serving",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployed model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MODEL_INFO_FAILED: {str(e)}"
        )


@router.get("/staging")
async def get_staging_model(
    clients: ServiceClients = Depends(get_clients)
):
    """
    Get information about the staging model (if deployed).
    
    Returns the model version in staging for A/B testing or validation.
    """
    try:
        # Get staging model info from MLflow
        model_info = await clients.mlflow.get_latest_model_info(
            settings.MODEL_NAME,
            stage="Staging"
        )
        
        if not model_info:
            return {
                "model_name": settings.MODEL_NAME,
                "stage": "Staging",
                "status": "not_deployed",
                "message": "No staging model is currently deployed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Get staging model metadata from KServe
        kserve_version = await clients.kserve.get_model_version("Staging")
        
        return {
            "model_name": settings.MODEL_NAME,
            "stage": "Staging",
            "version": model_info.get("version"),
            "run_id": model_info.get("run_id"),
            "created_at": model_info.get("creation_timestamp"),
            "kserve_version": kserve_version,
            "status": "serving" if kserve_version else "registered",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting staging model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MODEL_INFO_FAILED: {str(e)}"
        )


@router.get("")
async def get_model_summary(
    clients: ServiceClients = Depends(get_clients)
):
    """
    Get summary of all deployed models (production and staging).
    """
    try:
        # Get both production and staging info
        prod_info = await clients.mlflow.get_latest_model_info(
            settings.MODEL_NAME, "Production"
        )
        staging_info = await clients.mlflow.get_latest_model_info(
            settings.MODEL_NAME, "Staging"
        )
        
        return {
            "model_name": settings.MODEL_NAME,
            "production": {
                "version": prod_info.get("version") if prod_info else None,
                "status": "serving" if prod_info else "not_deployed"
            },
            "staging": {
                "version": staging_info.get("version") if staging_info else None,
                "status": "serving" if staging_info else "not_deployed"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MODEL_SUMMARY_FAILED: {str(e)}"
        )
