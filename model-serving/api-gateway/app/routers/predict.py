"""
Real-time prediction endpoints.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Header

from app.config import settings
from app.models.requests import PredictRequest, BatchSyncRequest, ModelStage
from app.models.responses import (
    PredictResponse,
    BatchSyncResponse,
    ConfidenceLevel,
    PredictionLabel,
)
from app.dependencies import ServiceClients, get_clients

logger = logging.getLogger(__name__)

router = APIRouter()


def get_confidence_level(probability: float) -> ConfidenceLevel:
    """Determine confidence level from probability."""
    if probability >= 0.9 or probability <= 0.1:
        return ConfidenceLevel.HIGH
    elif probability >= 0.7 or probability <= 0.3:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


# Classification threshold - adjusted higher due to model bias on Enron dataset
SPAM_THRESHOLD = 0.7


def get_prediction_label(spam_probability: float) -> PredictionLabel:
    """Get prediction label from probability."""
    return PredictionLabel.SPAM if spam_probability >= SPAM_THRESHOLD else PredictionLabel.HAM


async def extract_features(
    email: PredictRequest,
    feature_transformer_url: str
) -> tuple[List[float], Optional[Dict[str, Any]]]:
    """
    Extract features from email using feature transformer service.
    
    Returns:
        Tuple of (feature_vector, feature_dict) where feature_dict contains
        named features for logging purposes.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{feature_transformer_url}/transform",
            json={
                "email_id": email.email_id,
                "subject": email.subject,
                "body": email.body,
                "sender": email.sender
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Build feature_dict from feature_names and features
        features = result["features"]
        feature_names = result.get("feature_names", [])
        
        if feature_names and len(feature_names) == len(features):
            feature_dict = {name: float(val) for name, val in zip(feature_names, features)}
        else:
            # Fallback to indexed names
            feature_dict = {f"f{i}": float(v) for i, v in enumerate(features[:20])}
        
        return features, feature_dict


@router.post("", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    clients: ServiceClients = Depends(get_clients),
    x_environment: Optional[str] = Header(None, alias="X-Environment")
):
    """
    Make a single prediction for an email.
    
    This endpoint:
    1. Extracts features using the feature transformer
    2. Gets Feast features for sender domain
    3. Makes prediction via KServe/Triton
    
    Use X-Environment header to switch between staging/production:
    - X-Environment: staging (default)
    - X-Environment: production
    
    Returns spam/ham classification with probability and confidence.
    """
    # Determine environment from header (default to staging)
    environment = (x_environment or settings.DEFAULT_ENVIRONMENT).lower()
    if environment not in ["staging", "production"]:
        environment = "staging"
    model_stage = "Production" if environment == "production" else "Staging"
    start_time = time.perf_counter()
    inference_id = str(uuid.uuid4())
    
    try:
        # Extract features
        features, feature_dict = await extract_features(
            request,
            settings.FEATURE_TRANSFORMER_URL
        )
        
        # Get model version info
        model_version = await clients.kserve.get_model_version(model_stage)
        
        # Make prediction using selected environment
        result = await clients.kserve.predict(
            features=features,
            model_stage=model_stage
        )
        
        # Parse prediction result
        raw_output = result.get("raw_output", [])
        spam_probability = float(raw_output[0]) if raw_output else 0.5
        prediction_label = get_prediction_label(spam_probability)
        
        # Calculate total latency
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Log inference for monitoring (non-blocking)
        if clients.inference_logger:
            try:
                # feature_dict now always has named features from transformer
                await clients.inference_logger.log_prediction(
                    inference_id=inference_id,
                    model_name=settings.MODEL_NAME,
                    model_version=model_version or "unknown",
                    features=feature_dict,
                    prediction=prediction_label.value,
                    spam_probability=spam_probability,
                    latency_ms=total_latency_ms,
                )
                logger.debug(f"Logged inference {inference_id}")
            except Exception as e:
                # Don't fail prediction if logging fails
                logger.warning(f"Failed to log inference: {e}")
        else:
            logger.debug("Inference logger not available")
        
        return PredictResponse(
            email_id=request.email_id,
            prediction=prediction_label,
            spam_probability=round(spam_probability, 4),
            confidence=get_confidence_level(spam_probability),
            model_version=model_version or "unknown",
            model_stage=model_stage,
            latency_ms=round(total_latency_ms, 2),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"FEATURE_EXTRACTION_FAILED: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PREDICTION_FAILED: {str(e)}"
        )


@router.post("/batch-sync", response_model=BatchSyncResponse)
async def predict_batch_sync(
    request: BatchSyncRequest,
    clients: ServiceClients = Depends(get_clients)
):
    """
    Make predictions for a small batch of emails (max 100).
    
    This is a synchronous endpoint - it will wait for all predictions
    to complete before returning.
    
    For larger batches (>100 emails), use the async batch endpoint.
    """
    start_time = time.perf_counter()
    
    predictions: List[PredictResponse] = []
    errors: List[str] = []
    
    # Get model info once
    model_version = await clients.kserve.get_model_version(
        request.model_stage.value
    )
    
    for email in request.emails:
        email_start = time.perf_counter()
        inference_id = str(uuid.uuid4())
        
        try:
            # Extract features
            features, feature_dict = await extract_features(
                email,
                settings.FEATURE_TRANSFORMER_URL
            )
            
            # Make prediction
            result = await clients.kserve.predict(
                features=features,
                model_stage=request.model_stage.value
            )
            
            # Parse result
            raw_output = result.get("raw_output", [])
            spam_probability = float(raw_output[0]) if raw_output else 0.5
            prediction_label = get_prediction_label(spam_probability)
            
            email_latency = (time.perf_counter() - email_start) * 1000
            
            # Log inference for monitoring (non-blocking)
            if clients.inference_logger:
                try:
                    await clients.inference_logger.log_prediction(
                        inference_id=inference_id,
                        model_name=settings.MODEL_NAME,
                        model_version=model_version or "unknown",
                        features=feature_dict,
                        prediction=prediction_label.value,
                        spam_probability=spam_probability,
                        latency_ms=email_latency,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log inference: {e}")
            
            predictions.append(PredictResponse(
                email_id=email.email_id,
                prediction=prediction_label,
                spam_probability=round(spam_probability, 4),
                confidence=get_confidence_level(spam_probability),
                model_version=model_version or "unknown",
                model_stage=request.model_stage.value,
                latency_ms=round(email_latency, 2),
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
            
        except Exception as e:
            logger.error(f"Prediction failed for email {email.email_id}: {e}")
            errors.append(f"{email.email_id}: {str(e)}")
    
    total_latency = (time.perf_counter() - start_time) * 1000
    
    return BatchSyncResponse(
        predictions=predictions,
        total_count=len(request.emails),
        success_count=len(predictions),
        error_count=len(errors),
        total_latency_ms=round(total_latency, 2),
        avg_latency_ms=round(total_latency / len(request.emails), 2) if request.emails else 0,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.post("/staging", response_model=PredictResponse)
async def predict_staging(
    request: PredictRequest,
    clients: ServiceClients = Depends(get_clients)
):
    """
    Make a prediction using the staging model.
    
    Useful for A/B testing or validating new model versions.
    """
    start_time = time.perf_counter()
    inference_id = str(uuid.uuid4())
    
    try:
        # Extract features
        features, feature_dict = await extract_features(
            request,
            settings.FEATURE_TRANSFORMER_URL
        )
        
        # Get staging model version
        model_version = await clients.kserve.get_model_version("Staging")
        
        # Make prediction against staging
        result = await clients.kserve.predict(
            features=features,
            model_stage="Staging"
        )
        
        # Parse prediction result
        raw_output = result.get("raw_output", [])
        spam_probability = float(raw_output[0]) if raw_output else 0.5
        prediction_label = get_prediction_label(spam_probability)
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Log inference for monitoring (non-blocking)
        if clients.inference_logger:
            try:
                await clients.inference_logger.log_prediction(
                    inference_id=inference_id,
                    model_name=settings.MODEL_NAME,
                    model_version=model_version or "unknown",
                    features=feature_dict,
                    prediction=prediction_label.value,
                    spam_probability=spam_probability,
                    latency_ms=total_latency_ms,
                    metadata={"model_stage": "Staging"}
                )
            except Exception as e:
                logger.warning(f"Failed to log inference: {e}")
        
        return PredictResponse(
            email_id=request.email_id,
            prediction=prediction_label,
            spam_probability=round(spam_probability, 4),
            confidence=get_confidence_level(spam_probability),
            model_version=model_version or "unknown",
            model_stage="Staging",
            latency_ms=round(total_latency_ms, 2),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Staging prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"STAGING_PREDICTION_FAILED: {str(e)}"
        )
