"""
Pydantic models for API requests (serving-only).
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ModelStage(str, Enum):
    """Model stages for inference."""
    STAGING = "Staging"
    PRODUCTION = "Production"


# ============================================================================
# Prediction Requests
# ============================================================================

class PredictRequest(BaseModel):
    """Single prediction request."""
    email_id: str = Field(..., description="Unique identifier for the email")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    sender: str = Field(..., description="Sender email address")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "email_id": "abc123",
                "subject": "URGENT: You've won $1,000,000!",
                "body": "Click here to claim your prize now...",
                "sender": "winner@lottery-prize.com"
            }
        }
    }


class BatchSyncRequest(BaseModel):
    """Synchronous batch prediction request (small batches < 100)."""
    emails: List[PredictRequest] = Field(
        ..., 
        max_length=100,
        description="List of emails to predict (max 100)"
    )
    model_stage: ModelStage = Field(
        default=ModelStage.PRODUCTION,
        description="Model stage to use for prediction"
    )


class BatchAsyncRequest(BaseModel):
    """Asynchronous batch prediction request (large batches)."""
    input_path: str = Field(
        ...,
        description="Azure Blob Storage path to input data (Parquet format)"
    )
    output_path: str = Field(
        ...,
        description="Azure Blob Storage path for output predictions"
    )
    model_stage: ModelStage = Field(
        default=ModelStage.PRODUCTION,
        description="Model stage to use for prediction"
    )
    ray_resources: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional Ray resource configuration"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "input_path": "abfss://data@storage.dfs.core.windows.net/input/emails_jan.parquet",
                "output_path": "abfss://data@storage.dfs.core.windows.net/output/predictions_jan.parquet",
                "model_stage": "Production"
            }
        }
    }
    
    @field_validator("input_path", "output_path")
    @classmethod
    def validate_azure_path(cls, v: str) -> str:
        """Validate Azure Blob Storage path format."""
        if not (v.startswith("abfss://") or v.startswith("wasbs://") or v.startswith("az://")):
            raise ValueError("Path must be an Azure Blob Storage path (abfss://, wasbs://, or az://)")
        return v
