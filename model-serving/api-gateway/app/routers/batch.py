"""
Batch inference endpoints.
"""

import logging
import os
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import pandas as pd
from io import StringIO

from app.models.requests import BatchAsyncRequest
from app.models.responses import (
    BatchJobResponse,
    BatchJobStatusResponse,
    BatchJobResultsResponse,
    JobStatus,
)
from app.dependencies import ServiceClients, get_clients
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Batch Results Reader
# =============================================================================

def read_parquet_from_blob_sync(
    storage_account_name: str,
    storage_account_key: str,
    container: str,
    blob_path: str
) -> pd.DataFrame:
    """Read parquet file from Azure Blob Storage (synchronous)."""
    from adlfs import AzureBlobFileSystem
    
    fs = AzureBlobFileSystem(
        account_name=storage_account_name,
        account_key=storage_account_key
    )
    
    full_path = f"{container}/{blob_path}"
    logger.info(f"Reading parquet file from: {full_path}")
    
    with fs.open(full_path, "rb") as f:
        df = pd.read_parquet(f)
    
    logger.info(f"Successfully read parquet file with {len(df)} records")
    return df


@router.get("/results/data", summary="Get batch prediction data")
async def get_batch_prediction_data(
    output_path: str = Query(
        ..., 
        description="Azure Blob path to predictions parquet (e.g., predictions/batch_output.parquet)"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    filter_spam: Optional[bool] = Query(None, description="Filter by spam (true) or ham (false)"),
    format: str = Query("json", description="Output format: json, csv, or summary")
):
    """
    Read batch prediction results from Azure Blob Storage.
    
    Returns the prediction data in a human-readable format with:
    - Email ID
    - Subject/Preview
    - Spam probability
    - Spam label (spam/ham)
    
    Use `format=summary` to get aggregated statistics only.
    Use `format=csv` to download as CSV.
    """
    import asyncio
    
    try:
        # Run sync I/O in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            read_parquet_from_blob_sync,
            settings.AZURE_STORAGE_ACCOUNT_NAME,
            settings.AZURE_STORAGE_ACCOUNT_KEY,
            "datasets",
            output_path
        )
        
        # Apply spam/ham filter if specified
        if filter_spam is not None:
            spam_label = 1 if filter_spam else 0
            df = df[df["spam_label"] == spam_label]
        
        total_records = len(df)
        
        # Return summary only
        if format == "summary":
            spam_count = int((df["spam_label"] == 1).sum())
            ham_count = int((df["spam_label"] == 0).sum())
            
            # Calculate probability distribution
            prob_bins = {
                "very_low_0_20": int(len(df[df["spam_probability"] < 0.2])),
                "low_20_40": int(len(df[(df["spam_probability"] >= 0.2) & (df["spam_probability"] < 0.4)])),
                "medium_40_60": int(len(df[(df["spam_probability"] >= 0.4) & (df["spam_probability"] < 0.6)])),
                "high_60_80": int(len(df[(df["spam_probability"] >= 0.6) & (df["spam_probability"] < 0.8)])),
                "very_high_80_100": int(len(df[df["spam_probability"] >= 0.8])),
            }
            
            return {
                "summary": {
                    "total_records": int(total_records),
                    "spam_count": spam_count,
                    "ham_count": ham_count,
                    "spam_percentage": round(float(spam_count / total_records * 100), 2) if total_records > 0 else 0.0,
                    "avg_spam_probability": round(float(df["spam_probability"].mean()), 4) if total_records > 0 else 0.0,
                    "min_spam_probability": round(float(df["spam_probability"].min()), 4) if total_records > 0 else 0.0,
                    "max_spam_probability": round(float(df["spam_probability"].max()), 4) if total_records > 0 else 0.0,
                    "probability_distribution": prob_bins
                },
                "output_path": output_path
            }
        
        # Apply pagination
        df_page = df.iloc[offset:offset + limit]
        
        # Prepare data for response
        records = []
        for _, row in df_page.iterrows():
            # Get email ID from various possible columns
            email_id = str(row.get("message_id", row.get("email_id", row.get("id", ""))))
            # Get subject safely
            subject = str(row.get("subject", ""))[:100] if pd.notna(row.get("subject")) else ""
            # Get body preview safely - try body_text first, then body
            body = row.get("body_text", row.get("body", ""))
            body_preview = str(body)[:200] if pd.notna(body) else ""
            # Get spam probability safely
            spam_prob = float(row.get("spam_probability", 0))
            
            record = {
                "email_id": email_id,
                "subject": subject,
                "body_preview": body_preview,
                "spam_probability": round(spam_prob, 4),
                "spam_label": "spam" if row.get("spam_label", 0) == 1 else "ham",
                "confidence": "high" if spam_prob > 0.8 or spam_prob < 0.2 else "medium" if spam_prob > 0.6 or spam_prob < 0.4 else "low"
            }
            records.append(record)
        
        # Return as CSV
        if format == "csv":
            csv_df = pd.DataFrame(records)
            csv_buffer = StringIO()
            csv_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            return StreamingResponse(
                iter([csv_buffer.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=batch_predictions.csv"}
            )
        
        # Default: return as JSON
        return {
            "data": records,
            "pagination": {
                "total_records": total_records,
                "limit": limit,
                "offset": offset,
                "returned_records": len(records),
                "has_more": offset + limit < total_records
            },
            "output_path": output_path
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"OUTPUT_NOT_FOUND: Predictions file not found at '{output_path}'"
        )
    except Exception as e:
        logger.error(f"Failed to read batch predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"READ_FAILED: {str(e)}"
        )


@router.post("", response_model=BatchJobResponse)
async def submit_batch_job(
    request: BatchAsyncRequest,
    clients: ServiceClients = Depends(get_clients)
):
    """
    Submit an asynchronous batch prediction job.
    
    This endpoint submits a Ray job for large-scale batch inference.
    Use this for datasets larger than 100 records.
    
    The job will:
    1. Read input data from Azure Blob Storage (Parquet format)
    2. Extract features for all records
    3. Make predictions using the specified model
    4. Write results to the output path
    
    Returns a job_id that can be used to check status and retrieve results.
    """
    try:
        result = await clients.ray.submit_batch_job(
            input_path=request.input_path,
            output_path=request.output_path,
            model_stage=request.model_stage.value,
            resources=request.ray_resources
        )
        
        logger.info(f"Submitted batch job: {result['job_id']}")
        
        return BatchJobResponse(
            job_id=result["job_id"],
            status=JobStatus.SUBMITTED,
            estimated_duration_minutes=result.get("estimated_duration_minutes"),
            created_at=result["created_at"]
        )
        
    except Exception as e:
        logger.error(f"Failed to submit batch job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"BATCH_SUBMISSION_FAILED: {str(e)}"
        )


@router.get("/{job_id}", response_model=BatchJobStatusResponse)
async def get_batch_job_status(
    job_id: str,
    clients: ServiceClients = Depends(get_clients)
):
    """
    Get the status of a batch prediction job.
    
    Returns current status, progress, and timing information.
    """
    try:
        status = await clients.ray.get_job_status(job_id)
        
        if status.get("status") == "not_found":
            raise HTTPException(
                status_code=404,
                detail=f"JOB_NOT_FOUND: Batch job '{job_id}' not found"
            )
        
        return BatchJobStatusResponse(
            job_id=job_id,
            status=JobStatus(status["status"]),
            started_at=status.get("started_at"),
            completed_at=status.get("completed_at"),
            error_message=status.get("error_message"),
            records_processed=status.get("records_processed"),
            records_total=status.get("records_total")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch job status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"STATUS_CHECK_FAILED: {str(e)}"
        )


@router.get("/{job_id}/results", response_model=BatchJobResultsResponse)
async def get_batch_job_results(
    job_id: str,
    clients: ServiceClients = Depends(get_clients)
):
    """
    Get the results of a completed batch prediction job.
    
    Returns the output path and summary statistics.
    Only available after job has succeeded.
    """
    try:
        # First check status
        status = await clients.ray.get_job_status(job_id)
        
        if status.get("status") == "not_found":
            raise HTTPException(
                status_code=404,
                detail=f"JOB_NOT_FOUND: Batch job '{job_id}' not found"
            )
        
        if status["status"] != "succeeded":
            raise HTTPException(
                status_code=400,
                detail=f"JOB_NOT_COMPLETE: Job status is '{status['status']}', results not available"
            )
        
        # Get results
        results = await clients.ray.get_job_results(job_id)
        
        return BatchJobResultsResponse(
            job_id=job_id,
            status=JobStatus.SUCCEEDED,
            output_path=results.get("output_path", ""),
            records_processed=results.get("records_processed", 0),
            spam_count=results.get("spam_count", 0),
            ham_count=results.get("ham_count", 0),
            completed_at=results.get("completed_at", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch job results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RESULTS_FETCH_FAILED: {str(e)}"
        )


@router.delete("/{job_id}")
async def cancel_batch_job(
    job_id: str,
    clients: ServiceClients = Depends(get_clients)
):
    """
    Cancel a running batch prediction job.
    
    Requires `operator` role.
    """
    try:
        result = await clients.ray.cancel_job(job_id)
        
        if result.get("status") == "not_found":
            raise HTTPException(
                status_code=404,
                detail=f"JOB_NOT_FOUND: Batch job '{job_id}' not found"
            )
        
        logger.info(f"Cancelled batch job: {job_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel batch job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"CANCEL_FAILED: {str(e)}"
        )


@router.get("", summary="List batch jobs")
async def list_batch_jobs(
    clients: ServiceClients = Depends(get_clients),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum jobs to return")
):
    """
    List recent batch prediction jobs.
    """
    try:
        jobs = await clients.ray.list_jobs(limit=limit)
        
        # Apply status filter if provided
        if status_filter:
            jobs = [j for j in jobs if j["status"] == status_filter.lower()]
        
        return {
            "jobs": jobs,
            "total_count": len(jobs)
        }
        
    except Exception as e:
        logger.error(f"Failed to list batch jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LIST_JOBS_FAILED: {str(e)}"
        )
