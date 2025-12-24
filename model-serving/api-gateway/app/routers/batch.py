"""
Batch inference endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.models.requests import BatchAsyncRequest
from app.models.responses import (
    BatchJobResponse,
    BatchJobStatusResponse,
    BatchJobResultsResponse,
    JobStatus,
)
from app.dependencies import ServiceClients, get_clients

logger = logging.getLogger(__name__)

router = APIRouter()


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
