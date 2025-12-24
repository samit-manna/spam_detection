"""
Ray client for batch inference operations.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import httpx
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config

logger = logging.getLogger(__name__)


class RayClient:
    """Client for Ray batch inference operations."""
    
    def __init__(
        self,
        address: str,
        namespace: str = "ray",
        dashboard_url: Optional[str] = None,
        timeout: int = 60
    ):
        self.address = address
        self.namespace = namespace
        self.dashboard_url = dashboard_url or address.replace("ray://", "http://").replace(":10001", ":8265")
        self.timeout = timeout
        self._k8s_batch: Optional[k8s_client.BatchV1Api] = None
        self._k8s_custom: Optional[k8s_client.CustomObjectsApi] = None
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def connect(self):
        """Initialize Ray client connection."""
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
        
        self._k8s_batch = k8s_client.BatchV1Api()
        self._k8s_custom = k8s_client.CustomObjectsApi()
        self._http_client = httpx.AsyncClient(
            base_url=self.dashboard_url,
            timeout=self.timeout
        )
        logger.info(f"Connected to Ray in namespace {self.namespace}")
    
    async def close(self):
        """Close client connections."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("Ray client closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ray cluster connectivity."""
        try:
            response = await self._http_client.get("/api/version")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "version": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def submit_batch_job(
        self,
        input_path: str,
        output_path: str,
        model_stage: str = "Production",
        resources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Submit a batch inference job via RayJob CRD."""
        job_id = f"batch-{uuid.uuid4().hex[:8]}"
        
        # Default resources
        resources = resources or {
            "num_cpus": 4,
            "memory": "8Gi"
        }
        
        # Create RayJob manifest
        ray_job = {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "metadata": {
                "name": job_id,
                "namespace": self.namespace,
                "labels": {
                    "app": "batch-inference",
                    "job-id": job_id
                }
            },
            "spec": {
                "entrypoint": f"python batch_predict.py --input {input_path} --output {output_path} --model-stage {model_stage}",
                "runtimeEnvYAML": """
pip:
  - pandas
  - pyarrow
  - azure-storage-blob
  - onnxruntime
  - feast
""",
                "rayClusterSpec": {
                    "rayVersion": "2.9.0",
                    "headGroupSpec": {
                        "rayStartParams": {
                            "dashboard-host": "0.0.0.0"
                        },
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "ray-head",
                                        "image": "${ACR_NAME}/batch-inference:latest",
                                        "resources": {
                                            "limits": {
                                                "cpu": resources.get("num_cpus", 4),
                                                "memory": resources.get("memory", "8Gi")
                                            }
                                        },
                                        "env": [
                                            {"name": "INPUT_PATH", "value": input_path},
                                            {"name": "OUTPUT_PATH", "value": output_path},
                                            {"name": "MODEL_STAGE", "value": model_stage}
                                        ],
                                        "envFrom": [
                                            {"secretRef": {"name": "azure-storage-secret"}}
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                },
                "submitterPodTemplate": {
                    "spec": {
                        "containers": [
                            {
                                "name": "submitter",
                                "image": "${ACR_NAME}/batch-inference:latest",
                                "envFrom": [
                                    {"secretRef": {"name": "azure-storage-secret"}}
                                ]
                            }
                        ],
                        "restartPolicy": "Never"
                    }
                },
                "shutdownAfterJobFinishes": True,
                "ttlSecondsAfterFinished": 3600
            }
        }
        
        try:
            # Create the RayJob
            self._k8s_custom.create_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayjobs",
                body=ray_job
            )
            
            logger.info(f"Submitted batch job: {job_id}")
            
            # Estimate duration based on typical performance
            estimated_minutes = 15  # Default estimate
            
            return {
                "job_id": job_id,
                "status": "submitted",
                "estimated_duration_minutes": estimated_minutes,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
        except k8s_client.ApiException as e:
            logger.error(f"Failed to submit batch job: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a batch job."""
        try:
            ray_job = self._k8s_custom.get_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayjobs",
                name=job_id
            )
            
            status = ray_job.get("status", {})
            job_status = status.get("jobStatus", "PENDING")
            
            # Map Ray job status to our status
            status_map = {
                "PENDING": "pending",
                "RUNNING": "running",
                "SUCCEEDED": "succeeded",
                "FAILED": "failed",
                "STOPPED": "cancelled"
            }
            normalized_status = status_map.get(job_status, job_status.lower())
            
            # Get timing info
            start_time = status.get("startTime")
            end_time = status.get("endTime")
            
            # Get progress from job output if available
            message = status.get("message", "")
            
            return {
                "job_id": job_id,
                "status": normalized_status,
                "started_at": start_time,
                "completed_at": end_time,
                "error_message": status.get("failureInfo") if normalized_status == "failed" else None,
                "message": message
            }
            
        except k8s_client.ApiException as e:
            if e.status == 404:
                return {"job_id": job_id, "status": "not_found"}
            raise
    
    async def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get results of a completed batch job."""
        status = await self.get_job_status(job_id)
        
        if status["status"] != "succeeded":
            return {
                "job_id": job_id,
                "status": status["status"],
                "error": "Job has not completed successfully"
            }
        
        try:
            # Get the RayJob to find output path
            ray_job = self._k8s_custom.get_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayjobs",
                name=job_id
            )
            
            # Parse entrypoint to get output path
            entrypoint = ray_job.get("spec", {}).get("entrypoint", "")
            output_path = None
            for part in entrypoint.split():
                if part.startswith("abfss://") or part.startswith("wasbs://"):
                    if "output" in part.lower():
                        output_path = part
                        break
            
            return {
                "job_id": job_id,
                "status": "succeeded",
                "output_path": output_path,
                "completed_at": status.get("completed_at"),
                "records_processed": status.get("records_processed"),
                "spam_count": status.get("spam_count"),
                "ham_count": status.get("ham_count")
            }
            
        except Exception as e:
            logger.error(f"Error getting job results: {e}")
            raise
    
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running batch job."""
        try:
            # Delete the RayJob (this will stop it)
            self._k8s_custom.delete_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayjobs",
                name=job_id
            )
            
            logger.info(f"Cancelled batch job: {job_id}")
            
            return {
                "job_id": job_id,
                "status": "cancelled",
                "cancelled_at": datetime.now(timezone.utc).isoformat()
            }
            
        except k8s_client.ApiException as e:
            if e.status == 404:
                return {"job_id": job_id, "status": "not_found"}
            raise
    
    async def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent batch jobs."""
        try:
            result = self._k8s_custom.list_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayjobs",
                limit=limit
            )
            
            jobs = []
            for item in result.get("items", []):
                status = item.get("status", {})
                jobs.append({
                    "job_id": item["metadata"]["name"],
                    "status": status.get("jobStatus", "PENDING").lower(),
                    "created_at": item["metadata"].get("creationTimestamp"),
                    "started_at": status.get("startTime"),
                    "completed_at": status.get("endTime")
                })
            
            return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return []
