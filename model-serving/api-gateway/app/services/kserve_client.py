"""
KServe client for inference service operations.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import httpx
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config

logger = logging.getLogger(__name__)


class KServeClient:
    """Client for KServe inference service operations."""
    
    def __init__(
        self,
        namespace: str = "kserve",
        inference_service_name: str = "spam-detector",
        staging_service_name: str = "spam-detector-staging",
        production_service_name: str = "spam-detector",
        model_name: str = None,
        timeout: int = 30
    ):
        self.namespace = namespace
        self.inference_service_name = inference_service_name
        self.staging_service_name = staging_service_name
        self.production_service_name = production_service_name
        # Model name in Triton may differ from inference service name
        self.model_name = model_name or inference_service_name.replace("-staging", "")
        self.timeout = timeout
        self._k8s_custom: Optional[k8s_client.CustomObjectsApi] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._inference_url: Optional[str] = None
    
    async def connect(self):
        """Initialize KServe client connection."""
        try:
            # Try in-cluster config first
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            # Fall back to kubeconfig
            k8s_config.load_kube_config()
        
        self._k8s_custom = k8s_client.CustomObjectsApi()
        
        # Get inference service URL
        await self._refresh_inference_url()
        
        self._http_client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info(f"Connected to KServe in namespace {self.namespace}")
    
    async def close(self):
        """Close client connections."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("KServe client closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check KServe connectivity."""
        try:
            isvc = await self.get_inference_service(self.inference_service_name)
            if isvc:
                return {
                    "status": "healthy",
                    "inference_service": self.inference_service_name,
                    "ready": isvc.get("status", {}).get("conditions", [{}])[0].get("status") == "True"
                }
            return {"status": "unhealthy", "error": "InferenceService not found"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _refresh_inference_url(self, stage: str = "Production"):
        """Refresh inference URL from InferenceService status."""
        try:
            service_name = self.inference_service_name
            if stage == "Staging":
                service_name = f"{self.inference_service_name}-staging"
            
            isvc = await self.get_inference_service(service_name)
            if isvc:
                self._inference_url = isvc.get("status", {}).get("url")
                logger.info(f"Inference URL: {self._inference_url}")
        except Exception as e:
            logger.warning(f"Could not get inference URL: {e}")
    
    async def get_inference_service(
        self,
        name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get InferenceService details."""
        try:
            name = name or self.inference_service_name
            isvc = self._k8s_custom.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="inferenceservices",
                name=name
            )
            return isvc
        except k8s_client.ApiException as e:
            if e.status == 404:
                return None
            raise
    
    async def list_inference_services(self) -> List[Dict[str, Any]]:
        """List all InferenceServices."""
        result = self._k8s_custom.list_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=self.namespace,
            plural="inferenceservices"
        )
        return result.get("items", [])
    
    async def predict(
        self,
        features: List[float],
        model_stage: str = "Production"
    ) -> Dict[str, Any]:
        """Make a prediction using KServe."""
        start_time = time.perf_counter()
        
        # Determine service name based on stage
        if model_stage == "Staging":
            service_name = self.staging_service_name
        else:
            service_name = self.production_service_name
        
        # Get inference URL - check if service exists
        isvc = await self.get_inference_service(service_name)
        if not isvc:
            raise ValueError(f"InferenceService {service_name} not found")
        
        status_url = isvc.get("status", {}).get("url")
        if not status_url:
            raise ValueError(f"InferenceService {service_name} not ready")
        
        # Use internal cluster service URL instead of external sslip.io URL
        # Format: http://<service-name>-predictor.<namespace>.svc.cluster.local
        internal_url = f"http://{service_name}-predictor.{self.namespace}.svc.cluster.local"
        
        # Build request for Triton - use model_name (the actual model in Triton)
        predict_url = f"{internal_url}/v2/models/{self.model_name}/infer"
        
        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, len(features)],
                    "datatype": "FP32",
                    "data": features  # Flat array, not nested
                }
            ]
        }
        
        # Make prediction
        response = await self._http_client.post(predict_url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse Triton response - model outputs 'label' and 'probabilities'
        outputs = result.get("outputs", [])
        probabilities = []
        label = None
        for output in outputs:
            if output.get("name") == "probabilities":
                probabilities = output.get("data", [])
            elif output.get("name") == "label":
                label = output.get("data", [None])[0]
        
        # Return spam probability (index 1 in probabilities array)
        spam_prob = probabilities[1] if len(probabilities) > 1 else 0.5
        
        return {
            "raw_output": [spam_prob],
            "label": label,
            "probabilities": probabilities,
            "latency_ms": round(latency_ms, 2),
            "model_stage": model_stage,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def predict_v1(
        self,
        features: List[float],
        model_stage: str = "Production"
    ) -> Dict[str, Any]:
        """Make a prediction using KServe V1 protocol."""
        start_time = time.perf_counter()
        
        # Determine service name based on stage
        service_name = self.inference_service_name
        if model_stage == "Staging":
            service_name = f"{self.inference_service_name}-staging"
        
        # Check if service exists
        isvc = await self.get_inference_service(service_name)
        if not isvc:
            raise ValueError(f"InferenceService {service_name} not found")
        
        status_url = isvc.get("status", {}).get("url")
        if not status_url:
            raise ValueError(f"InferenceService {service_name} not ready")
        
        # Use internal cluster service URL
        internal_url = f"http://{service_name}-predictor.{self.namespace}.svc.cluster.local"
        
        # V1 protocol - use model_name (the actual model in Triton)
        predict_url = f"{internal_url}/v1/models/{self.model_name}:predict"
        
        payload = {
            "instances": [features]
        }
        
        response = await self._http_client.post(predict_url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "predictions": result.get("predictions", []),
            "latency_ms": round(latency_ms, 2),
            "model_stage": model_stage,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_model_metadata(
        self,
        model_stage: str = "Production"
    ) -> Dict[str, Any]:
        """Get model metadata from Triton."""
        service_name = self.inference_service_name
        if model_stage == "Staging":
            service_name = f"{self.inference_service_name}-staging"
        
        isvc = await self.get_inference_service(service_name)
        if not isvc:
            return {}
        
        status_url = isvc.get("status", {}).get("url")
        if not status_url:
            return {}
        
        try:
            # Use internal cluster service URL
            internal_url = f"http://{service_name}-predictor.{self.namespace}.svc.cluster.local"
            metadata_url = f"{internal_url}/v2/models/{self.model_name}"
            response = await self._http_client.get(metadata_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Could not get model metadata: {e}")
            return {}
    
    async def get_model_version(
        self,
        model_stage: str = "Production"
    ) -> Optional[str]:
        """Get current model version from InferenceService annotations."""
        service_name = self.inference_service_name
        if model_stage == "Staging":
            service_name = f"{self.inference_service_name}-staging"
        
        isvc = await self.get_inference_service(service_name)
        if not isvc:
            return None
        
        annotations = isvc.get("metadata", {}).get("annotations", {})
        return annotations.get("serving.kserve.io/model-version")
