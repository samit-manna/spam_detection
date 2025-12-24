"""
MLflow client for model registry operations.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import httpx
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)


class MLflowClient:
    """Client for MLflow tracking and model registry operations."""
    
    def __init__(
        self,
        tracking_uri: str,
        registry_uri: Optional[str] = None,
        timeout: int = 30
    ):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        self.timeout = timeout
        self._client: Optional[MlflowClient] = None
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def connect(self):
        """Initialize MLflow client connection."""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        self._client = MlflowClient(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri
        )
        
        self._http_client = httpx.AsyncClient(
            base_url=self.tracking_uri,
            timeout=self.timeout
        )
        
        logger.info(f"Connected to MLflow at {self.tracking_uri}")
    
    async def close(self):
        """Close client connections."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("MLflow client closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MLflow connectivity."""
        try:
            response = await self._http_client.get("/health")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "latency_ms": response.elapsed.total_seconds() * 1000
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        models = []
        
        for rm in self._client.search_registered_models():
            model_info = {
                "name": rm.name,
                "description": rm.description,
                "creation_timestamp": self._timestamp_to_iso(rm.creation_timestamp),
                "last_updated_timestamp": self._timestamp_to_iso(rm.last_updated_timestamp),
                "tags": dict(rm.tags) if rm.tags else {},
                "latest_versions": []
            }
            
            for mv in rm.latest_versions:
                model_info["latest_versions"].append(
                    self._format_model_version(mv)
                )
            
            models.append(model_info)
        
        return models
    
    async def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            rm = self._client.get_registered_model(model_name)
            return {
                "name": rm.name,
                "description": rm.description,
                "creation_timestamp": self._timestamp_to_iso(rm.creation_timestamp),
                "last_updated_timestamp": self._timestamp_to_iso(rm.last_updated_timestamp),
                "tags": dict(rm.tags) if rm.tags else {},
                "latest_versions": [
                    self._format_model_version(mv) for mv in rm.latest_versions
                ]
            }
        except Exception as e:
            logger.error(f"Error getting model {model_name}: {e}")
            return None
    
    async def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model."""
        versions = []
        
        for mv in self._client.search_model_versions(f"name='{model_name}'"):
            versions.append(self._format_model_version(mv))
        
        return sorted(versions, key=lambda x: int(x["version"]), reverse=True)
    
    async def get_latest_model_info(
        self,
        model_name: str,
        stage: str = "Production"
    ) -> Optional[Dict[str, Any]]:
        """Get information about the latest model version in a stage."""
        try:
            versions = self._client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return self._format_model_version(versions[0])
            return None
        except Exception as e:
            logger.error(f"Error getting latest model version: {e}")
            return None
    
    async def get_model_version(
        self,
        model_name: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific model version."""
        try:
            mv = self._client.get_model_version(model_name, version)
            return self._format_model_version(mv)
        except Exception as e:
            logger.error(f"Error getting model version: {e}")
            return None
    
    async def promote_model(
        self,
        model_name: str,
        version: str,
        to_stage: str,
        archive_existing: bool = True
    ) -> Dict[str, Any]:
        """Promote a model version to a new stage."""
        try:
            # Get current production version before promotion
            current_prod = await self.get_latest_model_info(model_name, to_stage)
            previous_version = current_prod["version"] if current_prod else None
            
            # Transition stage
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=to_stage,
                archive_existing_versions=archive_existing
            )
            
            logger.info(
                f"Promoted model {model_name} version {version} to {to_stage}"
            )
            
            return {
                "status": "promoted",
                "model_name": model_name,
                "version": version,
                "previous_production_version": previous_version,
                "promoted_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            raise
    
    async def rollback_model(
        self,
        model_name: str,
        target_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rollback to a previous model version."""
        try:
            # Get current production version
            current_prod = await self.get_latest_model_info(model_name, "Production")
            if not current_prod:
                raise ValueError("No production model to rollback from")
            
            current_version = current_prod["version"]
            
            # Determine target version
            if target_version is None:
                # Find previous production version from archived
                versions = await self.get_model_versions(model_name)
                for v in versions:
                    if v["version"] != current_version and v["stage"] == "Archived":
                        target_version = v["version"]
                        break
                
                if target_version is None:
                    raise ValueError("No previous version found to rollback to")
            
            # Perform rollback
            self._client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production",
                archive_existing_versions=True
            )
            
            logger.info(
                f"Rolled back model {model_name} from {current_version} to {target_version}"
            )
            
            return {
                "status": "rolled_back",
                "model_name": model_name,
                "rolled_back_version": target_version,
                "previous_version": current_version,
                "rolled_back_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
            raise
    
    async def get_model_metrics(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, float]:
        """Get metrics for a model version."""
        try:
            mv = self._client.get_model_version(model_name, version)
            if mv.run_id:
                run = self._client.get_run(mv.run_id)
                return dict(run.data.metrics)
            return {}
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {}
    
    def _format_model_version(self, mv: ModelVersion) -> Dict[str, Any]:
        """Format ModelVersion to dict."""
        return {
            "version": mv.version,
            "stage": mv.current_stage,
            "creation_timestamp": self._timestamp_to_iso(mv.creation_timestamp),
            "last_updated_timestamp": self._timestamp_to_iso(mv.last_updated_timestamp),
            "run_id": mv.run_id,
            "source": mv.source,
            "status": mv.status,
            "description": mv.description,
            "tags": dict(mv.tags) if mv.tags else {}
        }
    
    def _timestamp_to_iso(self, ts: Optional[int]) -> Optional[str]:
        """Convert MLflow timestamp to ISO format."""
        if ts is None:
            return None
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
