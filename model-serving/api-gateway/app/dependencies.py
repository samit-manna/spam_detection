"""
Dependency injection module for FastAPI.

This module contains shared dependencies that are used across routers,
separated from main.py to avoid circular imports.
"""

import asyncio
import logging
from typing import Optional

from app.config import settings
from app.services.mlflow_client import MLflowClient
from app.services.kserve_client import KServeClient
from app.services.ray_client import RayClient
from app.services.redis_client import RedisClient
from app.services.inference_logger import InferenceLogger

logger = logging.getLogger(__name__)


class ServiceClients:
    """Container for all service clients with connection pooling."""
    
    def __init__(self):
        self.mlflow: Optional[MLflowClient] = None
        self.kserve: Optional[KServeClient] = None
        self.ray: Optional[RayClient] = None
        self.redis: Optional[RedisClient] = None
        self.inference_logger: Optional["InferenceLogger"] = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize all service clients."""
        logger.info("Initializing service clients...")
        
        self.mlflow = MLflowClient(
            tracking_uri=settings.MLFLOW_TRACKING_URI,
            registry_uri=settings.MLFLOW_REGISTRY_URI
        )
        await self.mlflow.connect()
        
        self.kserve = KServeClient(
            namespace=settings.KSERVE_NAMESPACE,
            inference_service_name=settings.INFERENCE_SERVICE_NAME,
            staging_service_name=settings.INFERENCE_SERVICE_STAGING,
            production_service_name=settings.INFERENCE_SERVICE_PRODUCTION
        )
        await self.kserve.connect()
        
        self.ray = RayClient(
            address=settings.RAY_ADDRESS,
            namespace=settings.RAY_NAMESPACE
        )
        await self.ray.connect()
        
        self.redis = RedisClient(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            ssl=settings.REDIS_SSL
        )
        await self.redis.connect()
        
        # Initialize inference logger for monitoring
        if getattr(settings, 'INFERENCE_LOGGING_ENABLED', True):
            try:
                self.inference_logger = InferenceLogger(
                    storage_account_name=getattr(settings, 'AZURE_STORAGE_ACCOUNT_NAME', None),
                    storage_account_key=getattr(settings, 'AZURE_STORAGE_ACCOUNT_KEY', None),
                    container_name=getattr(settings, 'INFERENCE_LOG_CONTAINER', 'data'),
                    model_name=settings.MODEL_NAME,
                    buffer_size=getattr(settings, 'INFERENCE_LOG_BUFFER_SIZE', 100),
                    flush_interval_seconds=getattr(settings, 'INFERENCE_LOG_FLUSH_INTERVAL', 60),
                )
                await self.inference_logger.start()
                logger.info("Inference logger initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize inference logger: {e}")
                self.inference_logger = None
        else:
            logger.info("Inference logging disabled")
        
        logger.info("All service clients initialized")
    
    async def shutdown(self):
        """Gracefully shutdown all service clients."""
        logger.info("Shutting down service clients...")
        self._shutdown_event.set()
        
        # Shutdown inference logger first to flush pending logs
        if self.inference_logger:
            try:
                await self.inference_logger.stop()
                logger.info("Inference logger stopped")
            except Exception as e:
                logger.error(f"Error stopping inference logger: {e}")
        
        clients = [self.mlflow, self.kserve, self.ray, self.redis]
        for client in clients:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing client: {e}")
        
        logger.info("All service clients shut down")
    
    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()


# Global service clients container
clients = ServiceClients()


def get_clients() -> ServiceClients:
    """
    Dependency injection for service clients.
    
    Returns the global ServiceClients instance that is initialized
    during application startup.
    """
    return clients
