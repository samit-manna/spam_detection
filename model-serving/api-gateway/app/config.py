"""
Application configuration using Pydantic Settings.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    WORKERS: int = 4
    
    # Authentication
    API_KEYS: str = Field(default="", description="Comma-separated API keys")
    ENABLE_AZURE_AD: bool = False
    AZURE_AD_TENANT_ID: Optional[str] = None
    AZURE_AD_CLIENT_ID: Optional[str] = None
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow-service.mlflow.svc.cluster.local:5000"
    MLFLOW_REGISTRY_URI: Optional[str] = None
    MODEL_NAME: str = "spam-detector"
    
    # KServe
    KSERVE_NAMESPACE: str = "kserve"
    INFERENCE_SERVICE_NAME: str = "spam-detector"
    INFERENCE_SERVICE_STAGING: str = "spam-detector-staging"
    INFERENCE_SERVICE_PRODUCTION: str = "spam-detector"
    INFERENCE_SERVICE_URL: Optional[str] = None
    KSERVE_TIMEOUT: int = 30
    DEFAULT_ENVIRONMENT: str = "staging"  # Default to staging for safety
    
    # Ray (Batch Inference)
    RAY_ADDRESS: str = "ray://ray-cluster-head-svc.ray.svc.cluster.local:10001"
    RAY_NAMESPACE: str = "ray"
    RAY_DASHBOARD_URL: str = "http://ray-cluster-head-svc.ray.svc.cluster.local:8265"
    
    # Redis (Feast Online Store)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6380
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SSL: bool = True
    REDIS_DB: int = 0
    
    # Azure Storage
    AZURE_STORAGE_ACCOUNT_NAME: Optional[str] = None
    AZURE_STORAGE_ACCOUNT_KEY: Optional[str] = None
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None
    
    # Inference Logging (Monitoring)
    INFERENCE_LOGGING_ENABLED: bool = True
    INFERENCE_LOG_CONTAINER: str = "feast"  # Same container as baselines/drift-reports
    INFERENCE_LOG_BUFFER_SIZE: int = 100
    INFERENCE_LOG_FLUSH_INTERVAL: int = 60
    
    # Feature Transformer
    FEATURE_TRANSFORMER_URL: str = "http://feature-transformer.serving.svc.cluster.local:80"
    
    # Feast
    FEAST_REPO_PATH: str = "/feast"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60
    
    # Timeouts
    REQUEST_TIMEOUT: int = 60
    BATCH_JOB_CHECK_INTERVAL: int = 30
    
    @property
    def api_keys_list(self) -> List[str]:
        """Parse API keys from comma-separated string."""
        if not self.API_KEYS:
            return []
        return [key.strip() for key in self.API_KEYS.split(",") if key.strip()]
    
    @property
    def mlflow_registry_uri_resolved(self) -> str:
        """Get MLflow registry URI, defaulting to tracking URI."""
        return self.MLFLOW_REGISTRY_URI or self.MLFLOW_TRACKING_URI
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# API Key roles mapping (in production, this should be in a database)
API_KEY_ROLES = {
    # Format: "api_key": "role"
    # Roles: viewer, operator (serving-only API)
}


def load_api_key_roles():
    """Load API key roles from environment or config."""
    global API_KEY_ROLES
    
    # Load from environment variables
    # Format: API_KEY_ROLE_<key>=<role>
    for key, value in os.environ.items():
        if key.startswith("API_KEY_ROLE_"):
            api_key = key.replace("API_KEY_ROLE_", "")
            API_KEY_ROLES[api_key] = value
    
    # Also support comma-separated format
    # VIEWER_API_KEYS=key1,key2
    viewer_keys = os.environ.get("VIEWER_API_KEYS", "")
    for key in viewer_keys.split(","):
        if key.strip():
            API_KEY_ROLES[key.strip()] = "viewer"
    
    # OPERATOR_API_KEYS=key1,key2
    operator_keys = os.environ.get("OPERATOR_API_KEYS", "")
    for key in operator_keys.split(","):
        if key.strip():
            API_KEY_ROLES[key.strip()] = "operator"


# Load roles on import
load_api_key_roles()
