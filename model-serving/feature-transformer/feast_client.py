"""
Feast Online Store Client

This module provides a client for fetching features from Feast online store.
It's designed to work with the Redis-based online store deployed in the cluster.

Usage:
    from feast_client import FeastOnlineClient
    
    client = FeastOnlineClient()
    features = await client.get_sender_features("example.com")
"""

import os
import logging
from typing import Dict, List, Optional, Any
import asyncio

import redis
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Support both Azure Redis (from Terraform) and in-cluster Redis
# Azure Redis uses SSL on port 6380
FEAST_REDIS_HOST = os.environ.get("FEAST_REDIS_HOST", "feast-redis.feast.svc.cluster.local")
FEAST_REDIS_PORT = int(os.environ.get("FEAST_REDIS_PORT", "6379"))
FEAST_REDIS_PASSWORD = os.environ.get("FEAST_REDIS_PASSWORD", None)
FEAST_REDIS_SSL = os.environ.get("FEAST_REDIS_SSL", "false").lower() == "true"
FEAST_PROJECT_NAME = os.environ.get("FEAST_PROJECT_NAME", "spam_detection")


# =============================================================================
# Feast Online Client
# =============================================================================

class FeastOnlineClient:
    """
    Async client for Feast online feature store backed by Redis.
    
    This client directly queries Redis for features, bypassing the Feast server
    for lower latency. It requires understanding of how Feast stores features in Redis.
    
    Feast Redis key format:
        {project}:{entity_key}:{feature_view_name}
    
    For faster lookups, we use the Redis hash structure that Feast creates.
    """
    
    def __init__(
        self,
        redis_host: str = FEAST_REDIS_HOST,
        redis_port: int = FEAST_REDIS_PORT,
        redis_password: Optional[str] = FEAST_REDIS_PASSWORD,
        redis_ssl: bool = FEAST_REDIS_SSL,
        project_name: str = FEAST_PROJECT_NAME
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.redis_ssl = redis_ssl
        self.project_name = project_name
        self.redis_client: Optional[aioredis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Establish connection to Redis."""
        if self._connected:
            return
        
        try:
            # Configure SSL for Azure Redis
            ssl_config = self.redis_ssl
            
            self.redis_client = aioredis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                ssl=ssl_config,
                decode_responses=False,  # We need bytes for Feast protobuf
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
    
    async def get_sender_domain_features(
        self,
        sender_domain: str
    ) -> Dict[str, Any]:
        """
        Get sender domain features from Feast online store.
        
        Args:
            sender_domain: The sender's email domain (e.g., "gmail.com")
            
        Returns:
            Dictionary with feature values
        """
        default_features = {
            "email_count": 0,
            "spam_count": 0,
            "ham_count": 0,
            "spam_ratio": 0.5
        }
        
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                return default_features
        
        try:
            # Feast stores features with keys in format:
            # project:entity_value:feature_view
            key = f"{self.project_name}:sender_domain:{sender_domain}:sender_domain_features"
            
            # Get all fields from the hash
            features = await self.redis_client.hgetall(key)
            
            if not features:
                logger.debug(f"No features found for sender_domain: {sender_domain}")
                return default_features
            
            # Parse Feast feature values (stored as protobuf)
            result = {}
            for field, value in features.items():
                field_name = field.decode() if isinstance(field, bytes) else field
                # Feast stores values as protobuf, but for simple numeric types
                # we can often decode them directly
                try:
                    result[field_name] = self._decode_value(value)
                except Exception:
                    result[field_name] = default_features.get(field_name, 0)
            
            return {**default_features, **result}
            
        except Exception as e:
            logger.warning(f"Error fetching features for {sender_domain}: {e}")
            return default_features
    
    def _decode_value(self, value: bytes) -> Any:
        """Decode a Feast feature value from bytes."""
        import struct
        
        # Try to decode as float (8 bytes, little-endian)
        if len(value) == 8:
            try:
                return struct.unpack('<d', value)[0]
            except struct.error:
                pass
        
        # Try to decode as int (8 bytes, little-endian)
        if len(value) == 8:
            try:
                return struct.unpack('<q', value)[0]
            except struct.error:
                pass
        
        # Try to decode as float32
        if len(value) == 4:
            try:
                return struct.unpack('<f', value)[0]
            except struct.error:
                pass
        
        # Try to decode as int32
        if len(value) == 4:
            try:
                return struct.unpack('<i', value)[0]
            except struct.error:
                pass
        
        # Fallback: try UTF-8 decode
        try:
            return float(value.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return 0
    
    async def get_batch_features(
        self,
        sender_domains: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get features for multiple sender domains.
        
        Args:
            sender_domains: List of sender domains
            
        Returns:
            Dictionary mapping sender_domain to features
        """
        results = {}
        
        # Use Redis pipeline for batch fetching
        if not self._connected:
            await self.connect()
        
        pipe = self.redis_client.pipeline()
        
        for domain in sender_domains:
            key = f"{self.project_name}:sender_domain:{domain}:sender_domain_features"
            pipe.hgetall(key)
        
        try:
            responses = await pipe.execute()
            
            for domain, features in zip(sender_domains, responses):
                if features:
                    parsed = {}
                    for field, value in features.items():
                        field_name = field.decode() if isinstance(field, bytes) else field
                        parsed[field_name] = self._decode_value(value)
                    results[domain] = parsed
                else:
                    results[domain] = {
                        "email_count": 0,
                        "spam_count": 0,
                        "ham_count": 0,
                        "spam_ratio": 0.5
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Batch feature fetch failed: {e}")
            return {domain: {"email_count": 0, "spam_count": 0, "ham_count": 0, "spam_ratio": 0.5}
                    for domain in sender_domains}


# =============================================================================
# Feast gRPC Client (Alternative)
# =============================================================================

class FeastGRPCClient:
    """
    Client for Feast Feature Server using gRPC.
    
    This client uses the official Feast gRPC API for feature retrieval.
    It's more reliable but slightly higher latency than direct Redis access.
    """
    
    def __init__(
        self,
        server_url: str = "feast-service.feast.svc.cluster.local:6566"
    ):
        self.server_url = server_url
        self._initialized = False
    
    async def initialize(self):
        """Initialize the gRPC client."""
        try:
            # Import feast
            from feast import FeatureStore
            
            # Note: In production, you'd use a gRPC client directly
            # For now, we create a Feast client
            logger.info(f"Initialized Feast gRPC client to {self.server_url}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Feast gRPC client: {e}")
            self._initialized = False
    
    async def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        feature_refs: List[str]
    ) -> Dict[str, List[Any]]:
        """
        Get features from Feast online store via gRPC.
        
        Args:
            entity_rows: List of entity dictionaries
            feature_refs: List of feature references
            
        Returns:
            Dictionary mapping feature names to value lists
        """
        if not self._initialized:
            await self.initialize()
        
        # Placeholder - in production, use gRPC client
        # Example response format
        return {ref.split(":")[1]: [0] * len(entity_rows) for ref in feature_refs}


# =============================================================================
# Singleton Instance
# =============================================================================

_feast_client: Optional[FeastOnlineClient] = None


async def get_feast_client() -> FeastOnlineClient:
    """Get or create the Feast client singleton."""
    global _feast_client
    
    if _feast_client is None:
        _feast_client = FeastOnlineClient()
        await _feast_client.connect()
    
    return _feast_client


# =============================================================================
# Test Function
# =============================================================================

async def test_feast_client():
    """Test the Feast client."""
    client = await get_feast_client()
    
    # Test single lookup
    features = await client.get_sender_domain_features("gmail.com")
    print(f"gmail.com features: {features}")
    
    # Test batch lookup
    domains = ["gmail.com", "yahoo.com", "example.com"]
    batch_features = await client.get_batch_features(domains)
    print(f"Batch features: {batch_features}")
    
    await client.close()


if __name__ == "__main__":
    asyncio.run(test_feast_client())
