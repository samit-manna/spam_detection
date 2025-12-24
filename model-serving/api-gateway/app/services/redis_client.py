"""
Redis client for caching and Feast online store operations.
"""

import logging
from typing import Dict, Optional, Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client for caching and metrics."""
    
    def __init__(
        self,
        host: str,
        port: int = 6380,
        password: Optional[str] = None,
        ssl: bool = True,
        db: int = 0,
        timeout: int = 5
    ):
        self.host = host
        self.port = port
        self.password = password
        self.ssl = ssl
        self.db = db
        self.timeout = timeout
        self._client: Optional[redis.Redis] = None
        self._pool: Optional[redis.ConnectionPool] = None
    
    async def connect(self):
        """Initialize Redis connection pool."""
        # Build connection kwargs
        connection_kwargs = {
            "host": self.host,
            "port": self.port,
            "password": self.password,
            "db": self.db,
            "decode_responses": True,
            "socket_timeout": self.timeout,
            "socket_connect_timeout": self.timeout,
            "max_connections": 20
        }
        
        # For SSL connections (Azure Redis), use ssl_cert_reqs
        if self.ssl:
            import ssl as ssl_module
            connection_kwargs["connection_class"] = redis.SSLConnection
            connection_kwargs["ssl_cert_reqs"] = ssl_module.CERT_REQUIRED
        
        self._pool = redis.ConnectionPool(**connection_kwargs)
        
        self._client = redis.Redis(connection_pool=self._pool)
        
        # Test connection
        await self._client.ping()
        logger.info(f"Connected to Redis at {self.host}:{self.port}")
    
    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis client closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            latency = await self._measure_latency()
            info = await self._client.info("server")
            return {
                "status": "healthy",
                "latency_ms": latency,
                "version": info.get("redis_version")
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _measure_latency(self) -> float:
        """Measure Redis latency."""
        import time
        start = time.perf_counter()
        await self._client.ping()
        return (time.perf_counter() - start) * 1000
    
    # Cache operations
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        return await self._client.get(key)
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL."""
        if ttl:
            return await self._client.setex(key, ttl, value)
        return await self._client.set(key, value)
    
    async def delete(self, key: str) -> int:
        """Delete key from cache."""
        return await self._client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self._client.exists(key) > 0
    
    # Counter operations for metrics
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        return await self._client.incrby(key, amount)
    
    async def get_counter(self, key: str) -> int:
        """Get counter value."""
        value = await self._client.get(key)
        return int(value) if value else 0
    
    # Hash operations for storing structured data
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields."""
        return await self._client.hset(name, mapping=mapping)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field."""
        return await self._client.hget(name, key)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        return await self._client.hgetall(name)
    
    # Metrics storage
    async def store_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[str] = None
    ):
        """Store a metric value."""
        import json
        from datetime import datetime, timezone
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        key = f"metrics:{metric_name}"
        data = json.dumps({"value": value, "timestamp": timestamp})
        
        # Store as sorted set for time-series
        score = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        await self._client.zadd(key, {data: score})
        
        # Keep only last 24 hours
        cutoff = score - (24 * 60 * 60)
        await self._client.zremrangebyscore(key, "-inf", cutoff)
    
    async def get_metrics(
        self,
        metric_name: str,
        limit: int = 100
    ) -> list:
        """Get recent metric values."""
        import json
        
        key = f"metrics:{metric_name}"
        values = await self._client.zrevrange(key, 0, limit - 1)
        return [json.loads(v) for v in values]
    
    # Rate limiting
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check rate limit for a key.
        Returns (allowed, remaining_requests).
        """
        import time
        
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        # Use sorted set for sliding window
        rate_key = f"ratelimit:{key}"
        
        # Remove old entries
        await self._client.zremrangebyscore(rate_key, "-inf", window_start)
        
        # Count current requests
        current_count = await self._client.zcard(rate_key)
        
        if current_count >= limit:
            return False, 0
        
        # Add new request
        await self._client.zadd(rate_key, {str(current_time): current_time})
        await self._client.expire(rate_key, window_seconds)
        
        return True, limit - current_count - 1
