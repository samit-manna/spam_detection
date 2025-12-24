"""
Feature transformer client for real-time feature extraction.
"""

import logging
from typing import Dict, List, Optional, Any

import httpx

logger = logging.getLogger(__name__)


class FeatureTransformerClient:
    """Client for feature transformer service."""
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 30
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def connect(self):
        """Initialize HTTP client."""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
        logger.info(f"Connected to Feature Transformer at {self.base_url}")
    
    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("Feature Transformer client closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Feature Transformer connectivity."""
        try:
            response = await self._http_client.get("/health")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "latency_ms": response.elapsed.total_seconds() * 1000
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def extract_features(
        self,
        email_id: str,
        subject: str,
        body: str,
        sender: str
    ) -> Dict[str, Any]:
        """
        Extract features from email for prediction.
        
        Returns feature vector and metadata.
        """
        payload = {
            "email_id": email_id,
            "subject": subject,
            "body": body,
            "sender": sender
        }
        
        response = await self._http_client.post("/transform", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def batch_extract_features(
        self,
        emails: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Extract features for multiple emails.
        
        Args:
            emails: List of dicts with email_id, subject, body, sender
            
        Returns:
            List of feature extraction results
        """
        payload = {"emails": emails}
        
        response = await self._http_client.post("/transform/batch", json=payload)
        response.raise_for_status()
        
        return response.json().get("results", [])
