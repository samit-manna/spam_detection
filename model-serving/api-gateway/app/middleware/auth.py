"""
Authentication middleware for API key and Azure AD validation.
"""

import logging
from typing import Optional, Callable
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config import settings, API_KEY_ROLES

logger = logging.getLogger(__name__)

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = [
    "/health",
    "/health/live",
    "/health/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
]

# Role permissions (serving-only API - no admin role needed)
ROLE_PERMISSIONS = {
    "viewer": {
        "GET": ["*"],  # All GET endpoints (models, metrics, batch status)
        "POST": [],
        "PUT": [],
        "DELETE": [],
    },
    "operator": {
        "GET": ["*"],
        "POST": ["/predict", "/predict/batch-sync", "/predict/batch", "/metrics/refresh"],
        "PUT": [],
        "DELETE": ["/predict/batch"],  # Can cancel own batch jobs
    },
}


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for handling API key authentication and authorization."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Skip auth for Prometheus metrics scraping
        if request.url.path.startswith("/metrics"):
            return await call_next(request)
        
        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        
        # Try Azure AD if enabled and no API key
        if not api_key and settings.ENABLE_AZURE_AD:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user_info = await self._validate_azure_ad_token(token)
                if user_info:
                    request.state.user = user_info
                    request.state.auth_method = "azure_ad"
                    return await call_next(request)
        
        # Validate API key
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="AUTHENTICATION_REQUIRED",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        # Check if API key is valid
        role = self._get_api_key_role(api_key)
        if not role:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            raise HTTPException(
                status_code=401,
                detail="INVALID_API_KEY"
            )
        
        # Check authorization
        if not self._is_authorized(role, request.method, request.url.path):
            logger.warning(
                f"Authorization denied: role={role}, method={request.method}, "
                f"path={request.url.path}"
            )
            raise HTTPException(
                status_code=403,
                detail="INSUFFICIENT_PERMISSIONS"
            )
        
        # Set user info in request state
        request.state.api_key = api_key
        request.state.role = role
        request.state.auth_method = "api_key"
        
        return await call_next(request)
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public."""
        for endpoint in PUBLIC_ENDPOINTS:
            if path == endpoint or path.startswith(endpoint + "/"):
                return True
        return False
    
    def _get_api_key_role(self, api_key: str) -> Optional[str]:
        """Get role for an API key."""
        # Check configured API keys
        if api_key in API_KEY_ROLES:
            return API_KEY_ROLES[api_key]
        
        # Check against settings API keys (default to operator role)
        if api_key in settings.api_keys_list:
            return "operator"
        
        return None
    
    def _is_authorized(self, role: str, method: str, path: str) -> bool:
        """Check if role has permission for method and path."""
        if role not in ROLE_PERMISSIONS:
            return False
        
        permissions = ROLE_PERMISSIONS[role]
        allowed_paths = permissions.get(method, [])
        
        if "*" in allowed_paths:
            return True
        
        for allowed in allowed_paths:
            if path.startswith(allowed):
                return True
        
        return False
    
    async def _validate_azure_ad_token(self, token: str) -> Optional[dict]:
        """Validate Azure AD JWT token."""
        if not settings.ENABLE_AZURE_AD:
            return None
        
        try:
            import jwt
            from jwt import PyJWKClient
            
            # Azure AD JWKS URL
            jwks_url = (
                f"https://login.microsoftonline.com/"
                f"{settings.AZURE_AD_TENANT_ID}/discovery/v2.0/keys"
            )
            
            jwks_client = PyJWKClient(jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            # Decode and validate token
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=settings.AZURE_AD_CLIENT_ID,
                issuer=f"https://login.microsoftonline.com/{settings.AZURE_AD_TENANT_ID}/v2.0"
            )
            
            # Extract user info
            return {
                "user_id": payload.get("oid"),
                "email": payload.get("preferred_username") or payload.get("email"),
                "name": payload.get("name"),
                "roles": payload.get("roles", []),
            }
            
        except Exception as e:
            logger.warning(f"Azure AD token validation failed: {e}")
            return None


def get_current_user(request: Request) -> dict:
    """Get current user from request state."""
    return {
        "api_key": getattr(request.state, "api_key", None),
        "role": getattr(request.state, "role", None),
        "user": getattr(request.state, "user", None),
        "auth_method": getattr(request.state, "auth_method", None),
    }


def require_role(required_roles: list):
    """Decorator to require specific roles for an endpoint."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            role = getattr(request.state, "role", None)
            if role not in required_roles:
                raise HTTPException(
                    status_code=403,
                    detail="INSUFFICIENT_PERMISSIONS"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
