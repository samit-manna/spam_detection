"""
Routers package initialization.
"""

from app.routers import health, models, predict, batch, metrics, drift

__all__ = [
    "health",
    "models",
    "predict",
    "batch",
    "metrics",
    "drift",
]
