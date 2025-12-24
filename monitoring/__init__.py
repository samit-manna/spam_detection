"""
Model Monitoring Module for Spam Detection

Provides monitoring capabilities including:
- Baseline generation from training data
- Drift detection using Evidently AI
- Alerting via webhooks and logs

Components:
- baseline: Generate and store baseline statistics from training data
- drift_detector: Scheduled drift detection comparing production vs baseline

Note: Inference logging and drift API endpoints have been moved to 
the api-gateway for self-contained deployment. See:
- model-serving/api-gateway/app/services/inference_logger.py
- model-serving/api-gateway/app/routers/drift.py
"""

__version__ = "1.0.0"
