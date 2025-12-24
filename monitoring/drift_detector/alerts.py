"""
Alert Generation for Drift Detection

Generates alerts based on drift detection results and sends
notifications via webhooks or logs.
"""

import os
import json
import logging
import httpx
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from monitoring.baseline.schema import (
    DriftAlert,
    DriftSeverity,
    DriftReport,
)
from .config import DriftConfig

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alerting."""
    enabled: bool = True
    webhook_url: Optional[str] = None
    log_alerts: bool = True
    
    # Thresholds for alert severity
    drift_score_warning: float = 0.2
    drift_score_critical: float = 0.5
    features_drifted_warning: int = 3
    prediction_drift_critical: float = 0.3


class AlertGenerator:
    """
    Generates alerts from drift detection results.
    
    Creates structured alerts with severity levels and recommendations.
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
    
    def generate_alerts(
        self,
        drift_report: DriftReport,
        config: Optional[DriftConfig] = None,
    ) -> List[DriftAlert]:
        """
        Generate alerts from drift report.
        
        Args:
            drift_report: Completed drift detection report
            config: Optional drift config with custom thresholds
            
        Returns:
            List of DriftAlert objects
        """
        alerts = []
        cfg = config or DriftConfig()
        
        # Check overall drift score
        if drift_report.drift_score >= cfg.drift_score_critical_threshold:
            alerts.append(DriftAlert(
                severity=DriftSeverity.CRITICAL,
                message=f"Critical drift detected: overall drift score {drift_report.drift_score:.2%} exceeds threshold {cfg.drift_score_critical_threshold:.0%}",
                recommendation="Immediate investigation required. Consider pausing model predictions and initiating retraining.",
                metric_name="drift_score",
                metric_value=drift_report.drift_score,
                threshold=cfg.drift_score_critical_threshold,
            ))
        elif drift_report.drift_score >= cfg.drift_score_warning_threshold:
            alerts.append(DriftAlert(
                severity=DriftSeverity.WARNING,
                message=f"Drift detected: overall drift score {drift_report.drift_score:.2%} exceeds threshold {cfg.drift_score_warning_threshold:.0%}",
                recommendation="Monitor closely and consider scheduling model retraining.",
                metric_name="drift_score",
                metric_value=drift_report.drift_score,
                threshold=cfg.drift_score_warning_threshold,
            ))
        
        # Check prediction drift
        if drift_report.prediction_drift:
            if drift_report.prediction_drift.psi >= cfg.prediction_drift_critical_threshold:
                alerts.append(DriftAlert(
                    severity=DriftSeverity.CRITICAL,
                    feature="prediction_probability",
                    message=f"Critical prediction drift: PSI {drift_report.prediction_drift.psi:.3f} exceeds threshold {cfg.prediction_drift_critical_threshold}",
                    recommendation="Model predictions may be unreliable. Review recent model outputs and consider retraining.",
                    metric_name="prediction_psi",
                    metric_value=drift_report.prediction_drift.psi,
                    threshold=cfg.prediction_drift_critical_threshold,
                ))
        
        # Check number of drifted features
        if drift_report.features_drifted_count >= cfg.features_drifted_warning_count:
            alerts.append(DriftAlert(
                severity=DriftSeverity.WARNING,
                message=f"Multiple features drifted: {drift_report.features_drifted_count} features show drift",
                recommendation=f"Investigate data pipeline for changes. Drifted features: {', '.join(drift_report.features_drifted_names[:5])}",
                metric_name="features_drifted_count",
                metric_value=float(drift_report.features_drifted_count),
                threshold=float(cfg.features_drifted_warning_count),
            ))
        
        # Check individual feature drift
        for feature_name, feature_result in drift_report.feature_drift.items():
            if feature_result.drift_detected and feature_result.psi >= cfg.psi_threshold:
                alerts.append(DriftAlert(
                    severity=DriftSeverity.WARNING,
                    feature=feature_name,
                    message=f"Feature drift detected: {feature_name} has PSI {feature_result.psi:.3f}",
                    recommendation=f"Investigate changes in {feature_name}. Mean shifted from {feature_result.baseline_mean:.3f} to {feature_result.current_mean:.3f}",
                    metric_name="feature_psi",
                    metric_value=feature_result.psi,
                    threshold=cfg.psi_threshold,
                ))
        
        # Check data quality
        if drift_report.data_quality.issues_detected:
            if drift_report.data_quality.missing_values_pct > cfg.missing_values_threshold_pct:
                alerts.append(DriftAlert(
                    severity=DriftSeverity.WARNING,
                    message=f"Data quality issue: {drift_report.data_quality.missing_values_pct:.1f}% missing values",
                    recommendation="Check data pipeline for issues. Missing data may impact prediction quality.",
                    metric_name="missing_values_pct",
                    metric_value=drift_report.data_quality.missing_values_pct,
                    threshold=cfg.missing_values_threshold_pct,
                ))
            
            if drift_report.data_quality.out_of_range_pct > cfg.out_of_range_threshold_pct:
                alerts.append(DriftAlert(
                    severity=DriftSeverity.WARNING,
                    message=f"Data quality issue: {drift_report.data_quality.out_of_range_pct:.1f}% out-of-range values",
                    recommendation="Investigate data source for anomalies or schema changes.",
                    metric_name="out_of_range_pct",
                    metric_value=drift_report.data_quality.out_of_range_pct,
                    threshold=cfg.out_of_range_threshold_pct,
                ))
        
        return alerts


class AlertNotifier:
    """
    Sends alert notifications via various channels.
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
    
    async def send_alerts(
        self,
        alerts: List[DriftAlert],
        drift_report: DriftReport,
    ):
        """
        Send alerts through configured channels.
        
        Args:
            alerts: List of alerts to send
            drift_report: Full drift report for context
        """
        if not self.config.enabled or not alerts:
            return
        
        # Log alerts
        if self.config.log_alerts:
            self._log_alerts(alerts, drift_report)
        
        # Send webhook notification
        if self.config.webhook_url:
            await self._send_webhook(alerts, drift_report)
    
    def _log_alerts(
        self,
        alerts: List[DriftAlert],
        drift_report: DriftReport,
    ):
        """Log alerts with structured format."""
        for alert in alerts:
            log_data = {
                "alert_type": "MODEL_DRIFT",
                "severity": alert.severity.value,
                "model_name": drift_report.model_name,
                "model_version": drift_report.model_version,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "message": alert.message,
                "feature": alert.feature,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "report_id": drift_report.report_id,
            }
            
            if alert.severity == DriftSeverity.CRITICAL:
                logger.critical(f"DRIFT_ALERT: {json.dumps(log_data)}")
            elif alert.severity == DriftSeverity.WARNING:
                logger.warning(f"DRIFT_ALERT: {json.dumps(log_data)}")
            else:
                logger.info(f"DRIFT_ALERT: {json.dumps(log_data)}")
    
    async def _send_webhook(
        self,
        alerts: List[DriftAlert],
        drift_report: DriftReport,
    ):
        """Send alerts to webhook URL."""
        if not self.config.webhook_url:
            return
        
        # Determine overall severity
        severities = [a.severity for a in alerts]
        overall_severity = DriftSeverity.CRITICAL if DriftSeverity.CRITICAL in severities else DriftSeverity.WARNING
        
        payload = {
            "alert_type": "MODEL_DRIFT",
            "severity": overall_severity.value,
            "model_name": drift_report.model_name,
            "model_version": drift_report.model_version,
            "drift_score": drift_report.drift_score,
            "drifted_features": drift_report.features_drifted_names[:10],
            "alerts_count": len(alerts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report_id": drift_report.report_id,
            "alerts": [
                {
                    "severity": a.severity.value,
                    "message": a.message,
                    "feature": a.feature,
                    "metric": a.metric_name,
                    "value": a.metric_value,
                }
                for a in alerts[:10]  # Limit to 10 alerts in webhook
            ],
            "action_url": f"https://mlflow/models/{drift_report.model_name}",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.info(f"Sent webhook alert for {len(alerts)} alerts")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")


def create_slack_payload(
    alerts: List[DriftAlert],
    drift_report: DriftReport,
) -> Dict[str, Any]:
    """
    Create a Slack-formatted webhook payload.
    
    Args:
        alerts: List of alerts
        drift_report: Drift report
        
    Returns:
        Slack webhook payload
    """
    # Determine color based on severity
    has_critical = any(a.severity == DriftSeverity.CRITICAL for a in alerts)
    color = "#dc3545" if has_critical else "#ffc107"  # Red for critical, yellow for warning
    
    # Build alert text
    alert_lines = []
    for alert in alerts[:5]:
        emoji = "🔴" if alert.severity == DriftSeverity.CRITICAL else "🟡"
        alert_lines.append(f"{emoji} {alert.message}")
    
    if len(alerts) > 5:
        alert_lines.append(f"... and {len(alerts) - 5} more alerts")
    
    return {
        "attachments": [
            {
                "color": color,
                "title": f"Model Drift Alert - {drift_report.model_name}",
                "text": "\n".join(alert_lines),
                "fields": [
                    {
                        "title": "Drift Score",
                        "value": f"{drift_report.drift_score:.1%}",
                        "short": True,
                    },
                    {
                        "title": "Features Drifted",
                        "value": str(drift_report.features_drifted_count),
                        "short": True,
                    },
                    {
                        "title": "Model Version",
                        "value": drift_report.model_version,
                        "short": True,
                    },
                    {
                        "title": "Samples Analyzed",
                        "value": str(drift_report.analysis_window.num_samples),
                        "short": True,
                    },
                ],
                "footer": f"Report ID: {drift_report.report_id}",
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }
        ]
    }
