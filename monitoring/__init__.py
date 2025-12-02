"""
Monitoring Module - Metrics, alerts, and HTTP endpoints

Provides:
- Prometheus metrics exporter
- HTTP control endpoints
- Multi-channel alerting (Slack, Telegram, Email)
- Alert templates
"""

from .exporter import (
    MetricsExporter,
    start_metrics_server,
    start_http_server,
    TradeMetrics
)
from .alerting import (
    Alert,
    AlertLevel,
    AlertManager,
    AlertTemplates,
    SlackNotifier,
    TelegramNotifier,
    EmailNotifier
)

__all__ = [
    # Metrics
    "MetricsExporter",
    "start_metrics_server",
    "start_http_server",
    "TradeMetrics",
    
    # Alerting
    "Alert",
    "AlertLevel",
    "AlertManager",
    "AlertTemplates",
    "SlackNotifier",
    "TelegramNotifier",
    "EmailNotifier",
]
