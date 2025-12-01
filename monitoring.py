"""
Monitoring and Alerting Module for Delta Exchange Algo Trading Bot
Prometheus metrics, health checks, and multi-channel alerts
"""

import asyncio
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

import requests
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess, REGISTRY
)

from config import get_config
from logger import get_logger


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert message"""
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


class PrometheusMetrics:
    """
    Prometheus metrics for trading bot
    Exposes metrics at /metrics endpoint
    """
    
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.registry = registry
        
        # Trading metrics
        self.trades_total = Counter(
            'trading_bot_trades_total',
            'Total number of trades',
            ['side', 'status'],
            registry=registry
        )
        
        self.orders_total = Counter(
            'trading_bot_orders_total',
            'Total number of orders',
            ['type', 'side', 'status'],
            registry=registry
        )
        
        self.pnl_total = Gauge(
            'trading_bot_pnl_total',
            'Total realized PnL',
            registry=registry
        )
        
        self.pnl_daily = Gauge(
            'trading_bot_pnl_daily',
            'Daily realized PnL',
            registry=registry
        )
        
        self.unrealized_pnl = Gauge(
            'trading_bot_unrealized_pnl',
            'Current unrealized PnL',
            registry=registry
        )
        
        self.position_size = Gauge(
            'trading_bot_position_size',
            'Current position size',
            ['product_id', 'side'],
            registry=registry
        )
        
        self.open_positions = Gauge(
            'trading_bot_open_positions',
            'Number of open positions',
            registry=registry
        )
        
        self.open_orders = Gauge(
            'trading_bot_open_orders',
            'Number of open orders',
            registry=registry
        )
        
        # Market data metrics
        self.current_price = Gauge(
            'trading_bot_current_price',
            'Current market price',
            ['symbol'],
            registry=registry
        )
        
        self.spread = Gauge(
            'trading_bot_spread',
            'Current bid-ask spread',
            ['symbol'],
            registry=registry
        )
        
        self.orderbook_imbalance = Gauge(
            'trading_bot_orderbook_imbalance',
            'Orderbook imbalance (-1 to 1)',
            ['symbol'],
            registry=registry
        )
        
        # Sentiment metrics
        self.sentiment_score = Gauge(
            'trading_bot_sentiment_score',
            'Current sentiment score (-1 to 1)',
            registry=registry
        )
        
        # Strategy metrics
        self.signal_strength = Gauge(
            'trading_bot_signal_strength',
            'Current signal strength',
            ['direction'],
            registry=registry
        )
        
        self.composite_score = Gauge(
            'trading_bot_composite_score',
            'Composite strategy score',
            registry=registry
        )
        
        # System metrics
        self.api_requests = Counter(
            'trading_bot_api_requests_total',
            'Total API requests',
            ['endpoint', 'status'],
            registry=registry
        )
        
        self.api_latency = Histogram(
            'trading_bot_api_latency_seconds',
            'API request latency',
            ['endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=registry
        )
        
        self.errors_total = Counter(
            'trading_bot_errors_total',
            'Total errors',
            ['type'],
            registry=registry
        )
        
        self.kill_switch_active = Gauge(
            'trading_bot_kill_switch_active',
            'Kill switch status (1=active)',
            registry=registry
        )
        
        self.uptime_seconds = Gauge(
            'trading_bot_uptime_seconds',
            'Bot uptime in seconds',
            registry=registry
        )
        
        self.last_trade_timestamp = Gauge(
            'trading_bot_last_trade_timestamp',
            'Timestamp of last trade',
            registry=registry
        )
        
        self.health_status = Gauge(
            'trading_bot_health_status',
            'Health status (1=healthy)',
            registry=registry
        )
        
        # Risk metrics
        self.daily_loss_remaining = Gauge(
            'trading_bot_daily_loss_remaining',
            'Remaining daily loss allowance',
            registry=registry
        )
        
        self.win_rate = Gauge(
            'trading_bot_win_rate',
            'Win rate (0-1)',
            registry=registry
        )
        
        self._start_time = time.time()
    
    def update_uptime(self):
        """Update uptime metric"""
        self.uptime_seconds.set(time.time() - self._start_time)
    
    def record_trade(self, side: str, status: str, pnl: float = 0.0):
        """Record a trade"""
        self.trades_total.labels(side=side, status=status).inc()
        if status == "filled":
            self.pnl_total.inc(pnl)
    
    def record_order(self, order_type: str, side: str, status: str):
        """Record an order"""
        self.orders_total.labels(type=order_type, side=side, status=status).inc()
    
    def record_api_request(self, endpoint: str, status: str, latency: float):
        """Record an API request"""
        self.api_requests.labels(endpoint=endpoint, status=status).inc()
        self.api_latency.labels(endpoint=endpoint).observe(latency)
    
    def record_error(self, error_type: str):
        """Record an error"""
        self.errors_total.labels(type=error_type).inc()


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics and control endpoints"""
    
    metrics: PrometheusMetrics = None
    health_checker: Callable[[], bool] = None
    bot_instance = None  # Reference to trading bot for control
    
    def do_GET(self):
        if self.path == '/metrics':
            self._serve_metrics()
        elif self.path == '/health':
            self._serve_health()
        elif self.path == '/status':
            self._serve_status()
        elif self.path == '/kill-switch':
            self._serve_kill_switch_status()
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        if self.path == '/kill-switch/activate':
            self._activate_kill_switch()
        elif self.path == '/kill-switch/deactivate':
            self._deactivate_kill_switch()
        elif self.path == '/emergency-stop':
            self._emergency_stop()
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def _serve_metrics(self):
        """Serve Prometheus metrics"""
        try:
            output = generate_latest(REGISTRY)
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
    
    def _serve_health(self):
        """Serve health check"""
        try:
            is_healthy = self.health_checker() if self.health_checker else True
            status = 200 if is_healthy else 503
            
            response = {
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0"
            }
            
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception:
            self.send_response(500)
            self.end_headers()
    
    def _serve_status(self):
        """Serve comprehensive bot status"""
        try:
            from risk_manager import get_risk_manager
            from product_discovery import get_product_discovery
            from instrument_selector import get_instrument_selector
            
            risk_mgr = get_risk_manager()
            discovery = get_product_discovery()
            selector = get_instrument_selector()
            
            # Get various metrics
            risk_metrics = risk_mgr.get_risk_metrics()
            capital_metrics = risk_mgr.get_capital_metrics()
            discovery_summary = discovery.get_tradable_summary()
            selector_summary = selector.get_selection_summary()
            
            response = {
                "status": "running",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0",
                "risk": risk_metrics,
                "capital": capital_metrics,
                "instruments": discovery_summary,
                "selector": selector_summary,
                "health": {
                    "api_healthy": self.health_checker() if self.health_checker else True,
                    "kill_switch": risk_mgr.is_kill_switch_active
                }
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response, default=str).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _serve_kill_switch_status(self):
        """Serve kill switch status"""
        try:
            from risk_manager import get_risk_manager
            risk_mgr = get_risk_manager()
            
            response = {
                "active": risk_mgr.is_kill_switch_active,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _activate_kill_switch(self):
        """Activate kill switch via HTTP"""
        try:
            from risk_manager import get_risk_manager
            
            # Read reason from body if present
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode() if content_length > 0 else ""
            
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {}
            
            reason = data.get("reason", "HTTP API request")
            
            risk_mgr = get_risk_manager()
            risk_mgr.activate_kill_switch(reason)
            
            response = {
                "success": True,
                "message": "Kill switch activated",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _deactivate_kill_switch(self):
        """Deactivate kill switch via HTTP"""
        try:
            from risk_manager import get_risk_manager
            
            risk_mgr = get_risk_manager()
            risk_mgr.deactivate_kill_switch()
            
            response = {
                "success": True,
                "message": "Kill switch deactivated",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _emergency_stop(self):
        """Trigger emergency stop via HTTP"""
        try:
            # Read reason from body if present
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode() if content_length > 0 else ""
            
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {}
            
            reason = data.get("reason", "HTTP API emergency stop")
            
            # Call emergency stop if bot instance available
            if self.bot_instance and hasattr(self.bot_instance, 'emergency_stop'):
                self.bot_instance.emergency_stop(reason)
            else:
                # Fallback to just activating kill switch
                from risk_manager import get_risk_manager
                risk_mgr = get_risk_manager()
                risk_mgr.activate_kill_switch(reason)
            
            response = {
                "success": True,
                "message": "Emergency stop triggered",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def log_message(self, format, *args):
        """Suppress HTTP logs"""
        pass


class TelegramAlerter:
    """Send alerts via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = get_logger()
    
    def send(self, alert: Alert) -> bool:
        """Send alert to Telegram"""
        if not self.bot_token or not self.chat_id:
            return False
        
        try:
            # Format message with emoji based on severity
            emoji_map = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            emoji = emoji_map.get(alert.severity, "📊")
            
            message = f"{emoji} *{alert.title}*\n\n"
            message += f"{alert.message}\n\n"
            message += f"_Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}_"
            
            if alert.metadata:
                message += "\n\n*Details:*\n"
                for key, value in alert.metadata.items():
                    message += f"• {key}: `{value}`\n"
            
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                },
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error("Telegram alert failed", error=str(e))
            return False


class SlackAlerter:
    """Send alerts via Slack webhook"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = get_logger()
    
    def send(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        if not self.webhook_url:
            return False
        
        try:
            # Color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffa500",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }
            
            color = color_map.get(alert.severity, "#808080")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in (alert.metadata or {}).items()
                    ],
                    "footer": "Delta Algo Bot",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error("Slack alert failed", error=str(e))
            return False


class AlertManager:
    """
    Centralized alert management
    Routes alerts to configured channels
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
        # Initialize alerters
        self.telegram = None
        self.slack = None
        
        if self.config.alerting.telegram_bot_token and self.config.alerting.telegram_chat_id:
            self.telegram = TelegramAlerter(
                self.config.alerting.telegram_bot_token,
                self.config.alerting.telegram_chat_id
            )
            self.logger.info("Telegram alerting enabled")
        
        if self.config.alerting.slack_webhook_url:
            self.slack = SlackAlerter(self.config.alerting.slack_webhook_url)
            self.logger.info("Slack alerting enabled")
        
        # Alert history
        self._alert_history: List[Alert] = []
        self._max_history = 1000
        
        # Rate limiting
        self._last_alert_time: Dict[str, float] = {}
        self._min_interval = 60  # Minimum seconds between same alerts
    
    def _should_send(self, alert_key: str) -> bool:
        """Check rate limiting"""
        now = time.time()
        last_time = self._last_alert_time.get(alert_key, 0)
        
        if now - last_time < self._min_interval:
            return False
        
        self._last_alert_time[alert_key] = now
        return True
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: Dict[str, Any] = None,
        rate_limit_key: str = None
    ):
        """Send alert to all configured channels"""
        # Rate limiting
        if rate_limit_key and not self._should_send(rate_limit_key):
            return
        
        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        # Store in history
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical"
        }.get(severity, "info")
        
        getattr(self.logger, log_level)(f"ALERT: {title}", message=message, metadata=metadata)
        
        # Send to channels
        if self.telegram:
            self.telegram.send(alert)
        
        if self.slack:
            self.slack.send(alert)
    
    # Convenience methods for common alerts
    def alert_order_placed(self, order_data: Dict[str, Any]):
        self.send_alert(
            title="Order Placed",
            message=f"New {order_data.get('side', '')} order placed",
            severity=AlertSeverity.INFO,
            metadata=order_data,
            rate_limit_key="order_placed"
        )
    
    def alert_order_filled(self, order_data: Dict[str, Any]):
        self.send_alert(
            title="Order Filled",
            message=f"Order filled at {order_data.get('price', 'N/A')}",
            severity=AlertSeverity.INFO,
            metadata=order_data
        )
    
    def alert_stop_triggered(self, stop_data: Dict[str, Any]):
        self.send_alert(
            title="Stop Loss Triggered",
            message=f"Stop loss triggered at {stop_data.get('price', 'N/A')}",
            severity=AlertSeverity.WARNING,
            metadata=stop_data
        )
    
    def alert_daily_loss_limit(self, pnl: float, limit: float):
        self.send_alert(
            title="⚠️ Daily Loss Limit Hit",
            message=f"Daily loss limit reached. Trading halted.",
            severity=AlertSeverity.CRITICAL,
            metadata={"daily_pnl": pnl, "limit": limit}
        )
    
    def alert_kill_switch(self, reason: str):
        self.send_alert(
            title="🚨 KILL SWITCH ACTIVATED",
            message=f"Trading has been stopped: {reason}",
            severity=AlertSeverity.CRITICAL,
            metadata={"reason": reason}
        )
    
    def alert_error(self, error_type: str, error_message: str):
        self.send_alert(
            title=f"Error: {error_type}",
            message=error_message,
            severity=AlertSeverity.ERROR,
            metadata={"error_type": error_type},
            rate_limit_key=f"error_{error_type}"
        )
    
    def alert_startup(self, config_summary: Dict[str, Any]):
        self.send_alert(
            title="🤖 Trading Bot Started",
            message="Delta Exchange Algo Bot is now running",
            severity=AlertSeverity.INFO,
            metadata=config_summary
        )
    
    def alert_shutdown(self, reason: str = "Normal shutdown"):
        self.send_alert(
            title="Bot Shutdown",
            message=f"Trading bot is shutting down: {reason}",
            severity=AlertSeverity.WARNING,
            metadata={"reason": reason}
        )


class MonitoringServer:
    """
    HTTP server for metrics, health checks, and control endpoints
    
    Endpoints:
    - GET /metrics - Prometheus metrics
    - GET /health - Health check
    - GET /status - Comprehensive bot status
    - GET /kill-switch - Kill switch status
    - POST /kill-switch/activate - Activate kill switch
    - POST /kill-switch/deactivate - Deactivate kill switch
    - POST /emergency-stop - Trigger emergency stop
    """
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.logger = get_logger()
        self.metrics = PrometheusMetrics()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._health_checker: Optional[Callable[[], bool]] = None
    
    def set_health_checker(self, checker: Callable[[], bool]):
        """Set health check function"""
        self._health_checker = checker
        MetricsHTTPHandler.health_checker = checker
    
    def set_bot_instance(self, bot):
        """Set bot instance for control endpoints"""
        MetricsHTTPHandler.bot_instance = bot
    
    def start(self):
        """Start the monitoring server"""
        MetricsHTTPHandler.metrics = self.metrics
        
        try:
            self._server = HTTPServer(('0.0.0.0', self.port), MetricsHTTPHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            self.logger.info(
                "Monitoring server started",
                port=self.port,
                endpoints=["/metrics", "/health", "/status", "/kill-switch", "/emergency-stop"]
            )
        except Exception as e:
            self.logger.error("Failed to start monitoring server", error=str(e))
    
    def stop(self):
        """Stop the monitoring server"""
        if self._server:
            self._server.shutdown()
            self.logger.info("Monitoring server stopped")


# Singleton instances
_metrics: Optional[PrometheusMetrics] = None
_alert_manager: Optional[AlertManager] = None
_monitoring_server: Optional[MonitoringServer] = None


def get_metrics() -> PrometheusMetrics:
    """Get or create global metrics"""
    global _metrics
    if _metrics is None:
        _metrics = PrometheusMetrics()
    return _metrics


def get_alert_manager() -> AlertManager:
    """Get or create global alert manager"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def get_monitoring_server() -> MonitoringServer:
    """Get or create global monitoring server"""
    global _monitoring_server
    if _monitoring_server is None:
        config = get_config()
        _monitoring_server = MonitoringServer(port=config.monitoring.prometheus_port)
    return _monitoring_server


if __name__ == "__main__":
    # Test monitoring
    server = get_monitoring_server()
    server.start()
    
    metrics = get_metrics()
    alerts = get_alert_manager()
    
    # Record some test metrics
    metrics.current_price.labels(symbol="BTCUSD").set(50000)
    metrics.sentiment_score.set(0.5)
    metrics.health_status.set(1)
    metrics.update_uptime()
    
    print(f"Monitoring server running on port {server.port}")
    print("Endpoints: /metrics, /health")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            metrics.update_uptime()
    except KeyboardInterrupt:
        server.stop()
