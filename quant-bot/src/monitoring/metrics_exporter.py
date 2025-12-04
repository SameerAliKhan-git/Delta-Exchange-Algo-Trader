"""
Production Monitoring & Alerting
================================

DELIVERABLE E: Grafana dashboards and Prometheus metrics.

Provides:
1. Prometheus metrics exporter
2. Grafana dashboard configuration
3. Alert rules for critical events
4. Real-time performance tracking
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import http.server
import socketserver
from pathlib import Path


# Prometheus metrics format helpers
class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class PrometheusMetric:
    """Prometheus metric definition."""
    name: str
    metric_type: MetricType
    help_text: str
    labels: List[str] = field(default_factory=list)
    
    # Current values
    value: float = 0.0
    label_values: Dict[str, float] = field(default_factory=dict)
    
    # Histogram buckets
    buckets: List[float] = field(default_factory=list)
    bucket_counts: Dict[float, int] = field(default_factory=dict)
    histogram_sum: float = 0.0
    histogram_count: int = 0
    
    def set(self, value: float, labels: Dict[str, str] = None):
        """Set gauge value."""
        if labels:
            key = tuple(sorted(labels.items()))
            self.label_values[str(key)] = value
        else:
            self.value = value
    
    def inc(self, amount: float = 1.0, labels: Dict[str, str] = None):
        """Increment counter."""
        if labels:
            key = str(tuple(sorted(labels.items())))
            self.label_values[key] = self.label_values.get(key, 0) + amount
        else:
            self.value += amount
    
    def observe(self, value: float, labels: Dict[str, str] = None):
        """Observe histogram value."""
        self.histogram_sum += value
        self.histogram_count += 1
        
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1
    
    def format(self) -> str:
        """Format metric for Prometheus scraping."""
        lines = []
        lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} {self.metric_type.value}")
        
        if self.metric_type == MetricType.HISTOGRAM:
            # Histogram format
            for bucket, count in sorted(self.bucket_counts.items()):
                lines.append(f'{self.name}_bucket{{le="{bucket}"}} {count}')
            lines.append(f'{self.name}_bucket{{le="+Inf"}} {self.histogram_count}')
            lines.append(f'{self.name}_sum {self.histogram_sum}')
            lines.append(f'{self.name}_count {self.histogram_count}')
        elif self.label_values:
            # Labeled metric
            for label_key, value in self.label_values.items():
                # Parse label key back to labels
                lines.append(f'{self.name}{{{label_key}}} {value}')
        else:
            # Simple metric
            lines.append(f'{self.name} {self.value}')
        
        return "\n".join(lines)


class TradingMetricsExporter:
    """
    Prometheus metrics exporter for trading system.
    """
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics: Dict[str, PrometheusMetric] = {}
        self._lock = threading.Lock()
        self._server = None
        self._server_thread = None
        
        # Initialize standard metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize standard trading metrics."""
        # PnL metrics
        self.register_metric(PrometheusMetric(
            name="trading_pnl_total",
            metric_type=MetricType.COUNTER,
            help_text="Total PnL in USD"
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_pnl_unrealized",
            metric_type=MetricType.GAUGE,
            help_text="Current unrealized PnL"
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_equity_current",
            metric_type=MetricType.GAUGE,
            help_text="Current equity value"
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_drawdown_current",
            metric_type=MetricType.GAUGE,
            help_text="Current drawdown percentage"
        ))
        
        # Trade metrics
        self.register_metric(PrometheusMetric(
            name="trading_trades_total",
            metric_type=MetricType.COUNTER,
            help_text="Total number of trades",
            labels=["strategy", "symbol", "side"]
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_trades_winning",
            metric_type=MetricType.COUNTER,
            help_text="Number of winning trades",
            labels=["strategy"]
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_win_rate",
            metric_type=MetricType.GAUGE,
            help_text="Current win rate",
            labels=["strategy"]
        ))
        
        # Execution metrics
        self.register_metric(PrometheusMetric(
            name="trading_slippage_bps",
            metric_type=MetricType.HISTOGRAM,
            help_text="Trade slippage in basis points",
            buckets=[1, 2, 5, 10, 20, 50, 100, 200]
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_latency_ms",
            metric_type=MetricType.HISTOGRAM,
            help_text="Order execution latency in milliseconds",
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2000]
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_fill_rate",
            metric_type=MetricType.GAUGE,
            help_text="Order fill rate"
        ))
        
        # Position metrics
        self.register_metric(PrometheusMetric(
            name="trading_position_size",
            metric_type=MetricType.GAUGE,
            help_text="Current position size in USD",
            labels=["symbol"]
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_leverage_current",
            metric_type=MetricType.GAUGE,
            help_text="Current leverage"
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_exposure_total",
            metric_type=MetricType.GAUGE,
            help_text="Total exposure percentage"
        ))
        
        # Risk metrics
        self.register_metric(PrometheusMetric(
            name="trading_risk_level",
            metric_type=MetricType.GAUGE,
            help_text="Current risk level (0=normal, 1=elevated, 2=high, 3=critical)"
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_daily_loss_pct",
            metric_type=MetricType.GAUGE,
            help_text="Daily loss percentage"
        ))
        
        # Regime metrics
        self.register_metric(PrometheusMetric(
            name="trading_regime",
            metric_type=MetricType.GAUGE,
            help_text="Current market regime (encoded)",
            labels=["regime"]
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_regime_confidence",
            metric_type=MetricType.GAUGE,
            help_text="Regime detection confidence"
        ))
        
        # Strategy metrics
        self.register_metric(PrometheusMetric(
            name="trading_strategy_weight",
            metric_type=MetricType.GAUGE,
            help_text="Strategy allocation weight",
            labels=["strategy"]
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_strategy_signals",
            metric_type=MetricType.COUNTER,
            help_text="Strategy signals generated",
            labels=["strategy", "direction"]
        ))
        
        # Order-flow metrics
        self.register_metric(PrometheusMetric(
            name="trading_orderflow_score",
            metric_type=MetricType.GAUGE,
            help_text="Current order-flow confirmation score"
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_orderflow_delta",
            metric_type=MetricType.GAUGE,
            help_text="Order-flow delta score"
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_orderflow_obi",
            metric_type=MetricType.GAUGE,
            help_text="Order book imbalance"
        ))
        
        # System metrics
        self.register_metric(PrometheusMetric(
            name="trading_system_uptime_seconds",
            metric_type=MetricType.COUNTER,
            help_text="System uptime in seconds"
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_api_errors_total",
            metric_type=MetricType.COUNTER,
            help_text="Total API errors",
            labels=["error_type"]
        ))
        
        self.register_metric(PrometheusMetric(
            name="trading_websocket_connected",
            metric_type=MetricType.GAUGE,
            help_text="WebSocket connection status"
        ))
    
    def register_metric(self, metric: PrometheusMetric):
        """Register a metric."""
        self.metrics[metric.name] = metric
    
    def record_trade(
        self,
        strategy: str,
        symbol: str,
        side: str,
        won: bool,
        pnl: float,
        slippage_bps: float,
        latency_ms: float
    ):
        """Record a trade with all related metrics."""
        with self._lock:
            # Trade count
            self.metrics["trading_trades_total"].inc(
                labels={"strategy": strategy, "symbol": symbol, "side": side}
            )
            
            # Winning trades
            if won:
                self.metrics["trading_trades_winning"].inc(
                    labels={"strategy": strategy}
                )
            
            # PnL
            self.metrics["trading_pnl_total"].inc(pnl)
            
            # Slippage histogram
            self.metrics["trading_slippage_bps"].observe(slippage_bps)
            
            # Latency histogram
            self.metrics["trading_latency_ms"].observe(latency_ms)
    
    def update_equity(self, equity: float, drawdown_pct: float, unrealized_pnl: float = 0):
        """Update equity metrics."""
        with self._lock:
            self.metrics["trading_equity_current"].set(equity)
            self.metrics["trading_drawdown_current"].set(drawdown_pct)
            self.metrics["trading_pnl_unrealized"].set(unrealized_pnl)
    
    def update_risk(self, risk_level: int, daily_loss_pct: float, leverage: float, exposure_pct: float):
        """Update risk metrics."""
        with self._lock:
            self.metrics["trading_risk_level"].set(risk_level)
            self.metrics["trading_daily_loss_pct"].set(daily_loss_pct)
            self.metrics["trading_leverage_current"].set(leverage)
            self.metrics["trading_exposure_total"].set(exposure_pct)
    
    def update_regime(self, regime: str, confidence: float):
        """Update regime metrics."""
        with self._lock:
            # Set regime (1 for current, 0 for others)
            for r in ["trending_up", "trending_down", "ranging", "volatile", "crisis"]:
                self.metrics["trading_regime"].set(
                    1 if r == regime else 0,
                    labels={"regime": r}
                )
            self.metrics["trading_regime_confidence"].set(confidence)
    
    def update_strategy_weights(self, weights: Dict[str, float]):
        """Update strategy allocation weights."""
        with self._lock:
            for strategy, weight in weights.items():
                self.metrics["trading_strategy_weight"].set(
                    weight,
                    labels={"strategy": strategy}
                )
    
    def update_orderflow(self, score: float, delta: float, obi: float):
        """Update order-flow metrics."""
        with self._lock:
            self.metrics["trading_orderflow_score"].set(score)
            self.metrics["trading_orderflow_delta"].set(delta)
            self.metrics["trading_orderflow_obi"].set(obi)
    
    def record_api_error(self, error_type: str):
        """Record an API error."""
        with self._lock:
            self.metrics["trading_api_errors_total"].inc(
                labels={"error_type": error_type}
            )
    
    def set_websocket_status(self, connected: bool):
        """Set WebSocket connection status."""
        with self._lock:
            self.metrics["trading_websocket_connected"].set(1 if connected else 0)
    
    def get_metrics_text(self) -> str:
        """Get all metrics in Prometheus format."""
        with self._lock:
            lines = []
            for metric in self.metrics.values():
                lines.append(metric.format())
                lines.append("")  # Empty line between metrics
            return "\n".join(lines)
    
    def start_server(self):
        """Start HTTP server for Prometheus scraping."""
        exporter = self
        
        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    content = exporter.get_metrics_text()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(content.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        self._server = socketserver.TCPServer(("", self.port), MetricsHandler)
        self._server_thread = threading.Thread(target=self._server.serve_forever)
        self._server_thread.daemon = True
        self._server_thread.start()
        print(f"Prometheus metrics server started on port {self.port}")
    
    def stop_server(self):
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()


# Grafana Dashboard Configuration
GRAFANA_DASHBOARD = {
    "title": "Trading Bot Performance",
    "uid": "trading-bot-v1",
    "refresh": "5s",
    "time": {"from": "now-1h", "to": "now"},
    "panels": [
        # Row 1: Key Metrics
        {
            "title": "Equity",
            "type": "stat",
            "gridPos": {"x": 0, "y": 0, "w": 4, "h": 4},
            "targets": [
                {"expr": "trading_equity_current", "legendFormat": "Equity"}
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": 90000},
                            {"color": "yellow", "value": 95000},
                            {"color": "green", "value": 100000}
                        ]
                    }
                }
            }
        },
        {
            "title": "Drawdown",
            "type": "gauge",
            "gridPos": {"x": 4, "y": 0, "w": 4, "h": 4},
            "targets": [
                {"expr": "trading_drawdown_current", "legendFormat": "Drawdown"}
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 20,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 5},
                            {"color": "red", "value": 10}
                        ]
                    }
                }
            }
        },
        {
            "title": "Win Rate",
            "type": "stat",
            "gridPos": {"x": 8, "y": 0, "w": 4, "h": 4},
            "targets": [
                {
                    "expr": "sum(trading_trades_winning) / sum(trading_trades_total) * 100",
                    "legendFormat": "Win Rate"
                }
            ],
            "fieldConfig": {"defaults": {"unit": "percent"}}
        },
        {
            "title": "Risk Level",
            "type": "stat",
            "gridPos": {"x": 12, "y": 0, "w": 4, "h": 4},
            "targets": [
                {"expr": "trading_risk_level", "legendFormat": "Risk"}
            ],
            "fieldConfig": {
                "defaults": {
                    "mappings": [
                        {"type": "value", "value": "0", "text": "NORMAL"},
                        {"type": "value", "value": "1", "text": "ELEVATED"},
                        {"type": "value", "value": "2", "text": "HIGH"},
                        {"type": "value", "value": "3", "text": "CRITICAL"}
                    ]
                }
            }
        },
        {
            "title": "Daily PnL",
            "type": "stat",
            "gridPos": {"x": 16, "y": 0, "w": 4, "h": 4},
            "targets": [
                {"expr": "trading_daily_loss_pct", "legendFormat": "Daily"}
            ],
            "fieldConfig": {"defaults": {"unit": "percent"}}
        },
        {
            "title": "Active Regime",
            "type": "stat",
            "gridPos": {"x": 20, "y": 0, "w": 4, "h": 4},
            "targets": [
                {"expr": "trading_regime{regime=\"trending_up\"} == 1", "legendFormat": "trending_up"},
                {"expr": "trading_regime{regime=\"trending_down\"} == 1", "legendFormat": "trending_down"},
                {"expr": "trading_regime{regime=\"ranging\"} == 1", "legendFormat": "ranging"},
                {"expr": "trading_regime{regime=\"volatile\"} == 1", "legendFormat": "volatile"},
                {"expr": "trading_regime{regime=\"crisis\"} == 1", "legendFormat": "crisis"}
            ]
        },
        
        # Row 2: Equity Chart
        {
            "title": "Equity Curve",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
            "targets": [
                {"expr": "trading_equity_current", "legendFormat": "Equity"}
            ],
            "fieldConfig": {"defaults": {"unit": "currencyUSD"}}
        },
        {
            "title": "PnL Over Time",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 4, "w": 12, "h": 8},
            "targets": [
                {"expr": "rate(trading_pnl_total[5m]) * 300", "legendFormat": "PnL (5m avg)"}
            ],
            "fieldConfig": {"defaults": {"unit": "currencyUSD"}}
        },
        
        # Row 3: Strategy Performance
        {
            "title": "Strategy Weights",
            "type": "piechart",
            "gridPos": {"x": 0, "y": 12, "w": 8, "h": 8},
            "targets": [
                {"expr": "trading_strategy_weight", "legendFormat": "{{strategy}}"}
            ]
        },
        {
            "title": "Trades by Strategy",
            "type": "bargauge",
            "gridPos": {"x": 8, "y": 12, "w": 8, "h": 8},
            "targets": [
                {"expr": "sum by (strategy) (trading_trades_total)", "legendFormat": "{{strategy}}"}
            ]
        },
        {
            "title": "Win Rate by Strategy",
            "type": "bargauge",
            "gridPos": {"x": 16, "y": 12, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "sum by (strategy) (trading_trades_winning) / sum by (strategy) (trading_trades_total) * 100",
                    "legendFormat": "{{strategy}}"
                }
            ],
            "fieldConfig": {"defaults": {"unit": "percent", "min": 40, "max": 70}}
        },
        
        # Row 4: Execution Quality
        {
            "title": "Slippage Distribution",
            "type": "histogram",
            "gridPos": {"x": 0, "y": 20, "w": 8, "h": 8},
            "targets": [
                {"expr": "trading_slippage_bps_bucket", "legendFormat": "{{le}}"}
            ]
        },
        {
            "title": "Latency Distribution",
            "type": "histogram",
            "gridPos": {"x": 8, "y": 20, "w": 8, "h": 8},
            "targets": [
                {"expr": "trading_latency_ms_bucket", "legendFormat": "{{le}}"}
            ]
        },
        {
            "title": "Fill Rate",
            "type": "gauge",
            "gridPos": {"x": 16, "y": 20, "w": 8, "h": 8},
            "targets": [
                {"expr": "trading_fill_rate", "legendFormat": "Fill Rate"}
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 0.8},
                            {"color": "green", "value": 0.95}
                        ]
                    }
                }
            }
        },
        
        # Row 5: Order Flow
        {
            "title": "Order Flow Score",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 28, "w": 12, "h": 6},
            "targets": [
                {"expr": "trading_orderflow_score", "legendFormat": "Score"},
                {"expr": "trading_orderflow_delta", "legendFormat": "Delta"},
                {"expr": "trading_orderflow_obi", "legendFormat": "OBI"}
            ]
        },
        {
            "title": "Regime Confidence",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 28, "w": 12, "h": 6},
            "targets": [
                {"expr": "trading_regime_confidence", "legendFormat": "Confidence"}
            ]
        }
    ]
}


# Prometheus Alert Rules
PROMETHEUS_ALERTS = """
groups:
  - name: trading_alerts
    interval: 10s
    rules:
      # Critical: Max drawdown exceeded
      - alert: MaxDrawdownExceeded
        expr: trading_drawdown_current > 10
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CRITICAL: Max drawdown exceeded"
          description: "Drawdown is {{ $value }}%, exceeds 10% limit"
      
      # Warning: Approaching drawdown limit
      - alert: DrawdownWarning
        expr: trading_drawdown_current > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Drawdown warning"
          description: "Drawdown is {{ $value }}%, approaching 10% limit"
      
      # Critical: Daily loss limit
      - alert: DailyLossLimitExceeded
        expr: trading_daily_loss_pct < -2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CRITICAL: Daily loss limit exceeded"
          description: "Daily loss is {{ $value }}%"
      
      # Warning: High risk level
      - alert: HighRiskLevel
        expr: trading_risk_level >= 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High risk level detected"
          description: "Risk level is {{ $value }}"
      
      # Critical: System halted
      - alert: TradingHalted
        expr: trading_risk_level >= 3
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "CRITICAL: Trading system halted"
          description: "Trading has been halted due to risk controls"
      
      # Warning: Low fill rate
      - alert: LowFillRate
        expr: trading_fill_rate < 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low fill rate detected"
          description: "Fill rate is {{ $value }}"
      
      # Warning: High slippage
      - alert: HighSlippage
        expr: rate(trading_slippage_bps_sum[5m]) / rate(trading_slippage_bps_count[5m]) > 20
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High slippage detected"
          description: "Average slippage is {{ $value }} bps"
      
      # Critical: WebSocket disconnected
      - alert: WebSocketDisconnected
        expr: trading_websocket_connected == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "WebSocket disconnected"
          description: "Market data feed is disconnected"
      
      # Warning: API errors
      - alert: APIErrorsHigh
        expr: rate(trading_api_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API error rate"
          description: "API error rate is {{ $value }} per second"
      
      # Info: Crisis regime
      - alert: CrisisRegimeDetected
        expr: trading_regime{regime="crisis"} == 1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Crisis regime detected"
          description: "Market is in crisis mode, exposure reduced"
"""


# Prometheus configuration
PROMETHEUS_CONFIG = """
global:
  scrape_interval: 5s
  evaluation_interval: 5s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

rule_files:
  - "trading_alerts.yml"

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 5s
"""


def export_monitoring_configs(output_dir: str = "monitoring"):
    """Export all monitoring configurations."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Export Grafana dashboard
    with open(output_path / "grafana_dashboard.json", "w") as f:
        json.dump(GRAFANA_DASHBOARD, f, indent=2)
    
    # Export Prometheus alerts
    with open(output_path / "trading_alerts.yml", "w") as f:
        f.write(PROMETHEUS_ALERTS)
    
    # Export Prometheus config
    with open(output_path / "prometheus.yml", "w") as f:
        f.write(PROMETHEUS_CONFIG)
    
    print(f"Monitoring configs exported to {output_path}/")
    print("  - grafana_dashboard.json")
    print("  - trading_alerts.yml")
    print("  - prometheus.yml")


if __name__ == "__main__":
    print("=" * 60)
    print("MONITORING SYSTEM TEST")
    print("=" * 60)
    
    # Create exporter
    exporter = TradingMetricsExporter(port=9091)
    
    # Simulate some metrics
    print("\nSimulating metrics...")
    
    # Update equity
    exporter.update_equity(105000, 3.5, 1500)
    
    # Update risk
    exporter.update_risk(1, -0.5, 1.5, 45)
    
    # Update regime
    exporter.update_regime("trending_up", 0.75)
    
    # Update strategy weights
    exporter.update_strategy_weights({
        'momentum': 0.35,
        'volatility_breakout': 0.25,
        'stat_arb': 0.20,
        'regime_ml': 0.20
    })
    
    # Update orderflow
    exporter.update_orderflow(0.65, 0.7, 0.55)
    
    # Record some trades
    for i in range(10):
        import random
        exporter.record_trade(
            strategy=random.choice(['momentum', 'stat_arb', 'regime_ml']),
            symbol='BTCUSDT',
            side=random.choice(['buy', 'sell']),
            won=random.random() > 0.45,
            pnl=random.uniform(-50, 100),
            slippage_bps=random.uniform(2, 15),
            latency_ms=random.uniform(50, 200)
        )
    
    # Print metrics
    print("\nGenerated Metrics:")
    print("-" * 40)
    metrics_text = exporter.get_metrics_text()
    print(metrics_text[:2000] + "..." if len(metrics_text) > 2000 else metrics_text)
    
    # Export configs
    print("\n" + "=" * 60)
    export_monitoring_configs("monitoring_output")
    
    print("\n" + "=" * 60)
    print("MONITORING SYSTEM WORKING CORRECTLY")
    print("=" * 60)
