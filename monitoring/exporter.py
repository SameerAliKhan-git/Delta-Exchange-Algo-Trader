"""
Metrics Exporter - Prometheus metrics and HTTP endpoints

Provides:
- Trading metrics (trades, P&L, signals)
- System metrics (latency, errors)
- HTTP control endpoints
"""

import time
import json
import threading
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime


@dataclass
class TradeMetrics:
    """Trading metrics container"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Signals
    signals_generated: int = 0
    orders_placed: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    
    # Risk
    daily_pnl: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # Timing
    last_trade_time: float = 0.0
    last_signal_time: float = 0.0
    
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100


class MetricsExporter:
    """
    Export metrics to Prometheus format
    """
    
    def __init__(self):
        self.metrics = TradeMetrics()
        self._custom_metrics: Dict[str, float] = {}
        self._labels: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def set_label(self, key: str, value: str) -> None:
        """Set a label for metrics"""
        self._labels[key] = value
    
    def record_trade(self, pnl: float, is_win: bool) -> None:
        """Record completed trade"""
        with self._lock:
            self.metrics.total_trades += 1
            if is_win:
                self.metrics.winning_trades += 1
            else:
                self.metrics.losing_trades += 1
            self.metrics.realized_pnl += pnl
            self.metrics.total_pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl
            self.metrics.last_trade_time = time.time()
    
    def record_signal(self) -> None:
        """Record signal generated"""
        with self._lock:
            self.metrics.signals_generated += 1
            self.metrics.last_signal_time = time.time()
    
    def record_order(self, status: str) -> None:
        """Record order status"""
        with self._lock:
            self.metrics.orders_placed += 1
            if status == 'filled':
                self.metrics.orders_filled += 1
            elif status == 'rejected':
                self.metrics.orders_rejected += 1
    
    def update_unrealized_pnl(self, pnl: float) -> None:
        """Update unrealized P&L"""
        with self._lock:
            self.metrics.unrealized_pnl = pnl
            self.metrics.total_pnl = self.metrics.realized_pnl + pnl
    
    def update_risk_metrics(self, daily_pnl: float, drawdown: float) -> None:
        """Update risk metrics"""
        with self._lock:
            self.metrics.daily_pnl = daily_pnl
            self.metrics.current_drawdown = drawdown
            self.metrics.max_drawdown = max(self.metrics.max_drawdown, drawdown)
    
    def set_metric(self, name: str, value: float) -> None:
        """Set custom metric"""
        with self._lock:
            self._custom_metrics[name] = value
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        lines = []
        
        # Labels string
        labels = ','.join(f'{k}="{v}"' for k, v in self._labels.items())
        label_str = f'{{{labels}}}' if labels else ''
        
        with self._lock:
            # Trade metrics
            lines.append(f'delta_trades_total{label_str} {self.metrics.total_trades}')
            lines.append(f'delta_trades_winning{label_str} {self.metrics.winning_trades}')
            lines.append(f'delta_trades_losing{label_str} {self.metrics.losing_trades}')
            lines.append(f'delta_win_rate{label_str} {self.metrics.win_rate()}')
            
            # P&L metrics
            lines.append(f'delta_pnl_total{label_str} {self.metrics.total_pnl}')
            lines.append(f'delta_pnl_realized{label_str} {self.metrics.realized_pnl}')
            lines.append(f'delta_pnl_unrealized{label_str} {self.metrics.unrealized_pnl}')
            lines.append(f'delta_pnl_daily{label_str} {self.metrics.daily_pnl}')
            
            # Order metrics
            lines.append(f'delta_signals_total{label_str} {self.metrics.signals_generated}')
            lines.append(f'delta_orders_placed{label_str} {self.metrics.orders_placed}')
            lines.append(f'delta_orders_filled{label_str} {self.metrics.orders_filled}')
            lines.append(f'delta_orders_rejected{label_str} {self.metrics.orders_rejected}')
            
            # Risk metrics
            lines.append(f'delta_drawdown_current{label_str} {self.metrics.current_drawdown}')
            lines.append(f'delta_drawdown_max{label_str} {self.metrics.max_drawdown}')
            
            # Custom metrics
            for name, value in self._custom_metrics.items():
                lines.append(f'delta_{name}{label_str} {value}')
        
        return '\n'.join(lines) + '\n'
    
    def get_json_metrics(self) -> Dict:
        """Get metrics as JSON"""
        with self._lock:
            return {
                'trades': {
                    'total': self.metrics.total_trades,
                    'winning': self.metrics.winning_trades,
                    'losing': self.metrics.losing_trades,
                    'win_rate': self.metrics.win_rate()
                },
                'pnl': {
                    'total': self.metrics.total_pnl,
                    'realized': self.metrics.realized_pnl,
                    'unrealized': self.metrics.unrealized_pnl,
                    'daily': self.metrics.daily_pnl
                },
                'orders': {
                    'signals': self.metrics.signals_generated,
                    'placed': self.metrics.orders_placed,
                    'filled': self.metrics.orders_filled,
                    'rejected': self.metrics.orders_rejected
                },
                'risk': {
                    'current_drawdown': self.metrics.current_drawdown,
                    'max_drawdown': self.metrics.max_drawdown
                },
                'timing': {
                    'last_trade': self.metrics.last_trade_time,
                    'last_signal': self.metrics.last_signal_time
                },
                'custom': dict(self._custom_metrics)
            }


class HTTPControlHandler(BaseHTTPRequestHandler):
    """HTTP handler for control endpoints"""
    
    exporter: MetricsExporter = None
    kill_switch_callback: Callable = None
    resume_callback: Callable = None
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/metrics':
            self._send_metrics()
        elif self.path == '/health':
            self._send_health()
        elif self.path == '/status':
            self._send_status()
        else:
            self._send_404()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/kill':
            self._handle_kill()
        elif self.path == '/resume':
            self._handle_resume()
        else:
            self._send_404()
    
    def _send_metrics(self):
        """Send Prometheus metrics"""
        if self.exporter:
            content = self.exporter.get_prometheus_metrics()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(content.encode())
        else:
            self._send_500("Exporter not configured")
    
    def _send_health(self):
        """Send health check"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'healthy', 'timestamp': time.time()}).encode())
    
    def _send_status(self):
        """Send detailed status"""
        if self.exporter:
            content = json.dumps(self.exporter.get_json_metrics(), indent=2)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(content.encode())
        else:
            self._send_500("Exporter not configured")
    
    def _handle_kill(self):
        """Handle kill switch activation"""
        if self.kill_switch_callback:
            self.kill_switch_callback()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'kill_switch_activated'}).encode())
        else:
            self._send_500("Kill switch not configured")
    
    def _handle_resume(self):
        """Handle trading resume"""
        if self.resume_callback:
            self.resume_callback()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'trading_resumed'}).encode())
        else:
            self._send_500("Resume callback not configured")
    
    def _send_404(self):
        """Send 404 response"""
        self.send_response(404)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': 'not_found'}).encode())
    
    def _send_500(self, message: str):
        """Send 500 response"""
        self.send_response(500)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def start_metrics_server(
    exporter: MetricsExporter,
    port: int = 8000
) -> HTTPServer:
    """
    Start Prometheus metrics server
    
    Args:
        exporter: MetricsExporter instance
        port: Server port
    
    Returns:
        HTTPServer instance
    """
    HTTPControlHandler.exporter = exporter
    
    server = HTTPServer(('0.0.0.0', port), HTTPControlHandler)
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    print(f"Metrics server started on port {port}")
    
    return server


def start_http_server(
    exporter: MetricsExporter,
    port: int = 8080,
    kill_callback: Callable = None,
    resume_callback: Callable = None
) -> HTTPServer:
    """
    Start HTTP control server
    
    Args:
        exporter: MetricsExporter instance
        port: Server port
        kill_callback: Function to call on kill switch
        resume_callback: Function to call on resume
    
    Returns:
        HTTPServer instance
    """
    HTTPControlHandler.exporter = exporter
    HTTPControlHandler.kill_switch_callback = kill_callback
    HTTPControlHandler.resume_callback = resume_callback
    
    server = HTTPServer(('0.0.0.0', port), HTTPControlHandler)
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    print(f"HTTP control server started on port {port}")
    print(f"  GET /health - Health check")
    print(f"  GET /status - Trading status")
    print(f"  GET /metrics - Prometheus metrics")
    print(f"  POST /kill - Activate kill switch")
    print(f"  POST /resume - Resume trading")
    
    return server
