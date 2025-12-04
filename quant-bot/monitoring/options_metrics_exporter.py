"""
Options Greeks Dashboard - Prometheus Metrics Exporter
=======================================================
Exports real-time options Greeks and risk metrics to Prometheus
for Grafana visualization.

Usage:
    python monitoring/options_metrics_exporter.py --port 8000
    
Metrics Exposed:
    - options_delta_exposure (by asset, strategy)
    - options_gamma_exposure
    - options_theta_daily
    - options_vega_exposure
    - options_iv_rank
    - options_margin_utilization
    - options_gap_risk_exposure
    - options_pnl_by_greek (delta_pnl, gamma_pnl, theta_pnl, vega_pnl)

Author: Quant Bot
Version: 1.0.0
"""

import os
import sys
import time
import logging
import argparse
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptionsPosition:
    """Single options position."""
    symbol: str
    underlying: str
    option_type: str  # call/put
    strike: float
    expiry: str
    quantity: float
    entry_price: float
    current_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    
    @property
    def notional(self) -> float:
        return abs(self.quantity * self.current_price * 100)  # 100 multiplier
    
    @property
    def delta_exposure(self) -> float:
        return self.delta * self.quantity * 100
    
    @property
    def gamma_exposure(self) -> float:
        return self.gamma * self.quantity * 100
    
    @property
    def theta_exposure(self) -> float:
        return self.theta * self.quantity * 100
    
    @property
    def vega_exposure(self) -> float:
        return self.vega * self.quantity * 100


@dataclass
class GreeksSnapshot:
    """Portfolio Greeks snapshot."""
    timestamp: datetime
    
    # Net Greeks
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    net_rho: float = 0.0
    
    # Greeks by underlying
    delta_by_asset: Dict[str, float] = field(default_factory=dict)
    gamma_by_asset: Dict[str, float] = field(default_factory=dict)
    vega_by_asset: Dict[str, float] = field(default_factory=dict)
    
    # P&L Attribution
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0
    unexplained_pnl: float = 0.0
    
    # Risk Metrics
    margin_used: float = 0.0
    margin_available: float = 0.0
    margin_utilization: float = 0.0
    
    # Vol Metrics
    portfolio_weighted_iv: float = 0.0
    iv_rank_avg: float = 0.0
    
    # Gap Risk
    gap_risk_1pct: float = 0.0
    gap_risk_3pct: float = 0.0
    gap_risk_5pct: float = 0.0
    worst_case_loss: float = 0.0


class OptionsMetricsCollector:
    """Collect options metrics from trading engine."""
    
    def __init__(self):
        self.positions: List[OptionsPosition] = []
        self.current_snapshot: Optional[GreeksSnapshot] = None
        self.historical_snapshots: List[GreeksSnapshot] = []
        self.iv_history: Dict[str, List[float]] = {}  # For IV rank calculation
        
        # Configuration
        self.iv_lookback_days = 252
        self.gap_risk_multipliers = [0.01, 0.03, 0.05]  # 1%, 3%, 5% moves
        
    def update_positions(self, positions: List[Dict]) -> None:
        """Update positions from trading engine."""
        self.positions = [
            OptionsPosition(**pos) for pos in positions
        ]
        self._calculate_snapshot()
    
    def _calculate_snapshot(self) -> None:
        """Calculate current Greeks snapshot."""
        snapshot = GreeksSnapshot(timestamp=datetime.now())
        
        # Aggregate Greeks
        for pos in self.positions:
            snapshot.net_delta += pos.delta_exposure
            snapshot.net_gamma += pos.gamma_exposure
            snapshot.net_theta += pos.theta_exposure
            snapshot.net_vega += pos.vega_exposure
            
            # By asset
            underlying = pos.underlying
            snapshot.delta_by_asset[underlying] = snapshot.delta_by_asset.get(underlying, 0) + pos.delta_exposure
            snapshot.gamma_by_asset[underlying] = snapshot.gamma_by_asset.get(underlying, 0) + pos.gamma_exposure
            snapshot.vega_by_asset[underlying] = snapshot.vega_by_asset.get(underlying, 0) + pos.vega_exposure
        
        # Calculate gap risk
        for mult in self.gap_risk_multipliers:
            gap_loss = self._calculate_gap_risk(mult)
            if mult == 0.01:
                snapshot.gap_risk_1pct = gap_loss
            elif mult == 0.03:
                snapshot.gap_risk_3pct = gap_loss
            elif mult == 0.05:
                snapshot.gap_risk_5pct = gap_loss
        
        # Worst case (max of up/down moves)
        snapshot.worst_case_loss = max(
            abs(snapshot.gap_risk_5pct),
            abs(self._calculate_gap_risk(-0.05))
        )
        
        # Store snapshot
        self.current_snapshot = snapshot
        self.historical_snapshots.append(snapshot)
        
        # Keep only last 24h of snapshots
        cutoff = datetime.now() - timedelta(hours=24)
        self.historical_snapshots = [
            s for s in self.historical_snapshots 
            if s.timestamp > cutoff
        ]
    
    def _calculate_gap_risk(self, price_move_pct: float) -> float:
        """
        Calculate portfolio P&L for a given price move.
        Uses Taylor expansion: P&L ≈ Δ·S·move + 0.5·Γ·S²·move²
        """
        total_pnl = 0.0
        
        for pos in self.positions:
            # Assume underlying price = 1 for normalized calculation
            delta_pnl = pos.delta_exposure * price_move_pct
            gamma_pnl = 0.5 * pos.gamma_exposure * (price_move_pct ** 2)
            total_pnl += delta_pnl + gamma_pnl
        
        return total_pnl
    
    def calculate_iv_rank(self, symbol: str, current_iv: float) -> float:
        """Calculate IV rank (0-100) based on historical IV."""
        history = self.iv_history.get(symbol, [])
        if len(history) < 20:
            return 50.0  # Default to middle if insufficient history
        
        min_iv = min(history)
        max_iv = max(history)
        
        if max_iv == min_iv:
            return 50.0
        
        return ((current_iv - min_iv) / (max_iv - min_iv)) * 100
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format."""
        lines = []
        
        if not self.current_snapshot:
            return ""
        
        snap = self.current_snapshot
        
        # Net Greeks
        lines.append(f'options_net_delta {snap.net_delta}')
        lines.append(f'options_net_gamma {snap.net_gamma}')
        lines.append(f'options_net_theta {snap.net_theta}')
        lines.append(f'options_net_vega {snap.net_vega}')
        
        # Greeks by asset
        for asset, delta in snap.delta_by_asset.items():
            lines.append(f'options_delta_by_asset{{asset="{asset}"}} {delta}')
        for asset, gamma in snap.gamma_by_asset.items():
            lines.append(f'options_gamma_by_asset{{asset="{asset}"}} {gamma}')
        for asset, vega in snap.vega_by_asset.items():
            lines.append(f'options_vega_by_asset{{asset="{asset}"}} {vega}')
        
        # P&L Attribution
        lines.append(f'options_pnl_delta {snap.delta_pnl}')
        lines.append(f'options_pnl_gamma {snap.gamma_pnl}')
        lines.append(f'options_pnl_theta {snap.theta_pnl}')
        lines.append(f'options_pnl_vega {snap.vega_pnl}')
        lines.append(f'options_pnl_unexplained {snap.unexplained_pnl}')
        
        # Margin
        lines.append(f'options_margin_used {snap.margin_used}')
        lines.append(f'options_margin_available {snap.margin_available}')
        lines.append(f'options_margin_utilization {snap.margin_utilization}')
        
        # Vol metrics
        lines.append(f'options_portfolio_iv {snap.portfolio_weighted_iv}')
        lines.append(f'options_iv_rank_avg {snap.iv_rank_avg}')
        
        # Gap risk
        lines.append(f'options_gap_risk_1pct {snap.gap_risk_1pct}')
        lines.append(f'options_gap_risk_3pct {snap.gap_risk_3pct}')
        lines.append(f'options_gap_risk_5pct {snap.gap_risk_5pct}')
        lines.append(f'options_worst_case_loss {snap.worst_case_loss}')
        
        # Position counts
        lines.append(f'options_position_count {len(self.positions)}')
        
        # Per-position metrics
        for pos in self.positions:
            labels = f'symbol="{pos.symbol}",underlying="{pos.underlying}",type="{pos.option_type}",strike="{pos.strike}"'
            lines.append(f'options_position_delta{{{labels}}} {pos.delta_exposure}')
            lines.append(f'options_position_gamma{{{labels}}} {pos.gamma_exposure}')
            lines.append(f'options_position_theta{{{labels}}} {pos.theta_exposure}')
            lines.append(f'options_position_vega{{{labels}}} {pos.vega_exposure}')
            lines.append(f'options_position_iv{{{labels}}} {pos.iv}')
        
        return '\n'.join(lines) + '\n'


class ArbitrageMetricsCollector:
    """Collect arbitrage metrics."""
    
    def __init__(self):
        self.funding_rates: Dict[str, float] = {}
        self.basis_spreads: Dict[str, float] = {}
        self.execution_latencies: List[float] = []
        self.fill_ratios: List[float] = []
        self.transfer_failures: int = 0
        self.total_transfers: int = 0
        
        # Funding capture tracking
        self.theoretical_funding: float = 0.0
        self.realized_funding: float = 0.0
        
        # Cross-exchange metrics
        self.venue_latencies: Dict[str, List[float]] = {}
        self.arb_opportunities_found: int = 0
        self.arb_opportunities_executed: int = 0
        
    def update_funding_metrics(
        self,
        pair: str,
        funding_rate: float,
        basis_spread: float,
        theoretical_capture: float,
        realized_capture: float
    ) -> None:
        """Update funding arbitrage metrics."""
        self.funding_rates[pair] = funding_rate
        self.basis_spreads[pair] = basis_spread
        self.theoretical_funding += theoretical_capture
        self.realized_funding += realized_capture
    
    def record_execution(
        self,
        venue: str,
        latency_ms: float,
        filled: bool,
        fill_ratio: float
    ) -> None:
        """Record execution metrics."""
        self.execution_latencies.append(latency_ms)
        self.fill_ratios.append(fill_ratio)
        
        if venue not in self.venue_latencies:
            self.venue_latencies[venue] = []
        self.venue_latencies[venue].append(latency_ms)
        
        # Keep only last 1000 entries
        if len(self.execution_latencies) > 1000:
            self.execution_latencies = self.execution_latencies[-1000:]
        if len(self.fill_ratios) > 1000:
            self.fill_ratios = self.fill_ratios[-1000:]
    
    def record_transfer(self, success: bool) -> None:
        """Record transfer attempt."""
        self.total_transfers += 1
        if not success:
            self.transfer_failures += 1
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format."""
        lines = []
        
        # Funding rates by pair
        for pair, rate in self.funding_rates.items():
            lines.append(f'arb_funding_rate{{pair="{pair}"}} {rate}')
        
        # Basis spreads
        for pair, spread in self.basis_spreads.items():
            lines.append(f'arb_basis_spread{{pair="{pair}"}} {spread}')
        
        # Funding capture efficiency
        capture_ratio = self.realized_funding / self.theoretical_funding if self.theoretical_funding > 0 else 0
        lines.append(f'arb_funding_theoretical {self.theoretical_funding}')
        lines.append(f'arb_funding_realized {self.realized_funding}')
        lines.append(f'arb_funding_capture_ratio {capture_ratio}')
        
        # Execution metrics
        if self.execution_latencies:
            avg_latency = sum(self.execution_latencies) / len(self.execution_latencies)
            p99_latency = sorted(self.execution_latencies)[int(len(self.execution_latencies) * 0.99)] if len(self.execution_latencies) > 100 else max(self.execution_latencies)
            lines.append(f'arb_execution_latency_avg_ms {avg_latency}')
            lines.append(f'arb_execution_latency_p99_ms {p99_latency}')
        
        # Fill ratio
        if self.fill_ratios:
            avg_fill = sum(self.fill_ratios) / len(self.fill_ratios)
            lines.append(f'arb_fill_ratio_avg {avg_fill}')
        
        # Per-venue latencies
        for venue, latencies in self.venue_latencies.items():
            if latencies:
                avg = sum(latencies[-100:]) / len(latencies[-100:])
                lines.append(f'arb_venue_latency_ms{{venue="{venue}"}} {avg}')
        
        # Transfer metrics
        transfer_success_rate = (self.total_transfers - self.transfer_failures) / self.total_transfers if self.total_transfers > 0 else 1.0
        lines.append(f'arb_transfer_total {self.total_transfers}')
        lines.append(f'arb_transfer_failures {self.transfer_failures}')
        lines.append(f'arb_transfer_success_rate {transfer_success_rate}')
        
        # Opportunity metrics
        execution_rate = self.arb_opportunities_executed / self.arb_opportunities_found if self.arb_opportunities_found > 0 else 0
        lines.append(f'arb_opportunities_found {self.arb_opportunities_found}')
        lines.append(f'arb_opportunities_executed {self.arb_opportunities_executed}')
        lines.append(f'arb_execution_rate {execution_rate}')
        
        return '\n'.join(lines) + '\n'


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""
    
    options_collector: Optional[OptionsMetricsCollector] = None
    arb_collector: Optional[ArbitrageMetricsCollector] = None
    
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            
            metrics = ""
            if self.options_collector:
                metrics += self.options_collector.get_prometheus_metrics()
            if self.arb_collector:
                metrics += self.arb_collector.get_prometheus_metrics()
            
            self.wfile.write(metrics.encode('utf-8'))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging


def run_metrics_server(
    port: int = 8000,
    options_collector: Optional[OptionsMetricsCollector] = None,
    arb_collector: Optional[ArbitrageMetricsCollector] = None
):
    """Run the metrics HTTP server."""
    MetricsHTTPHandler.options_collector = options_collector
    MetricsHTTPHandler.arb_collector = arb_collector
    
    server = HTTPServer(('0.0.0.0', port), MetricsHTTPHandler)
    logger.info(f"Metrics server running on port {port}")
    logger.info(f"Endpoints: /metrics, /health")
    
    server.serve_forever()


def demo_with_sample_data():
    """Demo with sample options positions."""
    collector = OptionsMetricsCollector()
    
    # Sample positions
    sample_positions = [
        {
            "symbol": "BTC-20241220-100000-C",
            "underlying": "BTC",
            "option_type": "call",
            "strike": 100000,
            "expiry": "2024-12-20",
            "quantity": 10,
            "entry_price": 5000,
            "current_price": 5500,
            "delta": 0.55,
            "gamma": 0.00002,
            "theta": -50,
            "vega": 150,
            "iv": 0.65
        },
        {
            "symbol": "BTC-20241220-100000-P",
            "underlying": "BTC",
            "option_type": "put",
            "strike": 100000,
            "expiry": "2024-12-20",
            "quantity": -10,
            "entry_price": 4500,
            "current_price": 4200,
            "delta": -0.45,
            "gamma": 0.00002,
            "theta": -45,
            "vega": 140,
            "iv": 0.63
        },
        {
            "symbol": "ETH-20241220-4000-C",
            "underlying": "ETH",
            "option_type": "call",
            "strike": 4000,
            "expiry": "2024-12-20",
            "quantity": 50,
            "entry_price": 200,
            "current_price": 250,
            "delta": 0.60,
            "gamma": 0.0005,
            "theta": -10,
            "vega": 30,
            "iv": 0.70
        }
    ]
    
    collector.update_positions(sample_positions)
    
    # Print sample metrics
    print("=" * 60)
    print("SAMPLE OPTIONS METRICS OUTPUT")
    print("=" * 60)
    print(collector.get_prometheus_metrics())
    
    return collector


def main():
    parser = argparse.ArgumentParser(description="Options & Arbitrage Metrics Exporter")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--demo", action="store_true", help="Run with demo data")
    
    args = parser.parse_args()
    
    options_collector = OptionsMetricsCollector()
    arb_collector = ArbitrageMetricsCollector()
    
    if args.demo:
        # Load demo data
        options_collector = demo_with_sample_data()
    
    # Start metrics server
    run_metrics_server(
        port=args.port,
        options_collector=options_collector,
        arb_collector=arb_collector
    )


if __name__ == "__main__":
    main()
