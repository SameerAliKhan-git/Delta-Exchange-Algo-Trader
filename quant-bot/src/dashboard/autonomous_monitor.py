"""
Autonomous Trading Monitor Dashboard
=====================================
Real-time UI for monitoring all autonomous trading operations.

Features:
- Live position tracking
- ML model status & metrics
- Strategy selection visualization
- Risk state monitoring
- Alert management
- Performance analytics

Usage:
    python src/dashboard/autonomous_monitor.py --port 8080
    
Then open http://localhost:8080 in your browser.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import argparse

# Web framework
try:
    from aiohttp import web
    import aiohttp_cors
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Prometheus client
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SystemOverview:
    """Overall system status."""
    status: str  # "running", "paused", "error", "killed"
    uptime_hours: float
    mode: str  # "paper", "canary", "production"
    allocation_pct: float
    
    # Counts
    active_positions: int
    pending_orders: int
    active_strategies: int
    active_alerts: int
    
    # Performance
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    total_pnl: float
    
    # Risk
    current_drawdown_pct: float
    max_drawdown_pct: float
    risk_level: str  # "normal", "elevated", "high", "critical"
    
    # Last actions
    last_trade_time: Optional[str]
    last_signal_time: Optional[str]
    last_retrain_time: Optional[str]


@dataclass
class PositionInfo:
    """Single position details."""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    notional_usd: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: float
    take_profit: float
    entry_time: str
    holding_hours: float
    strategy: str
    
    # Greeks (for options)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None


@dataclass 
class StrategyStatus:
    """Strategy performance status."""
    name: str
    enabled: bool
    active_positions: int
    
    # Performance
    total_trades: int
    win_rate: float
    avg_pnl: float
    sharpe_ratio: float
    
    # Selection
    selection_weight: float
    recent_selection_pct: float
    
    # Regime compatibility
    current_regime: str
    regime_compatible: bool


@dataclass
class MLModelStatus:
    """ML model status."""
    name: str
    model_type: str
    last_trained: str
    training_samples: int
    
    # Metrics
    accuracy: float
    precision: float
    f1_score: float
    
    # Drift
    psi_score: float
    drift_detected: bool
    
    # Status
    status: str  # "active", "retraining", "degraded", "offline"


@dataclass
class AlertInfo:
    """Alert information."""
    id: str
    timestamp: str
    severity: str  # "info", "warning", "critical"
    category: str
    title: str
    description: str
    status: str  # "firing", "resolved", "acknowledged"
    acknowledged_by: Optional[str] = None


# =============================================================================
# Dashboard State Manager
# =============================================================================

class DashboardStateManager:
    """
    Manages dashboard state and provides data for UI.
    Connects to various system components.
    """
    
    def __init__(self):
        self._start_time = datetime.utcnow()
        self._positions: Dict[str, PositionInfo] = {}
        self._strategies: Dict[str, StrategyStatus] = {}
        self._models: Dict[str, MLModelStatus] = {}
        self._alerts: List[AlertInfo] = []
        self._trade_history: List[Dict] = []
        self._pnl_history: List[Dict] = []
        
        # Mock initial data for demo
        self._init_mock_data()
    
    def _init_mock_data(self):
        """Initialize with mock data for demo."""
        import random
        
        # Mock positions
        self._positions = {
            "BTCUSD-1": PositionInfo(
                symbol="BTCUSD",
                side="long",
                entry_price=98500,
                current_price=99200,
                quantity=0.5,
                notional_usd=49600,
                unrealized_pnl=350,
                unrealized_pnl_pct=0.71,
                stop_loss=96000,
                take_profit=105000,
                entry_time=(datetime.utcnow() - timedelta(hours=4)).isoformat(),
                holding_hours=4.0,
                strategy="momentum"
            ),
            "ETHUSD-1": PositionInfo(
                symbol="ETHUSD",
                side="short",
                entry_price=3650,
                current_price=3620,
                quantity=10,
                notional_usd=36200,
                unrealized_pnl=300,
                unrealized_pnl_pct=0.82,
                stop_loss=3800,
                take_profit=3400,
                entry_time=(datetime.utcnow() - timedelta(hours=2)).isoformat(),
                holding_hours=2.0,
                strategy="mean_reversion"
            )
        }
        
        # Mock strategies
        strategies = ["momentum", "mean_reversion", "stat_arb", "regime_ml", "funding_arb", "options_delta"]
        for i, name in enumerate(strategies):
            self._strategies[name] = StrategyStatus(
                name=name,
                enabled=True,
                active_positions=random.randint(0, 2),
                total_trades=random.randint(50, 200),
                win_rate=0.5 + random.uniform(0, 0.2),
                avg_pnl=random.uniform(10, 100),
                sharpe_ratio=0.8 + random.uniform(0, 1.5),
                selection_weight=1.0 / len(strategies),
                recent_selection_pct=random.uniform(0.1, 0.3),
                current_regime="trending",
                regime_compatible=random.choice([True, True, True, False])
            )
        
        # Mock models
        for symbol in ["BTCUSD", "ETHUSD"]:
            self._models[f"model_{symbol}"] = MLModelStatus(
                name=f"model_{symbol}",
                model_type="ensemble",
                last_trained=(datetime.utcnow() - timedelta(hours=12)).isoformat(),
                training_samples=5000,
                accuracy=0.58 + random.uniform(0, 0.1),
                precision=0.55 + random.uniform(0, 0.1),
                f1_score=0.56 + random.uniform(0, 0.1),
                psi_score=random.uniform(0.05, 0.15),
                drift_detected=False,
                status="active"
            )
        
        # Mock alerts
        self._alerts = [
            AlertInfo(
                id="alert-001",
                timestamp=datetime.utcnow().isoformat(),
                severity="warning",
                category="hedging",
                title="Hedging slippage elevated",
                description="Average hedge slippage is 12.5 bps over last 10m",
                status="firing"
            ),
            AlertInfo(
                id="alert-002",
                timestamp=(datetime.utcnow() - timedelta(hours=1)).isoformat(),
                severity="info",
                category="expiry",
                title="Options position expiring within 24h",
                description="BTC-27DEC24-100000-C expires in 18 hours",
                status="acknowledged",
                acknowledged_by="system"
            )
        ]
    
    def get_overview(self) -> SystemOverview:
        """Get system overview."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds() / 3600
        
        return SystemOverview(
            status="running",
            uptime_hours=uptime,
            mode="canary",
            allocation_pct=5.0,
            active_positions=len(self._positions),
            pending_orders=0,
            active_strategies=sum(1 for s in self._strategies.values() if s.enabled),
            active_alerts=sum(1 for a in self._alerts if a.status == "firing"),
            daily_pnl=650.0,
            daily_pnl_pct=0.65,
            weekly_pnl=2100.0,
            total_pnl=8500.0,
            current_drawdown_pct=1.2,
            max_drawdown_pct=3.5,
            risk_level="normal",
            last_trade_time=(datetime.utcnow() - timedelta(minutes=45)).isoformat(),
            last_signal_time=(datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            last_retrain_time=(datetime.utcnow() - timedelta(hours=12)).isoformat()
        )
    
    def get_positions(self) -> List[PositionInfo]:
        """Get all positions."""
        return list(self._positions.values())
    
    def get_strategies(self) -> List[StrategyStatus]:
        """Get all strategy statuses."""
        return list(self._strategies.values())
    
    def get_models(self) -> List[MLModelStatus]:
        """Get all model statuses."""
        return list(self._models.values())
    
    def get_alerts(self, include_resolved: bool = False) -> List[AlertInfo]:
        """Get alerts."""
        if include_resolved:
            return self._alerts
        return [a for a in self._alerts if a.status != "resolved"]
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.status = "acknowledged"
                alert.acknowledged_by = user
                return True
        return False
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history."""
        # Mock trade history
        import random
        trades = []
        for i in range(min(limit, 20)):
            trades.append({
                'timestamp': (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                'symbol': random.choice(['BTCUSD', 'ETHUSD']),
                'side': random.choice(['buy', 'sell']),
                'price': 98000 + random.uniform(-2000, 2000),
                'quantity': random.uniform(0.1, 1.0),
                'pnl': random.uniform(-100, 200),
                'strategy': random.choice(['momentum', 'mean_reversion', 'stat_arb'])
            })
        return trades
    
    def get_pnl_timeseries(self, hours: int = 24) -> List[Dict]:
        """Get PnL timeseries."""
        import random
        points = []
        cumulative = 0
        for i in range(hours * 4):  # 15-minute intervals
            pnl = random.uniform(-50, 75)
            cumulative += pnl
            points.append({
                'timestamp': (datetime.utcnow() - timedelta(minutes=15*(hours*4 - i))).isoformat(),
                'pnl': pnl,
                'cumulative_pnl': cumulative
            })
        return points


# =============================================================================
# Web Server
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Autonomous Trading Monitor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @keyframes pulse-green { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes pulse-red { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        .pulse-green { animation: pulse-green 2s infinite; }
        .pulse-red { animation: pulse-red 1s infinite; }
        .glass { backdrop-filter: blur(10px); background: rgba(17, 24, 39, 0.8); }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-6">
        <!-- Header -->
        <header class="flex items-center justify-between mb-8">
            <div class="flex items-center space-x-4">
                <div class="text-3xl">🤖</div>
                <div>
                    <h1 class="text-2xl font-bold">Autonomous Trading Monitor</h1>
                    <p class="text-gray-400 text-sm" id="last-update">Last update: --</p>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <div id="system-status" class="flex items-center space-x-2 px-4 py-2 rounded-lg bg-green-900/50">
                    <div class="w-3 h-3 rounded-full bg-green-500 pulse-green"></div>
                    <span class="text-green-400 font-medium">System Running</span>
                </div>
                <button onclick="toggleKillSwitch()" class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium">
                    🛑 Kill Switch
                </button>
            </div>
        </header>
        
        <!-- Overview Cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
            <div class="glass rounded-xl p-4">
                <div class="text-gray-400 text-sm">Mode</div>
                <div class="text-xl font-bold text-yellow-400" id="mode">Canary (5%)</div>
            </div>
            <div class="glass rounded-xl p-4">
                <div class="text-gray-400 text-sm">Positions</div>
                <div class="text-xl font-bold" id="positions-count">0</div>
            </div>
            <div class="glass rounded-xl p-4">
                <div class="text-gray-400 text-sm">Daily P&L</div>
                <div class="text-xl font-bold text-green-400" id="daily-pnl">$0.00</div>
            </div>
            <div class="glass rounded-xl p-4">
                <div class="text-gray-400 text-sm">Drawdown</div>
                <div class="text-xl font-bold" id="drawdown">0.0%</div>
            </div>
            <div class="glass rounded-xl p-4">
                <div class="text-gray-400 text-sm">Active Alerts</div>
                <div class="text-xl font-bold text-orange-400" id="alerts-count">0</div>
            </div>
            <div class="glass rounded-xl p-4">
                <div class="text-gray-400 text-sm">Risk Level</div>
                <div class="text-xl font-bold text-green-400" id="risk-level">Normal</div>
            </div>
        </div>
        
        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Positions Panel -->
            <div class="lg:col-span-2 glass rounded-xl p-6">
                <h2 class="text-lg font-bold mb-4">📊 Active Positions</h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead class="text-gray-400 border-b border-gray-700">
                            <tr>
                                <th class="text-left py-2">Symbol</th>
                                <th class="text-left py-2">Side</th>
                                <th class="text-right py-2">Entry</th>
                                <th class="text-right py-2">Current</th>
                                <th class="text-right py-2">Size</th>
                                <th class="text-right py-2">P&L</th>
                                <th class="text-left py-2">Strategy</th>
                                <th class="text-right py-2">Hours</th>
                            </tr>
                        </thead>
                        <tbody id="positions-table">
                            <tr><td colspan="8" class="text-center py-4 text-gray-500">No positions</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Alerts Panel -->
            <div class="glass rounded-xl p-6">
                <h2 class="text-lg font-bold mb-4">🚨 Active Alerts</h2>
                <div id="alerts-list" class="space-y-3 max-h-64 overflow-y-auto">
                    <div class="text-gray-500 text-center py-4">No active alerts</div>
                </div>
            </div>
        </div>
        
        <!-- Second Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <!-- Strategy Performance -->
            <div class="glass rounded-xl p-6">
                <h2 class="text-lg font-bold mb-4">🎯 Strategy Performance</h2>
                <div id="strategies-list" class="space-y-3">
                    <div class="text-gray-500 text-center py-4">Loading strategies...</div>
                </div>
            </div>
            
            <!-- ML Models Status -->
            <div class="glass rounded-xl p-6">
                <h2 class="text-lg font-bold mb-4">🧠 ML Models</h2>
                <div id="models-list" class="space-y-3">
                    <div class="text-gray-500 text-center py-4">Loading models...</div>
                </div>
            </div>
        </div>
        
        <!-- PnL Chart -->
        <div class="glass rounded-xl p-6 mt-6">
            <h2 class="text-lg font-bold mb-4">📈 P&L Timeline (24h)</h2>
            <canvas id="pnl-chart" height="100"></canvas>
        </div>
        
        <!-- Recent Trades -->
        <div class="glass rounded-xl p-6 mt-6">
            <h2 class="text-lg font-bold mb-4">📜 Recent Trades</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead class="text-gray-400 border-b border-gray-700">
                        <tr>
                            <th class="text-left py-2">Time</th>
                            <th class="text-left py-2">Symbol</th>
                            <th class="text-left py-2">Side</th>
                            <th class="text-right py-2">Price</th>
                            <th class="text-right py-2">Qty</th>
                            <th class="text-right py-2">P&L</th>
                            <th class="text-left py-2">Strategy</th>
                        </tr>
                    </thead>
                    <tbody id="trades-table">
                        <tr><td colspan="7" class="text-center py-4 text-gray-500">No trades yet</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        let pnlChart = null;
        
        async function fetchData() {
            try {
                const [overview, positions, strategies, models, alerts, trades, pnl] = await Promise.all([
                    fetch('/api/overview').then(r => r.json()),
                    fetch('/api/positions').then(r => r.json()),
                    fetch('/api/strategies').then(r => r.json()),
                    fetch('/api/models').then(r => r.json()),
                    fetch('/api/alerts').then(r => r.json()),
                    fetch('/api/trades').then(r => r.json()),
                    fetch('/api/pnl').then(r => r.json())
                ]);
                
                updateOverview(overview);
                updatePositions(positions);
                updateStrategies(strategies);
                updateModels(models);
                updateAlerts(alerts);
                updateTrades(trades);
                updatePnLChart(pnl);
                
                document.getElementById('last-update').textContent = 
                    'Last update: ' + new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Fetch error:', e);
            }
        }
        
        function updateOverview(data) {
            document.getElementById('mode').textContent = 
                data.mode.charAt(0).toUpperCase() + data.mode.slice(1) + ' (' + data.allocation_pct + '%)';
            document.getElementById('positions-count').textContent = data.active_positions;
            document.getElementById('daily-pnl').textContent = '$' + data.daily_pnl.toFixed(2);
            document.getElementById('daily-pnl').className = 
                'text-xl font-bold ' + (data.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400');
            document.getElementById('drawdown').textContent = data.current_drawdown_pct.toFixed(1) + '%';
            document.getElementById('alerts-count').textContent = data.active_alerts;
            document.getElementById('risk-level').textContent = data.risk_level.charAt(0).toUpperCase() + data.risk_level.slice(1);
            
            const riskColors = { normal: 'text-green-400', elevated: 'text-yellow-400', high: 'text-orange-400', critical: 'text-red-400' };
            document.getElementById('risk-level').className = 'text-xl font-bold ' + (riskColors[data.risk_level] || 'text-gray-400');
        }
        
        function updatePositions(positions) {
            const tbody = document.getElementById('positions-table');
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-gray-500">No positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(p => `
                <tr class="border-b border-gray-800">
                    <td class="py-2 font-medium">${p.symbol}</td>
                    <td class="py-2">
                        <span class="px-2 py-1 rounded text-xs font-medium ${p.side === 'long' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'}">
                            ${p.side.toUpperCase()}
                        </span>
                    </td>
                    <td class="py-2 text-right">$${p.entry_price.toLocaleString()}</td>
                    <td class="py-2 text-right">$${p.current_price.toLocaleString()}</td>
                    <td class="py-2 text-right">$${p.notional_usd.toLocaleString()}</td>
                    <td class="py-2 text-right ${p.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}">
                        $${p.unrealized_pnl.toFixed(2)} (${p.unrealized_pnl_pct.toFixed(2)}%)
                    </td>
                    <td class="py-2 text-gray-400">${p.strategy}</td>
                    <td class="py-2 text-right text-gray-400">${p.holding_hours.toFixed(1)}h</td>
                </tr>
            `).join('');
        }
        
        function updateStrategies(strategies) {
            const div = document.getElementById('strategies-list');
            div.innerHTML = strategies.map(s => `
                <div class="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                    <div class="flex items-center space-x-3">
                        <div class="w-2 h-2 rounded-full ${s.enabled ? (s.regime_compatible ? 'bg-green-500' : 'bg-yellow-500') : 'bg-gray-500'}"></div>
                        <div>
                            <div class="font-medium">${s.name}</div>
                            <div class="text-xs text-gray-400">${s.total_trades} trades | ${(s.win_rate * 100).toFixed(0)}% WR</div>
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="text-sm font-medium">Sharpe: ${s.sharpe_ratio.toFixed(2)}</div>
                        <div class="text-xs text-gray-400">Selection: ${(s.recent_selection_pct * 100).toFixed(0)}%</div>
                    </div>
                </div>
            `).join('');
        }
        
        function updateModels(models) {
            const div = document.getElementById('models-list');
            div.innerHTML = models.map(m => {
                const statusColors = { active: 'bg-green-500', retraining: 'bg-blue-500', degraded: 'bg-yellow-500', offline: 'bg-red-500' };
                return `
                    <div class="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                        <div class="flex items-center space-x-3">
                            <div class="w-2 h-2 rounded-full ${statusColors[m.status] || 'bg-gray-500'}"></div>
                            <div>
                                <div class="font-medium">${m.name}</div>
                                <div class="text-xs text-gray-400">${m.model_type} | ${m.training_samples} samples</div>
                            </div>
                        </div>
                        <div class="text-right">
                            <div class="text-sm font-medium">Acc: ${(m.accuracy * 100).toFixed(1)}%</div>
                            <div class="text-xs ${m.drift_detected ? 'text-red-400' : 'text-gray-400'}">
                                PSI: ${m.psi_score.toFixed(3)} ${m.drift_detected ? '⚠️' : ''}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function updateAlerts(alerts) {
            const div = document.getElementById('alerts-list');
            const firing = alerts.filter(a => a.status === 'firing');
            
            if (firing.length === 0) {
                div.innerHTML = '<div class="text-gray-500 text-center py-4">No active alerts ✓</div>';
                return;
            }
            
            div.innerHTML = firing.map(a => {
                const severityColors = { info: 'border-blue-500 bg-blue-900/20', warning: 'border-yellow-500 bg-yellow-900/20', critical: 'border-red-500 bg-red-900/20' };
                return `
                    <div class="p-3 rounded-lg border-l-4 ${severityColors[a.severity] || 'border-gray-500'}">
                        <div class="flex items-center justify-between">
                            <div class="font-medium text-sm">${a.title}</div>
                            <span class="text-xs px-2 py-1 rounded ${a.severity === 'critical' ? 'bg-red-600' : a.severity === 'warning' ? 'bg-yellow-600' : 'bg-blue-600'}">
                                ${a.severity.toUpperCase()}
                            </span>
                        </div>
                        <div class="text-xs text-gray-400 mt-1">${a.description}</div>
                        <button onclick="acknowledgeAlert('${a.id}')" class="text-xs text-blue-400 mt-2 hover:underline">
                            Acknowledge
                        </button>
                    </div>
                `;
            }).join('');
        }
        
        function updateTrades(trades) {
            const tbody = document.getElementById('trades-table');
            if (trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-gray-500">No trades yet</td></tr>';
                return;
            }
            
            tbody.innerHTML = trades.slice(0, 10).map(t => `
                <tr class="border-b border-gray-800">
                    <td class="py-2 text-gray-400">${new Date(t.timestamp).toLocaleTimeString()}</td>
                    <td class="py-2 font-medium">${t.symbol}</td>
                    <td class="py-2">
                        <span class="px-2 py-1 rounded text-xs font-medium ${t.side === 'buy' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'}">
                            ${t.side.toUpperCase()}
                        </span>
                    </td>
                    <td class="py-2 text-right">$${t.price.toFixed(2)}</td>
                    <td class="py-2 text-right">${t.quantity.toFixed(4)}</td>
                    <td class="py-2 text-right ${t.pnl >= 0 ? 'text-green-400' : 'text-red-400'}">$${t.pnl.toFixed(2)}</td>
                    <td class="py-2 text-gray-400">${t.strategy}</td>
                </tr>
            `).join('');
        }
        
        function updatePnLChart(data) {
            const ctx = document.getElementById('pnl-chart').getContext('2d');
            
            const labels = data.map(d => new Date(d.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
            const values = data.map(d => d.cumulative_pnl);
            
            if (pnlChart) {
                pnlChart.data.labels = labels;
                pnlChart.data.datasets[0].data = values;
                pnlChart.update();
            } else {
                pnlChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Cumulative P&L',
                            data: values,
                            borderColor: values[values.length - 1] >= 0 ? '#10b981' : '#ef4444',
                            backgroundColor: values[values.length - 1] >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#9ca3af' } },
                            y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#9ca3af', callback: v => '$' + v } }
                        }
                    }
                });
            }
        }
        
        async function acknowledgeAlert(id) {
            await fetch('/api/alerts/' + id + '/acknowledge', { method: 'POST' });
            fetchData();
        }
        
        async function toggleKillSwitch() {
            if (confirm('⚠️ DANGER: This will immediately halt ALL trading. Continue?')) {
                await fetch('/api/killswitch', { method: 'POST' });
                document.getElementById('system-status').innerHTML = `
                    <div class="w-3 h-3 rounded-full bg-red-500 pulse-red"></div>
                    <span class="text-red-400 font-medium">KILLED</span>
                `;
            }
        }
        
        // Initial fetch and auto-refresh
        fetchData();
        setInterval(fetchData, 5000);
    </script>
</body>
</html>
"""


class DashboardServer:
    """
    Aiohttp-based dashboard server.
    """
    
    def __init__(self, state_manager: DashboardStateManager, port: int = 8080):
        self.state = state_manager
        self.port = port
        self.app = None
    
    async def setup(self):
        """Setup the web app."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required: pip install aiohttp aiohttp-cors")
        
        self.app = web.Application()
        
        # CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        # Routes
        routes = [
            web.get('/', self.index),
            web.get('/api/overview', self.api_overview),
            web.get('/api/positions', self.api_positions),
            web.get('/api/strategies', self.api_strategies),
            web.get('/api/models', self.api_models),
            web.get('/api/alerts', self.api_alerts),
            web.get('/api/trades', self.api_trades),
            web.get('/api/pnl', self.api_pnl),
            web.post('/api/alerts/{id}/acknowledge', self.api_acknowledge_alert),
            web.post('/api/killswitch', self.api_killswitch),
        ]
        
        if PROMETHEUS_AVAILABLE:
            routes.append(web.get('/metrics', self.metrics))
        
        for route in routes:
            self.app.router.add_route(route.method, route.resource.canonical, route.handler)
            cors.add(self.app.router.routes()[-1])
        
        return self.app
    
    async def index(self, request):
        """Serve main HTML page."""
        return web.Response(text=HTML_TEMPLATE, content_type='text/html')
    
    async def api_overview(self, request):
        """API: Get system overview."""
        return web.json_response(asdict(self.state.get_overview()))
    
    async def api_positions(self, request):
        """API: Get positions."""
        return web.json_response([asdict(p) for p in self.state.get_positions()])
    
    async def api_strategies(self, request):
        """API: Get strategies."""
        return web.json_response([asdict(s) for s in self.state.get_strategies()])
    
    async def api_models(self, request):
        """API: Get models."""
        return web.json_response([asdict(m) for m in self.state.get_models()])
    
    async def api_alerts(self, request):
        """API: Get alerts."""
        return web.json_response([asdict(a) for a in self.state.get_alerts()])
    
    async def api_trades(self, request):
        """API: Get trade history."""
        return web.json_response(self.state.get_trade_history())
    
    async def api_pnl(self, request):
        """API: Get PnL timeseries."""
        return web.json_response(self.state.get_pnl_timeseries())
    
    async def api_acknowledge_alert(self, request):
        """API: Acknowledge alert."""
        alert_id = request.match_info['id']
        success = self.state.acknowledge_alert(alert_id)
        return web.json_response({'success': success})
    
    async def api_killswitch(self, request):
        """API: Trigger kill switch."""
        logger.critical("KILL SWITCH ACTIVATED VIA DASHBOARD")
        # In production, this would call the actual kill switch
        return web.json_response({'success': True, 'message': 'Kill switch activated'})
    
    async def metrics(self, request):
        """Prometheus metrics endpoint."""
        return web.Response(body=generate_latest(), content_type=CONTENT_TYPE_LATEST)
    
    async def run(self):
        """Run the server."""
        await self.setup()
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logger.info(f"Dashboard running at http://localhost:{self.port}")
        
        # Keep running
        while True:
            await asyncio.sleep(3600)


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description='Autonomous Trading Monitor')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    args = parser.parse_args()
    
    if not AIOHTTP_AVAILABLE:
        print("ERROR: aiohttp required. Install with: pip install aiohttp aiohttp-cors")
        return
    
    state_manager = DashboardStateManager()
    server = DashboardServer(state_manager, port=args.port)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║        🤖 AUTONOMOUS TRADING MONITOR DASHBOARD                    ║
╠═══════════════════════════════════════════════════════════════════╣
║  URL: http://localhost:{args.port:<5}                                   ║
║                                                                   ║
║  Features:                                                        ║
║   • Real-time position monitoring                                 ║
║   • Strategy performance tracking                                 ║
║   • ML model status & drift detection                             ║
║   • Alert management                                              ║
║   • P&L visualization                                             ║
║   • Kill switch control                                           ║
║                                                                   ║
║  Press Ctrl+C to stop                                             ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    await server.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
