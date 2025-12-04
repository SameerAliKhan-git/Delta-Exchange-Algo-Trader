#!/usr/bin/env python3
"""
src/dashboard/api_endpoints.py

Dashboard API Endpoints
=======================

FastAPI backend that serves the React dashboard:
- WebSocket for real-time events
- REST endpoints for system status
- Meta-learner control endpoints
- Position and P&L streaming

Usage:
  python -m src.dashboard.api_endpoints --port 8080
  uvicorn src.dashboard.api_endpoints:app --port 8080 --reload
"""

import asyncio
import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    from pydantic import BaseModel
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn websockets")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class SystemStatus(BaseModel):
    """System status response."""
    status: str  # healthy, degraded, error
    uptime_seconds: int
    mode: str  # paper, canary, production
    active_strategies: List[str]
    meta_learner_enabled: bool
    last_trade_time: Optional[str]
    error_count: int


class Position(BaseModel):
    """Position model."""
    symbol: str
    side: str  # long, short
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: float


class Trade(BaseModel):
    """Trade model."""
    id: str
    timestamp: str
    symbol: str
    side: str
    size: float
    price: float
    pnl: float
    fees: float
    strategy: str


class StrategyStats(BaseModel):
    """Strategy statistics."""
    name: str
    enabled: bool
    trades: int
    win_rate: float
    sharpe_ratio: float
    total_pnl: float
    last_signal: Optional[str]
    regime: str


class MetaLearnerStatus(BaseModel):
    """Meta-learner status."""
    enabled: bool
    mode: str  # thompson, ucb, epsilon_greedy
    current_strategy: Optional[str]
    exploration_rate: float
    regime: str
    arm_stats: Dict[str, Dict]


class AlertModel(BaseModel):
    """Alert model."""
    id: str
    severity: str  # info, warning, critical
    message: str
    timestamp: str
    acknowledged: bool


class MetaLearnerConfig(BaseModel):
    """Meta-learner configuration."""
    enabled: bool
    mode: str = "thompson"
    exploration_rate: float = 0.1


class TradeResult(BaseModel):
    """Trade result for meta-learner update."""
    strategy: str
    pnl: float
    fees: float
    is_win: bool


# ============================================================================
# Connection Manager for WebSockets
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept new connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = {"all"}
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, channel: str, message: Dict):
        """Broadcast message to subscribed clients."""
        payload = json.dumps({"channel": channel, "data": message, "ts": time.time()})
        
        disconnected = []
        for connection in self.active_connections:
            subs = self.subscriptions.get(connection, set())
            if "all" in subs or channel in subs:
                try:
                    await connection.send_text(payload)
                except:
                    disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    def subscribe(self, websocket: WebSocket, channels: List[str]):
        """Subscribe to specific channels."""
        if websocket in self.subscriptions:
            self.subscriptions[websocket].update(channels)


# ============================================================================
# Mock Data Store (replace with real data in production)
# ============================================================================

class DataStore:
    """Mock data store for demo purposes."""
    
    def __init__(self):
        self.start_time = time.time()
        self.mode = "paper"
        self.meta_learner_enabled = False
        self.meta_learner_mode = "thompson"
        self.exploration_rate = 0.1
        self.current_strategy = "momentum"
        self.regime = "trending"
        self.error_count = 0
        self.last_trade_time = None
        self.trades: List[Dict] = []
        self.alerts: List[Dict] = []
        
        # Strategy stats
        self.strategies = {
            "momentum": {
                "enabled": True, "trades": 145, "win_rate": 0.62,
                "sharpe": 1.85, "pnl": 12500.0, "last_signal": "long"
            },
            "mean_reversion": {
                "enabled": True, "trades": 89, "win_rate": 0.58,
                "sharpe": 1.42, "pnl": 8200.0, "last_signal": "neutral"
            },
            "iv_arbitrage": {
                "enabled": True, "trades": 34, "win_rate": 0.71,
                "sharpe": 2.15, "pnl": 15600.0, "last_signal": "long"
            },
            "funding_arb": {
                "enabled": True, "trades": 56, "win_rate": 0.82,
                "sharpe": 2.85, "pnl": 9800.0, "last_signal": "long"
            },
        }
        
        # Arm stats for meta-learner
        self.arm_stats = {
            name: {"alpha": 5 + random.randint(0, 10), "beta": 3 + random.randint(0, 5), "pulls": random.randint(20, 100)}
            for name in self.strategies
        }
        
        # Mock positions
        self.positions = [
            {"symbol": "BTCUSD", "side": "long", "size": 0.5, "entry_price": 67500,
             "current_price": 68200, "unrealized_pnl": 350, "realized_pnl": 1200, "leverage": 5},
            {"symbol": "ETHUSD", "side": "short", "size": 3.0, "entry_price": 3650,
             "current_price": 3620, "unrealized_pnl": 90, "realized_pnl": 450, "leverage": 3},
        ]
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            "status": "healthy",
            "uptime_seconds": int(time.time() - self.start_time),
            "mode": self.mode,
            "active_strategies": [k for k, v in self.strategies.items() if v["enabled"]],
            "meta_learner_enabled": self.meta_learner_enabled,
            "last_trade_time": self.last_trade_time,
            "error_count": self.error_count,
        }
    
    def get_positions(self) -> List[Dict]:
        """Get current positions with simulated updates."""
        for pos in self.positions:
            # Simulate price movement
            delta = random.uniform(-0.002, 0.002)
            pos["current_price"] *= (1 + delta)
            
            if pos["side"] == "long":
                pos["unrealized_pnl"] = (pos["current_price"] - pos["entry_price"]) * pos["size"]
            else:
                pos["unrealized_pnl"] = (pos["entry_price"] - pos["current_price"]) * pos["size"]
        
        return self.positions
    
    def get_strategy_stats(self) -> List[Dict]:
        """Get strategy statistics."""
        return [
            {
                "name": name,
                "enabled": stats["enabled"],
                "trades": stats["trades"],
                "win_rate": stats["win_rate"],
                "sharpe_ratio": stats["sharpe"],
                "total_pnl": stats["pnl"],
                "last_signal": stats["last_signal"],
                "regime": self.regime,
            }
            for name, stats in self.strategies.items()
        ]
    
    def get_meta_learner_status(self) -> Dict:
        """Get meta-learner status."""
        return {
            "enabled": self.meta_learner_enabled,
            "mode": self.meta_learner_mode,
            "current_strategy": self.current_strategy if self.meta_learner_enabled else None,
            "exploration_rate": self.exploration_rate,
            "regime": self.regime,
            "arm_stats": self.arm_stats,
        }
    
    def set_meta_learner(self, config: Dict):
        """Update meta-learner configuration."""
        self.meta_learner_enabled = config.get("enabled", self.meta_learner_enabled)
        self.meta_learner_mode = config.get("mode", self.meta_learner_mode)
        self.exploration_rate = config.get("exploration_rate", self.exploration_rate)
    
    def update_meta_learner(self, result: Dict):
        """Update meta-learner with trade result."""
        strategy = result["strategy"]
        if strategy in self.arm_stats:
            if result["is_win"]:
                self.arm_stats[strategy]["alpha"] += 1
            else:
                self.arm_stats[strategy]["beta"] += 1
            self.arm_stats[strategy]["pulls"] += 1
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades."""
        return self.trades[-limit:] if self.trades else []
    
    def add_trade(self, trade: Dict):
        """Add a new trade."""
        self.trades.append(trade)
        self.last_trade_time = datetime.utcnow().isoformat()
        
        # Update strategy stats
        strategy = trade.get("strategy")
        if strategy in self.strategies:
            self.strategies[strategy]["trades"] += 1
            self.strategies[strategy]["pnl"] += trade.get("pnl", 0)
    
    def get_alerts(self, acknowledged: Optional[bool] = None) -> List[Dict]:
        """Get alerts."""
        if acknowledged is None:
            return self.alerts
        return [a for a in self.alerts if a["acknowledged"] == acknowledged]
    
    def add_alert(self, severity: str, message: str):
        """Add a new alert."""
        self.alerts.append({
            "id": f"alert_{len(self.alerts) + 1}",
            "severity": severity,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "acknowledged": False,
        })
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                return True
        return False
    
    def get_pnl_history(self, days: int = 30) -> List[Dict]:
        """Get P&L history."""
        history = []
        base_pnl = 0
        
        for i in range(days):
            date = datetime.utcnow() - timedelta(days=days - i)
            daily_pnl = random.uniform(-500, 800)
            base_pnl += daily_pnl
            
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "daily_pnl": round(daily_pnl, 2),
                "cumulative_pnl": round(base_pnl, 2),
            })
        
        return history


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Quant Bot Dashboard API",
    description="API for the React dashboard - real-time trading data and control",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
manager = ConnectionManager()
data_store = DataStore()


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get system status."""
    return data_store.get_system_status()


@app.get("/api/positions")
async def get_positions():
    """Get current positions."""
    return {"positions": data_store.get_positions()}


@app.get("/api/trades")
async def get_trades(limit: int = Query(50, ge=1, le=500)):
    """Get recent trades."""
    return {"trades": data_store.get_recent_trades(limit)}


@app.get("/api/strategies")
async def get_strategies():
    """Get strategy statistics."""
    return {"strategies": data_store.get_strategy_stats()}


@app.post("/api/strategies/{name}/toggle")
async def toggle_strategy(name: str):
    """Toggle strategy enabled/disabled."""
    if name not in data_store.strategies:
        raise HTTPException(404, f"Strategy {name} not found")
    
    data_store.strategies[name]["enabled"] = not data_store.strategies[name]["enabled"]
    
    # Broadcast update
    await manager.broadcast("strategies", {"strategies": data_store.get_strategy_stats()})
    
    return {"strategy": name, "enabled": data_store.strategies[name]["enabled"]}


@app.get("/api/meta/status", response_model=MetaLearnerStatus)
async def get_meta_status():
    """Get meta-learner status."""
    return data_store.get_meta_learner_status()


@app.post("/api/meta/enable")
async def enable_meta_learner():
    """Enable meta-learner."""
    data_store.meta_learner_enabled = True
    
    await manager.broadcast("meta", data_store.get_meta_learner_status())
    
    return {"status": "enabled", "mode": data_store.meta_learner_mode}


@app.post("/api/meta/disable")
async def disable_meta_learner():
    """Disable meta-learner."""
    data_store.meta_learner_enabled = False
    
    await manager.broadcast("meta", data_store.get_meta_learner_status())
    
    return {"status": "disabled"}


@app.post("/api/meta/configure")
async def configure_meta_learner(config: MetaLearnerConfig):
    """Configure meta-learner."""
    data_store.set_meta_learner(config.dict())
    
    await manager.broadcast("meta", data_store.get_meta_learner_status())
    
    return data_store.get_meta_learner_status()


@app.post("/api/meta/update")
async def update_meta_learner(result: TradeResult):
    """Update meta-learner with trade result."""
    data_store.update_meta_learner(result.dict())
    
    return {"status": "updated", "arm_stats": data_store.arm_stats[result.strategy]}


@app.get("/api/meta/select")
async def select_strategy():
    """Get meta-learner's current strategy selection."""
    if not data_store.meta_learner_enabled:
        raise HTTPException(400, "Meta-learner is disabled")
    
    # Simple Thompson sampling selection
    best_strategy = None
    best_sample = -1
    
    for name, stats in data_store.arm_stats.items():
        if data_store.strategies.get(name, {}).get("enabled", False):
            # Sample from Beta distribution
            sample = random.betavariate(stats["alpha"], stats["beta"])
            if sample > best_sample:
                best_sample = sample
                best_strategy = name
    
    data_store.current_strategy = best_strategy
    
    return {
        "strategy": best_strategy,
        "confidence": best_sample,
        "regime": data_store.regime,
    }


@app.get("/api/alerts")
async def get_alerts(acknowledged: Optional[bool] = None):
    """Get alerts."""
    return {"alerts": data_store.get_alerts(acknowledged)}


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    if data_store.acknowledge_alert(alert_id):
        await manager.broadcast("alerts", {"alerts": data_store.get_alerts()})
        return {"status": "acknowledged"}
    raise HTTPException(404, f"Alert {alert_id} not found")


@app.get("/api/pnl/history")
async def get_pnl_history(days: int = Query(30, ge=1, le=365)):
    """Get P&L history."""
    return {"history": data_store.get_pnl_history(days)}


@app.get("/api/metrics")
async def get_metrics():
    """Get current performance metrics."""
    strategies = data_store.get_strategy_stats()
    total_pnl = sum(s["total_pnl"] for s in strategies)
    total_trades = sum(s["trades"] for s in strategies)
    avg_win_rate = sum(s["win_rate"] for s in strategies) / len(strategies) if strategies else 0
    
    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "avg_win_rate": avg_win_rate,
        "active_strategies": len([s for s in strategies if s["enabled"]]),
        "current_drawdown": random.uniform(0, 0.05),  # Mock
        "sharpe_ratio": random.uniform(1.5, 2.5),  # Mock
    }


@app.get("/api/mode")
async def get_mode():
    """Get current trading mode."""
    return {"mode": data_store.mode}


@app.post("/api/mode/{new_mode}")
async def set_mode(new_mode: str):
    """Set trading mode (paper, canary, production)."""
    if new_mode not in ["paper", "canary", "production"]:
        raise HTTPException(400, "Invalid mode")
    
    if new_mode == "production":
        # Require confirmation
        return {"status": "confirmation_required", "message": "Production mode requires confirmation"}
    
    data_store.mode = new_mode
    await manager.broadcast("status", data_store.get_system_status())
    
    return {"mode": data_store.mode}


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "channel": "init",
            "data": {
                "status": data_store.get_system_status(),
                "positions": data_store.get_positions(),
                "strategies": data_store.get_strategy_stats(),
                "meta": data_store.get_meta_learner_status(),
            },
            "ts": time.time(),
        }))
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                action = message.get("action")
                
                if action == "subscribe":
                    channels = message.get("channels", [])
                    manager.subscribe(websocket, channels)
                    await websocket.send_text(json.dumps({
                        "channel": "subscribed",
                        "data": {"channels": channels},
                        "ts": time.time(),
                    }))
                
                elif action == "ping":
                    await websocket.send_text(json.dumps({
                        "channel": "pong",
                        "ts": time.time(),
                    }))
                
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================================
# Background Tasks
# ============================================================================

async def broadcast_updates():
    """Background task to broadcast periodic updates."""
    while True:
        await asyncio.sleep(2)  # Every 2 seconds
        
        # Broadcast position updates
        await manager.broadcast("positions", {"positions": data_store.get_positions()})
        
        # Broadcast status
        await manager.broadcast("status", data_store.get_system_status())


@app.on_event("startup")
async def startup():
    """Start background tasks."""
    asyncio.create_task(broadcast_updates())
    
    # Add some initial alerts
    data_store.add_alert("info", "System started successfully")
    data_store.add_alert("warning", "Paper trading mode active")


# ============================================================================
# Static Files / React App
# ============================================================================

# Serve React build if available
react_build = PROJECT_ROOT / "dashboard" / "build"
if react_build.exists():
    app.mount("/static", StaticFiles(directory=str(react_build / "static")), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def serve_react():
        return FileResponse(str(react_build / "index.html"))


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dashboard API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                 QUANT BOT DASHBOARD API                      ║
╠══════════════════════════════════════════════════════════════╣
║  REST API:     http://{args.host}:{args.port}/api/*                     
║  WebSocket:    ws://{args.host}:{args.port}/ws                          
║  Docs:         http://{args.host}:{args.port}/docs                      
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.dashboard.api_endpoints:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
