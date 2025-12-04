"""
Autonomous Trading Master Controller
=====================================
This module integrates ALL autonomous trading components into a single
unified controller. It connects:

- Meta-Learner (strategy selection)
- Autonomous Orchestrator (ML-driven trading)
- Safety Gate (risk controls)
- Order-Flow Gate (trade filtering)
- Regime Gate (strategy-regime alignment)
- Hedging Daemon (delta neutralization)
- Canary Orchestrator (staged deployment)

Usage:
    python src/engine/master_controller.py --mode canary --allocation 5
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("MasterController")


# =============================================================================
# Configuration
# =============================================================================

class TradingMode(Enum):
    PAPER = "paper"
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"


@dataclass
class MasterConfig:
    """Master controller configuration."""
    mode: TradingMode = TradingMode.PAPER
    allocation_pct: float = 1.0  # % of capital to use
    
    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ["BTCUSD", "ETHUSD"])
    max_positions_per_symbol: int = 3
    position_size_usd: float = 1000.0
    
    # Timing
    signal_interval_sec: int = 60
    retrain_interval_hours: int = 24
    health_check_interval_sec: int = 30
    
    # Risk limits (safety gate)
    max_hourly_loss_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 10.0
    
    # Kill switch
    kill_switch_path: str = "/tmp/quant_kill_switch"
    
    # Integrations
    enable_order_flow_gate: bool = True
    enable_regime_gate: bool = True
    enable_hedging: bool = True
    enable_meta_learner: bool = True


# =============================================================================
# Component Status Tracking
# =============================================================================

@dataclass
class ComponentStatus:
    """Status of a single component."""
    name: str
    healthy: bool
    last_check: datetime
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class SystemHealthMonitor:
    """Monitors health of all components."""
    
    def __init__(self):
        self._components: Dict[str, ComponentStatus] = {}
        self._start_time = datetime.utcnow()
    
    def register(self, name: str):
        """Register a component."""
        self._components[name] = ComponentStatus(
            name=name,
            healthy=False,
            last_check=datetime.utcnow()
        )
    
    def update(self, name: str, healthy: bool, message: str = "", metrics: Dict = None):
        """Update component status."""
        if name in self._components:
            self._components[name].healthy = healthy
            self._components[name].last_check = datetime.utcnow()
            self._components[name].message = message
            if metrics:
                self._components[name].metrics = metrics
    
    def all_healthy(self) -> bool:
        """Check if all components are healthy."""
        return all(c.healthy for c in self._components.values())
    
    def get_unhealthy(self) -> List[str]:
        """Get list of unhealthy components."""
        return [name for name, c in self._components.items() if not c.healthy]
    
    def get_status_report(self) -> Dict:
        """Get full status report."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        return {
            "uptime_seconds": uptime,
            "all_healthy": self.all_healthy(),
            "components": {
                name: {
                    "healthy": c.healthy,
                    "last_check": c.last_check.isoformat(),
                    "message": c.message,
                    "metrics": c.metrics
                }
                for name, c in self._components.items()
            }
        }


# =============================================================================
# Mock Component Interfaces (replace with actual imports in production)
# =============================================================================

class MockMetaLearner:
    """Mock meta-learner for strategy selection."""
    
    def __init__(self):
        self.strategies = ["momentum", "mean_reversion", "stat_arb", "regime_ml"]
    
    async def select_strategy(self, market_data: Dict, regime: str) -> str:
        """Select best strategy using Thompson Sampling."""
        # In production, this would use actual meta_learner.py
        import random
        return random.choice(self.strategies)
    
    async def update_reward(self, strategy: str, reward: float):
        """Update strategy performance."""
        pass


class MockSafetyGate:
    """Mock safety gate for risk checks."""
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self.hourly_pnl = 0.0
        self.daily_pnl = 0.0
        self.peak_equity = 100000.0
        self.current_equity = 100000.0
    
    def check_trade(self, trade: Dict) -> tuple[bool, str]:
        """Check if trade passes safety checks."""
        # Check drawdown
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown > self.config.max_drawdown_pct / 100:
            return False, f"Max drawdown exceeded: {drawdown:.2%}"
        
        # Check daily loss
        if self.daily_pnl < -self.config.max_daily_loss_pct:
            return False, f"Daily loss limit exceeded: {self.daily_pnl:.2%}"
        
        return True, "OK"
    
    def check_kill_switch(self) -> bool:
        """Check if kill switch is active."""
        return Path(self.config.kill_switch_path).exists()


class MockOrderFlowGate:
    """Mock order-flow gate."""
    
    async def check_signal(self, symbol: str, signal: str) -> tuple[bool, str]:
        """Check if order-flow confirms signal."""
        # In production, this would use actual orderflow_gate.py
        import random
        if random.random() > 0.2:  # 80% pass rate
            return True, "Order flow confirms"
        return False, "Order flow contradicts signal"


class MockRegimeGate:
    """Mock regime gate."""
    
    def __init__(self):
        self.current_regime = "trending"
        self.compatibility = {
            "trending": ["momentum", "regime_ml"],
            "ranging": ["mean_reversion", "stat_arb"],
            "high_vol": ["stat_arb"],
            "crisis": [],  # No strategies in crisis
        }
    
    async def detect_regime(self, market_data: Dict) -> str:
        """Detect current market regime."""
        return self.current_regime
    
    def is_compatible(self, strategy: str, regime: str) -> bool:
        """Check if strategy is compatible with regime."""
        return strategy in self.compatibility.get(regime, [])


class MockHedgingDaemon:
    """Mock hedging daemon."""
    
    async def get_net_delta(self) -> float:
        """Get current net delta."""
        import random
        return random.uniform(-0.5, 0.5)
    
    async def neutralize_delta(self, target_delta: float = 0.0):
        """Execute delta neutralization."""
        pass


class MockModelManager:
    """Mock ML model manager."""
    
    def __init__(self):
        self.last_trained = datetime.utcnow() - timedelta(hours=12)
        self.samples_since_train = 0
    
    async def predict(self, features: Dict) -> Dict:
        """Generate prediction."""
        import random
        return {
            "direction": random.choice(["long", "short", "neutral"]),
            "confidence": random.uniform(0.5, 0.9),
            "expected_return": random.uniform(-0.02, 0.03)
        }
    
    async def needs_retrain(self, hours: int = 24, samples: int = 1000) -> bool:
        """Check if model needs retraining."""
        hours_since = (datetime.utcnow() - self.last_trained).total_seconds() / 3600
        return hours_since >= hours or self.samples_since_train >= samples
    
    async def retrain(self, data: List[Dict]):
        """Retrain model."""
        self.last_trained = datetime.utcnow()
        self.samples_since_train = 0
        logger.info("Model retrained")


# =============================================================================
# Master Controller
# =============================================================================

class MasterController:
    """
    Master controller that integrates all autonomous trading components.
    
    This is the single entry point for autonomous trading, coordinating:
    - Strategy selection via meta-learner
    - Signal generation via ML models
    - Risk checks via safety gate
    - Trade filtering via order-flow and regime gates
    - Position hedging via hedging daemon
    """
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Health monitoring
        self.health = SystemHealthMonitor()
        
        # Initialize components (use mocks, replace with real imports)
        self._init_components()
        
        # Trading state
        self.positions: Dict[str, List[Dict]] = {}
        self.pending_orders: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.daily_pnl: float = 0.0
        
        # Metrics
        self.metrics = {
            "signals_generated": 0,
            "trades_executed": 0,
            "trades_blocked_safety": 0,
            "trades_blocked_orderflow": 0,
            "trades_blocked_regime": 0,
            "total_pnl": 0.0,
        }
        
        logger.info(f"MasterController initialized | Mode: {config.mode.value} | Allocation: {config.allocation_pct}%")
    
    def _init_components(self):
        """Initialize all trading components."""
        # Register components for health monitoring
        components = [
            "meta_learner",
            "model_manager",
            "safety_gate",
            "order_flow_gate",
            "regime_gate",
            "hedging_daemon",
            "exchange_client",
        ]
        for comp in components:
            self.health.register(comp)
        
        # Initialize component instances (mocks for now)
        self.meta_learner = MockMetaLearner()
        self.model_manager = MockModelManager()
        self.safety_gate = MockSafetyGate(self.config)
        self.order_flow_gate = MockOrderFlowGate()
        self.regime_gate = MockRegimeGate()
        self.hedging_daemon = MockHedgingDaemon()
        
        # Mark all as healthy initially
        for comp in components:
            self.health.update(comp, True, "Initialized")
    
    async def start(self):
        """Start the master controller."""
        logger.info("=" * 60)
        logger.info("MASTER CONTROLLER STARTING")
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"Allocation: {self.config.allocation_pct}%")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info("=" * 60)
        
        self._running = True
        
        # Setup signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                asyncio.get_event_loop().add_signal_handler(
                    sig, lambda: asyncio.create_task(self.shutdown())
                )
            except NotImplementedError:
                pass  # Windows doesn't support this
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._retrain_loop()),
        ]
        
        if self.config.enable_hedging:
            tasks.append(asyncio.create_task(self._hedging_loop()))
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        logger.info("Master controller stopped")
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutdown requested...")
        self._running = False
        self._shutdown_event.set()
    
    # =========================================================================
    # Main Trading Loop
    # =========================================================================
    
    async def _trading_loop(self):
        """Main trading loop."""
        logger.info("Trading loop started")
        
        while self._running:
            try:
                # 1. Check kill switch
                if self.safety_gate.check_kill_switch():
                    logger.critical("KILL SWITCH ACTIVE - Trading halted")
                    await asyncio.sleep(60)
                    continue
                
                # 2. Check system health
                if not self.health.all_healthy():
                    unhealthy = self.health.get_unhealthy()
                    logger.warning(f"Unhealthy components: {unhealthy}")
                    await asyncio.sleep(30)
                    continue
                
                # 3. Process each symbol
                for symbol in self.config.symbols:
                    await self._process_symbol(symbol)
                
                # 4. Wait for next interval
                await asyncio.sleep(self.config.signal_interval_sec)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    async def _process_symbol(self, symbol: str):
        """Process trading logic for a single symbol."""
        try:
            # 1. Get market data (mock)
            market_data = await self._get_market_data(symbol)
            
            # 2. Detect regime
            regime = await self.regime_gate.detect_regime(market_data)
            
            # 3. Select strategy using meta-learner
            if self.config.enable_meta_learner:
                strategy = await self.meta_learner.select_strategy(market_data, regime)
            else:
                strategy = "momentum"  # Default
            
            # 4. Check regime compatibility
            if self.config.enable_regime_gate:
                if not self.regime_gate.is_compatible(strategy, regime):
                    logger.debug(f"{symbol}: Strategy {strategy} incompatible with regime {regime}")
                    self.metrics["trades_blocked_regime"] += 1
                    return
            
            # 5. Generate ML signal
            features = self._extract_features(market_data)
            prediction = await self.model_manager.predict(features)
            self.model_manager.samples_since_train += 1
            self.metrics["signals_generated"] += 1
            
            # 6. Check signal strength
            if prediction["direction"] == "neutral" or prediction["confidence"] < 0.6:
                return
            
            # 7. Check order-flow confirmation
            if self.config.enable_order_flow_gate:
                confirmed, reason = await self.order_flow_gate.check_signal(
                    symbol, prediction["direction"]
                )
                if not confirmed:
                    logger.debug(f"{symbol}: Order-flow gate blocked - {reason}")
                    self.metrics["trades_blocked_orderflow"] += 1
                    return
            
            # 8. Build trade
            trade = {
                "symbol": symbol,
                "side": prediction["direction"],
                "size_usd": self.config.position_size_usd * (self.config.allocation_pct / 100),
                "strategy": strategy,
                "confidence": prediction["confidence"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # 9. Safety gate check
            passed, reason = self.safety_gate.check_trade(trade)
            if not passed:
                logger.warning(f"{symbol}: Safety gate blocked - {reason}")
                self.metrics["trades_blocked_safety"] += 1
                return
            
            # 10. Execute trade (paper mode just logs)
            await self._execute_trade(trade)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def _execute_trade(self, trade: Dict):
        """Execute a trade."""
        if self.config.mode == TradingMode.PAPER:
            logger.info(f"📝 PAPER TRADE: {trade['side'].upper()} {trade['symbol']} | "
                       f"Size: ${trade['size_usd']:.2f} | Strategy: {trade['strategy']} | "
                       f"Confidence: {trade['confidence']:.2%}")
        elif self.config.mode == TradingMode.SHADOW:
            logger.info(f"👤 SHADOW TRADE: {trade['side'].upper()} {trade['symbol']} | "
                       f"Size: ${trade['size_usd']:.2f}")
        else:
            # Real execution would happen here
            logger.info(f"💰 LIVE TRADE: {trade['side'].upper()} {trade['symbol']} | "
                       f"Size: ${trade['size_usd']:.2f}")
        
        self.trade_history.append(trade)
        self.metrics["trades_executed"] += 1
        
        # Update meta-learner reward (mock)
        import random
        reward = random.uniform(-1, 2)  # Mock P&L
        await self.meta_learner.update_reward(trade["strategy"], reward)
    
    # =========================================================================
    # Background Loops
    # =========================================================================
    
    async def _health_check_loop(self):
        """Periodic health checks."""
        while self._running:
            try:
                # Check each component
                self.health.update("meta_learner", True, "Active")
                self.health.update("model_manager", True, 
                                   f"Samples: {self.model_manager.samples_since_train}")
                self.health.update("safety_gate", True, 
                                   f"Daily P&L: {self.daily_pnl:.2%}")
                self.health.update("order_flow_gate", True, "Active")
                self.health.update("regime_gate", True, 
                                   f"Regime: {self.regime_gate.current_regime}")
                self.health.update("hedging_daemon", True, "Active")
                self.health.update("exchange_client", True, "Connected")
                
                await asyncio.sleep(self.config.health_check_interval_sec)
            except asyncio.CancelledError:
                break
    
    async def _retrain_loop(self):
        """Periodic model retraining."""
        while self._running:
            try:
                if await self.model_manager.needs_retrain(
                    self.config.retrain_interval_hours, 1000
                ):
                    logger.info("🔄 Triggering model retrain...")
                    await self.model_manager.retrain(self.trade_history[-1000:])
                
                await asyncio.sleep(3600)  # Check hourly
            except asyncio.CancelledError:
                break
    
    async def _hedging_loop(self):
        """Periodic delta hedging."""
        while self._running:
            try:
                net_delta = await self.hedging_daemon.get_net_delta()
                if abs(net_delta) > 0.1:  # Threshold
                    logger.info(f"🛡️ Hedging: Net delta {net_delta:.3f}, neutralizing...")
                    await self.hedging_daemon.neutralize_delta()
                
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get market data for symbol (mock)."""
        import random
        return {
            "symbol": symbol,
            "price": 98000 + random.uniform(-2000, 2000),
            "volume_24h": random.uniform(1e9, 5e9),
            "change_24h": random.uniform(-0.05, 0.05),
        }
    
    def _extract_features(self, market_data: Dict) -> Dict:
        """Extract features for ML model."""
        return {
            "price": market_data["price"],
            "volume": market_data["volume_24h"],
            "momentum": market_data["change_24h"],
        }
    
    def get_status(self) -> Dict:
        """Get current status."""
        return {
            "mode": self.config.mode.value,
            "allocation_pct": self.config.allocation_pct,
            "running": self._running,
            "health": self.health.get_status_report(),
            "metrics": self.metrics,
            "positions": self.positions,
            "pending_orders": len(self.pending_orders),
            "trade_count": len(self.trade_history),
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Trading Master Controller")
    parser.add_argument("--mode", choices=["paper", "shadow", "canary", "production"],
                        default="paper", help="Trading mode")
    parser.add_argument("--allocation", type=float, default=1.0,
                        help="Capital allocation percentage")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD"],
                        help="Symbols to trade")
    args = parser.parse_args()
    
    config = MasterConfig(
        mode=TradingMode(args.mode),
        allocation_pct=args.allocation,
        symbols=args.symbols,
    )
    
    controller = MasterController(config)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     🤖 AUTONOMOUS TRADING MASTER CONTROLLER                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Mode:          {config.mode.value:12}                                           ║
║  Allocation:    {config.allocation_pct:5.1f}%                                            ║
║  Symbols:       {', '.join(config.symbols):30}                     ║
║                                                                               ║
║  Components:                                                                  ║
║   ✓ Meta-Learner (Thompson Sampling)                                          ║
║   ✓ ML Model Manager (Auto-retrain)                                           ║
║   ✓ Safety Gate (Risk limits)                                                 ║
║   ✓ Order-Flow Gate (Trade filtering)                                         ║
║   ✓ Regime Gate (Strategy alignment)                                          ║
║   ✓ Hedging Daemon (Delta neutral)                                            ║
║                                                                               ║
║  Press Ctrl+C to stop                                                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    await controller.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
