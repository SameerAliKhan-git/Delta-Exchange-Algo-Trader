"""
Automated Hedging Daemon
========================
Production service implementing Optuna-optimized params with automatic
rollback on slippage spikes. Runs as a background service.

Features:
- Loads optimal params from Optuna study or config
- Continuous delta-neutral hedging
- Slippage monitoring with automatic rollback
- Prometheus metrics integration
- Graceful shutdown handling

Usage:
    python src/services/hedging_daemon.py --config config/hedging.yml
    python src/services/hedging_daemon.py --study optuna_hedging_study
"""

import asyncio
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from threading import Lock
import argparse

# Optional imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class HedgeMode(Enum):
    """Hedging mode."""
    AGGRESSIVE = "aggressive"      # Tight thresholds, frequent hedging
    BALANCED = "balanced"          # Default mode
    CONSERVATIVE = "conservative"  # Wide thresholds, less hedging
    EMERGENCY = "emergency"        # Flatten everything


@dataclass
class HedgeParams:
    """Hedging parameters (from Optuna or config)."""
    delta_threshold: float = 0.05        # Hedge when abs(delta) > threshold
    hedge_interval_seconds: int = 60     # Check interval
    max_hedge_size_pct: float = 0.25     # Max hedge size per iteration
    slippage_limit_bps: float = 15.0     # Max acceptable slippage
    cooldown_seconds: int = 30           # Cooldown after hedge
    gamma_weight: float = 0.3            # Gamma consideration in sizing
    vega_weight: float = 0.2             # Vega consideration
    use_futures: bool = True             # Use futures for hedging
    use_perps: bool = True               # Use perpetuals
    prefer_maker: bool = True            # Prefer maker orders
    
    @classmethod
    def from_optuna_study(cls, study_name: str, storage: Optional[str] = None) -> 'HedgeParams':
        """Load best params from Optuna study."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed")
        
        storage = storage or "sqlite:///data/optuna_studies.db"
        study = optuna.load_study(study_name=study_name, storage=storage)
        best = study.best_params
        
        return cls(
            delta_threshold=best.get('delta_threshold', 0.05),
            hedge_interval_seconds=int(best.get('hedge_interval', 60)),
            max_hedge_size_pct=best.get('max_hedge_size', 0.25),
            slippage_limit_bps=best.get('slippage_limit', 15.0),
            cooldown_seconds=int(best.get('cooldown', 30)),
            gamma_weight=best.get('gamma_weight', 0.3),
            vega_weight=best.get('vega_weight', 0.2),
        )
    
    @classmethod
    def from_config(cls, config_path: str) -> 'HedgeParams':
        """Load params from YAML config."""
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        hedging = cfg.get('hedging', {})
        return cls(**{k: v for k, v in hedging.items() if hasattr(cls, k)})


@dataclass
class PortfolioGreeks:
    """Current portfolio Greeks."""
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    net_theta: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def needs_hedge(self) -> bool:
        """Check if delta exceeds typical threshold."""
        return abs(self.net_delta) > 0.05


@dataclass
class HedgeOrder:
    """Hedge order details."""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: str  # 'market' or 'limit'
    price: Optional[float] = None
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HedgeResult:
    """Result of a hedge execution."""
    success: bool
    order: HedgeOrder
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    slippage_bps: Optional[float] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class SlippageEvent:
    """Slippage tracking event."""
    timestamp: datetime
    expected_price: float
    actual_price: float
    slippage_bps: float
    size: float
    symbol: str


# =============================================================================
# Prometheus Metrics
# =============================================================================

if PROMETHEUS_AVAILABLE:
    HEDGE_EXECUTIONS = Counter(
        'hedging_daemon_executions_total',
        'Total hedge executions',
        ['status', 'symbol']
    )
    HEDGE_SLIPPAGE = Histogram(
        'hedging_daemon_slippage_bps',
        'Hedge slippage in basis points',
        buckets=[1, 2, 5, 10, 15, 20, 30, 50, 100]
    )
    HEDGE_LATENCY = Histogram(
        'hedging_daemon_latency_ms',
        'Hedge execution latency',
        buckets=[10, 25, 50, 100, 250, 500, 1000]
    )
    PORTFOLIO_DELTA = Gauge(
        'hedging_daemon_portfolio_delta',
        'Current portfolio net delta'
    )
    DAEMON_STATUS = Gauge(
        'hedging_daemon_status',
        'Daemon status (1=running, 0=stopped)'
    )
    ROLLBACK_COUNT = Counter(
        'hedging_daemon_rollbacks_total',
        'Total rollbacks triggered'
    )


# =============================================================================
# Hedging Daemon
# =============================================================================

class HedgingDaemon:
    """
    Production hedging daemon that maintains delta neutrality.
    
    Features:
    - Optuna-optimized parameters
    - Slippage monitoring with auto-rollback
    - Mode switching (aggressive/balanced/conservative)
    - Prometheus metrics
    - Graceful shutdown
    """
    
    def __init__(
        self,
        params: HedgeParams,
        exchange_client: Any = None,
        greeks_provider: Optional[Callable[[], PortfolioGreeks]] = None,
        mode: HedgeMode = HedgeMode.BALANCED,
    ):
        self.params = params
        self.exchange = exchange_client
        self.greeks_provider = greeks_provider
        self.mode = mode
        
        # State
        self._running = False
        self._lock = Lock()
        self._last_hedge_time: Optional[datetime] = None
        self._slippage_history: List[SlippageEvent] = []
        self._consecutive_high_slippage = 0
        
        # Rollback thresholds
        self._slippage_rollback_threshold = 3  # Consecutive high slippage events
        self._slippage_window = timedelta(minutes=30)
        
        # Callbacks
        self._on_rollback: Optional[Callable] = None
        self._on_hedge: Optional[Callable[[HedgeResult], None]] = None
        
        logger.info(f"HedgingDaemon initialized with mode={mode.value}")
        logger.info(f"Params: delta_threshold={params.delta_threshold}, "
                   f"interval={params.hedge_interval_seconds}s")
    
    def set_rollback_callback(self, callback: Callable) -> None:
        """Set callback for rollback events."""
        self._on_rollback = callback
    
    def set_hedge_callback(self, callback: Callable[[HedgeResult], None]) -> None:
        """Set callback for hedge events."""
        self._on_hedge = callback
    
    def set_mode(self, mode: HedgeMode) -> None:
        """Switch hedging mode."""
        old_mode = self.mode
        self.mode = mode
        logger.warning(f"Mode changed: {old_mode.value} -> {mode.value}")
        
        # Adjust params based on mode
        if mode == HedgeMode.AGGRESSIVE:
            self.params.delta_threshold = 0.02
            self.params.hedge_interval_seconds = 30
        elif mode == HedgeMode.CONSERVATIVE:
            self.params.delta_threshold = 0.10
            self.params.hedge_interval_seconds = 120
        elif mode == HedgeMode.EMERGENCY:
            self.params.delta_threshold = 0.001  # Hedge almost everything
            self.params.hedge_interval_seconds = 10
    
    async def start(self) -> None:
        """Start the hedging daemon."""
        self._running = True
        
        if PROMETHEUS_AVAILABLE:
            DAEMON_STATUS.set(1)
        
        logger.info("Hedging daemon started")
        
        while self._running:
            try:
                await self._hedge_cycle()
                await asyncio.sleep(self.params.hedge_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Hedge cycle error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
        
        if PROMETHEUS_AVAILABLE:
            DAEMON_STATUS.set(0)
        
        logger.info("Hedging daemon stopped")
    
    def stop(self) -> None:
        """Stop the hedging daemon."""
        self._running = False
        logger.info("Hedging daemon stop requested")
    
    async def _hedge_cycle(self) -> None:
        """Execute one hedge cycle."""
        # Get current Greeks
        greeks = await self._get_greeks()
        
        if PROMETHEUS_AVAILABLE:
            PORTFOLIO_DELTA.set(greeks.net_delta)
        
        # Check if hedge needed
        threshold = self._get_adjusted_threshold()
        if abs(greeks.net_delta) <= threshold:
            logger.debug(f"Delta {greeks.net_delta:.4f} within threshold {threshold}")
            return
        
        # Check cooldown
        if self._in_cooldown():
            logger.debug("In cooldown period, skipping hedge")
            return
        
        # Calculate hedge
        hedge_order = self._calculate_hedge(greeks)
        
        if hedge_order is None:
            return
        
        # Execute hedge
        result = await self._execute_hedge(hedge_order)
        
        # Process result
        await self._process_hedge_result(result)
        
        # Update last hedge time
        self._last_hedge_time = datetime.utcnow()
    
    async def _get_greeks(self) -> PortfolioGreeks:
        """Get current portfolio Greeks."""
        if self.greeks_provider:
            return self.greeks_provider()
        
        # Demo: simulate Greeks
        import random
        return PortfolioGreeks(
            net_delta=random.uniform(-0.15, 0.15),
            net_gamma=random.uniform(-0.01, 0.01),
            net_vega=random.uniform(-100, 100),
            net_theta=random.uniform(-50, 0),
        )
    
    def _get_adjusted_threshold(self) -> float:
        """Get delta threshold adjusted for mode."""
        base = self.params.delta_threshold
        
        if self.mode == HedgeMode.AGGRESSIVE:
            return base * 0.5
        elif self.mode == HedgeMode.CONSERVATIVE:
            return base * 2.0
        elif self.mode == HedgeMode.EMERGENCY:
            return 0.001
        return base
    
    def _in_cooldown(self) -> bool:
        """Check if in cooldown period."""
        if self._last_hedge_time is None:
            return False
        
        elapsed = (datetime.utcnow() - self._last_hedge_time).total_seconds()
        return elapsed < self.params.cooldown_seconds
    
    def _calculate_hedge(self, greeks: PortfolioGreeks) -> Optional[HedgeOrder]:
        """Calculate optimal hedge order."""
        # Target: neutralize delta
        delta_to_hedge = -greeks.net_delta
        
        # Adjust for gamma (hedge more if gamma is adverse)
        gamma_adj = 1.0 + self.params.gamma_weight * abs(greeks.net_gamma) * 100
        adjusted_size = delta_to_hedge * gamma_adj
        
        # Apply max size limit
        max_size = self.params.max_hedge_size_pct * abs(delta_to_hedge)
        if abs(adjusted_size) > max_size:
            adjusted_size = max_size if adjusted_size > 0 else -max_size
        
        # Determine symbol and side
        symbol = "BTCUSD"  # Primary hedge instrument
        side = "buy" if adjusted_size > 0 else "sell"
        
        # Determine order type
        order_type = "limit" if self.params.prefer_maker else "market"
        
        return HedgeOrder(
            symbol=symbol,
            side=side,
            size=abs(adjusted_size),
            order_type=order_type,
            reason=f"Delta neutralization: {greeks.net_delta:.4f} -> ~0"
        )
    
    async def _execute_hedge(self, order: HedgeOrder) -> HedgeResult:
        """Execute hedge order."""
        start_time = time.time()
        
        try:
            if self.exchange:
                # Real execution
                result = await self.exchange.place_order(
                    symbol=order.symbol,
                    side=order.side,
                    size=order.size,
                    order_type=order.order_type,
                    price=order.price,
                )
                fill_price = result.get('fill_price', order.price)
                fill_size = result.get('fill_size', order.size)
            else:
                # Demo execution
                import random
                fill_price = 100000 * (1 + random.uniform(-0.001, 0.001))
                fill_size = order.size
                await asyncio.sleep(0.05)  # Simulate latency
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate slippage
            expected_price = order.price or fill_price
            slippage_bps = abs(fill_price - expected_price) / expected_price * 10000
            
            # Record metrics
            if PROMETHEUS_AVAILABLE:
                HEDGE_EXECUTIONS.labels(status='success', symbol=order.symbol).inc()
                HEDGE_SLIPPAGE.observe(slippage_bps)
                HEDGE_LATENCY.observe(latency_ms)
            
            return HedgeResult(
                success=True,
                order=order,
                fill_price=fill_price,
                fill_size=fill_size,
                slippage_bps=slippage_bps,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                HEDGE_EXECUTIONS.labels(status='error', symbol=order.symbol).inc()
            
            return HedgeResult(
                success=False,
                order=order,
                error=str(e),
            )
    
    async def _process_hedge_result(self, result: HedgeResult) -> None:
        """Process hedge result and check for rollback triggers."""
        if not result.success:
            logger.error(f"Hedge failed: {result.error}")
            return
        
        logger.info(
            f"Hedge executed: {result.order.side} {result.fill_size:.4f} "
            f"{result.order.symbol} @ {result.fill_price:.2f} "
            f"(slippage: {result.slippage_bps:.1f} bps)"
        )
        
        # Callback
        if self._on_hedge:
            self._on_hedge(result)
        
        # Track slippage
        if result.slippage_bps and result.slippage_bps > self.params.slippage_limit_bps:
            self._consecutive_high_slippage += 1
            
            self._slippage_history.append(SlippageEvent(
                timestamp=datetime.utcnow(),
                expected_price=result.order.price or result.fill_price,
                actual_price=result.fill_price,
                slippage_bps=result.slippage_bps,
                size=result.fill_size,
                symbol=result.order.symbol,
            ))
            
            logger.warning(
                f"High slippage detected: {result.slippage_bps:.1f} bps "
                f"(limit: {self.params.slippage_limit_bps} bps) "
                f"[consecutive: {self._consecutive_high_slippage}]"
            )
            
            # Check rollback trigger
            if self._consecutive_high_slippage >= self._slippage_rollback_threshold:
                await self._trigger_rollback("Consecutive high slippage events")
        else:
            self._consecutive_high_slippage = 0
        
        # Clean old slippage history
        cutoff = datetime.utcnow() - self._slippage_window
        self._slippage_history = [
            e for e in self._slippage_history if e.timestamp > cutoff
        ]
    
    async def _trigger_rollback(self, reason: str) -> None:
        """Trigger rollback procedure."""
        logger.critical(f"ROLLBACK TRIGGERED: {reason}")
        
        if PROMETHEUS_AVAILABLE:
            ROLLBACK_COUNT.inc()
        
        # Switch to conservative mode
        self.set_mode(HedgeMode.CONSERVATIVE)
        
        # Reset consecutive counter
        self._consecutive_high_slippage = 0
        
        # Call rollback callback
        if self._on_rollback:
            self._on_rollback(reason)
        
        # Log slippage history
        logger.warning("Recent slippage events:")
        for event in self._slippage_history[-10:]:
            logger.warning(
                f"  {event.timestamp}: {event.symbol} "
                f"{event.slippage_bps:.1f} bps on {event.size:.4f}"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get daemon status."""
        return {
            'running': self._running,
            'mode': self.mode.value,
            'params': {
                'delta_threshold': self.params.delta_threshold,
                'hedge_interval': self.params.hedge_interval_seconds,
                'slippage_limit': self.params.slippage_limit_bps,
            },
            'last_hedge': self._last_hedge_time.isoformat() if self._last_hedge_time else None,
            'consecutive_high_slippage': self._consecutive_high_slippage,
            'slippage_events_30m': len(self._slippage_history),
        }


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hedging Daemon')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--study', type=str, help='Optuna study name')
    parser.add_argument('--mode', type=str, default='balanced',
                       choices=['aggressive', 'balanced', 'conservative'])
    parser.add_argument('--metrics-port', type=int, default=8001,
                       help='Prometheus metrics port')
    args = parser.parse_args()
    
    # Load params
    if args.study:
        params = HedgeParams.from_optuna_study(args.study)
        logger.info(f"Loaded params from Optuna study: {args.study}")
    elif args.config:
        params = HedgeParams.from_config(args.config)
        logger.info(f"Loaded params from config: {args.config}")
    else:
        params = HedgeParams()
        logger.info("Using default params")
    
    # Start metrics server
    if PROMETHEUS_AVAILABLE:
        start_http_server(args.metrics_port)
        logger.info(f"Prometheus metrics on port {args.metrics_port}")
    
    # Create daemon
    mode = HedgeMode(args.mode)
    daemon = HedgingDaemon(params=params, mode=mode)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, daemon.stop)
    
    # Run
    await daemon.start()


if __name__ == '__main__':
    asyncio.run(main())
