"""
Funding Arbitrage Auto-Scaler
=============================
Dynamic sizing controller that throttles funding arb positions based on
real-time slippage curve and capacity constraints.

Features:
- Real-time slippage monitoring
- Capacity curve estimation
- Dynamic position sizing
- Auto-throttling on degradation
- Prometheus metrics

Usage:
    from src.services.funding_arb_scaler import FundingArbAutoScaler
    
    scaler = FundingArbAutoScaler(config)
    await scaler.start()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
import json

# Optional imports
try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class ScalerMode(Enum):
    """Scaler operating mode."""
    AGGRESSIVE = "aggressive"      # Max capacity utilization
    NORMAL = "normal"              # Default mode
    CONSERVATIVE = "conservative"  # Reduced sizing
    HALTED = "halted"             # No new positions


@dataclass
class SlippageObservation:
    """Single slippage observation."""
    timestamp: datetime
    size_usd: float
    expected_rate_bps: float
    realized_rate_bps: float
    slippage_bps: float
    venue: str
    symbol: str
    
    @property
    def efficiency(self) -> float:
        """Execution efficiency (1.0 = perfect)."""
        if self.expected_rate_bps == 0:
            return 1.0
        return self.realized_rate_bps / self.expected_rate_bps


@dataclass
class CapacityCurve:
    """Estimated capacity curve."""
    # Size buckets (USD)
    size_buckets: List[float] = field(default_factory=lambda: [
        1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000
    ])
    # Expected slippage at each bucket (bps)
    slippage_at_bucket: List[float] = field(default_factory=lambda: [
        0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 50.0
    ])
    # Confidence at each bucket
    confidence: List[float] = field(default_factory=lambda: [
        0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4
    ])
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_expected_slippage(self, size_usd: float) -> Tuple[float, float]:
        """Get expected slippage and confidence for a given size."""
        if not NUMPY_AVAILABLE:
            # Simple linear interpolation
            for i, bucket in enumerate(self.size_buckets):
                if size_usd <= bucket:
                    return self.slippage_at_bucket[i], self.confidence[i]
            return self.slippage_at_bucket[-1], self.confidence[-1]
        
        # NumPy interpolation
        return (
            float(np.interp(size_usd, self.size_buckets, self.slippage_at_bucket)),
            float(np.interp(size_usd, self.size_buckets, self.confidence)),
        )
    
    def get_max_size_for_slippage(self, max_slippage_bps: float) -> float:
        """Get maximum size that keeps slippage under limit."""
        for i, slip in enumerate(self.slippage_at_bucket):
            if slip > max_slippage_bps:
                if i == 0:
                    return self.size_buckets[0] * (max_slippage_bps / slip)
                # Interpolate
                ratio = (max_slippage_bps - self.slippage_at_bucket[i-1]) / (slip - self.slippage_at_bucket[i-1])
                return self.size_buckets[i-1] + ratio * (self.size_buckets[i] - self.size_buckets[i-1])
        return self.size_buckets[-1]


@dataclass
class ScalerConfig:
    """Auto-scaler configuration."""
    # Slippage limits
    max_slippage_bps: float = 10.0           # Hard slippage limit
    target_slippage_bps: float = 5.0         # Target slippage
    slippage_buffer_bps: float = 2.0         # Buffer for estimation error
    
    # Capacity limits
    max_position_usd: float = 100000         # Max single position
    max_total_exposure_usd: float = 500000   # Max total exposure
    min_position_usd: float = 1000           # Min position size
    
    # Throttling
    throttle_on_consecutive_high: int = 3    # Throttle after N high slippage
    throttle_reduction_pct: float = 0.5      # Reduce size by this much
    throttle_recovery_minutes: int = 30      # Recovery period
    
    # Monitoring
    observation_window_hours: int = 24       # History window
    min_observations_for_curve: int = 20    # Min obs for curve estimation
    update_interval_seconds: int = 60        # Curve update interval
    
    # Profitability
    min_funding_rate_bps: float = 1.0        # Min funding to trade
    min_expected_profit_bps: float = 0.5     # After slippage
    
    # Prometheus
    metrics_port: int = 8003


# =============================================================================
# Prometheus Metrics
# =============================================================================

if PROMETHEUS_AVAILABLE:
    SCALER_POSITION_SIZE = Gauge(
        'funding_arb_scaler_position_size_usd',
        'Recommended position size',
        ['venue', 'symbol']
    )
    SCALER_SLIPPAGE_REALIZED = Histogram(
        'funding_arb_scaler_slippage_bps',
        'Realized slippage',
        ['venue'],
        buckets=[0.5, 1, 2, 5, 10, 20, 50, 100]
    )
    SCALER_MODE = Gauge(
        'funding_arb_scaler_mode',
        'Current scaler mode (0=halted, 1=conservative, 2=normal, 3=aggressive)'
    )
    SCALER_CAPACITY_UTILIZATION = Gauge(
        'funding_arb_scaler_capacity_utilization',
        'Current capacity utilization'
    )
    SCALER_THROTTLE_EVENTS = Counter(
        'funding_arb_scaler_throttle_events_total',
        'Total throttle events'
    )


# =============================================================================
# Auto-Scaler
# =============================================================================

class FundingArbAutoScaler:
    """
    Dynamic position sizer for funding arbitrage.
    
    Monitors real-time slippage and adjusts position sizes to maintain
    profitability while maximizing capacity utilization.
    """
    
    def __init__(
        self,
        config: ScalerConfig,
        exchange_client: Any = None,
        funding_provider: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        self.config = config
        self.exchange = exchange_client
        self.funding_provider = funding_provider
        
        # State
        self._mode = ScalerMode.NORMAL
        self._running = False
        self._observations: deque = deque(maxlen=10000)
        self._capacity_curve = CapacityCurve()
        self._consecutive_high_slippage = 0
        self._throttle_until: Optional[datetime] = None
        
        # Current recommendations
        self._current_sizes: Dict[str, float] = {}
        self._last_update: Optional[datetime] = None
        
        logger.info("FundingArbAutoScaler initialized")
    
    @property
    def mode(self) -> ScalerMode:
        return self._mode
    
    @mode.setter
    def mode(self, value: ScalerMode) -> None:
        old_mode = self._mode
        self._mode = value
        logger.warning(f"Scaler mode changed: {old_mode.value} -> {value.value}")
        
        if PROMETHEUS_AVAILABLE:
            mode_values = {
                ScalerMode.HALTED: 0,
                ScalerMode.CONSERVATIVE: 1,
                ScalerMode.NORMAL: 2,
                ScalerMode.AGGRESSIVE: 3,
            }
            SCALER_MODE.set(mode_values.get(value, 2))
    
    async def start(self) -> None:
        """Start the auto-scaler."""
        self._running = True
        logger.info("Auto-scaler started")
        
        while self._running:
            try:
                await self._update_cycle()
                await asyncio.sleep(self.config.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update cycle error: {e}")
                await asyncio.sleep(5)
        
        logger.info("Auto-scaler stopped")
    
    def stop(self) -> None:
        """Stop the auto-scaler."""
        self._running = False
    
    async def _update_cycle(self) -> None:
        """Execute one update cycle."""
        # Check throttle recovery
        if self._throttle_until and datetime.utcnow() > self._throttle_until:
            logger.info("Throttle period ended, recovering")
            self._throttle_until = None
            self._consecutive_high_slippage = 0
            if self._mode == ScalerMode.CONSERVATIVE:
                self.mode = ScalerMode.NORMAL
        
        # Update capacity curve
        if len(self._observations) >= self.config.min_observations_for_curve:
            self._update_capacity_curve()
        
        # Calculate recommended sizes
        await self._calculate_sizes()
        
        self._last_update = datetime.utcnow()
    
    def record_execution(
        self,
        size_usd: float,
        expected_rate_bps: float,
        realized_rate_bps: float,
        venue: str,
        symbol: str,
    ) -> None:
        """Record an execution for slippage tracking."""
        slippage = expected_rate_bps - realized_rate_bps
        
        obs = SlippageObservation(
            timestamp=datetime.utcnow(),
            size_usd=size_usd,
            expected_rate_bps=expected_rate_bps,
            realized_rate_bps=realized_rate_bps,
            slippage_bps=slippage,
            venue=venue,
            symbol=symbol,
        )
        
        self._observations.append(obs)
        
        # Update metrics
        if PROMETHEUS_AVAILABLE:
            SCALER_SLIPPAGE_REALIZED.labels(venue=venue).observe(slippage)
        
        # Check for high slippage
        if slippage > self.config.max_slippage_bps:
            self._consecutive_high_slippage += 1
            logger.warning(
                f"High slippage: {slippage:.1f} bps on ${size_usd:,.0f} "
                f"[consecutive: {self._consecutive_high_slippage}]"
            )
            
            # Trigger throttle
            if self._consecutive_high_slippage >= self.config.throttle_on_consecutive_high:
                self._trigger_throttle()
        else:
            self._consecutive_high_slippage = 0
        
        logger.debug(
            f"Recorded execution: {venue}/{symbol} ${size_usd:,.0f} "
            f"slippage={slippage:.1f}bps"
        )
    
    def _trigger_throttle(self) -> None:
        """Trigger throttling mode."""
        logger.warning("THROTTLE TRIGGERED: Switching to conservative mode")
        
        self.mode = ScalerMode.CONSERVATIVE
        self._throttle_until = datetime.utcnow() + timedelta(
            minutes=self.config.throttle_recovery_minutes
        )
        self._consecutive_high_slippage = 0
        
        if PROMETHEUS_AVAILABLE:
            SCALER_THROTTLE_EVENTS.inc()
    
    def _update_capacity_curve(self) -> None:
        """Update capacity curve from observations."""
        # Filter recent observations
        cutoff = datetime.utcnow() - timedelta(hours=self.config.observation_window_hours)
        recent = [o for o in self._observations if o.timestamp > cutoff]
        
        if len(recent) < self.config.min_observations_for_curve:
            return
        
        # Group by size bucket and calculate average slippage
        buckets = self._capacity_curve.size_buckets
        slippage_by_bucket: Dict[int, List[float]] = {i: [] for i in range(len(buckets))}
        
        for obs in recent:
            # Find bucket
            for i, bucket in enumerate(buckets):
                if obs.size_usd <= bucket:
                    slippage_by_bucket[i].append(obs.slippage_bps)
                    break
        
        # Update curve
        new_slippage = []
        new_confidence = []
        
        for i in range(len(buckets)):
            observations = slippage_by_bucket[i]
            if observations:
                # Use 75th percentile as conservative estimate
                if NUMPY_AVAILABLE:
                    new_slippage.append(float(np.percentile(observations, 75)))
                else:
                    sorted_obs = sorted(observations)
                    idx = int(len(sorted_obs) * 0.75)
                    new_slippage.append(sorted_obs[min(idx, len(sorted_obs)-1)])
                # Confidence based on sample size
                new_confidence.append(min(1.0, len(observations) / 50))
            else:
                # Keep existing estimate with reduced confidence
                new_slippage.append(self._capacity_curve.slippage_at_bucket[i])
                new_confidence.append(self._capacity_curve.confidence[i] * 0.8)
        
        self._capacity_curve.slippage_at_bucket = new_slippage
        self._capacity_curve.confidence = new_confidence
        self._capacity_curve.last_updated = datetime.utcnow()
        
        logger.info("Capacity curve updated from observations")
    
    async def _calculate_sizes(self) -> None:
        """Calculate recommended position sizes."""
        # Get current funding rates
        if self.funding_provider:
            funding_rates = self.funding_provider()
        else:
            # Demo rates
            funding_rates = {
                'BTCUSD': 0.0005,  # 5 bps
                'ETHUSD': 0.0008,  # 8 bps
            }
        
        for symbol, rate in funding_rates.items():
            rate_bps = rate * 10000
            
            # Skip if funding too low
            if abs(rate_bps) < self.config.min_funding_rate_bps:
                self._current_sizes[symbol] = 0
                continue
            
            # Calculate max size that keeps us profitable
            target_slip = self.config.target_slippage_bps
            if self._mode == ScalerMode.CONSERVATIVE:
                target_slip *= self.config.throttle_reduction_pct
            elif self._mode == ScalerMode.AGGRESSIVE:
                target_slip = self.config.max_slippage_bps
            elif self._mode == ScalerMode.HALTED:
                self._current_sizes[symbol] = 0
                continue
            
            # Account for round-trip (entry + exit)
            max_one_way_slip = (abs(rate_bps) - self.config.min_expected_profit_bps) / 2
            target_slip = min(target_slip, max_one_way_slip)
            
            if target_slip <= 0:
                self._current_sizes[symbol] = 0
                continue
            
            # Get max size from capacity curve
            max_size = self._capacity_curve.get_max_size_for_slippage(target_slip)
            
            # Apply limits
            max_size = min(max_size, self.config.max_position_usd)
            max_size = max(max_size, self.config.min_position_usd) if max_size > 0 else 0
            
            self._current_sizes[symbol] = max_size
            
            if PROMETHEUS_AVAILABLE:
                SCALER_POSITION_SIZE.labels(venue='delta', symbol=symbol).set(max_size)
            
            logger.debug(
                f"Size recommendation: {symbol} -> ${max_size:,.0f} "
                f"(funding={rate_bps:.1f}bps, target_slip={target_slip:.1f}bps)"
            )
        
        # Calculate capacity utilization
        total_recommended = sum(self._current_sizes.values())
        utilization = total_recommended / self.config.max_total_exposure_usd
        
        if PROMETHEUS_AVAILABLE:
            SCALER_CAPACITY_UTILIZATION.set(utilization)
    
    def get_recommended_size(self, symbol: str) -> float:
        """Get recommended position size for a symbol."""
        return self._current_sizes.get(symbol, 0)
    
    def get_all_sizes(self) -> Dict[str, float]:
        """Get all recommended sizes."""
        return self._current_sizes.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get scaler status."""
        return {
            'mode': self._mode.value,
            'running': self._running,
            'throttle_until': self._throttle_until.isoformat() if self._throttle_until else None,
            'consecutive_high_slippage': self._consecutive_high_slippage,
            'observations_count': len(self._observations),
            'current_sizes': self._current_sizes,
            'capacity_curve': {
                'size_buckets': self._capacity_curve.size_buckets,
                'slippage_at_bucket': self._capacity_curve.slippage_at_bucket,
                'confidence': self._capacity_curve.confidence,
                'last_updated': self._capacity_curve.last_updated.isoformat(),
            },
            'last_update': self._last_update.isoformat() if self._last_update else None,
        }
    
    def export_curve(self, path: str) -> None:
        """Export capacity curve to file."""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'curve': {
                'size_buckets': self._capacity_curve.size_buckets,
                'slippage_at_bucket': self._capacity_curve.slippage_at_bucket,
                'confidence': self._capacity_curve.confidence,
            },
            'observations_count': len(self._observations),
            'mode': self._mode.value,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Capacity curve exported to {path}")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Demo main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Funding Arb Auto-Scaler')
    parser.add_argument('--mode', type=str, default='normal',
                       choices=['aggressive', 'normal', 'conservative', 'halted'])
    parser.add_argument('--metrics-port', type=int, default=8003)
    args = parser.parse_args()
    
    # Start metrics server
    if PROMETHEUS_AVAILABLE:
        from prometheus_client import start_http_server
        start_http_server(args.metrics_port)
        logger.info(f"Metrics server on port {args.metrics_port}")
    
    config = ScalerConfig(metrics_port=args.metrics_port)
    scaler = FundingArbAutoScaler(config)
    scaler.mode = ScalerMode(args.mode)
    
    # Simulate some observations
    import random
    for _ in range(50):
        size = random.uniform(1000, 100000)
        base_slip = 0.5 + (size / 50000) ** 1.5  # Slippage increases with size
        realized_slip = base_slip * random.uniform(0.7, 1.3)
        
        scaler.record_execution(
            size_usd=size,
            expected_rate_bps=5.0,
            realized_rate_bps=5.0 - realized_slip,
            venue='delta',
            symbol='BTCUSD',
        )
    
    await scaler.start()


if __name__ == '__main__':
    asyncio.run(main())
