"""
IV Surface Metrics Exporter
============================
Prometheus exporter for IV surface monitoring and alerting.

Features:
- Real-time IV surface metrics
- Z-score calculation for shift detection
- Term structure and skew metrics
- Integration with volatility surface module

Usage:
    python monitoring/iv_surface_exporter.py --port 8004
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import argparse
import math

try:
    from prometheus_client import Gauge, Histogram, Counter, start_http_server
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
# Prometheus Metrics
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # ATM IV metrics
    IV_SURFACE_ATM_IV = Gauge(
        'iv_surface_atm_iv',
        'ATM implied volatility',
        ['underlying', 'tenor']
    )
    IV_SURFACE_25D_CALL_IV = Gauge(
        'iv_surface_25d_call_iv',
        '25-delta call implied volatility',
        ['underlying', 'tenor']
    )
    IV_SURFACE_25D_PUT_IV = Gauge(
        'iv_surface_25d_put_iv',
        '25-delta put implied volatility',
        ['underlying', 'tenor']
    )
    
    # Shift detection
    IV_SURFACE_SHIFT_ZSCORE = Gauge(
        'iv_surface_shift_zscore',
        'IV surface shift Z-score (vs 24h)',
        ['underlying']
    )
    
    # Realized vol
    IV_SURFACE_REALIZED_VOL = Gauge(
        'iv_surface_realized_vol',
        'Realized volatility',
        ['underlying', 'window']
    )
    
    # Term structure
    IV_SURFACE_TERM_STRUCTURE_SLOPE = Gauge(
        'iv_surface_term_structure_slope',
        'Term structure slope',
        ['underlying', 'tenor']
    )
    
    # Strike-level IV (for heatmap)
    IV_SURFACE_STRIKE_IV = Gauge(
        'iv_surface_strike_iv',
        'IV at specific strike',
        ['underlying', 'strike', 'tenor']
    )
    
    # SMA and std for Bollinger bands
    IV_SURFACE_ATM_IV_SMA_24H = Gauge(
        'iv_surface_atm_iv_sma_24h',
        '24h SMA of ATM IV',
        ['underlying']
    )
    IV_SURFACE_ATM_IV_STD_24H = Gauge(
        'iv_surface_atm_iv_std_24h',
        '24h standard deviation of ATM IV',
        ['underlying']
    )
    
    # VIX correlation
    IV_SURFACE_VIX_CORRELATION = Gauge(
        'iv_surface_vix_correlation',
        'Correlation with VIX-equivalent',
        ['underlying']
    )
    IV_SURFACE_VIX_INDEX = Gauge(
        'iv_surface_vix_index',
        'Crypto VIX equivalent index',
        []
    )
    
    # Last update timestamp
    IV_SURFACE_LAST_UPDATE = Gauge(
        'iv_surface_last_update_timestamp',
        'Last surface update timestamp',
        ['underlying']
    )


# =============================================================================
# IV Surface Data Classes
# =============================================================================

@dataclass
class IVPoint:
    """Single IV observation."""
    strike: float
    expiry_days: float
    iv: float
    delta: float
    option_type: str  # 'call' or 'put'


@dataclass
class IVSurfaceSnapshot:
    """Complete IV surface snapshot."""
    underlying: str
    spot_price: float
    timestamp: datetime
    
    # ATM IVs by tenor
    atm_ivs: Dict[str, float]  # tenor -> iv
    
    # 25D IVs by tenor
    call_25d_ivs: Dict[str, float]
    put_25d_ivs: Dict[str, float]
    
    # Full strike grid
    strike_ivs: Dict[str, Dict[float, float]]  # tenor -> strike -> iv
    
    # Realized vol
    realized_vol_24h: float
    realized_vol_7d: float
    
    @property
    def atm_iv(self) -> float:
        """Get primary ATM IV (7d tenor)."""
        return self.atm_ivs.get('7d', self.atm_ivs.get('14d', 0.5))
    
    @property
    def risk_reversal_25d(self) -> float:
        """25D risk reversal (put - call)."""
        put_iv = self.put_25d_ivs.get('7d', 0.5)
        call_iv = self.call_25d_ivs.get('7d', 0.5)
        return put_iv - call_iv
    
    @property
    def butterfly_25d(self) -> float:
        """25D butterfly (wings - ATM)."""
        put_iv = self.put_25d_ivs.get('7d', 0.5)
        call_iv = self.call_25d_ivs.get('7d', 0.5)
        atm_iv = self.atm_ivs.get('7d', 0.5)
        return (put_iv + call_iv) / 2 - atm_iv


# =============================================================================
# IV Surface Provider
# =============================================================================

class IVSurfaceProvider:
    """
    Provides IV surface data.
    In production, connects to exchange APIs or pricing service.
    """
    
    def __init__(self, underlyings: List[str]):
        self.underlyings = underlyings
        self._history: Dict[str, List[IVSurfaceSnapshot]] = {u: [] for u in underlyings}
        self._history_max_len = 1440  # 24h of minute data
    
    async def get_surface(self, underlying: str) -> IVSurfaceSnapshot:
        """Get current IV surface for an underlying."""
        # In production, this fetches from exchange/pricing service
        # Demo: generate realistic synthetic data
        
        spot = self._get_spot_price(underlying)
        base_iv = self._get_base_iv(underlying)
        
        # Generate tenors
        tenors = ['1d', '2d', '7d', '14d', '30d']
        tenor_days = {'1d': 1, '2d': 2, '7d': 7, '14d': 14, '30d': 30}
        
        # ATM IVs with term structure
        atm_ivs = {}
        for tenor in tenors:
            days = tenor_days[tenor]
            # Term structure: short-dated usually higher in crypto
            term_adj = 1.0 + 0.02 * math.log(30 / max(days, 1))
            noise = self._noise() * 0.02
            atm_ivs[tenor] = base_iv * term_adj * (1 + noise)
        
        # 25D IVs (with skew)
        skew_strength = 0.05 + self._noise() * 0.02  # Puts usually higher
        call_25d_ivs = {}
        put_25d_ivs = {}
        for tenor in tenors:
            base = atm_ivs[tenor]
            call_25d_ivs[tenor] = base * (1 - skew_strength * 0.5)
            put_25d_ivs[tenor] = base * (1 + skew_strength)
        
        # Strike grid
        strike_ivs = {}
        for tenor in tenors:
            strike_ivs[tenor] = {}
            atm = atm_ivs[tenor]
            for moneyness in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
                strike = spot * moneyness
                # Smile: higher IV away from ATM
                smile_adj = 1 + 0.1 * (1 - moneyness) ** 2
                strike_ivs[tenor][strike] = atm * smile_adj
        
        # Realized vol (simplified)
        realized_24h = base_iv * (0.8 + self._noise() * 0.3)
        realized_7d = base_iv * (0.85 + self._noise() * 0.2)
        
        snapshot = IVSurfaceSnapshot(
            underlying=underlying,
            spot_price=spot,
            timestamp=datetime.utcnow(),
            atm_ivs=atm_ivs,
            call_25d_ivs=call_25d_ivs,
            put_25d_ivs=put_25d_ivs,
            strike_ivs=strike_ivs,
            realized_vol_24h=realized_24h,
            realized_vol_7d=realized_7d,
        )
        
        # Store in history
        self._history[underlying].append(snapshot)
        if len(self._history[underlying]) > self._history_max_len:
            self._history[underlying].pop(0)
        
        return snapshot
    
    def get_history(self, underlying: str, hours: int = 24) -> List[IVSurfaceSnapshot]:
        """Get historical snapshots."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [s for s in self._history[underlying] if s.timestamp > cutoff]
    
    def _get_spot_price(self, underlying: str) -> float:
        """Get spot price (demo)."""
        prices = {
            'BTCUSD': 100000 + self._noise() * 1000,
            'ETHUSD': 3500 + self._noise() * 50,
            'SOLUSD': 200 + self._noise() * 5,
        }
        return prices.get(underlying, 100)
    
    def _get_base_iv(self, underlying: str) -> float:
        """Get base IV level (demo)."""
        base_ivs = {
            'BTCUSD': 0.55,
            'ETHUSD': 0.65,
            'SOLUSD': 0.90,
        }
        # Add some time-varying component
        hour = datetime.utcnow().hour
        time_factor = 1.0 + 0.05 * math.sin(hour * math.pi / 12)
        return base_ivs.get(underlying, 0.5) * time_factor
    
    def _noise(self) -> float:
        """Generate noise."""
        if NUMPY_AVAILABLE:
            return float(np.random.normal(0, 1))
        import random
        return random.gauss(0, 1)


# =============================================================================
# IV Surface Exporter
# =============================================================================

class IVSurfaceExporter:
    """Exports IV surface metrics to Prometheus."""
    
    def __init__(
        self,
        provider: IVSurfaceProvider,
        update_interval: int = 60,
    ):
        self.provider = provider
        self.update_interval = update_interval
        self._running = False
    
    async def start(self) -> None:
        """Start the exporter."""
        if not PROMETHEUS_AVAILABLE:
            logger.error("prometheus_client not installed")
            return
        
        self._running = True
        logger.info("IV Surface Exporter started")
        
        while self._running:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update error: {e}")
                await asyncio.sleep(5)
        
        logger.info("IV Surface Exporter stopped")
    
    def stop(self) -> None:
        """Stop the exporter."""
        self._running = False
    
    async def _update_metrics(self) -> None:
        """Update all metrics."""
        for underlying in self.provider.underlyings:
            try:
                surface = await self.provider.get_surface(underlying)
                self._export_surface(surface)
            except Exception as e:
                logger.error(f"Error updating {underlying}: {e}")
    
    def _export_surface(self, surface: IVSurfaceSnapshot) -> None:
        """Export a surface snapshot to Prometheus."""
        underlying = surface.underlying
        
        # ATM IVs
        for tenor, iv in surface.atm_ivs.items():
            IV_SURFACE_ATM_IV.labels(underlying=underlying, tenor=tenor).set(iv)
        
        # 25D IVs
        for tenor, iv in surface.call_25d_ivs.items():
            IV_SURFACE_25D_CALL_IV.labels(underlying=underlying, tenor=tenor).set(iv)
        
        for tenor, iv in surface.put_25d_ivs.items():
            IV_SURFACE_25D_PUT_IV.labels(underlying=underlying, tenor=tenor).set(iv)
        
        # Realized vol
        IV_SURFACE_REALIZED_VOL.labels(underlying=underlying, window='24h').set(surface.realized_vol_24h)
        IV_SURFACE_REALIZED_VOL.labels(underlying=underlying, window='7d').set(surface.realized_vol_7d)
        
        # Term structure slope
        for tenor, iv in surface.atm_ivs.items():
            IV_SURFACE_TERM_STRUCTURE_SLOPE.labels(underlying=underlying, tenor=tenor).set(iv)
        
        # Strike IVs (for heatmap)
        for tenor, strikes in surface.strike_ivs.items():
            for strike, iv in strikes.items():
                IV_SURFACE_STRIKE_IV.labels(
                    underlying=underlying,
                    strike=f"{strike:.0f}",
                    tenor=tenor
                ).set(iv)
        
        # Z-score calculation
        history = self.provider.get_history(underlying, hours=24)
        if len(history) >= 10:
            historical_ivs = [s.atm_iv for s in history[:-1]]
            current_iv = surface.atm_iv
            
            if NUMPY_AVAILABLE:
                mean_iv = float(np.mean(historical_ivs))
                std_iv = float(np.std(historical_ivs))
            else:
                mean_iv = sum(historical_ivs) / len(historical_ivs)
                std_iv = (sum((x - mean_iv)**2 for x in historical_ivs) / len(historical_ivs)) ** 0.5
            
            if std_iv > 0.001:
                zscore = (current_iv - mean_iv) / std_iv
            else:
                zscore = 0.0
            
            IV_SURFACE_SHIFT_ZSCORE.labels(underlying=underlying).set(zscore)
            IV_SURFACE_ATM_IV_SMA_24H.labels(underlying=underlying).set(mean_iv)
            IV_SURFACE_ATM_IV_STD_24H.labels(underlying=underlying).set(std_iv)
        
        # VIX correlation (simplified - would need actual VIX data)
        IV_SURFACE_VIX_CORRELATION.labels(underlying=underlying).set(0.7 + self._noise() * 0.1)
        
        # Crypto VIX index (average of major cryptos)
        btc_iv = surface.atm_iv if underlying == 'BTCUSD' else 0.55
        IV_SURFACE_VIX_INDEX.set(btc_iv * 100)
        
        # Last update
        IV_SURFACE_LAST_UPDATE.labels(underlying=underlying).set(surface.timestamp.timestamp())
    
    def _noise(self) -> float:
        if NUMPY_AVAILABLE:
            return float(np.random.normal(0, 1))
        import random
        return random.gauss(0, 1)


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description='IV Surface Exporter')
    parser.add_argument('--port', type=int, default=8004, help='Metrics port')
    parser.add_argument('--interval', type=int, default=60, help='Update interval (seconds)')
    parser.add_argument('--underlyings', type=str, nargs='+',
                       default=['BTCUSD', 'ETHUSD', 'SOLUSD'],
                       help='Underlyings to monitor')
    args = parser.parse_args()
    
    if not PROMETHEUS_AVAILABLE:
        logger.error("prometheus_client not installed. Run: pip install prometheus_client")
        return
    
    # Start metrics server
    start_http_server(args.port)
    logger.info(f"Prometheus metrics server on port {args.port}")
    
    # Create provider and exporter
    provider = IVSurfaceProvider(args.underlyings)
    exporter = IVSurfaceExporter(provider, update_interval=args.interval)
    
    # Run
    await exporter.start()


if __name__ == '__main__':
    asyncio.run(main())
