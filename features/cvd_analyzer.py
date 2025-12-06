"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         CUMULATIVE VOLUME DELTA (CVD) ANALYZER                                ║
║                                                                               ║
║  Tracks the cumulative difference between buy and sell volume                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CVD is a key order flow indicator that shows the net buying/selling pressure
over time. Rising CVD with rising price confirms bullish trend.
Divergences between price and CVD can signal reversals.

Features:
- Real-time CVD calculation
- CVD momentum and divergence detection
- Multi-timeframe CVD
- Normalized CVD
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("Features.CVD")


@dataclass
class TradeData:
    """Single trade for CVD calculation."""
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'


@dataclass
class CVDSnapshot:
    """CVD measurement at a point in time."""
    timestamp: datetime
    cvd: float
    cvd_delta: float  # Change since last snapshot
    buy_volume: float
    sell_volume: float
    net_volume: float
    trade_count: int


@dataclass
class CVDSignal:
    """CVD-based trading signal."""
    cvd_current: float
    cvd_ema: float
    cvd_momentum: float
    cvd_zscore: float
    price_cvd_divergence: float  # Positive = bullish divergence
    signal_strength: float
    signal_direction: str
    confidence: float


class CVDAnalyzer:
    """
    Cumulative Volume Delta Analyzer.
    
    Tracks the running sum of (buy_volume - sell_volume) to measure
    net order flow pressure over time.
    
    Usage:
        cvd = CVDAnalyzer()
        
        # Process trades
        for trade in trades:
            cvd.add_trade(trade.price, trade.size, trade.side)
        
        # Get signal
        signal = cvd.get_signal(current_price)
        print(f"CVD Signal: {signal.signal_direction}")
    """
    
    def __init__(
        self,
        ema_span: int = 50,
        zscore_window: int = 200,
        snapshot_interval_seconds: int = 60,
    ):
        """
        Initialize CVD analyzer.
        
        Args:
            ema_span: Span for EMA smoothing
            zscore_window: Window for z-score calculation
            snapshot_interval_seconds: Interval for snapshots
        """
        self.ema_span = ema_span
        self.zscore_window = zscore_window
        self.snapshot_interval = timedelta(seconds=snapshot_interval_seconds)
        
        # CVD state
        self._cvd: float = 0.0
        self._buy_volume: float = 0.0
        self._sell_volume: float = 0.0
        self._trade_count: int = 0
        
        # History
        self._snapshots: deque = deque(maxlen=1000)
        self._cvd_values: deque = deque(maxlen=zscore_window)
        self._price_history: deque = deque(maxlen=zscore_window)
        self._trades: deque = deque(maxlen=10000)
        
        # EMA state
        self._ema_alpha = 2 / (ema_span + 1)
        self._cvd_ema: Optional[float] = None
        
        # Last snapshot time
        self._last_snapshot_time: Optional[datetime] = None
        
        logger.info("CVD analyzer initialized")
    
    def add_trade(
        self,
        price: float,
        size: float,
        side: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Add a trade to CVD calculation.
        
        Args:
            price: Trade price
            size: Trade size
            side: 'buy' or 'sell'
            timestamp: Trade timestamp
        """
        ts = timestamp or datetime.now()
        
        trade = TradeData(
            timestamp=ts,
            price=price,
            size=size,
            side=side.lower(),
        )
        self._trades.append(trade)
        
        # Update CVD
        if side.lower() == 'buy':
            self._cvd += size
            self._buy_volume += size
        else:
            self._cvd -= size
            self._sell_volume += size
        
        self._trade_count += 1
        
        # Check if we should create a snapshot
        if (self._last_snapshot_time is None or 
            ts - self._last_snapshot_time >= self.snapshot_interval):
            self._create_snapshot(ts)
    
    def add_trades_batch(
        self,
        trades: List[Tuple[float, float, str]],  # [(price, size, side), ...]
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add multiple trades at once."""
        ts = timestamp or datetime.now()
        for price, size, side in trades:
            self.add_trade(price, size, side, ts)
    
    def infer_side(
        self,
        trade_price: float,
        bid_price: float,
        ask_price: float,
    ) -> str:
        """
        Infer trade side from price relative to bid/ask.
        
        Simple tick rule: trades at ask are buys, at bid are sells.
        """
        mid = (bid_price + ask_price) / 2
        if trade_price >= mid:
            return 'buy'
        else:
            return 'sell'
    
    def _create_snapshot(self, timestamp: datetime) -> CVDSnapshot:
        """Create a CVD snapshot."""
        last_cvd = self._snapshots[-1].cvd if self._snapshots else 0
        
        snapshot = CVDSnapshot(
            timestamp=timestamp,
            cvd=self._cvd,
            cvd_delta=self._cvd - last_cvd,
            buy_volume=self._buy_volume,
            sell_volume=self._sell_volume,
            net_volume=self._buy_volume - self._sell_volume,
            trade_count=self._trade_count,
        )
        
        self._snapshots.append(snapshot)
        self._cvd_values.append(self._cvd)
        self._last_snapshot_time = timestamp
        
        # Update EMA
        if self._cvd_ema is None:
            self._cvd_ema = self._cvd
        else:
            self._cvd_ema = self._ema_alpha * self._cvd + (1 - self._ema_alpha) * self._cvd_ema
        
        return snapshot
    
    def update_price(self, price: float) -> None:
        """Update current price for divergence detection."""
        self._price_history.append(price)
    
    def get_signal(self, current_price: Optional[float] = None) -> CVDSignal:
        """
        Get CVD-based trading signal.
        
        Args:
            current_price: Current market price for divergence detection
            
        Returns:
            CVDSignal with signal strength and direction
        """
        if len(self._cvd_values) < 2:
            return CVDSignal(
                cvd_current=self._cvd,
                cvd_ema=self._cvd,
                cvd_momentum=0,
                cvd_zscore=0,
                price_cvd_divergence=0,
                signal_strength=0,
                signal_direction='neutral',
                confidence=0,
            )
        
        cvd = self._cvd
        ema = self._cvd_ema or cvd
        
        # Calculate momentum
        values = list(self._cvd_values)
        if len(values) >= 10:
            momentum = values[-1] - values[-10]
        else:
            momentum = values[-1] - values[0]
        
        # Calculate z-score
        mean = np.mean(values)
        std = np.std(values)
        zscore = (cvd - mean) / (std + 1e-10)
        
        # Calculate price-CVD divergence
        divergence = 0.0
        if current_price and len(self._price_history) >= 10:
            prices = list(self._price_history)[-10:]
            cvds = list(self._cvd_values)[-10:]
            
            price_change = (prices[-1] - prices[0]) / (prices[0] + 1e-10)
            cvd_change = (cvds[-1] - cvds[0]) / (abs(cvds[0]) + 1e-10)
            
            # Bullish divergence: price falling but CVD rising
            # Bearish divergence: price rising but CVD falling
            if price_change < 0 and cvd_change > 0:
                divergence = abs(cvd_change - price_change)  # Bullish
            elif price_change > 0 and cvd_change < 0:
                divergence = -abs(cvd_change - price_change)  # Bearish
        
        # Calculate signal strength
        signal_strength = 0.0
        
        # CVD trend component (40%)
        cvd_normalized = np.tanh(momentum / (std + 1e-10))
        signal_strength += cvd_normalized * 0.4
        
        # EMA crossover (30%)
        crossover = cvd - ema
        crossover_normalized = np.tanh(crossover / (std + 1e-10))
        signal_strength += crossover_normalized * 0.3
        
        # Divergence component (30%)
        signal_strength += np.tanh(divergence) * 0.3
        
        # Clamp
        signal_strength = max(-1, min(1, signal_strength))
        
        # Direction
        if signal_strength > 0.1:
            direction = 'bullish'
        elif signal_strength < -0.1:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Confidence
        confidence = abs(signal_strength) * (1 - abs(zscore) / 5)
        confidence = max(0, min(1, confidence))
        
        return CVDSignal(
            cvd_current=cvd,
            cvd_ema=ema,
            cvd_momentum=momentum,
            cvd_zscore=zscore,
            price_cvd_divergence=divergence,
            signal_strength=signal_strength,
            signal_direction=direction,
            confidence=confidence,
        )
    
    def get_features(self) -> Dict[str, float]:
        """Get CVD features for ML models."""
        signal = self.get_signal()
        
        return {
            'cvd': self._cvd,
            'cvd_ema': signal.cvd_ema,
            'cvd_momentum': signal.cvd_momentum,
            'cvd_zscore': signal.cvd_zscore,
            'cvd_divergence': signal.price_cvd_divergence,
            'cvd_signal': signal.signal_strength,
            'buy_volume': self._buy_volume,
            'sell_volume': self._sell_volume,
            'buy_sell_ratio': self._buy_volume / (self._sell_volume + 1e-10),
        }
    
    def get_multi_timeframe(self) -> Dict[str, float]:
        """
        Get CVD values at multiple timeframes.
        
        Returns:
            Dict with CVD at 1m, 5m, 15m, 1h windows
        """
        now = datetime.now()
        result = {}
        
        timeframes = [
            ('1m', timedelta(minutes=1)),
            ('5m', timedelta(minutes=5)),
            ('15m', timedelta(minutes=15)),
            ('1h', timedelta(hours=1)),
        ]
        
        for name, window in timeframes:
            cutoff = now - window
            window_trades = [t for t in self._trades if t.timestamp >= cutoff]
            
            buy_vol = sum(t.size for t in window_trades if t.side == 'buy')
            sell_vol = sum(t.size for t in window_trades if t.side == 'sell')
            
            result[f'cvd_{name}'] = buy_vol - sell_vol
            result[f'volume_{name}'] = buy_vol + sell_vol
        
        return result
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self._cvd = 0.0
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._trade_count = 0
        self._snapshots.clear()
        self._cvd_values.clear()
        self._price_history.clear()
        self._trades.clear()
        self._cvd_ema = None
        self._last_snapshot_time = None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer
    cvd = CVDAnalyzer(snapshot_interval_seconds=1)
    
    # Simulate trades
    np.random.seed(42)
    price = 50000
    
    for i in range(500):
        # Random trade
        size = np.random.uniform(0.01, 1.0)
        
        # Bias toward buying
        if i > 250:
            side = 'buy' if np.random.random() < 0.6 else 'sell'
        else:
            side = 'buy' if np.random.random() < 0.5 else 'sell'
        
        # Price random walk
        price += np.random.randn() * 10
        
        cvd.add_trade(price, size, side)
        cvd.update_price(price)
        
        if (i + 1) % 100 == 0:
            signal = cvd.get_signal(price)
            print(f"Trade {i+1}:")
            print(f"  CVD: {signal.cvd_current:.2f}")
            print(f"  Signal: {signal.signal_direction} ({signal.signal_strength:.3f})")
    
    # Get features
    print("\nFeatures for ML:")
    for name, value in cvd.get_features().items():
        print(f"  {name}: {value:.4f}")
    
    # Multi-timeframe
    print("\nMulti-timeframe CVD:")
    for name, value in cvd.get_multi_timeframe().items():
        print(f"  {name}: {value:.4f}")
