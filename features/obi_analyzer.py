"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         ORDER BOOK IMBALANCE (OBI) ANALYZER                                   ║
║                                                                               ║
║  Measures the imbalance between bid and ask sides of the order book           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

OBI is a key microstructure signal that indicates buying/selling pressure.
Positive OBI suggests more bid pressure (bullish).
Negative OBI suggests more ask pressure (bearish).

Features:
- Multi-level OBI calculation
- Time-weighted OBI
- OBI momentum
- Normalized OBI
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger("Features.OBI")


@dataclass
class OBISnapshot:
    """Single OBI measurement."""
    timestamp: datetime
    obi_level1: float      # Top of book imbalance
    obi_level5: float      # Top 5 levels imbalance
    obi_level10: float     # Top 10 levels imbalance
    obi_weighted: float    # Volume-weighted imbalance
    bid_depth_total: float
    ask_depth_total: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'obi_level1': self.obi_level1,
            'obi_level5': self.obi_level5,
            'obi_level10': self.obi_level10,
            'obi_weighted': self.obi_weighted,
            'bid_depth_total': self.bid_depth_total,
            'ask_depth_total': self.ask_depth_total,
        }


@dataclass
class OBISignal:
    """OBI-based trading signal."""
    obi_current: float
    obi_ema: float
    obi_momentum: float
    obi_zscore: float
    signal_strength: float  # -1 to 1
    signal_direction: str   # 'bullish', 'bearish', 'neutral'
    confidence: float


class OrderBookImbalance:
    """
    Order Book Imbalance Analyzer.
    
    Calculates various OBI metrics from order book data to measure
    the balance of buying vs selling pressure.
    
    Usage:
        obi = OrderBookImbalance()
        
        # Update with order book data
        obi.update(bids, asks)
        
        # Get signal
        signal = obi.get_signal()
        print(f"OBI Signal: {signal.signal_direction} ({signal.signal_strength:.2f})")
    """
    
    def __init__(
        self,
        ema_span: int = 20,
        zscore_window: int = 100,
        levels_to_track: List[int] = [1, 5, 10, 20],
    ):
        """
        Initialize OBI analyzer.
        
        Args:
            ema_span: Span for EMA smoothing
            zscore_window: Window for z-score calculation
            levels_to_track: Order book levels to analyze
        """
        self.ema_span = ema_span
        self.zscore_window = zscore_window
        self.levels_to_track = levels_to_track
        
        # History
        self._history: deque = deque(maxlen=1000)
        self._obi_values: deque = deque(maxlen=zscore_window)
        
        # EMA state
        self._ema_alpha = 2 / (ema_span + 1)
        self._obi_ema: Optional[float] = None
        
        # Current state
        self._current_snapshot: Optional[OBISnapshot] = None
        
        logger.info(f"OBI analyzer initialized with levels: {levels_to_track}")
    
    def update(
        self,
        bids: List[Tuple[float, float]],  # [(price, size), ...]
        asks: List[Tuple[float, float]],
        timestamp: Optional[datetime] = None,
    ) -> OBISnapshot:
        """
        Update OBI with new order book data.
        
        Args:
            bids: List of (price, size) tuples, sorted descending by price
            asks: List of (price, size) tuples, sorted ascending by price
            timestamp: Optional timestamp
            
        Returns:
            OBISnapshot with calculated metrics
        """
        ts = timestamp or datetime.now()
        
        # Calculate OBI at different levels
        obi_level1 = self._calculate_obi(bids[:1], asks[:1])
        obi_level5 = self._calculate_obi(bids[:5], asks[:5])
        obi_level10 = self._calculate_obi(bids[:10], asks[:10])
        
        # Volume-weighted OBI (weights by distance from mid)
        obi_weighted = self._calculate_weighted_obi(bids[:20], asks[:20])
        
        # Total depths
        bid_depth = sum(size for _, size in bids)
        ask_depth = sum(size for _, size in asks)
        
        snapshot = OBISnapshot(
            timestamp=ts,
            obi_level1=obi_level1,
            obi_level5=obi_level5,
            obi_level10=obi_level10,
            obi_weighted=obi_weighted,
            bid_depth_total=bid_depth,
            ask_depth_total=ask_depth,
        )
        
        # Update history and EMA
        self._history.append(snapshot)
        self._obi_values.append(obi_level5)  # Use level5 as primary
        
        if self._obi_ema is None:
            self._obi_ema = obi_level5
        else:
            self._obi_ema = self._ema_alpha * obi_level5 + (1 - self._ema_alpha) * self._obi_ema
        
        self._current_snapshot = snapshot
        
        return snapshot
    
    def _calculate_obi(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> float:
        """
        Calculate Order Book Imbalance.
        
        OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Returns value in [-1, 1]:
        - Positive: more bid pressure (bullish)
        - Negative: more ask pressure (bearish)
        """
        bid_vol = sum(size for _, size in bids) if bids else 0
        ask_vol = sum(size for _, size in asks) if asks else 0
        
        total = bid_vol + ask_vol
        if total < 1e-10:
            return 0.0
        
        return (bid_vol - ask_vol) / total
    
    def _calculate_weighted_obi(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> float:
        """
        Calculate volume-weighted OBI.
        
        Weights levels by inverse distance from mid price.
        """
        if not bids or not asks:
            return 0.0
        
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        # Weighted bid volume
        weighted_bid = 0.0
        for i, (price, size) in enumerate(bids):
            distance = abs(mid_price - price) / mid_price
            weight = 1 / (1 + distance * 100)  # Closer = higher weight
            weighted_bid += size * weight
        
        # Weighted ask volume
        weighted_ask = 0.0
        for i, (price, size) in enumerate(asks):
            distance = abs(price - mid_price) / mid_price
            weight = 1 / (1 + distance * 100)
            weighted_ask += size * weight
        
        total = weighted_bid + weighted_ask
        if total < 1e-10:
            return 0.0
        
        return (weighted_bid - weighted_ask) / total
    
    def get_signal(self) -> OBISignal:
        """
        Get OBI-based trading signal.
        
        Returns:
            OBISignal with signal strength and direction
        """
        if not self._current_snapshot or len(self._obi_values) < 2:
            return OBISignal(
                obi_current=0,
                obi_ema=0,
                obi_momentum=0,
                obi_zscore=0,
                signal_strength=0,
                signal_direction='neutral',
                confidence=0,
            )
        
        current = self._current_snapshot.obi_level5
        ema = self._obi_ema or current
        
        # Calculate momentum (rate of change)
        if len(self._obi_values) >= 5:
            recent = list(self._obi_values)[-5:]
            momentum = recent[-1] - recent[0]
        else:
            momentum = 0
        
        # Calculate z-score
        values = list(self._obi_values)
        mean = np.mean(values)
        std = np.std(values)
        zscore = (current - mean) / (std + 1e-10)
        
        # Determine signal
        # Combine current OBI, EMA crossover, and z-score
        signal_strength = 0.0
        
        # Current OBI component (40%)
        signal_strength += current * 0.4
        
        # EMA crossover component (30%)
        crossover = current - ema
        signal_strength += crossover * 0.3
        
        # Z-score component (30%) - mean reversion signal
        # Extreme z-scores suggest potential reversal
        zscore_signal = np.tanh(zscore * 0.3)  # Bounded
        signal_strength += zscore_signal * 0.3
        
        # Clamp to [-1, 1]
        signal_strength = max(-1, min(1, signal_strength))
        
        # Determine direction
        if signal_strength > 0.1:
            direction = 'bullish'
        elif signal_strength < -0.1:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Confidence based on consistency
        confidence = abs(signal_strength) * (1 - abs(zscore) / 5)
        confidence = max(0, min(1, confidence))
        
        return OBISignal(
            obi_current=current,
            obi_ema=ema,
            obi_momentum=momentum,
            obi_zscore=zscore,
            signal_strength=signal_strength,
            signal_direction=direction,
            confidence=confidence,
        )
    
    def get_features(self) -> Dict[str, float]:
        """
        Get OBI features for ML models.
        
        Returns:
            Dictionary of feature name -> value
        """
        signal = self.get_signal()
        snapshot = self._current_snapshot
        
        if not snapshot:
            return {
                'obi_level1': 0,
                'obi_level5': 0,
                'obi_level10': 0,
                'obi_weighted': 0,
                'obi_ema': 0,
                'obi_momentum': 0,
                'obi_zscore': 0,
                'obi_signal': 0,
            }
        
        return {
            'obi_level1': snapshot.obi_level1,
            'obi_level5': snapshot.obi_level5,
            'obi_level10': snapshot.obi_level10,
            'obi_weighted': snapshot.obi_weighted,
            'obi_ema': signal.obi_ema,
            'obi_momentum': signal.obi_momentum,
            'obi_zscore': signal.obi_zscore,
            'obi_signal': signal.signal_strength,
        }
    
    def get_history(self, n: int = 100) -> List[Dict]:
        """Get recent OBI history."""
        return [s.to_dict() for s in list(self._history)[-n:]]
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self._history.clear()
        self._obi_values.clear()
        self._obi_ema = None
        self._current_snapshot = None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer
    obi = OrderBookImbalance()
    
    # Simulate order book updates
    np.random.seed(42)
    
    for i in range(100):
        mid = 50000 + i * 10
        
        # Generate random order book
        bids = [(mid - j * 5 - np.random.uniform(0, 2), np.random.uniform(0.1, 2.0)) 
                for j in range(20)]
        asks = [(mid + j * 5 + np.random.uniform(0, 2), np.random.uniform(0.1, 2.0)) 
                for j in range(20)]
        
        # Add some imbalance trend
        if i > 50:
            # More bid pressure
            bids = [(p, s * 1.5) for p, s in bids]
        
        snapshot = obi.update(bids, asks)
        
        if (i + 1) % 20 == 0:
            signal = obi.get_signal()
            print(f"Step {i+1}:")
            print(f"  OBI L5: {snapshot.obi_level5:.3f}")
            print(f"  Signal: {signal.signal_direction} ({signal.signal_strength:.3f})")
            print(f"  Confidence: {signal.confidence:.2f}")
    
    # Get features
    print("\nFeatures for ML:")
    for name, value in obi.get_features().items():
        print(f"  {name}: {value:.4f}")
