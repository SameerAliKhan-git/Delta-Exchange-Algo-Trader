"""
Strategy 3: Market Microstructure / Orderflow (HFT Edge)
========================================================

Why it works in crypto:
- Predictable order book imbalance
- Slow-moving retail orderbooks
- Inefficient market makers
- Hidden liquidity patterns

LOB-based ML models outperform OHLCV dramatically on crypto.

Concepts:
- Order book imbalance (OBI)
- Spread dynamics  
- Queue imbalance
- Whale detection
- Cumulative volume delta (CVD)

This is how many HFT crypto firms make money.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from .base import (
    BaseStrategy, StrategyConfig, Signal, SignalType,
    TechnicalIndicators
)


@dataclass
class MicrostructureConfig(StrategyConfig):
    """Configuration for microstructure strategies."""
    name: str = "microstructure"
    
    # Order book depth levels
    ob_levels: int = 10
    
    # Imbalance thresholds
    obi_threshold: float = 0.3  # 30% imbalance triggers signal
    obi_strong_threshold: float = 0.5  # 50% = strong signal
    
    # CVD settings
    cvd_lookback: int = 20
    cvd_threshold: float = 0.7  # Standardized CVD threshold
    
    # Spread analysis
    spread_lookback: int = 50
    spread_expansion_mult: float = 2.0  # Spread > 2x average
    
    # Whale detection
    whale_threshold_usd: float = 100000  # Orders > $100k
    whale_lookback: int = 10
    
    # Signal timing
    hold_seconds: int = 5  # Micro-holding period
    min_edge: float = 0.0001  # 1 bps minimum edge
    
    # Filters
    min_liquidity: float = 10000  # Min USD on each side
    max_spread_bps: float = 10  # Max 10 bps spread to trade
    
    # Regime
    allowed_regimes: List[str] = field(default_factory=lambda: ["all"])


@dataclass
class OrderBookSnapshot:
    """Single order book snapshot."""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    bid_total: float = 0.0
    ask_total: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0
    imbalance: float = 0.0
    
    def __post_init__(self):
        if self.bids and self.asks:
            self.bid_total = sum(size for _, size in self.bids)
            self.ask_total = sum(size for _, size in self.asks)
            best_bid = self.bids[0][0] if self.bids else 0
            best_ask = self.asks[0][0] if self.asks else 0
            self.mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            self.spread = best_ask - best_bid if best_bid and best_ask else 0
            total = self.bid_total + self.ask_total
            self.imbalance = (self.bid_total - self.ask_total) / total if total > 0 else 0


class OrderBookImbalance:
    """
    Order Book Imbalance (OBI) analyzer.
    
    OBI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
    
    Positive OBI = buying pressure = price likely to rise
    Negative OBI = selling pressure = price likely to fall
    
    This is one of the strongest short-term predictors in crypto.
    """
    
    def __init__(self, levels: int = 10, window: int = 50):
        """
        Initialize OBI analyzer.
        
        Args:
            levels: Number of order book levels to consider
            window: Rolling window for statistics
        """
        self.levels = levels
        self.window = window
        
        self._history: deque = deque(maxlen=window)
        self._obi_series: List[float] = []
    
    def update(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """
        Update with new order book snapshot.
        
        Returns current OBI metrics.
        """
        self._history.append(snapshot)
        self._obi_series.append(snapshot.imbalance)
        
        # Keep series bounded
        if len(self._obi_series) > self.window * 2:
            self._obi_series = self._obi_series[-self.window:]
        
        # Calculate metrics
        obi_array = np.array(self._obi_series[-self.window:])
        
        return {
            'obi': snapshot.imbalance,
            'obi_ma': np.mean(obi_array),
            'obi_std': np.std(obi_array),
            'obi_zscore': (snapshot.imbalance - np.mean(obi_array)) / (np.std(obi_array) + 1e-10),
            'obi_momentum': snapshot.imbalance - obi_array[-2] if len(obi_array) > 1 else 0,
            'bid_depth': snapshot.bid_total,
            'ask_depth': snapshot.ask_total,
            'spread_bps': snapshot.spread / snapshot.mid_price * 10000 if snapshot.mid_price > 0 else 0
        }
    
    def get_weighted_obi(self, snapshot: OrderBookSnapshot) -> float:
        """
        Calculate price-weighted OBI.
        
        Weights orders closer to mid-price more heavily.
        """
        if not snapshot.bids or not snapshot.asks:
            return 0.0
        
        mid = snapshot.mid_price
        
        weighted_bid = 0
        weighted_ask = 0
        
        for price, size in snapshot.bids[:self.levels]:
            distance = (mid - price) / mid
            weight = 1 / (1 + distance * 100)  # Decay with distance
            weighted_bid += size * weight
        
        for price, size in snapshot.asks[:self.levels]:
            distance = (price - mid) / mid
            weight = 1 / (1 + distance * 100)
            weighted_ask += size * weight
        
        total = weighted_bid + weighted_ask
        return (weighted_bid - weighted_ask) / total if total > 0 else 0
    
    def get_depth_imbalance_by_level(self, snapshot: OrderBookSnapshot) -> np.ndarray:
        """
        Get imbalance at each price level.
        
        Useful for ML features.
        """
        imbalances = []
        
        for i in range(min(self.levels, len(snapshot.bids), len(snapshot.asks))):
            bid_size = snapshot.bids[i][1]
            ask_size = snapshot.asks[i][1]
            total = bid_size + ask_size
            imb = (bid_size - ask_size) / total if total > 0 else 0
            imbalances.append(imb)
        
        return np.array(imbalances)


class CVDAnalyzer:
    """
    Cumulative Volume Delta analyzer.
    
    CVD = Cumulative(Buy Volume - Sell Volume)
    
    Tracks the battle between buyers and sellers.
    Divergences between CVD and price are powerful signals.
    """
    
    def __init__(self, lookback: int = 100):
        """
        Initialize CVD analyzer.
        
        Args:
            lookback: Period for analysis
        """
        self.lookback = lookback
        
        self._cvd: float = 0.0
        self._cvd_history: deque = deque(maxlen=lookback)
        self._price_history: deque = deque(maxlen=lookback)
        self._delta_history: deque = deque(maxlen=lookback)
    
    def update(self, buy_volume: float, sell_volume: float, price: float) -> Dict[str, float]:
        """
        Update CVD with new trade data.
        
        Args:
            buy_volume: Buy-side volume
            sell_volume: Sell-side volume
            price: Current price
        
        Returns CVD metrics.
        """
        delta = buy_volume - sell_volume
        self._cvd += delta
        
        self._cvd_history.append(self._cvd)
        self._price_history.append(price)
        self._delta_history.append(delta)
        
        cvd_array = np.array(self._cvd_history)
        price_array = np.array(self._price_history)
        delta_array = np.array(self._delta_history)
        
        # Normalize CVD for comparison
        cvd_normalized = (cvd_array - np.mean(cvd_array)) / (np.std(cvd_array) + 1e-10)
        price_normalized = (price_array - np.mean(price_array)) / (np.std(price_array) + 1e-10)
        
        # Divergence: CVD going up while price flat/down = bullish
        if len(cvd_array) > 10:
            cvd_trend = np.polyfit(range(10), cvd_array[-10:], 1)[0]
            price_trend = np.polyfit(range(10), price_array[-10:], 1)[0]
            divergence = cvd_trend - price_trend
        else:
            divergence = 0
        
        return {
            'cvd': self._cvd,
            'cvd_normalized': cvd_normalized[-1] if len(cvd_normalized) > 0 else 0,
            'delta': delta,
            'delta_ma': np.mean(delta_array) if len(delta_array) > 0 else 0,
            'divergence': divergence,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'volume_ratio': buy_volume / sell_volume if sell_volume > 0 else 1.0
        }
    
    def detect_absorption(self) -> Optional[str]:
        """
        Detect absorption patterns.
        
        Absorption = large volume traded but price doesn't move
        """
        if len(self._delta_history) < 10:
            return None
        
        recent_delta = sum(list(self._delta_history)[-10:])
        price_change = self._price_history[-1] - self._price_history[-10]
        price_change_pct = price_change / self._price_history[-10]
        
        # High delta but low price change = absorption
        if abs(recent_delta) > np.std(list(self._delta_history)) * 2:
            if abs(price_change_pct) < 0.001:  # Less than 0.1% price change
                if recent_delta > 0:
                    return "ask_absorption"  # Buyers absorbed by sellers
                else:
                    return "bid_absorption"  # Sellers absorbed by buyers
        
        return None


class WhaleDetector:
    """
    Whale order detection.
    
    Detects large orders that can move the market.
    """
    
    def __init__(self, threshold_usd: float = 100000, lookback: int = 100):
        """
        Initialize whale detector.
        
        Args:
            threshold_usd: USD threshold for whale orders
            lookback: History length
        """
        self.threshold_usd = threshold_usd
        self.lookback = lookback
        
        self._whale_orders: deque = deque(maxlen=lookback)
        self._recent_whales: List[Dict] = []
    
    def check_order(self, price: float, size: float, side: str) -> Optional[Dict]:
        """
        Check if order is a whale order.
        
        Returns whale info if detected.
        """
        usd_value = price * size
        
        if usd_value >= self.threshold_usd:
            whale = {
                'timestamp': datetime.now(),
                'price': price,
                'size': size,
                'side': side,
                'usd_value': usd_value
            }
            self._whale_orders.append(whale)
            self._recent_whales.append(whale)
            
            # Clean old whales
            cutoff = datetime.now().timestamp() - 300  # 5 min
            self._recent_whales = [
                w for w in self._recent_whales 
                if w['timestamp'].timestamp() > cutoff
            ]
            
            return whale
        
        return None
    
    def get_whale_pressure(self) -> Dict[str, float]:
        """Get net whale buying/selling pressure."""
        buy_volume = 0
        sell_volume = 0
        
        for whale in self._recent_whales:
            if whale['side'] == 'buy':
                buy_volume += whale['usd_value']
            else:
                sell_volume += whale['usd_value']
        
        total = buy_volume + sell_volume
        
        return {
            'whale_buy_volume': buy_volume,
            'whale_sell_volume': sell_volume,
            'whale_imbalance': (buy_volume - sell_volume) / total if total > 0 else 0,
            'whale_count': len(self._recent_whales)
        }
    
    def detect_iceberg(self, trades: List[Dict]) -> Optional[Dict]:
        """
        Detect iceberg orders (large hidden orders).
        
        Pattern: Multiple trades at same price in quick succession
        """
        if len(trades) < 5:
            return None
        
        # Group recent trades by price
        price_groups = {}
        for trade in trades[-20:]:
            price = round(trade['price'], 2)
            if price not in price_groups:
                price_groups[price] = []
            price_groups[price].append(trade)
        
        # Check for iceberg pattern
        for price, price_trades in price_groups.items():
            if len(price_trades) >= 5:
                total_size = sum(t['size'] for t in price_trades)
                if total_size * price >= self.threshold_usd:
                    sides = [t['side'] for t in price_trades]
                    dominant_side = max(set(sides), key=sides.count)
                    
                    return {
                        'type': 'iceberg',
                        'price': price,
                        'total_size': total_size,
                        'side': dominant_side,
                        'trade_count': len(price_trades)
                    }
        
        return None


class SpreadAnalyzer:
    """
    Bid-ask spread analyzer.
    
    Spread dynamics reveal market maker behavior and liquidity conditions.
    """
    
    def __init__(self, lookback: int = 100):
        """Initialize spread analyzer."""
        self.lookback = lookback
        
        self._spread_history: deque = deque(maxlen=lookback)
        self._mid_history: deque = deque(maxlen=lookback)
    
    def update(self, bid: float, ask: float) -> Dict[str, float]:
        """
        Update with new bid/ask.
        
        Returns spread metrics.
        """
        spread = ask - bid
        mid = (bid + ask) / 2
        spread_bps = spread / mid * 10000
        
        self._spread_history.append(spread_bps)
        self._mid_history.append(mid)
        
        spread_array = np.array(self._spread_history)
        
        return {
            'spread_bps': spread_bps,
            'spread_ma': np.mean(spread_array),
            'spread_std': np.std(spread_array),
            'spread_zscore': (spread_bps - np.mean(spread_array)) / (np.std(spread_array) + 1e-10),
            'spread_percentile': np.percentile(spread_array, (spread_array <= spread_bps).sum() / len(spread_array) * 100) if len(spread_array) > 10 else 50,
            'is_wide': spread_bps > np.mean(spread_array) + 2 * np.std(spread_array),
            'is_tight': spread_bps < np.mean(spread_array) - np.std(spread_array)
        }


class FootprintAnalyzer:
    """
    Footprint / Delta Analyzer.
    
    Tracks buying vs selling pressure within each candle.
    """
    def __init__(self, window: int = 20):
        self.window = window
        self.deltas = deque(maxlen=window)
        self.cumulative_delta = 0.0
        
    def update(self, buy_vol: float, sell_vol: float) -> Dict[str, float]:
        delta = buy_vol - sell_vol
        self.cumulative_delta += delta
        self.deltas.append(delta)
        
        deltas_arr = np.array(self.deltas)
        
        return {
            'bar_delta': delta,
            'cumulative_delta': self.cumulative_delta,
            'delta_zscore': (delta - np.mean(deltas_arr)) / (np.std(deltas_arr) + 1e-10) if len(deltas_arr) > 1 else 0,
            'is_buying_pressure': delta > 0 and delta > np.std(deltas_arr),
            'is_selling_pressure': delta < 0 and abs(delta) > np.std(deltas_arr)
        }


class VolumeProfileAnalyzer:
    """
    Volume Profile Analyzer.
    
    Identifies High Volume Nodes (HVN) and Low Volume Nodes (LVN).
    """
    def __init__(self, n_bins: int = 50, decay: float = 0.99):
        self.n_bins = n_bins
        self.decay = decay
        self.profile = {}  # price_bin -> volume
        self.poc = 0.0  # Point of Control
        self.vah = 0.0  # Value Area High
        self.val = 0.0  # Value Area Low
        
    def update(self, price: float, volume: float):
        # Decay old volume
        for k in self.profile:
            self.profile[k] *= self.decay
            
        # Add new volume
        bin_size = price * 0.001  # 0.1% bins
        price_bin = round(price / bin_size) * bin_size
        self.profile[price_bin] = self.profile.get(price_bin, 0) + volume
        
        # Recalculate POC/VA
        if not self.profile:
            return
            
        sorted_bins = sorted(self.profile.items(), key=lambda x: x[1], reverse=True)
        self.poc = sorted_bins[0][0]
        
        # Calculate Value Area (70% of volume)
        total_vol = sum(self.profile.values())
        target_vol = total_vol * 0.70
        current_vol = 0
        va_bins = []
        
        for p, v in sorted_bins:
            current_vol += v
            va_bins.append(p)
            if current_vol >= target_vol:
                break
                
        self.vah = max(va_bins) if va_bins else price
        self.val = min(va_bins) if va_bins else price
        
    def get_location(self, price: float) -> str:
        if price > self.vah: return "above_va"
        if price < self.val: return "below_va"
        return "inside_va"


class LiquidityAnalyzer:
    """
    Liquidity / Heatmap Analyzer.
    
    Detects buy/sell walls in the order book.
    """
    def __init__(self, threshold_ratio: float = 3.0):
        self.threshold_ratio = threshold_ratio
        
    def analyze(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        if not snapshot.bids or not snapshot.asks:
            return {}
            
        # Calculate average size per level
        avg_bid_size = np.mean([s for _, s in snapshot.bids])
        avg_ask_size = np.mean([s for _, s in snapshot.asks])
        
        # Find walls
        buy_walls = []
        for price, size in snapshot.bids:
            if size > avg_bid_size * self.threshold_ratio:
                buy_walls.append((price, size))
                
        sell_walls = []
        for price, size in snapshot.asks:
            if size > avg_ask_size * self.threshold_ratio:
                sell_walls.append((price, size))
                
        return {
            'buy_walls': buy_walls,
            'sell_walls': sell_walls,
            'nearest_support': buy_walls[0][0] if buy_walls else None,
            'nearest_resistance': sell_walls[0][0] if sell_walls else None,
            'support_strength': sum(s for _, s in buy_walls),
            'resistance_strength': sum(s for _, s in sell_walls)
        }


class MicrostructureStrategy(BaseStrategy):
    """
    Market Microstructure Strategy using Order Flow, OBI, and CVD.
    """
    def __init__(self, config: Optional[MicrostructureConfig] = None):
        super().__init__(config or MicrostructureConfig())
        self.config: MicrostructureConfig = self.config
        
        # Confirmation components
        self.footprint = FootprintAnalyzer()
        self.profile = VolumeProfileAnalyzer()
        self.liquidity = LiquidityAnalyzer()
        
        # Strategy components
        self.obi_analyzer = OrderBookImbalance(
            levels=self.config.ob_levels,
            window=50
        )
        self.cvd_analyzer = CVDAnalyzer(lookback=100)
        self.whale_detector = WhaleDetector(
            threshold_usd=self.config.whale_threshold_usd
        )
        self.spread_analyzer = SpreadAnalyzer()
        
        self._last_snapshot: Optional[OrderBookSnapshot] = None
        self._features: Dict[str, float] = {}
        self.is_initialized = False

    def update_snapshot(self, snapshot: OrderBookSnapshot, trade_price: float, trade_vol: float, side: str):
        # Update components
        buy_vol = trade_vol if side == 'buy' else 0
        sell_vol = trade_vol if side == 'sell' else 0
        
        self.footprint.update(buy_vol, sell_vol)
        self.profile.update(trade_price, trade_vol)
        
        self.last_snapshot = snapshot
        
    def confirm_trade(self, side: str, price: float) -> Tuple[bool, str]:
        """
        Confirm if a trade should be taken based on order flow.
        """
        reasons = []
        score = 0
        
        # 1. Check Liquidity Walls
        liq_metrics = self.liquidity.analyze(self.last_snapshot)
        if side == 'buy':
            # Don't buy if right below a sell wall
            if liq_metrics.get('nearest_resistance') and \
               liq_metrics['nearest_resistance'] < price * 1.001:
                reasons.append("Rejected: Buying into Resistance Wall")
                score -= 2
            # Good if above support
            if liq_metrics.get('nearest_support') and \
               liq_metrics['nearest_support'] > price * 0.999:
                score += 1
                
        elif side == 'sell':
            # Don't sell if right above a buy wall
            if liq_metrics.get('nearest_support') and \
               liq_metrics['nearest_support'] > price * 0.999:
                reasons.append("Rejected: Selling into Support Wall")
                score -= 2
        
        # 2. Check Volume Profile
        loc = self.profile.get_location(price)
        if side == 'buy' and loc == 'above_va':
            score += 1  # Breakout
        elif side == 'sell' and loc == 'below_va':
            score += 1  # Breakdown
            
        # 3. Check Footprint (Delta)
        fp_metrics = self.footprint.update(0, 0) # Get current state
        if side == 'buy' and fp_metrics['cumulative_delta'] > 0:
            score += 1
        elif side == 'sell' and fp_metrics['cumulative_delta'] < 0:
            score += 1
            
        if score >= 1:
            return True, "Confirmed by Order Flow"
        else:
            return False, f"Order Flow Weak (Score {score}): {'; '.join(reasons)}"
    

    def update(self, data: pd.DataFrame):
        """
        Update with OHLCV data.
        
        Note: For real microstructure trading, you need order book
        and trade data, not just OHLCV. This simulates with OHLCV.
        """
        self.current_bar = len(data) - 1
        
        # Simulate order book from OHLCV (in production use real L2 data)
        close = data['close'].iloc[-1]
        volume = data['volume'].iloc[-1] if 'volume' in data.columns else 1000
        
        # Simulate bid/ask
        spread = close * 0.0002  # 2 bps spread
        bid = close - spread / 2
        ask = close + spread / 2
        
        # Simulate order book depth
        bids = [(bid - i * spread, volume / 10) for i in range(self.config.ob_levels)]
        asks = [(ask + i * spread, volume / 10) for i in range(self.config.ob_levels)]
        
        # Create snapshot
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )
        self._last_snapshot = snapshot
        
        # Update analyzers
        obi_metrics = self.obi_analyzer.update(snapshot)
        spread_metrics = self.spread_analyzer.update(bid, ask)
        
        # Simulate buy/sell volume split
        buy_vol = volume * (0.5 + np.random.uniform(-0.2, 0.2))
        sell_vol = volume - buy_vol
        cvd_metrics = self.cvd_analyzer.update(buy_vol, sell_vol, close)
        
        # Combine features
        self._features = {**obi_metrics, **cvd_metrics, **spread_metrics}
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate microstructure signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        if self._last_snapshot is None:
            return None
        
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        obi = self._features.get('obi', 0)
        obi_zscore = self._features.get('obi_zscore', 0)
        cvd_normalized = self._features.get('cvd_normalized', 0)
        divergence = self._features.get('divergence', 0)
        spread_bps = self._features.get('spread_bps', 0)
        
        # Check spread is tradeable
        if spread_bps > self.config.max_spread_bps:
            return None
        
        signal_type = None
        reason = ""
        strength = 0.0
        confidence = 0.0
        
        # Strong OBI signal
        if abs(obi) > self.config.obi_strong_threshold:
            if obi > 0:
                signal_type = SignalType.LONG
                reason = f"Strong OBI={obi:.2f}, CVD={cvd_normalized:.2f}"
            else:
                signal_type = SignalType.SHORT
                reason = f"Strong OBI={obi:.2f}, CVD={cvd_normalized:.2f}"
            strength = min(1.0, abs(obi) / 0.6)
            confidence = 0.7
        
        # OBI + CVD confirmation
        elif abs(obi) > self.config.obi_threshold:
            if obi > 0 and cvd_normalized > 0:
                signal_type = SignalType.LONG
                reason = f"OBI={obi:.2f} + CVD={cvd_normalized:.2f} confirm"
                strength = min(1.0, (abs(obi) + abs(cvd_normalized)) / 1.5)
                confidence = 0.65
            elif obi < 0 and cvd_normalized < 0:
                signal_type = SignalType.SHORT
                reason = f"OBI={obi:.2f} + CVD={cvd_normalized:.2f} confirm"
                strength = min(1.0, (abs(obi) + abs(cvd_normalized)) / 1.5)
                confidence = 0.65
        
        # Divergence signal (CVD vs Price)
        elif abs(divergence) > 0.5:
            if divergence > 0:  # CVD rising faster than price
                signal_type = SignalType.LONG
                reason = f"Bullish divergence: CVD leading price"
            else:
                signal_type = SignalType.SHORT
                reason = f"Bearish divergence: CVD lagging price"
            strength = min(1.0, abs(divergence))
            confidence = 0.55
        
        if signal_type is None:
            return None
        
        if confidence < self.config.min_confidence:
            return None
        
        # Short hold time for microstructure
        atr = (data['high'].iloc[-1] - data['low'].iloc[-1]) * 0.1  # Tight stops
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=confidence,
            atr=atr,
            obi=obi,
            cvd=cvd_normalized
        )
    
    def get_features_for_ml(self) -> Dict[str, float]:
        """
        Get features formatted for ML model.
        
        These are the features that predict short-term price moves.
        """
        return {
            # OBI features
            'obi': self._features.get('obi', 0),
            'obi_ma': self._features.get('obi_ma', 0),
            'obi_zscore': self._features.get('obi_zscore', 0),
            'obi_momentum': self._features.get('obi_momentum', 0),
            
            # Depth features
            'bid_depth': self._features.get('bid_depth', 0),
            'ask_depth': self._features.get('ask_depth', 0),
            'depth_ratio': self._features.get('bid_depth', 1) / (self._features.get('ask_depth', 1) + 1e-10),
            
            # CVD features
            'cvd_normalized': self._features.get('cvd_normalized', 0),
            'delta': self._features.get('delta', 0),
            'delta_ma': self._features.get('delta_ma', 0),
            'divergence': self._features.get('divergence', 0),
            'volume_ratio': self._features.get('volume_ratio', 1),
            
            # Spread features
            'spread_bps': self._features.get('spread_bps', 0),
            'spread_zscore': self._features.get('spread_zscore', 0),
            'is_wide_spread': 1 if self._features.get('is_wide', False) else 0,
            'is_tight_spread': 1 if self._features.get('is_tight', False) else 0
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MARKET MICROSTRUCTURE STRATEGIES")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    n = 200
    
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    data = pd.DataFrame({
        'open': prices - np.random.uniform(0, 0.2, n),
        'high': prices + np.random.uniform(0, 0.3, n),
        'low': prices - np.random.uniform(0, 0.3, n),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n),
        'symbol': 'BTCUSDT'
    })
    
    print("\n1. ORDER BOOK IMBALANCE")
    print("-" * 50)
    
    # Simulate order book updates
    obi = OrderBookImbalance()
    
    for i in range(50):
        # Random order book
        bid = prices[i] - 0.1
        ask = prices[i] + 0.1
        bids = [(bid - j * 0.1, np.random.uniform(10, 100)) for j in range(10)]
        asks = [(ask + j * 0.1, np.random.uniform(10, 100)) for j in range(10)]
        
        snapshot = OrderBookSnapshot(datetime.now(), bids, asks)
        metrics = obi.update(snapshot)
    
    print(f"   Current OBI: {metrics['obi']:.3f}")
    print(f"   OBI Z-Score: {metrics['obi_zscore']:.3f}")
    print(f"   Bid Depth: {metrics['bid_depth']:.0f}")
    print(f"   Ask Depth: {metrics['ask_depth']:.0f}")
    
    print("\n2. CUMULATIVE VOLUME DELTA")
    print("-" * 50)
    
    cvd = CVDAnalyzer()
    
    for i in range(50):
        buy_vol = np.random.uniform(100, 500)
        sell_vol = np.random.uniform(100, 500)
        cvd_metrics = cvd.update(buy_vol, sell_vol, prices[i])
    
    print(f"   CVD: {cvd_metrics['cvd']:.0f}")
    print(f"   CVD Normalized: {cvd_metrics['cvd_normalized']:.3f}")
    print(f"   Divergence: {cvd_metrics['divergence']:.3f}")
    
    absorption = cvd.detect_absorption()
    if absorption:
        print(f"   Absorption detected: {absorption}")
    
    print("\n3. WHALE DETECTION")
    print("-" * 50)
    
    whale = WhaleDetector(threshold_usd=50000)
    
    # Simulate some trades
    for i in range(20):
        size = np.random.exponential(100)  # Most small, some large
        detected = whale.check_order(prices[i], size, 'buy' if np.random.random() > 0.5 else 'sell')
        if detected:
            print(f"   🐋 Whale detected: ${detected['usd_value']:,.0f} {detected['side']}")
    
    pressure = whale.get_whale_pressure()
    print(f"   Whale Imbalance: {pressure['whale_imbalance']:.2f}")
    
    print("\n4. MICROSTRUCTURE STRATEGY")
    print("-" * 50)
    
    micro = MicrostructureStrategy(MicrostructureConfig(
        obi_threshold=0.2,
        min_confidence=0.5
    ))
    micro.update(data)
    
    signals = []
    for i in range(50, len(data)):
        micro.current_bar = i
        signal = micro.generate_signal(data.iloc[:i+1])
        if signal:
            signals.append(signal)
    
    print(f"   Total signals: {len(signals)}")
    
    print("\n5. ML FEATURES")
    print("-" * 50)
    
    features = micro.get_features_for_ml()
    print("   Features for ML model:")
    for k, v in list(features.items())[:8]:
        print(f"     {k}: {v:.4f}")
    
    print("\n" + "="*70)
    print("HFT/MICROSTRUCTURE KEY INSIGHTS")
    print("="*70)
    print("""
1. ORDER BOOK IMBALANCE IS THE #1 PREDICTOR
   - OBI > 0.3 = strong buying pressure
   - OBI < -0.3 = strong selling pressure
   - Works best in 1-60 second timeframes
   
2. CVD SHOWS THE REAL BATTLE
   - Price going up but CVD flat = weak rally
   - Price flat but CVD rising = accumulation
   - Divergences are powerful signals
   
3. SPREAD DYNAMICS MATTER
   - Wide spreads = uncertainty, avoid
   - Tight spreads = good liquidity, trade
   - Spread expansion before moves = warning
   
4. WHALE ORDERS MOVE MARKETS
   - Track large orders (>$100k)
   - Iceberg detection catches hidden whales
   - Front-running whales is profitable but risky
   
5. FOR CRYPTO HFT:
   - Use L2 orderbook data from Binance/Bybit
   - Hold positions 1-60 seconds
   - Need sub-100ms execution
   - Profit per trade: 1-5 bps
   - Win rate: 55-65%
   
6. ML MODEL FEATURES (in order of importance):
   1. OBI at multiple levels
   2. OBI momentum
   3. CVD normalized
   4. Spread z-score
   5. Depth ratio
   6. Volume ratio
""")
