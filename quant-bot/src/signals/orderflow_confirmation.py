"""
Order-Flow Confirmation Engine
==============================

THE GAME CHANGER: Use with ANY strategy to 10× clarity.

Three Core Tools:
1. Footprint (Bid/Ask Dominance) → Who is winning the fight?
2. Heatmap Liquidity → Is real money defending or absorbing?
3. Volume Profile → Is trade at high-quality auction area?

THE BULLETPROOF RULE:
A level is only valid if at least 2 of 3 order-flow confirmations agree.

This eliminates low-quality trades and retains only smart-money-confirmed setups.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum


class FootprintSignal(Enum):
    """Footprint chart signals."""
    ABSORPTION = "absorption"          # Large limit orders stop aggressive buyers/sellers
    EXHAUSTION = "exhaustion"          # Decreasing delta → one side giving up
    IMBALANCE = "imbalance"            # >3:1 bid/ask confirms institutional activity
    DELTA_DIVERGENCE = "delta_div"     # Price new low but delta doesn't → reversal
    NEUTRAL = "neutral"


class LiquiditySignal(Enum):
    """Heatmap liquidity signals."""
    STACKING = "stacking"              # Liquidity appearing → institutional defense
    PULLING = "pulling"                # Liquidity pulling → fake breakout ahead
    ABSORPTION = "absorption"          # Absorbing aggression → trend continuation
    SPOOFING = "spoofing"              # Trap; do not trade
    VACUUM = "vacuum"                  # No liquidity → explosive breakout
    NEUTRAL = "neutral"


class VolumeProfileZone(Enum):
    """Volume profile location zones."""
    VAH = "value_area_high"            # Sell imbalance
    VAL = "value_area_low"             # Buy imbalance
    POC = "point_of_control"           # Mean reversion magnet
    LVN = "low_volume_node"            # Breakout acceleration
    HVN = "high_volume_node"           # Heavy chop; avoid unless mean reversion


@dataclass
class OrderFlowConfirmation:
    """Complete order-flow confirmation result."""
    timestamp: datetime
    
    # Individual scores (-1 to +1)
    delta_score: float
    obi_score: float
    liquidity_score: float
    location_score: float
    
    # Trade validity score
    trade_score: float
    is_valid: bool
    
    # Signals detected
    footprint_signal: FootprintSignal
    liquidity_signal: LiquiditySignal
    volume_zone: VolumeProfileZone
    
    # Confirmation count (how many of 3 agree)
    confirmations: int
    confirmation_details: Dict[str, bool]
    
    # Metadata
    reason: str
    confidence: float


@dataclass
class FootprintBar:
    """Single footprint bar data."""
    timestamp: datetime
    price: float
    bid_volume: float          # Volume traded at bid (sells)
    ask_volume: float          # Volume traded at ask (buys)
    delta: float               # ask_volume - bid_volume
    total_volume: float
    
    # Level-by-level data
    levels: Dict[float, Dict[str, float]] = field(default_factory=dict)
    # {price: {'bid': vol, 'ask': vol, 'delta': vol}}


class FootprintAnalyzer:
    """
    Footprint Chart Analysis = Confirmation of Intent
    
    Footprint reads aggression, which price alone can't show.
    
    Confirms:
    - Whether zone is defended
    - If breaker block is real
    - Displacement candles
    - If liquidity sweep has real intent
    - Breakout strength vs fakeout
    """
    
    def __init__(self, imbalance_threshold: float = 3.0, absorption_threshold: float = 2.0):
        """
        Initialize footprint analyzer.
        
        Args:
            imbalance_threshold: Ratio for bid/ask imbalance (default 3:1)
            absorption_threshold: Multiplier for absorption detection
        """
        self.imbalance_threshold = imbalance_threshold
        self.absorption_threshold = absorption_threshold
        
        self._footprint_history: deque = deque(maxlen=1000)
        self._delta_history: deque = deque(maxlen=100)
    
    def add_footprint_bar(self, bar: FootprintBar):
        """Add a footprint bar to history."""
        self._footprint_history.append(bar)
        self._delta_history.append(bar.delta)
    
    def create_footprint_from_trades(
        self, 
        trades: List[Dict],
        price_levels: int = 20
    ) -> FootprintBar:
        """
        Create footprint bar from trade data.
        
        trades: List of {'price', 'size', 'side', 'timestamp'}
        """
        if not trades:
            return FootprintBar(
                timestamp=datetime.now(),
                price=0, bid_volume=0, ask_volume=0,
                delta=0, total_volume=0
            )
        
        bid_volume = sum(t['size'] for t in trades if t['side'] == 'sell')
        ask_volume = sum(t['size'] for t in trades if t['side'] == 'buy')
        
        # Group by price levels
        prices = [t['price'] for t in trades]
        min_price, max_price = min(prices), max(prices)
        level_size = (max_price - min_price) / price_levels if max_price > min_price else 1
        
        levels = {}
        for t in trades:
            level = round(t['price'] / level_size) * level_size
            if level not in levels:
                levels[level] = {'bid': 0, 'ask': 0, 'delta': 0}
            
            if t['side'] == 'sell':
                levels[level]['bid'] += t['size']
            else:
                levels[level]['ask'] += t['size']
            levels[level]['delta'] = levels[level]['ask'] - levels[level]['bid']
        
        bar = FootprintBar(
            timestamp=trades[-1]['timestamp'] if isinstance(trades[-1]['timestamp'], datetime) else datetime.now(),
            price=trades[-1]['price'],
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            delta=ask_volume - bid_volume,
            total_volume=bid_volume + ask_volume,
            levels=levels
        )
        
        self.add_footprint_bar(bar)
        return bar
    
    def calculate_delta_score(self) -> float:
        """
        Calculate Delta Strength Score.
        
        delta_score = (aggressive_buy - aggressive_sell) / total_volume
        Range: -1 to +1
        """
        if not self._footprint_history:
            return 0.0
        
        recent = list(self._footprint_history)[-10:]
        
        total_bid = sum(bar.bid_volume for bar in recent)
        total_ask = sum(bar.ask_volume for bar in recent)
        total_vol = total_bid + total_ask
        
        if total_vol == 0:
            return 0.0
        
        return (total_ask - total_bid) / total_vol
    
    def detect_absorption(self, bar: Optional[FootprintBar] = None) -> bool:
        """
        Detect absorption pattern.
        
        Large limit orders stop aggressive buyers/sellers.
        High volume but small price movement.
        """
        if bar is None:
            if not self._footprint_history:
                return False
            bar = self._footprint_history[-1]
        
        if len(self._footprint_history) < 5:
            return False
        
        # Compare current volume to average
        avg_volume = np.mean([b.total_volume for b in list(self._footprint_history)[-20:]])
        
        # High volume relative to average
        high_volume = bar.total_volume > avg_volume * self.absorption_threshold
        
        # But balanced delta (neither side winning decisively)
        balanced_delta = abs(bar.delta) / bar.total_volume < 0.2 if bar.total_volume > 0 else False
        
        return high_volume and balanced_delta
    
    def detect_exhaustion(self) -> bool:
        """
        Detect exhaustion pattern.
        
        Decreasing delta → one side is giving up.
        """
        if len(self._delta_history) < 5:
            return False
        
        recent_deltas = list(self._delta_history)[-5:]
        
        # Check for decreasing absolute delta
        abs_deltas = [abs(d) for d in recent_deltas]
        
        # Trend of decreasing delta
        decreasing = all(abs_deltas[i] >= abs_deltas[i+1] for i in range(len(abs_deltas)-1))
        
        # Significant decrease
        significant = abs_deltas[0] > abs_deltas[-1] * 2 if abs_deltas[-1] > 0 else True
        
        return decreasing and significant
    
    def detect_imbalance(self, bar: Optional[FootprintBar] = None) -> Tuple[bool, str]:
        """
        Detect imbalance pattern.
        
        > 3:1 bid/ask at a level confirms institutional activity.
        Returns (is_imbalance, direction)
        """
        if bar is None:
            if not self._footprint_history:
                return False, "neutral"
            bar = self._footprint_history[-1]
        
        if not bar.levels:
            # Use bar-level data
            if bar.bid_volume > 0 and bar.ask_volume > 0:
                ratio = max(bar.ask_volume / bar.bid_volume, bar.bid_volume / bar.ask_volume)
                if ratio >= self.imbalance_threshold:
                    direction = "bullish" if bar.ask_volume > bar.bid_volume else "bearish"
                    return True, direction
            return False, "neutral"
        
        # Check each price level for imbalance
        imbalances = []
        for price, data in bar.levels.items():
            bid, ask = data['bid'], data['ask']
            if bid > 0 and ask > 0:
                ratio = max(ask / bid, bid / ask)
                if ratio >= self.imbalance_threshold:
                    direction = "bullish" if ask > bid else "bearish"
                    imbalances.append((price, ratio, direction))
        
        if imbalances:
            # Return strongest imbalance
            strongest = max(imbalances, key=lambda x: x[1])
            return True, strongest[2]
        
        return False, "neutral"
    
    def detect_delta_divergence(self, prices: pd.Series) -> Tuple[bool, str]:
        """
        Detect delta divergence.
        
        Price makes new low but delta doesn't → reversal signal.
        """
        if len(self._delta_history) < 10 or len(prices) < 10:
            return False, "neutral"
        
        recent_prices = prices.iloc[-10:].values
        recent_deltas = list(self._delta_history)[-10:]
        
        # Bullish divergence: Price new low, delta higher low
        price_new_low = recent_prices[-1] == min(recent_prices)
        delta_higher_low = recent_deltas[-1] > min(recent_deltas[:-1])
        
        if price_new_low and delta_higher_low:
            return True, "bullish"
        
        # Bearish divergence: Price new high, delta lower high
        price_new_high = recent_prices[-1] == max(recent_prices)
        delta_lower_high = recent_deltas[-1] < max(recent_deltas[:-1])
        
        if price_new_high and delta_lower_high:
            return True, "bearish"
        
        return False, "neutral"
    
    def get_footprint_signal(self, prices: Optional[pd.Series] = None) -> Tuple[FootprintSignal, float, str]:
        """
        Get overall footprint signal.
        
        Returns (signal, strength, reason)
        """
        if not self._footprint_history:
            return FootprintSignal.NEUTRAL, 0.0, "No footprint data"
        
        bar = self._footprint_history[-1]
        
        # Check for absorption
        if self.detect_absorption(bar):
            return FootprintSignal.ABSORPTION, 0.8, "Absorption detected - limit orders defending"
        
        # Check for exhaustion
        if self.detect_exhaustion():
            return FootprintSignal.EXHAUSTION, 0.7, "Exhaustion detected - one side giving up"
        
        # Check for imbalance
        is_imbalance, direction = self.detect_imbalance(bar)
        if is_imbalance:
            return FootprintSignal.IMBALANCE, 0.9, f"Imbalance detected - {direction} institutional activity"
        
        # Check for delta divergence
        if prices is not None:
            is_divergence, div_direction = self.detect_delta_divergence(prices)
            if is_divergence:
                return FootprintSignal.DELTA_DIVERGENCE, 0.85, f"Delta divergence - {div_direction} reversal signal"
        
        return FootprintSignal.NEUTRAL, 0.0, "No significant footprint signal"


class HeatmapAnalyzer:
    """
    Heatmap Liquidity Analysis = Confirmation of Liquidity & Trap Detection
    
    Shows resting limit orders and their behavior.
    In crypto, heatmap reading is a cheat code.
    
    Liquidity behavior is more predictive than price.
    """
    
    def __init__(
        self,
        significant_liquidity_usd: float = 1000000,  # $1M
        spoofing_threshold: float = 0.5  # 50% pullback = spoofing
    ):
        """
        Initialize heatmap analyzer.
        
        Args:
            significant_liquidity_usd: Minimum USD for significant level
            spoofing_threshold: Threshold for detecting spoofing
        """
        self.significant_liquidity_usd = significant_liquidity_usd
        self.spoofing_threshold = spoofing_threshold
        
        self._orderbook_history: deque = deque(maxlen=100)
        self._liquidity_levels: Dict[float, List[float]] = {}  # price -> [liquidity over time]
    
    def update_orderbook(self, orderbook: Dict):
        """
        Update with new orderbook snapshot.
        
        orderbook: {'bids': [(price, size), ...], 'asks': [(price, size), ...]}
        """
        self._orderbook_history.append({
            'timestamp': datetime.now(),
            'bids': orderbook.get('bids', []),
            'asks': orderbook.get('asks', [])
        })
        
        # Track liquidity at each level
        for price, size in orderbook.get('bids', []) + orderbook.get('asks', []):
            if price not in self._liquidity_levels:
                self._liquidity_levels[price] = []
            self._liquidity_levels[price].append(size)
            
            # Keep only recent history
            if len(self._liquidity_levels[price]) > 50:
                self._liquidity_levels[price] = self._liquidity_levels[price][-50:]
    
    def calculate_obi(self, levels: int = 10) -> float:
        """
        Calculate Orderbook Imbalance Ratio.
        
        obi = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity)
        Range: -1 (all asks) to +1 (all bids)
        
        Positive = bullish (more bids defending)
        Negative = bearish (more asks defending)
        """
        if not self._orderbook_history:
            return 0.0
        
        latest = self._orderbook_history[-1]
        
        bids = latest['bids'][:levels]
        asks = latest['asks'][:levels]
        
        bid_liquidity = sum(size for _, size in bids)
        ask_liquidity = sum(size for _, size in asks)
        
        total = bid_liquidity + ask_liquidity
        if total == 0:
            return 0.0
        
        return (bid_liquidity - ask_liquidity) / total
    
    def detect_liquidity_stacking(self, price: float, tolerance: float = 0.001) -> bool:
        """
        Detect liquidity stacking (appearing at level).
        
        Institutional defense - trend support.
        """
        if not self._liquidity_levels:
            return False
        
        # Find levels near the price
        relevant_levels = [
            (p, sizes) for p, sizes in self._liquidity_levels.items()
            if abs(p - price) / price < tolerance
        ]
        
        if not relevant_levels:
            return False
        
        # Check if liquidity is increasing
        for level_price, sizes in relevant_levels:
            if len(sizes) >= 3:
                # Increasing trend
                if sizes[-1] > sizes[-2] > sizes[-3]:
                    return True
                # Sudden large increase
                if len(sizes) >= 5 and sizes[-1] > np.mean(sizes[-5:-1]) * 2:
                    return True
        
        return False
    
    def detect_liquidity_pulling(self, price: float, tolerance: float = 0.001) -> bool:
        """
        Detect liquidity pulling (disappearing).
        
        Fake breakout ahead - trap risk.
        """
        if not self._liquidity_levels:
            return False
        
        relevant_levels = [
            (p, sizes) for p, sizes in self._liquidity_levels.items()
            if abs(p - price) / price < tolerance
        ]
        
        if not relevant_levels:
            return False
        
        for level_price, sizes in relevant_levels:
            if len(sizes) >= 3:
                # Decreasing trend
                if sizes[-1] < sizes[-2] * self.spoofing_threshold:
                    return True
                # Sudden disappearance
                if len(sizes) >= 5 and sizes[-1] < np.mean(sizes[-5:-1]) * 0.3:
                    return True
        
        return False
    
    def detect_spoofing(self) -> bool:
        """
        Detect spoofing patterns.
        
        Large orders appearing and disappearing = trap.
        """
        if len(self._orderbook_history) < 5:
            return False
        
        # Track large order appearances and disappearances
        large_order_changes = 0
        
        for i in range(1, min(5, len(self._orderbook_history))):
            prev = self._orderbook_history[-i-1]
            curr = self._orderbook_history[-i]
            
            # Check for large orders that appeared then disappeared
            prev_levels = {p: s for p, s in prev['bids'] + prev['asks']}
            curr_levels = {p: s for p, s in curr['bids'] + curr['asks']}
            
            for price in prev_levels:
                if price in curr_levels:
                    # Large decrease
                    if prev_levels[price] > self.significant_liquidity_usd and \
                       curr_levels[price] < prev_levels[price] * 0.3:
                        large_order_changes += 1
        
        # Multiple suspicious changes = spoofing
        return large_order_changes >= 2
    
    def detect_liquidity_vacuum(self, direction: str = 'both') -> bool:
        """
        Detect liquidity vacuum.
        
        No liquidity → explosive breakout potential.
        """
        if not self._orderbook_history:
            return False
        
        latest = self._orderbook_history[-1]
        
        # Check for thin orderbook
        bid_depth = sum(size for _, size in latest['bids'][:10])
        ask_depth = sum(size for _, size in latest['asks'][:10])
        
        threshold = self.significant_liquidity_usd * 0.1  # Very thin
        
        if direction == 'up':
            return ask_depth < threshold
        elif direction == 'down':
            return bid_depth < threshold
        else:
            return bid_depth < threshold or ask_depth < threshold
    
    def calculate_liquidity_score(self, current_price: float) -> float:
        """
        Calculate Liquidity Movement Score.
        
        +1 → Liquidity stacking (trend support)
        -1 → Liquidity pulling (fakeout risk)
        0 → Neutral
        """
        if not self._orderbook_history:
            return 0.0
        
        stacking = self.detect_liquidity_stacking(current_price)
        pulling = self.detect_liquidity_pulling(current_price)
        spoofing = self.detect_spoofing()
        
        if spoofing:
            return -1.0  # High risk
        
        if stacking and not pulling:
            return 1.0  # Trend support
        
        if pulling and not stacking:
            return -0.7  # Fakeout risk
        
        return 0.0
    
    def get_liquidity_signal(self, current_price: float) -> Tuple[LiquiditySignal, float, str]:
        """
        Get overall liquidity signal.
        
        Returns (signal, strength, reason)
        """
        if not self._orderbook_history:
            return LiquiditySignal.NEUTRAL, 0.0, "No orderbook data"
        
        # Check for spoofing first (highest priority)
        if self.detect_spoofing():
            return LiquiditySignal.SPOOFING, 0.9, "Spoofing detected - TRAP, do not trade"
        
        # Check for vacuum
        if self.detect_liquidity_vacuum():
            return LiquiditySignal.VACUUM, 0.8, "Liquidity vacuum - explosive breakout potential"
        
        # Check for stacking
        if self.detect_liquidity_stacking(current_price):
            return LiquiditySignal.STACKING, 0.7, "Liquidity stacking - institutional defense"
        
        # Check for pulling
        if self.detect_liquidity_pulling(current_price):
            return LiquiditySignal.PULLING, 0.7, "Liquidity pulling - fake breakout risk"
        
        # Check OBI for absorption
        obi = self.calculate_obi()
        if abs(obi) > 0.5:
            direction = "bid" if obi > 0 else "ask"
            return LiquiditySignal.ABSORPTION, abs(obi), f"Absorption on {direction} side"
        
        return LiquiditySignal.NEUTRAL, 0.0, "No significant liquidity signal"


class VolumeProfileAnalyzer:
    """
    Volume Profile Analysis = Confirmation of Location
    
    Location is everything.
    ICT/SMC give structure, but location determines risk.
    """
    
    def __init__(
        self,
        value_area_pct: float = 0.70,  # 70% of volume
        lvn_threshold: float = 0.3,     # Below 30% of avg = LVN
        hvn_threshold: float = 1.5      # Above 150% of avg = HVN
    ):
        """
        Initialize volume profile analyzer.
        
        Args:
            value_area_pct: Percentage for value area calculation
            lvn_threshold: Threshold for low volume nodes
            hvn_threshold: Threshold for high volume nodes
        """
        self.value_area_pct = value_area_pct
        self.lvn_threshold = lvn_threshold
        self.hvn_threshold = hvn_threshold
        
        self._profile: Dict[float, float] = {}  # price -> volume
        self._vah: Optional[float] = None
        self._val: Optional[float] = None
        self._poc: Optional[float] = None
    
    def build_profile(
        self,
        data: pd.DataFrame,
        price_levels: int = 50
    ):
        """
        Build volume profile from OHLCV data.
        
        Uses typical price and volume distribution.
        """
        if data.empty:
            return
        
        # Calculate typical price for each bar
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Define price levels
        price_min = data['low'].min()
        price_max = data['high'].max()
        level_size = (price_max - price_min) / price_levels
        
        # Distribute volume to levels
        self._profile = {}
        
        for i in range(len(data)):
            tp = typical_price.iloc[i]
            vol = data['volume'].iloc[i]
            
            # Assign to nearest level
            level = round((tp - price_min) / level_size) * level_size + price_min
            self._profile[level] = self._profile.get(level, 0) + vol
        
        # Calculate POC, VAH, VAL
        self._calculate_value_area()
    
    def build_profile_from_trades(self, trades: List[Dict], price_levels: int = 50):
        """
        Build volume profile from trade data.
        
        More accurate than OHLCV-based profile.
        """
        if not trades:
            return
        
        prices = [t['price'] for t in trades]
        price_min, price_max = min(prices), max(prices)
        level_size = (price_max - price_min) / price_levels if price_max > price_min else 1
        
        self._profile = {}
        
        for trade in trades:
            level = round((trade['price'] - price_min) / level_size) * level_size + price_min
            self._profile[level] = self._profile.get(level, 0) + trade['size']
        
        self._calculate_value_area()
    
    def _calculate_value_area(self):
        """Calculate POC, VAH, and VAL."""
        if not self._profile:
            return
        
        # POC = highest volume level
        self._poc = max(self._profile, key=self._profile.get)
        
        # Value Area = levels containing 70% of volume
        total_volume = sum(self._profile.values())
        target_volume = total_volume * self.value_area_pct
        
        # Sort levels by proximity to POC
        sorted_levels = sorted(
            self._profile.keys(),
            key=lambda x: abs(x - self._poc)
        )
        
        cumulative = 0
        value_area_levels = []
        
        for level in sorted_levels:
            cumulative += self._profile[level]
            value_area_levels.append(level)
            if cumulative >= target_volume:
                break
        
        if value_area_levels:
            self._vah = max(value_area_levels)
            self._val = min(value_area_levels)
    
    def get_zone(self, price: float) -> VolumeProfileZone:
        """
        Get the volume profile zone for a price.
        """
        if not self._profile or self._poc is None:
            return VolumeProfileZone.POC
        
        # Check proximity to key levels
        avg_vol = np.mean(list(self._profile.values()))
        
        # Find nearest profile level
        nearest_level = min(self._profile.keys(), key=lambda x: abs(x - price))
        level_vol = self._profile[nearest_level]
        
        # LVN check
        if level_vol < avg_vol * self.lvn_threshold:
            return VolumeProfileZone.LVN
        
        # HVN check
        if level_vol > avg_vol * self.hvn_threshold:
            return VolumeProfileZone.HVN
        
        # VAH/VAL/POC check
        if self._vah and abs(price - self._vah) < abs(price - self._poc):
            if abs(price - self._vah) < abs(price - self._val if self._val else float('inf')):
                return VolumeProfileZone.VAH
        
        if self._val and abs(price - self._val) < abs(price - self._poc):
            return VolumeProfileZone.VAL
        
        return VolumeProfileZone.POC
    
    def calculate_location_score(self, price: float, direction: str = 'long') -> float:
        """
        Calculate Volume Profile Location Score.
        
        +1 → VAL (for longs) / VAH (for shorts)
        -1 → VAH (for longs) / VAL (for shorts)
        +2 → LVN breakout zones
        """
        zone = self.get_zone(price)
        
        if zone == VolumeProfileZone.LVN:
            return 2.0  # Breakout acceleration
        
        if zone == VolumeProfileZone.HVN:
            return -0.5  # Heavy chop, avoid
        
        if direction == 'long':
            if zone == VolumeProfileZone.VAL:
                return 1.0  # Buy imbalance
            elif zone == VolumeProfileZone.VAH:
                return -1.0  # Sell imbalance
        else:  # short
            if zone == VolumeProfileZone.VAH:
                return 1.0  # Sell imbalance
            elif zone == VolumeProfileZone.VAL:
                return -1.0  # Buy imbalance
        
        return 0.0  # POC = neutral
    
    def get_profile_stats(self) -> Dict:
        """Get volume profile statistics."""
        return {
            'poc': self._poc,
            'vah': self._vah,
            'val': self._val,
            'total_volume': sum(self._profile.values()) if self._profile else 0,
            'level_count': len(self._profile)
        }


class OrderFlowConfirmationEngine:
    """
    Complete Order-Flow Confirmation Engine
    
    Combines:
    1. Footprint Analysis
    2. Heatmap/Liquidity Analysis
    3. Volume Profile Analysis
    
    THE BULLETPROOF RULE:
    A level is only valid if at least 2 of 3 order-flow confirmations agree.
    
    Trade Validity Score:
    trade_score = 0.4 * delta_score + 0.3 * obi + 0.2 * liquidity_score + 0.1 * location_score
    
    Only enter trades when trade_score > 0.5
    """
    
    def __init__(
        self,
        min_confirmations: int = 2,
        min_trade_score: float = 0.5,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize order-flow confirmation engine.
        
        Args:
            min_confirmations: Minimum confirmations required (default 2 of 3)
            min_trade_score: Minimum trade score to enter (default 0.5)
            weights: Custom weights for score calculation
        """
        self.min_confirmations = min_confirmations
        self.min_trade_score = min_trade_score
        
        # Default weights
        self.weights = weights or {
            'delta': 0.4,
            'obi': 0.3,
            'liquidity': 0.2,
            'location': 0.1
        }
        
        # Analyzers
        self.footprint = FootprintAnalyzer()
        self.heatmap = HeatmapAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
    
    def update_footprint(self, trades: List[Dict]):
        """Update footprint analyzer with trade data."""
        if trades:
            self.footprint.create_footprint_from_trades(trades)
    
    def update_orderbook(self, orderbook: Dict):
        """Update heatmap analyzer with orderbook data."""
        self.heatmap.update_orderbook(orderbook)
    
    def update_volume_profile(self, data: pd.DataFrame):
        """Update volume profile from OHLCV data."""
        self.volume_profile.build_profile(data)
    
    def get_confirmation(
        self,
        current_price: float,
        direction: str = 'long',
        prices: Optional[pd.Series] = None
    ) -> OrderFlowConfirmation:
        """
        Get complete order-flow confirmation.
        
        Args:
            current_price: Current market price
            direction: Trade direction ('long' or 'short')
            prices: Price series for divergence detection
        
        Returns:
            OrderFlowConfirmation with all signals and scores
        """
        timestamp = datetime.now()
        
        # 1. Footprint Analysis
        footprint_signal, fp_strength, fp_reason = self.footprint.get_footprint_signal(prices)
        delta_score = self.footprint.calculate_delta_score()
        
        # Adjust delta score for direction
        if direction == 'short':
            delta_score = -delta_score
        
        # 2. Heatmap/Liquidity Analysis
        liquidity_signal, liq_strength, liq_reason = self.heatmap.get_liquidity_signal(current_price)
        obi_score = self.heatmap.calculate_obi()
        liquidity_score = self.heatmap.calculate_liquidity_score(current_price)
        
        # Adjust OBI for direction
        if direction == 'short':
            obi_score = -obi_score
        
        # 3. Volume Profile Analysis
        volume_zone = self.volume_profile.get_zone(current_price)
        location_score = self.volume_profile.calculate_location_score(current_price, direction)
        
        # Normalize location score to -1 to 1 range
        location_score_normalized = max(-1, min(1, location_score / 2))
        
        # Calculate Trade Validity Score
        trade_score = (
            self.weights['delta'] * delta_score +
            self.weights['obi'] * obi_score +
            self.weights['liquidity'] * liquidity_score +
            self.weights['location'] * location_score_normalized
        )
        
        # Count confirmations
        confirmations = 0
        confirmation_details = {}
        
        # Footprint confirmation
        fp_confirms = footprint_signal in [
            FootprintSignal.ABSORPTION,
            FootprintSignal.IMBALANCE,
            FootprintSignal.DELTA_DIVERGENCE
        ] and delta_score > 0
        confirmation_details['footprint'] = fp_confirms
        if fp_confirms:
            confirmations += 1
        
        # Liquidity confirmation
        liq_confirms = liquidity_signal in [
            LiquiditySignal.STACKING,
            LiquiditySignal.ABSORPTION,
            LiquiditySignal.VACUUM
        ] and liquidity_signal != LiquiditySignal.SPOOFING
        confirmation_details['liquidity'] = liq_confirms
        if liq_confirms:
            confirmations += 1
        
        # Volume profile confirmation
        vp_confirms = volume_zone in [
            VolumeProfileZone.VAL if direction == 'long' else VolumeProfileZone.VAH,
            VolumeProfileZone.LVN
        ]
        confirmation_details['volume_profile'] = vp_confirms
        if vp_confirms:
            confirmations += 1
        
        # Determine validity
        is_valid = (
            confirmations >= self.min_confirmations and
            trade_score >= self.min_trade_score and
            liquidity_signal != LiquiditySignal.SPOOFING
        )
        
        # Generate reason
        reasons = []
        if fp_confirms:
            reasons.append(f"Footprint: {fp_reason}")
        if liq_confirms:
            reasons.append(f"Liquidity: {liq_reason}")
        if vp_confirms:
            reasons.append(f"Location: {volume_zone.value}")
        
        reason = " | ".join(reasons) if reasons else "No confirmations"
        
        # Calculate confidence
        confidence = min(1.0, (
            0.4 * fp_strength +
            0.3 * liq_strength +
            0.3 * (1 if vp_confirms else 0)
        ))
        
        return OrderFlowConfirmation(
            timestamp=timestamp,
            delta_score=delta_score,
            obi_score=obi_score,
            liquidity_score=liquidity_score,
            location_score=location_score,
            trade_score=trade_score,
            is_valid=is_valid,
            footprint_signal=footprint_signal,
            liquidity_signal=liquidity_signal,
            volume_zone=volume_zone,
            confirmations=confirmations,
            confirmation_details=confirmation_details,
            reason=reason,
            confidence=confidence
        )
    
    def should_enter_trade(
        self,
        current_price: float,
        direction: str = 'long',
        prices: Optional[pd.Series] = None
    ) -> Tuple[bool, OrderFlowConfirmation]:
        """
        Simple interface: Should we enter this trade?
        
        Returns (should_enter, confirmation_details)
        """
        confirmation = self.get_confirmation(current_price, direction, prices)
        return confirmation.is_valid, confirmation
    
    def validate_signal(
        self,
        signal_type: str,
        entry_price: float,
        prices: Optional[pd.Series] = None
    ) -> Dict:
        """
        Validate any trading signal with order-flow confirmation.
        
        Works with ICT/SMC signals, breakouts, etc.
        
        Args:
            signal_type: Type of signal (e.g., 'order_block', 'fvg', 'breakout')
            entry_price: Proposed entry price
            prices: Price series for context
        
        Returns:
            Validation result with recommendations
        """
        direction = 'long'  # Default, can be extended
        
        confirmation = self.get_confirmation(entry_price, direction, prices)
        
        # Signal-specific validation
        validation = {
            'signal_type': signal_type,
            'entry_price': entry_price,
            'is_valid': confirmation.is_valid,
            'trade_score': confirmation.trade_score,
            'confirmations': confirmation.confirmations,
            'confirmation_details': confirmation.confirmation_details,
            'recommendation': 'ENTER' if confirmation.is_valid else 'SKIP',
            'reason': confirmation.reason,
            'warnings': []
        }
        
        # Add warnings
        if confirmation.liquidity_signal == LiquiditySignal.SPOOFING:
            validation['warnings'].append("⚠️ SPOOFING DETECTED - High trap risk")
        
        if confirmation.liquidity_signal == LiquiditySignal.PULLING:
            validation['warnings'].append("⚠️ Liquidity pulling - Potential fakeout")
        
        if confirmation.volume_zone == VolumeProfileZone.HVN:
            validation['warnings'].append("⚠️ High volume node - Expect chop")
        
        if confirmation.footprint_signal == FootprintSignal.EXHAUSTION:
            validation['warnings'].append("⚠️ Exhaustion detected - Trend may reverse")
        
        return validation


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_orderflow_engine(
    significant_liquidity: float = 1000000,
    min_confirmations: int = 2,
    min_trade_score: float = 0.5
) -> OrderFlowConfirmationEngine:
    """Create a configured order-flow confirmation engine."""
    engine = OrderFlowConfirmationEngine(
        min_confirmations=min_confirmations,
        min_trade_score=min_trade_score
    )
    engine.heatmap.significant_liquidity_usd = significant_liquidity
    return engine


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ORDER-FLOW CONFIRMATION ENGINE")
    print("=" * 70)
    
    import numpy as np
    np.random.seed(42)
    
    # Create engine
    engine = create_orderflow_engine(
        significant_liquidity=1000000,
        min_confirmations=2,
        min_trade_score=0.5
    )
    
    print("\n1. FOOTPRINT ANALYSIS")
    print("-" * 50)
    
    # Simulate trade data
    trades = []
    base_price = 50000
    for i in range(100):
        trades.append({
            'price': base_price + np.random.randn() * 100,
            'size': np.random.exponential(1000),
            'side': 'buy' if np.random.random() > 0.4 else 'sell',
            'timestamp': datetime.now()
        })
    
    engine.update_footprint(trades)
    
    delta_score = engine.footprint.calculate_delta_score()
    print(f"   Delta Score: {delta_score:.3f}")
    
    fp_signal, fp_strength, fp_reason = engine.footprint.get_footprint_signal()
    print(f"   Footprint Signal: {fp_signal.value}")
    print(f"   Reason: {fp_reason}")
    
    print("\n2. HEATMAP/LIQUIDITY ANALYSIS")
    print("-" * 50)
    
    # Simulate orderbook
    orderbook = {
        'bids': [(49900 - i*10, np.random.exponential(50000)) for i in range(20)],
        'asks': [(50100 + i*10, np.random.exponential(40000)) for i in range(20)]
    }
    
    engine.update_orderbook(orderbook)
    
    obi = engine.heatmap.calculate_obi()
    print(f"   OBI Score: {obi:.3f}")
    
    liq_signal, liq_strength, liq_reason = engine.heatmap.get_liquidity_signal(50000)
    print(f"   Liquidity Signal: {liq_signal.value}")
    print(f"   Reason: {liq_reason}")
    
    print("\n3. VOLUME PROFILE ANALYSIS")
    print("-" * 50)
    
    # Create OHLCV data
    n = 200
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    
    data = pd.DataFrame({
        'high': prices + np.random.uniform(50, 150, n),
        'low': prices - np.random.uniform(50, 150, n),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n)
    })
    
    engine.update_volume_profile(data)
    
    stats = engine.volume_profile.get_profile_stats()
    print(f"   POC: ${stats['poc']:,.0f}")
    print(f"   VAH: ${stats['vah']:,.0f}")
    print(f"   VAL: ${stats['val']:,.0f}")
    
    zone = engine.volume_profile.get_zone(50000)
    print(f"   Current Zone: {zone.value}")
    
    print("\n4. COMPLETE CONFIRMATION")
    print("-" * 50)
    
    confirmation = engine.get_confirmation(
        current_price=50000,
        direction='long',
        prices=data['close']
    )
    
    print(f"   Trade Score: {confirmation.trade_score:.3f}")
    print(f"   Is Valid: {confirmation.is_valid}")
    print(f"   Confirmations: {confirmation.confirmations}/3")
    print(f"   Details: {confirmation.confirmation_details}")
    print(f"   Reason: {confirmation.reason}")
    
    print("\n5. SIGNAL VALIDATION EXAMPLE")
    print("-" * 50)
    
    # Validate an ICT Order Block signal
    validation = engine.validate_signal(
        signal_type='order_block',
        entry_price=49850,
        prices=data['close']
    )
    
    print(f"   Signal: {validation['signal_type']}")
    print(f"   Entry: ${validation['entry_price']:,.0f}")
    print(f"   Recommendation: {validation['recommendation']}")
    print(f"   Trade Score: {validation['trade_score']:.3f}")
    print(f"   Warnings: {validation['warnings']}")
    
    print("\n" + "=" * 70)
    print("KEY PRINCIPLES")
    print("=" * 70)
    print("""
THE BULLETPROOF RULE:
A level is only valid if at least 2 of 3 order-flow confirmations agree.

TRADE VALIDITY SCORE:
trade_score = 0.4 * delta_score + 0.3 * obi + 0.2 * liquidity_score + 0.1 * location_score

Only enter trades when trade_score > 0.5

WHY THIS WORKS:
✓ Cuts noise
✓ Confirms institutional intent  
✓ Eliminates fake breakouts
✓ Avoids low-liquidity traps
✓ Finds accurate entries
✓ Reduces stop-outs
✓ Improves RRR
✓ Aligns with smart money

This is the closest thing to a "real edge" retail traders can get.
""")
