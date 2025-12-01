"""
Order Flow Analysis - Real-Time Market Microstructure Analysis

Provides:
- Cumulative Volume Delta (CVD)
- Footprint chart analysis
- Whale detection and tracking
- Absorption detection
- Iceberg order detection
- Trade flow toxicity metrics
- Point of Control (POC) analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum


class TradeAggression(Enum):
    """Trade aggression classification"""
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"
    NEUTRAL = "neutral"


class FlowType(Enum):
    """Order flow type"""
    ABSORPTION = "absorption"
    EXHAUSTION = "exhaustion"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


@dataclass
class Trade:
    """Individual trade"""
    timestamp: float
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    aggression: TradeAggression = TradeAggression.NEUTRAL
    is_whale: bool = False


@dataclass
class FootprintLevel:
    """Single price level in footprint"""
    price: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    delta: float = 0.0
    trade_count: int = 0
    
    @property
    def total_volume(self) -> float:
        return self.bid_volume + self.ask_volume
    
    @property
    def imbalance(self) -> float:
        total = self.total_volume
        return self.delta / total if total > 0 else 0


@dataclass
class FootprintBar:
    """Footprint chart bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    levels: Dict[float, FootprintLevel]
    cvd: float  # Cumulative Volume Delta
    delta: float  # Bar delta
    total_volume: float
    buy_volume: float
    sell_volume: float
    poc: float  # Point of Control
    vah: float  # Value Area High
    val: float  # Value Area Low
    whale_trades: int
    absorption_detected: bool
    exhaustion_detected: bool


@dataclass
class OrderFlowSignal:
    """Order flow trading signal"""
    direction: int  # 1=long, -1=short, 0=neutral
    strength: float  # 0-1
    flow_type: FlowType
    confidence: float
    supporting_factors: List[str]
    whale_activity: str
    entry_price: float
    stop_loss: float
    take_profit: float


class OrderFlowAnalyzer:
    """
    Real-Time Order Flow Analysis Engine
    
    Analyzes tick-by-tick data to extract institutional
    order flow patterns and generate trading signals.
    """
    
    def __init__(
        self,
        tick_size: float = 0.01,
        bar_period: int = 60,  # seconds
        whale_threshold: float = 10.0,  # Multiple of average trade
        absorption_threshold: float = 3.0,
        cvd_smoothing: int = 20
    ):
        """
        Initialize order flow analyzer
        
        Args:
            tick_size: Price tick size
            bar_period: Period for footprint bars
            whale_threshold: Whale trade detection threshold
            absorption_threshold: Absorption detection threshold
            cvd_smoothing: CVD smoothing period
        """
        self.tick_size = tick_size
        self.bar_period = bar_period
        self.whale_threshold = whale_threshold
        self.absorption_threshold = absorption_threshold
        self.cvd_smoothing = cvd_smoothing
        
        # Trade buffer
        self._trades: deque = deque(maxlen=100000)
        self._avg_trade_size: float = 0.0
        
        # Footprint bars
        self._bars: List[FootprintBar] = []
        self._current_bar: Optional[FootprintBar] = None
        self._bar_start_time: float = 0
        
        # CVD tracking
        self._cvd: float = 0.0
        self._cvd_history: deque = deque(maxlen=1000)
        
        # Whale tracking
        self._whale_trades: deque = deque(maxlen=100)
        
        # Iceberg detection
        self._iceberg_levels: Dict[float, Dict] = {}
    
    def process_trade(self, trade: Trade) -> None:
        """
        Process incoming trade
        
        Args:
            trade: Trade data
        """
        # Update average trade size
        if self._avg_trade_size == 0:
            self._avg_trade_size = trade.size
        else:
            self._avg_trade_size = 0.99 * self._avg_trade_size + 0.01 * trade.size
        
        # Detect whale trades
        if trade.size >= self._avg_trade_size * self.whale_threshold:
            trade.is_whale = True
            self._whale_trades.append(trade)
        
        # Classify aggression
        trade.aggression = self._classify_aggression(trade)
        
        # Update CVD
        delta = trade.size if trade.side == 'buy' else -trade.size
        self._cvd += delta
        self._cvd_history.append((trade.timestamp, self._cvd))
        
        # Add to buffer
        self._trades.append(trade)
        
        # Update current footprint bar
        self._update_footprint(trade)
        
        # Detect iceberg orders
        self._detect_iceberg(trade)
    
    def process_trades_batch(self, trades: List[Dict]) -> None:
        """Process batch of trades"""
        for t in trades:
            trade = Trade(
                timestamp=t.get('timestamp', 0),
                price=t.get('price', 0),
                size=t.get('size', 0),
                side=t.get('side', 'buy')
            )
            self.process_trade(trade)
    
    def get_signal(self, orderbook: Optional[Dict] = None) -> OrderFlowSignal:
        """
        Generate trading signal from order flow analysis
        
        Args:
            orderbook: Current orderbook for context
        
        Returns:
            OrderFlowSignal with trade recommendation
        """
        if len(self._trades) < 100:
            return self._neutral_signal()
        
        # Analyze components
        cvd_signal = self._analyze_cvd()
        delta_signal = self._analyze_bar_delta()
        whale_signal = self._analyze_whale_activity()
        absorption = self._detect_absorption()
        exhaustion = self._detect_exhaustion()
        imbalance = self._analyze_imbalance()
        
        # Combine signals
        direction = 0
        strength = 0.0
        factors = []
        
        # CVD momentum
        if abs(cvd_signal) > 0.3:
            direction += np.sign(cvd_signal)
            strength += abs(cvd_signal) * 0.25
            factors.append(f"CVD {'bullish' if cvd_signal > 0 else 'bearish'}: {cvd_signal:.2f}")
        
        # Bar delta
        if abs(delta_signal) > 0.3:
            direction += np.sign(delta_signal)
            strength += abs(delta_signal) * 0.2
            factors.append(f"Delta {'bullish' if delta_signal > 0 else 'bearish'}")
        
        # Whale activity
        if abs(whale_signal) > 0.3:
            direction += np.sign(whale_signal) * 2  # Double weight
            strength += abs(whale_signal) * 0.3
            factors.append(f"Whale {'buying' if whale_signal > 0 else 'selling'}")
        
        # Absorption (reversal signal)
        if absorption['detected']:
            direction += absorption['direction']
            strength += 0.2
            factors.append(f"Absorption at {absorption['price']:.2f}")
        
        # Exhaustion (reversal signal)
        if exhaustion['detected']:
            direction += exhaustion['direction']
            strength += 0.15
            factors.append("Exhaustion detected")
        
        # Imbalance
        if abs(imbalance) > 0.4:
            direction += np.sign(imbalance)
            strength += abs(imbalance) * 0.1
            factors.append(f"Order imbalance: {imbalance:.2f}")
        
        # Normalize
        direction = np.sign(direction) if abs(direction) > 0 else 0
        strength = min(strength, 1.0)
        
        # Determine flow type
        flow_type = self._classify_flow(cvd_signal, delta_signal, absorption, exhaustion)
        
        # Calculate confidence
        confidence = strength * (0.5 + 0.5 * len(factors) / 5)
        
        # Calculate levels
        current_price = self._trades[-1].price if self._trades else 0
        atr = self._calculate_atr()
        
        if direction > 0:
            stop_loss = current_price - atr * 1.5
            take_profit = current_price + atr * 2.5
        elif direction < 0:
            stop_loss = current_price + atr * 1.5
            take_profit = current_price - atr * 2.5
        else:
            stop_loss = take_profit = current_price
        
        return OrderFlowSignal(
            direction=direction,
            strength=strength,
            flow_type=flow_type,
            confidence=confidence,
            supporting_factors=factors,
            whale_activity=f"{len(self._whale_trades)} whale trades, net {'buying' if whale_signal > 0 else 'selling'}",
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def get_footprint_bars(self, n: int = 20) -> List[FootprintBar]:
        """Get recent footprint bars"""
        return self._bars[-n:] if self._bars else []
    
    def get_cvd(self) -> Tuple[float, List[Tuple[float, float]]]:
        """Get current CVD and history"""
        return self._cvd, list(self._cvd_history)
    
    def get_whale_activity(self) -> Dict:
        """Get whale activity summary"""
        if not self._whale_trades:
            return {'buy_volume': 0, 'sell_volume': 0, 'net': 0, 'count': 0}
        
        buy_vol = sum(t.size for t in self._whale_trades if t.side == 'buy')
        sell_vol = sum(t.size for t in self._whale_trades if t.side == 'sell')
        
        return {
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'net': buy_vol - sell_vol,
            'count': len(self._whale_trades),
            'avg_size': np.mean([t.size for t in self._whale_trades])
        }
    
    def get_volume_profile(self, bars: int = 20) -> Dict[float, float]:
        """Get volume at price profile"""
        profile = {}
        
        for bar in self._bars[-bars:]:
            for level in bar.levels.values():
                price = level.price
                if price not in profile:
                    profile[price] = 0
                profile[price] += level.total_volume
        
        return dict(sorted(profile.items()))
    
    def get_poc(self, bars: int = 20) -> float:
        """Get Point of Control (highest volume price)"""
        profile = self.get_volume_profile(bars)
        if not profile:
            return 0
        return max(profile.items(), key=lambda x: x[1])[0]
    
    def get_value_area(self, bars: int = 20, pct: float = 0.7) -> Tuple[float, float]:
        """Get Value Area (70% of volume)"""
        profile = self.get_volume_profile(bars)
        if not profile:
            return 0, 0
        
        total_vol = sum(profile.values())
        target_vol = total_vol * pct
        
        # Find POC
        poc = self.get_poc(bars)
        
        # Expand from POC
        prices = sorted(profile.keys())
        poc_idx = prices.index(poc) if poc in prices else len(prices) // 2
        
        val = vah = poc
        current_vol = profile.get(poc, 0)
        
        left = poc_idx - 1
        right = poc_idx + 1
        
        while current_vol < target_vol and (left >= 0 or right < len(prices)):
            left_vol = profile.get(prices[left], 0) if left >= 0 else 0
            right_vol = profile.get(prices[right], 0) if right < len(prices) else 0
            
            if left_vol >= right_vol and left >= 0:
                val = prices[left]
                current_vol += left_vol
                left -= 1
            elif right < len(prices):
                vah = prices[right]
                current_vol += right_vol
                right += 1
            else:
                break
        
        return val, vah
    
    def _update_footprint(self, trade: Trade) -> None:
        """Update current footprint bar"""
        # Check if need new bar
        if self._current_bar is None or trade.timestamp - self._bar_start_time >= self.bar_period:
            # Finalize current bar
            if self._current_bar:
                self._finalize_bar()
            
            # Start new bar
            self._bar_start_time = trade.timestamp
            self._current_bar = FootprintBar(
                timestamp=datetime.fromtimestamp(trade.timestamp),
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                levels={},
                cvd=self._cvd,
                delta=0,
                total_volume=0,
                buy_volume=0,
                sell_volume=0,
                poc=trade.price,
                vah=trade.price,
                val=trade.price,
                whale_trades=0,
                absorption_detected=False,
                exhaustion_detected=False
            )
        
        bar = self._current_bar
        
        # Update OHLC
        bar.high = max(bar.high, trade.price)
        bar.low = min(bar.low, trade.price)
        bar.close = trade.price
        
        # Update levels
        level_price = round(trade.price / self.tick_size) * self.tick_size
        if level_price not in bar.levels:
            bar.levels[level_price] = FootprintLevel(price=level_price)
        
        level = bar.levels[level_price]
        if trade.side == 'buy':
            level.ask_volume += trade.size
            bar.buy_volume += trade.size
        else:
            level.bid_volume += trade.size
            bar.sell_volume += trade.size
        
        level.delta = level.ask_volume - level.bid_volume
        level.trade_count += 1
        
        # Update bar totals
        bar.total_volume = bar.buy_volume + bar.sell_volume
        bar.delta = bar.buy_volume - bar.sell_volume
        
        # Whale tracking
        if trade.is_whale:
            bar.whale_trades += 1
    
    def _finalize_bar(self) -> None:
        """Finalize current bar and calculate derived metrics"""
        if not self._current_bar:
            return
        
        bar = self._current_bar
        
        # Calculate POC
        if bar.levels:
            poc_level = max(bar.levels.values(), key=lambda x: x.total_volume)
            bar.poc = poc_level.price
        
        # Detect absorption
        bar.absorption_detected = self._check_bar_absorption(bar)
        
        # Detect exhaustion
        bar.exhaustion_detected = self._check_bar_exhaustion(bar)
        
        self._bars.append(bar)
        
        # Keep only recent bars
        if len(self._bars) > 1000:
            self._bars = self._bars[-500:]
    
    def _classify_aggression(self, trade: Trade) -> TradeAggression:
        """Classify trade aggression based on price movement"""
        if len(self._trades) < 2:
            return TradeAggression.NEUTRAL
        
        prev_trade = self._trades[-1]
        
        # Uptick = aggressive buying, Downtick = aggressive selling
        if trade.price > prev_trade.price:
            return TradeAggression.AGGRESSIVE if trade.side == 'buy' else TradeAggression.PASSIVE
        elif trade.price < prev_trade.price:
            return TradeAggression.AGGRESSIVE if trade.side == 'sell' else TradeAggression.PASSIVE
        else:
            return TradeAggression.NEUTRAL
    
    def _detect_iceberg(self, trade: Trade) -> None:
        """Detect potential iceberg orders"""
        level = round(trade.price / self.tick_size) * self.tick_size
        
        if level not in self._iceberg_levels:
            self._iceberg_levels[level] = {
                'count': 0,
                'volume': 0,
                'side': trade.side,
                'last_time': trade.timestamp
            }
        
        info = self._iceberg_levels[level]
        
        # Same side trades at same level within short time = potential iceberg
        if (trade.side == info['side'] and 
            trade.timestamp - info['last_time'] < 5):  # Within 5 seconds
            info['count'] += 1
            info['volume'] += trade.size
        else:
            info['count'] = 1
            info['volume'] = trade.size
            info['side'] = trade.side
        
        info['last_time'] = trade.timestamp
    
    def _analyze_cvd(self) -> float:
        """Analyze CVD trend"""
        if len(self._cvd_history) < self.cvd_smoothing:
            return 0
        
        recent = [h[1] for h in list(self._cvd_history)[-self.cvd_smoothing:]]
        
        # Trend direction
        if len(recent) >= 2:
            slope = (recent[-1] - recent[0]) / len(recent)
            max_range = max(recent) - min(recent)
            
            if max_range > 0:
                normalized_slope = slope / max_range * len(recent)
                return np.clip(normalized_slope, -1, 1)
        
        return 0
    
    def _analyze_bar_delta(self) -> float:
        """Analyze recent bar deltas"""
        if len(self._bars) < 5:
            return 0
        
        recent_deltas = [bar.delta for bar in self._bars[-5:]]
        avg_delta = np.mean(recent_deltas)
        total_volume = sum(bar.total_volume for bar in self._bars[-5:])
        
        if total_volume > 0:
            return np.clip(avg_delta / total_volume * 5, -1, 1)
        return 0
    
    def _analyze_whale_activity(self) -> float:
        """Analyze whale trading activity"""
        if not self._whale_trades:
            return 0
        
        buy_vol = sum(t.size for t in self._whale_trades if t.side == 'buy')
        sell_vol = sum(t.size for t in self._whale_trades if t.side == 'sell')
        total = buy_vol + sell_vol
        
        if total > 0:
            return (buy_vol - sell_vol) / total
        return 0
    
    def _detect_absorption(self) -> Dict:
        """Detect absorption patterns"""
        if len(self._bars) < 3:
            return {'detected': False}
        
        recent = self._bars[-3:]
        
        # Absorption: large volume but small price movement
        for bar in recent:
            if bar.total_volume > 0:
                efficiency = abs(bar.close - bar.open) / (bar.high - bar.low) if bar.high != bar.low else 1
                
                # Low efficiency with high volume = absorption
                if efficiency < 0.3 and bar.total_volume > self._avg_trade_size * 100:
                    # Determine direction based on delta
                    direction = 1 if bar.delta > 0 else -1
                    return {
                        'detected': True,
                        'price': bar.close,
                        'direction': direction,
                        'volume': bar.total_volume
                    }
        
        return {'detected': False}
    
    def _detect_exhaustion(self) -> Dict:
        """Detect exhaustion patterns"""
        if len(self._bars) < 5:
            return {'detected': False}
        
        recent = self._bars[-5:]
        
        # Exhaustion: decreasing delta with price continuation
        deltas = [bar.delta for bar in recent]
        prices = [bar.close for bar in recent]
        
        # Price trending but delta weakening
        price_trend = prices[-1] - prices[0]
        delta_trend = deltas[-1] - deltas[0]
        
        if abs(price_trend) > 0:
            # Divergence between price and delta
            if np.sign(price_trend) != np.sign(delta_trend):
                return {
                    'detected': True,
                    'direction': -np.sign(price_trend),  # Reversal
                    'strength': abs(delta_trend) / abs(price_trend) if price_trend != 0 else 0
                }
        
        return {'detected': False}
    
    def _analyze_imbalance(self) -> float:
        """Analyze order book style imbalance from trades"""
        if len(self._trades) < 50:
            return 0
        
        recent = list(self._trades)[-50:]
        
        buy_vol = sum(t.size for t in recent if t.side == 'buy')
        sell_vol = sum(t.size for t in recent if t.side == 'sell')
        total = buy_vol + sell_vol
        
        if total > 0:
            return (buy_vol - sell_vol) / total
        return 0
    
    def _check_bar_absorption(self, bar: FootprintBar) -> bool:
        """Check if bar shows absorption"""
        if bar.total_volume == 0:
            return False
        
        body = abs(bar.close - bar.open)
        range_size = bar.high - bar.low
        
        if range_size > 0:
            efficiency = body / range_size
            return efficiency < 0.3 and bar.total_volume > self._avg_trade_size * 50
        return False
    
    def _check_bar_exhaustion(self, bar: FootprintBar) -> bool:
        """Check if bar shows exhaustion"""
        if len(self._bars) < 3:
            return False
        
        prev_bars = self._bars[-3:-1]
        avg_delta = np.mean([b.delta for b in prev_bars])
        
        # Current delta much smaller than recent average
        if abs(avg_delta) > 0:
            delta_ratio = bar.delta / avg_delta
            return delta_ratio < 0.3  # Less than 30% of recent average
        
        return False
    
    def _classify_flow(
        self,
        cvd_signal: float,
        delta_signal: float,
        absorption: Dict,
        exhaustion: Dict
    ) -> FlowType:
        """Classify the type of order flow"""
        if absorption.get('detected'):
            return FlowType.ABSORPTION
        
        if exhaustion.get('detected'):
            return FlowType.EXHAUSTION
        
        if abs(cvd_signal) > 0.5 and np.sign(cvd_signal) == np.sign(delta_signal):
            return FlowType.MOMENTUM
        
        if abs(cvd_signal) > 0.3 and np.sign(cvd_signal) != np.sign(delta_signal):
            return FlowType.REVERSAL
        
        if cvd_signal > 0.2 and delta_signal < 0.1:
            return FlowType.ACCUMULATION
        
        if cvd_signal < -0.2 and delta_signal > -0.1:
            return FlowType.DISTRIBUTION
        
        return FlowType.MOMENTUM
    
    def _calculate_atr(self, period: int = 14) -> float:
        """Calculate ATR from bars"""
        if len(self._bars) < period:
            return 0
        
        recent = self._bars[-period:]
        trs = []
        
        for i, bar in enumerate(recent):
            if i == 0:
                tr = bar.high - bar.low
            else:
                prev = recent[i-1]
                tr = max(
                    bar.high - bar.low,
                    abs(bar.high - prev.close),
                    abs(bar.low - prev.close)
                )
            trs.append(tr)
        
        return np.mean(trs) if trs else 0
    
    def _neutral_signal(self) -> OrderFlowSignal:
        """Return neutral signal"""
        return OrderFlowSignal(
            direction=0,
            strength=0,
            flow_type=FlowType.MOMENTUM,
            confidence=0,
            supporting_factors=[],
            whale_activity="Insufficient data",
            entry_price=0,
            stop_loss=0,
            take_profit=0
        )
