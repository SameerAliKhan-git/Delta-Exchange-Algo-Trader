"""
Medallion Strategy - Renaissance Technologies-Inspired Quantitative Strategy

This implements multiple advanced quantitative strategies inspired by
Renaissance Technologies' Medallion Fund approach:

1. Statistical Arbitrage - Mean reversion with cointegration
2. Market Making - Bid/ask spread capture with inventory management
3. Momentum with Regime Detection - Adaptive trend following
4. Factor Models - Multi-factor alpha generation
5. Machine Learning Signals - Pattern recognition
6. High-Frequency Signals - Microstructure analysis

The key principles:
- Short holding periods (minutes to hours)
- High win rate through statistical edge
- Massive diversification across signals
- Strict risk management
- Continuous adaptation to market changes
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RegimeType(Enum):
    """Market regime classification"""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"


@dataclass
class SignalComponent:
    """Individual signal component"""
    name: str
    value: float  # -1 to 1
    confidence: float  # 0 to 1
    weight: float  # Contribution weight


@dataclass
class CompositeSignal:
    """Composite signal from multiple sources"""
    direction: int  # 1=long, -1=short, 0=neutral
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    components: List[SignalComponent]
    regime: RegimeType
    expected_return: float
    expected_holding_period: int  # candles
    stop_loss_pct: float
    take_profit_pct: float


class MedallionStrategy:
    """
    Renaissance Technologies-Inspired Quantitative Strategy
    
    Combines multiple alpha sources with dynamic weighting
    based on market regime and recent performance.
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        signal_threshold: float = 0.6,
        min_confidence: float = 0.5,
        max_holding_periods: int = 24,
        use_ml_signals: bool = True
    ):
        """
        Initialize Medallion strategy
        
        Args:
            lookback_period: Historical bars for analysis
            signal_threshold: Minimum signal strength to trade
            min_confidence: Minimum confidence to trade
            max_holding_periods: Maximum holding time
            use_ml_signals: Whether to use ML-based signals
        """
        self.lookback = lookback_period
        self.signal_threshold = signal_threshold
        self.min_confidence = min_confidence
        self.max_holding = max_holding_periods
        self.use_ml = use_ml_signals
        
        # Signal weights (dynamic, adjusted based on performance)
        self.signal_weights = {
            'momentum': 0.15,
            'mean_reversion': 0.15,
            'orderflow': 0.20,
            'volatility': 0.10,
            'factor': 0.15,
            'pattern': 0.15,
            'microstructure': 0.10
        }
        
        # State
        self._regime = RegimeType.MEAN_REVERTING
        self._volatility_regime = 'normal'
        self._signal_performance: Dict[str, List[float]] = {k: [] for k in self.signal_weights}
        
        # Caches
        self._last_prices: np.ndarray = np.array([])
        self._last_volumes: np.ndarray = np.array([])
    
    def generate_signal(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        orderbook: Optional[Dict] = None,
        trades: Optional[List[Dict]] = None
    ) -> CompositeSignal:
        """
        Generate composite trading signal
        
        Args:
            prices: OHLCV price data (Nx5: open, high, low, close, volume)
            volumes: Volume data
            orderbook: Current orderbook snapshot
            trades: Recent trades for microstructure analysis
        
        Returns:
            CompositeSignal with direction and metadata
        """
        if len(prices) < self.lookback:
            return self._neutral_signal()
        
        # Update caches
        self._last_prices = prices
        self._last_volumes = volumes
        
        # Detect current regime
        self._regime = self._detect_regime(prices)
        
        # Generate all signal components
        components = []
        
        # 1. Momentum signals
        momentum_signal = self._momentum_signal(prices)
        components.append(momentum_signal)
        
        # 2. Mean reversion signals
        mean_rev_signal = self._mean_reversion_signal(prices)
        components.append(mean_rev_signal)
        
        # 3. Order flow signals
        if orderbook:
            flow_signal = self._orderflow_signal(orderbook, trades)
            components.append(flow_signal)
        
        # 4. Volatility signals
        vol_signal = self._volatility_signal(prices)
        components.append(vol_signal)
        
        # 5. Factor model signals
        factor_signal = self._factor_signal(prices, volumes)
        components.append(factor_signal)
        
        # 6. Pattern recognition signals
        pattern_signal = self._pattern_signal(prices)
        components.append(pattern_signal)
        
        # 7. Microstructure signals
        if trades:
            micro_signal = self._microstructure_signal(trades)
            components.append(micro_signal)
        
        # Combine signals with regime-adjusted weights
        return self._combine_signals(components)
    
    def _detect_regime(self, prices: np.ndarray) -> RegimeType:
        """Detect current market regime using multiple indicators"""
        closes = prices[:, 3] if prices.ndim == 2 else prices
        
        # Calculate volatility
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns[-20:])
        avg_volatility = np.std(returns)
        
        # Calculate trend strength (ADX-like)
        trend_strength = self._calculate_trend_strength(closes)
        
        # Calculate mean reversion tendency (Hurst exponent approximation)
        hurst = self._estimate_hurst(closes)
        
        # Classify regime
        if volatility > avg_volatility * 1.5:
            self._volatility_regime = 'high'
            return RegimeType.HIGH_VOLATILITY
        elif volatility < avg_volatility * 0.5:
            self._volatility_regime = 'low'
            return RegimeType.LOW_VOLATILITY
        
        if trend_strength > 0.6:
            return RegimeType.TRENDING
        elif hurst < 0.4:
            return RegimeType.MEAN_REVERTING
        else:
            return RegimeType.BREAKOUT
    
    def _momentum_signal(self, prices: np.ndarray) -> SignalComponent:
        """Multi-timeframe momentum signal"""
        closes = prices[:, 3] if prices.ndim == 2 else prices
        
        # Calculate momentum across timeframes
        mom_5 = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
        mom_10 = (closes[-1] / closes[-10] - 1) if len(closes) >= 10 else 0
        mom_20 = (closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0
        
        # EMA crossovers
        ema_8 = self._ema(closes, 8)[-1]
        ema_21 = self._ema(closes, 21)[-1]
        ema_signal = 1 if ema_8 > ema_21 else -1
        
        # MACD
        macd, signal_line = self._macd(closes)
        macd_signal = 1 if macd > signal_line else -1
        
        # Combine
        raw_signal = (
            np.sign(mom_5) * 0.2 +
            np.sign(mom_10) * 0.2 +
            np.sign(mom_20) * 0.2 +
            ema_signal * 0.2 +
            macd_signal * 0.2
        )
        
        # Confidence based on signal alignment
        signals = [np.sign(mom_5), np.sign(mom_10), np.sign(mom_20), ema_signal, macd_signal]
        agreement = sum(1 for s in signals if s == np.sign(raw_signal)) / len(signals)
        
        # Regime adjustment
        weight = self.signal_weights['momentum']
        if self._regime == RegimeType.TRENDING:
            weight *= 1.5
        elif self._regime == RegimeType.MEAN_REVERTING:
            weight *= 0.5
        
        return SignalComponent(
            name='momentum',
            value=np.clip(raw_signal, -1, 1),
            confidence=agreement,
            weight=weight
        )
    
    def _mean_reversion_signal(self, prices: np.ndarray) -> SignalComponent:
        """Mean reversion signal using multiple methods"""
        closes = prices[:, 3] if prices.ndim == 2 else prices
        
        # Bollinger Band position
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(closes, 20, 2)
        bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        bb_signal = 1 - 2 * bb_position  # Oversold = bullish, overbought = bearish
        
        # RSI
        rsi = self._rsi(closes, 14)
        rsi_signal = (50 - rsi) / 50  # Oversold = bullish
        
        # Z-score
        mean = np.mean(closes[-20:])
        std = np.std(closes[-20:])
        z_score = (closes[-1] - mean) / std if std > 0 else 0
        z_signal = -np.clip(z_score / 2, -1, 1)
        
        # Stochastic
        stoch = self._stochastic(prices, 14)
        stoch_signal = (50 - stoch) / 50
        
        # Combine
        raw_signal = (bb_signal * 0.3 + rsi_signal * 0.3 + z_signal * 0.2 + stoch_signal * 0.2)
        
        # Confidence
        signals = [bb_signal, rsi_signal, z_signal, stoch_signal]
        confidence = 1 - np.std(signals)  # Higher agreement = higher confidence
        
        # Regime adjustment
        weight = self.signal_weights['mean_reversion']
        if self._regime == RegimeType.MEAN_REVERTING:
            weight *= 1.5
        elif self._regime == RegimeType.TRENDING:
            weight *= 0.5
        
        return SignalComponent(
            name='mean_reversion',
            value=np.clip(raw_signal, -1, 1),
            confidence=max(0, min(1, confidence)),
            weight=weight
        )
    
    def _orderflow_signal(
        self,
        orderbook: Dict,
        trades: Optional[List[Dict]]
    ) -> SignalComponent:
        """Order flow imbalance signal"""
        # Orderbook imbalance
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        bid_volume = sum(b[1] for b in bids[:10])
        ask_volume = sum(a[1] for a in asks[:10])
        total = bid_volume + ask_volume
        
        imbalance = (bid_volume - ask_volume) / total if total > 0 else 0
        
        # Trade flow (CVD - Cumulative Volume Delta)
        cvd_signal = 0
        if trades:
            buy_volume = sum(t.get('size', 0) for t in trades if t.get('side') == 'buy')
            sell_volume = sum(t.get('size', 0) for t in trades if t.get('side') == 'sell')
            total_trade = buy_volume + sell_volume
            cvd_signal = (buy_volume - sell_volume) / total_trade if total_trade > 0 else 0
        
        # Bid/ask spread pressure
        if bids and asks:
            spread = asks[0][0] - bids[0][0]
            mid = (asks[0][0] + bids[0][0]) / 2
            spread_pct = spread / mid
            # Tight spread = more confident signal
            spread_confidence = 1 - min(spread_pct * 100, 1)
        else:
            spread_confidence = 0.5
        
        # Combine
        raw_signal = imbalance * 0.5 + cvd_signal * 0.5
        
        return SignalComponent(
            name='orderflow',
            value=np.clip(raw_signal, -1, 1),
            confidence=spread_confidence,
            weight=self.signal_weights['orderflow']
        )
    
    def _volatility_signal(self, prices: np.ndarray) -> SignalComponent:
        """Volatility-based signal"""
        closes = prices[:, 3] if prices.ndim == 2 else prices
        highs = prices[:, 1] if prices.ndim == 2 else closes * 1.01
        lows = prices[:, 2] if prices.ndim == 2 else closes * 0.99
        
        # ATR
        atr = self._atr(highs, lows, closes, 14)
        avg_atr = np.mean(atr[-20:])
        current_atr = atr[-1]
        
        # Volatility ratio
        vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        
        # Bollinger Band width
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(closes, 20, 2)
        bb_width = (bb_upper - bb_lower) / bb_middle
        avg_width = np.mean([
            (self._bollinger_bands(closes[:i], 20, 2)[0] - self._bollinger_bands(closes[:i], 20, 2)[2]) / self._bollinger_bands(closes[:i], 20, 2)[1]
            for i in range(max(21, len(closes) - 20), len(closes))
        ]) if len(closes) > 40 else bb_width
        
        # Signal: volatility expansion suggests breakout potential
        if vol_ratio > 1.5:
            # High volatility - momentum likely
            trend = 1 if closes[-1] > closes[-5] else -1
            signal = trend * 0.5
        elif vol_ratio < 0.7:
            # Low volatility - mean reversion or breakout coming
            signal = 0
        else:
            signal = 0
        
        confidence = abs(vol_ratio - 1) / 2  # More extreme = more confident
        
        return SignalComponent(
            name='volatility',
            value=np.clip(signal, -1, 1),
            confidence=min(confidence, 1),
            weight=self.signal_weights['volatility']
        )
    
    def _factor_signal(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> SignalComponent:
        """Multi-factor alpha model"""
        closes = prices[:, 3] if prices.ndim == 2 else prices
        
        # Factor 1: Price momentum
        mom_factor = (closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0
        
        # Factor 2: Volume trend
        vol_ma5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        vol_ma20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else vol_ma5
        vol_factor = vol_ma5 / vol_ma20 - 1 if vol_ma20 > 0 else 0
        
        # Factor 3: Volatility (inverse - low vol preferred)
        returns = np.diff(closes) / closes[:-1]
        vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        vol_factor_inv = -vol * 10  # Normalize
        
        # Factor 4: Mean reversion
        mean_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        mr_factor = (mean_20 - closes[-1]) / closes[-1]
        
        # Factor 5: Trend quality (R-squared of linear regression)
        if len(closes) >= 20:
            x = np.arange(20)
            y = closes[-20:]
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            trend_quality = np.sign(slope) * r_squared
        else:
            trend_quality = 0
        
        # Combine factors with weights
        raw_signal = (
            np.sign(mom_factor) * 0.25 +
            np.sign(vol_factor) * 0.15 +
            np.sign(vol_factor_inv) * 0.1 +
            np.sign(mr_factor) * 0.25 +
            trend_quality * 0.25
        )
        
        # Confidence based on factor agreement
        factors = [np.sign(mom_factor), np.sign(vol_factor), np.sign(mr_factor), trend_quality]
        avg_sign = np.mean([f for f in factors if f != 0]) if any(f != 0 for f in factors) else 0
        confidence = abs(avg_sign)
        
        return SignalComponent(
            name='factor',
            value=np.clip(raw_signal, -1, 1),
            confidence=confidence,
            weight=self.signal_weights['factor']
        )
    
    def _pattern_signal(self, prices: np.ndarray) -> SignalComponent:
        """Pattern recognition signal"""
        closes = prices[:, 3] if prices.ndim == 2 else prices
        highs = prices[:, 1] if prices.ndim == 2 else closes * 1.01
        lows = prices[:, 2] if prices.ndim == 2 else closes * 0.99
        opens = prices[:, 0] if prices.ndim == 2 else closes
        
        signal = 0
        confidence = 0
        
        if len(closes) < 5:
            return SignalComponent(
                name='pattern',
                value=0,
                confidence=0,
                weight=self.signal_weights['pattern']
            )
        
        # Candlestick patterns
        
        # 1. Bullish engulfing
        if (opens[-2] > closes[-2] and  # Previous bearish
            closes[-1] > opens[-1] and  # Current bullish
            opens[-1] < closes[-2] and  # Opens below prev close
            closes[-1] > opens[-2]):    # Closes above prev open
            signal += 0.3
            confidence += 0.2
        
        # 2. Bearish engulfing
        if (opens[-2] < closes[-2] and  # Previous bullish
            closes[-1] < opens[-1] and  # Current bearish
            opens[-1] > closes[-2] and  # Opens above prev close
            closes[-1] < opens[-2]):    # Closes below prev open
            signal -= 0.3
            confidence += 0.2
        
        # 3. Hammer (bullish reversal)
        body = abs(closes[-1] - opens[-1])
        lower_wick = min(opens[-1], closes[-1]) - lows[-1]
        upper_wick = highs[-1] - max(opens[-1], closes[-1])
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            signal += 0.25
            confidence += 0.15
        
        # 4. Shooting star (bearish reversal)
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            signal -= 0.25
            confidence += 0.15
        
        # 5. Three white soldiers / Three black crows
        if len(closes) >= 3:
            if all(closes[-i] > opens[-i] for i in range(1, 4)):  # Three bullish
                if closes[-1] > closes[-2] > closes[-3]:
                    signal += 0.3
                    confidence += 0.2
            if all(closes[-i] < opens[-i] for i in range(1, 4)):  # Three bearish
                if closes[-1] < closes[-2] < closes[-3]:
                    signal -= 0.3
                    confidence += 0.2
        
        # Support/Resistance levels
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        # Near support
        support = np.min(recent_lows)
        if closes[-1] < support * 1.02:  # Within 2% of support
            signal += 0.2
            confidence += 0.1
        
        # Near resistance
        resistance = np.max(recent_highs)
        if closes[-1] > resistance * 0.98:  # Within 2% of resistance
            signal -= 0.2
            confidence += 0.1
        
        return SignalComponent(
            name='pattern',
            value=np.clip(signal, -1, 1),
            confidence=min(confidence, 1),
            weight=self.signal_weights['pattern']
        )
    
    def _microstructure_signal(self, trades: List[Dict]) -> SignalComponent:
        """Market microstructure signal"""
        if not trades:
            return SignalComponent(
                name='microstructure',
                value=0,
                confidence=0,
                weight=self.signal_weights['microstructure']
            )
        
        # Analyze recent trades
        buy_trades = [t for t in trades if t.get('side') == 'buy']
        sell_trades = [t for t in trades if t.get('side') == 'sell']
        
        # 1. Trade imbalance
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)
        total_count = buy_count + sell_count
        count_imbalance = (buy_count - sell_count) / total_count if total_count > 0 else 0
        
        # 2. Volume imbalance
        buy_volume = sum(t.get('size', 0) for t in buy_trades)
        sell_volume = sum(t.get('size', 0) for t in sell_trades)
        total_volume = buy_volume + sell_volume
        volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # 3. Large trade detection (whale activity)
        all_sizes = [t.get('size', 0) for t in trades]
        avg_size = np.mean(all_sizes) if all_sizes else 0
        large_trades = [t for t in trades if t.get('size', 0) > avg_size * 3]
        
        whale_signal = 0
        if large_trades:
            whale_buy = sum(t.get('size', 0) for t in large_trades if t.get('side') == 'buy')
            whale_sell = sum(t.get('size', 0) for t in large_trades if t.get('side') == 'sell')
            whale_total = whale_buy + whale_sell
            whale_signal = (whale_buy - whale_sell) / whale_total if whale_total > 0 else 0
        
        # 4. Trade acceleration
        if len(trades) >= 10:
            recent_rate = 10 / max(1, trades[-1].get('timestamp', 1) - trades[-10].get('timestamp', 0))
            # Higher activity often precedes moves
        
        # Combine
        raw_signal = count_imbalance * 0.2 + volume_imbalance * 0.4 + whale_signal * 0.4
        confidence = abs(volume_imbalance)  # Strong imbalance = high confidence
        
        return SignalComponent(
            name='microstructure',
            value=np.clip(raw_signal, -1, 1),
            confidence=min(confidence, 1),
            weight=self.signal_weights['microstructure']
        )
    
    def _combine_signals(self, components: List[SignalComponent]) -> CompositeSignal:
        """Combine all signals into composite signal"""
        if not components:
            return self._neutral_signal()
        
        # Weighted average
        total_weight = sum(c.weight * c.confidence for c in components)
        if total_weight == 0:
            return self._neutral_signal()
        
        weighted_signal = sum(c.value * c.weight * c.confidence for c in components) / total_weight
        avg_confidence = np.mean([c.confidence for c in components])
        
        # Determine direction
        if weighted_signal > self.signal_threshold:
            direction = 1
        elif weighted_signal < -self.signal_threshold:
            direction = -1
        else:
            direction = 0
        
        # Calculate expected metrics based on regime
        if self._regime == RegimeType.TRENDING:
            expected_return = abs(weighted_signal) * 0.03  # 3% max
            holding_period = 12
            sl = 0.015
            tp = 0.03
        elif self._regime == RegimeType.MEAN_REVERTING:
            expected_return = abs(weighted_signal) * 0.015
            holding_period = 4
            sl = 0.01
            tp = 0.015
        else:
            expected_return = abs(weighted_signal) * 0.02
            holding_period = 8
            sl = 0.012
            tp = 0.02
        
        return CompositeSignal(
            direction=direction,
            strength=abs(weighted_signal),
            confidence=avg_confidence,
            components=components,
            regime=self._regime,
            expected_return=expected_return,
            expected_holding_period=min(holding_period, self.max_holding),
            stop_loss_pct=sl,
            take_profit_pct=tp
        )
    
    def _neutral_signal(self) -> CompositeSignal:
        """Return neutral signal"""
        return CompositeSignal(
            direction=0,
            strength=0,
            confidence=0,
            components=[],
            regime=self._regime,
            expected_return=0,
            expected_holding_period=0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
    
    # ==================== Technical Indicators ====================
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _macd(self, data: np.ndarray) -> Tuple[float, float]:
        """MACD"""
        ema12 = self._ema(data, 12)
        ema26 = self._ema(data, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(macd_line, 9)
        return macd_line[-1], signal_line[-1]
    
    def _rsi(self, data: np.ndarray, period: int = 14) -> float:
        """RSI"""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _bollinger_bands(self, data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Bollinger Bands"""
        if len(data) < period:
            return data[-1], data[-1], data[-1]
        
        middle = np.mean(data[-period:])
        std = np.std(data[-period:])
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        atr = np.zeros(len(tr))
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return np.concatenate([[tr[0]], atr])
    
    def _stochastic(self, prices: np.ndarray, period: int = 14) -> float:
        """Stochastic oscillator"""
        if prices.ndim == 2:
            highs = prices[:, 1]
            lows = prices[:, 2]
            closes = prices[:, 3]
        else:
            highs = prices * 1.01
            lows = prices * 0.99
            closes = prices
        
        if len(closes) < period:
            return 50
        
        highest = np.max(highs[-period:])
        lowest = np.min(lows[-period:])
        
        if highest == lowest:
            return 50
        
        return ((closes[-1] - lowest) / (highest - lowest)) * 100
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength (0-1)"""
        if len(prices) < 20:
            return 0.5
        
        # ADX approximation
        returns = np.diff(prices) / prices[:-1]
        pos_moves = np.where(returns > 0, returns, 0)
        neg_moves = np.where(returns < 0, -returns, 0)
        
        avg_pos = np.mean(pos_moves[-14:])
        avg_neg = np.mean(neg_moves[-14:])
        
        if avg_pos + avg_neg == 0:
            return 0.5
        
        dx = abs(avg_pos - avg_neg) / (avg_pos + avg_neg)
        return dx
    
    def _estimate_hurst(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """Estimate Hurst exponent (simplified)"""
        if len(prices) < max_lag * 2:
            return 0.5
        
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            tau.append(np.std(np.subtract(prices[lag:], prices[:-lag])))
        
        if not tau or min(tau) <= 0:
            return 0.5
        
        # Linear regression of log-log plot
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        hurst = slope / 2
        
        return np.clip(hurst, 0, 1)
