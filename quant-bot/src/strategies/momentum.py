"""
Strategy 1: Momentum / Trend-Following
======================================

Why it works in crypto:
- Strong trending behavior
- High retail-driven momentum  
- Weak mean-reversion
- BTC, ETH, SOL, ADA show persistent momentum patterns

Concepts:
- Rolling returns
- Moving averages (5/20, 20/50)
- Volatility breakout
- RSI ≥ 55
- Trend + volatility filters
- ATR-based stop losses
- Regime detection

This captures large crypto trends (10–50%) and ignores choppy noise.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base import (
    BaseStrategy, StrategyConfig, Signal, SignalType,
    TechnicalIndicators
)


@dataclass
class TrendConfig(StrategyConfig):
    """Configuration for trend-following strategies."""
    name: str = "momentum"
    
    # EMA settings
    fast_ema: int = 20
    slow_ema: int = 50
    signal_ema: int = 9
    
    # Trend filters
    adx_period: int = 14
    adx_threshold: float = 25.0  # Only trade when ADX > 25
    rsi_period: int = 14
    rsi_long_threshold: float = 55.0  # RSI > 55 for long
    rsi_short_threshold: float = 45.0  # RSI < 45 for short
    
    # Volatility
    atr_period: int = 14
    volatility_filter: bool = True
    min_volatility: float = 0.01  # Minimum daily vol
    max_volatility: float = 0.15  # Maximum daily vol
    
    # Stops
    use_dynamic_stops: bool = True
    atr_stop_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0  # Risk-reward ratio
    trailing_stop: bool = True
    trailing_atr_mult: float = 2.5
    
    # Momentum confirmation
    momentum_lookback: int = 10
    min_momentum: float = 0.02  # 2% move required
    
    # Regime
    allowed_regimes: List[str] = field(default_factory=lambda: ["trending", "breakout"])


class MomentumStrategy(BaseStrategy):
    """
    Core momentum strategy using EMA crossovers with filters.
    
    Entry Logic:
    1. Fast EMA crosses above Slow EMA (long) or below (short)
    2. ADX > 25 (strong trend)
    3. RSI confirms direction (>55 long, <45 short)
    4. Volatility within acceptable range
    
    Exit Logic:
    1. Opposite crossover
    2. ATR trailing stop
    3. Take profit at 3x risk
    """
    
    def __init__(self, config: Optional[TrendConfig] = None):
        super().__init__(config or TrendConfig())
        self.config: TrendConfig = self.config
        
        # Cached indicators
        self._fast_ema: Optional[pd.Series] = None
        self._slow_ema: Optional[pd.Series] = None
        self._adx: Optional[pd.Series] = None
        self._rsi: Optional[pd.Series] = None
        self._atr: Optional[pd.Series] = None
        
        # State
        self._prev_position = 0  # -1, 0, 1
        self._trailing_stop: Optional[float] = None
    
    def update(self, data: pd.DataFrame):
        """Update indicators with new data."""
        self.current_bar = len(data) - 1
        
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Calculate indicators
        self._fast_ema = TechnicalIndicators.ema(close, self.config.fast_ema)
        self._slow_ema = TechnicalIndicators.ema(close, self.config.slow_ema)
        self._adx, self._plus_di, self._minus_di = TechnicalIndicators.adx(
            high, low, close, self.config.adx_period
        )
        self._rsi = TechnicalIndicators.rsi(close, self.config.rsi_period)
        self._atr = TechnicalIndicators.atr(high, low, close, self.config.atr_period)
        
        # Momentum
        self._momentum = close.pct_change(self.config.momentum_lookback)
        
        # Volatility
        self._volatility = close.pct_change().rolling(20).std() * np.sqrt(24)  # Daily vol
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate momentum signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        # Get latest values
        i = -1
        close = data['close'].iloc[i]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        fast_ema = self._fast_ema.iloc[i]
        slow_ema = self._slow_ema.iloc[i]
        fast_ema_prev = self._fast_ema.iloc[i-1]
        slow_ema_prev = self._slow_ema.iloc[i-1]
        
        adx = self._adx.iloc[i]
        rsi = self._rsi.iloc[i]
        atr = self._atr.iloc[i]
        momentum = self._momentum.iloc[i]
        volatility = self._volatility.iloc[i]
        
        # Check volatility filter
        if self.config.volatility_filter:
            if volatility < self.config.min_volatility or volatility > self.config.max_volatility:
                return None
        
        # Check trend strength
        if adx < self.config.adx_threshold:
            return None
        
        signal_type = None
        reason = ""
        strength = 0.0
        
        # Long signal: Fast crosses above Slow
        if fast_ema > slow_ema and fast_ema_prev <= slow_ema_prev:
            if rsi > self.config.rsi_long_threshold:
                if momentum > self.config.min_momentum:
                    signal_type = SignalType.LONG
                    reason = f"EMA crossover UP, ADX={adx:.1f}, RSI={rsi:.1f}"
                    strength = min(1.0, (adx - 25) / 25 + (rsi - 55) / 45)
        
        # Short signal: Fast crosses below Slow
        elif fast_ema < slow_ema and fast_ema_prev >= slow_ema_prev:
            if rsi < self.config.rsi_short_threshold:
                if momentum < -self.config.min_momentum:
                    signal_type = SignalType.SHORT
                    reason = f"EMA crossover DOWN, ADX={adx:.1f}, RSI={rsi:.1f}"
                    strength = min(1.0, (adx - 25) / 25 + (45 - rsi) / 45)
        
        if signal_type is None:
            return None
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(adx, rsi, momentum, volatility, signal_type)
        
        if confidence < self.config.min_confidence:
            return None
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=confidence,
            atr=atr,
            adx=adx,
            rsi=rsi,
            momentum=momentum
        )
    
    def _calculate_confidence(self, adx: float, rsi: float, momentum: float,
                            volatility: float, signal_type: SignalType) -> float:
        """Calculate signal confidence from multiple factors."""
        scores = []
        
        # ADX score (higher = stronger trend)
        adx_score = min(1.0, (adx - 20) / 30)
        scores.append(adx_score * 0.3)
        
        # RSI score (distance from neutral)
        if signal_type == SignalType.LONG:
            rsi_score = min(1.0, (rsi - 50) / 30)
        else:
            rsi_score = min(1.0, (50 - rsi) / 30)
        scores.append(rsi_score * 0.25)
        
        # Momentum score
        momentum_score = min(1.0, abs(momentum) / 0.05)
        scores.append(momentum_score * 0.25)
        
        # Volatility score (moderate is best)
        optimal_vol = 0.05
        vol_score = 1.0 - min(1.0, abs(volatility - optimal_vol) / optimal_vol)
        scores.append(vol_score * 0.2)
        
        return sum(scores)


class SupertrendStrategy(BaseStrategy):
    """
    Supertrend-based trend following strategy.
    
    Supertrend is particularly effective in crypto because:
    1. It adapts to volatility via ATR
    2. Provides clear entry/exit levels
    3. Works well in trending markets
    """
    
    def __init__(self, config: Optional[TrendConfig] = None):
        config = config or TrendConfig(name="supertrend")
        super().__init__(config)
        
        self.st_period = 10
        self.st_multiplier = 3.0
        
        self._supertrend: Optional[pd.Series] = None
        self._direction: Optional[pd.Series] = None
    
    def update(self, data: pd.DataFrame):
        """Update supertrend indicator."""
        self.current_bar = len(data) - 1
        
        self._supertrend, self._direction = TechnicalIndicators.supertrend(
            data['high'], data['low'], data['close'],
            self.st_period, self.st_multiplier
        )
        
        self._atr = TechnicalIndicators.atr(
            data['high'], data['low'], data['close'], 14
        )
        
        # Add ADX filter
        self._adx, _, _ = TechnicalIndicators.adx(
            data['high'], data['low'], data['close'], 14
        )
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate supertrend signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        current_dir = self._direction.iloc[-1]
        prev_dir = self._direction.iloc[-2]
        adx = self._adx.iloc[-1]
        atr = self._atr.iloc[-1]
        supertrend = self._supertrend.iloc[-1]
        
        # Only trade strong trends
        if adx < 25:
            return None
        
        signal_type = None
        reason = ""
        
        # Direction flip from -1 to 1 = Long
        if current_dir == 1 and prev_dir == -1:
            signal_type = SignalType.LONG
            reason = f"Supertrend flip UP at {supertrend:.2f}, ADX={adx:.1f}"
        
        # Direction flip from 1 to -1 = Short
        elif current_dir == -1 and prev_dir == 1:
            signal_type = SignalType.SHORT
            reason = f"Supertrend flip DOWN at {supertrend:.2f}, ADX={adx:.1f}"
        
        if signal_type is None:
            return None
        
        strength = min(1.0, (adx - 25) / 25)
        confidence = 0.5 + strength * 0.3
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=confidence,
            atr=atr,
            supertrend=supertrend
        )


class EMAcrossoverStrategy(BaseStrategy):
    """
    Classic EMA crossover with noise reduction.
    
    Uses multiple EMAs and requires price to be on the right side
    of ALL moving averages for confirmation.
    """
    
    def __init__(self, config: Optional[TrendConfig] = None):
        config = config or TrendConfig(name="ema_crossover")
        super().__init__(config)
        
        self.ema_periods = [5, 10, 20, 50]
        self._emas: Dict[int, pd.Series] = {}
    
    def update(self, data: pd.DataFrame):
        """Update EMA values."""
        self.current_bar = len(data) - 1
        
        close = data['close']
        
        for period in self.ema_periods:
            self._emas[period] = TechnicalIndicators.ema(close, period)
        
        self._atr = TechnicalIndicators.atr(
            data['high'], data['low'], data['close'], 14
        )
        
        self._volatility = close.pct_change().rolling(20).std()
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate EMA crossover signal with noise reduction."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        atr = self._atr.iloc[-1]
        
        # Get current EMA values
        ema_values = {p: self._emas[p].iloc[-1] for p in self.ema_periods}
        ema_values_prev = {p: self._emas[p].iloc[-2] for p in self.ema_periods}
        
        # Check if EMAs are properly stacked
        # Long: price > EMA5 > EMA10 > EMA20 > EMA50
        # Short: price < EMA5 < EMA10 < EMA20 < EMA50
        
        long_stack = all(
            close > ema_values[5] > ema_values[10] > ema_values[20] > ema_values[50]
            for _ in [1]
        )
        
        short_stack = all(
            close < ema_values[5] < ema_values[10] < ema_values[20] < ema_values[50]
            for _ in [1]
        )
        
        # Check for crossover (EMA5 crossing EMA10)
        ema5_cross_up = ema_values[5] > ema_values[10] and ema_values_prev[5] <= ema_values_prev[10]
        ema5_cross_down = ema_values[5] < ema_values[10] and ema_values_prev[5] >= ema_values_prev[10]
        
        signal_type = None
        reason = ""
        
        if ema5_cross_up and long_stack:
            signal_type = SignalType.LONG
            reason = f"EMA5 crossed above EMA10, bullish stack confirmed"
        
        elif ema5_cross_down and short_stack:
            signal_type = SignalType.SHORT
            reason = f"EMA5 crossed below EMA10, bearish stack confirmed"
        
        if signal_type is None:
            return None
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=0.7,
            confidence=0.65,
            atr=atr
        )


class ADXTrendStrategy(BaseStrategy):
    """
    ADX-based trend trading strategy.
    
    Only takes trades when ADX indicates a strong trend,
    then uses +DI/-DI crossovers for direction.
    """
    
    def __init__(self, config: Optional[TrendConfig] = None):
        config = config or TrendConfig(name="adx_trend")
        super().__init__(config)
        
        self._adx: Optional[pd.Series] = None
        self._plus_di: Optional[pd.Series] = None
        self._minus_di: Optional[pd.Series] = None
    
    def update(self, data: pd.DataFrame):
        """Update ADX indicators."""
        self.current_bar = len(data) - 1
        
        self._adx, self._plus_di, self._minus_di = TechnicalIndicators.adx(
            data['high'], data['low'], data['close'], self.config.adx_period
        )
        
        self._atr = TechnicalIndicators.atr(
            data['high'], data['low'], data['close'], 14
        )
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate ADX trend signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        adx = self._adx.iloc[-1]
        plus_di = self._plus_di.iloc[-1]
        minus_di = self._minus_di.iloc[-1]
        plus_di_prev = self._plus_di.iloc[-2]
        minus_di_prev = self._minus_di.iloc[-2]
        atr = self._atr.iloc[-1]
        
        # Only trade strong trends
        if adx < self.config.adx_threshold:
            return None
        
        signal_type = None
        reason = ""
        
        # +DI crosses above -DI = Long
        if plus_di > minus_di and plus_di_prev <= minus_di_prev:
            signal_type = SignalType.LONG
            reason = f"+DI crossed above -DI, ADX={adx:.1f}"
        
        # -DI crosses above +DI = Short
        elif minus_di > plus_di and minus_di_prev <= plus_di_prev:
            signal_type = SignalType.SHORT
            reason = f"-DI crossed above +DI, ADX={adx:.1f}"
        
        if signal_type is None:
            return None
        
        strength = min(1.0, (adx - 25) / 25)
        confidence = 0.5 + strength * 0.3
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=confidence,
            atr=atr,
            adx=adx
        )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MOMENTUM / TREND-FOLLOWING STRATEGIES")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    n = 500
    
    # Trending market simulation
    trend = np.cumsum(np.random.randn(n) * 0.02 + 0.001)  # Slight uptrend
    noise = np.random.randn(n) * 0.01
    prices = 100 * np.exp(trend + noise)
    
    data = pd.DataFrame({
        'open': prices * (1 - np.random.uniform(0, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n)),
        'low': prices * (1 - np.random.uniform(0, 0.01, n)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n),
        'symbol': 'BTCUSDT'
    })
    
    print("\n1. MOMENTUM STRATEGY")
    print("-" * 50)
    
    momentum = MomentumStrategy(TrendConfig(
        fast_ema=20,
        slow_ema=50,
        adx_threshold=25.0,
        volatility_filter=False  # For demo
    ))
    
    momentum.update(data)
    
    signals = []
    for i in range(100, len(data)):
        momentum.current_bar = i
        signal = momentum.generate_signal(data.iloc[:i+1])
        if signal:
            signals.append(signal)
    
    print(f"   Total signals: {len(signals)}")
    if signals:
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        print(f"   Long signals: {len(long_signals)}")
        print(f"   Short signals: {len(short_signals)}")
        print(f"   Last signal: {signals[-1].reason}")
    
    print("\n2. SUPERTREND STRATEGY")
    print("-" * 50)
    
    supertrend = SupertrendStrategy()
    supertrend.update(data)
    
    st_signals = []
    for i in range(100, len(data)):
        supertrend.current_bar = i
        signal = supertrend.generate_signal(data.iloc[:i+1])
        if signal:
            st_signals.append(signal)
    
    print(f"   Total signals: {len(st_signals)}")
    if st_signals:
        print(f"   Last signal: {st_signals[-1].reason}")
    
    print("\n3. EMA CROSSOVER STRATEGY")
    print("-" * 50)
    
    ema_cross = EMAcrossoverStrategy()
    ema_cross.update(data)
    
    ema_signals = []
    for i in range(100, len(data)):
        ema_cross.current_bar = i
        signal = ema_cross.generate_signal(data.iloc[:i+1])
        if signal:
            ema_signals.append(signal)
    
    print(f"   Total signals: {len(ema_signals)}")
    
    print("\n4. ADX TREND STRATEGY")
    print("-" * 50)
    
    adx_strat = ADXTrendStrategy()
    adx_strat.update(data)
    
    adx_signals = []
    for i in range(100, len(data)):
        adx_strat.current_bar = i
        signal = adx_strat.generate_signal(data.iloc[:i+1])
        if signal:
            adx_signals.append(signal)
    
    print(f"   Total signals: {len(adx_signals)}")
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS FOR CRYPTO MOMENTUM")
    print("="*70)
    print("""
1. ADX Filter is CRITICAL
   - Only trade when ADX > 25
   - Crypto has long choppy periods - avoid them
   
2. Use ATR-Based Stops, NOT Fixed Percentages
   - Crypto volatility varies 10x between periods
   - Fixed 2% stops will fail
   
3. RSI Confirmation Prevents Fakeouts
   - Long: RSI > 55 (bullish momentum)
   - Short: RSI < 45 (bearish momentum)
   
4. Multiple Timeframe Confirmation
   - Check trend on higher timeframe
   - Enter on lower timeframe

5. Best Crypto Pairs for Momentum:
   - BTC: Primary trend leader
   - ETH: Strong momentum characteristics
   - SOL: High momentum, higher volatility
   - BNB: More stable trends
""")
