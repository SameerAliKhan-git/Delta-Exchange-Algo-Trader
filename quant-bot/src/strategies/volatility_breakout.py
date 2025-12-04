"""
Strategy 2: Volatility Breakout
===============================

Why it works in crypto:
- Crypto has massive intraday expansions after quiet periods
- Volatility clustering is pronounced
- Breakouts tend to follow through more than in traditional markets

This is the "Holy Grail" for many profitable crypto bots.

Concepts:
- Donchian channels
- Range expansion indexing  
- ATR breakout
- Volatility clustering
- Conditional trading (only trade when vol compresses then expands)
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
class BreakoutConfig(StrategyConfig):
    """Configuration for breakout strategies."""
    name: str = "volatility_breakout"
    
    # Donchian settings
    donchian_period: int = 20
    
    # ATR settings
    atr_period: int = 14
    atr_breakout_mult: float = 1.5  # Price move > 1.5x ATR = breakout
    
    # Volatility compression settings
    volatility_lookback: int = 20
    compression_threshold: float = 0.5  # Vol is 50% below average
    expansion_threshold: float = 1.5  # Vol is 150% above average
    
    # Range settings
    range_period: int = 10
    range_breakout_pct: float = 0.02  # 2% breakout threshold
    
    # Keltner squeeze
    use_squeeze: bool = True
    squeeze_bb_period: int = 20
    squeeze_bb_mult: float = 2.0
    squeeze_kc_period: int = 20
    squeeze_kc_mult: float = 1.5
    
    # Volume confirmation
    use_volume_confirm: bool = True
    volume_mult: float = 1.5  # Volume > 1.5x average
    
    # Risk
    atr_stop_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0
    
    # Regime
    allowed_regimes: List[str] = field(default_factory=lambda: ["breakout", "high_volatility"])


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Core volatility breakout strategy.
    
    Logic:
    1. Detect volatility compression (squeeze)
    2. Wait for range breakout with volume confirmation
    3. Enter on expansion with tight stop
    4. Trail stop as momentum continues
    """
    
    def __init__(self, config: Optional[BreakoutConfig] = None):
        super().__init__(config or BreakoutConfig())
        self.config: BreakoutConfig = self.config
        
        # State
        self._in_squeeze = False
        self._squeeze_bars = 0
        self._recent_range_high: Optional[float] = None
        self._recent_range_low: Optional[float] = None
    
    def update(self, data: pd.DataFrame):
        """Update breakout indicators."""
        self.current_bar = len(data) - 1
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume'] if 'volume' in data.columns else pd.Series(np.ones(len(data)))
        
        # ATR
        self._atr = TechnicalIndicators.atr(high, low, close, self.config.atr_period)
        
        # Donchian channels
        self._dc_upper, self._dc_mid, self._dc_lower = TechnicalIndicators.donchian_channels(
            high, low, self.config.donchian_period
        )
        
        # Volatility
        self._returns = close.pct_change()
        self._volatility = self._returns.rolling(self.config.volatility_lookback).std()
        self._vol_ma = self._volatility.rolling(self.config.volatility_lookback * 2).mean()
        self._vol_ratio = self._volatility / self._vol_ma
        
        # Range
        self._range_high = high.rolling(self.config.range_period).max()
        self._range_low = low.rolling(self.config.range_period).min()
        self._range_size = (self._range_high - self._range_low) / close
        
        # Volume
        self._volume_ma = volume.rolling(20).mean()
        self._volume_ratio = volume / self._volume_ma
        
        # Squeeze detection (Bollinger inside Keltner)
        if self.config.use_squeeze:
            bb_upper, bb_mid, bb_lower = TechnicalIndicators.bollinger_bands(
                close, self.config.squeeze_bb_period, self.config.squeeze_bb_mult
            )
            kc_upper, kc_mid, kc_lower = TechnicalIndicators.keltner_channels(
                high, low, close, self.config.squeeze_kc_period, 
                self.config.atr_period, self.config.squeeze_kc_mult
            )
            
            # Squeeze: BB inside KC
            self._squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        else:
            self._squeeze = pd.Series(False, index=close.index)
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate breakout signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        i = -1
        close = data['close'].iloc[i]
        high = data['high'].iloc[i]
        low = data['low'].iloc[i]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        atr = self._atr.iloc[i]
        dc_upper = self._dc_upper.iloc[i]
        dc_lower = self._dc_lower.iloc[i]
        vol_ratio = self._vol_ratio.iloc[i]
        volume_ratio = self._volume_ratio.iloc[i]
        squeeze = self._squeeze.iloc[i]
        squeeze_prev = self._squeeze.iloc[i-1] if i > -len(data) else False
        
        # Update squeeze state
        if squeeze:
            self._squeeze_bars += 1
            self._in_squeeze = True
        else:
            if self._in_squeeze:
                # Squeeze released
                self._in_squeeze = False
        
        signal_type = None
        reason = ""
        strength = 0.0
        
        # Breakout conditions:
        # 1. Was in squeeze, now released
        # 2. OR volatility was compressed, now expanding
        # 3. Price breaks range
        
        breakout_condition = False
        
        # Squeeze release breakout
        if squeeze_prev and not squeeze:
            breakout_condition = True
            reason_prefix = "Squeeze release"
        
        # Volatility expansion breakout
        elif vol_ratio > self.config.expansion_threshold and self._vol_ratio.iloc[i-1] < 1.0:
            breakout_condition = True
            reason_prefix = "Vol expansion"
        
        # Donchian breakout
        elif close > dc_upper:
            breakout_condition = True
            reason_prefix = "Donchian upper"
        elif close < dc_lower:
            breakout_condition = True
            reason_prefix = "Donchian lower"
        
        if not breakout_condition:
            return None
        
        # Volume confirmation
        if self.config.use_volume_confirm:
            if volume_ratio < self.config.volume_mult:
                return None
        
        # Direction based on price action
        if close > dc_upper or (close > data['close'].iloc[i-1] and breakout_condition):
            signal_type = SignalType.LONG
            reason = f"{reason_prefix} LONG, vol_ratio={vol_ratio:.2f}"
        else:
            signal_type = SignalType.SHORT
            reason = f"{reason_prefix} SHORT, vol_ratio={vol_ratio:.2f}"
        
        # Strength based on volatility expansion
        strength = min(1.0, vol_ratio / 2)
        
        # Confidence based on multiple confirmations
        confidence = 0.5
        if volume_ratio > self.config.volume_mult:
            confidence += 0.15
        if self._squeeze_bars > 3:
            confidence += 0.15
        if vol_ratio > 2.0:
            confidence += 0.1
        
        self._squeeze_bars = 0  # Reset
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=confidence,
            atr=atr,
            vol_ratio=vol_ratio,
            volume_ratio=volume_ratio
        )


class DonchianBreakout(BaseStrategy):
    """
    Donchian Channel breakout strategy.
    
    Classic trend-following approach:
    - Long when price breaks above N-period high
    - Short when price breaks below N-period low
    - Exit at middle band or opposite breakout
    """
    
    def __init__(self, config: Optional[BreakoutConfig] = None):
        config = config or BreakoutConfig(name="donchian_breakout")
        super().__init__(config)
        
        self.entry_period = 20
        self.exit_period = 10
    
    def update(self, data: pd.DataFrame):
        """Update Donchian channels."""
        self.current_bar = len(data) - 1
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Entry channels
        self._entry_upper = high.rolling(self.entry_period).max()
        self._entry_lower = low.rolling(self.entry_period).min()
        
        # Exit channels (shorter period)
        self._exit_upper = high.rolling(self.exit_period).max()
        self._exit_lower = low.rolling(self.exit_period).min()
        
        self._atr = TechnicalIndicators.atr(high, low, close, 14)
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate Donchian breakout signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        close = data['close'].iloc[-1]
        high = data['high'].iloc[-1]
        low = data['low'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        entry_upper = self._entry_upper.iloc[-2]  # Use previous bar's channel
        entry_lower = self._entry_lower.iloc[-2]
        atr = self._atr.iloc[-1]
        
        signal_type = None
        reason = ""
        
        # Long: Close breaks above upper channel
        if close > entry_upper and prev_close <= entry_upper:
            signal_type = SignalType.LONG
            reason = f"Price broke above {self.entry_period}-bar high at {entry_upper:.2f}"
        
        # Short: Close breaks below lower channel
        elif close < entry_lower and prev_close >= entry_lower:
            signal_type = SignalType.SHORT
            reason = f"Price broke below {self.entry_period}-bar low at {entry_lower:.2f}"
        
        if signal_type is None:
            return None
        
        # Calculate breakout strength (how far past the level)
        if signal_type == SignalType.LONG:
            breakout_pct = (close - entry_upper) / entry_upper
        else:
            breakout_pct = (entry_lower - close) / entry_lower
        
        strength = min(1.0, breakout_pct / 0.02)  # Max at 2% breakout
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=0.65,
            atr=atr
        )


class ATRBreakout(BaseStrategy):
    """
    ATR-based breakout strategy.
    
    Triggers when price moves more than N x ATR from recent close.
    Adapts to volatility automatically.
    """
    
    def __init__(self, config: Optional[BreakoutConfig] = None):
        config = config or BreakoutConfig(name="atr_breakout")
        super().__init__(config)
        
        self.atr_mult = 2.0
        self.lookback = 5
    
    def update(self, data: pd.DataFrame):
        """Update ATR values."""
        self.current_bar = len(data) - 1
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        self._atr = TechnicalIndicators.atr(high, low, close, self.config.atr_period)
        self._ema = TechnicalIndicators.ema(close, 20)
        
        # Reference price (average of last N closes)
        self._ref_price = close.rolling(self.lookback).mean()
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate ATR breakout signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        atr = self._atr.iloc[-1]
        ref_price = self._ref_price.iloc[-1]
        ema = self._ema.iloc[-1]
        
        # Calculate breakout threshold
        upper_threshold = ref_price + self.atr_mult * atr
        lower_threshold = ref_price - self.atr_mult * atr
        
        signal_type = None
        reason = ""
        
        # Long breakout
        if close > upper_threshold and close > ema:
            signal_type = SignalType.LONG
            reason = f"ATR breakout UP: {close:.2f} > {upper_threshold:.2f} (ref + {self.atr_mult}xATR)"
        
        # Short breakout
        elif close < lower_threshold and close < ema:
            signal_type = SignalType.SHORT
            reason = f"ATR breakout DOWN: {close:.2f} < {lower_threshold:.2f} (ref - {self.atr_mult}xATR)"
        
        if signal_type is None:
            return None
        
        # Strength based on ATR multiples moved
        if signal_type == SignalType.LONG:
            atr_moves = (close - ref_price) / atr
        else:
            atr_moves = (ref_price - close) / atr
        
        strength = min(1.0, atr_moves / 3)  # Max at 3 ATR move
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=0.6,
            atr=atr
        )


class RangeExpansion(BaseStrategy):
    """
    Range expansion strategy.
    
    Detects when daily/hourly range expands significantly
    after a period of contraction.
    """
    
    def __init__(self, config: Optional[BreakoutConfig] = None):
        config = config or BreakoutConfig(name="range_expansion")
        super().__init__(config)
        
        self.range_period = 10
        self.expansion_mult = 2.0
        self.contraction_periods = 3
    
    def update(self, data: pd.DataFrame):
        """Update range indicators."""
        self.current_bar = len(data) - 1
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Bar range
        self._bar_range = high - low
        self._avg_range = self._bar_range.rolling(self.range_period).mean()
        self._range_ratio = self._bar_range / self._avg_range
        
        # ATR
        self._atr = TechnicalIndicators.atr(high, low, close, 14)
        
        # Detect contraction (consecutive small ranges)
        small_range = self._bar_range < self._avg_range * 0.7
        self._contraction = small_range.rolling(self.contraction_periods).sum() >= self.contraction_periods
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate range expansion signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        close = data['close'].iloc[-1]
        open_price = data['open'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        range_ratio = self._range_ratio.iloc[-1]
        was_contracted = self._contraction.iloc[-2]
        atr = self._atr.iloc[-1]
        
        # Need prior contraction
        if not was_contracted:
            return None
        
        # Need current expansion
        if range_ratio < self.expansion_mult:
            return None
        
        signal_type = None
        reason = ""
        
        # Direction based on close vs open
        if close > open_price:
            signal_type = SignalType.LONG
            reason = f"Range expansion UP: {range_ratio:.1f}x average after contraction"
        else:
            signal_type = SignalType.SHORT
            reason = f"Range expansion DOWN: {range_ratio:.1f}x average after contraction"
        
        strength = min(1.0, (range_ratio - 1) / 2)
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=0.65,
            atr=atr
        )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("VOLATILITY BREAKOUT STRATEGIES")
    print("="*70)
    
    # Generate sample data with volatility clusters
    np.random.seed(42)
    n = 500
    
    # Volatility regime (alternating low/high)
    vol_regime = np.zeros(n)
    vol_regime[100:150] = 1  # High vol
    vol_regime[250:350] = 1  # High vol
    vol_regime[400:450] = 1  # High vol
    
    returns = np.where(
        vol_regime == 1,
        np.random.randn(n) * 0.03,  # High vol
        np.random.randn(n) * 0.008   # Low vol
    )
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create realistic OHLC
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.003, 0.003, n)),
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.01 * (1 + vol_regime))),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.01 * (1 + vol_regime))),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n) * (1 + vol_regime),
        'symbol': 'BTCUSDT'
    })
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    print("\n1. VOLATILITY BREAKOUT STRATEGY")
    print("-" * 50)
    
    vb = VolatilityBreakoutStrategy(BreakoutConfig(
        use_volume_confirm=False  # For demo
    ))
    vb.update(data)
    
    signals = []
    for i in range(50, len(data)):
        vb.current_bar = i
        signal = vb.generate_signal(data.iloc[:i+1])
        if signal:
            signals.append(signal)
    
    print(f"   Total signals: {len(signals)}")
    
    print("\n2. DONCHIAN BREAKOUT")
    print("-" * 50)
    
    donchian = DonchianBreakout()
    donchian.update(data)
    
    dc_signals = []
    for i in range(50, len(data)):
        donchian.current_bar = i
        signal = donchian.generate_signal(data.iloc[:i+1])
        if signal:
            dc_signals.append(signal)
    
    print(f"   Total signals: {len(dc_signals)}")
    
    print("\n3. ATR BREAKOUT")
    print("-" * 50)
    
    atr_bo = ATRBreakout()
    atr_bo.update(data)
    
    atr_signals = []
    for i in range(50, len(data)):
        atr_bo.current_bar = i
        signal = atr_bo.generate_signal(data.iloc[:i+1])
        if signal:
            atr_signals.append(signal)
    
    print(f"   Total signals: {len(atr_signals)}")
    
    print("\n4. RANGE EXPANSION")
    print("-" * 50)
    
    re = RangeExpansion()
    re.update(data)
    
    re_signals = []
    for i in range(50, len(data)):
        re.current_bar = i
        signal = re.generate_signal(data.iloc[:i+1])
        if signal:
            re_signals.append(signal)
    
    print(f"   Total signals: {len(re_signals)}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS FOR VOLATILITY BREAKOUT")
    print("="*70)
    print("""
1. THE SQUEEZE IS CRUCIAL
   - Bollinger Bands inside Keltner Channels = compression
   - Release of squeeze = explosive move incoming
   - More bars in squeeze = stronger breakout

2. VOLUME CONFIRMS REAL BREAKOUTS
   - Volume spike (>1.5x average) validates breakout
   - Low volume breakout = likely false
   
3. ATR ADAPTS TO CHANGING VOLATILITY
   - Fixed pip/percentage stops FAIL in crypto
   - ATR-based everything (stops, targets, breakout detection)
   
4. WAIT FOR THE SETUP
   - Don't chase already-extended moves
   - Best entries: RIGHT when squeeze releases
   - Worst entries: After 3+ bars of momentum

5. CRYPTO-SPECIFIC:
   - Weekends often show compression → Monday breakouts
   - News events create best volatility clusters
   - BTC leads, alts follow with bigger moves
""")
