"""
Base Strategy Classes
=====================
Foundation classes for all trading strategies.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
    EXIT_LONG = 2
    EXIT_SHORT = -2


@dataclass
class Signal:
    """Trading signal with metadata."""
    signal_type: SignalType
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    timestamp: datetime
    price: float
    symbol: str
    strategy: str
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_entry(self) -> bool:
        return self.signal_type in [SignalType.LONG, SignalType.SHORT]
    
    @property
    def is_exit(self) -> bool:
        return self.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]
    
    @property
    def direction(self) -> int:
        if self.signal_type == SignalType.LONG:
            return 1
        elif self.signal_type == SignalType.SHORT:
            return -1
        return 0


@dataclass
class StrategyConfig:
    """Base configuration for strategies."""
    name: str = "base_strategy"
    enabled: bool = True
    weight: float = 1.0
    max_positions: int = 1
    position_size_pct: float = 0.1  # 10% of portfolio per position
    
    # Risk parameters
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    use_dynamic_stops: bool = True
    atr_stop_multiplier: float = 2.0
    
    # Signal parameters
    min_signal_strength: float = 0.5
    min_confidence: float = 0.6
    
    # Cooldown
    cooldown_bars: int = 5
    
    # Regime filters
    use_regime_filter: bool = True
    allowed_regimes: List[str] = field(default_factory=lambda: ["trending", "breakout"])


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - generate_signal(): Produce trading signals
    - update(): Update internal state with new data
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config or StrategyConfig()
        self.name = self.config.name
        
        # State
        self.is_initialized = False
        self.last_signal: Optional[Signal] = None
        self.last_signal_bar = -999
        self.current_bar = 0
        self.current_regime: Optional[str] = None
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_acted = 0
        self.cumulative_pnl = 0.0
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate trading signal from data.
        
        Args:
            data: OHLCV DataFrame with latest market data
            
        Returns:
            Signal object or None
        """
        pass
    
    @abstractmethod
    def update(self, data: pd.DataFrame):
        """
        Update strategy internal state.
        
        Args:
            data: Latest market data
        """
        pass
    
    def can_generate_signal(self) -> bool:
        """Check if signal generation is allowed."""
        # Check enabled
        if not self.config.enabled:
            return False
        
        # Check cooldown
        bars_since_signal = self.current_bar - self.last_signal_bar
        if bars_since_signal < self.config.cooldown_bars:
            return False
        
        # Check regime
        if self.config.use_regime_filter and self.current_regime:
            if self.current_regime not in self.config.allowed_regimes:
                return False
        
        return True
    
    def set_regime(self, regime: str):
        """Set current market regime."""
        self.current_regime = regime
    
    def calculate_stops(self, entry_price: float, direction: int,
                       atr: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            atr: ATR value for dynamic stops
            
        Returns:
            (stop_loss, take_profit)
        """
        if self.config.use_dynamic_stops and atr is not None:
            stop_distance = atr * self.config.atr_stop_multiplier
            tp_distance = stop_distance * 2  # 2:1 R:R
        else:
            stop_distance = entry_price * self.config.stop_loss_pct
            tp_distance = entry_price * self.config.take_profit_pct
        
        if direction > 0:  # Long
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def _create_signal(self, signal_type: SignalType, price: float,
                      symbol: str, reason: str, strength: float = 0.7,
                      confidence: float = 0.7, atr: Optional[float] = None,
                      **metadata) -> Signal:
        """Helper to create consistent signals."""
        direction = 1 if signal_type == SignalType.LONG else -1
        stop_loss, take_profit = self.calculate_stops(price, direction, atr)
        
        signal = Signal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(),
            price=price,
            symbol=symbol,
            strategy=self.name,
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata
        )
        
        self.last_signal = signal
        self.last_signal_bar = self.current_bar
        self.signals_generated += 1
        
        return signal
    
    def get_performance_stats(self) -> Dict:
        """Get strategy performance statistics."""
        return {
            'name': self.name,
            'signals_generated': self.signals_generated,
            'signals_acted': self.signals_acted,
            'cumulative_pnl': self.cumulative_pnl,
            'current_regime': self.current_regime
        }


# =============================================================================
# COMMON INDICATORS
# =============================================================================

class TechnicalIndicators:
    """Common technical indicators used across strategies."""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26,
            signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20,
                       std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index."""
        # +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                           index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                            index=high.index)
        
        # ATR
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        # Smooth DM
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """Supertrend indicator."""
        atr = TechnicalIndicators.atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            
            # Adjust bands
            if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
        
        return supertrend, direction
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series,
                         period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels."""
        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        middle = (upper + lower) / 2
        return upper, middle, lower
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                        ema_period: int = 20, atr_period: int = 10,
                        multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels."""
        middle = close.ewm(span=ema_period, adjust=False).mean()
        atr = TechnicalIndicators.atr(high, low, close, atr_period)
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        return upper, middle, lower
    
    @staticmethod
    def volatility_index(close: pd.Series, period: int = 20) -> pd.Series:
        """Rolling volatility as annualized percentage."""
        returns = close.pct_change()
        vol = returns.rolling(period).std() * np.sqrt(365 * 24)  # Hourly to annualized
        return vol
    
    @staticmethod
    def volume_profile(close: pd.Series, volume: pd.Series, 
                      period: int = 50, bins: int = 10) -> Dict[str, float]:
        """Volume profile analysis."""
        recent_close = close.iloc[-period:]
        recent_volume = volume.iloc[-period:]
        
        price_range = np.linspace(recent_close.min(), recent_close.max(), bins + 1)
        volume_at_price = []
        
        for i in range(bins):
            mask = (recent_close >= price_range[i]) & (recent_close < price_range[i + 1])
            volume_at_price.append(recent_volume[mask].sum())
        
        max_vol_idx = np.argmax(volume_at_price)
        poc = (price_range[max_vol_idx] + price_range[max_vol_idx + 1]) / 2  # Point of Control
        
        return {
            'poc': poc,
            'value_area_high': price_range[-1],
            'value_area_low': price_range[0],
            'volume_profile': volume_at_price
        }
