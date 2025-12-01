"""
Technical Indicators - Pure numpy implementations for speed

Common indicators used in trading strategies:
- Moving averages (EMA, SMA)
- Momentum indicators (RSI, Momentum, MACD)
- Volatility indicators (ATR, Bollinger Bands)
- Oscillators (Stochastic)
"""

import numpy as np
from typing import Tuple, Optional


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average
    
    Args:
        data: Price array
        period: Lookback period
    
    Returns:
        SMA values (NaN for first period-1 values)
    """
    if len(data) < period:
        return np.full(len(data), np.nan)
    
    result = np.full(len(data), np.nan)
    cumsum = np.cumsum(data)
    result[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period
    
    return result


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average
    
    Args:
        data: Price array
        period: Lookback period
    
    Returns:
        EMA values
    """
    if len(data) < period:
        return np.full(len(data), np.nan)
    
    alpha = 2 / (period + 1)
    result = np.zeros(len(data))
    
    # Initialize with SMA
    result[period-1] = np.mean(data[:period])
    
    # Calculate EMA
    for i in range(period, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    # Set initial values to NaN
    result[:period-1] = np.nan
    
    return result


def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index
    
    Args:
        data: Price array (closes)
        period: RSI period (default 14)
    
    Returns:
        RSI values (0-100)
    """
    if len(data) < period + 1:
        return np.full(len(data), np.nan)
    
    # Calculate price changes
    delta = np.diff(data)
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Calculate average gains and losses
    avg_gain = np.zeros(len(data))
    avg_loss = np.zeros(len(data))
    
    # Initial average (simple)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Smoothed average
    for i in range(period + 1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RSI
    rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain))
    rsi_values = 100 - (100 / (1 + rs))
    
    # Set initial values to NaN
    rsi_values[:period] = np.nan
    
    return rsi_values


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
    
    Returns:
        ATR values
    """
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    # Calculate True Range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate ATR (smoothed)
    atr_values = np.zeros(len(high))
    atr_values[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(high)):
        atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period
    
    atr_values[:period-1] = np.nan
    
    return atr_values


def momentum(data: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Price Momentum (rate of change)
    
    Args:
        data: Price array
        period: Lookback period
    
    Returns:
        Momentum as percentage change
    """
    if len(data) < period + 1:
        return np.full(len(data), np.nan)
    
    result = np.full(len(data), np.nan)
    result[period:] = (data[period:] - data[:-period]) / data[:-period]
    
    return result


def bollinger_bands(
    data: np.ndarray, 
    period: int = 20, 
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands
    
    Args:
        data: Price array
        period: SMA period
        num_std: Number of standard deviations
    
    Returns:
        (upper_band, middle_band, lower_band)
    """
    middle = sma(data, period)
    
    # Calculate rolling std
    std = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        std[i] = np.std(data[i-period+1:i+1])
    
    upper = middle + num_std * std
    lower = middle - num_std * std
    
    return upper, middle, lower


def macd(
    data: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence)
    
    Args:
        data: Price array
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
    
    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = ema(data, fast_period)
    ema_slow = ema(data, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period (smoothing)
    
    Returns:
        (%K, %D)
    """
    if len(high) < k_period:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    # Calculate %K
    k = np.full(len(high), np.nan)
    
    for i in range(k_period - 1, len(high)):
        highest = np.max(high[i-k_period+1:i+1])
        lowest = np.min(low[i-k_period+1:i+1])
        
        if highest != lowest:
            k[i] = 100 * (close[i] - lowest) / (highest - lowest)
        else:
            k[i] = 50
    
    # Calculate %D (SMA of %K)
    d = sma(k, d_period)
    
    return k, d


def vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray
) -> np.ndarray:
    """
    Volume Weighted Average Price
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
    
    Returns:
        VWAP values
    """
    typical_price = (high + low + close) / 3
    cumulative_tpv = np.cumsum(typical_price * volume)
    cumulative_vol = np.cumsum(volume)
    
    return np.divide(
        cumulative_tpv, 
        cumulative_vol, 
        where=cumulative_vol != 0,
        out=np.zeros_like(cumulative_tpv)
    )


def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
    
    Returns:
        (ADX, +DI, -DI)
    """
    if len(high) < period + 1:
        nan_arr = np.full(len(high), np.nan)
        return nan_arr, nan_arr, nan_arr
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    
    for i in range(1, len(high)):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    # Calculate TR
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Smooth with Wilder's method
    smoothed_tr = np.zeros(len(high))
    smoothed_plus_dm = np.zeros(len(high))
    smoothed_minus_dm = np.zeros(len(high))
    
    smoothed_tr[period] = np.sum(tr[1:period+1])
    smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period + 1, len(high)):
        smoothed_tr[i] = smoothed_tr[i-1] - smoothed_tr[i-1]/period + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - smoothed_plus_dm[i-1]/period + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - smoothed_minus_dm[i-1]/period + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = np.zeros(len(high))
    minus_di = np.zeros(len(high))
    
    for i in range(period, len(high)):
        if smoothed_tr[i] != 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(len(high))
    for i in range(period, len(high)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is smoothed DX
    adx_values = np.zeros(len(high))
    adx_values[2*period-1] = np.mean(dx[period:2*period])
    
    for i in range(2*period, len(high)):
        adx_values[i] = (adx_values[i-1] * (period - 1) + dx[i]) / period
    
    # Set NaN for warmup
    plus_di[:period] = np.nan
    minus_di[:period] = np.nan
    adx_values[:2*period-1] = np.nan
    
    return adx_values, plus_di, minus_di


def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supertrend Indicator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        multiplier: ATR multiplier
    
    Returns:
        (supertrend, direction) where direction is 1 (up) or -1 (down)
    """
    atr_values = atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr_values
    lower_band = hl2 - multiplier * atr_values
    
    supertrend = np.zeros(len(close))
    direction = np.zeros(len(close))
    
    supertrend[period-1] = upper_band[period-1]
    direction[period-1] = -1
    
    for i in range(period, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        
        # Adjust bands
        if direction[i] == 1 and lower_band[i] < supertrend[i-1]:
            supertrend[i] = supertrend[i-1]
        if direction[i] == -1 and upper_band[i] > supertrend[i-1]:
            supertrend[i] = supertrend[i-1]
    
    supertrend[:period-1] = np.nan
    direction[:period-1] = 0
    
    return supertrend, direction
