"""
Feature Engineering Module

Implements financial features including:
- Technical indicators
- Microstructure features
- Statistical features
- AFML features (fractional differentiation, CUSUM)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a dummy decorator if numba is not available
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator
    logger.warning("Numba not available. Performance may be reduced.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using pure Python implementations.")


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lookback_windows: List[int] = None
    volatility_window: int = 20
    volume_window: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    frac_diff_d: float = 0.5
    frac_diff_threshold: float = 1e-5
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 50, 100]


# ==============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# ==============================================================================

@njit
def _ewm_mean(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted moving average (Numba-accelerated)."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


@njit
def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum (Numba-accelerated)."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    for i in range(window - 1, n):
        result[i] = np.max(arr[i - window + 1:i + 1])
    return result


@njit
def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum (Numba-accelerated)."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    for i in range(window - 1, n):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result


@njit
def _compute_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI (Numba-accelerated)."""
    n = len(prices)
    result = np.empty(n)
    result[:period] = np.nan
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, n):
        # Smoothed averages
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


# ==============================================================================
# TECHNICAL INDICATORS
# ==============================================================================

def compute_returns(
    prices: pd.Series,
    periods: List[int] = [1, 5, 10, 20]
) -> pd.DataFrame:
    """
    Compute returns over various periods.
    
    Args:
        prices: Price series
        periods: List of lookback periods
        
    Returns:
        DataFrame with return columns
    """
    features = pd.DataFrame(index=prices.index)
    
    for period in periods:
        features[f'return_{period}'] = prices.pct_change(period)
        features[f'log_return_{period}'] = np.log(prices / prices.shift(period))
    
    return features


def compute_volatility(
    prices: pd.Series,
    windows: List[int] = [10, 20, 50]
) -> pd.DataFrame:
    """
    Compute realized volatility over various windows.
    
    Args:
        prices: Price series
        windows: List of lookback windows
        
    Returns:
        DataFrame with volatility columns
    """
    features = pd.DataFrame(index=prices.index)
    
    log_returns = np.log(prices / prices.shift(1))
    
    for window in windows:
        # Standard deviation of returns
        features[f'volatility_{window}'] = log_returns.rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (using high/low)
        # This would need high/low data
    
    return features


def compute_momentum(
    prices: pd.Series,
    periods: List[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Compute momentum indicators.
    
    Args:
        prices: Price series
        periods: List of lookback periods
        
    Returns:
        DataFrame with momentum columns
    """
    features = pd.DataFrame(index=prices.index)
    
    for period in periods:
        # Price momentum (ROC)
        features[f'momentum_{period}'] = prices.pct_change(period)
        
        # Z-score of returns
        returns = prices.pct_change(period)
        features[f'momentum_zscore_{period}'] = (
            returns - returns.rolling(50).mean()
        ) / returns.rolling(50).std()
    
    return features


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values
    """
    if TALIB_AVAILABLE:
        return pd.Series(
            talib.RSI(prices.values.astype(float), timeperiod=period),
            index=prices.index,
            name='rsi'
        )
    
    # Pure Python fallback
    return pd.Series(
        _compute_rsi(prices.values.astype(float), period),
        index=prices.index,
        name='rsi'
    )


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Compute MACD indicator.
    
    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        DataFrame with MACD, signal, and histogram
    """
    if TALIB_AVAILABLE:
        macd, signal_line, hist = talib.MACD(
            prices.values.astype(float),
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal
        )
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal_line,
            'macd_hist': hist
        }, index=prices.index)
    
    # Pure Python fallback
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    return pd.DataFrame({
        'macd': macd,
        'macd_signal': signal_line,
        'macd_hist': macd - signal_line
    }, index=prices.index)


def compute_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.
    
    Args:
        prices: Price series
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        DataFrame with upper, middle, lower bands and %B
    """
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(
            prices.values.astype(float),
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev
        )
    else:
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        upper, middle, lower = upper.values, middle.values, lower.values
    
    bb_width = (upper - lower) / middle
    bb_pct = (prices.values - lower) / (upper - lower)
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_width': bb_width,
        'bb_pct': bb_pct
    }, index=prices.index)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR values
    """
    if TALIB_AVAILABLE:
        return pd.Series(
            talib.ATR(
                high.values.astype(float),
                low.values.astype(float),
                close.values.astype(float),
                timeperiod=period
            ),
            index=close.index,
            name='atr'
        )
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(period).mean().rename('atr')


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D smoothing period
        
    Returns:
        DataFrame with %K and %D
    """
    if TALIB_AVAILABLE:
        slowk, slowd = talib.STOCH(
            high.values.astype(float),
            low.values.astype(float),
            close.values.astype(float),
            fastk_period=k_period,
            slowk_period=d_period,
            slowd_period=d_period
        )
        return pd.DataFrame({
            'stoch_k': slowk,
            'stoch_d': slowd
        }, index=close.index)
    
    # Pure Python
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    
    return pd.DataFrame({
        'stoch_k': k,
        'stoch_d': d
    }, index=close.index)


# ==============================================================================
# VOLUME FEATURES
# ==============================================================================

def compute_volume_features(
    close: pd.Series,
    volume: pd.Series,
    windows: List[int] = [10, 20, 50]
) -> pd.DataFrame:
    """
    Compute volume-based features.
    
    Args:
        close: Close prices
        volume: Volume series
        windows: Lookback windows
        
    Returns:
        DataFrame with volume features
    """
    features = pd.DataFrame(index=close.index)
    
    # Dollar volume
    dollar_volume = close * volume
    
    for window in windows:
        # Volume SMA
        features[f'volume_sma_{window}'] = volume.rolling(window).mean()
        
        # Relative volume
        features[f'relative_volume_{window}'] = volume / volume.rolling(window).mean()
        
        # Volume trend
        features[f'volume_trend_{window}'] = (
            volume.rolling(window // 2).mean() / volume.rolling(window).mean()
        )
        
        # Dollar volume
        features[f'dollar_volume_{window}'] = dollar_volume.rolling(window).mean()
    
    # On-Balance Volume (OBV)
    obv = (np.sign(close.diff()) * volume).cumsum()
    features['obv'] = obv
    features['obv_slope'] = obv.diff(10) / 10
    
    # Volume-Price Trend (VPT)
    vpt = (volume * close.pct_change()).cumsum()
    features['vpt'] = vpt
    
    # Accumulation/Distribution
    mfm = ((close - close.shift(1).rolling(2).min()) - 
           (close.shift(1).rolling(2).max() - close)) / \
          (close.shift(1).rolling(2).max() - close.shift(1).rolling(2).min() + 1e-10)
    features['ad_line'] = (mfm * volume).cumsum()
    
    return features


# ==============================================================================
# AFML FEATURES
# ==============================================================================

def compute_frac_diff(
    series: pd.Series,
    d: float = 0.5,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Compute fractionally differentiated series (AFML Ch. 5).
    
    Fractional differentiation preserves memory while achieving stationarity.
    
    Args:
        series: Input time series
        d: Differentiation order (0 < d < 1)
        threshold: Weight cutoff threshold
        
    Returns:
        Fractionally differentiated series
    """
    # Compute weights
    weights = [1.0]
    k = 1
    
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1
    
    weights = np.array(weights[::-1])
    width = len(weights)
    
    # Apply weights
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(width - 1, len(series)):
        result.iloc[i] = np.dot(weights, series.iloc[i - width + 1:i + 1].values)
    
    return result


def compute_cusum_filter(
    prices: pd.Series,
    threshold: float = None
) -> pd.Series:
    """
    CUSUM filter for event-driven sampling (AFML Ch. 2).
    
    Identifies significant price movements using cumulative sum.
    
    Args:
        prices: Price series
        threshold: Event threshold (default: daily std)
        
    Returns:
        Boolean series indicating event occurrences
    """
    if threshold is None:
        threshold = prices.pct_change().std()
    
    returns = prices.pct_change().fillna(0)
    
    # Initialize
    s_pos = 0
    s_neg = 0
    events = pd.Series(False, index=prices.index)
    
    for i, ret in enumerate(returns):
        s_pos = max(0, s_pos + ret)
        s_neg = min(0, s_neg + ret)
        
        if s_pos > threshold:
            events.iloc[i] = True
            s_pos = 0
        elif s_neg < -threshold:
            events.iloc[i] = True
            s_neg = 0
    
    return events


def compute_volatility_estimators(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series = None,
    window: int = 20
) -> pd.DataFrame:
    """
    Advanced volatility estimators.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        open_: Open prices (optional)
        window: Rolling window
        
    Returns:
        DataFrame with various volatility estimates
    """
    features = pd.DataFrame(index=close.index)
    
    # Parkinson volatility (high-low range)
    log_hl = np.log(high / low)
    features['vol_parkinson'] = np.sqrt(
        log_hl.pow(2).rolling(window).mean() / (4 * np.log(2))
    ) * np.sqrt(252)
    
    # Garman-Klass volatility
    log_hl2 = np.log(high / low).pow(2)
    log_co2 = np.log(close / close.shift(1)).pow(2)
    features['vol_garman_klass'] = np.sqrt(
        (0.5 * log_hl2 - (2 * np.log(2) - 1) * log_co2).rolling(window).mean()
    ) * np.sqrt(252)
    
    # Rogers-Satchell volatility
    if open_ is not None:
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_co = np.log(close / open_)
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        features['vol_rogers_satchell'] = np.sqrt(
            rs.rolling(window).mean()
        ) * np.sqrt(252)
    
    # Yang-Zhang volatility (most efficient)
    if open_ is not None:
        overnight = np.log(open_ / close.shift(1)).pow(2)
        openclose = np.log(close / open_).pow(2)
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        k = 0.34 / (1 + (window + 1) / (window - 1))
        features['vol_yang_zhang'] = np.sqrt(
            overnight.rolling(window).mean() + 
            k * openclose.rolling(window).mean() + 
            (1 - k) * rs.rolling(window).mean()
        ) * np.sqrt(252)
    
    return features


# ==============================================================================
# FEATURE PIPELINE
# ==============================================================================

class FeaturePipeline:
    """
    Complete feature engineering pipeline.
    
    Example:
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(df)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature pipeline.
        
        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            target_col: Main price column for features
            
        Returns:
            DataFrame with all features
        """
        logger.info(f"Generating features for {len(df)} rows...")
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df['volume']
        
        features = pd.DataFrame(index=df.index)
        
        # Returns
        logger.debug("Computing returns...")
        returns = compute_returns(close, self.config.lookback_windows)
        features = pd.concat([features, returns], axis=1)
        
        # Volatility
        logger.debug("Computing volatility...")
        volatility = compute_volatility(close, self.config.lookback_windows)
        features = pd.concat([features, volatility], axis=1)
        
        # Momentum
        logger.debug("Computing momentum...")
        momentum = compute_momentum(close, self.config.lookback_windows)
        features = pd.concat([features, momentum], axis=1)
        
        # RSI
        logger.debug("Computing RSI...")
        features['rsi'] = compute_rsi(close, self.config.rsi_period)
        features['rsi_normalized'] = (features['rsi'] - 50) / 50
        
        # MACD
        logger.debug("Computing MACD...")
        macd = compute_macd(
            close, 
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features = pd.concat([features, macd], axis=1)
        
        # Bollinger Bands
        logger.debug("Computing Bollinger Bands...")
        bb = compute_bollinger_bands(close, self.config.bb_period, self.config.bb_std)
        features = pd.concat([features, bb], axis=1)
        
        # ATR
        logger.debug("Computing ATR...")
        features['atr'] = compute_atr(high, low, close, self.config.atr_period)
        features['atr_pct'] = features['atr'] / close
        
        # Stochastic
        logger.debug("Computing Stochastic...")
        stoch = compute_stochastic(high, low, close)
        features = pd.concat([features, stoch], axis=1)
        
        # Volume features
        logger.debug("Computing volume features...")
        vol_features = compute_volume_features(
            close, volume, self.config.lookback_windows
        )
        features = pd.concat([features, vol_features], axis=1)
        
        # Advanced volatility
        logger.debug("Computing advanced volatility...")
        adv_vol = compute_volatility_estimators(high, low, close, open_)
        features = pd.concat([features, adv_vol], axis=1)
        
        # Fractional differentiation (AFML)
        logger.debug("Computing fractional differentiation...")
        features['close_frac_diff'] = compute_frac_diff(
            close,
            self.config.frac_diff_d,
            self.config.frac_diff_threshold
        )
        
        # CUSUM events
        logger.debug("Computing CUSUM filter...")
        features['cusum_event'] = compute_cusum_filter(close).astype(int)
        
        # Price relative to moving averages
        for window in self.config.lookback_windows:
            ma = close.rolling(window).mean()
            features[f'price_to_ma_{window}'] = close / ma - 1
        
        # High/Low relative features
        for window in self.config.lookback_windows:
            highest = high.rolling(window).max()
            lowest = low.rolling(window).min()
            features[f'price_to_high_{window}'] = close / highest - 1
            features[f'price_to_low_{window}'] = close / lowest - 1
        
        # Time features (if datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour'] = df.index.hour
            features['dayofweek'] = df.index.dayofweek
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Generated {len(self.feature_names)} features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("FEATURE ENGINEERING DEMO")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range('2020-01-01', periods=n, freq='1h')
    close = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n))
    high = close * (1 + np.random.uniform(0, 0.01, n))
    low = close * (1 - np.random.uniform(0, 0.01, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.exponential(1000000, n)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Generate features
    pipeline = FeaturePipeline()
    features = pipeline.fit_transform(df)
    
    print(f"\nGenerated {len(pipeline.get_feature_names())} features")
    print(f"\nFeature names: {pipeline.get_feature_names()[:10]}...")
    print(f"\nSample features:")
    print(features.dropna().head())
