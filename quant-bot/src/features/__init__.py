"""
Features module for quant-bot.

Includes:
- Feature engineering pipeline
- Technical indicators
- Signal processing (Kalman, HMM, Wavelets, GARCH)
"""

from .feature_engineer import (
    FeaturePipeline,
    FeatureConfig,
    compute_returns,
    compute_volatility,
    compute_momentum,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_stochastic,
    compute_volume_features,
    compute_frac_diff,
    compute_cusum_filter,
    compute_volatility_estimators
)

# Signal Processing - import only base classes that exist
from .signal_processing import (
    KalmanFilter,
    DynamicHedgeRatio,
    GaussianHMM,
    GARCH,
    WaveletDecomposition
)

__all__ = [
    # Feature Engineering
    'FeaturePipeline',
    'FeatureConfig',
    'compute_returns',
    'compute_volatility',
    'compute_momentum',
    'compute_rsi',
    'compute_macd',
    'compute_bollinger_bands',
    'compute_atr',
    'compute_stochastic',
    'compute_volume_features',
    'compute_frac_diff',
    'compute_cusum_filter',
    'compute_volatility_estimators',
    # Signal Processing
    'KalmanFilter',
    'DynamicHedgeRatio',
    'GaussianHMM',
    'GARCH',
    'WaveletDecomposition'
]
