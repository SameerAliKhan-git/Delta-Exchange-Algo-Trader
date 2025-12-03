"""Features module for quant-bot."""

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

__all__ = [
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
    'compute_volatility_estimators'
]
