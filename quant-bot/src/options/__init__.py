"""
Options Module - Production-Grade Options Trading Infrastructure
================================================================

This module provides complete options trading capabilities:

- Pricing Engine: Black-Scholes, Bachelier, Monte Carlo
- Greeks: Delta, Gamma, Theta, Vega, Rho + higher order
- Volatility Surface: SVI, SABR, interpolation
- IV Rank/Percentile: Historical IV analysis
- Strategies: All major options strategies
- Delta Hedging: Automated hedging system
- Gamma Scalping: Market-neutral gamma capture
- Risk Engine: Crypto-specific options risk management

Author: Quant Bot
Version: 1.0.0
"""

from .pricing_engine import (
    OptionsPricingEngine,
    OptionType,
    PricingModel,
    Greeks,
    OptionPrice,
    calculate_breakeven,
    calculate_max_profit_loss
)

from .volatility_surface import (
    VolatilitySurfaceEngine,
    VolSurface,
    VolSlice,
    VolPoint,
    IVMetrics,
    SurfaceModel,
    SVIParams
)

from .strategies import (
    OptionsStrategyEngine,
    StrategyType,
    StrategyPosition,
    StrategyAnalysis,
    OptionLeg,
    SpotLeg,
    DeltaHedgingEngine,
    GammaScalpingEngine
)

from .risk_engine import (
    OptionsRiskEngine,
    RiskReport,
    RiskLimit,
    RiskLevel,
    RiskType,
    VolatilityCrushRisk,
    GapRiskAnalysis
)

__all__ = [
    # Pricing
    'OptionsPricingEngine',
    'OptionType',
    'PricingModel',
    'Greeks',
    'OptionPrice',
    'calculate_breakeven',
    'calculate_max_profit_loss',
    
    # Volatility
    'VolatilitySurfaceEngine',
    'VolSurface',
    'VolSlice',
    'VolPoint',
    'IVMetrics',
    'SurfaceModel',
    'SVIParams',
    
    # Strategies
    'OptionsStrategyEngine',
    'StrategyType',
    'StrategyPosition',
    'StrategyAnalysis',
    'OptionLeg',
    'SpotLeg',
    'DeltaHedgingEngine',
    'GammaScalpingEngine',
    
    # Risk
    'OptionsRiskEngine',
    'RiskReport',
    'RiskLimit',
    'RiskLevel',
    'RiskType',
    'VolatilityCrushRisk',
    'GapRiskAnalysis',
]

__version__ = '1.0.0'
