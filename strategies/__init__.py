"""
Strategies Module - Professional Quant Strategy Framework
==========================================================

Provides Renaissance Technologies-inspired strategies:
- StrategyBase - Base class with lifecycle hooks
- MomentumStrategy - Multi-modal momentum strategy
- OptionsDirectionalStrategy - Options trading with delta-based selection
- MedallionStrategy - Multi-alpha Renaissance-inspired strategy
- StatArbStrategy - Statistical arbitrage with cointegration
- OptionsAlphaStrategy - Advanced options: volatility arb, gamma scalping
- EnsembleStrategy - ML-based strategy ensemble
- StrategyRegistry - Strategy registration and discovery
"""

from .base import StrategyBase, StrategyState, Position, Order
from .momentum import MomentumStrategy
from .options_directional import OptionsDirectionalStrategy

# Advanced quant strategies
from .registry import StrategyRegistry, StrategyPerformance, StrategyConfig
from .medallion import MedallionStrategy
from .stat_arb import StatisticalArbitrage
from .options_alpha import OptionsAlphaStrategy
from .ensemble import EnsembleEngine

__all__ = [
    # Base
    'StrategyBase',
    'StrategyState',
    'Position',
    'Order',
    
    # Basic Strategies
    'MomentumStrategy',
    'OptionsDirectionalStrategy',
    
    # Advanced Quant Strategies
    'MedallionStrategy',
    'StatisticalArbitrage',
    'OptionsAlphaStrategy',
    'EnsembleEngine',
    
    # Registry
    'StrategyRegistry',
    'StrategyPerformance',
    'StrategyConfig'
]

