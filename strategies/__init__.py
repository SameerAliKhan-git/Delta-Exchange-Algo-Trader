"""
Strategies Module - Jesse-like strategy framework

Provides:
- StrategyBase - Base class with lifecycle hooks
- MomentumStrategy - Multi-modal momentum strategy
- OptionsDirectionalStrategy - Options trading with delta-based selection
"""

from .base import StrategyBase, StrategyState, Position, Order
from .momentum import MomentumStrategy
from .options_directional import OptionsDirectionalStrategy

__all__ = [
    # Base
    'StrategyBase',
    'StrategyState',
    'Position',
    'Order',
    
    # Strategies
    'MomentumStrategy',
    'OptionsDirectionalStrategy'
]
