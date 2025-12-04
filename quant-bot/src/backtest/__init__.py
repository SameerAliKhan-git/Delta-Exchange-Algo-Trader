"""
Backtest module for quant-bot.

Includes:
- Backtest engine
- Walk-forward optimization
- Strategy validation (Monte Carlo, PBO, Deflated Sharpe)
"""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    Order,
    Position,
    Trade,
    Side,
    OrderType,
    fixed_slippage,
    volume_dependent_slippage,
    fixed_fraction_size,
    kelly_criterion_size,
    walk_forward_optimization
)

# Strategy Validation
from .validation import (
    ValidationResult,
    PBOResult,
    MonteCarloValidator,
    PBOAnalyzer,
    DeflatedSharpeRatio,
    BootstrapCI,
    StrategyValidator
)

__all__ = [
    # Backtest Engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'Order',
    'Position',
    'Trade',
    'Side',
    'OrderType',
    'fixed_slippage',
    'volume_dependent_slippage',
    'fixed_fraction_size',
    'kelly_criterion_size',
    'walk_forward_optimization',
    # Validation
    'ValidationResult',
    'PBOResult',
    'MonteCarloValidator',
    'PBOAnalyzer',
    'DeflatedSharpeRatio',
    'BootstrapCI',
    'StrategyValidator'
]
