"""Backtest module for quant-bot."""

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

__all__ = [
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
    'walk_forward_optimization'
]
