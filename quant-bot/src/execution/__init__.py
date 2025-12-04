"""
Execution module for quant-bot.

Includes:
- Execution simulator
- Paper trading engine
- Smart execution algorithms (TWAP, VWAP, IS, Iceberg, Adaptive)
"""

from .simulator import (
    ExecutionSimulator,
    ExecutionConfig,
    ExecutionMode,
    ExecutionOrder,
    OrderStatus,
    MarketSnapshot,
    PaperTradingEngine
)

# Execution Algorithms
from .execution_algos import (
    TWAPAlgo,
    VWAPAlgo,
    ImplementationShortfallAlgo,
    IcebergAlgo,
    AdaptiveAlgo,
    ExecutionEngine,
    ChildOrder,
    ExecutionReport,
    OrderSide
)

__all__ = [
    # Simulator
    'ExecutionSimulator',
    'ExecutionConfig',
    'ExecutionMode',
    'ExecutionOrder',
    'OrderStatus',
    'MarketSnapshot',
    'PaperTradingEngine',
    # Execution Algorithms
    'TWAPAlgo',
    'VWAPAlgo',
    'ImplementationShortfallAlgo',
    'IcebergAlgo',
    'AdaptiveAlgo',
    'ExecutionEngine',
    'ChildOrder',
    'ExecutionReport',
    'OrderSide'
]
