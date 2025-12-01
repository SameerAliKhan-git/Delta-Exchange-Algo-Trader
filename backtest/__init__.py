"""
Backtest Module - Historical backtesting engine

Provides:
- Strategy backtesting
- Performance metrics
- Trade analysis
"""

from .runner import (
    BacktestRunner,
    BacktestConfig,
    BacktestResult,
    run_backtest
)
from .metrics import (
    calculate_metrics,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_win_rate,
    PerformanceMetrics,
    Trade
)

__all__ = [
    # Runner
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    
    # Metrics
    "calculate_metrics",
    "calculate_sharpe",
    "calculate_max_drawdown",
    "calculate_win_rate",
    "PerformanceMetrics",
    "Trade",
]
