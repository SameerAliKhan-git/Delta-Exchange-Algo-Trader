"""
Backtest Module - Historical backtesting engine

Provides:
- Event-driven backtesting architecture
- Strategy backtesting with data replay
- Comprehensive performance metrics
- Monte Carlo simulation
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
from .event_engine import (
    Event,
    EventType,
    EventEngine,
    TickEvent,
    BarEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    create_event
)
from .data_handler import (
    HistoricalBar,
    DataSeries,
    DataHandler,
    MultiSymbolHandler
)
from .performance import (
    PerformanceAnalyzer,
    DrawdownPeriod,
    MonthlyReturn,
    RiskMetrics
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
    
    # Event Engine
    "Event",
    "EventType",
    "EventEngine",
    "TickEvent",
    "BarEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "create_event",
    
    # Data Handler
    "HistoricalBar",
    "DataSeries",
    "DataHandler",
    "MultiSymbolHandler",
    
    # Performance
    "PerformanceAnalyzer",
    "DrawdownPeriod",
    "MonthlyReturn",
    "RiskMetrics",
]
