"""
Engine Module - Real-Time Execution & Orchestration
====================================================
Quant-grade execution systems.
"""

from .realtime import (
    RealTimeExecutionEngine,
    ExecutionAlgorithm,
    OrderStatus,
    SmartOrder,
    ExecutionMetrics,
    OrderBookTracker,
    TWAPExecutor,
    VWAPExecutor,
    IcebergExecutor,
    SniperExecutor,
    AdaptiveExecutor,
    create_execution_engine
)

from .orchestrator import (
    MasterOrchestrator,
    TradingMode,
    MarketRegime,
    StrategyAllocation,
    RiskLimits,
    PortfolioState,
    CapitalAllocator,
    RegimeDetector,
    DrawdownProtection,
    PerformanceAttribution,
    create_orchestrator
)

__all__ = [
    # Execution
    'RealTimeExecutionEngine',
    'ExecutionAlgorithm',
    'OrderStatus',
    'SmartOrder',
    'ExecutionMetrics',
    'OrderBookTracker',
    'TWAPExecutor',
    'VWAPExecutor',
    'IcebergExecutor',
    'SniperExecutor',
    'AdaptiveExecutor',
    'create_execution_engine',
    # Orchestration
    'MasterOrchestrator',
    'TradingMode',
    'MarketRegime',
    'StrategyAllocation',
    'RiskLimits',
    'PortfolioState',
    'CapitalAllocator',
    'RegimeDetector',
    'DrawdownProtection',
    'PerformanceAttribution',
    'create_orchestrator'
]
