"""
Trading Module
==============
Core trading orchestration and paper trading.
"""

from .paper_trading_orchestrator import (
    PaperTradingOrchestrator,
    PaperTrade,
    DailyReport,
    TradingMode as PaperTradingMode
)

from .production_pipeline import (
    ProductionTradingPipeline,
    TradeDecision,
    SignalDecision,
    TradingMode,
    create_production_pipeline
)

__all__ = [
    # Paper trading
    'PaperTradingOrchestrator',
    'PaperTrade',
    'DailyReport',
    'PaperTradingMode',
    
    # Production pipeline
    'ProductionTradingPipeline',
    'TradeDecision',
    'SignalDecision',
    'TradingMode',
    'create_production_pipeline'
]
