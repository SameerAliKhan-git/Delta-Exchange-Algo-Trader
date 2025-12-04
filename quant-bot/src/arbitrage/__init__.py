"""
Arbitrage Module - Production-Grade Multi-Strategy Arbitrage
=============================================================

This module provides complete arbitrage trading capabilities:

- Funding Arbitrage: Spot-perp funding rate capture
- Statistical Arbitrage: Cointegration-based pairs trading
- Cross-Exchange: Multi-venue price arbitrage
- Triangular: Currency cycle arbitrage

Author: Quant Bot
Version: 1.0.0
"""

from .funding_arbitrage import (
    FundingArbitrageEngine,
    FundingRateEngine,
    BasisCalculator,
    CarryYieldTracker,
    FundingRate,
    BasisSpread,
    ArbitragePosition,
    ArbitrageOpportunity as FundingOpportunity,
    FundingDirection,
    PositionSide
)

from .statistical_arbitrage import (
    StatisticalArbitrageEngine,
    CointegrationTester,
    ZScoreMonitor,
    CointegrationResult,
    CointegrationMethod,
    SpreadMetrics,
    PairPosition,
    PairStatus,
    StatArbOpportunity,
    calculate_correlation,
    calculate_rolling_correlation,
    find_optimal_hedge_ratio
)

from .cross_exchange import (
    CrossExchangeArbitrageEngine,
    OrderBookAggregator,
    LatencyNormalizer,
    TransferTimeModel,
    FeesReconciler,
    Exchange,
    OrderSide,
    ArbitrageType,
    ExchangeConfig,
    OrderBookLevel,
    AggregatedBook,
    ArbitrageOpportunity,
    ArbitrageExecution
)

__all__ = [
    # Funding Arbitrage
    'FundingArbitrageEngine',
    'FundingRateEngine',
    'BasisCalculator',
    'CarryYieldTracker',
    'FundingRate',
    'BasisSpread',
    'ArbitragePosition',
    'FundingOpportunity',
    'FundingDirection',
    'PositionSide',
    
    # Statistical Arbitrage
    'StatisticalArbitrageEngine',
    'CointegrationTester',
    'ZScoreMonitor',
    'CointegrationResult',
    'CointegrationMethod',
    'SpreadMetrics',
    'PairPosition',
    'PairStatus',
    'StatArbOpportunity',
    'calculate_correlation',
    'calculate_rolling_correlation',
    'find_optimal_hedge_ratio',
    
    # Cross-Exchange
    'CrossExchangeArbitrageEngine',
    'OrderBookAggregator',
    'LatencyNormalizer',
    'TransferTimeModel',
    'FeesReconciler',
    'Exchange',
    'OrderSide',
    'ArbitrageType',
    'ExchangeConfig',
    'OrderBookLevel',
    'AggregatedBook',
    'ArbitrageOpportunity',
    'ArbitrageExecution',
]

__version__ = '1.0.0'
