"""
Risk module for quant-bot.

Includes:
- Risk management (limits, position sizing)
- Portfolio optimization (HRP, MVO, Black-Litterman, Risk Parity)
"""

from .risk_manager import (
    RiskManager,
    RiskLimits,
    RiskAction,
    PositionRisk,
    PortfolioRisk
)

# Portfolio Optimization
from .portfolio_optimization import (
    PortfolioResult,
    OptimizationMethod,
    CovarianceEstimator,
    MeanVarianceOptimizer,
    HierarchicalRiskParity,
    RiskParity,
    BlackLitterman,
    PortfolioOptimizer
)

__all__ = [
    # Risk Management
    'RiskManager',
    'RiskLimits',
    'RiskAction',
    'PositionRisk',
    'PortfolioRisk',
    # Portfolio Optimization
    'PortfolioResult',
    'OptimizationMethod',
    'CovarianceEstimator',
    'MeanVarianceOptimizer',
    'HierarchicalRiskParity',
    'RiskParity',
    'BlackLitterman',
    'PortfolioOptimizer'
]
