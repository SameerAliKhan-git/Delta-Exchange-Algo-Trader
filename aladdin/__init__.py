"""
ALADDIN AI - Autonomous Trading System
=======================================
Inspired by BlackRock's Aladdin (Asset, Liability, Debt and Derivative Investment Network)

This system provides:
- Autonomous market analysis
- News & sentiment processing
- Multi-strategy execution
- Risk management
- 24/7 automated trading
"""

from aladdin.core import AladdinAI
from aladdin.sentiment import SentimentAnalyzer
from aladdin.market_analyzer import MarketAnalyzer
from aladdin.risk_engine import RiskEngine
from aladdin.executor import TradeExecutor

__version__ = "1.0.0"
__all__ = ["AladdinAI", "SentimentAnalyzer", "MarketAnalyzer", "RiskEngine", "TradeExecutor"]
