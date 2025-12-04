"""
Signals module for trading signal generation.

Includes:
- FinBERT sentiment analysis
- Technical signals
- Order flow signals
- Fusion engine for signal combination
- ORDER-FLOW CONFIRMATION ENGINE (Use with ANY strategy)
"""

from .finbert_sentiment import (
    KeywordSentiment,
    FinBERTSentiment,
    NewsSentimentAnalyzer,
    FinancialEventExtractor,
    SentimentResult,
    NewsSentiment
)

from .orderflow_confirmation import (
    # Core Engine
    OrderFlowConfirmationEngine,
    OrderFlowConfirmation,
    create_orderflow_engine,
    
    # Analyzers
    FootprintAnalyzer,
    HeatmapAnalyzer,
    VolumeProfileAnalyzer,
    
    # Data Types
    FootprintBar,
    FootprintSignal,
    LiquiditySignal,
    VolumeProfileZone,
)

from .orderflow_enhanced import (
    OrderFlowEnhancedStrategy,
    QuickOrderFlowFilter,
    EnhancedSignal,
)

__all__ = [
    # Sentiment
    'KeywordSentiment',
    'FinBERTSentiment', 
    'NewsSentimentAnalyzer',
    'FinancialEventExtractor',
    'SentimentResult',
    'NewsSentiment',
    
    # Order-Flow Confirmation Engine
    'OrderFlowConfirmationEngine',
    'OrderFlowConfirmation',
    'create_orderflow_engine',
    'FootprintAnalyzer',
    'HeatmapAnalyzer',
    'VolumeProfileAnalyzer',
    'FootprintBar',
    'FootprintSignal',
    'LiquiditySignal',
    'VolumeProfileZone',
    
    # Order-Flow Enhanced Strategies
    'OrderFlowEnhancedStrategy',
    'QuickOrderFlowFilter',
    'EnhancedSignal',
]
