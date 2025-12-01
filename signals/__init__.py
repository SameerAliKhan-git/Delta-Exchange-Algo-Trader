"""
Signals Module - Technical, orderbook, and sentiment signal generators

Provides:
- Technical indicators (EMA, RSI, ATR, MACD, Bollinger Bands)
- Order book analysis (imbalance, depth, wall detection)
- Sentiment analysis (social, news, composite)
"""

from .technical import (
    ema, sma, rsi, atr, momentum, 
    bollinger_bands, macd, stochastic, adx, supertrend
)
from .orderbook import (
    compute_imbalance, compute_depth_ratio, 
    detect_walls, compute_vwap_deviation,
    OrderbookSnapshot, parse_orderbook,
    get_orderbook_signal
)
from .sentiment import (
    get_composite_sentiment, analyze_text_sentiment,
    get_social_sentiment, get_news_sentiment,
    lexicon_sentiment, sentiment_to_signal,
    get_fear_greed_index, SentimentResult
)

__all__ = [
    # Technical
    'ema', 'sma', 'rsi', 'atr', 'momentum',
    'bollinger_bands', 'macd', 'stochastic', 'adx', 'supertrend',
    
    # Orderbook
    'compute_imbalance', 'compute_depth_ratio',
    'detect_walls', 'compute_vwap_deviation',
    'OrderbookSnapshot', 'parse_orderbook', 'get_orderbook_signal',
    
    # Sentiment
    'get_composite_sentiment', 'analyze_text_sentiment',
    'get_social_sentiment', 'get_news_sentiment',
    'lexicon_sentiment', 'sentiment_to_signal',
    'get_fear_greed_index', 'SentimentResult'
]
