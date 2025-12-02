"""
ALADDIN - Data Ingest Pipeline
================================
Multi-source news and data ingestion system.
"""

from .news_aggregator import NewsAggregator, NewsSource
from .sentiment_engine import SentimentEngine
from .rss_parser import RSSParser
from .social_media import RedditConnector, TwitterConnector, TelegramConnector

__all__ = [
    'NewsAggregator',
    'NewsSource',
    'SentimentEngine',
    'RSSParser',
    'RedditConnector',
    'TwitterConnector',
    'TelegramConnector'
]
