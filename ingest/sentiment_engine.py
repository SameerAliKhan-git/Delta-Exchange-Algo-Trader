"""
ALADDIN - Sentiment Engine
=============================
Advanced NLP-based sentiment analysis for trading signals.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import math

from .news_aggregator import NewsItem, NewsAggregator, NewsSource
from .rss_parser import RSSParser
from .social_media import RedditConnector


class SentimentCategory(Enum):
    """Sentiment categories."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentScore:
    """Detailed sentiment score."""
    score: float  # -1 to 1
    category: SentimentCategory
    confidence: float  # 0 to 1
    bullish_signals: int = 0
    bearish_signals: int = 0
    source_breakdown: Dict[str, float] = field(default_factory=dict)
    keywords_detected: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class SentimentEngine:
    """
    Advanced sentiment analysis engine using keyword-based analysis.
    Designed for crypto market sentiment.
    """
    
    # Bullish keywords with weights
    BULLISH_KEYWORDS = {
        # Strong bullish
        'moon': 2.0, 'mooning': 2.0, 'bullish': 2.0, 'breakout': 2.0,
        'ath': 2.0, 'all-time high': 2.0, 'buy': 1.5, 'buying': 1.5,
        'pump': 1.5, 'pumping': 1.5, 'rally': 2.0, 'rallying': 2.0,
        'surge': 1.8, 'surging': 1.8, 'soar': 1.8, 'soaring': 1.8,
        
        # Moderate bullish
        'uptrend': 1.5, 'support': 1.2, 'bullish divergence': 2.0,
        'accumulation': 1.5, 'accumulating': 1.5, 'hodl': 1.3,
        'long': 1.2, 'going long': 1.5, 'bull run': 2.0,
        'adoption': 1.5, 'institutional': 1.3, 'etf': 1.5,
        'approval': 1.5, 'approved': 1.8, 'partnership': 1.2,
        
        # Mild bullish
        'positive': 1.0, 'growth': 1.0, 'gains': 1.0, 'profit': 1.0,
        'upgrade': 1.2, 'milestone': 1.2, 'bullish signal': 1.8,
    }
    
    # Bearish keywords with weights
    BEARISH_KEYWORDS = {
        # Strong bearish
        'crash': -2.0, 'crashing': -2.0, 'dump': -2.0, 'dumping': -2.0,
        'bearish': -2.0, 'collapse': -2.0, 'plunge': -2.0, 'plunging': -2.0,
        'sell': -1.5, 'selling': -1.5, 'capitulation': -2.5,
        'rekt': -2.0, 'liquidation': -1.8, 'liquidated': -2.0,
        
        # Moderate bearish
        'downtrend': -1.5, 'resistance': -1.2, 'bearish divergence': -2.0,
        'distribution': -1.5, 'short': -1.2, 'going short': -1.5,
        'bear market': -2.0, 'fud': -1.5, 'scam': -1.8,
        'hack': -2.0, 'hacked': -2.5, 'exploit': -2.0,
        'ban': -1.8, 'banned': -2.0, 'regulation': -1.0,
        
        # Mild bearish
        'negative': -1.0, 'decline': -1.0, 'loss': -1.0, 'losses': -1.0,
        'downgrade': -1.2, 'concern': -1.0, 'bearish signal': -1.8,
        'warning': -1.2, 'risk': -0.8, 'uncertainty': -1.0,
    }
    
    # Neutral/noise words to filter
    NOISE_WORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger('Aladdin.SentimentEngine')
        self.config = config or self._default_config()
        
        # Data sources
        self.news_aggregator = NewsAggregator()
        self.rss_parser = RSSParser()
        self.reddit = RedditConnector()
        
        # Cache
        self._sentiment_cache: Dict[str, SentimentScore] = {}
        self._last_fetch: Optional[datetime] = None
        
    def _default_config(self) -> Dict:
        return {
            'refresh_interval': 300,
            'decay_window': 3600,  # 1 hour
            'min_confidence': 0.3,
            'source_weights': {
                'NEWS_API': 1.0,
                'CRYPTOCOMPARE': 0.9,
                'RSS_FEED': 0.8,
                'REDDIT': 0.6,
                'TWITTER': 0.7,
                'TELEGRAM': 0.5,
            }
        }
    
    def refresh_data(self) -> int:
        """Fetch fresh data from all sources."""
        total_items = 0
        
        try:
            # News aggregator
            news_items = self.news_aggregator.fetch_all()
            total_items += len(news_items)
            
            # RSS feeds
            rss_items = self.rss_parser.fetch_all()
            total_items += len(rss_items)
            
            # Reddit
            reddit_items = self.reddit.fetch_all(limit=20)
            total_items += len(reddit_items)
            
            self._last_fetch = datetime.now()
            self.logger.info(f"Refreshed data: {total_items} total items")
            
        except Exception as e:
            self.logger.error(f"Error refreshing data: {e}")
        
        return total_items
    
    def analyze_text(self, text: str) -> Tuple[float, List[str]]:
        """
        Analyze sentiment of a text string.
        Returns (score, keywords_detected).
        """
        text_lower = text.lower()
        score = 0.0
        keywords = []
        
        # Check bullish keywords
        for keyword, weight in self.BULLISH_KEYWORDS.items():
            if keyword in text_lower:
                score += weight
                keywords.append(f"+{keyword}")
        
        # Check bearish keywords
        for keyword, weight in self.BEARISH_KEYWORDS.items():
            if keyword in text_lower:
                score += weight  # weight is already negative
                keywords.append(f"{keyword}")
        
        # Normalize to -1 to 1
        if keywords:
            score = max(-1.0, min(1.0, score / 5.0))
        
        return score, keywords
    
    def analyze_item(self, item: NewsItem) -> float:
        """Analyze sentiment of a news item."""
        # Combine title and content
        text = f"{item.title} {item.content}"
        score, keywords = self.analyze_text(text)
        
        # Apply source weight
        source_weight = self.config['source_weights'].get(item.source.name, 0.5)
        weighted_score = score * source_weight
        
        # Apply recency decay
        age_hours = (datetime.now() - item.published_at).total_seconds() / 3600
        decay = math.exp(-age_hours / 24)  # Half-life of 24 hours
        decayed_score = weighted_score * decay
        
        # Update item
        item.sentiment_score = decayed_score
        
        return decayed_score
    
    def get_sentiment(self, symbol: str = None, hours: int = 24) -> SentimentScore:
        """
        Get overall market sentiment, optionally for a specific symbol.
        """
        # Collect all items
        all_items = []
        
        # From news aggregator
        all_items.extend(self.news_aggregator.get_recent(hours=hours, symbols=[symbol] if symbol else None))
        
        # From RSS
        all_items.extend(self.rss_parser.get_recent(hours=hours))
        
        # From Reddit
        all_items.extend(list(self.reddit.items.values()))
        
        # Filter by symbol if specified
        if symbol:
            symbol_upper = symbol.upper()
            all_items = [i for i in all_items if symbol_upper in i.symbols]
        
        if not all_items:
            return SentimentScore(
                score=0.0,
                category=SentimentCategory.NEUTRAL,
                confidence=0.0
            )
        
        # Analyze each item
        scores = []
        all_keywords = []
        source_scores: Dict[str, List[float]] = {}
        
        for item in all_items:
            score = self.analyze_item(item)
            scores.append(score)
            
            _, keywords = self.analyze_text(f"{item.title} {item.content}")
            all_keywords.extend(keywords)
            
            source_name = item.source.name
            if source_name not in source_scores:
                source_scores[source_name] = []
            source_scores[source_name].append(score)
        
        # Calculate aggregate score
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Calculate confidence based on agreement
        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            confidence = max(0.1, 1.0 - variance)
        else:
            confidence = 0.5
        
        # Determine category
        category = self._score_to_category(avg_score)
        
        # Count signals
        bullish = sum(1 for s in scores if s > 0.1)
        bearish = sum(1 for s in scores if s < -0.1)
        
        # Source breakdown
        source_breakdown = {
            name: sum(sc) / len(sc) if sc else 0
            for name, sc in source_scores.items()
        }
        
        return SentimentScore(
            score=round(avg_score, 3),
            category=category,
            confidence=round(confidence, 2),
            bullish_signals=bullish,
            bearish_signals=bearish,
            source_breakdown=source_breakdown,
            keywords_detected=list(set(all_keywords))[:20]
        )
    
    def _score_to_category(self, score: float) -> SentimentCategory:
        """Convert numeric score to category."""
        if score > 0.5:
            return SentimentCategory.VERY_BULLISH
        elif score > 0.2:
            return SentimentCategory.BULLISH
        elif score < -0.5:
            return SentimentCategory.VERY_BEARISH
        elif score < -0.2:
            return SentimentCategory.BEARISH
        else:
            return SentimentCategory.NEUTRAL
    
    def get_trading_bias(self, symbol: str = None) -> Tuple[str, float]:
        """
        Get trading bias based on sentiment.
        Returns (direction, confidence).
        """
        sentiment = self.get_sentiment(symbol)
        
        if sentiment.score > 0.3 and sentiment.confidence > 0.5:
            return ("LONG", sentiment.confidence * abs(sentiment.score))
        elif sentiment.score < -0.3 and sentiment.confidence > 0.5:
            return ("SHORT", sentiment.confidence * abs(sentiment.score))
        else:
            return ("NEUTRAL", sentiment.confidence)
    
    def get_fear_greed(self) -> Dict:
        """
        Calculate Fear & Greed Index based on sentiment.
        Returns index from 0 (Extreme Fear) to 100 (Extreme Greed).
        """
        sentiment = self.get_sentiment()
        
        # Convert -1 to 1 score to 0 to 100
        index = (sentiment.score + 1) * 50
        index = max(0, min(100, index))
        
        if index < 25:
            label = "Extreme Fear"
        elif index < 45:
            label = "Fear"
        elif index < 55:
            label = "Neutral"
        elif index < 75:
            label = "Greed"
        else:
            label = "Extreme Greed"
        
        return {
            'index': round(index),
            'label': label,
            'sentiment_score': sentiment.score,
            'confidence': sentiment.confidence,
            'bullish_signals': sentiment.bullish_signals,
            'bearish_signals': sentiment.bearish_signals
        }
    
    def print_report(self, symbol: str = None):
        """Print sentiment report to console."""
        sentiment = self.get_sentiment(symbol)
        fear_greed = self.get_fear_greed()
        
        print("\n" + "="*70)
        print(f"🎭 SENTIMENT ANALYSIS REPORT" + (f" - {symbol}" if symbol else ""))
        print("="*70)
        
        # Overall score
        emoji = "🟢" if sentiment.score > 0.2 else "🔴" if sentiment.score < -0.2 else "⚪"
        print(f"\nOverall Sentiment: {emoji} {sentiment.category.value.upper()}")
        print(f"Score: {sentiment.score:+.3f} (Confidence: {sentiment.confidence:.0%})")
        
        # Fear & Greed
        fg_emoji = "😱" if fear_greed['index'] < 30 else "😨" if fear_greed['index'] < 45 else "😐" if fear_greed['index'] < 55 else "😏" if fear_greed['index'] < 75 else "🤑"
        print(f"\nFear & Greed Index: {fg_emoji} {fear_greed['index']} ({fear_greed['label']})")
        
        # Signal breakdown
        print(f"\nSignal Breakdown:")
        print(f"  🟢 Bullish Signals: {sentiment.bullish_signals}")
        print(f"  🔴 Bearish Signals: {sentiment.bearish_signals}")
        
        # Source breakdown
        if sentiment.source_breakdown:
            print(f"\nBy Source:")
            for source, score in sorted(sentiment.source_breakdown.items(), 
                                        key=lambda x: x[1], reverse=True):
                bar = "+" * int(abs(score) * 10) if score > 0 else "-" * int(abs(score) * 10)
                print(f"  {source:<15}: {score:+.2f} {bar}")
        
        # Keywords
        if sentiment.keywords_detected:
            print(f"\nKey Signals Detected:")
            bullish_kw = [k for k in sentiment.keywords_detected if k.startswith('+')]
            bearish_kw = [k for k in sentiment.keywords_detected if not k.startswith('+')]
            if bullish_kw:
                print(f"  🟢 {', '.join(bullish_kw[:5])}")
            if bearish_kw:
                print(f"  🔴 {', '.join(bearish_kw[:5])}")
        
        # Trading recommendation
        bias, conf = self.get_trading_bias(symbol)
        print(f"\n🎯 Trading Bias: {bias} (Strength: {conf:.0%})")
        
        print("="*70)


# Test the sentiment engine
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = SentimentEngine()
    print("\n🔄 Refreshing data sources...")
    engine.refresh_data()
    
    engine.print_report()
    engine.print_report('BTC')
