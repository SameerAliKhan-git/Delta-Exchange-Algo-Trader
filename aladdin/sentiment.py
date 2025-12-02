"""
SENTIMENT ANALYZER - News & Market Sentiment Analysis
======================================================
Analyzes global news, economic events, and market sentiment to inform trading decisions.

Data Sources:
- Crypto news feeds (RSS)
- Economic calendars
- Social sentiment (Twitter/Reddit trends)
- Fear & Greed Index
- Market correlations
"""

import time
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import hashlib

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("AladdinAI.Sentiment")


@dataclass
class NewsItem:
    """Single news item"""
    title: str
    source: str
    url: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    relevance: float  # 0 to 1
    keywords: List[str]
    impact: str  # "high", "medium", "low"


@dataclass
class EconomicEvent:
    """Economic calendar event"""
    name: str
    country: str
    timestamp: datetime
    impact: str  # "high", "medium", "low"
    forecast: Optional[float]
    previous: Optional[float]
    actual: Optional[float]


class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis for trading decisions
    """
    
    # Sentiment keywords
    BULLISH_KEYWORDS = [
        "surge", "soar", "rally", "breakout", "bullish", "moon", "ath", "all-time high",
        "adoption", "institutional", "etf approved", "accumulation", "buy signal",
        "support", "recovery", "growth", "positive", "upgrade", "outperform",
        "partnership", "launch", "innovation", "bullrun", "pump"
    ]
    
    BEARISH_KEYWORDS = [
        "crash", "dump", "plunge", "bearish", "selloff", "correction", "breakdown",
        "hack", "ban", "regulation", "lawsuit", "fraud", "scam", "ponzi",
        "resistance", "decline", "negative", "downgrade", "underperform",
        "bankruptcy", "default", "layoff", "recession", "collapse"
    ]
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._news_cache: List[NewsItem] = []
        self._economic_events: List[EconomicEvent] = []
        self._last_fetch = None
        self._cache_duration = 300  # 5 minutes
        
        # Fear & Greed Index simulation (in production, fetch from API)
        self._fear_greed = 50
        
        logger.info("Sentiment Analyzer initialized")
    
    def get_sentiment(self, symbol: str = "BTC") -> Dict:
        """
        Get comprehensive sentiment analysis
        
        Returns:
            Dict with sentiment metrics
        """
        # Fetch latest news if cache expired
        if self._should_refresh():
            self._fetch_news()
            self._fetch_economic_events()
        
        # Calculate sentiment components
        news_sentiment = self._calculate_news_sentiment(symbol)
        economic_outlook = self._analyze_economic_outlook()
        fear_greed = self._get_fear_greed_index()
        social_sentiment = self._get_social_sentiment(symbol)
        
        # Weighted composite sentiment
        composite = (
            news_sentiment * 0.35 +
            social_sentiment * 0.25 +
            (fear_greed - 50) / 50 * 0.25 +
            self._economic_to_score(economic_outlook) * 0.15
        )
        
        return {
            "news_sentiment": news_sentiment,
            "social_sentiment": social_sentiment,
            "fear_greed": fear_greed,
            "economic_outlook": economic_outlook,
            "composite": composite,
            "timestamp": datetime.now().isoformat()
        }
    
    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed"""
        if self._last_fetch is None:
            return True
        return (datetime.now() - self._last_fetch).seconds > self._cache_duration
    
    def _fetch_news(self):
        """Fetch crypto news from various sources"""
        logger.info("Fetching news...")
        
        news_sources = [
            ("https://min-api.cryptocompare.com/data/v2/news/?lang=EN", "cryptocompare"),
            # Add more sources as needed
        ]
        
        self._news_cache = []
        
        for url, source in news_sources:
            try:
                if requests:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get("Data", [])
                        
                        for article in articles[:20]:  # Limit to 20 most recent
                            news_item = NewsItem(
                                title=article.get("title", ""),
                                source=article.get("source", source),
                                url=article.get("url", ""),
                                timestamp=datetime.fromtimestamp(article.get("published_on", 0)),
                                sentiment_score=self._analyze_text_sentiment(article.get("title", "")),
                                relevance=1.0,
                                keywords=article.get("categories", "").split("|"),
                                impact="medium"
                            )
                            self._news_cache.append(news_item)
                            
            except Exception as e:
                logger.warning(f"Failed to fetch news from {source}: {e}")
        
        # If no news fetched, generate synthetic market data
        if not self._news_cache:
            self._generate_synthetic_news()
        
        self._last_fetch = datetime.now()
        logger.info(f"Fetched {len(self._news_cache)} news items")
    
    def _generate_synthetic_news(self):
        """Generate synthetic news for testing when APIs unavailable"""
        synthetic_news = [
            ("Bitcoin Holds Strong Above Key Support Level", 0.3),
            ("Institutional Interest in Crypto Continues to Grow", 0.5),
            ("Market Volatility Expected Ahead of Fed Decision", -0.1),
            ("Major Exchange Reports Record Trading Volume", 0.4),
            ("Crypto Adoption Accelerates in Emerging Markets", 0.6),
        ]
        
        for title, sentiment in synthetic_news:
            self._news_cache.append(NewsItem(
                title=title,
                source="synthetic",
                url="",
                timestamp=datetime.now(),
                sentiment_score=sentiment,
                relevance=0.8,
                keywords=["bitcoin", "crypto"],
                impact="medium"
            ))
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using keyword matching"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def _calculate_news_sentiment(self, symbol: str) -> float:
        """Calculate aggregate news sentiment for symbol"""
        if not self._news_cache:
            return 0.0
        
        symbol_lower = symbol.lower().replace("usd", "")
        relevant_news = [
            n for n in self._news_cache
            if symbol_lower in n.title.lower() or 
               symbol_lower in " ".join(n.keywords).lower() or
               "bitcoin" in n.title.lower() or
               "crypto" in n.title.lower()
        ]
        
        if not relevant_news:
            return 0.0
        
        # Weight by recency
        now = datetime.now()
        weighted_sum = 0
        weight_total = 0
        
        for news in relevant_news:
            age_hours = (now - news.timestamp).total_seconds() / 3600
            weight = 1 / (1 + age_hours / 24)  # Decay over 24 hours
            weighted_sum += news.sentiment_score * weight * news.relevance
            weight_total += weight
        
        return weighted_sum / weight_total if weight_total > 0 else 0.0
    
    def _fetch_economic_events(self):
        """Fetch economic calendar events"""
        # In production, fetch from economic calendar API
        # For now, generate key events
        
        self._economic_events = [
            EconomicEvent(
                name="Federal Reserve Interest Rate Decision",
                country="US",
                timestamp=datetime.now() + timedelta(days=7),
                impact="high",
                forecast=5.25,
                previous=5.25,
                actual=None
            ),
            EconomicEvent(
                name="US CPI Inflation",
                country="US",
                timestamp=datetime.now() + timedelta(days=3),
                impact="high",
                forecast=3.2,
                previous=3.4,
                actual=None
            ),
            EconomicEvent(
                name="Non-Farm Payrolls",
                country="US",
                timestamp=datetime.now() + timedelta(days=5),
                impact="high",
                forecast=180000,
                previous=216000,
                actual=None
            )
        ]
    
    def _analyze_economic_outlook(self) -> str:
        """Analyze economic outlook based on events"""
        if not self._economic_events:
            return "neutral"
        
        # Check for upcoming high-impact events
        upcoming_high_impact = [
            e for e in self._economic_events
            if e.impact == "high" and 
               e.timestamp > datetime.now() and
               e.timestamp < datetime.now() + timedelta(days=7)
        ]
        
        if len(upcoming_high_impact) >= 2:
            return "volatile"  # Expect volatility
        
        # Simplified outlook based on Fed policy
        for event in self._economic_events:
            if "interest rate" in event.name.lower():
                if event.forecast and event.previous:
                    if event.forecast > event.previous:
                        return "bearish"  # Higher rates = bearish for crypto
                    elif event.forecast < event.previous:
                        return "bullish"  # Lower rates = bullish for crypto
        
        return "neutral"
    
    def _economic_to_score(self, outlook: str) -> float:
        """Convert economic outlook to score"""
        mapping = {
            "bullish": 0.5,
            "neutral": 0.0,
            "bearish": -0.5,
            "volatile": -0.2
        }
        return mapping.get(outlook, 0.0)
    
    def _get_fear_greed_index(self) -> float:
        """Get Fear & Greed Index"""
        try:
            if requests:
                response = requests.get(
                    "https://api.alternative.me/fng/",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data"):
                        self._fear_greed = float(data["data"][0]["value"])
        except Exception as e:
            logger.debug(f"Failed to fetch Fear & Greed: {e}")
        
        return self._fear_greed
    
    def _get_social_sentiment(self, symbol: str) -> float:
        """Get social media sentiment (Twitter, Reddit)"""
        # In production, use Twitter/Reddit APIs
        # For now, return estimate based on fear/greed
        
        fear_greed = self._fear_greed
        
        # Convert to -1 to 1 scale
        return (fear_greed - 50) / 50
    
    def get_market_mood(self) -> str:
        """Get human-readable market mood"""
        fg = self._fear_greed
        
        if fg >= 80:
            return "Extreme Greed"
        elif fg >= 60:
            return "Greed"
        elif fg >= 40:
            return "Neutral"
        elif fg >= 20:
            return "Fear"
        else:
            return "Extreme Fear"
    
    def get_trading_bias(self) -> Tuple[str, float]:
        """Get trading bias based on sentiment"""
        sentiment = self.get_sentiment("BTC")
        composite = sentiment["composite"]
        
        if composite > 0.3:
            return "LONG", abs(composite)
        elif composite < -0.3:
            return "SHORT", abs(composite)
        else:
            return "NEUTRAL", abs(composite)
    
    def get_news_summary(self, limit: int = 5) -> List[Dict]:
        """Get summary of recent news"""
        if self._should_refresh():
            self._fetch_news()
        
        return [
            {
                "title": n.title,
                "source": n.source,
                "sentiment": n.sentiment_score,
                "time": n.timestamp.isoformat()
            }
            for n in self._news_cache[:limit]
        ]
    
    def print_sentiment_report(self, symbol: str = "BTC"):
        """Print formatted sentiment report"""
        sentiment = self.get_sentiment(symbol)
        bias, confidence = self.get_trading_bias()
        
        print("\n" + "=" * 60)
        print(f"SENTIMENT REPORT - {symbol}")
        print("=" * 60)
        print(f"\n  Market Mood: {self.get_market_mood()}")
        print(f"  Fear & Greed Index: {sentiment['fear_greed']:.0f}")
        print(f"  News Sentiment: {sentiment['news_sentiment']:.2f}")
        print(f"  Social Sentiment: {sentiment['social_sentiment']:.2f}")
        print(f"  Economic Outlook: {sentiment['economic_outlook']}")
        print(f"\n  COMPOSITE SCORE: {sentiment['composite']:.2f}")
        print(f"  TRADING BIAS: {bias} (Confidence: {confidence:.2%})")
        
        print("\n  Recent News:")
        for news in self.get_news_summary(3):
            sentiment_icon = "+" if news["sentiment"] > 0 else "-" if news["sentiment"] < 0 else " "
            print(f"  [{sentiment_icon}] {news['title'][:50]}...")
        
        print("=" * 60)
