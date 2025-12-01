"""
Data Ingestion Module for Delta Exchange Algo Trading Bot
Handles market data, news, and sentiment data collection
"""

import time
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from collections import deque
from dataclasses import dataclass, field
import threading

import numpy as np
import pandas as pd
import requests
from transformers import pipeline

from config import get_config
from logger import get_logger
from delta_client import get_delta_client


@dataclass
class PriceData:
    """Price data point"""
    timestamp: datetime
    price: float
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    

@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot"""
    timestamp: datetime
    bids: List[List[float]]  # [[price, size], ...]
    asks: List[List[float]]  # [[price, size], ...]
    
    @property
    def bid_volume(self) -> float:
        return sum(b[1] for b in self.bids) if self.bids else 0
    
    @property
    def ask_volume(self) -> float:
        return sum(a[1] for a in self.asks) if self.asks else 0
    
    @property
    def imbalance(self) -> float:
        """Calculate orderbook imbalance (-1 to 1)"""
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.0
        return (self.bid_volume - self.ask_volume) / total
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0][0] - self.bids[0][0]


@dataclass
class SentimentData:
    """Sentiment data point"""
    timestamp: datetime
    source: str
    score: float  # -1 to 1
    text: str
    confidence: float = 1.0
    influence_weight: float = 1.0


@dataclass
class MarketState:
    """Current market state aggregation"""
    timestamp: datetime
    price: float
    prices: List[float] = field(default_factory=list)
    orderbook: Optional[OrderbookSnapshot] = None
    sentiment_score: float = 0.0
    sentiment_count: int = 0
    
    # Technical indicators (computed)
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    atr: Optional[float] = None
    volume_ma: Optional[float] = None


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def fetch(self) -> Any:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class DeltaMarketDataSource(DataSource):
    """Market data from Delta Exchange"""
    
    def __init__(self, product_id: int, product_symbol: str):
        self.product_id = product_id
        self.product_symbol = product_symbol
        self.client = get_delta_client()
        self.logger = get_logger()
    
    def fetch(self) -> Optional[PriceData]:
        """Fetch current price data"""
        try:
            ticker = self.client.get_ticker(self.product_symbol)
            return PriceData(
                timestamp=datetime.utcnow(),
                price=float(ticker.get('close', ticker.get('last_price', 0))),
                volume=float(ticker.get('volume', 0)),
                bid=float(ticker.get('bid', 0)) if ticker.get('bid') else None,
                ask=float(ticker.get('ask', 0)) if ticker.get('ask') else None
            )
        except Exception as e:
            self.logger.error("Failed to fetch ticker", error=str(e))
            return None
    
    def fetch_orderbook(self, depth: int = 10) -> Optional[OrderbookSnapshot]:
        """Fetch orderbook snapshot"""
        try:
            ob = self.client.get_orderbook(self.product_id, depth=depth)
            return OrderbookSnapshot(
                timestamp=datetime.utcnow(),
                bids=[[float(b['price']), float(b['size'])] for b in ob.get('buy', [])],
                asks=[[float(a['price']), float(a['size'])] for a in ob.get('sell', [])]
            )
        except Exception as e:
            self.logger.error("Failed to fetch orderbook", error=str(e))
            return None
    
    def is_available(self) -> bool:
        return self.client.health_check()


class SentimentAnalyzer:
    """
    Sentiment analysis using transformer models
    Supports multiple text sources and aggregation
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.logger = get_logger()
        self.model_name = model_name
        self._pipeline = None
        self._load_lock = threading.Lock()
    
    @property
    def pipeline(self):
        """Lazy load the sentiment pipeline"""
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    self.logger.info("Loading sentiment model", model=self.model_name)
                    self._pipeline = pipeline(
                        "sentiment-analysis",
                        model=self.model_name,
                        device=-1  # CPU; set to 0 for GPU
                    )
        return self._pipeline
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of a single text
        Returns score from -1 (negative) to 1 (positive)
        """
        try:
            result = self.pipeline(text[:512])[0]  # Truncate to model max
            score = result['score']
            if result['label'] == 'NEGATIVE':
                score = -score
            return score
        except Exception as e:
            self.logger.error("Sentiment analysis failed", error=str(e))
            return 0.0
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Analyze sentiment of multiple texts"""
        if not texts:
            return []
        
        try:
            # Truncate texts
            truncated = [t[:512] for t in texts]
            results = self.pipeline(truncated)
            
            scores = []
            for result in results:
                score = result['score']
                if result['label'] == 'NEGATIVE':
                    score = -score
                scores.append(score)
            
            return scores
        except Exception as e:
            self.logger.error("Batch sentiment analysis failed", error=str(e))
            return [0.0] * len(texts)
    
    def aggregate_sentiment(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None
    ) -> float:
        """
        Aggregate sentiment from multiple texts
        Optional weights for influence-based scoring
        """
        if not texts:
            return 0.0
        
        scores = self.analyze_batch(texts)
        
        if weights is None:
            return float(np.mean(scores))
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        return float(np.average(scores, weights=weights))


class TwitterDataSource(DataSource):
    """Twitter/X data source for sentiment"""
    
    def __init__(
        self,
        bearer_token: Optional[str] = None,
        keywords: List[str] = None
    ):
        config = get_config()
        self.bearer_token = bearer_token or config.data_ingestion.twitter_bearer_token
        self.keywords = keywords or ["bitcoin", "btc", "crypto"]
        self.logger = get_logger()
        self.base_url = "https://api.twitter.com/2"
    
    def is_available(self) -> bool:
        return bool(self.bearer_token)
    
    def fetch(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent tweets matching keywords"""
        if not self.is_available():
            self.logger.warning("Twitter API not configured")
            return []
        
        try:
            query = " OR ".join(self.keywords) + " -is:retweet lang:en"
            
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            params = {
                "query": query,
                "max_results": min(max_results, 100),
                "tweet.fields": "created_at,public_metrics,author_id"
            }
            
            response = requests.get(
                f"{self.base_url}/tweets/search/recent",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            else:
                self.logger.error("Twitter API error", status=response.status_code)
                return []
                
        except Exception as e:
            self.logger.error("Twitter fetch failed", error=str(e))
            return []
    
    def get_sentiment_texts(self, max_results: int = 50) -> List[str]:
        """Get tweet texts for sentiment analysis"""
        tweets = self.fetch(max_results)
        return [t.get('text', '') for t in tweets if t.get('text')]


class RedditDataSource(DataSource):
    """Reddit data source for sentiment"""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "DeltaAlgoBot/1.0",
        subreddits: List[str] = None
    ):
        config = get_config()
        self.client_id = client_id or config.data_ingestion.reddit_client_id
        self.client_secret = client_secret or config.data_ingestion.reddit_client_secret
        self.user_agent = user_agent
        self.subreddits = subreddits or ["cryptocurrency", "bitcoin", "CryptoMarkets"]
        self.logger = get_logger()
        self._access_token = None
        self._token_expiry = None
    
    def is_available(self) -> bool:
        return bool(self.client_id and self.client_secret)
    
    def _get_access_token(self) -> Optional[str]:
        """Get Reddit OAuth access token"""
        if self._access_token and self._token_expiry and datetime.utcnow() < self._token_expiry:
            return self._access_token
        
        try:
            auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
            data = {"grant_type": "client_credentials"}
            headers = {"User-Agent": self.user_agent}
            
            response = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self._access_token = token_data.get('access_token')
                self._token_expiry = datetime.utcnow() + timedelta(seconds=token_data.get('expires_in', 3600) - 60)
                return self._access_token
            
            return None
        except Exception as e:
            self.logger.error("Reddit auth failed", error=str(e))
            return None
    
    def fetch(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent posts from subreddits"""
        if not self.is_available():
            self.logger.warning("Reddit API not configured")
            return []
        
        token = self._get_access_token()
        if not token:
            return []
        
        posts = []
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self.user_agent
        }
        
        for subreddit in self.subreddits:
            try:
                response = requests.get(
                    f"https://oauth.reddit.com/r/{subreddit}/hot",
                    headers=headers,
                    params={"limit": limit // len(self.subreddits)},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for post in data.get('data', {}).get('children', []):
                        posts.append(post.get('data', {}))
            except Exception as e:
                self.logger.error("Reddit fetch failed", subreddit=subreddit, error=str(e))
        
        return posts
    
    def get_sentiment_texts(self, limit: int = 50) -> List[str]:
        """Get post titles and texts for sentiment analysis"""
        posts = self.fetch(limit)
        texts = []
        for post in posts:
            title = post.get('title', '')
            selftext = post.get('selftext', '')[:500]  # Limit text length
            if title:
                texts.append(f"{title}. {selftext}" if selftext else title)
        return texts


class NewsAPIDataSource(DataSource):
    """News API data source"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        keywords: List[str] = None
    ):
        config = get_config()
        self.api_key = api_key or config.data_ingestion.newsapi_key
        self.keywords = keywords or ["bitcoin", "cryptocurrency", "crypto market"]
        self.logger = get_logger()
        self.base_url = "https://newsapi.org/v2"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def fetch(self, page_size: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent news articles"""
        if not self.is_available():
            self.logger.warning("NewsAPI not configured")
            return []
        
        try:
            query = " OR ".join(self.keywords)
            
            response = requests.get(
                f"{self.base_url}/everything",
                params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "language": "en",
                    "apiKey": self.api_key
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                self.logger.error("NewsAPI error", status=response.status_code)
                return []
                
        except Exception as e:
            self.logger.error("News fetch failed", error=str(e))
            return []
    
    def get_sentiment_texts(self, page_size: int = 30) -> List[str]:
        """Get article titles and descriptions for sentiment analysis"""
        articles = self.fetch(page_size)
        texts = []
        for article in articles:
            title = article.get('title', '')
            desc = article.get('description', '') or ''
            if title:
                texts.append(f"{title}. {desc[:200]}")
        return texts


class CryptoPanicDataSource(DataSource):
    """CryptoPanic news aggregator"""
    
    def __init__(self, api_key: Optional[str] = None):
        config = get_config()
        self.api_key = api_key or config.data_ingestion.cryptopanic_api_key
        self.logger = get_logger()
        self.base_url = "https://cryptopanic.com/api/v1"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def fetch(self, filter_type: str = "hot") -> List[Dict[str, Any]]:
        """Fetch news from CryptoPanic"""
        if not self.is_available():
            return []
        
        try:
            response = requests.get(
                f"{self.base_url}/posts/",
                params={
                    "auth_token": self.api_key,
                    "filter": filter_type,
                    "currencies": "BTC,ETH"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            return []
        except Exception as e:
            self.logger.error("CryptoPanic fetch failed", error=str(e))
            return []
    
    def get_sentiment_texts(self) -> List[str]:
        """Get news titles for sentiment analysis"""
        posts = self.fetch()
        return [p.get('title', '') for p in posts if p.get('title')]


class DataIngestor:
    """
    Main data ingestion orchestrator
    Aggregates data from multiple sources
    """
    
    def __init__(self):
        config = get_config()
        self.config = config
        self.logger = get_logger()
        
        # Initialize data sources
        self.market_source = DeltaMarketDataSource(
            product_id=config.trading.product_id,
            product_symbol=config.trading.product_symbol
        )
        
        # Initialize sentiment sources (will be lazy-loaded if credentials exist)
        self.twitter_source = TwitterDataSource()
        self.reddit_source = RedditDataSource()
        self.news_source = NewsAPIDataSource()
        self.cryptopanic_source = CryptoPanicDataSource()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Data buffers
        self.price_buffer: deque = deque(maxlen=config.strategy.price_buffer_size)
        self.orderbook_buffer: deque = deque(maxlen=100)
        self.sentiment_buffer: deque = deque(maxlen=1000)
        
        # Current state
        self._current_state: Optional[MarketState] = None
        self._last_sentiment_update = datetime.min
        self._sentiment_update_interval = timedelta(minutes=5)
    
    def fetch_price(self) -> Optional[PriceData]:
        """Fetch and buffer current price"""
        price_data = self.market_source.fetch()
        if price_data:
            self.price_buffer.append(price_data)
            self.logger.log_market_data(
                symbol=self.config.trading.product_symbol,
                price=price_data.price,
                volume=price_data.volume,
                bid=price_data.bid,
                ask=price_data.ask
            )
        return price_data
    
    def fetch_orderbook(self) -> Optional[OrderbookSnapshot]:
        """Fetch and buffer orderbook"""
        ob = self.market_source.fetch_orderbook(
            depth=self.config.data_ingestion.orderbook_depth
        )
        if ob:
            self.orderbook_buffer.append(ob)
        return ob
    
    def fetch_sentiment(self) -> float:
        """
        Fetch and aggregate sentiment from all available sources
        Caches result for efficiency
        """
        now = datetime.utcnow()
        if now - self._last_sentiment_update < self._sentiment_update_interval:
            # Return cached sentiment
            if self.sentiment_buffer:
                recent = [s for s in self.sentiment_buffer 
                         if now - s.timestamp < timedelta(hours=1)]
                if recent:
                    return float(np.mean([s.score for s in recent]))
            return 0.0
        
        self._last_sentiment_update = now
        all_texts = []
        all_weights = []
        
        # Gather texts from all sources
        sources = [
            ("twitter", self.twitter_source, 1.5),  # Higher weight for real-time
            ("reddit", self.reddit_source, 1.0),
            ("news", self.news_source, 1.2),
            ("cryptopanic", self.cryptopanic_source, 1.3),
        ]
        
        for source_name, source, weight in sources:
            if source.is_available():
                try:
                    texts = source.get_sentiment_texts()
                    all_texts.extend(texts)
                    all_weights.extend([weight] * len(texts))
                    self.logger.debug(f"Fetched {len(texts)} texts from {source_name}")
                except Exception as e:
                    self.logger.error(f"Failed to fetch from {source_name}", error=str(e))
        
        if not all_texts:
            # Use placeholder if no sources available
            self.logger.warning("No sentiment sources available, using neutral sentiment")
            return 0.0
        
        # Analyze sentiment
        score = self.sentiment_analyzer.aggregate_sentiment(all_texts, all_weights)
        
        # Buffer individual sentiment data
        scores = self.sentiment_analyzer.analyze_batch(all_texts)
        for text, s in zip(all_texts, scores):
            self.sentiment_buffer.append(SentimentData(
                timestamp=now,
                source="aggregated",
                score=s,
                text=text[:100]
            ))
        
        self.logger.log_sentiment(
            source="aggregated",
            score=score,
            sample_size=len(all_texts)
        )
        
        return score
    
    def get_prices_array(self) -> np.ndarray:
        """Get price history as numpy array"""
        if not self.price_buffer:
            return np.array([])
        return np.array([p.price for p in self.price_buffer])
    
    def get_current_state(self) -> MarketState:
        """
        Get current aggregated market state
        This is the main method for strategy consumption
        """
        # Fetch latest data
        price_data = self.fetch_price()
        orderbook = self.fetch_orderbook()
        sentiment = self.fetch_sentiment()
        
        prices = self.get_prices_array()
        
        state = MarketState(
            timestamp=datetime.utcnow(),
            price=price_data.price if price_data else 0.0,
            prices=list(prices),
            orderbook=orderbook,
            sentiment_score=sentiment,
            sentiment_count=len(self.sentiment_buffer)
        )
        
        # Compute technical indicators if we have enough data
        if len(prices) >= self.config.strategy.ema_slow_period:
            state.ema_fast = self._compute_ema(prices, self.config.strategy.ema_fast_period)
            state.ema_slow = self._compute_ema(prices, self.config.strategy.ema_slow_period)
        
        if len(prices) >= self.config.strategy.atr_period + 1:
            state.atr = self._compute_atr(prices, self.config.strategy.atr_period)
        
        self._current_state = state
        return state
    
    def _compute_ema(self, prices: np.ndarray, period: int) -> float:
        """Compute exponential moving average"""
        if len(prices) < period:
            return float(prices[-1]) if len(prices) > 0 else 0.0
        
        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])
    
    def _compute_atr(self, prices: np.ndarray, period: int) -> float:
        """
        Compute Average True Range
        Simplified version using close-to-close returns
        """
        if len(prices) < period + 1:
            return float(prices[-1] * 0.02)  # Default 2% if not enough data
        
        returns = np.abs(np.diff(prices) / prices[:-1])
        atr = np.mean(returns[-period:]) * prices[-1]
        return float(atr)
    
    def is_warmed_up(self) -> bool:
        """Check if we have enough data for trading"""
        return len(self.price_buffer) >= self.config.strategy.warmup_period
    
    @property
    def current_price(self) -> float:
        """Get most recent price"""
        if self.price_buffer:
            return self.price_buffer[-1].price
        return 0.0


# Singleton instance
_ingestor: Optional[DataIngestor] = None


def get_data_ingestor() -> DataIngestor:
    """Get or create the global data ingestor"""
    global _ingestor
    if _ingestor is None:
        _ingestor = DataIngestor()
    return _ingestor


if __name__ == "__main__":
    # Test data ingestion
    ingestor = get_data_ingestor()
    
    print("Fetching market state...")
    state = ingestor.get_current_state()
    
    print(f"Current price: {state.price}")
    print(f"Prices in buffer: {len(state.prices)}")
    print(f"Sentiment score: {state.sentiment_score:.3f}")
    
    if state.orderbook:
        print(f"Orderbook imbalance: {state.orderbook.imbalance:.3f}")
        print(f"Spread: {state.orderbook.spread:.2f}")
    
    print(f"Warmed up: {ingestor.is_warmed_up()}")
