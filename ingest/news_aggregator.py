"""
ALADDIN - News Aggregator
===========================
Aggregates news from multiple sources into a unified stream.
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import hashlib
import json
import time
import os


class NewsSource(Enum):
    """Supported news sources."""
    NEWS_API = auto()
    CRYPTOCOMPARE = auto()
    RSS_FEED = auto()
    TWITTER = auto()
    REDDIT = auto()
    TELEGRAM = auto()
    CUSTOM = auto()


@dataclass
class NewsItem:
    """Represents a single news item."""
    id: str
    title: str
    content: str
    source: NewsSource
    source_name: str
    url: str
    published_at: datetime
    author: Optional[str] = None
    image_url: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)  # Related crypto symbols
    sentiment_score: float = 0.0  # -1 to 1
    importance: float = 0.5  # 0 to 1
    raw_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content[:500],  # Truncate for storage
            'source': self.source.name,
            'source_name': self.source_name,
            'url': self.url,
            'published_at': self.published_at.isoformat(),
            'symbols': self.symbols,
            'sentiment_score': self.sentiment_score,
            'importance': self.importance
        }


class NewsAggregator:
    """
    Aggregates news from multiple sources with deduplication.
    """
    
    # NewsAPI endpoint
    NEWSAPI_URL = "https://newsapi.org/v2/everything"
    
    # CryptoCompare endpoint
    CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger('Aladdin.NewsAggregator')
        self.config = config or self._default_config()
        
        # News storage
        self.news_items: Dict[str, NewsItem] = {}
        self.last_fetch: Dict[NewsSource, datetime] = {}
        
        # Deduplication
        self._seen_hashes: set = set()
        
        # API keys from environment or config
        self.newsapi_key = os.environ.get('NEWSAPI_KEY', self.config.get('newsapi_key', ''))
        
    def _default_config(self) -> Dict:
        return {
            'newsapi_key': '',
            'refresh_interval': 300,  # 5 minutes
            'max_age_hours': 24,
            'keywords': ['bitcoin', 'ethereum', 'crypto', 'cryptocurrency', 'BTC', 'ETH'],
            'excluded_sources': [],
            'max_items_per_source': 50,
        }
    
    def fetch_all(self) -> List[NewsItem]:
        """Fetch news from all enabled sources."""
        all_news = []
        
        # CryptoCompare (always available, no API key needed)
        try:
            crypto_news = self.fetch_cryptocompare()
            all_news.extend(crypto_news)
            self.logger.info(f"Fetched {len(crypto_news)} items from CryptoCompare")
        except Exception as e:
            self.logger.error(f"CryptoCompare fetch error: {e}")
        
        # NewsAPI (requires API key)
        if self.newsapi_key:
            try:
                newsapi_items = self.fetch_newsapi()
                all_news.extend(newsapi_items)
                self.logger.info(f"Fetched {len(newsapi_items)} items from NewsAPI")
            except Exception as e:
                self.logger.error(f"NewsAPI fetch error: {e}")
        
        # Store and dedupe
        for item in all_news:
            self._add_item(item)
        
        return list(self.news_items.values())
    
    def fetch_cryptocompare(self) -> List[NewsItem]:
        """Fetch news from CryptoCompare."""
        try:
            params = {
                'lang': 'EN',
                'categories': 'BTC,ETH,Trading,Market',
                'excludeCategories': 'Sponsored'
            }
            
            response = requests.get(
                self.CRYPTOCOMPARE_NEWS_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            items = []
            for article in data.get('Data', []):
                item = NewsItem(
                    id=f"cc_{article['id']}",
                    title=article.get('title', ''),
                    content=article.get('body', ''),
                    source=NewsSource.CRYPTOCOMPARE,
                    source_name=article.get('source', 'CryptoCompare'),
                    url=article.get('url', ''),
                    published_at=datetime.fromtimestamp(article.get('published_on', 0)),
                    image_url=article.get('imageurl'),
                    categories=article.get('categories', '').split('|'),
                    symbols=self._extract_symbols(article.get('title', '') + ' ' + article.get('body', '')),
                    raw_data=article
                )
                items.append(item)
            
            self.last_fetch[NewsSource.CRYPTOCOMPARE] = datetime.now()
            return items
            
        except Exception as e:
            self.logger.error(f"CryptoCompare error: {e}")
            return []
    
    def fetch_newsapi(self, query: str = None) -> List[NewsItem]:
        """Fetch news from NewsAPI.org."""
        if not self.newsapi_key:
            self.logger.warning("NewsAPI key not configured")
            return []
        
        try:
            keywords = query or ' OR '.join(self.config['keywords'])
            
            params = {
                'q': keywords,
                'apiKey': self.newsapi_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': self.config['max_items_per_source']
            }
            
            response = requests.get(
                self.NEWSAPI_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                self.logger.error(f"NewsAPI error: {data.get('message')}")
                return []
            
            items = []
            for article in data.get('articles', []):
                # Skip if no content
                if not article.get('title'):
                    continue
                
                # Parse date
                pub_date = datetime.now()
                if article.get('publishedAt'):
                    try:
                        pub_date = datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        )
                    except:
                        pass
                
                item = NewsItem(
                    id=f"na_{hashlib.md5(article.get('url', '').encode()).hexdigest()[:12]}",
                    title=article.get('title', ''),
                    content=article.get('description', '') or article.get('content', ''),
                    source=NewsSource.NEWS_API,
                    source_name=article.get('source', {}).get('name', 'NewsAPI'),
                    url=article.get('url', ''),
                    published_at=pub_date,
                    author=article.get('author'),
                    image_url=article.get('urlToImage'),
                    symbols=self._extract_symbols(article.get('title', '') + ' ' + (article.get('description', '') or '')),
                    raw_data=article
                )
                items.append(item)
            
            self.last_fetch[NewsSource.NEWS_API] = datetime.now()
            return items
            
        except Exception as e:
            self.logger.error(f"NewsAPI error: {e}")
            return []
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract crypto symbols mentioned in text."""
        text_upper = text.upper()
        symbols = []
        
        # Common crypto symbols
        crypto_map = {
            'BITCOIN': 'BTC',
            'ETHEREUM': 'ETH',
            'BTC': 'BTC',
            'ETH': 'ETH',
            'XRP': 'XRP',
            'RIPPLE': 'XRP',
            'SOLANA': 'SOL',
            'SOL': 'SOL',
            'CARDANO': 'ADA',
            'ADA': 'ADA',
            'DOGECOIN': 'DOGE',
            'DOGE': 'DOGE',
            'POLYGON': 'MATIC',
            'MATIC': 'MATIC',
            'AVALANCHE': 'AVAX',
            'AVAX': 'AVAX',
            'CHAINLINK': 'LINK',
            'LINK': 'LINK',
        }
        
        for keyword, symbol in crypto_map.items():
            if keyword in text_upper:
                if symbol not in symbols:
                    symbols.append(symbol)
        
        return symbols
    
    def _add_item(self, item: NewsItem):
        """Add item with deduplication."""
        # Create hash for dedup
        content_hash = hashlib.md5(
            (item.title + item.source_name).encode()
        ).hexdigest()
        
        if content_hash in self._seen_hashes:
            return  # Duplicate
        
        self._seen_hashes.add(content_hash)
        self.news_items[item.id] = item
    
    def get_recent(self, hours: int = 24, symbols: List[str] = None) -> List[NewsItem]:
        """Get recent news, optionally filtered by symbols."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        items = [
            item for item in self.news_items.values()
            if item.published_at >= cutoff
        ]
        
        if symbols:
            items = [
                item for item in items
                if any(s in item.symbols for s in symbols)
            ]
        
        # Sort by recency
        items.sort(key=lambda x: x.published_at, reverse=True)
        
        return items
    
    def get_by_source(self, source: NewsSource) -> List[NewsItem]:
        """Get news from a specific source."""
        return [
            item for item in self.news_items.values()
            if item.source == source
        ]
    
    def get_stats(self) -> Dict:
        """Get aggregator statistics."""
        by_source = {}
        for source in NewsSource:
            by_source[source.name] = len(self.get_by_source(source))
        
        return {
            'total_items': len(self.news_items),
            'by_source': by_source,
            'last_fetch': {
                k.name: v.isoformat() 
                for k, v in self.last_fetch.items()
            },
            'unique_symbols': list(set(
                s for item in self.news_items.values()
                for s in item.symbols
            ))
        }
    
    def cleanup_old(self, hours: int = None):
        """Remove news older than threshold."""
        hours = hours or self.config['max_age_hours']
        cutoff = datetime.now() - timedelta(hours=hours)
        
        old_ids = [
            id for id, item in self.news_items.items()
            if item.published_at < cutoff
        ]
        
        for id in old_ids:
            del self.news_items[id]
        
        if old_ids:
            self.logger.info(f"Cleaned up {len(old_ids)} old news items")
    
    def print_summary(self):
        """Print news summary to console."""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("📰 NEWS AGGREGATOR SUMMARY")
        print("="*70)
        print(f"Total Items: {stats['total_items']}")
        print("\nBy Source:")
        for source, count in stats['by_source'].items():
            if count > 0:
                print(f"  • {source}: {count}")
        
        print(f"\nSymbols Mentioned: {', '.join(stats['unique_symbols'][:10])}")
        
        # Show recent headlines
        recent = self.get_recent(hours=6)[:5]
        if recent:
            print("\n📌 LATEST HEADLINES:")
            for item in recent:
                age = datetime.now() - item.published_at
                age_str = f"{int(age.total_seconds() / 60)}m ago"
                print(f"  [{age_str}] {item.title[:60]}...")
        
        print("="*70)


# Test the aggregator
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    aggregator = NewsAggregator()
    aggregator.fetch_all()
    aggregator.print_summary()
