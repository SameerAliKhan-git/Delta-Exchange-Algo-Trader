"""
ALADDIN - Social Media Connectors
===================================
Reddit, Twitter/X, and Telegram data ingestion.
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import os
import json
import time

from .news_aggregator import NewsItem, NewsSource


class RedditConnector:
    """
    Fetches crypto discussions from Reddit.
    Uses public JSON API (no authentication required for read-only).
    """
    
    # Default subreddits to monitor
    DEFAULT_SUBREDDITS = [
        'cryptocurrency',
        'bitcoin', 
        'ethereum',
        'CryptoMarkets',
        'ethtrader',
        'CryptoCurrency',
        'SatoshiStreetBets',
    ]
    
    def __init__(self, subreddits: List[str] = None):
        self.logger = logging.getLogger('Aladdin.Reddit')
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS
        self.items: Dict[str, NewsItem] = {}
        self.last_fetch: Optional[datetime] = None
        
        # Rate limiting
        self._last_request = 0
        self._min_interval = 2  # seconds between requests
    
    def fetch_all(self, limit: int = 25) -> List[NewsItem]:
        """Fetch from all configured subreddits."""
        all_items = []
        
        for subreddit in self.subreddits:
            try:
                items = self.fetch_subreddit(subreddit, limit=limit)
                all_items.extend(items)
                self.logger.info(f"Fetched {len(items)} posts from r/{subreddit}")
            except Exception as e:
                self.logger.error(f"Error fetching r/{subreddit}: {e}")
        
        self.last_fetch = datetime.now()
        return all_items
    
    def fetch_subreddit(self, subreddit: str, sort: str = 'hot', limit: int = 25) -> List[NewsItem]:
        """Fetch posts from a specific subreddit."""
        # Rate limiting
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        
        items = []
        
        try:
            headers = {
                'User-Agent': 'Aladdin Trading Bot/1.0 (by /u/aladdin_bot)'
            }
            
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {'limit': limit}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            self._last_request = time.time()
            
            if response.status_code == 429:
                self.logger.warning("Reddit rate limited, waiting...")
                time.sleep(60)
                return items
            
            response.raise_for_status()
            data = response.json()
            
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                
                # Skip stickied posts
                if post_data.get('stickied'):
                    continue
                
                # Create news item
                item = NewsItem(
                    id=f"reddit_{post_data.get('id', '')}",
                    title=post_data.get('title', ''),
                    content=post_data.get('selftext', '')[:1000],
                    source=NewsSource.REDDIT,
                    source_name=f"r/{subreddit}",
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    published_at=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                    author=post_data.get('author'),
                    symbols=self._extract_symbols(
                        post_data.get('title', '') + ' ' + post_data.get('selftext', '')
                    ),
                    importance=self._calculate_importance(post_data),
                    raw_data={
                        'score': post_data.get('score', 0),
                        'upvote_ratio': post_data.get('upvote_ratio', 0),
                        'num_comments': post_data.get('num_comments', 0),
                        'subreddit': subreddit
                    }
                )
                
                items.append(item)
                self.items[item.id] = item
                
        except Exception as e:
            self.logger.error(f"Error fetching r/{subreddit}: {e}")
        
        return items
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract crypto symbols from text."""
        text_upper = text.upper()
        symbols = []
        
        crypto_keywords = {
            'BITCOIN': 'BTC', 'BTC': 'BTC', '$BTC': 'BTC',
            'ETHEREUM': 'ETH', 'ETH': 'ETH', '$ETH': 'ETH',
            'XRP': 'XRP', 'RIPPLE': 'XRP',
            'SOLANA': 'SOL', 'SOL': 'SOL', '$SOL': 'SOL',
            'CARDANO': 'ADA', 'ADA': 'ADA',
            'DOGECOIN': 'DOGE', 'DOGE': 'DOGE',
            'POLYGON': 'MATIC', 'MATIC': 'MATIC',
        }
        
        for keyword, symbol in crypto_keywords.items():
            if keyword in text_upper and symbol not in symbols:
                symbols.append(symbol)
        
        return symbols
    
    def _calculate_importance(self, post_data: Dict) -> float:
        """Calculate post importance based on engagement."""
        score = post_data.get('score', 0)
        comments = post_data.get('num_comments', 0)
        ratio = post_data.get('upvote_ratio', 0.5)
        
        # Normalize to 0-1
        score_factor = min(1.0, score / 1000)
        comment_factor = min(1.0, comments / 100)
        
        importance = (score_factor * 0.4 + comment_factor * 0.3 + ratio * 0.3)
        return round(importance, 2)
    
    def get_trending(self, min_score: int = 100) -> List[NewsItem]:
        """Get trending posts above score threshold."""
        return [
            item for item in self.items.values()
            if item.raw_data.get('score', 0) >= min_score
        ]
    
    def get_sentiment_sample(self, symbol: str = None, limit: int = 50) -> List[str]:
        """Get text samples for sentiment analysis."""
        items = list(self.items.values())
        
        if symbol:
            items = [i for i in items if symbol.upper() in i.symbols]
        
        # Sort by importance
        items.sort(key=lambda x: x.importance, reverse=True)
        
        return [f"{i.title} {i.content}"[:500] for i in items[:limit]]


class TwitterConnector:
    """
    Twitter/X connector.
    Note: Requires Twitter API v2 Bearer Token for real usage.
    This is a placeholder that can be activated with API keys.
    """
    
    def __init__(self, bearer_token: str = None):
        self.logger = logging.getLogger('Aladdin.Twitter')
        self.bearer_token = bearer_token or os.environ.get('TWITTER_BEARER_TOKEN')
        self.items: Dict[str, NewsItem] = {}
        self.enabled = bool(self.bearer_token)
        
        if not self.enabled:
            self.logger.warning("Twitter connector disabled: No bearer token")
    
    def fetch_search(self, query: str, limit: int = 100) -> List[NewsItem]:
        """Search recent tweets."""
        if not self.enabled:
            return []
        
        # Twitter API v2 implementation would go here
        # For now, return empty list
        self.logger.info(f"Twitter search: {query} (placeholder)")
        return []
    
    def fetch_user_timeline(self, username: str, limit: int = 50) -> List[NewsItem]:
        """Fetch tweets from a specific user."""
        if not self.enabled:
            return []
        
        self.logger.info(f"Twitter user: {username} (placeholder)")
        return []
    
    def get_crypto_influencers(self) -> List[str]:
        """List of crypto influencers to follow."""
        return [
            'VitalikButerin',
            'caborello',
            'APompliano',
            'SBF_FTX',
            'CryptoCapo_',
            'CryptoCobain',
            'trader1sz',
            'TheCryptoDog',
        ]


class TelegramConnector:
    """
    Telegram channel monitor.
    Note: Requires Telegram Bot API token for real usage.
    This is a placeholder that can be activated with API keys.
    """
    
    # Popular crypto Telegram channels
    CRYPTO_CHANNELS = [
        'WhalAlert',
        'Cointelegraph',
        'CryptoNewsRoom',
        'BitcoinMagazine',
    ]
    
    def __init__(self, bot_token: str = None):
        self.logger = logging.getLogger('Aladdin.Telegram')
        self.bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.items: Dict[str, NewsItem] = {}
        self.enabled = bool(self.bot_token)
        
        if not self.enabled:
            self.logger.warning("Telegram connector disabled: No bot token")
    
    def fetch_channel(self, channel: str, limit: int = 50) -> List[NewsItem]:
        """Fetch messages from a Telegram channel."""
        if not self.enabled:
            return []
        
        # Telegram API implementation would go here
        self.logger.info(f"Telegram channel: {channel} (placeholder)")
        return []
    
    def fetch_all_channels(self) -> List[NewsItem]:
        """Fetch from all configured channels."""
        if not self.enabled:
            return []
        
        all_items = []
        for channel in self.CRYPTO_CHANNELS:
            items = self.fetch_channel(channel)
            all_items.extend(items)
        
        return all_items


# Test the connectors
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("🔌 SOCIAL MEDIA CONNECTORS TEST")
    print("="*60)
    
    # Test Reddit
    print("\n📱 Testing Reddit...")
    reddit = RedditConnector()
    reddit_items = reddit.fetch_subreddit('cryptocurrency', limit=10)
    print(f"  Fetched {len(reddit_items)} posts from r/cryptocurrency")
    
    if reddit_items:
        print("\n  Top Posts:")
        for item in reddit_items[:3]:
            score = item.raw_data.get('score', 0)
            print(f"    [{score}] {item.title[:50]}...")
    
    # Twitter status
    print("\n🐦 Twitter Status:")
    twitter = TwitterConnector()
    print(f"  Enabled: {twitter.enabled}")
    
    # Telegram status
    print("\n📨 Telegram Status:")
    telegram = TelegramConnector()
    print(f"  Enabled: {telegram.enabled}")
    
    print("\n" + "="*60)
