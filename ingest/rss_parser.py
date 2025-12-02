"""
ALADDIN - RSS Feed Parser
===========================
Parses RSS/Atom feeds from crypto news sources.
"""

import requests
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
import re

from .news_aggregator import NewsItem, NewsSource


@dataclass
class RSSFeed:
    """RSS feed configuration."""
    url: str
    name: str
    enabled: bool = True
    refresh_interval: int = 300  # seconds
    last_fetch: Optional[datetime] = None


class RSSParser:
    """
    Parses RSS/Atom feeds from crypto news sources.
    """
    
    # Default crypto RSS feeds
    DEFAULT_FEEDS = [
        RSSFeed(
            url="https://cointelegraph.com/rss",
            name="Cointelegraph"
        ),
        RSSFeed(
            url="https://cryptonews.com/news/feed/",
            name="CryptoNews"
        ),
        RSSFeed(
            url="https://news.bitcoin.com/feed/",
            name="Bitcoin.com"
        ),
        RSSFeed(
            url="https://bitcoinmagazine.com/feed",
            name="Bitcoin Magazine"
        ),
        RSSFeed(
            url="https://decrypt.co/feed",
            name="Decrypt"
        ),
        RSSFeed(
            url="https://www.coindesk.com/arc/outboundfeeds/rss/",
            name="CoinDesk"
        ),
        RSSFeed(
            url="https://thedefiant.io/api/feed",
            name="The Defiant"
        ),
    ]
    
    def __init__(self, feeds: List[RSSFeed] = None):
        self.logger = logging.getLogger('Aladdin.RSSParser')
        self.feeds = feeds or self.DEFAULT_FEEDS.copy()
        self.items: Dict[str, NewsItem] = {}
    
    def add_feed(self, url: str, name: str):
        """Add a new RSS feed."""
        feed = RSSFeed(url=url, name=name)
        self.feeds.append(feed)
        self.logger.info(f"Added RSS feed: {name}")
    
    def remove_feed(self, url: str):
        """Remove an RSS feed."""
        self.feeds = [f for f in self.feeds if f.url != url]
    
    def fetch_all(self) -> List[NewsItem]:
        """Fetch news from all enabled feeds."""
        all_items = []
        
        for feed in self.feeds:
            if not feed.enabled:
                continue
            
            try:
                items = self.fetch_feed(feed)
                all_items.extend(items)
                self.logger.info(f"Fetched {len(items)} items from {feed.name}")
            except Exception as e:
                self.logger.error(f"Error fetching {feed.name}: {e}")
        
        return all_items
    
    def fetch_feed(self, feed: RSSFeed) -> List[NewsItem]:
        """Fetch and parse a single RSS feed."""
        items = []
        
        try:
            headers = {
                'User-Agent': 'Aladdin Trading Bot/1.0',
                'Accept': 'application/rss+xml, application/xml, text/xml'
            }
            
            response = requests.get(
                feed.url,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Handle different feed formats
            if root.tag == 'rss':
                items = self._parse_rss(root, feed)
            elif root.tag.endswith('feed'):
                items = self._parse_atom(root, feed)
            else:
                self.logger.warning(f"Unknown feed format for {feed.name}")
            
            feed.last_fetch = datetime.now()
            
            # Store items
            for item in items:
                self.items[item.id] = item
            
        except Exception as e:
            self.logger.error(f"Error parsing {feed.name}: {e}")
        
        return items
    
    def _parse_rss(self, root: ET.Element, feed: RSSFeed) -> List[NewsItem]:
        """Parse RSS 2.0 format."""
        items = []
        channel = root.find('channel')
        
        if channel is None:
            return items
        
        for item in channel.findall('item'):
            try:
                title = item.findtext('title', '')
                link = item.findtext('link', '')
                description = item.findtext('description', '')
                pub_date = item.findtext('pubDate', '')
                author = item.findtext('author', '') or item.findtext('dc:creator', '')
                
                # Parse date
                pub_datetime = datetime.now()
                if pub_date:
                    try:
                        pub_datetime = parsedate_to_datetime(pub_date)
                    except:
                        pass
                
                # Clean HTML from description
                clean_desc = re.sub(r'<[^>]+>', '', description)
                
                # Create item
                news_item = NewsItem(
                    id=f"rss_{hashlib.md5(link.encode()).hexdigest()[:12]}",
                    title=title,
                    content=clean_desc,
                    source=NewsSource.RSS_FEED,
                    source_name=feed.name,
                    url=link,
                    published_at=pub_datetime,
                    author=author if author else None,
                    symbols=self._extract_symbols(title + ' ' + clean_desc)
                )
                items.append(news_item)
                
            except Exception as e:
                self.logger.debug(f"Error parsing RSS item: {e}")
        
        return items
    
    def _parse_atom(self, root: ET.Element, feed: RSSFeed) -> List[NewsItem]:
        """Parse Atom format."""
        items = []
        
        # Handle namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns) or root.findall('entry'):
            try:
                title_elem = entry.find('atom:title', ns) or entry.find('title')
                title = title_elem.text if title_elem is not None else ''
                
                link_elem = entry.find('atom:link', ns) or entry.find('link')
                link = link_elem.get('href', '') if link_elem is not None else ''
                
                content_elem = entry.find('atom:content', ns) or entry.find('content') or \
                              entry.find('atom:summary', ns) or entry.find('summary')
                content = content_elem.text if content_elem is not None else ''
                
                published_elem = entry.find('atom:published', ns) or entry.find('published') or \
                                entry.find('atom:updated', ns) or entry.find('updated')
                pub_date = published_elem.text if published_elem is not None else ''
                
                # Parse date
                pub_datetime = datetime.now()
                if pub_date:
                    try:
                        pub_datetime = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    except:
                        pass
                
                # Clean HTML
                clean_content = re.sub(r'<[^>]+>', '', content or '')
                
                news_item = NewsItem(
                    id=f"rss_{hashlib.md5(link.encode()).hexdigest()[:12]}",
                    title=title,
                    content=clean_content,
                    source=NewsSource.RSS_FEED,
                    source_name=feed.name,
                    url=link,
                    published_at=pub_datetime,
                    symbols=self._extract_symbols(title + ' ' + clean_content)
                )
                items.append(news_item)
                
            except Exception as e:
                self.logger.debug(f"Error parsing Atom entry: {e}")
        
        return items
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract crypto symbols from text."""
        text_upper = text.upper()
        symbols = []
        
        crypto_keywords = {
            'BITCOIN': 'BTC', 'BTC': 'BTC',
            'ETHEREUM': 'ETH', 'ETH': 'ETH',
            'XRP': 'XRP', 'RIPPLE': 'XRP',
            'SOLANA': 'SOL', 'SOL': 'SOL',
            'CARDANO': 'ADA', 'ADA': 'ADA',
            'DOGECOIN': 'DOGE', 'DOGE': 'DOGE',
            'POLYGON': 'MATIC', 'MATIC': 'MATIC',
        }
        
        for keyword, symbol in crypto_keywords.items():
            if keyword in text_upper and symbol not in symbols:
                symbols.append(symbol)
        
        return symbols
    
    def get_recent(self, hours: int = 24) -> List[NewsItem]:
        """Get recent items from all feeds."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=hours)
        
        items = [
            item for item in self.items.values()
            if item.published_at >= cutoff
        ]
        
        return sorted(items, key=lambda x: x.published_at, reverse=True)
    
    def get_stats(self) -> Dict:
        """Get parser statistics."""
        return {
            'total_feeds': len(self.feeds),
            'enabled_feeds': sum(1 for f in self.feeds if f.enabled),
            'total_items': len(self.items),
            'feeds': [
                {
                    'name': f.name,
                    'url': f.url,
                    'enabled': f.enabled,
                    'last_fetch': f.last_fetch.isoformat() if f.last_fetch else None
                }
                for f in self.feeds
            ]
        }
    
    def print_summary(self):
        """Print summary to console."""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("📡 RSS FEED PARSER")
        print("="*70)
        print(f"Configured Feeds: {stats['total_feeds']}")
        print(f"Total Items: {stats['total_items']}")
        
        print("\n📋 FEEDS:")
        for feed in stats['feeds']:
            status = "✅" if feed['enabled'] else "❌"
            print(f"  {status} {feed['name']}")
        
        # Show recent headlines
        recent = self.get_recent(hours=6)[:5]
        if recent:
            print("\n📌 LATEST FROM RSS:")
            for item in recent:
                print(f"  [{item.source_name}] {item.title[:55]}...")
        
        print("="*70)


# Test the parser
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = RSSParser()
    parser.fetch_all()
    parser.print_summary()
