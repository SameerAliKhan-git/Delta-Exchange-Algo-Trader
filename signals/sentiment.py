"""
Sentiment Analysis - Multi-source sentiment signals

Provides:
- Social media sentiment (Twitter, Reddit)
- News sentiment
- Composite sentiment scoring
- Text analysis with HuggingFace or lexicon fallback
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    source: str
    timestamp: datetime
    metadata: Dict = None


# Simple lexicon for fallback sentiment
BULLISH_WORDS = {
    'bullish', 'moon', 'pump', 'buy', 'long', 'breakout', 'rally', 'surge',
    'soar', 'rocket', 'lambo', 'hodl', 'diamond', 'hands', 'gains', 'profit',
    'green', 'up', 'high', 'ath', 'fomo', 'accumulate', 'dip', 'support'
}

BEARISH_WORDS = {
    'bearish', 'dump', 'sell', 'short', 'crash', 'tank', 'plunge', 'drop',
    'fall', 'rekt', 'red', 'down', 'low', 'fear', 'panic', 'capitulate',
    'resistance', 'breakdown', 'weak', 'exit', 'bubble', 'overvalued'
}

# Intensity modifiers
INTENSIFIERS = {'very', 'extremely', 'super', 'mega', 'ultra', 'insanely'}
NEGATORS = {'not', "n't", 'no', 'never', 'none', 'neither'}


def lexicon_sentiment(text: str) -> Tuple[float, float]:
    """
    Simple lexicon-based sentiment analysis
    
    Args:
        text: Text to analyze
    
    Returns:
        (score, confidence) where score is -1 to 1
    """
    if not text:
        return 0.0, 0.0
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return 0.0, 0.0
    
    bullish_count = 0
    bearish_count = 0
    
    # Track negation
    negate_next = False
    intensity = 1.0
    
    for word in words:
        if word in NEGATORS:
            negate_next = True
            continue
        
        if word in INTENSIFIERS:
            intensity = 1.5
            continue
        
        is_bullish = word in BULLISH_WORDS
        is_bearish = word in BEARISH_WORDS
        
        if negate_next:
            is_bullish, is_bearish = is_bearish, is_bullish
            negate_next = False
        
        if is_bullish:
            bullish_count += intensity
        elif is_bearish:
            bearish_count += intensity
        
        intensity = 1.0
    
    total = bullish_count + bearish_count
    if total == 0:
        return 0.0, 0.0
    
    score = (bullish_count - bearish_count) / total
    confidence = min(total / len(words), 1.0)  # More sentiment words = higher confidence
    
    return score, confidence


def analyze_text_sentiment(
    text: str,
    use_transformer: bool = True
) -> SentimentResult:
    """
    Analyze sentiment of text using HuggingFace or lexicon fallback
    
    Args:
        text: Text to analyze
        use_transformer: Whether to use HuggingFace model
    
    Returns:
        SentimentResult
    """
    if use_transformer:
        try:
            return _transformer_sentiment(text)
        except Exception:
            pass  # Fall back to lexicon
    
    score, confidence = lexicon_sentiment(text)
    
    return SentimentResult(
        score=score,
        confidence=confidence,
        source="lexicon",
        timestamp=datetime.utcnow()
    )


def _transformer_sentiment(text: str) -> SentimentResult:
    """
    Sentiment analysis using HuggingFace transformers
    
    Uses finbert or distilbert for crypto/financial text
    """
    try:
        from transformers import pipeline
        
        # Use cached pipeline
        if not hasattr(_transformer_sentiment, '_pipeline'):
            _transformer_sentiment._pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512
            )
        
        result = _transformer_sentiment._pipeline(text[:512])[0]
        
        # Convert label to score
        label = result['label'].lower()
        confidence = result['score']
        
        if label == 'positive':
            score = confidence
        elif label == 'negative':
            score = -confidence
        else:
            score = 0.0
        
        return SentimentResult(
            score=score,
            confidence=confidence,
            source="finbert",
            timestamp=datetime.utcnow()
        )
        
    except ImportError:
        raise Exception("transformers not installed")


def get_social_sentiment(
    symbol: str,
    sources: List[str] = None,
    lookback_hours: int = 24
) -> SentimentResult:
    """
    Get aggregated sentiment from social media
    
    Args:
        symbol: Trading symbol (e.g., 'BTC')
        sources: List of sources ['twitter', 'reddit']
        lookback_hours: Hours to look back
    
    Returns:
        Aggregated SentimentResult
    """
    sources = sources or ['twitter', 'reddit']
    results = []
    
    # Twitter sentiment
    if 'twitter' in sources:
        try:
            twitter_result = _get_twitter_sentiment(symbol, lookback_hours)
            if twitter_result:
                results.append(twitter_result)
        except Exception:
            pass
    
    # Reddit sentiment
    if 'reddit' in sources:
        try:
            reddit_result = _get_reddit_sentiment(symbol, lookback_hours)
            if reddit_result:
                results.append(reddit_result)
        except Exception:
            pass
    
    # Aggregate results
    if not results:
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            source="social_aggregate",
            timestamp=datetime.utcnow()
        )
    
    # Weighted average by confidence
    total_weight = sum(r.confidence for r in results)
    if total_weight == 0:
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            source="social_aggregate",
            timestamp=datetime.utcnow()
        )
    
    weighted_score = sum(r.score * r.confidence for r in results) / total_weight
    avg_confidence = total_weight / len(results)
    
    return SentimentResult(
        score=weighted_score,
        confidence=avg_confidence,
        source="social_aggregate",
        timestamp=datetime.utcnow(),
        metadata={"sources": [r.source for r in results]}
    )


def _get_twitter_sentiment(symbol: str, lookback_hours: int) -> Optional[SentimentResult]:
    """
    Get Twitter/X sentiment for symbol
    
    Note: Requires Twitter API credentials in config
    """
    # Placeholder - implement with Twitter API
    # Example using tweepy:
    # 
    # import tweepy
    # from config import get_config
    # 
    # config = get_config()
    # client = tweepy.Client(bearer_token=config.data_ingestion.twitter_bearer_token)
    # 
    # query = f"${symbol} OR #{symbol} -is:retweet lang:en"
    # tweets = client.search_recent_tweets(query=query, max_results=100)
    # 
    # scores = [analyze_text_sentiment(t.text).score for t in tweets.data]
    # return SentimentResult(score=np.mean(scores), ...)
    
    return None


def _get_reddit_sentiment(symbol: str, lookback_hours: int) -> Optional[SentimentResult]:
    """
    Get Reddit sentiment for symbol
    
    Note: Requires Reddit API credentials in config
    """
    # Placeholder - implement with PRAW
    # 
    # import praw
    # from config import get_config
    # 
    # config = get_config()
    # reddit = praw.Reddit(
    #     client_id=config.data_ingestion.reddit_client_id,
    #     client_secret=config.data_ingestion.reddit_client_secret,
    #     user_agent=config.data_ingestion.reddit_user_agent
    # )
    # 
    # subreddits = ['cryptocurrency', 'bitcoin', 'ethtrader']
    # posts = []
    # for sub in subreddits:
    #     posts.extend(reddit.subreddit(sub).search(symbol, time_filter='day'))
    # 
    # scores = [analyze_text_sentiment(p.title + " " + p.selftext).score for p in posts]
    # return SentimentResult(score=np.mean(scores), ...)
    
    return None


def get_news_sentiment(
    symbol: str,
    lookback_hours: int = 24
) -> SentimentResult:
    """
    Get sentiment from news sources
    
    Args:
        symbol: Trading symbol
        lookback_hours: Hours to look back
    
    Returns:
        SentimentResult from news
    """
    # Placeholder - implement with NewsAPI or CryptoPanic
    # 
    # from newsapi import NewsApiClient
    # from config import get_config
    # 
    # config = get_config()
    # newsapi = NewsApiClient(api_key=config.data_ingestion.newsapi_key)
    # 
    # articles = newsapi.get_everything(
    #     q=symbol,
    #     language='en',
    #     sort_by='publishedAt',
    #     from_param=(datetime.utcnow() - timedelta(hours=lookback_hours)).isoformat()
    # )
    # 
    # scores = [analyze_text_sentiment(a['title'] + " " + a['description']).score 
    #           for a in articles['articles']]
    
    return SentimentResult(
        score=0.0,
        confidence=0.0,
        source="news",
        timestamp=datetime.utcnow()
    )


def get_composite_sentiment(
    symbol: str,
    weights: Dict[str, float] = None
) -> SentimentResult:
    """
    Get composite sentiment from all sources
    
    Args:
        symbol: Trading symbol
        weights: Source weights {'social': 0.6, 'news': 0.4}
    
    Returns:
        Weighted composite SentimentResult
    """
    weights = weights or {'social': 0.6, 'news': 0.4}
    
    results = {}
    
    # Get social sentiment
    try:
        social = get_social_sentiment(symbol)
        results['social'] = social
    except Exception:
        pass
    
    # Get news sentiment
    try:
        news = get_news_sentiment(symbol)
        results['news'] = news
    except Exception:
        pass
    
    if not results:
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            source="composite",
            timestamp=datetime.utcnow()
        )
    
    # Weighted average
    total_weight = 0.0
    weighted_score = 0.0
    weighted_confidence = 0.0
    
    for source, result in results.items():
        weight = weights.get(source, 0.5)
        weighted_score += result.score * weight * result.confidence
        weighted_confidence += result.confidence * weight
        total_weight += weight
    
    if total_weight == 0:
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            source="composite",
            timestamp=datetime.utcnow()
        )
    
    return SentimentResult(
        score=weighted_score / total_weight,
        confidence=weighted_confidence / total_weight,
        source="composite",
        timestamp=datetime.utcnow(),
        metadata={"sources": list(results.keys())}
    )


def get_fear_greed_index() -> Tuple[int, str]:
    """
    Get Crypto Fear & Greed Index
    
    Returns:
        (value 0-100, classification string)
    """
    try:
        import requests
        
        response = requests.get(
            "https://api.alternative.me/fng/",
            timeout=10
        )
        data = response.json()
        
        if data.get('data'):
            value = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            return value, classification
        
    except Exception:
        pass
    
    return 50, "Neutral"


def sentiment_to_signal(sentiment: SentimentResult, thresholds: Dict = None) -> int:
    """
    Convert sentiment result to trading signal
    
    Args:
        sentiment: SentimentResult
        thresholds: {'bull': 0.3, 'bear': -0.3}
    
    Returns:
        1 (bullish), -1 (bearish), 0 (neutral)
    """
    thresholds = thresholds or {'bull': 0.3, 'bear': -0.3}
    
    # Require minimum confidence
    if sentiment.confidence < 0.3:
        return 0
    
    if sentiment.score > thresholds['bull']:
        return 1
    elif sentiment.score < thresholds['bear']:
        return -1
    else:
        return 0
