"""
FinBERT Sentiment Analysis
==========================
Transformer-based sentiment analysis for financial text.

This module provides:
- FinBERT model wrapper
- News sentiment scoring
- Event extraction
- Sentiment aggregation

For production use, requires: pip install transformers torch
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SENTIMENT DATA CLASSES
# =============================================================================

@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    scores: Dict[str, float]  # All class probabilities
    
    
@dataclass
class NewsSentiment:
    """Aggregated news sentiment for a symbol."""
    symbol: str
    timestamp: datetime
    overall_sentiment: float  # -1 to 1
    n_articles: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    articles: List[SentimentResult]


# =============================================================================
# KEYWORD-BASED SENTIMENT (FALLBACK)
# =============================================================================

class KeywordSentiment:
    """
    Keyword-based sentiment analysis (fallback when no GPU).
    
    Uses financial domain-specific word lists.
    """
    
    def __init__(self):
        # Positive financial words
        self.positive_words = {
            'bullish', 'bull', 'buy', 'long', 'growth', 'profit', 'gain',
            'surge', 'rally', 'breakout', 'uptrend', 'recovery', 'strong',
            'outperform', 'upgrade', 'beat', 'exceed', 'momentum', 'opportunity',
            'optimistic', 'positive', 'support', 'accumulate', 'undervalued',
            'earnings', 'revenue', 'dividend', 'expansion', 'innovation',
            'partnership', 'acquisition', 'approval', 'breakthrough', 'success',
            'record', 'high', 'peak', 'soar', 'jump', 'climb', 'rise', 'increase'
        }
        
        # Negative financial words
        self.negative_words = {
            'bearish', 'bear', 'sell', 'short', 'decline', 'loss', 'drop',
            'crash', 'plunge', 'breakdown', 'downtrend', 'recession', 'weak',
            'underperform', 'downgrade', 'miss', 'disappoint', 'risk', 'concern',
            'pessimistic', 'negative', 'resistance', 'distribute', 'overvalued',
            'debt', 'lawsuit', 'investigation', 'fraud', 'scandal', 'bankruptcy',
            'layoff', 'restructure', 'default', 'warning', 'failure', 'trouble',
            'low', 'bottom', 'fall', 'sink', 'tumble', 'slide', 'decrease'
        }
        
        # Intensifiers
        self.intensifiers = {
            'very', 'extremely', 'highly', 'significantly', 'strongly',
            'sharply', 'dramatically', 'substantially', 'massively', 'huge'
        }
        
        # Negations
        self.negations = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing',
            'nowhere', 'hardly', 'barely', 'scarcely', "n't", "don't", "doesn't"
        }
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text using keywords.
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = 0
        negative_count = 0
        total_words = len(words)
        
        # Count sentiment words with context
        for i, word in enumerate(words):
            # Check for negation in previous 3 words
            negated = any(words[max(0, i-3):i] and w in self.negations 
                         for w in words[max(0, i-3):i])
            
            # Check for intensifier
            intensified = any(w in self.intensifiers 
                            for w in words[max(0, i-2):i])
            
            multiplier = 1.5 if intensified else 1.0
            
            if word in self.positive_words:
                if negated:
                    negative_count += multiplier
                else:
                    positive_count += multiplier
                    
            elif word in self.negative_words:
                if negated:
                    positive_count += multiplier
                else:
                    negative_count += multiplier
        
        # Calculate scores
        total_sentiment = positive_count + negative_count
        
        if total_sentiment == 0:
            scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            sentiment = 'neutral'
            confidence = 0.5
        else:
            pos_ratio = positive_count / total_sentiment
            neg_ratio = negative_count / total_sentiment
            
            # Normalize with neutral component
            neutral_score = 1 - abs(pos_ratio - neg_ratio)
            
            scores = {
                'positive': pos_ratio * (1 - neutral_score * 0.5),
                'negative': neg_ratio * (1 - neutral_score * 0.5),
                'neutral': neutral_score * 0.5
            }
            
            # Normalize
            total = sum(scores.values())
            scores = {k: v/total for k, v in scores.items()}
            
            # Determine sentiment
            sentiment = max(scores, key=scores.get)
            confidence = scores[sentiment]
        
        return SentimentResult(
            text=text[:100] + '...' if len(text) > 100 else text,
            sentiment=sentiment,
            confidence=confidence,
            scores=scores
        )


# =============================================================================
# FINBERT SENTIMENT (REQUIRES TRANSFORMERS)
# =============================================================================

class FinBERTSentiment:
    """
    FinBERT-based sentiment analysis.
    
    FinBERT is a BERT model fine-tuned on financial text.
    Provides more accurate sentiment than keyword-based methods.
    
    Requires: pip install transformers torch
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert",
                 device: Optional[str] = None):
        """
        Initialize FinBERT model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None for auto
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        # Fallback to keyword sentiment
        self.keyword_sentiment = KeywordSentiment()
    
    def _load_model(self):
        """Load the FinBERT model."""
        if self._loaded:
            return True
            
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            return True
            
        except ImportError:
            print("Warning: transformers/torch not installed. Using keyword sentiment.")
            return False
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment using FinBERT.
        
        Falls back to keyword sentiment if model not available.
        """
        if not self._load_model():
            return self.keyword_sentiment.analyze(text)
        
        try:
            import torch
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            
            # Map to sentiment labels
            labels = ['positive', 'negative', 'neutral']
            scores = {label: float(prob) for label, prob in zip(labels, probs)}
            
            sentiment = max(scores, key=scores.get)
            confidence = scores[sentiment]
            
            return SentimentResult(
                text=text[:100] + '...' if len(text) > 100 else text,
                sentiment=sentiment,
                confidence=confidence,
                scores=scores
            )
            
        except Exception as e:
            print(f"FinBERT error: {e}. Falling back to keywords.")
            return self.keyword_sentiment.analyze(text)
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts efficiently.
        """
        if not self._load_model():
            return [self.keyword_sentiment.analyze(t) for t in texts]
        
        try:
            import torch
            
            # Tokenize all texts
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            # Create results
            labels = ['positive', 'negative', 'neutral']
            results = []
            
            for i, text in enumerate(texts):
                scores = {label: float(prob) for label, prob in zip(labels, probs[i])}
                sentiment = max(scores, key=scores.get)
                
                results.append(SentimentResult(
                    text=text[:100] + '...' if len(text) > 100 else text,
                    sentiment=sentiment,
                    confidence=scores[sentiment],
                    scores=scores
                ))
            
            return results
            
        except Exception as e:
            print(f"Batch analysis error: {e}")
            return [self.keyword_sentiment.analyze(t) for t in texts]


# =============================================================================
# NEWS AGGREGATOR WITH SENTIMENT
# =============================================================================

class NewsSentimentAnalyzer:
    """
    Aggregate news sentiment for trading signals.
    """
    
    def __init__(self, use_finbert: bool = True):
        """
        Initialize news sentiment analyzer.
        
        Args:
            use_finbert: Whether to use FinBERT (if available)
        """
        if use_finbert:
            self.sentiment_model = FinBERTSentiment()
        else:
            self.sentiment_model = KeywordSentiment()
    
    def analyze_news(self, articles: List[Dict]) -> NewsSentiment:
        """
        Analyze sentiment of news articles.
        
        Args:
            articles: List of dicts with 'title', 'text', 'symbol', 'timestamp'
        
        Returns:
            NewsSentiment with aggregated results
        """
        if not articles:
            return NewsSentiment(
                symbol='UNKNOWN',
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                n_articles=0,
                positive_ratio=0.33,
                negative_ratio=0.33,
                neutral_ratio=0.34,
                articles=[]
            )
        
        # Get symbol from first article
        symbol = articles[0].get('symbol', 'UNKNOWN')
        
        # Analyze each article
        results = []
        for article in articles:
            text = article.get('title', '') + ' ' + article.get('text', '')
            if text.strip():
                result = self.sentiment_model.analyze(text)
                results.append(result)
        
        if not results:
            return NewsSentiment(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                n_articles=0,
                positive_ratio=0.33,
                negative_ratio=0.33,
                neutral_ratio=0.34,
                articles=[]
            )
        
        # Aggregate
        sentiments = [r.sentiment for r in results]
        positive_ratio = sentiments.count('positive') / len(sentiments)
        negative_ratio = sentiments.count('negative') / len(sentiments)
        neutral_ratio = sentiments.count('neutral') / len(sentiments)
        
        # Overall sentiment score: weighted by confidence
        overall = 0
        total_weight = 0
        for r in results:
            if r.sentiment == 'positive':
                overall += r.confidence
            elif r.sentiment == 'negative':
                overall -= r.confidence
            total_weight += r.confidence
        
        overall_sentiment = overall / total_weight if total_weight > 0 else 0
        
        return NewsSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_sentiment=overall_sentiment,
            n_articles=len(results),
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio,
            articles=results
        )
    
    def get_trading_signal(self, news_sentiment: NewsSentiment,
                          threshold: float = 0.3) -> Dict:
        """
        Convert sentiment to trading signal.
        
        Args:
            news_sentiment: Aggregated news sentiment
            threshold: Minimum sentiment for signal
        
        Returns:
            Dictionary with signal and metadata
        """
        sentiment = news_sentiment.overall_sentiment
        
        if sentiment > threshold:
            signal = 1  # Long
            confidence = min(1.0, sentiment / threshold)
        elif sentiment < -threshold:
            signal = -1  # Short
            confidence = min(1.0, abs(sentiment) / threshold)
        else:
            signal = 0  # Neutral
            confidence = 1 - abs(sentiment) / threshold
        
        return {
            'signal': signal,
            'confidence': confidence,
            'sentiment_score': sentiment,
            'n_articles': news_sentiment.n_articles,
            'positive_ratio': news_sentiment.positive_ratio,
            'negative_ratio': news_sentiment.negative_ratio,
            'symbol': news_sentiment.symbol
        }


# =============================================================================
# EVENT EXTRACTION
# =============================================================================

class FinancialEventExtractor:
    """
    Extract financial events from news text.
    """
    
    def __init__(self):
        # Event patterns
        self.event_patterns = {
            'earnings': [
                r'earnings\s+(beat|miss|report|announce)',
                r'(beat|missed?)\s+estimates',
                r'(revenue|profit|income)\s+(increase|decrease|grow|decline)',
                r'EPS\s+of\s+\$?[\d.]+',
                r'quarterly\s+results'
            ],
            'merger_acquisition': [
                r'(acquire|acquisition|merger|merge|buy|purchase)\s+\w+',
                r'deal\s+worth\s+\$?[\d.]+\s*(million|billion)?',
                r'takeover\s+(bid|offer)',
                r'(strategic|definitive)\s+agreement'
            ],
            'analyst_rating': [
                r'(upgrade|downgrade|initiate|maintain)\s+rating',
                r'price\s+target\s+(raise|lower|cut|increase)',
                r'(buy|sell|hold|neutral)\s+rating',
                r'(outperform|underperform)\s+rating'
            ],
            'product_launch': [
                r'(launch|introduce|unveil|announce)\s+(new|product|service)',
                r'FDA\s+(approve|approval|reject)',
                r'patent\s+(grant|approve|file)',
                r'(expand|expansion)\s+into'
            ],
            'management_change': [
                r'(CEO|CFO|CTO|COO|president)\s+(step|resign|appoint)',
                r'(new|former|interim)\s+(CEO|CFO|CTO)',
                r'executive\s+(shakeup|change|departure)',
                r'board\s+(member|director)\s+(resign|appoint)'
            ],
            'regulatory': [
                r'(SEC|DOJ|FTC|FDA)\s+(investigation|probe|inquiry)',
                r'(settle|settlement)\s+with\s+(SEC|DOJ|FTC)',
                r'(fine|penalty)\s+of\s+\$?[\d.]+',
                r'regulatory\s+(approval|clearance|rejection)'
            ]
        }
    
    def extract_events(self, text: str) -> List[Dict]:
        """
        Extract financial events from text.
        
        Returns list of detected events.
        """
        text_lower = text.lower()
        events = []
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    events.append({
                        'type': event_type,
                        'pattern': pattern,
                        'matches': matches[:3],  # Limit matches
                        'confidence': min(1.0, len(matches) * 0.3 + 0.4)
                    })
                    break  # One match per event type
        
        return events
    
    def get_event_impact(self, event_type: str) -> Dict:
        """
        Get expected market impact for event type.
        """
        impact_map = {
            'earnings': {
                'volatility_impact': 'high',
                'duration': 'short',
                'direction': 'depends_on_content'
            },
            'merger_acquisition': {
                'volatility_impact': 'very_high',
                'duration': 'medium',
                'direction': 'positive_for_target'
            },
            'analyst_rating': {
                'volatility_impact': 'medium',
                'duration': 'short',
                'direction': 'depends_on_rating'
            },
            'product_launch': {
                'volatility_impact': 'medium',
                'duration': 'medium',
                'direction': 'positive'
            },
            'management_change': {
                'volatility_impact': 'medium',
                'duration': 'medium',
                'direction': 'uncertain'
            },
            'regulatory': {
                'volatility_impact': 'high',
                'duration': 'long',
                'direction': 'usually_negative'
            }
        }
        
        return impact_map.get(event_type, {
            'volatility_impact': 'unknown',
            'duration': 'unknown',
            'direction': 'unknown'
        })


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FINBERT SENTIMENT ANALYSIS")
    print("="*70)
    
    # Sample news articles
    articles = [
        {
            'title': 'Tesla Beats Q3 Earnings Estimates',
            'text': 'Tesla reported strong quarterly results, beating analyst expectations with EPS of $0.85 vs expected $0.72. Revenue grew 15% year-over-year.',
            'symbol': 'TSLA',
            'timestamp': datetime.now()
        },
        {
            'title': 'Apple Announces New iPhone Launch',
            'text': 'Apple unveiled its latest iPhone with revolutionary AI features. Analysts are optimistic about strong demand during holiday season.',
            'symbol': 'AAPL',
            'timestamp': datetime.now()
        },
        {
            'title': 'Fed Signals Rate Hikes May Continue',
            'text': 'The Federal Reserve indicated that interest rates may stay higher for longer due to persistent inflation concerns. Markets declined on the news.',
            'symbol': 'SPY',
            'timestamp': datetime.now()
        },
        {
            'title': 'Company XYZ Faces SEC Investigation',
            'text': 'XYZ Corp disclosed an SEC investigation into accounting practices. The company may face significant fines and penalties.',
            'symbol': 'XYZ',
            'timestamp': datetime.now()
        }
    ]
    
    # 1. Individual Sentiment Analysis
    print("\n1. Individual Article Sentiment (Keyword-based)")
    print("-" * 50)
    
    keyword_analyzer = KeywordSentiment()
    
    for article in articles:
        text = article['title'] + ' ' + article['text']
        result = keyword_analyzer.analyze(text)
        
        print(f"\n   [{article['symbol']}] {article['title'][:50]}...")
        print(f"   Sentiment: {result.sentiment.upper()} (confidence: {result.confidence:.2f})")
        print(f"   Scores: P={result.scores['positive']:.2f}, "
              f"N={result.scores['negative']:.2f}, "
              f"U={result.scores['neutral']:.2f}")
    
    # 2. FinBERT (if available)
    print("\n2. FinBERT Sentiment (if available)")
    print("-" * 50)
    
    finbert = FinBERTSentiment()
    
    # Try to load model
    if finbert._load_model():
        print("   FinBERT loaded successfully!")
        for article in articles[:2]:
            text = article['title'] + ' ' + article['text']
            result = finbert.analyze(text)
            print(f"\n   [{article['symbol']}] Sentiment: {result.sentiment.upper()}")
    else:
        print("   FinBERT not available (install transformers torch)")
        print("   Using keyword sentiment as fallback")
    
    # 3. News Aggregation
    print("\n3. Aggregated News Sentiment")
    print("-" * 50)
    
    analyzer = NewsSentimentAnalyzer(use_finbert=False)  # Use keywords for demo
    
    # Group by symbol
    from collections import defaultdict
    by_symbol = defaultdict(list)
    for article in articles:
        by_symbol[article['symbol']].append(article)
    
    for symbol, symbol_articles in by_symbol.items():
        sentiment = analyzer.analyze_news(symbol_articles)
        signal = analyzer.get_trading_signal(sentiment, threshold=0.3)
        
        print(f"\n   {symbol}:")
        print(f"     Articles: {sentiment.n_articles}")
        print(f"     Overall Sentiment: {sentiment.overall_sentiment:+.2f}")
        print(f"     Positive/Negative/Neutral: {sentiment.positive_ratio:.0%}/{sentiment.negative_ratio:.0%}/{sentiment.neutral_ratio:.0%}")
        print(f"     Trading Signal: {['SHORT', 'NEUTRAL', 'LONG'][signal['signal']+1]} (conf: {signal['confidence']:.2f})")
    
    # 4. Event Extraction
    print("\n4. Financial Event Extraction")
    print("-" * 50)
    
    extractor = FinancialEventExtractor()
    
    for article in articles:
        text = article['title'] + ' ' + article['text']
        events = extractor.extract_events(text)
        
        if events:
            print(f"\n   [{article['symbol']}] {article['title'][:40]}...")
            for event in events:
                impact = extractor.get_event_impact(event['type'])
                print(f"     Event: {event['type'].replace('_', ' ').title()}")
                print(f"     Impact: {impact['volatility_impact']} volatility, {impact['direction']}")
    
    print("\n" + "="*70)
    print("PRODUCTION SETUP")
    print("="*70)
    print("""
For production sentiment analysis:

1. Install FinBERT:
   pip install transformers torch
   
2. GPU Acceleration (recommended):
   pip install torch --index-url https://download.pytorch.org/whl/cu118

3. Alternative Models:
   - ProsusAI/finbert (default)
   - yiyanghkust/finbert-tone
   - ahmedrachid/FinancialBERT-Sentiment-Analysis

4. News Sources:
   - Alpha Vantage News API
   - NewsAPI.org
   - Benzinga
   - Reddit API (r/wallstreetbets, r/stocks)

5. Integration:
   from signals.finbert_sentiment import NewsSentimentAnalyzer
   
   analyzer = NewsSentimentAnalyzer(use_finbert=True)
   sentiment = analyzer.analyze_news(articles)
   signal = analyzer.get_trading_signal(sentiment)
   
   if signal['signal'] != 0 and signal['confidence'] > 0.6:
       execute_trade(signal)
""")
