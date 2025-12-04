"""
Strategy 7: Event-Driven Trading
================================

Crypto reacts STRONGLY to:
- Binance/FTX news
- Twitter & sentiment
- On-chain metrics
- Whale movements
- Liquidation cascades
- Funding spikes
- NFT hype cycles

These strategies WIN because crypto is NEWS-DRIVEN.

Concepts:
- NLP embeddings
- Twitter sentiment
- Crypto Fear & Greed Index
- Whale wallet tracking
- Liquidation map modeling
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from .base import (
    BaseStrategy, StrategyConfig, Signal, SignalType,
    TechnicalIndicators
)


class EventType(Enum):
    """Types of market-moving events."""
    NEWS_POSITIVE = "news_positive"
    NEWS_NEGATIVE = "news_negative"
    WHALE_BUY = "whale_buy"
    WHALE_SELL = "whale_sell"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    FUNDING_SPIKE = "funding_spike"
    EXCHANGE_LISTING = "exchange_listing"
    EXCHANGE_DELISTING = "exchange_delisting"
    HACK = "hack"
    REGULATORY = "regulatory"
    SOCIAL_VIRAL = "social_viral"
    FEAR_EXTREME = "fear_extreme"
    GREED_EXTREME = "greed_extreme"


@dataclass
class Event:
    """A market event."""
    timestamp: datetime
    event_type: EventType
    symbol: Optional[str]
    magnitude: float  # 0 to 1
    source: str
    description: str
    sentiment_score: float  # -1 to 1
    metadata: Dict = field(default_factory=dict)


@dataclass
class EventConfig(StrategyConfig):
    """Configuration for event-driven strategies."""
    name: str = "event_driven"
    
    # Sentiment thresholds
    sentiment_entry_threshold: float = 0.6  # Strong sentiment
    fear_greed_extreme: float = 20  # < 20 extreme fear, > 80 extreme greed
    
    # Whale detection
    whale_threshold_usd: float = 1000000  # $1M moves
    whale_lookback_hours: int = 24
    
    # Liquidation
    liquidation_threshold_usd: float = 10000000  # $10M liquidations
    liquidation_cascade_threshold: float = 50000000  # $50M = cascade
    
    # Event timing
    event_decay_hours: float = 24  # Events lose relevance
    min_event_magnitude: float = 0.5
    
    # Volatility backdrop
    prefer_low_vol_events: bool = True
    max_volatility_for_entry: float = 0.04
    
    # Regime
    allowed_regimes: List[str] = field(default_factory=lambda: ["all"])


class SentimentAnalyzer:
    """
    Crypto-specific sentiment analysis.
    
    Combines multiple sources:
    - News headlines
    - Twitter sentiment
    - Reddit sentiment
    - Fear & Greed Index
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self._sentiment_history: deque = deque(maxlen=1000)
        self._news_sentiment: float = 0.0
        self._social_sentiment: float = 0.0
        self._fear_greed_index: float = 50.0
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of text.
        
        Returns score from -1 (bearish) to 1 (bullish).
        
        In production, use FinBERT or similar.
        """
        # Crypto-specific keywords
        bullish_words = [
            'bullish', 'moon', 'pump', 'breakout', 'support', 'buy',
            'long', 'accumulate', 'undervalued', 'adoption', 'partnership',
            'institutional', 'approval', 'listing', 'upgrade', 'hodl',
            'ath', 'rally', 'surge'
        ]
        
        bearish_words = [
            'bearish', 'dump', 'crash', 'sell', 'short', 'resistance',
            'overvalued', 'hack', 'exploit', 'delisting', 'ban',
            'investigation', 'scam', 'rug', 'liquidation', 'capitulation',
            'fear', 'panic'
        ]
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def update_fear_greed(self, value: float):
        """Update Fear & Greed Index (0-100)."""
        self._fear_greed_index = value
    
    def update_news_sentiment(self, headlines: List[str]) -> float:
        """Analyze news headlines and update sentiment."""
        if not headlines:
            return self._news_sentiment
        
        scores = [self.analyze_text(h) for h in headlines]
        self._news_sentiment = np.mean(scores)
        
        return self._news_sentiment
    
    def update_social_sentiment(self, posts: List[Dict]) -> float:
        """
        Analyze social media sentiment.
        
        Posts should have 'text' and 'engagement' keys.
        """
        if not posts:
            return self._social_sentiment
        
        weighted_scores = []
        for post in posts:
            score = self.analyze_text(post.get('text', ''))
            engagement = post.get('engagement', 1)
            weighted_scores.append(score * np.log1p(engagement))
        
        total_engagement = sum(np.log1p(p.get('engagement', 1)) for p in posts)
        
        if total_engagement > 0:
            self._social_sentiment = sum(weighted_scores) / total_engagement
        
        return self._social_sentiment
    
    def get_composite_sentiment(self) -> Dict[str, float]:
        """Get weighted composite sentiment score."""
        # Weights for different sources
        weights = {
            'news': 0.3,
            'social': 0.3,
            'fear_greed': 0.4
        }
        
        # Normalize fear/greed to -1 to 1
        fg_normalized = (self._fear_greed_index - 50) / 50
        
        composite = (
            weights['news'] * self._news_sentiment +
            weights['social'] * self._social_sentiment +
            weights['fear_greed'] * fg_normalized
        )
        
        return {
            'composite': composite,
            'news': self._news_sentiment,
            'social': self._social_sentiment,
            'fear_greed': self._fear_greed_index,
            'fear_greed_normalized': fg_normalized
        }
    
    def detect_sentiment_extreme(self) -> Optional[EventType]:
        """Detect extreme sentiment conditions."""
        if self._fear_greed_index < 20:
            return EventType.FEAR_EXTREME
        elif self._fear_greed_index > 80:
            return EventType.GREED_EXTREME
        return None


class WhaleWatcher:
    """
    Track large wallet movements.
    
    Whale movements often precede price action:
    - Large exchange deposits = potential sell
    - Large exchange withdrawals = potential buy (cold storage)
    - Large transfers between wallets = repositioning
    """
    
    def __init__(self, threshold_usd: float = 1000000):
        """
        Initialize whale watcher.
        
        Args:
            threshold_usd: Minimum USD value to track
        """
        self.threshold_usd = threshold_usd
        
        self._whale_transactions: deque = deque(maxlen=1000)
        self._exchange_inflows: float = 0.0
        self._exchange_outflows: float = 0.0
    
    def add_transaction(self, tx: Dict):
        """
        Add whale transaction.
        
        tx should have: timestamp, value_usd, from_type, to_type, symbol
        """
        if tx.get('value_usd', 0) < self.threshold_usd:
            return
        
        self._whale_transactions.append(tx)
        
        # Track exchange flows
        if tx.get('to_type') == 'exchange':
            self._exchange_inflows += tx['value_usd']
        elif tx.get('from_type') == 'exchange':
            self._exchange_outflows += tx['value_usd']
    
    def get_flow_signal(self, hours: int = 24) -> Dict[str, float]:
        """
        Get exchange flow signal.
        
        High inflows = bearish (selling)
        High outflows = bullish (holding)
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_txs = [
            tx for tx in self._whale_transactions
            if tx.get('timestamp', datetime.min) > cutoff
        ]
        
        inflows = sum(
            tx['value_usd'] for tx in recent_txs 
            if tx.get('to_type') == 'exchange'
        )
        
        outflows = sum(
            tx['value_usd'] for tx in recent_txs
            if tx.get('from_type') == 'exchange'
        )
        
        net_flow = outflows - inflows
        
        return {
            'inflows_usd': inflows,
            'outflows_usd': outflows,
            'net_flow_usd': net_flow,
            'flow_signal': np.sign(net_flow) * np.log1p(abs(net_flow)) / 20,  # Normalized
            'is_bullish': net_flow > self.threshold_usd,
            'is_bearish': net_flow < -self.threshold_usd
        }
    
    def detect_whale_event(self) -> Optional[Event]:
        """Detect significant whale activity."""
        flows = self.get_flow_signal(hours=4)  # Short-term
        
        if flows['is_bullish'] and flows['net_flow_usd'] > self.threshold_usd * 5:
            return Event(
                timestamp=datetime.now(),
                event_type=EventType.WHALE_BUY,
                symbol=None,
                magnitude=min(1.0, flows['net_flow_usd'] / (self.threshold_usd * 10)),
                source='on_chain',
                description=f"Large exchange outflows: ${flows['net_flow_usd']:,.0f}",
                sentiment_score=0.7
            )
        
        elif flows['is_bearish'] and flows['net_flow_usd'] < -self.threshold_usd * 5:
            return Event(
                timestamp=datetime.now(),
                event_type=EventType.WHALE_SELL,
                symbol=None,
                magnitude=min(1.0, abs(flows['net_flow_usd']) / (self.threshold_usd * 10)),
                source='on_chain',
                description=f"Large exchange inflows: ${abs(flows['net_flow_usd']):,.0f}",
                sentiment_score=-0.7
            )
        
        return None


class LiquidationTracker:
    """
    Track liquidation events.
    
    Liquidation cascades create predictable price action:
    - Long liquidations = price drops further (cascade)
    - Short liquidations = price pumps further (short squeeze)
    """
    
    def __init__(self, cascade_threshold: float = 50000000):
        """
        Initialize liquidation tracker.
        
        Args:
            cascade_threshold: USD amount indicating cascade
        """
        self.cascade_threshold = cascade_threshold
        
        self._liquidations: deque = deque(maxlen=10000)
        self._hourly_liquidations: Dict[str, float] = {'long': 0, 'short': 0}
    
    def add_liquidation(self, liq: Dict):
        """
        Add liquidation event.
        
        liq should have: timestamp, side (long/short), value_usd, symbol, price
        """
        self._liquidations.append(liq)
        
        # Update hourly totals
        self._hourly_liquidations[liq.get('side', 'long')] += liq.get('value_usd', 0)
    
    def get_liquidation_pressure(self, hours: int = 1) -> Dict[str, float]:
        """Get recent liquidation pressure."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = [
            liq for liq in self._liquidations
            if liq.get('timestamp', datetime.min) > cutoff
        ]
        
        long_liqs = sum(l['value_usd'] for l in recent if l.get('side') == 'long')
        short_liqs = sum(l['value_usd'] for l in recent if l.get('side') == 'short')
        
        total = long_liqs + short_liqs
        
        return {
            'long_liquidations_usd': long_liqs,
            'short_liquidations_usd': short_liqs,
            'total_liquidations_usd': total,
            'ratio': long_liqs / (short_liqs + 1) if short_liqs > 0 else float('inf'),
            'is_cascade': total > self.cascade_threshold,
            'pressure_direction': 'down' if long_liqs > short_liqs else 'up'
        }
    
    def detect_cascade_event(self) -> Optional[Event]:
        """Detect liquidation cascade."""
        pressure = self.get_liquidation_pressure(hours=0.5)  # 30 min
        
        if pressure['is_cascade']:
            if pressure['pressure_direction'] == 'down':
                return Event(
                    timestamp=datetime.now(),
                    event_type=EventType.LIQUIDATION_CASCADE,
                    symbol=None,
                    magnitude=min(1.0, pressure['total_liquidations_usd'] / (self.cascade_threshold * 2)),
                    source='liquidations',
                    description=f"Long liquidation cascade: ${pressure['long_liquidations_usd']:,.0f}",
                    sentiment_score=-0.8,
                    metadata={'direction': 'down'}
                )
            else:
                return Event(
                    timestamp=datetime.now(),
                    event_type=EventType.LIQUIDATION_CASCADE,
                    symbol=None,
                    magnitude=min(1.0, pressure['total_liquidations_usd'] / (self.cascade_threshold * 2)),
                    source='liquidations',
                    description=f"Short squeeze: ${pressure['short_liquidations_usd']:,.0f}",
                    sentiment_score=0.8,
                    metadata={'direction': 'up'}
                )
        
        return None


class EventAggregator:
    """
    Aggregates events from multiple sources.
    
    Weighs and combines events to generate actionable signals.
    """
    
    def __init__(self, decay_hours: float = 24):
        """
        Initialize event aggregator.
        
        Args:
            decay_hours: Time for events to decay in relevance
        """
        self.decay_hours = decay_hours
        
        self._events: deque = deque(maxlen=1000)
        self._event_weights = {
            EventType.NEWS_POSITIVE: 0.7,
            EventType.NEWS_NEGATIVE: 0.7,
            EventType.WHALE_BUY: 0.8,
            EventType.WHALE_SELL: 0.8,
            EventType.LIQUIDATION_CASCADE: 0.9,
            EventType.FUNDING_SPIKE: 0.6,
            EventType.EXCHANGE_LISTING: 0.9,
            EventType.EXCHANGE_DELISTING: 0.95,
            EventType.HACK: 1.0,
            EventType.REGULATORY: 0.85,
            EventType.SOCIAL_VIRAL: 0.5,
            EventType.FEAR_EXTREME: 0.7,
            EventType.GREED_EXTREME: 0.7
        }
    
    def add_event(self, event: Event):
        """Add event to aggregator."""
        self._events.append(event)
    
    def _calculate_decay(self, event: Event) -> float:
        """Calculate time decay factor for event."""
        age_hours = (datetime.now() - event.timestamp).total_seconds() / 3600
        return np.exp(-age_hours / self.decay_hours)
    
    def get_aggregate_signal(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Get aggregate signal from all recent events.
        
        Args:
            symbol: Filter to specific symbol (None = all)
        
        Returns:
            Aggregate signal metrics
        """
        relevant_events = [
            e for e in self._events
            if symbol is None or e.symbol is None or e.symbol == symbol
        ]
        
        if not relevant_events:
            return {'signal': 0, 'confidence': 0, 'event_count': 0}
        
        weighted_sentiment = 0
        total_weight = 0
        
        for event in relevant_events:
            decay = self._calculate_decay(event)
            weight = self._event_weights.get(event.event_type, 0.5) * event.magnitude * decay
            
            weighted_sentiment += event.sentiment_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return {'signal': 0, 'confidence': 0, 'event_count': len(relevant_events)}
        
        signal = weighted_sentiment / total_weight
        confidence = min(1.0, total_weight / 5)  # Normalize
        
        return {
            'signal': signal,
            'confidence': confidence,
            'event_count': len(relevant_events),
            'total_weight': total_weight,
            'direction': 'bullish' if signal > 0 else 'bearish'
        }
    
    def get_strongest_event(self) -> Optional[Event]:
        """Get the most impactful recent event."""
        if not self._events:
            return None
        
        return max(
            self._events,
            key=lambda e: self._event_weights.get(e.event_type, 0.5) * e.magnitude * self._calculate_decay(e)
        )


class SentimentTrader(BaseStrategy):
    """
    Trade based on sentiment signals.
    
    Strategy:
    - Long on extreme fear (contrarian)
    - Short on extreme greed (contrarian)
    - Or momentum on strong directional sentiment
    """
    
    def __init__(self, config: Optional[EventConfig] = None, contrarian: bool = True):
        super().__init__(config or EventConfig())
        self.config: EventConfig = self.config
        
        self.contrarian = contrarian  # True = buy fear, sell greed
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def update(self, data: pd.DataFrame):
        """Update strategy state."""
        self.current_bar = len(data) - 1
        
        # In production, this would receive real sentiment data
        # Here we simulate based on price action
        returns = data['close'].pct_change()
        recent_return = returns.iloc[-10:].sum()
        
        # Simulate fear/greed based on recent returns
        simulated_fg = 50 + recent_return * 500  # Scale returns to FG
        simulated_fg = max(0, min(100, simulated_fg))
        
        self.sentiment_analyzer.update_fear_greed(simulated_fg)
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate sentiment-based signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        sentiment = self.sentiment_analyzer.get_composite_sentiment()
        fg = sentiment['fear_greed']
        
        signal_type = None
        reason = ""
        strength = 0.0
        
        if self.contrarian:
            # Contrarian: Buy fear, sell greed
            if fg < self.config.fear_greed_extreme:
                signal_type = SignalType.LONG
                reason = f"Extreme fear ({fg:.0f}) - contrarian BUY"
                strength = min(1.0, (self.config.fear_greed_extreme - fg) / 20)
            
            elif fg > (100 - self.config.fear_greed_extreme):
                signal_type = SignalType.SHORT
                reason = f"Extreme greed ({fg:.0f}) - contrarian SELL"
                strength = min(1.0, (fg - (100 - self.config.fear_greed_extreme)) / 20)
        else:
            # Momentum: Follow sentiment
            if fg > 60 and sentiment['composite'] > 0.5:
                signal_type = SignalType.LONG
                reason = f"Bullish sentiment ({fg:.0f}, {sentiment['composite']:.2f})"
                strength = min(1.0, sentiment['composite'])
            
            elif fg < 40 and sentiment['composite'] < -0.5:
                signal_type = SignalType.SHORT
                reason = f"Bearish sentiment ({fg:.0f}, {sentiment['composite']:.2f})"
                strength = min(1.0, abs(sentiment['composite']))
        
        if signal_type is None:
            return None
        
        atr = TechnicalIndicators.atr(
            data['high'], data['low'], data['close'], 14
        ).iloc[-1]
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=0.6,
            atr=atr,
            fear_greed=fg
        )


class EventDrivenStrategy(BaseStrategy):
    """
    Complete event-driven trading strategy.
    
    Combines:
    - Sentiment analysis
    - Whale watching
    - Liquidation tracking
    - Event aggregation
    
    Trade on events + volatility backdrop.
    """
    
    def __init__(self, config: Optional[EventConfig] = None):
        super().__init__(config or EventConfig())
        self.config: EventConfig = self.config
        
        self.sentiment_analyzer = SentimentAnalyzer()
        self.whale_watcher = WhaleWatcher(self.config.whale_threshold_usd)
        self.liquidation_tracker = LiquidationTracker(self.config.liquidation_cascade_threshold)
        self.event_aggregator = EventAggregator(self.config.event_decay_hours)
        
        self._volatility: Optional[float] = None
    
    def update(self, data: pd.DataFrame):
        """Update strategy state."""
        self.current_bar = len(data) - 1
        
        # Calculate volatility
        returns = data['close'].pct_change()
        self._volatility = returns.iloc[-20:].std() * np.sqrt(24)  # Daily
        
        # Simulate events based on price action (in production, use real data)
        self._simulate_events(data)
        
        self.is_initialized = True
    
    def _simulate_events(self, data: pd.DataFrame):
        """Simulate events for demonstration."""
        close = data['close']
        returns = close.pct_change()
        
        recent_return = returns.iloc[-5:].sum()
        volatility = returns.iloc[-20:].std()
        
        # Simulate whale events
        if abs(recent_return) > 0.05:
            if recent_return > 0:
                event = Event(
                    timestamp=datetime.now(),
                    event_type=EventType.WHALE_BUY,
                    symbol=None,
                    magnitude=min(1.0, abs(recent_return) / 0.1),
                    source='simulated',
                    description="Simulated whale buy",
                    sentiment_score=0.7
                )
            else:
                event = Event(
                    timestamp=datetime.now(),
                    event_type=EventType.WHALE_SELL,
                    symbol=None,
                    magnitude=min(1.0, abs(recent_return) / 0.1),
                    source='simulated',
                    description="Simulated whale sell",
                    sentiment_score=-0.7
                )
            self.event_aggregator.add_event(event)
        
        # Simulate sentiment events
        fg = self.sentiment_analyzer._fear_greed_index
        if fg < 20:
            event = Event(
                timestamp=datetime.now(),
                event_type=EventType.FEAR_EXTREME,
                symbol=None,
                magnitude=0.8,
                source='fear_greed',
                description=f"Extreme fear: {fg}",
                sentiment_score=-0.5  # Negative sentiment but contrarian opportunity
            )
            self.event_aggregator.add_event(event)
        elif fg > 80:
            event = Event(
                timestamp=datetime.now(),
                event_type=EventType.GREED_EXTREME,
                symbol=None,
                magnitude=0.8,
                source='fear_greed',
                description=f"Extreme greed: {fg}",
                sentiment_score=0.5
            )
            self.event_aggregator.add_event(event)
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate event-driven signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        # Check volatility backdrop
        if self.config.prefer_low_vol_events:
            if self._volatility > self.config.max_volatility_for_entry:
                return None
        
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        # Get aggregate event signal
        event_signal = self.event_aggregator.get_aggregate_signal(symbol)
        
        if event_signal['event_count'] == 0:
            return None
        
        if event_signal['confidence'] < 0.3:
            return None
        
        signal_type = None
        reason = ""
        
        # Strong bullish events
        if event_signal['signal'] > self.config.sentiment_entry_threshold:
            signal_type = SignalType.LONG
            reason = f"Bullish events ({event_signal['event_count']}): signal={event_signal['signal']:.2f}"
        
        # Strong bearish events
        elif event_signal['signal'] < -self.config.sentiment_entry_threshold:
            signal_type = SignalType.SHORT
            reason = f"Bearish events ({event_signal['event_count']}): signal={event_signal['signal']:.2f}"
        
        # Check for contrarian extreme fear/greed
        strongest = self.event_aggregator.get_strongest_event()
        if strongest and strongest.event_type == EventType.FEAR_EXTREME:
            signal_type = SignalType.LONG
            reason = f"Extreme fear contrarian: {strongest.description}"
        elif strongest and strongest.event_type == EventType.GREED_EXTREME:
            signal_type = SignalType.SHORT
            reason = f"Extreme greed contrarian: {strongest.description}"
        
        if signal_type is None:
            return None
        
        atr = TechnicalIndicators.atr(
            data['high'], data['low'], data['close'], 14
        ).iloc[-1]
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=abs(event_signal['signal']),
            confidence=event_signal['confidence'],
            atr=atr,
            event_count=event_signal['event_count'],
            volatility=self._volatility
        )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EVENT-DRIVEN TRADING STRATEGIES")
    print("="*70)
    
    np.random.seed(42)
    
    print("\n1. SENTIMENT ANALYSIS")
    print("-" * 50)
    
    analyzer = SentimentAnalyzer()
    
    # Test headlines
    headlines = [
        "Bitcoin surges past $50k on institutional adoption",
        "Ethereum upgrade successful, bullish momentum continues",
        "SEC investigation into major crypto exchange",
        "Whale wallet moves $100M to Coinbase",
        "Record liquidations hit crypto markets"
    ]
    
    for headline in headlines:
        score = analyzer.analyze_text(headline)
        print(f"   '{headline[:50]}...'")
        print(f"      Sentiment: {score:+.2f}")
    
    # Update Fear & Greed
    analyzer.update_fear_greed(15)  # Extreme fear
    composite = analyzer.get_composite_sentiment()
    print(f"\n   Composite sentiment: {composite['composite']:.2f}")
    print(f"   Fear & Greed: {composite['fear_greed']}")
    
    print("\n2. WHALE WATCHING")
    print("-" * 50)
    
    whale_watcher = WhaleWatcher(threshold_usd=1000000)
    
    # Simulate whale transactions
    transactions = [
        {'timestamp': datetime.now(), 'value_usd': 5000000, 'from_type': 'exchange', 'to_type': 'wallet'},
        {'timestamp': datetime.now(), 'value_usd': 3000000, 'from_type': 'exchange', 'to_type': 'wallet'},
        {'timestamp': datetime.now(), 'value_usd': 2000000, 'from_type': 'wallet', 'to_type': 'exchange'},
    ]
    
    for tx in transactions:
        whale_watcher.add_transaction(tx)
    
    flows = whale_watcher.get_flow_signal(hours=24)
    print(f"   Exchange inflows: ${flows['inflows_usd']:,.0f}")
    print(f"   Exchange outflows: ${flows['outflows_usd']:,.0f}")
    print(f"   Net flow signal: {flows['flow_signal']:.3f}")
    print(f"   Is bullish: {flows['is_bullish']}")
    
    print("\n3. LIQUIDATION TRACKING")
    print("-" * 50)
    
    liq_tracker = LiquidationTracker(cascade_threshold=50000000)
    
    # Simulate liquidations
    for _ in range(100):
        liq = {
            'timestamp': datetime.now(),
            'side': 'long' if np.random.random() > 0.3 else 'short',
            'value_usd': np.random.exponential(500000),
            'symbol': 'BTCUSDT'
        }
        liq_tracker.add_liquidation(liq)
    
    pressure = liq_tracker.get_liquidation_pressure(hours=1)
    print(f"   Long liquidations: ${pressure['long_liquidations_usd']:,.0f}")
    print(f"   Short liquidations: ${pressure['short_liquidations_usd']:,.0f}")
    print(f"   Ratio (long/short): {pressure['ratio']:.2f}")
    print(f"   Pressure direction: {pressure['pressure_direction']}")
    
    print("\n4. EVENT AGGREGATION")
    print("-" * 50)
    
    aggregator = EventAggregator()
    
    events = [
        Event(datetime.now(), EventType.WHALE_BUY, 'BTCUSDT', 0.8, 'chain', 'Large outflow', 0.7),
        Event(datetime.now() - timedelta(hours=2), EventType.NEWS_POSITIVE, None, 0.6, 'news', 'ETF approval', 0.8),
        Event(datetime.now() - timedelta(hours=10), EventType.FEAR_EXTREME, None, 0.9, 'fg', 'Fear at 15', -0.3),
    ]
    
    for event in events:
        aggregator.add_event(event)
    
    agg_signal = aggregator.get_aggregate_signal()
    print(f"   Aggregate signal: {agg_signal['signal']:.2f}")
    print(f"   Confidence: {agg_signal['confidence']:.2f}")
    print(f"   Event count: {agg_signal['event_count']}")
    print(f"   Direction: {agg_signal['direction']}")
    
    print("\n5. COMPLETE STRATEGY")
    print("-" * 50)
    
    n = 200
    prices = 50000 + np.cumsum(np.random.randn(n) * 500)
    
    data = pd.DataFrame({
        'open': prices - 100,
        'high': prices + 200,
        'low': prices - 200,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n),
        'symbol': 'BTCUSDT'
    })
    
    strategy = EventDrivenStrategy(EventConfig(
        sentiment_entry_threshold=0.4,
        prefer_low_vol_events=False  # For demo
    ))
    
    signals = []
    for i in range(50, len(data)):
        strategy.current_bar = i
        strategy.update(data.iloc[:i+1])
        signal = strategy.generate_signal(data.iloc[:i+1])
        if signal:
            signals.append(signal)
    
    print(f"   Total signals: {len(signals)}")
    if signals:
        print(f"   Last signal: {signals[-1].reason}")
    
    print("\n" + "="*70)
    print("EVENT-DRIVEN TRADING KEY INSIGHTS")
    print("="*70)
    print("""
1. CRYPTO IS NEWS-DRIVEN
   Unlike equities, crypto reacts violently to:
   - Twitter posts from influencers
   - Exchange news (listings, hacks)
   - Regulatory announcements
   - Whale movements
   
2. KEY EVENT SOURCES:
   a) Sentiment:
      - Fear & Greed Index
      - Twitter sentiment
      - Reddit/Discord buzz
   
   b) On-Chain:
      - Whale wallet movements
      - Exchange inflows/outflows
      - Stablecoin flows
   
   c) Market:
      - Liquidation cascades
      - Funding rate spikes
      - Open interest changes
   
3. TRADING APPROACHES:
   
   CONTRARIAN (Higher win rate):
   - Buy extreme fear (<20)
   - Sell extreme greed (>80)
   - Requires patience
   
   MOMENTUM (Higher profit per trade):
   - Follow strong sentiment
   - Ride news-driven moves
   - Faster entry/exit
   
4. VOLATILITY BACKDROP MATTERS:
   - Low vol + bullish event = STRONG long
   - High vol + bullish event = Risky long
   - Best: Event in quiet market
   
5. EVENT DECAY:
   - Events lose relevance over time
   - 24-hour decay for most events
   - Breaking news: First 1-4 hours critical
   
6. PRACTICAL IMPLEMENTATION:
   - Use Coinglass for liquidation data
   - Use alternative.me for Fear & Greed
   - Track whale wallets via Whale Alert
   - Parse crypto Twitter for sentiment
   
7. RISK MANAGEMENT:
   - Size DOWN during high event uncertainty
   - Wider stops for event trades
   - Don't chase after event is priced in
""")
