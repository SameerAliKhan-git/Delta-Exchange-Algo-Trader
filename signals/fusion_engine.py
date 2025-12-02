"""
ALADDIN - Signal Fusion Engine
================================
Multi-modal signal aggregation with confidence scoring and decay windows.
Combines technical, sentiment, and orderbook signals into actionable trading signals.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import numpy as np


class SignalType(Enum):
    """Types of signals."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    ORDERBOOK = "orderbook"
    ORDERFLOW = "orderflow"
    NEWS = "news"
    SOCIAL = "social"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"


class SignalDirection(Enum):
    """Signal direction."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class SignalComponent:
    """Individual signal component from a single source."""
    source: SignalType
    direction: SignalDirection
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    weight: float  # Configured weight for this source
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    decay_window: int = 300  # seconds before signal decays
    
    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds()
    
    @property
    def decayed_strength(self) -> float:
        """Get strength with time decay applied."""
        if self.age_seconds >= self.decay_window:
            return 0.0
        decay_factor = math.exp(-self.age_seconds / (self.decay_window / 2))
        return self.strength * decay_factor
    
    @property
    def weighted_strength(self) -> float:
        """Get decayed strength multiplied by weight and confidence."""
        return self.decayed_strength * self.weight * self.confidence


@dataclass
class FusedSignal:
    """Aggregated signal from multiple sources."""
    symbol: str
    direction: SignalDirection
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    components: List[SignalComponent] = field(default_factory=list)
    agreement_ratio: float = 0.0  # How many sources agree
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: List[str] = field(default_factory=list)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on."""
        return (
            self.direction != SignalDirection.NEUTRAL and
            abs(self.strength) >= 0.3 and
            self.confidence >= 0.5 and
            self.agreement_ratio >= 0.5
        )
    
    @property
    def trade_direction(self) -> str:
        """Get string direction for trading."""
        if self.direction == SignalDirection.LONG:
            return "long"
        elif self.direction == SignalDirection.SHORT:
            return "short"
        return "neutral"
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'strength': round(self.strength, 4),
            'confidence': round(self.confidence, 4),
            'agreement_ratio': round(self.agreement_ratio, 4),
            'is_actionable': self.is_actionable,
            'reasoning': self.reasoning,
            'components': len(self.components),
            'timestamp': self.timestamp.isoformat()
        }


class SignalFusion:
    """
    Multi-modal signal fusion engine.
    
    Combines signals from:
    - Technical analysis (trend, momentum, volatility)
    - Sentiment analysis (news, social, fear/greed)
    - Order book analysis (imbalance, depth, walls)
    - Order flow analysis (CVD, absorption, whale detection)
    """
    
    # Default weights for each signal type
    DEFAULT_WEIGHTS = {
        SignalType.TECHNICAL: 0.35,
        SignalType.SENTIMENT: 0.20,
        SignalType.ORDERBOOK: 0.20,
        SignalType.ORDERFLOW: 0.15,
        SignalType.NEWS: 0.05,
        SignalType.SOCIAL: 0.05,
    }
    
    # Default decay windows (seconds)
    DEFAULT_DECAY_WINDOWS = {
        SignalType.TECHNICAL: 300,    # 5 minutes
        SignalType.SENTIMENT: 3600,   # 1 hour
        SignalType.ORDERBOOK: 60,     # 1 minute
        SignalType.ORDERFLOW: 120,    # 2 minutes
        SignalType.NEWS: 7200,        # 2 hours
        SignalType.SOCIAL: 1800,      # 30 minutes
    }
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger('Aladdin.SignalFusion')
        self.config = config or {}
        
        # Configure weights
        self.weights = self.config.get('weights', self.DEFAULT_WEIGHTS.copy())
        self.decay_windows = self.config.get('decay_windows', self.DEFAULT_DECAY_WINDOWS.copy())
        
        # Thresholds
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.min_agreement = self.config.get('min_agreement', 0.5)
        self.min_strength = self.config.get('min_strength', 0.3)
        
        # Signal storage
        self._signals: Dict[str, List[SignalComponent]] = {}  # symbol -> signals
        
    def add_signal(self, symbol: str, signal: SignalComponent):
        """Add a signal component."""
        if symbol not in self._signals:
            self._signals[symbol] = []
        
        # Apply decay window from config
        if signal.source in self.decay_windows:
            signal.decay_window = self.decay_windows[signal.source]
        
        # Apply weight from config
        if signal.source in self.weights:
            signal.weight = self.weights[signal.source]
        
        self._signals[symbol].append(signal)
        self._cleanup_old_signals(symbol)
    
    def _cleanup_old_signals(self, symbol: str):
        """Remove fully decayed signals."""
        if symbol not in self._signals:
            return
        
        self._signals[symbol] = [
            s for s in self._signals[symbol]
            if s.age_seconds < s.decay_window * 2
        ]
    
    def fuse(self, symbol: str) -> FusedSignal:
        """
        Fuse all signals for a symbol into a single actionable signal.
        
        Uses weighted average with:
        - Source weights
        - Confidence scores
        - Time decay
        - Agreement bonus
        """
        if symbol not in self._signals or not self._signals[symbol]:
            return FusedSignal(
                symbol=symbol,
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                confidence=0.0
            )
        
        signals = self._signals[symbol]
        
        # Calculate weighted strength
        total_weight = 0.0
        weighted_sum = 0.0
        reasoning = []
        
        # Track direction votes
        long_votes = 0
        short_votes = 0
        neutral_votes = 0
        
        # Track by source type
        by_source: Dict[SignalType, List[SignalComponent]] = {}
        
        for sig in signals:
            weighted_strength = sig.weighted_strength
            
            if abs(weighted_strength) > 0.01:
                weighted_sum += weighted_strength
                total_weight += sig.weight * sig.confidence
                
                # Count votes
                if sig.direction == SignalDirection.LONG:
                    long_votes += sig.confidence
                elif sig.direction == SignalDirection.SHORT:
                    short_votes += sig.confidence
                else:
                    neutral_votes += sig.confidence
                
                # Group by source
                if sig.source not in by_source:
                    by_source[sig.source] = []
                by_source[sig.source].append(sig)
                
                # Add reasoning
                reasoning.append(
                    f"{sig.source.value}: {sig.direction.value} "
                    f"({sig.strength:+.2f}, conf={sig.confidence:.0%})"
                )
        
        # Calculate final strength
        if total_weight > 0:
            final_strength = weighted_sum / total_weight
        else:
            final_strength = 0.0
        
        # Clamp to [-1, 1]
        final_strength = max(-1.0, min(1.0, final_strength))
        
        # Determine direction
        total_votes = long_votes + short_votes + neutral_votes
        if total_votes > 0:
            if long_votes > short_votes and long_votes > neutral_votes:
                direction = SignalDirection.LONG
                agreement_ratio = long_votes / total_votes
            elif short_votes > long_votes and short_votes > neutral_votes:
                direction = SignalDirection.SHORT
                agreement_ratio = short_votes / total_votes
            else:
                direction = SignalDirection.NEUTRAL
                agreement_ratio = neutral_votes / total_votes
        else:
            direction = SignalDirection.NEUTRAL
            agreement_ratio = 0.0
        
        # Calculate confidence
        # Based on: signal agreement, number of sources, average confidence
        source_diversity = len(by_source) / len(self.weights)
        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0
        confidence = (agreement_ratio * 0.4 + source_diversity * 0.3 + avg_confidence * 0.3)
        
        # Apply agreement bonus to strength
        if agreement_ratio > 0.7:
            final_strength *= 1.2
            final_strength = max(-1.0, min(1.0, final_strength))
        
        return FusedSignal(
            symbol=symbol,
            direction=direction,
            strength=final_strength,
            confidence=confidence,
            components=signals.copy(),
            agreement_ratio=agreement_ratio,
            reasoning=reasoning
        )
    
    def add_technical_signal(self, symbol: str, trend: float, momentum: float, 
                            volatility: float, rsi: float):
        """Add technical analysis signals."""
        # Trend signal
        trend_direction = SignalDirection.LONG if trend > 0.2 else (
            SignalDirection.SHORT if trend < -0.2 else SignalDirection.NEUTRAL
        )
        
        trend_sig = SignalComponent(
            source=SignalType.TECHNICAL,
            direction=trend_direction,
            strength=trend,
            confidence=min(1.0, abs(trend) + 0.3),
            weight=self.weights[SignalType.TECHNICAL],
            metadata={'indicator': 'trend', 'value': trend}
        )
        self.add_signal(symbol, trend_sig)
        
        # Momentum signal
        mom_direction = SignalDirection.LONG if momentum > 0.3 else (
            SignalDirection.SHORT if momentum < -0.3 else SignalDirection.NEUTRAL
        )
        
        mom_sig = SignalComponent(
            source=SignalType.MOMENTUM,
            direction=mom_direction,
            strength=momentum,
            confidence=min(1.0, abs(momentum) + 0.2),
            weight=self.weights.get(SignalType.MOMENTUM, 0.1),
            metadata={'indicator': 'momentum', 'value': momentum}
        )
        self.add_signal(symbol, mom_sig)
        
        # RSI signal (mean reversion)
        if rsi < 30:
            rsi_direction = SignalDirection.LONG  # Oversold
            rsi_strength = (30 - rsi) / 30
        elif rsi > 70:
            rsi_direction = SignalDirection.SHORT  # Overbought
            rsi_strength = -(rsi - 70) / 30
        else:
            rsi_direction = SignalDirection.NEUTRAL
            rsi_strength = 0
        
        rsi_sig = SignalComponent(
            source=SignalType.TECHNICAL,
            direction=rsi_direction,
            strength=rsi_strength,
            confidence=0.7 if abs(rsi_strength) > 0.5 else 0.4,
            weight=self.weights[SignalType.TECHNICAL] * 0.5,
            metadata={'indicator': 'rsi', 'value': rsi}
        )
        self.add_signal(symbol, rsi_sig)
    
    def add_sentiment_signal(self, symbol: str, sentiment_score: float, 
                            fear_greed: float, news_sentiment: float):
        """Add sentiment analysis signals."""
        # Overall sentiment
        sent_direction = SignalDirection.LONG if sentiment_score > 0.2 else (
            SignalDirection.SHORT if sentiment_score < -0.2 else SignalDirection.NEUTRAL
        )
        
        sent_sig = SignalComponent(
            source=SignalType.SENTIMENT,
            direction=sent_direction,
            strength=sentiment_score,
            confidence=0.6,
            weight=self.weights[SignalType.SENTIMENT],
            metadata={'fear_greed': fear_greed}
        )
        self.add_signal(symbol, sent_sig)
        
        # Fear/Greed contrarian signal
        # Extreme fear = buying opportunity, extreme greed = selling opportunity
        if fear_greed < 25:
            fg_direction = SignalDirection.LONG
            fg_strength = (25 - fear_greed) / 25
        elif fear_greed > 75:
            fg_direction = SignalDirection.SHORT
            fg_strength = -(fear_greed - 75) / 25
        else:
            fg_direction = SignalDirection.NEUTRAL
            fg_strength = 0
        
        fg_sig = SignalComponent(
            source=SignalType.SENTIMENT,
            direction=fg_direction,
            strength=fg_strength,
            confidence=0.5,
            weight=self.weights[SignalType.SENTIMENT] * 0.5,
            metadata={'fear_greed': fear_greed, 'contrarian': True}
        )
        self.add_signal(symbol, fg_sig)
        
        # News sentiment
        if abs(news_sentiment) > 0.1:
            news_direction = SignalDirection.LONG if news_sentiment > 0 else SignalDirection.SHORT
            
            news_sig = SignalComponent(
                source=SignalType.NEWS,
                direction=news_direction,
                strength=news_sentiment,
                confidence=0.5,
                weight=self.weights.get(SignalType.NEWS, 0.05),
                metadata={'news_sentiment': news_sentiment}
            )
            self.add_signal(symbol, news_sig)
    
    def add_orderbook_signal(self, symbol: str, imbalance: float, 
                            bid_depth: float, ask_depth: float,
                            wall_detected: Optional[str] = None):
        """Add order book analysis signals."""
        # Imbalance signal
        imb_direction = SignalDirection.LONG if imbalance > 0.2 else (
            SignalDirection.SHORT if imbalance < -0.2 else SignalDirection.NEUTRAL
        )
        
        imb_sig = SignalComponent(
            source=SignalType.ORDERBOOK,
            direction=imb_direction,
            strength=imbalance,
            confidence=min(1.0, abs(imbalance) * 1.5),
            weight=self.weights[SignalType.ORDERBOOK],
            metadata={
                'imbalance': imbalance,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth
            }
        )
        self.add_signal(symbol, imb_sig)
        
        # Wall detection signal
        if wall_detected:
            wall_direction = SignalDirection.SHORT if wall_detected == 'ask' else SignalDirection.LONG
            wall_sig = SignalComponent(
                source=SignalType.ORDERBOOK,
                direction=wall_direction,
                strength=0.3 if wall_detected == 'ask' else 0.3,
                confidence=0.6,
                weight=self.weights[SignalType.ORDERBOOK] * 0.5,
                metadata={'wall': wall_detected}
            )
            self.add_signal(symbol, wall_sig)
    
    def add_orderflow_signal(self, symbol: str, cvd: float, 
                            delta: float, absorption: float):
        """Add order flow analysis signals."""
        # CVD trend
        cvd_direction = SignalDirection.LONG if cvd > 0 else (
            SignalDirection.SHORT if cvd < 0 else SignalDirection.NEUTRAL
        )
        
        cvd_sig = SignalComponent(
            source=SignalType.ORDERFLOW,
            direction=cvd_direction,
            strength=max(-1, min(1, cvd / 1000)),  # Normalize CVD
            confidence=0.7,
            weight=self.weights[SignalType.ORDERFLOW],
            metadata={'cvd': cvd, 'delta': delta}
        )
        self.add_signal(symbol, cvd_sig)
        
        # Absorption signal
        if abs(absorption) > 0.5:
            abs_direction = SignalDirection.LONG if absorption > 0 else SignalDirection.SHORT
            abs_sig = SignalComponent(
                source=SignalType.ORDERFLOW,
                direction=abs_direction,
                strength=absorption,
                confidence=0.6,
                weight=self.weights[SignalType.ORDERFLOW] * 0.7,
                metadata={'absorption': absorption}
            )
            self.add_signal(symbol, abs_sig)
    
    def get_signal(self, symbol: str) -> FusedSignal:
        """Get fused signal for symbol."""
        return self.fuse(symbol)
    
    def get_all_signals(self) -> Dict[str, FusedSignal]:
        """Get fused signals for all tracked symbols."""
        return {symbol: self.fuse(symbol) for symbol in self._signals}
    
    def print_report(self, symbol: str):
        """Print signal fusion report."""
        signal = self.fuse(symbol)
        
        print("\n" + "="*70)
        print(f"⚡ SIGNAL FUSION REPORT - {symbol}")
        print("="*70)
        
        # Direction with emoji
        if signal.direction == SignalDirection.LONG:
            dir_str = "🟢 LONG"
        elif signal.direction == SignalDirection.SHORT:
            dir_str = "🔴 SHORT"
        else:
            dir_str = "⚪ NEUTRAL"
        
        print(f"\nFused Signal: {dir_str}")
        print(f"Strength: {signal.strength:+.4f}")
        print(f"Confidence: {signal.confidence:.1%}")
        print(f"Agreement: {signal.agreement_ratio:.1%}")
        print(f"Actionable: {'✅ YES' if signal.is_actionable else '❌ NO'}")
        
        # Component breakdown
        if signal.components:
            print(f"\n📊 COMPONENTS ({len(signal.components)}):")
            by_source: Dict[str, List[SignalComponent]] = {}
            for comp in signal.components:
                key = comp.source.value
                if key not in by_source:
                    by_source[key] = []
                by_source[key].append(comp)
            
            for source, comps in by_source.items():
                avg_strength = np.mean([c.weighted_strength for c in comps])
                print(f"  {source:<12}: {avg_strength:+.3f} ({len(comps)} signals)")
        
        # Reasoning
        if signal.reasoning:
            print(f"\n📝 REASONING:")
            for r in signal.reasoning[:5]:
                print(f"  • {r}")
        
        print("="*70)


# Test the fusion engine
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fusion = SignalFusion()
    
    # Add sample signals for BTCUSD
    fusion.add_technical_signal('BTCUSD', trend=0.6, momentum=0.4, volatility=0.3, rsi=35)
    fusion.add_sentiment_signal('BTCUSD', sentiment_score=0.3, fear_greed=30, news_sentiment=0.2)
    fusion.add_orderbook_signal('BTCUSD', imbalance=0.4, bid_depth=1000000, ask_depth=800000)
    fusion.add_orderflow_signal('BTCUSD', cvd=500, delta=200, absorption=0.3)
    
    fusion.print_report('BTCUSD')
