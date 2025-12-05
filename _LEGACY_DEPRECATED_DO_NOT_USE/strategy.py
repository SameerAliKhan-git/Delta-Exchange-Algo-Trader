"""
Trading Strategy Module for Delta Exchange Algo Trading Bot
Multi-modal strategy combining momentum, technicals, and sentiment
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from config import get_config
from logger import get_logger
from data_ingest import MarketState, get_data_ingestor


class SignalDirection(Enum):
    """Trading signal direction"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Signal:
    """Trading signal"""
    direction: SignalDirection
    strength: float  # 0 to 1
    price: float
    timestamp: datetime
    reasons: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.NEUTRAL and self.strength >= 0.5


@dataclass
class StrategyState:
    """Internal strategy state"""
    last_signal: Optional[Signal] = None
    momentum_score: float = 0.0
    sentiment_score: float = 0.0
    orderbook_score: float = 0.0
    technical_score: float = 0.0
    composite_score: float = 0.0


class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.logger = get_logger()
        self.config = get_config()
    
    @abstractmethod
    def evaluate(self, market_state: MarketState) -> Signal:
        """Evaluate market state and return trading signal"""
        pass
    
    @abstractmethod
    def is_ready(self, market_state: MarketState) -> bool:
        """Check if strategy has enough data to evaluate"""
        pass


class MomentumIndicator:
    """Momentum-based indicators"""
    
    @staticmethod
    def ema_crossover(
        price: float,
        ema_fast: float,
        ema_slow: float,
        threshold: float = 0.002
    ) -> Tuple[float, str]:
        """
        EMA crossover signal
        Returns (score, reason) where score is -1 to 1
        """
        if ema_fast is None or ema_slow is None:
            return 0.0, "Insufficient EMA data"
        
        # Bullish: price > ema_fast > ema_slow
        bullish = price > ema_fast > ema_slow
        # Bearish: price < ema_fast < ema_slow
        bearish = price < ema_fast < ema_slow
        
        # Calculate strength based on distance
        distance_pct = (price - ema_fast) / price if price > 0 else 0
        
        if bullish and distance_pct > threshold:
            return min(1.0, distance_pct / (threshold * 2)), "Bullish EMA crossover"
        elif bearish and distance_pct < -threshold:
            return max(-1.0, distance_pct / (threshold * 2)), "Bearish EMA crossover"
        
        return 0.0, "No clear EMA signal"
    
    @staticmethod
    def price_momentum(prices: List[float], lookback: int = 20) -> Tuple[float, str]:
        """
        Price momentum over lookback period
        Returns (score, reason) where score is -1 to 1
        """
        if len(prices) < lookback:
            return 0.0, "Insufficient price data"
        
        recent = np.array(prices[-lookback:])
        returns = (recent[-1] - recent[0]) / recent[0]
        
        # Normalize to -1 to 1 (assume 5% move is max)
        score = np.clip(returns / 0.05, -1, 1)
        
        if score > 0.2:
            return score, f"Strong upward momentum ({returns*100:.1f}%)"
        elif score < -0.2:
            return score, f"Strong downward momentum ({returns*100:.1f}%)"
        
        return score, f"Weak momentum ({returns*100:.1f}%)"
    
    @staticmethod
    def volatility_breakout(
        price: float,
        prices: List[float],
        atr: float,
        atr_multiplier: float = 1.5
    ) -> Tuple[float, str]:
        """
        Volatility breakout detection
        Returns (score, reason)
        """
        if len(prices) < 20 or atr is None:
            return 0.0, "Insufficient data for volatility"
        
        recent_high = max(prices[-20:])
        recent_low = min(prices[-20:])
        
        # Check for breakout
        upper_band = recent_high - atr * atr_multiplier
        lower_band = recent_low + atr * atr_multiplier
        
        if price > upper_band:
            return 0.7, "Upward volatility breakout"
        elif price < lower_band:
            return -0.7, "Downward volatility breakout"
        
        return 0.0, "No volatility breakout"


class SentimentIndicator:
    """Sentiment-based indicators"""
    
    @staticmethod
    def sentiment_score(
        score: float,
        bull_threshold: float = 0.35,
        bear_threshold: float = -0.35
    ) -> Tuple[float, str]:
        """
        Convert sentiment score to signal
        Returns (signal_score, reason)
        """
        if score > bull_threshold:
            return score, f"Bullish sentiment ({score:.2f})"
        elif score < bear_threshold:
            return score, f"Bearish sentiment ({score:.2f})"
        
        return 0.0, f"Neutral sentiment ({score:.2f})"


class OrderbookIndicator:
    """Orderbook-based indicators"""
    
    @staticmethod
    def imbalance_signal(
        imbalance: float,
        threshold: float = 0.05
    ) -> Tuple[float, str]:
        """
        Orderbook imbalance signal
        Returns (score, reason)
        """
        if imbalance > threshold:
            return imbalance, f"Buy pressure ({imbalance:.2f})"
        elif imbalance < -threshold:
            return imbalance, f"Sell pressure ({imbalance:.2f})"
        
        return 0.0, f"Balanced orderbook ({imbalance:.2f})"
    
    @staticmethod
    def spread_signal(spread: float, price: float) -> Tuple[float, str]:
        """
        Spread-based liquidity signal
        Returns (score, reason) - narrow spread = good liquidity
        """
        spread_pct = spread / price if price > 0 else 0
        
        if spread_pct < 0.001:  # Very tight spread
            return 1.0, "Excellent liquidity"
        elif spread_pct < 0.003:
            return 0.5, "Good liquidity"
        elif spread_pct > 0.01:
            return -0.5, "Poor liquidity - wide spread"
        
        return 0.0, "Normal liquidity"


class MultiModalStrategy(BaseStrategy):
    """
    Multi-modal trading strategy
    Combines momentum, technicals, sentiment, and orderbook signals
    
    Only trades when multiple modalities align
    """
    
    def __init__(
        self,
        min_agreeing_signals: int = 2,
        min_strength: float = 0.5
    ):
        super().__init__(name="MultiModalStrategy")
        self.min_agreeing_signals = min_agreeing_signals
        self.min_strength = min_strength
        self.state = StrategyState()
        
        # Weights for different signal types
        self.weights = {
            "momentum": 0.3,
            "sentiment": 0.25,
            "orderbook": 0.25,
            "technical": 0.2
        }
    
    def is_ready(self, market_state: MarketState) -> bool:
        """Check if we have enough data"""
        if len(market_state.prices) < self.config.strategy.warmup_period:
            return False
        if market_state.ema_fast is None or market_state.ema_slow is None:
            return False
        return True
    
    def evaluate(self, market_state: MarketState) -> Signal:
        """
        Evaluate market state using multiple modalities
        
        Signal components:
        1. Momentum: EMA crossover + price momentum
        2. Sentiment: Aggregated social/news sentiment
        3. Orderbook: Bid/ask imbalance
        4. Technical: ATR volatility breakout
        
        Trade only when >= min_agreeing_signals agree on direction
        """
        if not self.is_ready(market_state):
            return Signal(
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                price=market_state.price,
                timestamp=market_state.timestamp,
                reasons=["Warming up - insufficient data"]
            )
        
        indicators = {}
        reasons = []
        
        # 1. Momentum signals
        ema_score, ema_reason = MomentumIndicator.ema_crossover(
            market_state.price,
            market_state.ema_fast,
            market_state.ema_slow,
            self.config.strategy.momentum_threshold
        )
        
        momentum_score, momentum_reason = MomentumIndicator.price_momentum(
            market_state.prices,
            lookback=20
        )
        
        combined_momentum = (ema_score + momentum_score) / 2
        self.state.momentum_score = combined_momentum
        indicators["ema_score"] = ema_score
        indicators["momentum_score"] = momentum_score
        indicators["combined_momentum"] = combined_momentum
        
        if abs(combined_momentum) > 0.2:
            reasons.append(f"Momentum: {ema_reason}, {momentum_reason}")
        
        # 2. Sentiment signal
        sent_score, sent_reason = SentimentIndicator.sentiment_score(
            market_state.sentiment_score,
            self.config.strategy.sentiment_bull_threshold,
            self.config.strategy.sentiment_bear_threshold
        )
        self.state.sentiment_score = sent_score
        indicators["sentiment"] = market_state.sentiment_score
        
        if abs(sent_score) > 0.2:
            reasons.append(f"Sentiment: {sent_reason}")
        
        # 3. Orderbook signal
        ob_score = 0.0
        if market_state.orderbook:
            ob_score, ob_reason = OrderbookIndicator.imbalance_signal(
                market_state.orderbook.imbalance,
                self.config.strategy.ob_imbalance_threshold
            )
            indicators["orderbook_imbalance"] = market_state.orderbook.imbalance
            
            if abs(ob_score) > 0.1:
                reasons.append(f"Orderbook: {ob_reason}")
        
        self.state.orderbook_score = ob_score
        
        # 4. Technical/volatility signal
        tech_score = 0.0
        if market_state.atr:
            tech_score, tech_reason = MomentumIndicator.volatility_breakout(
                market_state.price,
                market_state.prices,
                market_state.atr
            )
            indicators["atr"] = market_state.atr
            
            if abs(tech_score) > 0.3:
                reasons.append(f"Technical: {tech_reason}")
        
        self.state.technical_score = tech_score
        
        # Calculate composite signal
        scores = {
            "momentum": combined_momentum,
            "sentiment": sent_score,
            "orderbook": ob_score,
            "technical": tech_score
        }
        
        # Count agreeing bullish/bearish signals
        bullish_count = sum(1 for s in scores.values() if s > 0.2)
        bearish_count = sum(1 for s in scores.values() if s < -0.2)
        
        # Weighted composite score
        composite = sum(
            scores[k] * self.weights[k]
            for k in scores.keys()
        )
        self.state.composite_score = composite
        indicators["composite"] = composite
        indicators["bullish_signals"] = bullish_count
        indicators["bearish_signals"] = bearish_count
        
        # Determine signal direction
        direction = SignalDirection.NEUTRAL
        strength = abs(composite)
        
        if bullish_count >= self.min_agreeing_signals and composite > 0:
            direction = SignalDirection.LONG
            reasons.insert(0, f"LONG signal: {bullish_count} bullish indicators")
        elif bearish_count >= self.min_agreeing_signals and composite < 0:
            direction = SignalDirection.SHORT
            reasons.insert(0, f"SHORT signal: {bearish_count} bearish indicators")
        else:
            reasons.insert(0, f"No consensus: {bullish_count} bull, {bearish_count} bear signals")
        
        # Build signal
        signal = Signal(
            direction=direction,
            strength=strength,
            price=market_state.price,
            timestamp=market_state.timestamp,
            reasons=reasons,
            indicators=indicators
        )
        
        self.state.last_signal = signal
        
        # Log signal
        self.logger.log_signal(
            signal_type="composite",
            direction=direction.value,
            strength=strength,
            price=market_state.price,
            indicators=indicators
        )
        
        return signal
    
    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state"""
        return {
            "momentum_score": self.state.momentum_score,
            "sentiment_score": self.state.sentiment_score,
            "orderbook_score": self.state.orderbook_score,
            "technical_score": self.state.technical_score,
            "composite_score": self.state.composite_score,
            "last_signal": self.state.last_signal.direction.value if self.state.last_signal else None
        }


class ScalpingStrategy(BaseStrategy):
    """
    High-frequency scalping strategy
    Uses orderbook imbalance and micro-momentum
    """
    
    def __init__(self):
        super().__init__(name="ScalpingStrategy")
        self.min_imbalance = 0.15  # Higher threshold for scalping
        self.lookback = 5  # Short lookback for micro-momentum
    
    def is_ready(self, market_state: MarketState) -> bool:
        return len(market_state.prices) >= self.lookback and market_state.orderbook is not None
    
    def evaluate(self, market_state: MarketState) -> Signal:
        if not self.is_ready(market_state):
            return Signal(
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                price=market_state.price,
                timestamp=market_state.timestamp,
                reasons=["Insufficient data for scalping"]
            )
        
        indicators = {}
        reasons = []
        
        # Orderbook imbalance
        ob = market_state.orderbook
        imbalance = ob.imbalance
        indicators["imbalance"] = imbalance
        
        # Micro momentum
        recent_prices = market_state.prices[-self.lookback:]
        micro_return = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        indicators["micro_return"] = micro_return
        
        # Signal logic
        direction = SignalDirection.NEUTRAL
        strength = 0.0
        
        if imbalance > self.min_imbalance and micro_return > 0:
            direction = SignalDirection.LONG
            strength = min(1.0, imbalance)
            reasons.append(f"Strong buy pressure with upward micro-momentum")
        elif imbalance < -self.min_imbalance and micro_return < 0:
            direction = SignalDirection.SHORT
            strength = min(1.0, abs(imbalance))
            reasons.append(f"Strong sell pressure with downward micro-momentum")
        else:
            reasons.append("No scalping opportunity")
        
        return Signal(
            direction=direction,
            strength=strength,
            price=market_state.price,
            timestamp=market_state.timestamp,
            reasons=reasons,
            indicators=indicators
        )


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy
    Trades when price deviates significantly from moving average
    """
    
    def __init__(self, deviation_threshold: float = 2.0):
        super().__init__(name="MeanReversionStrategy")
        self.deviation_threshold = deviation_threshold  # ATR multiplier
    
    def is_ready(self, market_state: MarketState) -> bool:
        return (
            market_state.ema_slow is not None and
            market_state.atr is not None and
            len(market_state.prices) >= 50
        )
    
    def evaluate(self, market_state: MarketState) -> Signal:
        if not self.is_ready(market_state):
            return Signal(
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                price=market_state.price,
                timestamp=market_state.timestamp,
                reasons=["Insufficient data for mean reversion"]
            )
        
        price = market_state.price
        mean = market_state.ema_slow
        atr = market_state.atr
        
        # Calculate z-score using ATR
        deviation = (price - mean) / atr
        
        indicators = {
            "price": price,
            "mean": mean,
            "atr": atr,
            "deviation": deviation
        }
        
        direction = SignalDirection.NEUTRAL
        strength = 0.0
        reasons = []
        
        if deviation > self.deviation_threshold:
            # Price far above mean - expect reversion down
            direction = SignalDirection.SHORT
            strength = min(1.0, deviation / (self.deviation_threshold * 2))
            reasons.append(f"Price {deviation:.1f} ATR above mean - expect reversion")
        elif deviation < -self.deviation_threshold:
            # Price far below mean - expect reversion up
            direction = SignalDirection.LONG
            strength = min(1.0, abs(deviation) / (self.deviation_threshold * 2))
            reasons.append(f"Price {abs(deviation):.1f} ATR below mean - expect reversion")
        else:
            reasons.append(f"Price within normal range ({deviation:.1f} ATR from mean)")
        
        return Signal(
            direction=direction,
            strength=strength,
            price=price,
            timestamp=market_state.timestamp,
            reasons=reasons,
            indicators=indicators
        )


class StrategyManager:
    """
    Strategy manager for running multiple strategies
    Can combine signals or switch between strategies
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategy: Optional[str] = None
        
        # Register default strategies
        self.register_strategy("multimodal", MultiModalStrategy())
        self.register_strategy("scalping", ScalpingStrategy())
        self.register_strategy("meanreversion", MeanReversionStrategy())
        
        # Set default
        self.set_active_strategy("multimodal")
    
    def register_strategy(self, name: str, strategy: BaseStrategy):
        """Register a strategy"""
        self.strategies[name] = strategy
        self.logger.info("Strategy registered", name=name)
    
    def set_active_strategy(self, name: str):
        """Set the active strategy"""
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy: {name}")
        self.active_strategy = name
        self.logger.info("Active strategy set", name=name)
    
    def evaluate(self, market_state: MarketState) -> Signal:
        """Evaluate using active strategy"""
        if not self.active_strategy:
            raise RuntimeError("No active strategy set")
        
        strategy = self.strategies[self.active_strategy]
        return strategy.evaluate(market_state)
    
    def evaluate_all(self, market_state: MarketState) -> Dict[str, Signal]:
        """Evaluate all strategies and return signals"""
        return {
            name: strategy.evaluate(market_state)
            for name, strategy in self.strategies.items()
        }
    
    def get_consensus_signal(self, market_state: MarketState) -> Signal:
        """
        Get consensus signal from all strategies
        Requires majority agreement
        """
        signals = self.evaluate_all(market_state)
        
        long_votes = sum(1 for s in signals.values() if s.direction == SignalDirection.LONG)
        short_votes = sum(1 for s in signals.values() if s.direction == SignalDirection.SHORT)
        total = len(signals)
        
        # Majority vote
        if long_votes > total / 2:
            avg_strength = np.mean([s.strength for s in signals.values() if s.direction == SignalDirection.LONG])
            return Signal(
                direction=SignalDirection.LONG,
                strength=avg_strength,
                price=market_state.price,
                timestamp=market_state.timestamp,
                reasons=[f"Consensus LONG: {long_votes}/{total} strategies agree"],
                indicators={"votes": {"long": long_votes, "short": short_votes}}
            )
        elif short_votes > total / 2:
            avg_strength = np.mean([s.strength for s in signals.values() if s.direction == SignalDirection.SHORT])
            return Signal(
                direction=SignalDirection.SHORT,
                strength=avg_strength,
                price=market_state.price,
                timestamp=market_state.timestamp,
                reasons=[f"Consensus SHORT: {short_votes}/{total} strategies agree"],
                indicators={"votes": {"long": long_votes, "short": short_votes}}
            )
        
        return Signal(
            direction=SignalDirection.NEUTRAL,
            strength=0.0,
            price=market_state.price,
            timestamp=market_state.timestamp,
            reasons=[f"No consensus: {long_votes} long, {short_votes} short"],
            indicators={"votes": {"long": long_votes, "short": short_votes}}
        )


# Singleton instance
_strategy_manager: Optional[StrategyManager] = None


def get_strategy_manager() -> StrategyManager:
    """Get or create the global strategy manager"""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = StrategyManager()
    return _strategy_manager


if __name__ == "__main__":
    # Test strategy
    from data_ingest import get_data_ingestor
    
    ingestor = get_data_ingestor()
    manager = get_strategy_manager()
    
    print("Fetching market state...")
    state = ingestor.get_current_state()
    
    print(f"\nMarket State:")
    print(f"  Price: {state.price}")
    print(f"  EMA Fast: {state.ema_fast}")
    print(f"  EMA Slow: {state.ema_slow}")
    print(f"  Sentiment: {state.sentiment_score}")
    
    if state.orderbook:
        print(f"  OB Imbalance: {state.orderbook.imbalance}")
    
    print("\nEvaluating signals...")
    signal = manager.evaluate(state)
    
    print(f"\nSignal:")
    print(f"  Direction: {signal.direction.value}")
    print(f"  Strength: {signal.strength:.3f}")
    print(f"  Actionable: {signal.is_actionable}")
    print(f"  Reasons: {signal.reasons}")
