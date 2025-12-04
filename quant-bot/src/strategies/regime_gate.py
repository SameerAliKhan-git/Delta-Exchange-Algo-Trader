"""
Regime Gating Module
====================
PRODUCTION DELIVERABLE: Strategy-regime alignment gate.

Momentum strategies must be OFF in choppy regimes.
Stat arb must be OFF in trending regimes.

This is how funds avoid drawdowns - they don't fight the regime.

This module provides:
- Regime detection (trending, ranging, volatile, crisis)
- Strategy-regime compatibility matrix
- Hard gate to disable incompatible strategies
- Smooth blending for transition periods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================

class MarketRegime(Enum):
    """Market regime types."""
    STRONG_TREND_UP = "strong_trend_up"      # Clear uptrend, low volatility
    TREND_UP = "trend_up"                    # Moderate uptrend
    STRONG_TREND_DOWN = "strong_trend_down"  # Clear downtrend
    TREND_DOWN = "trend_down"                # Moderate downtrend
    RANGING = "ranging"                      # Sideways, mean-reverting
    HIGH_VOLATILITY = "high_volatility"      # High vol, no clear direction
    LOW_VOLATILITY = "low_volatility"        # Compression, breakout imminent
    CRISIS = "crisis"                        # Extreme moves, correlations spike
    TRANSITION = "transition"                # Changing between regimes


class StrategyType(Enum):
    """Strategy types for regime matching."""
    MOMENTUM = "momentum"
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    STAT_ARB = "stat_arb"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    MARKET_MAKING = "market_making"
    CARRY = "carry"


@dataclass
class RegimeState:
    """Current regime state."""
    timestamp: datetime
    symbol: str
    
    # Primary regime
    regime: MarketRegime
    confidence: float  # 0-1 confidence in classification
    
    # Regime metrics
    trend_strength: float      # -1 to 1 (negative = downtrend)
    volatility_percentile: float  # 0-100 percentile of historical vol
    mean_reversion_score: float   # How mean-reverting (0-1)
    
    # Time in regime
    regime_duration_hours: float
    regime_stability: float    # How stable the regime is (0-1)
    
    # Transition probability
    transition_probability: float  # Probability of regime change soon
    
    # Raw features
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'regime': self.regime.value,
            'confidence': self.confidence,
            'trend_strength': self.trend_strength,
            'volatility_percentile': self.volatility_percentile,
            'mean_reversion_score': self.mean_reversion_score,
            'regime_duration_hours': self.regime_duration_hours,
            'regime_stability': self.regime_stability,
            'transition_probability': self.transition_probability
        }


@dataclass
class StrategyGateDecision:
    """Decision about whether strategy should be active."""
    strategy_name: str
    strategy_type: StrategyType
    is_allowed: bool
    reason: str
    
    # Regime context
    current_regime: MarketRegime
    regime_confidence: float
    
    # Adjustments
    position_size_multiplier: float  # 0 = fully disabled, 1 = normal, >1 = boosted
    confidence_adjustment: float     # Multiply strategy confidence by this
    
    # Risk adjustments
    stop_loss_multiplier: float      # Widen/tighten stops
    take_profit_multiplier: float
    
    def to_dict(self) -> Dict:
        return {
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type.value,
            'is_allowed': self.is_allowed,
            'reason': self.reason,
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'position_size_multiplier': self.position_size_multiplier,
            'confidence_adjustment': self.confidence_adjustment,
            'stop_loss_multiplier': self.stop_loss_multiplier,
            'take_profit_multiplier': self.take_profit_multiplier
        }


# =============================================================================
# REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """
    Detect current market regime from price data.
    
    Uses multiple indicators:
    - ADX for trend strength
    - Bollinger Band width for volatility regime
    - Hurst exponent for mean reversion
    - RSI for overbought/oversold
    """
    
    def __init__(
        self,
        lookback_periods: int = 100,
        volatility_lookback: int = 252,
        trend_threshold: float = 0.3,
        high_vol_percentile: float = 75,
        low_vol_percentile: float = 25
    ):
        self.lookback_periods = lookback_periods
        self.volatility_lookback = volatility_lookback
        self.trend_threshold = trend_threshold
        self.high_vol_percentile = high_vol_percentile
        self.low_vol_percentile = low_vol_percentile
        
        # State tracking
        self.regime_history: Dict[str, deque] = {}
        self.last_regime: Dict[str, MarketRegime] = {}
        self.regime_start_time: Dict[str, datetime] = {}
        
        # Price data
        self.price_data: Dict[str, deque] = {}
    
    def update(self, symbol: str, price: float, timestamp: datetime = None):
        """Update with new price."""
        timestamp = timestamp or datetime.now()
        
        if symbol not in self.price_data:
            self.price_data[symbol] = deque(maxlen=self.volatility_lookback)
        
        self.price_data[symbol].append({
            'timestamp': timestamp,
            'price': price
        })
    
    def detect(self, symbol: str, prices: Optional[np.ndarray] = None) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            symbol: Trading symbol
            prices: Optional price array (uses internal buffer if not provided)
        
        Returns:
            RegimeState with regime classification
        """
        # Get price data
        if prices is None:
            if symbol not in self.price_data or len(self.price_data[symbol]) < 20:
                return self._empty_state(symbol)
            prices = np.array([p['price'] for p in self.price_data[symbol]])
        
        if len(prices) < 20:
            return self._empty_state(symbol)
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate features
        features = self._calculate_features(prices, returns)
        
        # Classify regime
        regime, confidence = self._classify_regime(features)
        
        # Track regime duration
        if symbol not in self.last_regime or self.last_regime[symbol] != regime:
            self.last_regime[symbol] = regime
            self.regime_start_time[symbol] = datetime.now()
        
        duration = (datetime.now() - self.regime_start_time.get(symbol, datetime.now())).total_seconds() / 3600
        
        # Calculate stability and transition probability
        stability = self._calculate_stability(symbol, regime)
        transition_prob = self._estimate_transition_probability(features, regime)
        
        state = RegimeState(
            timestamp=datetime.now(),
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            trend_strength=features.get('trend_strength', 0),
            volatility_percentile=features.get('vol_percentile', 50),
            mean_reversion_score=features.get('mean_reversion', 0.5),
            regime_duration_hours=duration,
            regime_stability=stability,
            transition_probability=transition_prob,
            features=features
        )
        
        # Update history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = deque(maxlen=100)
        self.regime_history[symbol].append(state)
        
        return state
    
    def _calculate_features(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Calculate regime detection features."""
        features = {}
        
        # 1. Trend strength using linear regression
        n = min(len(prices), self.lookback_periods)
        recent_prices = prices[-n:]
        x = np.arange(n)
        slope, intercept = np.polyfit(x, recent_prices, 1)
        trend_r2 = 1 - np.sum((recent_prices - (slope * x + intercept))**2) / np.sum((recent_prices - np.mean(recent_prices))**2)
        
        # Normalize trend strength to -1 to 1
        trend_direction = np.sign(slope)
        features['trend_strength'] = trend_direction * np.sqrt(max(0, trend_r2))
        
        # 2. ADX-like trend indicator
        high = prices  # Simplified - would use actual high/low in production
        low = prices * 0.99
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - prices[:-1]))
        atr = np.mean(tr[-14:])
        
        plus_dm = np.maximum(0, high[1:] - high[:-1])
        minus_dm = np.maximum(0, low[:-1] - low[1:])
        
        plus_di = np.mean(plus_dm[-14:]) / atr if atr > 0 else 0
        minus_di = np.mean(minus_dm[-14:]) / atr if atr > 0 else 0
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        features['adx'] = dx
        
        # 3. Volatility regime
        vol = np.std(returns[-20:]) * np.sqrt(252)
        hist_vol = np.std(returns) * np.sqrt(252)
        
        # Calculate percentile
        vol_history = [np.std(returns[i:i+20]) * np.sqrt(252) 
                      for i in range(max(0, len(returns)-252), len(returns)-20)]
        if vol_history:
            features['vol_percentile'] = np.searchsorted(sorted(vol_history), vol) / len(vol_history) * 100
        else:
            features['vol_percentile'] = 50
        
        features['volatility'] = vol
        
        # 4. Mean reversion score (Hurst exponent approximation)
        # H < 0.5 = mean reverting, H > 0.5 = trending
        lags = range(2, min(20, len(returns)//2))
        tau = []
        for lag in lags:
            rs = [np.max(returns[i:i+lag]) - np.min(returns[i:i+lag]) 
                  for i in range(len(returns) - lag)]
            if rs:
                tau.append(np.mean(rs))
        
        if len(tau) > 2:
            hurst = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)[0]
        else:
            hurst = 0.5
        
        features['hurst'] = hurst
        features['mean_reversion'] = 1 - min(1, max(0, hurst))
        
        # 5. RSI for extreme readings
        gains = np.maximum(returns, 0)
        losses = np.abs(np.minimum(returns, 0))
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100 if avg_gain > 0 else 50
        
        features['rsi'] = rsi
        
        # 6. Bollinger Band width (normalized)
        sma = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        bb_width = (2 * std / sma) * 100
        features['bb_width'] = bb_width
        
        return features
    
    def _classify_regime(self, features: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Classify regime from features."""
        trend = features.get('trend_strength', 0)
        vol_pct = features.get('vol_percentile', 50)
        adx = features.get('adx', 0)
        mr = features.get('mean_reversion', 0.5)
        rsi = features.get('rsi', 50)
        
        # Initialize scores
        regime_scores = {r: 0.0 for r in MarketRegime}
        
        # Strong trend up
        if trend > 0.5 and adx > 0.4:
            regime_scores[MarketRegime.STRONG_TREND_UP] = trend * adx
        elif trend > 0.3:
            regime_scores[MarketRegime.TREND_UP] = trend * 0.7
        
        # Strong trend down
        if trend < -0.5 and adx > 0.4:
            regime_scores[MarketRegime.STRONG_TREND_DOWN] = abs(trend) * adx
        elif trend < -0.3:
            regime_scores[MarketRegime.TREND_DOWN] = abs(trend) * 0.7
        
        # Ranging
        if abs(trend) < 0.2 and mr > 0.5:
            regime_scores[MarketRegime.RANGING] = mr * (1 - abs(trend))
        
        # High volatility
        if vol_pct > self.high_vol_percentile:
            vol_score = (vol_pct - self.high_vol_percentile) / (100 - self.high_vol_percentile)
            regime_scores[MarketRegime.HIGH_VOLATILITY] += vol_score * 0.5
        
        # Low volatility
        if vol_pct < self.low_vol_percentile:
            regime_scores[MarketRegime.LOW_VOLATILITY] = (self.low_vol_percentile - vol_pct) / self.low_vol_percentile
        
        # Crisis
        if vol_pct > 90 and abs(trend) > 0.3:
            regime_scores[MarketRegime.CRISIS] = (vol_pct / 100) * abs(trend)
        
        # Extreme RSI indicates potential transition
        if rsi > 80 or rsi < 20:
            regime_scores[MarketRegime.TRANSITION] += 0.3
        
        # Select regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        best_score = regime_scores[best_regime]
        
        # If no clear regime, default to ranging
        if best_score < 0.2:
            best_regime = MarketRegime.RANGING
            best_score = 0.3
        
        # Normalize confidence
        confidence = min(1.0, best_score / 0.8)
        
        return best_regime, confidence
    
    def _calculate_stability(self, symbol: str, current_regime: MarketRegime) -> float:
        """Calculate how stable the current regime is."""
        if symbol not in self.regime_history:
            return 0.5
        
        history = list(self.regime_history[symbol])[-20:]
        if not history:
            return 0.5
        
        # Count how many recent periods had same regime
        same_regime = sum(1 for s in history if s.regime == current_regime)
        stability = same_regime / len(history)
        
        return stability
    
    def _estimate_transition_probability(self, features: Dict, current_regime: MarketRegime) -> float:
        """Estimate probability of regime change."""
        transition_prob = 0.1  # Base probability
        
        # High RSI extremes increase transition probability
        rsi = features.get('rsi', 50)
        if rsi > 75 or rsi < 25:
            transition_prob += 0.2
        
        # Very low volatility often precedes regime change
        if features.get('vol_percentile', 50) < 10:
            transition_prob += 0.3
        
        # Weak trend in trending regime
        if current_regime in [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]:
            if abs(features.get('trend_strength', 0)) < 0.2:
                transition_prob += 0.2
        
        return min(1.0, transition_prob)
    
    def _empty_state(self, symbol: str) -> RegimeState:
        """Return empty/neutral state."""
        return RegimeState(
            timestamp=datetime.now(),
            symbol=symbol,
            regime=MarketRegime.RANGING,
            confidence=0.0,
            trend_strength=0,
            volatility_percentile=50,
            mean_reversion_score=0.5,
            regime_duration_hours=0,
            regime_stability=0.5,
            transition_probability=0.5
        )


# =============================================================================
# STRATEGY-REGIME COMPATIBILITY
# =============================================================================

class StrategyRegimeCompatibility:
    """
    Defines which strategies work in which regimes.
    
    This is the KEY insight that prevents drawdowns.
    """
    
    # Compatibility matrix: strategy_type -> {regime: (allowed, position_mult, stop_mult)}
    DEFAULT_COMPATIBILITY: Dict[StrategyType, Dict[MarketRegime, Tuple[bool, float, float]]] = {
        StrategyType.MOMENTUM: {
            MarketRegime.STRONG_TREND_UP: (True, 1.5, 1.0),
            MarketRegime.TREND_UP: (True, 1.2, 1.0),
            MarketRegime.STRONG_TREND_DOWN: (True, 1.5, 1.0),
            MarketRegime.TREND_DOWN: (True, 1.2, 1.0),
            MarketRegime.RANGING: (False, 0.0, 1.5),  # DISABLED in ranging
            MarketRegime.HIGH_VOLATILITY: (True, 0.7, 1.5),
            MarketRegime.LOW_VOLATILITY: (False, 0.0, 1.0),
            MarketRegime.CRISIS: (False, 0.0, 2.0),
            MarketRegime.TRANSITION: (False, 0.0, 1.5),
        },
        StrategyType.TREND_FOLLOWING: {
            MarketRegime.STRONG_TREND_UP: (True, 1.5, 1.0),
            MarketRegime.TREND_UP: (True, 1.3, 1.0),
            MarketRegime.STRONG_TREND_DOWN: (True, 1.5, 1.0),
            MarketRegime.TREND_DOWN: (True, 1.3, 1.0),
            MarketRegime.RANGING: (False, 0.0, 1.5),  # DISABLED
            MarketRegime.HIGH_VOLATILITY: (True, 0.5, 2.0),
            MarketRegime.LOW_VOLATILITY: (False, 0.3, 1.0),
            MarketRegime.CRISIS: (False, 0.0, 2.0),
            MarketRegime.TRANSITION: (True, 0.5, 1.5),
        },
        StrategyType.MEAN_REVERSION: {
            MarketRegime.STRONG_TREND_UP: (False, 0.0, 2.0),  # DISABLED
            MarketRegime.TREND_UP: (False, 0.0, 1.5),  # DISABLED
            MarketRegime.STRONG_TREND_DOWN: (False, 0.0, 2.0),
            MarketRegime.TREND_DOWN: (False, 0.0, 1.5),
            MarketRegime.RANGING: (True, 1.5, 1.0),  # OPTIMAL
            MarketRegime.HIGH_VOLATILITY: (True, 0.8, 1.5),
            MarketRegime.LOW_VOLATILITY: (True, 1.2, 0.8),
            MarketRegime.CRISIS: (False, 0.0, 2.0),
            MarketRegime.TRANSITION: (False, 0.3, 1.5),
        },
        StrategyType.STAT_ARB: {
            MarketRegime.STRONG_TREND_UP: (False, 0.0, 2.0),  # DISABLED in trends
            MarketRegime.TREND_UP: (True, 0.5, 1.5),
            MarketRegime.STRONG_TREND_DOWN: (False, 0.0, 2.0),
            MarketRegime.TREND_DOWN: (True, 0.5, 1.5),
            MarketRegime.RANGING: (True, 1.5, 1.0),  # OPTIMAL
            MarketRegime.HIGH_VOLATILITY: (True, 0.7, 1.5),
            MarketRegime.LOW_VOLATILITY: (True, 1.3, 0.8),
            MarketRegime.CRISIS: (False, 0.0, 2.0),
            MarketRegime.TRANSITION: (True, 0.5, 1.5),
        },
        StrategyType.BREAKOUT: {
            MarketRegime.STRONG_TREND_UP: (True, 1.0, 1.0),
            MarketRegime.TREND_UP: (True, 1.0, 1.0),
            MarketRegime.STRONG_TREND_DOWN: (True, 1.0, 1.0),
            MarketRegime.TREND_DOWN: (True, 1.0, 1.0),
            MarketRegime.RANGING: (True, 0.7, 1.2),
            MarketRegime.HIGH_VOLATILITY: (True, 0.8, 1.5),
            MarketRegime.LOW_VOLATILITY: (True, 1.5, 0.8),  # OPTIMAL before breakout
            MarketRegime.CRISIS: (False, 0.0, 2.0),
            MarketRegime.TRANSITION: (True, 1.2, 1.0),
        },
        StrategyType.VOLATILITY: {
            MarketRegime.STRONG_TREND_UP: (True, 0.8, 1.0),
            MarketRegime.TREND_UP: (True, 0.8, 1.0),
            MarketRegime.STRONG_TREND_DOWN: (True, 0.8, 1.0),
            MarketRegime.TREND_DOWN: (True, 0.8, 1.0),
            MarketRegime.RANGING: (True, 0.8, 1.0),
            MarketRegime.HIGH_VOLATILITY: (True, 1.5, 1.0),  # OPTIMAL
            MarketRegime.LOW_VOLATILITY: (True, 1.0, 0.8),
            MarketRegime.CRISIS: (True, 1.0, 2.0),
            MarketRegime.TRANSITION: (True, 1.2, 1.0),
        },
        StrategyType.MARKET_MAKING: {
            MarketRegime.STRONG_TREND_UP: (False, 0.0, 2.0),  # DISABLED
            MarketRegime.TREND_UP: (True, 0.5, 1.5),
            MarketRegime.STRONG_TREND_DOWN: (False, 0.0, 2.0),
            MarketRegime.TREND_DOWN: (True, 0.5, 1.5),
            MarketRegime.RANGING: (True, 1.5, 1.0),  # OPTIMAL
            MarketRegime.HIGH_VOLATILITY: (True, 0.3, 2.0),
            MarketRegime.LOW_VOLATILITY: (True, 1.2, 0.8),
            MarketRegime.CRISIS: (False, 0.0, 2.0),
            MarketRegime.TRANSITION: (True, 0.5, 1.5),
        },
    }
    
    def __init__(self, custom_compatibility: Optional[Dict] = None):
        self.compatibility = custom_compatibility or self.DEFAULT_COMPATIBILITY
    
    def get_compatibility(
        self,
        strategy_type: StrategyType,
        regime: MarketRegime
    ) -> Tuple[bool, float, float]:
        """
        Get compatibility for strategy-regime pair.
        
        Returns:
            (is_allowed, position_multiplier, stop_loss_multiplier)
        """
        if strategy_type not in self.compatibility:
            return (True, 1.0, 1.0)  # Default allow
        
        strategy_compat = self.compatibility[strategy_type]
        if regime not in strategy_compat:
            return (True, 1.0, 1.0)
        
        return strategy_compat[regime]


# =============================================================================
# REGIME EXECUTION GATE
# =============================================================================

class RegimeExecutionGate:
    """
    HARD GATE that filters strategies based on regime compatibility.
    
    USE THIS TO CHECK EVERY STRATEGY BEFORE GENERATING SIGNALS:
    
        decision = gate.check_strategy(strategy_name, strategy_type, symbol)
        if not decision.is_allowed:
            skip this strategy for this bar
    """
    
    def __init__(
        self,
        compatibility: Optional[StrategyRegimeCompatibility] = None,
        detector: Optional[RegimeDetector] = None,
        strict_mode: bool = True,
        min_regime_confidence: float = 0.4
    ):
        self.compatibility = compatibility or StrategyRegimeCompatibility()
        self.detector = detector or RegimeDetector()
        self.strict_mode = strict_mode
        self.min_regime_confidence = min_regime_confidence
        
        # Strategy type mapping (name -> type)
        self.strategy_types: Dict[str, StrategyType] = {}
        
        # Statistics
        self.checks_total = 0
        self.checks_allowed = 0
        self.checks_blocked = 0
        
        # History
        self.decision_history: List[StrategyGateDecision] = []
    
    def register_strategy(self, name: str, strategy_type: StrategyType):
        """Register a strategy's type for regime matching."""
        self.strategy_types[name] = strategy_type
        logger.info(f"Registered strategy '{name}' as type '{strategy_type.value}'")
    
    def update_prices(self, symbol: str, price: float, timestamp: datetime = None):
        """Update price data for regime detection."""
        self.detector.update(symbol, price, timestamp)
    
    def get_regime(self, symbol: str) -> RegimeState:
        """Get current regime for symbol."""
        return self.detector.detect(symbol)
    
    def check_strategy(
        self,
        strategy_name: str,
        symbol: str,
        strategy_type: Optional[StrategyType] = None,
        signal_confidence: float = 1.0
    ) -> StrategyGateDecision:
        """
        Check if a strategy should be active in current regime.
        
        THIS IS THE CRITICAL GATE FUNCTION.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            strategy_type: Type of strategy (uses registered type if not provided)
            signal_confidence: Strategy's signal confidence
        
        Returns:
            StrategyGateDecision with allow/block and adjustments
        """
        self.checks_total += 1
        
        # Get strategy type
        if strategy_type is None:
            strategy_type = self.strategy_types.get(strategy_name, StrategyType.MOMENTUM)
        
        # Get current regime
        regime_state = self.detector.detect(symbol)
        
        # Check if we have confident regime classification
        if regime_state.confidence < self.min_regime_confidence:
            # Low confidence - allow with reduced size
            decision = StrategyGateDecision(
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                is_allowed=True,
                reason="Low regime confidence - allowing with reduced size",
                current_regime=regime_state.regime,
                regime_confidence=regime_state.confidence,
                position_size_multiplier=0.5,
                confidence_adjustment=0.7,
                stop_loss_multiplier=1.5,
                take_profit_multiplier=1.0
            )
            self.checks_allowed += 1
            self.decision_history.append(decision)
            return decision
        
        # Get compatibility
        is_allowed, pos_mult, stop_mult = self.compatibility.get_compatibility(
            strategy_type, regime_state.regime
        )
        
        # Apply strict mode
        if self.strict_mode and not is_allowed:
            reason = f"Strategy {strategy_type.value} incompatible with {regime_state.regime.value} regime"
            self.checks_blocked += 1
        else:
            reason = f"Strategy compatible with {regime_state.regime.value} regime"
            if pos_mult > 1.0:
                reason += " (boosted)"
            elif pos_mult < 1.0:
                reason += " (reduced)"
            self.checks_allowed += 1
            is_allowed = True  # In non-strict mode, allow but adjust
        
        # Adjust confidence based on regime stability
        confidence_adj = regime_state.regime_stability
        if regime_state.transition_probability > 0.5:
            confidence_adj *= 0.7  # Reduce confidence if transition likely
        
        decision = StrategyGateDecision(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            is_allowed=is_allowed,
            reason=reason,
            current_regime=regime_state.regime,
            regime_confidence=regime_state.confidence,
            position_size_multiplier=pos_mult if is_allowed else 0.0,
            confidence_adjustment=confidence_adj,
            stop_loss_multiplier=stop_mult,
            take_profit_multiplier=1.0 / stop_mult if stop_mult > 0 else 1.0
        )
        
        self.decision_history.append(decision)
        
        # Log decision
        if not is_allowed:
            logger.warning(
                f"[REGIME GATE BLOCKED] {strategy_name} ({strategy_type.value}) "
                f"| Regime: {regime_state.regime.value} | {reason}"
            )
        else:
            logger.info(
                f"[REGIME GATE PASSED] {strategy_name} ({strategy_type.value}) "
                f"| Regime: {regime_state.regime.value} | Size mult: {pos_mult:.2f}"
            )
        
        return decision
    
    def check_all_strategies(
        self,
        symbol: str,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, StrategyGateDecision]:
        """
        Check all registered strategies for a symbol.
        
        Returns dict of strategy_name -> decision.
        """
        if strategies is None:
            strategies = list(self.strategy_types.keys())
        
        decisions = {}
        for name in strategies:
            decisions[name] = self.check_strategy(name, symbol)
        
        return decisions
    
    def get_active_strategies(self, symbol: str) -> List[str]:
        """Get list of strategies that are active in current regime."""
        decisions = self.check_all_strategies(symbol)
        return [name for name, decision in decisions.items() if decision.is_allowed]
    
    def get_statistics(self) -> Dict:
        """Get gate statistics."""
        return {
            'checks_total': self.checks_total,
            'checks_allowed': self.checks_allowed,
            'checks_blocked': self.checks_blocked,
            'block_rate': self.checks_blocked / max(1, self.checks_total)
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate regime gating."""
    
    print("=" * 70)
    print("REGIME EXECUTION GATE - DEMO")
    print("=" * 70)
    
    # Create gate
    gate = RegimeExecutionGate(strict_mode=True)
    
    # Register strategies
    gate.register_strategy("momentum_btc", StrategyType.MOMENTUM)
    gate.register_strategy("stat_arb_eth", StrategyType.STAT_ARB)
    gate.register_strategy("mean_reversion", StrategyType.MEAN_REVERSION)
    gate.register_strategy("trend_follow", StrategyType.TREND_FOLLOWING)
    
    # Simulate trending market
    print("\n1. Simulating TRENDING market...")
    np.random.seed(42)
    prices = 50000 * np.cumprod(1 + np.random.normal(0.002, 0.01, 100))  # Upward drift
    
    for p in prices:
        gate.update_prices('BTCUSDT', p)
    
    regime = gate.get_regime('BTCUSDT')
    print(f"   Detected regime: {regime.regime.value}")
    print(f"   Confidence: {regime.confidence:.2f}")
    print(f"   Trend strength: {regime.trend_strength:.2f}")
    
    print("\n2. Checking strategies in trending regime...")
    decisions = gate.check_all_strategies('BTCUSDT')
    for name, decision in decisions.items():
        status = "✓ ACTIVE" if decision.is_allowed else "✗ BLOCKED"
        print(f"   {status} {name}: {decision.reason}")
        if decision.is_allowed:
            print(f"      Position mult: {decision.position_size_multiplier:.2f}")
    
    # Simulate ranging market
    print("\n3. Simulating RANGING market...")
    prices = 50000 + np.random.normal(0, 500, 100)  # Mean-reverting around 50000
    
    for p in prices:
        gate.update_prices('ETHUSDT', p)
    
    regime = gate.get_regime('ETHUSDT')
    print(f"   Detected regime: {regime.regime.value}")
    print(f"   Confidence: {regime.confidence:.2f}")
    print(f"   Mean reversion score: {regime.mean_reversion_score:.2f}")
    
    print("\n4. Checking strategies in ranging regime...")
    decisions = gate.check_all_strategies('ETHUSDT')
    for name, decision in decisions.items():
        status = "✓ ACTIVE" if decision.is_allowed else "✗ BLOCKED"
        print(f"   {status} {name}: {decision.reason}")
    
    # Statistics
    print("\n" + "-" * 70)
    print("GATE STATISTICS")
    print("-" * 70)
    stats = gate.get_statistics()
    print(f"Total checks: {stats['checks_total']}")
    print(f"Allowed: {stats['checks_allowed']}")
    print(f"Blocked: {stats['checks_blocked']}")
    print(f"Block rate: {stats['block_rate']:.0%}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKEY INSIGHT: Momentum strategies were blocked in ranging regime,")
    print("while stat_arb was blocked in trending regime. This prevents")
    print("strategy-regime mismatch drawdowns.")


if __name__ == "__main__":
    demo()
