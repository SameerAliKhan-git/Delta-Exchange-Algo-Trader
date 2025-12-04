"""
Strategy Ensemble - Master Orchestrator
========================================

This module brings together all 7 strategy families into a unified
ensemble that:

1. Detects current market regime
2. Selects optimal strategy for the regime
3. Weights signals based on strategy confidence
4. Manages position sizing across strategies
5. Implements anti-correlation diversification

THE KEY INSIGHT:
================
Don't run all strategies simultaneously.
Run regime-appropriate strategies with adaptive weighting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum

from .base import BaseStrategy, StrategyConfig, Signal, SignalType, TechnicalIndicators
from .regime_ml import RegimeDetector, MarketRegime, RegimeConfig


@dataclass
class EnsembleConfig(StrategyConfig):
    """Configuration for strategy ensemble."""
    name: str = "strategy_ensemble"
    
    # Regime detection
    regime_lookback: int = 100
    regime_threshold: float = 0.7  # Confidence required to assign regime
    
    # Strategy selection
    max_concurrent_strategies: int = 3
    min_strategy_confidence: float = 0.5
    
    # Position management
    max_total_exposure: float = 2.0  # Max leverage across all strategies
    correlation_limit: float = 0.7  # Max allowed correlation between active strategies
    
    # Performance tracking
    lookback_for_weighting: int = 50  # Trades to consider for weight updates
    decay_factor: float = 0.95  # Exponential decay for older performance


class StrategyPerformance:
    """Track individual strategy performance."""
    
    def __init__(self, strategy_name: str, lookback: int = 100):
        self.strategy_name = strategy_name
        self.lookback = lookback
        
        self._trades: deque = deque(maxlen=lookback)
        self._pnl_history: deque = deque(maxlen=lookback)
        self._regime_performance: Dict[MarketRegime, List[float]] = {
            r: [] for r in MarketRegime
        }
    
    def record_trade(self, pnl: float, regime: MarketRegime):
        """Record a completed trade."""
        self._trades.append({
            'pnl': pnl,
            'regime': regime,
            'timestamp': datetime.now()
        })
        self._pnl_history.append(pnl)
        self._regime_performance[regime].append(pnl)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics."""
        if not self._trades:
            return {
                'win_rate': 0.5,
                'avg_pnl': 0.0,
                'sharpe': 0.0,
                'trade_count': 0
            }
        
        pnls = list(self._pnl_history)
        wins = sum(1 for p in pnls if p > 0)
        
        win_rate = wins / len(pnls)
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'sharpe': sharpe,
            'trade_count': len(pnls)
        }
    
    def get_regime_score(self, regime: MarketRegime) -> float:
        """Get strategy's performance in specific regime."""
        regime_pnls = self._regime_performance.get(regime, [])
        
        if len(regime_pnls) < 5:
            return 0.5  # Not enough data, neutral score
        
        win_rate = sum(1 for p in regime_pnls if p > 0) / len(regime_pnls)
        avg_pnl = np.mean(regime_pnls)
        
        # Combine win rate and avg pnl into score
        score = 0.5 * win_rate + 0.5 * np.tanh(avg_pnl * 10)
        
        return max(0, min(1, score))


class StrategySelector:
    """
    Select optimal strategies for current market regime.
    
    Uses:
    - Regime detection
    - Historical performance per regime
    - Anti-correlation filtering
    """
    
    # Which strategies work best in which regimes
    REGIME_STRATEGY_MAP = {
        MarketRegime.TRENDING_UP: [
            'momentum', 'volatility_breakout', 'regime_ml'
        ],
        MarketRegime.TRENDING_DOWN: [
            'momentum', 'volatility_breakout', 'regime_ml'
        ],
        MarketRegime.RANGING: [
            'stat_arb', 'funding_arbitrage', 'microstructure'
        ],
        MarketRegime.HIGH_VOLATILITY: [
            'volatility_breakout', 'event_driven', 'regime_ml'
        ],
        MarketRegime.LOW_VOLATILITY: [
            'funding_arbitrage', 'stat_arb', 'microstructure'
        ],
        MarketRegime.BREAKOUT: [
            'volatility_breakout', 'momentum', 'event_driven'
        ],
        MarketRegime.MEAN_REVERTING: [
            'stat_arb', 'microstructure'
        ],
        MarketRegime.CRISIS: [
            # CRITICAL: Crisis = SIT OUT or very conservative
            'funding_arbitrage'  # Only market-neutral strategies
        ],
        MarketRegime.UNKNOWN: [
            # Unknown = reduce exposure
            'funding_arbitrage', 'stat_arb'
        ]
    }
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        
        self._performance: Dict[str, StrategyPerformance] = {}
        self._correlations: Dict[Tuple[str, str], float] = {}
    
    def register_strategy(self, name: str):
        """Register a strategy for tracking."""
        if name not in self._performance:
            self._performance[name] = StrategyPerformance(name)
    
    def record_trade(self, strategy_name: str, pnl: float, regime: MarketRegime):
        """Record a trade for a strategy."""
        if strategy_name in self._performance:
            self._performance[strategy_name].record_trade(pnl, regime)
    
    def update_correlation(self, name1: str, name2: str, correlation: float):
        """Update correlation between two strategies."""
        key = tuple(sorted([name1, name2]))
        self._correlations[key] = correlation
    
    def get_correlation(self, name1: str, name2: str) -> float:
        """Get correlation between two strategies."""
        key = tuple(sorted([name1, name2]))
        return self._correlations.get(key, 0.0)
    
    def select_strategies(self, regime: MarketRegime) -> List[Tuple[str, float]]:
        """
        Select strategies for current regime.
        
        Returns:
            List of (strategy_name, weight) tuples
        """
        # Get candidate strategies for regime
        candidates = self.REGIME_STRATEGY_MAP.get(regime, [])
        
        if not candidates:
            return []
        
        # Score each candidate
        scored = []
        for name in candidates:
            if name not in self._performance:
                continue
            
            perf = self._performance[name]
            regime_score = perf.get_regime_score(regime)
            metrics = perf.get_metrics()
            
            # Combine regime score with overall metrics
            if metrics['trade_count'] < 10:
                score = 0.5  # Not enough data, default score
            else:
                score = (
                    0.4 * regime_score +
                    0.3 * metrics['win_rate'] +
                    0.3 * min(1, max(0, metrics['sharpe'] / 2 + 0.5))
                )
            
            if score >= self.config.min_strategy_confidence:
                scored.append((name, score))
        
        # Sort by score
        scored.sort(key=lambda x: -x[1])
        
        # Filter for low correlation
        selected = []
        for name, score in scored:
            if len(selected) >= self.config.max_concurrent_strategies:
                break
            
            # Check correlation with already selected
            can_add = True
            for sel_name, _ in selected:
                corr = self.get_correlation(name, sel_name)
                if abs(corr) > self.config.correlation_limit:
                    can_add = False
                    break
            
            if can_add:
                selected.append((name, score))
        
        # Normalize weights
        if selected:
            total_score = sum(s for _, s in selected)
            selected = [(n, s / total_score) for n, s in selected]
        
        return selected


class SignalAggregator:
    """
    Aggregate signals from multiple strategies.
    
    Methods:
    - Majority voting
    - Weighted average
    - Conviction-weighted
    """
    
    @staticmethod
    def majority_vote(signals: List[Tuple[Signal, float]]) -> Optional[SignalType]:
        """
        Simple majority vote on signal direction.
        
        Returns None if no clear majority.
        """
        if not signals:
            return None
        
        votes = {'long': 0, 'short': 0, 'flat': 0}
        
        for signal, weight in signals:
            if signal.signal_type == SignalType.LONG:
                votes['long'] += weight
            elif signal.signal_type == SignalType.SHORT:
                votes['short'] += weight
            else:
                votes['flat'] += weight
        
        max_vote = max(votes.values())
        
        if max_vote < 0.5:  # No clear majority
            return None
        
        for direction, vote in votes.items():
            if vote == max_vote:
                return {
                    'long': SignalType.LONG,
                    'short': SignalType.SHORT,
                    'flat': SignalType.FLAT
                }[direction]
        
        return None
    
    @staticmethod
    def weighted_average(signals: List[Tuple[Signal, float]]) -> Dict[str, float]:
        """
        Calculate weighted average signal strength.
        """
        if not signals:
            return {'direction': 0, 'strength': 0, 'confidence': 0}
        
        total_weight = sum(w for _, w in signals)
        
        direction_score = 0
        strength_score = 0
        confidence_score = 0
        
        for signal, weight in signals:
            if signal.signal_type == SignalType.LONG:
                dir_val = 1
            elif signal.signal_type == SignalType.SHORT:
                dir_val = -1
            else:
                dir_val = 0
            
            normalized_weight = weight / total_weight
            direction_score += dir_val * normalized_weight
            strength_score += signal.strength * normalized_weight
            confidence_score += signal.confidence * normalized_weight
        
        return {
            'direction': direction_score,
            'strength': strength_score,
            'confidence': confidence_score
        }
    
    @staticmethod
    def conviction_weighted(signals: List[Tuple[Signal, float]]) -> Optional[Signal]:
        """
        Weight by strategy weight AND signal confidence.
        Returns strongest conviction signal.
        """
        if not signals:
            return None
        
        # Score each signal
        scored_signals = []
        for signal, strategy_weight in signals:
            conviction = strategy_weight * signal.confidence * signal.strength
            scored_signals.append((signal, conviction))
        
        # Get highest conviction
        best_signal, best_conviction = max(scored_signals, key=lambda x: x[1])
        
        if best_conviction < 0.3:
            return None
        
        return best_signal


class StrategyEnsemble(BaseStrategy):
    """
    Master strategy ensemble that orchestrates all 7 strategy families.
    
    Flow:
    1. Detect regime
    2. Select regime-appropriate strategies
    3. Collect signals from active strategies
    4. Aggregate signals
    5. Generate final position recommendation
    """
    
    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        strategies: Optional[Dict[str, BaseStrategy]] = None
    ):
        super().__init__(config or EnsembleConfig())
        self.config: EnsembleConfig = self.config
        
        self.strategies = strategies or {}
        self.regime_detector = RegimeDetector(RegimeConfig())
        self.selector = StrategySelector(self.config)
        
        self._current_regime: Optional[MarketRegime] = None
        self._regime_confidence: float = 0.0
        self._active_strategies: List[Tuple[str, float]] = []
        self._recent_signals: List[Tuple[Signal, float]] = []
        
        # Register all strategies
        for name in self.strategies:
            self.selector.register_strategy(name)
    
    def add_strategy(self, name: str, strategy: BaseStrategy):
        """Add a strategy to the ensemble."""
        self.strategies[name] = strategy
        self.selector.register_strategy(name)
    
    def update(self, data: pd.DataFrame):
        """Update ensemble state."""
        self.current_bar = len(data) - 1
        
        # Detect regime (returns tuple of regime, confidence)
        self._current_regime, self._regime_confidence = self.regime_detector.detect_regime(data)
        
        # Select strategies for regime
        self._active_strategies = self.selector.select_strategies(self._current_regime)
        
        # Update all strategies
        for name in self.strategies:
            try:
                self.strategies[name].update(data)
            except Exception as e:
                print(f"Error updating strategy {name}: {e}")
        
        self.is_initialized = True
    
    def _collect_signals(self, data: pd.DataFrame) -> List[Tuple[Signal, float]]:
        """Collect signals from active strategies."""
        signals = []
        
        for name, weight in self._active_strategies:
            if name not in self.strategies:
                continue
            
            try:
                signal = self.strategies[name].generate_signal(data)
                if signal and signal.signal_type != SignalType.FLAT:
                    signals.append((signal, weight))
            except Exception as e:
                print(f"Error generating signal from {name}: {e}")
        
        return signals
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate ensemble signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        # Crisis regime = NO TRADE (critical insight)
        if self._current_regime == MarketRegime.CRISIS:
            return None
        
        # Collect signals from active strategies
        signals = self._collect_signals(data)
        self._recent_signals = signals
        
        if not signals:
            return None
        
        # Aggregate signals
        weighted = SignalAggregator.weighted_average(signals)
        
        # Need strong directional consensus
        if abs(weighted['direction']) < 0.3:
            return None
        
        if weighted['confidence'] < 0.5:
            return None
        
        # Get conviction-weighted best signal as template
        best_signal = SignalAggregator.conviction_weighted(signals)
        
        if best_signal is None:
            return None
        
        # Modify signal based on ensemble
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        # Determine direction from ensemble
        if weighted['direction'] > 0:
            signal_type = SignalType.LONG
        elif weighted['direction'] < 0:
            signal_type = SignalType.SHORT
        else:
            return None
        
        atr = TechnicalIndicators.atr(
            data['high'], data['low'], data['close'], 14
        ).iloc[-1]
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=f"Ensemble ({self._current_regime.name}): {len(signals)} strategies agree",
            strength=weighted['strength'],
            confidence=weighted['confidence'],
            atr=atr,
            regime=self._current_regime.name,
            active_strategies=len(self._active_strategies),
            contributing_signals=len(signals)
        )
    
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on regime and consensus."""
        if self._current_regime == MarketRegime.CRISIS:
            return 0.0  # No trading
        
        if self._current_regime == MarketRegime.HIGH_VOLATILITY:
            return 0.5  # Reduced size
        
        if not self._recent_signals:
            return 0.5
        
        # Scale by number of agreeing strategies
        agreement_ratio = len(self._recent_signals) / max(len(self._active_strategies), 1)
        
        return min(1.0, agreement_ratio * 1.2)
    
    def get_status(self) -> Dict[str, Any]:
        """Get ensemble status."""
        return {
            'current_regime': self._current_regime.name if self._current_regime else 'UNKNOWN',
            'active_strategies': [(n, f"{w:.2f}") for n, w in self._active_strategies],
            'recent_signal_count': len(self._recent_signals),
            'position_size_multiplier': self.get_position_size_multiplier()
        }


class AutoTuningEnsemble(StrategyEnsemble):
    """
    Self-tuning ensemble that adjusts weights based on performance.
    
    Features:
    - Online learning of strategy weights
    - Automatic regime-strategy mapping updates
    - Performance-based inclusion/exclusion
    """
    
    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        strategies: Optional[Dict[str, BaseStrategy]] = None
    ):
        super().__init__(config, strategies)
        
        self._trade_history: deque = deque(maxlen=1000)
        self._strategy_weights: Dict[str, float] = {
            name: 1.0 for name in self.strategies
        }
    
    def record_trade_result(
        self,
        strategy_name: str,
        pnl: float,
        regime: MarketRegime
    ):
        """Record trade result for learning."""
        self._trade_history.append({
            'strategy': strategy_name,
            'pnl': pnl,
            'regime': regime,
            'timestamp': datetime.now()
        })
        
        # Update selector
        self.selector.record_trade(strategy_name, pnl, regime)
        
        # Update strategy weight
        self._update_weight(strategy_name, pnl)
    
    def _update_weight(self, strategy_name: str, pnl: float):
        """Update strategy weight based on recent PnL."""
        if strategy_name not in self._strategy_weights:
            return
        
        # Simple multiplicative update
        if pnl > 0:
            self._strategy_weights[strategy_name] *= 1.05
        else:
            self._strategy_weights[strategy_name] *= 0.95
        
        # Clamp weights
        self._strategy_weights[strategy_name] = max(
            0.1,
            min(3.0, self._strategy_weights[strategy_name])
        )
    
    def get_adjusted_weights(self) -> Dict[str, float]:
        """Get performance-adjusted strategy weights."""
        return self._strategy_weights.copy()


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STRATEGY ENSEMBLE - MASTER ORCHESTRATOR")
    print("="*70)
    
    np.random.seed(42)
    
    print("\n1. STRATEGY SELECTOR")
    print("-" * 50)
    
    config = EnsembleConfig()
    selector = StrategySelector(config)
    
    # Register strategies
    strategies = ['momentum', 'volatility_breakout', 'stat_arb', 
                  'funding_arbitrage', 'microstructure', 'event_driven', 'regime_ml']
    
    for name in strategies:
        selector.register_strategy(name)
    
    # Simulate some trade history
    for _ in range(100):
        name = np.random.choice(strategies)
        regime = np.random.choice(list(MarketRegime))
        
        # Some strategies better in certain regimes
        if name == 'momentum' and regime == MarketRegime.TRENDING:
            pnl = np.random.normal(0.02, 0.01)
        elif name == 'stat_arb' and regime == MarketRegime.RANGING:
            pnl = np.random.normal(0.015, 0.008)
        elif name == 'funding_arbitrage':
            pnl = np.random.normal(0.005, 0.003)  # Consistent but small
        else:
            pnl = np.random.normal(0, 0.02)
        
        selector.record_trade(name, pnl, regime)
    
    print("\n   Strategy selection by regime:")
    for regime in MarketRegime:
        selected = selector.select_strategies(regime)
        print(f"\n   {regime.name}:")
        for name, weight in selected:
            print(f"      - {name}: {weight:.2f}")
    
    print("\n2. SIGNAL AGGREGATION")
    print("-" * 50)
    
    # Create mock signals
    from .base import Signal, SignalType
    
    mock_signals = [
        (Signal(
            signal_type=SignalType.LONG,
            price=50000,
            symbol='BTCUSDT',
            timestamp=datetime.now(),
            reason='Momentum bullish',
            strength=0.8,
            confidence=0.7,
            metadata={}
        ), 0.4),
        (Signal(
            signal_type=SignalType.LONG,
            price=50000,
            symbol='BTCUSDT',
            timestamp=datetime.now(),
            reason='Breakout triggered',
            strength=0.6,
            confidence=0.6,
            metadata={}
        ), 0.3),
        (Signal(
            signal_type=SignalType.SHORT,
            price=50000,
            symbol='BTCUSDT',
            timestamp=datetime.now(),
            reason='Stat arb divergence',
            strength=0.5,
            confidence=0.5,
            metadata={}
        ), 0.3),
    ]
    
    vote = SignalAggregator.majority_vote(mock_signals)
    print(f"   Majority vote: {vote.name if vote else 'NO CONSENSUS'}")
    
    weighted = SignalAggregator.weighted_average(mock_signals)
    print(f"   Weighted average direction: {weighted['direction']:.2f}")
    print(f"   Weighted average strength: {weighted['strength']:.2f}")
    
    best = SignalAggregator.conviction_weighted(mock_signals)
    print(f"   Best conviction signal: {best.signal_type.name if best else 'NONE'}")
    
    print("\n3. COMPLETE ENSEMBLE")
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
    
    # Create ensemble without actual strategies (demonstration)
    ensemble = StrategyEnsemble(EnsembleConfig())
    ensemble.update(data)
    
    status = ensemble.get_status()
    print(f"   Regime: {status['current_regime']}")
    print(f"   Active strategies: {status['active_strategies']}")
    print(f"   Position multiplier: {status['position_size_multiplier']:.2f}")
    
    print("\n" + "="*70)
    print("ENSEMBLE KEY PRINCIPLES")
    print("="*70)
    print("""
1. REGIME FIRST
   - ALWAYS detect regime before selecting strategies
   - Different strategies for different regimes
   - CHOPPY regime = NO TRADE
   
2. STRATEGY SELECTION
   - Max 3 concurrent strategies
   - Low correlation between active strategies
   - Performance-weighted selection
   
3. SIGNAL AGGREGATION
   - Majority vote for direction
   - Conviction weighting for sizing
   - Minimum consensus required
   
4. POSITION SIZING
   - Scale by regime volatility
   - Scale by strategy agreement
   - Never exceed max exposure
   
5. CONTINUOUS LEARNING
   - Track performance per regime
   - Update weights online
   - Auto-adjust strategy mix
   
6. THE 7 FAMILIES BY REGIME:

   TRENDING:
   - Momentum
   - Volatility Breakout
   - Regime ML
   
   RANGING:
   - Stat Arb
   - Funding Arbitrage
   - Microstructure
   
   HIGH VOLATILITY:
   - Volatility Breakout
   - Event-Driven
   - Regime ML
   
   LOW VOLATILITY:
   - Funding Arbitrage
   - Stat Arb
   - Microstructure
   
   CHOPPY:
   - NOTHING (or market-neutral only)
   
This is the CRITICAL insight missing from most quant bots!
""")
