"""
Ensemble Strategy Engine - Combines Multiple Alpha Sources

Implements machine learning-based ensemble methods:
- Dynamic strategy weighting
- Performance-based adaptation
- Regime-aware combination
- Risk parity allocation
- Meta-learning for optimal blending
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum


class CombinationMethod(Enum):
    """Strategy combination methods"""
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHT = "performance_weight"
    RISK_PARITY = "risk_parity"
    META_LEARNING = "meta_learning"
    REGIME_BASED = "regime_based"
    MOMENTUM_OF_MOMENTUM = "momentum_of_momentum"


@dataclass
class StrategySignal:
    """Individual strategy signal"""
    strategy_name: str
    direction: int  # -1, 0, 1
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: datetime


@dataclass
class StrategyPerformance:
    """Track individual strategy performance"""
    strategy_name: str
    returns: deque = field(default_factory=lambda: deque(maxlen=100))
    signals: deque = field(default_factory=lambda: deque(maxlen=100))
    hit_rate: float = 0.5
    sharpe: float = 0.0
    recent_pnl: float = 0.0
    weight: float = 1.0
    drawdown: float = 0.0
    
    def update(self, signal: int, realized_return: float) -> None:
        """Update performance metrics"""
        self.returns.append(realized_return)
        self.signals.append(signal)
        
        if len(self.returns) >= 10:
            returns_arr = np.array(self.returns)
            
            # Hit rate
            correct = sum(1 for s, r in zip(self.signals, self.returns) 
                         if (s > 0 and r > 0) or (s < 0 and r < 0))
            self.hit_rate = correct / len(self.signals)
            
            # Sharpe ratio
            if np.std(returns_arr) > 0:
                self.sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
            
            # Recent P&L
            self.recent_pnl = np.sum(returns_arr[-20:])
            
            # Drawdown
            cumulative = np.cumsum(returns_arr)
            peak = np.maximum.accumulate(cumulative)
            dd = peak - cumulative
            self.drawdown = np.max(dd) if len(dd) > 0 else 0


@dataclass
class EnsembleSignal:
    """Combined ensemble signal"""
    direction: int
    strength: float
    confidence: float
    component_signals: Dict[str, StrategySignal]
    weights: Dict[str, float]
    method: CombinationMethod
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float


class EnsembleEngine:
    """
    Ensemble Strategy Engine
    
    Dynamically combines multiple strategies using various
    ensemble methods to maximize risk-adjusted returns.
    """
    
    def __init__(
        self,
        combination_method: CombinationMethod = CombinationMethod.META_LEARNING,
        min_strategies: int = 2,
        rebalance_period: int = 24,  # hours
        min_weight: float = 0.05,
        max_weight: float = 0.5,
        risk_target: float = 0.02  # 2% daily risk
    ):
        """
        Initialize ensemble engine
        
        Args:
            combination_method: Method for combining strategies
            min_strategies: Minimum strategies for signal
            rebalance_period: Hours between weight updates
            min_weight: Minimum strategy weight
            max_weight: Maximum strategy weight
            risk_target: Target daily risk
        """
        self.method = combination_method
        self.min_strategies = min_strategies
        self.rebalance_period = rebalance_period
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_target = risk_target
        
        # Strategy tracking
        self._strategies: Dict[str, Callable] = {}
        self._performance: Dict[str, StrategyPerformance] = {}
        self._weights: Dict[str, float] = {}
        
        # State
        self._last_rebalance: datetime = datetime.utcnow()
        self._current_regime: str = "neutral"
        
        # Meta-learning parameters
        self._meta_weights: np.ndarray = None
        self._feature_history: deque = deque(maxlen=1000)
    
    def register_strategy(
        self,
        name: str,
        signal_func: Callable,
        initial_weight: float = None
    ) -> None:
        """
        Register a strategy
        
        Args:
            name: Strategy name
            signal_func: Function that returns StrategySignal
            initial_weight: Initial weight (default: equal)
        """
        self._strategies[name] = signal_func
        self._performance[name] = StrategyPerformance(strategy_name=name)
        
        # Initialize weights
        n_strategies = len(self._strategies)
        if initial_weight:
            self._weights[name] = initial_weight
        else:
            # Equal weight
            for s in self._strategies:
                self._weights[s] = 1.0 / n_strategies
    
    def generate_signal(
        self,
        market_data: Dict,
        regime: str = None
    ) -> EnsembleSignal:
        """
        Generate combined ensemble signal
        
        Args:
            market_data: Market data for strategies
            regime: Current market regime
        
        Returns:
            EnsembleSignal with combined recommendation
        """
        if regime:
            self._current_regime = regime
        
        # Check for rebalance
        if self._should_rebalance():
            self._rebalance_weights()
        
        # Collect signals from all strategies
        signals = {}
        for name, func in self._strategies.items():
            try:
                signal = func(market_data)
                signals[name] = signal
            except Exception as e:
                # Strategy error - use neutral signal
                signals[name] = StrategySignal(
                    strategy_name=name,
                    direction=0,
                    strength=0,
                    confidence=0,
                    timestamp=datetime.utcnow()
                )
        
        # Check minimum strategies
        active_signals = {k: v for k, v in signals.items() if v.direction != 0}
        if len(active_signals) < self.min_strategies:
            return self._neutral_signal(signals)
        
        # Combine based on method
        if self.method == CombinationMethod.EQUAL_WEIGHT:
            combined = self._equal_weight_combine(signals)
        elif self.method == CombinationMethod.PERFORMANCE_WEIGHT:
            combined = self._performance_weight_combine(signals)
        elif self.method == CombinationMethod.RISK_PARITY:
            combined = self._risk_parity_combine(signals)
        elif self.method == CombinationMethod.META_LEARNING:
            combined = self._meta_learning_combine(signals, market_data)
        elif self.method == CombinationMethod.REGIME_BASED:
            combined = self._regime_based_combine(signals)
        else:
            combined = self._momentum_of_momentum_combine(signals)
        
        return combined
    
    def update_performance(
        self,
        strategy_name: str,
        signal: int,
        realized_return: float
    ) -> None:
        """Update strategy performance after trade"""
        if strategy_name in self._performance:
            self._performance[strategy_name].update(signal, realized_return)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self._weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary for all strategies"""
        summary = {}
        for name, perf in self._performance.items():
            summary[name] = {
                'weight': self._weights.get(name, 0),
                'hit_rate': perf.hit_rate,
                'sharpe': perf.sharpe,
                'recent_pnl': perf.recent_pnl,
                'drawdown': perf.drawdown
            }
        return summary
    
    def _should_rebalance(self) -> bool:
        """Check if weights should be rebalanced"""
        elapsed = (datetime.utcnow() - self._last_rebalance).total_seconds() / 3600
        return elapsed >= self.rebalance_period
    
    def _rebalance_weights(self) -> None:
        """Rebalance strategy weights based on performance"""
        self._last_rebalance = datetime.utcnow()
        
        # Calculate performance scores
        scores = {}
        for name, perf in self._performance.items():
            if len(perf.returns) < 10:
                scores[name] = 1.0  # New strategy gets neutral score
            else:
                # Score based on Sharpe, hit rate, and recent performance
                score = (
                    max(0, perf.sharpe) * 0.4 +
                    perf.hit_rate * 0.3 +
                    (1 + perf.recent_pnl) * 0.3
                )
                scores[name] = max(0.1, score)  # Minimum score
        
        # Normalize to weights
        total = sum(scores.values())
        for name in self._strategies:
            raw_weight = scores.get(name, 1.0) / total if total > 0 else 1.0 / len(self._strategies)
            self._weights[name] = np.clip(raw_weight, self.min_weight, self.max_weight)
        
        # Renormalize
        total = sum(self._weights.values())
        if total > 0:
            for name in self._weights:
                self._weights[name] /= total
    
    def _equal_weight_combine(self, signals: Dict[str, StrategySignal]) -> EnsembleSignal:
        """Equal weight combination"""
        n = len(signals)
        weights = {k: 1.0/n for k in signals}
        
        return self._weighted_combine(signals, weights)
    
    def _performance_weight_combine(self, signals: Dict[str, StrategySignal]) -> EnsembleSignal:
        """Performance-based weight combination"""
        return self._weighted_combine(signals, self._weights)
    
    def _risk_parity_combine(self, signals: Dict[str, StrategySignal]) -> EnsembleSignal:
        """Risk parity weight combination"""
        # Calculate inverse volatility weights
        inv_vol_weights = {}
        
        for name in signals:
            perf = self._performance.get(name)
            if perf and len(perf.returns) >= 10:
                vol = np.std(perf.returns)
                inv_vol_weights[name] = 1.0 / max(vol, 0.001)
            else:
                inv_vol_weights[name] = 1.0
        
        # Normalize
        total = sum(inv_vol_weights.values())
        weights = {k: v/total for k, v in inv_vol_weights.items()}
        
        return self._weighted_combine(signals, weights)
    
    def _meta_learning_combine(
        self,
        signals: Dict[str, StrategySignal],
        market_data: Dict
    ) -> EnsembleSignal:
        """Meta-learning based combination"""
        # Extract features
        features = self._extract_features(market_data)
        
        # If no meta-model trained, use performance weights
        if self._meta_weights is None or len(self._feature_history) < 100:
            self._feature_history.append((features, signals))
            return self._performance_weight_combine(signals)
        
        # Predict optimal weights
        predicted_weights = self._predict_weights(features)
        
        # Store for learning
        self._feature_history.append((features, signals))
        
        return self._weighted_combine(signals, predicted_weights)
    
    def _regime_based_combine(self, signals: Dict[str, StrategySignal]) -> EnsembleSignal:
        """Regime-based weight adjustment"""
        weights = self._weights.copy()
        
        # Adjust based on regime
        if self._current_regime == "trending":
            # Favor momentum strategies
            for name in weights:
                if "momentum" in name.lower():
                    weights[name] *= 1.5
                elif "mean_reversion" in name.lower():
                    weights[name] *= 0.5
        
        elif self._current_regime == "mean_reverting":
            # Favor mean reversion strategies
            for name in weights:
                if "mean_reversion" in name.lower():
                    weights[name] *= 1.5
                elif "momentum" in name.lower():
                    weights[name] *= 0.5
        
        elif self._current_regime == "high_volatility":
            # Favor volatility and options strategies
            for name in weights:
                if "volatility" in name.lower() or "options" in name.lower():
                    weights[name] *= 1.3
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return self._weighted_combine(signals, weights)
    
    def _momentum_of_momentum_combine(self, signals: Dict[str, StrategySignal]) -> EnsembleSignal:
        """Weight by recent strategy momentum"""
        weights = {}
        
        for name in signals:
            perf = self._performance.get(name)
            if perf and len(perf.returns) >= 5:
                recent_mom = sum(perf.returns[-5:])
                weights[name] = max(0.1, 1 + recent_mom)  # Higher weight for positive momentum
            else:
                weights[name] = 1.0
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return self._weighted_combine(signals, weights)
    
    def _weighted_combine(
        self,
        signals: Dict[str, StrategySignal],
        weights: Dict[str, float]
    ) -> EnsembleSignal:
        """Combine signals with weights"""
        weighted_direction = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for name, signal in signals.items():
            w = weights.get(name, 0)
            if w > 0:
                weighted_direction += signal.direction * w * signal.confidence
                weighted_strength += signal.strength * w
                weighted_confidence += signal.confidence * w
                total_weight += w
        
        if total_weight == 0:
            return self._neutral_signal(signals)
        
        # Normalize
        avg_direction = weighted_direction / total_weight
        avg_strength = weighted_strength / total_weight
        avg_confidence = weighted_confidence / total_weight
        
        # Determine final direction
        if abs(avg_direction) > 0.3:  # Threshold
            direction = 1 if avg_direction > 0 else -1
        else:
            direction = 0
        
        # Calculate position size based on confidence and risk target
        position_size = min(
            self.risk_target * avg_confidence,
            self.max_weight  # Cap at max weight
        )
        
        # Estimate stop/take profit from strongest signal
        best_signal = max(signals.values(), key=lambda s: s.confidence * s.strength)
        current_price = 50000  # Placeholder
        atr_pct = 0.02  # 2% ATR assumption
        
        if direction > 0:
            stop_loss = current_price * (1 - atr_pct * 1.5)
            take_profit = current_price * (1 + atr_pct * 2.5)
        elif direction < 0:
            stop_loss = current_price * (1 + atr_pct * 1.5)
            take_profit = current_price * (1 - atr_pct * 2.5)
        else:
            stop_loss = take_profit = current_price
        
        return EnsembleSignal(
            direction=direction,
            strength=avg_strength,
            confidence=avg_confidence,
            component_signals=signals,
            weights=weights,
            method=self.method,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size
        )
    
    def _extract_features(self, market_data: Dict) -> np.ndarray:
        """Extract features for meta-learning"""
        features = []
        
        # Price features
        prices = market_data.get('prices', [])
        if len(prices) >= 20:
            returns = np.diff(prices) / prices[:-1]
            features.extend([
                np.mean(returns[-5:]),  # Recent momentum
                np.mean(returns[-20:]),  # Longer momentum
                np.std(returns[-20:]),  # Volatility
                (prices[-1] - np.mean(prices[-20:])) / np.std(prices[-20:]) if np.std(prices[-20:]) > 0 else 0,  # Z-score
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Volume features
        volumes = market_data.get('volumes', [])
        if len(volumes) >= 10:
            features.extend([
                volumes[-1] / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) > 0 else 1,  # Volume ratio
            ])
        else:
            features.append(1)
        
        # Orderbook features
        orderbook = market_data.get('orderbook', {})
        if orderbook:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            bid_vol = sum(b[1] for b in bids[:5]) if bids else 0
            ask_vol = sum(a[1] for a in asks[:5]) if asks else 0
            total = bid_vol + ask_vol
            features.append((bid_vol - ask_vol) / total if total > 0 else 0)
        else:
            features.append(0)
        
        return np.array(features)
    
    def _predict_weights(self, features: np.ndarray) -> Dict[str, float]:
        """Predict optimal weights using meta-model"""
        # Simple linear prediction (can be replaced with more complex model)
        if self._meta_weights is None:
            return self._weights
        
        # Compute score for each strategy
        scores = {}
        for i, name in enumerate(self._strategies):
            if i < len(self._meta_weights):
                score = np.dot(features, self._meta_weights[i])
                scores[name] = max(0.1, score)
            else:
                scores[name] = 1.0
        
        # Normalize
        total = sum(scores.values())
        return {k: np.clip(v/total, self.min_weight, self.max_weight) for k, v in scores.items()}
    
    def train_meta_model(self) -> None:
        """Train meta-learning model on historical data"""
        if len(self._feature_history) < 100:
            return
        
        # Collect training data
        features_list = []
        outcomes_list = []
        
        for features, signals in self._feature_history:
            features_list.append(features)
            
            # Outcome: which strategies performed well
            outcomes = []
            for name in self._strategies:
                perf = self._performance.get(name)
                if perf and len(perf.returns) > 0:
                    outcomes.append(perf.returns[-1] if len(perf.returns) > 0 else 0)
                else:
                    outcomes.append(0)
            outcomes_list.append(outcomes)
        
        # Simple ridge regression for each strategy
        X = np.array(features_list)
        Y = np.array(outcomes_list)
        
        if X.shape[0] > 0 and Y.shape[0] > 0:
            self._meta_weights = []
            for i in range(Y.shape[1]):
                y = Y[:, i]
                # Ridge regression: (X'X + λI)^-1 X'y
                lambda_reg = 0.1
                XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
                Xty = X.T @ y
                try:
                    w = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    w = np.zeros(X.shape[1])
                self._meta_weights.append(w)
            
            self._meta_weights = np.array(self._meta_weights)
    
    def _neutral_signal(self, signals: Dict[str, StrategySignal]) -> EnsembleSignal:
        """Return neutral signal"""
        return EnsembleSignal(
            direction=0,
            strength=0,
            confidence=0,
            component_signals=signals,
            weights=self._weights,
            method=self.method,
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            position_size_pct=0
        )
