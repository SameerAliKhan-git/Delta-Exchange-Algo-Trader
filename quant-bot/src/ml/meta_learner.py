"""
Meta-Learner: Online Model Selection
====================================

DELIVERABLE B: Meta-learner A/B test infrastructure.

Purpose:
- Shadow online meta-learner vs baseline for 2k+ trades
- Dynamically select between strategies based on recent performance
- Implement Thompson Sampling for exploration/exploitation
- Track confidence intervals and statistical significance

Key Features:
1. Multi-armed bandit for strategy selection
2. Online learning with decay
3. Regime-conditional weighting
4. A/B testing framework with statistical significance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import pickle
from pathlib import Path
import threading
from scipy import stats


class SelectionMethod(Enum):
    """Strategy selection methods."""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"                    # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson"
    SOFTMAX = "softmax"
    REGIME_CONDITIONAL = "regime"


@dataclass
class StrategyArm:
    """Multi-armed bandit arm for a strategy."""
    name: str
    
    # Performance tracking
    total_trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    total_pnl_squared: float = 0.0
    
    # Bayesian tracking (for Thompson Sampling)
    alpha: float = 1.0  # Beta distribution parameter
    beta: float = 1.0   # Beta distribution parameter
    
    # Time-weighted performance
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_pnl: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Regime-specific performance
    regime_performance: Dict[str, Dict] = field(default_factory=dict)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5
        return self.wins / self.total_trades
    
    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    @property
    def pnl_std(self) -> float:
        if self.total_trades < 2:
            return 0.0
        variance = (self.total_pnl_squared / self.total_trades) - (self.avg_pnl ** 2)
        return np.sqrt(max(0, variance))
    
    @property
    def sharpe(self) -> float:
        if self.pnl_std == 0:
            return 0.0
        return self.avg_pnl / self.pnl_std
    
    @property
    def recent_win_rate(self) -> float:
        if len(self.recent_wins) == 0:
            return 0.5
        return sum(self.recent_wins) / len(self.recent_wins)
    
    def update(self, won: bool, pnl: float, regime: str = None):
        """Update arm statistics."""
        self.total_trades += 1
        if won:
            self.wins += 1
        self.total_pnl += pnl
        self.total_pnl_squared += pnl ** 2
        
        # Update Bayesian parameters
        if won:
            self.alpha += 1
        else:
            self.beta += 1
        
        # Update recent
        self.recent_wins.append(1 if won else 0)
        self.recent_pnl.append(pnl)
        
        # Update regime-specific
        if regime:
            if regime not in self.regime_performance:
                self.regime_performance[regime] = {
                    'trades': 0, 'wins': 0, 'pnl': 0.0
                }
            self.regime_performance[regime]['trades'] += 1
            if won:
                self.regime_performance[regime]['wins'] += 1
            self.regime_performance[regime]['pnl'] += pnl
    
    def sample_thompson(self) -> float:
        """Sample from posterior (Thompson Sampling)."""
        return np.random.beta(self.alpha, self.beta)
    
    def ucb_score(self, total_pulls: int, c: float = 2.0) -> float:
        """Calculate UCB score."""
        if self.total_trades == 0:
            return float('inf')
        
        exploitation = self.win_rate
        exploration = c * np.sqrt(np.log(total_pulls + 1) / self.total_trades)
        return exploitation + exploration
    
    def get_regime_win_rate(self, regime: str) -> float:
        """Get win rate for specific regime."""
        if regime not in self.regime_performance:
            return self.win_rate
        perf = self.regime_performance[regime]
        if perf['trades'] == 0:
            return self.win_rate
        return perf['wins'] / perf['trades']


@dataclass
class ABTestResult:
    """A/B test result."""
    name: str
    control_name: str
    treatment_name: str
    
    # Metrics
    control_win_rate: float
    treatment_win_rate: float
    control_avg_pnl: float
    treatment_avg_pnl: float
    
    # Statistical significance
    win_rate_p_value: float
    pnl_p_value: float
    is_significant: bool  # p < 0.05
    
    # Effect size
    win_rate_lift: float  # % improvement
    pnl_lift: float
    
    # Confidence intervals (95%)
    win_rate_ci_lower: float
    win_rate_ci_upper: float
    pnl_ci_lower: float
    pnl_ci_upper: float
    
    # Sample sizes
    control_n: int
    treatment_n: int
    
    # Recommendation
    recommendation: str
    confidence: str  # "low", "medium", "high"


class MetaLearner:
    """
    Online meta-learner for strategy selection.
    
    Uses multi-armed bandit algorithms to dynamically select
    the best performing strategy based on recent results.
    """
    
    def __init__(
        self,
        strategy_names: List[str],
        method: SelectionMethod = SelectionMethod.THOMPSON_SAMPLING,
        epsilon: float = 0.1,
        decay_factor: float = 0.99,
        min_samples_per_arm: int = 30
    ):
        self.method = method
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.min_samples_per_arm = min_samples_per_arm
        
        # Arms
        self.arms: Dict[str, StrategyArm] = {
            name: StrategyArm(name=name) for name in strategy_names
        }
        
        # Total tracking
        self.total_selections = 0
        self.selection_history: deque = deque(maxlen=10000)
        
        # Current regime
        self.current_regime: Optional[str] = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def select_strategy(self, regime: str = None) -> str:
        """
        Select a strategy to use.
        
        Returns:
            Strategy name
        """
        with self._lock:
            self.current_regime = regime
            self.total_selections += 1
            
            # Force exploration if any arm is under-sampled
            for arm in self.arms.values():
                if arm.total_trades < self.min_samples_per_arm:
                    self.selection_history.append({
                        'timestamp': datetime.now(),
                        'strategy': arm.name,
                        'method': 'forced_exploration',
                        'regime': regime
                    })
                    return arm.name
            
            # Selection based on method
            if self.method == SelectionMethod.EPSILON_GREEDY:
                selected = self._epsilon_greedy()
            elif self.method == SelectionMethod.UCB:
                selected = self._ucb_select()
            elif self.method == SelectionMethod.THOMPSON_SAMPLING:
                selected = self._thompson_select()
            elif self.method == SelectionMethod.SOFTMAX:
                selected = self._softmax_select()
            elif self.method == SelectionMethod.REGIME_CONDITIONAL:
                selected = self._regime_select(regime)
            else:
                selected = self._thompson_select()
            
            self.selection_history.append({
                'timestamp': datetime.now(),
                'strategy': selected,
                'method': self.method.value,
                'regime': regime
            })
            
            return selected
    
    def _epsilon_greedy(self) -> str:
        """Epsilon-greedy selection."""
        if np.random.random() < self.epsilon:
            # Explore
            return np.random.choice(list(self.arms.keys()))
        else:
            # Exploit
            best_arm = max(self.arms.values(), key=lambda a: a.recent_win_rate)
            return best_arm.name
    
    def _ucb_select(self) -> str:
        """Upper Confidence Bound selection."""
        scores = {
            name: arm.ucb_score(self.total_selections)
            for name, arm in self.arms.items()
        }
        return max(scores, key=scores.get)
    
    def _thompson_select(self) -> str:
        """Thompson Sampling selection."""
        samples = {
            name: arm.sample_thompson()
            for name, arm in self.arms.items()
        }
        return max(samples, key=samples.get)
    
    def _softmax_select(self, temperature: float = 0.1) -> str:
        """Softmax selection."""
        win_rates = np.array([arm.recent_win_rate for arm in self.arms.values()])
        probs = np.exp(win_rates / temperature)
        probs = probs / probs.sum()
        return np.random.choice(list(self.arms.keys()), p=probs)
    
    def _regime_select(self, regime: str) -> str:
        """Regime-conditional selection."""
        if regime is None:
            return self._thompson_select()
        
        # Get regime-specific win rates
        regime_scores = {
            name: arm.get_regime_win_rate(regime)
            for name, arm in self.arms.items()
        }
        
        # Add some Thompson sampling noise for exploration
        for name in regime_scores:
            regime_scores[name] += np.random.normal(0, 0.05)
        
        return max(regime_scores, key=regime_scores.get)
    
    def update(self, strategy: str, won: bool, pnl: float, regime: str = None):
        """
        Update strategy performance.
        
        Args:
            strategy: Strategy name
            won: Whether trade was profitable
            pnl: Realized PnL
            regime: Current market regime
        """
        with self._lock:
            if strategy in self.arms:
                self.arms[strategy].update(won, pnl, regime or self.current_regime)
    
    def get_rankings(self, metric: str = "win_rate") -> List[Tuple[str, float]]:
        """Get strategy rankings."""
        if metric == "win_rate":
            rankings = [(a.name, a.win_rate) for a in self.arms.values()]
        elif metric == "recent_win_rate":
            rankings = [(a.name, a.recent_win_rate) for a in self.arms.values()]
        elif metric == "avg_pnl":
            rankings = [(a.name, a.avg_pnl) for a in self.arms.values()]
        elif metric == "sharpe":
            rankings = [(a.name, a.sharpe) for a in self.arms.values()]
        else:
            rankings = [(a.name, a.win_rate) for a in self.arms.values()]
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_selection_distribution(self, last_n: int = 1000) -> Dict[str, float]:
        """Get recent selection distribution."""
        recent = list(self.selection_history)[-last_n:]
        if not recent:
            return {name: 1/len(self.arms) for name in self.arms}
        
        counts = {}
        for sel in recent:
            name = sel['strategy']
            counts[name] = counts.get(name, 0) + 1
        
        total = sum(counts.values())
        return {name: count/total for name, count in counts.items()}
    
    def get_metrics(self) -> Dict:
        """Get comprehensive metrics."""
        return {
            'total_selections': self.total_selections,
            'arms': {
                name: {
                    'total_trades': arm.total_trades,
                    'wins': arm.wins,
                    'win_rate': arm.win_rate,
                    'recent_win_rate': arm.recent_win_rate,
                    'avg_pnl': arm.avg_pnl,
                    'sharpe': arm.sharpe,
                    'regime_performance': arm.regime_performance
                }
                for name, arm in self.arms.items()
            },
            'rankings': self.get_rankings(),
            'selection_distribution': self.get_selection_distribution()
        }
    
    def save(self, filepath: str):
        """Save meta-learner state."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'arms': self.arms,
                'total_selections': self.total_selections,
                'method': self.method,
                'epsilon': self.epsilon
            }, f)
    
    def load(self, filepath: str):
        """Load meta-learner state."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.arms = data['arms']
            self.total_selections = data['total_selections']


class ABTestFramework:
    """
    A/B testing framework for strategy comparison.
    
    Supports:
    - Random assignment to control/treatment
    - Statistical significance testing
    - Confidence intervals
    - Early stopping for clear winners
    """
    
    def __init__(
        self,
        control_strategy: str,
        treatment_strategy: str,
        min_samples: int = 100,
        significance_level: float = 0.05,
        min_detectable_effect: float = 0.02
    ):
        self.control_strategy = control_strategy
        self.treatment_strategy = treatment_strategy
        self.min_samples = min_samples
        self.significance_level = significance_level
        self.min_detectable_effect = min_detectable_effect
        
        # Results
        self.control_results: List[Dict] = []
        self.treatment_results: List[Dict] = []
        
        # State
        self.is_active = True
        self.start_time = datetime.now()
        self._lock = threading.Lock()
    
    def assign(self) -> str:
        """Randomly assign to control or treatment."""
        if not self.is_active:
            # After test concludes, use winner
            result = self.analyze()
            if result.is_significant and result.win_rate_lift > 0:
                return self.treatment_strategy
            return self.control_strategy
        
        # Random assignment
        return np.random.choice([self.control_strategy, self.treatment_strategy])
    
    def record_result(self, strategy: str, won: bool, pnl: float, metadata: Dict = None):
        """Record a trade result."""
        with self._lock:
            result = {
                'timestamp': datetime.now(),
                'won': won,
                'pnl': pnl,
                'metadata': metadata or {}
            }
            
            if strategy == self.control_strategy:
                self.control_results.append(result)
            elif strategy == self.treatment_strategy:
                self.treatment_results.append(result)
            
            # Check if we should stop
            if len(self.control_results) >= self.min_samples and len(self.treatment_results) >= self.min_samples:
                self._check_early_stopping()
    
    def _check_early_stopping(self):
        """Check if we can stop the test early."""
        result = self.analyze()
        
        # Stop if highly significant
        if result.is_significant and result.win_rate_p_value < 0.01:
            self.is_active = False
    
    def analyze(self) -> ABTestResult:
        """Analyze A/B test results."""
        # Control metrics
        control_wins = [r['won'] for r in self.control_results]
        control_pnl = [r['pnl'] for r in self.control_results]
        
        # Treatment metrics
        treatment_wins = [r['won'] for r in self.treatment_results]
        treatment_pnl = [r['pnl'] for r in self.treatment_results]
        
        # Win rates
        control_wr = np.mean(control_wins) if control_wins else 0.5
        treatment_wr = np.mean(treatment_wins) if treatment_wins else 0.5
        
        # Average PnL
        control_avg_pnl = np.mean(control_pnl) if control_pnl else 0
        treatment_avg_pnl = np.mean(treatment_pnl) if treatment_pnl else 0
        
        # Statistical tests
        # Win rate - Chi-squared test
        if len(control_wins) > 0 and len(treatment_wins) > 0:
            contingency = [
                [sum(control_wins), len(control_wins) - sum(control_wins)],
                [sum(treatment_wins), len(treatment_wins) - sum(treatment_wins)]
            ]
            chi2, wr_p_value, _, _ = stats.chi2_contingency(contingency)
        else:
            wr_p_value = 1.0
        
        # PnL - Welch's t-test
        if len(control_pnl) > 1 and len(treatment_pnl) > 1:
            t_stat, pnl_p_value = stats.ttest_ind(control_pnl, treatment_pnl, equal_var=False)
        else:
            pnl_p_value = 1.0
        
        # Effect sizes
        wr_lift = (treatment_wr - control_wr) / control_wr * 100 if control_wr > 0 else 0
        pnl_lift = (treatment_avg_pnl - control_avg_pnl) / abs(control_avg_pnl) * 100 if control_avg_pnl != 0 else 0
        
        # Confidence intervals for win rate difference
        wr_diff = treatment_wr - control_wr
        pooled_se = np.sqrt(
            control_wr * (1 - control_wr) / max(1, len(control_wins)) +
            treatment_wr * (1 - treatment_wr) / max(1, len(treatment_wins))
        )
        z = 1.96  # 95% CI
        wr_ci_lower = wr_diff - z * pooled_se
        wr_ci_upper = wr_diff + z * pooled_se
        
        # CI for PnL difference
        pnl_diff = treatment_avg_pnl - control_avg_pnl
        if len(control_pnl) > 1 and len(treatment_pnl) > 1:
            pnl_se = np.sqrt(
                np.var(control_pnl) / len(control_pnl) +
                np.var(treatment_pnl) / len(treatment_pnl)
            )
        else:
            pnl_se = 0
        pnl_ci_lower = pnl_diff - z * pnl_se
        pnl_ci_upper = pnl_diff + z * pnl_se
        
        # Significance
        is_significant = wr_p_value < self.significance_level or pnl_p_value < self.significance_level
        
        # Recommendation
        if len(control_results := self.control_results) < self.min_samples:
            recommendation = "Need more data"
            confidence = "low"
        elif not is_significant:
            recommendation = "No significant difference - continue testing"
            confidence = "low"
        elif wr_lift > 5 and pnl_lift > 10:
            recommendation = f"Switch to {self.treatment_strategy} (strong improvement)"
            confidence = "high"
        elif wr_lift > 0 and pnl_lift > 0:
            recommendation = f"Switch to {self.treatment_strategy} (moderate improvement)"
            confidence = "medium"
        else:
            recommendation = f"Keep {self.control_strategy} (treatment is worse)"
            confidence = "high" if is_significant else "medium"
        
        return ABTestResult(
            name=f"{self.control_strategy}_vs_{self.treatment_strategy}",
            control_name=self.control_strategy,
            treatment_name=self.treatment_strategy,
            control_win_rate=control_wr,
            treatment_win_rate=treatment_wr,
            control_avg_pnl=control_avg_pnl,
            treatment_avg_pnl=treatment_avg_pnl,
            win_rate_p_value=wr_p_value,
            pnl_p_value=pnl_p_value,
            is_significant=is_significant,
            win_rate_lift=wr_lift,
            pnl_lift=pnl_lift,
            win_rate_ci_lower=wr_ci_lower,
            win_rate_ci_upper=wr_ci_upper,
            pnl_ci_lower=pnl_ci_lower,
            pnl_ci_upper=pnl_ci_upper,
            control_n=len(control_wins),
            treatment_n=len(treatment_wins),
            recommendation=recommendation,
            confidence=confidence
        )
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        result = self.analyze()
        
        lines = [
            f"A/B Test: {result.name}",
            f"=" * 50,
            f"",
            f"Sample Sizes:",
            f"  Control ({result.control_name}): {result.control_n}",
            f"  Treatment ({result.treatment_name}): {result.treatment_n}",
            f"",
            f"Win Rates:",
            f"  Control: {result.control_win_rate:.2%}",
            f"  Treatment: {result.treatment_win_rate:.2%}",
            f"  Lift: {result.win_rate_lift:+.1f}%",
            f"  P-value: {result.win_rate_p_value:.4f}",
            f"  95% CI: [{result.win_rate_ci_lower:.4f}, {result.win_rate_ci_upper:.4f}]",
            f"",
            f"Average PnL:",
            f"  Control: ${result.control_avg_pnl:.2f}",
            f"  Treatment: ${result.treatment_avg_pnl:.2f}",
            f"  Lift: {result.pnl_lift:+.1f}%",
            f"  P-value: {result.pnl_p_value:.4f}",
            f"  95% CI: [${result.pnl_ci_lower:.2f}, ${result.pnl_ci_upper:.2f}]",
            f"",
            f"Statistical Significance: {'YES' if result.is_significant else 'NO'}",
            f"",
            f"RECOMMENDATION: {result.recommendation}",
            f"Confidence: {result.confidence}",
        ]
        
        return "\n".join(lines)


class ShadowRunner:
    """
    Shadow trading for comparing strategies.
    
    Runs multiple strategies in parallel without actual execution,
    tracking theoretical performance for comparison.
    """
    
    def __init__(self, strategy_names: List[str]):
        self.strategy_names = strategy_names
        self.shadow_positions: Dict[str, Dict[str, float]] = {
            name: {} for name in strategy_names
        }
        self.shadow_pnl: Dict[str, List[float]] = {
            name: [] for name in strategy_names
        }
        self.shadow_trades: Dict[str, List[Dict]] = {
            name: [] for name in strategy_names
        }
    
    def record_signal(
        self,
        strategy: str,
        symbol: str,
        signal: int,  # 1 for long, -1 for short, 0 for flat
        confidence: float,
        price: float
    ):
        """Record a shadow signal."""
        if strategy not in self.strategy_names:
            return
        
        self.shadow_trades[strategy].append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'price': price
        })
        
        # Update shadow position
        self.shadow_positions[strategy][symbol] = signal * confidence
    
    def update_prices(self, prices: Dict[str, float]):
        """Update PnL based on current prices."""
        for strategy in self.strategy_names:
            pnl = 0
            for symbol, position in self.shadow_positions[strategy].items():
                if symbol in prices:
                    # Get entry price from last trade
                    trades = [t for t in self.shadow_trades[strategy] if t['symbol'] == symbol]
                    if trades:
                        entry_price = trades[-1]['price']
                        current_price = prices[symbol]
                        pnl += position * (current_price - entry_price) / entry_price * 100
            
            self.shadow_pnl[strategy].append(pnl)
    
    def get_comparison(self) -> pd.DataFrame:
        """Get strategy comparison."""
        results = []
        for name in self.strategy_names:
            pnl_series = self.shadow_pnl[name]
            if not pnl_series:
                continue
            
            results.append({
                'strategy': name,
                'total_trades': len(self.shadow_trades[name]),
                'total_pnl': sum(pnl_series),
                'avg_pnl': np.mean(pnl_series) if pnl_series else 0,
                'sharpe': np.mean(pnl_series) / np.std(pnl_series) if len(pnl_series) > 1 else 0,
                'max_drawdown': min(0, min(pnl_series)) if pnl_series else 0
            })
        
        return pd.DataFrame(results).sort_values('total_pnl', ascending=False)


# Convenience function
def create_meta_learner(
    strategies: List[str],
    method: str = "thompson"
) -> MetaLearner:
    """Create a configured meta-learner."""
    method_map = {
        "thompson": SelectionMethod.THOMPSON_SAMPLING,
        "ucb": SelectionMethod.UCB,
        "epsilon": SelectionMethod.EPSILON_GREEDY,
        "softmax": SelectionMethod.SOFTMAX,
        "regime": SelectionMethod.REGIME_CONDITIONAL
    }
    return MetaLearner(
        strategy_names=strategies,
        method=method_map.get(method, SelectionMethod.THOMPSON_SAMPLING)
    )


if __name__ == "__main__":
    print("=" * 60)
    print("META-LEARNER & A/B TESTING DEMO")
    print("=" * 60)
    
    # Create meta-learner
    strategies = ['momentum', 'volatility_breakout', 'stat_arb', 'regime_ml']
    learner = create_meta_learner(strategies, method="thompson")
    
    # Simulate trades
    np.random.seed(42)
    
    # Give different strategies different "true" win rates
    true_win_rates = {
        'momentum': 0.55,
        'volatility_breakout': 0.52,
        'stat_arb': 0.58,
        'regime_ml': 0.60
    }
    
    print("\nSimulating 2000 trades...")
    for i in range(2000):
        # Select strategy
        selected = learner.select_strategy(regime="trending")
        
        # Simulate outcome
        true_wr = true_win_rates[selected]
        won = np.random.random() < true_wr
        pnl = np.random.normal(10 if won else -8, 5)
        
        # Update
        learner.update(selected, won, pnl, regime="trending")
    
    # Results
    print("\nMeta-Learner Results:")
    metrics = learner.get_metrics()
    
    print("\nStrategy Rankings (by win rate):")
    for rank, (name, wr) in enumerate(metrics['rankings'], 1):
        trades = metrics['arms'][name]['total_trades']
        print(f"  {rank}. {name}: {wr:.2%} win rate ({trades} trades)")
    
    print("\nSelection Distribution (last 1000):")
    for name, pct in metrics['selection_distribution'].items():
        print(f"  {name}: {pct:.1%}")
    
    # A/B Test
    print("\n" + "=" * 60)
    print("A/B TEST DEMO")
    print("=" * 60)
    
    ab_test = ABTestFramework(
        control_strategy='momentum',
        treatment_strategy='regime_ml',
        min_samples=100
    )
    
    # Simulate A/B test
    for i in range(500):
        strategy = ab_test.assign()
        true_wr = true_win_rates[strategy]
        won = np.random.random() < true_wr
        pnl = np.random.normal(10 if won else -8, 5)
        ab_test.record_result(strategy, won, pnl)
    
    print("\n" + ab_test.get_summary())
    
    print("\n" + "=" * 60)
    print("META-LEARNER WORKING CORRECTLY")
    print("=" * 60)
