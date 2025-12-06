"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         CAPITAL ALLOCATION META-LEARNER (CAML) - Allocator                    ║
║                                                                               ║
║  Thompson Sampling with Gaussian Process for capacity-aware allocation        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CAML optimizes capital allocation across strategies using:
1. Constrained Thompson Sampling with GP prior
2. After-cost Sharpe maximization
3. Capacity constraints
4. Turnover penalties

State Vector:
    s_t = (regime_posterior, realized_slippage_1h, turnover_1h, 
           open_pnl, margin_usage, liq_distance)

Action:
    a_t ∈ [0,1]^N  (fractional weight per strategy)  Σa_t = 1

Reward:
    r_t = Δ(post-cost Sharpe) – 5×Δ(drawdown) – λ|a_t – a_{t-1}|_1
"""

import time
import threading
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json

logger = logging.getLogger("CAML")


@dataclass
class CAMLConfig:
    """Configuration for CAML."""
    
    # Update frequency
    update_interval_minutes: int = 30
    
    # Replay buffer
    replay_buffer_size: int = 500
    min_samples_for_update: int = 50
    
    # Thompson Sampling parameters
    prior_alpha: float = 1.0  # Beta prior alpha
    prior_beta: float = 1.0   # Beta prior beta
    
    # Reward weights
    sharpe_weight: float = 1.0
    drawdown_weight: float = 5.0
    turnover_penalty: float = 0.1
    
    # Constraints
    max_weight_multiplier: float = 3.0  # Max weight = multiplier × backtest max
    min_weight: float = 0.0
    max_weight: float = 0.5  # No single strategy > 50%
    
    # Auto-archive
    zero_weight_days_to_archive: int = 3
    
    # Exploration
    exploration_bonus: float = 0.1


@dataclass
class AllocationState:
    """Current state for allocation decision."""
    regime_posterior: np.ndarray  # Probability distribution over regimes
    realized_slippage_1h: float   # Realized slippage in last hour (bps)
    turnover_1h: float            # Portfolio turnover in last hour
    open_pnl: float               # Current open P&L
    margin_usage: float           # Current margin utilization (0-1)
    liq_distance: float           # Distance to liquidation (ATR units)
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.concatenate([
            self.regime_posterior,
            np.array([
                self.realized_slippage_1h,
                self.turnover_1h,
                self.open_pnl,
                self.margin_usage,
                self.liq_distance,
            ])
        ])


@dataclass
class StrategyArm:
    """Thompson Sampling arm for a strategy."""
    name: str
    alpha: float = 1.0  # Beta distribution alpha (successes + prior)
    beta: float = 1.0   # Beta distribution beta (failures + prior)
    
    # Performance tracking
    total_returns: float = 0.0
    total_trades: int = 0
    sharpe_estimate: float = 0.0
    
    # Regime-specific performance
    regime_performance: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Backtest reference
    backtest_max_weight: float = 0.25
    backtest_sharpe: float = 1.0
    
    # Auto-archive tracking
    zero_weight_start: Optional[datetime] = None
    is_archived: bool = False
    
    def sample(self) -> float:
        """Sample from posterior (Thompson Sampling)."""
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, reward: float) -> None:
        """Update posterior with observed reward."""
        # Map reward to [0,1] for Beta distribution
        success_prob = 1.0 / (1.0 + np.exp(-reward))  # Sigmoid
        
        # Update counts
        self.alpha += success_prob
        self.beta += (1 - success_prob)
        
        self.total_returns += reward
        self.total_trades += 1
    
    def get_mean(self) -> float:
        """Get posterior mean."""
        return self.alpha / (self.alpha + self.beta)
    
    def get_uncertainty(self) -> float:
        """Get posterior uncertainty (std)."""
        a, b = self.alpha, self.beta
        return np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))


@dataclass
class ReplayExperience:
    """Single experience for replay buffer."""
    state: AllocationState
    action: np.ndarray  # Allocation weights
    reward: float
    next_state: AllocationState
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'state': self.state.to_vector().tolist(),
            'action': self.action.tolist(),
            'reward': float(self.reward),
            'timestamp': self.timestamp.isoformat(),
        }


class CapitalAllocationMetaLearner:
    """
    Capital Allocation Meta-Learner (CAML)
    
    Uses Thompson Sampling with capacity constraints to allocate
    capital across strategies optimally.
    
    Usage:
        caml = CapitalAllocationMetaLearner(strategies=['momentum', 'stat_arb', 'funding'])
        
        # Get allocation
        state = AllocationState(...)
        weights = caml.get_allocation(state)
        
        # Update with observed reward
        caml.update(state, weights, reward, next_state)
    """
    
    def __init__(
        self,
        strategies: List[str],
        config: Optional[CAMLConfig] = None,
        backtest_weights: Optional[Dict[str, float]] = None,
        backtest_sharpes: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize CAML.
        
        Args:
            strategies: List of strategy names
            config: CAML configuration
            backtest_weights: Maximum weights from backtest per strategy
            backtest_sharpes: Sharpe ratios from backtest per strategy
        """
        self.strategies = strategies
        self.config = config or CAMLConfig()
        self.n_strategies = len(strategies)
        
        # Initialize arms
        self.arms: Dict[str, StrategyArm] = {}
        for name in strategies:
            arm = StrategyArm(
                name=name,
                alpha=self.config.prior_alpha,
                beta=self.config.prior_beta,
            )
            if backtest_weights:
                arm.backtest_max_weight = backtest_weights.get(name, 0.25)
            if backtest_sharpes:
                arm.backtest_sharpe = backtest_sharpes.get(name, 1.0)
            self.arms[name] = arm
        
        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=self.config.replay_buffer_size)
        
        # Current allocation
        self.current_allocation: Dict[str, float] = {
            name: 1.0 / self.n_strategies for name in strategies
        }
        self.last_allocation_time: Optional[datetime] = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'updates': 0,
            'allocations': 0,
            'archived_strategies': 0,
        }
        
        logger.info(f"CAML initialized with {self.n_strategies} strategies")
    
    def get_allocation(
        self,
        state: AllocationState,
        use_thompson: bool = True,
    ) -> Dict[str, float]:
        """
        Get optimal allocation for current state.
        
        Args:
            state: Current allocation state
            use_thompson: Whether to use Thompson Sampling (vs greedy)
            
        Returns:
            Dict mapping strategy name to weight
        """
        with self._lock:
            self._stats['allocations'] += 1
            
            # Get regime (highest posterior)
            regime_idx = np.argmax(state.regime_posterior)
            regime_name = f"regime_{regime_idx}"
            
            # Sample from each arm
            samples = {}
            for name, arm in self.arms.items():
                if arm.is_archived:
                    samples[name] = 0.0
                    continue
                
                if use_thompson:
                    # Thompson Sampling with exploration bonus
                    base_sample = arm.sample()
                    uncertainty_bonus = self.config.exploration_bonus * arm.get_uncertainty()
                    samples[name] = base_sample + uncertainty_bonus
                else:
                    # Greedy
                    samples[name] = arm.get_mean()
                
                # Adjust for regime-specific performance
                if regime_name in arm.regime_performance:
                    regime_mean, _ = arm.regime_performance[regime_name]
                    samples[name] *= (1 + regime_mean)
            
            # Normalize to sum to 1
            total = sum(samples.values())
            if total > 0:
                weights = {name: s / total for name, s in samples.items()}
            else:
                # Equal weights fallback
                active = [n for n, a in self.arms.items() if not a.is_archived]
                weights = {name: 1.0 / len(active) if name in active else 0.0 
                          for name in self.strategies}
            
            # Apply constraints
            weights = self._apply_constraints(weights)
            
            # Update current allocation
            self.current_allocation = weights
            self.last_allocation_time = datetime.now()
            
            return weights
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply allocation constraints."""
        constrained = {}
        
        for name, weight in weights.items():
            arm = self.arms[name]
            
            # Min/max weight
            weight = max(self.config.min_weight, weight)
            weight = min(self.config.max_weight, weight)
            
            # Backtest maximum constraint
            max_allowed = arm.backtest_max_weight * self.config.max_weight_multiplier
            weight = min(weight, max_allowed)
            
            constrained[name] = weight
        
        # Re-normalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {n: w / total for n, w in constrained.items()}
        
        return constrained
    
    def update(
        self,
        state: AllocationState,
        action: Dict[str, float],
        reward: float,
        next_state: AllocationState,
    ) -> None:
        """
        Update CAML with observed experience.
        
        Args:
            state: State when allocation was made
            action: Allocation weights used
            reward: Observed reward (post-cost Sharpe delta, etc.)
            next_state: State after allocation period
        """
        with self._lock:
            self._stats['updates'] += 1
            
            # Add to replay buffer
            experience = ReplayExperience(
                state=state,
                action=np.array([action.get(n, 0.0) for n in self.strategies]),
                reward=reward,
                next_state=next_state,
                timestamp=datetime.now(),
            )
            self.replay_buffer.append(experience)
            
            # Calculate per-strategy reward attribution
            strategy_rewards = self._attribute_rewards(action, reward)
            
            # Update each arm
            for name, arm in self.arms.items():
                if name in strategy_rewards:
                    arm.update(strategy_rewards[name])
            
            # Check for archiving
            self._check_archive_conditions()
            
            # Periodic batch update from replay buffer
            if len(self.replay_buffer) >= self.config.min_samples_for_update:
                if self._stats['updates'] % 10 == 0:
                    self._batch_update_from_replay()
    
    def _attribute_rewards(
        self,
        action: Dict[str, float],
        total_reward: float,
    ) -> Dict[str, float]:
        """
        Attribute total reward to individual strategies.
        
        Uses weight-proportional attribution with Shapley-inspired adjustment.
        """
        # Simple weight-proportional for now
        # In production, use actual strategy-level P&L
        rewards = {}
        for name, weight in action.items():
            if weight > 0:
                rewards[name] = total_reward * weight
        return rewards
    
    def _check_archive_conditions(self) -> None:
        """Check if any strategies should be archived."""
        for name, arm in self.arms.items():
            if arm.is_archived:
                continue
            
            current_weight = self.current_allocation.get(name, 0.0)
            
            if current_weight < 0.01:  # Effectively zero
                if arm.zero_weight_start is None:
                    arm.zero_weight_start = datetime.now()
                else:
                    days_at_zero = (datetime.now() - arm.zero_weight_start).days
                    if days_at_zero >= self.config.zero_weight_days_to_archive:
                        arm.is_archived = True
                        self._stats['archived_strategies'] += 1
                        logger.warning(
                            f"Strategy '{name}' archived after {days_at_zero} days at zero weight"
                        )
            else:
                arm.zero_weight_start = None
    
    def _batch_update_from_replay(self) -> None:
        """Batch update from replay buffer."""
        if len(self.replay_buffer) < self.config.min_samples_for_update:
            return
        
        # Sample from replay buffer
        sample_size = min(50, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), sample_size, replace=False)
        
        for idx in indices:
            exp = self.replay_buffer[idx]
            
            # Calculate advantage-weighted update
            for i, name in enumerate(self.strategies):
                weight = exp.action[i]
                if weight > 0:
                    strategy_reward = exp.reward * weight
                    self.arms[name].update(strategy_reward * 0.1)  # Smaller updates for replay
    
    def calculate_reward(
        self,
        sharpe_delta: float,
        drawdown_delta: float,
        turnover: float,
    ) -> float:
        """
        Calculate CAML reward from components.
        
        reward = Δ(post-cost Sharpe) – 5×Δ(drawdown) – λ×turnover
        """
        reward = (
            self.config.sharpe_weight * sharpe_delta
            - self.config.drawdown_weight * drawdown_delta
            - self.config.turnover_penalty * turnover
        )
        return reward
    
    def unarchive_strategy(self, name: str) -> bool:
        """
        Unarchive a strategy (trigger retrain pipeline).
        
        Args:
            name: Strategy name
            
        Returns:
            bool: True if unarchived
        """
        if name not in self.arms:
            return False
        
        arm = self.arms[name]
        if not arm.is_archived:
            return False
        
        # Reset arm
        arm.is_archived = False
        arm.alpha = self.config.prior_alpha
        arm.beta = self.config.prior_beta
        arm.zero_weight_start = None
        
        logger.info(f"Strategy '{name}' unarchived and reset")
        return True
    
    def get_status(self) -> Dict:
        """Get current CAML status."""
        with self._lock:
            arms_status = {}
            for name, arm in self.arms.items():
                arms_status[name] = {
                    'mean': arm.get_mean(),
                    'uncertainty': arm.get_uncertainty(),
                    'alpha': arm.alpha,
                    'beta': arm.beta,
                    'total_trades': arm.total_trades,
                    'is_archived': arm.is_archived,
                    'current_weight': self.current_allocation.get(name, 0.0),
                }
            
            return {
                'current_allocation': self.current_allocation.copy(),
                'arms': arms_status,
                'stats': self._stats.copy(),
                'replay_buffer_size': len(self.replay_buffer),
            }
    
    def save_state(self, filepath: str) -> None:
        """Save CAML state to file."""
        state = {
            'strategies': self.strategies,
            'arms': {
                name: {
                    'alpha': arm.alpha,
                    'beta': arm.beta,
                    'total_returns': arm.total_returns,
                    'total_trades': arm.total_trades,
                    'is_archived': arm.is_archived,
                    'backtest_max_weight': arm.backtest_max_weight,
                    'backtest_sharpe': arm.backtest_sharpe,
                }
                for name, arm in self.arms.items()
            },
            'current_allocation': self.current_allocation,
            'stats': self._stats,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"CAML state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load CAML state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        for name, data in state.get('arms', {}).items():
            if name in self.arms:
                arm = self.arms[name]
                arm.alpha = data['alpha']
                arm.beta = data['beta']
                arm.total_returns = data['total_returns']
                arm.total_trades = data['total_trades']
                arm.is_archived = data['is_archived']
                arm.backtest_max_weight = data['backtest_max_weight']
                arm.backtest_sharpe = data['backtest_sharpe']
        
        self.current_allocation = state.get('current_allocation', self.current_allocation)
        self._stats = state.get('stats', self._stats)
        
        logger.info(f"CAML state loaded from {filepath}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create CAML with 4 strategies
    strategies = ['momentum', 'stat_arb', 'funding', 'microstructure']
    
    caml = CapitalAllocationMetaLearner(
        strategies=strategies,
        backtest_weights={'momentum': 0.3, 'stat_arb': 0.25, 'funding': 0.3, 'microstructure': 0.2},
        backtest_sharpes={'momentum': 1.2, 'stat_arb': 1.5, 'funding': 2.0, 'microstructure': 0.8},
    )
    
    print("Initial status:")
    print(json.dumps(caml.get_status(), indent=2))
    
    # Simulate allocation decisions
    for i in range(100):
        # Create state
        state = AllocationState(
            regime_posterior=np.array([0.6, 0.3, 0.1]),  # 3 regimes
            realized_slippage_1h=np.random.uniform(0.5, 2.0),
            turnover_1h=np.random.uniform(0.1, 0.5),
            open_pnl=np.random.uniform(-0.02, 0.02),
            margin_usage=np.random.uniform(0.2, 0.5),
            liq_distance=np.random.uniform(2, 10),
        )
        
        # Get allocation
        weights = caml.get_allocation(state)
        
        # Simulate reward
        sharpe_delta = np.random.normal(0.01, 0.05)
        drawdown_delta = np.random.uniform(0, 0.01)
        turnover = sum(abs(weights.get(s, 0) - 0.25) for s in strategies)
        reward = caml.calculate_reward(sharpe_delta, drawdown_delta, turnover)
        
        # Create next state
        next_state = AllocationState(
            regime_posterior=np.array([0.5, 0.4, 0.1]),
            realized_slippage_1h=np.random.uniform(0.5, 2.0),
            turnover_1h=np.random.uniform(0.1, 0.5),
            open_pnl=np.random.uniform(-0.02, 0.02),
            margin_usage=np.random.uniform(0.2, 0.5),
            liq_distance=np.random.uniform(2, 10),
        )
        
        # Update
        caml.update(state, weights, reward, next_state)
        
        if (i + 1) % 20 == 0:
            print(f"\nAfter {i + 1} iterations:")
            for s, w in weights.items():
                arm = caml.arms[s]
                print(f"  {s}: weight={w:.3f}, mean={arm.get_mean():.3f}, trades={arm.total_trades}")
    
    print("\nFinal status:")
    status = caml.get_status()
    print(f"Total allocations: {status['stats']['allocations']}")
    print(f"Replay buffer size: {status['replay_buffer_size']}")
    
    # Save state
    caml.save_state("caml_state.json")
    print("\nState saved")
