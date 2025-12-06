"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         CAML STATE MANAGER - Strategy Performance Tracking                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Manages strategy performance tracking, regime transitions, and state persistence.
"""

import logging
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

logger = logging.getLogger("CAML.StateManager")


@dataclass
class StrategyPerformance:
    """Performance tracking for a single strategy."""
    name: str
    
    # Returns tracking
    returns_1h: deque = field(default_factory=lambda: deque(maxlen=24))
    returns_24h: deque = field(default_factory=lambda: deque(maxlen=288))
    returns_7d: deque = field(default_factory=lambda: deque(maxlen=2016))
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    
    # Timing
    last_trade_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    
    # Regime-specific
    regime_returns: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_return(self, ret: float, regime: Optional[str] = None) -> None:
        """Add a return observation."""
        self.returns_1h.append(ret)
        self.returns_24h.append(ret)
        self.returns_7d.append(ret)
        self.total_pnl += ret
        self.last_update_time = datetime.now()
        
        if regime:
            if regime not in self.regime_returns:
                self.regime_returns[regime] = []
            self.regime_returns[regime].append(ret)
    
    def add_trade(self, pnl: float) -> None:
        """Record a trade."""
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        self.last_trade_time = datetime.now()
    
    def get_sharpe(self, period: str = '24h') -> float:
        """Calculate Sharpe ratio for period."""
        if period == '1h':
            returns = list(self.returns_1h)
        elif period == '24h':
            returns = list(self.returns_24h)
        elif period == '7d':
            returns = list(self.returns_7d)
        else:
            returns = list(self.returns_24h)
        
        if len(returns) < 2:
            return 0.0
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-10:
            return 0.0
        
        # Annualize (assuming 5-minute bars)
        periods_per_year = 365 * 24 * 12
        return (mean_ret / std_ret) * np.sqrt(periods_per_year)
    
    def get_win_rate(self) -> float:
        """Get win rate."""
        if self.total_trades == 0:
            return 0.5
        return self.winning_trades / self.total_trades
    
    def get_regime_sharpe(self, regime: str) -> float:
        """Get Sharpe for a specific regime."""
        returns = self.regime_returns.get(regime, [])
        if len(returns) < 2:
            return 0.0
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-10:
            return 0.0
        
        return (mean_ret / std_ret) * np.sqrt(365 * 24 * 12)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': self.total_pnl,
            'sharpe_1h': self.get_sharpe('1h'),
            'sharpe_24h': self.get_sharpe('24h'),
            'sharpe_7d': self.get_sharpe('7d'),
            'win_rate': self.get_win_rate(),
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
        }


class StateManager:
    """
    Manages CAML state including strategy performance and regime transitions.
    
    Features:
    - Per-strategy performance tracking
    - Regime transition detection
    - State persistence
    - Performance aggregation
    """
    
    def __init__(
        self,
        strategies: List[str],
        n_regimes: int = 3,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize StateManager.
        
        Args:
            strategies: List of strategy names
            n_regimes: Number of market regimes
            persist_path: Path for state persistence
        """
        self.strategies = strategies
        self.n_regimes = n_regimes
        self.persist_path = persist_path
        
        # Strategy performance
        self.performance: Dict[str, StrategyPerformance] = {
            name: StrategyPerformance(name=name)
            for name in strategies
        }
        
        # Regime tracking
        self.current_regime: int = 0
        self.regime_history: deque = deque(maxlen=1000)
        self.regime_transition_times: List[datetime] = []
        
        # Aggregate metrics
        self.portfolio_returns: deque = deque(maxlen=2016)  # ~7 days
        self.allocation_history: deque = deque(maxlen=1000)
        
        # Load persisted state if available
        if persist_path and Path(persist_path).exists():
            self.load()
    
    def update_performance(
        self,
        strategy: str,
        return_value: float,
        regime: Optional[int] = None,
    ) -> None:
        """
        Update strategy performance.
        
        Args:
            strategy: Strategy name
            return_value: Return value to add
            regime: Current market regime (optional)
        """
        if strategy not in self.performance:
            logger.warning(f"Unknown strategy: {strategy}")
            return
        
        regime_str = f"regime_{regime}" if regime is not None else None
        self.performance[strategy].add_return(return_value, regime_str)
    
    def record_trade(
        self,
        strategy: str,
        pnl: float,
    ) -> None:
        """
        Record a trade for a strategy.
        
        Args:
            strategy: Strategy name
            pnl: Trade P&L
        """
        if strategy in self.performance:
            self.performance[strategy].add_trade(pnl)
    
    def update_regime(self, new_regime: int) -> bool:
        """
        Update current regime.
        
        Args:
            new_regime: New regime index
            
        Returns:
            bool: True if regime changed
        """
        changed = new_regime != self.current_regime
        
        if changed:
            self.regime_transition_times.append(datetime.now())
            logger.info(f"Regime transition: {self.current_regime} -> {new_regime}")
        
        self.current_regime = new_regime
        self.regime_history.append((datetime.now(), new_regime))
        
        return changed
    
    def record_allocation(self, allocation: Dict[str, float]) -> None:
        """Record an allocation decision."""
        self.allocation_history.append({
            'timestamp': datetime.now().isoformat(),
            'allocation': allocation.copy(),
        })
    
    def get_strategy_rankings(self, metric: str = 'sharpe_24h') -> List[Tuple[str, float]]:
        """
        Get strategies ranked by a metric.
        
        Args:
            metric: 'sharpe_1h', 'sharpe_24h', 'sharpe_7d', 'win_rate', 'pnl'
            
        Returns:
            List of (strategy_name, metric_value) sorted descending
        """
        rankings = []
        
        for name, perf in self.performance.items():
            if metric == 'sharpe_1h':
                value = perf.get_sharpe('1h')
            elif metric == 'sharpe_24h':
                value = perf.get_sharpe('24h')
            elif metric == 'sharpe_7d':
                value = perf.get_sharpe('7d')
            elif metric == 'win_rate':
                value = perf.get_win_rate()
            elif metric == 'pnl':
                value = perf.total_pnl
            else:
                value = 0.0
            
            rankings.append((name, value))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_regime_performance(self, regime: int) -> Dict[str, float]:
        """Get strategy Sharpes for a specific regime."""
        regime_str = f"regime_{regime}"
        return {
            name: perf.get_regime_sharpe(regime_str)
            for name, perf in self.performance.items()
        }
    
    def get_aggregate_stats(self) -> Dict:
        """Get aggregate portfolio statistics."""
        all_returns = list(self.portfolio_returns)
        
        if len(all_returns) < 2:
            return {
                'portfolio_sharpe': 0.0,
                'portfolio_return': 0.0,
                'portfolio_volatility': 0.0,
                'total_strategies': len(self.strategies),
                'regime_transitions_7d': 0,
                'current_regime': self.current_regime,
            }
        
        mean_ret = np.mean(all_returns)
        std_ret = np.std(all_returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(365 * 24 * 12) if std_ret > 0 else 0
        
        # Count recent regime transitions
        week_ago = datetime.now() - timedelta(days=7)
        recent_transitions = sum(
            1 for t in self.regime_transition_times
            if t >= week_ago
        )
        
        return {
            'portfolio_sharpe': sharpe,
            'portfolio_return': sum(all_returns),
            'portfolio_volatility': std_ret,
            'total_strategies': len(self.strategies),
            'regime_transitions_7d': recent_transitions,
            'current_regime': self.current_regime,
        }
    
    def get_full_status(self) -> Dict:
        """Get complete state manager status."""
        return {
            'aggregate': self.get_aggregate_stats(),
            'strategies': {
                name: perf.to_dict()
                for name, perf in self.performance.items()
            },
            'rankings': {
                'by_sharpe_24h': self.get_strategy_rankings('sharpe_24h'),
                'by_pnl': self.get_strategy_rankings('pnl'),
            },
            'regime_info': {
                'current': self.current_regime,
                'history_length': len(self.regime_history),
                'transition_count': len(self.regime_transition_times),
            },
        }
    
    def save(self) -> None:
        """Save state to disk."""
        if not self.persist_path:
            return
        
        state = {
            'strategies': self.strategies,
            'n_regimes': self.n_regimes,
            'current_regime': self.current_regime,
            'performance': {
                name: {
                    'total_trades': perf.total_trades,
                    'winning_trades': perf.winning_trades,
                    'total_pnl': perf.total_pnl,
                    'regime_returns': perf.regime_returns,
                }
                for name, perf in self.performance.items()
            },
            'regime_transition_times': [
                t.isoformat() for t in self.regime_transition_times[-100:]
            ],
            'timestamp': datetime.now().isoformat(),
        }
        
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {self.persist_path}")
    
    def load(self) -> bool:
        """Load state from disk."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return False
        
        try:
            with open(self.persist_path, 'r') as f:
                state = json.load(f)
            
            self.current_regime = state.get('current_regime', 0)
            
            for name, data in state.get('performance', {}).items():
                if name in self.performance:
                    perf = self.performance[name]
                    perf.total_trades = data.get('total_trades', 0)
                    perf.winning_trades = data.get('winning_trades', 0)
                    perf.total_pnl = data.get('total_pnl', 0.0)
                    perf.regime_returns = data.get('regime_returns', {})
            
            self.regime_transition_times = [
                datetime.fromisoformat(t)
                for t in state.get('regime_transition_times', [])
            ]
            
            logger.info(f"State loaded from {self.persist_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create state manager
    strategies = ['momentum', 'stat_arb', 'funding', 'microstructure']
    manager = StateManager(strategies, n_regimes=3)
    
    # Simulate performance updates
    import random
    
    for i in range(500):
        # Random strategy
        strategy = random.choice(strategies)
        
        # Random return
        ret = random.gauss(0.0001, 0.001)
        
        # Update
        manager.update_performance(strategy, ret, regime=manager.current_regime)
        
        # Occasional trade
        if random.random() < 0.1:
            pnl = random.gauss(0, 0.01)
            manager.record_trade(strategy, pnl)
        
        # Occasional regime change
        if random.random() < 0.02:
            new_regime = random.randint(0, 2)
            manager.update_regime(new_regime)
    
    # Print status
    print("\n" + "=" * 60)
    print("STATE MANAGER STATUS")
    print("=" * 60)
    
    status = manager.get_full_status()
    
    print("\nAggregate Stats:")
    for key, value in status['aggregate'].items():
        print(f"  {key}: {value}")
    
    print("\nStrategy Performance:")
    for name, data in status['strategies'].items():
        print(f"  {name}:")
        print(f"    Sharpe (24h): {data['sharpe_24h']:.3f}")
        print(f"    Win Rate: {data['win_rate']:.1%}")
        print(f"    Total PnL: {data['total_pnl']:.4f}")
    
    print("\nRankings by Sharpe (24h):")
    for name, value in status['rankings']['by_sharpe_24h']:
        print(f"  {name}: {value:.3f}")
