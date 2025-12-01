"""
Strategy Registry - Central registry for all trading strategies

Provides:
- Strategy discovery and registration
- Performance tracking and ranking
- Dynamic strategy selection based on market regime
"""

from typing import Dict, List, Optional, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class StrategyPerformance:
    """Track strategy performance metrics"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Regime-specific performance
    regime_performance: Dict[MarketRegime, float] = field(default_factory=dict)
    
    def update_metrics(self, trades: List[Dict]) -> None:
        """Update metrics from trade history"""
        if not trades:
            return
        
        self.total_trades = len(trades)
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        self.total_pnl = sum(t.get('pnl', 0) for t in trades)
        
        if wins:
            self.avg_win = sum(t['pnl'] for t in wins) / len(wins)
        if losses:
            self.avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses))
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Expectancy
        self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)
        
        self.last_updated = datetime.utcnow()


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    strategy_class: Type
    enabled: bool = True
    weight: float = 1.0
    max_positions: int = 3
    preferred_regimes: List[MarketRegime] = field(default_factory=list)
    min_capital: float = 1000.0
    risk_per_trade: float = 0.02


class StrategyRegistry:
    """
    Central registry for all trading strategies
    
    Tracks performance, selects optimal strategies based on
    market conditions, and manages strategy lifecycle.
    """
    
    def __init__(self):
        self._strategies: Dict[str, StrategyConfig] = {}
        self._performance: Dict[str, StrategyPerformance] = {}
        self._current_regime: MarketRegime = MarketRegime.RANGING
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []
    
    def register(
        self,
        name: str,
        strategy_class: Type,
        weight: float = 1.0,
        preferred_regimes: List[MarketRegime] = None,
        **kwargs
    ) -> None:
        """Register a strategy"""
        config = StrategyConfig(
            name=name,
            strategy_class=strategy_class,
            weight=weight,
            preferred_regimes=preferred_regimes or [],
            **kwargs
        )
        self._strategies[name] = config
        self._performance[name] = StrategyPerformance(strategy_name=name)
    
    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """Get strategy by name"""
        return self._strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, StrategyConfig]:
        """Get all registered strategies"""
        return self._strategies.copy()
    
    def get_enabled_strategies(self) -> List[StrategyConfig]:
        """Get all enabled strategies"""
        return [s for s in self._strategies.values() if s.enabled]
    
    def get_optimal_strategies(
        self,
        regime: MarketRegime = None,
        top_n: int = 3
    ) -> List[StrategyConfig]:
        """
        Get optimal strategies for current market regime
        
        Ranks by:
        1. Regime suitability
        2. Recent performance
        3. Risk-adjusted returns
        """
        regime = regime or self._current_regime
        
        scored_strategies = []
        for name, config in self._strategies.items():
            if not config.enabled:
                continue
            
            score = 0.0
            perf = self._performance.get(name)
            
            # Regime suitability (0-30 points)
            if regime in config.preferred_regimes:
                score += 30
            elif not config.preferred_regimes:
                score += 15  # Neutral
            
            # Performance score (0-40 points)
            if perf and perf.total_trades > 10:
                # Win rate contribution
                score += perf.win_rate * 15
                
                # Profit factor contribution
                score += min(perf.profit_factor * 5, 15)
                
                # Sharpe ratio contribution
                score += min(max(perf.sharpe_ratio, 0) * 5, 10)
            else:
                score += 20  # New strategy gets neutral score
            
            # Weight contribution (0-30 points)
            score += config.weight * 30
            
            scored_strategies.append((config, score))
        
        # Sort by score descending
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        
        return [s[0] for s in scored_strategies[:top_n]]
    
    def update_regime(self, regime: MarketRegime) -> None:
        """Update current market regime"""
        if regime != self._current_regime:
            self._regime_history.append((datetime.utcnow(), regime))
            self._current_regime = regime
    
    def update_performance(self, strategy_name: str, trades: List[Dict]) -> None:
        """Update strategy performance metrics"""
        if strategy_name in self._performance:
            self._performance[strategy_name].update_metrics(trades)
    
    def get_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get strategy performance"""
        return self._performance.get(strategy_name)
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get strategy rankings by performance"""
        rankings = []
        
        for name, perf in self._performance.items():
            if perf.total_trades > 0:
                # Composite score
                score = (
                    perf.win_rate * 0.2 +
                    min(perf.profit_factor / 3, 1.0) * 0.3 +
                    min(max(perf.sharpe_ratio, 0) / 3, 1.0) * 0.3 +
                    min(perf.expectancy / 100, 1.0) * 0.2
                )
                rankings.append((name, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


# Global registry instance
_registry = StrategyRegistry()


def get_registry() -> StrategyRegistry:
    """Get the global strategy registry"""
    return _registry


def register_strategy(
    name: str,
    strategy_class: Type,
    **kwargs
) -> None:
    """Register a strategy with the global registry"""
    _registry.register(name, strategy_class, **kwargs)
