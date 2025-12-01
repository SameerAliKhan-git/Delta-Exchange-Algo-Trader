"""
Master Trading Orchestrator - Renaissance-Level System Coordinator
===================================================================
The brain that coordinates all strategies, manages capital allocation,
and ensures optimal performance across the entire trading system.

Inspired by how Renaissance Technologies coordinates their Medallion fund.

Features:
- Multi-strategy coordination
- Dynamic capital allocation
- Risk aggregation and limits
- Performance attribution
- Real-time strategy switching
- Drawdown protection
- Correlation-aware position sizing
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import deque
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading modes"""
    AGGRESSIVE = "aggressive"  # Max alpha, higher risk
    NORMAL = "normal"  # Balanced
    CONSERVATIVE = "conservative"  # Capital preservation
    RISK_OFF = "risk_off"  # Minimal exposure
    EMERGENCY = "emergency"  # Exit all positions


class MarketRegime(Enum):
    """Market regime classification"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGING = "ranging"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CRISIS = "crisis"


@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy"""
    strategy_name: str
    weight: float  # % of capital
    min_weight: float = 0.0
    max_weight: float = 0.5
    current_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    correlation_to_portfolio: float = 0.0
    is_active: bool = True


@dataclass
class RiskLimits:
    """Portfolio-level risk limits"""
    max_portfolio_var: float = 0.02  # 2% daily VaR
    max_drawdown: float = 0.10  # 10% max drawdown
    max_position_size: float = 0.20  # 20% in single position
    max_correlation: float = 0.70  # Max strategy correlation
    max_leverage: float = 3.0
    min_cash_buffer: float = 0.10  # 10% cash minimum
    daily_loss_limit: float = 0.03  # 3% daily loss limit


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_equity: float = 0.0
    cash: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_exposure: float = 0.0
    leverage: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)
    position_values: Dict[str, float] = field(default_factory=dict)
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0


class CapitalAllocator:
    """Dynamic capital allocation across strategies"""
    
    def __init__(self, allocations: List[StrategyAllocation]):
        self.allocations = {a.strategy_name: a for a in allocations}
        self.rebalance_threshold = 0.05  # Rebalance if drift > 5%
        self.performance_lookback = 30  # Days
        
    def optimize_weights(
        self,
        returns: Dict[str, List[float]],
        regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Optimize strategy weights using modified mean-variance optimization.
        Accounts for regime and recent performance.
        """
        strategies = list(returns.keys())
        n = len(strategies)
        
        if n == 0:
            return {}
            
        # Calculate expected returns and covariance
        ret_matrix = []
        for s in strategies:
            rets = returns.get(s, [])
            if len(rets) < 2:
                rets = [0.0] * 10
            ret_matrix.append(rets[-self.performance_lookback:])
            
        # Pad to same length
        max_len = max(len(r) for r in ret_matrix)
        ret_matrix = [
            r + [0.0] * (max_len - len(r)) for r in ret_matrix
        ]
        
        ret_array = np.array(ret_matrix)
        expected_returns = np.mean(ret_array, axis=1)
        cov_matrix = np.cov(ret_array)
        
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]])
            
        # Risk parity with performance tilt
        vol = np.sqrt(np.diag(cov_matrix) + 1e-8)
        inv_vol = 1.0 / vol
        
        # Performance tilt - favor recent winners moderately
        perf_tilt = 1.0 + np.clip(expected_returns, -0.5, 0.5)
        
        # Regime adjustments
        regime_mult = self._get_regime_multipliers(strategies, regime)
        
        # Combined weights
        raw_weights = inv_vol * perf_tilt * regime_mult
        
        # Apply constraints
        weights = {}
        for i, s in enumerate(strategies):
            alloc = self.allocations.get(s)
            if alloc:
                w = raw_weights[i]
                w = max(alloc.min_weight, min(alloc.max_weight, w))
                weights[s] = w
            else:
                weights[s] = raw_weights[i]
                
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
            
        return weights
        
    def _get_regime_multipliers(
        self,
        strategies: List[str],
        regime: MarketRegime
    ) -> np.ndarray:
        """Get regime-based weight multipliers"""
        multipliers = np.ones(len(strategies))
        
        regime_preferences = {
            MarketRegime.BULL_TREND: {
                'momentum': 1.5, 'trend': 1.5,
                'mean_reversion': 0.7, 'arbitrage': 1.0
            },
            MarketRegime.BEAR_TREND: {
                'momentum': 1.3, 'trend': 1.2,
                'mean_reversion': 0.5, 'hedging': 2.0
            },
            MarketRegime.RANGING: {
                'mean_reversion': 1.5, 'arbitrage': 1.3,
                'momentum': 0.5, 'trend': 0.5
            },
            MarketRegime.HIGH_VOL: {
                'volatility': 1.5, 'options': 1.5,
                'momentum': 0.7, 'leverage': 0.3
            },
            MarketRegime.CRISIS: {
                'hedging': 2.0, 'cash': 2.0,
                'momentum': 0.3, 'arbitrage': 0.5
            }
        }
        
        prefs = regime_preferences.get(regime, {})
        
        for i, s in enumerate(strategies):
            for key, mult in prefs.items():
                if key.lower() in s.lower():
                    multipliers[i] = mult
                    break
                    
        return multipliers
        
    def should_rebalance(self, current_weights: Dict[str, float]) -> bool:
        """Check if rebalancing is needed"""
        for s, target in self.allocations.items():
            current = current_weights.get(s, 0)
            if abs(current - target.weight) > self.rebalance_threshold:
                return True
        return False


class RegimeDetector:
    """Market regime detection"""
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.price_history: deque = deque(maxlen=lookback * 2)
        self.vol_history: deque = deque(maxlen=lookback)
        
    def add_data(self, price: float, volume: float = 0):
        """Add price data"""
        self.price_history.append(price)
        
    def detect_regime(self) -> MarketRegime:
        """Detect current market regime"""
        if len(self.price_history) < 20:
            return MarketRegime.RANGING
            
        prices = np.array(list(self.price_history))
        returns = np.diff(prices) / prices[:-1]
        
        # Trend detection
        sma_short = np.mean(prices[-10:])
        sma_long = np.mean(prices[-30:]) if len(prices) >= 30 else sma_short
        trend_strength = (sma_short - sma_long) / sma_long
        
        # Volatility
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
        avg_vol = np.std(returns) if len(returns) > 0 else 0.02
        vol_ratio = volatility / (avg_vol + 1e-8)
        
        # Classify
        if vol_ratio > 2.0:
            if volatility > 0.05:
                return MarketRegime.CRISIS
            return MarketRegime.HIGH_VOL
            
        if vol_ratio < 0.5:
            return MarketRegime.LOW_VOL
            
        if trend_strength > 0.02:
            return MarketRegime.BULL_TREND
        elif trend_strength < -0.02:
            return MarketRegime.BEAR_TREND
            
        return MarketRegime.RANGING


class DrawdownProtection:
    """Drawdown protection and circuit breakers"""
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.daily_start_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.triggered_at: Optional[float] = None
        
    def update(self, equity: float) -> TradingMode:
        """Update and check drawdown protection"""
        self.peak_equity = max(self.peak_equity, equity)
        
        if self.daily_start_equity == 0:
            self.daily_start_equity = equity
            
        # Calculate drawdowns
        current_dd = (self.peak_equity - equity) / self.peak_equity
        daily_pnl = (equity - self.daily_start_equity) / self.daily_start_equity
        
        # Check limits
        if current_dd > self.limits.max_drawdown * 0.9:
            logger.warning(f"🚨 EMERGENCY: Near max drawdown {current_dd:.1%}")
            return TradingMode.EMERGENCY
            
        if current_dd > self.limits.max_drawdown * 0.7:
            logger.warning(f"⚠️ High drawdown: {current_dd:.1%}")
            return TradingMode.RISK_OFF
            
        if daily_pnl < -self.limits.daily_loss_limit:
            logger.warning(f"⚠️ Daily loss limit hit: {daily_pnl:.1%}")
            return TradingMode.RISK_OFF
            
        if daily_pnl < -self.limits.daily_loss_limit * 0.7:
            return TradingMode.CONSERVATIVE
            
        return TradingMode.NORMAL
        
    def reset_daily(self, equity: float):
        """Reset daily tracking"""
        self.daily_start_equity = equity


class PerformanceAttribution:
    """Track performance by strategy and factor"""
    
    def __init__(self):
        self.strategy_returns: Dict[str, List[float]] = {}
        self.factor_returns: Dict[str, List[float]] = {}
        self.timestamps: List[float] = []
        
    def record(self, strategy: str, pnl: float, factors: Dict[str, float] = None):
        """Record performance"""
        if strategy not in self.strategy_returns:
            self.strategy_returns[strategy] = []
        self.strategy_returns[strategy].append(pnl)
        
        if factors:
            for factor, value in factors.items():
                if factor not in self.factor_returns:
                    self.factor_returns[factor] = []
                self.factor_returns[factor].append(value)
                
        self.timestamps.append(time.time())
        
    def get_attribution(self) -> Dict[str, Dict]:
        """Get performance attribution report"""
        report = {}
        
        for strategy, returns in self.strategy_returns.items():
            if len(returns) > 0:
                report[strategy] = {
                    'total_return': sum(returns),
                    'avg_return': np.mean(returns),
                    'volatility': np.std(returns) if len(returns) > 1 else 0,
                    'sharpe': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
                    'max_drawdown': self._calculate_max_dd(returns),
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns)
                }
                
        return report
        
    def _calculate_max_dd(self, returns: List[float]) -> float:
        """Calculate max drawdown from returns"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


class MasterOrchestrator:
    """
    Master Trading Orchestrator
    ============================
    Coordinates all strategies and manages the entire trading operation.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_limits: Optional[RiskLimits] = None
    ):
        self.initial_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()
        
        # State
        self.portfolio = PortfolioState(
            total_equity=initial_capital,
            cash=initial_capital,
            peak_equity=initial_capital
        )
        self.mode = TradingMode.NORMAL
        self.regime = MarketRegime.RANGING
        
        # Components
        self.strategies: Dict[str, Any] = {}
        self.allocator: Optional[CapitalAllocator] = None
        self.regime_detector = RegimeDetector()
        self.drawdown_protection = DrawdownProtection(self.risk_limits)
        self.attribution = PerformanceAttribution()
        
        # Execution
        self.execution_engine: Optional[Any] = None
        
        # Signals
        self.pending_signals: List[Dict] = []
        
        # State
        self.running = False
        self._tasks: List[asyncio.Task] = []
        
        # Callbacks
        self.on_signal: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        self.on_mode_change: Optional[Callable] = None
        
        logger.info(f"🎯 Master Orchestrator initialized with ${initial_capital:,.0f}")
        
    def register_strategy(
        self,
        name: str,
        strategy: Any,
        allocation: StrategyAllocation
    ):
        """Register a strategy with the orchestrator"""
        self.strategies[name] = strategy
        
        if self.allocator is None:
            self.allocator = CapitalAllocator([allocation])
        else:
            self.allocator.allocations[name] = allocation
            
        logger.info(f"📊 Strategy registered: {name} (weight: {allocation.weight:.1%})")
        
    def set_execution_engine(self, engine: Any):
        """Set the execution engine"""
        self.execution_engine = engine
        
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        logger.info("🚀 Master Orchestrator started")
        
        # Start main loop
        self._tasks.append(asyncio.create_task(self._main_loop()))
        
        # Start risk monitor
        self._tasks.append(asyncio.create_task(self._risk_monitor()))
        
        # Start allocation rebalancer
        self._tasks.append(asyncio.create_task(self._rebalance_loop()))
        
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Close all positions if in emergency mode
        if self.mode == TradingMode.EMERGENCY:
            await self._emergency_close_all()
            
        logger.info("🛑 Master Orchestrator stopped")
        
    async def _main_loop(self):
        """Main orchestration loop"""
        while self.running:
            try:
                # Check mode
                if self.mode == TradingMode.EMERGENCY:
                    await asyncio.sleep(1)
                    continue
                    
                # Process pending signals
                await self._process_signals()
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(1)
                
    async def _risk_monitor(self):
        """Continuous risk monitoring"""
        while self.running:
            try:
                # Update portfolio state
                await self._update_portfolio_state()
                
                # Check drawdown protection
                new_mode = self.drawdown_protection.update(self.portfolio.total_equity)
                
                if new_mode != self.mode:
                    await self._change_mode(new_mode)
                    
                # Check leverage
                if self.portfolio.leverage > self.risk_limits.max_leverage:
                    logger.warning(f"⚠️ Leverage too high: {self.portfolio.leverage:.1f}x")
                    await self._reduce_exposure()
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Risk monitor error: {e}")
                await asyncio.sleep(5)
                
    async def _rebalance_loop(self):
        """Periodic portfolio rebalancing"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if self.mode in {TradingMode.RISK_OFF, TradingMode.EMERGENCY}:
                    continue
                    
                if self.allocator:
                    # Get current weights
                    current_weights = self._calculate_current_weights()
                    
                    # Check if rebalance needed
                    if self.allocator.should_rebalance(current_weights):
                        await self._rebalance()
                        
            except Exception as e:
                logger.error(f"Rebalance loop error: {e}")
                
    def receive_market_data(self, symbol: str, price: float, volume: float = 0):
        """Receive market data update"""
        self.regime_detector.add_data(price, volume)
        
        # Check for regime change
        new_regime = self.regime_detector.detect_regime()
        if new_regime != self.regime:
            logger.info(f"📊 Regime change: {self.regime.value} -> {new_regime.value}")
            self.regime = new_regime
            
    def receive_signal(
        self,
        strategy_name: str,
        symbol: str,
        direction: str,  # 'long', 'short', 'close'
        strength: float,  # 0-1
        metadata: Dict = None
    ):
        """Receive a trading signal from a strategy"""
        if self.mode == TradingMode.EMERGENCY:
            return
            
        signal = {
            'strategy': strategy_name,
            'symbol': symbol,
            'direction': direction,
            'strength': strength,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        self.pending_signals.append(signal)
        logger.debug(f"📡 Signal received: {strategy_name} -> {direction} {symbol}")
        
    async def _process_signals(self):
        """Process pending signals with risk checks"""
        if not self.pending_signals:
            return
            
        signals = self.pending_signals.copy()
        self.pending_signals.clear()
        
        # Aggregate signals for same symbol
        aggregated = self._aggregate_signals(signals)
        
        for symbol, agg_signal in aggregated.items():
            try:
                await self._execute_signal(symbol, agg_signal)
            except Exception as e:
                logger.error(f"Signal execution error: {e}")
                
    def _aggregate_signals(self, signals: List[Dict]) -> Dict[str, Dict]:
        """Aggregate signals from multiple strategies"""
        by_symbol: Dict[str, List[Dict]] = {}
        
        for s in signals:
            symbol = s['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(s)
            
        aggregated = {}
        
        for symbol, symbol_signals in by_symbol.items():
            # Weight by allocation and strength
            long_score = 0.0
            short_score = 0.0
            total_weight = 0.0
            
            for s in symbol_signals:
                strategy = s['strategy']
                alloc = self.allocator.allocations.get(strategy) if self.allocator else None
                weight = alloc.weight if alloc else 0.1
                
                if s['direction'] == 'long':
                    long_score += s['strength'] * weight
                elif s['direction'] == 'short':
                    short_score += s['strength'] * weight
                    
                total_weight += weight
                
            if total_weight > 0:
                net_score = (long_score - short_score) / total_weight
            else:
                net_score = 0
                
            if abs(net_score) > 0.2:  # Minimum consensus threshold
                aggregated[symbol] = {
                    'direction': 'long' if net_score > 0 else 'short',
                    'strength': abs(net_score),
                    'sources': len(symbol_signals)
                }
                
        return aggregated
        
    async def _execute_signal(self, symbol: str, signal: Dict):
        """Execute an aggregated signal"""
        direction = signal['direction']
        strength = signal['strength']
        
        # Calculate position size
        size = self._calculate_position_size(symbol, strength)
        
        if size <= 0:
            return
            
        # Risk checks
        if not self._check_risk_limits(symbol, size, direction):
            logger.warning(f"⚠️ Risk check failed for {symbol}")
            return
            
        # Execute via engine
        if self.execution_engine:
            from engine.realtime import ExecutionAlgorithm
            
            side = 'buy' if direction == 'long' else 'sell'
            
            # Select algorithm based on size and urgency
            algo = ExecutionAlgorithm.ADAPTIVE
            if strength > 0.8:
                algo = ExecutionAlgorithm.MARKET
            elif size > self.portfolio.total_equity * 0.1:
                algo = ExecutionAlgorithm.TWAP
                
            await self.execution_engine.submit_order(
                symbol=symbol,
                side=side,
                size=size,
                algorithm=algo,
                urgency=strength
            )
            
            logger.info(f"🎯 Order submitted: {side} {size:.4f} {symbol}")
            
            if self.on_trade:
                self.on_trade(symbol, side, size)
                
    def _calculate_position_size(self, symbol: str, strength: float) -> float:
        """Calculate position size using Kelly-inspired sizing"""
        # Base size from portfolio
        base_allocation = self.portfolio.total_equity * 0.05  # 5% base
        
        # Adjust by signal strength
        strength_mult = strength ** 2  # Convex in strength
        
        # Mode adjustments
        mode_mult = {
            TradingMode.AGGRESSIVE: 1.5,
            TradingMode.NORMAL: 1.0,
            TradingMode.CONSERVATIVE: 0.5,
            TradingMode.RISK_OFF: 0.1,
            TradingMode.EMERGENCY: 0.0
        }.get(self.mode, 1.0)
        
        # Regime adjustments
        regime_mult = {
            MarketRegime.BULL_TREND: 1.2,
            MarketRegime.BEAR_TREND: 0.8,
            MarketRegime.RANGING: 1.0,
            MarketRegime.HIGH_VOL: 0.6,
            MarketRegime.CRISIS: 0.2
        }.get(self.regime, 1.0)
        
        size = base_allocation * strength_mult * mode_mult * regime_mult
        
        # Ensure within limits
        max_size = self.portfolio.total_equity * self.risk_limits.max_position_size
        size = min(size, max_size)
        
        return size
        
    def _check_risk_limits(self, symbol: str, size: float, direction: str) -> bool:
        """Check if trade passes risk limits"""
        # Check position concentration
        current = self.portfolio.position_values.get(symbol, 0)
        new_exposure = current + size if direction == 'long' else current - size
        
        if abs(new_exposure) / self.portfolio.total_equity > self.risk_limits.max_position_size:
            return False
            
        # Check total exposure
        new_total = self.portfolio.total_exposure + size
        if new_total / self.portfolio.total_equity > self.risk_limits.max_leverage:
            return False
            
        # Check cash buffer
        if self.portfolio.cash - size < self.portfolio.total_equity * self.risk_limits.min_cash_buffer:
            return False
            
        return True
        
    def _calculate_current_weights(self) -> Dict[str, float]:
        """Calculate current strategy weights based on positions"""
        # This would need actual position tracking per strategy
        # Simplified version
        weights = {}
        for name in self.strategies:
            alloc = self.allocator.allocations.get(name) if self.allocator else None
            weights[name] = alloc.weight if alloc else 0.1
        return weights
        
    async def _rebalance(self):
        """Rebalance portfolio to target weights"""
        if not self.allocator:
            return
            
        # Get strategy returns
        returns = {}
        for name in self.strategies:
            rets = self.attribution.strategy_returns.get(name, [])
            returns[name] = rets
            
        # Optimize weights
        new_weights = self.allocator.optimize_weights(returns, self.regime)
        
        # Update allocations
        for name, weight in new_weights.items():
            if name in self.allocator.allocations:
                old_weight = self.allocator.allocations[name].weight
                self.allocator.allocations[name].weight = weight
                
                if abs(weight - old_weight) > 0.01:
                    logger.info(f"📊 Rebalance: {name} {old_weight:.1%} -> {weight:.1%}")
                    
    async def _change_mode(self, new_mode: TradingMode):
        """Change trading mode"""
        old_mode = self.mode
        self.mode = new_mode
        
        logger.warning(f"🔄 Mode change: {old_mode.value} -> {new_mode.value}")
        
        if self.on_mode_change:
            self.on_mode_change(old_mode, new_mode)
            
        if new_mode == TradingMode.EMERGENCY:
            await self._emergency_close_all()
        elif new_mode == TradingMode.RISK_OFF:
            await self._reduce_exposure(target_pct=0.5)
            
    async def _reduce_exposure(self, target_pct: float = 0.8):
        """Reduce overall exposure"""
        if not self.execution_engine:
            return
            
        for symbol, value in self.portfolio.position_values.items():
            if value > 0:
                reduce_size = value * (1 - target_pct)
                if reduce_size > 0:
                    await self.execution_engine.submit_order(
                        symbol=symbol,
                        side='sell',
                        size=reduce_size,
                        urgency=0.8
                    )
                    
    async def _emergency_close_all(self):
        """Emergency close all positions"""
        logger.critical("🚨 EMERGENCY: Closing all positions!")
        
        if self.execution_engine:
            # Cancel all open orders
            await self.execution_engine.cancel_all()
            
            # Close all positions
            for symbol, size in self.portfolio.positions.items():
                if size != 0:
                    side = 'sell' if size > 0 else 'buy'
                    await self.execution_engine.submit_order(
                        symbol=symbol,
                        side=side,
                        size=abs(size),
                        urgency=1.0
                    )
                    
    async def _update_portfolio_state(self):
        """Update portfolio state from execution engine"""
        if self.execution_engine:
            # Get positions
            self.portfolio.positions = dict(self.execution_engine.positions)
            
        # Update metrics
        self.portfolio.peak_equity = max(
            self.portfolio.peak_equity,
            self.portfolio.total_equity
        )
        self.portfolio.current_drawdown = (
            (self.portfolio.peak_equity - self.portfolio.total_equity) /
            self.portfolio.peak_equity
        )
        
    def get_status(self) -> Dict:
        """Get orchestrator status"""
        return {
            'mode': self.mode.value,
            'regime': self.regime.value,
            'equity': self.portfolio.total_equity,
            'cash': self.portfolio.cash,
            'leverage': self.portfolio.leverage,
            'drawdown': f"{self.portfolio.current_drawdown:.1%}",
            'daily_pnl': f"{self.portfolio.daily_pnl:.1%}",
            'strategies_active': len(self.strategies),
            'pending_signals': len(self.pending_signals)
        }
        
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        attribution = self.attribution.get_attribution()
        
        return {
            'portfolio': {
                'total_return': f"{(self.portfolio.total_equity / self.initial_capital - 1) * 100:.2f}%",
                'peak_equity': self.portfolio.peak_equity,
                'max_drawdown': f"{self.portfolio.current_drawdown:.1%}",
                'current_mode': self.mode.value,
                'current_regime': self.regime.value
            },
            'strategies': attribution,
            'risk_metrics': {
                'leverage': self.portfolio.leverage,
                'position_count': len(self.portfolio.positions),
                'cash_ratio': self.portfolio.cash / self.portfolio.total_equity
            }
        }


# Factory function
def create_orchestrator(
    capital: float = 100000.0,
    risk_limits: Optional[RiskLimits] = None
) -> MasterOrchestrator:
    """Create a configured orchestrator"""
    return MasterOrchestrator(
        initial_capital=capital,
        risk_limits=risk_limits
    )
