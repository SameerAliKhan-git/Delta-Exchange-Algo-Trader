"""
Risk Manager Module

Implements comprehensive risk management:
- Position sizing (Kelly, fixed fraction)
- Portfolio risk limits
- Drawdown controls
- Correlation-aware position management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger


class RiskAction(Enum):
    """Risk management actions."""
    ALLOW = "allow"
    REDUCE = "reduce"
    REJECT = "reject"
    CLOSE_ALL = "close_all"
    HALT = "halt"


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    # Position limits
    max_position_size: float = 0.20  # 20% of portfolio per position
    max_portfolio_leverage: float = 2.0
    max_positions: int = 10
    max_correlated_exposure: float = 0.50  # 50% in correlated assets
    
    # Loss limits
    max_drawdown: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 0.03  # 3% daily loss limit
    weekly_loss_limit: float = 0.08  # 8% weekly loss limit
    trade_loss_limit: float = 0.02  # 2% max loss per trade
    
    # Risk metrics
    max_var_1d: float = 0.05  # 5% daily VaR limit
    max_volatility: float = 0.30  # 30% annualized volatility
    
    # Time limits
    max_holding_period: int = 100  # Max bars to hold
    forced_close_age: int = 200  # Force close after this many bars
    
    # Concentration
    max_sector_exposure: float = 0.40  # 40% max sector exposure
    min_diversification: int = 3  # Minimum number of positions


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    weight: float  # Portfolio weight
    volatility: float
    var_1d: float
    holding_period: int
    stop_loss: float
    stop_distance: float


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_value: float
    cash: float
    invested: float
    leverage: float
    
    # PnL
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    
    # Risk metrics
    portfolio_volatility: float
    portfolio_var_1d: float
    portfolio_sharpe: float
    current_drawdown: float
    max_drawdown: float
    
    # Exposure
    long_exposure: float
    short_exposure: float
    gross_exposure: float
    net_exposure: float
    
    # Limits
    positions_count: int
    limit_breaches: List[str] = field(default_factory=list)


class RiskManager:
    """
    Portfolio risk manager.
    
    Monitors and controls risk at position and portfolio levels.
    
    Example:
        risk_mgr = RiskManager(limits=RiskLimits())
        
        # Check if trade is allowed
        action, reason = risk_mgr.check_trade(
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            price=50000
        )
        
        if action == RiskAction.ALLOW:
            execute_trade()
    """
    
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize risk manager.
        
        Args:
            limits: Risk limit configuration
            initial_capital: Starting capital
        """
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital
        
        # State
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.capital = initial_capital
        self.equity_history: List[Tuple[datetime, float]] = []
        
        # Tracking
        self.peak_equity = initial_capital
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.last_date = None
        self.last_week = None
        
        # Risk flags
        self.is_halted = False
        self.halt_reason = ""
        
        # Correlation matrix (to be updated)
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Price history for volatility calculation
        self.price_history: Dict[str, List[float]] = {}
    
    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """
        Update equity tracking.
        
        Args:
            equity: Current portfolio equity
            timestamp: Current timestamp
        """
        self.equity_history.append((timestamp, equity))
        
        # Update peak equity
        self.peak_equity = max(self.peak_equity, equity)
        
        # Reset daily PnL on new day
        current_date = timestamp.date() if hasattr(timestamp, 'date') else None
        if current_date != self.last_date:
            self.daily_pnl = 0.0
            self.last_date = current_date
        
        # Reset weekly PnL on new week
        current_week = timestamp.isocalendar()[1] if hasattr(timestamp, 'isocalendar') else None
        if current_week != self.last_week:
            self.weekly_pnl = 0.0
            self.last_week = current_week
    
    def update_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        current_price: float,
        stop_loss: Optional[float] = None
    ) -> None:
        """
        Update position information.
        
        Args:
            symbol: Asset symbol
            size: Position size (positive for long, negative for short)
            entry_price: Entry price
            current_price: Current price
            stop_loss: Stop loss price
        """
        if size == 0:
            if symbol in self.positions:
                del self.positions[symbol]
            return
        
        pnl = (current_price - entry_price) * size
        pnl_pct = (current_price / entry_price - 1) * np.sign(size) if entry_price > 0 else 0
        
        # Calculate position weight
        position_value = abs(size * current_price)
        equity = self._get_equity()
        weight = position_value / equity if equity > 0 else 0
        
        # Calculate volatility if we have history
        volatility = self._calculate_volatility(symbol)
        
        # Calculate VaR
        var_1d = position_value * volatility * 2.33 / np.sqrt(252)  # 99% VaR
        
        # Stop distance
        stop_distance = abs(current_price - stop_loss) / current_price if stop_loss else self.limits.trade_loss_limit
        
        self.positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'current_price': current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'weight': weight,
            'volatility': volatility,
            'var_1d': var_1d,
            'stop_loss': stop_loss,
            'stop_distance': stop_distance,
            'holding_period': self.positions.get(symbol, {}).get('holding_period', 0) + 1
        }
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(current_price)
        
        # Keep last 100 prices
        self.price_history[symbol] = self.price_history[symbol][-100:]
    
    def _get_equity(self) -> float:
        """Get current equity estimate."""
        unrealized_pnl = sum(p.get('pnl', 0) for p in self.positions.values())
        return self.capital + unrealized_pnl
    
    def _calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """Calculate rolling volatility for a symbol."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < window:
            return 0.20  # Default 20% volatility
        
        prices = np.array(self.price_history[symbol][-window:])
        returns = np.diff(np.log(prices))
        
        if len(returns) == 0:
            return 0.20
        
        return np.std(returns) * np.sqrt(252 * 24)  # Annualized (assuming hourly)
    
    def get_portfolio_risk(self) -> PortfolioRisk:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Returns:
            PortfolioRisk object with all metrics
        """
        equity = self._get_equity()
        
        # Calculate exposures
        long_exposure = sum(
            p['size'] * p['current_price']
            for p in self.positions.values() if p['size'] > 0
        )
        short_exposure = sum(
            abs(p['size']) * p['current_price']
            for p in self.positions.values() if p['size'] < 0
        )
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        leverage = gross_exposure / equity if equity > 0 else 0
        
        # Calculate unrealized PnL
        unrealized_pnl = sum(p['pnl'] for p in self.positions.values())
        
        # Calculate portfolio volatility (simplified - sum of weighted variances)
        portfolio_var = sum(
            (p['weight'] ** 2) * (p['volatility'] ** 2)
            for p in self.positions.values()
        )
        portfolio_volatility = np.sqrt(portfolio_var)
        
        # Calculate portfolio VaR
        portfolio_var_1d = equity * portfolio_volatility * 2.33 / np.sqrt(252)
        
        # Calculate drawdown
        current_drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Check limit breaches
        limit_breaches = []
        
        if leverage > self.limits.max_portfolio_leverage:
            limit_breaches.append(f"leverage: {leverage:.2f} > {self.limits.max_portfolio_leverage}")
        
        if current_drawdown > self.limits.max_drawdown:
            limit_breaches.append(f"drawdown: {current_drawdown:.2%} > {self.limits.max_drawdown:.2%}")
        
        if abs(self.daily_pnl / self.initial_capital) > self.limits.daily_loss_limit:
            limit_breaches.append(f"daily_loss: {self.daily_pnl / self.initial_capital:.2%}")
        
        if portfolio_var_1d / equity > self.limits.max_var_1d:
            limit_breaches.append(f"var: {portfolio_var_1d / equity:.2%} > {self.limits.max_var_1d:.2%}")
        
        return PortfolioRisk(
            total_value=equity,
            cash=self.capital,
            invested=gross_exposure,
            leverage=leverage,
            realized_pnl=0.0,  # Would need trade history
            unrealized_pnl=unrealized_pnl,
            daily_pnl=self.daily_pnl,
            portfolio_volatility=portfolio_volatility,
            portfolio_var_1d=portfolio_var_1d,
            portfolio_sharpe=0.0,  # Would need return history
            current_drawdown=current_drawdown,
            max_drawdown=current_drawdown,  # Simplified
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            positions_count=len(self.positions),
            limit_breaches=limit_breaches
        )
    
    def check_trade(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        size: float,
        price: float,
        stop_loss: Optional[float] = None
    ) -> Tuple[RiskAction, str]:
        """
        Check if a trade is allowed under risk limits.
        
        Args:
            symbol: Asset symbol
            side: Trade side ('buy' or 'sell')
            size: Trade size
            price: Trade price
            stop_loss: Stop loss price
            
        Returns:
            Tuple of (RiskAction, reason_string)
        """
        if self.is_halted:
            return RiskAction.HALT, f"Trading halted: {self.halt_reason}"
        
        equity = self._get_equity()
        trade_value = size * price
        
        # Get portfolio risk
        portfolio_risk = self.get_portfolio_risk()
        
        # Check for limit breaches
        if portfolio_risk.limit_breaches:
            # Allow closing trades, reject opening trades
            if symbol in self.positions:
                current_pos = self.positions[symbol]['size']
                if (side == 'sell' and current_pos > 0) or (side == 'buy' and current_pos < 0):
                    return RiskAction.ALLOW, "Closing position allowed despite limits"
            
            return RiskAction.REJECT, f"Limit breaches: {portfolio_risk.limit_breaches}"
        
        # Check position size limit
        position_weight = trade_value / equity if equity > 0 else 1.0
        if position_weight > self.limits.max_position_size:
            recommended_size = self.limits.max_position_size * equity / price
            return RiskAction.REDUCE, f"Position too large: {position_weight:.2%} > {self.limits.max_position_size:.2%}. Recommended: {recommended_size:.4f}"
        
        # Check number of positions
        if len(self.positions) >= self.limits.max_positions and symbol not in self.positions:
            return RiskAction.REJECT, f"Max positions reached: {len(self.positions)}"
        
        # Check leverage
        new_gross = portfolio_risk.gross_exposure + trade_value
        new_leverage = new_gross / equity if equity > 0 else float('inf')
        if new_leverage > self.limits.max_portfolio_leverage:
            return RiskAction.REJECT, f"Leverage limit: {new_leverage:.2f} > {self.limits.max_portfolio_leverage}"
        
        # Check trade loss limit
        if stop_loss:
            if side == 'buy':
                max_loss = (price - stop_loss) / price
            else:
                max_loss = (stop_loss - price) / price
            
            if max_loss > self.limits.trade_loss_limit:
                return RiskAction.REDUCE, f"Trade risk too high: {max_loss:.2%} > {self.limits.trade_loss_limit:.2%}"
        
        # Check drawdown
        if portfolio_risk.current_drawdown > self.limits.max_drawdown * 0.8:
            # Near drawdown limit - reduce position sizes
            return RiskAction.REDUCE, f"Near drawdown limit: {portfolio_risk.current_drawdown:.2%}"
        
        return RiskAction.ALLOW, "Trade within risk limits"
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        stop_loss: float,
        method: str = 'fixed_fraction'
    ) -> float:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Asset symbol
            price: Entry price
            stop_loss: Stop loss price
            method: Sizing method ('fixed_fraction', 'kelly', 'volatility_target')
            
        Returns:
            Recommended position size
        """
        equity = self._get_equity()
        
        # Calculate stop distance
        stop_distance = abs(price - stop_loss) / price
        
        if method == 'fixed_fraction':
            # Risk a fixed percentage per trade
            risk_amount = equity * self.limits.trade_loss_limit
            position_value = risk_amount / stop_distance if stop_distance > 0 else 0
            position_size = position_value / price
        
        elif method == 'kelly':
            # Kelly criterion (requires win rate and win/loss ratio)
            # Use conservative defaults
            win_rate = 0.50
            win_loss_ratio = 1.5
            kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
            kelly_pct = max(0, kelly_pct * 0.25)  # Use quarter Kelly
            
            position_value = equity * kelly_pct
            position_size = position_value / price
        
        elif method == 'volatility_target':
            # Target a specific portfolio volatility contribution
            target_vol = 0.02  # 2% contribution to portfolio vol
            symbol_vol = self._calculate_volatility(symbol)
            
            if symbol_vol > 0:
                position_value = (target_vol / symbol_vol) * equity
                position_size = position_value / price
            else:
                position_size = 0
        
        else:
            position_size = 0
        
        # Apply position size limits
        max_position_value = equity * self.limits.max_position_size
        max_position_size = max_position_value / price
        
        position_size = min(position_size, max_position_size)
        
        return max(0, position_size)
    
    def check_exit_conditions(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if position should be exited.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if symbol not in self.positions:
            return False, "No position"
        
        pos = self.positions[symbol]
        
        # Check stop loss
        if pos['stop_loss']:
            if pos['size'] > 0 and pos['current_price'] <= pos['stop_loss']:
                return True, "Stop loss hit (long)"
            if pos['size'] < 0 and pos['current_price'] >= pos['stop_loss']:
                return True, "Stop loss hit (short)"
        
        # Check holding period
        if pos['holding_period'] >= self.limits.forced_close_age:
            return True, f"Forced close: held {pos['holding_period']} bars"
        
        if pos['holding_period'] >= self.limits.max_holding_period:
            return True, f"Max holding period: {pos['holding_period']} bars"
        
        # Check position loss limit
        if pos['pnl_pct'] < -self.limits.trade_loss_limit:
            return True, f"Position loss limit: {pos['pnl_pct']:.2%}"
        
        return False, "No exit condition"
    
    def halt_trading(self, reason: str) -> None:
        """Halt all trading."""
        self.is_halted = True
        self.halt_reason = reason
        logger.warning(f"Trading halted: {reason}")
    
    def resume_trading(self) -> None:
        """Resume trading."""
        self.is_halted = False
        self.halt_reason = ""
        logger.info("Trading resumed")
    
    def get_summary(self) -> Dict:
        """Get risk summary."""
        portfolio_risk = self.get_portfolio_risk()
        
        return {
            'equity': portfolio_risk.total_value,
            'cash': portfolio_risk.cash,
            'leverage': portfolio_risk.leverage,
            'drawdown': portfolio_risk.current_drawdown,
            'daily_pnl': portfolio_risk.daily_pnl,
            'positions_count': portfolio_risk.positions_count,
            'gross_exposure': portfolio_risk.gross_exposure,
            'net_exposure': portfolio_risk.net_exposure,
            'portfolio_volatility': portfolio_risk.portfolio_volatility,
            'portfolio_var_1d': portfolio_risk.portfolio_var_1d,
            'limit_breaches': portfolio_risk.limit_breaches,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason
        }


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("RISK MANAGER DEMO")
    print("=" * 60)
    
    # Create risk manager
    limits = RiskLimits(
        max_position_size=0.10,  # 10% max per position
        max_portfolio_leverage=1.5,
        max_drawdown=0.10,
        trade_loss_limit=0.01
    )
    
    risk_mgr = RiskManager(limits=limits, initial_capital=100000)
    
    # Simulate some positions
    risk_mgr.update_position(
        symbol="BTC/USD",
        size=0.5,
        entry_price=50000,
        current_price=51000,
        stop_loss=49000
    )
    
    risk_mgr.update_position(
        symbol="ETH/USD",
        size=5.0,
        entry_price=3000,
        current_price=3100,
        stop_loss=2900
    )
    
    # Check a new trade
    action, reason = risk_mgr.check_trade(
        symbol="XRP/USD",
        side="buy",
        size=10000,
        price=0.50,
        stop_loss=0.48
    )
    
    print(f"\nTrade check: {action.value}")
    print(f"Reason: {reason}")
    
    # Calculate position size
    recommended_size = risk_mgr.calculate_position_size(
        symbol="XRP/USD",
        price=0.50,
        stop_loss=0.48,
        method='fixed_fraction'
    )
    
    print(f"\nRecommended position size: {recommended_size:.2f}")
    
    # Get portfolio risk
    portfolio_risk = risk_mgr.get_portfolio_risk()
    print(f"\nPortfolio Risk:")
    print(f"  Equity: ${portfolio_risk.total_value:,.2f}")
    print(f"  Leverage: {portfolio_risk.leverage:.2f}x")
    print(f"  Drawdown: {portfolio_risk.current_drawdown:.2%}")
    print(f"  Long Exposure: ${portfolio_risk.long_exposure:,.2f}")
    print(f"  Short Exposure: ${portfolio_risk.short_exposure:,.2f}")
    print(f"  Volatility: {portfolio_risk.portfolio_volatility:.2%}")
    
    # Get summary
    summary = risk_mgr.get_summary()
    print(f"\nSummary: {summary}")
