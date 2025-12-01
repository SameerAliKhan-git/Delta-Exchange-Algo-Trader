"""
Risk Manager - Comprehensive risk management

Provides:
- Capital-aware position sizing (futures & options)
- Daily loss limits and drawdown protection
- Kill switch functionality
- Position and exposure limits
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date
from collections import defaultdict


class RiskState(Enum):
    """Risk management state"""
    ACTIVE = "active"
    REDUCED = "reduced"
    STOPPED = "stopped"
    KILL_SWITCH = "kill_switch"


@dataclass
class RiskConfig:
    """Risk configuration"""
    # Capital allocation
    max_capital_per_trade_pct: float = 0.02  # 2% per trade
    max_total_exposure_pct: float = 0.3  # 30% max exposure
    
    # Position limits
    max_positions: int = 5
    max_size_per_position: float = 100.0
    
    # Loss limits
    max_daily_loss_pct: float = 0.05  # 5% daily loss limit
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    
    # Leverage
    max_leverage: int = 10
    default_leverage: int = 3
    
    # Options specific
    max_premium_pct: float = 0.005  # 0.5% of capital for option premium
    min_option_delta: float = 0.2
    max_option_delta: float = 0.5
    
    # Kill switch
    cooldown_minutes: int = 60
    
    # ATR sizing
    atr_risk_multiple: float = 2.0  # Stop loss at 2x ATR


@dataclass
class PositionSize:
    """Position size calculation result"""
    size: float
    contracts: int
    notional: float
    margin_required: float
    risk_amount: float
    leverage: int
    valid: bool = True
    reason: str = ""


@dataclass
class DailyPnL:
    """Track daily P&L"""
    date: date
    starting_balance: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades: int = 0


class RiskManager:
    """
    Comprehensive risk management system
    """
    
    def __init__(
        self,
        config: RiskConfig = None,
        initial_capital: float = 10000.0
    ):
        """
        Initialize risk manager
        
        Args:
            config: Risk configuration
            initial_capital: Starting capital
        """
        self.config = config or RiskConfig()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # State tracking
        self.state = RiskState.ACTIVE
        self.kill_switch_time: float = 0.0
        
        # P&L tracking
        self.daily_pnl: Dict[date, DailyPnL] = {}
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        
        # Position tracking
        self.open_positions: Dict[str, float] = {}  # symbol -> size
        self.position_pnl: Dict[str, float] = {}  # symbol -> pnl
        
        # Trade history
        self.trade_count_today = 0
        self.last_trade_time = 0.0
    
    # ==================== Capital & Sizing ====================
    
    def update_capital(self, new_capital: float) -> None:
        """Update current capital from exchange balance"""
        self.current_capital = new_capital
        
        # Update peak and drawdown
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        
        self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital
        
        # Check drawdown limit
        if self.current_drawdown >= self.config.max_drawdown_pct:
            self.trigger_kill_switch("Max drawdown reached")
    
    def compute_futures_size(
        self,
        price: float,
        atr: float,
        contract_size: float = 1.0,
        leverage: int = None
    ) -> PositionSize:
        """
        Compute position size for futures using ATR-based risk
        
        Args:
            price: Current price
            atr: Average True Range
            contract_size: Contract size
            leverage: Leverage (uses default if not specified)
        
        Returns:
            PositionSize calculation
        """
        if not self.can_trade():
            return PositionSize(
                size=0, contracts=0, notional=0, margin_required=0,
                risk_amount=0, leverage=0, valid=False,
                reason=f"Trading disabled: {self.state.value}"
            )
        
        leverage = leverage or self.config.default_leverage
        leverage = min(leverage, self.config.max_leverage)
        
        # Calculate risk per trade
        risk_amount = self.current_capital * self.config.max_capital_per_trade_pct
        
        # Stop loss distance using ATR
        stop_distance = atr * self.config.atr_risk_multiple
        
        if stop_distance == 0:
            return PositionSize(
                size=0, contracts=0, notional=0, margin_required=0,
                risk_amount=risk_amount, leverage=leverage, valid=False,
                reason="ATR is zero"
            )
        
        # Position size = risk / stop distance
        size = risk_amount / stop_distance
        
        # Convert to contracts
        contracts = int(size / contract_size)
        
        # Calculate notional and margin
        notional = contracts * contract_size * price
        margin_required = notional / leverage
        
        # Check exposure limit
        total_exposure = self._get_total_exposure() + notional
        max_exposure = self.current_capital * self.config.max_total_exposure_pct
        
        if total_exposure > max_exposure:
            # Reduce size to fit within limit
            available_exposure = max_exposure - self._get_total_exposure()
            contracts = int(available_exposure / (contract_size * price))
            notional = contracts * contract_size * price
            margin_required = notional / leverage
        
        # Check max size
        size = min(contracts * contract_size, self.config.max_size_per_position)
        contracts = int(size / contract_size)
        notional = contracts * contract_size * price
        margin_required = notional / leverage
        
        return PositionSize(
            size=size,
            contracts=contracts,
            notional=notional,
            margin_required=margin_required,
            risk_amount=risk_amount,
            leverage=leverage,
            valid=contracts > 0,
            reason="" if contracts > 0 else "Size too small"
        )
    
    def compute_option_contracts(
        self,
        premium: float,
        contract_size: float = 1.0
    ) -> PositionSize:
        """
        Compute number of option contracts based on premium budget
        
        Args:
            premium: Option premium per contract
            contract_size: Contract size
        
        Returns:
            PositionSize calculation
        """
        if not self.can_trade():
            return PositionSize(
                size=0, contracts=0, notional=0, margin_required=0,
                risk_amount=0, leverage=1, valid=False,
                reason=f"Trading disabled: {self.state.value}"
            )
        
        # Budget for option premium
        premium_budget = self.current_capital * self.config.max_premium_pct
        
        if premium <= 0:
            return PositionSize(
                size=0, contracts=0, notional=0, margin_required=0,
                risk_amount=premium_budget, leverage=1, valid=False,
                reason="Invalid premium"
            )
        
        # Calculate contracts within budget
        total_premium_per_contract = premium * contract_size
        contracts = int(premium_budget / total_premium_per_contract)
        
        # Notional is the premium paid
        notional = contracts * total_premium_per_contract
        
        return PositionSize(
            size=contracts * contract_size,
            contracts=contracts,
            notional=notional,
            margin_required=notional,  # For options buyer, margin = premium
            risk_amount=notional,  # Max loss = premium paid
            leverage=1,
            valid=contracts > 0,
            reason="" if contracts > 0 else "Premium budget too small"
        )
    
    # ==================== Risk Checks ====================
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if self.state == RiskState.KILL_SWITCH:
            # Check if cooldown expired
            if time.time() - self.kill_switch_time > self.config.cooldown_minutes * 60:
                self.state = RiskState.ACTIVE
            else:
                return False
        
        return self.state in (RiskState.ACTIVE, RiskState.REDUCED)
    
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if new position can be opened
        
        Args:
            symbol: Trading symbol
        
        Returns:
            (allowed, reason)
        """
        if not self.can_trade():
            return False, f"Trading disabled: {self.state.value}"
        
        # Check position count
        if len(self.open_positions) >= self.config.max_positions:
            if symbol not in self.open_positions:
                return False, f"Max positions ({self.config.max_positions}) reached"
        
        # Check daily loss
        today = date.today()
        if today in self.daily_pnl:
            daily = self.daily_pnl[today]
            loss_pct = abs(min(0, daily.realized_pnl + daily.unrealized_pnl)) / self.initial_capital
            if loss_pct >= self.config.max_daily_loss_pct:
                return False, "Daily loss limit reached"
        
        # Check drawdown
        if self.current_drawdown >= self.config.max_drawdown_pct * 0.8:
            if self.state != RiskState.REDUCED:
                self.state = RiskState.REDUCED
            # Allow but with reduced size
            return True, "Position allowed but size reduced (high drawdown)"
        
        return True, "OK"
    
    def validate_order(
        self,
        symbol: str,
        size: float,
        price: float,
        side: str
    ) -> Tuple[bool, str, float]:
        """
        Validate order before submission
        
        Args:
            symbol: Trading symbol
            size: Order size
            price: Order price
            side: 'buy' or 'sell'
        
        Returns:
            (valid, reason, adjusted_size)
        """
        can_open, reason = self.can_open_position(symbol)
        
        if not can_open and "reduced" not in reason.lower():
            return False, reason, 0.0
        
        # Apply size limits
        adjusted_size = min(size, self.config.max_size_per_position)
        
        # Reduce size if in REDUCED state
        if self.state == RiskState.REDUCED:
            adjusted_size *= 0.5
        
        # Check if closing position (always allowed)
        if symbol in self.open_positions:
            current_size = self.open_positions[symbol]
            if (current_size > 0 and side == 'sell') or (current_size < 0 and side == 'buy'):
                # Closing position - always allowed
                return True, "Closing position", min(adjusted_size, abs(current_size))
        
        return True, "OK", adjusted_size
    
    # ==================== Position Tracking ====================
    
    def on_position_opened(
        self,
        symbol: str,
        size: float,
        entry_price: float
    ) -> None:
        """Track new position"""
        self.open_positions[symbol] = size
        self.position_pnl[symbol] = 0.0
        self.trade_count_today += 1
        self.last_trade_time = time.time()
        
        # Update daily stats
        self._update_daily_pnl()
    
    def on_position_closed(
        self,
        symbol: str,
        realized_pnl: float
    ) -> None:
        """Track closed position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
        if symbol in self.position_pnl:
            del self.position_pnl[symbol]
        
        # Update daily realized P&L
        today = date.today()
        if today not in self.daily_pnl:
            self._init_daily_pnl()
        
        self.daily_pnl[today].realized_pnl += realized_pnl
        self.daily_pnl[today].trades += 1
        
        # Check if loss limit hit
        if self.daily_pnl[today].realized_pnl < 0:
            loss_pct = abs(self.daily_pnl[today].realized_pnl) / self.initial_capital
            if loss_pct >= self.config.max_daily_loss_pct:
                self.trigger_kill_switch("Daily loss limit hit")
    
    def update_position_pnl(self, symbol: str, unrealized_pnl: float) -> None:
        """Update unrealized P&L for position"""
        self.position_pnl[symbol] = unrealized_pnl
        self._update_daily_pnl()
    
    # ==================== Kill Switch ====================
    
    def trigger_kill_switch(self, reason: str) -> None:
        """Activate kill switch"""
        self.state = RiskState.KILL_SWITCH
        self.kill_switch_time = time.time()
        print(f"[RISK] Kill switch activated: {reason}")
    
    def reset_kill_switch(self) -> None:
        """Manually reset kill switch"""
        self.state = RiskState.ACTIVE
        self.kill_switch_time = 0.0
    
    # ==================== Reporting ====================
    
    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report"""
        today = date.today()
        daily = self.daily_pnl.get(today, DailyPnL(today, self.current_capital))
        
        return {
            'state': self.state.value,
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'peak': self.peak_capital,
                'drawdown_pct': self.current_drawdown * 100
            },
            'daily': {
                'realized_pnl': daily.realized_pnl,
                'unrealized_pnl': daily.unrealized_pnl,
                'total_pnl': daily.realized_pnl + daily.unrealized_pnl,
                'trades': daily.trades
            },
            'positions': {
                'count': len(self.open_positions),
                'max': self.config.max_positions,
                'symbols': list(self.open_positions.keys()),
                'total_exposure': self._get_total_exposure()
            },
            'limits': {
                'max_daily_loss_pct': self.config.max_daily_loss_pct * 100,
                'max_drawdown_pct': self.config.max_drawdown_pct * 100,
                'max_exposure_pct': self.config.max_total_exposure_pct * 100,
                'max_leverage': self.config.max_leverage
            }
        }
    
    # ==================== Internal ====================
    
    def _get_total_exposure(self) -> float:
        """Get total position exposure"""
        # This should be updated with actual position values
        return sum(abs(size) for size in self.open_positions.values()) * 50000  # Rough estimate
    
    def _init_daily_pnl(self) -> None:
        """Initialize today's P&L tracking"""
        today = date.today()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = DailyPnL(
                date=today,
                starting_balance=self.current_capital
            )
    
    def _update_daily_pnl(self) -> None:
        """Update daily unrealized P&L"""
        self._init_daily_pnl()
        today = date.today()
        self.daily_pnl[today].unrealized_pnl = sum(self.position_pnl.values())


# ==================== Convenience Functions ====================

def compute_futures_size(
    capital: float,
    price: float,
    atr: float,
    risk_pct: float = 0.02,
    atr_multiple: float = 2.0,
    leverage: int = 3,
    contract_size: float = 1.0
) -> PositionSize:
    """
    Standalone function to compute futures position size
    
    Args:
        capital: Available capital
        price: Current price
        atr: Average True Range
        risk_pct: Risk per trade as percentage
        atr_multiple: ATR multiple for stop loss
        leverage: Leverage to use
        contract_size: Contract size
    
    Returns:
        PositionSize calculation
    """
    risk_amount = capital * risk_pct
    stop_distance = atr * atr_multiple
    
    if stop_distance == 0:
        return PositionSize(
            size=0, contracts=0, notional=0, margin_required=0,
            risk_amount=risk_amount, leverage=leverage, valid=False,
            reason="ATR is zero"
        )
    
    size = risk_amount / stop_distance
    contracts = int(size / contract_size)
    notional = contracts * contract_size * price
    margin_required = notional / leverage
    
    return PositionSize(
        size=contracts * contract_size,
        contracts=contracts,
        notional=notional,
        margin_required=margin_required,
        risk_amount=risk_amount,
        leverage=leverage,
        valid=contracts > 0
    )


def compute_option_contracts(
    capital: float,
    premium: float,
    max_premium_pct: float = 0.005,
    contract_size: float = 1.0
) -> PositionSize:
    """
    Standalone function to compute option contracts
    
    Args:
        capital: Available capital
        premium: Option premium per contract
        max_premium_pct: Max percentage of capital for premium
        contract_size: Contract size
    
    Returns:
        PositionSize calculation
    """
    premium_budget = capital * max_premium_pct
    
    if premium <= 0:
        return PositionSize(
            size=0, contracts=0, notional=0, margin_required=0,
            risk_amount=premium_budget, leverage=1, valid=False,
            reason="Invalid premium"
        )
    
    total_premium = premium * contract_size
    contracts = int(premium_budget / total_premium)
    notional = contracts * total_premium
    
    return PositionSize(
        size=contracts * contract_size,
        contracts=contracts,
        notional=notional,
        margin_required=notional,
        risk_amount=notional,
        leverage=1,
        valid=contracts > 0
    )
