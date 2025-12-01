"""
Risk Manager for Delta Exchange Algo Trading Bot
Enforces strict risk controls and position sizing
"""

import time
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import threading

from config import get_config
from logger import get_logger, get_audit_logger
from delta_client import get_delta_client


class RiskViolation(Enum):
    """Types of risk violations"""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_POSITION_SIZE = "max_position_size"
    MAX_OPEN_POSITIONS = "max_open_positions"
    COOLDOWN_ACTIVE = "cooldown_active"
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    INVALID_SIZE = "invalid_size"


@dataclass
class RiskCheckResult:
    """Result of a risk check"""
    allowed: bool
    violation: Optional[RiskViolation] = None
    message: str = ""
    adjusted_size: Optional[float] = None


@dataclass
class TradeRecord:
    """Record of a completed trade"""
    trade_id: str
    timestamp: datetime
    product_id: int
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    is_closed: bool = False


@dataclass 
class DailyStats:
    """Daily trading statistics"""
    date: date
    total_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0


class RiskManager:
    """
    Risk management engine
    
    Responsibilities:
    - Position sizing based on ATR and risk per trade
    - Daily loss limit enforcement
    - Maximum position size limits
    - Maximum concurrent positions
    - Trade cooldown management
    - Kill switch monitoring
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.audit = get_audit_logger()
        self.client = get_delta_client()
        
        # Risk parameters from config
        self.risk_per_trade = self.config.risk.risk_per_trade_inr
        self.max_daily_loss = self.config.risk.max_daily_loss_inr
        self.max_open_positions = self.config.risk.max_open_positions
        self.min_cooldown_seconds = self.config.risk.min_trade_cooldown_seconds
        self.max_position_size = self.config.risk.max_position_size
        self.default_stop_loss_pct = self.config.risk.default_stop_loss_pct
        
        # State tracking
        self._daily_stats: Dict[date, DailyStats] = {}
        self._trade_records: List[TradeRecord] = []
        self._last_trade_time: float = 0
        self._open_position_count: int = 0
        self._kill_switch_active: bool = False
        self._lock = threading.Lock()
        
        # Initialize today's stats
        self._get_or_create_daily_stats(date.today())
    
    def _get_or_create_daily_stats(self, day: date) -> DailyStats:
        """Get or create daily statistics"""
        if day not in self._daily_stats:
            self._daily_stats[day] = DailyStats(date=day)
        return self._daily_stats[day]
    
    @property
    def today_stats(self) -> DailyStats:
        """Get today's statistics"""
        return self._get_or_create_daily_stats(date.today())
    
    @property
    def daily_pnl(self) -> float:
        """Get today's PnL"""
        return self.today_stats.total_pnl
    
    @property
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active"""
        return self._kill_switch_active or self._check_kill_switch_file()
    
    def _check_kill_switch_file(self) -> bool:
        """Check if kill switch file exists"""
        import os
        kill_switch_path = self.config.system.kill_switch_file
        return os.path.exists(kill_switch_path)
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate the kill switch"""
        self._kill_switch_active = True
        self.logger.critical("KILL SWITCH ACTIVATED", reason=reason)
        self.audit.log_kill_switch(reason)
        
        # Create kill switch file
        try:
            with open(self.config.system.kill_switch_file, 'w') as f:
                f.write(f"Kill switch activated: {datetime.utcnow().isoformat()}\nReason: {reason}")
        except Exception as e:
            self.logger.error("Failed to create kill switch file", error=str(e))
    
    def deactivate_kill_switch(self):
        """Deactivate the kill switch"""
        import os
        self._kill_switch_active = False
        
        # Remove kill switch file
        try:
            kill_switch_path = self.config.system.kill_switch_file
            if os.path.exists(kill_switch_path):
                os.remove(kill_switch_path)
            self.logger.info("Kill switch deactivated")
        except Exception as e:
            self.logger.error("Failed to remove kill switch file", error=str(e))
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        risk_amount: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk per trade and stop distance
        
        Formula: size = risk_amount / (entry_price - stop_price)
        
        Args:
            entry_price: Expected entry price
            stop_price: Stop loss price
            risk_amount: Amount to risk (defaults to config)
        
        Returns:
            Position size (capped at max_position_size)
        """
        risk = risk_amount or self.risk_per_trade
        stop_distance = abs(entry_price - stop_price)
        
        if stop_distance <= 0:
            self.logger.warning("Invalid stop distance", entry=entry_price, stop=stop_price)
            return 0.0
        
        # Calculate size
        size = risk / stop_distance
        
        # Apply floor and ceiling
        min_size = 0.0001  # Minimum tradeable size
        size = max(min_size, min(size, self.max_position_size))
        
        self.logger.debug(
            "Position size calculated",
            entry_price=entry_price,
            stop_price=stop_price,
            risk=risk,
            size=size
        )
        
        return round(size, 4)
    
    def calculate_position_size_atr(
        self,
        price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        risk_amount: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Calculate position size and stop price based on ATR
        
        Args:
            price: Current price
            atr: Average True Range
            atr_multiplier: ATR multiplier for stop distance
            risk_amount: Amount to risk
        
        Returns:
            (position_size, stop_price)
        """
        stop_distance = atr * atr_multiplier
        stop_price = price - stop_distance  # For long; flip for short
        
        size = self.calculate_position_size(price, stop_price, risk_amount)
        
        return size, stop_price
    
    def check_can_trade(self) -> RiskCheckResult:
        """
        Comprehensive check if trading is allowed
        
        Checks:
        - Kill switch
        - Daily loss limit
        - Cooldown period
        - Max open positions
        """
        with self._lock:
            # Check kill switch
            if self.is_kill_switch_active:
                return RiskCheckResult(
                    allowed=False,
                    violation=RiskViolation.KILL_SWITCH_ACTIVE,
                    message="Kill switch is active - trading disabled"
                )
            
            # Check daily loss limit
            if self.daily_pnl <= -abs(self.max_daily_loss):
                self.logger.critical(
                    "Daily loss limit reached",
                    daily_pnl=self.daily_pnl,
                    limit=self.max_daily_loss
                )
                self.audit.log_risk_violation({
                    "type": "daily_loss_limit",
                    "daily_pnl": self.daily_pnl,
                    "limit": self.max_daily_loss
                })
                return RiskCheckResult(
                    allowed=False,
                    violation=RiskViolation.DAILY_LOSS_LIMIT,
                    message=f"Daily loss limit reached: {self.daily_pnl:.2f} / -{self.max_daily_loss:.2f}"
                )
            
            # Check cooldown
            time_since_last = time.time() - self._last_trade_time
            if time_since_last < self.min_cooldown_seconds:
                remaining = self.min_cooldown_seconds - time_since_last
                return RiskCheckResult(
                    allowed=False,
                    violation=RiskViolation.COOLDOWN_ACTIVE,
                    message=f"Cooldown active: {remaining:.0f}s remaining"
                )
            
            # Check max open positions
            open_positions = self._get_open_position_count()
            if open_positions >= self.max_open_positions:
                return RiskCheckResult(
                    allowed=False,
                    violation=RiskViolation.MAX_OPEN_POSITIONS,
                    message=f"Max positions reached: {open_positions}/{self.max_open_positions}"
                )
            
            return RiskCheckResult(allowed=True, message="All checks passed")
    
    def check_position_size(self, size: float, price: float) -> RiskCheckResult:
        """Check if position size is within limits"""
        if size <= 0:
            return RiskCheckResult(
                allowed=False,
                violation=RiskViolation.INVALID_SIZE,
                message="Position size must be positive"
            )
        
        if size > self.max_position_size:
            return RiskCheckResult(
                allowed=False,
                violation=RiskViolation.MAX_POSITION_SIZE,
                message=f"Size {size} exceeds max {self.max_position_size}",
                adjusted_size=self.max_position_size
            )
        
        # Check if we have sufficient balance (basic check)
        try:
            balances = self.client.get_wallet_balances()
            # This is a simplified check - actual margin requirements may vary
            available = sum(float(b.get('available_balance', 0)) for b in balances)
            required = size * price * 0.1  # Assume 10x leverage
            
            if required > available:
                return RiskCheckResult(
                    allowed=False,
                    violation=RiskViolation.INSUFFICIENT_BALANCE,
                    message=f"Insufficient balance: need {required:.2f}, have {available:.2f}"
                )
        except Exception as e:
            self.logger.warning("Balance check failed", error=str(e))
        
        return RiskCheckResult(allowed=True, message="Size check passed")
    
    def _get_open_position_count(self) -> int:
        """Get count of open positions from exchange"""
        try:
            positions = self.client.get_positions()
            count = sum(1 for p in positions if abs(float(p.get('size', 0))) > 0)
            self._open_position_count = count
            return count
        except Exception as e:
            self.logger.error("Failed to get positions", error=str(e))
            return self._open_position_count
    
    def record_trade_entry(
        self,
        trade_id: str,
        product_id: int,
        side: str,
        size: float,
        entry_price: float
    ):
        """Record a trade entry"""
        with self._lock:
            record = TradeRecord(
                trade_id=trade_id,
                timestamp=datetime.utcnow(),
                product_id=product_id,
                side=side,
                size=size,
                entry_price=entry_price
            )
            self._trade_records.append(record)
            self._last_trade_time = time.time()
            
            self.logger.log_trade(
                action="entry",
                trade_id=trade_id,
                product_symbol=str(product_id),
                side=side,
                size=size,
                entry_price=entry_price
            )
    
    def record_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float
    ):
        """Record a trade exit and update PnL"""
        with self._lock:
            # Find and update the trade record
            for record in reversed(self._trade_records):
                if record.trade_id == trade_id and not record.is_closed:
                    record.exit_price = exit_price
                    record.pnl = pnl
                    record.is_closed = True
                    break
            
            # Update daily stats
            stats = self.today_stats
            stats.total_pnl += pnl
            stats.trade_count += 1
            
            if pnl > 0:
                stats.win_count += 1
                stats.largest_win = max(stats.largest_win, pnl)
            else:
                stats.loss_count += 1
                stats.largest_loss = min(stats.largest_loss, pnl)
            
            self.logger.log_trade(
                action="exit",
                trade_id=trade_id,
                exit_price=exit_price,
                pnl=pnl
            )
            
            self.logger.log_risk_event(
                event_type="daily_pnl_update",
                details={
                    "trade_pnl": pnl,
                    "daily_pnl": stats.total_pnl,
                    "trade_count": stats.trade_count,
                    "win_rate": stats.win_count / stats.trade_count if stats.trade_count > 0 else 0
                }
            )
            
            # Check if daily loss limit hit after this trade
            if stats.total_pnl <= -abs(self.max_daily_loss):
                self.activate_kill_switch(f"Daily loss limit hit: {stats.total_pnl:.2f}")
    
    def get_stop_price(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
        stop_pct: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            side: 'buy' (long) or 'sell' (short)
            atr: ATR for dynamic stop (optional)
            stop_pct: Fixed stop percentage (optional)
        
        Returns:
            Stop loss price
        """
        pct = stop_pct or self.default_stop_loss_pct
        
        if atr:
            # ATR-based stop (2x ATR by default)
            stop_distance = max(atr * 2, entry_price * pct)
        else:
            # Percentage-based stop
            stop_distance = entry_price * pct
        
        if side == 'buy':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def get_take_profit_price(
        self,
        entry_price: float,
        side: str,
        risk_reward_ratio: float = 2.0,
        stop_price: Optional[float] = None
    ) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            side: 'buy' (long) or 'sell' (short)
            risk_reward_ratio: Target R:R ratio
            stop_price: Stop price (for R:R calculation)
        
        Returns:
            Take profit price
        """
        if stop_price:
            stop_distance = abs(entry_price - stop_price)
            tp_distance = stop_distance * risk_reward_ratio
        else:
            tp_pct = self.config.risk.default_take_profit_pct
            tp_distance = entry_price * tp_pct
        
        if side == 'buy':
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance
    
    def get_trailing_stop_params(
        self,
        entry_price: float,
        side: str
    ) -> Dict[str, float]:
        """Get trailing stop parameters"""
        trail_pct = self.config.risk.trailing_stop_pct
        
        return {
            "trail_amount": entry_price * trail_pct,
            "activation_price": entry_price * (1 + trail_pct) if side == 'buy' else entry_price * (1 - trail_pct)
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        stats = self.today_stats
        
        return {
            "daily_pnl": stats.total_pnl,
            "daily_loss_limit": self.max_daily_loss,
            "daily_loss_remaining": self.max_daily_loss + stats.total_pnl,
            "trade_count_today": stats.trade_count,
            "win_rate": stats.win_count / stats.trade_count if stats.trade_count > 0 else 0,
            "largest_win": stats.largest_win,
            "largest_loss": stats.largest_loss,
            "open_positions": self._open_position_count,
            "max_positions": self.max_open_positions,
            "cooldown_remaining": max(0, self.min_cooldown_seconds - (time.time() - self._last_trade_time)),
            "kill_switch_active": self.is_kill_switch_active
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new trading day)"""
        today = date.today()
        if today not in self._daily_stats:
            self._daily_stats[today] = DailyStats(date=today)
            self.logger.info("Daily stats reset for new trading day")


# Singleton instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get or create the global risk manager"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager


if __name__ == "__main__":
    # Test risk manager
    rm = get_risk_manager()
    
    # Test position sizing
    size, stop = rm.calculate_position_size_atr(
        price=50000,
        atr=1000,
        atr_multiplier=2.0
    )
    print(f"Position size: {size}, Stop: {stop}")
    
    # Test can trade check
    result = rm.check_can_trade()
    print(f"Can trade: {result.allowed}, Message: {result.message}")
    
    # Test risk metrics
    metrics = rm.get_risk_metrics()
    print(f"Risk metrics: {metrics}")
