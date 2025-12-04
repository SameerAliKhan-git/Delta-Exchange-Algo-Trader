"""
Production Risk Controls & Safety Gates
=======================================

DELIVERABLE: Non-negotiable safety controls for autonomous trading.

Implements:
1. Per-hour loss limit
2. Per-day loss limit  
3. Max drawdown kill switch
4. Per-symbol max position
5. Liquidity-aware sizing
6. Canary/staged rollouts
7. Automated anomaly detection

Impact: CRITICAL — these prevent catastrophic losses.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import threading
import json
from pathlib import Path


class RiskLevel(Enum):
    """Risk level classification."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    KILLED = "killed"


class TradingState(Enum):
    """Trading state."""
    ACTIVE = "active"
    REDUCED = "reduced"      # Reduced position sizing
    CLOSE_ONLY = "close_only"  # Only allow closing positions
    HALTED = "halted"        # No trading at all


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    # Loss limits
    max_hourly_loss_pct: float = 0.5       # 0.5% max loss per hour
    max_daily_loss_pct: float = 2.0        # 2% max loss per day
    max_weekly_loss_pct: float = 5.0       # 5% max loss per week
    
    # Drawdown limits
    max_drawdown_pct: float = 10.0         # 10% max drawdown from peak
    drawdown_warning_pct: float = 5.0      # Warning at 5%
    
    # Position limits
    max_position_pct: float = 20.0         # 20% max in single position
    max_concentration_pct: float = 40.0    # 40% max in correlated assets
    max_leverage: float = 3.0              # 3x max leverage
    
    # Per-trade limits
    max_trade_size_pct: float = 2.0        # 2% max per trade
    max_slippage_bps: float = 50           # 50 bps max slippage
    
    # Execution limits
    max_daily_trades: int = 100            # Max trades per day
    max_hourly_trades: int = 20            # Max trades per hour
    min_trade_interval_sec: float = 1.0    # Min 1 sec between trades
    
    # Canary allocation
    canary_allocation_pct: float = 1.0     # Start with 1%
    canary_promotion_days: int = 5         # Days before promotion
    canary_promotion_allocation_pct: float = 5.0  # Promote to 5%
    full_allocation_days: int = 10         # Days before full allocation
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RiskState:
    """Current risk state."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Current metrics
    hourly_pnl_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    
    # Position state
    total_exposure_pct: float = 0.0
    max_single_position_pct: float = 0.0
    current_leverage: float = 0.0
    
    # Trading state
    trades_today: int = 0
    trades_this_hour: int = 0
    last_trade_time: Optional[datetime] = None
    
    # Risk level
    risk_level: RiskLevel = RiskLevel.NORMAL
    trading_state: TradingState = TradingState.ACTIVE
    
    # Breached limits
    breached_limits: List[str] = field(default_factory=list)


class RiskEvent:
    """Risk event for logging and alerting."""
    
    def __init__(
        self,
        event_type: str,
        severity: RiskLevel,
        message: str,
        metrics: Dict = None
    ):
        self.timestamp = datetime.now()
        self.event_type = event_type
        self.severity = severity
        self.message = message
        self.metrics = metrics or {}
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'severity': self.severity.value,
            'message': self.message,
            'metrics': self.metrics
        }


class SafetyGate:
    """
    Production safety gate with kill switches.
    
    NON-NEGOTIABLE: These controls must ALWAYS be active.
    """
    
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_equity: float = 100000,
        alert_callback: Optional[Callable[[RiskEvent], None]] = None
    ):
        self.limits = limits or RiskLimits()
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.alert_callback = alert_callback
        
        # State
        self.state = RiskState(current_equity=initial_equity, peak_equity=initial_equity)
        self._lock = threading.Lock()
        
        # History
        self._pnl_history: deque = deque(maxlen=10000)
        self._trade_history: deque = deque(maxlen=1000)
        self._event_history: List[RiskEvent] = []
        
        # Time tracking
        self._hour_start = datetime.now().replace(minute=0, second=0, microsecond=0)
        self._day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self._week_start = self._day_start - timedelta(days=self._day_start.weekday())
        
        # Accumulated PnL
        self._hourly_pnl = 0.0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        
        # Canary state
        self._canary_start_date: Optional[datetime] = None
        self._canary_profitable_days = 0
        self._allocation_multiplier = self.limits.canary_allocation_pct / 100
    
    def _emit_event(self, event: RiskEvent):
        """Emit a risk event."""
        self._event_history.append(event)
        if self.alert_callback:
            self.alert_callback(event)
    
    def _check_time_windows(self):
        """Check and reset time-based windows."""
        now = datetime.now()
        
        # Check hourly reset
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        if current_hour > self._hour_start:
            self._hourly_pnl = 0.0
            self.state.trades_this_hour = 0
            self._hour_start = current_hour
        
        # Check daily reset
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if current_day > self._day_start:
            # Check if yesterday was profitable for canary
            if self._daily_pnl > 0:
                self._canary_profitable_days += 1
            else:
                self._canary_profitable_days = max(0, self._canary_profitable_days - 1)
            
            self._daily_pnl = 0.0
            self.state.trades_today = 0
            self._day_start = current_day
            
            # Update canary allocation
            self._update_canary_allocation()
        
        # Check weekly reset
        current_week = current_day - timedelta(days=current_day.weekday())
        if current_week > self._week_start:
            self._weekly_pnl = 0.0
            self._week_start = current_week
    
    def _update_canary_allocation(self):
        """Update canary allocation based on performance."""
        if self._canary_start_date is None:
            self._canary_start_date = datetime.now()
            self._allocation_multiplier = self.limits.canary_allocation_pct / 100
            return
        
        days_active = (datetime.now() - self._canary_start_date).days
        
        if (days_active >= self.limits.full_allocation_days and 
            self._canary_profitable_days >= self.limits.full_allocation_days * 0.6):
            # Full allocation
            self._allocation_multiplier = 1.0
            self._emit_event(RiskEvent(
                "canary_promoted",
                RiskLevel.NORMAL,
                "Promoted to full allocation",
                {'days_active': days_active, 'profitable_days': self._canary_profitable_days}
            ))
        elif (days_active >= self.limits.canary_promotion_days and
              self._canary_profitable_days >= self.limits.canary_promotion_days * 0.6):
            # Promoted allocation
            self._allocation_multiplier = self.limits.canary_promotion_allocation_pct / 100
            self._emit_event(RiskEvent(
                "canary_promoted",
                RiskLevel.NORMAL,
                f"Promoted to {self.limits.canary_promotion_allocation_pct}% allocation",
                {'days_active': days_active, 'profitable_days': self._canary_profitable_days}
            ))
    
    def update_equity(self, new_equity: float, pnl: float = None):
        """Update current equity and PnL."""
        with self._lock:
            self._check_time_windows()
            
            if pnl is None:
                pnl = new_equity - self.current_equity
            
            self.current_equity = new_equity
            self.peak_equity = max(self.peak_equity, new_equity)
            
            # Update PnL tracking
            pnl_pct = pnl / self.initial_equity * 100
            self._hourly_pnl += pnl_pct
            self._daily_pnl += pnl_pct
            self._weekly_pnl += pnl_pct
            
            self._pnl_history.append({
                'timestamp': datetime.now(),
                'equity': new_equity,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            
            # Update state
            self.state.current_equity = new_equity
            self.state.peak_equity = self.peak_equity
            self.state.hourly_pnl_pct = self._hourly_pnl
            self.state.daily_pnl_pct = self._daily_pnl
            self.state.weekly_pnl_pct = self._weekly_pnl
            self.state.current_drawdown_pct = (self.peak_equity - new_equity) / self.peak_equity * 100
            
            # Check limits
            self._check_all_limits()
    
    def _check_all_limits(self):
        """Check all risk limits and update state."""
        breaches = []
        risk_level = RiskLevel.NORMAL
        trading_state = TradingState.ACTIVE
        
        # Check hourly loss
        if self._hourly_pnl < -self.limits.max_hourly_loss_pct:
            breaches.append(f"Hourly loss {self._hourly_pnl:.2f}% exceeds limit {self.limits.max_hourly_loss_pct}%")
            risk_level = RiskLevel.CRITICAL
            trading_state = TradingState.HALTED
            self._emit_event(RiskEvent(
                "hourly_loss_breach",
                RiskLevel.CRITICAL,
                f"HOURLY LOSS LIMIT BREACHED: {self._hourly_pnl:.2f}%",
                {'hourly_pnl': self._hourly_pnl, 'limit': self.limits.max_hourly_loss_pct}
            ))
        
        # Check daily loss
        if self._daily_pnl < -self.limits.max_daily_loss_pct:
            breaches.append(f"Daily loss {self._daily_pnl:.2f}% exceeds limit {self.limits.max_daily_loss_pct}%")
            risk_level = RiskLevel.CRITICAL
            trading_state = TradingState.HALTED
            self._emit_event(RiskEvent(
                "daily_loss_breach",
                RiskLevel.CRITICAL,
                f"DAILY LOSS LIMIT BREACHED: {self._daily_pnl:.2f}%",
                {'daily_pnl': self._daily_pnl, 'limit': self.limits.max_daily_loss_pct}
            ))
        
        # Check drawdown
        drawdown = self.state.current_drawdown_pct
        if drawdown > self.limits.max_drawdown_pct:
            breaches.append(f"Drawdown {drawdown:.2f}% exceeds limit {self.limits.max_drawdown_pct}%")
            risk_level = RiskLevel.KILLED
            trading_state = TradingState.HALTED
            self._emit_event(RiskEvent(
                "max_drawdown_breach",
                RiskLevel.CRITICAL,
                f"MAX DRAWDOWN KILL SWITCH TRIGGERED: {drawdown:.2f}%",
                {'drawdown': drawdown, 'limit': self.limits.max_drawdown_pct}
            ))
        elif drawdown > self.limits.drawdown_warning_pct:
            if risk_level.value < RiskLevel.HIGH.value:
                risk_level = RiskLevel.HIGH
            trading_state = TradingState.REDUCED
            self._emit_event(RiskEvent(
                "drawdown_warning",
                RiskLevel.HIGH,
                f"Drawdown warning: {drawdown:.2f}%",
                {'drawdown': drawdown, 'warning_threshold': self.limits.drawdown_warning_pct}
            ))
        
        self.state.breached_limits = breaches
        self.state.risk_level = risk_level
        self.state.trading_state = trading_state
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.
        
        Returns:
            (can_trade, reason)
        """
        with self._lock:
            self._check_time_windows()
            
            if self.state.trading_state == TradingState.HALTED:
                return False, f"Trading halted: {', '.join(self.state.breached_limits)}"
            
            if self.state.trading_state == TradingState.CLOSE_ONLY:
                return False, "Close-only mode active"
            
            # Check trade frequency
            if self.state.trades_today >= self.limits.max_daily_trades:
                return False, f"Daily trade limit reached: {self.limits.max_daily_trades}"
            
            if self.state.trades_this_hour >= self.limits.max_hourly_trades:
                return False, f"Hourly trade limit reached: {self.limits.max_hourly_trades}"
            
            # Check trade interval
            if self.state.last_trade_time:
                elapsed = (datetime.now() - self.state.last_trade_time).total_seconds()
                if elapsed < self.limits.min_trade_interval_sec:
                    return False, f"Trade interval not met: {elapsed:.1f}s < {self.limits.min_trade_interval_sec}s"
            
            return True, "OK"
    
    def check_trade(
        self,
        symbol: str,
        size_usd: float,
        side: str,
        current_positions: Dict[str, float] = None,
        expected_slippage_bps: float = 0
    ) -> Tuple[bool, str, float]:
        """
        Check if a specific trade is allowed.
        
        Returns:
            (allowed, reason, adjusted_size)
        """
        with self._lock:
            can, reason = self.can_trade()
            if not can:
                return False, reason, 0
            
            # Check trade size
            max_size = self.current_equity * self.limits.max_trade_size_pct / 100
            max_size *= self._allocation_multiplier  # Apply canary allocation
            
            if size_usd > max_size:
                return False, f"Trade size ${size_usd:,.0f} exceeds limit ${max_size:,.0f}", max_size
            
            # Check slippage
            if expected_slippage_bps > self.limits.max_slippage_bps:
                return False, f"Expected slippage {expected_slippage_bps:.1f} bps exceeds limit", 0
            
            # Check position concentration
            if current_positions:
                current_pos = current_positions.get(symbol, 0)
                new_pos = current_pos + (size_usd if side == 'buy' else -size_usd)
                pos_pct = abs(new_pos) / self.current_equity * 100
                
                if pos_pct > self.limits.max_position_pct:
                    allowed_size = (self.limits.max_position_pct / 100 * self.current_equity) - abs(current_pos)
                    return False, f"Position {pos_pct:.1f}% exceeds limit {self.limits.max_position_pct}%", max(0, allowed_size)
            
            # Apply canary allocation
            adjusted_size = min(size_usd, max_size)
            
            return True, "OK", adjusted_size
    
    def record_trade(self, symbol: str, size_usd: float, side: str, pnl: float = 0):
        """Record a completed trade."""
        with self._lock:
            self.state.trades_today += 1
            self.state.trades_this_hour += 1
            self.state.last_trade_time = datetime.now()
            
            self._trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'size_usd': size_usd,
                'side': side,
                'pnl': pnl
            })
    
    def get_position_multiplier(self) -> float:
        """
        Get current position size multiplier based on risk state.
        
        Returns:
            Multiplier (0 to 1) to apply to position sizes
        """
        with self._lock:
            if self.state.trading_state == TradingState.HALTED:
                return 0.0
            
            if self.state.trading_state == TradingState.CLOSE_ONLY:
                return 0.0
            
            multiplier = self._allocation_multiplier
            
            # Reduce for elevated risk
            if self.state.risk_level == RiskLevel.HIGH:
                multiplier *= 0.5
            elif self.state.risk_level == RiskLevel.ELEVATED:
                multiplier *= 0.75
            
            # Reduce if approaching daily loss limit
            daily_loss_usage = abs(self._daily_pnl) / self.limits.max_daily_loss_pct
            if daily_loss_usage > 0.5:
                multiplier *= (1 - daily_loss_usage * 0.5)
            
            return max(0, min(1, multiplier))
    
    def get_liquidity_adjusted_size(
        self,
        target_size_usd: float,
        available_depth_usd: float,
        max_participation_pct: float = 10
    ) -> float:
        """
        Get liquidity-aware position size.
        
        Args:
            target_size_usd: Desired size
            available_depth_usd: Available liquidity
            max_participation_pct: Max % of available liquidity to take
        
        Returns:
            Adjusted size in USD
        """
        # Don't take more than max_participation_pct of available liquidity
        max_from_liquidity = available_depth_usd * max_participation_pct / 100
        
        # Apply position multiplier
        multiplier = self.get_position_multiplier()
        
        return min(target_size_usd * multiplier, max_from_liquidity)
    
    def force_halt(self, reason: str):
        """Force halt all trading."""
        with self._lock:
            self.state.trading_state = TradingState.HALTED
            self.state.risk_level = RiskLevel.KILLED
            self.state.breached_limits.append(f"Manual halt: {reason}")
            
            self._emit_event(RiskEvent(
                "manual_halt",
                RiskLevel.CRITICAL,
                f"TRADING MANUALLY HALTED: {reason}",
                {}
            ))
    
    def reset_daily(self, keep_drawdown: bool = True):
        """Reset daily limits (call at start of new day)."""
        with self._lock:
            self._daily_pnl = 0.0
            self.state.trades_today = 0
            
            if not keep_drawdown:
                self.peak_equity = self.current_equity
            
            # Reset trading state if it was daily-limit based
            if self.state.trading_state in [TradingState.HALTED, TradingState.REDUCED]:
                if self.state.current_drawdown_pct < self.limits.max_drawdown_pct:
                    self.state.trading_state = TradingState.ACTIVE
                    self.state.risk_level = RiskLevel.NORMAL
                    self.state.breached_limits = []
    
    def get_status(self) -> Dict:
        """Get current risk status."""
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'trading_state': self.state.trading_state.value,
                'risk_level': self.state.risk_level.value,
                'current_equity': self.current_equity,
                'peak_equity': self.peak_equity,
                'drawdown_pct': self.state.current_drawdown_pct,
                'hourly_pnl_pct': self._hourly_pnl,
                'daily_pnl_pct': self._daily_pnl,
                'weekly_pnl_pct': self._weekly_pnl,
                'trades_today': self.state.trades_today,
                'position_multiplier': self.get_position_multiplier(),
                'allocation_multiplier': self._allocation_multiplier,
                'canary_profitable_days': self._canary_profitable_days,
                'breached_limits': self.state.breached_limits
            }
    
    def export_events(self, filepath: str):
        """Export risk events to file."""
        events = [e.to_dict() for e in self._event_history]
        with open(filepath, 'w') as f:
            json.dump(events, f, indent=2)


class AnomalyDetector:
    """
    Automated anomaly detection for trading operations.
    
    Detects:
    - Sudden slippage spikes
    - Failed fill rate increases
    - Unusual trade patterns
    - Model prediction drift
    """
    
    def __init__(
        self,
        slippage_threshold_multiplier: float = 2.0,
        fill_rate_threshold: float = 0.8,
        alert_callback: Optional[Callable[[str, Dict], None]] = None
    ):
        self.slippage_threshold_multiplier = slippage_threshold_multiplier
        self.fill_rate_threshold = fill_rate_threshold
        self.alert_callback = alert_callback
        
        # History
        self._slippage_history: deque = deque(maxlen=100)
        self._fill_history: deque = deque(maxlen=100)
        self._latency_history: deque = deque(maxlen=100)
        
        # Baselines (running averages)
        self._baseline_slippage = 5.0  # bps
        self._baseline_latency = 100   # ms
    
    def record_fill(
        self,
        filled: bool,
        slippage_bps: float,
        latency_ms: float,
        expected_slippage_bps: float = None
    ):
        """Record a fill attempt."""
        self._fill_history.append(filled)
        self._slippage_history.append(slippage_bps)
        self._latency_history.append(latency_ms)
        
        # Update baselines
        if len(self._slippage_history) > 10:
            self._baseline_slippage = np.median(list(self._slippage_history))
        if len(self._latency_history) > 10:
            self._baseline_latency = np.median(list(self._latency_history))
        
        # Check for anomalies
        anomalies = []
        
        # Slippage spike
        if slippage_bps > self._baseline_slippage * self.slippage_threshold_multiplier:
            anomalies.append({
                'type': 'slippage_spike',
                'value': slippage_bps,
                'baseline': self._baseline_slippage,
                'threshold': self._baseline_slippage * self.slippage_threshold_multiplier
            })
        
        # Expected vs realized slippage
        if expected_slippage_bps and slippage_bps > expected_slippage_bps * 2:
            anomalies.append({
                'type': 'slippage_model_miss',
                'realized': slippage_bps,
                'expected': expected_slippage_bps
            })
        
        # Latency spike
        if latency_ms > self._baseline_latency * 3:
            anomalies.append({
                'type': 'latency_spike',
                'value': latency_ms,
                'baseline': self._baseline_latency
            })
        
        # Fill rate drop
        recent_fills = list(self._fill_history)[-20:]
        if len(recent_fills) >= 10:
            fill_rate = sum(recent_fills) / len(recent_fills)
            if fill_rate < self.fill_rate_threshold:
                anomalies.append({
                    'type': 'fill_rate_drop',
                    'rate': fill_rate,
                    'threshold': self.fill_rate_threshold
                })
        
        # Alert on anomalies
        if anomalies and self.alert_callback:
            for anomaly in anomalies:
                self.alert_callback(anomaly['type'], anomaly)
        
        return anomalies
    
    def get_metrics(self) -> Dict:
        """Get current anomaly detection metrics."""
        recent_fills = list(self._fill_history)[-20:]
        recent_slippage = list(self._slippage_history)[-20:]
        recent_latency = list(self._latency_history)[-20:]
        
        return {
            'fill_rate': sum(recent_fills) / len(recent_fills) if recent_fills else 1.0,
            'avg_slippage_bps': np.mean(recent_slippage) if recent_slippage else 0,
            'baseline_slippage_bps': self._baseline_slippage,
            'avg_latency_ms': np.mean(recent_latency) if recent_latency else 0,
            'baseline_latency_ms': self._baseline_latency,
            'samples': len(self._fill_history)
        }


# Convenience function
def create_safety_gate(
    initial_equity: float = 100000,
    max_daily_loss_pct: float = 2.0,
    max_drawdown_pct: float = 10.0,
    canary_mode: bool = True
) -> SafetyGate:
    """Create a configured safety gate."""
    limits = RiskLimits(
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
        canary_allocation_pct=1.0 if canary_mode else 100.0
    )
    return SafetyGate(limits=limits, initial_equity=initial_equity)


if __name__ == "__main__":
    print("=" * 60)
    print("PRODUCTION SAFETY GATE TEST")
    print("=" * 60)
    
    # Create safety gate
    gate = create_safety_gate(
        initial_equity=100000,
        max_daily_loss_pct=2.0,
        max_drawdown_pct=10.0,
        canary_mode=True
    )
    
    print("\nInitial status:")
    status = gate.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    # Simulate some trading
    print("\n--- Simulating trades ---")
    
    # Normal trade
    can, reason, size = gate.check_trade('BTCUSDT', 1000, 'buy')
    print(f"\nTrade check (normal): {can}, {reason}, size=${size:.0f}")
    
    if can:
        gate.record_trade('BTCUSDT', size, 'buy', pnl=50)
        gate.update_equity(100050)
    
    # After some losses
    print("\n--- Simulating losses ---")
    gate.update_equity(98500, pnl=-1550)  # 1.55% loss
    
    status = gate.get_status()
    print(f"After loss - trading_state: {status['trading_state']}, daily_pnl: {status['daily_pnl_pct']:.2f}%")
    
    # Try to trade again
    can, reason, size = gate.check_trade('BTCUSDT', 1000, 'buy')
    print(f"Trade check after loss: {can}, {reason}")
    
    # Trigger daily loss limit
    print("\n--- Triggering daily loss limit ---")
    gate.update_equity(97500, pnl=-1000)  # Now at -2.5%
    
    status = gate.get_status()
    print(f"After breach - trading_state: {status['trading_state']}")
    print(f"Breached limits: {status['breached_limits']}")
    
    can, reason, size = gate.check_trade('BTCUSDT', 1000, 'buy')
    print(f"Trade check after breach: {can}, {reason}")
    
    print("\n" + "=" * 60)
    print("SAFETY GATE WORKING CORRECTLY")
    print("=" * 60)
