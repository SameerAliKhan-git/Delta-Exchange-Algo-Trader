"""
ALADDIN - Enhanced Risk Engine
================================
Advanced risk management with circuit breakers and kill switch.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import json


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    HALF_OPEN = "half_open"  # Testing recovery
    OPEN = "open"          # Trading halted


@dataclass
class RiskAlert:
    """Risk alert event."""
    level: RiskLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class CircuitBreaker:
    """Individual circuit breaker for a specific risk metric."""
    name: str
    threshold: float
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    trip_count: int = 0
    last_trip: Optional[datetime] = None
    reset_after: int = 300  # seconds to reset
    
    def check(self, current_value: float) -> bool:
        """
        Check if circuit breaker should trip.
        Returns True if trading should be halted.
        """
        if self.state == CircuitBreakerState.OPEN:
            # Check if we should try half-open
            if self.last_trip and (datetime.now() - self.last_trip).seconds > self.reset_after:
                self.state = CircuitBreakerState.HALF_OPEN
                return False  # Allow test trade
            return True  # Still open
        
        if abs(current_value) >= self.threshold:
            self.trip()
            return True
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Successful trade, close circuit
            self.state = CircuitBreakerState.CLOSED
        
        return False
    
    def trip(self):
        """Trip the circuit breaker."""
        self.state = CircuitBreakerState.OPEN
        self.trip_count += 1
        self.last_trip = datetime.now()
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.last_trip = None


class RiskDesk:
    """
    Advanced risk management desk.
    
    Features:
    - Kill switch for emergency trading halt
    - Circuit breakers for individual risk metrics
    - Position and exposure limits
    - Real-time risk monitoring
    - Alert system
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger('Aladdin.RiskDesk')
        self.config = config or self._default_config()
        
        # Kill switch
        self._kill_switch_active = False
        self._kill_switch_reason = ""
        self._kill_switch_time = None
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._init_circuit_breakers()
        
        # Alerts
        self.alerts: List[RiskAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Metrics tracking
        self._metrics: Dict[str, float] = {}
        self._last_update = datetime.now()
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def _default_config(self) -> Dict:
        return {
            # Position limits
            'max_positions': 5,
            'max_position_value_pct': 20.0,  # Max 20% of portfolio per position
            'max_total_exposure_pct': 100.0,  # Max 100% exposure
            
            # Loss limits
            'max_daily_loss_pct': 5.0,
            'max_weekly_loss_pct': 10.0,
            'max_monthly_loss_pct': 15.0,
            'max_drawdown_pct': 20.0,
            
            # Volatility limits
            'max_portfolio_var': 0.05,  # 5% VaR
            'max_single_trade_loss_pct': 2.0,
            
            # Circuit breaker thresholds
            'circuit_breakers': {
                'daily_loss': {'threshold': 3.0, 'reset_after': 3600},
                'hourly_loss': {'threshold': 2.0, 'reset_after': 1800},
                'consecutive_losses': {'threshold': 5, 'reset_after': 3600},
                'volatility_spike': {'threshold': 2.0, 'reset_after': 900},
                'exposure_limit': {'threshold': 80.0, 'reset_after': 300},
            },
            
            # Kill switch
            'auto_kill_switch': True,
            'kill_switch_threshold_pct': 10.0,  # Auto kill at 10% daily loss
        }
    
    def _init_circuit_breakers(self):
        """Initialize circuit breakers from config."""
        cb_config = self.config.get('circuit_breakers', {})
        
        for name, params in cb_config.items():
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                threshold=params.get('threshold', 100),
                reset_after=params.get('reset_after', 300)
            )
    
    # ==================== Kill Switch ====================
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate emergency kill switch to halt all trading."""
        with self._lock:
            self._kill_switch_active = True
            self._kill_switch_reason = reason
            self._kill_switch_time = datetime.now()
            
            self.logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
            self._create_alert(
                RiskLevel.EMERGENCY,
                f"Kill switch activated: {reason}",
                "kill_switch",
                1.0,
                0.0
            )
    
    def deactivate_kill_switch(self, confirmation: str = ""):
        """Deactivate kill switch. Requires confirmation."""
        if confirmation != "CONFIRM_RESUME_TRADING":
            self.logger.warning("Kill switch deactivation requires confirmation: 'CONFIRM_RESUME_TRADING'")
            return False
        
        with self._lock:
            self._kill_switch_active = False
            self._kill_switch_reason = ""
            self._kill_switch_time = None
            
            self.logger.info("✅ Kill switch deactivated - trading resumed")
            return True
    
    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active
    
    def get_kill_switch_status(self) -> Dict:
        return {
            'active': self._kill_switch_active,
            'reason': self._kill_switch_reason,
            'activated_at': self._kill_switch_time.isoformat() if self._kill_switch_time else None,
            'duration_minutes': (datetime.now() - self._kill_switch_time).seconds // 60 if self._kill_switch_time else 0
        }
    
    # ==================== Circuit Breakers ====================
    
    def check_circuit_breakers(self, metrics: Dict[str, float]) -> bool:
        """
        Check all circuit breakers with current metrics.
        Returns True if any breaker is tripped (trading should halt).
        """
        any_tripped = False
        
        for name, breaker in self.circuit_breakers.items():
            if name in metrics:
                if breaker.check(metrics[name]):
                    any_tripped = True
                    self.logger.warning(f"⚡ Circuit breaker '{name}' tripped at {metrics[name]}")
                    self._create_alert(
                        RiskLevel.HIGH,
                        f"Circuit breaker '{name}' tripped",
                        name,
                        metrics[name],
                        breaker.threshold
                    )
        
        return any_tripped
    
    def get_circuit_breaker_status(self) -> Dict:
        return {
            name: {
                'state': cb.state.value,
                'threshold': cb.threshold,
                'trip_count': cb.trip_count,
                'last_trip': cb.last_trip.isoformat() if cb.last_trip else None
            }
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_circuit_breaker(self, name: str):
        """Manually reset a specific circuit breaker."""
        if name in self.circuit_breakers:
            self.circuit_breakers[name].reset()
            self.logger.info(f"Circuit breaker '{name}' manually reset")
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()
        self.logger.info("All circuit breakers reset")
    
    # ==================== Risk Checks ====================
    
    def can_trade(self, portfolio: Dict, proposed_trade: Dict = None) -> Tuple[bool, str]:
        """
        Comprehensive check if trading is allowed.
        
        Args:
            portfolio: Current portfolio state
            proposed_trade: Details of proposed trade (optional)
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Kill switch check
        if self._kill_switch_active:
            return False, f"Kill switch active: {self._kill_switch_reason}"
        
        # Extract portfolio values
        equity = portfolio.get('total_equity', 0)
        daily_pnl = portfolio.get('daily_pnl', 0)
        positions = portfolio.get('open_positions', {})
        
        # Daily loss check
        if equity > 0:
            daily_loss_pct = abs(daily_pnl) / equity * 100 if daily_pnl < 0 else 0
            if daily_loss_pct >= self.config['max_daily_loss_pct']:
                return False, f"Daily loss limit reached: {daily_loss_pct:.2f}%"
            
            # Auto kill switch check
            if self.config['auto_kill_switch'] and daily_loss_pct >= self.config['kill_switch_threshold_pct']:
                self.activate_kill_switch(f"Auto-triggered: Daily loss {daily_loss_pct:.2f}%")
                return False, "Kill switch auto-activated"
        
        # Position count check
        if len(positions) >= self.config['max_positions']:
            return False, f"Max positions reached: {len(positions)}"
        
        # Drawdown check
        max_dd = portfolio.get('max_drawdown', 0)
        if max_dd >= self.config['max_drawdown_pct']:
            return False, f"Max drawdown reached: {max_dd:.2f}%"
        
        # Circuit breaker check
        metrics = {
            'daily_loss': abs(daily_pnl) / equity * 100 if equity > 0 and daily_pnl < 0 else 0,
            'exposure_limit': sum(
                pos.get('value', 0) for pos in positions.values()
            ) / equity * 100 if equity > 0 else 0
        }
        
        if self.check_circuit_breakers(metrics):
            return False, "Circuit breaker tripped"
        
        # Proposed trade checks
        if proposed_trade:
            trade_value = proposed_trade.get('value', 0)
            if equity > 0:
                trade_pct = trade_value / equity * 100
                if trade_pct > self.config['max_position_value_pct']:
                    return False, f"Trade too large: {trade_pct:.1f}% > {self.config['max_position_value_pct']}%"
        
        return True, "Trading allowed"
    
    def assess_risk_level(self, portfolio: Dict) -> RiskLevel:
        """Assess current overall risk level."""
        if self._kill_switch_active:
            return RiskLevel.EMERGENCY
        
        equity = portfolio.get('total_equity', 0)
        daily_pnl = portfolio.get('daily_pnl', 0)
        drawdown = portfolio.get('max_drawdown', 0)
        
        daily_loss_pct = abs(daily_pnl) / equity * 100 if equity > 0 and daily_pnl < 0 else 0
        
        # Check for open circuit breakers
        open_breakers = sum(1 for cb in self.circuit_breakers.values() 
                          if cb.state == CircuitBreakerState.OPEN)
        
        if open_breakers >= 2 or daily_loss_pct >= self.config['kill_switch_threshold_pct']:
            return RiskLevel.CRITICAL
        elif open_breakers >= 1 or daily_loss_pct >= self.config['max_daily_loss_pct'] * 0.8:
            return RiskLevel.HIGH
        elif drawdown >= self.config['max_drawdown_pct'] * 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    # ==================== Alerts ====================
    
    def _create_alert(self, level: RiskLevel, message: str, 
                     metric_name: str, current: float, threshold: float):
        """Create and dispatch a risk alert."""
        alert = RiskAlert(
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current,
            threshold_value=threshold
        )
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Dispatch to callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback to receive risk alerts."""
        self.alert_callbacks.append(callback)
    
    def get_recent_alerts(self, limit: int = 10, level: RiskLevel = None) -> List[RiskAlert]:
        """Get recent alerts, optionally filtered by level."""
        alerts = self.alerts.copy()
        if level:
            alerts = [a for a in alerts if a.level == level]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
    
    # ==================== Reporting ====================
    
    def get_risk_report(self, portfolio: Dict) -> Dict:
        """Generate comprehensive risk report."""
        risk_level = self.assess_risk_level(portfolio)
        can_trade, reason = self.can_trade(portfolio)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_level': risk_level.value,
            'can_trade': can_trade,
            'trade_restriction_reason': reason if not can_trade else None,
            'kill_switch': self.get_kill_switch_status(),
            'circuit_breakers': self.get_circuit_breaker_status(),
            'recent_alerts': [
                {
                    'level': a.level.value,
                    'message': a.message,
                    'time': a.timestamp.isoformat()
                }
                for a in self.get_recent_alerts(5)
            ],
            'limits': {
                'daily_loss_pct': self.config['max_daily_loss_pct'],
                'max_drawdown_pct': self.config['max_drawdown_pct'],
                'max_positions': self.config['max_positions']
            }
        }
    
    def print_status(self, portfolio: Dict):
        """Print risk desk status."""
        report = self.get_risk_report(portfolio)
        
        print("\n" + "="*70)
        print("🛡️ RISK DESK STATUS")
        print("="*70)
        
        # Risk level with color emoji
        level = report['risk_level']
        level_emoji = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴',
            'emergency': '🚨'
        }
        print(f"\n  Risk Level: {level_emoji.get(level, '⚪')} {level.upper()}")
        print(f"  Can Trade: {'✅ YES' if report['can_trade'] else '❌ NO'}")
        
        if not report['can_trade']:
            print(f"  Reason: {report['trade_restriction_reason']}")
        
        # Kill switch
        ks = report['kill_switch']
        print(f"\n  Kill Switch: {'🚨 ACTIVE' if ks['active'] else '✅ Inactive'}")
        if ks['active']:
            print(f"    Reason: {ks['reason']}")
            print(f"    Duration: {ks['duration_minutes']} minutes")
        
        # Circuit breakers
        print(f"\n  Circuit Breakers:")
        for name, cb in report['circuit_breakers'].items():
            state = cb['state']
            state_emoji = '🔴' if state == 'open' else '🟡' if state == 'half_open' else '🟢'
            print(f"    {state_emoji} {name}: {state} (trips: {cb['trip_count']})")
        
        # Recent alerts
        if report['recent_alerts']:
            print(f"\n  Recent Alerts:")
            for alert in report['recent_alerts'][:3]:
                print(f"    ⚠️ [{alert['level']}] {alert['message']}")
        
        print("="*70)


# For compatibility, re-export from risk_engine
from typing import Tuple


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    risk_desk = RiskDesk()
    
    # Sample portfolio
    portfolio = {
        'total_equity': 10000,
        'daily_pnl': -150,  # -1.5% loss
        'max_drawdown': 3.5,
        'open_positions': {
            'BTCUSD': {'value': 2000, 'direction': 'long'},
            'ETHUSD': {'value': 1500, 'direction': 'long'}
        }
    }
    
    risk_desk.print_status(portfolio)
    
    # Test circuit breaker
    print("\n🧪 Testing circuit breakers...")
    risk_desk.check_circuit_breakers({'daily_loss': 3.5})  # Should trip
    
    risk_desk.print_status(portfolio)
    
    # Test kill switch
    print("\n🧪 Testing kill switch...")
    risk_desk.activate_kill_switch("Test activation")
    
    can_trade, reason = risk_desk.can_trade(portfolio)
    print(f"Can trade: {can_trade}, Reason: {reason}")
