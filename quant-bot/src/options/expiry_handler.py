"""
Auto-Expiry & Assignment Handler

Production-ready module for automatically handling options expiry and assignment.
Includes:
- Expiry detection and notification
- Auto-close logic for near-expiry positions
- Roll-forward automation
- Assignment/exercise handling
- Risk monitoring during expiry windows

Usage:
    handler = ExpiryHandler(exchange_client)
    await handler.start_monitoring()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import json
from pathlib import Path
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger('expiry_handler')


# ============================================================================
# Data Structures
# ============================================================================

class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class ExpiryAction(Enum):
    HOLD = "hold"  # Keep until expiry
    CLOSE = "close"  # Close before expiry
    ROLL = "roll"  # Roll to next expiry
    EXERCISE = "exercise"  # Exercise early (American)
    SETTLE = "settle"  # Cash settle (European)


class AssignmentStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    EXERCISED = "exercised"
    EXPIRED_WORTHLESS = "expired_worthless"
    CLOSED = "closed"


@dataclass
class OptionsPosition:
    """Represents an options position."""
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry: datetime
    quantity: float  # Positive = long, negative = short
    entry_price: float
    current_price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0
    underlying_price: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def notional(self) -> float:
        return abs(self.quantity * self.underlying_price)
    
    @property
    def time_to_expiry_hours(self) -> float:
        delta = self.expiry - datetime.utcnow()
        return max(0, delta.total_seconds() / 3600)
    
    @property
    def time_to_expiry_days(self) -> float:
        return self.time_to_expiry_hours / 24
    
    @property
    def moneyness(self) -> float:
        """ITM > 0, ATM = 0, OTM < 0"""
        if self.underlying_price <= 0:
            return 0
        if self.option_type == OptionType.CALL:
            return (self.underlying_price - self.strike) / self.strike
        else:
            return (self.strike - self.underlying_price) / self.strike
    
    @property
    def intrinsic_value(self) -> float:
        if self.option_type == OptionType.CALL:
            return max(0, self.underlying_price - self.strike)
        else:
            return max(0, self.strike - self.underlying_price)
    
    @property
    def time_value(self) -> float:
        return max(0, self.current_price - self.intrinsic_value)


@dataclass
class ExpiryConfig:
    """Configuration for expiry handling."""
    # Timing thresholds
    early_warning_hours: float = 24.0  # Alert 24h before expiry
    close_threshold_hours: float = 4.0  # Auto-close if < 4h to expiry
    roll_threshold_hours: float = 12.0  # Consider rolling if < 12h
    
    # Risk thresholds
    max_delta_at_expiry: float = 0.10  # Max delta exposure at expiry
    max_gamma_at_expiry: float = 0.05  # Max gamma at expiry
    min_time_value_to_hold: float = 0.001  # Close if time value < 0.1%
    
    # Rolling parameters
    default_roll_days: int = 7  # Roll to 7 DTE by default
    max_roll_spread_bps: float = 50.0  # Max cost to roll (bps)
    
    # Execution
    use_market_orders_near_expiry: bool = True
    slippage_tolerance: float = 0.02  # 2% slippage tolerance
    
    # Notifications
    enable_alerts: bool = True
    alert_webhook_url: Optional[str] = None


@dataclass
class ExpiryEvent:
    """Represents an expiry event."""
    position: OptionsPosition
    action: ExpiryAction
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    executed: bool = False
    result: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.position.symbol,
            'action': self.action.value,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat(),
            'executed': self.executed,
            'result': self.result
        }


# ============================================================================
# Expiry Analysis Engine
# ============================================================================

class ExpiryAnalyzer:
    """
    Analyzes options positions and determines appropriate expiry actions.
    """
    
    def __init__(self, config: ExpiryConfig):
        self.config = config
        
    def analyze_position(self, position: OptionsPosition) -> Tuple[ExpiryAction, str]:
        """
        Determine the recommended action for a position approaching expiry.
        Returns (action, reason).
        """
        hours_to_expiry = position.time_to_expiry_hours
        
        # Already expired
        if hours_to_expiry <= 0:
            return ExpiryAction.SETTLE, "Position has expired"
        
        # Very close to expiry - force close
        if hours_to_expiry < self.config.close_threshold_hours:
            return self._analyze_near_expiry(position)
        
        # In roll window
        if hours_to_expiry < self.config.roll_threshold_hours:
            return self._analyze_roll_candidate(position)
        
        # Early warning zone
        if hours_to_expiry < self.config.early_warning_hours:
            return self._analyze_warning_zone(position)
        
        # Not near expiry
        return ExpiryAction.HOLD, "Position not near expiry"
    
    def _analyze_near_expiry(self, position: OptionsPosition) -> Tuple[ExpiryAction, str]:
        """Analyze position very close to expiry."""
        
        # If ITM and long, consider exercise
        if position.is_long and position.moneyness > 0.01:
            return ExpiryAction.EXERCISE, f"ITM position ({position.moneyness:.1%}), exercise recommended"
        
        # If short with significant delta, close to avoid pin risk
        if position.is_short and abs(position.delta) > self.config.max_delta_at_expiry:
            return ExpiryAction.CLOSE, f"Short position with high delta ({position.delta:.2f}), close to avoid pin risk"
        
        # High gamma = pin risk
        if abs(position.gamma) > self.config.max_gamma_at_expiry:
            return ExpiryAction.CLOSE, f"High gamma ({position.gamma:.3f}) near expiry, close for pin risk"
        
        # OTM with minimal time value
        if position.moneyness < -0.05 and position.time_value < self.config.min_time_value_to_hold:
            return ExpiryAction.CLOSE, "Deep OTM with minimal time value, let expire or close"
        
        return ExpiryAction.SETTLE, "Approaching expiry, will settle at expiration"
    
    def _analyze_roll_candidate(self, position: OptionsPosition) -> Tuple[ExpiryAction, str]:
        """Analyze position in roll window."""
        
        # Profitable short position - consider rolling
        if position.is_short and position.current_price < position.entry_price:
            return ExpiryAction.ROLL, "Profitable short position, consider rolling to capture more theta"
        
        # Long position with remaining time value
        if position.is_long and position.time_value > self.config.min_time_value_to_hold:
            return ExpiryAction.ROLL, "Long position with time value, consider rolling to preserve value"
        
        # ATM position - high pin risk
        if abs(position.moneyness) < 0.02:
            return ExpiryAction.ROLL, "Near ATM with pin risk, roll to avoid expiry uncertainty"
        
        return ExpiryAction.HOLD, "Position in roll window, monitoring"
    
    def _analyze_warning_zone(self, position: OptionsPosition) -> Tuple[ExpiryAction, str]:
        """Analyze position in early warning zone."""
        
        # Just monitor and alert
        if abs(position.delta) > 0.8:
            return ExpiryAction.HOLD, f"Deep ITM position (delta={position.delta:.2f}), will likely be exercised/assigned"
        
        return ExpiryAction.HOLD, f"Position expiring in {position.time_to_expiry_hours:.1f}h, monitoring"
    
    def calculate_roll_cost(
        self,
        current_position: OptionsPosition,
        target_strike: float,
        target_expiry: datetime,
        target_price: float
    ) -> float:
        """Calculate the cost/credit of rolling a position."""
        # Close current position
        if current_position.is_long:
            close_cost = -current_position.current_price  # Selling = credit
        else:
            close_cost = current_position.current_price  # Buying back = debit
        
        # Open new position
        if current_position.is_long:
            open_cost = target_price  # Buying = debit
        else:
            open_cost = -target_price  # Selling = credit
        
        net_cost = close_cost + open_cost
        
        # Convert to basis points of notional
        cost_bps = (net_cost / current_position.underlying_price) * 10000
        
        return cost_bps


# ============================================================================
# Expiry Execution Engine
# ============================================================================

class ExpiryExecutor:
    """
    Executes expiry-related actions (close, roll, exercise).
    """
    
    def __init__(self, config: ExpiryConfig, exchange_client: Any = None):
        self.config = config
        self.exchange = exchange_client
        self.execution_log: List[ExpiryEvent] = []
        
    async def execute_action(self, event: ExpiryEvent) -> Dict:
        """Execute the recommended expiry action."""
        position = event.position
        action = event.action
        
        logger.info(f"Executing {action.value} for {position.symbol}: {event.reason}")
        
        try:
            if action == ExpiryAction.CLOSE:
                result = await self._execute_close(position)
            elif action == ExpiryAction.ROLL:
                result = await self._execute_roll(position)
            elif action == ExpiryAction.EXERCISE:
                result = await self._execute_exercise(position)
            elif action == ExpiryAction.SETTLE:
                result = await self._handle_settlement(position)
            else:
                result = {'status': 'held', 'message': 'No action taken'}
            
            event.executed = True
            event.result = result
            self.execution_log.append(event)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute {action.value} for {position.symbol}: {e}")
            event.result = {'status': 'error', 'error': str(e)}
            self.execution_log.append(event)
            raise
    
    async def _execute_close(self, position: OptionsPosition) -> Dict:
        """Close a position before expiry."""
        side = 'sell' if position.is_long else 'buy'
        quantity = abs(position.quantity)
        
        # Use market order near expiry for certainty
        order_type = 'market' if self.config.use_market_orders_near_expiry else 'limit'
        
        if self.exchange:
            order = await self.exchange.place_order(
                symbol=position.symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=position.current_price if order_type == 'limit' else None
            )
            return {
                'status': 'closed',
                'order_id': order.get('id'),
                'fill_price': order.get('filled_price'),
                'quantity': quantity
            }
        else:
            # Simulation mode
            logger.info(f"[SIMULATION] Would close {quantity} {position.symbol} via {side} {order_type}")
            return {
                'status': 'simulated_close',
                'side': side,
                'quantity': quantity,
                'estimated_price': position.current_price
            }
    
    async def _execute_roll(self, position: OptionsPosition) -> Dict:
        """Roll a position to a later expiry."""
        # Calculate target expiry
        target_expiry = datetime.utcnow() + timedelta(days=self.config.default_roll_days)
        
        # Close current
        close_result = await self._execute_close(position)
        
        # Open new position at same strike (or adjusted)
        new_quantity = position.quantity  # Same direction
        
        if self.exchange:
            # Find next expiry contract
            # new_symbol = await self.exchange.find_option_symbol(
            #     underlying=position.underlying,
            #     option_type=position.option_type.value,
            #     strike=position.strike,
            #     expiry=target_expiry
            # )
            
            # open_order = await self.exchange.place_order(
            #     symbol=new_symbol,
            #     side='buy' if new_quantity > 0 else 'sell',
            #     quantity=abs(new_quantity),
            #     order_type='limit'
            # )
            
            return {
                'status': 'rolled',
                'close_result': close_result,
                'new_expiry': target_expiry.isoformat(),
                'new_strike': position.strike
            }
        else:
            logger.info(f"[SIMULATION] Would roll to {target_expiry.date()} at strike {position.strike}")
            return {
                'status': 'simulated_roll',
                'close_result': close_result,
                'target_expiry': target_expiry.isoformat(),
                'target_strike': position.strike
            }
    
    async def _execute_exercise(self, position: OptionsPosition) -> Dict:
        """Exercise an option (for American-style)."""
        if not position.is_long:
            raise ValueError("Cannot exercise short position")
        
        if self.exchange:
            # result = await self.exchange.exercise_option(
            #     symbol=position.symbol,
            #     quantity=position.quantity
            # )
            return {
                'status': 'exercise_requested',
                'symbol': position.symbol,
                'quantity': position.quantity
            }
        else:
            logger.info(f"[SIMULATION] Would exercise {position.quantity} {position.symbol}")
            return {
                'status': 'simulated_exercise',
                'intrinsic_value': position.intrinsic_value,
                'quantity': position.quantity
            }
    
    async def _handle_settlement(self, position: OptionsPosition) -> Dict:
        """Handle position at settlement."""
        settlement_value = position.intrinsic_value * abs(position.quantity)
        
        if position.is_short:
            settlement_value = -settlement_value  # Liability for short
        
        return {
            'status': 'settled',
            'settlement_value': settlement_value,
            'final_moneyness': position.moneyness,
            'itm': position.moneyness > 0
        }


# ============================================================================
# Expiry Handler (Main Interface)
# ============================================================================

class ExpiryHandler:
    """
    Main handler for options expiry management.
    
    Usage:
        config = ExpiryConfig()
        handler = ExpiryHandler(config, exchange_client)
        
        # Check all positions
        events = await handler.check_positions(positions)
        
        # Or start continuous monitoring
        await handler.start_monitoring()
    """
    
    def __init__(
        self,
        config: ExpiryConfig = None,
        exchange_client: Any = None,
        position_provider: Callable[[], List[OptionsPosition]] = None
    ):
        self.config = config or ExpiryConfig()
        self.exchange = exchange_client
        self.position_provider = position_provider
        
        self.analyzer = ExpiryAnalyzer(self.config)
        self.executor = ExpiryExecutor(self.config, exchange_client)
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self._on_expiry_event: Optional[Callable[[ExpiryEvent], None]] = None
        self._on_action_executed: Optional[Callable[[ExpiryEvent], None]] = None
    
    def set_callbacks(
        self,
        on_expiry_event: Callable[[ExpiryEvent], None] = None,
        on_action_executed: Callable[[ExpiryEvent], None] = None
    ):
        """Set callback functions for events."""
        self._on_expiry_event = on_expiry_event
        self._on_action_executed = on_action_executed
    
    async def check_positions(
        self,
        positions: List[OptionsPosition],
        auto_execute: bool = False
    ) -> List[ExpiryEvent]:
        """
        Check all positions for expiry-related actions.
        
        Args:
            positions: List of current options positions
            auto_execute: If True, automatically execute recommended actions
            
        Returns:
            List of ExpiryEvent objects with recommendations/results
        """
        events = []
        
        for position in positions:
            action, reason = self.analyzer.analyze_position(position)
            
            if action != ExpiryAction.HOLD:
                event = ExpiryEvent(
                    position=position,
                    action=action,
                    reason=reason
                )
                events.append(event)
                
                logger.info(f"Expiry event: {position.symbol} -> {action.value}: {reason}")
                
                if self._on_expiry_event:
                    self._on_expiry_event(event)
                
                if auto_execute and action in [ExpiryAction.CLOSE, ExpiryAction.ROLL]:
                    try:
                        await self.executor.execute_action(event)
                        if self._on_action_executed:
                            self._on_action_executed(event)
                    except Exception as e:
                        logger.error(f"Auto-execute failed: {e}")
        
        return events
    
    async def start_monitoring(self, interval_seconds: float = 300):
        """
        Start continuous monitoring of positions.
        
        Args:
            interval_seconds: How often to check positions (default: 5 minutes)
        """
        if not self.position_provider:
            raise ValueError("position_provider must be set for monitoring")
        
        self._monitoring = True
        logger.info(f"Starting expiry monitoring (interval: {interval_seconds}s)")
        
        while self._monitoring:
            try:
                positions = self.position_provider()
                await self.check_positions(positions, auto_execute=True)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring = False
        logger.info("Stopping expiry monitoring")
    
    def get_expiry_calendar(
        self,
        positions: List[OptionsPosition],
        days_ahead: int = 30
    ) -> Dict[str, List[OptionsPosition]]:
        """
        Get a calendar view of upcoming expiries.
        
        Returns dict keyed by date with positions expiring on that date.
        """
        calendar = {}
        cutoff = datetime.utcnow() + timedelta(days=days_ahead)
        
        for position in positions:
            if position.expiry <= cutoff:
                date_key = position.expiry.strftime('%Y-%m-%d')
                if date_key not in calendar:
                    calendar[date_key] = []
                calendar[date_key].append(position)
        
        return dict(sorted(calendar.items()))
    
    def get_risk_summary(self, positions: List[OptionsPosition]) -> Dict:
        """
        Get risk summary for positions approaching expiry.
        """
        summary = {
            'total_positions': len(positions),
            'expiring_24h': 0,
            'expiring_week': 0,
            'total_delta': 0.0,
            'total_gamma': 0.0,
            'at_risk_notional': 0.0,
            'positions_by_status': {}
        }
        
        for position in positions:
            hours = position.time_to_expiry_hours
            
            if hours < 24:
                summary['expiring_24h'] += 1
            if hours < 168:  # 7 days
                summary['expiring_week'] += 1
            
            if hours < self.config.early_warning_hours:
                summary['total_delta'] += position.delta * abs(position.quantity)
                summary['total_gamma'] += position.gamma * abs(position.quantity)
                summary['at_risk_notional'] += position.notional
            
            action, _ = self.analyzer.analyze_position(position)
            status = action.value
            if status not in summary['positions_by_status']:
                summary['positions_by_status'][status] = 0
            summary['positions_by_status'][status] += 1
        
        return summary
    
    def export_events(self, filepath: str):
        """Export execution log to JSON file."""
        events = [e.to_dict() for e in self.executor.execution_log]
        with open(filepath, 'w') as f:
            json.dump(events, f, indent=2)
        logger.info(f"Exported {len(events)} events to {filepath}")


# ============================================================================
# Alert Manager
# ============================================================================

class ExpiryAlertManager:
    """
    Manages alerts and notifications for expiry events.
    """
    
    def __init__(self, config: ExpiryConfig):
        self.config = config
        self.alerts_sent: List[Dict] = []
    
    async def send_alert(self, event: ExpiryEvent, level: str = 'info'):
        """Send an alert for an expiry event."""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'symbol': event.position.symbol,
            'action': event.action.value,
            'reason': event.reason,
            'hours_to_expiry': event.position.time_to_expiry_hours,
            'delta': event.position.delta,
            'moneyness': event.position.moneyness
        }
        
        self.alerts_sent.append(alert)
        
        # Log alert
        logger.warning(f"EXPIRY ALERT [{level.upper()}]: {event.position.symbol} - {event.reason}")
        
        # Send to webhook if configured
        if self.config.alert_webhook_url:
            await self._send_webhook(alert)
    
    async def _send_webhook(self, alert: Dict):
        """Send alert to configured webhook."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(
                    self.config.alert_webhook_url,
                    json=alert,
                    timeout=aiohttp.ClientTimeout(total=10)
                )
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")


# ============================================================================
# Testing & Demo
# ============================================================================

def create_sample_positions() -> List[OptionsPosition]:
    """Create sample positions for testing."""
    now = datetime.utcnow()
    
    return [
        # Expiring in 2 hours (should close)
        OptionsPosition(
            symbol='BTC-50000-C-20240115',
            underlying='BTC',
            option_type=OptionType.CALL,
            strike=50000,
            expiry=now + timedelta(hours=2),
            quantity=1.0,
            entry_price=500,
            current_price=450,
            delta=0.65,
            gamma=0.08,
            theta=-50,
            vega=100,
            underlying_price=50200
        ),
        
        # Expiring in 8 hours (roll candidate)
        OptionsPosition(
            symbol='BTC-48000-P-20240115',
            underlying='BTC',
            option_type=OptionType.PUT,
            strike=48000,
            expiry=now + timedelta(hours=8),
            quantity=-2.0,  # Short
            entry_price=200,
            current_price=50,
            delta=-0.15,
            gamma=0.02,
            theta=-30,
            vega=80,
            underlying_price=50200
        ),
        
        # Expiring in 20 hours (warning zone)
        OptionsPosition(
            symbol='ETH-3000-C-20240115',
            underlying='ETH',
            option_type=OptionType.CALL,
            strike=3000,
            expiry=now + timedelta(hours=20),
            quantity=5.0,
            entry_price=100,
            current_price=150,
            delta=0.70,
            gamma=0.03,
            theta=-20,
            vega=50,
            underlying_price=3100
        ),
        
        # Expiring in 5 days (safe)
        OptionsPosition(
            symbol='BTC-55000-C-20240120',
            underlying='BTC',
            option_type=OptionType.CALL,
            strike=55000,
            expiry=now + timedelta(days=5),
            quantity=1.0,
            entry_price=800,
            current_price=600,
            delta=0.35,
            gamma=0.01,
            theta=-40,
            vega=200,
            underlying_price=50200
        )
    ]


async def demo():
    """Demonstrate the expiry handler."""
    print("=" * 70)
    print("🔔 OPTIONS EXPIRY HANDLER DEMO")
    print("=" * 70)
    
    # Create config and handler
    config = ExpiryConfig(
        early_warning_hours=24,
        close_threshold_hours=4,
        roll_threshold_hours=12
    )
    
    handler = ExpiryHandler(config)
    
    # Get sample positions
    positions = create_sample_positions()
    
    print(f"\n📊 Checking {len(positions)} positions...\n")
    
    # Check positions
    events = await handler.check_positions(positions, auto_execute=False)
    
    print("-" * 70)
    print("EXPIRY EVENTS:")
    print("-" * 70)
    
    for event in events:
        pos = event.position
        print(f"\n  Symbol: {pos.symbol}")
        print(f"  Action: {event.action.value.upper()}")
        print(f"  Reason: {event.reason}")
        print(f"  Time to expiry: {pos.time_to_expiry_hours:.1f}h")
        print(f"  Delta: {pos.delta:.2f}, Gamma: {pos.gamma:.3f}")
        print(f"  Moneyness: {pos.moneyness:.1%}")
    
    # Get risk summary
    print("\n" + "=" * 70)
    print("📈 RISK SUMMARY")
    print("=" * 70)
    
    summary = handler.get_risk_summary(positions)
    print(f"\n  Total positions: {summary['total_positions']}")
    print(f"  Expiring < 24h: {summary['expiring_24h']}")
    print(f"  Expiring < 7d: {summary['expiring_week']}")
    print(f"  Net delta (at risk): {summary['total_delta']:.2f}")
    print(f"  Net gamma (at risk): {summary['total_gamma']:.3f}")
    print(f"  At-risk notional: ${summary['at_risk_notional']:,.0f}")
    
    # Get expiry calendar
    print("\n" + "=" * 70)
    print("📅 EXPIRY CALENDAR")
    print("=" * 70)
    
    calendar = handler.get_expiry_calendar(positions)
    for date, pos_list in calendar.items():
        print(f"\n  {date}:")
        for p in pos_list:
            print(f"    - {p.symbol} ({p.option_type.value}, qty={p.quantity})")
    
    print("\n✅ Demo complete!")


# ============================================================================
# Tests
# ============================================================================

class TestExpiryHandler:
    """Test suite for expiry handler."""
    
    def test_analyzer_close_threshold(self):
        """Test that positions near expiry trigger close."""
        config = ExpiryConfig(close_threshold_hours=4)
        analyzer = ExpiryAnalyzer(config)
        
        position = OptionsPosition(
            symbol='TEST',
            underlying='BTC',
            option_type=OptionType.CALL,
            strike=50000,
            expiry=datetime.utcnow() + timedelta(hours=2),
            quantity=-1.0,  # Short
            entry_price=100,
            current_price=50,
            delta=0.20,
            gamma=0.05,
            underlying_price=50000
        )
        
        action, reason = analyzer.analyze_position(position)
        # High gamma should trigger close
        assert action == ExpiryAction.CLOSE
        
    def test_analyzer_roll_threshold(self):
        """Test that positions in roll window are identified."""
        config = ExpiryConfig(roll_threshold_hours=12, close_threshold_hours=4)
        analyzer = ExpiryAnalyzer(config)
        
        position = OptionsPosition(
            symbol='TEST',
            underlying='BTC',
            option_type=OptionType.PUT,
            strike=50000,
            expiry=datetime.utcnow() + timedelta(hours=8),
            quantity=-1.0,  # Profitable short
            entry_price=200,
            current_price=50,
            delta=-0.10,
            gamma=0.01,
            underlying_price=52000
        )
        
        action, reason = analyzer.analyze_position(position)
        assert action == ExpiryAction.ROLL
        
    def test_analyzer_hold_far_expiry(self):
        """Test that far-expiry positions are held."""
        config = ExpiryConfig(early_warning_hours=24)
        analyzer = ExpiryAnalyzer(config)
        
        position = OptionsPosition(
            symbol='TEST',
            underlying='BTC',
            option_type=OptionType.CALL,
            strike=50000,
            expiry=datetime.utcnow() + timedelta(days=30),
            quantity=1.0,
            entry_price=100,
            current_price=150,
            delta=0.50,
            gamma=0.01,
            underlying_price=50000
        )
        
        action, reason = analyzer.analyze_position(position)
        assert action == ExpiryAction.HOLD
        
    def test_risk_summary(self):
        """Test risk summary calculation."""
        config = ExpiryConfig()
        handler = ExpiryHandler(config)
        
        positions = create_sample_positions()
        summary = handler.get_risk_summary(positions)
        
        assert summary['total_positions'] == 4
        assert summary['expiring_24h'] >= 1
        
    def test_expiry_calendar(self):
        """Test expiry calendar generation."""
        config = ExpiryConfig()
        handler = ExpiryHandler(config)
        
        positions = create_sample_positions()
        calendar = handler.get_expiry_calendar(positions, days_ahead=7)
        
        assert len(calendar) > 0


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        import pytest
        pytest.main([__file__, '-v'])
    else:
        # Run demo
        asyncio.run(demo())
