"""
Event Engine - Event-driven architecture for backtesting

Provides:
- Event queue management
- Event types for market data, signals, orders, fills
- Pub/sub pattern for loose coupling
"""

import logging
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from queue import Queue, PriorityQueue
from collections import defaultdict
import heapq

logger = logging.getLogger("Aladdin.EventEngine")


class EventType(Enum):
    """Event types in the trading system"""
    # Market events
    TICK = auto()           # Price tick update
    BAR = auto()            # OHLCV bar/candle
    ORDERBOOK = auto()      # Orderbook update
    
    # Trading events
    SIGNAL = auto()         # Trading signal generated
    ORDER = auto()          # Order created
    FILL = auto()           # Order filled
    CANCEL = auto()         # Order cancelled
    REJECT = auto()         # Order rejected
    
    # System events
    START = auto()          # Backtest start
    END = auto()            # Backtest end
    HEARTBEAT = auto()      # System heartbeat
    ERROR = auto()          # Error occurred
    
    # Risk events
    RISK_CHECK = auto()     # Risk check triggered
    RISK_BREACH = auto()    # Risk limit breached
    STOP_TRIGGERED = auto() # Stop loss triggered
    
    # Portfolio events
    POSITION_UPDATE = auto()  # Position changed
    PNL_UPDATE = auto()       # P&L updated
    MARGIN_CALL = auto()      # Margin call warning


@dataclass
class Event:
    """
    Base event class with priority ordering
    
    Events are ordered by (priority, timestamp) for processing
    """
    priority: int = field(default=5)
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: EventType = field(default=EventType.HEARTBEAT)
    data: Dict = field(default_factory=dict)
    source: str = field(default="")
    _seq: int = field(default=0, repr=False)  # Sequence number for stable sorting
    
    # Class variable for sequence counter
    _counter: int = 0
    
    def __lt__(self, other: 'Event') -> bool:
        """Compare events by priority, then timestamp, then sequence"""
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self._seq < other._seq
    
    def __le__(self, other: 'Event') -> bool:
        return self == other or self < other
    
    def __post_init__(self):
        """Set priority based on event type if not specified"""
        # Assign sequence number for stable sorting
        Event._counter += 1
        object.__setattr__(self, '_seq', Event._counter)
        
        priority_map = {
            EventType.START: 0,
            EventType.RISK_BREACH: 1,
            EventType.STOP_TRIGGERED: 1,
            EventType.MARGIN_CALL: 1,
            EventType.TICK: 2,
            EventType.BAR: 2,
            EventType.ORDERBOOK: 2,
            EventType.SIGNAL: 3,
            EventType.ORDER: 4,
            EventType.FILL: 4,
            EventType.CANCEL: 4,
            EventType.REJECT: 4,
            EventType.POSITION_UPDATE: 5,
            EventType.PNL_UPDATE: 5,
            EventType.RISK_CHECK: 6,
            EventType.HEARTBEAT: 7,
            EventType.END: 8,
            EventType.ERROR: 9,
        }
        if self.priority == 5:
            self.priority = priority_map.get(self.event_type, 5)


@dataclass
class TickEvent(Event):
    """Tick/price update event"""
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.TICK
        super().__post_init__()
        self.data = {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask
        }


@dataclass
class BarEvent(Event):
    """OHLCV bar/candle event"""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    timeframe: str = "1h"
    
    def __post_init__(self):
        self.event_type = EventType.BAR
        super().__post_init__()
        self.data = {
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe
        }


@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    symbol: str = ""
    direction: int = 0  # 1=long, -1=short, 0=neutral
    strength: float = 0.0
    confidence: float = 0.0
    strategy: str = ""
    target_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.SIGNAL
        super().__post_init__()
        self.data = {
            'symbol': self.symbol,
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


@dataclass
class OrderEvent(Event):
    """Order event"""
    order_id: str = ""
    symbol: str = ""
    side: str = ""  # 'buy' or 'sell'
    order_type: str = "market"  # 'market', 'limit', 'stop'
    quantity: float = 0.0
    price: float = 0.0
    stop_price: float = 0.0
    time_in_force: str = "GTC"
    
    def __post_init__(self):
        self.event_type = EventType.ORDER
        super().__post_init__()
        self.data = {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force
        }


@dataclass
class FillEvent(Event):
    """Order fill event"""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.FILL
        super().__post_init__()
        self.data = {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'fill_price': self.fill_price,
            'commission': self.commission,
            'slippage': self.slippage
        }


class EventEngine:
    """
    Central event engine for event-driven backtesting
    
    Features:
    - Priority queue for event ordering
    - Pub/sub pattern for event handlers
    - Event history for replay/debugging
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize event engine
        
        Args:
            max_history: Maximum events to keep in history
        """
        self._queue: List = []  # Priority queue (heap)
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._global_handlers: List[Callable] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._running = False
        self._event_count = 0
        
        logger.info("Event engine initialized")
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        """
        Subscribe to specific event type
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callback function
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"Handler subscribed to {event_type.name}")
    
    def subscribe_all(self, handler: Callable[[Event], None]) -> None:
        """
        Subscribe to all event types
        
        Args:
            handler: Callback function
        """
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)
            logger.debug("Global handler subscribed")
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        """
        Unsubscribe from event type
        
        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
    
    def publish(self, event: Event) -> None:
        """
        Publish event to queue
        
        Args:
            event: Event to publish
        """
        heapq.heappush(self._queue, event)
        self._event_count += 1
    
    def process_next(self) -> Optional[Event]:
        """
        Process next event in queue
        
        Returns:
            Processed event or None if queue empty
        """
        if not self._queue:
            return None
        
        event = heapq.heappop(self._queue)
        
        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        # Dispatch to handlers
        self._dispatch(event)
        
        return event
    
    def process_all(self) -> int:
        """
        Process all events in queue
        
        Returns:
            Number of events processed
        """
        count = 0
        while self._queue:
            self.process_next()
            count += 1
        return count
    
    def run(self) -> None:
        """
        Run event loop until stopped
        """
        self._running = True
        logger.info("Event engine started")
        
        while self._running and self._queue:
            self.process_next()
        
        logger.info(f"Event engine stopped. Processed {self._event_count} events")
    
    def stop(self) -> None:
        """Stop event loop"""
        self._running = False
    
    def _dispatch(self, event: Event) -> None:
        """
        Dispatch event to all subscribed handlers
        
        Args:
            event: Event to dispatch
        """
        # Call specific handlers
        for handler in self._handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.event_type.name}: {e}")
        
        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Global handler error: {e}")
    
    def clear(self) -> None:
        """Clear event queue"""
        self._queue = []
    
    def clear_history(self) -> None:
        """Clear event history"""
        self._history = []
    
    @property
    def queue_size(self) -> int:
        """Current queue size"""
        return len(self._queue)
    
    @property
    def history(self) -> List[Event]:
        """Event history"""
        return self._history.copy()
    
    @property
    def event_count(self) -> int:
        """Total events processed"""
        return self._event_count
    
    def get_history_by_type(self, event_type: EventType) -> List[Event]:
        """
        Get history filtered by event type
        
        Args:
            event_type: Type to filter by
        
        Returns:
            List of matching events
        """
        return [e for e in self._history if e.event_type == event_type]
    
    def replay_history(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> None:
        """
        Replay events from history
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (exclusive)
        """
        if end_idx is None:
            end_idx = len(self._history)
        
        for event in self._history[start_idx:end_idx]:
            self.publish(event)
    
    def get_stats(self) -> Dict:
        """
        Get event engine statistics
        
        Returns:
            Dict with statistics
        """
        type_counts = defaultdict(int)
        for event in self._history:
            type_counts[event.event_type.name] += 1
        
        return {
            'total_events': self._event_count,
            'queue_size': self.queue_size,
            'history_size': len(self._history),
            'events_by_type': dict(type_counts),
            'running': self._running
        }


# Convenience function for creating events
def create_event(
    event_type: EventType,
    timestamp: datetime = None,
    **kwargs
) -> Event:
    """
    Create event with type and data
    
    Args:
        event_type: Type of event
        timestamp: Event timestamp
        **kwargs: Event-specific data
    
    Returns:
        Created event
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return Event(
        event_type=event_type,
        timestamp=timestamp,
        data=kwargs
    )


if __name__ == "__main__":
    # Test event engine
    logging.basicConfig(level=logging.INFO)
    
    engine = EventEngine()
    
    # Track processed events
    processed = []
    
    def on_bar(event: Event):
        processed.append(('BAR', event.data))
        print(f"BAR: {event.data.get('symbol')} @ {event.data.get('close')}")
    
    def on_signal(event: Event):
        processed.append(('SIGNAL', event.data))
        print(f"SIGNAL: {event.data.get('direction')} with {event.data.get('confidence')}% confidence")
    
    def on_fill(event: Event):
        processed.append(('FILL', event.data))
        print(f"FILL: {event.data.get('symbol')} @ {event.data.get('fill_price')}")
    
    # Subscribe handlers
    engine.subscribe(EventType.BAR, on_bar)
    engine.subscribe(EventType.SIGNAL, on_signal)
    engine.subscribe(EventType.FILL, on_fill)
    
    # Publish events (note: will be processed in priority order)
    engine.publish(BarEvent(
        symbol="BTCUSD",
        open=100000, high=101000, low=99000, close=100500,
        volume=1000
    ))
    
    engine.publish(SignalEvent(
        symbol="BTCUSD",
        direction=1,
        strength=0.8,
        confidence=0.85,
        strategy="momentum"
    ))
    
    engine.publish(FillEvent(
        order_id="order_001",
        symbol="BTCUSD",
        side="buy",
        quantity=0.1,
        fill_price=100520,
        commission=6.03
    ))
    
    # Process all events
    print("\n" + "=" * 60)
    print("PROCESSING EVENTS (in priority order)")
    print("=" * 60)
    
    count = engine.process_all()
    
    print("\n" + "=" * 60)
    print(f"Processed {count} events")
    print(f"Stats: {engine.get_stats()}")
