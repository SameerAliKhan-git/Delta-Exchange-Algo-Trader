"""
ALADDIN - Order Manager
=========================
Idempotent order execution with state management.
"""

import logging
import hashlib
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "pending"           # Created, not submitted
    SUBMITTED = "submitted"       # Sent to exchange
    OPEN = "open"                 # Acknowledged by exchange
    PARTIALLY_FILLED = "partial"  # Partially executed
    FILLED = "filled"             # Fully executed
    CANCELLED = "cancelled"       # Cancelled
    REJECTED = "rejected"         # Rejected by exchange
    EXPIRED = "expired"           # Time in force expired
    FAILED = "failed"             # System error


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order entity with full lifecycle tracking."""
    # Core identifiers
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    
    # Pricing
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    
    # Fills
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Metadata
    strategy: str = ""
    signal_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    last_error: str = ""
    
    # Idempotency
    idempotency_key: str = ""
    
    def __post_init__(self):
        if not self.idempotency_key:
            self.idempotency_key = self._generate_idempotency_key()
    
    def _generate_idempotency_key(self) -> str:
        """Generate unique idempotency key based on order parameters."""
        key_data = f"{self.symbol}:{self.side.value}:{self.order_type.value}:" \
                   f"{self.quantity}:{self.price}:{self.signal_id}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED
        )
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (can be cancelled)."""
        return self.status in (
            OrderStatus.SUBMITTED,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED
        )
    
    @property
    def unfilled_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def fill_percentage(self) -> float:
        return (self.filled_quantity / self.quantity * 100) if self.quantity > 0 else 0
    
    def to_dict(self) -> Dict:
        return {
            'client_order_id': self.client_order_id,
            'exchange_order_id': self.exchange_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_qty': self.filled_quantity,
            'avg_price': self.average_fill_price,
            'fill_pct': f"{self.fill_percentage:.1f}%",
            'created_at': self.created_at.isoformat(),
            'strategy': self.strategy
        }


class OrderManager:
    """
    Manages order lifecycle with idempotent execution.
    
    Features:
    - Idempotent order submission (prevents duplicates)
    - Automatic retries with exponential backoff
    - Order state persistence
    - Bracket order support (stop loss + take profit)
    - Order tracking and history
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger('Aladdin.OrderManager')
        self.config = config or self._default_config()
        
        # Order storage
        self.orders: Dict[str, Order] = {}  # client_order_id -> Order
        self.idempotency_cache: Dict[str, str] = {}  # idempotency_key -> client_order_id
        
        # Locks for thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'orders_created': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'orders_failed': 0,
            'total_retries': 0
        }
    
    def _default_config(self) -> Dict:
        return {
            'max_retries': 3,
            'retry_delay_base': 1.0,  # seconds
            'retry_delay_max': 30.0,
            'order_timeout': 60,  # seconds
            'idempotency_cache_ttl': 3600,  # 1 hour
            'default_time_in_force': 'GTC',
        }
    
    def create_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: float = None,
                    stop_loss: float = None, take_profit: float = None,
                    strategy: str = "", signal_id: str = "") -> Order:
        """
        Create a new order with idempotency check.
        
        Returns existing order if duplicate detected.
        """
        with self._lock:
            # Create order object
            order = Order(
                client_order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=OrderSide(side),
                order_type=OrderType(order_type),
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=strategy,
                signal_id=signal_id,
                max_retries=self.config['max_retries']
            )
            
            # Check for duplicate (idempotency)
            if order.idempotency_key in self.idempotency_cache:
                existing_id = self.idempotency_cache[order.idempotency_key]
                if existing_id in self.orders:
                    existing = self.orders[existing_id]
                    if not existing.is_terminal:
                        self.logger.warning(f"Duplicate order detected, returning existing: {existing_id}")
                        return existing
            
            # Store order
            self.orders[order.client_order_id] = order
            self.idempotency_cache[order.idempotency_key] = order.client_order_id
            self.stats['orders_created'] += 1
            
            self.logger.info(f"Created order: {order.client_order_id} - {side} {quantity} {symbol}")
            
            return order
    
    def update_order_status(self, client_order_id: str, 
                           status: OrderStatus,
                           exchange_order_id: str = None,
                           filled_quantity: float = None,
                           average_price: float = None,
                           error: str = None):
        """Update order status from exchange response."""
        with self._lock:
            if client_order_id not in self.orders:
                self.logger.error(f"Order not found: {client_order_id}")
                return
            
            order = self.orders[client_order_id]
            order.status = status
            order.updated_at = datetime.now()
            
            if exchange_order_id:
                order.exchange_order_id = exchange_order_id
            
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
            
            if average_price is not None:
                order.average_fill_price = average_price
            
            if error:
                order.last_error = error
            
            # Update timestamps
            if status == OrderStatus.SUBMITTED:
                order.submitted_at = datetime.now()
            elif status == OrderStatus.FILLED:
                order.filled_at = datetime.now()
                self.stats['orders_filled'] += 1
            elif status == OrderStatus.CANCELLED:
                self.stats['orders_cancelled'] += 1
            elif status == OrderStatus.REJECTED:
                self.stats['orders_rejected'] += 1
            elif status == OrderStatus.FAILED:
                self.stats['orders_failed'] += 1
            
            self.logger.info(f"Order {client_order_id} updated: {status.value}")
    
    def increment_retry(self, client_order_id: str) -> bool:
        """
        Increment retry count for an order.
        Returns True if retry is allowed, False if max retries exceeded.
        """
        with self._lock:
            if client_order_id not in self.orders:
                return False
            
            order = self.orders[client_order_id]
            order.retry_count += 1
            self.stats['total_retries'] += 1
            
            if order.retry_count > order.max_retries:
                order.status = OrderStatus.FAILED
                order.last_error = "Max retries exceeded"
                return False
            
            return True
    
    def get_retry_delay(self, order: Order) -> float:
        """Calculate retry delay with exponential backoff."""
        base = self.config['retry_delay_base']
        max_delay = self.config['retry_delay_max']
        
        delay = base * (2 ** order.retry_count)
        return min(delay, max_delay)
    
    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        return self.orders.get(client_order_id)
    
    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[Order]:
        """Get order by exchange order ID."""
        for order in self.orders.values():
            if order.exchange_order_id == exchange_order_id:
                return order
        return None
    
    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """Get all active (non-terminal) orders."""
        orders = [o for o in self.orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders by status."""
        return [o for o in self.orders.values() if o.status == status]
    
    def get_orders_by_strategy(self, strategy: str) -> List[Order]:
        """Get orders for a specific strategy."""
        return [o for o in self.orders.values() if o.strategy == strategy]
    
    def cancel_order(self, client_order_id: str) -> bool:
        """Mark order for cancellation."""
        with self._lock:
            if client_order_id not in self.orders:
                return False
            
            order = self.orders[client_order_id]
            if not order.is_active:
                self.logger.warning(f"Cannot cancel order in state: {order.status.value}")
                return False
            
            # Note: Actual cancellation is done by ExecutionEngine
            return True
    
    def cancel_all(self, symbol: str = None) -> List[str]:
        """Mark all active orders for cancellation."""
        cancelled = []
        for order in self.get_active_orders(symbol):
            if self.cancel_order(order.client_order_id):
                cancelled.append(order.client_order_id)
        return cancelled
    
    def cleanup_old_orders(self, max_age_hours: int = 24):
        """Remove old terminal orders from cache."""
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            
            old_orders = [
                oid for oid, order in self.orders.items()
                if order.is_terminal and order.updated_at < cutoff
            ]
            
            for oid in old_orders:
                order = self.orders.pop(oid)
                # Also remove from idempotency cache
                if order.idempotency_key in self.idempotency_cache:
                    del self.idempotency_cache[order.idempotency_key]
            
            if old_orders:
                self.logger.info(f"Cleaned up {len(old_orders)} old orders")
    
    def get_statistics(self) -> Dict:
        """Get order manager statistics."""
        return {
            **self.stats,
            'active_orders': len(self.get_active_orders()),
            'total_orders': len(self.orders),
            'fill_rate': (
                self.stats['orders_filled'] / self.stats['orders_created'] * 100
                if self.stats['orders_created'] > 0 else 0
            ),
            'idempotency_cache_size': len(self.idempotency_cache)
        }
    
    def print_orders(self, status_filter: OrderStatus = None):
        """Print order summary."""
        orders = self.orders.values()
        if status_filter:
            orders = [o for o in orders if o.status == status_filter]
        
        orders = sorted(orders, key=lambda o: o.created_at, reverse=True)
        
        print("\n" + "="*90)
        print("📋 ORDER BOOK")
        print("="*90)
        print(f"{'ID':<12} {'Symbol':<10} {'Side':<5} {'Type':<10} {'Qty':>10} {'Price':>10} {'Status':<12} {'Fill%':>6}")
        print("-"*90)
        
        for order in orders[:20]:
            print(f"{order.client_order_id[:12]:<12} "
                  f"{order.symbol:<10} "
                  f"{order.side.value:<5} "
                  f"{order.order_type.value:<10} "
                  f"{order.quantity:>10.4f} "
                  f"{order.price if order.price else 'MARKET':>10} "
                  f"{order.status.value:<12} "
                  f"{order.fill_percentage:>5.1f}%")
        
        if len(orders) > 20:
            print(f"  ... and {len(orders) - 20} more orders")
        
        print("="*90)
        
        # Statistics
        stats = self.get_statistics()
        print(f"\n📊 STATISTICS:")
        print(f"  Created: {stats['orders_created']}, Filled: {stats['orders_filled']}, "
              f"Cancelled: {stats['orders_cancelled']}, Failed: {stats['orders_failed']}")
        print(f"  Fill Rate: {stats['fill_rate']:.1f}%, Active: {stats['active_orders']}")


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = OrderManager()
    
    # Create some orders
    order1 = manager.create_order(
        symbol='BTCUSD',
        side='buy',
        order_type='limit',
        quantity=0.1,
        price=85000,
        strategy='trend_following',
        signal_id='sig_001'
    )
    
    order2 = manager.create_order(
        symbol='ETHUSD',
        side='buy',
        order_type='market',
        quantity=1.0,
        strategy='momentum',
        signal_id='sig_002'
    )
    
    # Test duplicate detection
    order3 = manager.create_order(
        symbol='BTCUSD',
        side='buy',
        order_type='limit',
        quantity=0.1,
        price=85000,
        strategy='trend_following',
        signal_id='sig_001'  # Same signal
    )
    
    print(f"Order 1 ID: {order1.client_order_id}")
    print(f"Order 3 ID: {order3.client_order_id}")  # Should be same as order1
    
    # Update status
    manager.update_order_status(order1.client_order_id, OrderStatus.SUBMITTED, 'EX123')
    manager.update_order_status(order2.client_order_id, OrderStatus.FILLED, 'EX124', 1.0, 3200.50)
    
    manager.print_orders()
