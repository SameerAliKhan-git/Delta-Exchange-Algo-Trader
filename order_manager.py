"""
Order Manager for Delta Exchange Algo Trading Bot
Handles order lifecycle, stop losses, and position management
"""

import uuid
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

from config import get_config
from logger import get_logger, get_audit_logger
from delta_client import get_delta_client, DeltaOrderError
from risk_manager import get_risk_manager


class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order types"""
    MARKET = "market_order"
    LIMIT = "limit_order"
    STOP_MARKET = "stop_market_order"
    STOP_LIMIT = "stop_limit_order"


@dataclass
class Order:
    """Order representation"""
    client_order_id: str
    product_id: int
    side: str
    size: float
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    
    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)
    
    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)


@dataclass
class Position:
    """Position representation"""
    product_id: int
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    trailing_stop_active: bool = False
    opened_at: datetime = field(default_factory=datetime.utcnow)


class OrderManager:
    """
    Order management engine
    
    Responsibilities:
    - Order placement with idempotency
    - Order tracking and status updates
    - Stop loss and take profit management
    - Position tracking
    - Retry logic for failed orders
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.audit = get_audit_logger()
        self.client = get_delta_client()
        self.risk_manager = get_risk_manager()
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[int, Position] = {}
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_fill_callbacks: List[Callable[[Order], None]] = []
        self._on_stop_triggered_callbacks: List[Callable[[Order], None]] = []
    
    def generate_order_id(self) -> str:
        """Generate a unique client order ID"""
        return f"delta_algo_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    def on_fill(self, callback: Callable[[Order], None]):
        """Register callback for order fills"""
        self._on_fill_callbacks.append(callback)
    
    def on_stop_triggered(self, callback: Callable[[Order], None]):
        """Register callback for stop order triggers"""
        self._on_stop_triggered_callbacks.append(callback)
    
    def place_market_order(
        self,
        product_id: int,
        side: str,
        size: float,
        reduce_only: bool = False,
        attach_stop: bool = True,
        stop_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Place a market order with optional stop loss and take profit
        
        Args:
            product_id: Product to trade
            side: 'buy' or 'sell'
            size: Order size
            reduce_only: Only reduce existing position
            attach_stop: Automatically attach stop loss
            stop_price: Stop loss price (auto-calculated if None)
            take_profit_price: Take profit price (optional)
        
        Returns:
            Order object if successful, None otherwise
        """
        # Check if trading is allowed
        risk_check = self.risk_manager.check_can_trade()
        if not risk_check.allowed:
            self.logger.warning(
                "Trade blocked by risk manager",
                violation=risk_check.violation.value if risk_check.violation else None,
                message=risk_check.message
            )
            return None
        
        client_order_id = self.generate_order_id()
        
        order = Order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=side,
            size=size,
            order_type=OrderType.MARKET
        )
        
        with self._lock:
            self._orders[client_order_id] = order
        
        try:
            # Place the order
            response = self.client.place_market_order(
                product_id=product_id,
                size=size,
                side=side,
                reduce_only=reduce_only,
                client_order_id=client_order_id
            )
            
            # Update order with response
            order.exchange_order_id = str(response.get('id', ''))
            order.status = self._parse_order_status(response.get('state', ''))
            order.filled_size = float(response.get('size', 0)) - float(response.get('unfilled_size', 0))
            order.avg_fill_price = float(response.get('average_fill_price', 0)) or self.client.get_current_price(product_id)
            order.updated_at = datetime.utcnow()
            
            self.logger.log_order(
                action="placed",
                order_id=order.exchange_order_id,
                product_symbol=str(product_id),
                side=side,
                size=size,
                order_type="market",
                status=order.status.value
            )
            
            # Record trade entry with risk manager
            if order.status == OrderStatus.FILLED:
                self.risk_manager.record_trade_entry(
                    trade_id=client_order_id,
                    product_id=product_id,
                    side=side,
                    size=size,
                    entry_price=order.avg_fill_price
                )
                
                # Update position tracking
                self._update_position(order)
                
                # Attach stop loss if requested
                if attach_stop and not reduce_only:
                    self._attach_stop_loss(order, stop_price, take_profit_price)
                
                # Trigger callbacks
                for callback in self._on_fill_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        self.logger.error("Fill callback error", error=str(e))
            
            return order
            
        except DeltaOrderError as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.logger.error("Order placement failed", error=str(e))
            return None
    
    def place_limit_order(
        self,
        product_id: int,
        side: str,
        size: float,
        limit_price: float,
        post_only: bool = False,
        reduce_only: bool = False
    ) -> Optional[Order]:
        """Place a limit order"""
        # Check if trading is allowed
        risk_check = self.risk_manager.check_can_trade()
        if not risk_check.allowed:
            self.logger.warning("Trade blocked", message=risk_check.message)
            return None
        
        client_order_id = self.generate_order_id()
        
        order = Order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=side,
            size=size,
            order_type=OrderType.LIMIT,
            limit_price=limit_price
        )
        
        with self._lock:
            self._orders[client_order_id] = order
        
        try:
            response = self.client.place_limit_order(
                product_id=product_id,
                size=size,
                side=side,
                limit_price=limit_price,
                post_only=post_only,
                reduce_only=reduce_only,
                client_order_id=client_order_id
            )
            
            order.exchange_order_id = str(response.get('id', ''))
            order.status = self._parse_order_status(response.get('state', ''))
            order.updated_at = datetime.utcnow()
            
            return order
            
        except DeltaOrderError as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return None
    
    def place_stop_order(
        self,
        product_id: int,
        side: str,
        size: float,
        stop_price: float,
        reduce_only: bool = True,
        is_stop_limit: bool = False,
        limit_price: Optional[float] = None
    ) -> Optional[Order]:
        """Place a stop loss order"""
        client_order_id = self.generate_order_id()
        
        order_type = OrderType.STOP_LIMIT if is_stop_limit else OrderType.STOP_MARKET
        
        order = Order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=side,
            size=size,
            order_type=order_type,
            stop_price=stop_price,
            limit_price=limit_price
        )
        
        with self._lock:
            self._orders[client_order_id] = order
        
        try:
            if is_stop_limit and limit_price:
                response = self.client.place_stop_limit_order(
                    product_id=product_id,
                    size=size,
                    side=side,
                    stop_price=stop_price,
                    limit_price=limit_price,
                    reduce_only=reduce_only,
                    client_order_id=client_order_id
                )
            else:
                response = self.client.place_stop_market_order(
                    product_id=product_id,
                    size=size,
                    side=side,
                    stop_price=stop_price,
                    reduce_only=reduce_only,
                    client_order_id=client_order_id
                )
            
            order.exchange_order_id = str(response.get('id', ''))
            order.status = self._parse_order_status(response.get('state', ''))
            order.updated_at = datetime.utcnow()
            
            self.logger.log_order(
                action="stop_placed",
                order_id=order.exchange_order_id,
                product_symbol=str(product_id),
                side=side,
                size=size,
                price=stop_price,
                order_type=order_type.value
            )
            
            return order
            
        except DeltaOrderError as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.logger.error("Stop order failed", error=str(e))
            return None
    
    def _attach_stop_loss(
        self,
        entry_order: Order,
        stop_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ):
        """Attach stop loss and optionally take profit to a filled order"""
        if not entry_order.avg_fill_price:
            return
        
        # Determine stop and TP prices
        if stop_price is None:
            stop_price = self.risk_manager.get_stop_price(
                entry_price=entry_order.avg_fill_price,
                side=entry_order.side
            )
        
        # Stop side is opposite of entry
        stop_side = 'sell' if entry_order.side == 'buy' else 'buy'
        
        # Place stop loss
        stop_order = self.place_stop_order(
            product_id=entry_order.product_id,
            side=stop_side,
            size=entry_order.filled_size,
            stop_price=stop_price,
            reduce_only=True
        )
        
        if stop_order:
            position = self._positions.get(entry_order.product_id)
            if position:
                position.stop_order_id = stop_order.client_order_id
        
        # Place take profit if specified
        if take_profit_price:
            self.place_limit_order(
                product_id=entry_order.product_id,
                side=stop_side,
                size=entry_order.filled_size,
                limit_price=take_profit_price,
                reduce_only=True
            )
    
    def _update_position(self, order: Order):
        """Update position tracking after a fill"""
        with self._lock:
            product_id = order.product_id
            
            if product_id not in self._positions:
                # New position
                self._positions[product_id] = Position(
                    product_id=product_id,
                    side='long' if order.side == 'buy' else 'short',
                    size=order.filled_size,
                    entry_price=order.avg_fill_price
                )
            else:
                position = self._positions[product_id]
                
                # Check if same direction or closing
                is_same_direction = (
                    (position.side == 'long' and order.side == 'buy') or
                    (position.side == 'short' and order.side == 'sell')
                )
                
                if is_same_direction:
                    # Adding to position
                    total_size = position.size + order.filled_size
                    position.entry_price = (
                        (position.entry_price * position.size + order.avg_fill_price * order.filled_size)
                        / total_size
                    )
                    position.size = total_size
                else:
                    # Reducing or closing position
                    if order.filled_size >= position.size:
                        # Position closed
                        pnl = self._calculate_pnl(position, order.avg_fill_price)
                        self.risk_manager.record_trade_exit(
                            trade_id=order.client_order_id,
                            exit_price=order.avg_fill_price,
                            pnl=pnl
                        )
                        del self._positions[product_id]
                    else:
                        # Partial close
                        position.size -= order.filled_size
    
    def _calculate_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate realized PnL for a position"""
        if position.side == 'long':
            return (exit_price - position.entry_price) * position.size
        else:
            return (position.entry_price - exit_price) * position.size
    
    def _parse_order_status(self, state: str) -> OrderStatus:
        """Parse exchange order state to OrderStatus"""
        state = state.lower()
        if state in ('filled', 'closed'):
            return OrderStatus.FILLED
        elif state in ('pending', 'open'):
            return OrderStatus.SUBMITTED
        elif state == 'cancelled':
            return OrderStatus.CANCELLED
        elif state in ('rejected', 'failed'):
            return OrderStatus.REJECTED
        elif 'partial' in state:
            return OrderStatus.PARTIALLY_FILLED
        return OrderStatus.PENDING
    
    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an order by client order ID"""
        with self._lock:
            order = self._orders.get(client_order_id)
            if not order or not order.exchange_order_id:
                return False
            
            try:
                self.client.cancel_order(order.exchange_order_id, order.product_id)
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow()
                
                self.audit.log_order_cancelled({
                    "client_order_id": client_order_id,
                    "exchange_order_id": order.exchange_order_id,
                    "product_id": order.product_id
                })
                
                return True
            except Exception as e:
                self.logger.error("Cancel order failed", error=str(e))
                return False
    
    def cancel_all_orders(self, product_id: Optional[int] = None) -> int:
        """Cancel all open orders"""
        try:
            self.client.cancel_all_orders(product_id)
            
            cancelled_count = 0
            with self._lock:
                for order in self._orders.values():
                    if order.is_active:
                        if product_id is None or order.product_id == product_id:
                            order.status = OrderStatus.CANCELLED
                            cancelled_count += 1
            
            self.logger.info("Cancelled all orders", count=cancelled_count)
            return cancelled_count
            
        except Exception as e:
            self.logger.error("Cancel all orders failed", error=str(e))
            return 0
    
    def close_position(self, product_id: int) -> Optional[Order]:
        """Close a position by placing a reducing market order"""
        with self._lock:
            position = self._positions.get(product_id)
            if not position:
                self.logger.info("No position to close", product_id=product_id)
                return None
        
        # Cancel any existing stop orders for this position
        if position.stop_order_id:
            self.cancel_order(position.stop_order_id)
        
        # Place closing order
        close_side = 'sell' if position.side == 'long' else 'buy'
        
        return self.place_market_order(
            product_id=product_id,
            side=close_side,
            size=position.size,
            reduce_only=True,
            attach_stop=False
        )
    
    def close_all_positions(self) -> int:
        """Close all open positions"""
        closed_count = 0
        product_ids = list(self._positions.keys())
        
        for product_id in product_ids:
            if self.close_position(product_id):
                closed_count += 1
        
        self.logger.info("Closed all positions", count=closed_count)
        return closed_count
    
    def sync_orders(self):
        """Sync order statuses with exchange"""
        try:
            open_orders = self.client.get_open_orders()
            
            # Update tracked orders
            exchange_order_ids = {str(o.get('id')) for o in open_orders}
            
            with self._lock:
                for order in self._orders.values():
                    if order.exchange_order_id:
                        if order.exchange_order_id not in exchange_order_ids and order.is_active:
                            # Order no longer open - likely filled or cancelled
                            try:
                                order_info = self.client.get_order(order.exchange_order_id)
                                order.status = self._parse_order_status(order_info.get('state', ''))
                                order.filled_size = float(order_info.get('size', 0)) - float(order_info.get('unfilled_size', 0))
                                order.avg_fill_price = float(order_info.get('average_fill_price', 0))
                                order.updated_at = datetime.utcnow()
                            except Exception:
                                pass
        except Exception as e:
            self.logger.error("Sync orders failed", error=str(e))
    
    def sync_positions(self):
        """Sync positions with exchange"""
        try:
            positions = self.client.get_positions()
            
            with self._lock:
                # Update existing and add new positions
                active_product_ids = set()
                
                for pos in positions:
                    size = float(pos.get('size', 0))
                    if abs(size) > 0:
                        product_id = int(pos.get('product_id', 0))
                        active_product_ids.add(product_id)
                        
                        if product_id not in self._positions:
                            self._positions[product_id] = Position(
                                product_id=product_id,
                                side='long' if size > 0 else 'short',
                                size=abs(size),
                                entry_price=float(pos.get('entry_price', 0))
                            )
                        else:
                            existing = self._positions[product_id]
                            existing.size = abs(size)
                            existing.entry_price = float(pos.get('entry_price', 0))
                            existing.unrealized_pnl = float(pos.get('unrealized_pnl', 0))
                
                # Remove closed positions
                for product_id in list(self._positions.keys()):
                    if product_id not in active_product_ids:
                        del self._positions[product_id]
                        
        except Exception as e:
            self.logger.error("Sync positions failed", error=str(e))
    
    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get an order by client order ID"""
        return self._orders.get(client_order_id)
    
    def get_position(self, product_id: int) -> Optional[Position]:
        """Get position for a product"""
        return self._positions.get(product_id)
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self._positions.values())
    
    def get_open_orders(self) -> List[Order]:
        """Get all active orders"""
        return [o for o in self._orders.values() if o.is_active]
    
    def has_position(self, product_id: int) -> bool:
        """Check if we have a position in a product"""
        return product_id in self._positions


# Singleton instance
_order_manager: Optional[OrderManager] = None


def get_order_manager() -> OrderManager:
    """Get or create the global order manager"""
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager


if __name__ == "__main__":
    # Test order manager
    om = get_order_manager()
    
    print("Order manager initialized")
    print(f"Open orders: {len(om.get_open_orders())}")
    print(f"Open positions: {len(om.get_all_positions())}")
