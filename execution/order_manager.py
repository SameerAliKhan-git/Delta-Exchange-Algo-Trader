"""
Order Manager - Handle order lifecycle and execution

Provides:
- Order submission with validation
- Order tracking and state management
- Bracket orders (take profit / stop loss)
"""

import time
import uuid
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .client import DeltaClient, OrderSide, OrderType, TimeInForce, OrderStatus


@dataclass
class OrderRequest:
    """Order request"""
    symbol: str
    product_id: int
    side: OrderSide
    size: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float = None
    stop_price: float = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    post_only: bool = False
    take_profit: float = None
    stop_loss: float = None
    client_order_id: str = None
    
    def __post_init__(self):
        if self.client_order_id is None:
            self.client_order_id = str(uuid.uuid4())[:8]


@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: int = None
    client_order_id: str = None
    status: OrderStatus = None
    filled_size: float = 0.0
    avg_price: float = 0.0
    error: str = None
    timestamp: float = field(default_factory=time.time)
    
    # Bracket order IDs
    take_profit_id: int = None
    stop_loss_id: int = None


class OrderManager:
    """
    Manage order lifecycle and execution
    """
    
    def __init__(
        self,
        client: DeltaClient,
        max_retries: int = 3,
        on_order_update: Callable[[OrderResult], None] = None
    ):
        """
        Initialize order manager
        
        Args:
            client: Delta Exchange client
            max_retries: Max retry attempts
            on_order_update: Callback for order updates
        """
        self.client = client
        self.max_retries = max_retries
        self.on_order_update = on_order_update
        
        # Track orders
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.active_orders: Dict[int, OrderResult] = {}
        self.order_history: List[OrderResult] = []
    
    def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit order to exchange
        
        Args:
            request: OrderRequest
        
        Returns:
            OrderResult
        """
        # Validate request
        validation = self._validate_request(request)
        if not validation['valid']:
            return OrderResult(
                success=False,
                client_order_id=request.client_order_id,
                error=validation['error']
            )
        
        # Track pending
        self.pending_orders[request.client_order_id] = request
        
        # Submit main order
        result = self._execute_order(request)
        
        # Handle bracket orders
        if result.success and (request.take_profit or request.stop_loss):
            self._submit_bracket_orders(request, result)
        
        # Update tracking
        del self.pending_orders[request.client_order_id]
        if result.order_id:
            self.active_orders[result.order_id] = result
        self.order_history.append(result)
        
        # Callback
        if self.on_order_update:
            self.on_order_update(result)
        
        return result
    
    def cancel_order(self, order_id: int, product_id: int) -> bool:
        """
        Cancel order
        
        Args:
            order_id: Order ID
            product_id: Product ID
        
        Returns:
            Success status
        """
        try:
            response = self.client.cancel_order(order_id, product_id)
            
            if 'result' in response:
                if order_id in self.active_orders:
                    self.active_orders[order_id].status = OrderStatus.CANCELLED
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def cancel_all(self, product_id: int = None) -> int:
        """
        Cancel all orders
        
        Args:
            product_id: Optional product filter
        
        Returns:
            Number of orders cancelled
        """
        try:
            response = self.client.cancel_all_orders(product_id)
            
            cancelled = 0
            if 'result' in response:
                cancelled = len(response['result'].get('cancelled_orders', []))
            
            # Update tracking
            for order_id in list(self.active_orders.keys()):
                self.active_orders[order_id].status = OrderStatus.CANCELLED
            
            return cancelled
            
        except Exception:
            return 0
    
    def modify_order(
        self,
        order_id: int,
        product_id: int,
        size: float = None,
        limit_price: float = None
    ) -> bool:
        """
        Modify existing order
        
        Args:
            order_id: Order ID
            product_id: Product ID
            size: New size
            limit_price: New price
        
        Returns:
            Success status
        """
        try:
            response = self.client.modify_order(
                order_id,
                product_id,
                size=size,
                limit_price=limit_price
            )
            
            return 'result' in response
            
        except Exception:
            return False
    
    def get_order_status(self, order_id: int) -> Optional[OrderResult]:
        """
        Get current order status
        
        Args:
            order_id: Order ID
        
        Returns:
            OrderResult or None
        """
        try:
            response = self.client.get_order(order_id)
            
            if 'result' not in response:
                return None
            
            order = response['result']
            
            return OrderResult(
                success=True,
                order_id=order['id'],
                client_order_id=order.get('client_order_id'),
                status=self._parse_status(order.get('state')),
                filled_size=float(order.get('size_filled', 0)),
                avg_price=float(order.get('average_fill_price', 0))
            )
            
        except Exception:
            return None
    
    def sync_orders(self, product_id: int = None) -> None:
        """
        Sync order state with exchange
        
        Args:
            product_id: Optional product filter
        """
        try:
            response = self.client.get_orders(product_id, state='open')
            
            if 'result' not in response:
                return
            
            # Update active orders
            exchange_orders = {o['id']: o for o in response['result']}
            
            for order_id in list(self.active_orders.keys()):
                if order_id in exchange_orders:
                    order = exchange_orders[order_id]
                    self.active_orders[order_id].status = self._parse_status(order.get('state'))
                    self.active_orders[order_id].filled_size = float(order.get('size_filled', 0))
                else:
                    # Order no longer open - mark as completed/cancelled
                    del self.active_orders[order_id]
                    
        except Exception:
            pass
    
    def _execute_order(self, request: OrderRequest) -> OrderResult:
        """Execute order with retries"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.place_order(
                    product_id=request.product_id,
                    side=request.side,
                    size=request.size,
                    order_type=request.order_type,
                    limit_price=request.limit_price,
                    stop_price=request.stop_price,
                    time_in_force=request.time_in_force,
                    reduce_only=request.reduce_only,
                    post_only=request.post_only,
                    client_order_id=request.client_order_id
                )
                
                if 'result' in response:
                    order = response['result']
                    return OrderResult(
                        success=True,
                        order_id=order.get('id'),
                        client_order_id=request.client_order_id,
                        status=self._parse_status(order.get('state')),
                        filled_size=float(order.get('size_filled', 0)),
                        avg_price=float(order.get('average_fill_price', 0))
                    )
                
                last_error = response.get('error', 'Unknown error')
                
            except Exception as e:
                last_error = str(e)
                time.sleep(0.5)
        
        return OrderResult(
            success=False,
            client_order_id=request.client_order_id,
            error=last_error
        )
    
    def _submit_bracket_orders(self, request: OrderRequest, main_result: OrderResult) -> None:
        """Submit take profit and stop loss orders"""
        opposite_side = OrderSide.SELL if request.side == OrderSide.BUY else OrderSide.BUY
        
        # Take profit order
        if request.take_profit:
            try:
                tp_response = self.client.place_order(
                    product_id=request.product_id,
                    side=opposite_side,
                    size=request.size,
                    order_type=OrderType.LIMIT,
                    limit_price=request.take_profit,
                    reduce_only=True
                )
                if 'result' in tp_response:
                    main_result.take_profit_id = tp_response['result'].get('id')
            except Exception:
                pass
        
        # Stop loss order
        if request.stop_loss:
            try:
                sl_response = self.client.place_order(
                    product_id=request.product_id,
                    side=opposite_side,
                    size=request.size,
                    order_type=OrderType.STOP_MARKET,
                    stop_price=request.stop_loss,
                    reduce_only=True
                )
                if 'result' in sl_response:
                    main_result.stop_loss_id = sl_response['result'].get('id')
            except Exception:
                pass
    
    def _validate_request(self, request: OrderRequest) -> Dict:
        """Validate order request"""
        if request.size <= 0:
            return {'valid': False, 'error': 'Size must be positive'}
        
        if request.order_type == OrderType.LIMIT and request.limit_price is None:
            return {'valid': False, 'error': 'Limit price required for limit orders'}
        
        if request.order_type in (OrderType.STOP_LIMIT, OrderType.STOP_MARKET):
            if request.stop_price is None:
                return {'valid': False, 'error': 'Stop price required for stop orders'}
        
        if request.take_profit and request.stop_loss:
            if request.side == OrderSide.BUY:
                if request.take_profit <= request.stop_loss:
                    return {'valid': False, 'error': 'TP must be above SL for long'}
            else:
                if request.take_profit >= request.stop_loss:
                    return {'valid': False, 'error': 'TP must be below SL for short'}
        
        return {'valid': True}
    
    def _parse_status(self, state: str) -> OrderStatus:
        """Parse order status from API response"""
        status_map = {
            'pending': OrderStatus.PENDING,
            'open': OrderStatus.OPEN,
            'filled': OrderStatus.FILLED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'cancelled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED
        }
        return status_map.get(state, OrderStatus.PENDING)
