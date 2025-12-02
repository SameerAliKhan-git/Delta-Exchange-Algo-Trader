"""
ALADDIN - Execution Engine
============================
Handles actual order execution with Delta Exchange API.
"""

import logging
import requests
import hmac
import hashlib
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.credentials import API_KEY, API_SECRET, BASE_URL

from .order_manager import Order, OrderStatus, OrderType, OrderSide, OrderManager


@dataclass
class ExecutionResult:
    """Result of order execution attempt."""
    success: bool
    order: Order
    exchange_order_id: Optional[str] = None
    message: str = ""
    error_code: Optional[str] = None
    raw_response: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ExecutionEngine:
    """
    Executes orders against Delta Exchange API.
    
    Features:
    - Authenticated API calls with HMAC signing
    - Automatic retry with backoff
    - Smart order routing
    - Paper trading mode
    - Bracket order support
    """
    
    def __init__(self, order_manager: OrderManager = None, paper_mode: bool = True):
        self.logger = logging.getLogger('Aladdin.Execution')
        self.order_manager = order_manager or OrderManager()
        self.paper_mode = paper_mode
        
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.base_url = BASE_URL
        
        # Paper trading simulation
        self._paper_fills: Dict[str, Dict] = {}
        
        if paper_mode:
            self.logger.info("Execution Engine initialized in PAPER MODE")
        else:
            self.logger.info("Execution Engine initialized in LIVE MODE")
    
    def _sign_request(self, method: str, endpoint: str, 
                     body: str = "", timestamp: str = None) -> Dict[str, str]:
        """Generate authentication headers for Delta Exchange API."""
        timestamp = timestamp or str(int(time.time()))
        
        # Create signature payload
        signature_payload = f"{method}{timestamp}{endpoint}{body}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'api-key': self.api_key,
            'signature': signature,
            'timestamp': timestamp,
            'Content-Type': 'application/json'
        }
    
    def execute(self, order: Order) -> ExecutionResult:
        """
        Execute an order.
        
        Handles both paper and live execution.
        """
        self.order_manager.update_order_status(
            order.client_order_id,
            OrderStatus.SUBMITTED
        )
        
        if self.paper_mode:
            return self._paper_execute(order)
        else:
            return self._live_execute(order)
    
    def _paper_execute(self, order: Order) -> ExecutionResult:
        """Simulate order execution for paper trading."""
        try:
            # Get current price for the symbol
            price = self._get_current_price(order.symbol)
            
            if price is None:
                return ExecutionResult(
                    success=False,
                    order=order,
                    message="Could not get price for symbol",
                    error_code="PRICE_UNAVAILABLE"
                )
            
            # Simulate fill
            fill_price = price
            if order.order_type == OrderType.LIMIT:
                # For limit orders, use the limit price if it would fill
                if order.side == OrderSide.BUY and order.price >= price:
                    fill_price = order.price
                elif order.side == OrderSide.SELL and order.price <= price:
                    fill_price = order.price
                else:
                    # Limit order wouldn't fill at current price
                    self.order_manager.update_order_status(
                        order.client_order_id,
                        OrderStatus.OPEN,
                        f"PAPER_{order.client_order_id[:8]}"
                    )
                    return ExecutionResult(
                        success=True,
                        order=order,
                        exchange_order_id=f"PAPER_{order.client_order_id[:8]}",
                        message="Limit order placed (unfilled)"
                    )
            
            # Simulate market or filled limit order
            paper_order_id = f"PAPER_{order.client_order_id[:8]}"
            
            self.order_manager.update_order_status(
                order.client_order_id,
                OrderStatus.FILLED,
                paper_order_id,
                order.quantity,
                fill_price
            )
            
            # Store paper fill
            self._paper_fills[order.client_order_id] = {
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': fill_price,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"📝 PAPER FILL: {order.side.value} {order.quantity} {order.symbol} @ {fill_price}")
            
            return ExecutionResult(
                success=True,
                order=order,
                exchange_order_id=paper_order_id,
                message=f"Paper fill @ {fill_price}"
            )
            
        except Exception as e:
            self.logger.error(f"Paper execution error: {e}")
            self.order_manager.update_order_status(
                order.client_order_id,
                OrderStatus.FAILED,
                error=str(e)
            )
            return ExecutionResult(
                success=False,
                order=order,
                message=str(e),
                error_code="PAPER_ERROR"
            )
    
    def _live_execute(self, order: Order) -> ExecutionResult:
        """Execute order against live Delta Exchange API."""
        try:
            # Build order payload
            payload = self._build_order_payload(order)
            
            # Send order
            endpoint = "/v2/orders"
            body = json.dumps(payload)
            headers = self._sign_request("POST", endpoint, body)
            
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                data=body,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                result_order = data.get('result', {})
                exchange_id = str(result_order.get('id', ''))
                
                self.order_manager.update_order_status(
                    order.client_order_id,
                    OrderStatus.OPEN,
                    exchange_id
                )
                
                self.logger.info(f"🚀 LIVE ORDER: {order.side.value} {order.quantity} {order.symbol} - ID: {exchange_id}")
                
                return ExecutionResult(
                    success=True,
                    order=order,
                    exchange_order_id=exchange_id,
                    message="Order submitted",
                    raw_response=data
                )
            else:
                error_msg = data.get('error', {}).get('message', 'Unknown error')
                
                self.order_manager.update_order_status(
                    order.client_order_id,
                    OrderStatus.REJECTED,
                    error=error_msg
                )
                
                return ExecutionResult(
                    success=False,
                    order=order,
                    message=error_msg,
                    error_code=data.get('error', {}).get('code'),
                    raw_response=data
                )
                
        except Exception as e:
            self.logger.error(f"Live execution error: {e}")
            
            # Check if we should retry
            if self.order_manager.increment_retry(order.client_order_id):
                delay = self.order_manager.get_retry_delay(order)
                self.logger.info(f"Retrying in {delay:.1f}s (attempt {order.retry_count})")
                time.sleep(delay)
                return self._live_execute(order)
            
            self.order_manager.update_order_status(
                order.client_order_id,
                OrderStatus.FAILED,
                error=str(e)
            )
            
            return ExecutionResult(
                success=False,
                order=order,
                message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    def _build_order_payload(self, order: Order) -> Dict:
        """Build order payload for Delta Exchange API."""
        payload = {
            'product_id': self._get_product_id(order.symbol),
            'size': order.quantity,
            'side': order.side.value,
            'order_type': order.order_type.value.replace('_', ' '),
            'client_order_id': order.client_order_id
        }
        
        # Add price for limit orders
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if order.price:
                payload['limit_price'] = str(order.price)
        
        # Add stop price for stop orders
        if order.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            if order.stop_price:
                payload['stop_price'] = str(order.stop_price)
        
        # Add bracket orders if specified
        if order.stop_loss:
            payload['stop_loss_price'] = str(order.stop_loss)
        
        if order.take_profit:
            payload['take_profit_price'] = str(order.take_profit)
        
        return payload
    
    def _get_product_id(self, symbol: str) -> int:
        """Get product ID for a symbol."""
        # This would normally come from the catalog
        # For now, use a simple mapping for common symbols
        product_ids = {
            'BTCUSD': 139,
            'ETHUSD': 140,
            'SOLUSD': 141,
        }
        return product_ids.get(symbol, 139)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            response = requests.get(
                f"{self.base_url}/v2/tickers/{symbol}",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and 'result' in data:
                return float(data['result'].get('mark_price', 0))
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def cancel_order(self, order: Order) -> ExecutionResult:
        """Cancel an open order."""
        if self.paper_mode:
            # Paper cancel
            self.order_manager.update_order_status(
                order.client_order_id,
                OrderStatus.CANCELLED
            )
            return ExecutionResult(
                success=True,
                order=order,
                message="Paper order cancelled"
            )
        
        try:
            # Live cancel
            endpoint = f"/v2/orders/{order.exchange_order_id}"
            headers = self._sign_request("DELETE", endpoint)
            
            response = requests.delete(
                f"{self.base_url}{endpoint}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                self.order_manager.update_order_status(
                    order.client_order_id,
                    OrderStatus.CANCELLED
                )
                return ExecutionResult(
                    success=True,
                    order=order,
                    message="Order cancelled"
                )
            else:
                return ExecutionResult(
                    success=False,
                    order=order,
                    message=data.get('error', {}).get('message', 'Cancel failed')
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                order=order,
                message=str(e)
            )
    
    def cancel_all_orders(self, symbol: str = None) -> List[ExecutionResult]:
        """Cancel all active orders."""
        results = []
        
        for order in self.order_manager.get_active_orders(symbol):
            result = self.cancel_order(order)
            results.append(result)
        
        return results
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions (paper or live)."""
        if self.paper_mode:
            return self._get_paper_positions()
        else:
            return self._get_live_positions()
    
    def _get_paper_positions(self) -> Dict[str, Dict]:
        """Calculate paper positions from fills."""
        positions = {}
        
        for fill in self._paper_fills.values():
            symbol = fill['symbol']
            qty = fill['quantity']
            if fill['side'] == 'sell':
                qty = -qty
            
            if symbol in positions:
                positions[symbol]['quantity'] += qty
                # Recalculate average price
                positions[symbol]['avg_price'] = (
                    positions[symbol]['avg_price'] + fill['price']
                ) / 2
            else:
                positions[symbol] = {
                    'quantity': qty,
                    'avg_price': fill['price'],
                    'side': 'long' if qty > 0 else 'short'
                }
        
        # Remove zero positions
        positions = {k: v for k, v in positions.items() if abs(v['quantity']) > 0.0001}
        
        return positions
    
    def _get_live_positions(self) -> Dict[str, Dict]:
        """Get live positions from Delta Exchange."""
        try:
            endpoint = "/v2/positions"
            headers = self._sign_request("GET", endpoint)
            
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                positions = {}
                for pos in data.get('result', []):
                    symbol = pos.get('product', {}).get('symbol', '')
                    positions[symbol] = {
                        'quantity': float(pos.get('size', 0)),
                        'avg_price': float(pos.get('entry_price', 0)),
                        'side': pos.get('side', ''),
                        'unrealized_pnl': float(pos.get('unrealized_pnl', 0))
                    }
                return positions
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def print_status(self):
        """Print execution engine status."""
        print("\n" + "="*60)
        print(f"⚙️ EXECUTION ENGINE STATUS")
        print("="*60)
        
        print(f"\n  Mode: {'📝 PAPER' if self.paper_mode else '🔴 LIVE'}")
        
        # Positions
        positions = self.get_positions()
        if positions:
            print(f"\n  📊 POSITIONS:")
            for symbol, pos in positions.items():
                side_emoji = "📈" if pos['side'] == 'long' else "📉"
                print(f"    {side_emoji} {symbol}: {pos['quantity']:.4f} @ ${pos['avg_price']:.2f}")
        else:
            print("\n  No open positions")
        
        # Order stats
        stats = self.order_manager.get_statistics()
        print(f"\n  📋 ORDER STATS:")
        print(f"    Created: {stats['orders_created']}")
        print(f"    Filled: {stats['orders_filled']} ({stats['fill_rate']:.1f}%)")
        print(f"    Active: {stats['active_orders']}")
        print(f"    Retries: {stats['total_retries']}")
        
        print("="*60)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = ExecutionEngine(paper_mode=True)
    
    # Create and execute test order
    order = engine.order_manager.create_order(
        symbol='BTCUSD',
        side='buy',
        order_type='market',
        quantity=0.01,
        strategy='test'
    )
    
    result = engine.execute(order)
    print(f"Execution result: {result.success} - {result.message}")
    
    engine.print_status()
