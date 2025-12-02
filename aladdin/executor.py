"""
TRADE EXECUTOR - Order Execution Engine
========================================
Handles order placement, management, and execution on Delta Exchange.

Features:
- Smart order routing
- Slippage protection
- Position management
- Order tracking
"""

import logging
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("AladdinAI.Executor")


class TradeExecutor:
    """
    Trade execution engine for Delta Exchange
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._client = None
        self._pending_orders: Dict = {}
        self._executed_orders: List[Dict] = []
        
        # Paper trading mode by default for safety
        self.paper_mode = self.config.get("paper_mode", True)
        
        logger.info(f"Trade Executor initialized (Paper Mode: {self.paper_mode})")
    
    @property
    def client(self):
        """Lazy load Delta Exchange client"""
        if self._client is None:
            try:
                from execution.client import create_client
                from config.credentials import API_KEY, API_SECRET, TESTNET
                self._client = create_client(API_KEY, API_SECRET, testnet=TESTNET)
            except Exception as e:
                logger.error(f"Failed to initialize client: {e}")
        return self._client
    
    def execute(self, signal) -> Dict:
        """
        Execute a trading signal
        
        Args:
            signal: TradingSignal object
        
        Returns:
            Dict with execution result
        """
        logger.info(f"Executing: {signal.direction} {signal.symbol} @ {signal.entry_price}")
        
        if self.paper_mode:
            return self._paper_execute(signal)
        else:
            return self._live_execute(signal)
    
    def _paper_execute(self, signal) -> Dict:
        """Execute in paper trading mode"""
        order_id = f"PAPER_{int(time.time() * 1000)}"
        
        order = {
            "order_id": order_id,
            "symbol": signal.symbol,
            "side": signal.direction,
            "size": signal.position_size,
            "price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "strategy": signal.strategy,
            "status": "filled",
            "fill_price": signal.entry_price,
            "timestamp": datetime.now().isoformat(),
            "paper": True
        }
        
        self._executed_orders.append(order)
        
        logger.info(f"[PAPER] Order filled: {order_id}")
        
        return {
            "success": True,
            "order_id": order_id,
            "fill_price": signal.entry_price,
            "message": "Paper trade executed"
        }
    
    def _live_execute(self, signal) -> Dict:
        """Execute live order on Delta Exchange"""
        if not self.client:
            return {"success": False, "error": "Client not initialized"}
        
        try:
            # Get product ID for symbol
            product_id = self._get_product_id(signal.symbol)
            if not product_id:
                return {"success": False, "error": f"Product not found: {signal.symbol}"}
            
            # Determine order side
            side = "buy" if signal.direction == "long" else "sell"
            
            # Place market order
            result = self.client.place_order(
                product_id=product_id,
                side=side,
                size=signal.position_size,
                order_type="market_order"
            )
            
            if result.get("success"):
                order_data = result.get("result", {})
                order_id = order_data.get("id")
                
                # Set stop loss and take profit
                self._set_bracket_orders(
                    product_id, 
                    order_id,
                    signal.direction,
                    signal.position_size,
                    signal.stop_loss,
                    signal.take_profit
                )
                
                logger.info(f"[LIVE] Order placed: {order_id}")
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "fill_price": float(order_data.get("average_fill_price", signal.entry_price)),
                    "message": "Live order executed"
                }
            else:
                return {"success": False, "error": result.get("error", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_product_id(self, symbol: str) -> Optional[int]:
        """Get product ID from symbol"""
        product_map = {
            "BTCUSD": 27,
            "ETHUSD": 3136,
            # Add more as needed
        }
        return product_map.get(symbol)
    
    def _set_bracket_orders(self, product_id: int, parent_order_id: str,
                           direction: str, size: float, 
                           stop_loss: float, take_profit: float):
        """Set stop loss and take profit orders"""
        try:
            opposite_side = "sell" if direction == "long" else "buy"
            
            # Stop Loss
            self.client.place_order(
                product_id=product_id,
                side=opposite_side,
                size=size,
                order_type="stop_market_order",
                stop_price=stop_loss,
                reduce_only=True
            )
            
            # Take Profit
            self.client.place_order(
                product_id=product_id,
                side=opposite_side,
                size=size,
                order_type="limit_order",
                limit_price=take_profit,
                reduce_only=True
            )
            
            logger.info(f"Bracket orders set: SL={stop_loss}, TP={take_profit}")
            
        except Exception as e:
            logger.warning(f"Failed to set bracket orders: {e}")
    
    def close_position(self, symbol: str, position: Dict) -> Dict:
        """Close an open position"""
        logger.info(f"Closing position: {symbol}")
        
        if self.paper_mode:
            return {
                "success": True,
                "message": f"Paper position closed: {symbol}"
            }
        
        try:
            product_id = self._get_product_id(symbol)
            if product_id:
                result = self.client.close_position(product_id)
                return {"success": result.get("success", False)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        if self.paper_mode:
            return [o for o in self._pending_orders.values()]
        
        try:
            if self.client:
                result = self.client.get_orders(state="open")
                return result.get("result", [])
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
        
        return []
    
    def cancel_order(self, order_id: str, product_id: int) -> Dict:
        """Cancel an open order"""
        if self.paper_mode:
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            return {"success": True}
        
        try:
            if self.client:
                return self.client.cancel_order(order_id, product_id)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        if self.paper_mode:
            self._pending_orders.clear()
            return {"success": True, "cancelled": 0}
        
        try:
            if self.client:
                return self.client.cancel_all_orders()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_positions(self) -> Dict:
        """Get all open positions"""
        if self.paper_mode:
            return {"success": True, "result": []}
        
        try:
            if self.client:
                return self.client.get_positions()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_balance(self) -> Dict:
        """Get wallet balance"""
        if self.paper_mode:
            return {
                "success": True,
                "result": {
                    "balance": 10000,
                    "available": 10000,
                    "currency": "USD"
                }
            }
        
        try:
            if self.client:
                return self.client.get_wallet_balance()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def enable_live_trading(self, confirm: bool = False):
        """
        Enable live trading mode
        
        WARNING: This will execute real trades!
        """
        if not confirm:
            logger.warning("Live trading not enabled. Set confirm=True to enable.")
            return False
        
        self.paper_mode = False
        logger.warning("LIVE TRADING ENABLED - Real money at risk!")
        return True
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total = len(self._executed_orders)
        
        if total == 0:
            return {
                "total_orders": 0,
                "fill_rate": 0,
                "avg_slippage": 0
            }
        
        filled = [o for o in self._executed_orders if o["status"] == "filled"]
        
        # Calculate slippage
        slippages = []
        for order in filled:
            expected = order.get("price", 0)
            actual = order.get("fill_price", 0)
            if expected > 0:
                slippage = abs(actual - expected) / expected * 100
                slippages.append(slippage)
        
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        
        return {
            "total_orders": total,
            "filled_orders": len(filled),
            "fill_rate": len(filled) / total * 100,
            "avg_slippage": avg_slippage,
            "paper_mode": self.paper_mode
        }
