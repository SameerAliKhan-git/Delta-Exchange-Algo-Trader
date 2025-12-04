#!/usr/bin/env python3
"""
Delta Exchange API Client

Full implementation for Delta Exchange India:
- REST API for order management, positions, account
- WebSocket for real-time market data
- Paper trading mode for testing
"""

import os
import time
import hmac
import hashlib
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

import requests
import aiohttp
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit_order"
    MARKET = "market_order"


class TimeInForce(Enum):
    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class DeltaConfig:
    """Delta Exchange configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("DELTA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("DELTA_API_SECRET", ""))
    base_url: str = field(default_factory=lambda: os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange"))
    ws_url: str = "wss://socket.india.delta.exchange"
    testnet: bool = True
    rate_limit_per_second: int = 10


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float]
    status: str
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    """Position representation."""
    symbol: str
    size: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: float


class DeltaExchangeClient:
    """
    Delta Exchange API Client.
    
    Usage:
        client = DeltaExchangeClient()
        
        # Get markets
        markets = client.get_products()
        
        # Place order
        order = client.place_order(
            symbol="BTCUSD",
            side=OrderSide.BUY,
            size=0.01,
            price=50000
        )
        
        # Get positions
        positions = client.get_positions()
    """
    
    def __init__(self, config: Optional[DeltaConfig] = None):
        self.config = config or DeltaConfig()
        self._session = requests.Session()
        self._last_request_time = 0
        
        if not self.config.api_key or not self.config.api_secret:
            logger.warning("API credentials not set - running in read-only mode")
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        min_interval = 1.0 / self.config.rate_limit_per_second
        elapsed = now - self._last_request_time
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def _sign_request(self, method: str, path: str, timestamp: str, body: str = "") -> str:
        """Generate HMAC signature for request."""
        message = method + timestamp + path + body
        signature = hmac.new(
            self.config.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        auth: bool = False
    ) -> Dict:
        """Make HTTP request to Delta Exchange API."""
        self._rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if auth:
            timestamp = str(int(time.time()))
            body = json.dumps(data) if data else ""
            signature = self._sign_request(method.upper(), endpoint, timestamp, body)
            
            headers.update({
                "api-key": self.config.api_key,
                "timestamp": timestamp,
                "signature": signature
            })
        
        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                json=data if data else None,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    # =========================================================================
    # Public Endpoints
    # =========================================================================
    
    def get_products(self) -> List[Dict]:
        """Get all available products/instruments."""
        response = self._make_request("GET", "/v2/products")
        return response.get("result", [])
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker for a symbol."""
        response = self._make_request("GET", f"/v2/tickers/{symbol}")
        return response.get("result", {})
    
    def get_tickers(self) -> List[Dict]:
        """Get all tickers."""
        response = self._make_request("GET", "/v2/tickers")
        return response.get("result", [])
    
    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """Get orderbook for a symbol."""
        response = self._make_request(
            "GET",
            f"/v2/l2orderbook/{symbol}",
            params={"depth": depth}
        )
        return response.get("result", {})
    
    def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol."""
        response = self._make_request(
            "GET",
            f"/v2/trades/{symbol}",
            params={"size": limit}
        )
        return response.get("result", [])
    
    def get_candles(
        self,
        symbol: str,
        resolution: str = "1h",
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Dict]:
        """
        Get OHLCV candles.
        
        Args:
            symbol: Product symbol
            resolution: Candle resolution (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 1d, 7d, 30d)
            start: Start timestamp (Unix seconds)
            end: End timestamp (Unix seconds)
        """
        params = {"resolution": resolution}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        
        response = self._make_request(
            "GET",
            f"/v2/history/candles",
            params={"symbol": symbol, **params}
        )
        return response.get("result", [])
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate for a perpetual."""
        response = self._make_request("GET", f"/v2/funding_rate/{symbol}")
        return response.get("result", {})
    
    # =========================================================================
    # Private Endpoints (Require Authentication)
    # =========================================================================
    
    def get_wallet_balance(self) -> Dict:
        """Get wallet balance."""
        response = self._make_request("GET", "/v2/wallet/balances", auth=True)
        return response.get("result", {})
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        response = self._make_request("GET", "/v2/positions", auth=True)
        
        positions = []
        for pos in response.get("result", []):
            if pos.get("size", 0) != 0:
                positions.append(Position(
                    symbol=pos.get("product_symbol", ""),
                    size=float(pos.get("size", 0)),
                    entry_price=float(pos.get("entry_price", 0)),
                    mark_price=float(pos.get("mark_price", 0)),
                    liquidation_price=float(pos.get("liquidation_price", 0)),
                    unrealized_pnl=float(pos.get("unrealized_pnl", 0)),
                    realized_pnl=float(pos.get("realized_pnl", 0)),
                    leverage=float(pos.get("leverage", 1))
                ))
        
        return positions
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        params = {}
        if symbol:
            params["product_id"] = symbol
        
        response = self._make_request("GET", "/v2/orders", params=params, auth=True)
        
        orders = []
        for order_data in response.get("result", []):
            orders.append(Order(
                id=str(order_data.get("id", "")),
                symbol=order_data.get("product_symbol", ""),
                side=OrderSide(order_data.get("side", "buy")),
                order_type=OrderType(order_data.get("order_type", "limit_order")),
                size=float(order_data.get("size", 0)),
                price=float(order_data.get("limit_price", 0)) if order_data.get("limit_price") else None,
                status=order_data.get("state", "unknown"),
                filled_size=float(order_data.get("filled_size", 0)),
                average_fill_price=float(order_data.get("average_fill_price", 0))
            ))
        
        return orders
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> Order:
        """
        Place a new order.
        
        Args:
            symbol: Product symbol (e.g., "BTCUSD")
            side: Order side (BUY or SELL)
            size: Order size
            order_type: Order type (LIMIT or MARKET)
            price: Limit price (required for limit orders)
            time_in_force: Time in force
            reduce_only: If True, only reduces position
            client_order_id: Custom order ID
        """
        data = {
            "product_id": self._get_product_id(symbol),
            "side": side.value,
            "size": size,
            "order_type": order_type.value,
            "time_in_force": time_in_force.value,
            "reduce_only": reduce_only
        }
        
        if order_type == OrderType.LIMIT and price:
            data["limit_price"] = str(price)
        
        if client_order_id:
            data["client_order_id"] = client_order_id
        
        response = self._make_request("POST", "/v2/orders", data=data, auth=True)
        order_data = response.get("result", {})
        
        return Order(
            id=str(order_data.get("id", "")),
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            status=order_data.get("state", "pending")
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self._make_request(
                "DELETE",
                f"/v2/orders/{order_id}",
                auth=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders. Returns number cancelled."""
        data = {}
        if symbol:
            data["product_id"] = self._get_product_id(symbol)
        
        response = self._make_request(
            "DELETE",
            "/v2/orders/all",
            data=data if data else None,
            auth=True
        )
        return response.get("result", {}).get("cancelled", 0)
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a product."""
        data = {
            "product_id": self._get_product_id(symbol),
            "leverage": leverage
        }
        
        try:
            self._make_request("POST", "/v2/positions/change_leverage", data=data, auth=True)
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
    
    def _get_product_id(self, symbol: str) -> int:
        """Get product ID from symbol."""
        # Cache products
        if not hasattr(self, "_products_cache"):
            products = self.get_products()
            self._products_cache = {p["symbol"]: p["id"] for p in products}
        
        if symbol in self._products_cache:
            return self._products_cache[symbol]
        
        # Try to find partial match
        for sym, pid in self._products_cache.items():
            if symbol.upper() in sym.upper():
                return pid
        
        raise ValueError(f"Unknown symbol: {symbol}")
    
    # =========================================================================
    # Historical Data
    # =========================================================================
    
    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> 'pd.DataFrame':
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Product symbol
            timeframe: Candle resolution
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max candles to fetch
        """
        import pandas as pd
        
        # Convert dates to timestamps
        start_ts = None
        end_ts = None
        
        if start_date:
            start_ts = int(pd.Timestamp(start_date).timestamp())
        if end_date:
            end_ts = int(pd.Timestamp(end_date).timestamp())
        
        all_candles = []
        current_end = end_ts or int(time.time())
        
        while len(all_candles) < limit:
            candles = self.get_candles(
                symbol=symbol,
                resolution=timeframe,
                end=current_end
            )
            
            if not candles:
                break
            
            all_candles.extend(candles)
            
            # Get oldest timestamp for next iteration
            oldest_ts = min(c.get("time", current_end) for c in candles)
            
            if start_ts and oldest_ts <= start_ts:
                break
            
            current_end = oldest_ts - 1
            
            if len(candles) < 100:  # Likely reached end of data
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles)
        
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Rename columns
        df = df.rename(columns={
            "time": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        
        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")
        
        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Sort and remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        
        return df[["open", "high", "low", "close", "volume"]]


class DeltaWebSocket:
    """
    Delta Exchange WebSocket client for real-time data.
    
    Usage:
        ws = DeltaWebSocket()
        
        @ws.on_ticker
        async def handle_ticker(data):
            print(f"Ticker update: {data}")
        
        await ws.connect()
        await ws.subscribe_ticker("BTCUSD")
    """
    
    def __init__(self, config: Optional[DeltaConfig] = None):
        self.config = config or DeltaConfig()
        self._ws = None
        self._running = False
        self._subscriptions = set()
        
        # Callbacks
        self._ticker_callbacks: List[Callable] = []
        self._orderbook_callbacks: List[Callable] = []
        self._trade_callbacks: List[Callable] = []
        self._order_callbacks: List[Callable] = []
        self._position_callbacks: List[Callable] = []
    
    async def connect(self):
        """Connect to WebSocket."""
        self._ws = await aiohttp.ClientSession().ws_connect(
            self.config.ws_url,
            heartbeat=30
        )
        self._running = True
        logger.info(f"Connected to Delta WebSocket: {self.config.ws_url}")
        
        # Start message handler
        asyncio.create_task(self._handle_messages())
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        while self._running and self._ws:
            try:
                msg = await self._ws.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._dispatch_message(data)
                    
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"WebSocket closed: {msg.type}")
                    break
                    
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
    
    async def _dispatch_message(self, data: Dict):
        """Dispatch message to appropriate callbacks."""
        msg_type = data.get("type")
        
        if msg_type == "ticker":
            for callback in self._ticker_callbacks:
                await callback(data.get("data", {}))
                
        elif msg_type == "l2_orderbook":
            for callback in self._orderbook_callbacks:
                await callback(data.get("data", {}))
                
        elif msg_type == "recent_trade":
            for callback in self._trade_callbacks:
                await callback(data.get("data", {}))
                
        elif msg_type == "orders":
            for callback in self._order_callbacks:
                await callback(data.get("data", {}))
                
        elif msg_type == "positions":
            for callback in self._position_callbacks:
                await callback(data.get("data", {}))
    
    async def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates."""
        await self._send({
            "type": "subscribe",
            "payload": {
                "channels": [{"name": "ticker", "symbols": [symbol]}]
            }
        })
        self._subscriptions.add(f"ticker:{symbol}")
    
    async def subscribe_orderbook(self, symbol: str):
        """Subscribe to orderbook updates."""
        await self._send({
            "type": "subscribe",
            "payload": {
                "channels": [{"name": "l2_orderbook", "symbols": [symbol]}]
            }
        })
        self._subscriptions.add(f"orderbook:{symbol}")
    
    async def subscribe_trades(self, symbol: str):
        """Subscribe to trade updates."""
        await self._send({
            "type": "subscribe",
            "payload": {
                "channels": [{"name": "recent_trade", "symbols": [symbol]}]
            }
        })
        self._subscriptions.add(f"trades:{symbol}")
    
    async def _send(self, data: Dict):
        """Send message to WebSocket."""
        if self._ws:
            await self._ws.send_json(data)
    
    def on_ticker(self, callback: Callable):
        """Register ticker callback."""
        self._ticker_callbacks.append(callback)
        return callback
    
    def on_orderbook(self, callback: Callable):
        """Register orderbook callback."""
        self._orderbook_callbacks.append(callback)
        return callback
    
    def on_trade(self, callback: Callable):
        """Register trade callback."""
        self._trade_callbacks.append(callback)
        return callback


class PaperTradingClient:
    """
    Paper trading client that simulates Delta Exchange.
    
    Uses the same interface as DeltaExchangeClient but doesn't execute real trades.
    """
    
    def __init__(self, initial_balance: float = 100000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Dict] = []
        self._order_counter = 0
        self._real_client = DeltaExchangeClient()  # For market data
    
    def get_products(self) -> List[Dict]:
        """Get products from real exchange."""
        return self._real_client.get_products()
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker from real exchange."""
        return self._real_client.get_ticker(symbol)
    
    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """Get orderbook from real exchange."""
        return self._real_client.get_orderbook(symbol, depth)
    
    def get_wallet_balance(self) -> Dict:
        """Get simulated wallet balance."""
        return {
            "balance": self.balance,
            "available_balance": self.balance - self._margin_used(),
            "unrealized_pnl": self._total_unrealized_pnl()
        }
    
    def get_positions(self) -> List[Position]:
        """Get simulated positions."""
        return list(self.positions.values())
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Place simulated order."""
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter}"
        
        # Get current market price
        ticker = self.get_ticker(symbol)
        market_price = float(ticker.get("mark_price", ticker.get("close", 0)))
        
        # Simulate fill
        fill_price = price if price else market_price
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=fill_price,
            status="filled",
            filled_size=size,
            average_fill_price=fill_price
        )
        
        # Update position
        self._update_position(symbol, side, size, fill_price)
        
        # Record trade
        self.trades.append({
            "order_id": order_id,
            "symbol": symbol,
            "side": side.value,
            "size": size,
            "price": fill_price,
            "timestamp": datetime.utcnow().isoformat(),
            "pnl": 0  # PnL calculated on close
        })
        
        logger.info(f"[PAPER] Order filled: {side.value} {size} {symbol} @ {fill_price}")
        
        return order
    
    def _update_position(self, symbol: str, side: OrderSide, size: float, price: float):
        """Update position after trade."""
        if symbol not in self.positions:
            # New position
            pos_size = size if side == OrderSide.BUY else -size
            self.positions[symbol] = Position(
                symbol=symbol,
                size=pos_size,
                entry_price=price,
                mark_price=price,
                liquidation_price=0,
                unrealized_pnl=0,
                realized_pnl=0,
                leverage=1
            )
        else:
            pos = self.positions[symbol]
            old_size = pos.size
            
            if side == OrderSide.BUY:
                new_size = old_size + size
            else:
                new_size = old_size - size
            
            # Calculate PnL if reducing/closing
            if (old_size > 0 and new_size < old_size) or (old_size < 0 and new_size > old_size):
                pnl = abs(min(abs(old_size), abs(new_size))) * (price - pos.entry_price)
                if old_size < 0:
                    pnl = -pnl
                pos.realized_pnl += pnl
                self.balance += pnl
            
            if new_size == 0:
                del self.positions[symbol]
            else:
                # Update entry price (average)
                if abs(new_size) > abs(old_size):
                    pos.entry_price = (pos.entry_price * abs(old_size) + price * size) / abs(new_size)
                pos.size = new_size
    
    def _margin_used(self) -> float:
        """Calculate total margin used."""
        margin = 0
        for pos in self.positions.values():
            margin += abs(pos.size) * pos.entry_price / pos.leverage
        return margin
    
    def _total_unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL."""
        pnl = 0
        for pos in self.positions.values():
            pnl += pos.unrealized_pnl
        return pnl
    
    def get_trade_history(self) -> List[Dict]:
        """Get paper trade history."""
        return self.trades
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        total_pnl = self.balance - self.initial_balance
        total_trades = len(self.trades)
        
        winning_trades = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.balance,
            "total_pnl": total_pnl,
            "pnl_pct": total_pnl / self.initial_balance * 100,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "open_positions": len(self.positions)
        }


def create_client(paper_trading: bool = True, **kwargs) -> DeltaExchangeClient:
    """
    Factory function to create appropriate client.
    
    Args:
        paper_trading: If True, creates paper trading client
        **kwargs: Additional configuration
    """
    if paper_trading:
        initial_balance = kwargs.get("initial_balance", 100000.0)
        return PaperTradingClient(initial_balance=initial_balance)
    else:
        config = DeltaConfig(**kwargs)
        return DeltaExchangeClient(config)


# =============================================================================
# Demo
# =============================================================================

async def demo():
    """Demonstrate Delta Exchange client."""
    print("=" * 60)
    print("DELTA EXCHANGE CLIENT DEMO")
    print("=" * 60)
    
    # Create paper trading client
    client = create_client(paper_trading=True, initial_balance=100000)
    
    print("\n1. Getting products...")
    try:
        products = client.get_products()
        print(f"   Found {len(products)} products")
        
        # Find BTCUSD
        btc_products = [p for p in products if "BTC" in p.get("symbol", "")]
        if btc_products:
            print(f"   BTC products: {[p['symbol'] for p in btc_products[:5]]}")
    except Exception as e:
        print(f"   Error getting products: {e}")
    
    print("\n2. Getting ticker...")
    try:
        ticker = client.get_ticker("BTCUSD")
        print(f"   BTCUSD price: ${ticker.get('mark_price', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Paper trading demo...")
    try:
        # Place buy order
        order = client.place_order(
            symbol="BTCUSD",
            side=OrderSide.BUY,
            size=0.1,
            order_type=OrderType.MARKET
        )
        print(f"   Buy order: {order.id} - {order.status}")
        
        # Check position
        positions = client.get_positions()
        if positions:
            pos = positions[0]
            print(f"   Position: {pos.size} {pos.symbol} @ {pos.entry_price}")
        
        # Check balance
        balance = client.get_wallet_balance()
        print(f"   Balance: ${balance.get('balance', 0):,.2f}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Performance summary...")
    if hasattr(client, "get_performance_summary"):
        summary = client.get_performance_summary()
        for k, v in summary.items():
            print(f"   {k}: {v}")
    
    print("\n✅ Demo complete")


if __name__ == "__main__":
    asyncio.run(demo())
