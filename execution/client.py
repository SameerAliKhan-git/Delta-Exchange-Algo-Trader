"""
Delta Exchange Client - API wrapper for Delta Exchange India

Provides:
- REST API wrapper with authentication
- Rate limiting
- Error handling and retries
"""

import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    LIMIT = "limit_order"
    MARKET = "market_order"
    STOP_LIMIT = "stop_limit_order"
    STOP_MARKET = "stop_market_order"


class TimeInForce(Enum):
    """Time in force"""
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class APIConfig:
    """API configuration"""
    api_key: str
    api_secret: str
    testnet: bool = True
    timeout: int = 30
    max_retries: int = 3
    
    @property
    def base_url(self) -> str:
        if self.testnet:
            return "https://cdn-ind.testnet.deltaex.org"
        return "https://api.india.delta.exchange"


class DeltaClient:
    """
    Delta Exchange India API Client
    
    Provides authenticated access to Delta Exchange REST API
    """
    
    def __init__(self, config: APIConfig = None, **kwargs):
        """
        Initialize client
        
        Args:
            config: APIConfig object
            **kwargs: Alternative config (api_key, api_secret, testnet)
        """
        if config:
            self.config = config
        else:
            self.config = APIConfig(
                api_key=kwargs.get('api_key', ''),
                api_secret=kwargs.get('api_secret', ''),
                testnet=kwargs.get('testnet', True)
            )
        
        self.session = requests.Session() if requests else None
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0
    
    # ==================== Public Endpoints ====================
    
    def get_products(self) -> Dict:
        """Get all tradable products"""
        return self._get("/v2/products")
    
    def get_product(self, product_id: int) -> Dict:
        """Get single product by ID"""
        return self._get(f"/v2/products/{product_id}")
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker for symbol"""
        return self._get("/v2/tickers", params={"symbol": symbol})
    
    def get_tickers(self) -> Dict:
        """Get all tickers"""
        return self._get("/v2/tickers")
    
    def get_orderbook(self, product_id: int, depth: int = 20) -> Dict:
        """Get order book"""
        return self._get(f"/v2/l2orderbook/{product_id}", params={"depth": depth})
    
    def get_candles(
        self,
        symbol: str,
        resolution: int,
        start: int = None,
        end: int = None
    ) -> Dict:
        """
        Get OHLCV candles
        
        Args:
            symbol: Product symbol
            resolution: Resolution in seconds
            start: Start timestamp
            end: End timestamp
        """
        params = {
            "symbol": symbol,
            "resolution": resolution
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        
        return self._get("/v2/history/candles", params=params)
    
    def get_recent_trades(self, product_id: int) -> Dict:
        """Get recent trades"""
        return self._get(f"/v2/trades/{product_id}")
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """Get funding rate for perpetual"""
        return self._get("/v2/funding_rate", params={"symbol": symbol})
    
    # ==================== Private Endpoints ====================
    
    def get_wallet_balance(self) -> Dict:
        """Get wallet balances"""
        return self._get("/v2/wallet/balances", auth=True)
    
    def get_positions(self) -> Dict:
        """Get all open positions"""
        return self._get("/v2/positions", auth=True)
    
    def get_position(self, product_id: int) -> Dict:
        """Get position for product"""
        return self._get(f"/v2/positions/{product_id}", auth=True)
    
    def get_orders(self, product_id: int = None, state: str = None) -> Dict:
        """
        Get orders
        
        Args:
            product_id: Filter by product
            state: Filter by state ('open', 'pending', 'closed')
        """
        params = {}
        if product_id:
            params["product_id"] = product_id
        if state:
            params["state"] = state
        
        return self._get("/v2/orders", params=params, auth=True)
    
    def get_order(self, order_id: int) -> Dict:
        """Get order by ID"""
        return self._get(f"/v2/orders/{order_id}", auth=True)
    
    def place_order(
        self,
        product_id: int,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float = None,
        stop_price: float = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        post_only: bool = False,
        client_order_id: str = None
    ) -> Dict:
        """
        Place order
        
        Args:
            product_id: Product ID
            side: Buy or sell
            size: Order size
            order_type: Order type
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            reduce_only: Only reduce position
            post_only: Post only (maker)
            client_order_id: Client order ID
        
        Returns:
            Order result
        """
        payload = {
            "product_id": product_id,
            "side": side.value if isinstance(side, OrderSide) else side,
            "size": size,
            "order_type": order_type.value if isinstance(order_type, OrderType) else order_type,
            "time_in_force": time_in_force.value if isinstance(time_in_force, TimeInForce) else time_in_force,
            "reduce_only": reduce_only,
            "post_only": post_only
        }
        
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)
        
        if client_order_id:
            payload["client_order_id"] = client_order_id
        
        return self._post("/v2/orders", payload=payload, auth=True)
    
    def cancel_order(self, order_id: int, product_id: int) -> Dict:
        """Cancel order"""
        return self._delete(
            f"/v2/orders/{order_id}",
            params={"product_id": product_id},
            auth=True
        )
    
    def cancel_all_orders(self, product_id: int = None) -> Dict:
        """Cancel all orders"""
        params = {}
        if product_id:
            params["product_id"] = product_id
        
        return self._delete("/v2/orders/all", params=params, auth=True)
    
    def modify_order(
        self,
        order_id: int,
        product_id: int,
        size: float = None,
        limit_price: float = None
    ) -> Dict:
        """Modify existing order"""
        payload = {"product_id": product_id}
        
        if size is not None:
            payload["size"] = size
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        
        return self._put(f"/v2/orders/{order_id}", payload=payload, auth=True)
    
    def close_position(self, product_id: int) -> Dict:
        """Close position (market order)"""
        return self._post(
            "/v2/positions/close",
            payload={"product_id": product_id},
            auth=True
        )
    
    def set_leverage(self, product_id: int, leverage: int) -> Dict:
        """Set leverage for product"""
        return self._post(
            "/v2/orders/leverage",
            payload={"product_id": product_id, "leverage": leverage},
            auth=True
        )
    
    # ==================== Options Specific ====================
    
    def get_option_chain(self, underlying: str) -> Dict:
        """Get options chain for underlying"""
        return self._get("/v2/option_chain", params={"underlying_asset": underlying})
    
    # ==================== Internal Methods ====================
    
    def _get(self, path: str, params: Dict = None, auth: bool = False) -> Dict:
        """GET request"""
        return self._request("GET", path, params=params, auth=auth)
    
    def _post(self, path: str, payload: Dict = None, auth: bool = False) -> Dict:
        """POST request"""
        return self._request("POST", path, payload=payload, auth=auth)
    
    def _put(self, path: str, payload: Dict = None, auth: bool = False) -> Dict:
        """PUT request"""
        return self._request("PUT", path, payload=payload, auth=auth)
    
    def _delete(self, path: str, params: Dict = None, auth: bool = False) -> Dict:
        """DELETE request"""
        return self._request("DELETE", path, params=params, auth=auth)
    
    def _request(
        self,
        method: str,
        path: str,
        params: Dict = None,
        payload: Dict = None,
        auth: bool = False
    ) -> Dict:
        """Make HTTP request"""
        if self.session is None:
            raise ImportError("requests library not installed")
        
        url = f"{self.config.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        
        if auth:
            headers.update(self._get_auth_headers(method, path, payload))
        
        # Rate limit check
        self._check_rate_limit()
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout
                )
                
                # Update rate limit info
                self._update_rate_limit(response.headers)
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = int(response.headers.get('Retry-After', 5))
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(1)
        
        return {"error": "Max retries exceeded"}
    
    def _get_auth_headers(self, method: str, path: str, payload: Dict = None) -> Dict:
        """Generate authentication headers"""
        timestamp = str(int(time.time()))
        
        # Build signature payload
        message = method + timestamp + path
        if payload:
            message += json.dumps(payload, separators=(',', ':'))
        
        # Sign with HMAC-SHA256
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "api-key": self.config.api_key,
            "timestamp": timestamp,
            "signature": signature
        }
    
    def _check_rate_limit(self) -> None:
        """Check and wait for rate limit"""
        if self._rate_limit_remaining <= 0:
            wait_time = max(0, self._rate_limit_reset - time.time())
            if wait_time > 0:
                time.sleep(wait_time)
    
    def _update_rate_limit(self, headers: Dict) -> None:
        """Update rate limit from response headers"""
        if 'X-RateLimit-Remaining' in headers:
            self._rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in headers:
            self._rate_limit_reset = int(headers['X-RateLimit-Reset'])


def create_client(
    api_key: str,
    api_secret: str,
    testnet: bool = True
) -> DeltaClient:
    """
    Create Delta Exchange client
    
    Args:
        api_key: API key
        api_secret: API secret
        testnet: Use testnet (default True)
    
    Returns:
        DeltaClient instance
    """
    config = APIConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )
    return DeltaClient(config)
