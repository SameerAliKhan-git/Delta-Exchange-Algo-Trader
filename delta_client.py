"""
Delta Exchange REST Client Wrapper
Provides a robust interface to Delta Exchange India API
with error handling, rate limiting, and logging
"""

import time
import hashlib
import hmac
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from functools import wraps

import requests
from ratelimit import limits, sleep_and_retry

from logger import get_logger, get_audit_logger
from config import get_config


class DeltaClientError(Exception):
    """Base exception for Delta client errors"""
    pass


class DeltaAuthError(DeltaClientError):
    """Authentication error"""
    pass


class DeltaRateLimitError(DeltaClientError):
    """Rate limit exceeded"""
    pass


class DeltaOrderError(DeltaClientError):
    """Order-related error"""
    pass


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except DeltaRateLimitError:
                    # Don't retry rate limits immediately
                    time.sleep(delay * (attempt + 1) * 2)
                    last_exception = DeltaRateLimitError("Rate limit exceeded")
                except requests.exceptions.RequestException as e:
                    time.sleep(delay * (attempt + 1))
                    last_exception = e
                except DeltaClientError as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
                    last_exception = e
            raise last_exception or DeltaClientError("Max retries exceeded")
        return wrapper
    return decorator


class DeltaExchangeClient:
    """
    Production-ready wrapper for Delta Exchange India REST API
    
    Features:
    - Automatic signature generation
    - Rate limiting
    - Retry logic with exponential backoff
    - Comprehensive logging
    - Testnet/production switching
    """
    
    # Rate limits: 10 requests per second (conservative)
    RATE_LIMIT_CALLS = 10
    RATE_LIMIT_PERIOD = 1
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        config = get_config()
        
        self.api_key = api_key or config.delta.api_key
        self.api_secret = api_secret or config.delta.api_secret
        self.base_url = (base_url or config.delta.base_url).rstrip('/')
        self.timeout = timeout
        
        self.logger = get_logger()
        self.audit = get_audit_logger()
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        self._is_testnet = 'testnet' in self.base_url.lower()
        
        self.logger.info(
            "Delta client initialized",
            base_url=self.base_url,
            is_testnet=self._is_testnet
        )
    
    @property
    def is_testnet(self) -> bool:
        return self._is_testnet
    
    def _generate_signature(
        self,
        method: str,
        endpoint: str,
        timestamp: str,
        payload: str = ""
    ) -> str:
        """Generate HMAC signature for authenticated requests"""
        signature_data = f"{method}{timestamp}{endpoint}{payload}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_auth_headers(
        self,
        method: str,
        endpoint: str,
        payload: str = ""
    ) -> Dict[str, str]:
        """Generate authentication headers"""
        timestamp = str(int(time.time()))
        signature = self._generate_signature(method, endpoint, timestamp, payload)
        
        return {
            'api-key': self.api_key,
            'timestamp': timestamp,
            'signature': signature
        }
    
    @sleep_and_retry
    @limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        auth_required: bool = False
    ) -> Dict[str, Any]:
        """Make an API request with rate limiting"""
        url = f"{self.base_url}{endpoint}"
        payload = json.dumps(data) if data else ""
        
        headers = {}
        if auth_required:
            headers = self._get_auth_headers(method, endpoint, payload)
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=payload if data else None,
                headers=headers,
                timeout=self.timeout
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                self.logger.warning("Rate limit hit", endpoint=endpoint)
                raise DeltaRateLimitError("Rate limit exceeded")
            
            # Handle auth errors
            if response.status_code == 401:
                self.logger.error("Authentication failed", endpoint=endpoint)
                raise DeltaAuthError("Invalid API credentials")
            
            # Handle other errors
            if response.status_code >= 400:
                error_msg = response.text
                self.logger.error(
                    "API error",
                    status_code=response.status_code,
                    endpoint=endpoint,
                    error=error_msg
                )
                raise DeltaClientError(f"API error {response.status_code}: {error_msg}")
            
            return response.json()
            
        except requests.exceptions.Timeout:
            self.logger.error("Request timeout", endpoint=endpoint)
            raise DeltaClientError(f"Request timeout: {endpoint}")
        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed", endpoint=endpoint, error=str(e))
            raise
    
    # ==================== PUBLIC ENDPOINTS ====================
    
    @retry_on_error()
    def get_products(self) -> List[Dict[str, Any]]:
        """Get all available products"""
        response = self._request("GET", "/v2/products")
        return response.get('result', [])
    
    @retry_on_error()
    def get_product(self, product_id: int) -> Dict[str, Any]:
        """Get a specific product by ID"""
        response = self._request("GET", f"/v2/products/{product_id}")
        return response.get('result', {})
    
    @retry_on_error()
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker for a symbol"""
        response = self._request("GET", f"/v2/tickers/{symbol}")
        return response.get('result', {})
    
    @retry_on_error()
    def get_tickers(self) -> List[Dict[str, Any]]:
        """Get all tickers"""
        response = self._request("GET", "/v2/tickers")
        return response.get('result', [])
    
    @retry_on_error()
    def get_orderbook(self, product_id: int, depth: int = 20) -> Dict[str, Any]:
        """Get L2 orderbook for a product"""
        response = self._request(
            "GET",
            f"/v2/l2orderbook/{product_id}",
            params={'depth': depth}
        )
        return response.get('result', {})
    
    @retry_on_error()
    def get_recent_trades(self, product_id: int) -> List[Dict[str, Any]]:
        """Get recent trades for a product"""
        response = self._request("GET", f"/v2/trades/{product_id}")
        return response.get('result', [])
    
    @retry_on_error()
    def get_mark_price(self, product_id: int) -> Dict[str, Any]:
        """Get mark price for a product"""
        response = self._request("GET", f"/v2/products/{product_id}/mark_price")
        return response.get('result', {})
    
    # ==================== AUTHENTICATED ENDPOINTS ====================
    
    @retry_on_error()
    def get_wallet_balances(self) -> List[Dict[str, Any]]:
        """Get wallet balances"""
        response = self._request("GET", "/v2/wallet/balances", auth_required=True)
        return response.get('result', [])
    
    @retry_on_error()
    def get_positions(self, product_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get open positions"""
        params = {'product_id': product_id} if product_id else None
        response = self._request("GET", "/v2/positions", params=params, auth_required=True)
        return response.get('result', [])
    
    @retry_on_error()
    def get_position(self, product_id: int) -> Dict[str, Any]:
        """Get position for a specific product"""
        response = self._request("GET", f"/v2/positions/{product_id}", auth_required=True)
        return response.get('result', {})
    
    @retry_on_error()
    def get_open_orders(self, product_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        params = {'product_id': product_id} if product_id else None
        response = self._request("GET", "/v2/orders", params=params, auth_required=True)
        return response.get('result', [])
    
    @retry_on_error()
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get a specific order"""
        response = self._request("GET", f"/v2/orders/{order_id}", auth_required=True)
        return response.get('result', {})
    
    @retry_on_error()
    def get_order_history(
        self,
        product_id: Optional[int] = None,
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history"""
        params = {'page_size': page_size}
        if product_id:
            params['product_id'] = product_id
        response = self._request("GET", "/v2/orders/history", params=params, auth_required=True)
        return response.get('result', [])
    
    @retry_on_error()
    def get_fills(
        self,
        product_id: Optional[int] = None,
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Get trade fills"""
        params = {'page_size': page_size}
        if product_id:
            params['product_id'] = product_id
        response = self._request("GET", "/v2/fills", params=params, auth_required=True)
        return response.get('result', [])
    
    # ==================== ORDER MANAGEMENT ====================
    
    def place_order(
        self,
        product_id: int,
        size: float,
        side: str,
        order_type: str = "market_order",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "gtc",
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place an order
        
        Args:
            product_id: Product ID to trade
            size: Order size
            side: 'buy' or 'sell'
            order_type: 'market_order', 'limit_order', 'stop_market_order', 'stop_limit_order'
            limit_price: Limit price (required for limit orders)
            stop_price: Stop/trigger price (required for stop orders)
            time_in_force: 'gtc', 'ioc', 'fok'
            post_only: Post-only order flag
            reduce_only: Reduce-only order flag
            client_order_id: Custom order ID for idempotency
        
        Returns:
            Order response from exchange
        """
        order_data = {
            'product_id': product_id,
            'size': size,
            'side': side,
            'order_type': order_type,
            'time_in_force': time_in_force,
            'post_only': post_only,
            'reduce_only': reduce_only
        }
        
        if limit_price is not None:
            order_data['limit_price'] = str(limit_price)
        
        if stop_price is not None:
            order_data['stop_price'] = str(stop_price)
        
        if client_order_id:
            order_data['client_order_id'] = client_order_id
        
        self.logger.log_order(
            action="placing",
            product_symbol=str(product_id),
            side=side,
            size=size,
            price=limit_price,
            order_type=order_type
        )
        
        # Audit log before placing
        self.audit.log_order_placed({
            **order_data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        try:
            response = self._request("POST", "/v2/orders", data=order_data, auth_required=True)
            result = response.get('result', {})
            
            self.logger.log_order(
                action="placed",
                order_id=result.get('id'),
                product_symbol=str(product_id),
                side=side,
                size=size,
                price=limit_price,
                order_type=order_type,
                status=result.get('state')
            )
            
            return result
            
        except DeltaClientError as e:
            self.logger.error(
                "Order placement failed",
                product_id=product_id,
                side=side,
                size=size,
                error=str(e)
            )
            raise DeltaOrderError(f"Order placement failed: {e}")
    
    def place_market_order(
        self,
        product_id: int,
        size: float,
        side: str,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Place a market order"""
        return self.place_order(
            product_id=product_id,
            size=size,
            side=side,
            order_type="market_order",
            reduce_only=reduce_only,
            client_order_id=client_order_id
        )
    
    def place_limit_order(
        self,
        product_id: int,
        size: float,
        side: str,
        limit_price: float,
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Place a limit order"""
        return self.place_order(
            product_id=product_id,
            size=size,
            side=side,
            order_type="limit_order",
            limit_price=limit_price,
            post_only=post_only,
            reduce_only=reduce_only,
            client_order_id=client_order_id
        )
    
    def place_stop_market_order(
        self,
        product_id: int,
        size: float,
        side: str,
        stop_price: float,
        reduce_only: bool = True,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Place a stop market order"""
        return self.place_order(
            product_id=product_id,
            size=size,
            side=side,
            order_type="stop_market_order",
            stop_price=stop_price,
            reduce_only=reduce_only,
            client_order_id=client_order_id
        )
    
    def place_stop_limit_order(
        self,
        product_id: int,
        size: float,
        side: str,
        stop_price: float,
        limit_price: float,
        reduce_only: bool = True,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Place a stop limit order"""
        return self.place_order(
            product_id=product_id,
            size=size,
            side=side,
            order_type="stop_limit_order",
            stop_price=stop_price,
            limit_price=limit_price,
            reduce_only=reduce_only,
            client_order_id=client_order_id
        )
    
    @retry_on_error()
    def cancel_order(self, order_id: str, product_id: int) -> Dict[str, Any]:
        """Cancel an order"""
        self.logger.log_order(action="cancelling", order_id=order_id)
        
        response = self._request(
            "DELETE",
            f"/v2/orders/{order_id}",
            data={'product_id': product_id},
            auth_required=True
        )
        result = response.get('result', {})
        
        self.audit.log_order_cancelled({
            'order_id': order_id,
            'product_id': product_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        self.logger.log_order(action="cancelled", order_id=order_id, status="cancelled")
        return result
    
    @retry_on_error()
    def cancel_all_orders(self, product_id: Optional[int] = None) -> Dict[str, Any]:
        """Cancel all open orders"""
        data = {'product_id': product_id} if product_id else {}
        
        self.logger.info("Cancelling all orders", product_id=product_id)
        
        response = self._request(
            "DELETE",
            "/v2/orders/all",
            data=data if data else None,
            auth_required=True
        )
        
        return response.get('result', {})
    
    # ==================== POSITION MANAGEMENT ====================
    
    @retry_on_error()
    def close_position(self, product_id: int) -> Dict[str, Any]:
        """Close a position by placing a reducing market order"""
        position = self.get_position(product_id)
        
        if not position or position.get('size', 0) == 0:
            self.logger.info("No position to close", product_id=product_id)
            return {}
        
        size = abs(float(position.get('size', 0)))
        side = 'sell' if float(position.get('size', 0)) > 0 else 'buy'
        
        self.logger.info(
            "Closing position",
            product_id=product_id,
            size=size,
            side=side
        )
        
        return self.place_market_order(
            product_id=product_id,
            size=size,
            side=side,
            reduce_only=True
        )
    
    @retry_on_error()
    def set_leverage(self, product_id: int, leverage: int) -> Dict[str, Any]:
        """Set leverage for a product"""
        response = self._request(
            "POST",
            "/v2/orders/leverage",
            data={'product_id': product_id, 'leverage': leverage},
            auth_required=True
        )
        
        self.logger.info("Leverage set", product_id=product_id, leverage=leverage)
        return response.get('result', {})
    
    # ==================== UTILITY METHODS ====================
    
    def get_product_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Find a product by its symbol"""
        products = self.get_products()
        for product in products:
            if product.get('symbol') == symbol:
                return product
        return None
    
    def get_current_price(self, product_id: int) -> float:
        """Get current mark price for a product"""
        try:
            mark_price = self.get_mark_price(product_id)
            return float(mark_price.get('mark_price', 0))
        except Exception:
            # Fallback to ticker
            products = self.get_products()
            for p in products:
                if p.get('id') == product_id:
                    ticker = self.get_ticker(p.get('symbol'))
                    return float(ticker.get('close', 0))
            return 0.0
    
    def health_check(self) -> bool:
        """Check if the API is accessible"""
        try:
            self.get_products()
            return True
        except Exception:
            return False


# Singleton instance
_client: Optional[DeltaExchangeClient] = None


def get_delta_client() -> DeltaExchangeClient:
    """Get or create the global Delta client"""
    global _client
    if _client is None:
        _client = DeltaExchangeClient()
    return _client


if __name__ == "__main__":
    # Test the client
    client = get_delta_client()
    
    print(f"Is testnet: {client.is_testnet}")
    print(f"Health check: {client.health_check()}")
    
    # Get products
    products = client.get_products()
    print(f"Found {len(products)} products")
    
    if products:
        product = products[0]
        print(f"First product: {product.get('symbol')} (ID: {product.get('id')})")
