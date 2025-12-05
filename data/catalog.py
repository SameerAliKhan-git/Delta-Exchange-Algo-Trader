"""
Product Catalog - Discover and filter tradable instruments

Provides:
- Futures catalog with liquidity filtering
- Options catalog with expiry/strike filtering
- Dynamic instrument selection
"""

import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class ProductType(Enum):
    """Types of tradable products"""
    PERPETUAL = "perpetual"
    FUTURES = "futures"
    CALL_OPTION = "call_option"
    PUT_OPTION = "put_option"
    SPOT = "spot"


@dataclass
class ProductInfo:
    """Information about a tradable product"""
    product_id: int
    symbol: str
    underlying: str
    product_type: ProductType
    contract_size: float
    tick_size: float
    min_size: float
    
    # Futures/Options specific
    expiry: Optional[datetime] = None
    strike: Optional[float] = None
    
    # Liquidity metrics (populated by catalog)
    volume_24h: float = 0.0
    open_interest: float = 0.0
    bid_ask_spread: float = 0.0
    
    # Metadata
    is_active: bool = True
    trading_status: str = "trading"
    
    @property
    def is_perpetual(self) -> bool:
        return self.product_type == ProductType.PERPETUAL
    
    @property
    def is_option(self) -> bool:
        return self.product_type in (ProductType.CALL_OPTION, ProductType.PUT_OPTION)
    
    @property
    def is_call(self) -> bool:
        return self.product_type == ProductType.CALL_OPTION
    
    @property
    def is_put(self) -> bool:
        return self.product_type == ProductType.PUT_OPTION
    
    @property
    def days_to_expiry(self) -> Optional[int]:
        if self.expiry is None:
            return None
        delta = self.expiry - datetime.utcnow()
        return max(0, delta.days)


@dataclass
class ProductCatalog:
    """
    Catalog of tradable products with filtering capabilities
    """
    products: Dict[str, ProductInfo] = field(default_factory=dict)
    underlyings: Set[str] = field(default_factory=set)
    last_updated: float = 0.0
    
    def add_product(self, product: ProductInfo) -> None:
        """Add product to catalog"""
        self.products[product.symbol] = product
        self.underlyings.add(product.underlying)
    
    def get_product(self, symbol: str) -> Optional[ProductInfo]:
        """Get product by symbol"""
        return self.products.get(symbol)
    
    def get_perpetuals(self) -> List[ProductInfo]:
        """Get all perpetual futures"""
        return [
            p for p in self.products.values()
            if p.is_perpetual and p.is_active
        ]
    
    def get_futures(self, underlying: str = None) -> List[ProductInfo]:
        """Get all futures (perpetual + dated)"""
        products = [
            p for p in self.products.values()
            if p.product_type in (ProductType.PERPETUAL, ProductType.FUTURES)
            and p.is_active
        ]
        
        if underlying:
            products = [p for p in products if p.underlying == underlying]
        
        return products
    
    def get_options(
        self,
        underlying: str = None,
        option_type: str = None,
        min_days: int = None,
        max_days: int = None,
        min_strike: float = None,
        max_strike: float = None
    ) -> List[ProductInfo]:
        """
        Get options with filters
        
        Args:
            underlying: Filter by underlying asset
            option_type: 'call' or 'put'
            min_days: Minimum days to expiry
            max_days: Maximum days to expiry
            min_strike: Minimum strike price
            max_strike: Maximum strike price
        
        Returns:
            List of matching options
        """
        options = [p for p in self.products.values() if p.is_option and p.is_active]
        
        if underlying:
            options = [p for p in options if p.underlying == underlying]
        
        if option_type == 'call':
            options = [p for p in options if p.is_call]
        elif option_type == 'put':
            options = [p for p in options if p.is_put]
        
        if min_days is not None:
            options = [p for p in options if p.days_to_expiry and p.days_to_expiry >= min_days]
        
        if max_days is not None:
            options = [p for p in options if p.days_to_expiry and p.days_to_expiry <= max_days]
        
        if min_strike is not None:
            options = [p for p in options if p.strike and p.strike >= min_strike]
        
        if max_strike is not None:
            options = [p for p in options if p.strike and p.strike <= max_strike]
        
        return options
    
    def filter_by_liquidity(
        self,
        products: List[ProductInfo],
        min_volume: float = 0,
        min_oi: float = 0,
        max_spread_pct: float = 1.0
    ) -> List[ProductInfo]:
        """
        Filter products by liquidity metrics
        
        Args:
            products: Products to filter
            min_volume: Minimum 24h volume
            min_oi: Minimum open interest
            max_spread_pct: Maximum bid-ask spread percentage
        
        Returns:
            Filtered products
        """
        result = []
        
        for p in products:
            if p.volume_24h < min_volume:
                continue
            if p.open_interest < min_oi:
                continue
            if p.bid_ask_spread > max_spread_pct:
                continue
            result.append(p)
        
        return result
    
    def get_most_liquid(
        self,
        underlying: str,
        product_type: ProductType = ProductType.PERPETUAL,
        count: int = 5
    ) -> List[ProductInfo]:
        """
        Get most liquid products by volume
        
        Args:
            underlying: Underlying asset
            product_type: Type of product
            count: Number of results
        
        Returns:
            Top products by volume
        """
        products = [
            p for p in self.products.values()
            if p.underlying == underlying
            and p.product_type == product_type
            and p.is_active
        ]
        
        products.sort(key=lambda x: x.volume_24h, reverse=True)
        
        return products[:count]


def build_catalog_from_api(delta_client) -> ProductCatalog:
    """
    Build product catalog from Delta Exchange API
    
    Args:
        delta_client: Delta Exchange client
    
    Returns:
        Populated ProductCatalog
    """
    catalog = ProductCatalog()
    
    try:
        # Get all products
        response = delta_client.get_products()
        
        if not response or 'result' not in response:
            return catalog
        
        for product in response['result']:
            product_type = _parse_product_type(product)
            
            info = ProductInfo(
                product_id=product.get('id', 0),
                symbol=product.get('symbol', ''),
                underlying=product.get('underlying_asset', {}).get('symbol', ''),
                product_type=product_type,
                contract_size=float(product.get('contract_value', 1)),
                tick_size=float(product.get('tick_size', 0.01)),
                min_size=float(product.get('min_size', 0.001)),
                expiry=_parse_expiry(product.get('settlement_time')),
                strike=float(product.get('strike_price', 0)) if product.get('strike_price') else None,
                is_active=product.get('trading_status') == 'trading',
                trading_status=product.get('trading_status', 'unknown')
            )
            
            catalog.add_product(info)
        
        catalog.last_updated = time.time()
        
    except Exception as e:
        print(f"Error building catalog: {e}")
    
    return catalog


def _parse_product_type(product: Dict) -> ProductType:
    """Parse product type from API response"""
    contract_type = product.get('contract_type', '')
    
    if contract_type == 'perpetual_futures':
        return ProductType.PERPETUAL
    elif contract_type == 'futures':
        return ProductType.FUTURES
    elif contract_type == 'call_options':
        return ProductType.CALL_OPTION
    elif contract_type == 'put_options':
        return ProductType.PUT_OPTION
    elif contract_type == 'spot':
        return ProductType.SPOT
    else:
        return ProductType.PERPETUAL


def _parse_expiry(settlement_time: str) -> Optional[datetime]:
    """Parse expiry datetime from settlement time"""
    if not settlement_time:
        return None
    
    try:
        return datetime.fromisoformat(settlement_time.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None


def get_futures_catalog(delta_client, underlying: str = None) -> List[ProductInfo]:
    """
    Convenience function to get futures catalog
    
    Args:
        delta_client: Delta Exchange client
        underlying: Optional underlying filter
    
    Returns:
        List of futures products
    """
    catalog = build_catalog_from_api(delta_client)
    return catalog.get_futures(underlying)


def get_options_catalog(
    delta_client,
    underlying: str,
    option_type: str = None,
    min_days: int = 3,
    max_days: int = 30
) -> List[ProductInfo]:
    """
    Convenience function to get options catalog
    
    Args:
        delta_client: Delta Exchange client
        underlying: Underlying asset
        option_type: 'call' or 'put'
        min_days: Minimum days to expiry
        max_days: Maximum days to expiry
    
    Returns:
        List of options products
    """
    catalog = build_catalog_from_api(delta_client)
    return catalog.get_options(
        underlying=underlying,
        option_type=option_type,
        min_days=min_days,
        max_days=max_days
    )


def filter_by_liquidity(
    products: List[ProductInfo],
    min_volume: float = 100000,
    min_oi: float = 10000,
    max_spread_pct: float = 0.5
) -> List[ProductInfo]:
    """
    Filter products by liquidity
    
    Args:
        products: Products to filter
        min_volume: Minimum 24h volume in USD
        min_oi: Minimum open interest
        max_spread_pct: Maximum spread percentage
    
    Returns:
        Filtered list
    """
    catalog = ProductCatalog()
    return catalog.filter_by_liquidity(products, min_volume, min_oi, max_spread_pct)
