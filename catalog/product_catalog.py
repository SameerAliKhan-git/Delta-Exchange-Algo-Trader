"""
ALADDIN - Product Catalog
===========================
Auto-discovers and classifies all tradeable instruments from Delta Exchange.
"""

import requests
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.credentials import API_KEY, API_SECRET, BASE_URL


class ProductType(Enum):
    """Classification of product types."""
    PERPETUAL = auto()
    FUTURE = auto()
    CALL_OPTION = auto()
    PUT_OPTION = auto()
    SPOT = auto()
    UNKNOWN = auto()


@dataclass
class Product:
    """
    Represents a tradeable product on Delta Exchange.
    """
    # Core identifiers
    id: int
    symbol: str
    description: str
    product_type: ProductType
    
    # Underlying asset info
    underlying_asset: str
    quote_asset: str
    settling_asset: str
    
    # Contract specs
    contract_size: float
    tick_size: float
    min_size: float
    lot_size: float
    
    # State
    is_active: bool
    trading_status: str
    
    # For futures/options
    expiry_date: Optional[datetime] = None
    strike_price: Optional[float] = None
    
    # Liquidity metrics (populated by LiquidityFilter)
    volume_24h: float = 0.0
    open_interest: float = 0.0
    bid_ask_spread: float = 0.0
    
    # Greeks for options
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    raw_data: Dict = field(default_factory=dict)
    
    @property
    def is_perpetual(self) -> bool:
        return self.product_type == ProductType.PERPETUAL
    
    @property
    def is_future(self) -> bool:
        return self.product_type == ProductType.FUTURE
    
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
        if self.expiry_date:
            # Handle timezone-aware datetimes
            now = datetime.now()
            expiry = self.expiry_date
            if expiry.tzinfo is not None:
                expiry = expiry.replace(tzinfo=None)
            return (expiry - now).days
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'product_type': self.product_type.name,
            'underlying_asset': self.underlying_asset,
            'contract_size': self.contract_size,
            'tick_size': self.tick_size,
            'volume_24h': self.volume_24h,
            'open_interest': self.open_interest,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'strike_price': self.strike_price,
            'delta': self.delta,
            'iv': self.iv
        }


class ProductCatalog:
    """
    Discovers, classifies, and maintains all tradeable products.
    Polls Delta Exchange /products endpoint.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('Aladdin.Catalog')
        self.base_url = BASE_URL
        
        # Product storage
        self.products: Dict[str, Product] = {}
        self.perpetuals: Dict[str, Product] = {}
        self.futures: Dict[str, Product] = {}
        self.options: Dict[str, Product] = {}
        
        # Refresh tracking
        self.last_refresh: Optional[datetime] = None
        self.refresh_interval = timedelta(minutes=5)
    
    def refresh(self) -> bool:
        """
        Fetch and classify all products from Delta Exchange.
        Returns True if successful.
        """
        try:
            self.logger.info("Refreshing product catalog...")
            
            # Fetch all products
            response = requests.get(
                f"{self.base_url}/v2/products",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and 'result' in data:
                products = data['result']
                self._process_products(products)
                self.last_refresh = datetime.now()
                self.logger.info(f"Catalog refreshed: {len(self.products)} products "
                               f"({len(self.perpetuals)} perps, {len(self.futures)} futures, "
                               f"{len(self.options)} options)")
                return True
            else:
                self.logger.error(f"Failed to fetch products: {data}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error refreshing catalog: {e}")
            return False
    
    def _process_products(self, products: List[Dict]):
        """Process and classify raw product data."""
        self.products.clear()
        self.perpetuals.clear()
        self.futures.clear()
        self.options.clear()
        
        for p in products:
            try:
                product = self._create_product(p)
                if product and product.is_active:
                    self.products[product.symbol] = product
                    
                    if product.is_perpetual:
                        self.perpetuals[product.symbol] = product
                    elif product.is_future:
                        self.futures[product.symbol] = product
                    elif product.is_option:
                        self.options[product.symbol] = product
                        
            except Exception as e:
                self.logger.debug(f"Error processing product {p.get('symbol')}: {e}")
    
    def _create_product(self, data: Dict) -> Optional[Product]:
        """Create a Product object from raw API data."""
        try:
            # Determine product type
            product_type = self._classify_product(data)
            
            # Parse expiry date if present
            expiry_date = None
            if data.get('settlement_time'):
                try:
                    expiry_date = datetime.fromisoformat(
                        data['settlement_time'].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            # Parse strike price for options
            strike_price = None
            if data.get('strike_price'):
                try:
                    strike_price = float(data['strike_price'])
                except:
                    pass
            
            return Product(
                id=data.get('id', 0),
                symbol=data.get('symbol', ''),
                description=data.get('description', ''),
                product_type=product_type,
                underlying_asset=data.get('underlying_asset', {}).get('symbol', ''),
                quote_asset=data.get('quoting_asset', {}).get('symbol', 'USD'),
                settling_asset=data.get('settling_asset', {}).get('symbol', 'USD'),
                contract_size=float(data.get('contract_value', 1)),
                tick_size=float(data.get('tick_size', 0.01)),
                min_size=float(data.get('minimum_order_size', 1)),
                lot_size=float(data.get('position_size_limit', 1)),
                is_active=data.get('state', '').lower() == 'live',
                trading_status=data.get('state', 'unknown'),
                expiry_date=expiry_date,
                strike_price=strike_price,
                raw_data=data
            )
        except Exception as e:
            self.logger.debug(f"Error creating product: {e}")
            return None
    
    def _classify_product(self, data: Dict) -> ProductType:
        """Classify product type from raw data."""
        contract_type = data.get('contract_type', '').lower()
        symbol = data.get('symbol', '').upper()
        
        if contract_type == 'perpetual_futures':
            return ProductType.PERPETUAL
        elif contract_type == 'futures':
            return ProductType.FUTURE
        elif contract_type == 'call_options':
            return ProductType.CALL_OPTION
        elif contract_type == 'put_options':
            return ProductType.PUT_OPTION
        elif 'spot' in contract_type:
            return ProductType.SPOT
        else:
            # Infer from symbol
            if 'PERP' in symbol:
                return ProductType.PERPETUAL
            elif '-C-' in symbol:
                return ProductType.CALL_OPTION
            elif '-P-' in symbol:
                return ProductType.PUT_OPTION
            else:
                return ProductType.UNKNOWN
    
    def get_product(self, symbol: str) -> Optional[Product]:
        """Get a product by symbol."""
        return self.products.get(symbol)
    
    def get_perpetuals(self, underlying: Optional[str] = None) -> List[Product]:
        """Get all perpetual contracts, optionally filtered by underlying."""
        perps = list(self.perpetuals.values())
        if underlying:
            perps = [p for p in perps if p.underlying_asset.upper() == underlying.upper()]
        return perps
    
    def get_futures(self, underlying: Optional[str] = None, 
                   min_dte: int = 0, max_dte: int = 365) -> List[Product]:
        """Get futures contracts filtered by underlying and days to expiry."""
        futures = list(self.futures.values())
        
        if underlying:
            futures = [f for f in futures if f.underlying_asset.upper() == underlying.upper()]
        
        # Filter by DTE
        futures = [f for f in futures 
                  if f.days_to_expiry is not None 
                  and min_dte <= f.days_to_expiry <= max_dte]
        
        return sorted(futures, key=lambda x: x.expiry_date or datetime.max)
    
    def get_options(self, underlying: Optional[str] = None,
                   option_type: Optional[str] = None,
                   min_dte: int = 1, max_dte: int = 30,
                   min_strike: float = 0, max_strike: float = float('inf')) -> List[Product]:
        """
        Get options filtered by various criteria.
        
        Args:
            underlying: Filter by underlying asset (e.g., 'BTC')
            option_type: 'call', 'put', or None for both
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            min_strike: Minimum strike price
            max_strike: Maximum strike price
        """
        options = list(self.options.values())
        
        if underlying:
            options = [o for o in options if o.underlying_asset.upper() == underlying.upper()]
        
        if option_type:
            if option_type.lower() == 'call':
                options = [o for o in options if o.is_call]
            elif option_type.lower() == 'put':
                options = [o for o in options if o.is_put]
        
        # Filter by DTE
        options = [o for o in options 
                  if o.days_to_expiry is not None 
                  and min_dte <= o.days_to_expiry <= max_dte]
        
        # Filter by strike
        options = [o for o in options 
                  if o.strike_price is not None 
                  and min_strike <= o.strike_price <= max_strike]
        
        return sorted(options, key=lambda x: (x.expiry_date or datetime.max, x.strike_price or 0))
    
    def get_option_chain(self, underlying: str, expiry_date: Optional[datetime] = None) -> Dict[str, List[Product]]:
        """
        Get complete option chain for an underlying.
        Returns dict with 'calls' and 'puts' lists sorted by strike.
        """
        options = self.get_options(underlying=underlying)
        
        if expiry_date:
            options = [o for o in options 
                      if o.expiry_date and o.expiry_date.date() == expiry_date.date()]
        
        calls = sorted([o for o in options if o.is_call], key=lambda x: x.strike_price or 0)
        puts = sorted([o for o in options if o.is_put], key=lambda x: x.strike_price or 0)
        
        return {'calls': calls, 'puts': puts}
    
    def get_liquid_products(self, min_volume: float = 100000,
                           min_oi: float = 100) -> List[Product]:
        """Get products meeting liquidity thresholds."""
        return [p for p in self.products.values()
                if p.volume_24h >= min_volume and p.open_interest >= min_oi]
    
    def needs_refresh(self) -> bool:
        """Check if catalog needs refreshing."""
        if not self.last_refresh:
            return True
        return datetime.now() - self.last_refresh > self.refresh_interval
    
    def get_summary(self) -> Dict:
        """Get catalog summary statistics."""
        return {
            'total_products': len(self.products),
            'perpetuals': len(self.perpetuals),
            'futures': len(self.futures),
            'options': len(self.options),
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'needs_refresh': self.needs_refresh()
        }
    
    def print_summary(self):
        """Print catalog summary to console."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("📊 PRODUCT CATALOG SUMMARY")
        print("="*60)
        print(f"Total Products: {summary['total_products']}")
        print(f"  • Perpetuals: {summary['perpetuals']}")
        print(f"  • Futures:    {summary['futures']}")
        print(f"  • Options:    {summary['options']}")
        print(f"Last Refresh:   {summary['last_refresh']}")
        print("="*60)
        
        # Show top perpetuals
        if self.perpetuals:
            print("\n📈 PERPETUAL CONTRACTS:")
            for symbol in list(self.perpetuals.keys())[:5]:
                p = self.perpetuals[symbol]
                print(f"  • {symbol}: tick={p.tick_size}, min_size={p.min_size}")
        
        # Show options expiries
        if self.options:
            expiries = set()
            for opt in self.options.values():
                if opt.expiry_date:
                    expiries.add(opt.expiry_date.date())
            print(f"\n📅 OPTIONS EXPIRY DATES: {len(expiries)} dates")
            for exp in sorted(expiries)[:5]:
                print(f"  • {exp}")


# Test the catalog
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    catalog = ProductCatalog()
    catalog.refresh()
    catalog.print_summary()
    
    # Test getting BTC options
    print("\n🎯 BTC OPTIONS (7-14 DTE):")
    btc_options = catalog.get_options(underlying='BTC', min_dte=7, max_dte=14)
    for opt in btc_options[:10]:
        print(f"  {opt.symbol}: strike={opt.strike_price}, DTE={opt.days_to_expiry}")
