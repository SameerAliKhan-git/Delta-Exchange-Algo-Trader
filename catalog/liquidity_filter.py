"""
ALADDIN - Liquidity Filter
=============================
Filters products by liquidity metrics and enriches with market data.
"""

import requests
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.credentials import API_KEY, API_SECRET, BASE_URL

from .product_catalog import Product, ProductCatalog


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a product."""
    symbol: str
    volume_24h: float
    open_interest: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread_pct: float
    depth_score: float  # 0-100
    liquidity_grade: str  # A, B, C, D, F
    
    @property
    def is_liquid(self) -> bool:
        return self.liquidity_grade in ('A', 'B')


class LiquidityFilter:
    """
    Enriches products with liquidity data and filters by thresholds.
    """
    
    # Liquidity grade thresholds
    GRADE_THRESHOLDS = {
        'A': {'volume': 1000000, 'spread': 0.05, 'oi': 500},
        'B': {'volume': 100000, 'spread': 0.10, 'oi': 100},
        'C': {'volume': 10000, 'spread': 0.25, 'oi': 50},
        'D': {'volume': 1000, 'spread': 0.50, 'oi': 10},
    }
    
    def __init__(self, catalog: ProductCatalog):
        self.logger = logging.getLogger('Aladdin.Liquidity')
        self.catalog = catalog
        self.base_url = BASE_URL
        
        # Cache liquidity metrics
        self.metrics: Dict[str, LiquidityMetrics] = {}
        self.last_update: Optional[datetime] = None
        self.update_interval = timedelta(minutes=1)
    
    def update_liquidity(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Update liquidity metrics for products.
        If symbols not specified, updates all products.
        """
        try:
            if symbols is None:
                symbols = list(self.catalog.products.keys())
            
            self.logger.info(f"Updating liquidity for {len(symbols)} products...")
            
            # Fetch ticker data for all products at once
            response = requests.get(
                f"{self.base_url}/v2/tickers",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and 'result' in data:
                tickers = {t['symbol']: t for t in data['result']}
                
                for symbol in symbols:
                    if symbol in tickers:
                        self._process_ticker(symbol, tickers[symbol])
                
                self.last_update = datetime.now()
                self.logger.info(f"Updated liquidity for {len(self.metrics)} products")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating liquidity: {e}")
            return False
    
    def _process_ticker(self, symbol: str, ticker: Dict):
        """Process ticker data into liquidity metrics."""
        try:
            volume_24h = float(ticker.get('volume', 0))
            open_interest = float(ticker.get('open_interest', 0) or 0)
            
            bid_price = float(ticker.get('bid', 0) or 0)
            ask_price = float(ticker.get('ask', 0) or 0)
            
            # Calculate spread
            if bid_price > 0 and ask_price > 0:
                mid_price = (bid_price + ask_price) / 2
                spread_pct = ((ask_price - bid_price) / mid_price) * 100
            else:
                spread_pct = 100.0
            
            # Calculate depth score (simplified)
            depth_score = min(100, (volume_24h / 10000) + (open_interest / 10))
            
            # Determine grade
            grade = self._calculate_grade(volume_24h, spread_pct, open_interest)
            
            metrics = LiquidityMetrics(
                symbol=symbol,
                volume_24h=volume_24h,
                open_interest=open_interest,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=float(ticker.get('bid_size', 0) or 0),
                ask_size=float(ticker.get('ask_size', 0) or 0),
                spread_pct=spread_pct,
                depth_score=depth_score,
                liquidity_grade=grade
            )
            
            self.metrics[symbol] = metrics
            
            # Update the product in catalog
            if symbol in self.catalog.products:
                product = self.catalog.products[symbol]
                product.volume_24h = volume_24h
                product.open_interest = open_interest
                product.bid_ask_spread = spread_pct
                
        except Exception as e:
            self.logger.debug(f"Error processing ticker for {symbol}: {e}")
    
    def _calculate_grade(self, volume: float, spread: float, oi: float) -> str:
        """Calculate liquidity grade based on thresholds."""
        for grade, thresholds in self.GRADE_THRESHOLDS.items():
            if (volume >= thresholds['volume'] and 
                spread <= thresholds['spread'] and
                oi >= thresholds['oi']):
                return grade
        return 'F'
    
    def get_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """Get liquidity metrics for a symbol."""
        return self.metrics.get(symbol)
    
    def filter_liquid(self, products: List[Product],
                     min_grade: str = 'C',
                     min_volume: float = 0,
                     max_spread: float = 100) -> List[Product]:
        """
        Filter products by liquidity criteria.
        
        Args:
            products: List of products to filter
            min_grade: Minimum liquidity grade (A, B, C, D)
            min_volume: Minimum 24h volume
            max_spread: Maximum bid-ask spread percentage
        """
        grade_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        min_grade_value = grade_order.get(min_grade, 0)
        
        filtered = []
        for product in products:
            metrics = self.metrics.get(product.symbol)
            if metrics:
                grade_value = grade_order.get(metrics.liquidity_grade, 0)
                if (grade_value >= min_grade_value and
                    metrics.volume_24h >= min_volume and
                    metrics.spread_pct <= max_spread):
                    filtered.append(product)
        
        # Sort by liquidity grade (best first)
        filtered.sort(key=lambda p: grade_order.get(
            self.metrics.get(p.symbol, LiquidityMetrics('', 0, 0, 0, 0, 0, 0, 100, 0, 'F')).liquidity_grade, 0
        ), reverse=True)
        
        return filtered
    
    def get_most_liquid(self, product_type: Optional[str] = None,
                       underlying: Optional[str] = None,
                       top_n: int = 10) -> List[Product]:
        """
        Get the most liquid products.
        
        Args:
            product_type: 'perpetual', 'future', 'option', or None for all
            underlying: Filter by underlying asset
            top_n: Number of products to return
        """
        # Get products by type
        if product_type == 'perpetual':
            products = list(self.catalog.perpetuals.values())
        elif product_type == 'future':
            products = list(self.catalog.futures.values())
        elif product_type == 'option':
            products = list(self.catalog.options.values())
        else:
            products = list(self.catalog.products.values())
        
        # Filter by underlying
        if underlying:
            products = [p for p in products if p.underlying_asset.upper() == underlying.upper()]
        
        # Sort by volume
        products.sort(key=lambda p: self.metrics.get(p.symbol, LiquidityMetrics('', 0, 0, 0, 0, 0, 0, 100, 0, 'F')).volume_24h, reverse=True)
        
        return products[:top_n]
    
    def get_tradeable_symbols(self, min_grade: str = 'C') -> List[str]:
        """Get list of symbols that meet minimum liquidity requirements."""
        grade_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        min_grade_value = grade_order.get(min_grade, 0)
        
        return [symbol for symbol, metrics in self.metrics.items()
                if grade_order.get(metrics.liquidity_grade, 0) >= min_grade_value]
    
    def print_liquidity_report(self, top_n: int = 20):
        """Print liquidity report to console."""
        print("\n" + "="*80)
        print("💧 LIQUIDITY REPORT")
        print("="*80)
        print(f"{'Symbol':<20} {'Volume 24h':>15} {'OI':>10} {'Spread':>8} {'Grade':>6}")
        print("-"*80)
        
        # Sort by volume
        sorted_metrics = sorted(
            self.metrics.values(),
            key=lambda m: m.volume_24h,
            reverse=True
        )[:top_n]
        
        for m in sorted_metrics:
            vol_str = f"${m.volume_24h:,.0f}"
            oi_str = f"{m.open_interest:,.0f}"
            spread_str = f"{m.spread_pct:.2f}%"
            print(f"{m.symbol:<20} {vol_str:>15} {oi_str:>10} {spread_str:>8} {m.liquidity_grade:>6}")
        
        print("="*80)
        
        # Grade distribution
        grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for m in self.metrics.values():
            grades[m.liquidity_grade] = grades.get(m.liquidity_grade, 0) + 1
        
        print("\n📊 GRADE DISTRIBUTION:")
        for grade, count in grades.items():
            bar = "█" * (count // 2)
            print(f"  Grade {grade}: {count:>4} {bar}")


# Test the liquidity filter
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create catalog and filter
    catalog = ProductCatalog()
    catalog.refresh()
    
    liq_filter = LiquidityFilter(catalog)
    liq_filter.update_liquidity()
    liq_filter.print_liquidity_report()
    
    # Get most liquid perpetuals
    print("\n🔥 MOST LIQUID PERPETUALS:")
    top_perps = liq_filter.get_most_liquid(product_type='perpetual', top_n=5)
    for p in top_perps:
        m = liq_filter.get_metrics(p.symbol)
        if m:
            print(f"  {p.symbol}: Vol=${m.volume_24h:,.0f}, Grade={m.liquidity_grade}")
