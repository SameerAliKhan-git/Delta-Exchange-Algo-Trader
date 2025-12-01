"""
Product Discovery Module for Delta Exchange Algo Trading Bot
Automatically discovers and classifies all tradable instruments
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading

from config import get_config
from logger import get_logger
from delta_client import get_delta_client


class InstrumentType(Enum):
    """Types of tradable instruments"""
    PERPETUAL = "perpetual"
    FUTURE = "future"
    CALL_OPTION = "call_option"
    PUT_OPTION = "put_option"
    SPOT = "spot"
    UNKNOWN = "unknown"


@dataclass
class Instrument:
    """Represents a tradable instrument"""
    product_id: int
    symbol: str
    instrument_type: InstrumentType
    underlying: str
    quote_currency: str
    
    # Contract specs
    contract_size: float = 1.0
    tick_size: float = 0.01
    lot_size: float = 1.0
    min_size: float = 0.0001
    max_size: float = 1000000.0
    
    # Option-specific
    strike_price: Optional[float] = None
    expiry: Optional[datetime] = None
    option_type: Optional[str] = None  # 'call' or 'put'
    
    # Market data (updated dynamically)
    last_price: float = 0.0
    mark_price: float = 0.0
    index_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume_24h: float = 0.0
    open_interest: float = 0.0
    
    # Option greeks (for options)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    # Liquidity metrics
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    spread_pct: float = 0.0
    
    # Status
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_option(self) -> bool:
        return self.instrument_type in (InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION)
    
    @property
    def is_future(self) -> bool:
        return self.instrument_type in (InstrumentType.FUTURE, InstrumentType.PERPETUAL)
    
    @property
    def days_to_expiry(self) -> Optional[int]:
        if self.expiry:
            return max(0, (self.expiry - datetime.utcnow()).days)
        return None
    
    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last_price or self.mark_price
    
    def meets_liquidity_threshold(self, min_volume_usd: float, max_spread_pct: float = 0.02) -> bool:
        """Check if instrument meets liquidity requirements"""
        if self.volume_24h < min_volume_usd:
            return False
        if self.spread_pct > max_spread_pct:
            return False
        return True


@dataclass
class OptionsChain:
    """Options chain for an underlying"""
    underlying: str
    underlying_price: float
    options: List[Instrument] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_calls(self) -> List[Instrument]:
        return [o for o in self.options if o.instrument_type == InstrumentType.CALL_OPTION]
    
    def get_puts(self) -> List[Instrument]:
        return [o for o in self.options if o.instrument_type == InstrumentType.PUT_OPTION]
    
    def get_by_expiry(self, min_days: int = 3, max_days: int = 30) -> List[Instrument]:
        return [
            o for o in self.options 
            if o.days_to_expiry is not None and min_days <= o.days_to_expiry <= max_days
        ]
    
    def get_by_delta(
        self, 
        option_type: str,  # 'call' or 'put'
        target_delta: float = 0.35,
        tolerance: float = 0.15
    ) -> List[Instrument]:
        """Get options within delta range"""
        if option_type == 'call':
            options = self.get_calls()
        else:
            options = self.get_puts()
        
        return [
            o for o in options
            if o.delta is not None and abs(abs(o.delta) - target_delta) <= tolerance
        ]
    
    def get_atm_strike(self) -> float:
        """Get at-the-money strike price"""
        if not self.options:
            return self.underlying_price
        
        strikes = sorted(set(o.strike_price for o in self.options if o.strike_price))
        if not strikes:
            return self.underlying_price
        
        # Find closest strike to underlying price
        return min(strikes, key=lambda s: abs(s - self.underlying_price))


class ProductDiscovery:
    """
    Discovers and classifies all tradable instruments on Delta Exchange
    Maintains a live catalog with market data
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.client = get_delta_client()
        
        # Product catalog
        self._instruments: Dict[int, Instrument] = {}  # product_id -> Instrument
        self._by_symbol: Dict[str, Instrument] = {}  # symbol -> Instrument
        self._by_underlying: Dict[str, List[Instrument]] = {}  # underlying -> [Instruments]
        self._options_chains: Dict[str, OptionsChain] = {}  # underlying -> OptionsChain
        
        # Discovery state
        self._last_discovery: datetime = datetime.min
        self._discovery_interval = timedelta(minutes=5)
        self._lock = threading.Lock()
        
        self.logger.info("Product discovery initialized")
    
    def discover_all(self, force: bool = False) -> Dict[int, Instrument]:
        """
        Discover all products from Delta Exchange
        
        Returns:
            Dictionary of product_id -> Instrument
        """
        now = datetime.utcnow()
        if not force and (now - self._last_discovery) < self._discovery_interval:
            return self._instruments
        
        self.logger.info("Starting product discovery...")
        
        try:
            products = self.client.get_products()
            
            with self._lock:
                self._instruments.clear()
                self._by_symbol.clear()
                self._by_underlying.clear()
                
                for product in products:
                    instrument = self._parse_product(product)
                    if instrument:
                        self._instruments[instrument.product_id] = instrument
                        self._by_symbol[instrument.symbol] = instrument
                        
                        # Group by underlying
                        if instrument.underlying not in self._by_underlying:
                            self._by_underlying[instrument.underlying] = []
                        self._by_underlying[instrument.underlying].append(instrument)
                
                self._last_discovery = now
            
            # Log summary
            type_counts = {}
            for inst in self._instruments.values():
                t = inst.instrument_type.value
                type_counts[t] = type_counts.get(t, 0) + 1
            
            self.logger.info(
                "Product discovery complete",
                total=len(self._instruments),
                by_type=type_counts
            )
            
            return self._instruments
            
        except Exception as e:
            self.logger.error("Product discovery failed", error=str(e))
            return self._instruments
    
    def _parse_product(self, product: Dict[str, Any]) -> Optional[Instrument]:
        """Parse Delta product data into Instrument"""
        try:
            product_id = int(product.get('id', 0))
            symbol = product.get('symbol', '')
            
            if not product_id or not symbol:
                return None
            
            # Determine instrument type
            product_type = product.get('product_type', '').lower()
            contract_type = product.get('contract_type', '').lower()
            
            if 'option' in product_type or 'option' in contract_type:
                # Determine call or put
                if 'call' in symbol.lower() or product.get('option_type', '').lower() == 'call':
                    instrument_type = InstrumentType.CALL_OPTION
                    option_type = 'call'
                else:
                    instrument_type = InstrumentType.PUT_OPTION
                    option_type = 'put'
            elif 'perpetual' in product_type or 'perpetual' in contract_type:
                instrument_type = InstrumentType.PERPETUAL
                option_type = None
            elif 'future' in product_type or 'future' in contract_type:
                instrument_type = InstrumentType.FUTURE
                option_type = None
            elif 'spot' in product_type:
                instrument_type = InstrumentType.SPOT
                option_type = None
            else:
                instrument_type = InstrumentType.UNKNOWN
                option_type = None
            
            # Parse underlying
            underlying = product.get('underlying_asset', {})
            if isinstance(underlying, dict):
                underlying_symbol = underlying.get('symbol', symbol.split('-')[0])
            else:
                underlying_symbol = str(underlying) if underlying else symbol.split('-')[0]
            
            # Parse quote currency
            quote = product.get('quoting_asset', {})
            if isinstance(quote, dict):
                quote_currency = quote.get('symbol', 'USD')
            else:
                quote_currency = str(quote) if quote else 'USD'
            
            # Parse strike and expiry for options
            strike_price = None
            expiry = None
            
            if instrument_type in (InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION):
                strike_price = float(product.get('strike_price', 0)) or None
                expiry_str = product.get('settlement_time') or product.get('expiry_time')
                if expiry_str:
                    try:
                        if isinstance(expiry_str, str):
                            expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                        elif isinstance(expiry_str, (int, float)):
                            expiry = datetime.fromtimestamp(expiry_str)
                    except Exception:
                        pass
            
            # Contract specifications
            contract_size = float(product.get('contract_value', 1) or 1)
            tick_size = float(product.get('tick_size', 0.01) or 0.01)
            lot_size = float(product.get('lot_size', 1) or 1)
            min_size = float(product.get('min_size', 0.0001) or 0.0001)
            max_size = float(product.get('max_size', 1000000) or 1000000)
            
            # Check if active
            state = product.get('state', '').lower()
            is_active = state in ('live', 'active', '') or product.get('is_active', True)
            
            return Instrument(
                product_id=product_id,
                symbol=symbol,
                instrument_type=instrument_type,
                underlying=underlying_symbol,
                quote_currency=quote_currency,
                contract_size=contract_size,
                tick_size=tick_size,
                lot_size=lot_size,
                min_size=min_size,
                max_size=max_size,
                strike_price=strike_price,
                expiry=expiry,
                option_type=option_type,
                is_active=is_active
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse product: {e}")
            return None
    
    def update_market_data(self, product_ids: List[int] = None):
        """
        Update market data for instruments
        
        Args:
            product_ids: Specific products to update, or None for all
        """
        if product_ids is None:
            product_ids = list(self._instruments.keys())
        
        for product_id in product_ids:
            if product_id not in self._instruments:
                continue
            
            instrument = self._instruments[product_id]
            
            try:
                # Get ticker data
                ticker = self.client.get_ticker(instrument.symbol)
                
                if ticker:
                    instrument.last_price = float(ticker.get('close', ticker.get('last_price', 0)) or 0)
                    instrument.mark_price = float(ticker.get('mark_price', instrument.last_price) or instrument.last_price)
                    instrument.bid = float(ticker.get('bid', 0) or 0)
                    instrument.ask = float(ticker.get('ask', 0) or 0)
                    instrument.volume_24h = float(ticker.get('volume', ticker.get('turnover_usd', 0)) or 0)
                    instrument.open_interest = float(ticker.get('open_interest', 0) or 0)
                    
                    # Calculate spread
                    if instrument.bid > 0 and instrument.ask > 0:
                        instrument.spread_pct = (instrument.ask - instrument.bid) / instrument.mid_price
                    
                    # Get greeks for options
                    if instrument.is_option:
                        instrument.delta = float(ticker.get('greeks', {}).get('delta', 0) or 0)
                        instrument.gamma = float(ticker.get('greeks', {}).get('gamma', 0) or 0)
                        instrument.theta = float(ticker.get('greeks', {}).get('theta', 0) or 0)
                        instrument.vega = float(ticker.get('greeks', {}).get('vega', 0) or 0)
                        instrument.implied_volatility = float(ticker.get('greeks', {}).get('iv', 0) or 0)
                    
                    instrument.last_updated = datetime.utcnow()
                    
            except Exception as e:
                self.logger.debug(f"Failed to update market data for {instrument.symbol}: {e}")
    
    def get_options_chain(self, underlying: str, refresh: bool = False) -> Optional[OptionsChain]:
        """
        Get options chain for an underlying asset
        
        Args:
            underlying: Underlying asset symbol (e.g., 'BTC')
            refresh: Force refresh market data
        
        Returns:
            OptionsChain or None if no options found
        """
        # Ensure discovery is done
        self.discover_all()
        
        underlying = underlying.upper()
        
        if underlying not in self._by_underlying:
            return None
        
        # Get all options for this underlying
        options = [
            inst for inst in self._by_underlying[underlying]
            if inst.is_option and inst.is_active
        ]
        
        if not options:
            return None
        
        # Get underlying price
        perpetual = self.get_perpetual(underlying)
        underlying_price = perpetual.mark_price if perpetual else 0
        
        if not underlying_price:
            # Try to get from options themselves
            for opt in options:
                if opt.mark_price > 0:
                    # Estimate underlying from ATM option
                    if opt.strike_price and abs(opt.delta or 0) > 0.4:
                        underlying_price = opt.strike_price
                        break
        
        if refresh:
            option_ids = [o.product_id for o in options]
            self.update_market_data(option_ids)
        
        chain = OptionsChain(
            underlying=underlying,
            underlying_price=underlying_price,
            options=options,
            last_updated=datetime.utcnow()
        )
        
        self._options_chains[underlying] = chain
        return chain
    
    def get_perpetual(self, underlying: str) -> Optional[Instrument]:
        """Get perpetual contract for an underlying"""
        self.discover_all()
        
        underlying = underlying.upper()
        
        if underlying not in self._by_underlying:
            return None
        
        for inst in self._by_underlying[underlying]:
            if inst.instrument_type == InstrumentType.PERPETUAL and inst.is_active:
                return inst
        
        return None
    
    def get_futures(self, underlying: str) -> List[Instrument]:
        """Get all futures contracts for an underlying"""
        self.discover_all()
        
        underlying = underlying.upper()
        
        if underlying not in self._by_underlying:
            return []
        
        return [
            inst for inst in self._by_underlying[underlying]
            if inst.instrument_type == InstrumentType.FUTURE and inst.is_active
        ]
    
    def get_liquid_instruments(
        self,
        min_volume_usd: float = 1000,
        max_spread_pct: float = 0.02,
        instrument_types: List[InstrumentType] = None
    ) -> List[Instrument]:
        """
        Get instruments meeting liquidity thresholds
        
        Args:
            min_volume_usd: Minimum 24h volume in USD
            max_spread_pct: Maximum bid-ask spread percentage
            instrument_types: Filter by instrument types
        
        Returns:
            List of liquid instruments
        """
        self.discover_all()
        
        liquid = []
        for inst in self._instruments.values():
            if not inst.is_active:
                continue
            
            if instrument_types and inst.instrument_type not in instrument_types:
                continue
            
            if inst.meets_liquidity_threshold(min_volume_usd, max_spread_pct):
                liquid.append(inst)
        
        # Sort by volume
        liquid.sort(key=lambda x: x.volume_24h, reverse=True)
        return liquid
    
    def get_instrument(self, product_id: int) -> Optional[Instrument]:
        """Get instrument by product ID"""
        self.discover_all()
        return self._instruments.get(product_id)
    
    def get_instrument_by_symbol(self, symbol: str) -> Optional[Instrument]:
        """Get instrument by symbol"""
        self.discover_all()
        return self._by_symbol.get(symbol.upper())
    
    def get_all_underlyings(self) -> List[str]:
        """Get list of all underlying assets"""
        self.discover_all()
        return list(self._by_underlying.keys())
    
    def get_tradable_summary(self) -> Dict[str, Any]:
        """Get summary of tradable instruments"""
        self.discover_all()
        
        summary = {
            "total_instruments": len(self._instruments),
            "active_instruments": sum(1 for i in self._instruments.values() if i.is_active),
            "underlyings": list(self._by_underlying.keys()),
            "by_type": {},
            "liquid_count": 0
        }
        
        for inst in self._instruments.values():
            t = inst.instrument_type.value
            summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
        
        # Count liquid instruments
        min_vol = self.config.data_ingestion.ticker_poll_interval  # Use as proxy
        summary["liquid_count"] = len(self.get_liquid_instruments(min_volume_usd=1000))
        
        return summary


# Singleton instance
_product_discovery: Optional[ProductDiscovery] = None


def get_product_discovery() -> ProductDiscovery:
    """Get or create the global product discovery"""
    global _product_discovery
    if _product_discovery is None:
        _product_discovery = ProductDiscovery()
    return _product_discovery


if __name__ == "__main__":
    # Test product discovery
    discovery = get_product_discovery()
    
    # Discover all products
    instruments = discovery.discover_all(force=True)
    print(f"Discovered {len(instruments)} instruments")
    
    # Get summary
    summary = discovery.get_tradable_summary()
    print(f"\nSummary: {summary}")
    
    # Get liquid instruments
    liquid = discovery.get_liquid_instruments(min_volume_usd=1000)
    print(f"\nLiquid instruments: {len(liquid)}")
    
    for inst in liquid[:5]:
        print(f"  {inst.symbol} ({inst.instrument_type.value}): vol={inst.volume_24h:.0f}")
    
    # Get perpetual for BTC
    btc_perp = discovery.get_perpetual("BTC")
    if btc_perp:
        print(f"\nBTC Perpetual: {btc_perp.symbol} (ID: {btc_perp.product_id})")
    
    # Get options chain
    chain = discovery.get_options_chain("BTC")
    if chain:
        print(f"\nBTC Options Chain: {len(chain.options)} options")
        print(f"  Calls: {len(chain.get_calls())}")
        print(f"  Puts: {len(chain.get_puts())}")
