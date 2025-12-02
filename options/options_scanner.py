"""
ALADDIN - Options Scanner
===========================
Scans options chain for trading opportunities based on delta buckets,
implied volatility, and liquidity.
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.credentials import API_KEY, API_SECRET, BASE_URL
from catalog.product_catalog import ProductCatalog, Product, ProductType


@dataclass
class Greeks:
    """Option Greeks."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass
class OptionQuote:
    """Option quote with greeks and pricing."""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    
    # Pricing
    mark_price: float
    bid: float
    ask: float
    last_price: float
    
    # Volume
    volume_24h: float
    open_interest: float
    
    # Greeks
    greeks: Greeks = field(default_factory=Greeks)
    iv: float = 0.0
    
    # Calculated
    moneyness: str = "atm"  # itm, atm, otm
    intrinsic_value: float = 0.0
    time_value: float = 0.0
    
    @property
    def days_to_expiry(self) -> int:
        now = datetime.now()
        exp = self.expiry
        if exp.tzinfo is not None:
            exp = exp.replace(tzinfo=None)
        return max(0, (exp - now).days)
    
    @property
    def spread_pct(self) -> float:
        if self.bid > 0:
            return ((self.ask - self.bid) / self.bid) * 100
        return float('inf')
    
    @property
    def is_liquid(self) -> bool:
        return self.spread_pct < 5.0 and self.open_interest >= 10
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'strike': self.strike,
            'expiry': self.expiry.strftime('%Y-%m-%d'),
            'type': self.option_type,
            'mark': self.mark_price,
            'bid': self.bid,
            'ask': self.ask,
            'iv': f"{self.iv:.1f}%",
            'delta': f"{self.greeks.delta:.3f}",
            'dte': self.days_to_expiry,
            'oi': self.open_interest,
            'moneyness': self.moneyness
        }


class DeltaBucket(Enum):
    """Delta bucket classification for options."""
    DEEP_ITM = "deep_itm"      # |delta| > 0.80
    ITM = "itm"                # 0.60 < |delta| <= 0.80
    ATM = "atm"                # 0.40 < |delta| <= 0.60
    OTM = "otm"                # 0.20 < |delta| <= 0.40
    DEEP_OTM = "deep_otm"      # |delta| <= 0.20


class OptionsScanner:
    """
    Scans options chains for trading opportunities.
    
    Features:
    - Delta bucket classification
    - Implied volatility analysis
    - Liquidity filtering
    - Spread opportunity detection
    """
    
    def __init__(self, catalog: ProductCatalog = None):
        self.logger = logging.getLogger('Aladdin.OptionsScanner')
        self.catalog = catalog or ProductCatalog()
        self.base_url = BASE_URL
        
        # Refresh catalog if needed
        if self.catalog.needs_refresh():
            self.catalog.refresh()
        
        # Cache quotes
        self._quotes: Dict[str, OptionQuote] = {}
        self._last_refresh: Optional[datetime] = None
    
    def refresh_quotes(self) -> int:
        """Fetch latest option quotes from Delta Exchange."""
        try:
            # Get all options
            options = list(self.catalog.options.values())
            
            # Fetch ticker data
            response = requests.get(
                f"{self.base_url}/v2/tickers",
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                return 0
            
            tickers = {t['symbol']: t for t in data.get('result', [])}
            
            # Process each option
            for opt in options:
                if opt.symbol in tickers:
                    quote = self._create_quote(opt, tickers[opt.symbol])
                    if quote:
                        self._quotes[quote.symbol] = quote
            
            self._last_refresh = datetime.now()
            self.logger.info(f"Refreshed {len(self._quotes)} option quotes")
            return len(self._quotes)
            
        except Exception as e:
            self.logger.error(f"Error refreshing quotes: {e}")
            return 0
    
    def _create_quote(self, product: Product, ticker: Dict) -> Optional[OptionQuote]:
        """Create OptionQuote from product and ticker data."""
        try:
            # Get pricing
            mark = float(ticker.get('mark_price', 0) or 0)
            bid = float(ticker.get('bid', 0) or 0)
            ask = float(ticker.get('ask', 0) or 0)
            last = float(ticker.get('close', 0) or 0)
            
            if mark == 0 and bid == 0 and ask == 0:
                return None
            
            # Determine option type
            if product.is_call:
                option_type = 'call'
            elif product.is_put:
                option_type = 'put'
            else:
                return None
            
            # Get spot price for moneyness
            spot = float(ticker.get('spot_price', 0) or 0)
            strike = product.strike_price or 0
            
            # Calculate moneyness
            if spot > 0 and strike > 0:
                if option_type == 'call':
                    if strike < spot * 0.97:
                        moneyness = 'itm'
                    elif strike > spot * 1.03:
                        moneyness = 'otm'
                    else:
                        moneyness = 'atm'
                else:  # put
                    if strike > spot * 1.03:
                        moneyness = 'itm'
                    elif strike < spot * 0.97:
                        moneyness = 'otm'
                    else:
                        moneyness = 'atm'
            else:
                moneyness = 'atm'
            
            # Get Greeks from ticker
            greeks = Greeks(
                delta=float(ticker.get('greeks', {}).get('delta', 0) or 0),
                gamma=float(ticker.get('greeks', {}).get('gamma', 0) or 0),
                theta=float(ticker.get('greeks', {}).get('theta', 0) or 0),
                vega=float(ticker.get('greeks', {}).get('vega', 0) or 0),
                rho=float(ticker.get('greeks', {}).get('rho', 0) or 0),
            )
            
            # Get IV
            iv = float(ticker.get('greeks', {}).get('iv', 0) or 0)
            
            return OptionQuote(
                symbol=product.symbol,
                underlying=product.underlying_asset,
                strike=strike,
                expiry=product.expiry_date or datetime.now(),
                option_type=option_type,
                mark_price=mark or ((bid + ask) / 2 if bid and ask else last),
                bid=bid,
                ask=ask,
                last_price=last,
                volume_24h=float(ticker.get('volume', 0) or 0),
                open_interest=float(ticker.get('open_interest', 0) or 0),
                greeks=greeks,
                iv=iv * 100 if iv < 10 else iv,  # Convert to percentage if needed
                moneyness=moneyness
            )
            
        except Exception as e:
            self.logger.debug(f"Error creating quote for {product.symbol}: {e}")
            return None
    
    def get_delta_bucket(self, delta: float) -> DeltaBucket:
        """Classify option by delta bucket."""
        abs_delta = abs(delta)
        
        if abs_delta > 0.80:
            return DeltaBucket.DEEP_ITM
        elif abs_delta > 0.60:
            return DeltaBucket.ITM
        elif abs_delta > 0.40:
            return DeltaBucket.ATM
        elif abs_delta > 0.20:
            return DeltaBucket.OTM
        else:
            return DeltaBucket.DEEP_OTM
    
    def scan_by_delta(self, underlying: str, target_delta: float,
                     tolerance: float = 0.10,
                     option_type: str = None,
                     min_dte: int = 1, max_dte: int = 30) -> List[OptionQuote]:
        """
        Find options matching target delta.
        
        Args:
            underlying: Underlying asset (e.g., 'BTC')
            target_delta: Target delta (0.30 for ~30 delta)
            tolerance: Delta tolerance range
            option_type: 'call', 'put', or None for both
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
        """
        matches = []
        
        for quote in self._quotes.values():
            # Filter by underlying
            if quote.underlying.upper() != underlying.upper():
                continue
            
            # Filter by option type
            if option_type and quote.option_type != option_type:
                continue
            
            # Filter by DTE
            dte = quote.days_to_expiry
            if dte < min_dte or dte > max_dte:
                continue
            
            # Check delta match
            abs_delta = abs(quote.greeks.delta)
            if abs(abs_delta - target_delta) <= tolerance:
                matches.append(quote)
        
        # Sort by how close to target delta
        matches.sort(key=lambda q: abs(abs(q.greeks.delta) - target_delta))
        
        return matches
    
    def scan_by_bucket(self, underlying: str, bucket: DeltaBucket,
                      option_type: str = None,
                      min_dte: int = 1, max_dte: int = 30) -> List[OptionQuote]:
        """Find options in a specific delta bucket."""
        matches = []
        
        for quote in self._quotes.values():
            if quote.underlying.upper() != underlying.upper():
                continue
            
            if option_type and quote.option_type != option_type:
                continue
            
            dte = quote.days_to_expiry
            if dte < min_dte or dte > max_dte:
                continue
            
            if self.get_delta_bucket(quote.greeks.delta) == bucket:
                matches.append(quote)
        
        return sorted(matches, key=lambda q: abs(q.greeks.delta), reverse=True)
    
    def scan_high_iv(self, underlying: str = None, min_iv: float = 50,
                    option_type: str = None,
                    min_dte: int = 1, max_dte: int = 30) -> List[OptionQuote]:
        """Find options with high implied volatility (for selling strategies)."""
        matches = []
        
        for quote in self._quotes.values():
            if underlying and quote.underlying.upper() != underlying.upper():
                continue
            
            if option_type and quote.option_type != option_type:
                continue
            
            dte = quote.days_to_expiry
            if dte < min_dte or dte > max_dte:
                continue
            
            if quote.iv >= min_iv:
                matches.append(quote)
        
        return sorted(matches, key=lambda q: q.iv, reverse=True)
    
    def scan_liquid(self, underlying: str = None, 
                   min_volume: float = 10,
                   min_oi: float = 50,
                   max_spread_pct: float = 5.0) -> List[OptionQuote]:
        """Find liquid options for trading."""
        matches = []
        
        for quote in self._quotes.values():
            if underlying and quote.underlying.upper() != underlying.upper():
                continue
            
            if (quote.volume_24h >= min_volume and
                quote.open_interest >= min_oi and
                quote.spread_pct <= max_spread_pct):
                matches.append(quote)
        
        return sorted(matches, key=lambda q: q.open_interest, reverse=True)
    
    def get_chain(self, underlying: str, expiry: datetime = None) -> Dict[str, List[OptionQuote]]:
        """
        Get complete options chain for an underlying.
        Returns dict with 'calls' and 'puts' sorted by strike.
        """
        calls = []
        puts = []
        
        for quote in self._quotes.values():
            if quote.underlying.upper() != underlying.upper():
                continue
            
            if expiry and quote.expiry.date() != expiry.date():
                continue
            
            if quote.option_type == 'call':
                calls.append(quote)
            else:
                puts.append(quote)
        
        return {
            'calls': sorted(calls, key=lambda q: q.strike),
            'puts': sorted(puts, key=lambda q: q.strike)
        }
    
    def get_atm_options(self, underlying: str, spot_price: float,
                       expiry: datetime = None) -> Dict[str, Optional[OptionQuote]]:
        """Find ATM call and put closest to spot price."""
        chain = self.get_chain(underlying, expiry)
        
        atm_call = None
        atm_put = None
        min_call_diff = float('inf')
        min_put_diff = float('inf')
        
        for call in chain['calls']:
            diff = abs(call.strike - spot_price)
            if diff < min_call_diff:
                min_call_diff = diff
                atm_call = call
        
        for put in chain['puts']:
            diff = abs(put.strike - spot_price)
            if diff < min_put_diff:
                min_put_diff = diff
                atm_put = put
        
        return {'call': atm_call, 'put': atm_put}
    
    def get_expiries(self, underlying: str) -> List[datetime]:
        """Get available expiry dates for an underlying."""
        expiries = set()
        
        for quote in self._quotes.values():
            if quote.underlying.upper() == underlying.upper():
                expiries.add(quote.expiry.date())
        
        return sorted(expiries)
    
    def print_chain(self, underlying: str, expiry: datetime = None):
        """Print options chain to console."""
        chain = self.get_chain(underlying, expiry)
        
        print("\n" + "="*90)
        print(f"📊 OPTIONS CHAIN - {underlying}")
        if expiry:
            print(f"   Expiry: {expiry.strftime('%Y-%m-%d')}")
        print("="*90)
        
        # Header
        print(f"\n{'CALLS':<45} │ {'PUTS':<45}")
        print("-"*45 + "─┼─" + "-"*45)
        print(f"{'Strike':>8} {'Bid':>8} {'Ask':>8} {'Delta':>7} {'IV':>6} │ "
              f"{'Strike':>8} {'Bid':>8} {'Ask':>8} {'Delta':>7} {'IV':>6}")
        print("-"*45 + "─┼─" + "-"*45)
        
        # Match calls and puts by strike
        all_strikes = sorted(set(
            [c.strike for c in chain['calls']] + 
            [p.strike for p in chain['puts']]
        ))
        
        calls_by_strike = {c.strike: c for c in chain['calls']}
        puts_by_strike = {p.strike: p for p in chain['puts']}
        
        for strike in all_strikes[:20]:  # Limit display
            call = calls_by_strike.get(strike)
            put = puts_by_strike.get(strike)
            
            if call:
                call_str = f"{call.strike:>8.0f} {call.bid:>8.2f} {call.ask:>8.2f} {call.greeks.delta:>7.3f} {call.iv:>5.0f}%"
            else:
                call_str = " " * 45
            
            if put:
                put_str = f"{put.strike:>8.0f} {put.bid:>8.2f} {put.ask:>8.2f} {put.greeks.delta:>7.3f} {put.iv:>5.0f}%"
            else:
                put_str = " " * 45
            
            print(f"{call_str} │ {put_str}")
        
        print("="*90)
    
    def print_scan_results(self, title: str, options: List[OptionQuote], limit: int = 10):
        """Print scan results to console."""
        print("\n" + "="*80)
        print(f"🔍 {title}")
        print("="*80)
        
        if not options:
            print("  No options found matching criteria")
            return
        
        print(f"{'Symbol':<25} {'Type':<5} {'Strike':>10} {'DTE':>5} {'Delta':>7} {'IV':>6} {'OI':>8}")
        print("-"*80)
        
        for opt in options[:limit]:
            print(f"{opt.symbol:<25} {opt.option_type:<5} {opt.strike:>10.0f} "
                  f"{opt.days_to_expiry:>5} {opt.greeks.delta:>7.3f} {opt.iv:>5.0f}% {opt.open_interest:>8.0f}")
        
        if len(options) > limit:
            print(f"  ... and {len(options) - limit} more")
        
        print("="*80)


# Test the scanner
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scanner = OptionsScanner()
    scanner.refresh_quotes()
    
    # Show available expiries
    expiries = scanner.get_expiries('BTC')
    print(f"\n📅 BTC Expiries: {[e.strftime('%Y-%m-%d') for e in expiries[:5]]}")
    
    # Scan for 30-delta options
    delta_30 = scanner.scan_by_delta('BTC', target_delta=0.30, min_dte=5, max_dte=14)
    scanner.print_scan_results("30-Delta BTC Options (5-14 DTE)", delta_30)
    
    # Scan for high IV
    high_iv = scanner.scan_high_iv(underlying='BTC', min_iv=60, min_dte=3, max_dte=14)
    scanner.print_scan_results("High IV BTC Options (>60%)", high_iv)
    
    # Print chain for nearest expiry
    if expiries:
        scanner.print_chain('BTC', datetime.combine(expiries[0], datetime.min.time()))
