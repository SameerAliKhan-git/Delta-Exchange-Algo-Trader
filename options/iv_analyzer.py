"""
ALADDIN - IV Analyzer
======================
Implied Volatility surface analysis and term structure.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from .options_scanner import OptionsScanner, OptionQuote


@dataclass
class IVPoint:
    """Single IV data point."""
    strike: float
    expiry: datetime
    iv: float
    delta: float
    option_type: str
    moneyness: float  # strike / spot


@dataclass
class IVSurface:
    """Implied volatility surface data."""
    underlying: str
    spot_price: float
    points: List[IVPoint] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Derived metrics
    atm_iv: float = 0.0
    iv_skew: float = 0.0  # Put IV - Call IV at same delta
    term_structure_slope: float = 0.0
    
    def get_smile(self, expiry: datetime) -> List[IVPoint]:
        """Get volatility smile for a specific expiry."""
        return sorted(
            [p for p in self.points if p.expiry.date() == expiry.date()],
            key=lambda x: x.strike
        )
    
    def get_term_structure(self, strike: float = None, 
                          delta: float = 0.5) -> List[Tuple[int, float]]:
        """Get IV term structure (by DTE)."""
        if strike:
            points = [p for p in self.points if abs(p.strike - strike) < self.spot_price * 0.02]
        else:
            # Use ATM options
            points = [p for p in self.points if 0.4 < abs(p.delta) < 0.6]
        
        by_dte = {}
        for p in points:
            dte = (p.expiry - datetime.now()).days
            if dte > 0:
                if dte not in by_dte:
                    by_dte[dte] = []
                by_dte[dte].append(p.iv)
        
        return sorted([(dte, np.mean(ivs)) for dte, ivs in by_dte.items()])


class IVAnalyzer:
    """
    Analyzes implied volatility patterns for trading opportunities.
    
    Features:
    - IV surface construction
    - Volatility smile analysis
    - Term structure analysis
    - Skew detection
    - IV percentile ranking
    """
    
    def __init__(self, scanner: OptionsScanner = None):
        self.logger = logging.getLogger('Aladdin.IVAnalyzer')
        self.scanner = scanner or OptionsScanner()
        
        # Historical IV for percentile calculation
        self._iv_history: Dict[str, List[float]] = {}
    
    def build_surface(self, underlying: str, spot_price: float) -> IVSurface:
        """Build IV surface from current option quotes."""
        surface = IVSurface(
            underlying=underlying,
            spot_price=spot_price
        )
        
        # Get all options for underlying
        chain = self.scanner.get_chain(underlying)
        
        for option_type, options in [('call', chain['calls']), ('put', chain['puts'])]:
            for opt in options:
                if opt.iv > 0:
                    point = IVPoint(
                        strike=opt.strike,
                        expiry=opt.expiry,
                        iv=opt.iv,
                        delta=opt.greeks.delta,
                        option_type=option_type,
                        moneyness=opt.strike / spot_price
                    )
                    surface.points.append(point)
        
        # Calculate derived metrics
        self._calculate_surface_metrics(surface)
        
        return surface
    
    def _calculate_surface_metrics(self, surface: IVSurface):
        """Calculate derived metrics for the IV surface."""
        if not surface.points:
            return
        
        # ATM IV (average of ~50 delta options)
        atm_points = [p for p in surface.points if 0.4 < abs(p.delta) < 0.6]
        if atm_points:
            surface.atm_iv = np.mean([p.iv for p in atm_points])
        
        # IV Skew (25 delta put IV - 25 delta call IV)
        puts_25d = [p for p in surface.points 
                   if p.option_type == 'put' and 0.20 < abs(p.delta) < 0.30]
        calls_25d = [p for p in surface.points 
                    if p.option_type == 'call' and 0.20 < abs(p.delta) < 0.30]
        
        if puts_25d and calls_25d:
            put_iv = np.mean([p.iv for p in puts_25d])
            call_iv = np.mean([p.iv for p in calls_25d])
            surface.iv_skew = put_iv - call_iv
        
        # Term structure slope
        term = surface.get_term_structure()
        if len(term) >= 2:
            dtes, ivs = zip(*term)
            if len(set(dtes)) > 1:
                slope = np.polyfit(dtes, ivs, 1)[0]
                surface.term_structure_slope = slope
    
    def get_iv_percentile(self, underlying: str, current_iv: float, 
                         lookback_days: int = 30) -> float:
        """
        Calculate IV percentile rank.
        Returns 0-100 indicating where current IV ranks historically.
        """
        key = f"{underlying}_{lookback_days}"
        history = self._iv_history.get(key, [])
        
        if not history:
            return 50.0  # Default to median if no history
        
        # Calculate percentile
        count_below = sum(1 for iv in history if iv < current_iv)
        percentile = (count_below / len(history)) * 100
        
        return percentile
    
    def record_iv(self, underlying: str, iv: float):
        """Record IV observation for historical tracking."""
        key = f"{underlying}_30"  # 30-day lookback
        if key not in self._iv_history:
            self._iv_history[key] = []
        
        self._iv_history[key].append(iv)
        
        # Keep last 252 observations (1 trading year)
        if len(self._iv_history[key]) > 252:
            self._iv_history[key] = self._iv_history[key][-252:]
    
    def analyze_skew(self, surface: IVSurface) -> Dict:
        """
        Analyze volatility skew for trading signals.
        
        Positive skew (puts > calls): Bearish sentiment, hedge demand
        Negative skew (calls > puts): Bullish sentiment, call buying
        """
        skew = surface.iv_skew
        
        if skew > 5:
            signal = "BEARISH"
            description = "Strong put demand - market hedging"
        elif skew > 2:
            signal = "SLIGHTLY_BEARISH"
            description = "Moderate put demand"
        elif skew < -5:
            signal = "BULLISH"
            description = "Strong call demand - upside speculation"
        elif skew < -2:
            signal = "SLIGHTLY_BULLISH"
            description = "Moderate call demand"
        else:
            signal = "NEUTRAL"
            description = "Balanced put/call demand"
        
        return {
            'skew': skew,
            'signal': signal,
            'description': description,
            'atm_iv': surface.atm_iv,
            'term_slope': surface.term_structure_slope
        }
    
    def analyze_term_structure(self, surface: IVSurface) -> Dict:
        """
        Analyze term structure for trading signals.
        
        Contango (upward slope): Normal market conditions
        Backwardation (inverted): Near-term uncertainty/events
        """
        term = surface.get_term_structure()
        
        if not term:
            return {'signal': 'UNKNOWN', 'description': 'Insufficient data'}
        
        slope = surface.term_structure_slope
        
        if slope > 0.5:
            signal = "CONTANGO"
            description = "Normal term structure - no near-term events expected"
        elif slope < -0.5:
            signal = "BACKWARDATION"
            description = "Inverted - near-term uncertainty or event expected"
        else:
            signal = "FLAT"
            description = "Flat term structure"
        
        return {
            'slope': slope,
            'signal': signal,
            'description': description,
            'term_structure': term[:5]  # First 5 points
        }
    
    def find_iv_opportunities(self, underlying: str, spot_price: float) -> List[Dict]:
        """
        Find IV-based trading opportunities.
        
        Looks for:
        - High IV (sell premium)
        - Low IV (buy options)
        - Skew trades
        - Calendar spread opportunities
        """
        opportunities = []
        surface = self.build_surface(underlying, spot_price)
        
        # High IV opportunity
        percentile = self.get_iv_percentile(underlying, surface.atm_iv)
        
        if percentile > 80:
            opportunities.append({
                'type': 'SELL_PREMIUM',
                'description': f'IV at {percentile:.0f}th percentile - consider selling options',
                'strategy': 'Iron Condor or Strangle',
                'iv': surface.atm_iv,
                'confidence': min(1.0, (percentile - 50) / 50)
            })
        elif percentile < 20:
            opportunities.append({
                'type': 'BUY_OPTIONS',
                'description': f'IV at {percentile:.0f}th percentile - options are cheap',
                'strategy': 'Long Straddle or Strangle',
                'iv': surface.atm_iv,
                'confidence': min(1.0, (50 - percentile) / 50)
            })
        
        # Skew trade
        skew_analysis = self.analyze_skew(surface)
        if abs(surface.iv_skew) > 5:
            opportunities.append({
                'type': 'SKEW_TRADE',
                'description': skew_analysis['description'],
                'strategy': 'Put Spread' if surface.iv_skew > 0 else 'Call Spread',
                'skew': surface.iv_skew,
                'confidence': min(1.0, abs(surface.iv_skew) / 10)
            })
        
        # Calendar opportunity
        term_analysis = self.analyze_term_structure(surface)
        if term_analysis['signal'] == 'BACKWARDATION':
            opportunities.append({
                'type': 'CALENDAR_SPREAD',
                'description': 'Term structure inverted - calendar spread opportunity',
                'strategy': 'Sell near-term, Buy far-term',
                'slope': term_analysis['slope'],
                'confidence': min(1.0, abs(term_analysis['slope']))
            })
        
        return opportunities
    
    def print_analysis(self, underlying: str, spot_price: float):
        """Print comprehensive IV analysis."""
        surface = self.build_surface(underlying, spot_price)
        
        print("\n" + "="*70)
        print(f"📈 IV ANALYSIS - {underlying}")
        print(f"   Spot: ${spot_price:,.2f}")
        print("="*70)
        
        print(f"\n📊 SURFACE METRICS:")
        print(f"  ATM IV: {surface.atm_iv:.1f}%")
        print(f"  25Δ Skew: {surface.iv_skew:+.1f}%")
        print(f"  Term Slope: {surface.term_structure_slope:+.3f}")
        
        # Skew analysis
        skew = self.analyze_skew(surface)
        print(f"\n🎯 SKEW SIGNAL: {skew['signal']}")
        print(f"   {skew['description']}")
        
        # Term structure
        term = self.analyze_term_structure(surface)
        print(f"\n📅 TERM STRUCTURE: {term['signal']}")
        print(f"   {term['description']}")
        
        # Opportunities
        opps = self.find_iv_opportunities(underlying, spot_price)
        if opps:
            print(f"\n💡 OPPORTUNITIES:")
            for opp in opps:
                print(f"  • {opp['type']}: {opp['description']}")
                print(f"    Strategy: {opp['strategy']} (Conf: {opp['confidence']:.0%})")
        
        print("="*70)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scanner = OptionsScanner()
    scanner.refresh_quotes()
    
    analyzer = IVAnalyzer(scanner)
    analyzer.print_analysis('BTC', 86000)
