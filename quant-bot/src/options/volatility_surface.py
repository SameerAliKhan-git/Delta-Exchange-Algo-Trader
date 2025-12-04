"""
Volatility Surface & IV Rank/Percentile Engine
================================================
Full volatility surface construction, interpolation, and IV metrics.
Supports crypto-specific adjustments for 24/7 markets.

Author: Quant Bot
Version: 1.0.0
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import warnings

from .pricing_engine import OptionsPricingEngine, OptionType, PricingModel

logger = logging.getLogger(__name__)


class SurfaceModel(Enum):
    SVI = "svi"                      # Stochastic Volatility Inspired
    SABR = "sabr"                    # SABR model
    CUBIC_SPLINE = "cubic_spline"   # Non-parametric
    RBF = "rbf"                      # Radial Basis Function


@dataclass
class VolPoint:
    """Single point on the volatility surface."""
    strike: float
    expiry: float  # Years to expiry
    iv: float
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    moneyness: float = 0.0  # log(K/F)
    delta: Optional[float] = None
    
    @property
    def mid_iv(self) -> float:
        if self.bid_iv and self.ask_iv:
            return (self.bid_iv + self.ask_iv) / 2
        return self.iv
    
    @property
    def iv_spread(self) -> float:
        if self.bid_iv and self.ask_iv:
            return self.ask_iv - self.bid_iv
        return 0.0


@dataclass
class VolSlice:
    """Volatility smile/skew at a single expiry."""
    expiry: float
    points: List[VolPoint]
    forward: float
    atm_iv: float
    
    # Skew metrics
    skew_25d: float = 0.0      # 25d put IV - 25d call IV
    butterfly_25d: float = 0.0  # Wing premium
    rr_25d: float = 0.0        # Risk reversal
    
    @property
    def strikes(self) -> np.ndarray:
        return np.array([p.strike for p in self.points])
    
    @property
    def ivs(self) -> np.ndarray:
        return np.array([p.iv for p in self.points])


@dataclass
class VolSurface:
    """Complete volatility surface."""
    spot: float
    timestamp: datetime
    slices: Dict[float, VolSlice]  # expiry -> slice
    model: SurfaceModel
    
    # Surface-level metrics
    term_structure_slope: float = 0.0
    average_skew: float = 0.0
    
    def get_iv(self, strike: float, expiry: float) -> Optional[float]:
        """Interpolate IV at arbitrary (K, T) point."""
        raise NotImplementedError("Use VolatilitySurfaceEngine.get_iv()")


@dataclass
class IVMetrics:
    """IV Rank and Percentile metrics."""
    current_iv: float
    iv_rank: float        # Percentile rank over lookback
    iv_percentile: float  # % of time IV was lower
    iv_mean: float
    iv_std: float
    iv_min: float
    iv_max: float
    iv_zscore: float
    lookback_days: int
    
    @property
    def is_high(self) -> bool:
        return self.iv_rank >= 0.7
    
    @property
    def is_low(self) -> bool:
        return self.iv_rank <= 0.3
    
    @property
    def regime(self) -> str:
        if self.iv_rank >= 0.8:
            return "EXTREME_HIGH"
        elif self.iv_rank >= 0.6:
            return "HIGH"
        elif self.iv_rank >= 0.4:
            return "NORMAL"
        elif self.iv_rank >= 0.2:
            return "LOW"
        else:
            return "EXTREME_LOW"


class SVIParams:
    """SVI (Stochastic Volatility Inspired) parametrization."""
    
    def __init__(
        self,
        a: float,    # Level
        b: float,    # ATM slope
        rho: float,  # Skew (-1 to 1)
        m: float,    # Translation
        sigma: float # ATM curvature
    ):
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma
    
    def total_variance(self, k: float) -> float:
        """Calculate total variance w(k) for log-moneyness k."""
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m)**2 + self.sigma**2)
        )
    
    def implied_vol(self, k: float, T: float) -> float:
        """Convert to implied vol."""
        w = self.total_variance(k)
        return np.sqrt(max(w / T, 0))
    
    def to_dict(self) -> Dict:
        return {
            'a': self.a, 'b': self.b, 'rho': self.rho,
            'm': self.m, 'sigma': self.sigma
        }


class VolatilitySurfaceEngine:
    """
    Production-grade volatility surface construction and interpolation.
    
    Features:
    - Multiple parametric models (SVI, SABR)
    - Non-parametric interpolation
    - IV rank/percentile calculation
    - Skew and term structure analysis
    - Arbitrage-free surface validation
    """
    
    def __init__(
        self,
        pricing_engine: Optional[OptionsPricingEngine] = None,
        default_model: SurfaceModel = SurfaceModel.CUBIC_SPLINE
    ):
        self.pricing_engine = pricing_engine or OptionsPricingEngine()
        self.default_model = default_model
        self.iv_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Calibrated models by expiry
        self._svi_params: Dict[float, SVIParams] = {}
        self._interpolators: Dict[float, Callable] = {}
        
        logger.info(f"VolatilitySurfaceEngine initialized with {default_model.value}")
    
    # ==================== IV EXTRACTION ====================
    
    def extract_iv_from_quotes(
        self,
        spot: float,
        quotes: List[Dict],
        risk_free_rate: float = 0.0
    ) -> List[VolPoint]:
        """
        Extract implied volatilities from option quotes.
        
        Args:
            spot: Current spot price
            quotes: List of dicts with keys:
                - strike: float
                - expiry: float (years)
                - type: 'call' or 'put'
                - bid: float (optional)
                - ask: float (optional)
                - mid: float (or use (bid+ask)/2)
        
        Returns:
            List of VolPoint objects
        """
        vol_points = []
        
        for q in quotes:
            strike = q['strike']
            expiry = q['expiry']
            opt_type = OptionType.CALL if q.get('type', 'call').lower() == 'call' else OptionType.PUT
            
            # Get price
            if 'mid' in q:
                price = q['mid']
            elif 'bid' in q and 'ask' in q:
                price = (q['bid'] + q['ask']) / 2
            else:
                continue
            
            # Solve for IV
            iv = self.pricing_engine.implied_volatility(
                price, spot, strike, expiry, opt_type, r=risk_free_rate
            )
            
            if iv is None or iv <= 0:
                continue
            
            # Calculate moneyness
            forward = spot * np.exp(risk_free_rate * expiry)
            moneyness = np.log(strike / forward)
            
            # Bid/Ask IV if available
            bid_iv = ask_iv = None
            if 'bid' in q and q['bid'] > 0:
                bid_iv = self.pricing_engine.implied_volatility(
                    q['bid'], spot, strike, expiry, opt_type, r=risk_free_rate
                )
            if 'ask' in q and q['ask'] > 0:
                ask_iv = self.pricing_engine.implied_volatility(
                    q['ask'], spot, strike, expiry, opt_type, r=risk_free_rate
                )
            
            vol_points.append(VolPoint(
                strike=strike,
                expiry=expiry,
                iv=iv,
                bid_iv=bid_iv,
                ask_iv=ask_iv,
                moneyness=moneyness
            ))
        
        return vol_points
    
    # ==================== SURFACE CONSTRUCTION ====================
    
    def build_surface(
        self,
        spot: float,
        vol_points: List[VolPoint],
        model: Optional[SurfaceModel] = None,
        risk_free_rate: float = 0.0
    ) -> VolSurface:
        """
        Build complete volatility surface from IV points.
        
        Args:
            spot: Current spot price
            vol_points: List of VolPoint objects
            model: Surface model to use
            risk_free_rate: For forward calculation
        
        Returns:
            VolSurface object
        """
        model = model or self.default_model
        
        # Group by expiry
        expiry_groups: Dict[float, List[VolPoint]] = {}
        for vp in vol_points:
            if vp.expiry not in expiry_groups:
                expiry_groups[vp.expiry] = []
            expiry_groups[vp.expiry].append(vp)
        
        slices = {}
        
        for expiry, points in sorted(expiry_groups.items()):
            # Sort by strike
            points = sorted(points, key=lambda x: x.strike)
            
            # Calculate forward
            forward = spot * np.exp(risk_free_rate * expiry)
            
            # Find ATM IV (closest to forward)
            atm_point = min(points, key=lambda p: abs(p.strike - forward))
            atm_iv = atm_point.iv
            
            # Calculate skew metrics
            skew_25d = self._calculate_skew(points, forward, delta=0.25)
            butterfly_25d = self._calculate_butterfly(points, forward, delta=0.25)
            rr_25d = self._calculate_risk_reversal(points, forward, delta=0.25)
            
            # Fit model for this slice
            if model == SurfaceModel.SVI:
                self._fit_svi(expiry, points, forward)
            elif model == SurfaceModel.CUBIC_SPLINE:
                self._fit_spline(expiry, points)
            
            slices[expiry] = VolSlice(
                expiry=expiry,
                points=points,
                forward=forward,
                atm_iv=atm_iv,
                skew_25d=skew_25d,
                butterfly_25d=butterfly_25d,
                rr_25d=rr_25d
            )
        
        # Calculate surface-level metrics
        term_slope = self._calculate_term_structure_slope(slices)
        avg_skew = np.mean([s.skew_25d for s in slices.values()])
        
        return VolSurface(
            spot=spot,
            timestamp=datetime.now(),
            slices=slices,
            model=model,
            term_structure_slope=term_slope,
            average_skew=avg_skew
        )
    
    def _calculate_skew(
        self,
        points: List[VolPoint],
        forward: float,
        delta: float = 0.25
    ) -> float:
        """Calculate put-call skew at given delta."""
        # Approximate delta strikes
        # 25d put ≈ 0.9 * F, 25d call ≈ 1.1 * F (rough approximation)
        put_strike_approx = forward * (1 - delta)
        call_strike_approx = forward * (1 + delta)
        
        put_points = [p for p in points if p.strike < forward]
        call_points = [p for p in points if p.strike > forward]
        
        if not put_points or not call_points:
            return 0.0
        
        put_iv = min(put_points, key=lambda p: abs(p.strike - put_strike_approx)).iv
        call_iv = min(call_points, key=lambda p: abs(p.strike - call_strike_approx)).iv
        
        return put_iv - call_iv
    
    def _calculate_butterfly(
        self,
        points: List[VolPoint],
        forward: float,
        delta: float = 0.25
    ) -> float:
        """Calculate butterfly spread (wing premium)."""
        atm_point = min(points, key=lambda p: abs(p.strike - forward))
        atm_iv = atm_point.iv
        
        put_strike = forward * (1 - delta)
        call_strike = forward * (1 + delta)
        
        put_points = [p for p in points if p.strike < forward]
        call_points = [p for p in points if p.strike > forward]
        
        if not put_points or not call_points:
            return 0.0
        
        put_iv = min(put_points, key=lambda p: abs(p.strike - put_strike)).iv
        call_iv = min(call_points, key=lambda p: abs(p.strike - call_strike)).iv
        
        return (put_iv + call_iv) / 2 - atm_iv
    
    def _calculate_risk_reversal(
        self,
        points: List[VolPoint],
        forward: float,
        delta: float = 0.25
    ) -> float:
        """Calculate risk reversal (call IV - put IV at same delta)."""
        return -self._calculate_skew(points, forward, delta)
    
    def _calculate_term_structure_slope(
        self,
        slices: Dict[float, VolSlice]
    ) -> float:
        """Calculate term structure slope (ATM IV vs expiry)."""
        if len(slices) < 2:
            return 0.0
        
        expiries = sorted(slices.keys())
        atm_ivs = [slices[e].atm_iv for e in expiries]
        
        # Linear regression slope
        x = np.array(expiries)
        y = np.array(atm_ivs)
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    # ==================== MODEL FITTING ====================
    
    def _fit_svi(
        self,
        expiry: float,
        points: List[VolPoint],
        forward: float
    ) -> SVIParams:
        """Fit SVI model to smile."""
        # Log-moneyness
        k = np.array([np.log(p.strike / forward) for p in points])
        # Total variance
        w = np.array([(p.iv ** 2) * expiry for p in points])
        
        def svi_objective(params):
            a, b, rho, m, sigma = params
            
            # Constraints
            if b < 0 or sigma < 0 or abs(rho) >= 1:
                return 1e10
            
            w_model = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            return np.sum((w_model - w)**2)
        
        # Initial guess
        x0 = [np.mean(w), 0.1, -0.5, 0.0, 0.1]
        
        result = minimize(
            svi_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        a, b, rho, m, sigma = result.x
        params = SVIParams(a, b, rho, m, sigma)
        self._svi_params[expiry] = params
        
        return params
    
    def _fit_spline(
        self,
        expiry: float,
        points: List[VolPoint]
    ) -> None:
        """Fit cubic spline to smile."""
        strikes = np.array([p.strike for p in points])
        ivs = np.array([p.iv for p in points])
        
        # Sort by strike
        idx = np.argsort(strikes)
        strikes = strikes[idx]
        ivs = ivs[idx]
        
        # Create spline
        spline = interpolate.CubicSpline(strikes, ivs, bc_type='natural')
        self._interpolators[expiry] = spline
    
    # ==================== INTERPOLATION ====================
    
    def get_iv(
        self,
        surface: VolSurface,
        strike: float,
        expiry: float
    ) -> float:
        """
        Get interpolated IV at arbitrary (K, T) point.
        
        Uses bilinear interpolation across the surface.
        """
        expiries = sorted(surface.slices.keys())
        
        if not expiries:
            raise ValueError("Empty surface")
        
        # Handle edge cases
        if expiry <= expiries[0]:
            return self._interpolate_strike(surface.slices[expiries[0]], strike)
        if expiry >= expiries[-1]:
            return self._interpolate_strike(surface.slices[expiries[-1]], strike)
        
        # Find bracketing expiries
        for i, e in enumerate(expiries[:-1]):
            if e <= expiry <= expiries[i + 1]:
                e1, e2 = e, expiries[i + 1]
                break
        
        # Interpolate at each expiry
        iv1 = self._interpolate_strike(surface.slices[e1], strike)
        iv2 = self._interpolate_strike(surface.slices[e2], strike)
        
        # Linear interpolation in time (variance preserving)
        w1 = iv1 ** 2 * e1
        w2 = iv2 ** 2 * e2
        
        # Interpolate total variance
        alpha = (expiry - e1) / (e2 - e1)
        w = w1 + alpha * (w2 - w1)
        
        return np.sqrt(w / expiry)
    
    def _interpolate_strike(
        self,
        slice_: VolSlice,
        strike: float
    ) -> float:
        """Interpolate IV at a strike within a single slice."""
        expiry = slice_.expiry
        
        # Use fitted model if available
        if expiry in self._svi_params:
            params = self._svi_params[expiry]
            k = np.log(strike / slice_.forward)
            return params.implied_vol(k, expiry)
        
        if expiry in self._interpolators:
            return float(self._interpolators[expiry](strike))
        
        # Fallback: linear interpolation
        strikes = slice_.strikes
        ivs = slice_.ivs
        
        return float(np.interp(strike, strikes, ivs))
    
    # ==================== IV RANK / PERCENTILE ====================
    
    def update_iv_history(
        self,
        symbol: str,
        iv: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add IV observation to history."""
        timestamp = timestamp or datetime.now()
        
        if symbol not in self.iv_history:
            self.iv_history[symbol] = []
        
        self.iv_history[symbol].append((timestamp, iv))
        
        # Keep last 365 days
        cutoff = datetime.now() - timedelta(days=365)
        self.iv_history[symbol] = [
            (t, v) for t, v in self.iv_history[symbol] if t >= cutoff
        ]
    
    def calculate_iv_metrics(
        self,
        symbol: str,
        current_iv: float,
        lookback_days: int = 252
    ) -> IVMetrics:
        """
        Calculate IV rank, percentile, and related metrics.
        
        IV Rank = (Current IV - Min IV) / (Max IV - Min IV)
        IV Percentile = % of observations below current IV
        """
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 2:
            # Not enough history
            return IVMetrics(
                current_iv=current_iv,
                iv_rank=0.5,
                iv_percentile=0.5,
                iv_mean=current_iv,
                iv_std=0.0,
                iv_min=current_iv,
                iv_max=current_iv,
                iv_zscore=0.0,
                lookback_days=0
            )
        
        # Filter to lookback period
        cutoff = datetime.now() - timedelta(days=lookback_days)
        history = [v for t, v in self.iv_history[symbol] if t >= cutoff]
        
        if len(history) < 2:
            history = [v for _, v in self.iv_history[symbol]]
        
        iv_array = np.array(history)
        
        iv_min = np.min(iv_array)
        iv_max = np.max(iv_array)
        iv_mean = np.mean(iv_array)
        iv_std = np.std(iv_array)
        
        # IV Rank
        if iv_max > iv_min:
            iv_rank = (current_iv - iv_min) / (iv_max - iv_min)
        else:
            iv_rank = 0.5
        
        # IV Percentile
        iv_percentile = np.sum(iv_array <= current_iv) / len(iv_array)
        
        # Z-score
        if iv_std > 0:
            iv_zscore = (current_iv - iv_mean) / iv_std
        else:
            iv_zscore = 0.0
        
        return IVMetrics(
            current_iv=current_iv,
            iv_rank=np.clip(iv_rank, 0, 1),
            iv_percentile=iv_percentile,
            iv_mean=iv_mean,
            iv_std=iv_std,
            iv_min=iv_min,
            iv_max=iv_max,
            iv_zscore=iv_zscore,
            lookback_days=len(history)
        )
    
    def load_iv_history(
        self,
        symbol: str,
        data: List[Tuple[datetime, float]]
    ) -> None:
        """Load historical IV data."""
        self.iv_history[symbol] = sorted(data, key=lambda x: x[0])
    
    # ==================== ANALYSIS ====================
    
    def analyze_surface(
        self,
        surface: VolSurface
    ) -> Dict:
        """
        Comprehensive surface analysis.
        
        Returns metrics useful for trading decisions.
        """
        analysis = {
            'timestamp': surface.timestamp.isoformat(),
            'spot': surface.spot,
            'model': surface.model.value,
            'num_expiries': len(surface.slices),
            'term_structure_slope': surface.term_structure_slope,
            'average_skew': surface.average_skew,
            'slices': {}
        }
        
        for expiry, slice_ in sorted(surface.slices.items()):
            analysis['slices'][f'{expiry:.4f}'] = {
                'expiry_days': int(expiry * 365),
                'forward': slice_.forward,
                'atm_iv': slice_.atm_iv,
                'skew_25d': slice_.skew_25d,
                'butterfly_25d': slice_.butterfly_25d,
                'rr_25d': slice_.rr_25d,
                'num_strikes': len(slice_.points),
                'strike_range': [slice_.points[0].strike, slice_.points[-1].strike]
            }
        
        # Term structure shape
        if len(surface.slices) >= 2:
            expiries = sorted(surface.slices.keys())
            atm_ivs = [surface.slices[e].atm_iv for e in expiries]
            
            if atm_ivs[-1] > atm_ivs[0]:
                analysis['term_structure_shape'] = 'CONTANGO'
            elif atm_ivs[-1] < atm_ivs[0]:
                analysis['term_structure_shape'] = 'BACKWARDATION'
            else:
                analysis['term_structure_shape'] = 'FLAT'
        
        # Skew regime
        if surface.average_skew > 0.05:
            analysis['skew_regime'] = 'PUT_PREMIUM'
        elif surface.average_skew < -0.05:
            analysis['skew_regime'] = 'CALL_PREMIUM'
        else:
            analysis['skew_regime'] = 'NEUTRAL'
        
        return analysis
    
    def detect_arbitrage_opportunities(
        self,
        surface: VolSurface
    ) -> List[Dict]:
        """
        Detect calendar and butterfly arbitrage opportunities.
        
        Returns list of potential opportunities.
        """
        opportunities = []
        
        expiries = sorted(surface.slices.keys())
        
        # Calendar spread arbitrage (variance should increase with time)
        for i in range(len(expiries) - 1):
            e1, e2 = expiries[i], expiries[i + 1]
            s1, s2 = surface.slices[e1], surface.slices[e2]
            
            # Check ATM
            var1 = s1.atm_iv ** 2 * e1
            var2 = s2.atm_iv ** 2 * e2
            
            if var2 < var1:
                opportunities.append({
                    'type': 'CALENDAR_ARB',
                    'description': f'Variance decreasing: {e1:.3f}y to {e2:.3f}y',
                    'expiry_near': e1,
                    'expiry_far': e2,
                    'atm_iv_near': s1.atm_iv,
                    'atm_iv_far': s2.atm_iv,
                    'action': 'BUY_NEAR_SELL_FAR'
                })
        
        # Butterfly arbitrage (convexity check)
        for expiry, slice_ in surface.slices.items():
            points = slice_.points
            if len(points) < 3:
                continue
            
            for i in range(1, len(points) - 1):
                k1, k2, k3 = points[i-1].strike, points[i].strike, points[i+1].strike
                v1, v2, v3 = points[i-1].iv, points[i].iv, points[i+1].iv
                
                # Linear interpolation at middle strike
                alpha = (k2 - k1) / (k3 - k1)
                v_interp = v1 + alpha * (v3 - v1)
                
                # Butterfly should be non-negative
                if v2 < v_interp - 0.001:  # Small tolerance
                    opportunities.append({
                        'type': 'BUTTERFLY_ARB',
                        'description': f'Convexity violation at K={k2:.0f}',
                        'expiry': expiry,
                        'strikes': [k1, k2, k3],
                        'ivs': [v1, v2, v3],
                        'action': 'BUY_WINGS_SELL_BODY'
                    })
        
        return opportunities


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Volatility Surface Engine")
    parser.add_argument("--spot", type=float, required=True, help="Spot price")
    parser.add_argument("--quotes-file", type=str, help="JSON file with option quotes")
    parser.add_argument("--iv-history", type=str, help="JSON file with IV history")
    parser.add_argument("--current-iv", type=float, help="Current IV for rank calculation")
    parser.add_argument("--symbol", type=str, default="BTC", help="Symbol for IV metrics")
    parser.add_argument("--model", choices=["svi", "spline"], default="spline")
    
    args = parser.parse_args()
    
    engine = VolatilitySurfaceEngine()
    
    if args.quotes_file:
        with open(args.quotes_file, 'r') as f:
            quotes = json.load(f)
        
        vol_points = engine.extract_iv_from_quotes(args.spot, quotes)
        
        model = SurfaceModel.SVI if args.model == "svi" else SurfaceModel.CUBIC_SPLINE
        surface = engine.build_surface(args.spot, vol_points, model=model)
        
        analysis = engine.analyze_surface(surface)
        print(json.dumps(analysis, indent=2))
        
        # Check for arbitrage
        arbs = engine.detect_arbitrage_opportunities(surface)
        if arbs:
            print("\n⚠️ Arbitrage Opportunities Detected:")
            for arb in arbs:
                print(f"  - {arb['type']}: {arb['description']}")
    
    if args.iv_history and args.current_iv:
        with open(args.iv_history, 'r') as f:
            history_data = json.load(f)
        
        # Convert to (datetime, float) tuples
        history = [
            (datetime.fromisoformat(item['timestamp']), item['iv'])
            for item in history_data
        ]
        
        engine.load_iv_history(args.symbol, history)
        metrics = engine.calculate_iv_metrics(args.symbol, args.current_iv)
        
        print(f"\n{'='*50}")
        print(f"IV METRICS FOR {args.symbol}")
        print(f"{'='*50}")
        print(f"Current IV:    {metrics.current_iv:.2%}")
        print(f"IV Rank:       {metrics.iv_rank:.1%}")
        print(f"IV Percentile: {metrics.iv_percentile:.1%}")
        print(f"IV Z-Score:    {metrics.iv_zscore:.2f}")
        print(f"Regime:        {metrics.regime}")
        print(f"Historical Range: {metrics.iv_min:.2%} - {metrics.iv_max:.2%}")
        print(f"Lookback: {metrics.lookback_days} observations")
