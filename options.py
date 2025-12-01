"""
Options Module for Delta Exchange Algo Trading Bot
Option chain scanner with delta-based selection and premium affordability checks
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config import get_config
from logger import get_logger
from product_discovery import (
    get_product_discovery, 
    Instrument, 
    InstrumentType, 
    OptionsChain
)
from risk_manager import get_risk_manager


class OptionSelectionCriteria(Enum):
    """Criteria for option selection"""
    DELTA = "delta"
    PREMIUM = "premium"
    LIQUIDITY = "liquidity"
    EXPIRY = "expiry"


@dataclass
class OptionCandidate:
    """A candidate option for trading"""
    instrument: Instrument
    score: float = 0.0
    
    # Selection criteria scores
    delta_score: float = 0.0
    premium_score: float = 0.0
    liquidity_score: float = 0.0
    expiry_score: float = 0.0
    
    # Computed values
    contracts: int = 0
    total_premium: float = 0.0
    max_loss: float = 0.0
    break_even: float = 0.0
    
    # Risk/reward
    risk_reward_ratio: float = 0.0
    expected_value: float = 0.0


@dataclass
class OptionTrade:
    """Option trade parameters"""
    instrument: Instrument
    direction: str  # 'call' or 'put'
    contracts: int
    premium: float
    strike: float
    expiry: datetime
    underlying_price: float
    
    # Greeks at entry
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0
    
    # Risk metrics
    max_loss: float = 0.0
    break_even: float = 0.0


class OptionsScanner:
    """
    Scans option chains and selects optimal options based on:
    - Delta targeting (0.30-0.45 for directional trades)
    - Premium affordability (max 0.5% of capital)
    - Liquidity requirements
    - Expiry window (3-30 days)
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.discovery = get_product_discovery()
        self.risk_mgr = get_risk_manager()
        
        # Selection parameters
        self.target_delta = getattr(self.config.risk, 'option_target_delta', 0.35)
        self.delta_tolerance = 0.15  # +/- from target
        self.min_delta = 0.20
        self.max_delta = 0.50
        
        # Expiry window
        self.min_expiry_days = 3
        self.max_expiry_days = 30
        
        # Premium limits (as % of capital)
        self.max_premium_pct = getattr(self.config.risk, 'option_max_premium_pct', 0.005)
        
        # Liquidity requirements
        self.min_volume_usd = getattr(self.config.risk, 'min_liquidity_usd', 1000)
        self.max_spread_pct = 0.05  # 5% max bid-ask spread
        
        self.logger.info(
            "Options scanner initialized",
            target_delta=self.target_delta,
            expiry_window=f"{self.min_expiry_days}-{self.max_expiry_days}d",
            max_premium_pct=self.max_premium_pct
        )
    
    def select_directional_option(
        self,
        underlying: str,
        direction: str,  # 'call' or 'put'
        target_delta: Optional[float] = None,
        expiry_window: Optional[Tuple[int, int]] = None,
        max_premium_pct: Optional[float] = None
    ) -> Optional[OptionTrade]:
        """
        Select optimal option for directional trade
        
        Args:
            underlying: Underlying asset (e.g., 'BTC', 'ETH')
            direction: 'call' for bullish, 'put' for bearish
            target_delta: Target delta (default: 0.35)
            expiry_window: (min_days, max_days) to expiry
            max_premium_pct: Max premium as % of capital
        
        Returns:
            OptionTrade parameters or None if no suitable option found
        """
        # Get option chain
        chain = self.discovery.get_options_chain(underlying, refresh=True)
        
        if not chain or not chain.options:
            self.logger.warning(f"No options chain found for {underlying}")
            return None
        
        # Use defaults
        tgt_delta = target_delta or self.target_delta
        min_days, max_days = expiry_window or (self.min_expiry_days, self.max_expiry_days)
        prem_pct = max_premium_pct or self.max_premium_pct
        
        # Get current capital for premium check
        account = self.risk_mgr.fetch_account_state()
        max_premium_spend = account.available_balance * prem_pct
        
        if max_premium_spend <= 0:
            self.logger.warning("No capital available for options")
            return None
        
        # Filter options by type
        if direction == 'call':
            options = chain.get_calls()
        else:
            options = chain.get_puts()
        
        if not options:
            self.logger.warning(f"No {direction} options found for {underlying}")
            return None
        
        # Filter by expiry
        options = [
            o for o in options
            if o.days_to_expiry is not None 
            and min_days <= o.days_to_expiry <= max_days
        ]
        
        if not options:
            self.logger.warning(f"No options in expiry window {min_days}-{max_days}d")
            return None
        
        # Score and rank options
        candidates = []
        for opt in options:
            candidate = self._score_option(
                opt, 
                chain.underlying_price, 
                tgt_delta, 
                max_premium_spend
            )
            if candidate:
                candidates.append(candidate)
        
        if not candidates:
            self.logger.warning("No suitable options found after filtering")
            return None
        
        # Sort by score (highest first)
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        # Select best candidate
        best = candidates[0]
        
        self.logger.info(
            f"Selected {direction} option",
            symbol=best.instrument.symbol,
            strike=best.instrument.strike_price,
            expiry=best.instrument.expiry,
            delta=best.instrument.delta,
            premium=best.instrument.ask,
            contracts=best.contracts,
            score=best.score
        )
        
        return OptionTrade(
            instrument=best.instrument,
            direction=direction,
            contracts=best.contracts,
            premium=best.instrument.ask or best.instrument.last_price,
            strike=best.instrument.strike_price or 0,
            expiry=best.instrument.expiry or datetime.utcnow(),
            underlying_price=chain.underlying_price,
            delta=best.instrument.delta or 0,
            gamma=best.instrument.gamma or 0,
            theta=best.instrument.theta or 0,
            vega=best.instrument.vega or 0,
            iv=best.instrument.implied_volatility or 0,
            max_loss=best.max_loss,
            break_even=best.break_even
        )
    
    def _score_option(
        self,
        option: Instrument,
        underlying_price: float,
        target_delta: float,
        max_premium_spend: float
    ) -> Optional[OptionCandidate]:
        """
        Score an option based on selection criteria
        
        Returns None if option doesn't meet minimum requirements
        """
        # Check if we have required data
        if not option.is_active:
            return None
        
        premium = option.ask or option.last_price or option.mark_price
        if premium <= 0:
            return None
        
        delta = abs(option.delta or 0)
        
        # Delta filter - reject if too far from target
        if delta < self.min_delta or delta > self.max_delta:
            return None
        
        # Premium affordability - reject if can't afford even 1 contract
        if premium > max_premium_spend:
            return None
        
        # Liquidity check
        if option.volume_24h < self.min_volume_usd:
            return None
        
        if option.spread_pct > self.max_spread_pct:
            return None
        
        # Calculate scores (0-1 scale)
        
        # Delta score: closer to target = higher score
        delta_diff = abs(delta - target_delta)
        delta_score = max(0, 1 - (delta_diff / self.delta_tolerance))
        
        # Premium score: cheaper = higher score (relative to max spend)
        premium_ratio = premium / max_premium_spend
        premium_score = max(0, 1 - premium_ratio)
        
        # Liquidity score: higher volume = higher score
        vol_ratio = min(option.volume_24h / (self.min_volume_usd * 10), 1)
        liquidity_score = vol_ratio
        
        # Expiry score: prefer middle of window (not too close, not too far)
        if option.days_to_expiry:
            mid_days = (self.min_expiry_days + self.max_expiry_days) / 2
            expiry_diff = abs(option.days_to_expiry - mid_days)
            max_diff = (self.max_expiry_days - self.min_expiry_days) / 2
            expiry_score = max(0, 1 - (expiry_diff / max_diff))
        else:
            expiry_score = 0.5
        
        # Weighted composite score
        score = (
            delta_score * 0.35 +
            premium_score * 0.25 +
            liquidity_score * 0.25 +
            expiry_score * 0.15
        )
        
        # Calculate contracts and risk metrics
        contracts, _ = self.risk_mgr.compute_option_contracts(
            option_premium=premium,
            max_premium_pct=self.max_premium_pct
        )
        
        total_premium = contracts * premium
        max_loss = total_premium  # Max loss for buying options
        
        # Break-even calculation
        strike = option.strike_price or 0
        if option.instrument_type == InstrumentType.CALL_OPTION:
            break_even = strike + premium
        else:
            break_even = strike - premium
        
        return OptionCandidate(
            instrument=option,
            score=score,
            delta_score=delta_score,
            premium_score=premium_score,
            liquidity_score=liquidity_score,
            expiry_score=expiry_score,
            contracts=contracts,
            total_premium=total_premium,
            max_loss=max_loss,
            break_even=break_even
        )
    
    def get_atm_options(
        self,
        underlying: str,
        direction: str = 'both'
    ) -> List[Instrument]:
        """
        Get at-the-money options for an underlying
        
        Args:
            underlying: Underlying asset
            direction: 'call', 'put', or 'both'
        
        Returns:
            List of ATM options
        """
        chain = self.discovery.get_options_chain(underlying)
        if not chain:
            return []
        
        atm_strike = chain.get_atm_strike()
        
        # Filter by direction
        if direction == 'call':
            options = chain.get_calls()
        elif direction == 'put':
            options = chain.get_puts()
        else:
            options = chain.options
        
        # Get options at or near ATM strike
        tolerance = atm_strike * 0.02  # 2% tolerance
        atm_options = [
            o for o in options
            if o.strike_price and abs(o.strike_price - atm_strike) <= tolerance
        ]
        
        return atm_options
    
    def get_options_by_delta(
        self,
        underlying: str,
        target_delta: float,
        direction: str,
        tolerance: float = 0.10
    ) -> List[Instrument]:
        """
        Get options within a delta range
        
        Args:
            underlying: Underlying asset
            target_delta: Target delta (0-1)
            direction: 'call' or 'put'
            tolerance: Delta tolerance
        
        Returns:
            List of options matching delta criteria
        """
        chain = self.discovery.get_options_chain(underlying, refresh=True)
        if not chain:
            return []
        
        return chain.get_by_delta(direction, target_delta, tolerance)
    
    def analyze_option_chain(self, underlying: str) -> Dict[str, Any]:
        """
        Analyze option chain for an underlying
        
        Returns summary statistics and metrics
        """
        chain = self.discovery.get_options_chain(underlying, refresh=True)
        if not chain:
            return {"error": f"No options chain for {underlying}"}
        
        calls = chain.get_calls()
        puts = chain.get_puts()
        
        # Calculate IV metrics
        call_ivs = [c.implied_volatility for c in calls if c.implied_volatility]
        put_ivs = [p.implied_volatility for p in puts if p.implied_volatility]
        
        avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else 0
        avg_put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else 0
        
        # Put-Call ratio
        call_oi = sum(c.open_interest for c in calls if c.open_interest)
        put_oi = sum(p.open_interest for p in puts if p.open_interest)
        pc_ratio = put_oi / call_oi if call_oi > 0 else 0
        
        # Volume analysis
        call_vol = sum(c.volume_24h for c in calls)
        put_vol = sum(p.volume_24h for p in puts)
        
        # Get expiries
        expiries = sorted(set(
            o.expiry for o in chain.options 
            if o.expiry is not None
        ))
        
        # Get strikes
        strikes = sorted(set(
            o.strike_price for o in chain.options 
            if o.strike_price is not None
        ))
        
        return {
            "underlying": underlying,
            "underlying_price": chain.underlying_price,
            "total_options": len(chain.options),
            "calls": len(calls),
            "puts": len(puts),
            "expiries": len(expiries),
            "nearest_expiry": expiries[0].isoformat() if expiries else None,
            "farthest_expiry": expiries[-1].isoformat() if expiries else None,
            "strikes": len(strikes),
            "atm_strike": chain.get_atm_strike(),
            "avg_call_iv": avg_call_iv,
            "avg_put_iv": avg_put_iv,
            "iv_skew": avg_put_iv - avg_call_iv,  # Positive = put skew
            "put_call_oi_ratio": pc_ratio,
            "call_volume": call_vol,
            "put_volume": put_vol,
            "volume_ratio": put_vol / call_vol if call_vol > 0 else 0
        }
    
    def should_trade_options(
        self,
        underlying: str,
        iv_threshold: float = 0.50
    ) -> Tuple[bool, str]:
        """
        Determine if options trading is favorable
        
        High IV often makes options expensive but good for selling
        Low IV makes options cheap and good for buying directional
        
        Args:
            underlying: Underlying asset
            iv_threshold: IV threshold for decision
        
        Returns:
            (should_trade, reason)
        """
        analysis = self.analyze_option_chain(underlying)
        
        if "error" in analysis:
            return False, analysis["error"]
        
        avg_iv = (analysis.get("avg_call_iv", 0) + analysis.get("avg_put_iv", 0)) / 2
        
        # Check liquidity
        if analysis["call_volume"] + analysis["put_volume"] < self.min_volume_usd * 2:
            return False, "Insufficient options volume"
        
        # Check IV
        if avg_iv > iv_threshold:
            return True, f"High IV ({avg_iv:.1%}) - consider selling options"
        elif avg_iv < iv_threshold * 0.5:
            return True, f"Low IV ({avg_iv:.1%}) - good for buying directional"
        else:
            return True, f"Moderate IV ({avg_iv:.1%}) - options trading feasible"


# Singleton instance
_options_scanner: Optional[OptionsScanner] = None


def get_options_scanner() -> OptionsScanner:
    """Get or create the global options scanner"""
    global _options_scanner
    if _options_scanner is None:
        _options_scanner = OptionsScanner()
    return _options_scanner


if __name__ == "__main__":
    # Test options module
    scanner = get_options_scanner()
    
    print("Testing Options Scanner")
    print("=" * 50)
    
    # Analyze BTC options
    analysis = scanner.analyze_option_chain("BTC")
    print(f"\nBTC Options Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Check if should trade options
    should_trade, reason = scanner.should_trade_options("BTC")
    print(f"\nShould trade BTC options: {should_trade}")
    print(f"Reason: {reason}")
    
    # Select a call option
    call_trade = scanner.select_directional_option("BTC", "call")
    if call_trade:
        print(f"\nSelected BTC Call:")
        print(f"  Symbol: {call_trade.instrument.symbol}")
        print(f"  Strike: {call_trade.strike}")
        print(f"  Premium: {call_trade.premium}")
        print(f"  Contracts: {call_trade.contracts}")
        print(f"  Delta: {call_trade.delta}")
        print(f"  Max Loss: {call_trade.max_loss}")
    else:
        print("\nNo suitable BTC call options found")
