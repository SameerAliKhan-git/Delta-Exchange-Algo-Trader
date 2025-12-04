"""
Options Strategy Module - Production-Grade Implementation
==========================================================
Complete options strategy library including spreads, condors,
delta hedging, and gamma scalping.

Author: Quant Bot
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging

from .pricing_engine import (
    OptionsPricingEngine, OptionType, OptionPrice, Greeks, PricingModel
)

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    # Single leg
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    
    # Income strategies
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    
    # Spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    
    # Volatility strategies
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    
    # Complex
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    
    # Hedging
    PROTECTIVE_PUT = "protective_put"
    COLLAR = "collar"
    
    # Advanced
    GAMMA_SCALP = "gamma_scalp"
    DELTA_HEDGE = "delta_hedge"


@dataclass
class OptionLeg:
    """Single option leg in a strategy."""
    strike: float
    expiry: float  # Years to expiry
    option_type: OptionType
    quantity: int  # Positive = long, negative = short
    premium: Optional[float] = None
    iv: Optional[float] = None
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class SpotLeg:
    """Spot/futures leg for delta hedging."""
    quantity: float  # Positive = long, negative = short
    entry_price: float


@dataclass
class StrategyPosition:
    """Complete strategy position."""
    strategy_type: StrategyType
    symbol: str
    option_legs: List[OptionLeg]
    spot_leg: Optional[SpotLeg] = None
    
    # Calculated fields
    net_premium: float = 0.0  # Positive = paid, negative = received
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_points: List[float] = field(default_factory=list)
    net_greeks: Greeks = field(default_factory=Greeks)
    
    # Risk metrics
    margin_required: float = 0.0
    probability_of_profit: float = 0.0
    expected_value: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyAnalysis:
    """Analysis results for a strategy."""
    position: StrategyPosition
    current_value: float
    pnl: float
    pnl_percent: float
    days_to_expiry: float
    
    # Greeks
    delta: float
    gamma: float
    theta: float  # Per day
    vega: float
    
    # Risk analysis
    delta_exposure_usd: float
    gamma_exposure_usd: float
    theta_decay_usd: float
    
    # Scenario analysis
    price_scenarios: Dict[float, float]  # spot_price -> pnl
    vol_scenarios: Dict[float, float]    # vol_change -> pnl


class OptionsStrategyEngine:
    """
    Production-grade options strategy construction and management.
    
    Supports:
    - All major strategy types
    - Greeks aggregation
    - Risk analysis
    - Delta hedging
    - Gamma scalping
    """
    
    def __init__(
        self,
        pricing_engine: Optional[OptionsPricingEngine] = None
    ):
        self.pricing_engine = pricing_engine or OptionsPricingEngine()
        self.positions: Dict[str, StrategyPosition] = {}
        
        logger.info("OptionsStrategyEngine initialized")
    
    # ==================== STRATEGY BUILDERS ====================
    
    def create_covered_call(
        self,
        symbol: str,
        spot: float,
        strike: float,
        expiry: float,
        iv: float,
        spot_quantity: float = 1.0
    ) -> StrategyPosition:
        """
        Covered Call: Long spot + Short call.
        Income strategy with limited upside.
        """
        call_price = self.pricing_engine.black_scholes_price(
            spot, strike, expiry, iv, OptionType.CALL
        )
        
        option_leg = OptionLeg(
            strike=strike,
            expiry=expiry,
            option_type=OptionType.CALL,
            quantity=-int(spot_quantity),  # Short calls
            premium=call_price,
            iv=iv
        )
        
        spot_leg = SpotLeg(
            quantity=spot_quantity,
            entry_price=spot
        )
        
        position = StrategyPosition(
            strategy_type=StrategyType.COVERED_CALL,
            symbol=symbol,
            option_legs=[option_leg],
            spot_leg=spot_leg,
            net_premium=-call_price * abs(option_leg.quantity),  # Premium received
        )
        
        # Calculate risk/reward
        position.max_profit = (strike - spot + call_price) * spot_quantity
        position.max_loss = (spot - call_price) * spot_quantity  # If spot goes to 0
        position.breakeven_points = [spot - call_price]
        
        self._calculate_greeks(position, spot, iv)
        
        return position
    
    def create_protective_put(
        self,
        symbol: str,
        spot: float,
        strike: float,
        expiry: float,
        iv: float,
        spot_quantity: float = 1.0
    ) -> StrategyPosition:
        """
        Protective Put: Long spot + Long put.
        Insurance against downside.
        """
        put_price = self.pricing_engine.black_scholes_price(
            spot, strike, expiry, iv, OptionType.PUT
        )
        
        option_leg = OptionLeg(
            strike=strike,
            expiry=expiry,
            option_type=OptionType.PUT,
            quantity=int(spot_quantity),  # Long puts
            premium=put_price,
            iv=iv
        )
        
        spot_leg = SpotLeg(
            quantity=spot_quantity,
            entry_price=spot
        )
        
        position = StrategyPosition(
            strategy_type=StrategyType.PROTECTIVE_PUT,
            symbol=symbol,
            option_legs=[option_leg],
            spot_leg=spot_leg,
            net_premium=put_price * abs(option_leg.quantity),  # Premium paid
        )
        
        position.max_profit = float('inf')
        position.max_loss = (spot - strike + put_price) * spot_quantity
        position.breakeven_points = [spot + put_price]
        
        self._calculate_greeks(position, spot, iv)
        
        return position
    
    def create_bull_call_spread(
        self,
        symbol: str,
        spot: float,
        lower_strike: float,
        upper_strike: float,
        expiry: float,
        iv: float,
        quantity: int = 1
    ) -> StrategyPosition:
        """
        Bull Call Spread: Long lower strike call + Short higher strike call.
        Bullish with limited risk/reward.
        """
        lower_call_price = self.pricing_engine.black_scholes_price(
            spot, lower_strike, expiry, iv, OptionType.CALL
        )
        upper_call_price = self.pricing_engine.black_scholes_price(
            spot, upper_strike, expiry, iv, OptionType.CALL
        )
        
        long_leg = OptionLeg(
            strike=lower_strike,
            expiry=expiry,
            option_type=OptionType.CALL,
            quantity=quantity,
            premium=lower_call_price,
            iv=iv
        )
        
        short_leg = OptionLeg(
            strike=upper_strike,
            expiry=expiry,
            option_type=OptionType.CALL,
            quantity=-quantity,
            premium=upper_call_price,
            iv=iv
        )
        
        net_debit = (lower_call_price - upper_call_price) * quantity
        
        position = StrategyPosition(
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            symbol=symbol,
            option_legs=[long_leg, short_leg],
            net_premium=net_debit,
        )
        
        position.max_profit = (upper_strike - lower_strike - net_debit / quantity) * quantity
        position.max_loss = net_debit
        position.breakeven_points = [lower_strike + net_debit / quantity]
        
        self._calculate_greeks(position, spot, iv)
        
        return position
    
    def create_bear_put_spread(
        self,
        symbol: str,
        spot: float,
        upper_strike: float,
        lower_strike: float,
        expiry: float,
        iv: float,
        quantity: int = 1
    ) -> StrategyPosition:
        """
        Bear Put Spread: Long higher strike put + Short lower strike put.
        Bearish with limited risk/reward.
        """
        upper_put_price = self.pricing_engine.black_scholes_price(
            spot, upper_strike, expiry, iv, OptionType.PUT
        )
        lower_put_price = self.pricing_engine.black_scholes_price(
            spot, lower_strike, expiry, iv, OptionType.PUT
        )
        
        long_leg = OptionLeg(
            strike=upper_strike,
            expiry=expiry,
            option_type=OptionType.PUT,
            quantity=quantity,
            premium=upper_put_price,
            iv=iv
        )
        
        short_leg = OptionLeg(
            strike=lower_strike,
            expiry=expiry,
            option_type=OptionType.PUT,
            quantity=-quantity,
            premium=lower_put_price,
            iv=iv
        )
        
        net_debit = (upper_put_price - lower_put_price) * quantity
        
        position = StrategyPosition(
            strategy_type=StrategyType.BEAR_PUT_SPREAD,
            symbol=symbol,
            option_legs=[long_leg, short_leg],
            net_premium=net_debit,
        )
        
        position.max_profit = (upper_strike - lower_strike - net_debit / quantity) * quantity
        position.max_loss = net_debit
        position.breakeven_points = [upper_strike - net_debit / quantity]
        
        self._calculate_greeks(position, spot, iv)
        
        return position
    
    def create_straddle(
        self,
        symbol: str,
        spot: float,
        strike: float,
        expiry: float,
        iv: float,
        quantity: int = 1,
        is_long: bool = True
    ) -> StrategyPosition:
        """
        Straddle: Call + Put at same strike.
        Long = volatility play, Short = income/range bound.
        """
        call_price = self.pricing_engine.black_scholes_price(
            spot, strike, expiry, iv, OptionType.CALL
        )
        put_price = self.pricing_engine.black_scholes_price(
            spot, strike, expiry, iv, OptionType.PUT
        )
        
        qty = quantity if is_long else -quantity
        
        call_leg = OptionLeg(
            strike=strike,
            expiry=expiry,
            option_type=OptionType.CALL,
            quantity=qty,
            premium=call_price,
            iv=iv
        )
        
        put_leg = OptionLeg(
            strike=strike,
            expiry=expiry,
            option_type=OptionType.PUT,
            quantity=qty,
            premium=put_price,
            iv=iv
        )
        
        total_premium = (call_price + put_price) * abs(quantity)
        
        position = StrategyPosition(
            strategy_type=StrategyType.LONG_STRADDLE if is_long else StrategyType.SHORT_STRADDLE,
            symbol=symbol,
            option_legs=[call_leg, put_leg],
            net_premium=total_premium if is_long else -total_premium,
        )
        
        if is_long:
            position.max_profit = float('inf')
            position.max_loss = total_premium
        else:
            position.max_profit = total_premium
            position.max_loss = float('inf')
        
        breakeven_lower = strike - (call_price + put_price)
        breakeven_upper = strike + (call_price + put_price)
        position.breakeven_points = [breakeven_lower, breakeven_upper]
        
        self._calculate_greeks(position, spot, iv)
        
        return position
    
    def create_strangle(
        self,
        symbol: str,
        spot: float,
        put_strike: float,
        call_strike: float,
        expiry: float,
        iv: float,
        quantity: int = 1,
        is_long: bool = True
    ) -> StrategyPosition:
        """
        Strangle: OTM Call + OTM Put.
        Cheaper than straddle, needs bigger move.
        """
        call_price = self.pricing_engine.black_scholes_price(
            spot, call_strike, expiry, iv, OptionType.CALL
        )
        put_price = self.pricing_engine.black_scholes_price(
            spot, put_strike, expiry, iv, OptionType.PUT
        )
        
        qty = quantity if is_long else -quantity
        
        call_leg = OptionLeg(
            strike=call_strike,
            expiry=expiry,
            option_type=OptionType.CALL,
            quantity=qty,
            premium=call_price,
            iv=iv
        )
        
        put_leg = OptionLeg(
            strike=put_strike,
            expiry=expiry,
            option_type=OptionType.PUT,
            quantity=qty,
            premium=put_price,
            iv=iv
        )
        
        total_premium = (call_price + put_price) * abs(quantity)
        
        position = StrategyPosition(
            strategy_type=StrategyType.LONG_STRANGLE if is_long else StrategyType.SHORT_STRANGLE,
            symbol=symbol,
            option_legs=[call_leg, put_leg],
            net_premium=total_premium if is_long else -total_premium,
        )
        
        if is_long:
            position.max_profit = float('inf')
            position.max_loss = total_premium
        else:
            position.max_profit = total_premium
            position.max_loss = float('inf')
        
        position.breakeven_points = [
            put_strike - (call_price + put_price),
            call_strike + (call_price + put_price)
        ]
        
        self._calculate_greeks(position, spot, iv)
        
        return position
    
    def create_iron_condor(
        self,
        symbol: str,
        spot: float,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        expiry: float,
        iv: float,
        quantity: int = 1
    ) -> StrategyPosition:
        """
        Iron Condor: Bull put spread + Bear call spread.
        Profit from low volatility / range-bound market.
        """
        # Bull put spread (credit)
        put_long_price = self.pricing_engine.black_scholes_price(
            spot, put_long_strike, expiry, iv, OptionType.PUT
        )
        put_short_price = self.pricing_engine.black_scholes_price(
            spot, put_short_strike, expiry, iv, OptionType.PUT
        )
        
        # Bear call spread (credit)
        call_short_price = self.pricing_engine.black_scholes_price(
            spot, call_short_strike, expiry, iv, OptionType.CALL
        )
        call_long_price = self.pricing_engine.black_scholes_price(
            spot, call_long_strike, expiry, iv, OptionType.CALL
        )
        
        legs = [
            OptionLeg(put_long_strike, expiry, OptionType.PUT, quantity, put_long_price, iv),
            OptionLeg(put_short_strike, expiry, OptionType.PUT, -quantity, put_short_price, iv),
            OptionLeg(call_short_strike, expiry, OptionType.CALL, -quantity, call_short_price, iv),
            OptionLeg(call_long_strike, expiry, OptionType.CALL, quantity, call_long_price, iv),
        ]
        
        net_credit = (
            put_short_price - put_long_price + 
            call_short_price - call_long_price
        ) * quantity
        
        position = StrategyPosition(
            strategy_type=StrategyType.IRON_CONDOR,
            symbol=symbol,
            option_legs=legs,
            net_premium=-net_credit,  # Credit received
        )
        
        # Max profit = net credit
        position.max_profit = net_credit
        
        # Max loss = width of either spread - net credit
        put_spread_width = put_short_strike - put_long_strike
        call_spread_width = call_long_strike - call_short_strike
        max_spread_width = max(put_spread_width, call_spread_width)
        position.max_loss = (max_spread_width - net_credit / quantity) * quantity
        
        position.breakeven_points = [
            put_short_strike - net_credit / quantity,
            call_short_strike + net_credit / quantity
        ]
        
        self._calculate_greeks(position, spot, iv)
        
        return position
    
    def create_iron_butterfly(
        self,
        symbol: str,
        spot: float,
        lower_strike: float,
        middle_strike: float,
        upper_strike: float,
        expiry: float,
        iv: float,
        quantity: int = 1
    ) -> StrategyPosition:
        """
        Iron Butterfly: Short straddle + Long strangle wings.
        Max profit at middle strike, defined risk.
        """
        put_long_price = self.pricing_engine.black_scholes_price(
            spot, lower_strike, expiry, iv, OptionType.PUT
        )
        put_short_price = self.pricing_engine.black_scholes_price(
            spot, middle_strike, expiry, iv, OptionType.PUT
        )
        call_short_price = self.pricing_engine.black_scholes_price(
            spot, middle_strike, expiry, iv, OptionType.CALL
        )
        call_long_price = self.pricing_engine.black_scholes_price(
            spot, upper_strike, expiry, iv, OptionType.CALL
        )
        
        legs = [
            OptionLeg(lower_strike, expiry, OptionType.PUT, quantity, put_long_price, iv),
            OptionLeg(middle_strike, expiry, OptionType.PUT, -quantity, put_short_price, iv),
            OptionLeg(middle_strike, expiry, OptionType.CALL, -quantity, call_short_price, iv),
            OptionLeg(upper_strike, expiry, OptionType.CALL, quantity, call_long_price, iv),
        ]
        
        net_credit = (
            put_short_price - put_long_price +
            call_short_price - call_long_price
        ) * quantity
        
        position = StrategyPosition(
            strategy_type=StrategyType.IRON_BUTTERFLY,
            symbol=symbol,
            option_legs=legs,
            net_premium=-net_credit,
        )
        
        position.max_profit = net_credit
        wing_width = middle_strike - lower_strike
        position.max_loss = (wing_width - net_credit / quantity) * quantity
        position.breakeven_points = [
            middle_strike - net_credit / quantity,
            middle_strike + net_credit / quantity
        ]
        
        self._calculate_greeks(position, spot, iv)
        
        return position
    
    def create_calendar_spread(
        self,
        symbol: str,
        spot: float,
        strike: float,
        near_expiry: float,
        far_expiry: float,
        iv_near: float,
        iv_far: float,
        option_type: OptionType = OptionType.CALL,
        quantity: int = 1
    ) -> StrategyPosition:
        """
        Calendar Spread: Short near-term, long far-term at same strike.
        Profits from time decay differential and vol changes.
        """
        near_price = self.pricing_engine.black_scholes_price(
            spot, strike, near_expiry, iv_near, option_type
        )
        far_price = self.pricing_engine.black_scholes_price(
            spot, strike, far_expiry, iv_far, option_type
        )
        
        near_leg = OptionLeg(
            strike=strike,
            expiry=near_expiry,
            option_type=option_type,
            quantity=-quantity,  # Short near
            premium=near_price,
            iv=iv_near
        )
        
        far_leg = OptionLeg(
            strike=strike,
            expiry=far_expiry,
            option_type=option_type,
            quantity=quantity,  # Long far
            premium=far_price,
            iv=iv_far
        )
        
        net_debit = (far_price - near_price) * quantity
        
        position = StrategyPosition(
            strategy_type=StrategyType.CALENDAR_SPREAD,
            symbol=symbol,
            option_legs=[near_leg, far_leg],
            net_premium=net_debit,
        )
        
        # Calendar max profit is complex, occurs when spot = strike at near expiry
        position.max_loss = net_debit  # If both expire worthless
        position.breakeven_points = []  # Complex, depends on vol
        
        self._calculate_greeks(position, spot, iv_far)
        
        return position
    
    # ==================== GREEKS CALCULATION ====================
    
    def _calculate_greeks(
        self,
        position: StrategyPosition,
        spot: float,
        sigma: float
    ) -> None:
        """Calculate aggregated Greeks for the position."""
        total_greeks = Greeks()
        
        for leg in position.option_legs:
            leg_greeks = self.pricing_engine.black_scholes_greeks(
                spot, leg.strike, leg.expiry, leg.iv or sigma, leg.option_type
            )
            total_greeks = total_greeks + leg_greeks.scale(leg.quantity)
        
        # Add spot leg delta if present
        if position.spot_leg:
            total_greeks.delta += position.spot_leg.quantity
        
        position.net_greeks = total_greeks
    
    # ==================== ANALYSIS ====================
    
    def analyze_position(
        self,
        position: StrategyPosition,
        current_spot: float,
        current_iv: float,
        current_time: Optional[float] = None
    ) -> StrategyAnalysis:
        """
        Analyze current state of a position.
        
        Args:
            position: Strategy position to analyze
            current_spot: Current spot price
            current_iv: Current implied volatility
            current_time: Current time to expiry (None = use original)
        """
        # Calculate current value
        current_value = 0.0
        
        for leg in position.option_legs:
            T = current_time if current_time is not None else leg.expiry
            if T <= 0:
                # At expiry
                if leg.option_type == OptionType.CALL:
                    intrinsic = max(current_spot - leg.strike, 0)
                else:
                    intrinsic = max(leg.strike - current_spot, 0)
                current_value += intrinsic * leg.quantity
            else:
                price = self.pricing_engine.black_scholes_price(
                    current_spot, leg.strike, T, current_iv, leg.option_type
                )
                current_value += price * leg.quantity
        
        # Add spot leg
        if position.spot_leg:
            current_value += (current_spot - position.spot_leg.entry_price) * position.spot_leg.quantity
        
        # PnL
        pnl = current_value - position.net_premium
        pnl_percent = (pnl / abs(position.net_premium)) * 100 if position.net_premium != 0 else 0
        
        # Calculate current Greeks
        self._calculate_greeks(position, current_spot, current_iv)
        greeks = position.net_greeks
        
        # Dollar exposures
        delta_exposure = greeks.delta * current_spot
        gamma_exposure = 0.5 * greeks.gamma * (current_spot * 0.01) ** 2 * 10000  # For 1% move
        theta_decay = greeks.theta  # Already in $ per day
        
        # Scenario analysis
        price_scenarios = self._calculate_price_scenarios(
            position, current_spot, current_iv, current_time
        )
        vol_scenarios = self._calculate_vol_scenarios(
            position, current_spot, current_iv, current_time
        )
        
        return StrategyAnalysis(
            position=position,
            current_value=current_value,
            pnl=pnl,
            pnl_percent=pnl_percent,
            days_to_expiry=(current_time or position.option_legs[0].expiry) * 365,
            delta=greeks.delta,
            gamma=greeks.gamma,
            theta=greeks.theta,
            vega=greeks.vega,
            delta_exposure_usd=delta_exposure,
            gamma_exposure_usd=gamma_exposure,
            theta_decay_usd=theta_decay,
            price_scenarios=price_scenarios,
            vol_scenarios=vol_scenarios
        )
    
    def _calculate_price_scenarios(
        self,
        position: StrategyPosition,
        spot: float,
        iv: float,
        time: Optional[float]
    ) -> Dict[float, float]:
        """Calculate P&L at different spot prices."""
        scenarios = {}
        
        for pct_move in [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20]:
            scenario_spot = spot * (1 + pct_move / 100)
            
            scenario_value = 0.0
            for leg in position.option_legs:
                T = time if time is not None else leg.expiry
                if T <= 0:
                    if leg.option_type == OptionType.CALL:
                        intrinsic = max(scenario_spot - leg.strike, 0)
                    else:
                        intrinsic = max(leg.strike - scenario_spot, 0)
                    scenario_value += intrinsic * leg.quantity
                else:
                    price = self.pricing_engine.black_scholes_price(
                        scenario_spot, leg.strike, T, iv, leg.option_type
                    )
                    scenario_value += price * leg.quantity
            
            if position.spot_leg:
                scenario_value += (scenario_spot - position.spot_leg.entry_price) * position.spot_leg.quantity
            
            scenarios[scenario_spot] = scenario_value - position.net_premium
        
        return scenarios
    
    def _calculate_vol_scenarios(
        self,
        position: StrategyPosition,
        spot: float,
        iv: float,
        time: Optional[float]
    ) -> Dict[float, float]:
        """Calculate P&L at different volatility levels."""
        scenarios = {}
        
        for vol_change in [-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20]:
            scenario_iv = iv * (1 + vol_change)
            
            scenario_value = 0.0
            for leg in position.option_legs:
                T = time if time is not None else leg.expiry
                if T > 0:
                    price = self.pricing_engine.black_scholes_price(
                        spot, leg.strike, T, scenario_iv, leg.option_type
                    )
                    scenario_value += price * leg.quantity
            
            if position.spot_leg:
                scenario_value += 0  # Vol doesn't affect spot directly
            
            scenarios[scenario_iv] = scenario_value - position.net_premium
        
        return scenarios
    
    def calculate_probability_of_profit(
        self,
        position: StrategyPosition,
        spot: float,
        iv: float,
        n_simulations: int = 10000
    ) -> float:
        """
        Monte Carlo estimate of probability of profit.
        
        Returns probability (0-1) that position is profitable at expiry.
        """
        if not position.option_legs:
            return 0.5
        
        expiry = position.option_legs[0].expiry
        
        # Simulate terminal prices
        drift = -0.5 * iv ** 2 * expiry
        diffusion = iv * np.sqrt(expiry)
        Z = np.random.standard_normal(n_simulations)
        S_T = spot * np.exp(drift + diffusion * Z)
        
        # Calculate payoffs
        payoffs = np.zeros(n_simulations)
        
        for leg in position.option_legs:
            if leg.option_type == OptionType.CALL:
                leg_payoff = np.maximum(S_T - leg.strike, 0)
            else:
                leg_payoff = np.maximum(leg.strike - S_T, 0)
            payoffs += leg_payoff * leg.quantity
        
        if position.spot_leg:
            payoffs += (S_T - position.spot_leg.entry_price) * position.spot_leg.quantity
        
        # Subtract net premium
        pnl = payoffs - position.net_premium
        
        return np.mean(pnl > 0)


# ==================== DELTA HEDGING ENGINE ====================

class DeltaHedgingEngine:
    """
    Production-grade delta hedging system.
    
    Supports:
    - Continuous delta hedging
    - Band-based hedging (hedge only when delta exceeds threshold)
    - Time-based hedging
    - Cost-aware hedging
    """
    
    def __init__(
        self,
        strategy_engine: Optional[OptionsStrategyEngine] = None,
        hedge_threshold: float = 0.05,  # Hedge when |delta| > threshold
        min_hedge_interval: float = 60.0,  # Minimum seconds between hedges
        hedge_cost_bps: float = 5.0  # Transaction cost in basis points
    ):
        self.strategy_engine = strategy_engine or OptionsStrategyEngine()
        self.hedge_threshold = hedge_threshold
        self.min_hedge_interval = min_hedge_interval
        self.hedge_cost_bps = hedge_cost_bps
        
        self.last_hedge_time: Optional[datetime] = None
        self.hedge_history: List[Dict] = []
        self.cumulative_hedge_cost: float = 0.0
        
        logger.info("DeltaHedgingEngine initialized")
    
    def calculate_hedge_quantity(
        self,
        position: StrategyPosition,
        current_spot: float,
        current_iv: float
    ) -> Tuple[float, float]:
        """
        Calculate required hedge quantity.
        
        Returns:
            Tuple of (hedge_quantity, current_delta)
            Positive hedge_quantity = buy spot, negative = sell spot
        """
        # Update Greeks
        self.strategy_engine._calculate_greeks(position, current_spot, current_iv)
        
        current_delta = position.net_greeks.delta
        
        # Target is delta-neutral
        hedge_quantity = -current_delta
        
        return hedge_quantity, current_delta
    
    def should_hedge(
        self,
        position: StrategyPosition,
        current_spot: float,
        current_iv: float
    ) -> Tuple[bool, str]:
        """
        Determine if hedging is needed.
        
        Returns:
            Tuple of (should_hedge, reason)
        """
        hedge_qty, current_delta = self.calculate_hedge_quantity(
            position, current_spot, current_iv
        )
        
        # Check time constraint
        if self.last_hedge_time:
            elapsed = (datetime.now() - self.last_hedge_time).total_seconds()
            if elapsed < self.min_hedge_interval:
                return False, f"Too soon since last hedge ({elapsed:.0f}s < {self.min_hedge_interval}s)"
        
        # Check threshold
        if abs(current_delta) < self.hedge_threshold:
            return False, f"Delta {current_delta:.4f} within threshold ±{self.hedge_threshold}"
        
        # Calculate cost vs benefit
        hedge_cost = abs(hedge_qty) * current_spot * (self.hedge_cost_bps / 10000)
        gamma_risk = 0.5 * position.net_greeks.gamma * (current_spot * 0.01) ** 2
        
        if hedge_cost > gamma_risk * 10:  # Cost exceeds 10x gamma risk
            return False, f"Hedge cost ${hedge_cost:.2f} exceeds risk benefit"
        
        return True, f"Delta {current_delta:.4f} exceeds threshold"
    
    def execute_hedge(
        self,
        position: StrategyPosition,
        current_spot: float,
        current_iv: float,
        dry_run: bool = False
    ) -> Dict:
        """
        Execute delta hedge.
        
        Returns hedge execution details.
        """
        hedge_qty, current_delta = self.calculate_hedge_quantity(
            position, current_spot, current_iv
        )
        
        # Calculate costs
        notional = abs(hedge_qty) * current_spot
        cost = notional * (self.hedge_cost_bps / 10000)
        
        hedge_record = {
            'timestamp': datetime.now().isoformat(),
            'pre_delta': current_delta,
            'hedge_quantity': hedge_qty,
            'spot_price': current_spot,
            'notional': notional,
            'cost': cost,
            'post_delta': current_delta + hedge_qty,  # Should be ~0
            'dry_run': dry_run
        }
        
        if not dry_run:
            # Update position
            if position.spot_leg is None:
                position.spot_leg = SpotLeg(
                    quantity=hedge_qty,
                    entry_price=current_spot
                )
            else:
                # Update average price
                total_qty = position.spot_leg.quantity + hedge_qty
                if total_qty != 0:
                    position.spot_leg.entry_price = (
                        position.spot_leg.entry_price * position.spot_leg.quantity +
                        current_spot * hedge_qty
                    ) / total_qty
                position.spot_leg.quantity = total_qty
            
            self.last_hedge_time = datetime.now()
            self.cumulative_hedge_cost += cost
            self.hedge_history.append(hedge_record)
            
            # Recalculate Greeks
            self.strategy_engine._calculate_greeks(position, current_spot, current_iv)
        
        logger.info(
            f"Delta hedge: {hedge_qty:+.4f} @ ${current_spot:.2f} "
            f"(pre-delta: {current_delta:.4f}, cost: ${cost:.2f})"
        )
        
        return hedge_record
    
    def get_hedge_summary(self) -> Dict:
        """Get summary of hedging activity."""
        if not self.hedge_history:
            return {'total_hedges': 0}
        
        total_bought = sum(h['hedge_quantity'] for h in self.hedge_history if h['hedge_quantity'] > 0)
        total_sold = sum(abs(h['hedge_quantity']) for h in self.hedge_history if h['hedge_quantity'] < 0)
        
        return {
            'total_hedges': len(self.hedge_history),
            'total_bought': total_bought,
            'total_sold': total_sold,
            'net_position': total_bought - total_sold,
            'total_turnover': sum(h['notional'] for h in self.hedge_history),
            'total_costs': self.cumulative_hedge_cost,
            'avg_cost_per_hedge': self.cumulative_hedge_cost / len(self.hedge_history)
        }


# ==================== GAMMA SCALPING ENGINE ====================

class GammaScalpingEngine:
    """
    Gamma Scalping: Profit from gamma by buying options and hedging delta.
    
    Strategy:
    1. Buy ATM straddle (long gamma, long vega)
    2. Delta hedge continuously
    3. Profit from realized vol > implied vol
    """
    
    def __init__(
        self,
        strategy_engine: Optional[OptionsStrategyEngine] = None,
        hedge_threshold: float = 0.10,
        scalp_threshold: float = 0.01  # Minimum spot move to consider scalping
    ):
        self.strategy_engine = strategy_engine or OptionsStrategyEngine()
        self.delta_hedger = DeltaHedgingEngine(
            strategy_engine=self.strategy_engine,
            hedge_threshold=hedge_threshold
        )
        self.scalp_threshold = scalp_threshold
        
        self.scalp_history: List[Dict] = []
        self.realized_pnl: float = 0.0
        self.last_spot: Optional[float] = None
        
        logger.info("GammaScalpingEngine initialized")
    
    def setup_gamma_position(
        self,
        symbol: str,
        spot: float,
        expiry: float,
        iv: float,
        notional: float = 10000
    ) -> StrategyPosition:
        """
        Set up gamma scalping position (long straddle).
        
        Args:
            symbol: Trading symbol
            spot: Current spot price
            expiry: Time to expiry (years)
            iv: Implied volatility
            notional: Dollar exposure target
        
        Returns:
            StrategyPosition with long straddle
        """
        # Calculate quantity to achieve target gamma exposure
        # Use ATM strike
        strike = spot
        
        # Create long straddle
        position = self.strategy_engine.create_straddle(
            symbol=symbol,
            spot=spot,
            strike=strike,
            expiry=expiry,
            iv=iv,
            quantity=1,
            is_long=True
        )
        
        # Scale to target notional
        if position.net_greeks.gamma > 0:
            gamma_dollar = position.net_greeks.gamma * spot ** 2 / 100
            scale_factor = notional / (gamma_dollar * 100) if gamma_dollar > 0 else 1
            
            for leg in position.option_legs:
                leg.quantity = int(leg.quantity * scale_factor)
            
            position.net_premium *= scale_factor
            self.strategy_engine._calculate_greeks(position, spot, iv)
        
        self.last_spot = spot
        
        logger.info(
            f"Gamma scalp position: {symbol} straddle @ K={strike:.2f}, "
            f"gamma=${position.net_greeks.gamma * spot**2 / 100:.2f}/1%"
        )
        
        return position
    
    def process_price_update(
        self,
        position: StrategyPosition,
        current_spot: float,
        current_iv: float
    ) -> Optional[Dict]:
        """
        Process price update and execute gamma scalp if appropriate.
        
        Returns scalp details if executed, None otherwise.
        """
        if self.last_spot is None:
            self.last_spot = current_spot
            return None
        
        # Calculate spot move
        spot_return = (current_spot - self.last_spot) / self.last_spot
        
        if abs(spot_return) < self.scalp_threshold:
            return None
        
        # Check if we should hedge
        should_hedge, reason = self.delta_hedger.should_hedge(
            position, current_spot, current_iv
        )
        
        if not should_hedge:
            return None
        
        # Calculate gamma P&L before hedging
        gamma_pnl = 0.5 * position.net_greeks.gamma * (
            (current_spot - self.last_spot) ** 2
        )
        
        # Execute hedge
        hedge_result = self.delta_hedger.execute_hedge(
            position, current_spot, current_iv
        )
        
        # Net P&L (gamma gain - hedge cost)
        net_pnl = gamma_pnl - hedge_result['cost']
        self.realized_pnl += net_pnl
        
        scalp_record = {
            'timestamp': datetime.now().isoformat(),
            'prev_spot': self.last_spot,
            'current_spot': current_spot,
            'spot_return': spot_return,
            'gamma_pnl': gamma_pnl,
            'hedge_cost': hedge_result['cost'],
            'net_pnl': net_pnl,
            'cumulative_pnl': self.realized_pnl,
            'hedge_details': hedge_result
        }
        
        self.scalp_history.append(scalp_record)
        self.last_spot = current_spot
        
        logger.info(
            f"Gamma scalp: spot {spot_return:+.2%}, "
            f"gamma P&L: ${gamma_pnl:.2f}, net: ${net_pnl:.2f}"
        )
        
        return scalp_record
    
    def get_scalp_summary(self) -> Dict:
        """Get summary of gamma scalping activity."""
        if not self.scalp_history:
            return {'total_scalps': 0}
        
        gamma_pnls = [s['gamma_pnl'] for s in self.scalp_history]
        hedge_costs = [s['hedge_cost'] for s in self.scalp_history]
        
        return {
            'total_scalps': len(self.scalp_history),
            'total_gamma_pnl': sum(gamma_pnls),
            'total_hedge_costs': sum(hedge_costs),
            'net_realized_pnl': self.realized_pnl,
            'avg_gamma_pnl': np.mean(gamma_pnls),
            'avg_hedge_cost': np.mean(hedge_costs),
            'win_rate': np.mean([s['net_pnl'] > 0 for s in self.scalp_history]),
            'largest_scalp': max(gamma_pnls),
            'hedge_summary': self.delta_hedger.get_hedge_summary()
        }


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Options Strategy Engine")
    parser.add_argument("--strategy", type=str, required=True,
                       choices=['covered_call', 'protective_put', 'bull_call', 'bear_put',
                               'straddle', 'strangle', 'iron_condor', 'iron_butterfly'],
                       help="Strategy type")
    parser.add_argument("--spot", type=float, required=True, help="Spot price")
    parser.add_argument("--strike", type=float, help="Strike price (for single strike strategies)")
    parser.add_argument("--strike-lower", type=float, help="Lower strike")
    parser.add_argument("--strike-upper", type=float, help="Upper strike")
    parser.add_argument("--expiry", type=float, required=True, help="Time to expiry (years)")
    parser.add_argument("--iv", type=float, required=True, help="Implied volatility")
    parser.add_argument("--quantity", type=int, default=1, help="Position size")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    
    args = parser.parse_args()
    
    engine = OptionsStrategyEngine()
    
    # Build strategy based on type
    if args.strategy == 'covered_call':
        position = engine.create_covered_call(
            "BTC", args.spot, args.strike or args.spot * 1.05,
            args.expiry, args.iv, args.quantity
        )
    elif args.strategy == 'protective_put':
        position = engine.create_protective_put(
            "BTC", args.spot, args.strike or args.spot * 0.95,
            args.expiry, args.iv, args.quantity
        )
    elif args.strategy == 'straddle':
        position = engine.create_straddle(
            "BTC", args.spot, args.strike or args.spot,
            args.expiry, args.iv, args.quantity, is_long=True
        )
    elif args.strategy == 'iron_condor':
        # Default strikes for IC
        s = args.spot
        position = engine.create_iron_condor(
            "BTC", s,
            put_long_strike=s * 0.85,
            put_short_strike=s * 0.90,
            call_short_strike=s * 1.10,
            call_long_strike=s * 1.15,
            expiry=args.expiry, iv=args.iv, quantity=args.quantity
        )
    else:
        print(f"Strategy {args.strategy} not fully implemented in CLI")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"STRATEGY: {position.strategy_type.value.upper()}")
    print(f"{'='*60}")
    print(f"Net Premium: ${position.net_premium:.2f}")
    print(f"Max Profit:  ${position.max_profit:.2f}" if position.max_profit != float('inf') else "Max Profit:  Unlimited")
    print(f"Max Loss:    ${position.max_loss:.2f}" if position.max_loss != float('inf') else "Max Loss:    Unlimited")
    print(f"Breakeven:   {position.breakeven_points}")
    print(f"\nGreeks:")
    print(f"  Delta: {position.net_greeks.delta:+.4f}")
    print(f"  Gamma: {position.net_greeks.gamma:.6f}")
    print(f"  Theta: ${position.net_greeks.theta:.4f}/day")
    print(f"  Vega:  ${position.net_greeks.vega:.4f}/1%vol")
    
    if args.analyze:
        analysis = engine.analyze_position(position, args.spot, args.iv)
        print(f"\nScenario Analysis (P&L at different spots):")
        for spot_price, pnl in sorted(analysis.price_scenarios.items()):
            pct_change = (spot_price / args.spot - 1) * 100
            print(f"  {pct_change:+6.1f}%: ${pnl:+.2f}")
