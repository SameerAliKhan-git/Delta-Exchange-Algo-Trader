"""
Options Alpha Strategy - Advanced Options Trading

Implements sophisticated options strategies:
- Delta-neutral positions
- Gamma scalping
- Volatility arbitrage
- IV surface analysis
- Greeks-based portfolio management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class StrategyType(Enum):
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    VERTICAL_SPREAD = "vertical_spread"
    CALENDAR_SPREAD = "calendar_spread"
    DELTA_NEUTRAL = "delta_neutral"
    GAMMA_SCALP = "gamma_scalp"


@dataclass
class Greeks:
    """Option Greeks"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass
class OptionContract:
    """Option contract details"""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    price: float
    iv: float  # Implied volatility
    greeks: Greeks = field(default_factory=Greeks)
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0


@dataclass
class OptionsPosition:
    """Options position"""
    contracts: List[Tuple[OptionContract, int]]  # (contract, quantity)
    entry_price: float
    current_value: float
    pnl: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float


@dataclass
class OptionsSignal:
    """Options trading signal"""
    strategy_type: StrategyType
    contracts: List[Tuple[OptionContract, int]]  # (contract, quantity)
    expected_profit: float
    max_loss: float
    probability_of_profit: float
    breakeven_points: List[float]
    greeks: Greeks
    confidence: float
    rationale: str


class OptionsAlphaStrategy:
    """
    Advanced Options Trading Strategy
    
    Implements multiple options strategies with Greeks management
    and volatility-based entry/exit signals.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        min_iv_percentile: float = 20,
        max_iv_percentile: float = 80,
        target_delta: float = 0.3,
        max_portfolio_delta: float = 0.5,
        max_portfolio_gamma: float = 0.2
    ):
        """
        Initialize options strategy
        
        Args:
            risk_free_rate: Risk-free interest rate
            min_iv_percentile: Min IV percentile for long vol
            max_iv_percentile: Max IV percentile for short vol
            target_delta: Target delta for directional trades
            max_portfolio_delta: Maximum portfolio delta
            max_portfolio_gamma: Maximum portfolio gamma
        """
        self.rf_rate = risk_free_rate
        self.min_iv_pct = min_iv_percentile
        self.max_iv_pct = max_iv_percentile
        self.target_delta = target_delta
        self.max_delta = max_portfolio_delta
        self.max_gamma = max_portfolio_gamma
        
        # IV history for percentile calculations
        self._iv_history: Dict[str, List[float]] = {}
        
        # Current positions
        self._positions: Dict[str, OptionsPosition] = {}
    
    def analyze_opportunity(
        self,
        underlying_price: float,
        option_chain: List[OptionContract],
        direction: int = 0,  # 0=neutral, 1=bullish, -1=bearish
        volatility_view: int = 0  # 0=neutral, 1=long vol, -1=short vol
    ) -> List[OptionsSignal]:
        """
        Analyze options chain for trading opportunities
        
        Args:
            underlying_price: Current underlying price
            option_chain: Available option contracts
            direction: Directional bias
            volatility_view: Volatility outlook
        
        Returns:
            List of trading signals ranked by expected value
        """
        signals = []
        
        # Get IV percentile
        iv_percentile = self._calculate_iv_percentile(option_chain)
        
        # Strategy selection based on views
        if volatility_view == 1 or iv_percentile < self.min_iv_pct:
            # Long volatility strategies
            signals.extend(self._long_vol_strategies(underlying_price, option_chain, direction))
        elif volatility_view == -1 or iv_percentile > self.max_iv_pct:
            # Short volatility strategies
            signals.extend(self._short_vol_strategies(underlying_price, option_chain, direction))
        
        # Directional strategies if strong view
        if abs(direction) == 1:
            signals.extend(self._directional_strategies(underlying_price, option_chain, direction))
        
        # Delta-neutral strategies
        signals.extend(self._delta_neutral_strategies(underlying_price, option_chain))
        
        # Rank by expected value and filter
        signals.sort(key=lambda x: x.expected_profit * x.probability_of_profit, reverse=True)
        
        return signals[:5]  # Top 5 opportunities
    
    def create_delta_neutral_position(
        self,
        underlying_price: float,
        option: OptionContract,
        contracts: int
    ) -> Tuple[int, str]:
        """
        Create delta-neutral position
        
        Returns:
            (hedge_quantity, hedge_direction) for underlying
        """
        position_delta = option.greeks.delta * contracts
        
        # Hedge with underlying
        hedge_qty = -int(position_delta * 100)  # Assuming 100 shares per contract
        hedge_direction = "buy" if hedge_qty > 0 else "sell"
        
        return abs(hedge_qty), hedge_direction
    
    def calculate_gamma_scalp_levels(
        self,
        position: OptionsPosition,
        underlying_price: float,
        scalp_threshold: float = 0.01  # 1% move
    ) -> Tuple[float, float]:
        """
        Calculate price levels for gamma scalping
        
        Returns:
            (buy_level, sell_level)
        """
        move = underlying_price * scalp_threshold
        
        # If long gamma, we profit from rebalancing
        if position.net_gamma > 0:
            buy_level = underlying_price - move
            sell_level = underlying_price + move
        else:
            # Short gamma - need to hedge larger moves
            buy_level = underlying_price - move * 2
            sell_level = underlying_price + move * 2
        
        return buy_level, sell_level
    
    def find_volatility_arbitrage(
        self,
        option_chain: List[OptionContract],
        realized_vol: float
    ) -> List[OptionsSignal]:
        """
        Find volatility arbitrage opportunities
        
        Compares implied volatility to realized volatility
        """
        signals = []
        
        for option in option_chain:
            vol_diff = option.iv - realized_vol
            
            # IV significantly higher than RV - sell options
            if vol_diff > 0.1:  # 10% higher
                strategy = self._create_short_vol_signal(option)
                signals.append(strategy)
            
            # IV significantly lower than RV - buy options
            elif vol_diff < -0.1:
                strategy = self._create_long_vol_signal(option)
                signals.append(strategy)
        
        return signals
    
    def analyze_iv_surface(
        self,
        option_chain: List[OptionContract],
        underlying_price: float
    ) -> Dict[str, any]:
        """
        Analyze implied volatility surface
        
        Returns:
            IV surface analysis with skew and term structure
        """
        # Group by expiry
        by_expiry = {}
        for opt in option_chain:
            exp_key = opt.expiry.strftime("%Y-%m-%d")
            if exp_key not in by_expiry:
                by_expiry[exp_key] = []
            by_expiry[exp_key].append(opt)
        
        analysis = {
            'atm_iv': {},
            'skew': {},
            'term_structure': [],
            'smile': {}
        }
        
        for expiry, options in by_expiry.items():
            # Find ATM options
            atm_options = sorted(options, key=lambda x: abs(x.strike - underlying_price))[:2]
            atm_iv = np.mean([o.iv for o in atm_options])
            analysis['atm_iv'][expiry] = atm_iv
            
            # Calculate skew (25 delta put IV - 25 delta call IV)
            puts = [o for o in options if o.option_type == OptionType.PUT]
            calls = [o for o in options if o.option_type == OptionType.CALL]
            
            if puts and calls:
                # Find ~25 delta options
                put_25d = min(puts, key=lambda x: abs(abs(x.greeks.delta) - 0.25), default=None)
                call_25d = min(calls, key=lambda x: abs(x.greeks.delta - 0.25), default=None)
                
                if put_25d and call_25d:
                    analysis['skew'][expiry] = put_25d.iv - call_25d.iv
            
            # IV smile for this expiry
            analysis['smile'][expiry] = {
                'strikes': [o.strike for o in sorted(options, key=lambda x: x.strike)],
                'ivs': [o.iv for o in sorted(options, key=lambda x: x.strike)]
            }
        
        # Term structure (sorted by expiry)
        sorted_expiries = sorted(analysis['atm_iv'].items())
        analysis['term_structure'] = [
            {'expiry': e, 'iv': iv} for e, iv in sorted_expiries
        ]
        
        return analysis
    
    def _long_vol_strategies(
        self,
        price: float,
        chain: List[OptionContract],
        direction: int
    ) -> List[OptionsSignal]:
        """Generate long volatility strategies"""
        signals = []
        
        # Find ATM options
        atm_call = self._find_atm_option(chain, price, OptionType.CALL)
        atm_put = self._find_atm_option(chain, price, OptionType.PUT)
        
        if atm_call and atm_put:
            # Long Straddle
            cost = atm_call.price + atm_put.price
            breakeven_up = price + cost
            breakeven_down = price - cost
            
            # Calculate expected profit based on IV
            expected_move = price * atm_call.iv * np.sqrt(self._days_to_expiry(atm_call) / 365)
            expected_profit = max(0, expected_move - cost)
            
            signals.append(OptionsSignal(
                strategy_type=StrategyType.LONG_STRADDLE,
                contracts=[(atm_call, 1), (atm_put, 1)],
                expected_profit=expected_profit,
                max_loss=cost,
                probability_of_profit=0.4,  # Straddles have ~40% POP
                breakeven_points=[breakeven_down, breakeven_up],
                greeks=Greeks(
                    delta=atm_call.greeks.delta + atm_put.greeks.delta,
                    gamma=atm_call.greeks.gamma + atm_put.greeks.gamma,
                    theta=atm_call.greeks.theta + atm_put.greeks.theta,
                    vega=atm_call.greeks.vega + atm_put.greeks.vega
                ),
                confidence=0.6 if expected_move > cost else 0.4,
                rationale=f"Long vol play: IV at {atm_call.iv:.1%}, expecting {expected_move/price:.1%} move"
            ))
        
        # Long Strangle (OTM options, cheaper)
        otm_call = self._find_option_by_delta(chain, 0.25, OptionType.CALL)
        otm_put = self._find_option_by_delta(chain, -0.25, OptionType.PUT)
        
        if otm_call and otm_put:
            cost = otm_call.price + otm_put.price
            
            signals.append(OptionsSignal(
                strategy_type=StrategyType.LONG_STRANGLE,
                contracts=[(otm_call, 1), (otm_put, 1)],
                expected_profit=cost * 0.5,  # Target 50% return
                max_loss=cost,
                probability_of_profit=0.35,
                breakeven_points=[otm_put.strike - cost, otm_call.strike + cost],
                greeks=Greeks(
                    delta=otm_call.greeks.delta + otm_put.greeks.delta,
                    gamma=otm_call.greeks.gamma + otm_put.greeks.gamma,
                    theta=otm_call.greeks.theta + otm_put.greeks.theta,
                    vega=otm_call.greeks.vega + otm_put.greeks.vega
                ),
                confidence=0.5,
                rationale="Long strangle for vol expansion"
            ))
        
        return signals
    
    def _short_vol_strategies(
        self,
        price: float,
        chain: List[OptionContract],
        direction: int
    ) -> List[OptionsSignal]:
        """Generate short volatility strategies"""
        signals = []
        
        # Iron Condor
        put_sell = self._find_option_by_delta(chain, -0.2, OptionType.PUT)
        put_buy = self._find_option_by_delta(chain, -0.1, OptionType.PUT)
        call_sell = self._find_option_by_delta(chain, 0.2, OptionType.CALL)
        call_buy = self._find_option_by_delta(chain, 0.1, OptionType.CALL)
        
        if all([put_sell, put_buy, call_sell, call_buy]):
            credit = (put_sell.price - put_buy.price + call_sell.price - call_buy.price)
            max_loss = min(
                put_sell.strike - put_buy.strike,
                call_buy.strike - call_sell.strike
            ) - credit
            
            signals.append(OptionsSignal(
                strategy_type=StrategyType.IRON_CONDOR,
                contracts=[
                    (put_buy, 1), (put_sell, -1),
                    (call_sell, -1), (call_buy, 1)
                ],
                expected_profit=credit * 0.5,  # Target 50% of credit
                max_loss=max_loss,
                probability_of_profit=0.7,  # Iron condors ~70% POP
                breakeven_points=[
                    put_sell.strike - credit,
                    call_sell.strike + credit
                ],
                greeks=Greeks(
                    delta=sum(o.greeks.delta * q for o, q in [
                        (put_buy, 1), (put_sell, -1), (call_sell, -1), (call_buy, 1)
                    ]),
                    gamma=sum(o.greeks.gamma * q for o, q in [
                        (put_buy, 1), (put_sell, -1), (call_sell, -1), (call_buy, 1)
                    ]),
                    theta=sum(o.greeks.theta * q for o, q in [
                        (put_buy, 1), (put_sell, -1), (call_sell, -1), (call_buy, 1)
                    ]),
                    vega=sum(o.greeks.vega * q for o, q in [
                        (put_buy, 1), (put_sell, -1), (call_sell, -1), (call_buy, 1)
                    ])
                ),
                confidence=0.7,
                rationale=f"Iron condor for premium collection, {credit/max_loss:.1%} risk/reward"
            ))
        
        # Short Strangle (undefined risk - for experienced traders)
        atm_call = self._find_atm_option(chain, price, OptionType.CALL)
        atm_put = self._find_atm_option(chain, price, OptionType.PUT)
        
        if atm_call and atm_put:
            credit = atm_call.price + atm_put.price
            
            signals.append(OptionsSignal(
                strategy_type=StrategyType.SHORT_STRADDLE,
                contracts=[(atm_call, -1), (atm_put, -1)],
                expected_profit=credit * 0.3,  # Manage at 30%
                max_loss=float('inf'),  # Undefined risk
                probability_of_profit=0.6,
                breakeven_points=[price - credit, price + credit],
                greeks=Greeks(
                    delta=-(atm_call.greeks.delta + atm_put.greeks.delta),
                    gamma=-(atm_call.greeks.gamma + atm_put.greeks.gamma),
                    theta=-(atm_call.greeks.theta + atm_put.greeks.theta),
                    vega=-(atm_call.greeks.vega + atm_put.greeks.vega)
                ),
                confidence=0.6,
                rationale="Short straddle - high theta, manage risk carefully"
            ))
        
        return signals
    
    def _directional_strategies(
        self,
        price: float,
        chain: List[OptionContract],
        direction: int
    ) -> List[OptionsSignal]:
        """Generate directional strategies"""
        signals = []
        
        if direction == 1:  # Bullish
            # Bull Call Spread
            atm_call = self._find_atm_option(chain, price, OptionType.CALL)
            otm_call = self._find_option_by_delta(chain, 0.2, OptionType.CALL)
            
            if atm_call and otm_call:
                cost = atm_call.price - otm_call.price
                max_profit = otm_call.strike - atm_call.strike - cost
                
                signals.append(OptionsSignal(
                    strategy_type=StrategyType.VERTICAL_SPREAD,
                    contracts=[(atm_call, 1), (otm_call, -1)],
                    expected_profit=max_profit * 0.4,
                    max_loss=cost,
                    probability_of_profit=0.5,
                    breakeven_points=[atm_call.strike + cost],
                    greeks=Greeks(
                        delta=atm_call.greeks.delta - otm_call.greeks.delta,
                        gamma=atm_call.greeks.gamma - otm_call.greeks.gamma,
                        theta=atm_call.greeks.theta - otm_call.greeks.theta,
                        vega=atm_call.greeks.vega - otm_call.greeks.vega
                    ),
                    confidence=0.55,
                    rationale=f"Bullish call spread, {max_profit/cost:.1%} max return"
                ))
        
        elif direction == -1:  # Bearish
            # Bear Put Spread
            atm_put = self._find_atm_option(chain, price, OptionType.PUT)
            otm_put = self._find_option_by_delta(chain, -0.2, OptionType.PUT)
            
            if atm_put and otm_put:
                cost = atm_put.price - otm_put.price
                max_profit = atm_put.strike - otm_put.strike - cost
                
                signals.append(OptionsSignal(
                    strategy_type=StrategyType.VERTICAL_SPREAD,
                    contracts=[(atm_put, 1), (otm_put, -1)],
                    expected_profit=max_profit * 0.4,
                    max_loss=cost,
                    probability_of_profit=0.5,
                    breakeven_points=[atm_put.strike - cost],
                    greeks=Greeks(
                        delta=atm_put.greeks.delta - otm_put.greeks.delta,
                        gamma=atm_put.greeks.gamma - otm_put.greeks.gamma,
                        theta=atm_put.greeks.theta - otm_put.greeks.theta,
                        vega=atm_put.greeks.vega - otm_put.greeks.vega
                    ),
                    confidence=0.55,
                    rationale=f"Bearish put spread, {max_profit/cost:.1%} max return"
                ))
        
        return signals
    
    def _delta_neutral_strategies(
        self,
        price: float,
        chain: List[OptionContract]
    ) -> List[OptionsSignal]:
        """Generate delta-neutral strategies"""
        signals = []
        
        # Gamma scalping setup
        atm_call = self._find_atm_option(chain, price, OptionType.CALL)
        
        if atm_call and atm_call.greeks.gamma > 0:
            # Long gamma position for scalping
            hedge_delta = -int(atm_call.greeks.delta * 100)
            
            signals.append(OptionsSignal(
                strategy_type=StrategyType.GAMMA_SCALP,
                contracts=[(atm_call, 1)],
                expected_profit=atm_call.price * 0.2,  # Target 20% of premium
                max_loss=atm_call.price,
                probability_of_profit=0.45,
                breakeven_points=[],  # Dynamic with scalping
                greeks=Greeks(
                    delta=0,  # Hedged
                    gamma=atm_call.greeks.gamma,
                    theta=atm_call.greeks.theta,
                    vega=atm_call.greeks.vega
                ),
                confidence=0.5,
                rationale=f"Gamma scalp: hedge {hedge_delta} shares, scalp at {price*0.01:.0f} intervals"
            ))
        
        return signals
    
    def _find_atm_option(
        self,
        chain: List[OptionContract],
        price: float,
        opt_type: OptionType
    ) -> Optional[OptionContract]:
        """Find at-the-money option"""
        options = [o for o in chain if o.option_type == opt_type]
        if not options:
            return None
        return min(options, key=lambda x: abs(x.strike - price))
    
    def _find_option_by_delta(
        self,
        chain: List[OptionContract],
        target_delta: float,
        opt_type: OptionType
    ) -> Optional[OptionContract]:
        """Find option by delta"""
        options = [o for o in chain if o.option_type == opt_type]
        if not options:
            return None
        return min(options, key=lambda x: abs(x.greeks.delta - target_delta))
    
    def _calculate_iv_percentile(self, chain: List[OptionContract]) -> float:
        """Calculate current IV percentile"""
        if not chain:
            return 50
        
        current_iv = np.mean([o.iv for o in chain])
        
        # Use history if available
        underlying = chain[0].underlying if chain else ''
        if underlying in self._iv_history and self._iv_history[underlying]:
            history = self._iv_history[underlying]
            percentile = sum(1 for iv in history if iv < current_iv) / len(history) * 100
            return percentile
        
        return 50  # Default to 50th percentile
    
    def _days_to_expiry(self, option: OptionContract) -> float:
        """Calculate days to expiry"""
        return max(0, (option.expiry - datetime.utcnow()).days)
    
    def _create_short_vol_signal(self, option: OptionContract) -> OptionsSignal:
        """Create short volatility signal"""
        return OptionsSignal(
            strategy_type=StrategyType.DELTA_NEUTRAL,
            contracts=[(option, -1)],
            expected_profit=option.price * 0.3,
            max_loss=float('inf'),
            probability_of_profit=0.6,
            breakeven_points=[],
            greeks=Greeks(
                delta=-option.greeks.delta,
                gamma=-option.greeks.gamma,
                theta=-option.greeks.theta,
                vega=-option.greeks.vega
            ),
            confidence=0.55,
            rationale=f"Short vol: IV {option.iv:.1%} above realized"
        )
    
    def _create_long_vol_signal(self, option: OptionContract) -> OptionsSignal:
        """Create long volatility signal"""
        return OptionsSignal(
            strategy_type=StrategyType.DELTA_NEUTRAL,
            contracts=[(option, 1)],
            expected_profit=option.price * 0.5,
            max_loss=option.price,
            probability_of_profit=0.45,
            breakeven_points=[],
            greeks=option.greeks,
            confidence=0.55,
            rationale=f"Long vol: IV {option.iv:.1%} below realized"
        )
    
    def update_iv_history(self, underlying: str, iv: float) -> None:
        """Update IV history for percentile calculations"""
        if underlying not in self._iv_history:
            self._iv_history[underlying] = []
        
        self._iv_history[underlying].append(iv)
        
        # Keep last 252 trading days
        if len(self._iv_history[underlying]) > 252:
            self._iv_history[underlying] = self._iv_history[underlying][-252:]


# Black-Scholes Option Pricing
def black_scholes(
    S: float,  # Underlying price
    K: float,  # Strike price
    T: float,  # Time to expiry (years)
    r: float,  # Risk-free rate
    sigma: float,  # Volatility
    option_type: str = 'call'
) -> Tuple[float, Greeks]:
    """
    Black-Scholes option pricing with Greeks
    
    Returns:
        (price, greeks)
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0), Greeks()
        else:
            return max(K - S, 0), Greeks()
    
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Standard normal CDF and PDF
    from scipy.stats import norm
    N = norm.cdf
    n = norm.pdf
    
    if option_type == 'call':
        price = S * N(d1) - K * np.exp(-r * T) * N(d2)
        delta = N(d1)
    else:
        price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
        delta = N(d1) - 1
    
    # Greeks
    gamma = n(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * n(d1) * sigma / (2 * np.sqrt(T)) - 
             r * K * np.exp(-r * T) * N(d2 if option_type == 'call' else -d2)) / 365
    vega = S * n(d1) * np.sqrt(T) / 100  # Per 1% IV change
    rho = K * T * np.exp(-r * T) * N(d2 if option_type == 'call' else -d2) / 100
    
    greeks = Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho
    )
    
    return price, greeks


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call'
) -> float:
    """
    Calculate implied volatility using Newton-Raphson
    """
    sigma = 0.3  # Initial guess
    
    for _ in range(100):
        bs_price, greeks = black_scholes(S, K, T, r, sigma, option_type)
        diff = bs_price - price
        
        if abs(diff) < 0.001:
            return sigma
        
        if greeks.vega == 0:
            break
        
        sigma -= diff / (greeks.vega * 100)
        sigma = max(0.01, min(5.0, sigma))
    
    return sigma
