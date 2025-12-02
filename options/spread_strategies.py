"""
ALADDIN - Spread Strategies
==============================
Option spread strategy builder and analyzer.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from .options_scanner import OptionsScanner, OptionQuote


class SpreadType(Enum):
    """Types of option spreads."""
    VERTICAL_CALL = "vertical_call"
    VERTICAL_PUT = "vertical_put"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    CALENDAR = "calendar"
    RATIO = "ratio"


@dataclass
class SpreadLeg:
    """Single leg of an option spread."""
    option: OptionQuote
    quantity: int  # Positive = long, Negative = short
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def cost(self) -> float:
        """Cost of this leg (positive = debit, negative = credit)."""
        if self.is_long:
            return self.option.ask * abs(self.quantity)
        else:
            return -self.option.bid * abs(self.quantity)


@dataclass
class OptionSpread:
    """Complete option spread trade."""
    spread_type: SpreadType
    underlying: str
    legs: List[SpreadLeg] = field(default_factory=list)
    
    # Calculated risk/reward
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_points: List[float] = field(default_factory=list)
    probability_profit: float = 0.0
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def net_cost(self) -> float:
        """Net cost of spread (positive = debit, negative = credit)."""
        return sum(leg.cost for leg in self.legs)
    
    @property
    def is_credit_spread(self) -> bool:
        return self.net_cost < 0
    
    @property
    def risk_reward_ratio(self) -> float:
        if self.max_loss != 0:
            return abs(self.max_profit / self.max_loss)
        return float('inf')
    
    def to_dict(self) -> Dict:
        return {
            'type': self.spread_type.value,
            'underlying': self.underlying,
            'legs': len(self.legs),
            'net_cost': self.net_cost,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'risk_reward': self.risk_reward_ratio,
            'breakevens': self.breakeven_points,
            'prob_profit': f"{self.probability_profit:.0%}",
            'description': self.description
        }
    
    def print_details(self):
        """Print spread details."""
        print(f"\n{'='*60}")
        print(f"📊 {self.spread_type.value.upper()}")
        print(f"   {self.underlying}")
        print(f"{'='*60}")
        
        print(f"\n🦵 LEGS:")
        for i, leg in enumerate(self.legs, 1):
            action = "BUY " if leg.is_long else "SELL"
            print(f"  {i}. {action} {abs(leg.quantity)}x {leg.option.symbol}")
            print(f"     Strike: {leg.option.strike:.0f}, "
                  f"Type: {leg.option.option_type}, "
                  f"Price: ${leg.option.mark_price:.2f}")
        
        print(f"\n💰 RISK/REWARD:")
        cost_type = "Debit" if self.net_cost > 0 else "Credit"
        print(f"  Net {cost_type}: ${abs(self.net_cost):.2f}")
        print(f"  Max Profit: ${self.max_profit:.2f}")
        print(f"  Max Loss: ${abs(self.max_loss):.2f}")
        print(f"  Risk/Reward: {self.risk_reward_ratio:.2f}:1")
        
        if self.breakeven_points:
            print(f"\n📍 BREAKEVEN: {', '.join(f'${b:.0f}' for b in self.breakeven_points)}")
        
        print(f"  Prob of Profit: {self.probability_profit:.0%}")
        print(f"{'='*60}")


class SpreadStrategies:
    """
    Option spread strategy builder.
    
    Builds and analyzes various option spread strategies:
    - Vertical spreads (bull/bear call/put spreads)
    - Iron condors
    - Iron butterflies
    - Straddles/Strangles
    - Calendar spreads
    """
    
    def __init__(self, scanner: OptionsScanner = None):
        self.logger = logging.getLogger('Aladdin.SpreadStrategies')
        self.scanner = scanner or OptionsScanner()
    
    def build_bull_call_spread(self, underlying: str, 
                               lower_strike: float, upper_strike: float,
                               expiry: datetime = None) -> Optional[OptionSpread]:
        """
        Build bull call spread (buy lower strike call, sell higher strike call).
        Bullish strategy with limited risk and limited reward.
        """
        chain = self.scanner.get_chain(underlying, expiry)
        calls = chain['calls']
        
        # Find matching options
        long_call = None
        short_call = None
        
        for call in calls:
            if abs(call.strike - lower_strike) < 1:
                long_call = call
            if abs(call.strike - upper_strike) < 1:
                short_call = call
        
        if not long_call or not short_call:
            return None
        
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_CALL,
            underlying=underlying,
            legs=[
                SpreadLeg(option=long_call, quantity=1),
                SpreadLeg(option=short_call, quantity=-1)
            ],
            description=f"Bull Call Spread: {lower_strike}/{upper_strike}"
        )
        
        self._calculate_vertical_risk_reward(spread)
        return spread
    
    def build_bear_put_spread(self, underlying: str,
                              upper_strike: float, lower_strike: float,
                              expiry: datetime = None) -> Optional[OptionSpread]:
        """
        Build bear put spread (buy higher strike put, sell lower strike put).
        Bearish strategy with limited risk and limited reward.
        """
        chain = self.scanner.get_chain(underlying, expiry)
        puts = chain['puts']
        
        long_put = None
        short_put = None
        
        for put in puts:
            if abs(put.strike - upper_strike) < 1:
                long_put = put
            if abs(put.strike - lower_strike) < 1:
                short_put = put
        
        if not long_put or not short_put:
            return None
        
        spread = OptionSpread(
            spread_type=SpreadType.VERTICAL_PUT,
            underlying=underlying,
            legs=[
                SpreadLeg(option=long_put, quantity=1),
                SpreadLeg(option=short_put, quantity=-1)
            ],
            description=f"Bear Put Spread: {upper_strike}/{lower_strike}"
        )
        
        self._calculate_vertical_risk_reward(spread)
        return spread
    
    def build_iron_condor(self, underlying: str, spot_price: float,
                         wing_width: float = 0.05,  # 5% wings
                         body_width: float = 0.03,  # 3% body
                         expiry: datetime = None) -> Optional[OptionSpread]:
        """
        Build iron condor (sell OTM call spread + sell OTM put spread).
        Neutral strategy profiting from time decay.
        """
        # Calculate strikes
        put_short_strike = spot_price * (1 - body_width)
        put_long_strike = spot_price * (1 - body_width - wing_width)
        call_short_strike = spot_price * (1 + body_width)
        call_long_strike = spot_price * (1 + body_width + wing_width)
        
        chain = self.scanner.get_chain(underlying, expiry)
        
        # Find closest strikes
        def find_closest(options, target_strike):
            return min(options, key=lambda o: abs(o.strike - target_strike), default=None)
        
        put_long = find_closest(chain['puts'], put_long_strike)
        put_short = find_closest(chain['puts'], put_short_strike)
        call_short = find_closest(chain['calls'], call_short_strike)
        call_long = find_closest(chain['calls'], call_long_strike)
        
        if not all([put_long, put_short, call_short, call_long]):
            return None
        
        spread = OptionSpread(
            spread_type=SpreadType.IRON_CONDOR,
            underlying=underlying,
            legs=[
                SpreadLeg(option=put_long, quantity=1),
                SpreadLeg(option=put_short, quantity=-1),
                SpreadLeg(option=call_short, quantity=-1),
                SpreadLeg(option=call_long, quantity=1)
            ],
            description=f"Iron Condor: {put_long.strike:.0f}/{put_short.strike:.0f}/{call_short.strike:.0f}/{call_long.strike:.0f}"
        )
        
        self._calculate_iron_condor_risk_reward(spread, spot_price)
        return spread
    
    def build_straddle(self, underlying: str, strike: float,
                      is_long: bool = True,
                      expiry: datetime = None) -> Optional[OptionSpread]:
        """
        Build straddle (buy/sell ATM call + ATM put at same strike).
        Volatility strategy betting on big move (long) or no move (short).
        """
        chain = self.scanner.get_chain(underlying, expiry)
        
        # Find ATM options
        atm_call = min(chain['calls'], key=lambda o: abs(o.strike - strike), default=None)
        atm_put = min(chain['puts'], key=lambda o: abs(o.strike - strike), default=None)
        
        if not atm_call or not atm_put:
            return None
        
        qty = 1 if is_long else -1
        
        spread = OptionSpread(
            spread_type=SpreadType.STRADDLE,
            underlying=underlying,
            legs=[
                SpreadLeg(option=atm_call, quantity=qty),
                SpreadLeg(option=atm_put, quantity=qty)
            ],
            description=f"{'Long' if is_long else 'Short'} Straddle @ {strike:.0f}"
        )
        
        self._calculate_straddle_risk_reward(spread, strike, is_long)
        return spread
    
    def build_strangle(self, underlying: str, spot_price: float,
                      width: float = 0.05,  # 5% OTM
                      is_long: bool = True,
                      expiry: datetime = None) -> Optional[OptionSpread]:
        """
        Build strangle (buy/sell OTM call + OTM put).
        Similar to straddle but cheaper with wider breakeven.
        """
        call_strike = spot_price * (1 + width)
        put_strike = spot_price * (1 - width)
        
        chain = self.scanner.get_chain(underlying, expiry)
        
        otm_call = min(chain['calls'], key=lambda o: abs(o.strike - call_strike), default=None)
        otm_put = min(chain['puts'], key=lambda o: abs(o.strike - put_strike), default=None)
        
        if not otm_call or not otm_put:
            return None
        
        qty = 1 if is_long else -1
        
        spread = OptionSpread(
            spread_type=SpreadType.STRANGLE,
            underlying=underlying,
            legs=[
                SpreadLeg(option=otm_call, quantity=qty),
                SpreadLeg(option=otm_put, quantity=qty)
            ],
            description=f"{'Long' if is_long else 'Short'} Strangle: {otm_put.strike:.0f}/{otm_call.strike:.0f}"
        )
        
        self._calculate_strangle_risk_reward(spread, is_long)
        return spread
    
    def _calculate_vertical_risk_reward(self, spread: OptionSpread):
        """Calculate risk/reward for vertical spreads."""
        net_cost = spread.net_cost
        
        # For bull call spread
        if spread.spread_type == SpreadType.VERTICAL_CALL:
            lower_strike = min(leg.option.strike for leg in spread.legs)
            upper_strike = max(leg.option.strike for leg in spread.legs)
            width = upper_strike - lower_strike
            
            spread.max_loss = abs(net_cost)
            spread.max_profit = width - abs(net_cost)
            spread.breakeven_points = [lower_strike + abs(net_cost)]
        
        # For bear put spread
        elif spread.spread_type == SpreadType.VERTICAL_PUT:
            lower_strike = min(leg.option.strike for leg in spread.legs)
            upper_strike = max(leg.option.strike for leg in spread.legs)
            width = upper_strike - lower_strike
            
            spread.max_loss = abs(net_cost)
            spread.max_profit = width - abs(net_cost)
            spread.breakeven_points = [upper_strike - abs(net_cost)]
        
        # Estimate probability (simplified using delta)
        long_leg = next(leg for leg in spread.legs if leg.is_long)
        spread.probability_profit = abs(long_leg.option.greeks.delta)
    
    def _calculate_iron_condor_risk_reward(self, spread: OptionSpread, spot_price: float):
        """Calculate risk/reward for iron condor."""
        credit = abs(spread.net_cost)  # Should be credit
        
        # Find wing width (assume equal)
        put_strikes = sorted([l.option.strike for l in spread.legs if l.option.option_type == 'put'])
        call_strikes = sorted([l.option.strike for l in spread.legs if l.option.option_type == 'call'])
        
        wing_width = max(
            put_strikes[1] - put_strikes[0] if len(put_strikes) >= 2 else 0,
            call_strikes[1] - call_strikes[0] if len(call_strikes) >= 2 else 0
        )
        
        spread.max_profit = credit
        spread.max_loss = wing_width - credit
        
        # Breakevens
        short_put = max(put_strikes) if put_strikes else spot_price
        short_call = min(call_strikes) if call_strikes else spot_price
        
        spread.breakeven_points = [
            short_put - credit,
            short_call + credit
        ]
        
        # Probability of profit (price stays between breakevens)
        # Simplified: use short option deltas
        spread.probability_profit = 0.65  # Typical for balanced IC
    
    def _calculate_straddle_risk_reward(self, spread: OptionSpread, strike: float, is_long: bool):
        """Calculate risk/reward for straddle."""
        total_premium = sum(
            leg.option.mark_price * abs(leg.quantity) for leg in spread.legs
        )
        
        if is_long:
            spread.max_loss = total_premium
            spread.max_profit = float('inf')  # Unlimited
            spread.breakeven_points = [strike - total_premium, strike + total_premium]
            spread.probability_profit = 0.35  # Typical for long straddle
        else:
            spread.max_profit = total_premium
            spread.max_loss = float('inf')  # Unlimited
            spread.breakeven_points = [strike - total_premium, strike + total_premium]
            spread.probability_profit = 0.65  # Typical for short straddle
    
    def _calculate_strangle_risk_reward(self, spread: OptionSpread, is_long: bool):
        """Calculate risk/reward for strangle."""
        call_leg = next(l for l in spread.legs if l.option.option_type == 'call')
        put_leg = next(l for l in spread.legs if l.option.option_type == 'put')
        
        total_premium = (call_leg.option.mark_price + put_leg.option.mark_price)
        
        if is_long:
            spread.max_loss = total_premium
            spread.max_profit = float('inf')
            spread.breakeven_points = [
                put_leg.option.strike - total_premium,
                call_leg.option.strike + total_premium
            ]
            spread.probability_profit = 0.30
        else:
            spread.max_profit = total_premium
            spread.max_loss = float('inf')
            spread.breakeven_points = [
                put_leg.option.strike - total_premium,
                call_leg.option.strike + total_premium
            ]
            spread.probability_profit = 0.70
    
    def suggest_strategy(self, underlying: str, spot_price: float,
                        outlook: str = 'neutral',  # bullish, bearish, neutral, volatile
                        iv_percentile: float = 50) -> List[OptionSpread]:
        """
        Suggest appropriate option strategies based on outlook and IV.
        
        Args:
            underlying: Underlying asset
            spot_price: Current spot price
            outlook: Market outlook (bullish, bearish, neutral, volatile)
            iv_percentile: Current IV percentile (0-100)
        """
        suggestions = []
        
        # Get first available expiry
        expiries = self.scanner.get_expiries(underlying)
        if not expiries:
            return suggestions
        
        # Use 2-3 week expiry if available
        target_dte = 14
        closest_expiry = min(expiries, key=lambda e: abs((datetime.combine(e, datetime.min.time()) - datetime.now()).days - target_dte))
        expiry_dt = datetime.combine(closest_expiry, datetime.min.time())
        
        if outlook == 'bullish':
            # Bull call spread if IV high, long call if IV low
            if iv_percentile > 50:
                spread = self.build_bull_call_spread(
                    underlying,
                    spot_price * 0.98,  # Slightly ITM
                    spot_price * 1.05,  # OTM
                    expiry_dt
                )
                if spread:
                    suggestions.append(spread)
            else:
                # Just get ATM call
                chain = self.scanner.get_chain(underlying, expiry_dt)
                atm_call = min(chain['calls'], key=lambda o: abs(o.strike - spot_price), default=None)
                if atm_call:
                    spread = OptionSpread(
                        spread_type=SpreadType.VERTICAL_CALL,
                        underlying=underlying,
                        legs=[SpreadLeg(option=atm_call, quantity=1)],
                        description="Long ATM Call"
                    )
                    spread.max_loss = atm_call.ask
                    spread.max_profit = float('inf')
                    suggestions.append(spread)
        
        elif outlook == 'bearish':
            spread = self.build_bear_put_spread(
                underlying,
                spot_price * 1.02,  # Slightly ITM
                spot_price * 0.95,  # OTM
                expiry_dt
            )
            if spread:
                suggestions.append(spread)
        
        elif outlook == 'neutral':
            # Iron condor if IV high
            if iv_percentile > 50:
                spread = self.build_iron_condor(underlying, spot_price, expiry=expiry_dt)
                if spread:
                    suggestions.append(spread)
            
            # Short strangle if IV very high
            if iv_percentile > 70:
                spread = self.build_strangle(underlying, spot_price, is_long=False, expiry=expiry_dt)
                if spread:
                    suggestions.append(spread)
        
        elif outlook == 'volatile':
            # Long straddle/strangle if IV low
            if iv_percentile < 50:
                straddle = self.build_straddle(underlying, spot_price, is_long=True, expiry=expiry_dt)
                if straddle:
                    suggestions.append(straddle)
                
                strangle = self.build_strangle(underlying, spot_price, is_long=True, expiry=expiry_dt)
                if strangle:
                    suggestions.append(strangle)
        
        return suggestions


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scanner = OptionsScanner()
    scanner.refresh_quotes()
    
    strategies = SpreadStrategies(scanner)
    
    # Suggest strategies for BTC
    spot = 86000
    print("\n🎯 STRATEGY SUGGESTIONS FOR BTC:")
    
    for outlook in ['bullish', 'bearish', 'neutral', 'volatile']:
        print(f"\n📌 {outlook.upper()} Outlook:")
        suggestions = strategies.suggest_strategy('BTC', spot, outlook=outlook, iv_percentile=60)
        for s in suggestions:
            print(f"  • {s.description}")
            print(f"    Max Profit: ${s.max_profit:.2f}, Max Loss: ${s.max_loss:.2f}")
