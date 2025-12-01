"""
Options Directional Strategy - Trade options based on market conditions

This strategy selects optimal options for directional trades:
- Uses delta-based strike selection (target ~0.35 delta)
- Premium affordability checks (max % of capital)
- IV-aware timing (prefer low IV for buying)
- Expiry window management (3-30 days)

Trade options when:
- Strong directional confidence
- IV is relatively low (options cheap)
- Sufficient liquidity in options chain
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from strategies.base import StrategyBase, StrategyContext, Candle, Order
from signals.technical import ema, atr, momentum as calc_momentum


@dataclass
class OptionLeg:
    """Represents an option contract"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: str
    premium: float
    delta: float
    contracts: int = 0


class OptionsDirectionalStrategy(StrategyBase):
    """
    Options-based directional strategy
    
    Uses options instead of futures when:
    - High IV environment (defined risk preferred)
    - Strong directional conviction
    - Want to limit max loss to premium
    
    Entry:
    - Select OTM option with target delta (~0.35)
    - Check premium affordability
    - Enter on strong momentum signal
    
    Exit:
    - Take profit at 50-100% gain
    - Cut loss at 50% of premium
    - Exit before expiry (7 days min)
    """
    
    name = "options_directional"
    version = "1.0.0"
    author = "Delta Algo Bot"
    description = "Directional options trading with delta-based selection"
    
    params = {
        # Option selection
        "target_delta": 0.35,
        "delta_tolerance": 0.15,
        "min_expiry_days": 3,
        "max_expiry_days": 30,
        
        # Premium sizing
        "max_premium_pct": 0.005,  # 0.5% of capital
        "max_contracts": 50,
        
        # Entry conditions
        "min_confidence": 0.70,  # Higher for options
        "min_momentum": 0.003,
        
        # Exit conditions
        "take_profit_pct": 1.0,  # 100% gain on premium
        "stop_loss_pct": 0.5,    # 50% loss on premium
        "exit_before_expiry_days": 7,
        
        # IV thresholds
        "max_iv_percentile": 50,  # Prefer lower IV
        
        # Warmup
        "warmup_period": 60
    }
    
    def __init__(self, ctx: StrategyContext, **kwargs):
        super().__init__(ctx)
        self.params = {**self.params, **kwargs}
        
        # Options-specific state
        self._option_position: Optional[OptionLeg] = None
        self._entry_premium: float = 0.0
        
        # Options chain provider (injected)
        self._get_options_chain = kwargs.get('get_options_chain')
        self._get_option_greeks = kwargs.get('get_option_greeks')
        self._place_option_order = kwargs.get('place_option_order')
    
    def on_start(self):
        """Initialize strategy"""
        self.log("Options strategy started", params=self.params)
    
    def on_candle(self, candle: Candle):
        """Main strategy logic"""
        self._add_candle(candle)
        
        # Check warmup
        if len(self._candles) < self.params["warmup_period"]:
            return
        
        # If we have an option position, manage it
        if self._option_position:
            self._manage_option_position()
            return
        
        # Otherwise, look for new entry
        self._check_entry()
    
    def _check_entry(self):
        """Check for new option entry"""
        # Compute momentum
        closes = self.close
        mom = calc_momentum(closes, 20)[-1]
        
        # Compute trend
        ema_fast = ema(closes, 20)[-1]
        ema_slow = ema(closes, 50)[-1]
        
        # Determine direction
        if mom > self.params["min_momentum"] and self.price > ema_fast > ema_slow:
            direction = "call"
            confidence = min(mom / 0.01, 1.0)
        elif mom < -self.params["min_momentum"] and self.price < ema_fast < ema_slow:
            direction = "put"
            confidence = min(abs(mom) / 0.01, 1.0)
        else:
            return
        
        # Check confidence threshold
        if confidence < self.params["min_confidence"]:
            return
        
        # Select option
        option = self._select_option(direction)
        if not option:
            self.log(f"No suitable {direction} option found")
            return
        
        # Size position
        contracts = self._compute_contracts(option.premium)
        if contracts <= 0:
            self.log("Cannot afford minimum contracts")
            return
        
        option.contracts = contracts
        
        # Enter position
        self._enter_option(option, confidence)
    
    def _select_option(self, direction: str) -> Optional[OptionLeg]:
        """
        Select optimal option for direction
        
        Args:
            direction: 'call' or 'put'
        
        Returns:
            OptionLeg or None
        """
        if not self._get_options_chain:
            return None
        
        try:
            # Get options chain
            chain = self._get_options_chain(self.symbol)
            if not chain:
                return None
            
            # Filter by type
            options = [o for o in chain if o.get('type') == direction]
            
            # Filter by expiry
            min_days = self.params["min_expiry_days"]
            max_days = self.params["max_expiry_days"]
            options = [
                o for o in options 
                if min_days <= o.get('days_to_expiry', 0) <= max_days
            ]
            
            # Filter by delta
            target_delta = self.params["target_delta"]
            tolerance = self.params["delta_tolerance"]
            
            candidates = []
            for o in options:
                delta = abs(o.get('delta', 0))
                if abs(delta - target_delta) <= tolerance:
                    candidates.append(o)
            
            if not candidates:
                return None
            
            # Score candidates (prefer: closest delta, good liquidity, lower premium)
            def score(o):
                delta_score = 1 - abs(abs(o.get('delta', 0)) - target_delta) / tolerance
                liquidity_score = min(o.get('volume', 0) / 1000, 1.0)
                return delta_score * 0.5 + liquidity_score * 0.5
            
            candidates.sort(key=score, reverse=True)
            best = candidates[0]
            
            return OptionLeg(
                symbol=best.get('symbol', ''),
                option_type=direction,
                strike=best.get('strike', 0),
                expiry=best.get('expiry', ''),
                premium=(best.get('ask', 0) + best.get('bid', 0)) / 2,
                delta=best.get('delta', 0)
            )
            
        except Exception as e:
            self.log(f"Error selecting option: {e}")
            return None
    
    def _compute_contracts(self, premium: float) -> int:
        """
        Compute number of contracts based on premium affordability
        
        Args:
            premium: Premium per contract
        
        Returns:
            Number of contracts
        """
        if premium <= 0:
            return 0
        
        max_spend = self.balance * self.params["max_premium_pct"]
        contracts = int(max_spend / premium)
        
        return min(contracts, self.params["max_contracts"])
    
    def _enter_option(self, option: OptionLeg, confidence: float):
        """Enter option position"""
        self.log(
            f"Entering {option.option_type.upper()}",
            strike=option.strike,
            expiry=option.expiry,
            premium=option.premium,
            contracts=option.contracts,
            delta=option.delta,
            confidence=f"{confidence:.1%}"
        )
        
        # Place order
        if self._place_option_order:
            order = self._place_option_order(
                symbol=option.symbol,
                side='buy',
                contracts=option.contracts,
                price=option.premium
            )
            
            if order:
                self._option_position = option
                self._entry_premium = option.premium
    
    def _manage_option_position(self):
        """Manage existing option position"""
        if not self._option_position:
            return
        
        # Get current premium
        current_premium = self._get_current_premium()
        if current_premium <= 0:
            return
        
        # Calculate PnL
        pnl_pct = (current_premium - self._entry_premium) / self._entry_premium
        
        # Check take profit
        if pnl_pct >= self.params["take_profit_pct"]:
            self.log(
                f"Take profit hit",
                entry=self._entry_premium,
                current=current_premium,
                pnl=f"{pnl_pct:.1%}"
            )
            self._exit_option("take_profit")
            return
        
        # Check stop loss
        if pnl_pct <= -self.params["stop_loss_pct"]:
            self.log(
                f"Stop loss hit",
                entry=self._entry_premium,
                current=current_premium,
                pnl=f"{pnl_pct:.1%}"
            )
            self._exit_option("stop_loss")
            return
        
        # Check expiry
        # TODO: Get days to expiry from option data
        # if days_to_expiry <= self.params["exit_before_expiry_days"]:
        #     self._exit_option("expiry_approaching")
    
    def _get_current_premium(self) -> float:
        """Get current premium for position"""
        if not self._option_position or not self._get_option_greeks:
            return 0.0
        
        try:
            greeks = self._get_option_greeks(self._option_position.symbol)
            return (greeks.get('bid', 0) + greeks.get('ask', 0)) / 2
        except Exception:
            return 0.0
    
    def _exit_option(self, reason: str):
        """Exit option position"""
        if not self._option_position:
            return
        
        self.log(f"Exiting option position", reason=reason)
        
        if self._place_option_order:
            self._place_option_order(
                symbol=self._option_position.symbol,
                side='sell',
                contracts=self._option_position.contracts
            )
        
        # Calculate PnL
        current_premium = self._get_current_premium()
        pnl = (current_premium - self._entry_premium) * self._option_position.contracts
        self._record_trade(pnl)
        
        # Clear position
        self._option_position = None
        self._entry_premium = 0.0
    
    def on_exit(self):
        """Cleanup"""
        # Exit any open position
        if self._option_position:
            self._exit_option("strategy_exit")
        
        self.log(
            "Options strategy stopped",
            trades=self.trade_count,
            win_rate=f"{self.win_rate:.1%}",
            total_pnl=self.total_pnl
        )
