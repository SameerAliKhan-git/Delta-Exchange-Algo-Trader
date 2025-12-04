"""
Options Pricing Engine - Production-Grade Implementation
=========================================================
Black-Scholes, Bachelier, and Monte Carlo pricing with full Greeks calculation.
Optimized for crypto options with 24/7 markets and high volatility.

Author: Quant Bot
Version: 1.0.0
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq, newton
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import warnings

logger = logging.getLogger(__name__)


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class PricingModel(Enum):
    BLACK_SCHOLES = "black_scholes"
    BACHELIER = "bachelier"  # Better for crypto near-zero rates
    MONTE_CARLO = "monte_carlo"


@dataclass
class Greeks:
    """Complete Greeks for an option position."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0  # Per day
    vega: float = 0.0   # Per 1% vol move
    rho: float = 0.0    # Per 1% rate move
    vanna: float = 0.0  # d(delta)/d(vol)
    volga: float = 0.0  # d(vega)/d(vol), aka vomma
    charm: float = 0.0  # d(delta)/d(time)
    speed: float = 0.0  # d(gamma)/d(spot)
    color: float = 0.0  # d(gamma)/d(time)
    
    def scale(self, quantity: float) -> 'Greeks':
        """Scale Greeks by position size."""
        return Greeks(
            delta=self.delta * quantity,
            gamma=self.gamma * quantity,
            theta=self.theta * quantity,
            vega=self.vega * quantity,
            rho=self.rho * quantity,
            vanna=self.vanna * quantity,
            volga=self.volga * quantity,
            charm=self.charm * quantity,
            speed=self.speed * quantity,
            color=self.color * quantity
        )
    
    def __add__(self, other: 'Greeks') -> 'Greeks':
        """Aggregate Greeks across positions."""
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            vanna=self.vanna + other.vanna,
            volga=self.volga + other.volga,
            charm=self.charm + other.charm,
            speed=self.speed + other.speed,
            color=self.color + other.color
        )


@dataclass
class OptionPrice:
    """Complete option pricing result."""
    price: float
    intrinsic: float
    extrinsic: float
    greeks: Greeks
    model: PricingModel
    iv: Optional[float] = None
    moneyness: float = 0.0
    time_to_expiry: float = 0.0
    
    @property
    def is_itm(self) -> bool:
        return self.intrinsic > 0
    
    @property
    def is_atm(self) -> bool:
        return abs(self.moneyness - 1.0) < 0.02
    
    @property
    def is_otm(self) -> bool:
        return self.intrinsic == 0


class OptionsPricingEngine:
    """
    Production-grade options pricing engine.
    
    Supports:
    - Black-Scholes (European options)
    - Bachelier (normal model, better for crypto)
    - Monte Carlo (exotic payoffs)
    - Full Greeks calculation (1st, 2nd, 3rd order)
    - IV solving via Brent's method
    """
    
    # Constants
    TRADING_DAYS_PER_YEAR = 365  # Crypto is 24/7
    MIN_TIME_TO_EXPIRY = 1e-10
    MIN_VOLATILITY = 0.001
    MAX_VOLATILITY = 10.0  # 1000% annualized
    IV_TOLERANCE = 1e-8
    IV_MAX_ITERATIONS = 100
    
    def __init__(
        self,
        default_model: PricingModel = PricingModel.BLACK_SCHOLES,
        risk_free_rate: float = 0.0,  # Typically 0 for crypto
        dividend_yield: float = 0.0
    ):
        self.default_model = default_model
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        
        logger.info(f"OptionsPricingEngine initialized with {default_model.value} model")
    
    # ==================== BLACK-SCHOLES ====================
    
    def _d1_d2(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes."""
        if T <= self.MIN_TIME_TO_EXPIRY:
            T = self.MIN_TIME_TO_EXPIRY
        if sigma <= self.MIN_VOLATILITY:
            sigma = self.MIN_VOLATILITY
            
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        return d1, d2
    
    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: OptionType,
        r: Optional[float] = None,
        q: Optional[float] = None
    ) -> float:
        """
        Black-Scholes option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            sigma: Volatility (annualized)
            option_type: CALL or PUT
            r: Risk-free rate (default: instance rate)
            q: Dividend yield (default: instance yield)
        
        Returns:
            Option price
        """
        r = r if r is not None else self.risk_free_rate
        q = q if q is not None else self.dividend_yield
        
        if T <= self.MIN_TIME_TO_EXPIRY:
            # At expiry
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma, q)
        
        if option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
        
        return max(price, 0.0)
    
    def black_scholes_greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: OptionType,
        r: Optional[float] = None,
        q: Optional[float] = None
    ) -> Greeks:
        """
        Calculate all Greeks using Black-Scholes.
        
        Returns full Greeks including second and third order.
        """
        r = r if r is not None else self.risk_free_rate
        q = q if q is not None else self.dividend_yield
        
        if T <= self.MIN_TIME_TO_EXPIRY:
            # At expiry - delta is 0 or 1
            if option_type == OptionType.CALL:
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            return Greeks(delta=delta)
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma, q)
        sqrt_T = np.sqrt(T)
        
        # PDF at d1
        n_d1 = stats.norm.pdf(d1)
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        
        # Discount factors
        exp_qT = np.exp(-q * T)
        exp_rT = np.exp(-r * T)
        
        # ===== First Order Greeks =====
        
        # Delta
        if option_type == OptionType.CALL:
            delta = exp_qT * N_d1
        else:
            delta = exp_qT * (N_d1 - 1)
        
        # Gamma (same for calls and puts)
        gamma = exp_qT * n_d1 / (S * sigma * sqrt_T)
        
        # Theta (per year, we'll convert to per day)
        if option_type == OptionType.CALL:
            theta = (
                -exp_qT * S * n_d1 * sigma / (2 * sqrt_T)
                - r * K * exp_rT * N_d2
                + q * S * exp_qT * N_d1
            )
        else:
            theta = (
                -exp_qT * S * n_d1 * sigma / (2 * sqrt_T)
                + r * K * exp_rT * stats.norm.cdf(-d2)
                - q * S * exp_qT * stats.norm.cdf(-d1)
            )
        theta_daily = theta / self.TRADING_DAYS_PER_YEAR
        
        # Vega (per 1% vol move)
        vega = S * exp_qT * sqrt_T * n_d1 / 100  # Divide by 100 for 1% convention
        
        # Rho (per 1% rate move)
        if option_type == OptionType.CALL:
            rho = K * T * exp_rT * N_d2 / 100
        else:
            rho = -K * T * exp_rT * stats.norm.cdf(-d2) / 100
        
        # ===== Second Order Greeks =====
        
        # Vanna: d(delta)/d(vol) = d(vega)/d(spot)
        vanna = -exp_qT * n_d1 * d2 / sigma
        
        # Volga (Vomma): d(vega)/d(vol)
        volga = vega * d1 * d2 / sigma
        
        # Charm: d(delta)/d(time)
        charm = exp_qT * n_d1 * (
            2 * (r - q) * T - d2 * sigma * sqrt_T
        ) / (2 * T * sigma * sqrt_T)
        if option_type == OptionType.PUT:
            charm = charm + q * exp_qT * stats.norm.cdf(-d1)
        else:
            charm = charm - q * exp_qT * N_d1
        
        # ===== Third Order Greeks =====
        
        # Speed: d(gamma)/d(spot)
        speed = -gamma / S * (1 + d1 / (sigma * sqrt_T))
        
        # Color: d(gamma)/d(time)
        color = -exp_qT * n_d1 / (2 * S * T * sigma * sqrt_T) * (
            2 * q * T + 1 + d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T)
        )
        
        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta_daily,
            vega=vega,
            rho=rho,
            vanna=vanna,
            volga=volga,
            charm=charm,
            speed=speed,
            color=color
        )
    
    # ==================== BACHELIER MODEL ====================
    
    def bachelier_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma_n: float,  # Normal vol (not lognormal)
        option_type: OptionType,
        r: Optional[float] = None
    ) -> float:
        """
        Bachelier (normal) model price.
        Better for crypto when spot can be very volatile.
        
        Args:
            S: Spot price
            K: Strike price  
            T: Time to expiry (years)
            sigma_n: Normal volatility (dollar terms)
            option_type: CALL or PUT
            r: Risk-free rate
        
        Returns:
            Option price
        """
        r = r if r is not None else self.risk_free_rate
        
        if T <= self.MIN_TIME_TO_EXPIRY:
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        sqrt_T = np.sqrt(T)
        d = (S - K) / (sigma_n * sqrt_T)
        
        if option_type == OptionType.CALL:
            price = np.exp(-r * T) * (
                (S - K) * stats.norm.cdf(d) + sigma_n * sqrt_T * stats.norm.pdf(d)
            )
        else:
            price = np.exp(-r * T) * (
                (K - S) * stats.norm.cdf(-d) + sigma_n * sqrt_T * stats.norm.pdf(d)
            )
        
        return max(price, 0.0)
    
    def bachelier_greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma_n: float,
        option_type: OptionType,
        r: Optional[float] = None
    ) -> Greeks:
        """Calculate Greeks under Bachelier model."""
        r = r if r is not None else self.risk_free_rate
        
        if T <= self.MIN_TIME_TO_EXPIRY:
            if option_type == OptionType.CALL:
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            return Greeks(delta=delta)
        
        sqrt_T = np.sqrt(T)
        d = (S - K) / (sigma_n * sqrt_T)
        exp_rT = np.exp(-r * T)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = exp_rT * stats.norm.cdf(d)
        else:
            delta = -exp_rT * stats.norm.cdf(-d)
        
        # Gamma
        gamma = exp_rT * stats.norm.pdf(d) / (sigma_n * sqrt_T)
        
        # Vega (per 1% normal vol move)
        vega = exp_rT * sqrt_T * stats.norm.pdf(d) / 100
        
        # Theta
        price = self.bachelier_price(S, K, T, sigma_n, option_type, r)
        theta = r * price - exp_rT * sigma_n * stats.norm.pdf(d) / (2 * sqrt_T)
        theta_daily = theta / self.TRADING_DAYS_PER_YEAR
        
        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta_daily,
            vega=vega,
            rho=0.0  # Simplified
        )
    
    # ==================== MONTE CARLO ====================
    
    def monte_carlo_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: OptionType,
        r: Optional[float] = None,
        q: Optional[float] = None,
        n_simulations: int = 100000,
        n_steps: int = 252,
        antithetic: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Monte Carlo option pricing.
        
        Returns:
            Tuple of (price, standard_error)
        """
        r = r if r is not None else self.risk_free_rate
        q = q if q is not None else self.dividend_yield
        
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate paths
        if antithetic:
            n_half = n_simulations // 2
            Z = np.random.standard_normal((n_half, n_steps))
            Z = np.concatenate([Z, -Z], axis=0)
        else:
            Z = np.random.standard_normal((n_simulations, n_steps))
        
        # Simulate
        log_returns = drift + diffusion * Z
        log_paths = np.cumsum(log_returns, axis=1)
        S_T = S * np.exp(log_paths[:, -1])
        
        # Calculate payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(n_simulations)
        
        return price, std_error
    
    # ==================== IMPLIED VOLATILITY ====================
    
    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: OptionType,
        r: Optional[float] = None,
        q: Optional[float] = None,
        model: PricingModel = PricingModel.BLACK_SCHOLES,
        initial_guess: float = 0.5
    ) -> Optional[float]:
        """
        Solve for implied volatility using Brent's method.
        
        Args:
            market_price: Observed market price
            S, K, T: Option parameters
            option_type: CALL or PUT
            model: Pricing model to use
            initial_guess: Starting point for solver
        
        Returns:
            Implied volatility or None if solution not found
        """
        r = r if r is not None else self.risk_free_rate
        q = q if q is not None else self.dividend_yield
        
        # Validate inputs
        intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
        if market_price < intrinsic - 1e-10:
            logger.warning(f"Market price {market_price} below intrinsic {intrinsic}")
            return None
        
        if T <= self.MIN_TIME_TO_EXPIRY:
            return None
        
        def objective(sigma: float) -> float:
            if model == PricingModel.BLACK_SCHOLES:
                theoretical = self.black_scholes_price(S, K, T, sigma, option_type, r, q)
            elif model == PricingModel.BACHELIER:
                theoretical = self.bachelier_price(S, K, T, sigma, option_type, r)
            else:
                theoretical, _ = self.monte_carlo_price(
                    S, K, T, sigma, option_type, r, q, n_simulations=10000
                )
            return theoretical - market_price
        
        try:
            # Try Brent's method with wide bounds
            iv = brentq(
                objective,
                self.MIN_VOLATILITY,
                self.MAX_VOLATILITY,
                xtol=self.IV_TOLERANCE,
                maxiter=self.IV_MAX_ITERATIONS
            )
            return iv
        except ValueError:
            # Brent failed, try Newton
            try:
                iv = newton(objective, initial_guess, maxiter=self.IV_MAX_ITERATIONS)
                if self.MIN_VOLATILITY <= iv <= self.MAX_VOLATILITY:
                    return iv
            except (RuntimeError, ValueError):
                pass
        
        logger.warning(f"IV solver failed for K={K}, T={T:.4f}, price={market_price}")
        return None
    
    # ==================== UNIFIED INTERFACE ====================
    
    def price_option(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: OptionType,
        model: Optional[PricingModel] = None,
        r: Optional[float] = None,
        q: Optional[float] = None
    ) -> OptionPrice:
        """
        Unified option pricing interface.
        
        Returns complete OptionPrice with price, Greeks, and metadata.
        """
        model = model or self.default_model
        r = r if r is not None else self.risk_free_rate
        q = q if q is not None else self.dividend_yield
        
        # Calculate price
        if model == PricingModel.BLACK_SCHOLES:
            price = self.black_scholes_price(S, K, T, sigma, option_type, r, q)
            greeks = self.black_scholes_greeks(S, K, T, sigma, option_type, r, q)
        elif model == PricingModel.BACHELIER:
            price = self.bachelier_price(S, K, T, sigma, option_type, r)
            greeks = self.bachelier_greeks(S, K, T, sigma, option_type, r)
        else:
            price, _ = self.monte_carlo_price(S, K, T, sigma, option_type, r, q)
            # Use BS Greeks as approximation for MC
            greeks = self.black_scholes_greeks(S, K, T, sigma, option_type, r, q)
        
        # Calculate intrinsic/extrinsic
        if option_type == OptionType.CALL:
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
        extrinsic = price - intrinsic
        
        # Moneyness
        moneyness = S / K
        
        return OptionPrice(
            price=price,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            greeks=greeks,
            model=model,
            iv=sigma,
            moneyness=moneyness,
            time_to_expiry=T
        )
    
    def price_portfolio(
        self,
        positions: List[Dict],
        S: float,
        sigma: float,
        r: Optional[float] = None,
        q: Optional[float] = None
    ) -> Tuple[float, Greeks]:
        """
        Price a portfolio of options.
        
        Args:
            positions: List of dicts with keys:
                - strike: float
                - expiry: float (years)
                - type: 'call' or 'put'
                - quantity: float (positive=long, negative=short)
            S: Current spot price
            sigma: Volatility (or dict of {(K,T): sigma} for surface)
            r, q: Rates
        
        Returns:
            Tuple of (total_value, aggregated_greeks)
        """
        total_value = 0.0
        total_greeks = Greeks()
        
        for pos in positions:
            K = pos['strike']
            T = pos['expiry']
            opt_type = OptionType.CALL if pos['type'].lower() == 'call' else OptionType.PUT
            qty = pos['quantity']
            
            # Get vol (from surface if dict)
            if isinstance(sigma, dict):
                vol = sigma.get((K, T), list(sigma.values())[0])
            else:
                vol = sigma
            
            result = self.price_option(S, K, T, vol, opt_type, r=r, q=q)
            
            total_value += result.price * qty
            total_greeks = total_greeks + result.greeks.scale(qty)
        
        return total_value, total_greeks


# ==================== CONVENIENCE FUNCTIONS ====================

def calculate_breakeven(
    premium: float,
    strike: float,
    option_type: OptionType,
    quantity: int = 1
) -> float:
    """Calculate breakeven price for an option position."""
    cost_per_contract = premium * abs(quantity)
    
    if quantity > 0:  # Long position
        if option_type == OptionType.CALL:
            return strike + premium
        else:
            return strike - premium
    else:  # Short position
        if option_type == OptionType.CALL:
            return strike + premium
        else:
            return strike - premium


def calculate_max_profit_loss(
    premium: float,
    strike: float,
    option_type: OptionType,
    quantity: int
) -> Tuple[float, float]:
    """
    Calculate max profit and max loss for a position.
    
    Returns:
        Tuple of (max_profit, max_loss) - positive values
    """
    if quantity > 0:  # Long
        max_loss = premium * quantity
        if option_type == OptionType.CALL:
            max_profit = float('inf')
        else:
            max_profit = (strike - premium) * quantity
    else:  # Short
        max_profit = premium * abs(quantity)
        if option_type == OptionType.CALL:
            max_loss = float('inf')
        else:
            max_loss = (strike - premium) * abs(quantity)
    
    return max_profit, max_loss


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Options Pricing Engine")
    parser.add_argument("--spot", type=float, required=True, help="Spot price")
    parser.add_argument("--strike", type=float, required=True, help="Strike price")
    parser.add_argument("--expiry", type=float, required=True, help="Time to expiry (years)")
    parser.add_argument("--vol", type=float, required=True, help="Volatility (annualized)")
    parser.add_argument("--type", choices=["call", "put"], required=True, help="Option type")
    parser.add_argument("--rate", type=float, default=0.0, help="Risk-free rate")
    parser.add_argument("--model", choices=["bs", "bachelier", "mc"], default="bs")
    parser.add_argument("--iv-price", type=float, help="Market price (for IV solve)")
    
    args = parser.parse_args()
    
    engine = OptionsPricingEngine(risk_free_rate=args.rate)
    opt_type = OptionType.CALL if args.type == "call" else OptionType.PUT
    
    model_map = {
        "bs": PricingModel.BLACK_SCHOLES,
        "bachelier": PricingModel.BACHELIER,
        "mc": PricingModel.MONTE_CARLO
    }
    
    if args.iv_price:
        iv = engine.implied_volatility(
            args.iv_price, args.spot, args.strike, args.expiry, opt_type
        )
        print(f"Implied Volatility: {iv:.4%}" if iv else "IV solve failed")
    else:
        result = engine.price_option(
            args.spot, args.strike, args.expiry, args.vol, opt_type,
            model=model_map[args.model]
        )
        
        print(f"\n{'='*50}")
        print(f"OPTIONS PRICING RESULT")
        print(f"{'='*50}")
        print(f"Model: {result.model.value}")
        print(f"Price: ${result.price:.4f}")
        print(f"Intrinsic: ${result.intrinsic:.4f}")
        print(f"Extrinsic: ${result.extrinsic:.4f}")
        print(f"Moneyness: {result.moneyness:.4f}")
        print(f"\nGreeks:")
        print(f"  Delta: {result.greeks.delta:.4f}")
        print(f"  Gamma: {result.greeks.gamma:.6f}")
        print(f"  Theta: ${result.greeks.theta:.4f}/day")
        print(f"  Vega:  ${result.greeks.vega:.4f}/1%vol")
        print(f"  Vanna: {result.greeks.vanna:.6f}")
        print(f"  Volga: {result.greeks.volga:.6f}")
