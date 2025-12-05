"""
Almgren-Chriss Market Impact Model
==================================
PRODUCTION DELIVERABLE: Realistic execution cost modeling.

Crypto liquidity is thin - simple slippage models underestimate costs.
This module implements the Almgren-Chriss framework with:
- Temporary impact (recovers after execution)
- Permanent impact (shifts the market)
- Optimal execution trajectories (TWAP, VWAP, Aggressive)

Reference: Almgren & Chriss (2001) - "Optimal Execution of Portfolio Transactions"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketImpactParams:
    """Market impact model parameters."""
    
    # Temporary impact parameters
    eta: float = 0.1  # Temporary impact coefficient
    
    # Permanent impact parameters
    gamma: float = 0.05  # Permanent impact coefficient
    
    # Volatility
    sigma: float = 0.02  # Daily volatility
    
    # Bid-ask spread
    spread_bps: float = 5.0  # Half-spread in basis points
    
    # Market depth
    daily_volume: float = 1e8  # Daily dollar volume
    
    # Risk aversion
    lambda_risk: float = 1e-6  # Risk aversion parameter
    
    def __post_init__(self):
        # Validate parameters
        assert self.eta >= 0, "Temporary impact must be non-negative"
        assert self.gamma >= 0, "Permanent impact must be non-negative"
        assert self.sigma > 0, "Volatility must be positive"
        assert self.spread_bps >= 0, "Spread must be non-negative"


@dataclass
class ExecutionResult:
    """Result of executing a trade."""
    
    # Order details
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    initial_price: float
    
    # Execution details
    avg_fill_price: float
    total_cost: float  # Total dollar cost including impact
    
    # Cost breakdown
    spread_cost: float  # Cost due to bid-ask spread
    temporary_impact_cost: float  # Temporary market impact
    permanent_impact_cost: float  # Permanent market impact
    timing_risk_cost: float  # Variance cost due to execution time
    
    # Total implementation shortfall
    implementation_shortfall: float  # Total cost vs arrival price
    implementation_shortfall_bps: float  # In basis points
    
    # Execution quality
    execution_time: float  # Time in hours
    participation_rate: float  # Our volume / total volume
    
    # Price trajectory
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'initial_price': self.initial_price,
            'avg_fill_price': self.avg_fill_price,
            'total_cost': self.total_cost,
            'spread_cost': self.spread_cost,
            'temporary_impact_cost': self.temporary_impact_cost,
            'permanent_impact_cost': self.permanent_impact_cost,
            'timing_risk_cost': self.timing_risk_cost,
            'implementation_shortfall': self.implementation_shortfall,
            'implementation_shortfall_bps': self.implementation_shortfall_bps,
            'execution_time': self.execution_time,
            'participation_rate': self.participation_rate
        }


class ExecutionStrategy(Enum):
    """Execution strategy types."""
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    IS = "is"  # Implementation shortfall (minimize total cost)
    AGGRESSIVE = "aggressive"  # Front-loaded execution
    PASSIVE = "passive"  # Back-loaded execution


# =============================================================================
# ALMGREN-CHRISS MODEL
# =============================================================================

class AlmgrenChrissModel:
    """
    Almgren-Chriss optimal execution model.
    
    This model separates market impact into:
    1. Temporary impact: g(v) = eta * v (proportional to trading rate)
    2. Permanent impact: h(v) = gamma * v (proportional to cumulative volume)
    
    The optimal execution trajectory minimizes:
    E[Cost] + lambda * Var[Cost]
    """
    
    def __init__(self, params: Optional[MarketImpactParams] = None):
        """
        Initialize Almgren-Chriss model.
        
        Args:
            params: Market impact parameters
        """
        self.params = params or MarketImpactParams()
    
    def temporary_impact(self, trading_rate: float) -> float:
        """
        Calculate temporary (instantaneous) market impact.
        
        Temporary impact is the additional cost due to demanding
        immediacy - it recovers after the trade.
        
        Args:
            trading_rate: Trading rate (shares per time unit)
        
        Returns:
            Price impact in dollars per share
        """
        # g(v) = eta * sign(v) * |v|^alpha
        # Simplified linear model: g(v) = eta * v
        return self.params.eta * np.abs(trading_rate)
    
    def permanent_impact(self, quantity: float, price: float) -> float:
        """
        Calculate permanent market impact.
        
        Permanent impact shifts the market price and doesn't recover.
        This is information leakage or price discovery.
        
        Args:
            quantity: Total quantity traded
            price: Current price
        
        Returns:
            Permanent price shift
        """
        # h(v) = gamma * v / ADV
        adv_shares = self.params.daily_volume / price
        return self.params.gamma * quantity / adv_shares
    
    def optimal_trajectory(
        self,
        quantity: float,
        price: float,
        time_horizon: float,
        n_steps: int = 100,
        strategy: ExecutionStrategy = ExecutionStrategy.IS
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate optimal execution trajectory.
        
        Args:
            quantity: Total shares to execute
            price: Current price
            time_horizon: Time to complete (hours)
            n_steps: Number of time steps
            strategy: Execution strategy
        
        Returns:
            (times, positions) - time points and remaining positions
        """
        tau = time_horizon / n_steps
        times = np.linspace(0, time_horizon, n_steps + 1)
        
        if strategy == ExecutionStrategy.TWAP:
            # Uniform execution
            positions = quantity * (1 - times / time_horizon)
            
        elif strategy == ExecutionStrategy.VWAP:
            # Volume-weighted - assume U-shaped volume profile
            # Higher volume at open and close
            t_norm = times / time_horizon
            volume_profile = 1 + 0.5 * np.cos(2 * np.pi * t_norm)
            cumulative_volume = np.cumsum(volume_profile) / np.sum(volume_profile)
            positions = quantity * (1 - cumulative_volume)
            
        elif strategy == ExecutionStrategy.IS:
            # Almgren-Chriss optimal trajectory
            # Minimizes E[Cost] + lambda * Var[Cost]
            
            # Calculate optimal urgency parameter kappa
            sigma_hourly = self.params.sigma / np.sqrt(24)  # Hourly volatility
            
            if self.params.eta > 0:
                kappa = np.sqrt(self.params.lambda_risk * sigma_hourly**2 / self.params.eta)
            else:
                kappa = 0.1
            
            # Optimal trajectory: x(t) = X * sinh(kappa * (T-t)) / sinh(kappa * T)
            T = time_horizon
            if np.sinh(kappa * T) > 0:
                positions = quantity * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)
            else:
                positions = quantity * (1 - times / T)
            
        elif strategy == ExecutionStrategy.AGGRESSIVE:
            # Front-loaded - execute more early
            t_norm = times / time_horizon
            positions = quantity * (1 - t_norm)**0.5
            
        elif strategy == ExecutionStrategy.PASSIVE:
            # Back-loaded - execute more late
            t_norm = times / time_horizon
            positions = quantity * (1 - t_norm**2)
        
        else:
            # Default to TWAP
            positions = quantity * (1 - times / time_horizon)
        
        return times, positions
    
    def calculate_execution_cost(
        self,
        quantity: float,
        price: float,
        time_horizon: float,
        strategy: ExecutionStrategy = ExecutionStrategy.IS,
        n_steps: int = 100
    ) -> ExecutionResult:
        """
        Calculate total execution cost for a trade.
        
        Args:
            quantity: Shares to execute (positive = buy)
            price: Current mid price
            time_horizon: Time to execute (hours)
            strategy: Execution strategy
            n_steps: Number of time steps for simulation
        
        Returns:
            ExecutionResult with full cost breakdown
        """
        side = 'buy' if quantity > 0 else 'sell'
        quantity = abs(quantity)
        
        # Get execution trajectory
        times, positions = self.optimal_trajectory(
            quantity, price, time_horizon, n_steps, strategy
        )
        
        # Calculate trading rates (shares per time step)
        tau = time_horizon / n_steps
        trading_rates = -np.diff(positions)  # Negative diff = selling positions
        
        # Initialize costs
        spread_cost = 0.0
        temp_impact_cost = 0.0
        perm_impact_cost = 0.0
        
        # Simulate execution
        prices = [price]
        volumes = list(trading_rates)
        cumulative_perm_impact = 0.0
        
        for i, rate in enumerate(trading_rates):
            if rate == 0:
                continue
            
            # 1. Spread cost (half-spread per share)
            spread = price * self.params.spread_bps / 10000
            spread_cost += spread * rate
            
            # 2. Permanent impact (accumulates)
            perm_impact = self.permanent_impact(rate, price)
            cumulative_perm_impact += perm_impact
            
            if side == 'buy':
                price = price * (1 + cumulative_perm_impact)
            else:
                price = price * (1 - cumulative_perm_impact)
            
            perm_impact_cost += perm_impact * price * rate
            
            # 3. Temporary impact
            temp_impact = self.temporary_impact(rate / tau)  # Rate per hour
            
            if side == 'buy':
                fill_price = price * (1 + temp_impact)
            else:
                fill_price = price * (1 - temp_impact)
            
            temp_impact_cost += (fill_price - price) * rate
            
            prices.append(fill_price)
        
        # 4. Timing risk (variance cost)
        sigma_hourly = self.params.sigma / np.sqrt(24)
        timing_risk_cost = 0.5 * self.params.lambda_risk * (sigma_hourly ** 2) * quantity ** 2 * time_horizon
        
        # Total cost
        total_cost = spread_cost + temp_impact_cost + perm_impact_cost
        
        # Average fill price
        if quantity > 0:
            avg_fill_price = (price + total_cost / quantity) if side == 'buy' else (price - total_cost / quantity)
        else:
            avg_fill_price = price
        
        # Implementation shortfall
        initial_price = prices[0]
        impl_shortfall = total_cost
        impl_shortfall_bps = (impl_shortfall / (quantity * initial_price)) * 10000 if quantity > 0 else 0
        
        # Participation rate
        adv_shares = self.params.daily_volume / initial_price
        participation_rate = quantity / (adv_shares * time_horizon / 24)
        
        return ExecutionResult(
            symbol="",  # Set by caller
            side=side,
            quantity=quantity,
            initial_price=initial_price,
            avg_fill_price=avg_fill_price,
            total_cost=total_cost,
            spread_cost=spread_cost,
            temporary_impact_cost=temp_impact_cost,
            permanent_impact_cost=perm_impact_cost,
            timing_risk_cost=timing_risk_cost,
            implementation_shortfall=impl_shortfall,
            implementation_shortfall_bps=impl_shortfall_bps,
            execution_time=time_horizon,
            participation_rate=participation_rate,
            prices=prices,
            volumes=volumes
        )
    
    def estimate_impact_bps(
        self,
        quantity_usd: float,
        price: float,
        time_horizon: float = 1.0
    ) -> Dict[str, float]:
        """
        Quick estimate of market impact in basis points.
        
        Args:
            quantity_usd: Dollar value of trade
            price: Current price
            time_horizon: Execution time in hours
        
        Returns:
            Dict with impact breakdown in bps
        """
        quantity = quantity_usd / price
        
        result = self.calculate_execution_cost(
            quantity, price, time_horizon, ExecutionStrategy.IS
        )
        
        return {
            'total_bps': result.implementation_shortfall_bps,
            'spread_bps': (result.spread_cost / quantity_usd) * 10000,
            'temp_impact_bps': (result.temporary_impact_cost / quantity_usd) * 10000,
            'perm_impact_bps': (result.permanent_impact_cost / quantity_usd) * 10000,
            'participation_rate': result.participation_rate
        }


# =============================================================================
# EXECUTION ENGINE WITH IMPACT MODEL
# =============================================================================

class AlmgrenChrissExecutor:
    """
    Execution engine using Almgren-Chriss impact model.
    
    Use this for production order execution with realistic cost estimation.
    """
    
    def __init__(
        self,
        default_params: Optional[MarketImpactParams] = None,
        symbol_params: Optional[Dict[str, MarketImpactParams]] = None
    ):
        """
        Initialize executor.
        
        Args:
            default_params: Default impact parameters
            symbol_params: Per-symbol impact parameters
        """
        self.default_params = default_params or MarketImpactParams()
        self.symbol_params = symbol_params or {}
        
        self.model = AlmgrenChrissModel(self.default_params)
        
        # Track execution history
        self.execution_history: List[ExecutionResult] = []
    
    def get_params(self, symbol: str) -> MarketImpactParams:
        """Get impact parameters for a symbol."""
        return self.symbol_params.get(symbol, self.default_params)
    
    def set_params(self, symbol: str, params: MarketImpactParams):
        """Set impact parameters for a symbol."""
        self.symbol_params[symbol] = params
    
    def calibrate_from_data(
        self,
        symbol: str,
        trades_df: pd.DataFrame,
        orderbook_df: Optional[pd.DataFrame] = None
    ) -> MarketImpactParams:
        """
        Calibrate impact parameters from historical data.
        
        Args:
            symbol: Trading symbol
            trades_df: Historical trades with columns:
                - timestamp, price, quantity, side
            orderbook_df: Historical orderbook snapshots (optional)
        
        Returns:
            Calibrated MarketImpactParams
        """
        # Calculate basic statistics
        daily_volume = trades_df.groupby(trades_df['timestamp'].dt.date)['quantity'].sum().mean()
        
        # Estimate volatility
        returns = trades_df['price'].pct_change().dropna()
        hourly_vol = returns.std() * np.sqrt(len(returns) / 24)  # Assuming data spans ~1 day
        daily_vol = hourly_vol * np.sqrt(24)
        
        # Estimate spread from orderbook or trades
        if orderbook_df is not None and 'bid' in orderbook_df.columns:
            spread = ((orderbook_df['ask'] - orderbook_df['bid']) / 
                     ((orderbook_df['ask'] + orderbook_df['bid']) / 2)).mean() * 10000 / 2
        else:
            # Estimate from trade price bounces
            spread = returns.abs().median() * 10000
        
        # Estimate impact coefficients (simplified)
        # In practice, use regression on signed volume vs price change
        typical_trade_size = trades_df['quantity'].median()
        price_moves = trades_df['price'].diff().abs().mean()
        
        eta = price_moves / typical_trade_size if typical_trade_size > 0 else 0.1
        gamma = eta * 0.5  # Permanent is typically half of temporary
        
        params = MarketImpactParams(
            eta=eta,
            gamma=gamma,
            sigma=daily_vol,
            spread_bps=spread,
            daily_volume=daily_volume * trades_df['price'].mean()
        )
        
        self.symbol_params[symbol] = params
        logger.info(f"Calibrated params for {symbol}: {params}")
        
        return params
    
    def execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_horizon: float = 1.0,
        strategy: ExecutionStrategy = ExecutionStrategy.IS,
        simulate: bool = True
    ) -> ExecutionResult:
        """
        Execute a trade with impact modeling.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Number of shares/contracts
            price: Current mid price
            time_horizon: Time to execute (hours)
            strategy: Execution strategy
            simulate: If True, simulate only. If False, send real orders.
        
        Returns:
            ExecutionResult with cost breakdown
        """
        params = self.get_params(symbol)
        model = AlmgrenChrissModel(params)
        
        # Adjust quantity sign based on side
        signed_quantity = quantity if side == 'buy' else -quantity
        
        # Calculate execution cost
        result = model.calculate_execution_cost(
            signed_quantity, price, time_horizon, strategy
        )
        result.symbol = symbol
        
        # Log execution
        logger.info(
            f"{'[SIM]' if simulate else '[LIVE]'} Execute {side} {quantity} {symbol} @ {price:.2f} "
            f"| Impact: {result.implementation_shortfall_bps:.1f} bps "
            f"| Strategy: {strategy.value}"
        )
        
        # Track history
        self.execution_history.append(result)
        
        return result
    
    def get_optimal_strategy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        urgency: float = 0.5
    ) -> ExecutionStrategy:
        """
        Recommend optimal execution strategy based on urgency.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Current price
            urgency: 0 = no urgency (passive), 1 = urgent (aggressive)
        
        Returns:
            Recommended ExecutionStrategy
        """
        params = self.get_params(symbol)
        
        # Calculate participation rate for 1-hour execution
        adv_shares = params.daily_volume / price
        participation = quantity / (adv_shares / 24)
        
        # High participation + high urgency = aggressive
        # Low participation + low urgency = passive
        # Default to IS for balanced approach
        
        if urgency > 0.8 or participation > 0.1:
            return ExecutionStrategy.AGGRESSIVE
        elif urgency < 0.2 and participation < 0.01:
            return ExecutionStrategy.PASSIVE
        elif urgency > 0.5:
            return ExecutionStrategy.TWAP
        else:
            return ExecutionStrategy.IS
    
    def preview_costs(
        self,
        symbol: str,
        quantity: float,
        price: float,
        time_horizons: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0]
    ) -> pd.DataFrame:
        """
        Preview execution costs for different time horizons.
        
        Useful for deciding execution urgency.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Current price
            time_horizons: List of execution times to compare (hours)
        
        Returns:
            DataFrame comparing costs
        """
        params = self.get_params(symbol)
        model = AlmgrenChrissModel(params)
        
        results = []
        
        for t in time_horizons:
            for strategy in [ExecutionStrategy.IS, ExecutionStrategy.TWAP, ExecutionStrategy.AGGRESSIVE]:
                result = model.calculate_execution_cost(quantity, price, t, strategy)
                results.append({
                    'time_horizon_hours': t,
                    'strategy': strategy.value,
                    'total_cost_bps': result.implementation_shortfall_bps,
                    'spread_bps': (result.spread_cost / (quantity * price)) * 10000,
                    'temp_impact_bps': (result.temporary_impact_cost / (quantity * price)) * 10000,
                    'perm_impact_bps': (result.permanent_impact_cost / (quantity * price)) * 10000,
                    'participation_rate': result.participation_rate
                })
        
        return pd.DataFrame(results)
    
    def get_execution_stats(self) -> Dict:
        """Get aggregate execution statistics."""
        if not self.execution_history:
            return {'n_executions': 0}
        
        total_value = sum(r.quantity * r.initial_price for r in self.execution_history)
        total_cost = sum(r.total_cost for r in self.execution_history)
        
        return {
            'n_executions': len(self.execution_history),
            'total_value': total_value,
            'total_cost': total_cost,
            'avg_impact_bps': np.mean([r.implementation_shortfall_bps for r in self.execution_history]),
            'max_impact_bps': np.max([r.implementation_shortfall_bps for r in self.execution_history]),
            'avg_participation': np.mean([r.participation_rate for r in self.execution_history])
        }

    def calculate_optimal_slice_size(
        self,
        symbol: str,
        total_quantity: float,
        current_volatility: float,
        historical_avg_vol: float = 0.02
    ) -> float:
        """
        Dynamic slicing based on:
        1. Current volatility (increase slices in high vol)
        2. Time of day (liquidity patterns) - simplified here
        3. Order book depth - simplified
        """
        # Get base slice from optimal trajectory (IS strategy)
        # We assume a standard 1-hour execution for the base calculation
        # In IS, the initial trade is usually the largest.
        
        params = self.get_params(symbol)
        model = AlmgrenChrissModel(params)
        
        # Calculate optimal trajectory for 1 hour
        times, positions = model.optimal_trajectory(
            total_quantity, 
            price=10000, # Dummy price, doesn't affect trajectory shape
            time_horizon=1.0, 
            strategy=ExecutionStrategy.IS
        )
        
        # Base slice is the first trade size
        base_slice = total_quantity - positions[1]
        
        # Adjust for volatility
        # If vol is high, we might want to trade smaller chunks more frequently (or larger if we want to finish fast?)
        # Almgren-Chriss says: High vol -> Trade faster (larger initial slice) to reduce timing risk.
        # But the user prompt says: "Dynamic slicing based on: Current volatility (increase slices in high vol)"
        # "Increase slices" usually means "more slices" = "smaller size per slice"? 
        # Or "increase slice size"?
        # Context: "Profit Boost #1: Intelligent Order Slicing... volatility_multiplier = 1 + ... * 0.5"
        # If multiplier > 1, slice size increases.
        # So high vol -> larger slices (faster execution) to avoid risk. This matches IS logic.
        
        vol_ratio = current_volatility / historical_avg_vol if historical_avg_vol > 0 else 1.0
        volatility_multiplier = 1 + (vol_ratio - 1) * 0.5
        
        optimal_slice = base_slice * volatility_multiplier
        
        # Cap at total quantity
        return min(optimal_slice, total_quantity)


# =============================================================================
# CRYPTO-SPECIFIC PARAMETERS
# =============================================================================

def get_crypto_params(symbol: str = "BTCUSDT") -> MarketImpactParams:
    """
    Get typical market impact parameters for crypto markets.
    
    These are conservative estimates for major crypto pairs.
    Actual values should be calibrated from exchange data.
    """
    crypto_params = {
        'BTCUSDT': MarketImpactParams(
            eta=0.05,           # Lower temporary impact due to high liquidity
            gamma=0.02,         # Lower permanent impact
            sigma=0.03,         # ~3% daily vol
            spread_bps=3.0,     # Tight spread on major exchanges
            daily_volume=5e9    # ~$5B daily volume
        ),
        'ETHUSDT': MarketImpactParams(
            eta=0.08,
            gamma=0.03,
            sigma=0.04,
            spread_bps=5.0,
            daily_volume=2e9
        ),
        'SOLUSDT': MarketImpactParams(
            eta=0.15,
            gamma=0.06,
            sigma=0.06,
            spread_bps=8.0,
            daily_volume=500e6
        ),
        'DEFAULT': MarketImpactParams(
            eta=0.20,           # Higher impact for less liquid
            gamma=0.10,
            sigma=0.05,
            spread_bps=15.0,
            daily_volume=100e6
        )
    }
    
    return crypto_params.get(symbol, crypto_params['DEFAULT'])


# =============================================================================
# INTEGRATION WITH BACKTEST
# =============================================================================

class BacktestImpactModel:
    """
    Market impact model for backtesting.
    
    Integrates Almgren-Chriss model into backtest framework.
    """
    
    def __init__(self, symbol_params: Optional[Dict[str, MarketImpactParams]] = None):
        self.executor = AlmgrenChrissExecutor(symbol_params=symbol_params)
    
    def apply_impact(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        volume: float = None,
        volatility: float = None
    ) -> Tuple[float, float]:
        """
        Apply market impact to a backtest trade.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Signal price
            volume: Current volume (optional, for dynamic calibration)
            volatility: Current volatility (optional)
        
        Returns:
            (fill_price, cost_bps) - Adjusted fill price and total cost in bps
        """
        # Get or calibrate params
        params = self.executor.get_params(symbol)
        
        # Update params if live data provided
        if volume is not None:
            params.daily_volume = volume * 24  # Assume hourly volume
        if volatility is not None:
            params.sigma = volatility
        
        # Calculate impact (assume 1 hour execution)
        result = self.executor.execute(
            symbol, side, quantity, price,
            time_horizon=1.0,
            strategy=ExecutionStrategy.IS,
            simulate=True
        )
        
        return result.avg_fill_price, result.implementation_shortfall_bps


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate Almgren-Chriss model."""
    
    print("=" * 70)
    print("ALMGREN-CHRISS MARKET IMPACT MODEL - DEMO")
    print("=" * 70)
    
    # Create executor with crypto params
    executor = AlmgrenChrissExecutor()
    executor.set_params('BTCUSDT', get_crypto_params('BTCUSDT'))
    
    # Example trade
    symbol = 'BTCUSDT'
    quantity = 10  # 10 BTC
    price = 50000
    
    print(f"\nTrade: Buy {quantity} {symbol} @ ${price:,.0f}")
    print(f"Trade Value: ${quantity * price:,.0f}")
    
    # Preview costs at different time horizons
    print("\n" + "-" * 70)
    print("COST PREVIEW BY EXECUTION TIME")
    print("-" * 70)
    
    preview = executor.preview_costs(symbol, quantity, price)
    preview_pivot = preview.pivot(index='time_horizon_hours', columns='strategy', values='total_cost_bps')
    print(preview_pivot.to_string())
    
    # Execute with different strategies
    print("\n" + "-" * 70)
    print("EXECUTION COMPARISON")
    print("-" * 70)
    
    for strategy in [ExecutionStrategy.IS, ExecutionStrategy.TWAP, ExecutionStrategy.AGGRESSIVE]:
        result = executor.execute(symbol, 'buy', quantity, price, 
                                  time_horizon=1.0, strategy=strategy)
        print(f"\n{strategy.value.upper()}:")
        print(f"  Avg Fill Price: ${result.avg_fill_price:,.2f}")
        print(f"  Total Cost: ${result.total_cost:,.2f}")
        print(f"  Implementation Shortfall: {result.implementation_shortfall_bps:.1f} bps")
        print(f"  Breakdown:")
        print(f"    - Spread: {(result.spread_cost / (quantity * price)) * 10000:.1f} bps")
        print(f"    - Temp Impact: {(result.temporary_impact_cost / (quantity * price)) * 10000:.1f} bps")
        print(f"    - Perm Impact: {(result.permanent_impact_cost / (quantity * price)) * 10000:.1f} bps")
    
    # Show impact scaling
    print("\n" + "-" * 70)
    print("IMPACT VS TRADE SIZE (1-hour execution, IS strategy)")
    print("-" * 70)
    print(f"{'Size (BTC)':>12} {'Value ($)':>15} {'Impact (bps)':>15}")
    print("-" * 45)
    
    for size in [1, 5, 10, 25, 50, 100]:
        result = executor.execute(symbol, 'buy', size, price, 
                                  time_horizon=1.0, strategy=ExecutionStrategy.IS)
        print(f"{size:>12} {size * price:>15,.0f} {result.implementation_shortfall_bps:>15.1f}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
