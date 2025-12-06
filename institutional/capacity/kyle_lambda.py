"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         KYLE LAMBDA ESTIMATOR - Market Impact Modeling                        ║
║                                                                               ║
║  Estimates market impact coefficient for capacity management                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Kyle's Lambda (λ) measures the price impact of trading:
    Δprice = λ × signed_volume

Higher λ means more market impact per unit traded.
This is critical for:
1. Scaling strategy capacity
2. Predicting execution slippage
3. Throttling during illiquid periods

Goal: Keep market impact < 20% of expected alpha
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
from scipy import stats
import logging

logger = logging.getLogger("Capacity.KyleLambda")


@dataclass
class LambdaEstimate:
    """Kyle lambda estimate for a symbol."""
    symbol: str
    lambda_value: float
    lambda_std: float
    r_squared: float
    sample_size: int
    timestamp: datetime
    
    # Percentiles from rolling history
    lambda_percentile: float = 50.0  # Current λ percentile vs history
    is_elevated: bool = False  # λ > 2 × median
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'lambda': self.lambda_value,
            'lambda_std': self.lambda_std,
            'r_squared': self.r_squared,
            'sample_size': self.sample_size,
            'timestamp': self.timestamp.isoformat(),
            'percentile': self.lambda_percentile,
            'is_elevated': self.is_elevated,
        }


@dataclass
class ImpactPrediction:
    """Predicted market impact for an order."""
    order_size: float
    predicted_impact_bps: float
    impact_cost_usd: float
    is_acceptable: bool
    max_acceptable_size: float
    recommendation: str


class MarketImpactModel:
    """
    Market impact model based on Kyle's lambda.
    
    Estimates the expected price impact of an order based on:
    - Order size
    - Current market conditions
    - Historical impact coefficient
    """
    
    def __init__(
        self,
        impact_threshold_bps: float = 5.0,  # Max acceptable impact in bps
        max_alpha_fraction: float = 0.2,    # Impact < 20% of alpha
        expected_alpha_bps: float = 10.0,   # Expected alpha per trade
    ):
        """
        Initialize market impact model.
        
        Args:
            impact_threshold_bps: Maximum acceptable impact in basis points
            max_alpha_fraction: Maximum fraction of alpha consumed by impact
            expected_alpha_bps: Expected alpha per trade in basis points
        """
        self.impact_threshold_bps = impact_threshold_bps
        self.max_alpha_fraction = max_alpha_fraction
        self.expected_alpha_bps = expected_alpha_bps
    
    def predict_impact(
        self,
        order_size: float,
        lambda_estimate: LambdaEstimate,
        current_price: float,
        daily_volume: Optional[float] = None,
    ) -> ImpactPrediction:
        """
        Predict market impact for an order.
        
        Args:
            order_size: Order size in base currency units
            lambda_estimate: Current lambda estimate
            current_price: Current market price
            daily_volume: Optional daily volume for sizing
            
        Returns:
            ImpactPrediction with expected impact and recommendation
        """
        # Expected price impact: Δprice = λ × signed_volume
        price_impact = lambda_estimate.lambda_value * order_size
        impact_bps = (price_impact / current_price) * 10000
        impact_cost_usd = abs(price_impact * order_size)
        
        # Check against thresholds
        max_impact = min(
            self.impact_threshold_bps,
            self.expected_alpha_bps * self.max_alpha_fraction
        )
        is_acceptable = abs(impact_bps) <= max_impact
        
        # Calculate max acceptable size
        if lambda_estimate.lambda_value > 0:
            max_size = (max_impact / 10000 * current_price) / lambda_estimate.lambda_value
        else:
            max_size = order_size * 10  # No constraint
        
        # Generate recommendation
        if is_acceptable:
            recommendation = "Order size acceptable"
        elif order_size > max_size * 2:
            recommendation = f"Split order into {int(order_size / max_size) + 1} parts using TWAP"
        else:
            recommendation = f"Reduce order size to {max_size:.4f}"
        
        return ImpactPrediction(
            order_size=order_size,
            predicted_impact_bps=impact_bps,
            impact_cost_usd=impact_cost_usd,
            is_acceptable=is_acceptable,
            max_acceptable_size=max_size,
            recommendation=recommendation,
        )


class KyleLambdaEstimator:
    """
    Kyle Lambda Estimator.
    
    Estimates market impact coefficient using linear regression:
        Δprice = λ × signed_volume + ε
    
    Usage:
        estimator = KyleLambdaEstimator()
        
        # Add trade data
        for trade in trades:
            estimator.add_trade(symbol, price, size, side)
        
        # Get estimate
        estimate = estimator.get_estimate(symbol)
        print(f"Lambda: {estimate.lambda_value:.6f}")
    """
    
    def __init__(
        self,
        estimation_window_minutes: int = 5,
        history_days: int = 30,
        min_samples: int = 50,
        elevated_threshold_multiplier: float = 2.0,
    ):
        """
        Initialize lambda estimator.
        
        Args:
            estimation_window_minutes: Window for each λ estimate
            history_days: Days of λ history to maintain
            min_samples: Minimum samples for estimation
            elevated_threshold_multiplier: λ > multiplier × median = elevated
        """
        self.estimation_window = timedelta(minutes=estimation_window_minutes)
        self.history_days = history_days
        self.min_samples = min_samples
        self.elevated_multiplier = elevated_threshold_multiplier
        
        # Trade data per symbol
        self._trades: Dict[str, deque] = {}
        
        # Lambda history per symbol
        self._lambda_history: Dict[str, deque] = {}
        
        # Last estimate per symbol
        self._last_estimate: Dict[str, LambdaEstimate] = {}
        
        # Impact model
        self.impact_model = MarketImpactModel()
        
        logger.info("KyleLambdaEstimator initialized")
    
    def add_trade(
        self,
        symbol: str,
        price: float,
        size: float,
        side: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Add a trade for lambda estimation.
        
        Args:
            symbol: Trading symbol
            price: Trade price
            size: Trade size
            side: 'buy' or 'sell'
            timestamp: Trade timestamp
        """
        ts = timestamp or datetime.now()
        
        # Initialize if needed
        if symbol not in self._trades:
            self._trades[symbol] = deque(maxlen=100000)
            self._lambda_history[symbol] = deque(maxlen=self.history_days * 288)  # 5-min windows
        
        # Signed volume
        signed_volume = size if side.lower() == 'buy' else -size
        
        self._trades[symbol].append({
            'timestamp': ts,
            'price': price,
            'size': size,
            'signed_volume': signed_volume,
        })
    
    def add_trades_batch(
        self,
        symbol: str,
        trades: List[Tuple[float, float, str]],  # [(price, size, side), ...]
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add multiple trades at once."""
        ts = timestamp or datetime.now()
        for price, size, side in trades:
            self.add_trade(symbol, price, size, side, ts)
    
    def estimate(self, symbol: str) -> Optional[LambdaEstimate]:
        """
        Estimate Kyle's lambda for a symbol.
        
        Uses linear regression: Δprice = λ × signed_volume
        
        Args:
            symbol: Trading symbol
            
        Returns:
            LambdaEstimate or None if insufficient data
        """
        if symbol not in self._trades:
            return None
        
        trades = list(self._trades[symbol])
        
        if len(trades) < self.min_samples:
            logger.debug(f"Insufficient trades for {symbol}: {len(trades)} < {self.min_samples}")
            return None
        
        # Get recent trades within estimation window
        now = datetime.now()
        cutoff = now - self.estimation_window
        recent_trades = [t for t in trades if t['timestamp'] >= cutoff]
        
        if len(recent_trades) < 20:
            # Use all trades if window too narrow
            recent_trades = trades[-self.min_samples:]
        
        # Extract price changes and signed volumes
        prices = [t['price'] for t in recent_trades]
        signed_volumes = [t['signed_volume'] for t in recent_trades]
        
        # Calculate price changes
        price_changes = np.diff(prices)
        cumulative_signed_volume = np.cumsum(signed_volumes[:-1])
        
        if len(price_changes) < 10:
            return None
        
        # Linear regression: Δprice = λ × cumulative_signed_volume
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            cumulative_signed_volume, price_changes
        )
        
        # Lambda is the slope
        lambda_value = max(0, slope)  # Lambda should be non-negative
        
        # Create estimate
        estimate = LambdaEstimate(
            symbol=symbol,
            lambda_value=lambda_value,
            lambda_std=std_err,
            r_squared=r_value ** 2,
            sample_size=len(recent_trades),
            timestamp=now,
        )
        
        # Add to history
        self._lambda_history[symbol].append(lambda_value)
        
        # Calculate percentile
        history = list(self._lambda_history[symbol])
        if len(history) > 1:
            estimate.lambda_percentile = stats.percentileofscore(history, lambda_value)
            median = np.median(history)
            estimate.is_elevated = lambda_value > self.elevated_multiplier * median
        
        self._last_estimate[symbol] = estimate
        
        return estimate
    
    def get_estimate(self, symbol: str) -> Optional[LambdaEstimate]:
        """Get latest estimate for a symbol."""
        return self._last_estimate.get(symbol)
    
    def get_capacity_adjustment(self, symbol: str) -> float:
        """
        Get capacity adjustment factor based on current λ.
        
        Returns:
            Multiplier for strategy capacity (0.5 = halve, 1.0 = unchanged)
        """
        estimate = self.get_estimate(symbol)
        
        if not estimate:
            return 1.0
        
        if estimate.is_elevated:
            # Halve capacity when λ is elevated
            return 0.5
        
        # Gradual adjustment based on percentile
        if estimate.lambda_percentile > 75:
            return 0.75
        elif estimate.lambda_percentile > 90:
            return 0.5
        
        return 1.0
    
    def predict_impact(
        self,
        symbol: str,
        order_size: float,
        current_price: float,
    ) -> Optional[ImpactPrediction]:
        """
        Predict market impact for an order.
        
        Args:
            symbol: Trading symbol
            order_size: Order size
            current_price: Current price
            
        Returns:
            ImpactPrediction or None if no estimate available
        """
        estimate = self.get_estimate(symbol)
        
        if not estimate:
            # Estimate first
            estimate = self.estimate(symbol)
            if not estimate:
                return None
        
        return self.impact_model.predict_impact(
            order_size=order_size,
            lambda_estimate=estimate,
            current_price=current_price,
        )
    
    def get_lambda_history(
        self,
        symbol: str,
        window_hours: float = 24,
    ) -> List[float]:
        """Get lambda history for analysis."""
        if symbol not in self._lambda_history:
            return []
        
        return list(self._lambda_history[symbol])
    
    def get_summary(self) -> Dict:
        """Get summary of all symbols."""
        return {
            symbol: estimate.to_dict()
            for symbol, estimate in self._last_estimate.items()
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create estimator
    estimator = KyleLambdaEstimator(
        estimation_window_minutes=5,
        min_samples=30,
    )
    
    # Simulate trades
    np.random.seed(42)
    price = 50000
    
    for i in range(200):
        # Random trade
        size = np.random.uniform(0.01, 0.5)
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        
        # Price moves with trades (simulating impact)
        impact = 0.1 * (size if side == 'buy' else -size)
        price += impact + np.random.randn() * 5
        
        estimator.add_trade("BTCUSD", price, size, side)
        
        if (i + 1) % 50 == 0:
            estimate = estimator.estimate("BTCUSD")
            if estimate:
                print(f"Step {i+1}: λ = {estimate.lambda_value:.6f}, "
                      f"R² = {estimate.r_squared:.3f}")
    
    # Predict impact
    print("\n=== Impact Predictions ===")
    
    for order_size in [0.1, 0.5, 1.0, 5.0]:
        prediction = estimator.predict_impact("BTCUSD", order_size, price)
        if prediction:
            print(f"Order {order_size:.1f} BTC:")
            print(f"  Impact: {prediction.predicted_impact_bps:.2f} bps")
            print(f"  Acceptable: {prediction.is_acceptable}")
            print(f"  Recommendation: {prediction.recommendation}")
    
    # Summary
    print("\n=== Summary ===")
    summary = estimator.get_summary()
    for symbol, data in summary.items():
        print(f"{symbol}: λ={data['lambda']:.6f}, elevated={data['is_elevated']}")
