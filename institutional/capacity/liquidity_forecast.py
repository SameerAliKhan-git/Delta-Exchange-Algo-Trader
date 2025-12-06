"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         LIQUIDITY FORECASTER - Predict Future Market Conditions               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Predicts liquidity conditions 1 hour ahead to proactively throttle allocation.
Uses simplified model (TFT-like) for lambda prediction.

Goal: Anticipate periods of high market impact and reduce position size proactively.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("Capacity.LiquidityForecast")


@dataclass
class LiquidityForecast:
    """Liquidity forecast for a symbol."""
    symbol: str
    forecast_time: datetime
    horizon_hours: float
    
    # Lambda forecasts
    lambda_current: float
    lambda_forecast: float
    lambda_forecast_std: float
    
    # Derived signals
    expected_impact_change: float  # % change in impact
    capacity_multiplier: float  # Recommended capacity adjustment
    
    # Confidence
    confidence: float
    model_used: str
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'forecast_time': self.forecast_time.isoformat(),
            'horizon_hours': self.horizon_hours,
            'lambda_current': self.lambda_current,
            'lambda_forecast': self.lambda_forecast,
            'lambda_forecast_std': self.lambda_forecast_std,
            'expected_impact_change': self.expected_impact_change,
            'capacity_multiplier': self.capacity_multiplier,
            'confidence': self.confidence,
            'model_used': self.model_used,
        }


class LiquidityForecaster:
    """
    Liquidity Forecaster.
    
    Predicts future market impact (Kyle's lambda) using historical patterns
    and current market conditions.
    
    Uses a simplified approach:
    1. Time-of-day seasonality
    2. Autoregressive component
    3. Volatility spillover
    
    In production, this could use a full TFT model.
    
    Usage:
        forecaster = LiquidityForecaster()
        
        # Update with lambda estimates
        forecaster.update(symbol, lambda_value, volatility)
        
        # Get forecast
        forecast = forecaster.predict(symbol, horizon_hours=1)
        print(f"Expected λ in 1h: {forecast.lambda_forecast:.6f}")
    """
    
    def __init__(
        self,
        history_hours: int = 168,  # 1 week
        ar_order: int = 5,
        seasonality_hours: int = 24,
    ):
        """
        Initialize forecaster.
        
        Args:
            history_hours: Hours of history to maintain
            ar_order: Autoregressive order
            seasonality_hours: Seasonality period in hours
        """
        self.history_hours = history_hours
        self.ar_order = ar_order
        self.seasonality_hours = seasonality_hours
        
        # Data storage
        self._lambda_history: Dict[str, deque] = {}
        self._volatility_history: Dict[str, deque] = {}
        self._timestamps: Dict[str, deque] = {}
        
        # Seasonality patterns (learned)
        self._seasonality_pattern: Dict[str, np.ndarray] = {}
        
        # AR coefficients
        self._ar_coefficients: Dict[str, np.ndarray] = {}
        
        logger.info("LiquidityForecaster initialized")
    
    def update(
        self,
        symbol: str,
        lambda_value: float,
        volatility: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update with new lambda estimate.
        
        Args:
            symbol: Trading symbol
            lambda_value: Current lambda estimate
            volatility: Current volatility estimate
            timestamp: Observation timestamp
        """
        ts = timestamp or datetime.now()
        
        # Initialize if needed
        if symbol not in self._lambda_history:
            max_len = self.history_hours * 12  # 5-min resolution
            self._lambda_history[symbol] = deque(maxlen=max_len)
            self._volatility_history[symbol] = deque(maxlen=max_len)
            self._timestamps[symbol] = deque(maxlen=max_len)
        
        self._lambda_history[symbol].append(lambda_value)
        self._volatility_history[symbol].append(volatility or 0.0)
        self._timestamps[symbol].append(ts)
        
        # Update seasonal pattern periodically
        if len(self._lambda_history[symbol]) % 100 == 0:
            self._update_seasonality(symbol)
            self._update_ar_coefficients(symbol)
    
    def _update_seasonality(self, symbol: str) -> None:
        """Update seasonality pattern from historical data."""
        if symbol not in self._lambda_history:
            return
        
        lambdas = list(self._lambda_history[symbol])
        timestamps = list(self._timestamps[symbol])
        
        if len(lambdas) < self.seasonality_hours * 2:
            return
        
        # Group by hour of day
        hourly_values = [[] for _ in range(24)]
        
        for ts, lam in zip(timestamps, lambdas):
            hour = ts.hour
            hourly_values[hour].append(lam)
        
        # Average per hour
        pattern = np.zeros(24)
        for hour in range(24):
            if hourly_values[hour]:
                pattern[hour] = np.mean(hourly_values[hour])
        
        # Normalize to mean 1
        mean_pattern = np.mean(pattern[pattern > 0]) or 1.0
        self._seasonality_pattern[symbol] = pattern / mean_pattern
    
    def _update_ar_coefficients(self, symbol: str) -> None:
        """Update AR coefficients using OLS."""
        if symbol not in self._lambda_history:
            return
        
        lambdas = np.array(self._lambda_history[symbol])
        
        if len(lambdas) < self.ar_order * 2:
            return
        
        # Create design matrix for AR(p)
        n = len(lambdas)
        X = np.zeros((n - self.ar_order, self.ar_order))
        y = lambdas[self.ar_order:]
        
        for i in range(n - self.ar_order):
            X[i] = lambdas[i:i + self.ar_order][::-1]
        
        # OLS estimation
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            self._ar_coefficients[symbol] = coeffs
        except Exception:
            self._ar_coefficients[symbol] = np.ones(self.ar_order) / self.ar_order
    
    def predict(
        self,
        symbol: str,
        horizon_hours: float = 1.0,
    ) -> Optional[LiquidityForecast]:
        """
        Predict future lambda.
        
        Args:
            symbol: Trading symbol
            horizon_hours: Forecast horizon in hours
            
        Returns:
            LiquidityForecast or None if insufficient data
        """
        if symbol not in self._lambda_history:
            return None
        
        lambdas = list(self._lambda_history[symbol])
        
        if len(lambdas) < self.ar_order:
            return None
        
        current_lambda = lambdas[-1]
        
        # Start with AR forecast
        if symbol in self._ar_coefficients:
            ar_coeffs = self._ar_coefficients[symbol]
            recent = np.array(lambdas[-self.ar_order:])[::-1]
            ar_forecast = np.dot(ar_coeffs, recent)
        else:
            ar_forecast = current_lambda
        
        # Apply seasonality
        now = datetime.now()
        future_hour = (now + timedelta(hours=horizon_hours)).hour
        
        if symbol in self._seasonality_pattern:
            current_seasonal = self._seasonality_pattern[symbol][now.hour]
            future_seasonal = self._seasonality_pattern[symbol][future_hour]
            
            if current_seasonal > 0:
                seasonal_adjustment = future_seasonal / current_seasonal
            else:
                seasonal_adjustment = 1.0
        else:
            seasonal_adjustment = 1.0
        
        # Combined forecast
        lambda_forecast = ar_forecast * seasonal_adjustment
        lambda_forecast = max(0, lambda_forecast)  # Non-negative
        
        # Estimate uncertainty
        if len(lambdas) >= 20:
            lambda_std = np.std(lambdas[-20:]) * np.sqrt(horizon_hours)
        else:
            lambda_std = current_lambda * 0.2
        
        # Calculate expected impact change
        impact_change = (lambda_forecast - current_lambda) / (current_lambda + 1e-10)
        
        # Capacity multiplier
        if lambda_forecast > current_lambda * 1.5:
            capacity_mult = 0.5
        elif lambda_forecast > current_lambda * 1.2:
            capacity_mult = 0.75
        elif lambda_forecast < current_lambda * 0.8:
            capacity_mult = 1.2
        else:
            capacity_mult = 1.0
        
        # Confidence based on R² of AR model
        confidence = 0.7  # Base confidence
        
        return LiquidityForecast(
            symbol=symbol,
            forecast_time=now,
            horizon_hours=horizon_hours,
            lambda_current=current_lambda,
            lambda_forecast=lambda_forecast,
            lambda_forecast_std=lambda_std,
            expected_impact_change=impact_change * 100,  # percentage
            capacity_multiplier=capacity_mult,
            confidence=confidence,
            model_used="AR+Seasonality",
        )
    
    def get_capacity_recommendation(
        self,
        symbol: str,
        horizon_hours: float = 1.0,
    ) -> float:
        """
        Get recommended capacity multiplier.
        
        Args:
            symbol: Trading symbol
            horizon_hours: Forecast horizon
            
        Returns:
            Capacity multiplier (1.0 = no change)
        """
        forecast = self.predict(symbol, horizon_hours)
        
        if forecast:
            return forecast.capacity_multiplier
        
        return 1.0
    
    def get_all_forecasts(
        self,
        horizon_hours: float = 1.0,
    ) -> Dict[str, LiquidityForecast]:
        """Get forecasts for all symbols."""
        forecasts = {}
        
        for symbol in self._lambda_history.keys():
            forecast = self.predict(symbol, horizon_hours)
            if forecast:
                forecasts[symbol] = forecast
        
        return forecasts


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create forecaster
    forecaster = LiquidityForecaster()
    
    # Simulate historical data
    np.random.seed(42)
    
    # Base lambda with time-of-day pattern
    base_lambda = 0.0001
    
    for hour in range(72):  # 3 days
        # Time-of-day pattern: higher during market hours
        hour_of_day = hour % 24
        if 8 <= hour_of_day <= 16:
            seasonal_factor = 1.5
        elif 0 <= hour_of_day <= 4:
            seasonal_factor = 0.5
        else:
            seasonal_factor = 1.0
        
        for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
            lambda_value = base_lambda * seasonal_factor * (1 + 0.2 * np.random.randn())
            lambda_value = max(0.00001, lambda_value)
            
            volatility = 0.02 * seasonal_factor * (1 + 0.1 * np.random.randn())
            
            ts = datetime.now() - timedelta(hours=72-hour, minutes=60-minute)
            forecaster.update("BTCUSD", lambda_value, volatility, ts)
    
    # Get forecast
    print("=== 1-Hour Forecast ===")
    forecast = forecaster.predict("BTCUSD", horizon_hours=1)
    
    if forecast:
        print(f"Current λ: {forecast.lambda_current:.6f}")
        print(f"Forecast λ: {forecast.lambda_forecast:.6f}")
        print(f"Expected change: {forecast.expected_impact_change:+.1f}%")
        print(f"Capacity multiplier: {forecast.capacity_multiplier:.2f}")
        print(f"Confidence: {forecast.confidence:.1%}")
    
    # Get capacity recommendation
    cap_mult = forecaster.get_capacity_recommendation("BTCUSD")
    print(f"\nCapacity recommendation: {cap_mult:.2f}x")
