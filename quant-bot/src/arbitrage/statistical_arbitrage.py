"""
Statistical Arbitrage Module
=============================
Cointegration-based pairs trading with Z-score monitoring
and mean reversion strategies.

Author: Quant Bot
Version: 1.0.0
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class CointegrationMethod(Enum):
    ENGLE_GRANGER = "engle_granger"
    JOHANSEN = "johansen"


class PairStatus(Enum):
    NEUTRAL = "neutral"
    LONG_SPREAD = "long_spread"    # Long Y, short X
    SHORT_SPREAD = "short_spread"  # Short Y, long X


@dataclass
class CointegrationResult:
    """Results from cointegration test."""
    method: CointegrationMethod
    is_cointegrated: bool
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    half_life: float
    r_squared: float
    
    @property
    def confidence(self) -> str:
        if self.p_value < 0.01:
            return "HIGH"
        elif self.p_value < 0.05:
            return "MEDIUM"
        elif self.p_value < 0.10:
            return "LOW"
        return "NONE"


@dataclass
class SpreadMetrics:
    """Current spread metrics."""
    spread: float
    z_score: float
    mean: float
    std: float
    half_life: float
    current_side: PairStatus
    
    @property
    def is_extreme(self) -> bool:
        return abs(self.z_score) > 2.0
    
    @property
    def entry_signal(self) -> Optional[PairStatus]:
        if self.z_score > 2.0:
            return PairStatus.SHORT_SPREAD
        elif self.z_score < -2.0:
            return PairStatus.LONG_SPREAD
        return None
    
    @property
    def exit_signal(self) -> bool:
        return abs(self.z_score) < 0.5


@dataclass
class PairPosition:
    """Active pairs trading position."""
    pair_id: str
    asset_x: str
    asset_y: str
    side: PairStatus
    
    # Position details
    x_quantity: float
    y_quantity: float
    x_entry_price: float
    y_entry_price: float
    hedge_ratio: float
    entry_z_score: float
    entry_time: datetime
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    @property
    def notional(self) -> float:
        return abs(self.x_quantity * self.x_entry_price) + abs(self.y_quantity * self.y_entry_price)


@dataclass
class StatArbOpportunity:
    """Detected statistical arbitrage opportunity."""
    pair_id: str
    asset_x: str
    asset_y: str
    signal: PairStatus
    z_score: float
    expected_return: float
    confidence: float
    half_life_days: float
    cointegration_pvalue: float


class CointegrationTester:
    """
    Test for cointegration between asset pairs.
    
    Implements:
    - Engle-Granger two-step method
    - ADF test for stationarity
    - Half-life estimation
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        min_observations: int = 100
    ):
        self.significance_level = significance_level
        self.min_observations = min_observations
        
        # Critical values for ADF test
        self.adf_critical_values = {
            '1%': -3.43,
            '5%': -2.86,
            '10%': -2.57
        }
        
        logger.info("CointegrationTester initialized")
    
    def engle_granger_test(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> CointegrationResult:
        """
        Engle-Granger two-step cointegration test.
        
        Step 1: Regress y on x to get hedge ratio
        Step 2: Test residuals for stationarity (ADF test)
        
        Args:
            x: Price series for asset X
            y: Price series for asset Y
        
        Returns:
            CointegrationResult
        """
        if len(x) < self.min_observations:
            raise ValueError(f"Need at least {self.min_observations} observations")
        
        if len(x) != len(y):
            raise ValueError("Series must have same length")
        
        # Step 1: OLS regression to find hedge ratio
        # y = alpha + beta * x + epsilon
        x_with_const = np.column_stack([np.ones(len(x)), x])
        beta, residuals, rank, s = np.linalg.lstsq(x_with_const, y, rcond=None)
        
        alpha, hedge_ratio = beta[0], beta[1]
        
        # Calculate residuals (spread)
        spread = y - (alpha + hedge_ratio * x)
        
        # Step 2: ADF test on residuals
        adf_stat, p_value = self._adf_test(spread)
        
        # Calculate R-squared
        ss_res = np.sum(spread ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate half-life
        half_life = self._calculate_half_life(spread)
        
        # Determine if cointegrated
        is_cointegrated = p_value < self.significance_level
        
        return CointegrationResult(
            method=CointegrationMethod.ENGLE_GRANGER,
            is_cointegrated=is_cointegrated,
            test_statistic=adf_stat,
            p_value=p_value,
            critical_values=self.adf_critical_values,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            r_squared=r_squared
        )
    
    def _adf_test(
        self,
        series: np.ndarray,
        max_lag: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        Tests H0: series has unit root (non-stationary)
        Regression: Δy_t = α + γ*y_{t-1} + Σβ_i*Δy_{t-i} + ε_t
        Test statistic: t-stat on γ (should be negative for stationarity)
        
        Returns:
            Tuple of (test_statistic, p_value)
        """
        n = len(series)
        
        if max_lag is None:
            max_lag = int(np.floor(12 * (n / 100) ** 0.25))
        
        # Ensure we have enough data
        max_lag = min(max_lag, n // 4)
        
        # First difference: Δy_t = y_t - y_{t-1}
        diff = np.diff(series)  # length n-1
        
        # Build regression matrices
        # y: Δy_t (dependent variable)
        # X: [1, y_{t-1}, Δy_{t-1}, Δy_{t-2}, ...]
        
        # Start index to ensure all lags are available
        start_idx = max_lag
        
        # Dependent variable: Δy from start_idx onwards
        y = diff[start_idx:]
        
        # Regressors
        X_list = []
        
        # Constant term
        X_list.append(np.ones(len(y)))
        
        # Lagged level: y_{t-1} aligned with Δy_t
        # When Δy_t = diff[t], we need y_{t-1} = series[t]
        lagged_level = series[start_idx:-1]  # y_{t-1} for t from start_idx to n-1
        X_list.append(lagged_level)
        
        # Lagged differences: Δy_{t-1}, Δy_{t-2}, ...
        for lag in range(1, max_lag + 1):
            lagged_diff = diff[start_idx - lag:-lag]
            X_list.append(lagged_diff)
        
        X = np.column_stack(X_list)
        
        # OLS
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            residuals = y - y_hat
            
            # Standard error of gamma (coefficient on lagged level, index 1)
            n_obs = len(y)
            k = X.shape[1]  # number of regressors
            sigma_sq = np.sum(residuals ** 2) / (n_obs - k)
            
            # Covariance matrix of coefficients
            XtX_inv = np.linalg.inv(X.T @ X)
            var_beta = sigma_sq * XtX_inv
            se_gamma = np.sqrt(var_beta[1, 1])
            
            # ADF statistic (t-stat on gamma)
            gamma = beta[1]
            adf_stat = gamma / se_gamma
            
            # P-value using MacKinnon critical values
            # For cointegration residuals (no constant in cointegrating regression)
            # Critical values are more negative
            if adf_stat < self.adf_critical_values['1%']:
                p_value = 0.005
            elif adf_stat < self.adf_critical_values['5%']:
                p_value = 0.03
            elif adf_stat < self.adf_critical_values['10%']:
                p_value = 0.08
            else:
                # Interpolate for larger (less negative) values
                # Approximate using standard normal for very high values
                if adf_stat > 0:
                    p_value = 0.99
                else:
                    # Linear interpolation between 10% and 50%
                    p_value = 0.1 + 0.4 * (adf_stat - self.adf_critical_values['10%']) / (0 - self.adf_critical_values['10%'])
                    p_value = min(0.99, max(0.1, p_value))
            
        except np.linalg.LinAlgError:
            adf_stat = 0
            p_value = 1.0
        
        return adf_stat, p_value
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calculate half-life of mean reversion using AR(1).
        
        spread_t = phi * spread_{t-1} + epsilon
        half_life = -log(2) / log(phi)
        """
        spread_lag = spread[:-1]
        spread_diff = spread[1:] - spread[:-1]
        
        # Regression: spread_diff = theta * spread_lag + epsilon
        # Where theta = phi - 1
        try:
            theta = np.sum(spread_lag * spread_diff) / np.sum(spread_lag ** 2)
            phi = theta + 1
            
            if phi <= 0 or phi >= 1:
                return float('inf')
            
            half_life = -np.log(2) / np.log(phi)
            return max(half_life, 0)
        except:
            return float('inf')
    
    def rolling_cointegration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: int = 252
    ) -> List[CointegrationResult]:
        """
        Rolling window cointegration test.
        
        Returns list of results for each window.
        """
        results = []
        
        for i in range(window, len(x) + 1):
            x_window = x[i - window:i]
            y_window = y[i - window:i]
            
            try:
                result = self.engle_granger_test(x_window, y_window)
                results.append(result)
            except:
                pass
        
        return results


class ZScoreMonitor:
    """
    Monitor Z-score of spread for trading signals.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 4.0
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        
        self.spread_history: deque = deque(maxlen=lookback * 2)
        
        logger.info("ZScoreMonitor initialized")
    
    def update(
        self,
        x_price: float,
        y_price: float,
        hedge_ratio: float,
        intercept: float = 0
    ) -> SpreadMetrics:
        """
        Update spread and calculate Z-score.
        
        Args:
            x_price: Current price of asset X
            y_price: Current price of asset Y
            hedge_ratio: Beta from cointegration
            intercept: Alpha from cointegration
        
        Returns:
            SpreadMetrics
        """
        # Calculate spread
        spread = y_price - (intercept + hedge_ratio * x_price)
        self.spread_history.append(spread)
        
        # Calculate rolling statistics
        if len(self.spread_history) < self.lookback:
            window = list(self.spread_history)
        else:
            window = list(self.spread_history)[-self.lookback:]
        
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            z_score = (spread - mean) / std
        else:
            z_score = 0
        
        # Determine current side
        if z_score > self.entry_threshold:
            current_side = PairStatus.SHORT_SPREAD
        elif z_score < -self.entry_threshold:
            current_side = PairStatus.LONG_SPREAD
        else:
            current_side = PairStatus.NEUTRAL
        
        # Estimate half-life
        if len(self.spread_history) >= 30:
            spreads = np.array(list(self.spread_history)[-60:])
            half_life = self._quick_half_life(spreads)
        else:
            half_life = float('inf')
        
        return SpreadMetrics(
            spread=spread,
            z_score=z_score,
            mean=mean,
            std=std,
            half_life=half_life,
            current_side=current_side
        )
    
    def _quick_half_life(self, spreads: np.ndarray) -> float:
        """Quick half-life calculation."""
        try:
            spread_lag = spreads[:-1]
            spread_current = spreads[1:]
            
            # AR(1) coefficient
            phi = np.corrcoef(spread_lag, spread_current)[0, 1]
            
            if phi <= 0 or phi >= 1:
                return float('inf')
            
            return -np.log(2) / np.log(phi)
        except:
            return float('inf')
    
    def get_signal(
        self,
        metrics: SpreadMetrics,
        current_position: PairStatus = PairStatus.NEUTRAL
    ) -> Tuple[str, Optional[PairStatus]]:
        """
        Get trading signal.
        
        Returns:
            Tuple of (action, new_position)
            action: 'ENTER', 'EXIT', 'STOP_LOSS', 'HOLD'
        """
        z = metrics.z_score
        
        # Stop loss check
        if abs(z) > self.stop_loss_threshold:
            if current_position != PairStatus.NEUTRAL:
                return 'STOP_LOSS', PairStatus.NEUTRAL
        
        # Exit check
        if current_position != PairStatus.NEUTRAL:
            if abs(z) < self.exit_threshold:
                return 'EXIT', PairStatus.NEUTRAL
            
            # Check if spread moved against us significantly
            if current_position == PairStatus.LONG_SPREAD and z > self.entry_threshold:
                return 'STOP_LOSS', PairStatus.NEUTRAL
            if current_position == PairStatus.SHORT_SPREAD and z < -self.entry_threshold:
                return 'STOP_LOSS', PairStatus.NEUTRAL
        
        # Entry check
        if current_position == PairStatus.NEUTRAL:
            if z > self.entry_threshold:
                return 'ENTER', PairStatus.SHORT_SPREAD
            elif z < -self.entry_threshold:
                return 'ENTER', PairStatus.LONG_SPREAD
        
        return 'HOLD', current_position


class StatisticalArbitrageEngine:
    """
    Production-grade statistical arbitrage engine.
    
    Features:
    - Automatic pair identification
    - Cointegration monitoring
    - Dynamic hedge ratio adjustment
    - Z-score based entry/exit
    - Risk management
    """
    
    def __init__(
        self,
        coint_tester: Optional[CointegrationTester] = None,
        min_cointegration_pvalue: float = 0.05,
        min_half_life: float = 1,
        max_half_life: float = 60,
        position_sizing_method: str = 'volatility'
    ):
        self.coint_tester = coint_tester or CointegrationTester()
        self.min_cointegration_pvalue = min_cointegration_pvalue
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.position_sizing_method = position_sizing_method
        
        # Pair tracking
        self.pairs: Dict[str, Dict] = {}  # pair_id -> pair info
        self.monitors: Dict[str, ZScoreMonitor] = {}
        self.positions: Dict[str, PairPosition] = {}
        
        # Price history
        self.price_history: Dict[str, deque] = {}
        
        logger.info("StatisticalArbitrageEngine initialized")
    
    def add_pair(
        self,
        asset_x: str,
        asset_y: str,
        lookback: int = 60
    ) -> str:
        """
        Add a pair to monitor.
        
        Returns pair_id.
        """
        pair_id = f"{asset_x}_{asset_y}"
        
        self.pairs[pair_id] = {
            'asset_x': asset_x,
            'asset_y': asset_y,
            'cointegration': None,
            'last_test': None
        }
        
        self.monitors[pair_id] = ZScoreMonitor(lookback=lookback)
        self.price_history[asset_x] = deque(maxlen=500)
        self.price_history[asset_y] = deque(maxlen=500)
        
        logger.info(f"Added pair: {pair_id}")
        return pair_id
    
    def update_prices(
        self,
        prices: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update prices for all assets."""
        timestamp = timestamp or datetime.now()
        
        for asset, price in prices.items():
            if asset in self.price_history:
                self.price_history[asset].append((timestamp, price))
    
    def test_cointegration(
        self,
        pair_id: str,
        force: bool = False
    ) -> Optional[CointegrationResult]:
        """
        Test cointegration for a pair.
        
        Args:
            pair_id: Pair identifier
            force: Force retest even if recent
        
        Returns:
            CointegrationResult or None
        """
        if pair_id not in self.pairs:
            return None
        
        pair = self.pairs[pair_id]
        
        # Check if we need to retest
        if not force and pair['last_test']:
            hours_since_test = (datetime.now() - pair['last_test']).total_seconds() / 3600
            if hours_since_test < 24:  # Don't retest within 24h
                return pair['cointegration']
        
        # Get price history
        x_prices = [p for _, p in self.price_history[pair['asset_x']]]
        y_prices = [p for _, p in self.price_history[pair['asset_y']]]
        
        if len(x_prices) < 100 or len(y_prices) < 100:
            return None
        
        # Align series
        min_len = min(len(x_prices), len(y_prices))
        x = np.array(x_prices[-min_len:])
        y = np.array(y_prices[-min_len:])
        
        # Run test
        try:
            result = self.coint_tester.engle_granger_test(x, y)
            pair['cointegration'] = result
            pair['last_test'] = datetime.now()
            
            logger.info(
                f"Cointegration test {pair_id}: "
                f"p-value={result.p_value:.4f}, "
                f"hedge_ratio={result.hedge_ratio:.4f}, "
                f"half_life={result.half_life:.1f}"
            )
            
            return result
        except Exception as e:
            logger.error(f"Cointegration test failed for {pair_id}: {e}")
            return None
    
    def update_spread(
        self,
        pair_id: str,
        x_price: float,
        y_price: float
    ) -> Optional[SpreadMetrics]:
        """
        Update spread for a pair and get current metrics.
        """
        if pair_id not in self.pairs:
            return None
        
        pair = self.pairs[pair_id]
        coint = pair.get('cointegration')
        
        if coint is None:
            # Try to establish cointegration first
            coint = self.test_cointegration(pair_id)
            if coint is None:
                return None
        
        # Update monitor
        metrics = self.monitors[pair_id].update(
            x_price, y_price,
            hedge_ratio=coint.hedge_ratio
        )
        
        return metrics
    
    def detect_opportunity(
        self,
        pair_id: str,
        x_price: float,
        y_price: float
    ) -> Optional[StatArbOpportunity]:
        """
        Detect statistical arbitrage opportunity.
        """
        metrics = self.update_spread(pair_id, x_price, y_price)
        
        if metrics is None:
            return None
        
        pair = self.pairs[pair_id]
        coint = pair.get('cointegration')
        
        if coint is None or not coint.is_cointegrated:
            return None
        
        # Check half-life bounds
        if not self.min_half_life <= metrics.half_life <= self.max_half_life:
            return None
        
        # Check for entry signal
        signal = metrics.entry_signal
        if signal is None:
            return None
        
        # Calculate expected return (based on mean reversion)
        expected_return = abs(metrics.z_score) * metrics.std / (x_price + y_price) / 2
        
        # Confidence based on cointegration strength and z-score
        confidence = (
            (1 - coint.p_value) * 0.5 +
            min(abs(metrics.z_score) / 4, 0.25) +
            (1 - metrics.half_life / self.max_half_life) * 0.25
        )
        
        return StatArbOpportunity(
            pair_id=pair_id,
            asset_x=pair['asset_x'],
            asset_y=pair['asset_y'],
            signal=signal,
            z_score=metrics.z_score,
            expected_return=expected_return,
            confidence=confidence,
            half_life_days=metrics.half_life,
            cointegration_pvalue=coint.p_value
        )
    
    def calculate_position_sizes(
        self,
        pair_id: str,
        x_price: float,
        y_price: float,
        notional: float
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for dollar-neutral entry.
        
        Args:
            pair_id: Pair identifier
            x_price: Current X price
            y_price: Current Y price
            notional: Total notional exposure
        
        Returns:
            Tuple of (x_quantity, y_quantity)
        """
        pair = self.pairs[pair_id]
        coint = pair.get('cointegration')
        
        if coint is None:
            return 0, 0
        
        hedge_ratio = coint.hedge_ratio
        
        # Dollar-neutral sizing
        # We want: x_qty * x_price ≈ y_qty * y_price * hedge_ratio
        # And: x_qty * x_price + y_qty * y_price = notional
        
        # Solve system of equations
        # Let x_notional = x_qty * x_price
        # Let y_notional = y_qty * y_price
        # x_notional = hedge_ratio * y_notional
        # x_notional + y_notional = notional
        
        y_notional = notional / (1 + abs(hedge_ratio))
        x_notional = notional - y_notional
        
        x_quantity = x_notional / x_price
        y_quantity = y_notional / y_price
        
        return x_quantity, y_quantity
    
    def open_position(
        self,
        opportunity: StatArbOpportunity,
        x_price: float,
        y_price: float,
        notional: float
    ) -> Optional[PairPosition]:
        """
        Open pairs trading position.
        """
        pair_id = opportunity.pair_id
        
        if pair_id in self.positions:
            logger.warning(f"Already have position in {pair_id}")
            return None
        
        pair = self.pairs[pair_id]
        coint = pair.get('cointegration')
        
        if coint is None:
            return None
        
        # Calculate sizes
        x_qty, y_qty = self.calculate_position_sizes(
            pair_id, x_price, y_price, notional
        )
        
        # Adjust signs based on signal
        if opportunity.signal == PairStatus.LONG_SPREAD:
            # Long Y, Short X
            x_qty = -x_qty
        else:
            # Short Y, Long X
            y_qty = -y_qty
        
        position = PairPosition(
            pair_id=pair_id,
            asset_x=opportunity.asset_x,
            asset_y=opportunity.asset_y,
            side=opportunity.signal,
            x_quantity=x_qty,
            y_quantity=y_qty,
            x_entry_price=x_price,
            y_entry_price=y_price,
            hedge_ratio=coint.hedge_ratio,
            entry_z_score=opportunity.z_score,
            entry_time=datetime.now()
        )
        
        self.positions[pair_id] = position
        
        logger.info(
            f"Opened {opportunity.signal.value}: {pair_id}, "
            f"x={x_qty:.4f}@{x_price:.2f}, y={y_qty:.4f}@{y_price:.2f}, "
            f"z-score={opportunity.z_score:.2f}"
        )
        
        return position
    
    def update_position(
        self,
        pair_id: str,
        x_price: float,
        y_price: float
    ) -> Optional[Tuple[str, SpreadMetrics]]:
        """
        Update position P&L and check for signals.
        
        Returns:
            Tuple of (action, metrics) or None
        """
        if pair_id not in self.positions:
            return None
        
        position = self.positions[pair_id]
        
        # Calculate P&L
        x_pnl = position.x_quantity * (x_price - position.x_entry_price)
        y_pnl = position.y_quantity * (y_price - position.y_entry_price)
        position.unrealized_pnl = x_pnl + y_pnl
        
        # Track drawdown
        if position.unrealized_pnl < position.max_drawdown:
            position.max_drawdown = position.unrealized_pnl
        
        # Update spread metrics
        metrics = self.update_spread(pair_id, x_price, y_price)
        
        if metrics is None:
            return None
        
        # Check for exit signal
        action, _ = self.monitors[pair_id].get_signal(metrics, position.side)
        
        return action, metrics
    
    def close_position(
        self,
        pair_id: str,
        x_price: float,
        y_price: float,
        reason: str = "exit_signal"
    ) -> Optional[PairPosition]:
        """
        Close pairs position.
        """
        if pair_id not in self.positions:
            return None
        
        position = self.positions[pair_id]
        
        # Final P&L
        x_pnl = position.x_quantity * (x_price - position.x_entry_price)
        y_pnl = position.y_quantity * (y_price - position.y_entry_price)
        position.unrealized_pnl = x_pnl + y_pnl
        
        del self.positions[pair_id]
        
        logger.info(
            f"Closed {pair_id}: P&L=${position.unrealized_pnl:.2f}, "
            f"reason={reason}"
        )
        
        return position
    
    def scan_pairs(
        self,
        assets: List[str],
        price_data: Dict[str, np.ndarray]
    ) -> List[Tuple[str, str, CointegrationResult]]:
        """
        Scan all pairs for cointegration.
        
        Args:
            assets: List of asset names
            price_data: Dict of asset -> price array
        
        Returns:
            List of (asset_x, asset_y, result) for cointegrated pairs
        """
        results = []
        n = len(assets)
        
        for i in range(n):
            for j in range(i + 1, n):
                asset_x, asset_y = assets[i], assets[j]
                
                if asset_x not in price_data or asset_y not in price_data:
                    continue
                
                x = price_data[asset_x]
                y = price_data[asset_y]
                
                try:
                    result = self.coint_tester.engle_granger_test(x, y)
                    
                    if result.is_cointegrated:
                        if self.min_half_life <= result.half_life <= self.max_half_life:
                            results.append((asset_x, asset_y, result))
                except:
                    pass
        
        # Sort by p-value
        results.sort(key=lambda x: x[2].p_value)
        
        return results
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary of all positions."""
        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        total_notional = sum(p.notional for p in self.positions.values())
        
        return {
            'num_positions': len(self.positions),
            'total_notional': total_notional,
            'total_unrealized_pnl': total_pnl,
            'positions': {
                pid: {
                    'side': pos.side.value,
                    'x_qty': pos.x_quantity,
                    'y_qty': pos.y_quantity,
                    'entry_z': pos.entry_z_score,
                    'pnl': pos.unrealized_pnl
                }
                for pid, pos in self.positions.items()
            }
        }


# ==================== UTILITY FUNCTIONS ====================

def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation."""
    return np.corrcoef(x, y)[0, 1]


def calculate_rolling_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 60
) -> np.ndarray:
    """Calculate rolling correlation."""
    n = len(x)
    corrs = np.zeros(n - window + 1)
    
    for i in range(n - window + 1):
        corrs[i] = calculate_correlation(
            x[i:i + window],
            y[i:i + window]
        )
    
    return corrs


def find_optimal_hedge_ratio(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'ols'
) -> float:
    """
    Find optimal hedge ratio.
    
    Methods:
    - 'ols': Ordinary Least Squares
    - 'tls': Total Least Squares
    """
    if method == 'ols':
        # y = beta * x
        beta = np.sum(x * y) / np.sum(x ** 2)
    elif method == 'tls':
        # Total Least Squares (Deming regression)
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        sxx = np.sum((x - x_mean) ** 2)
        syy = np.sum((y - y_mean) ** 2)
        sxy = np.sum((x - x_mean) * (y - y_mean))
        
        beta = (syy - sxx + np.sqrt((syy - sxx)**2 + 4 * sxy**2)) / (2 * sxy)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return beta


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Statistical Arbitrage Engine")
    parser.add_argument("--action", type=str, required=True,
                       choices=['test', 'scan', 'simulate'],
                       help="Action to perform")
    parser.add_argument("--asset-x", type=str, help="Asset X")
    parser.add_argument("--asset-y", type=str, help="Asset Y")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback period")
    
    args = parser.parse_args()
    
    # Generate sample data for demo
    np.random.seed(42)
    n = 500
    
    # Create cointegrated pair
    x = np.cumsum(np.random.randn(n) * 0.02) + 100
    noise = np.random.randn(n) * 0.5
    y = 0.8 * x + 20 + noise  # y = 0.8x + 20 + noise
    
    engine = StatisticalArbitrageEngine()
    
    if args.action == 'test':
        result = engine.coint_tester.engle_granger_test(x, y)
        
        print(f"\n{'='*60}")
        print("COINTEGRATION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Method:          {result.method.value}")
        print(f"Is Cointegrated: {'✅ YES' if result.is_cointegrated else '❌ NO'}")
        print(f"Test Statistic:  {result.test_statistic:.4f}")
        print(f"P-Value:         {result.p_value:.4f}")
        print(f"Confidence:      {result.confidence}")
        print(f"Hedge Ratio:     {result.hedge_ratio:.4f}")
        print(f"Half-Life:       {result.half_life:.1f} periods")
        print(f"R-Squared:       {result.r_squared:.4f}")
        print(f"\nCritical Values:")
        for level, value in result.critical_values.items():
            print(f"  {level}: {value:.4f}")
    
    elif args.action == 'simulate':
        # Add pair
        pair_id = engine.add_pair('ASSET_X', 'ASSET_Y')
        
        # Feed price history
        for i in range(len(x)):
            engine.update_prices({
                'ASSET_X': x[i],
                'ASSET_Y': y[i]
            })
        
        # Test cointegration
        engine.test_cointegration(pair_id)
        
        # Simulate trading
        trades = []
        position = None
        
        for i in range(100, len(x)):
            x_price, y_price = x[i], y[i]
            
            if position is None:
                # Look for entry
                opp = engine.detect_opportunity(pair_id, x_price, y_price)
                if opp and opp.confidence > 0.6:
                    position = engine.open_position(opp, x_price, y_price, 10000)
            else:
                # Check for exit
                result = engine.update_position(pair_id, x_price, y_price)
                if result:
                    action, metrics = result
                    if action in ['EXIT', 'STOP_LOSS']:
                        closed = engine.close_position(pair_id, x_price, y_price, action)
                        trades.append(closed.unrealized_pnl)
                        position = None
        
        # Close any remaining position
        if position:
            closed = engine.close_position(pair_id, x[-1], y[-1], 'end_of_data')
            trades.append(closed.unrealized_pnl)
        
        print(f"\n{'='*60}")
        print("SIMULATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades:    {len(trades)}")
        print(f"Total P&L:       ${sum(trades):.2f}")
        print(f"Avg P&L/Trade:   ${np.mean(trades):.2f}" if trades else "N/A")
        print(f"Win Rate:        {np.mean([t > 0 for t in trades]):.1%}" if trades else "N/A")
