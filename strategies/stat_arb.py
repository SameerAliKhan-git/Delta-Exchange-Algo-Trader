"""
Statistical Arbitrage Strategy

Implements statistical arbitrage techniques:
- Pairs trading with cointegration
- Mean reversion with half-life optimization
- Z-score based entry/exit
- Dynamic hedge ratios
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PairMetrics:
    """Metrics for a trading pair"""
    symbol_a: str
    symbol_b: str
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    half_life: float
    spread_zscore: float
    is_cointegrated: bool


@dataclass
class StatArbSignal:
    """Statistical arbitrage signal"""
    action: str  # 'long_spread', 'short_spread', 'close', 'none'
    symbol_long: str
    symbol_short: str
    size_ratio: float
    zscore: float
    confidence: float
    expected_reversion_periods: int


class StatisticalArbitrage:
    """
    Statistical Arbitrage Strategy
    
    Implements pairs trading with rigorous statistical testing.
    """
    
    def __init__(
        self,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        zscore_stop: float = 4.0,
        min_half_life: int = 2,
        max_half_life: int = 50,
        lookback: int = 100,
        cointegration_pvalue: float = 0.05
    ):
        """
        Initialize stat arb strategy
        
        Args:
            zscore_entry: Z-score threshold for entry
            zscore_exit: Z-score threshold for exit
            zscore_stop: Z-score stop loss
            min_half_life: Minimum half-life for mean reversion
            max_half_life: Maximum half-life
            lookback: Lookback period for calculations
            cointegration_pvalue: P-value threshold for cointegration
        """
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.zscore_stop = zscore_stop
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.lookback = lookback
        self.coint_pvalue = cointegration_pvalue
        
        # Track pairs
        self._pair_metrics: Dict[Tuple[str, str], PairMetrics] = {}
        self._spread_history: Dict[Tuple[str, str], List[float]] = {}
    
    def analyze_pair(
        self,
        symbol_a: str,
        prices_a: np.ndarray,
        symbol_b: str,
        prices_b: np.ndarray
    ) -> PairMetrics:
        """
        Analyze a pair for trading potential
        
        Args:
            symbol_a: First symbol
            prices_a: Price series for first symbol
            symbol_b: Second symbol
            prices_b: Price series for second symbol
        
        Returns:
            PairMetrics with analysis results
        """
        # Correlation
        correlation = np.corrcoef(prices_a, prices_b)[0, 1]
        
        # Cointegration test (simplified Engle-Granger)
        coint_pvalue, hedge_ratio = self._test_cointegration(prices_a, prices_b)
        
        # Calculate spread
        spread = prices_a - hedge_ratio * prices_b
        
        # Half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        # Current z-score
        zscore = (spread[-1] - np.mean(spread)) / np.std(spread) if np.std(spread) > 0 else 0
        
        metrics = PairMetrics(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=correlation,
            cointegration_pvalue=coint_pvalue,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            spread_zscore=zscore,
            is_cointegrated=(
                coint_pvalue < self.coint_pvalue and
                self.min_half_life <= half_life <= self.max_half_life
            )
        )
        
        self._pair_metrics[(symbol_a, symbol_b)] = metrics
        self._spread_history[(symbol_a, symbol_b)] = list(spread)
        
        return metrics
    
    def generate_signal(
        self,
        symbol_a: str,
        price_a: float,
        symbol_b: str,
        price_b: float
    ) -> StatArbSignal:
        """
        Generate trading signal for a pair
        
        Args:
            symbol_a: First symbol
            price_a: Current price of first symbol
            symbol_b: Second symbol
            price_b: Current price of second symbol
        
        Returns:
            StatArbSignal with trade recommendation
        """
        key = (symbol_a, symbol_b)
        
        if key not in self._pair_metrics:
            return StatArbSignal(
                action='none',
                symbol_long='',
                symbol_short='',
                size_ratio=1.0,
                zscore=0,
                confidence=0,
                expected_reversion_periods=0
            )
        
        metrics = self._pair_metrics[key]
        
        if not metrics.is_cointegrated:
            return StatArbSignal(
                action='none',
                symbol_long='',
                symbol_short='',
                size_ratio=1.0,
                zscore=0,
                confidence=0,
                expected_reversion_periods=0
            )
        
        # Calculate current spread and z-score
        spread = price_a - metrics.hedge_ratio * price_b
        
        history = self._spread_history.get(key, [])
        if len(history) < 20:
            return StatArbSignal(
                action='none',
                symbol_long='',
                symbol_short='',
                size_ratio=1.0,
                zscore=0,
                confidence=0,
                expected_reversion_periods=0
            )
        
        mean = np.mean(history[-self.lookback:])
        std = np.std(history[-self.lookback:])
        
        if std == 0:
            return StatArbSignal(
                action='none',
                symbol_long='',
                symbol_short='',
                size_ratio=1.0,
                zscore=0,
                confidence=0,
                expected_reversion_periods=0
            )
        
        zscore = (spread - mean) / std
        
        # Determine action
        action = 'none'
        symbol_long = ''
        symbol_short = ''
        
        if zscore > self.zscore_entry:
            # Spread too high - short A, long B
            action = 'short_spread'
            symbol_long = symbol_b
            symbol_short = symbol_a
        elif zscore < -self.zscore_entry:
            # Spread too low - long A, short B
            action = 'long_spread'
            symbol_long = symbol_a
            symbol_short = symbol_b
        elif abs(zscore) < self.zscore_exit:
            action = 'close'
        elif abs(zscore) > self.zscore_stop:
            action = 'stop_loss'
        
        # Confidence based on cointegration strength and z-score
        confidence = (1 - metrics.cointegration_pvalue) * min(abs(zscore) / self.zscore_entry, 1)
        
        # Expected reversion time
        expected_periods = int(metrics.half_life * 2)  # ~2 half-lives for full reversion
        
        return StatArbSignal(
            action=action,
            symbol_long=symbol_long,
            symbol_short=symbol_short,
            size_ratio=metrics.hedge_ratio,
            zscore=zscore,
            confidence=confidence,
            expected_reversion_periods=expected_periods
        )
    
    def find_best_pairs(
        self,
        symbols: List[str],
        prices: Dict[str, np.ndarray],
        top_n: int = 5
    ) -> List[PairMetrics]:
        """
        Find best trading pairs from universe
        
        Args:
            symbols: List of symbols to analyze
            prices: Dictionary of price series
            top_n: Number of top pairs to return
        
        Returns:
            List of best pair metrics
        """
        pairs = []
        
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i+1:]:
                if sym_a not in prices or sym_b not in prices:
                    continue
                
                try:
                    metrics = self.analyze_pair(
                        sym_a, prices[sym_a],
                        sym_b, prices[sym_b]
                    )
                    if metrics.is_cointegrated:
                        pairs.append(metrics)
                except Exception:
                    continue
        
        # Sort by cointegration p-value (lower is better)
        pairs.sort(key=lambda x: x.cointegration_pvalue)
        
        return pairs[:top_n]
    
    def _test_cointegration(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        Test for cointegration (simplified Engle-Granger)
        
        Returns:
            (p-value, hedge_ratio)
        """
        # OLS regression: A = beta * B + residual
        X = np.column_stack([np.ones(len(prices_b)), prices_b])
        beta = np.linalg.lstsq(X, prices_a, rcond=None)[0]
        hedge_ratio = beta[1]
        
        # Residuals
        residuals = prices_a - (beta[0] + hedge_ratio * prices_b)
        
        # ADF test on residuals (simplified)
        pvalue = self._adf_test(residuals)
        
        return pvalue, hedge_ratio
    
    def _adf_test(self, series: np.ndarray) -> float:
        """
        Simplified ADF test
        
        Returns approximate p-value
        """
        n = len(series)
        if n < 20:
            return 1.0
        
        # Calculate test statistic
        y = series[1:]
        y_lag = series[:-1]
        
        # Regression: delta_y = alpha + beta * y_lag + error
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        delta_y = y - y_lag
        
        try:
            beta = np.linalg.lstsq(X, delta_y, rcond=None)[0]
            
            # t-statistic for beta[1]
            residuals = delta_y - X @ beta
            mse = np.sum(residuals**2) / (n - 2)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(var_beta))
            
            t_stat = beta[1] / se_beta[1] if se_beta[1] > 0 else 0
            
            # Approximate p-value (critical values for ADF)
            # -3.43 (1%), -2.86 (5%), -2.57 (10%)
            if t_stat < -3.43:
                return 0.01
            elif t_stat < -2.86:
                return 0.05
            elif t_stat < -2.57:
                return 0.10
            else:
                return 0.5
        except Exception:
            return 1.0
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calculate half-life of mean reversion
        
        Uses Ornstein-Uhlenbeck model
        """
        n = len(spread)
        if n < 20:
            return 100  # Default high value
        
        # Regression: spread[t] - spread[t-1] = lambda * (spread[t-1] - mean) + noise
        y = spread[1:] - spread[:-1]
        X = spread[:-1] - np.mean(spread)
        
        try:
            # OLS
            lambda_est = np.sum(X * y) / np.sum(X * X)
            
            if lambda_est >= 0:
                return 100  # No mean reversion
            
            half_life = -np.log(2) / lambda_est
            return max(1, min(half_life, 100))
        except Exception:
            return 100
    
    def update_spread_history(
        self,
        symbol_a: str,
        price_a: float,
        symbol_b: str,
        price_b: float
    ) -> None:
        """Update spread history with new prices"""
        key = (symbol_a, symbol_b)
        
        if key not in self._pair_metrics:
            return
        
        metrics = self._pair_metrics[key]
        spread = price_a - metrics.hedge_ratio * price_b
        
        if key not in self._spread_history:
            self._spread_history[key] = []
        
        self._spread_history[key].append(spread)
        
        # Keep only recent history
        if len(self._spread_history[key]) > self.lookback * 2:
            self._spread_history[key] = self._spread_history[key][-self.lookback:]
