"""
Strategy Validation & Statistical Testing
==========================================
Advanced validation techniques to prevent overfitting:
- Monte Carlo Simulation
- Probability of Backtest Overfitting (PBO)
- Deflated Sharpe Ratio
- Bootstrap Confidence Intervals
- CSCV (Combinatorially Symmetric Cross-Validation)

Reference: AFML Ch. 11-12, Bailey & López de Prado papers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    from scipy.special import comb
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationResult:
    """Results from strategy validation."""
    metric: str
    value: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    n_simulations: int
    metadata: Dict


@dataclass
class PBOResult:
    """Results from Probability of Backtest Overfitting analysis."""
    pbo: float  # Probability of Backtest Overfitting
    performance_degradation: float  # Expected OOS performance drop
    rank_correlation: float  # IS vs OOS rank correlation
    logits: np.ndarray
    is_overfit: bool
    confidence: str  # 'high', 'medium', 'low'


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloValidator:
    """
    Monte Carlo simulation for strategy validation.
    
    Methods:
    - Path shuffling: Test if order matters
    - Block bootstrap: Preserve autocorrelation
    - Synthetic data: Test against null model
    """
    
    def __init__(self, returns: np.ndarray, n_simulations: int = 1000,
                 block_size: int = 20, seed: int = 42):
        """
        Initialize Monte Carlo validator.
        
        Args:
            returns: Strategy returns
            n_simulations: Number of MC simulations
            block_size: Size of blocks for block bootstrap
            seed: Random seed
        """
        self.returns = np.asarray(returns)
        self.n_simulations = n_simulations
        self.block_size = block_size
        self.seed = seed
        
        np.random.seed(seed)
    
    def shuffle_simulation(self) -> Dict:
        """
        Test if strategy depends on return order.
        
        If shuffled returns produce similar results, the strategy
        may just be capturing general market exposure, not timing.
        """
        original_sharpe = self._calculate_sharpe(self.returns)
        original_total_return = np.prod(1 + self.returns) - 1
        
        shuffled_sharpes = []
        shuffled_returns = []
        
        for _ in range(self.n_simulations):
            shuffled = np.random.permutation(self.returns)
            shuffled_sharpes.append(self._calculate_sharpe(shuffled))
            shuffled_returns.append(np.prod(1 + shuffled) - 1)
        
        # P-value: fraction of shuffled >= original
        p_value_sharpe = (np.array(shuffled_sharpes) >= original_sharpe).mean()
        p_value_return = (np.array(shuffled_returns) >= original_total_return).mean()
        
        return {
            'original_sharpe': original_sharpe,
            'shuffled_sharpe_mean': np.mean(shuffled_sharpes),
            'shuffled_sharpe_std': np.std(shuffled_sharpes),
            'p_value_sharpe': p_value_sharpe,
            'original_return': original_total_return,
            'shuffled_return_mean': np.mean(shuffled_returns),
            'p_value_return': p_value_return,
            'significant': p_value_sharpe < 0.05
        }
    
    def block_bootstrap(self, statistic_fn: Optional[Callable] = None) -> Dict:
        """
        Block bootstrap to preserve autocorrelation.
        
        More appropriate for time series than i.i.d. bootstrap.
        """
        if statistic_fn is None:
            statistic_fn = self._calculate_sharpe
        
        original_stat = statistic_fn(self.returns)
        n = len(self.returns)
        n_blocks = int(np.ceil(n / self.block_size))
        
        bootstrap_stats = []
        
        for _ in range(self.n_simulations):
            # Sample blocks with replacement
            indices = []
            for _ in range(n_blocks):
                start = np.random.randint(0, n - self.block_size + 1)
                indices.extend(range(start, start + self.block_size))
            
            indices = indices[:n]  # Trim to original length
            bootstrap_sample = self.returns[indices]
            bootstrap_stats.append(statistic_fn(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_stats, 2.5)
        ci_upper = np.percentile(bootstrap_stats, 97.5)
        
        # Standard error
        se = bootstrap_stats.std()
        
        return {
            'original': original_stat,
            'bootstrap_mean': bootstrap_stats.mean(),
            'bootstrap_std': se,
            'ci_95': (ci_lower, ci_upper),
            'significant': ci_lower > 0  # For Sharpe ratio
        }
    
    def null_hypothesis_test(self, alpha: float = 0.05) -> Dict:
        """
        Test strategy against random walk null hypothesis.
        
        H0: Returns come from random walk (no predictability)
        """
        # Generate null distribution (random returns with same vol)
        vol = self.returns.std()
        mean = self.returns.mean()
        
        null_sharpes = []
        
        for _ in range(self.n_simulations):
            null_returns = np.random.normal(0, vol, len(self.returns))  # Zero mean
            null_sharpes.append(self._calculate_sharpe(null_returns))
        
        original_sharpe = self._calculate_sharpe(self.returns)
        
        # P-value
        p_value = (np.array(null_sharpes) >= original_sharpe).mean()
        
        # Critical value
        critical_value = np.percentile(null_sharpes, (1 - alpha) * 100)
        
        return {
            'original_sharpe': original_sharpe,
            'null_mean': np.mean(null_sharpes),
            'null_std': np.std(null_sharpes),
            'critical_value': critical_value,
            'p_value': p_value,
            'reject_null': p_value < alpha,
            'alpha': alpha
        }
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)


# =============================================================================
# PROBABILITY OF BACKTEST OVERFITTING (PBO)
# =============================================================================

class PBOAnalyzer:
    """
    Probability of Backtest Overfitting (Bailey & López de Prado).
    
    Estimates probability that in-sample optimal strategy
    will underperform out-of-sample.
    
    Uses Combinatorially Symmetric Cross-Validation (CSCV).
    """
    
    def __init__(self, strategy_returns: pd.DataFrame, n_splits: int = 16):
        """
        Initialize PBO analyzer.
        
        Args:
            strategy_returns: DataFrame with returns of multiple strategy configurations
                              (rows = time, columns = strategy variants)
            n_splits: Number of splits for CSCV (must be even)
        """
        self.strategy_returns = strategy_returns
        self.n_splits = n_splits
        
        if n_splits % 2 != 0:
            raise ValueError("n_splits must be even for CSCV")
    
    def calculate_pbo(self) -> PBOResult:
        """
        Calculate Probability of Backtest Overfitting.
        
        Returns:
            PBOResult with PBO estimate and diagnostics
        """
        returns = self.strategy_returns.values
        n_samples, n_strategies = returns.shape
        
        # Split into S subsets
        subset_size = n_samples // self.n_splits
        subsets = [returns[i*subset_size:(i+1)*subset_size] for i in range(self.n_splits)]
        
        # Generate all combinations where half are IS, half are OOS
        n_combinations = int(comb(self.n_splits, self.n_splits // 2, exact=True))
        
        logits = []
        is_sharpes = []
        oos_sharpes = []
        
        # Sample combinations (for computational efficiency)
        max_combinations = min(n_combinations, 1000)
        
        for _ in range(max_combinations):
            # Random split into IS and OOS
            indices = np.random.permutation(self.n_splits)
            is_indices = indices[:self.n_splits // 2]
            oos_indices = indices[self.n_splits // 2:]
            
            # Combine subsets
            is_returns = np.vstack([subsets[i] for i in is_indices])
            oos_returns = np.vstack([subsets[i] for i in oos_indices])
            
            # Calculate Sharpe for each strategy
            is_sharpe = np.array([self._sharpe(is_returns[:, i]) for i in range(n_strategies)])
            oos_sharpe = np.array([self._sharpe(oos_returns[:, i]) for i in range(n_strategies)])
            
            # Find IS optimal strategy
            best_is_idx = np.argmax(is_sharpe)
            
            # Check if OOS performance is below median
            oos_median = np.median(oos_sharpe)
            is_overfit = oos_sharpe[best_is_idx] < oos_median
            
            # Calculate logit
            rank = (oos_sharpe < oos_sharpe[best_is_idx]).sum() / n_strategies
            logit = np.log((rank + 1e-10) / (1 - rank + 1e-10))
            
            logits.append(logit)
            is_sharpes.append(is_sharpe[best_is_idx])
            oos_sharpes.append(oos_sharpe[best_is_idx])
        
        logits = np.array(logits)
        
        # PBO = probability that logit < 0 (OOS rank < 0.5)
        pbo = (logits < 0).mean()
        
        # Performance degradation
        perf_degradation = np.mean(np.array(is_sharpes) - np.array(oos_sharpes))
        
        # Rank correlation
        rank_corr = np.corrcoef(is_sharpes, oos_sharpes)[0, 1]
        
        # Confidence assessment
        if pbo < 0.25:
            confidence = 'high'
            is_overfit = False
        elif pbo < 0.50:
            confidence = 'medium'
            is_overfit = False
        else:
            confidence = 'low'
            is_overfit = True
        
        return PBOResult(
            pbo=pbo,
            performance_degradation=perf_degradation,
            rank_correlation=rank_corr,
            logits=logits,
            is_overfit=is_overfit,
            confidence=confidence
        )
    
    def _sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)


# =============================================================================
# DEFLATED SHARPE RATIO
# =============================================================================

class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado).
    
    Adjusts Sharpe ratio for multiple testing.
    When you try many strategies, you expect some to look good by chance.
    """
    
    def __init__(self, returns: np.ndarray, n_trials: int,
                 var_sharpe: Optional[float] = None):
        """
        Initialize DSR calculator.
        
        Args:
            returns: Strategy returns
            n_trials: Number of strategy configurations tried
            var_sharpe: Variance of Sharpe ratio estimates (estimated if None)
        """
        self.returns = np.asarray(returns)
        self.n_trials = n_trials
        
        # Estimate Sharpe variance if not provided
        if var_sharpe is None:
            self.var_sharpe = self._estimate_sharpe_variance()
        else:
            self.var_sharpe = var_sharpe
    
    def _estimate_sharpe_variance(self) -> float:
        """
        Estimate variance of Sharpe ratio.
        
        Using Lo (2002) formula.
        """
        n = len(self.returns)
        sr = self._calculate_sharpe()
        
        # Variance of Sharpe estimator
        var_sr = (1 + 0.5 * sr ** 2) / n
        
        return var_sr
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio."""
        if self.returns.std() == 0:
            return 0
        return self.returns.mean() / self.returns.std() * np.sqrt(252)
    
    def calculate_dsr(self) -> Dict:
        """
        Calculate Deflated Sharpe Ratio.
        
        Returns:
            Dictionary with DSR and related metrics
        """
        sr = self._calculate_sharpe()
        
        # Expected maximum Sharpe under null (trying n strategies)
        # Using Euler-Mascheroni constant approximation
        gamma = 0.5772156649
        expected_max_sr = np.sqrt(self.var_sharpe) * (
            (1 - gamma) * stats.norm.ppf(1 - 1 / self.n_trials) +
            gamma * stats.norm.ppf(1 - 1 / (self.n_trials * np.e))
        ) if SCIPY_AVAILABLE else np.sqrt(2 * np.log(self.n_trials)) * np.sqrt(self.var_sharpe)
        
        # Deflated Sharpe Ratio
        dsr = sr - expected_max_sr
        
        # P-value
        if SCIPY_AVAILABLE:
            z_score = sr / np.sqrt(self.var_sharpe)
            p_value = 1 - stats.norm.cdf(z_score)
            
            # Adjusted p-value for multiple testing
            adjusted_p_value = 1 - (1 - p_value) ** self.n_trials
        else:
            p_value = None
            adjusted_p_value = None
        
        return {
            'sharpe_ratio': sr,
            'deflated_sharpe': dsr,
            'expected_max_sharpe': expected_max_sr,
            'n_trials': self.n_trials,
            'var_sharpe': self.var_sharpe,
            'p_value': p_value,
            'adjusted_p_value': adjusted_p_value,
            'significant': dsr > 0
        }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

class BootstrapCI:
    """
    Bootstrap confidence intervals for performance metrics.
    """
    
    def __init__(self, returns: np.ndarray, n_bootstrap: int = 10000,
                 block_size: int = 20, seed: int = 42):
        """
        Initialize bootstrap CI calculator.
        
        Args:
            returns: Strategy returns
            n_bootstrap: Number of bootstrap samples
            block_size: Block size for block bootstrap
            seed: Random seed
        """
        self.returns = np.asarray(returns)
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        
        np.random.seed(seed)
    
    def calculate_ci(self, metric: str = 'sharpe', 
                     confidence: float = 0.95) -> ValidationResult:
        """
        Calculate confidence interval for metric.
        
        Args:
            metric: 'sharpe', 'total_return', 'max_drawdown', 'sortino'
            confidence: Confidence level
        
        Returns:
            ValidationResult with CI
        """
        # Metric function
        metric_fns = {
            'sharpe': self._sharpe,
            'total_return': lambda r: np.prod(1 + r) - 1,
            'max_drawdown': self._max_drawdown,
            'sortino': self._sortino
        }
        
        if metric not in metric_fns:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_fn = metric_fns[metric]
        
        # Original metric
        original = metric_fn(self.returns)
        
        # Bootstrap samples (block bootstrap)
        bootstrap_metrics = []
        n = len(self.returns)
        n_blocks = int(np.ceil(n / self.block_size))
        
        for _ in range(self.n_bootstrap):
            indices = []
            for _ in range(n_blocks):
                start = np.random.randint(0, max(1, n - self.block_size + 1))
                indices.extend(range(start, min(start + self.block_size, n)))
            
            indices = indices[:n]
            bootstrap_sample = self.returns[indices]
            bootstrap_metrics.append(metric_fn(bootstrap_sample))
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        
        # Confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
        
        # P-value (two-sided test against zero)
        if metric in ['sharpe', 'total_return', 'sortino']:
            p_value = 2 * min(
                (bootstrap_metrics <= 0).mean(),
                (bootstrap_metrics >= 0).mean()
            )
        else:
            p_value = None
        
        return ValidationResult(
            metric=metric,
            value=original,
            p_value=p_value if p_value else 1.0,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=ci_lower > 0 if metric in ['sharpe', 'total_return', 'sortino'] else True,
            n_simulations=self.n_bootstrap,
            metadata={
                'bootstrap_mean': bootstrap_metrics.mean(),
                'bootstrap_std': bootstrap_metrics.std()
            }
        )
    
    def _sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return self._sharpe(returns)
        return returns.mean() / downside.std() * np.sqrt(252)
    
    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return drawdown.max()


# =============================================================================
# STRATEGY VALIDATOR (UNIFIED INTERFACE)
# =============================================================================

class StrategyValidator:
    """
    Comprehensive strategy validation.
    
    Combines multiple validation methods.
    """
    
    def __init__(self, returns: np.ndarray, n_trials: int = 1):
        """
        Initialize validator.
        
        Args:
            returns: Strategy returns
            n_trials: Number of strategy configurations tried
        """
        self.returns = np.asarray(returns)
        self.n_trials = n_trials
    
    def full_validation(self, confidence: float = 0.95,
                        n_simulations: int = 1000) -> Dict:
        """
        Run comprehensive validation.
        
        Returns dictionary with all validation results.
        """
        results = {}
        
        # 1. Basic statistics
        results['basic'] = {
            'n_observations': len(self.returns),
            'total_return': np.prod(1 + self.returns) - 1,
            'annualized_return': self.returns.mean() * 252,
            'annualized_vol': self.returns.std() * np.sqrt(252),
            'sharpe_ratio': self.returns.mean() / self.returns.std() * np.sqrt(252) if self.returns.std() > 0 else 0,
            'max_drawdown': self._max_drawdown(),
            'skewness': stats.skew(self.returns) if SCIPY_AVAILABLE else 0,
            'kurtosis': stats.kurtosis(self.returns) if SCIPY_AVAILABLE else 0
        }
        
        # 2. Monte Carlo tests
        mc = MonteCarloValidator(self.returns, n_simulations=n_simulations)
        results['monte_carlo'] = {
            'shuffle_test': mc.shuffle_simulation(),
            'null_test': mc.null_hypothesis_test(),
            'block_bootstrap': mc.block_bootstrap()
        }
        
        # 3. Deflated Sharpe Ratio
        if self.n_trials > 1:
            dsr = DeflatedSharpeRatio(self.returns, self.n_trials)
            results['deflated_sharpe'] = dsr.calculate_dsr()
        
        # 4. Bootstrap confidence intervals
        bootstrap = BootstrapCI(self.returns, n_bootstrap=n_simulations)
        results['confidence_intervals'] = {
            'sharpe': bootstrap.calculate_ci('sharpe', confidence),
            'total_return': bootstrap.calculate_ci('total_return', confidence),
            'max_drawdown': bootstrap.calculate_ci('max_drawdown', confidence)
        }
        
        # 5. Overall assessment
        results['assessment'] = self._overall_assessment(results)
        
        return results
    
    def _max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + self.returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return drawdown.max()
    
    def _overall_assessment(self, results: Dict) -> Dict:
        """Generate overall assessment from all tests."""
        warnings_list = []
        
        # Check Monte Carlo
        if results['monte_carlo']['shuffle_test']['p_value_sharpe'] > 0.05:
            warnings_list.append("Strategy may depend on market direction, not timing")
        
        if results['monte_carlo']['null_test']['p_value'] > 0.05:
            warnings_list.append("Cannot reject random walk null hypothesis")
        
        # Check bootstrap CI
        sharpe_ci = results['confidence_intervals']['sharpe']
        if sharpe_ci.confidence_interval[0] <= 0:
            warnings_list.append("Sharpe ratio confidence interval includes zero")
        
        # Check deflated Sharpe
        if 'deflated_sharpe' in results:
            if not results['deflated_sharpe']['significant']:
                warnings_list.append(f"Deflated Sharpe is negative after adjusting for {self.n_trials} trials")
        
        # Overall verdict
        n_warnings = len(warnings_list)
        
        if n_warnings == 0:
            verdict = "STRONG"
            confidence = "High confidence in strategy validity"
        elif n_warnings == 1:
            verdict = "MODERATE"
            confidence = "Some concerns about strategy validity"
        else:
            verdict = "WEAK"
            confidence = "Significant concerns about overfitting"
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'warnings': warnings_list,
            'n_warnings': n_warnings
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STRATEGY VALIDATION MODULE")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate sample strategy returns
    n_days = 500
    
    # Strategy with real alpha
    alpha = 0.0003  # Daily alpha
    noise = np.random.randn(n_days) * 0.02
    good_returns = alpha + noise
    
    # Strategy with no alpha (looks good by chance)
    bad_returns = np.random.randn(n_days) * 0.02
    bad_returns[::20] += 0.05  # Lucky streaks
    
    print("\n1. Good Strategy (has alpha)")
    print("-" * 50)
    
    validator = StrategyValidator(good_returns, n_trials=1)
    results = validator.full_validation(n_simulations=500)
    
    print(f"   Total Return: {results['basic']['total_return']:.2%}")
    print(f"   Sharpe Ratio: {results['basic']['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {results['basic']['max_drawdown']:.2%}")
    print(f"\n   Monte Carlo Shuffle p-value: {results['monte_carlo']['shuffle_test']['p_value_sharpe']:.4f}")
    print(f"   Null Hypothesis p-value: {results['monte_carlo']['null_test']['p_value']:.4f}")
    print(f"   Sharpe 95% CI: [{results['confidence_intervals']['sharpe'].confidence_interval[0]:.2f}, "
          f"{results['confidence_intervals']['sharpe'].confidence_interval[1]:.2f}]")
    print(f"\n   Assessment: {results['assessment']['verdict']}")
    print(f"   {results['assessment']['confidence']}")
    
    print("\n2. Bad Strategy (no alpha, lucky)")
    print("-" * 50)
    
    validator = StrategyValidator(bad_returns, n_trials=100)  # Simulating 100 trials
    results = validator.full_validation(n_simulations=500)
    
    print(f"   Total Return: {results['basic']['total_return']:.2%}")
    print(f"   Sharpe Ratio: {results['basic']['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {results['basic']['max_drawdown']:.2%}")
    print(f"\n   Monte Carlo Shuffle p-value: {results['monte_carlo']['shuffle_test']['p_value_sharpe']:.4f}")
    print(f"   Null Hypothesis p-value: {results['monte_carlo']['null_test']['p_value']:.4f}")
    
    if 'deflated_sharpe' in results:
        print(f"   Deflated Sharpe: {results['deflated_sharpe']['deflated_sharpe']:.2f}")
        print(f"   Expected Max Sharpe (by chance): {results['deflated_sharpe']['expected_max_sharpe']:.2f}")
    
    print(f"\n   Assessment: {results['assessment']['verdict']}")
    for warning in results['assessment']['warnings']:
        print(f"   ⚠️  {warning}")
    
    # PBO Demo
    print("\n3. Probability of Backtest Overfitting (PBO)")
    print("-" * 50)
    
    # Generate returns for multiple strategy variants
    n_strategies = 20
    strategy_returns = pd.DataFrame({
        f'strategy_{i}': alpha * (0.5 + np.random.rand()) + np.random.randn(n_days) * 0.02
        for i in range(n_strategies)
    })
    
    pbo_analyzer = PBOAnalyzer(strategy_returns, n_splits=8)
    pbo_result = pbo_analyzer.calculate_pbo()
    
    print(f"   Number of strategies tested: {n_strategies}")
    print(f"   PBO (Probability of Overfitting): {pbo_result.pbo:.2%}")
    print(f"   Performance Degradation: {pbo_result.performance_degradation:.4f}")
    print(f"   IS-OOS Rank Correlation: {pbo_result.rank_correlation:.2f}")
    print(f"   Confidence: {pbo_result.confidence.upper()}")
    print(f"   Is Overfit: {'Yes ⚠️' if pbo_result.is_overfit else 'No ✓'}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
1. MONTE CARLO SHUFFLE TEST:
   - p-value < 0.05: Strategy timing matters (good)
   - p-value > 0.05: Returns independent of order (may be just beta)

2. NULL HYPOTHESIS TEST:
   - p-value < 0.05: Reject random walk (strategy has edge)
   - p-value > 0.05: Cannot distinguish from random

3. DEFLATED SHARPE RATIO:
   - Positive: Strategy survives multiple testing adjustment
   - Negative: Expected by chance given number of trials

4. PROBABILITY OF BACKTEST OVERFITTING (PBO):
   - PBO < 0.25: Low risk of overfitting
   - PBO 0.25-0.50: Moderate risk
   - PBO > 0.50: High risk, likely overfit

5. BOOTSTRAP CONFIDENCE INTERVALS:
   - If CI excludes zero, metric is statistically significant
   - Wider CI = more uncertainty

RED FLAGS:
- Shuffle test p-value > 0.10
- Null test p-value > 0.10  
- Deflated Sharpe < 0
- PBO > 0.50
- Sharpe CI includes zero
""")
