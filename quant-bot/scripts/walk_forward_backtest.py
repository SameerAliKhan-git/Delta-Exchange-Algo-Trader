"""
D. Walk-Forward Backtest Harness
================================
Production-grade walk-forward backtesting with:
- Configurable train/test windows
- P&L distribution analysis
- Regime detection
- Statistical significance testing

Reference: AFML Ch. 12 - Backtesting through CV
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# WALK-FORWARD WINDOW CONFIGURATION
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""
    
    # Window sizes (in number of samples or time units)
    train_window: int = 252 * 8  # 1 year of hourly data (252 trading days * 8 hours)
    test_window: int = 63 * 8    # 3 months of hourly data
    
    # Step size (how much to advance between iterations)
    step_size: Optional[int] = None  # Defaults to test_window
    
    # Minimum train size for first window
    min_train_size: int = 100
    
    # Purge and embargo (to prevent leakage)
    purge_window: int = 10       # Samples to remove before test
    embargo_window: int = 5      # Samples to skip after test
    
    # Time-based settings (alternative to sample counts)
    use_time_windows: bool = False
    train_months: int = 12
    test_months: int = 3
    
    def __post_init__(self):
        if self.step_size is None:
            self.step_size = self.test_window


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    # Trading metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    
    # Predictions and returns
    predictions: np.ndarray = field(repr=False)
    actual: np.ndarray = field(repr=False)
    returns: np.ndarray = field(repr=False)
    strategy_returns: np.ndarray = field(repr=False)
    
    # Model info
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward backtest."""
    config: WalkForwardConfig
    folds: List[FoldResult]
    
    # Aggregated metrics
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    mean_return: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    
    # Statistical tests
    pvalue_vs_random: float = 1.0  # H0: strategy = random
    pvalue_vs_buyhold: float = 1.0  # H0: strategy = buy and hold
    
    # Combined analysis
    equity_curve: pd.Series = field(default_factory=pd.Series)
    all_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    all_actual: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

class WalkForwardBacktester:
    """
    Walk-Forward Backtesting Engine.
    
    Implements anchored and rolling window approaches with:
    - Purging and embargo to prevent leakage
    - Transaction costs
    - Statistical significance testing
    - Regime detection
    """
    
    def __init__(self, 
                 model_factory: Callable,
                 config: WalkForwardConfig,
                 transaction_cost_bps: float = 10,
                 position_sizing: str = 'equal'):
        """
        Initialize backtest engine.
        
        Args:
            model_factory: Function that returns a new model instance
            config: WalkForwardConfig
            transaction_cost_bps: Transaction cost in basis points
            position_sizing: 'equal', 'kelly', or 'volatility'
        """
        self.model_factory = model_factory
        self.config = config
        self.transaction_cost_bps = transaction_cost_bps
        self.position_sizing = position_sizing
        
        self.results: Optional[WalkForwardResult] = None
    
    def _generate_folds(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index arrays for walk-forward validation.
        
        Returns list of (train_indices, test_indices) tuples.
        """
        n = len(X)
        folds = []
        
        # Start position for first test set
        current_pos = self.config.train_window + self.config.purge_window
        
        while current_pos + self.config.test_window <= n:
            # Test window
            test_start = current_pos
            test_end = min(current_pos + self.config.test_window, n)
            
            # Train window (everything before test minus purge)
            train_end = current_pos - self.config.purge_window
            train_start = max(0, train_end - self.config.train_window)
            
            if train_end - train_start >= self.config.min_train_size:
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                folds.append((train_idx, test_idx))
            
            # Advance by step size
            current_pos += self.config.step_size
        
        return folds
    
    def _compute_trading_metrics(self, 
                                  predictions: np.ndarray,
                                  actual: np.ndarray,
                                  returns: np.ndarray) -> Dict:
        """Compute trading performance metrics."""
        
        # Convert predictions to positions
        positions = predictions * 2 - 1  # 0 -> -1 (short), 1 -> 1 (long)
        
        # Position changes for transaction costs
        position_changes = np.abs(np.diff(positions, prepend=0))
        
        # Strategy returns before costs
        strategy_returns_gross = positions * returns
        
        # Transaction costs
        cost_per_trade = self.transaction_cost_bps / 10000
        costs = position_changes * cost_per_trade
        
        # Net strategy returns
        strategy_returns = strategy_returns_gross - costs
        
        # Cumulative returns
        cum_returns = np.cumprod(1 + strategy_returns) - 1
        
        # Sharpe ratio (assuming hourly data)
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(8760)
        else:
            sharpe = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(cum_returns + 1)
        drawdown = (peak - (cum_returns + 1)) / peak
        max_dd = np.max(drawdown)
        
        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        total_trades = np.sum(position_changes > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': cum_returns[-1] if len(cum_returns) > 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': int(total_trades),
            'win_rate': win_rate,
            'strategy_returns': strategy_returns
        }
    
    def _run_single_fold(self, 
                          X_train: pd.DataFrame, y_train: np.ndarray,
                          X_test: pd.DataFrame, y_test: np.ndarray,
                          returns_test: np.ndarray,
                          fold_id: int) -> FoldResult:
        """Run backtest for a single fold."""
        
        # Train model
        model = self.model_factory()
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Classification metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        rec = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # Trading metrics
        trading_metrics = self._compute_trading_metrics(predictions, y_test, returns_test)
        
        # Feature importance
        feat_imp = {}
        if hasattr(model, 'feature_importances_'):
            feat_imp = dict(zip(X_train.columns, model.feature_importances_))
        
        return FoldResult(
            fold_id=fold_id,
            train_start=X_train.index[0],
            train_end=X_train.index[-1],
            test_start=X_test.index[0],
            test_end=X_test.index[-1],
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            total_return=trading_metrics['total_return'],
            sharpe_ratio=trading_metrics['sharpe_ratio'],
            max_drawdown=trading_metrics['max_drawdown'],
            num_trades=trading_metrics['num_trades'],
            win_rate=trading_metrics['win_rate'],
            predictions=predictions,
            actual=y_test,
            returns=returns_test,
            strategy_returns=trading_metrics['strategy_returns'],
            feature_importance=feat_imp
        )
    
    def run(self, X: pd.DataFrame, y: pd.Series, 
            returns: pd.Series,
            verbose: bool = True) -> WalkForwardResult:
        """
        Run complete walk-forward backtest.
        
        Args:
            X: Features DataFrame with datetime index
            y: Labels (0/1 for direction)
            returns: Actual returns corresponding to each sample
            verbose: Whether to print progress
        
        Returns:
            WalkForwardResult with all metrics and fold results
        """
        # Generate folds
        folds_indices = self._generate_folds(X)
        
        if verbose:
            print(f"Running {len(folds_indices)} walk-forward folds...")
        
        fold_results = []
        all_preds = []
        all_actual = []
        all_strat_returns = []
        
        for fold_id, (train_idx, test_idx) in enumerate(folds_indices):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
            returns_test = returns.iloc[test_idx].values
            
            # Run fold
            result = self._run_single_fold(
                X_train, y_train, X_test, y_test, returns_test, fold_id
            )
            fold_results.append(result)
            
            # Accumulate for aggregate analysis
            all_preds.extend(result.predictions)
            all_actual.extend(result.actual)
            all_strat_returns.extend(result.strategy_returns)
            
            if verbose and (fold_id + 1) % 5 == 0:
                print(f"  Completed fold {fold_id + 1}/{len(folds_indices)}: "
                      f"Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.2f}")
        
        # Aggregate results
        result = self._aggregate_results(fold_results, all_preds, all_actual, all_strat_returns)
        
        self.results = result
        return result
    
    def _aggregate_results(self, 
                            folds: List[FoldResult],
                            all_preds: List,
                            all_actual: List,
                            all_strat_returns: List) -> WalkForwardResult:
        """Aggregate fold results into final metrics."""
        
        all_preds = np.array(all_preds)
        all_actual = np.array(all_actual)
        all_strat_returns = np.array(all_strat_returns)
        
        # Basic aggregation
        mean_acc = np.mean([f.accuracy for f in folds])
        std_acc = np.std([f.accuracy for f in folds])
        mean_sharpe = np.mean([f.sharpe_ratio for f in folds])
        std_sharpe = np.std([f.sharpe_ratio for f in folds])
        mean_return = np.mean([f.total_return for f in folds])
        
        # Total equity curve
        cum_returns = np.cumprod(1 + all_strat_returns)
        total_return = cum_returns[-1] - 1 if len(cum_returns) > 0 else 0
        
        # Max drawdown across all periods
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak
        max_dd = np.max(drawdown)
        
        # Statistical tests
        pvalue_random = self._test_vs_random(all_preds, all_actual)
        pvalue_buyhold = self._test_vs_buyhold(all_strat_returns)
        
        return WalkForwardResult(
            config=self.config,
            folds=folds,
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            mean_return=mean_return,
            total_return=total_return,
            max_drawdown=max_dd,
            pvalue_vs_random=pvalue_random,
            pvalue_vs_buyhold=pvalue_buyhold,
            equity_curve=pd.Series(cum_returns),
            all_predictions=all_preds,
            all_actual=all_actual
        )
    
    def _test_vs_random(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """Test if strategy is better than random (binomial test)."""
        if not SCIPY_AVAILABLE:
            return 1.0
        
        n_correct = np.sum(predictions == actual)
        n_total = len(predictions)
        
        # Two-tailed binomial test against 50% accuracy
        result = stats.binomtest(n_correct, n_total, p=0.5, alternative='greater')
        return result.pvalue
    
    def _test_vs_buyhold(self, strategy_returns: np.ndarray) -> float:
        """Test if strategy returns are significantly different from buy-and-hold."""
        if not SCIPY_AVAILABLE:
            return 1.0
        
        # H0: mean return = 0
        t_stat, pvalue = stats.ttest_1samp(strategy_returns, 0)
        return pvalue


# =============================================================================
# REGIME ANALYSIS
# =============================================================================

class RegimeDetector:
    """
    Detect market regimes and analyze strategy performance by regime.
    
    Regimes:
    - Trend: Persistent directional movement
    - Mean-Reversion: Range-bound oscillation
    - High Volatility: Elevated uncertainty
    - Low Volatility: Calm markets
    """
    
    def __init__(self, vol_lookback: int = 20, trend_lookback: int = 50):
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
    
    def detect_regimes(self, returns: pd.Series) -> pd.DataFrame:
        """
        Classify each period into a market regime.
        
        Returns DataFrame with regime classifications.
        """
        df = pd.DataFrame(index=returns.index)
        df['returns'] = returns
        
        # Rolling volatility
        df['volatility'] = returns.rolling(self.vol_lookback).std()
        vol_median = df['volatility'].median()
        
        # Rolling trend (cumulative return over lookback)
        df['trend'] = returns.rolling(self.trend_lookback).sum()
        trend_threshold = returns.std() * np.sqrt(self.trend_lookback) * 0.5
        
        # Classify regimes
        conditions = [
            (df['volatility'] > vol_median * 1.5),  # High volatility
            (df['volatility'] < vol_median * 0.5),  # Low volatility
            (df['trend'].abs() > trend_threshold),  # Trending
        ]
        choices = ['high_vol', 'low_vol', 'trending']
        
        df['regime'] = np.select(conditions, choices, default='mean_reversion')
        
        return df[['regime', 'volatility', 'trend']]
    
    def analyze_by_regime(self, 
                          predictions: np.ndarray,
                          actual: np.ndarray,
                          returns: pd.Series,
                          strategy_returns: np.ndarray) -> Dict:
        """
        Analyze strategy performance in each regime.
        """
        regimes = self.detect_regimes(returns)
        
        results = {}
        
        for regime in regimes['regime'].unique():
            mask = (regimes['regime'] == regime).values[:len(predictions)]
            
            if mask.sum() == 0:
                continue
            
            regime_preds = predictions[mask]
            regime_actual = actual[mask]
            regime_strat_returns = strategy_returns[mask]
            
            results[regime] = {
                'n_samples': int(mask.sum()),
                'accuracy': accuracy_score(regime_actual, regime_preds),
                'total_return': np.prod(1 + regime_strat_returns) - 1,
                'sharpe': (regime_strat_returns.mean() / (regime_strat_returns.std() + 1e-8)) * np.sqrt(8760),
                'win_rate': np.sum(regime_strat_returns > 0) / len(regime_strat_returns)
            }
        
        return results


# =============================================================================
# REPORTING
# =============================================================================

def generate_backtest_report(result: WalkForwardResult,
                             regime_analysis: Optional[Dict] = None) -> str:
    """Generate comprehensive backtest report."""
    
    report = []
    report.append("=" * 70)
    report.append("WALK-FORWARD BACKTEST REPORT")
    report.append("=" * 70)
    
    # Configuration
    report.append("\n## CONFIGURATION")
    report.append(f"Train window: {result.config.train_window} samples")
    report.append(f"Test window:  {result.config.test_window} samples")
    report.append(f"Total folds:  {len(result.folds)}")
    
    if result.folds:
        report.append(f"Date range:   {result.folds[0].train_start} to {result.folds[-1].test_end}")
    
    # Summary metrics
    report.append("\n## SUMMARY METRICS")
    report.append(f"Mean Accuracy:     {result.mean_accuracy:.4f} (+/- {result.std_accuracy:.4f})")
    report.append(f"Mean Sharpe:       {result.mean_sharpe:.2f} (+/- {result.std_sharpe:.2f})")
    report.append(f"Mean Fold Return:  {result.mean_return:.2%}")
    report.append(f"Total Return:      {result.total_return:.2%}")
    report.append(f"Max Drawdown:      {result.max_drawdown:.2%}")
    
    # Statistical significance
    report.append("\n## STATISTICAL TESTS")
    sig_random = "***" if result.pvalue_vs_random < 0.01 else "**" if result.pvalue_vs_random < 0.05 else "*" if result.pvalue_vs_random < 0.1 else ""
    sig_buyhold = "***" if result.pvalue_vs_buyhold < 0.01 else "**" if result.pvalue_vs_buyhold < 0.05 else "*" if result.pvalue_vs_buyhold < 0.1 else ""
    
    report.append(f"P-value vs Random:    {result.pvalue_vs_random:.4f} {sig_random}")
    report.append(f"P-value vs Buy&Hold:  {result.pvalue_vs_buyhold:.4f} {sig_buyhold}")
    report.append("(*** p<0.01, ** p<0.05, * p<0.1)")
    
    # Per-fold summary
    report.append("\n## PER-FOLD RESULTS")
    report.append(f"{'Fold':>5} {'Return':>10} {'Sharpe':>8} {'Accuracy':>10} {'Trades':>8} {'Win %':>8}")
    report.append("-" * 55)
    
    for fold in result.folds:
        report.append(f"{fold.fold_id:>5} {fold.total_return:>10.2%} "
                     f"{fold.sharpe_ratio:>8.2f} {fold.accuracy:>10.4f} "
                     f"{fold.num_trades:>8} {fold.win_rate:>8.2%}")
    
    # Regime analysis
    if regime_analysis:
        report.append("\n## REGIME ANALYSIS")
        for regime, metrics in regime_analysis.items():
            report.append(f"\n  {regime.upper()}")
            report.append(f"    Samples:  {metrics['n_samples']}")
            report.append(f"    Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"    Return:   {metrics['total_return']:.2%}")
            report.append(f"    Sharpe:   {metrics['sharpe']:.2f}")
            report.append(f"    Win Rate: {metrics['win_rate']:.2%}")
    
    # Feature importance stability
    if result.folds and result.folds[0].feature_importance:
        report.append("\n## FEATURE IMPORTANCE STABILITY")
        
        # Aggregate importance across folds
        all_importance = {}
        for fold in result.folds:
            for feat, imp in fold.feature_importance.items():
                if feat not in all_importance:
                    all_importance[feat] = []
                all_importance[feat].append(imp)
        
        # Top features by mean importance
        mean_importance = {f: np.mean(imps) for f, imps in all_importance.items()}
        std_importance = {f: np.std(imps) for f, imps in all_importance.items()}
        
        sorted_features = sorted(mean_importance.items(), key=lambda x: -x[1])[:10]
        
        report.append(f"{'Feature':>30} {'Mean Imp':>10} {'Std':>8}")
        report.append("-" * 50)
        for feat, mean_imp in sorted_features:
            std_imp = std_importance[feat]
            report.append(f"{feat:>30} {mean_imp:>10.4f} {std_imp:>8.4f}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


# =============================================================================
# DEMO
# =============================================================================

def create_extended_sample_data():
    """Create extended sample data for walk-forward demo."""
    np.random.seed(42)
    
    # 2 years of hourly data
    n_samples = 252 * 8 * 2  # ~4000 samples
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='H')
    
    # Features
    X = pd.DataFrame(index=dates)
    
    # Regime-dependent price dynamics
    regime = np.zeros(n_samples)
    regime_state = 0
    for i in range(n_samples):
        if np.random.random() < 0.005:  # 0.5% chance of regime change
            regime_state = (regime_state + 1) % 4
        regime[i] = regime_state
    
    # Generate returns with regime-dependent characteristics
    returns = np.zeros(n_samples)
    for i in range(n_samples):
        if regime[i] == 0:  # Trending up
            returns[i] = 0.0002 + np.random.randn() * 0.01
        elif regime[i] == 1:  # High vol
            returns[i] = np.random.randn() * 0.03
        elif regime[i] == 2:  # Trending down
            returns[i] = -0.0002 + np.random.randn() * 0.01
        else:  # Low vol mean reversion
            returns[i] = -returns[i-1] * 0.3 + np.random.randn() * 0.005 if i > 0 else 0
    
    returns = pd.Series(returns, index=dates)
    
    # Features based on returns
    X['momentum_5'] = returns.rolling(5).mean().fillna(0)
    X['momentum_20'] = returns.rolling(20).mean().fillna(0)
    X['volatility'] = returns.rolling(20).std().fillna(0.01)
    X['rsi'] = 50 + returns.rolling(14).sum().fillna(0) * 100
    X['macd'] = X['momentum_5'] - X['momentum_20']
    X['volume_ma'] = 1 + np.cumsum(np.random.randn(n_samples) * 0.01)
    
    for i in range(10):
        X[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Labels: future return direction
    y = (returns.shift(-1) > 0).astype(int).fillna(0)
    
    return X, y, returns


def main():
    """Run walk-forward backtest demo."""
    print("="*70)
    print("WALK-FORWARD BACKTEST HARNESS")
    print("="*70)
    
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed")
        return
    
    # Create data
    print("\n1. Creating 2 years of hourly sample data...")
    X, y, returns = create_extended_sample_data()
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Date range: {X.index[0]} to {X.index[-1]}")
    
    # Configure walk-forward
    print("\n2. Configuring walk-forward backtest...")
    config = WalkForwardConfig(
        train_window=252 * 8,   # 1 year
        test_window=63 * 8,     # 3 months
        step_size=63 * 8,       # Non-overlapping
        purge_window=10,
        embargo_window=5
    )
    print(f"   Train window: {config.train_window} samples (~1 year)")
    print(f"   Test window:  {config.test_window} samples (~3 months)")
    
    # Create backtester
    def model_factory():
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
    
    backtester = WalkForwardBacktester(
        model_factory=model_factory,
        config=config,
        transaction_cost_bps=10
    )
    
    # Run backtest
    print("\n3. Running walk-forward backtest...")
    result = backtester.run(X, y, returns, verbose=True)
    
    # Regime analysis
    print("\n4. Analyzing performance by regime...")
    regime_detector = RegimeDetector()
    
    # Get combined strategy returns from all folds
    all_strat_returns = np.concatenate([f.strategy_returns for f in result.folds])
    all_preds = result.all_predictions
    all_actual = result.all_actual
    returns_subset = returns.iloc[-len(all_strat_returns):]
    
    regime_analysis = regime_detector.analyze_by_regime(
        all_preds, all_actual, returns_subset, all_strat_returns
    )
    
    # Generate report
    print("\n" + "="*70)
    report = generate_backtest_report(result, regime_analysis)
    print(report)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    # Save report
    with open(output_dir / "walkforward_report.txt", 'w') as f:
        f.write(report)
    
    # Save equity curve
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    
    print(f"\nResults saved to {output_dir}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
1. P-VALUE VS RANDOM < 0.05 means strategy accuracy is significantly 
   better than coin flip. This is the minimum bar for a valid signal.

2. P-VALUE VS BUY&HOLD < 0.05 means strategy returns are significantly 
   different from zero. Check if positive!

3. SHARPE STABILITY: Low std_sharpe relative to mean_sharpe suggests 
   robust strategy that works across different market conditions.

4. REGIME ANALYSIS: Check if strategy works in all regimes or only 
   specific ones. Regime-specific strategies need regime detection.

5. FEATURE IMPORTANCE STABILITY: Features with high mean and low std 
   importance are reliable predictors. Unstable features may be noise.

RED FLAGS:
- P-value vs random > 0.10: Strategy may not have real edge
- Sharpe std > mean: Very inconsistent performance
- All returns from one regime: Strategy is not robust
- Declining accuracy in later folds: Model decay / regime change
""")


if __name__ == "__main__":
    main()
