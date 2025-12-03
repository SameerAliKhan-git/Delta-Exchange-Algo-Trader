"""
AFML Labeling Module

Implements labeling methods from Advances in Financial Machine Learning:
- Triple-Barrier Method
- Meta-Labeling
- Sample Weights
- Label Uniqueness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from numba import njit
from loguru import logger


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    profit_taking: float = 0.02  # 2% profit target
    stop_loss: float = 0.01  # 1% stop loss
    max_holding_period: int = 20  # Maximum bars to hold
    min_return: float = 0.0001  # Minimum return threshold
    vertical_barrier_times: Optional[pd.Series] = None  # Custom vertical barriers


# ==============================================================================
# TRIPLE-BARRIER METHOD (AFML Ch. 3)
# ==============================================================================

@njit
def _apply_triple_barrier_numba(
    prices: np.ndarray,
    barriers_upper: np.ndarray,
    barriers_lower: np.ndarray,
    max_periods: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated triple barrier application.
    
    Returns:
        labels: -1, 0, 1 for stop loss, time exit, profit taking
        exit_indices: Index where barrier was hit
        returns: Actual return at exit
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int32)
    exit_indices = np.zeros(n, dtype=np.int32)
    returns = np.zeros(n, dtype=np.float64)
    
    for i in range(n - 1):
        entry_price = prices[i]
        upper = barriers_upper[i]
        lower = barriers_lower[i]
        
        # Look forward until barrier hit or max_periods
        for j in range(1, min(max_periods + 1, n - i)):
            current_price = prices[i + j]
            current_return = (current_price / entry_price) - 1
            
            # Check upper barrier (profit taking)
            if current_price >= upper:
                labels[i] = 1
                exit_indices[i] = i + j
                returns[i] = current_return
                break
            
            # Check lower barrier (stop loss)
            if current_price <= lower:
                labels[i] = -1
                exit_indices[i] = i + j
                returns[i] = current_return
                break
            
            # Check vertical barrier (time expiry)
            if j == max_periods:
                # Label based on return sign
                if current_return > 0:
                    labels[i] = 1
                elif current_return < 0:
                    labels[i] = -1
                else:
                    labels[i] = 0
                exit_indices[i] = i + j
                returns[i] = current_return
                break
    
    return labels, exit_indices, returns


class TripleBarrierLabeler:
    """
    Triple-Barrier Labeling Method (AFML Ch. 3).
    
    Labels observations based on which barrier is touched first:
    - Upper barrier: Profit-taking level (label = 1)
    - Lower barrier: Stop-loss level (label = -1)
    - Vertical barrier: Time expiration (label based on return sign)
    
    Example:
        labeler = TripleBarrierLabeler(
            profit_taking=0.02,  # 2% target
            stop_loss=0.01,      # 1% stop
            max_holding=20       # 20 bars max
        )
        labels = labeler.fit_transform(prices, volatility)
    """
    
    def __init__(
        self,
        profit_taking: float = 0.02,
        stop_loss: float = 0.01,
        max_holding_period: int = 20,
        volatility_scaling: bool = True,
        min_return: float = 0.0001
    ):
        """
        Initialize labeler.
        
        Args:
            profit_taking: Profit target as fraction (or multiplier if volatility_scaling)
            stop_loss: Stop loss as fraction (or multiplier if volatility_scaling)
            max_holding_period: Maximum bars to hold position
            volatility_scaling: If True, barriers are multiples of volatility
            min_return: Minimum return to consider for labeling
        """
        self.profit_taking = profit_taking
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period
        self.volatility_scaling = volatility_scaling
        self.min_return = min_return
        
        # Results storage
        self.labels_: Optional[pd.Series] = None
        self.exit_times_: Optional[pd.Series] = None
        self.returns_: Optional[pd.Series] = None
        self.barriers_: Optional[pd.DataFrame] = None
    
    def _compute_barriers(
        self,
        prices: pd.Series,
        volatility: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute upper and lower barriers.
        
        Args:
            prices: Price series
            volatility: Optional volatility series for scaling
            
        Returns:
            Tuple of (upper_barriers, lower_barriers)
        """
        if self.volatility_scaling and volatility is not None:
            # Barriers are multiples of volatility
            upper = prices * (1 + self.profit_taking * volatility)
            lower = prices * (1 - self.stop_loss * volatility)
        else:
            # Fixed percentage barriers
            upper = prices * (1 + self.profit_taking)
            lower = prices * (1 - self.stop_loss)
        
        return upper, lower
    
    def fit_transform(
        self,
        prices: pd.Series,
        volatility: Optional[pd.Series] = None,
        events: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Generate labels using triple-barrier method.
        
        Args:
            prices: Close price series
            volatility: Optional daily volatility for scaling barriers
            events: Optional series indicating which bars to label
            
        Returns:
            Series with labels (-1, 0, 1)
        """
        logger.info(f"Applying triple-barrier labeling to {len(prices)} samples")
        
        if volatility is None:
            # Compute rolling volatility
            volatility = prices.pct_change().rolling(20).std()
        
        # Compute barriers
        upper, lower = self._compute_barriers(prices, volatility)
        
        # Apply triple barrier using Numba
        labels, exit_indices, returns = _apply_triple_barrier_numba(
            prices.values.astype(np.float64),
            upper.values.astype(np.float64),
            lower.values.astype(np.float64),
            self.max_holding_period
        )
        
        # Create output series
        self.labels_ = pd.Series(labels, index=prices.index, name='label')
        self.exit_times_ = pd.Series(exit_indices, index=prices.index, name='exit_idx')
        self.returns_ = pd.Series(returns, index=prices.index, name='return')
        
        # Store barriers
        self.barriers_ = pd.DataFrame({
            'price': prices,
            'upper': upper,
            'lower': lower
        }, index=prices.index)
        
        # Filter by events if provided
        if events is not None:
            self.labels_ = self.labels_[events]
        
        # Log statistics
        label_counts = self.labels_.value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        return self.labels_
    
    def get_barriers(self) -> pd.DataFrame:
        """Return computed barriers."""
        return self.barriers_
    
    def get_returns(self) -> pd.Series:
        """Return actual returns at exit."""
        return self.returns_


# ==============================================================================
# META-LABELING (AFML Ch. 3)
# ==============================================================================

class MetaLabeler:
    """
    Meta-Labeling (AFML Ch. 3).
    
    Two-stage labeling approach:
    1. Primary model predicts direction (long/short)
    2. Meta model predicts if primary model is correct
    
    This allows:
    - Bet sizing based on meta-model probability
    - Filtering out low-confidence primary signals
    - Secondary model can focus on different features
    
    Example:
        meta = MetaLabeler(primary_model)
        meta_labels = meta.fit_transform(prices, primary_signals)
    """
    
    def __init__(self, side_threshold: float = 0.0):
        """
        Initialize meta-labeler.
        
        Args:
            side_threshold: Threshold for primary signal (default 0 = any signal)
        """
        self.side_threshold = side_threshold
        self.meta_labels_: Optional[pd.Series] = None
    
    def fit_transform(
        self,
        prices: pd.Series,
        primary_signals: pd.Series,
        triple_barrier_labels: pd.Series
    ) -> pd.Series:
        """
        Generate meta-labels.
        
        Meta-label = 1 if primary signal direction matches actual outcome
        Meta-label = 0 if primary signal direction was wrong
        
        Args:
            prices: Price series
            primary_signals: Primary model signals (+1 long, -1 short, 0 neutral)
            triple_barrier_labels: Labels from triple-barrier method
            
        Returns:
            Meta-labels (0 or 1)
        """
        # Align indices
        common_idx = prices.index.intersection(
            primary_signals.index
        ).intersection(triple_barrier_labels.index)
        
        signals = primary_signals.loc[common_idx]
        tb_labels = triple_barrier_labels.loc[common_idx]
        
        # Filter by threshold
        active_signals = signals.abs() > self.side_threshold
        
        # Meta-label: 1 if signal direction matches outcome
        meta_labels = pd.Series(0, index=common_idx)
        
        # Long signal correct if price went up
        long_correct = (signals > 0) & (tb_labels > 0)
        
        # Short signal correct if price went down
        short_correct = (signals < 0) & (tb_labels < 0)
        
        meta_labels[long_correct | short_correct] = 1
        
        # Only keep labels where primary signal was active
        self.meta_labels_ = meta_labels[active_signals]
        
        logger.info(f"Meta-labeling: {self.meta_labels_.sum()}/{len(self.meta_labels_)} correct signals")
        
        return self.meta_labels_


# ==============================================================================
# SAMPLE WEIGHTS (AFML Ch. 4)
# ==============================================================================

def compute_sample_weights(
    labels: pd.Series,
    exit_times: pd.Series,
    close: pd.Series,
    decay: float = 1.0
) -> pd.Series:
    """
    Compute sample weights based on uniqueness (AFML Ch. 4).
    
    Samples that overlap less with other samples get higher weights.
    This reduces the impact of redundant observations.
    
    Args:
        labels: Label series
        exit_times: Exit time indices
        close: Close prices (for computing returns)
        decay: Time decay factor (1 = no decay)
        
    Returns:
        Sample weights
    """
    # Compute concurrent labels at each time
    t1 = exit_times.dropna()
    
    # Count number of labels active at each timestamp
    concurrent = pd.Series(0, index=close.index)
    
    for start_idx, end_idx in zip(t1.index, t1.values):
        # Get actual datetime indices
        start_loc = close.index.get_loc(start_idx)
        end_loc = int(end_idx) if end_idx < len(close) else len(close) - 1
        
        # Mark concurrent period
        concurrent.iloc[start_loc:end_loc + 1] += 1
    
    # Compute average uniqueness for each sample
    weights = pd.Series(index=labels.index, dtype=float)
    
    for start_idx, end_idx in zip(t1.index, t1.values):
        start_loc = close.index.get_loc(start_idx)
        end_loc = int(end_idx) if end_idx < len(close) else len(close) - 1
        
        # Average uniqueness = 1 / average concurrent labels
        avg_concurrent = concurrent.iloc[start_loc:end_loc + 1].mean()
        weights[start_idx] = 1.0 / avg_concurrent if avg_concurrent > 0 else 0
    
    # Apply time decay
    if decay < 1.0:
        time_weights = decay ** np.arange(len(weights))[::-1]
        weights = weights * time_weights
    
    # Normalize weights
    weights = weights / weights.sum()
    
    return weights


def compute_return_attribution(
    labels: pd.Series,
    returns: pd.Series,
    weights: pd.Series
) -> pd.DataFrame:
    """
    Attribute returns to different label types.
    
    Args:
        labels: Label series (-1, 0, 1)
        returns: Return series
        weights: Sample weights
        
    Returns:
        DataFrame with return attribution by label type
    """
    df = pd.DataFrame({
        'label': labels,
        'return': returns,
        'weight': weights
    }).dropna()
    
    attribution = df.groupby('label').agg({
        'return': ['sum', 'mean', 'std', 'count'],
        'weight': 'sum'
    })
    
    attribution.columns = ['total_return', 'mean_return', 'std_return', 'count', 'total_weight']
    
    return attribution


# ==============================================================================
# TREND SCANNING (AFML Ch. 3)
# ==============================================================================

@njit
def _trend_scanning_numba(
    prices: np.ndarray,
    t_events: np.ndarray,
    look_forward: int,
    min_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated trend scanning.
    
    Returns:
        t_values: T-statistic of trend
        labels: Direction labels
    """
    n = len(t_events)
    t_values = np.zeros(n)
    labels = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        idx = t_events[i]
        
        if idx + look_forward >= len(prices):
            continue
        
        # Get forward window
        window = prices[idx:idx + look_forward]
        
        if len(window) < min_samples:
            continue
        
        # Compute trend via linear regression
        x = np.arange(len(window))
        x_mean = x.mean()
        y_mean = window.mean()
        
        numerator = np.sum((x - x_mean) * (window - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            continue
        
        slope = numerator / denominator
        
        # Compute t-statistic
        y_pred = x_mean + slope * (x - x_mean)
        residuals = window - y_pred
        mse = np.sum(residuals ** 2) / (len(window) - 2)
        se_slope = np.sqrt(mse / denominator) if mse > 0 and denominator > 0 else 1e-10
        
        t_stat = slope / se_slope if se_slope > 0 else 0
        
        t_values[i] = t_stat
        labels[i] = np.sign(t_stat)
    
    return t_values, labels


def trend_scanning_labels(
    prices: pd.Series,
    events: pd.Series = None,
    look_forward: int = 20,
    min_samples: int = 5,
    t_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Trend Scanning Labels (AFML Ch. 3).
    
    Labels based on statistical significance of forward trend.
    
    Args:
        prices: Price series
        events: Event indices (or all indices if None)
        look_forward: Forward window for trend detection
        min_samples: Minimum samples for regression
        t_threshold: T-statistic threshold for significance
        
    Returns:
        DataFrame with t_value, label columns
    """
    if events is None:
        events = pd.Series(range(len(prices)), index=prices.index)
    
    t_events = events.values.astype(np.int64)
    
    t_values, labels = _trend_scanning_numba(
        prices.values.astype(np.float64),
        t_events,
        look_forward,
        min_samples
    )
    
    result = pd.DataFrame({
        't_value': t_values,
        'label': labels
    }, index=events.index)
    
    # Apply threshold
    result.loc[result['t_value'].abs() < t_threshold, 'label'] = 0
    
    return result


# ==============================================================================
# BINARY LABELS FOR CLASSIFICATION
# ==============================================================================

def to_binary_labels(
    labels: pd.Series,
    positive_class: int = 1
) -> pd.Series:
    """
    Convert multi-class labels to binary.
    
    Args:
        labels: Multi-class labels (-1, 0, 1)
        positive_class: Which class to treat as positive
        
    Returns:
        Binary labels (0, 1)
    """
    return (labels == positive_class).astype(int)


def to_multiclass_labels(
    labels: pd.Series,
    n_bins: int = 3
) -> pd.Series:
    """
    Convert continuous labels to multi-class.
    
    Args:
        labels: Continuous values
        n_bins: Number of bins/classes
        
    Returns:
        Multi-class labels
    """
    return pd.qcut(labels, q=n_bins, labels=range(n_bins))


# ==============================================================================
# LABEL ANALYSIS
# ==============================================================================

def analyze_labels(
    labels: pd.Series,
    returns: pd.Series,
    prices: pd.Series
) -> Dict:
    """
    Analyze label quality and distribution.
    
    Args:
        labels: Label series
        returns: Return series
        prices: Price series
        
    Returns:
        Dictionary with analysis metrics
    """
    analysis = {}
    
    # Label distribution
    counts = labels.value_counts()
    analysis['label_counts'] = counts.to_dict()
    analysis['label_percentages'] = (counts / len(labels) * 100).to_dict()
    
    # Return by label
    df = pd.DataFrame({'label': labels, 'return': returns}).dropna()
    by_label = df.groupby('label')['return'].agg(['mean', 'std', 'count'])
    analysis['return_by_label'] = by_label.to_dict()
    
    # Win rates
    if 1 in labels.values:
        long_returns = df[df['label'] == 1]['return']
        analysis['long_win_rate'] = (long_returns > 0).mean()
        analysis['long_avg_return'] = long_returns.mean()
    
    if -1 in labels.values:
        short_returns = df[df['label'] == -1]['return']
        analysis['short_win_rate'] = (short_returns < 0).mean()  # Short wins when negative
        analysis['short_avg_return'] = short_returns.mean()
    
    # Class imbalance
    max_count = counts.max()
    min_count = counts.min()
    analysis['class_imbalance_ratio'] = max_count / min_count if min_count > 0 else np.inf
    
    return analysis


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("AFML LABELING DEMO")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range('2020-01-01', periods=n, freq='1h')
    prices = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n)),
        index=dates
    )
    
    # Apply triple-barrier labeling
    labeler = TripleBarrierLabeler(
        profit_taking=0.02,
        stop_loss=0.01,
        max_holding_period=20,
        volatility_scaling=False
    )
    
    labels = labeler.fit_transform(prices)
    
    print("\n--- Triple-Barrier Labels ---")
    print(f"Label distribution:\n{labels.value_counts()}")
    
    # Analyze labels
    analysis = analyze_labels(labels, labeler.returns_, prices)
    print(f"\nLabel analysis:\n{analysis}")
    
    # Compute sample weights
    weights = compute_sample_weights(
        labels,
        labeler.exit_times_,
        prices
    )
    print(f"\nSample weights (first 10): {weights.head(10).values}")
    
    # Trend scanning
    trend = trend_scanning_labels(prices, look_forward=20, t_threshold=2.0)
    print(f"\nTrend scanning labels:\n{trend['label'].value_counts()}")
