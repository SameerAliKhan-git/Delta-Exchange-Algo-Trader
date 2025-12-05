"""
Concept Drift Detection
=======================
Automated monitoring for model performance degradation.

This module provides:
- Statistical drift detection (PSI, KS test)
- Feature drift monitoring
- Model performance tracking
- Automated retraining triggers

Based on concepts from:
- López de Prado's ML asset allocation
- Production ML monitoring best practices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DRIFT DETECTION DATA CLASSES
# =============================================================================

@dataclass
class DriftResult:
    """Result from drift detection."""
    is_drift: bool
    metric_name: str
    drift_score: float
    threshold: float
    p_value: Optional[float]
    reference_stats: Dict
    current_stats: Dict
    timestamp: datetime
    
    
@dataclass
class PerformanceDegradation:
    """Performance degradation alert."""
    is_degraded: bool
    metric_name: str
    reference_value: float
    current_value: float
    degradation_pct: float
    threshold_pct: float
    timestamp: datetime


# =============================================================================
# STATISTICAL DRIFT TESTS
# =============================================================================

class StatisticalDriftTests:
    """
    Statistical tests for distribution drift.
    """
    
    @staticmethod
    def psi(reference: np.ndarray, current: np.ndarray,
            n_bins: int = 10) -> float:
        """
        Population Stability Index (PSI).
        
        Measures how much a distribution has shifted.
        
        PSI < 0.1: No significant shift
        PSI 0.1-0.25: Moderate shift
        PSI > 0.25: Significant shift
        
        Args:
            reference: Reference (training) distribution
            current: Current (production) distribution
            n_bins: Number of bins for histogram
        
        Returns:
            PSI score
        """
        # Create bins from reference
        _, bin_edges = np.histogram(reference, bins=n_bins)
        
        # Count in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.clip(ref_props, 1e-10, 1)
        cur_props = np.clip(cur_props, 1e-10, 1)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return psi
    
    @staticmethod
    def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution difference.
        
        Returns:
            (statistic, p_value)
        """
        try:
            from scipy.stats import ks_2samp
            stat, p_value = ks_2samp(reference, current)
            return stat, p_value
        except ImportError:
            # Manual KS test
            ref_sorted = np.sort(reference)
            cur_sorted = np.sort(current)
            
            all_values = np.sort(np.concatenate([ref_sorted, cur_sorted]))
            
            ref_cdf = np.searchsorted(ref_sorted, all_values, side='right') / len(reference)
            cur_cdf = np.searchsorted(cur_sorted, all_values, side='right') / len(current)
            
            statistic = np.max(np.abs(ref_cdf - cur_cdf))
            
            # Approximate p-value (simplified)
            n = len(reference) * len(current) / (len(reference) + len(current))
            p_value = np.exp(-2 * statistic ** 2 * n)
            
            return statistic, p_value
    
    @staticmethod
    def wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
        """
        Earth Mover's Distance (Wasserstein distance).
        
        Measures minimum "work" to transform one distribution to another.
        """
        try:
            from scipy.stats import wasserstein_distance
            return wasserstein_distance(reference, current)
        except ImportError:
            # Simplified 1D Wasserstein
            ref_sorted = np.sort(reference)
            cur_sorted = np.sort(current)
            
            # Resample to same length
            n = min(len(ref_sorted), len(cur_sorted))
            ref_resampled = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(ref_sorted)),
                ref_sorted
            )
            cur_resampled = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(cur_sorted)),
                cur_sorted
            )
            
            return np.mean(np.abs(ref_resampled - cur_resampled))
    
    @staticmethod
    def chi_square_test(reference: np.ndarray, current: np.ndarray,
                        n_bins: int = 10) -> Tuple[float, float]:
        """
        Chi-square test for categorical/binned data.
        """
        try:
            from scipy.stats import chi2_contingency
            
            _, bin_edges = np.histogram(reference, bins=n_bins)
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Avoid zero counts
            ref_counts = ref_counts + 1
            cur_counts = cur_counts + 1
            
            contingency = np.array([ref_counts, cur_counts])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            
            return chi2, p_value
        except ImportError:
            # Simplified chi-square
            _, bin_edges = np.histogram(reference, bins=n_bins)
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)
            
            expected = ref_counts * len(current) / len(reference)
            expected = np.clip(expected, 1, None)
            
            chi2 = np.sum((cur_counts - expected) ** 2 / expected)
            
            # Approximate p-value
            df = n_bins - 1
            p_value = 1 - min(chi2 / (2 * df), 1)  # Very rough approximation
            
            return chi2, p_value


# =============================================================================
# FEATURE DRIFT MONITOR
# =============================================================================

class FeatureDriftMonitor:
    """
    Monitor feature distributions for drift.
    """
    
    def __init__(self, reference_data: pd.DataFrame,
                 psi_threshold: float = 0.25,
                 ks_alpha: float = 0.05):
        """
        Initialize with reference data.
        
        Args:
            reference_data: Training/reference data
            psi_threshold: PSI threshold for drift alert
            ks_alpha: Significance level for KS test
        """
        self.reference_data = reference_data
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        
        # Store reference statistics
        self.reference_stats = {}
        for col in reference_data.columns:
            self.reference_stats[col] = {
                'mean': reference_data[col].mean(),
                'std': reference_data[col].std(),
                'min': reference_data[col].min(),
                'max': reference_data[col].max(),
                'median': reference_data[col].median(),
                'q25': reference_data[col].quantile(0.25),
                'q75': reference_data[col].quantile(0.75)
            }
    
    def check_drift(self, current_data: pd.DataFrame,
                   features: Optional[List[str]] = None) -> Dict[str, DriftResult]:
        """
        Check for drift in features.
        
        Returns dict of feature -> DriftResult
        """
        if features is None:
            features = [c for c in current_data.columns 
                       if c in self.reference_data.columns]
        
        results = {}
        
        for feature in features:
            if feature not in self.reference_data.columns:
                continue
            
            ref_values = self.reference_data[feature].dropna().values
            cur_values = current_data[feature].dropna().values
            
            if len(cur_values) < 10:
                continue
            
            # Calculate PSI
            psi = StatisticalDriftTests.psi(ref_values, cur_values)
            
            # KS test
            ks_stat, ks_pvalue = StatisticalDriftTests.ks_test(ref_values, cur_values)
            
            # Determine if drift detected
            is_drift = psi > self.psi_threshold or ks_pvalue < self.ks_alpha
            
            # Current statistics
            current_stats = {
                'mean': np.mean(cur_values),
                'std': np.std(cur_values),
                'min': np.min(cur_values),
                'max': np.max(cur_values),
                'median': np.median(cur_values)
            }
            
            results[feature] = DriftResult(
                is_drift=is_drift,
                metric_name=feature,
                drift_score=psi,
                threshold=self.psi_threshold,
                p_value=ks_pvalue,
                reference_stats=self.reference_stats[feature],
                current_stats=current_stats,
                timestamp=datetime.now()
            )
        
        return results
    
    def get_drift_summary(self, drift_results: Dict[str, DriftResult]) -> pd.DataFrame:
        """Get summary DataFrame of drift results."""
        summary = []
        
        for feature, result in drift_results.items():
            summary.append({
                'feature': feature,
                'is_drift': result.is_drift,
                'psi': result.drift_score,
                'p_value': result.p_value,
                'ref_mean': result.reference_stats['mean'],
                'cur_mean': result.current_stats['mean'],
                'mean_shift': result.current_stats['mean'] - result.reference_stats['mean']
            })
        
        return pd.DataFrame(summary).sort_values('psi', ascending=False)


# =============================================================================
# MODEL PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """
    Monitor model performance metrics over time.
    """
    
    def __init__(self, reference_metrics: Dict[str, float],
                 degradation_threshold: float = 0.20,
                 window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            reference_metrics: Baseline metrics from validation
            degradation_threshold: Percentage drop to trigger alert
            window_size: Rolling window for metric calculation
        """
        self.reference_metrics = reference_metrics
        self.degradation_threshold = degradation_threshold
        self.window_size = window_size
        
        # Metric history
        self.metric_history: Dict[str, deque] = {
            metric: deque(maxlen=window_size * 10)
            for metric in reference_metrics
        }
        
        # Prediction history
        self.predictions: deque = deque(maxlen=window_size * 100)
        self.actuals: deque = deque(maxlen=window_size * 100)
        self.timestamps: deque = deque(maxlen=window_size * 100)
    
    def log_prediction(self, prediction: float, actual: float,
                      timestamp: Optional[datetime] = None):
        """Log a prediction and actual value."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.now())
    
    def calculate_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
        """Calculate current metrics from recent predictions."""
        if len(self.predictions) < 10:
            return {}
        
        n = window or self.window_size
        preds = np.array(list(self.predictions)[-n:])
        actuals = np.array(list(self.actuals)[-n:])
        
        # Calculate various metrics
        metrics = {}
        
        # Directional accuracy (for trading)
        if np.any(actuals != 0):
            direction_correct = np.sign(preds) == np.sign(actuals)
            metrics['hit_rate'] = np.mean(direction_correct)
        
        # MSE
        metrics['mse'] = np.mean((preds - actuals) ** 2)
        
        # MAE
        metrics['mae'] = np.mean(np.abs(preds - actuals))
        
        # Correlation
        if np.std(preds) > 0 and np.std(actuals) > 0:
            metrics['correlation'] = np.corrcoef(preds, actuals)[0, 1]
        
        # Trading-specific: Sharpe-like metric
        returns = preds * actuals  # Profit if prediction correct
        if len(returns) > 0 and np.std(returns) > 0:
            metrics['sharpe_proxy'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return metrics
    
    def check_degradation(self) -> List[PerformanceDegradation]:
        """Check for performance degradation."""
        current_metrics = self.calculate_metrics()
        alerts = []
        
        for metric, reference_value in self.reference_metrics.items():
            if metric not in current_metrics:
                continue
            
            current_value = current_metrics[metric]
            
            # Calculate degradation percentage
            if reference_value != 0:
                degradation_pct = (reference_value - current_value) / abs(reference_value)
            else:
                degradation_pct = abs(current_value)
            
            # For metrics where higher is better (hit_rate, sharpe)
            if metric in ['hit_rate', 'correlation', 'sharpe_proxy']:
                is_degraded = degradation_pct > self.degradation_threshold
            # For metrics where lower is better (mse, mae)
            else:
                is_degraded = -degradation_pct > self.degradation_threshold
            
            alerts.append(PerformanceDegradation(
                is_degraded=is_degraded,
                metric_name=metric,
                reference_value=reference_value,
                current_value=current_value,
                degradation_pct=degradation_pct,
                threshold_pct=self.degradation_threshold,
                timestamp=datetime.now()
            ))
        
        return alerts


# =============================================================================
# CUSUM DRIFT DETECTOR
# =============================================================================

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change detection.
    
    Detects shifts in the mean of a time series.
    Good for detecting concept drift in real-time.
    """
    
    def __init__(self, threshold: float = 5.0, drift: float = 0.5,
                 warm_up: int = 50):
        """
        Initialize CUSUM detector.
        
        Args:
            threshold: Detection threshold (h)
            drift: Allowable drift (k)
            warm_up: Number of samples for mean estimation
        """
        self.threshold = threshold
        self.drift = drift
        self.warm_up = warm_up
        
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.values: List[float] = []
        self.g_plus = 0
        self.g_minus = 0
        self.mean = None
        self.std = None
        self.change_detected = False
        self.change_points: List[int] = []
    
    def update(self, value: float) -> bool:
        """
        Update with new value.
        
        Returns True if change detected.
        """
        self.values.append(value)
        
        # Warm-up period
        if len(self.values) <= self.warm_up:
            if len(self.values) == self.warm_up:
                self.mean = np.mean(self.values)
                self.std = np.std(self.values)
                if self.std < 1e-10:
                    self.std = 1.0
            return False
        
        # Standardize
        z = (value - self.mean) / self.std
        
        # Update CUSUM
        self.g_plus = max(0, self.g_plus + z - self.drift)
        self.g_minus = max(0, self.g_minus - z - self.drift)
        
        # Check for change
        if self.g_plus > self.threshold or self.g_minus > self.threshold:
            self.change_detected = True
            self.change_points.append(len(self.values))
            
            # Reset after detection
            self.g_plus = 0
            self.g_minus = 0
            
            return True
        
        return False
    
    def get_status(self) -> Dict:
        """Get current detector status."""
        return {
            'change_detected': self.change_detected,
            'n_changes': len(self.change_points),
            'change_points': self.change_points,
            'current_g_plus': self.g_plus,
            'current_g_minus': self.g_minus,
            'reference_mean': self.mean,
            'reference_std': self.std
        }


# =============================================================================
# DRIFT ALERTING SYSTEM
# =============================================================================

class DriftAlertSystem:
    """
    Comprehensive drift alerting system.
    
    Combines multiple drift detection methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize alerting system.
        
        Args:
            config: Configuration dict with thresholds
        """
        self.config = config or {
            'psi_threshold': 0.25,
            'ks_alpha': 0.05,
            'performance_degradation': 0.20,
            'cusum_threshold': 5.0,
            'min_samples': 100
        }
        
        self.feature_monitor: Optional[FeatureDriftMonitor] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.cusum_detectors: Dict[str, CUSUMDetector] = {}
        
        # Alert history
        self.alerts: List[Dict] = []
    
    def initialize_feature_monitor(self, reference_data: pd.DataFrame):
        """Initialize feature drift monitoring."""
        self.feature_monitor = FeatureDriftMonitor(
            reference_data,
            psi_threshold=self.config['psi_threshold'],
            ks_alpha=self.config['ks_alpha']
        )
    
    def initialize_performance_monitor(self, reference_metrics: Dict[str, float]):
        """Initialize performance monitoring."""
        self.performance_monitor = PerformanceMonitor(
            reference_metrics,
            degradation_threshold=self.config['performance_degradation']
        )
    
    def add_cusum_monitor(self, name: str):
        """Add CUSUM monitor for a metric."""
        self.cusum_detectors[name] = CUSUMDetector(
            threshold=self.config['cusum_threshold']
        )
    
    def check_all(self, current_features: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run all drift checks.
        
        Returns comprehensive drift report.
        """
        report = {
            'timestamp': datetime.now(),
            'feature_drift': None,
            'performance_degradation': None,
            'cusum_changes': {},
            'requires_retraining': False,
            'alerts': []
        }
        
        # Feature drift
        if self.feature_monitor is not None and current_features is not None:
            drift_results = self.feature_monitor.check_drift(current_features)
            
            drifted_features = [
                name for name, result in drift_results.items() 
                if result.is_drift
            ]
            
            report['feature_drift'] = {
                'n_features_checked': len(drift_results),
                'n_features_drifted': len(drifted_features),
                'drifted_features': drifted_features,
                'drift_scores': {
                    name: result.drift_score 
                    for name, result in drift_results.items()
                }
            }
            
            if drifted_features:
                report['alerts'].append({
                    'type': 'feature_drift',
                    'severity': 'high' if len(drifted_features) > 3 else 'medium',
                    'message': f"Drift detected in {len(drifted_features)} features: {drifted_features[:5]}"
                })
        
        # Performance degradation
        if self.performance_monitor is not None:
            degradations = self.performance_monitor.check_degradation()
            
            degraded_metrics = [
                d for d in degradations if d.is_degraded
            ]
            
            report['performance_degradation'] = {
                'n_metrics_checked': len(degradations),
                'n_metrics_degraded': len(degraded_metrics),
                'degradations': [
                    {
                        'metric': d.metric_name,
                        'reference': d.reference_value,
                        'current': d.current_value,
                        'degradation_pct': d.degradation_pct
                    }
                    for d in degraded_metrics
                ]
            }
            
            if degraded_metrics:
                report['alerts'].append({
                    'type': 'performance_degradation',
                    'severity': 'critical',
                    'message': f"Performance degradation detected: {[d.metric_name for d in degraded_metrics]}"
                })
        
        # CUSUM changes
        for name, detector in self.cusum_detectors.items():
            status = detector.get_status()
            report['cusum_changes'][name] = {
                'change_detected': status['change_detected'],
                'n_changes': status['n_changes']
            }
            
            if status['change_detected']:
                report['alerts'].append({
                    'type': 'cusum_change',
                    'severity': 'high',
                    'message': f"CUSUM detected change in {name}"
                })
        
        # Determine if retraining needed
        report['requires_retraining'] = (
            len(report['alerts']) > 0 and 
            any(a['severity'] in ['high', 'critical'] for a in report['alerts'])
        )
        
        # Store alerts
        self.alerts.extend(report['alerts'])
        
        return report


# =============================================================================
# RETRAINING TRIGGER
# =============================================================================

class RetrainingTrigger:
    """
    Automated retraining trigger based on drift detection.
    """
    
    def __init__(self, drift_system: DriftAlertSystem,
                 cooldown_hours: int = 24,
                 min_samples: int = 1000):
        """
        Initialize retraining trigger.
        
        Args:
            drift_system: Drift alerting system
            cooldown_hours: Minimum hours between retraining
            min_samples: Minimum new samples before retraining
        """
        self.drift_system = drift_system
        self.cooldown_hours = cooldown_hours
        self.min_samples = min_samples
        
        self.last_retrain: Optional[datetime] = None
        self.samples_since_retrain = 0
        self.retrain_history: List[Dict] = []
    
    def log_sample(self, n: int = 1):
        """Log new samples."""
        self.samples_since_retrain += n
    
    def should_retrain(self, drift_report: Dict) -> Tuple[bool, str]:
        """
        Determine if retraining should be triggered.
        
        Returns (should_retrain, reason)
        """
        # Check cooldown
        if self.last_retrain is not None:
            hours_since = (datetime.now() - self.last_retrain).total_seconds() / 3600
            if hours_since < self.cooldown_hours:
                return False, f"In cooldown ({hours_since:.1f}h since last retrain)"
        
        # Check minimum samples
        if self.samples_since_retrain < self.min_samples:
            return False, f"Not enough samples ({self.samples_since_retrain}/{self.min_samples})"
        
        # Check drift report
        if not drift_report.get('requires_retraining', False):
            return False, "No significant drift detected"
        
        # Determine reason
        reasons = []
        
        if drift_report.get('feature_drift', {}).get('n_features_drifted', 0) > 0:
            reasons.append("feature drift")
        
        if drift_report.get('performance_degradation', {}).get('n_metrics_degraded', 0) > 0:
            reasons.append("performance degradation")
        
        if any(v.get('change_detected') for v in drift_report.get('cusum_changes', {}).values()):
            reasons.append("CUSUM change detection")
        
        return True, f"Triggered by: {', '.join(reasons)}"
    
    def trigger_retrain(self, reason: str):
        """Mark that retraining was triggered."""
        self.last_retrain = datetime.now()
        self.samples_since_retrain = 0
        
        self.retrain_history.append({
            'timestamp': self.last_retrain,
            'reason': reason
        })


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CONCEPT DRIFT DETECTION")
    print("="*70)
    
    np.random.seed(42)
    
    # 1. Create reference and current data with drift
    print("\n1. Simulating Data with Drift")
    print("-" * 50)
    
    # Reference data (training period)
    n_reference = 1000
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_reference),
        'feature_2': np.random.normal(5, 2, n_reference),
        'feature_3': np.random.exponential(1, n_reference),
        'feature_stable': np.random.uniform(0, 1, n_reference)
    })
    
    # Current data with drift in some features
    n_current = 200
    current_data = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, n_current),  # Mean and std shift
        'feature_2': np.random.normal(6, 2.5, n_current),    # Slight drift
        'feature_3': np.random.exponential(2, n_current),    # Distribution shift
        'feature_stable': np.random.uniform(0, 1, n_current) # No drift
    })
    
    print(f"   Reference data: {n_reference} samples")
    print(f"   Current data: {n_current} samples")
    
    # 2. Statistical Drift Tests
    print("\n2. Statistical Drift Tests")
    print("-" * 50)
    
    for col in reference_data.columns:
        ref = reference_data[col].values
        cur = current_data[col].values
        
        psi = StatisticalDriftTests.psi(ref, cur)
        ks_stat, ks_pvalue = StatisticalDriftTests.ks_test(ref, cur)
        
        drift_flag = "🔴 DRIFT" if psi > 0.25 or ks_pvalue < 0.05 else "🟢 OK"
        
        print(f"   {col}:")
        print(f"      PSI: {psi:.4f}, KS p-value: {ks_pvalue:.4f} {drift_flag}")
    
    # 3. Feature Drift Monitor
    print("\n3. Feature Drift Monitor")
    print("-" * 50)
    
    monitor = FeatureDriftMonitor(reference_data, psi_threshold=0.2)
    drift_results = monitor.check_drift(current_data)
    
    summary = monitor.get_drift_summary(drift_results)
    print(summary.to_string(index=False))
    
    # 4. Performance Monitor
    print("\n4. Performance Monitor")
    print("-" * 50)
    
    reference_metrics = {
        'hit_rate': 0.55,
        'sharpe_proxy': 1.8,
        'mse': 0.01
    }
    
    perf_monitor = PerformanceMonitor(reference_metrics, degradation_threshold=0.15)
    
    # Simulate predictions
    for i in range(100):
        # Good performance initially
        actual = np.random.choice([-1, 1])
        pred = actual * (1 if np.random.rand() < 0.55 else -1)
        perf_monitor.log_prediction(pred, actual)
    
    # Add some bad predictions to simulate degradation
    for i in range(50):
        actual = np.random.choice([-1, 1])
        pred = actual * (1 if np.random.rand() < 0.45 else -1)  # Worse hit rate
        perf_monitor.log_prediction(pred, actual)
    
    current_metrics = perf_monitor.calculate_metrics()
    degradations = perf_monitor.check_degradation()
    
    print(f"   Reference metrics: {reference_metrics}")
    print(f"   Current metrics: {current_metrics}")
    
    for d in degradations:
        status = "🔴 DEGRADED" if d.is_degraded else "🟢 OK"
        print(f"   {d.metric_name}: {d.current_value:.4f} (ref: {d.reference_value:.4f}) {status}")
    
    # 5. CUSUM Detector
    print("\n5. CUSUM Change Detection")
    print("-" * 50)
    
    cusum = CUSUMDetector(threshold=4.0, drift=0.5, warm_up=30)
    
    # Generate data with a mean shift
    data = np.concatenate([
        np.random.normal(0, 1, 50),   # Initial period
        np.random.normal(0, 1, 50),   # Stable
        np.random.normal(1, 1, 50)    # Mean shift at t=100
    ])
    
    changes = []
    for i, val in enumerate(data):
        if cusum.update(val):
            changes.append(i)
    
    print(f"   Total samples: {len(data)}")
    print(f"   True change point: ~100")
    print(f"   Detected change points: {changes}")
    
    status = cusum.get_status()
    print(f"   Detector status: {status['n_changes']} changes detected")
    
    # 6. Complete Alert System
    print("\n6. Drift Alert System")
    print("-" * 50)
    
    alert_system = DriftAlertSystem({
        'psi_threshold': 0.2,
        'ks_alpha': 0.05,
        'performance_degradation': 0.15,
        'cusum_threshold': 4.0
    })
    
    alert_system.initialize_feature_monitor(reference_data)
    alert_system.initialize_performance_monitor(reference_metrics)
    alert_system.add_cusum_monitor('returns')
    
    # Run checks
    report = alert_system.check_all(current_data)
    
    print(f"   Features checked: {report['feature_drift']['n_features_checked']}")
    print(f"   Features drifted: {report['feature_drift']['n_features_drifted']}")
    print(f"   Drifted features: {report['feature_drift']['drifted_features']}")
    print(f"   Requires retraining: {report['requires_retraining']}")
    
    if report['alerts']:
        print(f"\n   Alerts:")
        for alert in report['alerts']:
            print(f"      [{alert['severity'].upper()}] {alert['message']}")
    
    # 7. Retraining Trigger
    print("\n7. Retraining Trigger")
    print("-" * 50)
    
    trigger = RetrainingTrigger(alert_system, cooldown_hours=24, min_samples=100)
    
    # Log samples
    trigger.log_sample(150)
    
    should_retrain, reason = trigger.should_retrain(report)
    print(f"   Should retrain: {should_retrain}")
    print(f"   Reason: {reason}")
    
    if should_retrain:
        trigger.trigger_retrain(reason)
        print(f"   Retraining triggered at {trigger.last_retrain}")
    
    print("\n" + "="*70)
    print("INTEGRATION GUIDE")
    print("="*70)
    print("""
1. Setup in training pipeline:
   from drift_detection import DriftAlertSystem, RetrainingTrigger
   
   # After training
   alert_system = DriftAlertSystem()
   alert_system.initialize_feature_monitor(X_train)
   alert_system.initialize_performance_monitor({
       'sharpe': backtest_sharpe,
       'hit_rate': backtest_hit_rate
   })

2. Monitor in production:
   trigger = RetrainingTrigger(alert_system)
   
   for batch in production_data:
       trigger.log_sample(len(batch))
       
       # Check periodically
       report = alert_system.check_all(batch)
       
       should_retrain, reason = trigger.should_retrain(report)
       if should_retrain:
           retrain_model()
           trigger.trigger_retrain(reason)

3. Recommended thresholds:
   - PSI > 0.25: Significant feature drift
   - KS p-value < 0.05: Distribution change
   - Performance degradation > 20%: Model needs update
   - Check frequency: Every 1-4 hours
""")
