"""
B. Meta-Labeler Implementation
==============================
Build a secondary model that predicts the probability of the primary model's
prediction being correct. Accept trades only when meta_prob > threshold.

Key concept: The meta-labeler doesn't predict direction - it predicts 
whether the primary model's directional prediction will be profitable.

Reference: AFML Chapter 3 - The Meta-Labeling Method
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, classification_report
    )
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not installed")


# =============================================================================
# META-LABELING FRAMEWORK
# =============================================================================

@dataclass
class MetaLabelResult:
    """Results from meta-labeling analysis."""
    primary_accuracy: float
    meta_accuracy: float
    meta_precision: float
    meta_recall: float
    filtered_accuracy: float
    filtered_trades: int
    total_trades: int
    acceptance_rate: float
    lift: float  # Improvement from meta-labeling


class MetaLabeler:
    """
    Meta-Labeling system that learns when to trust the primary model.
    
    The meta-labeler is trained on:
    - Primary model's predictions (but NOT the actual direction)
    - Market conditions at time of prediction
    - Historical accuracy in similar conditions
    
    Output: Probability that primary model's prediction will be correct.
    """
    
    def __init__(self, 
                 primary_model,
                 meta_model_type: str = 'logistic',
                 threshold: float = 0.6,
                 calibrate: bool = True):
        """
        Initialize meta-labeler.
        
        Args:
            primary_model: Trained primary model with predict_proba method
            meta_model_type: 'logistic' or 'random_forest'
            threshold: Minimum meta-probability to accept trade
            calibrate: Whether to calibrate probabilities
        """
        self.primary_model = primary_model
        self.meta_model_type = meta_model_type
        self.threshold = threshold
        self.calibrate = calibrate
        
        # Initialize meta-model
        if meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        else:
            self.meta_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_meta_features(self, X: pd.DataFrame, y_primary_pred: np.ndarray,
                              y_primary_prob: np.ndarray) -> pd.DataFrame:
        """
        Create features for the meta-model.
        
        Does NOT include the primary prediction direction, only:
        - Confidence of primary prediction
        - Market conditions
        - Recent returns and volatility
        """
        meta_features = pd.DataFrame(index=X.index)
        
        # 1. Primary model confidence (max probability)
        meta_features['primary_confidence'] = np.max(y_primary_prob, axis=1)
        
        # 2. Prediction entropy (uncertainty)
        eps = 1e-10
        entropy = -np.sum(y_primary_prob * np.log(y_primary_prob + eps), axis=1)
        meta_features['prediction_entropy'] = entropy
        
        # 3. Probability margin (difference between top 2 classes)
        sorted_probs = np.sort(y_primary_prob, axis=1)
        meta_features['prob_margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # 4. Market features (if available in X)
        market_features = ['volatility', 'trend', 'volume', 'momentum', 
                          'rsi', 'macd', 'atr', 'bb_width']
        
        for feat in market_features:
            matching = [c for c in X.columns if feat.lower() in c.lower()]
            if matching:
                meta_features[f'market_{feat}'] = X[matching[0]].values
        
        # 5. Recent performance indicators (rolling window features)
        if 'returns' in X.columns or any('ret' in c.lower() for c in X.columns):
            ret_col = 'returns' if 'returns' in X.columns else \
                      [c for c in X.columns if 'ret' in c.lower()][0]
            
            # Recent return volatility
            if isinstance(X.index, pd.DatetimeIndex):
                meta_features['recent_vol'] = X[ret_col].rolling(10).std().fillna(0)
                meta_features['recent_trend'] = X[ret_col].rolling(10).mean().fillna(0)
        
        # 6. Absolute values of key features (regime indicators)
        for i, col in enumerate(X.columns[:5]):  # First 5 features
            meta_features[f'abs_feature_{i}'] = np.abs(X[col])
        
        return meta_features.fillna(0)
    
    def fit(self, X: pd.DataFrame, y_true: np.ndarray,
            y_primary_pred: np.ndarray, y_primary_prob: np.ndarray):
        """
        Train the meta-model.
        
        Args:
            X: Original features used by primary model
            y_true: Actual labels (1=profitable, 0=not profitable)
            y_primary_pred: Primary model's predictions
            y_primary_prob: Primary model's prediction probabilities
        """
        # Create meta-labels: Was the primary model correct?
        y_meta = (y_primary_pred == y_true).astype(int)
        
        # Create meta-features
        meta_features = self.create_meta_features(X, y_primary_pred, y_primary_prob)
        
        # Scale features
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        
        # Train meta-model
        self.meta_model.fit(meta_features_scaled, y_meta)
        self.is_fitted = True
        
        # Store feature names for importance analysis
        self.meta_feature_names = meta_features.columns.tolist()
        
        return self
    
    def predict_meta_prob(self, X: pd.DataFrame, 
                          y_primary_pred: np.ndarray,
                          y_primary_prob: np.ndarray) -> np.ndarray:
        """
        Predict probability that primary model is correct.
        
        Returns:
            Array of probabilities [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted. Call fit() first.")
        
        meta_features = self.create_meta_features(X, y_primary_pred, y_primary_prob)
        meta_features_scaled = self.scaler.transform(meta_features)
        
        return self.meta_model.predict_proba(meta_features_scaled)[:, 1]
    
    def filter_trades(self, X: pd.DataFrame,
                      y_primary_pred: np.ndarray,
                      y_primary_prob: np.ndarray,
                      threshold: Optional[float] = None) -> np.ndarray:
        """
        Filter predictions based on meta-probability.
        
        Args:
            X: Features
            y_primary_pred: Primary model predictions
            y_primary_prob: Primary model probabilities
            threshold: Meta-probability threshold (default: self.threshold)
        
        Returns:
            Boolean array where True = accept trade
        """
        threshold = threshold or self.threshold
        meta_probs = self.predict_meta_prob(X, y_primary_pred, y_primary_prob)
        return meta_probs >= threshold
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get meta-model feature importance."""
        if not self.is_fitted:
            return {}
        
        if hasattr(self.meta_model, 'feature_importances_'):
            importance = self.meta_model.feature_importances_
        elif hasattr(self.meta_model, 'coef_'):
            importance = np.abs(self.meta_model.coef_[0])
        else:
            return {}
        
        return dict(sorted(
            zip(self.meta_feature_names, importance),
            key=lambda x: -x[1]
        ))


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_meta_labeler(meta_labeler: MetaLabeler,
                          X_test: pd.DataFrame,
                          y_test: np.ndarray,
                          thresholds: List[float] = None) -> Dict:
    """
    Comprehensive evaluation of meta-labeling at various thresholds.
    
    Args:
        meta_labeler: Fitted MetaLabeler instance
        X_test: Test features
        y_test: True labels
        thresholds: List of thresholds to evaluate
    
    Returns:
        Dictionary with metrics at each threshold
    """
    if thresholds is None:
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    # Get primary predictions
    y_primary_pred = meta_labeler.primary_model.predict(X_test)
    y_primary_prob = meta_labeler.primary_model.predict_proba(X_test)
    
    # Primary model baseline metrics
    primary_accuracy = accuracy_score(y_test, y_primary_pred)
    
    results = {
        'primary': {
            'accuracy': primary_accuracy,
            'precision': precision_score(y_test, y_primary_pred, zero_division=0),
            'recall': recall_score(y_test, y_primary_pred, zero_division=0),
            'f1': f1_score(y_test, y_primary_pred, zero_division=0),
            'trades': len(y_test)
        },
        'thresholds': {}
    }
    
    # Meta-model predictions
    meta_probs = meta_labeler.predict_meta_prob(X_test, y_primary_pred, y_primary_prob)
    y_meta_true = (y_primary_pred == y_test).astype(int)
    
    # Meta-model intrinsic performance
    meta_pred = (meta_probs >= 0.5).astype(int)
    results['meta_model'] = {
        'accuracy': accuracy_score(y_meta_true, meta_pred),
        'roc_auc': roc_auc_score(y_meta_true, meta_probs),
        'precision': precision_score(y_meta_true, meta_pred, zero_division=0),
        'recall': recall_score(y_meta_true, meta_pred, zero_division=0)
    }
    
    # Evaluate at each threshold
    for threshold in thresholds:
        mask = meta_probs >= threshold
        n_filtered = mask.sum()
        
        if n_filtered > 0:
            filtered_accuracy = accuracy_score(y_test[mask], y_primary_pred[mask])
            filtered_precision = precision_score(y_test[mask], y_primary_pred[mask], zero_division=0)
            filtered_recall = precision_score(y_test[mask], y_primary_pred[mask], zero_division=0)
            lift = filtered_accuracy / primary_accuracy if primary_accuracy > 0 else 0
        else:
            filtered_accuracy = 0
            filtered_precision = 0
            filtered_recall = 0
            lift = 0
        
        results['thresholds'][threshold] = {
            'accuracy': filtered_accuracy,
            'precision': filtered_precision,
            'recall': filtered_recall,
            'trades': n_filtered,
            'acceptance_rate': n_filtered / len(y_test),
            'lift': lift
        }
    
    return results


def find_optimal_threshold(results: Dict, 
                           min_trades_pct: float = 0.2,
                           min_lift: float = 1.05) -> float:
    """
    Find optimal threshold balancing accuracy lift and trade frequency.
    
    Args:
        results: Output from evaluate_meta_labeler
        min_trades_pct: Minimum percentage of trades to accept
        min_lift: Minimum required accuracy lift
    
    Returns:
        Optimal threshold value
    """
    valid_thresholds = []
    
    for threshold, metrics in results['thresholds'].items():
        if metrics['acceptance_rate'] >= min_trades_pct and metrics['lift'] >= min_lift:
            # Score = lift * sqrt(acceptance_rate)
            score = metrics['lift'] * np.sqrt(metrics['acceptance_rate'])
            valid_thresholds.append((threshold, score, metrics))
    
    if not valid_thresholds:
        print("WARNING: No threshold meets criteria, using 0.5")
        return 0.5
    
    # Return threshold with highest score
    best = max(valid_thresholds, key=lambda x: x[1])
    return best[0]


# =============================================================================
# DEMO WITH VISUALIZATION
# =============================================================================

def create_sample_data_for_meta():
    """Create sample data that simulates a realistic trading scenario."""
    np.random.seed(42)
    n_samples = 3000
    
    # Create datetime index
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='H')
    
    # Market features
    X = pd.DataFrame(index=dates)
    
    # Price-based features
    X['returns'] = np.random.randn(n_samples) * 0.02
    X['momentum'] = np.cumsum(X['returns'].values * 0.1 + np.random.randn(n_samples) * 0.01)
    X['volatility'] = pd.Series(np.abs(X['returns'])).rolling(20).std().fillna(0.01)
    
    # Technical features
    X['rsi'] = 50 + np.cumsum(np.random.randn(n_samples) * 2)
    X['rsi'] = X['rsi'].clip(0, 100)
    X['macd'] = np.random.randn(n_samples) * 0.5
    X['volume'] = 1000 + np.abs(np.cumsum(np.random.randn(n_samples) * 50))
    
    # Additional features
    for i in range(20):
        X[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Generate true labels with regime-dependent signal
    regime = np.sin(np.arange(n_samples) / 200) > 0  # Alternating regimes
    
    # Signal strength varies by regime
    signal = np.where(
        regime,
        0.5 * X['momentum'] + 0.3 * X['macd'] + np.random.randn(n_samples) * 0.3,
        0.2 * X['momentum'] - 0.2 * X['macd'] + np.random.randn(n_samples) * 0.5
    )
    
    y = (signal > 0).astype(int)
    
    return X, y


def main():
    """Run meta-labeling demo."""
    print("="*70)
    print("META-LABELER DEMO")
    print("="*70)
    
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed")
        return
    
    # Create data
    print("\n1. Creating sample data...")
    X, y = create_sample_data_for_meta()
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data
    train_size = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\n2. Training primary model...")
    # Train primary model
    primary_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    primary_model.fit(X_train, y_train)
    
    # Primary model performance
    y_train_pred = primary_model.predict(X_train)
    y_train_prob = primary_model.predict_proba(X_train)
    y_test_pred = primary_model.predict(X_test)
    y_test_prob = primary_model.predict_proba(X_test)
    
    print(f"   Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"   Test Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    
    print(f"\n3. Training meta-labeler...")
    # Create and train meta-labeler
    meta_labeler = MetaLabeler(
        primary_model=primary_model,
        meta_model_type='logistic',
        threshold=0.6
    )
    
    meta_labeler.fit(X_train, y_train, y_train_pred, y_train_prob)
    
    print(f"\n4. Evaluating at different thresholds...")
    # Evaluate
    results = evaluate_meta_labeler(
        meta_labeler, X_test, y_test,
        thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    )
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nPrimary Model (no filtering):")
    print(f"  Accuracy:  {results['primary']['accuracy']:.4f}")
    print(f"  Precision: {results['primary']['precision']:.4f}")
    print(f"  F1 Score:  {results['primary']['f1']:.4f}")
    print(f"  Trades:    {results['primary']['trades']}")
    
    print(f"\nMeta-Model Intrinsic Performance:")
    print(f"  Accuracy:  {results['meta_model']['accuracy']:.4f}")
    print(f"  ROC AUC:   {results['meta_model']['roc_auc']:.4f}")
    print(f"  Precision: {results['meta_model']['precision']:.4f}")
    
    print(f"\nThreshold Analysis:")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Lift':>8} {'Trades':>8} {'Accept %':>10}")
    print("-" * 50)
    
    for threshold, metrics in results['thresholds'].items():
        print(f"{threshold:>10.2f} {metrics['accuracy']:>10.4f} "
              f"{metrics['lift']:>8.2f}x {metrics['trades']:>8} "
              f"{metrics['acceptance_rate']*100:>9.1f}%")
    
    # Find optimal threshold
    optimal = find_optimal_threshold(results, min_trades_pct=0.2, min_lift=1.02)
    print(f"\nOptimal Threshold: {optimal}")
    
    # Feature importance
    print(f"\nMeta-Feature Importance:")
    importance = meta_labeler.get_feature_importance()
    for i, (feat, imp) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {feat}: {imp:.4f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. Meta-labeling improves accuracy by filtering low-confidence trades
2. Higher thresholds give better accuracy but fewer trades
3. The 'lift' metric shows improvement over baseline
4. Trade-off: Better win rate vs fewer opportunities

PRODUCTION USAGE:
- Set threshold based on risk tolerance and capital constraints
- Monitor meta-model calibration over time
- Retrain periodically as market regimes change
- Combine with position sizing for risk management
""")


if __name__ == "__main__":
    main()
