"""
A. Optuna Hyperparameter Tuning Script
=======================================
With PurgedKFold support, walk-forward validation, and metrics logging.

Priority: HIGH (run after sanity checks pass)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna not installed. Run: pip install optuna")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# =============================================================================
# PURGED K-FOLD WITH EMBARGO
# =============================================================================

class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation with Embargo (AFML Ch. 7).
    
    Prevents information leakage by:
    1. Purging: Removes training samples that overlap with test period
    2. Embargo: Adds gap after test set before allowing training samples
    """
    
    def __init__(self, n_splits: int = 5, purge_pct: float = 0.01, 
                 embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging and embargo.
        
        Args:
            X: Features with datetime index
            y: Labels (optional)
            groups: Event end times for each sample (for purging overlap)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate purge and embargo sizes
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Split into folds
        fold_size = n_samples // self.n_splits
        
        splits = []
        for fold in range(self.n_splits):
            # Test indices for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]
            
            # Train indices: exclude test, purge zone, and embargo
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Remove test samples
            train_mask[test_start:test_end] = False
            
            # Remove purge zone before test (samples that might overlap)
            purge_start = max(0, test_start - purge_size)
            train_mask[purge_start:test_start] = False
            
            # Remove embargo zone after test
            embargo_end = min(n_samples, test_end + embargo_size)
            train_mask[test_end:embargo_end] = False
            
            train_idx = indices[train_mask]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation for time series.
    
    Simulates realistic training: train on past, test on future.
    """
    
    def __init__(self, n_splits: int = 5, train_pct: float = 0.6,
                 min_train_size: int = 100, gap: int = 0):
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.min_train_size = min_train_size
        self.gap = gap
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate window sizes
        total_test_size = int(n_samples * (1 - self.train_pct))
        test_size = total_test_size // self.n_splits
        
        splits = []
        for fold in range(self.n_splits):
            # Test window for this fold
            test_end = n_samples - (self.n_splits - fold - 1) * test_size
            test_start = test_end - test_size
            
            # Train on everything before test (minus gap)
            train_end = test_start - self.gap
            train_start = max(0, train_end - int(train_end * self.train_pct / (1 - self.train_pct)))
            train_start = max(train_start, 0)
            
            if train_end - train_start >= self.min_train_size and test_end > test_start:
                train_idx = indices[train_start:train_end]
                test_idx = indices[test_start:test_end]
                splits.append((train_idx, test_idx))
        
        return splits


# =============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =============================================================================

@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""
    best_params: Dict
    best_score: float
    all_trials: List[Dict]
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    study_stats: Dict


class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner with multiple model support.
    """
    
    def __init__(self, model_type: str = 'xgboost', cv_method: str = 'purged',
                 n_splits: int = 5, purge_pct: float = 0.01, embargo_pct: float = 0.01,
                 metric: str = 'f1', direction: str = 'maximize'):
        self.model_type = model_type
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.metric = metric
        self.direction = direction
        
        # Initialize CV
        if cv_method == 'purged':
            self.cv = PurgedKFoldCV(n_splits, purge_pct, embargo_pct)
        else:
            self.cv = WalkForwardCV(n_splits, gap=int(purge_pct * 1000))
    
    def _create_xgboost_objective(self, X: pd.DataFrame, y: pd.Series,
                                   sample_weights: Optional[pd.Series] = None):
        """Create objective function for XGBoost tuning."""
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
            }
            
            # Cross-validation
            scores = []
            splits = self.cv.split(X) if self.cv_method == 'walk_forward' else self.cv.split(X, y)
            
            for train_idx, test_idx in splits:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                weights = sample_weights.iloc[train_idx] if sample_weights is not None else None
                
                model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1,
                                          use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train, sample_weight=weights)
                
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                score = self._calculate_metric(y_test, y_pred, y_prob)
                scores.append(score)
            
            return np.mean(scores)
        
        return objective
    
    def _create_lightgbm_objective(self, X: pd.DataFrame, y: pd.Series,
                                    sample_weights: Optional[pd.Series] = None):
        """Create objective function for LightGBM tuning."""
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
            }
            
            scores = []
            splits = self.cv.split(X) if self.cv_method == 'walk_forward' else self.cv.split(X, y)
            
            for train_idx, test_idx in splits:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                weights = sample_weights.iloc[train_idx] if sample_weights is not None else None
                
                model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
                model.fit(X_train, y_train, sample_weight=weights)
                
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                score = self._calculate_metric(y_test, y_pred, y_prob)
                scores.append(score)
            
            return np.mean(scores)
        
        return objective
    
    def _create_rf_objective(self, X: pd.DataFrame, y: pd.Series,
                              sample_weights: Optional[pd.Series] = None):
        """Create objective function for Random Forest tuning."""
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            }
            
            scores = []
            splits = self.cv.split(X) if self.cv_method == 'walk_forward' else self.cv.split(X, y)
            
            for train_idx, test_idx in splits:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                weights = sample_weights.iloc[train_idx] if sample_weights is not None else None
                
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train, sample_weight=weights)
                
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                score = self._calculate_metric(y_test, y_pred, y_prob)
                scores.append(score)
            
            return np.mean(scores)
        
        return objective
    
    def _calculate_metric(self, y_true, y_pred, y_prob) -> float:
        """Calculate the optimization metric."""
        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'precision':
            return precision_score(y_true, y_pred, zero_division=0)
        elif self.metric == 'recall':
            return recall_score(y_true, y_pred, zero_division=0)
        elif self.metric == 'f1':
            return f1_score(y_true, y_pred, zero_division=0)
        elif self.metric == 'log_loss':
            return -log_loss(y_true, y_prob)  # Negative because we maximize
        else:
            return f1_score(y_true, y_pred, zero_division=0)
    
    def tune(self, X: pd.DataFrame, y: pd.Series, 
             sample_weights: Optional[pd.Series] = None,
             n_trials: int = 100, timeout: int = 3600,
             show_progress: bool = True) -> TuningResult:
        """
        Run hyperparameter tuning.
        
        Args:
            X: Features
            y: Labels
            sample_weights: Optional sample weights
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds
            show_progress: Whether to show progress bar
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed")
        
        # Create objective based on model type
        if self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed")
            objective = self._create_xgboost_objective(X, y, sample_weights)
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed")
            objective = self._create_lightgbm_objective(X, y, sample_weights)
        elif self.model_type == 'random_forest':
            objective = self._create_rf_objective(X, y, sample_weights)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name=f"{self.model_type}_tuning"
        )
        
        # Optuna verbosity
        optuna.logging.set_verbosity(
            optuna.logging.INFO if show_progress else optuna.logging.WARNING
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )
        
        # Get best model and feature importance
        best_params = study.best_params
        feature_importance = self._get_feature_importance(X, y, best_params, sample_weights)
        
        # Compile results
        all_trials = [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
        
        return TuningResult(
            best_params=best_params,
            best_score=study.best_value,
            all_trials=all_trials,
            cv_scores=[t.value for t in study.trials if t.value is not None],
            feature_importance=feature_importance,
            study_stats={
                'n_trials': len(study.trials),
                'n_complete': len([t for t in study.trials if t.state.name == 'COMPLETE']),
                'best_trial': study.best_trial.number,
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration)
            }
        )
    
    def _get_feature_importance(self, X: pd.DataFrame, y: pd.Series,
                                 params: Dict, sample_weights: Optional[pd.Series]) -> Dict[str, float]:
        """Train final model and get feature importance."""
        if self.model_type == 'xgboost':
            model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1,
                                      use_label_encoder=False, eval_metric='logloss')
        elif self.model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        else:
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        weights = sample_weights.values if sample_weights is not None else None
        model.fit(X, y, sample_weight=weights)
        
        importance = dict(zip(X.columns, model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: -x[1]))


def save_tuning_results(result: TuningResult, filepath: str):
    """Save tuning results to JSON."""
    output = {
        'best_params': result.best_params,
        'best_score': result.best_score,
        'study_stats': result.study_stats,
        'feature_importance': result.feature_importance,
        'all_trials': result.all_trials
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Results saved to {filepath}")


# =============================================================================
# DEMO
# =============================================================================

def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 2000
    n_features = 50
    
    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=pd.date_range('2022-01-01', periods=n_samples, freq='H')
    )
    
    # Generate labels with some signal
    signal = X['feature_0'] + 0.5 * X['feature_1'] - 0.3 * X['feature_2']
    y = pd.Series(
        (signal + np.random.randn(n_samples) * 0.5 > 0).astype(int),
        index=X.index
    )
    
    # Generate sample weights (higher for recent samples)
    weights = pd.Series(
        np.linspace(0.5, 1.5, n_samples),
        index=X.index
    )
    
    return X, y, weights


def main():
    """Run Optuna tuning demo."""
    print("="*70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("="*70)
    
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna not installed. Run: pip install optuna")
        return
    
    # Create sample data
    print("\n1. Creating sample data...")
    X, y, weights = create_sample_data()
    print(f"   Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"   Label distribution: {y.value_counts().to_dict()}")
    
    # Choose model type based on availability
    if XGBOOST_AVAILABLE:
        model_type = 'xgboost'
    elif LIGHTGBM_AVAILABLE:
        model_type = 'lightgbm'
    else:
        model_type = 'random_forest'
    
    print(f"\n2. Tuning {model_type.upper()} with Purged K-Fold CV...")
    
    # Create tuner
    tuner = HyperparameterTuner(
        model_type=model_type,
        cv_method='purged',
        n_splits=5,
        purge_pct=0.01,
        embargo_pct=0.01,
        metric='f1'
    )
    
    # Run tuning (reduced trials for demo)
    result = tuner.tune(
        X, y,
        sample_weights=weights,
        n_trials=20,  # Increase to 100+ for real use
        timeout=300,
        show_progress=True
    )
    
    # Print results
    print("\n" + "="*70)
    print("TUNING RESULTS")
    print("="*70)
    print(f"\nBest Score: {result.best_score:.4f}")
    print(f"\nBest Parameters:")
    for param, value in result.best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nStudy Statistics:")
    for stat, value in result.study_stats.items():
        print(f"  {stat}: {value}")
    
    print(f"\nTop 10 Features:")
    for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10]):
        print(f"  {i+1}. {feat}: {imp:.4f}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "tuning_results.json"
    output_path.parent.mkdir(exist_ok=True)
    save_tuning_results(result, str(output_path))
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Increase n_trials to 100+ for production tuning
2. Use walk-forward CV for time series validation
3. Add custom metrics (Sharpe, net P&L) as objectives
4. Combine with meta-labeling for trade selection
5. Run sanity checks on final model
""")


if __name__ == "__main__":
    main()
