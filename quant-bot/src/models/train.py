"""
Model Training Module

Implements ML training harness with:
- Purged K-Fold Cross-Validation
- Hyperparameter optimization
- Model selection (XGBoost, LightGBM, Random Forest)
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from loguru import logger

try:
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, log_loss, confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available")


# ==============================================================================
# PURGED K-FOLD CROSS-VALIDATION (AFML Ch. 7)
# ==============================================================================

class PurgedKFold:
    """
    Purged K-Fold Cross-Validation (AFML Ch. 7).
    
    Prevents information leakage between train/test sets by:
    1. Purging: Remove training samples that overlap with test samples
    2. Embargo: Add a gap after test samples before training resumes
    
    This is critical for financial time series where labels may span
    multiple time periods.
    
    Example:
        cv = PurgedKFold(n_splits=5, purge_gap=10, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X, labels, exit_times):
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[test_idx], y[test_idx])
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.0
    ):
        """
        Initialize purged K-fold.
        
        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to purge around test set
            embargo_pct: Percentage of samples to embargo after test set
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        exit_times: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate train/test indices.
        
        Args:
            X: Feature DataFrame
            y: Labels (optional)
            exit_times: Exit time indices for purging (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        indices = np.arange(len(X))
        fold_size = len(X) // self.n_splits
        embargo_size = int(len(X) * self.embargo_pct)
        
        for i in range(self.n_splits):
            # Test set indices
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else len(X)
            test_idx = indices[test_start:test_end]
            
            # Training set (excluding test, purge, and embargo)
            train_mask = np.ones(len(X), dtype=bool)
            
            # Remove test indices
            train_mask[test_start:test_end] = False
            
            # Apply purge gap before test set
            purge_start = max(0, test_start - self.purge_gap)
            train_mask[purge_start:test_start] = False
            
            # Apply embargo after test set
            embargo_end = min(len(X), test_end + embargo_size)
            train_mask[test_end:embargo_end] = False
            
            # Additional purging based on exit times
            if exit_times is not None:
                for idx in test_idx:
                    if idx in exit_times.index:
                        exit_idx = int(exit_times[idx])
                        # Remove training samples that overlap with this test sample
                        train_mask[max(0, idx - self.purge_gap):min(len(X), exit_idx + 1)] = False
            
            train_idx = indices[train_mask]
            
            logger.debug(f"Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}")
            
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (AFML Ch. 12).
    
    Generates all combinations of N groups into training and testing,
    allowing for more paths through the data and better backtesting.
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.0
    ):
        """
        Initialize combinatorial CV.
        
        Args:
            n_splits: Number of groups to divide data into
            n_test_splits: Number of groups to use for testing in each combination
            purge_gap: Samples to purge around test boundaries
            embargo_pct: Embargo percentage
        """
        from itertools import combinations
        
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        
        # Generate all combinations
        self.combinations = list(combinations(range(n_splits), n_test_splits))
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        exit_times: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate train/test indices for each combination."""
        indices = np.arange(len(X))
        group_size = len(X) // self.n_splits
        embargo_size = int(len(X) * self.embargo_pct)
        
        for combo in self.combinations:
            # Determine test groups
            test_mask = np.zeros(len(X), dtype=bool)
            for g in combo:
                start = g * group_size
                end = (g + 1) * group_size if g < self.n_splits - 1 else len(X)
                test_mask[start:end] = True
            
            test_idx = indices[test_mask]
            
            # Training mask with purge and embargo
            train_mask = ~test_mask.copy()
            
            # Apply purge around test boundaries
            test_changes = np.diff(test_mask.astype(int))
            boundaries = np.where(test_changes != 0)[0]
            
            for boundary in boundaries:
                purge_start = max(0, boundary - self.purge_gap)
                purge_end = min(len(X), boundary + self.purge_gap + embargo_size)
                train_mask[purge_start:purge_end] = False
            
            train_idx = indices[train_mask]
            
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        """Return number of combinations."""
        return len(self.combinations)


# ==============================================================================
# MODEL WRAPPER
# ==============================================================================

class BaseModel(ABC):
    """Abstract base class for models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseModel":
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass


class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        self.params = params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42
        }
        self.model = None
        self.feature_names: List[str] = []
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "XGBoostModel":
        """Train XGBoost model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight, **kwargs)
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate probability predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return dict(enumerate(importance))


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        self.params = params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "verbose": -1
        }
        self.model = None
        self.feature_names: List[str] = []
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "LightGBMModel":
        """Train LightGBM model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight, **kwargs)
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate probability predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return dict(enumerate(importance))


class RandomForestModel(BaseModel):
    """Random Forest model wrapper."""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize Random Forest model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")
        
        self.params = params or {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1
        }
        self.model = None
        self.feature_names: List[str] = []
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "RandomForestModel":
        """Train Random Forest model."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate probability predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return dict(enumerate(importance))


# ==============================================================================
# MODEL TRAINER
# ==============================================================================

@dataclass
class TrainingResults:
    """Container for training results."""
    model: Any
    cv_scores: List[float]
    mean_score: float
    std_score: float
    feature_importance: Dict[str, float]
    best_params: Dict
    confusion_matrices: List[np.ndarray] = None
    
    def summary(self) -> str:
        """Return summary string."""
        return (
            f"CV Score: {self.mean_score:.4f} (+/- {self.std_score:.4f})\n"
            f"Best Params: {self.best_params}\n"
            f"Top Features: {sorted(self.feature_importance.items(), key=lambda x: -x[1])[:5]}"
        )


class ModelTrainer:
    """
    Model training harness with cross-validation and hyperparameter optimization.
    
    Example:
        trainer = ModelTrainer(model_type='xgboost')
        results = trainer.train(X, y, sample_weights=weights, optimize=True)
    """
    
    MODEL_CLASSES = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'random_forest': RandomForestModel
    }
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        params: Optional[Dict] = None,
        cv_type: str = 'purged',
        n_splits: int = 5,
        purge_gap: int = 10,
        embargo_pct: float = 0.01
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest')
            params: Model parameters
            cv_type: Cross-validation type ('purged', 'combinatorial')
            n_splits: Number of CV splits
            purge_gap: Samples to purge
            embargo_pct: Embargo percentage
        """
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        self.params = params
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        
        # Results
        self.trained_model: Optional[BaseModel] = None
        self.results: Optional[TrainingResults] = None
    
    def _get_cv(self) -> Union[PurgedKFold, CombinatorialPurgedCV]:
        """Get cross-validator."""
        if self.cv_type == 'combinatorial':
            return CombinatorialPurgedCV(
                n_splits=self.n_splits,
                n_test_splits=2,
                purge_gap=self.purge_gap,
                embargo_pct=self.embargo_pct
            )
        return PurgedKFold(
            n_splits=self.n_splits,
            purge_gap=self.purge_gap,
            embargo_pct=self.embargo_pct
        )
    
    def _create_model(self, params: Optional[Dict] = None) -> BaseModel:
        """Create model instance."""
        model_class = self.MODEL_CLASSES[self.model_type]
        return model_class(params or self.params)
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        exit_times: Optional[pd.Series] = None
    ) -> Tuple[List[float], List[np.ndarray]]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            sample_weights: Sample weights
            exit_times: Exit times for purging
            
        Returns:
            Tuple of (scores, confusion_matrices)
        """
        cv = self._get_cv()
        scores = []
        cms = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, exit_times)):
            # Get train/test data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Handle sample weights
            weights = None
            if sample_weights is not None:
                weights = sample_weights.iloc[train_idx].values
            
            # Train model
            model = self._create_model()
            model.fit(X_train, y_train.values, sample_weight=weights)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Compute AUC if binary classification
            if len(np.unique(y_test)) == 2:
                score = roc_auc_score(y_test, y_proba[:, 1])
            else:
                score = accuracy_score(y_test, y_pred)
            
            scores.append(score)
            cms.append(confusion_matrix(y_test, y_pred))
            
            logger.debug(f"Fold {fold + 1}: score={score:.4f}")
        
        return scores, cms
    
    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        sample_weights: Optional[pd.Series] = None,
        exit_times: Optional[pd.Series] = None
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Features
            y: Labels
            n_trials: Number of optimization trials
            sample_weights: Sample weights
            exit_times: Exit times
            
        Returns:
            Best parameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using default parameters.")
            return self.params or {}
        
        def objective(trial):
            # Define search space based on model type
            if self.model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'use_label_encoder': False,
                    'random_state': 42
                }
            elif self.model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'objective': 'binary',
                    'metric': 'auc',
                    'random_state': 42,
                    'verbose': -1
                }
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42,
                    'n_jobs': -1
                }
            
            self.params = params
            scores, _ = self.cross_validate(X, y, sample_weights, exit_times)
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        return study.best_trial.params
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        exit_times: Optional[pd.Series] = None,
        optimize: bool = False,
        n_trials: int = 50
    ) -> TrainingResults:
        """
        Train model with cross-validation.
        
        Args:
            X: Features DataFrame
            y: Labels Series
            sample_weights: Sample weights
            exit_times: Exit times for purging
            optimize: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            TrainingResults object
        """
        logger.info(f"Training {self.model_type} model on {len(X)} samples, {X.shape[1]} features")
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if sample_weights is not None:
            sample_weights = sample_weights.loc[common_idx]
        if exit_times is not None:
            exit_times = exit_times.loc[common_idx.intersection(exit_times.index)]
        
        # Remove NaN rows
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if sample_weights is not None:
            sample_weights = sample_weights[valid_mask]
        
        logger.info(f"After cleaning: {len(X)} samples")
        
        # Optimize hyperparameters if requested
        best_params = self.params or {}
        if optimize:
            best_params = self.optimize_hyperparameters(
                X, y, n_trials, sample_weights, exit_times
            )
            self.params = best_params
        
        # Cross-validate
        scores, cms = self.cross_validate(X, y, sample_weights, exit_times)
        
        # Train final model on all data
        weights = sample_weights.values if sample_weights is not None else None
        self.trained_model = self._create_model(self.params)
        self.trained_model.fit(X, y.values, sample_weight=weights)
        
        # Get feature importance
        feature_importance = self.trained_model.get_feature_importance()
        
        self.results = TrainingResults(
            model=self.trained_model,
            cv_scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            feature_importance=feature_importance,
            best_params=best_params,
            confusion_matrices=cms
        )
        
        logger.info(f"Training complete: {self.results.summary()}")
        
        return self.results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.trained_model is None:
            raise ValueError("Model not trained")
        return self.trained_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        if self.trained_model is None:
            raise ValueError("Model not trained")
        return self.trained_model.predict_proba(X)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("MODEL TRAINING DEMO")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create labels with some signal
    signal = X['feature_0'] + X['feature_1'] * 0.5 + np.random.randn(n) * 0.5
    y = pd.Series((signal > 0).astype(int), name='label')
    
    # Train model
    trainer = ModelTrainer(model_type='xgboost', n_splits=5)
    results = trainer.train(X, y, optimize=False)
    
    print(f"\n{results.summary()}")
    
    # Make predictions
    predictions = trainer.predict(X[:10])
    print(f"\nSample predictions: {predictions}")
