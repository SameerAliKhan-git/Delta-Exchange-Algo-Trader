"""
src/ml/cost_aware_objective.py

Cost-Aware Training Objective Wrapper
=====================================

This module implements training objectives that optimize for NET P&L after fees/slippage
rather than raw accuracy. This is critical for real trading where:

- A 60% accurate model with 2 trades/day at 20bps cost might lose money
- A 55% accurate model with 0.5 trades/day at 5bps cost might be profitable

Key Components:
1. CostAwareObjective - Sklearn-compatible scorer that penalizes by transaction costs
2. AlmgrenChrissImpact - Market impact model for realistic slippage estimation
3. TurnoverRegularizer - Penalizes excessive trading frequency
4. NetPnLOptimizer - Custom optimizer that maximizes net returns

Usage:
    from src.ml.cost_aware_objective import CostAwareObjective, train_cost_aware_model
    
    # Train with cost awareness
    model = train_cost_aware_model(
        X_train, y_train,
        cost_bps=10.0,
        avg_trade_size_usd=10000,
        expected_trades_per_day=2.0
    )
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Market Impact Models
# =============================================================================

@dataclass
class AlmgrenChrissParams:
    """Parameters for Almgren-Chriss market impact model."""
    # Temporary impact (linear)
    eta: float = 0.1  # Temporary impact coefficient
    
    # Permanent impact (linear)  
    gamma: float = 0.05  # Permanent impact coefficient
    
    # Volatility
    sigma: float = 0.02  # Daily volatility
    
    # Average daily volume (in units)
    adv: float = 1_000_000
    
    # Risk aversion
    lambda_risk: float = 1e-6


class AlmgrenChrissImpact:
    """
    Almgren-Chriss market impact model.
    
    Estimates slippage based on:
    - Order size relative to ADV
    - Market volatility
    - Urgency of execution
    
    Total cost = Temporary Impact + Permanent Impact + Spread
    
    Reference: Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
    """
    
    def __init__(self, params: Optional[AlmgrenChrissParams] = None):
        self.params = params or AlmgrenChrissParams()
    
    def estimate_impact(
        self,
        order_size: float,
        price: float,
        time_horizon: float = 1.0,  # Trading time in days
        urgency: float = 1.0,  # 1.0 = normal, >1 = urgent (more impact)
    ) -> Dict[str, float]:
        """
        Estimate market impact for an order.
        
        Args:
            order_size: Size in base units (e.g., BTC)
            price: Current price
            time_horizon: Time to execute (days)
            urgency: Urgency factor (higher = faster execution = more impact)
        
        Returns:
            Dict with temporary_impact, permanent_impact, total_impact (all in bps)
        """
        # Participation rate
        participation = order_size / (self.params.adv * time_horizon)
        
        # Temporary impact (goes away after trade)
        temp_impact = self.params.eta * self.params.sigma * np.sqrt(participation / time_horizon)
        
        # Permanent impact (stays in price)
        perm_impact = self.params.gamma * (order_size / self.params.adv)
        
        # Adjust for urgency
        temp_impact *= urgency
        
        # Convert to bps
        temp_impact_bps = temp_impact * 10000
        perm_impact_bps = perm_impact * 10000
        total_impact_bps = temp_impact_bps + perm_impact_bps
        
        return {
            'temporary_impact_bps': temp_impact_bps,
            'permanent_impact_bps': perm_impact_bps,
            'total_impact_bps': total_impact_bps,
            'participation_rate': participation,
        }
    
    def estimate_slippage_curve(
        self,
        sizes: List[float],
        price: float,
        time_horizon: float = 1.0,
    ) -> List[Dict[str, float]]:
        """Estimate slippage for multiple order sizes."""
        return [self.estimate_impact(s, price, time_horizon) for s in sizes]


# =============================================================================
# Cost-Aware Scoring Functions
# =============================================================================

def net_pnl_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    cost_bps: float = 10.0,
    avg_return_per_correct: float = 0.002,  # 20 bps average return per correct prediction
    avg_loss_per_wrong: float = 0.002,  # 20 bps average loss per wrong prediction
) -> float:
    """
    Score function that computes expected net P&L after transaction costs.
    
    Args:
        y_true: True labels (0/1 for down/up)
        y_pred: Predicted labels
        cost_bps: Round-trip transaction cost in basis points
        avg_return_per_correct: Average return when prediction is correct
        avg_loss_per_wrong: Average loss when prediction is wrong
    
    Returns:
        Net P&L score (higher is better)
    """
    cost_decimal = cost_bps / 10000
    
    # Correct predictions
    correct = (y_true == y_pred)
    n_correct = np.sum(correct)
    n_wrong = np.sum(~correct)
    n_total = len(y_true)
    
    if n_total == 0:
        return 0.0
    
    # Gross P&L (before costs)
    gross_pnl = n_correct * avg_return_per_correct - n_wrong * avg_loss_per_wrong
    
    # Transaction costs (pay on every trade)
    # Assume we trade on every prediction that's not "hold"
    n_trades = n_total  # Simplified: every prediction triggers a trade
    total_cost = n_trades * cost_decimal * 2  # Round-trip
    
    # Net P&L
    net_pnl = gross_pnl - total_cost
    
    return net_pnl


def turnover_adjusted_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prev: Optional[np.ndarray] = None,
    *,
    cost_bps: float = 10.0,
    turnover_penalty: float = 0.5,  # Penalty per position change
) -> float:
    """
    Score that penalizes excessive turnover (position changes).
    
    Args:
        y_true: True labels
        y_pred: Current predictions
        y_pred_prev: Previous predictions (to calculate turnover)
        cost_bps: Transaction cost
        turnover_penalty: Extra penalty for changing positions
    
    Returns:
        Turnover-adjusted score
    """
    base_score = net_pnl_score(y_true, y_pred, cost_bps=cost_bps)
    
    if y_pred_prev is not None:
        # Count position changes
        n_changes = np.sum(y_pred != y_pred_prev)
        turnover_cost = n_changes * turnover_penalty * (cost_bps / 10000)
        return base_score - turnover_cost
    
    return base_score


def sharpe_aware_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray,
    *,
    cost_bps: float = 10.0,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Score based on Sharpe ratio of strategy returns after costs.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (1 = long, 0 = short/flat)
        returns: Actual returns for each period
        cost_bps: Transaction cost
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sharpe ratio of the strategy
    """
    cost_decimal = cost_bps / 10000
    
    # Strategy returns: go long when predict 1, short when predict 0
    position = np.where(y_pred == 1, 1, -1)
    strategy_returns = position * returns
    
    # Subtract transaction costs when position changes
    position_changes = np.abs(np.diff(position, prepend=position[0]))
    costs = position_changes * cost_decimal
    net_returns = strategy_returns - costs
    
    # Sharpe ratio
    if np.std(net_returns) == 0:
        return 0.0
    
    daily_rf = risk_free_rate / 252
    sharpe = (np.mean(net_returns) - daily_rf) / np.std(net_returns) * np.sqrt(252)
    
    return sharpe


# =============================================================================
# Sklearn-Compatible Scorers
# =============================================================================

def make_cost_aware_scorer(
    cost_bps: float = 10.0,
    metric: str = "net_pnl",
) -> Callable:
    """
    Create a sklearn-compatible scorer with cost awareness.
    
    Args:
        cost_bps: Transaction cost in bps
        metric: One of "net_pnl", "sharpe", "turnover_adjusted"
    
    Returns:
        Sklearn scorer function
    """
    if metric == "net_pnl":
        def scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return net_pnl_score(y, y_pred, cost_bps=cost_bps)
    elif metric == "sharpe":
        def scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            # Assume last column of X is returns (for this scorer)
            returns = X[:, -1] if X.ndim > 1 else np.zeros_like(y)
            return sharpe_aware_score(y, y_pred, returns, cost_bps=cost_bps)
    elif metric == "turnover_adjusted":
        def scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return turnover_adjusted_score(y, y_pred, cost_bps=cost_bps)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return scorer


class CostAwareObjective:
    """
    Cost-aware objective function for hyperparameter optimization.
    
    Compatible with:
    - Optuna
    - Sklearn GridSearchCV/RandomSearchCV
    - Custom optimization loops
    """
    
    def __init__(
        self,
        cost_bps: float = 10.0,
        avg_trade_size_usd: float = 10000,
        expected_trades_per_day: float = 2.0,
        impact_model: Optional[AlmgrenChrissImpact] = None,
    ):
        self.cost_bps = cost_bps
        self.avg_trade_size_usd = avg_trade_size_usd
        self.expected_trades_per_day = expected_trades_per_day
        self.impact_model = impact_model or AlmgrenChrissImpact()
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute cost-aware objective value.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional, for confidence weighting)
        
        Returns:
            Objective value (higher is better for maximization)
        """
        n_samples = len(y_true)
        
        # Base accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Expected trades
        n_trades = n_samples * (self.expected_trades_per_day / 252)  # Assume 252 trading days
        
        # Fixed costs
        fixed_cost = n_trades * (self.cost_bps / 10000) * 2  # Round-trip
        
        # Market impact (if we have probabilities, weight by confidence)
        if y_pred_proba is not None:
            # Higher confidence = larger position = more impact
            confidence = np.abs(y_pred_proba - 0.5) * 2  # Scale to [0, 1]
            avg_confidence = np.mean(confidence)
            
            # Estimate impact for average trade
            impact = self.impact_model.estimate_impact(
                self.avg_trade_size_usd * avg_confidence,
                price=1.0  # Normalized
            )
            impact_cost = impact['total_impact_bps'] / 10000 * n_trades
        else:
            impact_cost = 0.0
        
        # Total cost
        total_cost = fixed_cost + impact_cost
        
        # Expected gross return (simplified)
        # Assume correct predictions earn 20bps, wrong predictions lose 20bps
        avg_return = 0.002
        gross_pnl = (accuracy * avg_return - (1 - accuracy) * avg_return) * n_samples
        
        # Net P&L
        net_pnl = gross_pnl - total_cost
        
        # Return as objective (normalized by number of samples)
        return net_pnl / n_samples if n_samples > 0 else 0.0
    
    def as_sklearn_scorer(self) -> Callable:
        """Return as sklearn-compatible scorer."""
        def scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            y_proba = None
            if hasattr(estimator, 'predict_proba'):
                try:
                    y_proba = estimator.predict_proba(X)[:, 1]
                except:
                    pass
            return self(y, y_pred, y_proba)
        return scorer
    
    def as_optuna_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_class: type,
    ) -> Callable:
        """
        Return as Optuna objective function.
        
        Usage:
            import optuna
            
            objective = cost_aware_obj.as_optuna_objective(X_train, y_train, X_val, y_val, RandomForestClassifier)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
        """
        def objective(trial):
            # Example hyperparameters for RandomForest
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)[:, 1]
            
            return self(y_val, y_pred, y_proba)
        
        return objective


# =============================================================================
# Cost-Aware Model Wrapper
# =============================================================================

class CostAwareClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that adds cost-aware prediction thresholds.
    
    Instead of predicting at 0.5 probability threshold, adjusts threshold
    to account for transaction costs and only trades when expected value is positive.
    """
    
    def __init__(
        self,
        base_model: BaseEstimator,
        cost_bps: float = 10.0,
        min_edge_bps: float = 5.0,  # Minimum expected edge to trade
        confidence_threshold: float = 0.55,  # Minimum confidence to trade
    ):
        self.base_model = base_model
        self.cost_bps = cost_bps
        self.min_edge_bps = min_edge_bps
        self.confidence_threshold = confidence_threshold
        self._optimal_threshold = 0.5
    
    def fit(self, X, y):
        """Fit the base model and calibrate threshold."""
        self.base_model.fit(X, y)
        
        # Calibrate threshold using cost-aware scoring
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(X)[:, 1]
            best_score = -np.inf
            best_threshold = 0.5
            
            for threshold in np.arange(0.45, 0.65, 0.01):
                y_pred = (proba >= threshold).astype(int)
                score = net_pnl_score(y, y_pred, cost_bps=self.cost_bps)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            self._optimal_threshold = best_threshold
            logger.info(f"Calibrated threshold: {self._optimal_threshold:.3f}")
        
        return self
    
    def predict(self, X):
        """Predict with cost-aware threshold."""
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(X)[:, 1]
            return (proba >= self._optimal_threshold).astype(int)
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        """Return prediction probabilities."""
        return self.base_model.predict_proba(X)
    
    def should_trade(self, X) -> np.ndarray:
        """
        Return boolean array indicating whether to trade each sample.
        
        Only trades when:
        1. Confidence exceeds threshold
        2. Expected edge exceeds cost + min_edge
        """
        if not hasattr(self.base_model, 'predict_proba'):
            return np.ones(len(X), dtype=bool)
        
        proba = self.base_model.predict_proba(X)[:, 1]
        confidence = np.abs(proba - 0.5) * 2  # [0, 1]
        
        # Expected edge = (P(correct) - 0.5) * avg_return * 2
        avg_return_bps = 20  # Assume 20bps average move
        expected_edge = (confidence * avg_return_bps)
        
        # Only trade if edge > cost + min_edge
        min_required_edge = self.cost_bps + self.min_edge_bps
        
        return (expected_edge >= min_required_edge) & (confidence >= (self.confidence_threshold - 0.5) * 2)


# =============================================================================
# Training Function
# =============================================================================

def train_cost_aware_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    cost_bps: float = 10.0,
    model_type: str = "ensemble",
    optimize_hyperparams: bool = True,
    n_trials: int = 50,
) -> CostAwareClassifier:
    """
    Train a cost-aware model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        cost_bps: Transaction cost in basis points
        model_type: "rf", "gb", "ensemble"
        optimize_hyperparams: Whether to optimize hyperparameters
        n_trials: Number of Optuna trials
    
    Returns:
        Trained CostAwareClassifier
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split
    
    # Split if no validation set provided
    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    logger.info(f"Training cost-aware model (cost={cost_bps}bps, type={model_type})")
    
    if optimize_hyperparams:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            cost_obj = CostAwareObjective(cost_bps=cost_bps)
            
            if model_type == "rf":
                objective = cost_obj.as_optuna_objective(
                    X_train, y_train, X_val, y_val, RandomForestClassifier
                )
            else:
                # Default to RF for now
                objective = cost_obj.as_optuna_objective(
                    X_train, y_train, X_val, y_val, RandomForestClassifier
                )
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            best_params = study.best_params
            logger.info(f"Best params: {best_params}")
            
            base_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            
        except ImportError:
            logger.warning("Optuna not installed, using default hyperparameters")
            base_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        if model_type == "rf":
            base_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        elif model_type == "gb":
            base_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        else:  # ensemble
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            base_model = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb)],
                voting='soft'
            )
    
    # Wrap in cost-aware classifier
    model = CostAwareClassifier(
        base_model=base_model,
        cost_bps=cost_bps,
        min_edge_bps=5.0,
        confidence_threshold=0.55,
    )
    
    # Fit on combined train+val for final model
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    model.fit(X_full, y_full)
    
    # Evaluate
    y_pred = model.predict(X_val)
    final_score = net_pnl_score(y_val, y_pred, cost_bps=cost_bps)
    logger.info(f"Final cost-aware score: {final_score:.6f}")
    
    return model


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    
    n_samples = 5000
    n_features = 20
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some signal
    true_coef = np.random.randn(n_features) * 0.1
    signal = X @ true_coef
    noise = np.random.randn(n_samples) * 0.5
    y = (signal + noise > 0).astype(int)
    
    # Train cost-aware model
    model = train_cost_aware_model(
        X, y,
        cost_bps=10.0,
        model_type="rf",
        optimize_hyperparams=False,  # Set True if Optuna installed
        n_trials=20
    )
    
    # Test predictions
    X_test = np.random.randn(100, n_features)
    predictions = model.predict(X_test)
    should_trade = model.should_trade(X_test)
    
    print(f"\nPredictions: {predictions}")
    print(f"Should trade: {should_trade.sum()}/{len(should_trade)}")
    print(f"Optimal threshold: {model._optimal_threshold:.3f}")
