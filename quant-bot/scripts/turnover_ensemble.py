"""
C. Turnover-Aware Training + Ensemble
=====================================
Train models that penalize excessive turnover (frequent position changes).
Combine multiple strategies in an ensemble with turnover regularization.

Key Concepts:
1. Penalize models that flip positions too frequently
2. Use transaction costs as implicit regularization
3. Build ensemble that diversifies signal sources
4. Weight ensemble members by risk-adjusted returns net of costs

Reference: AFML Ch. 8 - Feature Importance, Ch. 10 - Bet Sizing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# TURNOVER-AWARE LOSS AND TRAINING
# =============================================================================

class TurnoverAwareModel:
    """
    Model wrapper that incorporates turnover penalty into training.
    
    Turnover penalty approaches:
    1. Sample weighting: Down-weight samples following position changes
    2. Custom loss: Add penalty term for prediction changes
    3. Post-hoc smoothing: Apply hysteresis to predictions
    """
    
    def __init__(self, 
                 base_model,
                 turnover_penalty: float = 0.5,
                 method: str = 'sample_weighting',
                 hysteresis_threshold: float = 0.1,
                 transaction_cost_bps: float = 10):
        """
        Initialize turnover-aware model.
        
        Args:
            base_model: Base sklearn-compatible model
            turnover_penalty: Penalty multiplier for turnover (0-1)
            method: 'sample_weighting', 'hysteresis', or 'both'
            hysteresis_threshold: Threshold for prediction changes
            transaction_cost_bps: Transaction cost in basis points
        """
        self.base_model = base_model
        self.turnover_penalty = turnover_penalty
        self.method = method
        self.hysteresis_threshold = hysteresis_threshold
        self.transaction_cost_bps = transaction_cost_bps
        
        self.last_prediction = None
        self.is_fitted = False
    
    def _compute_turnover_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights that penalize position changes.
        
        Samples immediately after a position change get lower weights,
        encouraging the model to learn stable patterns.
        """
        n = len(y)
        weights = np.ones(n)
        
        # Identify position changes
        changes = np.diff(y.astype(int), prepend=y[0])
        change_indices = np.where(changes != 0)[0]
        
        # Apply exponential decay penalty after each change
        for idx in change_indices:
            # Decay window: 5 samples after change
            for offset in range(5):
                if idx + offset < n:
                    # Weight decreases closer to the change
                    decay = np.exp(-0.5 * (offset))
                    weights[idx + offset] *= (1 - self.turnover_penalty * decay)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None):
        """
        Train model with turnover-aware sample weighting.
        """
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Compute turnover weights
        turnover_weights = self._compute_turnover_weights(y_array)
        
        # Combine with provided sample weights
        if sample_weight is not None:
            combined_weights = sample_weight * turnover_weights
        else:
            combined_weights = turnover_weights
        
        # Fit base model
        self.base_model.fit(X, y_array, sample_weight=combined_weights)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get raw prediction probabilities."""
        return self.base_model.predict_proba(X)
    
    def predict(self, X: pd.DataFrame, 
                apply_hysteresis: bool = True) -> np.ndarray:
        """
        Predict with optional hysteresis to reduce turnover.
        
        Hysteresis: Only change prediction if probability difference
        exceeds threshold, reducing spurious position flips.
        """
        proba = self.predict_proba(X)
        
        if not apply_hysteresis or self.method == 'sample_weighting':
            return np.argmax(proba, axis=1)
        
        predictions = []
        current_position = None
        
        for i, p in enumerate(proba):
            pred_class = np.argmax(p)
            pred_confidence = p[pred_class]
            
            if current_position is None:
                # First prediction
                current_position = pred_class
            else:
                # Only change if confidence exceeds threshold
                opposite_prob = p[1 - current_position] if proba.shape[1] == 2 else 0
                
                if opposite_prob > 0.5 + self.hysteresis_threshold:
                    current_position = 1 - current_position
            
            predictions.append(current_position)
        
        return np.array(predictions)
    
    def compute_expected_turnover(self, X: pd.DataFrame) -> float:
        """Compute expected turnover rate from predictions."""
        predictions = self.predict(X, apply_hysteresis=False)
        changes = np.sum(np.abs(np.diff(predictions)))
        return changes / len(predictions)
    
    def compute_net_returns(self, X: pd.DataFrame, 
                            returns: pd.Series,
                            apply_hysteresis: bool = True) -> pd.Series:
        """
        Compute returns net of transaction costs.
        
        Args:
            X: Features
            returns: Future returns for each prediction
            apply_hysteresis: Whether to apply turnover reduction
        """
        predictions = self.predict(X, apply_hysteresis=apply_hysteresis)
        
        # Convert predictions to positions (-1, 0, 1)
        positions = predictions * 2 - 1  # 0 -> -1, 1 -> 1
        
        # Compute position changes for transaction costs
        position_changes = np.abs(np.diff(positions, prepend=0))
        
        # Transaction cost per change (in decimal)
        cost_per_trade = self.transaction_cost_bps / 10000
        
        # Gross returns from strategy
        gross_returns = positions * returns.values
        
        # Net returns after costs
        net_returns = gross_returns - (position_changes * cost_per_trade)
        
        return pd.Series(net_returns, index=returns.index)


# =============================================================================
# ENSEMBLE WITH TURNOVER OPTIMIZATION
# =============================================================================

@dataclass
class EnsembleMember:
    """Single model in the ensemble."""
    name: str
    model: TurnoverAwareModel
    weight: float = 1.0
    performance: Dict = field(default_factory=dict)


class TurnoverOptimizedEnsemble:
    """
    Ensemble that optimizes for risk-adjusted returns net of transaction costs.
    
    Features:
    1. Multiple base models with different characteristics
    2. Weighted voting based on historical performance
    3. Dynamic weight adjustment based on regime
    4. Consensus filtering to reduce spurious signals
    """
    
    def __init__(self,
                 models: List[Tuple[str, any]],
                 turnover_penalty: float = 0.5,
                 consensus_threshold: float = 0.6,
                 transaction_cost_bps: float = 10,
                 rebalance_frequency: int = 20):
        """
        Initialize ensemble.
        
        Args:
            models: List of (name, model) tuples
            turnover_penalty: Turnover penalty for all models
            consensus_threshold: Minimum agreement for signal
            transaction_cost_bps: Transaction cost in basis points
            rebalance_frequency: How often to rebalance weights
        """
        self.consensus_threshold = consensus_threshold
        self.transaction_cost_bps = transaction_cost_bps
        self.rebalance_frequency = rebalance_frequency
        
        # Wrap models with turnover awareness
        self.members: List[EnsembleMember] = []
        for name, model in models:
            wrapped = TurnoverAwareModel(
                model,
                turnover_penalty=turnover_penalty,
                method='both',
                transaction_cost_bps=transaction_cost_bps
            )
            self.members.append(EnsembleMember(name=name, model=wrapped))
        
        self.is_fitted = False
        self.performance_history = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: Optional[np.ndarray] = None):
        """Train all ensemble members."""
        for member in self.members:
            member.model.fit(X, y, sample_weight)
        
        self.is_fitted = True
        return self
    
    def _compute_member_weights(self, 
                                 X_val: pd.DataFrame,
                                 y_val: pd.Series,
                                 returns_val: pd.Series) -> np.ndarray:
        """
        Compute optimal weights for ensemble members based on validation performance.
        
        Weights are proportional to Sharpe ratio net of costs.
        """
        weights = []
        
        for member in self.members:
            # Compute predictions
            preds = member.model.predict(X_val)
            
            # Accuracy
            acc = accuracy_score(y_val, preds)
            
            # Net returns
            net_returns = member.model.compute_net_returns(X_val, returns_val)
            
            # Sharpe ratio (annualized assuming hourly data)
            sharpe = (net_returns.mean() / (net_returns.std() + 1e-8)) * np.sqrt(8760)
            
            # Turnover
            turnover = member.model.compute_expected_turnover(X_val)
            
            # Store performance
            member.performance = {
                'accuracy': acc,
                'sharpe': sharpe,
                'turnover': turnover,
                'net_return': net_returns.sum()
            }
            
            # Weight = max(0, sharpe) to avoid negative weights
            weight = max(0.1, sharpe)  # Minimum weight of 0.1
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return weights
    
    def update_weights(self, X_val: pd.DataFrame, y_val: pd.Series,
                       returns_val: pd.Series):
        """Update member weights based on recent performance."""
        weights = self._compute_member_weights(X_val, y_val, returns_val)
        
        for member, weight in zip(self.members, weights):
            member.weight = weight
        
        # Log performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'weights': {m.name: m.weight for m in self.members},
            'performance': {m.name: m.performance for m in self.members}
        })
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get weighted ensemble probability predictions.
        """
        probas = []
        weights = []
        
        for member in self.members:
            proba = member.model.predict_proba(X)
            probas.append(proba)
            weights.append(member.weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(probas[0])
        for proba, weight in zip(probas, weights):
            ensemble_proba += weight * proba
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame,
                use_consensus: bool = True) -> np.ndarray:
        """
        Get ensemble predictions with optional consensus filtering.
        
        Args:
            X: Features
            use_consensus: If True, only signal when consensus_threshold 
                          fraction of models agree
        
        Returns:
            Predictions: 1 (long), 0 (short), or -1 (no trade) if consensus required
        """
        if not use_consensus:
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        
        # Get individual predictions
        all_preds = []
        all_weights = []
        
        for member in self.members:
            preds = member.model.predict(X)
            all_preds.append(preds)
            all_weights.append(member.weight)
        
        all_preds = np.array(all_preds)  # Shape: (n_models, n_samples)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()
        
        # Compute weighted vote for each sample
        ensemble_preds = []
        for i in range(X.shape[0]):
            sample_preds = all_preds[:, i]
            
            # Weighted sum of predictions
            weighted_vote = np.sum(sample_preds * all_weights)
            
            # Check consensus
            if weighted_vote >= self.consensus_threshold:
                ensemble_preds.append(1)  # Strong long signal
            elif weighted_vote <= (1 - self.consensus_threshold):
                ensemble_preds.append(0)  # Strong short signal
            else:
                ensemble_preds.append(-1)  # No consensus - no trade
        
        return np.array(ensemble_preds)
    
    def get_model_agreement(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze agreement between ensemble members.
        
        Returns DataFrame with agreement statistics.
        """
        all_preds = {}
        
        for member in self.members:
            preds = member.model.predict(X)
            all_preds[member.name] = preds
        
        df = pd.DataFrame(all_preds, index=X.index)
        
        # Add agreement metrics
        df['agreement_pct'] = df.apply(lambda x: x.value_counts().max() / len(self.members), axis=1)
        df['consensus'] = df['agreement_pct'] >= self.consensus_threshold
        
        return df
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of member performance."""
        return pd.DataFrame([
            {
                'model': m.name,
                'weight': m.weight,
                **m.performance
            }
            for m in self.members
        ])


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def analyze_turnover_impact(model: TurnoverAwareModel,
                            X: pd.DataFrame,
                            y: pd.Series,
                            returns: pd.Series) -> Dict:
    """
    Analyze impact of turnover penalty on model performance.
    
    Compares:
    1. Raw predictions (no turnover penalty)
    2. With turnover penalty in training
    3. With hysteresis in prediction
    """
    # Raw predictions (no hysteresis)
    raw_preds = model.predict(X, apply_hysteresis=False)
    raw_turnover = np.sum(np.abs(np.diff(raw_preds))) / len(raw_preds)
    raw_returns = model.compute_net_returns(X, returns, apply_hysteresis=False)
    
    # With hysteresis
    smooth_preds = model.predict(X, apply_hysteresis=True)
    smooth_turnover = np.sum(np.abs(np.diff(smooth_preds))) / len(smooth_preds)
    smooth_returns = model.compute_net_returns(X, returns, apply_hysteresis=True)
    
    return {
        'raw': {
            'accuracy': accuracy_score(y, raw_preds),
            'turnover': raw_turnover,
            'total_return': raw_returns.sum(),
            'sharpe': (raw_returns.mean() / (raw_returns.std() + 1e-8)) * np.sqrt(8760)
        },
        'with_hysteresis': {
            'accuracy': accuracy_score(y, smooth_preds),
            'turnover': smooth_turnover,
            'total_return': smooth_returns.sum(),
            'sharpe': (smooth_returns.mean() / (smooth_returns.std() + 1e-8)) * np.sqrt(8760)
        },
        'improvement': {
            'turnover_reduction': (raw_turnover - smooth_turnover) / raw_turnover if raw_turnover > 0 else 0,
            'return_change': smooth_returns.sum() - raw_returns.sum(),
            'sharpe_change': (
                (smooth_returns.mean() / (smooth_returns.std() + 1e-8)) - 
                (raw_returns.mean() / (raw_returns.std() + 1e-8))
            ) * np.sqrt(8760)
        }
    }


# =============================================================================
# DEMO
# =============================================================================

def create_sample_data_with_returns():
    """Create sample data with returns for turnover analysis."""
    np.random.seed(42)
    n_samples = 3000
    
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='H')
    
    # Features
    X = pd.DataFrame(index=dates)
    
    # Price-based features
    noise = np.random.randn(n_samples) * 0.02
    trend = np.sin(np.arange(n_samples) / 100) * 0.01
    X['momentum'] = np.cumsum(trend + noise * 0.1)
    X['volatility'] = pd.Series(np.abs(noise)).rolling(20).std().fillna(0.01)
    X['rsi'] = 50 + np.cumsum(np.random.randn(n_samples) * 2).clip(-50, 50)
    
    for i in range(15):
        X[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Returns (what we're trying to predict direction of)
    returns = pd.Series(
        trend + noise,
        index=dates,
        name='returns'
    )
    
    # Labels: future return direction
    y = pd.Series(
        (returns.shift(-1) > 0).astype(int),
        index=dates,
        name='label'
    ).fillna(0).astype(int)
    
    return X, y, returns


def main():
    """Run turnover-aware ensemble demo."""
    print("="*70)
    print("TURNOVER-AWARE TRAINING + ENSEMBLE")
    print("="*70)
    
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed")
        return
    
    # Create data
    print("\n1. Creating sample data...")
    X, y, returns = create_sample_data_with_returns()
    
    # Split data
    train_end = int(len(X) * 0.6)
    val_end = int(len(X) * 0.8)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    returns_val = returns.iloc[train_end:val_end]
    returns_test = returns.iloc[val_end:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create individual turnover-aware model
    print("\n2. Training turnover-aware Random Forest...")
    ta_model = TurnoverAwareModel(
        base_model=RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        turnover_penalty=0.5,
        method='both',
        hysteresis_threshold=0.1,
        transaction_cost_bps=10
    )
    ta_model.fit(X_train, y_train)
    
    # Analyze turnover impact
    print("\n3. Analyzing turnover impact...")
    impact = analyze_turnover_impact(ta_model, X_test, y_test, returns_test)
    
    print(f"\n   Raw Predictions:")
    print(f"     Accuracy:    {impact['raw']['accuracy']:.4f}")
    print(f"     Turnover:    {impact['raw']['turnover']:.4f}")
    print(f"     Total Return:{impact['raw']['total_return']:.4f}")
    print(f"     Sharpe:      {impact['raw']['sharpe']:.2f}")
    
    print(f"\n   With Hysteresis:")
    print(f"     Accuracy:    {impact['with_hysteresis']['accuracy']:.4f}")
    print(f"     Turnover:    {impact['with_hysteresis']['turnover']:.4f}")
    print(f"     Total Return:{impact['with_hysteresis']['total_return']:.4f}")
    print(f"     Sharpe:      {impact['with_hysteresis']['sharpe']:.2f}")
    
    print(f"\n   Improvement:")
    print(f"     Turnover Reduction: {impact['improvement']['turnover_reduction']*100:.1f}%")
    print(f"     Return Change:      {impact['improvement']['return_change']:.4f}")
    print(f"     Sharpe Change:      {impact['improvement']['sharpe_change']:.2f}")
    
    # Create ensemble
    print("\n4. Building turnover-optimized ensemble...")
    models = [
        ('RF', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ('GB', GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)),
        ('LR', LogisticRegression(max_iter=1000, random_state=42)),
    ]
    
    ensemble = TurnoverOptimizedEnsemble(
        models=models,
        turnover_penalty=0.5,
        consensus_threshold=0.6,
        transaction_cost_bps=10
    )
    
    ensemble.fit(X_train, y_train)
    ensemble.update_weights(X_val, y_val, returns_val)
    
    print("\n5. Ensemble Performance:")
    print(ensemble.get_performance_summary().to_string())
    
    # Test ensemble
    print("\n6. Test Set Evaluation:")
    ensemble_preds = ensemble.predict(X_test, use_consensus=True)
    
    # Filter out no-trade signals
    valid_mask = ensemble_preds >= 0
    valid_preds = ensemble_preds[valid_mask]
    valid_y = y_test.values[valid_mask]
    
    if len(valid_preds) > 0:
        acc = accuracy_score(valid_y, valid_preds)
        print(f"   Accuracy (consensus trades only): {acc:.4f}")
        print(f"   Trades taken: {len(valid_preds)} / {len(y_test)} ({100*len(valid_preds)/len(y_test):.1f}%)")
        
        # Agreement analysis
        agreement = ensemble.get_model_agreement(X_test)
        print(f"   Average agreement: {agreement['agreement_pct'].mean():.2f}")
        print(f"   Consensus rate: {agreement['consensus'].mean()*100:.1f}%")
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. Turnover penalty reduces position changes by down-weighting
   samples immediately following position flips during training.

2. Hysteresis in prediction requires higher confidence to change
   positions, further reducing spurious trades.

3. Transaction costs significantly impact net returns - models
   must be trained with realistic cost assumptions.

4. Ensemble consensus filtering trades quality for quantity,
   taking only high-conviction signals.

5. Weight ensemble members by risk-adjusted returns NET of costs
   to properly account for turnover differences.

PRODUCTION RECOMMENDATIONS:
- Set transaction_cost_bps to match your actual trading costs
- Use turnover_penalty = 0.3-0.7 depending on your cost structure
- Require consensus_threshold >= 0.6 for signal validity
- Rebalance weights monthly or after significant regime changes
""")


if __name__ == "__main__":
    main()
