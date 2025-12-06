"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         SHAP EXPLAINER - Model Interpretability                               ║
║                                                                               ║
║  SHAP (SHapley Additive exPlanations) for understanding model predictions    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Provides interpretable explanations for:
1. Why a specific prediction was made
2. Which features are most important globally
3. How features interact

This is critical for:
- Regulatory compliance
- Model debugging
- Building trust in autonomous decisions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
import json

logger = logging.getLogger("Explainability.SHAP")

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed - using fallback importance")


@dataclass
class FeatureImportance:
    """Feature importance measurement."""
    feature_name: str
    importance: float  # Absolute importance
    direction: str     # 'positive', 'negative', or 'neutral'
    shap_value: float  # Raw SHAP value
    feature_value: float
    contribution_pct: float


@dataclass
class PredictionExplanation:
    """Complete explanation for a single prediction."""
    prediction_id: str
    model_id: str
    timestamp: datetime
    
    # Prediction details
    predicted_value: float
    predicted_class: Optional[str] = None
    probability: Optional[float] = None
    
    # Base value (expected prediction)
    base_value: float = 0.0
    
    # Feature contributions
    feature_contributions: List[FeatureImportance] = field(default_factory=list)
    
    # Top contributors
    top_positive: List[str] = field(default_factory=list)
    top_negative: List[str] = field(default_factory=list)
    
    # Summary
    explanation_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'prediction_id': self.prediction_id,
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'predicted_value': self.predicted_value,
            'predicted_class': self.predicted_class,
            'probability': self.probability,
            'base_value': self.base_value,
            'feature_contributions': [
                {
                    'feature': fc.feature_name,
                    'importance': fc.importance,
                    'direction': fc.direction,
                    'shap_value': fc.shap_value,
                    'feature_value': fc.feature_value,
                    'contribution_pct': fc.contribution_pct,
                }
                for fc in self.feature_contributions
            ],
            'top_positive': self.top_positive,
            'top_negative': self.top_negative,
            'explanation_text': self.explanation_text,
        }


class SHAPExplainer:
    """
    SHAP-based model explainer.
    
    Provides interpretable explanations for model predictions using
    SHAP (SHapley Additive exPlanations) values.
    
    Usage:
        explainer = SHAPExplainer(model, feature_names)
        
        # Fit on background data
        explainer.fit(X_background)
        
        # Explain a prediction
        explanation = explainer.explain(X_single)
        print(explanation.explanation_text)
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_id: str = "unknown",
        model_type: str = "tree",  # 'tree', 'linear', 'deep', 'kernel'
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The model to explain
            feature_names: List of feature names
            model_id: Identifier for the model
            model_type: Type of model for appropriate explainer
        """
        self.model = model
        self.feature_names = feature_names
        self.model_id = model_id
        self.model_type = model_type
        
        self._explainer = None
        self._background_data = None
        self._global_importance: Optional[Dict[str, float]] = None
        
        logger.info(f"SHAPExplainer initialized for model '{model_id}' with {len(feature_names)} features")
    
    def fit(self, X_background: np.ndarray, sample_size: int = 100) -> None:
        """
        Fit the explainer on background data.
        
        Args:
            X_background: Background dataset for SHAP calculations
            sample_size: Number of samples to use (for efficiency)
        """
        # Sample if needed
        if len(X_background) > sample_size:
            indices = np.random.choice(len(X_background), sample_size, replace=False)
            self._background_data = X_background[indices]
        else:
            self._background_data = X_background
        
        if SHAP_AVAILABLE:
            try:
                if self.model_type == 'tree':
                    self._explainer = shap.TreeExplainer(self.model)
                elif self.model_type == 'linear':
                    self._explainer = shap.LinearExplainer(self.model, self._background_data)
                elif self.model_type == 'deep':
                    self._explainer = shap.DeepExplainer(self.model, self._background_data)
                else:
                    self._explainer = shap.KernelExplainer(
                        self.model.predict if hasattr(self.model, 'predict') else self.model,
                        self._background_data
                    )
                logger.info(f"SHAP explainer fitted with {len(self._background_data)} background samples")
            except Exception as e:
                logger.error(f"Failed to create SHAP explainer: {e}")
                self._explainer = None
        else:
            logger.info("SHAP not available, using fallback importance")
    
    def explain(
        self,
        X: np.ndarray,
        prediction_id: Optional[str] = None,
    ) -> PredictionExplanation:
        """
        Explain a prediction.
        
        Args:
            X: Input features (single sample or batch)
            prediction_id: Optional ID for the prediction
            
        Returns:
            PredictionExplanation with feature contributions
        """
        import uuid
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get prediction
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(X)[0]
        else:
            prediction = self.model(X)[0]
        
        # Get SHAP values
        if SHAP_AVAILABLE and self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(X)
                
                # Handle multi-output
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                shap_values = shap_values[0] if shap_values.ndim > 1 else shap_values
                base_value = self._explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0]
                    
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
                shap_values, base_value = self._fallback_importance(X[0])
        else:
            shap_values, base_value = self._fallback_importance(X[0])
        
        # Create feature contributions
        total_abs = np.sum(np.abs(shap_values)) + 1e-10
        contributions = []
        
        for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            direction = 'positive' if shap_val > 0 else 'negative' if shap_val < 0 else 'neutral'
            
            contributions.append(FeatureImportance(
                feature_name=name,
                importance=abs(shap_val),
                direction=direction,
                shap_value=float(shap_val),
                feature_value=float(X[0, i]) if i < X.shape[1] else 0,
                contribution_pct=abs(shap_val) / total_abs * 100,
            ))
        
        # Sort by importance
        contributions.sort(key=lambda x: x.importance, reverse=True)
        
        # Find top contributors
        top_positive = [c.feature_name for c in contributions if c.direction == 'positive'][:3]
        top_negative = [c.feature_name for c in contributions if c.direction == 'negative'][:3]
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            prediction, contributions, top_positive, top_negative
        )
        
        return PredictionExplanation(
            prediction_id=prediction_id or str(uuid.uuid4())[:8],
            model_id=self.model_id,
            timestamp=datetime.now(),
            predicted_value=float(prediction),
            base_value=float(base_value),
            feature_contributions=contributions,
            top_positive=top_positive,
            top_negative=top_negative,
            explanation_text=explanation_text,
        )
    
    def _fallback_importance(self, X: np.ndarray) -> tuple:
        """Fallback importance when SHAP is not available."""
        # Simple perturbation-based importance
        if self._background_data is None:
            return np.zeros(len(self.feature_names)), 0
        
        base_pred = self.model.predict(X.reshape(1, -1))[0] if hasattr(self.model, 'predict') else 0
        importances = []
        
        for i in range(len(self.feature_names)):
            X_perturbed = X.copy()
            # Perturb to background mean
            X_perturbed[i] = np.mean(self._background_data[:, i])
            
            try:
                perturbed_pred = self.model.predict(X_perturbed.reshape(1, -1))[0]
                importance = base_pred - perturbed_pred
            except Exception:
                importance = 0
            
            importances.append(importance)
        
        return np.array(importances), np.mean(self._background_data[:, 0])
    
    def _generate_explanation_text(
        self,
        prediction: float,
        contributions: List[FeatureImportance],
        top_positive: List[str],
        top_negative: List[str],
    ) -> str:
        """Generate human-readable explanation."""
        lines = []
        
        lines.append(f"Prediction: {prediction:.4f}")
        lines.append("")
        
        if top_positive:
            lines.append(f"Top bullish factors: {', '.join(top_positive)}")
        if top_negative:
            lines.append(f"Top bearish factors: {', '.join(top_negative)}")
        
        lines.append("")
        lines.append("Top feature contributions:")
        
        for i, c in enumerate(contributions[:5]):
            sign = "+" if c.direction == 'positive' else "-"
            lines.append(f"  {i+1}. {c.feature_name}: {sign}{c.importance:.4f} ({c.contribution_pct:.1f}%)")
        
        return "\n".join(lines)
    
    def get_global_importance(
        self,
        X: np.ndarray,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Calculate global feature importance.
        
        Args:
            X: Dataset to explain
            n_samples: Number of samples to use
            
        Returns:
            Dict mapping feature name to importance
        """
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
        
        if SHAP_AVAILABLE and self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Mean absolute SHAP value per feature
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                
                self._global_importance = {
                    name: float(imp)
                    for name, imp in zip(self.feature_names, mean_abs)
                }
                
            except Exception as e:
                logger.error(f"Failed to calculate global importance: {e}")
                self._global_importance = {name: 0.0 for name in self.feature_names}
        else:
            # Random importance for fallback
            self._global_importance = {name: 1.0 / len(self.feature_names) for name in self.feature_names}
        
        return self._global_importance
    
    def feature_importance_report(self) -> str:
        """Generate a feature importance report."""
        if not self._global_importance:
            return "No global importance calculated. Call get_global_importance() first."
        
        sorted_features = sorted(
            self._global_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        lines = [
            "# Feature Importance Report",
            f"Model: {self.model_id}",
            f"Features: {len(self.feature_names)}",
            "",
            "## Ranking",
            "",
            "| Rank | Feature | Importance |",
            "|------|---------|------------|",
        ]
        
        for i, (name, imp) in enumerate(sorted_features):
            lines.append(f"| {i+1} | {name} | {imp:.4f} |")
        
        return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple mock model
    class MockModel:
        def predict(self, X):
            return X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2
    
    model = MockModel()
    feature_names = ['rsi', 'momentum', 'volatility', 'obi', 'cvd']
    
    # Create explainer
    explainer = SHAPExplainer(
        model=model,
        feature_names=feature_names,
        model_id="mock_momentum_v1",
        model_type="kernel",
    )
    
    # Create background data
    np.random.seed(42)
    X_background = np.random.randn(100, 5)
    explainer.fit(X_background)
    
    # Explain a prediction
    X_test = np.array([[0.7, 0.5, 0.3, 0.6, 0.4]])
    explanation = explainer.explain(X_test)
    
    print("=== Prediction Explanation ===")
    print(explanation.explanation_text)
    
    # Get global importance
    importance = explainer.get_global_importance(X_background)
    
    print("\n=== Global Importance ===")
    for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")
    
    # Generate report
    print("\n=== Report ===")
    print(explainer.feature_importance_report())
    
    # JSON output
    print("\n=== JSON ===")
    print(json.dumps(explanation.to_dict(), indent=2))
