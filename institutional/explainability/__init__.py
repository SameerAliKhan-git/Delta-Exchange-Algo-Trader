"""
Explainability - SHAP/XAI for Model Interpretation
===================================================

Tools for understanding model decisions.
"""

from .shap_explainer import SHAPExplainer, FeatureImportance
from .model_cards import ModelCard, ModelCardBuilder

__all__ = [
    'SHAPExplainer',
    'FeatureImportance',
    'ModelCard',
    'ModelCardBuilder',
]
