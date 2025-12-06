"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         MODEL CARDS - Standardized Model Documentation                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Model Cards provide standardized documentation for ML models, including:
- Model details and intended use
- Training data and methodology
- Performance metrics
- Limitations and ethical considerations
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger("Explainability.ModelCards")


@dataclass
class ModelCard:
    """
    Standardized model documentation card.
    
    Based on Google's Model Cards for Model Reporting.
    """
    # Model Details
    model_id: str
    model_name: str
    version: str
    model_type: str
    description: str
    
    # Development Info
    developers: List[str] = field(default_factory=list)
    organization: str = ""
    created_date: str = ""
    last_updated: str = ""
    
    # Intended Use
    primary_intended_uses: List[str] = field(default_factory=list)
    primary_intended_users: List[str] = field(default_factory=list)
    out_of_scope_uses: List[str] = field(default_factory=list)
    
    # Training Data
    training_data_description: str = ""
    training_data_size: Optional[int] = None
    training_data_date_range: str = ""
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_count: int = 0
    features: List[str] = field(default_factory=list)
    
    # Architecture
    architecture_description: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance Metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    evaluation_data_description: str = ""
    validation_approach: str = ""
    
    # Limitations
    known_limitations: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    
    # Ethical Considerations
    ethical_considerations: List[str] = field(default_factory=list)
    
    # Deployment
    deployment_status: str = ""  # development, staging, production
    hardware_requirements: str = ""
    inference_latency_ms: Optional[float] = None
    
    # Monitoring
    monitoring_metrics: List[str] = field(default_factory=list)
    retraining_frequency: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'model_details': {
                'model_id': self.model_id,
                'model_name': self.model_name,
                'version': self.version,
                'model_type': self.model_type,
                'description': self.description,
            },
            'development': {
                'developers': self.developers,
                'organization': self.organization,
                'created_date': self.created_date,
                'last_updated': self.last_updated,
            },
            'intended_use': {
                'primary_intended_uses': self.primary_intended_uses,
                'primary_intended_users': self.primary_intended_users,
                'out_of_scope_uses': self.out_of_scope_uses,
            },
            'training_data': {
                'description': self.training_data_description,
                'size': self.training_data_size,
                'date_range': self.training_data_date_range,
                'preprocessing_steps': self.preprocessing_steps,
                'feature_count': self.feature_count,
                'features': self.features,
            },
            'architecture': {
                'description': self.architecture_description,
                'hyperparameters': self.hyperparameters,
            },
            'performance': {
                'metrics': self.performance_metrics,
                'evaluation_data': self.evaluation_data_description,
                'validation_approach': self.validation_approach,
            },
            'limitations': {
                'known_limitations': self.known_limitations,
                'failure_modes': self.failure_modes,
            },
            'ethical_considerations': self.ethical_considerations,
            'deployment': {
                'status': self.deployment_status,
                'hardware_requirements': self.hardware_requirements,
                'inference_latency_ms': self.inference_latency_ms,
            },
            'monitoring': {
                'metrics': self.monitoring_metrics,
                'retraining_frequency': self.retraining_frequency,
            },
        }
    
    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        lines = [
            f"# Model Card: {self.model_name}",
            "",
            f"**Model ID:** {self.model_id}",
            f"**Version:** {self.version}",
            f"**Last Updated:** {self.last_updated}",
            "",
            "---",
            "",
            "## Model Details",
            "",
            f"**Type:** {self.model_type}",
            "",
            f"**Description:** {self.description}",
            "",
            f"**Developers:** {', '.join(self.developers)}",
            f"**Organization:** {self.organization}",
            "",
            "---",
            "",
            "## Intended Use",
            "",
            "### Primary Intended Uses",
            "",
        ]
        
        for use in self.primary_intended_uses:
            lines.append(f"- {use}")
        
        lines.extend([
            "",
            "### Primary Intended Users",
            "",
        ])
        
        for user in self.primary_intended_users:
            lines.append(f"- {user}")
        
        lines.extend([
            "",
            "### Out-of-Scope Uses",
            "",
        ])
        
        for use in self.out_of_scope_uses:
            lines.append(f"- ⚠️ {use}")
        
        lines.extend([
            "",
            "---",
            "",
            "## Training Data",
            "",
            f"**Description:** {self.training_data_description}",
            "",
            f"**Size:** {self.training_data_size:,} samples" if self.training_data_size else "",
            f"**Date Range:** {self.training_data_date_range}",
            "",
            "### Features",
            "",
            f"**Count:** {self.feature_count}",
            "",
            "| Feature Name |",
            "|--------------|",
        ])
        
        for feat in self.features[:20]:
            lines.append(f"| {feat} |")
        
        if len(self.features) > 20:
            lines.append(f"| ... and {len(self.features) - 20} more |")
        
        lines.extend([
            "",
            "### Preprocessing",
            "",
        ])
        
        for i, step in enumerate(self.preprocessing_steps):
            lines.append(f"{i+1}. {step}")
        
        lines.extend([
            "",
            "---",
            "",
            "## Architecture",
            "",
            f"{self.architecture_description}",
            "",
            "### Hyperparameters",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
        ])
        
        for name, value in self.hyperparameters.items():
            lines.append(f"| {name} | {value} |")
        
        lines.extend([
            "",
            "---",
            "",
            "## Performance Metrics",
            "",
            f"**Validation Approach:** {self.validation_approach}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        
        for name, value in self.performance_metrics.items():
            lines.append(f"| {name} | {value:.4f} |")
        
        lines.extend([
            "",
            f"**Evaluation Data:** {self.evaluation_data_description}",
            "",
            "---",
            "",
            "## Limitations",
            "",
            "### Known Limitations",
            "",
        ])
        
        for lim in self.known_limitations:
            lines.append(f"- {lim}")
        
        lines.extend([
            "",
            "### Failure Modes",
            "",
        ])
        
        for mode in self.failure_modes:
            lines.append(f"- ⚠️ {mode}")
        
        lines.extend([
            "",
            "---",
            "",
            "## Ethical Considerations",
            "",
        ])
        
        for consideration in self.ethical_considerations:
            lines.append(f"- {consideration}")
        
        lines.extend([
            "",
            "---",
            "",
            "## Deployment",
            "",
            f"**Status:** {self.deployment_status}",
            f"**Hardware:** {self.hardware_requirements}",
            f"**Inference Latency:** {self.inference_latency_ms}ms" if self.inference_latency_ms else "",
            "",
            "---",
            "",
            "## Monitoring",
            "",
            f"**Retraining Frequency:** {self.retraining_frequency}",
            "",
            "### Monitored Metrics",
            "",
        ])
        
        for metric in self.monitoring_metrics:
            lines.append(f"- {metric}")
        
        return "\n".join(lines)
    
    def save(self, filepath: str) -> None:
        """Save model card to file."""
        if filepath.endswith('.md'):
            with open(filepath, 'w') as f:
                f.write(self.to_markdown())
        else:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Model card saved to {filepath}")


class ModelCardBuilder:
    """
    Builder for creating model cards.
    
    Usage:
        card = (
            ModelCardBuilder()
            .basic_info("momentum_v2", "Momentum Strategy", "2.1.0", "LSTM")
            .description("LSTM-based momentum prediction model")
            .training_data("Historical BTC/ETH data", 500000, "2020-2024")
            .performance({'sharpe': 1.5, 'accuracy': 0.58})
            .build()
        )
    """
    
    def __init__(self):
        self._data = {}
    
    def basic_info(
        self,
        model_id: str,
        model_name: str,
        version: str,
        model_type: str,
    ) -> 'ModelCardBuilder':
        self._data['model_id'] = model_id
        self._data['model_name'] = model_name
        self._data['version'] = version
        self._data['model_type'] = model_type
        return self
    
    def description(self, desc: str) -> 'ModelCardBuilder':
        self._data['description'] = desc
        return self
    
    def developers(self, devs: List[str], org: str = "") -> 'ModelCardBuilder':
        self._data['developers'] = devs
        self._data['organization'] = org
        self._data['created_date'] = datetime.now().strftime("%Y-%m-%d")
        self._data['last_updated'] = datetime.now().strftime("%Y-%m-%d")
        return self
    
    def intended_use(
        self,
        uses: List[str],
        users: List[str],
        out_of_scope: Optional[List[str]] = None,
    ) -> 'ModelCardBuilder':
        self._data['primary_intended_uses'] = uses
        self._data['primary_intended_users'] = users
        self._data['out_of_scope_uses'] = out_of_scope or []
        return self
    
    def training_data(
        self,
        description: str,
        size: int,
        date_range: str,
        features: Optional[List[str]] = None,
        preprocessing: Optional[List[str]] = None,
    ) -> 'ModelCardBuilder':
        self._data['training_data_description'] = description
        self._data['training_data_size'] = size
        self._data['training_data_date_range'] = date_range
        self._data['features'] = features or []
        self._data['feature_count'] = len(features) if features else 0
        self._data['preprocessing_steps'] = preprocessing or []
        return self
    
    def architecture(
        self,
        description: str,
        hyperparameters: Optional[Dict] = None,
    ) -> 'ModelCardBuilder':
        self._data['architecture_description'] = description
        self._data['hyperparameters'] = hyperparameters or {}
        return self
    
    def performance(
        self,
        metrics: Dict[str, float],
        eval_data: str = "",
        validation: str = "",
    ) -> 'ModelCardBuilder':
        self._data['performance_metrics'] = metrics
        self._data['evaluation_data_description'] = eval_data
        self._data['validation_approach'] = validation
        return self
    
    def limitations(
        self,
        known: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
    ) -> 'ModelCardBuilder':
        self._data['known_limitations'] = known or []
        self._data['failure_modes'] = failures or []
        return self
    
    def ethical(self, considerations: List[str]) -> 'ModelCardBuilder':
        self._data['ethical_considerations'] = considerations
        return self
    
    def deployment(
        self,
        status: str,
        hardware: str = "",
        latency: Optional[float] = None,
    ) -> 'ModelCardBuilder':
        self._data['deployment_status'] = status
        self._data['hardware_requirements'] = hardware
        self._data['inference_latency_ms'] = latency
        return self
    
    def monitoring(
        self,
        metrics: List[str],
        retrain_freq: str = "",
    ) -> 'ModelCardBuilder':
        self._data['monitoring_metrics'] = metrics
        self._data['retraining_frequency'] = retrain_freq
        return self
    
    def build(self) -> ModelCard:
        return ModelCard(**self._data)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Build a model card
    card = (
        ModelCardBuilder()
        .basic_info(
            "momentum_lstm_v2",
            "Momentum LSTM Predictor",
            "2.1.0",
            "LSTM Neural Network"
        )
        .description(
            "LSTM-based model for predicting short-term price momentum in crypto markets. "
            "Outputs a probability of upward movement in the next 5-minute candle."
        )
        .developers(["Trading Team"], "Delta Algo Trader")
        .intended_use(
            uses=[
                "Generate momentum signals for BTC and ETH perpetual futures",
                "Provide probability estimates for trend continuation",
            ],
            users=["Automated trading system", "Strategy ensemble"],
            out_of_scope=[
                "Long-term (>1 hour) predictions",
                "Other asset classes (stocks, forex)",
                "Markets with <$10M daily volume",
            ]
        )
        .training_data(
            description="Historical 1-minute OHLCV data from Binance and Delta Exchange",
            size=2_500_000,
            date_range="2021-01-01 to 2024-11-30",
            features=['close', 'volume', 'rsi', 'macd', 'obi', 'cvd', 'atr', 'bb_width'],
            preprocessing=[
                "Missing value forward-fill",
                "Z-score normalization (rolling 100-bar window)",
                "Sequence creation (20 timesteps)",
                "Train/val/test split with purging",
            ]
        )
        .architecture(
            description="2-layer LSTM with attention mechanism, followed by dense layers",
            hyperparameters={
                'lstm_units': [64, 32],
                'dropout': 0.2,
                'attention': True,
                'dense_units': [16, 8],
                'learning_rate': 0.001,
                'batch_size': 256,
            }
        )
        .performance(
            metrics={
                'sharpe_ratio': 1.52,
                'accuracy': 0.548,
                'precision': 0.562,
                'recall': 0.534,
                'profit_factor': 1.35,
            },
            eval_data="Out-of-sample test set (2024-06-01 to 2024-11-30)",
            validation="Purged K-Fold (5 folds, 100-bar gap)",
        )
        .limitations(
            known=[
                "Performance degrades in low-volatility regimes",
                "Accuracy drops during major news events",
                "Model lag in rapidly changing markets (< 50ms)",
            ],
            failures=[
                "May generate overconfident signals during regime transitions",
                "Volume-based features unreliable during exchange maintenance",
            ]
        )
        .ethical([
            "Model should not be used for market manipulation",
            "Position sizing must respect liquidity constraints",
            "Kill switch must be enabled for all live deployments",
        ])
        .deployment(
            status="production",
            hardware="GPU (T4 or better recommended)",
            latency=5.2,
        )
        .monitoring(
            metrics=[
                "Prediction accuracy (rolling 24h)",
                "Sharpe ratio (rolling 7d)",
                "Feature drift (KL divergence)",
                "Latency p99",
            ],
            retrain_freq="Weekly (or on regime change)",
        )
        .build()
    )
    
    # Print markdown
    print(card.to_markdown())
    
    # Save
    card.save("model_card_momentum_lstm.md")
