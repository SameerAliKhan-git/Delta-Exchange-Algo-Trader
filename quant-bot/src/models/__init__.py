"""
Models module for quant-bot.

Includes:
- Traditional ML models (XGBoost, LightGBM, RandomForest)
- Deep Learning (LSTM, GRU, TCN, Transformer)
- Reinforcement Learning (DQN, PPO, A2C)
- Drift Detection
- Cross-validation (PurgedKFold)
"""

from .train import (
    PurgedKFold,
    CombinatorialPurgedCV,
    ModelTrainer,
    TrainingResults,
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    BaseModel
)

# Deep Learning models
from .deep_learning import (
    LSTMCell,
    GRUCell,
    LSTMNetwork,
    Attention,
    MultiHeadAttention,
    TransformerBlock,
    TransformerEncoder,
    Autoencoder,
    DeepTradingModel
)

# Reinforcement Learning
from .reinforcement_learning import (
    TradingState,
    StepResult,
    TradingEnvironment,
    ReplayBuffer,
    DQNAgent,
    PolicyGradientAgent,
    RLTrainer
)

# Drift Detection
from .drift_detection import (
    DriftResult,
    PerformanceDegradation,
    StatisticalDriftTests,
    FeatureDriftMonitor,
    PerformanceMonitor,
    CUSUMDetector,
    DriftAlertSystem,
    RetrainingTrigger
)

__all__ = [
    # Traditional ML
    'PurgedKFold',
    'CombinatorialPurgedCV',
    'ModelTrainer',
    'TrainingResults',
    'XGBoostModel',
    'LightGBMModel',
    'RandomForestModel',
    'BaseModel',
    # Deep Learning
    'LSTMCell',
    'GRUCell',
    'LSTMNetwork',
    'Attention',
    'MultiHeadAttention',
    'TransformerBlock',
    'TransformerEncoder',
    'Autoencoder',
    'DeepTradingModel',
    # Reinforcement Learning
    'TradingState',
    'StepResult',
    'TradingEnvironment',
    'ReplayBuffer',
    'DQNAgent',
    'PolicyGradientAgent',
    'RLTrainer',
    # Drift Detection
    'DriftResult',
    'PerformanceDegradation',
    'StatisticalDriftTests',
    'FeatureDriftMonitor',
    'PerformanceMonitor',
    'CUSUMDetector',
    'DriftAlertSystem',
    'RetrainingTrigger'
]
