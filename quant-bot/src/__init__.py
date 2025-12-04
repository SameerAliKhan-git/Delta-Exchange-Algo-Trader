"""
Quant-Bot: Autonomous ML Trading System
=======================================

A comprehensive machine learning trading framework implementing:
- AFML (Advances in Financial Machine Learning) concepts
- Deep Learning (LSTM, GRU, TCN, Transformers)
- Reinforcement Learning (DQN, PPO, A2C)
- Portfolio Optimization (HRP, MVO, Black-Litterman)
- Advanced Signal Processing (Kalman, HMM, Wavelets)
- NLP/Sentiment Analysis (FinBERT)
- Automated Drift Detection & Retraining
- Smart Execution Algorithms (TWAP, VWAP, IS)

Master Concept Implementation Status:
1. ✓ Alternative Bar Sampling (data/bar_types.py)
2. ✓ Feature Engineering & Transformation
3. ✓ Signal Processing (features/signal_processing.py)
4. ✓ Machine Learning Models (model selection, meta-labeling)
5. ✓ Deep Learning (models/deep_learning.py)
6. ✓ Cross-Validation (PurgedKFold)
7. ✓ Portfolio Optimization (risk/portfolio_optimization.py)
8. ✓ Strategy Validation (backtest/validation.py)
9. ✓ Reinforcement Learning (models/reinforcement_learning.py)
10. ✓ Execution Algorithms (execution/execution_algos.py)
11. ✓ Risk Management (CVaR, position sizing)
12. ✓ NLP/Sentiment (signals/finbert_sentiment.py)
13. ✓ MLOps & Tracking (infrastructure/mlflow_tracking.py)
14. ✓ Drift Detection (models/drift_detection.py)
15. ✓ Autonomous Orchestration (engine/autonomous_orchestrator.py)
"""

from . import data
from . import features
from . import labeling
from . import models
from . import backtest
from . import execution
from . import risk
from . import utils
from .config import Config, get_config, load_config

__version__ = "2.0.0"
__author__ = "Quant-Bot Team"

__all__ = [
    'data',
    'features',
    'labeling',
    'models',
    'backtest',
    'execution',
    'risk',
    'utils',
    'Config',
    'get_config',
    'load_config',
    # Convenience functions
    'get_orchestrator',
    'get_model_trainer',
    'get_risk_optimizer',
    'get_execution_engine',
    'get_sentiment_analyzer',
    'get_drift_monitor',
    'get_experiment_tracker',
    'quick_start'
]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_orchestrator(config_dict: dict = None):
    """
    Get a configured autonomous orchestrator.
    
    Args:
        config_dict: Optional configuration dictionary
    
    Returns:
        AutonomousOrchestrator instance
    """
    from .engine.autonomous_orchestrator import AutonomousOrchestrator, OrchestratorConfig
    
    if config_dict:
        config = OrchestratorConfig(**config_dict)
    else:
        config = OrchestratorConfig()
    
    return AutonomousOrchestrator(config)


def get_model_trainer():
    """Get model training utilities."""
    from .models.deep_learning import train_deep_model
    from .models.reinforcement_learning import RLTrader
    from .backtest.validation import StrategyValidator
    
    return {
        'train_deep_model': train_deep_model,
        'rl_trader': RLTrader,
        'validator': StrategyValidator
    }


def get_risk_optimizer():
    """Get portfolio optimization utilities."""
    from .risk.portfolio_optimization import PortfolioOptimizer
    return PortfolioOptimizer


def get_execution_engine():
    """Get execution algorithm engine."""
    from .execution.execution_algos import ExecutionEngine
    return ExecutionEngine()


def get_sentiment_analyzer(use_finbert: bool = True):
    """Get sentiment analysis utilities."""
    from .signals.finbert_sentiment import NewsSentimentAnalyzer
    return NewsSentimentAnalyzer(use_finbert=use_finbert)


def get_drift_monitor():
    """Get drift detection utilities."""
    from .models.drift_detection import DriftAlertSystem
    return DriftAlertSystem()


def get_experiment_tracker(tracking_uri: str = None):
    """Get MLflow experiment tracker."""
    from .infrastructure.mlflow_tracking import MLflowTracker
    return MLflowTracker(tracking_uri=tracking_uri)


def quick_start():
    """Print quick start guide."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUANT-BOT AUTONOMOUS TRADING SYSTEM v2.0                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Quick Start:                                                                ║
║  ───────────                                                                 ║
║  1. Install dependencies:                                                    ║
║     pip install -r requirements_full.txt                                     ║
║                                                                              ║
║  2. Configure system:                                                        ║
║     from src import get_orchestrator                                         ║
║     orchestrator = get_orchestrator({                                        ║
║         'symbols': ['BTCUSDT', 'ETHUSDT'],                                   ║
║         'position_size_usd': 1000.0,                                         ║
║         'max_positions': 5                                                   ║
║     })                                                                       ║
║                                                                              ║
║  3. Run trading:                                                             ║
║     import asyncio                                                           ║
║     asyncio.run(orchestrator.run())                                          ║
║                                                                              ║
║  Key Components:                                                             ║
║  ───────────────                                                             ║
║  • AutonomousOrchestrator - Main trading controller                          ║
║  • ModelManager - ML model lifecycle                                         ║
║  • RiskController - Position & risk management                               ║
║  • ExecutionEngine - Smart order execution                                   ║
║  • DriftAlertSystem - Model performance monitoring                           ║
║  • MLflowTracker - Experiment tracking                                       ║
║                                                                              ║
║  See individual modules for detailed documentation.                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
