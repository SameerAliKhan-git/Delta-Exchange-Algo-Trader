"""
Configuration Management for Quant-Bot

Centralized configuration with environment variable support and validation.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DataConfig:
    """Data loading configuration."""
    cache_dir: str = "./data/cache"
    default_start: str = "2020-01-01"
    default_end: str = "2024-12-31"
    ohlcv_columns: List[str] = field(default_factory=lambda: [
        "timestamp", "open", "high", "low", "close", "volume"
    ])
    datetime_col: str = "timestamp"
    resample_freq: str = "1h"  # Default timeframe
    
    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    volatility_window: int = 20
    volume_window: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Fractional differentiation
    frac_diff_d: float = 0.5  # Differencing order
    frac_diff_threshold: float = 1e-5  # Weight cutoff threshold


@dataclass 
class LabelingConfig:
    """Triple-barrier labeling configuration."""
    profit_taking_multiplier: float = 2.0  # Multiplier of daily volatility
    stop_loss_multiplier: float = 1.0
    max_holding_period: int = 20  # Maximum bars to hold
    min_return: float = 0.0001  # Minimum return to consider
    
    # Meta-labeling
    meta_labeling: bool = True
    primary_model_threshold: float = 0.5


@dataclass
class ModelConfig:
    """Model training configuration."""
    model_type: str = "xgboost"  # xgboost, lightgbm, random_forest
    
    # Cross-validation
    n_splits: int = 5
    purge_gap: int = 10  # Bars to purge between train/test
    embargo_pct: float = 0.01  # Embargo percentage
    
    # XGBoost defaults
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42
    })
    
    # LightGBM defaults  
    lgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "objective": "binary",
        "metric": "auc",
        "random_state": 42,
        "verbose": -1
    })


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    
    # Position sizing
    position_sizing: str = "fixed_fraction"  # fixed_fraction, kelly, equal_weight
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_pct: float = 0.20  # 20% max single position
    
    # Risk limits
    max_drawdown: float = 0.15  # 15% max drawdown circuit breaker
    daily_loss_limit: float = 0.03  # 3% daily loss limit
    
    # Execution
    fill_probability: float = 0.95  # 95% fill rate
    latency_ms: int = 50  # 50ms latency


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_portfolio_leverage: float = 2.0
    max_single_position: float = 0.20  # 20% of portfolio
    max_sector_exposure: float = 0.40  # 40% sector limit
    correlation_threshold: float = 0.7  # Reduce size for correlated assets
    
    # Circuit breakers
    drawdown_limit: float = 0.10  # 10% drawdown halt
    daily_var_limit: float = 0.05  # 5% daily VaR limit
    position_time_limit: int = 100  # Max bars to hold
    
    # Kelly criterion
    kelly_fraction: float = 0.25  # Use 1/4 Kelly


@dataclass
class ExchangeConfig:
    """Exchange API configuration."""
    exchange: str = "delta_exchange"
    api_key: str = os.getenv("DELTA_API_KEY", "")
    api_secret: str = os.getenv("DELTA_API_SECRET", "")
    base_url: str = "https://api.india.delta.exchange"
    ws_url: str = "wss://socket.india.delta.exchange"
    testnet: bool = True
    
    # Rate limits
    rate_limit_per_second: int = 10
    order_timeout_seconds: int = 30


@dataclass
class Config:
    """Master configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    
    # Global settings
    random_seed: int = 42
    log_level: str = "INFO"
    output_dir: str = "./output"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'data' in data:
            for k, v in data['data'].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)
        
        if 'features' in data:
            for k, v in data['features'].items():
                if hasattr(config.features, k):
                    setattr(config.features, k, v)
        
        if 'labeling' in data:
            for k, v in data['labeling'].items():
                if hasattr(config.labeling, k):
                    setattr(config.labeling, k, v)
                    
        if 'model' in data:
            for k, v in data['model'].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
                    
        if 'backtest' in data:
            for k, v in data['backtest'].items():
                if hasattr(config.backtest, k):
                    setattr(config.backtest, k, v)
        
        if 'risk' in data:
            for k, v in data['risk'].items():
                if hasattr(config.risk, k):
                    setattr(config.risk, k, v)
        
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        with open(path, 'w') as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration."""
    global _config
    _config = config


def load_config(path: str) -> Config:
    """Load and set configuration from file."""
    config = Config.from_yaml(path)
    set_config(config)
    return config


if __name__ == "__main__":
    # Demo
    config = get_config()
    print(f"Initial capital: ${config.backtest.initial_capital:,.2f}")
    print(f"Risk per trade: {config.backtest.risk_per_trade:.1%}")
    print(f"Max drawdown: {config.backtest.max_drawdown:.1%}")
