"""
Configuration Management for Delta Exchange Algo Trading Bot
Uses Pydantic for validation and type safety
"""

import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DeltaExchangeConfig(BaseSettings):
    """Delta Exchange API configuration"""
    api_key: str = Field(default="", alias="DELTA_API_KEY")
    api_secret: str = Field(default="", alias="DELTA_API_SECRET")
    base_url: str = Field(
        default="https://cdn-ind.testnet.deltaex.org",
        alias="DELTA_BASE_URL"
    )
    
    @property
    def is_testnet(self) -> bool:
        return "testnet" in self.base_url.lower()


class TradingConfig(BaseSettings):
    """Trading configuration"""
    product_symbol: str = Field(default="BTCUSD", alias="PRODUCT_SYMBOL")
    product_id: int = Field(default=139, alias="PRODUCT_ID")
    trading_mode: str = Field(default="paper", alias="TRADING_MODE")  # paper or live
    
    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, v):
        if v not in ("paper", "live"):
            raise ValueError("trading_mode must be 'paper' or 'live'")
        return v
    
    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"


class RiskConfig(BaseSettings):
    """Risk management configuration - CRITICAL SETTINGS"""
    risk_per_trade_inr: float = Field(default=500.0, alias="RISK_PER_TRADE_INR")
    max_daily_loss_inr: float = Field(default=5000.0, alias="MAX_DAILY_LOSS_INR")
    max_open_positions: int = Field(default=3, alias="MAX_OPEN_POSITIONS")
    min_trade_cooldown_seconds: int = Field(default=60, alias="MIN_TRADE_COOLDOWN_SECONDS")
    default_stop_loss_pct: float = Field(default=0.02, alias="DEFAULT_STOP_LOSS_PCT")
    default_take_profit_pct: float = Field(default=0.04, alias="DEFAULT_TAKE_PROFIT_PCT")
    enable_trailing_stop: bool = Field(default=True, alias="ENABLE_TRAILING_STOP")
    trailing_stop_pct: float = Field(default=0.015, alias="TRAILING_STOP_PCT")
    max_position_size: float = Field(default=10.0, alias="MAX_POSITION_SIZE")
    
    # Capital-aware sizing (V2)
    exposure_pct: float = Field(default=0.25, alias="EXPOSURE_PCT")
    option_max_premium_pct: float = Field(default=0.005, alias="OPTION_MAX_PREMIUM_PCT")
    confidence_threshold: float = Field(default=0.65, alias="CONFIDENCE_THRESHOLD")
    min_liquidity_usd: float = Field(default=1000.0, alias="MIN_LIQUIDITY_USD")
    option_target_delta: float = Field(default=0.35, alias="OPTION_TARGET_DELTA")
    option_min_expiry_days: int = Field(default=3, alias="OPTION_MIN_EXPIRY_DAYS")
    option_max_expiry_days: int = Field(default=30, alias="OPTION_MAX_EXPIRY_DAYS")
    
    @field_validator("risk_per_trade_inr", "max_daily_loss_inr")
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    @field_validator("exposure_pct", "option_max_premium_pct", "confidence_threshold")
    @classmethod
    def validate_percentage(cls, v):
        if not 0 < v <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        return v


class StrategyConfig(BaseSettings):
    """Strategy parameters"""
    ema_fast_period: int = Field(default=20, alias="EMA_FAST_PERIOD")
    ema_slow_period: int = Field(default=50, alias="EMA_SLOW_PERIOD")
    atr_period: int = Field(default=14, alias="ATR_PERIOD")
    sentiment_bull_threshold: float = Field(default=0.35, alias="SENTIMENT_BULL_THRESHOLD")
    sentiment_bear_threshold: float = Field(default=-0.35, alias="SENTIMENT_BEAR_THRESHOLD")
    ob_imbalance_threshold: float = Field(default=0.05, alias="OB_IMBALANCE_THRESHOLD")
    momentum_threshold: float = Field(default=0.002, alias="MOMENTUM_THRESHOLD")
    price_buffer_size: int = Field(default=500, alias="PRICE_BUFFER_SIZE")
    warmup_period: int = Field(default=60, alias="WARMUP_PERIOD")


class DataIngestionConfig(BaseSettings):
    """Data ingestion configuration"""
    ticker_poll_interval: int = Field(default=5, alias="TICKER_POLL_INTERVAL")
    orderbook_depth: int = Field(default=10, alias="ORDERBOOK_DEPTH")
    
    # Twitter API
    twitter_bearer_token: Optional[str] = Field(default=None, alias="TWITTER_BEARER_TOKEN")
    twitter_api_key: Optional[str] = Field(default=None, alias="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(default=None, alias="TWITTER_API_SECRET")
    
    # Reddit API
    reddit_client_id: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="DeltaAlgoBot/1.0", alias="REDDIT_USER_AGENT")
    
    # News APIs
    newsapi_key: Optional[str] = Field(default=None, alias="NEWSAPI_KEY")
    cryptopanic_api_key: Optional[str] = Field(default=None, alias="CRYPTOPANIC_API_KEY")


class AlertingConfig(BaseSettings):
    """Alerting configuration"""
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")
    slack_webhook_url: Optional[str] = Field(default=None, alias="SLACK_WEBHOOK_URL")
    alert_events: List[str] = Field(
        default=["order_placed", "order_filled", "stop_triggered", "daily_loss_limit", "error", "kill_switch"],
        alias="ALERT_EVENTS"
    )
    
    @field_validator("alert_events", mode="before")
    @classmethod
    def parse_alert_events(cls, v):
        if isinstance(v, str):
            return [e.strip() for e in v.split(",")]
        return v


class MonitoringConfig(BaseSettings):
    """Monitoring configuration"""
    prometheus_port: int = Field(default=8000, alias="PROMETHEUS_PORT")
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    database_url: str = Field(default="sqlite:///./data/trades.db", alias="DATABASE_URL")


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/trading.log", alias="LOG_FILE")
    json_logging: bool = Field(default=False, alias="JSON_LOGGING")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class SystemConfig(BaseSettings):
    """System configuration"""
    kill_switch_file: str = Field(default="./KILL_SWITCH", alias="KILL_SWITCH_FILE")
    health_check_interval: int = Field(default=30, alias="HEALTH_CHECK_INTERVAL")
    timezone: str = Field(default="Asia/Kolkata", alias="TIMEZONE")


class BacktestConfig(BaseSettings):
    """Backtest configuration"""
    backtest_data_file: str = Field(default="./data/historical_prices.csv", alias="BACKTEST_DATA_FILE")
    backtest_start_date: str = Field(default="2024-01-01", alias="BACKTEST_START_DATE")
    backtest_end_date: str = Field(default="2024-12-01", alias="BACKTEST_END_DATE")
    backtest_initial_capital: float = Field(default=100000.0, alias="BACKTEST_INITIAL_CAPITAL")


class AppConfig(BaseSettings):
    """Main application configuration - aggregates all configs"""
    
    delta: DeltaExchangeConfig = Field(default_factory=DeltaExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    data_ingestion: DataIngestionConfig = Field(default_factory=DataIngestionConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_config() -> AppConfig:
    """Get cached application configuration"""
    return AppConfig()


def reload_config() -> AppConfig:
    """Force reload configuration (clears cache)"""
    get_config.cache_clear()
    return get_config()


# Ensure data and log directories exist
def ensure_directories():
    """Create necessary directories"""
    directories = ["./data", "./logs", "./backtest_results"]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration loading
    ensure_directories()
    config = get_config()
    print(f"Delta Base URL: {config.delta.base_url}")
    print(f"Is Testnet: {config.delta.is_testnet}")
    print(f"Trading Mode: {config.trading.trading_mode}")
    print(f"Risk Per Trade: ₹{config.risk.risk_per_trade_inr}")
    print(f"Max Daily Loss: ₹{config.risk.max_daily_loss_inr}")
