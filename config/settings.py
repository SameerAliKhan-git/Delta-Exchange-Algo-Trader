"""
Settings - Application configuration with Pydantic

Loads configuration from environment variables and .env file.
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class APIConfig:
    """Delta Exchange API configuration"""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    timeout: int = 30
    max_retries: int = 3
    
    @property
    def base_url(self) -> str:
        if self.testnet:
            return "https://cdn-ind.testnet.deltaex.org"
        return "https://api.india.delta.exchange"


@dataclass
class TradingConfig:
    """Trading configuration"""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSD"])
    default_timeframe: str = "1h"
    default_leverage: int = 3
    max_leverage: int = 10
    
    # Order settings
    default_order_type: str = "market"
    use_reduce_only: bool = True
    
    # Strategy settings
    strategy: str = "momentum"
    signal_threshold: float = 0.6


@dataclass  
class RiskConfig:
    """Risk management configuration"""
    max_capital_per_trade_pct: float = 0.02
    max_total_exposure_pct: float = 0.3
    max_positions: int = 5
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    
    # ATR-based sizing
    atr_risk_multiple: float = 2.0
    
    # Options
    max_premium_pct: float = 0.005
    
    # Kill switch
    cooldown_minutes: int = 60


@dataclass
class DataConfig:
    """Data ingestion configuration"""
    cache_enabled: bool = True
    cache_dir: str = "./data_cache"
    
    # Sentiment sources
    use_twitter: bool = False
    use_reddit: bool = False
    use_news: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    
    http_enabled: bool = True
    http_port: int = 8080
    
    log_level: str = "INFO"
    log_file: str = "trader.log"


@dataclass
class Settings:
    """Complete application settings"""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def get_settings() -> Settings:
    """
    Load settings from environment variables
    
    Returns:
        Settings instance
    """
    # Try to load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    settings = Settings(
        api=APIConfig(
            api_key=os.getenv("DELTA_API_KEY", ""),
            api_secret=os.getenv("DELTA_API_SECRET", ""),
            testnet=os.getenv("DELTA_TESTNET", "true").lower() == "true",
            timeout=int(os.getenv("API_TIMEOUT", "30")),
        ),
        trading=TradingConfig(
            symbols=os.getenv("TRADING_SYMBOLS", "BTCUSD").split(","),
            default_timeframe=os.getenv("DEFAULT_TIMEFRAME", "1h"),
            default_leverage=int(os.getenv("DEFAULT_LEVERAGE", "3")),
            max_leverage=int(os.getenv("MAX_LEVERAGE", "10")),
            strategy=os.getenv("STRATEGY", "momentum"),
            signal_threshold=float(os.getenv("SIGNAL_THRESHOLD", "0.6")),
        ),
        risk=RiskConfig(
            max_capital_per_trade_pct=float(os.getenv("MAX_CAPITAL_PER_TRADE_PCT", "0.02")),
            max_total_exposure_pct=float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.3")),
            max_positions=int(os.getenv("MAX_POSITIONS", "5")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.15")),
            cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "60")),
        ),
        data=DataConfig(
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            cache_dir=os.getenv("CACHE_DIR", "./data_cache"),
        ),
        monitoring=MonitoringConfig(
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8000")),
            http_enabled=os.getenv("HTTP_ENABLED", "true").lower() == "true",
            http_port=int(os.getenv("HTTP_PORT", "8080")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        ),
    )
    
    return settings


# Convenience function for quick access
def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_settings().api


def get_trading_config() -> TradingConfig:
    """Get trading configuration"""
    return get_settings().trading


def get_risk_config() -> RiskConfig:
    """Get risk configuration"""
    return get_settings().risk
