"""
Structured Logging for Delta Exchange Algo Trading Bot
Provides consistent, queryable log output for production use
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict
import json
from functools import lru_cache

import structlog
from structlog.typing import EventDict


class TradingLogger:
    """
    Centralized logging for the trading bot
    Supports both human-readable and JSON formats
    """
    
    def __init__(
        self,
        name: str = "delta_algo_bot",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        json_logging: bool = False
    ):
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file
        self.json_logging = json_logging
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure structlog and standard logging"""
        
        # Ensure log directory exists
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure standard logging
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        handlers.append(console_handler)
        
        # File handler (if configured)
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.log_level)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            handlers=handlers,
            format="%(message)s"
        )
        
        # Configure structlog
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]
        
        if self.json_logging:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(self.log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(self.name)
    
    def bind(self, **kwargs) -> "TradingLogger":
        """Bind context variables to logger"""
        self.logger = self.logger.bind(**kwargs)
        return self
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)
    
    # Trading-specific log methods
    def log_order(
        self,
        action: str,
        order_id: Optional[str] = None,
        product_symbol: str = "",
        side: str = "",
        size: float = 0.0,
        price: Optional[float] = None,
        order_type: str = "market",
        status: str = "",
        **kwargs
    ):
        """Log order-related events"""
        self.logger.info(
            f"ORDER_{action.upper()}",
            order_id=order_id,
            product_symbol=product_symbol,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            status=status,
            **kwargs
        )
    
    def log_trade(
        self,
        action: str,
        trade_id: Optional[str] = None,
        product_symbol: str = "",
        side: str = "",
        size: float = 0.0,
        entry_price: float = 0.0,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        **kwargs
    ):
        """Log trade-related events"""
        self.logger.info(
            f"TRADE_{action.upper()}",
            trade_id=trade_id,
            product_symbol=product_symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            **kwargs
        )
    
    def log_signal(
        self,
        signal_type: str,
        direction: str,
        strength: float,
        price: float,
        indicators: Dict[str, Any],
        **kwargs
    ):
        """Log trading signals"""
        self.logger.info(
            f"SIGNAL_{signal_type.upper()}",
            direction=direction,
            strength=strength,
            price=price,
            indicators=indicators,
            **kwargs
        )
    
    def log_risk_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        **kwargs
    ):
        """Log risk management events"""
        level = logging.WARNING if "limit" in event_type.lower() else logging.INFO
        self.logger.log(
            level,
            f"RISK_{event_type.upper()}",
            details=details,
            **kwargs
        )
    
    def log_market_data(
        self,
        symbol: str,
        price: float,
        volume: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        **kwargs
    ):
        """Log market data updates (use sparingly - high volume)"""
        self.logger.debug(
            "MARKET_DATA",
            symbol=symbol,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            **kwargs
        )
    
    def log_sentiment(
        self,
        source: str,
        score: float,
        sample_size: int,
        **kwargs
    ):
        """Log sentiment analysis results"""
        self.logger.info(
            "SENTIMENT",
            source=source,
            score=score,
            sample_size=sample_size,
            **kwargs
        )
    
    def log_system_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        **kwargs
    ):
        """Log system events (startup, shutdown, errors)"""
        self.logger.info(
            f"SYSTEM_{event_type.upper()}",
            details=details,
            **kwargs
        )


class AuditLogger:
    """
    Immutable audit trail logger for compliance and debugging
    Writes to separate audit log file with checksums
    """
    
    def __init__(self, audit_file: str = "./data/audit.log"):
        self.audit_file = audit_file
        Path(audit_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Set up audit file handler
        self.handler = logging.FileHandler(audit_file, mode='a')
        self.handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.addHandler(self.handler)
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.propagate = False
    
    def log(self, event_type: str, data: Dict[str, Any]):
        """
        Write an immutable audit record
        Includes timestamp and event hash for integrity
        """
        import hashlib
        
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "data": data
        }
        
        # Create hash of the record for integrity verification
        record_str = json.dumps(record, sort_keys=True)
        record["hash"] = hashlib.sha256(record_str.encode()).hexdigest()[:16]
        
        self.audit_logger.info(json.dumps(record))
    
    def log_order_placed(self, order_data: Dict[str, Any]):
        """Audit log for order placement"""
        self.log("ORDER_PLACED", order_data)
    
    def log_order_filled(self, order_data: Dict[str, Any]):
        """Audit log for order fill"""
        self.log("ORDER_FILLED", order_data)
    
    def log_order_cancelled(self, order_data: Dict[str, Any]):
        """Audit log for order cancellation"""
        self.log("ORDER_CANCELLED", order_data)
    
    def log_position_opened(self, position_data: Dict[str, Any]):
        """Audit log for position opening"""
        self.log("POSITION_OPENED", position_data)
    
    def log_position_closed(self, position_data: Dict[str, Any]):
        """Audit log for position closing"""
        self.log("POSITION_CLOSED", position_data)
    
    def log_risk_violation(self, violation_data: Dict[str, Any]):
        """Audit log for risk violations"""
        self.log("RISK_VIOLATION", violation_data)
    
    def log_kill_switch(self, reason: str):
        """Audit log for kill switch activation"""
        self.log("KILL_SWITCH_ACTIVATED", {"reason": reason})
    
    def log_config_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Audit log for configuration changes"""
        self.log("CONFIG_CHANGE", {"old": old_config, "new": new_config})


# Global logger instances
_logger: Optional[TradingLogger] = None
_audit_logger: Optional[AuditLogger] = None


def get_logger() -> TradingLogger:
    """Get or create the global trading logger"""
    global _logger
    if _logger is None:
        from config import get_config
        config = get_config()
        _logger = TradingLogger(
            log_level=config.logging.log_level,
            log_file=config.logging.log_file,
            json_logging=config.logging.json_logging
        )
    return _logger


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_logging: bool = False
) -> TradingLogger:
    """
    Initialize logging with custom settings
    Call this early in application startup
    """
    global _logger
    _logger = TradingLogger(
        log_level=log_level,
        log_file=log_file,
        json_logging=json_logging
    )
    return _logger


if __name__ == "__main__":
    # Test logging
    logger = setup_logging(log_level="DEBUG")
    
    logger.info("Starting trading bot", version="1.0.0")
    logger.log_order(
        action="placed",
        order_id="12345",
        product_symbol="BTCUSD",
        side="buy",
        size=0.1,
        price=50000.0
    )
    logger.log_signal(
        signal_type="composite",
        direction="long",
        strength=0.75,
        price=50000.0,
        indicators={"ema_cross": True, "sentiment": 0.4, "ob_imbalance": 0.08}
    )
    logger.log_risk_event(
        event_type="daily_loss_check",
        details={"current_loss": -2500, "limit": -5000, "remaining": 2500}
    )
    
    # Test audit logger
    audit = get_audit_logger()
    audit.log_order_placed({
        "order_id": "12345",
        "symbol": "BTCUSD",
        "side": "buy",
        "size": 0.1,
        "price": 50000.0
    })
