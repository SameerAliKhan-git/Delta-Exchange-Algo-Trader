"""
Alert Manager - Trading alerts and notifications

Provides:
- Multi-channel alerting (Slack, Telegram, Email)
- Alert severity levels
- Rate limiting and deduplication
- Alert templates
"""

import logging
import json
import time
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

logger = logging.getLogger("Aladdin.Alerts")


class AlertLevel(Enum):
    """Alert severity levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class Alert:
    """Alert definition"""
    level: AlertLevel
    title: str
    message: str
    source: str = "Aladdin"
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict = field(default_factory=dict)
    dedupe_key: str = ""
    
    def __post_init__(self):
        if not self.dedupe_key:
            # Generate deduplication key from title and source
            self.dedupe_key = hashlib.md5(
                f"{self.source}:{self.title}".encode()
            ).hexdigest()[:12]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'level': self.level.name,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'dedupe_key': self.dedupe_key
        }
    
    def format_slack(self) -> Dict:
        """Format for Slack webhook"""
        color_map = {
            AlertLevel.DEBUG: "#808080",
            AlertLevel.INFO: "#2196F3",
            AlertLevel.WARNING: "#FF9800",
            AlertLevel.ERROR: "#F44336",
            AlertLevel.CRITICAL: "#B71C1C"
        }
        
        emoji_map = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "🚨",
            AlertLevel.CRITICAL: "🔴"
        }
        
        return {
            "attachments": [{
                "color": color_map[self.level],
                "title": f"{emoji_map[self.level]} {self.title}",
                "text": self.message,
                "fields": [
                    {"title": "Source", "value": self.source, "short": True},
                    {"title": "Level", "value": self.level.name, "short": True},
                    {"title": "Time", "value": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                ] + [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in list(self.data.items())[:4]
                ],
                "footer": f"Aladdin Trading Bot | {self.dedupe_key}"
            }]
        }
    
    def format_telegram(self) -> str:
        """Format for Telegram"""
        emoji_map = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "🚨",
            AlertLevel.CRITICAL: "🔴"
        }
        
        lines = [
            f"{emoji_map[self.level]} *{self.title}*",
            f"_{self.level.name}_",
            "",
            self.message,
            "",
            f"📍 Source: {self.source}",
            f"🕐 Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        if self.data:
            lines.append("")
            for k, v in list(self.data.items())[:5]:
                lines.append(f"• {k}: {v}")
        
        return "\n".join(lines)


class SlackNotifier:
    """Slack webhook notifier"""
    
    def __init__(self, webhook_url: str, channel: str = None):
        """
        Initialize Slack notifier
        
        Args:
            webhook_url: Slack webhook URL
            channel: Optional channel override
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self._enabled = True
    
    def send(self, alert: Alert) -> bool:
        """
        Send alert to Slack
        
        Args:
            alert: Alert to send
        
        Returns:
            True if successful
        """
        if not self._enabled or not self.webhook_url:
            return False
        
        try:
            payload = alert.format_slack()
            if self.channel:
                payload['channel'] = self.channel
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"Slack alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Slack error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False
    
    def enable(self) -> None:
        """Enable notifier"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable notifier"""
        self._enabled = False


class TelegramNotifier:
    """Telegram bot notifier"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token
            chat_id: Target chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = True
        self._base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send(self, alert: Alert) -> bool:
        """
        Send alert to Telegram
        
        Args:
            alert: Alert to send
        
        Returns:
            True if successful
        """
        if not self._enabled or not self.bot_token or not self.chat_id:
            return False
        
        try:
            message = alert.format_telegram()
            
            response = requests.post(
                f"{self._base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"Telegram alert sent: {alert.title}")
                return True
            else:
                logger.error(f"Telegram error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")
            return False
    
    def enable(self) -> None:
        """Enable notifier"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable notifier"""
        self._enabled = False


class EmailNotifier:
    """Email notifier via SMTP"""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True
    ):
        """
        Initialize email notifier
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_addr: From email address
            to_addrs: List of recipient addresses
            use_tls: Use TLS encryption
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls
        self._enabled = True
    
    def send(self, alert: Alert) -> bool:
        """
        Send alert via email
        
        Args:
            alert: Alert to send
        
        Returns:
            True if successful
        """
        if not self._enabled:
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = f"[{alert.level.name}] {alert.title}"
            
            # Email body
            body = f"""
Aladdin Trading Alert

Level: {alert.level.name}
Title: {alert.title}
Source: {alert.source}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}

Additional Data:
{json.dumps(alert.data, indent=2)}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.debug(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    def enable(self) -> None:
        """Enable notifier"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable notifier"""
        self._enabled = False


class AlertManager:
    """
    Central alert management system
    
    Features:
    - Multiple notification channels
    - Rate limiting
    - Deduplication
    - Alert history
    """
    
    def __init__(
        self,
        min_level: AlertLevel = AlertLevel.INFO,
        rate_limit_seconds: int = 60,
        dedupe_window_minutes: int = 30
    ):
        """
        Initialize alert manager
        
        Args:
            min_level: Minimum alert level to process
            rate_limit_seconds: Minimum seconds between same alerts
            dedupe_window_minutes: Window for deduplication
        """
        self.min_level = min_level
        self.rate_limit_seconds = rate_limit_seconds
        self.dedupe_window = timedelta(minutes=dedupe_window_minutes)
        
        self._notifiers: List = []
        self._history: List[Alert] = []
        self._last_sent: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._enabled = True
        
        logger.info("Alert manager initialized")
    
    def add_notifier(self, notifier) -> None:
        """Add notification channel"""
        self._notifiers.append(notifier)
        logger.info(f"Added notifier: {type(notifier).__name__}")
    
    def alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str = "Aladdin",
        data: Dict = None,
        force: bool = False
    ) -> bool:
        """
        Send alert
        
        Args:
            level: Alert severity
            title: Alert title
            message: Alert message
            source: Source component
            data: Additional data
            force: Force send (bypass rate limiting)
        
        Returns:
            True if alert was sent
        """
        if not self._enabled:
            return False
        
        if level.value < self.min_level.value:
            return False
        
        alert = Alert(
            level=level,
            title=title,
            message=message,
            source=source,
            data=data or {}
        )
        
        with self._lock:
            # Check rate limiting
            if not force and alert.dedupe_key in self._last_sent:
                elapsed = datetime.now() - self._last_sent[alert.dedupe_key]
                if elapsed.total_seconds() < self.rate_limit_seconds:
                    logger.debug(f"Alert rate limited: {title}")
                    return False
            
            # Update tracking
            self._last_sent[alert.dedupe_key] = datetime.now()
            self._history.append(alert)
            
            # Trim history
            if len(self._history) > 1000:
                self._history = self._history[-500:]
        
        # Send to all notifiers
        success = False
        for notifier in self._notifiers:
            try:
                if notifier.send(alert):
                    success = True
            except Exception as e:
                logger.error(f"Notifier error: {e}")
        
        logger.log(
            level.value * 10 + 10,  # Map to logging levels
            f"[{level.name}] {title}: {message}"
        )
        
        return success
    
    # Convenience methods
    def debug(self, title: str, message: str, **kwargs) -> bool:
        return self.alert(AlertLevel.DEBUG, title, message, **kwargs)
    
    def info(self, title: str, message: str, **kwargs) -> bool:
        return self.alert(AlertLevel.INFO, title, message, **kwargs)
    
    def warning(self, title: str, message: str, **kwargs) -> bool:
        return self.alert(AlertLevel.WARNING, title, message, **kwargs)
    
    def error(self, title: str, message: str, **kwargs) -> bool:
        return self.alert(AlertLevel.ERROR, title, message, **kwargs)
    
    def critical(self, title: str, message: str, **kwargs) -> bool:
        return self.alert(AlertLevel.CRITICAL, title, message, **kwargs)
    
    # Trading-specific alerts
    def trade_opened(self, symbol: str, side: str, size: float, price: float) -> bool:
        """Alert for new trade"""
        return self.info(
            "Trade Opened",
            f"{side.upper()} {size} {symbol} @ ${price:,.2f}",
            source="OrderManager",
            data={'symbol': symbol, 'side': side, 'size': size, 'price': price}
        )
    
    def trade_closed(self, symbol: str, pnl: float, pnl_pct: float) -> bool:
        """Alert for closed trade"""
        emoji = "✅" if pnl > 0 else "❌"
        level = AlertLevel.INFO if pnl > 0 else AlertLevel.WARNING
        return self.alert(
            level,
            f"Trade Closed {emoji}",
            f"{symbol}: ${pnl:+,.2f} ({pnl_pct:+.2f}%)",
            source="OrderManager",
            data={'symbol': symbol, 'pnl': pnl, 'pnl_pct': pnl_pct}
        )
    
    def risk_breach(self, rule: str, value: float, limit: float) -> bool:
        """Alert for risk limit breach"""
        return self.critical(
            "Risk Limit Breached",
            f"{rule}: Current {value:.2f} > Limit {limit:.2f}",
            source="RiskDesk",
            data={'rule': rule, 'value': value, 'limit': limit},
            force=True
        )
    
    def kill_switch_activated(self, reason: str) -> bool:
        """Alert for kill switch activation"""
        return self.critical(
            "🛑 KILL SWITCH ACTIVATED",
            f"All trading halted: {reason}",
            source="RiskDesk",
            force=True
        )
    
    def daily_summary(self, pnl: float, trades: int, win_rate: float) -> bool:
        """Daily performance summary"""
        emoji = "📈" if pnl > 0 else "📉"
        return self.info(
            f"Daily Summary {emoji}",
            f"P&L: ${pnl:+,.2f} | Trades: {trades} | Win Rate: {win_rate:.1f}%",
            source="Analytics",
            data={'pnl': pnl, 'trades': trades, 'win_rate': win_rate}
        )
    
    def get_history(
        self,
        level: AlertLevel = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get alert history
        
        Args:
            level: Filter by level
            since: Filter by time
            limit: Maximum results
        
        Returns:
            List of alerts
        """
        with self._lock:
            alerts = self._history.copy()
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts[-limit:]
    
    def enable(self) -> None:
        """Enable alert manager"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable alert manager"""
        self._enabled = False
    
    def clear_rate_limits(self) -> None:
        """Clear rate limiting cache"""
        with self._lock:
            self._last_sent.clear()


# Alert templates
class AlertTemplates:
    """Pre-defined alert templates"""
    
    @staticmethod
    def signal_generated(
        symbol: str,
        direction: str,
        confidence: float,
        strategy: str
    ) -> Alert:
        return Alert(
            level=AlertLevel.INFO,
            title="Signal Generated",
            message=f"{direction.upper()} {symbol} (Confidence: {confidence:.1%})",
            source=strategy,
            data={
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'strategy': strategy
            }
        )
    
    @staticmethod
    def margin_warning(
        current_margin: float,
        required_margin: float,
        utilization_pct: float
    ) -> Alert:
        return Alert(
            level=AlertLevel.WARNING,
            title="Margin Warning",
            message=f"Margin utilization at {utilization_pct:.1f}%",
            source="RiskDesk",
            data={
                'current_margin': current_margin,
                'required_margin': required_margin,
                'utilization': utilization_pct
            }
        )
    
    @staticmethod
    def system_error(error: str, component: str) -> Alert:
        return Alert(
            level=AlertLevel.ERROR,
            title="System Error",
            message=error,
            source=component
        )


if __name__ == "__main__":
    # Test alert manager
    logging.basicConfig(level=logging.DEBUG)
    
    manager = AlertManager(min_level=AlertLevel.DEBUG)
    
    # Note: These would need actual credentials to work
    # manager.add_notifier(SlackNotifier("https://hooks.slack.com/..."))
    # manager.add_notifier(TelegramNotifier("bot_token", "chat_id"))
    
    print("=" * 60)
    print("ALERT MANAGER TEST")
    print("=" * 60)
    
    # Send test alerts
    manager.debug("Debug Test", "This is a debug message")
    manager.info("Info Test", "System started successfully")
    manager.warning("Warning Test", "High CPU usage detected", data={'cpu': 85})
    manager.error("Error Test", "Failed to connect to API")
    manager.critical("Critical Test", "Database connection lost", force=True)
    
    # Trading alerts
    manager.trade_opened("BTCUSD", "buy", 0.1, 100000)
    manager.trade_closed("BTCUSD", 150.00, 1.5)
    manager.risk_breach("Max Drawdown", 12.5, 10.0)
    
    # Test rate limiting
    print("\nTesting rate limiting:")
    result1 = manager.info("Same Alert", "First call")
    print(f"First call: {'sent' if result1 else 'blocked'}")
    result2 = manager.info("Same Alert", "Second call (should be blocked)")
    print(f"Second call: {'sent' if result2 else 'blocked'}")
    
    # History
    print(f"\nAlert history: {len(manager.get_history())} alerts")
    for alert in manager.get_history(limit=5):
        print(f"  [{alert.level.name}] {alert.title}")
    
    print("\n✅ Alert manager test complete!")
