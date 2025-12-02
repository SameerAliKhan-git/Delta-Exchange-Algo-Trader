"""
RISK ENGINE - Portfolio Risk Management
========================================
Controls risk exposure, position sizing, and protects capital.

Features:
- Dynamic position sizing
- Maximum drawdown protection
- Correlation-aware exposure
- Stop loss management
- Daily loss limits
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("AladdinAI.RiskEngine")


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_exposure: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    risk_reward_ratio: float = 0.0


class RiskEngine:
    """
    Portfolio risk management engine
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.metrics = RiskMetrics()
        self._trade_history: List[Dict] = []
        self._equity_curve: List[float] = []
        self._peak_equity: float = 0
        
        logger.info("Risk Engine initialized")
    
    def _default_config(self) -> Dict:
        return {
            "max_daily_loss_pct": 5.0,
            "max_drawdown_pct": 15.0,
            "risk_per_trade_pct": 2.0,
            "max_leverage": 10,
            "max_positions": 5,
            "max_correlation_exposure": 0.7,
            "min_risk_reward": 1.5,
        }
    
    def can_trade(self, portfolio) -> bool:
        """
        Check if trading is allowed based on risk limits
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            bool: Whether trading is allowed
        """
        # Check daily loss limit
        if portfolio.daily_pnl < 0:
            daily_loss_pct = abs(portfolio.daily_pnl) / portfolio.total_equity * 100
            if daily_loss_pct >= self.config["max_daily_loss_pct"]:
                logger.warning(f"Daily loss limit reached: {daily_loss_pct:.2f}%")
                return False
        
        # Check max drawdown
        if portfolio.max_drawdown >= self.config["max_drawdown_pct"]:
            logger.warning(f"Max drawdown reached: {portfolio.max_drawdown:.2f}%")
            return False
        
        # Check max positions
        if len(portfolio.open_positions) >= self.config["max_positions"]:
            logger.warning(f"Max positions reached: {len(portfolio.open_positions)}")
            return False
        
        return True
    
    def approve_signal(self, signal, portfolio) -> bool:
        """
        Approve or reject a trading signal based on risk assessment
        
        Args:
            signal: Trading signal to evaluate
            portfolio: Current portfolio state
        
        Returns:
            bool: Whether signal is approved
        """
        # Check risk/reward ratio
        entry = signal.entry_price
        stop = signal.stop_loss
        target = signal.take_profit
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if risk > 0:
            rr_ratio = reward / risk
            if rr_ratio < self.config["min_risk_reward"]:
                logger.warning(f"Risk/reward too low: {rr_ratio:.2f}")
                return False
        
        # Check position size vs max
        max_position_value = portfolio.total_equity * 0.1  # 10% max
        position_value = signal.position_size * entry
        
        if position_value > max_position_value:
            logger.warning(f"Position too large: ${position_value:.2f}")
            return False
        
        # Check correlation (if same direction positions exist)
        for symbol, pos in portfolio.open_positions.items():
            if pos["direction"] == signal.direction:
                # Same direction = increased risk
                if len(portfolio.open_positions) >= 2:
                    logger.warning("Too many same-direction positions")
                    return False
        
        # Check confidence threshold
        if signal.confidence < 0.5:
            logger.warning(f"Confidence too low: {signal.confidence:.2f}")
            return False
        
        return True
    
    def calculate_position_size(self, entry: float, stop_loss: float, 
                                portfolio_value: float) -> float:
        """
        Calculate optimal position size based on risk
        
        Uses Kelly Criterion modified for conservative risk management
        """
        risk_per_trade = portfolio_value * (self.config["risk_per_trade_pct"] / 100)
        risk_per_unit = abs(entry - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        # Base position size
        position_size = risk_per_trade / risk_per_unit
        
        # Apply leverage cap
        max_leverage_position = portfolio_value * self.config["max_leverage"] / entry
        position_size = min(position_size, max_leverage_position)
        
        # Apply win rate adjustment (Kelly-inspired)
        win_rate = self.metrics.win_rate or 0.5
        kelly_fraction = (win_rate - (1 - win_rate)) / 1  # Simplified
        kelly_fraction = max(0.1, min(0.5, kelly_fraction))  # Cap between 10-50%
        
        position_size *= kelly_fraction
        
        return position_size
    
    def update_metrics(self, portfolio, trade_result: Dict = None):
        """Update risk metrics after trade or periodically"""
        
        # Update equity curve
        self._equity_curve.append(portfolio.total_equity)
        
        # Update peak equity
        if portfolio.total_equity > self._peak_equity:
            self._peak_equity = portfolio.total_equity
        
        # Calculate drawdown
        if self._peak_equity > 0:
            self.metrics.current_drawdown = (
                (self._peak_equity - portfolio.total_equity) / self._peak_equity * 100
            )
            self.metrics.max_drawdown = max(
                self.metrics.max_drawdown, 
                self.metrics.current_drawdown
            )
        
        # Update trade statistics if trade completed
        if trade_result:
            self._trade_history.append(trade_result)
            self._calculate_trade_stats()
        
        # Update exposure metrics
        self.metrics.total_exposure = sum(
            pos.get("size", 0) * pos.get("entry_price", 0)
            for pos in portfolio.open_positions.values()
        )
    
    def _calculate_trade_stats(self):
        """Calculate trading statistics"""
        if not self._trade_history:
            return
        
        wins = [t for t in self._trade_history if t.get("pnl", 0) > 0]
        losses = [t for t in self._trade_history if t.get("pnl", 0) < 0]
        
        # Win rate
        self.metrics.win_rate = len(wins) / len(self._trade_history)
        
        # Average win/loss
        if wins:
            self.metrics.avg_win = sum(t["pnl"] for t in wins) / len(wins)
        if losses:
            self.metrics.avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))
        
        # Risk/Reward ratio
        if self.metrics.avg_loss > 0:
            self.metrics.risk_reward_ratio = self.metrics.avg_win / self.metrics.avg_loss
        
        # Sharpe ratio (simplified)
        if len(self._equity_curve) > 1:
            returns = [
                (self._equity_curve[i] - self._equity_curve[i-1]) / self._equity_curve[i-1]
                for i in range(1, len(self._equity_curve))
            ]
            if returns:
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                if std_return > 0:
                    self.metrics.sharpe_ratio = (avg_return * 252 ** 0.5) / std_return
    
    def get_risk_score(self) -> float:
        """
        Calculate overall risk score (0-100)
        Lower is better/safer
        """
        score = 0
        
        # Drawdown component (40% weight)
        dd_pct = self.metrics.current_drawdown / self.config["max_drawdown_pct"]
        score += min(40, dd_pct * 40)
        
        # Exposure component (30% weight)
        # Assuming max exposure is 5x portfolio
        exposure_pct = self.metrics.total_exposure / 100000  # Normalize
        score += min(30, exposure_pct * 30)
        
        # Trade frequency component (15% weight)
        recent_trades = [
            t for t in self._trade_history
            if datetime.fromisoformat(t.get("timestamp", "2000-01-01")) > 
               datetime.now() - timedelta(hours=24)
        ]
        trade_frequency = len(recent_trades) / 20  # Normalize to 20 trades/day
        score += min(15, trade_frequency * 15)
        
        # Win rate inverse (15% weight)
        wr_component = (1 - self.metrics.win_rate) * 15
        score += wr_component
        
        return min(100, score)
    
    def get_recommended_action(self) -> str:
        """Get recommended action based on risk score"""
        score = self.get_risk_score()
        
        if score < 20:
            return "AGGRESSIVE"  # Can take more risk
        elif score < 40:
            return "NORMAL"  # Standard trading
        elif score < 60:
            return "CAUTIOUS"  # Reduce position sizes
        elif score < 80:
            return "DEFENSIVE"  # Close some positions
        else:
            return "STOP_TRADING"  # Too risky
    
    def print_risk_report(self, portfolio):
        """Print formatted risk report"""
        self.update_metrics(portfolio)
        
        print("\n" + "=" * 60)
        print("RISK MANAGEMENT REPORT")
        print("=" * 60)
        print(f"\n  Portfolio Value: ${portfolio.total_equity:,.2f}")
        print(f"  Total Exposure: ${self.metrics.total_exposure:,.2f}")
        print(f"  Open Positions: {len(portfolio.open_positions)}")
        
        print(f"\n  Drawdown:")
        print(f"    Current: {self.metrics.current_drawdown:.2f}%")
        print(f"    Maximum: {self.metrics.max_drawdown:.2f}%")
        print(f"    Limit: {self.config['max_drawdown_pct']:.2f}%")
        
        print(f"\n  Performance:")
        print(f"    Win Rate: {self.metrics.win_rate:.1%}")
        print(f"    Avg Win: ${self.metrics.avg_win:,.2f}")
        print(f"    Avg Loss: ${self.metrics.avg_loss:,.2f}")
        print(f"    Risk/Reward: {self.metrics.risk_reward_ratio:.2f}")
        print(f"    Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}")
        
        print(f"\n  Risk Assessment:")
        print(f"    Risk Score: {self.get_risk_score():.1f}/100")
        print(f"    Recommendation: {self.get_recommended_action()}")
        
        print("=" * 60)
