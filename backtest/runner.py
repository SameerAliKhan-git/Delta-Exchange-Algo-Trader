"""
Backtest Runner - Execute backtests on historical data

Provides:
- Event-driven backtesting engine
- Strategy execution simulation
- Position and P&L tracking
"""

import time
import numpy as np
from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from strategies.base import StrategyBase, StrategyState
from data.loader import CandleData
from .metrics import Trade, PerformanceMetrics, calculate_metrics


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 10000.0
    commission_rate: float = 0.0006  # 0.06% taker fee
    slippage_pct: float = 0.0005  # 0.05% slippage
    leverage: int = 1
    
    # Risk settings
    max_position_size: float = 1.0
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2%
    use_take_profit: bool = True
    take_profit_pct: float = 0.04  # 4%
    
    # Data settings
    warmup_periods: int = 100


@dataclass
class BacktestResult:
    """Backtest result"""
    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    
    # Summary
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    
    def summary(self) -> str:
        """Generate summary string"""
        return f"""
Backtest Results
================
Period: {datetime.fromtimestamp(self.timestamps[0])} to {datetime.fromtimestamp(self.timestamps[-1])}
Initial Capital: ${self.config.initial_capital:,.2f}
Final Equity: ${self.final_equity:,.2f}
Total Return: {self.total_return_pct:.2f}%

Performance Metrics
-------------------
Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}
Max Drawdown: {self.metrics.max_drawdown_pct:.2f}%
Win Rate: {self.metrics.win_rate:.1f}%
Profit Factor: {self.metrics.profit_factor:.2f}

Trades
------
Total Trades: {self.metrics.total_trades}
Winning: {self.metrics.winning_trades}
Losing: {self.metrics.losing_trades}
Avg Win: ${self.metrics.avg_win:.2f}
Avg Loss: ${self.metrics.avg_loss:.2f}
"""


class BacktestRunner:
    """
    Event-driven backtest engine
    """
    
    def __init__(
        self,
        strategy: StrategyBase,
        config: BacktestConfig = None
    ):
        """
        Initialize backtest runner
        
        Args:
            strategy: Strategy instance to test
            config: Backtest configuration
        """
        self.strategy = strategy
        self.config = config or BacktestConfig()
        
        # State
        self.equity = self.config.initial_capital
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.entry_time = 0.0
        
        # Tracking
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[float] = []
        
        # Stop/Take profit tracking
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
    
    def run(self, data: CandleData) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: Historical OHLCV data
        
        Returns:
            BacktestResult
        """
        if len(data) < self.config.warmup_periods:
            raise ValueError(f"Insufficient data: need {self.config.warmup_periods} periods, got {len(data)}")
        
        # Reset state
        self._reset()
        
        # Initialize strategy
        self.strategy.on_start()
        
        # Run through data
        for i in range(self.config.warmup_periods, len(data)):
            # Extract candle
            timestamp = data.timestamps[i]
            open_price = data.opens[i]
            high = data.highs[i]
            low = data.lows[i]
            close = data.closes[i]
            volume = data.volumes[i]
            
            # Build Candle object for strategy
            from strategies.base import Candle
            from datetime import datetime
            candle = Candle(
                timestamp=datetime.fromtimestamp(timestamp),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            )
            
            # Check stops first (using high/low of candle)
            self._check_stops(high, low, timestamp)
            
            # Call strategy
            signal = self.strategy.on_candle(candle)
            
            # Process signal
            if signal is not None:
                self._process_signal(signal, close, timestamp)
            
            # Update equity with unrealized P&L
            unrealized = self._calculate_unrealized_pnl(close)
            self.equity_curve.append(self.equity + unrealized)
            self.timestamps.append(timestamp)
        
        # Close any open position at end
        if self.position != 0:
            self._close_position(data.closes[-1], data.timestamps[-1], "end_of_backtest")
        
        # Call strategy end
        self.strategy.on_exit()
        
        # Calculate metrics
        equity_array = np.array(self.equity_curve)
        timestamps_array = np.array(self.timestamps)
        metrics = calculate_metrics(self.trades, equity_array, self.config.initial_capital)
        
        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            equity_curve=equity_array,
            timestamps=timestamps_array,
            final_equity=self.equity,
            total_return_pct=((self.equity - self.config.initial_capital) / self.config.initial_capital) * 100
        )
    
    def _reset(self) -> None:
        """Reset backtest state"""
        self.equity = self.config.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
    
    def _process_signal(self, signal: int, price: float, timestamp: float) -> None:
        """Process trading signal"""
        if signal == 0:
            return
        
        # Apply slippage
        if signal > 0:
            exec_price = price * (1 + self.config.slippage_pct)
        else:
            exec_price = price * (1 - self.config.slippage_pct)
        
        # Check if closing existing position
        if self.position != 0:
            if (self.position > 0 and signal < 0) or (self.position < 0 and signal > 0):
                self._close_position(exec_price, timestamp, "signal")
        
        # Open new position
        if self.position == 0 and signal != 0:
            self._open_position(signal, exec_price, timestamp)
    
    def _open_position(self, direction: int, price: float, timestamp: float) -> None:
        """Open new position"""
        # Calculate position size (simple: use max allowed)
        size = self.config.max_position_size * direction
        
        # Apply commission
        commission = abs(size) * price * self.config.commission_rate
        self.equity -= commission
        
        self.position = size
        self.entry_price = price
        self.entry_time = timestamp
        
        # Set stops
        if self.config.use_stop_loss:
            if direction > 0:
                self.stop_loss_price = price * (1 - self.config.stop_loss_pct)
            else:
                self.stop_loss_price = price * (1 + self.config.stop_loss_pct)
        
        if self.config.use_take_profit:
            if direction > 0:
                self.take_profit_price = price * (1 + self.config.take_profit_pct)
            else:
                self.take_profit_price = price * (1 - self.config.take_profit_pct)
    
    def _close_position(self, price: float, timestamp: float, reason: str) -> None:
        """Close position and record trade"""
        if self.position == 0:
            return
        
        # Calculate P&L
        if self.position > 0:
            pnl = (price - self.entry_price) * abs(self.position)
        else:
            pnl = (self.entry_price - price) * abs(self.position)
        
        # Apply commission
        commission = abs(self.position) * price * self.config.commission_rate
        pnl -= commission
        
        # Record trade
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=timestamp,
            entry_price=self.entry_price,
            exit_price=price,
            size=abs(self.position),
            side="long" if self.position > 0 else "short",
            pnl=pnl,
            pnl_pct=(pnl / (self.entry_price * abs(self.position))) * 100,
            exit_reason=reason
        )
        self.trades.append(trade)
        
        # Update equity
        self.equity += pnl
        
        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
    
    def _check_stops(self, high: float, low: float, timestamp: float) -> None:
        """Check stop loss and take profit"""
        if self.position == 0:
            return
        
        # Long position
        if self.position > 0:
            if self.stop_loss_price > 0 and low <= self.stop_loss_price:
                self._close_position(self.stop_loss_price, timestamp, "stop_loss")
            elif self.take_profit_price > 0 and high >= self.take_profit_price:
                self._close_position(self.take_profit_price, timestamp, "take_profit")
        
        # Short position
        elif self.position < 0:
            if self.stop_loss_price > 0 and high >= self.stop_loss_price:
                self._close_position(self.stop_loss_price, timestamp, "stop_loss")
            elif self.take_profit_price > 0 and low <= self.take_profit_price:
                self._close_position(self.take_profit_price, timestamp, "take_profit")
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.position == 0:
            return 0.0
        
        if self.position > 0:
            return (current_price - self.entry_price) * abs(self.position)
        else:
            return (self.entry_price - current_price) * abs(self.position)


def run_backtest(
    strategy_class: Type[StrategyBase],
    data: CandleData,
    config: BacktestConfig = None,
    strategy_params: Dict = None
) -> BacktestResult:
    """
    Convenience function to run backtest
    
    Args:
        strategy_class: Strategy class to instantiate
        data: Historical data
        config: Backtest configuration
        strategy_params: Parameters to pass to strategy
    
    Returns:
        BacktestResult
    """
    # Instantiate strategy
    params = strategy_params or {}
    strategy = strategy_class(**params)
    
    # Create runner and execute
    runner = BacktestRunner(strategy, config)
    return runner.run(data)
