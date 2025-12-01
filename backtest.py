"""
Backtest Engine for Delta Exchange Algo Trading Bot
Historical simulation and strategy validation
"""

import csv
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd

from config import get_config
from logger import get_logger
from strategy import (
    BaseStrategy, MultiModalStrategy, Signal, SignalDirection,
    MeanReversionStrategy, ScalpingStrategy
)
from data_ingest import MarketState, OrderbookSnapshot


@dataclass
class BacktestTrade:
    """Record of a backtest trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None


@dataclass
class BacktestResults:
    """Backtest results summary"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_duration: timedelta
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "total_return_pct": f"{self.total_return_pct:.2%}",
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": f"{self.win_rate:.2%}",
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": f"{self.max_drawdown_pct:.2%}",
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_trade_duration": str(self.avg_trade_duration)
        }
    
    def print_summary(self):
        """Print formatted results summary"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"Final Capital: ₹{self.final_capital:,.2f}")
        print("-"*60)
        print(f"Total Return: ₹{self.total_return:,.2f} ({self.total_return_pct:.2%})")
        print(f"Max Drawdown: ₹{self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2%})")
        print("-"*60)
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.win_rate:.2%}")
        print(f"Avg Win: ₹{self.avg_win:,.2f}")
        print(f"Avg Loss: ₹{self.avg_loss:,.2f}")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print("-"*60)
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {self.calmar_ratio:.2f}")
        print(f"Avg Trade Duration: {self.avg_trade_duration}")
        print("="*60)


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 100000
    risk_per_trade_pct: float = 0.01  # 1% risk per trade
    max_position_pct: float = 0.1  # 10% of capital max position
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    slippage_pct: float = 0.001  # 0.1% slippage
    commission_pct: float = 0.0005  # 0.05% commission
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.015


class HistoricalDataLoader:
    """Load historical price data for backtesting"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def load_csv(
        self,
        filepath: str,
        date_column: str = 'timestamp',
        price_column: str = 'close',
        volume_column: str = 'volume'
    ) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath, parse_dates=[date_column])
            df = df.sort_values(date_column)
            df = df.rename(columns={
                date_column: 'timestamp',
                price_column: 'close'
            })
            
            # Ensure required columns
            if 'open' not in df.columns:
                df['open'] = df['close']
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            if volume_column in df.columns:
                df = df.rename(columns={volume_column: 'volume'})
            else:
                df['volume'] = 0
            
            self.logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def generate_synthetic_data(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_price: float = 50000,
        volatility: float = 0.02,
        trend: float = 0.0001,
        interval_minutes: int = 5
    ) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(minutes=interval_minutes)
        
        n = len(timestamps)
        
        # Generate returns with trend and volatility
        returns = np.random.normal(trend, volatility, n)
        prices = initial_price * np.cumprod(1 + returns)
        
        # Add some noise for OHLC
        high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': np.roll(prices, 1),
            'high': high,
            'low': low,
            'close': prices,
            'volume': np.random.randint(100, 10000, n)
        })
        
        df.loc[0, 'open'] = initial_price
        
        self.logger.info(f"Generated {len(df)} synthetic data points")
        return df
    
    def add_sentiment_column(
        self,
        df: pd.DataFrame,
        correlation: float = 0.3,
        lag_periods: int = 2
    ) -> pd.DataFrame:
        """Add synthetic sentiment data correlated with price movements"""
        returns = df['close'].pct_change()
        
        # Create sentiment that's partially correlated with future returns
        noise = np.random.normal(0, 0.3, len(df))
        shifted_returns = returns.shift(-lag_periods).fillna(0)
        
        sentiment = correlation * (shifted_returns / shifted_returns.std()) + (1 - correlation) * noise
        sentiment = np.clip(sentiment, -1, 1)
        
        df['sentiment'] = sentiment
        return df


class BacktestEngine:
    """
    Backtest engine for strategy validation
    
    Simulates trading on historical data with realistic execution
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig = None
    ):
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.logger = get_logger()
        self.data_loader = HistoricalDataLoader()
        
        # State
        self._reset_state()
    
    def _reset_state(self):
        """Reset backtest state"""
        self.capital = self.config.initial_capital
        self.position = None  # Current position
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.peak_equity = self.config.initial_capital
        self.max_drawdown = 0
    
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResults:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            BacktestResults with performance metrics
        """
        self._reset_state()
        
        # Filter data by date
        if start_date:
            data = data[data['timestamp'] >= start_date]
        if end_date:
            data = data[data['timestamp'] <= end_date]
        
        if len(data) < 100:
            raise ValueError("Insufficient data for backtest (need at least 100 rows)")
        
        self.logger.info(f"Starting backtest with {len(data)} data points")
        
        # Build price buffer for strategy warm-up
        prices = []
        
        for idx, row in data.iterrows():
            timestamp = row['timestamp']
            price = row['close']
            prices.append(price)
            
            # Record equity
            current_equity = self._calculate_equity(price)
            self.equity_curve.append(current_equity)
            
            # Track drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            drawdown = self.peak_equity - current_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
            
            # Check stop loss / take profit for open position
            if self.position:
                exit_signal = self._check_exit(row)
                if exit_signal:
                    self._close_position(timestamp, price, exit_signal)
            
            # Skip if not enough data
            if len(prices) < 60:
                continue
            
            # Build market state
            market_state = self._build_market_state(
                timestamp=timestamp,
                prices=prices[-500:],  # Last 500 prices
                current_row=row
            )
            
            # Get strategy signal
            signal = self.strategy.evaluate(market_state)
            
            # Execute signal if no position
            if signal.is_actionable and not self.position:
                self._open_position(signal, timestamp, price)
        
        # Close any remaining position
        if self.position:
            final_price = data.iloc[-1]['close']
            self._close_position(data.iloc[-1]['timestamp'], final_price, "end_of_backtest")
        
        # Calculate results
        return self._calculate_results(data)
    
    def _build_market_state(
        self,
        timestamp: datetime,
        prices: List[float],
        current_row: pd.Series
    ) -> MarketState:
        """Build MarketState from historical data"""
        price = current_row['close']
        
        # Calculate EMAs
        price_series = pd.Series(prices)
        ema_fast = price_series.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_slow = price_series.ewm(span=50, adjust=False).mean().iloc[-1]
        
        # Calculate ATR
        if len(prices) > 14:
            returns = price_series.pct_change().abs()
            atr = returns.tail(14).mean() * price
        else:
            atr = price * 0.02
        
        # Get sentiment if available
        sentiment = current_row.get('sentiment', 0.0)
        
        # Create synthetic orderbook
        orderbook = OrderbookSnapshot(
            timestamp=timestamp,
            bids=[[price * 0.999, 100], [price * 0.998, 200]],
            asks=[[price * 1.001, 100], [price * 1.002, 200]]
        )
        
        return MarketState(
            timestamp=timestamp,
            price=price,
            prices=prices,
            orderbook=orderbook,
            sentiment_score=sentiment,
            sentiment_count=1,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            atr=atr
        )
    
    def _open_position(self, signal: Signal, timestamp: datetime, price: float):
        """Open a new position"""
        # Apply slippage
        slippage = price * self.config.slippage_pct
        if signal.direction == SignalDirection.LONG:
            entry_price = price + slippage
            side = 'long'
        else:
            entry_price = price - slippage
            side = 'short'
        
        # Calculate position size based on risk
        stop_distance = entry_price * self.config.stop_loss_pct
        risk_amount = self.capital * self.config.risk_per_trade_pct
        size = risk_amount / stop_distance
        
        # Cap at max position size
        max_size = (self.capital * self.config.max_position_pct) / entry_price
        size = min(size, max_size)
        
        # Apply commission
        commission = entry_price * size * self.config.commission_pct
        self.capital -= commission
        
        self.position = BacktestTrade(
            entry_time=timestamp,
            exit_time=None,
            side=side,
            entry_price=entry_price,
            exit_price=None,
            size=size
        )
        
        self.logger.debug(
            f"Opened {side} position",
            price=entry_price,
            size=size,
            timestamp=timestamp
        )
    
    def _check_exit(self, row: pd.Series) -> Optional[str]:
        """Check if position should be closed"""
        if not self.position:
            return None
        
        price = row['close']
        high = row.get('high', price)
        low = row.get('low', price)
        
        entry = self.position.entry_price
        
        if self.position.side == 'long':
            # Stop loss
            stop_price = entry * (1 - self.config.stop_loss_pct)
            if low <= stop_price:
                return 'stop_loss'
            
            # Take profit
            tp_price = entry * (1 + self.config.take_profit_pct)
            if high >= tp_price:
                return 'take_profit'
        else:
            # Stop loss for short
            stop_price = entry * (1 + self.config.stop_loss_pct)
            if high >= stop_price:
                return 'stop_loss'
            
            # Take profit for short
            tp_price = entry * (1 - self.config.take_profit_pct)
            if low <= tp_price:
                return 'take_profit'
        
        return None
    
    def _close_position(self, timestamp: datetime, price: float, reason: str):
        """Close current position"""
        if not self.position:
            return
        
        # Apply slippage
        slippage = price * self.config.slippage_pct
        if self.position.side == 'long':
            exit_price = price - slippage
        else:
            exit_price = price + slippage
        
        # Calculate PnL
        if self.position.side == 'long':
            pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.size
        
        # Apply commission
        commission = exit_price * self.position.size * self.config.commission_pct
        pnl -= commission
        
        # Update position
        self.position.exit_time = timestamp
        self.position.exit_price = exit_price
        self.position.pnl = pnl
        self.position.pnl_pct = pnl / (self.position.entry_price * self.position.size)
        self.position.exit_reason = reason
        
        # Update capital
        self.capital += pnl
        
        # Record trade
        self.trades.append(self.position)
        
        self.logger.debug(
            f"Closed {self.position.side} position",
            exit_price=exit_price,
            pnl=pnl,
            reason=reason
        )
        
        self.position = None
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open position"""
        equity = self.capital
        
        if self.position:
            if self.position.side == 'long':
                unrealized = (current_price - self.position.entry_price) * self.position.size
            else:
                unrealized = (self.position.entry_price - current_price) * self.position.size
            equity += unrealized
        
        return equity
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResults:
        """Calculate backtest results and metrics"""
        final_capital = self.capital
        total_return = final_capital - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital
        
        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.is_winner)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        max_drawdown_pct = self.max_drawdown / self.peak_equity if self.peak_equity > 0 else 0
        
        # Risk-adjusted returns
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Sharpe Ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino_ratio = 0
        
        # Calmar Ratio
        annual_return = total_return_pct * (252 / len(data)) if len(data) > 0 else 0
        calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Average trade duration
        durations = [t.duration for t in self.trades if t.duration]
        avg_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta()
        
        return BacktestResults(
            start_date=data.iloc[0]['timestamp'],
            end_date=data.iloc[-1]['timestamp'],
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_trade_duration=avg_duration,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
    
    def save_results(self, results: BacktestResults, filepath: str):
        """Save backtest results to file"""
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Save trades
        trades_file = filepath.replace('.json', '_trades.csv')
        with open(trades_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'entry_time', 'exit_time', 'side', 'entry_price',
                'exit_price', 'size', 'pnl', 'pnl_pct', 'exit_reason'
            ])
            for trade in results.trades:
                writer.writerow([
                    trade.entry_time, trade.exit_time, trade.side,
                    trade.entry_price, trade.exit_price, trade.size,
                    trade.pnl, trade.pnl_pct, trade.exit_reason
                ])
        
        # Save equity curve
        equity_file = filepath.replace('.json', '_equity.csv')
        pd.DataFrame({'equity': results.equity_curve}).to_csv(equity_file, index=False)
        
        self.logger.info(f"Results saved to {filepath}")


def run_backtest_example():
    """Example backtest run"""
    # Create strategy
    strategy = MultiModalStrategy(min_agreeing_signals=2)
    
    # Create backtest config
    config = BacktestConfig(
        initial_capital=100000,
        risk_per_trade_pct=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    # Create engine
    engine = BacktestEngine(strategy, config)
    
    # Generate synthetic data for testing
    loader = HistoricalDataLoader()
    data = loader.generate_synthetic_data(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 1),
        initial_price=50000,
        volatility=0.015,
        trend=0.0001
    )
    
    # Add sentiment
    data = loader.add_sentiment_column(data)
    
    # Run backtest
    results = engine.run(data)
    
    # Print results
    results.print_summary()
    
    # Save results
    engine.save_results(results, './backtest_results/backtest_results.json')
    
    return results


if __name__ == "__main__":
    results = run_backtest_example()
