"""
Backtesting Engine

Implements event-driven backtesting with:
- Realistic slippage and commission modeling
- Position sizing strategies
- Risk management integration
- Performance analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class Side(Enum):
    """Trade side enumeration."""
    LONG = 1
    SHORT = -1
    FLAT = 0


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Order object."""
    timestamp: datetime
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0


@dataclass
class Position:
    """Position object."""
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    side: Side
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    bars_held: int
    exit_reason: str = ""


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_pct: float = 0.20  # 20% max position
    max_positions: int = 5
    use_leverage: bool = False
    max_leverage: float = 1.0
    fill_probability: float = 1.0
    
    # Risk limits
    max_drawdown: float = 0.15  # 15% circuit breaker
    daily_loss_limit: float = 0.03  # 3% daily loss limit


@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    
    # Performance metrics
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_trade_pnl_pct: float = 0.0
    avg_bars_held: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_commission: float = 0.0
    
    def summary(self) -> str:
        """Return summary string."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                     BACKTEST RESULTS                         ║
╠══════════════════════════════════════════════════════════════╣
║  Total Return:      {self.total_return:>10.2%}                          ║
║  CAGR:              {self.cagr:>10.2%}                          ║
║  Sharpe Ratio:      {self.sharpe_ratio:>10.2f}                          ║
║  Sortino Ratio:     {self.sortino_ratio:>10.2f}                          ║
║  Max Drawdown:      {self.max_drawdown:>10.2%}                          ║
║  Calmar Ratio:      {self.calmar_ratio:>10.2f}                          ║
╠══════════════════════════════════════════════════════════════╣
║  Total Trades:      {self.total_trades:>10d}                          ║
║  Win Rate:          {self.win_rate:>10.2%}                          ║
║  Profit Factor:     {self.profit_factor:>10.2f}                          ║
║  Avg Trade P&L:     {self.avg_trade_pnl_pct:>10.2%}                          ║
║  Avg Bars Held:     {self.avg_bars_held:>10.1f}                          ║
║  Total Commission:  ${self.total_commission:>9.2f}                          ║
╚══════════════════════════════════════════════════════════════╝
"""


# ==============================================================================
# SLIPPAGE MODELS
# ==============================================================================

def fixed_slippage(price: float, slippage_pct: float, side: Side) -> float:
    """
    Fixed percentage slippage model.
    
    Args:
        price: Base price
        slippage_pct: Slippage percentage
        side: Trade side
        
    Returns:
        Adjusted price after slippage
    """
    if side == Side.LONG:
        return price * (1 + slippage_pct)
    elif side == Side.SHORT:
        return price * (1 - slippage_pct)
    return price


def volume_dependent_slippage(
    price: float,
    quantity: float,
    volume: float,
    impact_factor: float = 0.1,
    side: Side = Side.LONG
) -> float:
    """
    Volume-dependent slippage model.
    
    Slippage increases with trade size relative to volume.
    
    Args:
        price: Base price
        quantity: Trade quantity
        volume: Bar volume
        impact_factor: Impact multiplier
        side: Trade side
        
    Returns:
        Adjusted price
    """
    if volume == 0:
        return price
    
    participation = quantity / volume
    slippage = impact_factor * participation
    
    return fixed_slippage(price, slippage, side)


# ==============================================================================
# POSITION SIZING
# ==============================================================================

def fixed_fraction_size(
    capital: float,
    price: float,
    risk_pct: float = 0.02,
    stop_distance_pct: float = 0.01
) -> float:
    """
    Fixed fractional position sizing.
    
    Size = (Capital * Risk%) / (Price * Stop Distance%)
    
    Args:
        capital: Available capital
        price: Entry price
        risk_pct: Risk per trade as fraction
        stop_distance_pct: Stop loss distance as fraction
        
    Returns:
        Position size in units
    """
    risk_amount = capital * risk_pct
    position_size = risk_amount / (price * stop_distance_pct)
    return position_size


def kelly_criterion_size(
    capital: float,
    price: float,
    win_rate: float,
    win_loss_ratio: float,
    kelly_fraction: float = 0.25
) -> float:
    """
    Kelly Criterion position sizing.
    
    Kelly % = W - [(1-W) / R]
    Where W = win rate, R = win/loss ratio
    
    Args:
        capital: Available capital
        price: Entry price
        win_rate: Historical win rate
        win_loss_ratio: Average win / average loss
        kelly_fraction: Fraction of Kelly to use (default 1/4)
        
    Returns:
        Position size in units
    """
    if win_loss_ratio == 0:
        return 0
    
    kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
    kelly_pct = max(0, kelly_pct * kelly_fraction)
    
    position_value = capital * kelly_pct
    return position_value / price


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================

class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Example:
        engine = BacktestEngine(initial_capital=100000)
        results = engine.run(data, model, features)
    """
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        slippage_model: Optional[Callable] = None,
        sizing_model: Optional[Callable] = None
    ):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
            slippage_model: Custom slippage function
            sizing_model: Custom position sizing function
        """
        self.config = config or BacktestConfig()
        self.slippage_model = slippage_model or fixed_slippage
        self.sizing_model = sizing_model or fixed_fraction_size
        
        # State
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        # Risk tracking
        self.peak_equity = self.config.initial_capital
        self.daily_pnl = 0.0
        self.last_date = None
    
    def reset(self) -> None:
        """Reset engine state."""
        self.capital = self.config.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_history = []
        self.peak_equity = self.config.initial_capital
        self.daily_pnl = 0.0
        self.last_date = None
    
    def _get_equity(self) -> float:
        """Calculate total equity (capital + unrealized PnL)."""
        equity = self.capital
        for pos in self.positions.values():
            equity += pos.unrealized_pnl
        return equity
    
    def _update_positions(self, bar: pd.Series) -> None:
        """Update position values with current prices."""
        for symbol, pos in self.positions.items():
            if 'close' in bar.index:
                pos.current_price = bar['close']
                
                if pos.side == Side.LONG:
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.quantity
    
    def _check_stop_loss(self, symbol: str, bar: pd.Series) -> bool:
        """Check if stop loss was hit."""
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        
        if pos.side == Side.LONG:
            # Check if low touched stop loss
            stop_price = pos.entry_price * (1 - self.config.risk_per_trade)
            if bar['low'] <= stop_price:
                return True
        else:
            # Check if high touched stop loss
            stop_price = pos.entry_price * (1 + self.config.risk_per_trade)
            if bar['high'] >= stop_price:
                return True
        
        return False
    
    def _check_take_profit(self, symbol: str, bar: pd.Series, target_pct: float = 0.03) -> bool:
        """Check if take profit was hit."""
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        
        if pos.side == Side.LONG:
            target_price = pos.entry_price * (1 + target_pct)
            if bar['high'] >= target_price:
                return True
        else:
            target_price = pos.entry_price * (1 - target_pct)
            if bar['low'] <= target_price:
                return True
        
        return False
    
    def _calculate_fill_price(
        self,
        bar: pd.Series,
        side: Side,
        order_type: OrderType = OrderType.MARKET
    ) -> float:
        """Calculate fill price with slippage."""
        base_price = bar['open']  # Assume fill at open
        
        return self.slippage_model(
            base_price,
            self.config.slippage_pct,
            side
        )
    
    def _calculate_position_size(
        self,
        price: float,
        side: Side
    ) -> float:
        """Calculate position size based on sizing model."""
        equity = self._get_equity()
        max_position_value = equity * self.config.max_position_pct
        
        size = self.sizing_model(
            equity,
            price,
            self.config.risk_per_trade
        )
        
        # Limit to max position size
        size = min(size, max_position_value / price)
        
        return size
    
    def _open_position(
        self,
        symbol: str,
        timestamp: datetime,
        bar: pd.Series,
        side: Side,
        signal_strength: float = 1.0
    ) -> bool:
        """Open a new position."""
        # Check if already in position
        if symbol in self.positions:
            return False
        
        # Check position limit
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # Calculate fill price
        fill_price = self._calculate_fill_price(bar, side)
        
        # Calculate position size
        quantity = self._calculate_position_size(fill_price, side)
        quantity *= signal_strength  # Scale by confidence
        
        # Check minimum size
        if quantity * fill_price < 100:  # Minimum $100 position
            return False
        
        # Calculate commission
        commission = quantity * fill_price * self.config.commission_pct
        
        # Check if we have enough capital
        required = quantity * fill_price + commission
        if required > self.capital:
            return False
        
        # Create position
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp,
            current_price=fill_price
        )
        
        # Deduct capital (for cash account)
        self.capital -= required
        
        logger.debug(f"Opened {side.name} position: {symbol} @ {fill_price:.4f} x {quantity:.4f}")
        
        return True
    
    def _close_position(
        self,
        symbol: str,
        timestamp: datetime,
        bar: pd.Series,
        reason: str = ""
    ) -> Optional[Trade]:
        """Close an existing position."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Calculate fill price (opposite side)
        close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
        fill_price = self._calculate_fill_price(bar, close_side)
        
        # Calculate PnL
        if pos.side == Side.LONG:
            pnl = (fill_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - fill_price) * pos.quantity
        
        # Calculate commission
        commission = pos.quantity * fill_price * self.config.commission_pct
        pnl -= commission
        
        # Add commission from entry
        entry_commission = pos.quantity * pos.entry_price * self.config.commission_pct
        pnl -= entry_commission
        total_commission = commission + entry_commission
        
        # Calculate bars held
        # This is approximate - would need bar counting in actual implementation
        bars_held = 1
        
        # Calculate percentage PnL
        position_value = pos.quantity * pos.entry_price
        pnl_pct = pnl / position_value if position_value > 0 else 0
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=total_commission,
            bars_held=bars_held,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        
        # Return capital + PnL
        self.capital += position_value + pnl + total_commission
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(f"Closed {pos.side.name} position: {symbol} @ {fill_price:.4f}, PnL: {pnl:.2f}")
        
        return trade
    
    def _check_risk_limits(self, timestamp: datetime) -> bool:
        """Check if risk limits are breached."""
        equity = self._get_equity()
        
        # Check max drawdown
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity
        
        if drawdown >= self.config.max_drawdown:
            logger.warning(f"Max drawdown breached: {drawdown:.2%}")
            return True
        
        # Check daily loss limit
        current_date = timestamp.date() if hasattr(timestamp, 'date') else None
        if current_date != self.last_date:
            self.daily_pnl = 0.0
            self.last_date = current_date
        
        if self.daily_pnl / self.peak_equity <= -self.config.daily_loss_limit:
            logger.warning(f"Daily loss limit breached: {self.daily_pnl:.2f}")
            return True
        
        return False
    
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "ASSET"
    ) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            data: OHLCV DataFrame with datetime index
            signals: Series with signals (+1 long, -1 short, 0 flat)
            symbol: Asset symbol
            
        Returns:
            BacktestResult object
        """
        self.reset()
        
        logger.info(f"Running backtest on {len(data)} bars...")
        
        # Ensure aligned indices
        common_idx = data.index.intersection(signals.index)
        data = data.loc[common_idx]
        signals = signals.loc[common_idx]
        
        # Main loop
        for i, (timestamp, bar) in enumerate(data.iterrows()):
            signal = signals.loc[timestamp] if timestamp in signals.index else 0
            
            # Update positions
            self._update_positions(bar)
            
            # Record equity
            equity = self._get_equity()
            self.equity_history.append((timestamp, equity))
            
            # Check risk limits
            if self._check_risk_limits(timestamp):
                # Close all positions
                for sym in list(self.positions.keys()):
                    self._close_position(sym, timestamp, bar, "risk_limit")
                continue
            
            # Check stop loss / take profit
            if symbol in self.positions:
                if self._check_stop_loss(symbol, bar):
                    self._close_position(symbol, timestamp, bar, "stop_loss")
                elif self._check_take_profit(symbol, bar):
                    self._close_position(symbol, timestamp, bar, "take_profit")
            
            # Process signals
            if signal > 0 and symbol not in self.positions:
                self._open_position(symbol, timestamp, bar, Side.LONG, abs(signal))
            elif signal < 0 and symbol not in self.positions:
                self._open_position(symbol, timestamp, bar, Side.SHORT, abs(signal))
            elif signal == 0 and symbol in self.positions:
                self._close_position(symbol, timestamp, bar, "signal_flat")
        
        # Close any remaining positions
        last_bar = data.iloc[-1]
        last_timestamp = data.index[-1]
        for sym in list(self.positions.keys()):
            self._close_position(sym, last_timestamp, last_bar, "end_of_backtest")
        
        # Calculate results
        return self._calculate_results(data)
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate backtest performance metrics."""
        # Build equity curve
        equity_df = pd.DataFrame(
            self.equity_history,
            columns=['timestamp', 'equity']
        ).set_index('timestamp')
        
        equity_curve = equity_df['equity']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_curve.iloc[-1] / self.config.initial_capital) - 1
        
        # CAGR
        n_years = len(equity_curve) / (252 * 24)  # Assuming hourly data
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if n_years > 0 else 0
        
        # Sharpe ratio (assuming hourly returns, annualized)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)
        else:
            sharpe = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = returns.mean() / downside_returns.std() * np.sqrt(252 * 24)
        else:
            sortino = 0
        
        # Max drawdown
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Calmar ratio
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        total_trades = len(self.trades)
        if total_trades > 0:
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = len([t for t in self.trades if t.pnl <= 0])
            win_rate = winning_trades / total_trades
            
            gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_trade_pnl = sum(t.pnl for t in self.trades) / total_trades
            avg_trade_pnl_pct = sum(t.pnl_pct for t in self.trades) / total_trades
            avg_bars_held = sum(t.bars_held for t in self.trades) / total_trades
            total_commission = sum(t.commission for t in self.trades)
        else:
            winning_trades = losing_trades = 0
            win_rate = profit_factor = 0
            avg_trade_pnl = avg_trade_pnl_pct = avg_bars_held = 0
            total_commission = 0
        
        # Build positions DataFrame (simplified)
        positions_df = pd.DataFrame()
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            returns=returns,
            positions=positions_df,
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_trade_pnl_pct=avg_trade_pnl_pct,
            avg_bars_held=avg_bars_held,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_commission=total_commission
        )


# ==============================================================================
# WALK-FORWARD OPTIMIZATION
# ==============================================================================

def walk_forward_optimization(
    data: pd.DataFrame,
    model_trainer: Any,
    feature_pipeline: Any,
    labeler: Any,
    n_splits: int = 6,
    train_pct: float = 0.6,
    optimize_trials: int = 20
) -> List[BacktestResult]:
    """
    Walk-forward analysis with rolling optimization.
    
    Args:
        data: Full OHLCV dataset
        model_trainer: ModelTrainer instance
        feature_pipeline: FeaturePipeline instance
        labeler: Labeler instance
        n_splits: Number of walk-forward periods
        train_pct: Training period percentage
        optimize_trials: Hyperparameter optimization trials
        
    Returns:
        List of BacktestResult for each period
    """
    results = []
    
    split_size = len(data) // n_splits
    
    for i in range(1, n_splits):
        logger.info(f"Walk-forward period {i}/{n_splits - 1}")
        
        # Training period
        train_end_idx = i * split_size
        train_start_idx = int(train_end_idx * (1 - train_pct))
        train_data = data.iloc[train_start_idx:train_end_idx]
        
        # Test period
        test_start_idx = train_end_idx
        test_end_idx = min((i + 1) * split_size, len(data))
        test_data = data.iloc[test_start_idx:test_end_idx]
        
        logger.info(f"Training: {len(train_data)} bars, Testing: {len(test_data)} bars")
        
        # Generate features and labels for training
        train_features = feature_pipeline.fit_transform(train_data)
        train_labels = labeler.fit_transform(train_data['close'])
        
        # Train model
        training_result = model_trainer.train(
            train_features.dropna(),
            train_labels.dropna(),
            optimize=optimize_trials > 0,
            n_trials=optimize_trials
        )
        
        # Generate features for testing
        test_features = feature_pipeline.fit_transform(test_data)
        
        # Generate signals
        proba = model_trainer.predict_proba(test_features.dropna())[:, 1]
        signals = pd.Series(
            np.where(proba > 0.6, 1, np.where(proba < 0.4, -1, 0)),
            index=test_features.dropna().index
        )
        
        # Backtest
        engine = BacktestEngine()
        result = engine.run(test_data.loc[signals.index], signals)
        
        results.append(result)
        logger.info(f"Period {i} result: {result.total_return:.2%}")
    
    return results


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("BACKTEST ENGINE DEMO")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range('2020-01-01', periods=n, freq='1h')
    close = 100 * np.cumprod(1 + np.random.normal(0.0002, 0.01, n))
    high = close * (1 + np.random.uniform(0, 0.01, n))
    low = close * (1 - np.random.uniform(0, 0.01, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.exponential(1000000, n)
    
    data = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Create random signals
    signals = pd.Series(
        np.random.choice([-1, 0, 0, 0, 1], size=n),
        index=dates
    )
    
    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission_pct=0.001,
        slippage_pct=0.0005
    )
    
    engine = BacktestEngine(config)
    result = engine.run(data, signals)
    
    print(result.summary())
