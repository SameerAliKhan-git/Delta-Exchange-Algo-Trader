"""
Performance Metrics - Calculate trading performance metrics

Provides:
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate, profit factor
- Trade statistics
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    """Individual trade record"""
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    size: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    exit_reason: str = ""
    
    @property
    def duration_hours(self) -> float:
        return (self.exit_time - self.entry_time) / 3600
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class PerformanceMetrics:
    """Complete performance metrics"""
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration_days: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0
    
    # Trade duration
    avg_trade_duration_hours: float = 0.0
    avg_winning_duration_hours: float = 0.0
    avg_losing_duration_hours: float = 0.0
    
    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


def calculate_metrics(
    trades: List[Trade],
    equity_curve: np.ndarray,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics
    
    Args:
        trades: List of completed trades
        equity_curve: Equity values over time
        initial_capital: Starting capital
        risk_free_rate: Risk-free rate for Sharpe calculation
        periods_per_year: Trading periods per year (252 for daily)
    
    Returns:
        PerformanceMetrics
    """
    metrics = PerformanceMetrics()
    
    if len(equity_curve) == 0:
        return metrics
    
    # Calculate returns
    final_equity = equity_curve[-1]
    metrics.total_return = final_equity - initial_capital
    metrics.total_return_pct = (metrics.total_return / initial_capital) * 100
    
    # Calculate daily returns
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    if len(returns) > 0:
        # Annualized return
        total_periods = len(returns)
        metrics.annualized_return = ((final_equity / initial_capital) ** (periods_per_year / total_periods) - 1) * 100
        
        # Sharpe ratio
        metrics.sharpe_ratio = calculate_sharpe(returns, risk_free_rate, periods_per_year)
        
        # Sortino ratio
        metrics.sortino_ratio = calculate_sortino(returns, risk_free_rate, periods_per_year)
        
        # Drawdown metrics
        dd_metrics = calculate_drawdown_metrics(equity_curve)
        metrics.max_drawdown = dd_metrics['max_drawdown']
        metrics.max_drawdown_pct = dd_metrics['max_drawdown_pct']
        metrics.avg_drawdown = dd_metrics['avg_drawdown']
        metrics.max_drawdown_duration_days = dd_metrics['max_duration_periods']
        
        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_pct
    
    # Trade statistics
    if len(trades) > 0:
        metrics.total_trades = len(trades)
        
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
        
        # P&L
        metrics.gross_profit = sum(t.pnl for t in winners)
        metrics.gross_loss = abs(sum(t.pnl for t in losers))
        
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        
        if len(winners) > 0:
            metrics.avg_win = metrics.gross_profit / len(winners)
            metrics.largest_win = max(t.pnl for t in winners)
        
        if len(losers) > 0:
            metrics.avg_loss = metrics.gross_loss / len(losers)
            metrics.largest_loss = min(t.pnl for t in losers)
        
        metrics.avg_trade = metrics.total_return / metrics.total_trades
        
        # Trade duration
        durations = [t.duration_hours for t in trades]
        metrics.avg_trade_duration_hours = np.mean(durations)
        
        if winners:
            metrics.avg_winning_duration_hours = np.mean([t.duration_hours for t in winners])
        if losers:
            metrics.avg_losing_duration_hours = np.mean([t.duration_hours for t in losers])
        
        # Streaks
        streak_stats = calculate_streaks(trades)
        metrics.max_consecutive_wins = streak_stats['max_wins']
        metrics.max_consecutive_losses = streak_stats['max_losses']
    
    return metrics


def calculate_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns - rf_period
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (only considers downside volatility)
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_period
    
    # Downside returns only
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown percentage
    
    Args:
        equity_curve: Array of equity values
    
    Returns:
        Max drawdown as decimal (e.g., 0.15 for 15%)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    
    return np.max(drawdown)


def calculate_drawdown_metrics(equity_curve: np.ndarray) -> Dict:
    """
    Calculate comprehensive drawdown metrics
    
    Args:
        equity_curve: Array of equity values
    
    Returns:
        Dict with drawdown metrics
    """
    if len(equity_curve) == 0:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'avg_drawdown': 0.0,
            'max_duration_periods': 0
        }
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    drawdown_abs = peak - equity_curve
    
    max_dd = np.max(drawdown)
    max_dd_abs = np.max(drawdown_abs)
    avg_dd = np.mean(drawdown[drawdown > 0]) if np.any(drawdown > 0) else 0.0
    
    # Calculate max drawdown duration
    max_duration = 0
    current_duration = 0
    
    for i in range(len(equity_curve)):
        if equity_curve[i] < peak[i]:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return {
        'max_drawdown': max_dd_abs,
        'max_drawdown_pct': max_dd * 100,
        'avg_drawdown': avg_dd * 100,
        'max_duration_periods': max_duration
    }


def calculate_win_rate(trades: List[Trade]) -> float:
    """
    Calculate win rate
    
    Args:
        trades: List of trades
    
    Returns:
        Win rate as percentage
    """
    if len(trades) == 0:
        return 0.0
    
    winners = sum(1 for t in trades if t.is_winner)
    return (winners / len(trades)) * 100


def calculate_streaks(trades: List[Trade]) -> Dict:
    """
    Calculate winning and losing streaks
    
    Args:
        trades: List of trades
    
    Returns:
        Dict with streak statistics
    """
    if len(trades) == 0:
        return {'max_wins': 0, 'max_losses': 0}
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for trade in trades:
        if trade.is_winner:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
    
    return {
        'max_wins': max_wins,
        'max_losses': max_losses
    }


def calculate_expectancy(trades: List[Trade]) -> float:
    """
    Calculate trade expectancy
    
    Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    
    Args:
        trades: List of trades
    
    Returns:
        Expectancy per trade
    """
    if len(trades) == 0:
        return 0.0
    
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]
    
    win_rate = len(winners) / len(trades)
    loss_rate = len(losers) / len(trades)
    
    avg_win = np.mean([t.pnl for t in winners]) if winners else 0.0
    avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0.0
    
    return (win_rate * avg_win) - (loss_rate * avg_loss)


def generate_trade_report(trades: List[Trade]) -> str:
    """
    Generate detailed trade report
    
    Args:
        trades: List of trades
    
    Returns:
        Formatted report string
    """
    if len(trades) == 0:
        return "No trades to report."
    
    lines = [
        "Trade Report",
        "=" * 80,
        f"{'#':<4} {'Entry':<20} {'Exit':<20} {'Side':<6} {'Size':<8} {'P&L':>10} {'%':>8} {'Reason':<15}",
        "-" * 80
    ]
    
    for i, trade in enumerate(trades, 1):
        entry = datetime.fromtimestamp(trade.entry_time).strftime("%Y-%m-%d %H:%M")
        exit_time = datetime.fromtimestamp(trade.exit_time).strftime("%Y-%m-%d %H:%M")
        
        lines.append(
            f"{i:<4} {entry:<20} {exit_time:<20} {trade.side:<6} {trade.size:<8.4f} "
            f"${trade.pnl:>9.2f} {trade.pnl_pct:>7.2f}% {trade.exit_reason:<15}"
        )
    
    lines.append("-" * 80)
    
    total_pnl = sum(t.pnl for t in trades)
    lines.append(f"Total P&L: ${total_pnl:,.2f}")
    
    return "\n".join(lines)
