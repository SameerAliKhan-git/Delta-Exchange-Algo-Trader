"""
Performance Metrics Module

Comprehensive trading performance metrics including:
- Return metrics (CAGR, Sharpe, Sortino)
- Risk metrics (VaR, CVaR, Max Drawdown)
- Trade analysis (win rate, profit factor)
- Benchmark comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    # Return metrics
    total_return: float
    cagr: float
    annualized_return: float
    annualized_volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    var_99: float
    cvar_95: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade: float
    avg_holding_period: float
    
    # Additional metrics
    skewness: float
    kurtosis: float
    tail_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'cagr': self.cagr,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from prices.
    
    Args:
        prices: Price series
        
    Returns:
        Return series
    """
    return prices.pct_change().dropna()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from prices.
    
    Args:
        prices: Price series
        
    Returns:
        Log return series
    """
    return np.log(prices / prices.shift(1)).dropna()


def calculate_total_return(returns: pd.Series) -> float:
    """
    Calculate total cumulative return.
    
    Args:
        returns: Return series
        
    Returns:
        Total return as decimal
    """
    return (1 + returns).prod() - 1


def calculate_cagr(
    returns: pd.Series,
    periods_per_year: float = 252
) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        returns: Return series
        periods_per_year: Number of periods per year
        
    Returns:
        CAGR as decimal
    """
    total_return = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    
    if n_years <= 0:
        return 0.0
    
    return total_return ** (1 / n_years) - 1


def calculate_annualized_volatility(
    returns: pd.Series,
    periods_per_year: float = 252
) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Return series
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized volatility as decimal
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Sharpe = (E[R] - Rf) / σ
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    return (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino Ratio.
    
    Sortino = (E[R] - target) / σ_downside
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        target_return: Target return (default 0)
        
    Returns:
        Sortino ratio
    """
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if returns.mean() > target_return else 0.0
    
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    excess_return = (returns.mean() - risk_free_rate / periods_per_year) * periods_per_year
    
    return excess_return / downside_std


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, int, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Return series
        
    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx, duration)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    
    # Find peak before trough
    peak_idx = cumulative[:trough_idx].idxmax()
    
    # Find recovery (when drawdown returns to 0)
    after_trough = drawdown[trough_idx:]
    recovery_mask = after_trough >= 0
    
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        duration = len(returns.loc[peak_idx:recovery_idx])
    else:
        duration = len(returns.loc[peak_idx:])
    
    return abs(max_dd), peak_idx, trough_idx, duration


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk.
    
    Args:
        returns: Return series
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: 'historical' or 'parametric'
        
    Returns:
        VaR as positive number
    """
    if method == 'historical':
        return -np.percentile(returns, (1 - confidence) * 100)
    elif method == 'parametric':
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        return -(returns.mean() + z_score * returns.std())
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    CVaR is the expected loss given that loss exceeds VaR.
    
    Args:
        returns: Return series
        confidence: Confidence level
        
    Returns:
        CVaR as positive number
    """
    var = calculate_var(returns, confidence)
    tail_returns = returns[returns < -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -tail_returns.mean()


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: float = 252
) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar = CAGR / Max Drawdown
    
    Args:
        returns: Return series
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(returns, periods_per_year)
    max_dd, _, _, _ = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return float('inf') if cagr > 0 else 0.0
    
    return cagr / max_dd


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega Ratio.
    
    Omega = E[max(R - threshold, 0)] / E[max(threshold - R, 0)]
    
    Args:
        returns: Return series
        threshold: Threshold return
        
    Returns:
        Omega ratio
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    
    if losses.sum() == 0:
        return float('inf') if gains.sum() > 0 else 0.0
    
    return gains.sum() / losses.sum()


def calculate_profit_factor(
    trade_returns: pd.Series
) -> float:
    """
    Calculate Profit Factor.
    
    Profit Factor = Gross Profit / Gross Loss
    
    Args:
        trade_returns: Return per trade
        
    Returns:
        Profit factor
    """
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_win_rate(trade_returns: pd.Series) -> float:
    """
    Calculate win rate.
    
    Args:
        trade_returns: Return per trade
        
    Returns:
        Win rate as decimal
    """
    if len(trade_returns) == 0:
        return 0.0
    
    return (trade_returns > 0).mean()


def calculate_expectancy(trade_returns: pd.Series) -> float:
    """
    Calculate trade expectancy.
    
    Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    
    Args:
        trade_returns: Return per trade
        
    Returns:
        Expected return per trade
    """
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]
    
    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
    loss_rate = 1 - win_rate
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    return win_rate * avg_win - loss_rate * avg_loss


def calculate_skewness(returns: pd.Series) -> float:
    """
    Calculate return skewness.
    
    Positive skew = more extreme positive returns
    Negative skew = more extreme negative returns
    
    Args:
        returns: Return series
        
    Returns:
        Skewness
    """
    return returns.skew()


def calculate_kurtosis(returns: pd.Series) -> float:
    """
    Calculate return kurtosis.
    
    Higher kurtosis = fatter tails
    Normal distribution has kurtosis = 3
    
    Args:
        returns: Return series
        
    Returns:
        Kurtosis
    """
    return returns.kurtosis()


def calculate_tail_ratio(
    returns: pd.Series,
    percentile: float = 5
) -> float:
    """
    Calculate tail ratio.
    
    Tail Ratio = |95th percentile| / |5th percentile|
    
    Higher is better (larger right tail relative to left tail)
    
    Args:
        returns: Return series
        percentile: Percentile to use (default 5)
        
    Returns:
        Tail ratio
    """
    right_tail = np.percentile(returns, 100 - percentile)
    left_tail = np.percentile(returns, percentile)
    
    if left_tail == 0:
        return float('inf') if right_tail > 0 else 0.0
    
    return abs(right_tail / left_tail)


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    
    Args:
        returns: Return series
        
    Returns:
        Drawdown series (negative values)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def generate_performance_report(
    returns: pd.Series,
    trade_returns: Optional[pd.Series] = None,
    periods_per_year: float = 252,
    risk_free_rate: float = 0.0
) -> PerformanceReport:
    """
    Generate comprehensive performance report.
    
    Args:
        returns: Strategy return series
        trade_returns: Per-trade return series (optional)
        periods_per_year: Number of periods per year
        risk_free_rate: Annual risk-free rate
        
    Returns:
        PerformanceReport object
    """
    # Return metrics
    total_return = calculate_total_return(returns)
    cagr = calculate_cagr(returns, periods_per_year)
    ann_return = returns.mean() * periods_per_year
    ann_vol = calculate_annualized_volatility(returns, periods_per_year)
    
    # Risk-adjusted returns
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(returns, periods_per_year)
    omega = calculate_omega_ratio(returns)
    
    # Risk metrics
    max_dd, _, _, dd_duration = calculate_max_drawdown(returns)
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    
    # Trade statistics
    if trade_returns is not None and len(trade_returns) > 0:
        total_trades = len(trade_returns)
        winning_trades = (trade_returns > 0).sum()
        losing_trades = (trade_returns <= 0).sum()
        win_rate = calculate_win_rate(trade_returns)
        profit_factor = calculate_profit_factor(trade_returns)
        
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns <= 0]
        
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        largest_win = wins.max() if len(wins) > 0 else 0.0
        largest_loss = abs(losses.min()) if len(losses) > 0 else 0.0
        avg_trade = trade_returns.mean()
        avg_holding = 1.0  # Would need actual holding period data
    else:
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        largest_win = 0.0
        largest_loss = 0.0
        avg_trade = 0.0
        avg_holding = 0.0
    
    # Distribution metrics
    skewness = calculate_skewness(returns)
    kurtosis = calculate_kurtosis(returns)
    tail_ratio = calculate_tail_ratio(returns)
    
    return PerformanceReport(
        total_return=total_return,
        cagr=cagr,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_trade=avg_trade,
        avg_holding_period=avg_holding,
        skewness=skewness,
        kurtosis=kurtosis,
        tail_ratio=tail_ratio
    )


def print_performance_report(report: PerformanceReport) -> None:
    """Print formatted performance report."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  PERFORMANCE REPORT                          ║
╠══════════════════════════════════════════════════════════════╣
║  RETURN METRICS                                              ║
║  ─────────────────────────────────────────────────────────── ║""")
    print(f"║  Total Return:           {report.total_return:>10.2%}                    ║")
    print(f"║  CAGR:                   {report.cagr:>10.2%}                    ║")
    print(f"║  Annualized Return:      {report.annualized_return:>10.2%}                    ║")
    print(f"║  Annualized Volatility:  {report.annualized_volatility:>10.2%}                    ║")
    print("""║                                                              ║
║  RISK-ADJUSTED METRICS                                       ║
║  ─────────────────────────────────────────────────────────── ║""")
    print(f"║  Sharpe Ratio:           {report.sharpe_ratio:>10.2f}                    ║")
    print(f"║  Sortino Ratio:          {report.sortino_ratio:>10.2f}                    ║")
    print(f"║  Calmar Ratio:           {report.calmar_ratio:>10.2f}                    ║")
    print(f"║  Omega Ratio:            {report.omega_ratio:>10.2f}                    ║")
    print("""║                                                              ║
║  RISK METRICS                                                ║
║  ─────────────────────────────────────────────────────────── ║""")
    print(f"║  Max Drawdown:           {report.max_drawdown:>10.2%}                    ║")
    print(f"║  Max DD Duration:        {report.max_drawdown_duration:>10d} bars              ║")
    print(f"║  VaR (95%):              {report.var_95:>10.2%}                    ║")
    print(f"║  VaR (99%):              {report.var_99:>10.2%}                    ║")
    print(f"║  CVaR (95%):             {report.cvar_95:>10.2%}                    ║")
    print("""║                                                              ║
║  TRADE STATISTICS                                            ║
║  ─────────────────────────────────────────────────────────── ║""")
    print(f"║  Total Trades:           {report.total_trades:>10d}                    ║")
    print(f"║  Win Rate:               {report.win_rate:>10.2%}                    ║")
    print(f"║  Profit Factor:          {report.profit_factor:>10.2f}                    ║")
    print(f"║  Avg Win:                {report.avg_win:>10.2%}                    ║")
    print(f"║  Avg Loss:               {report.avg_loss:>10.2%}                    ║")
    print(f"║  Largest Win:            {report.largest_win:>10.2%}                    ║")
    print(f"║  Largest Loss:           {report.largest_loss:>10.2%}                    ║")
    print("""║                                                              ║
║  DISTRIBUTION METRICS                                        ║
║  ─────────────────────────────────────────────────────────── ║""")
    print(f"║  Skewness:               {report.skewness:>10.2f}                    ║")
    print(f"║  Kurtosis:               {report.kurtosis:>10.2f}                    ║")
    print(f"║  Tail Ratio:             {report.tail_ratio:>10.2f}                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("PERFORMANCE METRICS DEMO")
    print("=" * 60)
    
    # Generate sample returns
    np.random.seed(42)
    n = 252 * 2  # 2 years daily
    
    returns = pd.Series(np.random.normal(0.0003, 0.015, n))  # ~7% annual, 24% vol
    trade_returns = pd.Series(np.random.normal(0.005, 0.02, 100))  # 100 trades
    
    # Generate report
    report = generate_performance_report(
        returns,
        trade_returns,
        periods_per_year=252
    )
    
    print_performance_report(report)
