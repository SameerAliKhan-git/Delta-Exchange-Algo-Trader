"""
Performance Analyzer - Advanced backtesting analytics

Provides:
- Detailed performance breakdown
- Risk-adjusted metrics
- Drawdown analysis
- Trade clustering
- Monte Carlo simulation
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .metrics import Trade, PerformanceMetrics, calculate_metrics

logger = logging.getLogger("Aladdin.Performance")


@dataclass
class DrawdownPeriod:
    """Drawdown period details"""
    start_time: datetime
    end_time: Optional[datetime]
    recovery_time: Optional[datetime]
    peak_equity: float
    trough_equity: float
    drawdown_pct: float
    duration_days: float
    recovery_days: Optional[float]


@dataclass
class MonthlyReturn:
    """Monthly return record"""
    year: int
    month: int
    return_pct: float
    num_trades: int
    win_rate: float


@dataclass
class RiskMetrics:
    """Advanced risk metrics"""
    # Volatility
    daily_volatility: float = 0.0
    annualized_volatility: float = 0.0
    
    # Value at Risk
    var_95: float = 0.0  # 95% VaR (1-day)
    var_99: float = 0.0  # 99% VaR (1-day)
    cvar_95: float = 0.0  # Conditional VaR
    
    # Tail risk
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Downside metrics
    downside_deviation: float = 0.0
    ulcer_index: float = 0.0


class PerformanceAnalyzer:
    """
    Advanced performance analytics for backtesting
    
    Features:
    - Monthly/yearly breakdown
    - Drawdown analysis
    - Risk metrics (VaR, CVaR)
    - Monte Carlo simulation
    - Trade clustering analysis
    """
    
    def __init__(
        self,
        trades: List[Trade],
        equity_curve: np.ndarray,
        timestamps: np.ndarray,
        initial_capital: float
    ):
        """
        Initialize analyzer
        
        Args:
            trades: List of trades
            equity_curve: Equity values over time
            timestamps: Timestamps for equity curve
            initial_capital: Starting capital
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.timestamps = timestamps
        self.initial_capital = initial_capital
        
        # Calculate basic metrics
        self.metrics = calculate_metrics(trades, equity_curve, initial_capital)
    
    def get_monthly_returns(self) -> List[MonthlyReturn]:
        """
        Calculate monthly returns
        
        Returns:
            List of MonthlyReturn
        """
        if len(self.equity_curve) == 0:
            return []
        
        # Group by month
        monthly: Dict[Tuple[int, int], Dict] = {}
        
        prev_equity = self.initial_capital
        
        for i, ts in enumerate(self.timestamps):
            dt = datetime.fromtimestamp(ts)
            key = (dt.year, dt.month)
            
            if key not in monthly:
                monthly[key] = {
                    'start_equity': prev_equity,
                    'end_equity': self.equity_curve[i],
                    'trades': []
                }
            else:
                monthly[key]['end_equity'] = self.equity_curve[i]
            
            prev_equity = self.equity_curve[i]
        
        # Add trades to months
        for trade in self.trades:
            dt = datetime.fromtimestamp(trade.exit_time)
            key = (dt.year, dt.month)
            if key in monthly:
                monthly[key]['trades'].append(trade)
        
        # Calculate returns
        results = []
        for (year, month), data in sorted(monthly.items()):
            trades = data['trades']
            winners = [t for t in trades if t.is_winner]
            
            return_pct = ((data['end_equity'] - data['start_equity']) / data['start_equity']) * 100
            win_rate = (len(winners) / len(trades) * 100) if trades else 0
            
            results.append(MonthlyReturn(
                year=year,
                month=month,
                return_pct=return_pct,
                num_trades=len(trades),
                win_rate=win_rate
            ))
        
        return results
    
    def get_drawdown_periods(self, min_drawdown_pct: float = 5.0) -> List[DrawdownPeriod]:
        """
        Identify significant drawdown periods
        
        Args:
            min_drawdown_pct: Minimum drawdown to consider
        
        Returns:
            List of DrawdownPeriod
        """
        if len(self.equity_curve) == 0:
            return []
        
        periods = []
        peak = self.equity_curve[0]
        peak_idx = 0
        in_drawdown = False
        trough = peak
        trough_idx = 0
        
        for i in range(len(self.equity_curve)):
            equity = self.equity_curve[i]
            
            if equity > peak:
                # New peak - close any open drawdown
                if in_drawdown:
                    dd_pct = ((peak - trough) / peak) * 100
                    if dd_pct >= min_drawdown_pct:
                        periods.append(DrawdownPeriod(
                            start_time=datetime.fromtimestamp(self.timestamps[peak_idx]),
                            end_time=datetime.fromtimestamp(self.timestamps[trough_idx]),
                            recovery_time=datetime.fromtimestamp(self.timestamps[i]),
                            peak_equity=peak,
                            trough_equity=trough,
                            drawdown_pct=dd_pct,
                            duration_days=(self.timestamps[trough_idx] - self.timestamps[peak_idx]) / 86400,
                            recovery_days=(self.timestamps[i] - self.timestamps[trough_idx]) / 86400
                        ))
                    in_drawdown = False
                
                peak = equity
                peak_idx = i
                trough = equity
                trough_idx = i
            
            elif equity < trough:
                trough = equity
                trough_idx = i
                in_drawdown = True
        
        # Handle open drawdown at end
        if in_drawdown:
            dd_pct = ((peak - trough) / peak) * 100
            if dd_pct >= min_drawdown_pct:
                periods.append(DrawdownPeriod(
                    start_time=datetime.fromtimestamp(self.timestamps[peak_idx]),
                    end_time=datetime.fromtimestamp(self.timestamps[trough_idx]),
                    recovery_time=None,
                    peak_equity=peak,
                    trough_equity=trough,
                    drawdown_pct=dd_pct,
                    duration_days=(self.timestamps[trough_idx] - self.timestamps[peak_idx]) / 86400,
                    recovery_days=None
                ))
        
        return periods
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """
        Calculate advanced risk metrics
        
        Returns:
            RiskMetrics
        """
        metrics = RiskMetrics()
        
        if len(self.equity_curve) < 2:
            return metrics
        
        # Calculate returns
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # Volatility
        metrics.daily_volatility = np.std(returns) * 100
        metrics.annualized_volatility = metrics.daily_volatility * np.sqrt(252)
        
        # VaR and CVaR
        metrics.var_95 = np.percentile(returns, 5) * self.equity_curve[-1] * -1
        metrics.var_99 = np.percentile(returns, 1) * self.equity_curve[-1] * -1
        
        # CVaR (Expected Shortfall)
        below_var = returns[returns <= np.percentile(returns, 5)]
        if len(below_var) > 0:
            metrics.cvar_95 = np.mean(below_var) * self.equity_curve[-1] * -1
        
        # Higher moments
        metrics.skewness = float(self._calculate_skewness(returns))
        metrics.kurtosis = float(self._calculate_kurtosis(returns))
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics.downside_deviation = np.std(negative_returns) * 100
        
        # Ulcer Index
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        metrics.ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100
        
        return metrics
    
    def monte_carlo_simulation(
        self,
        num_simulations: int = 1000,
        num_periods: int = 252
    ) -> Dict:
        """
        Run Monte Carlo simulation on returns
        
        Args:
            num_simulations: Number of simulation paths
            num_periods: Number of periods to simulate
        
        Returns:
            Dict with simulation results
        """
        if len(self.equity_curve) < 2:
            return {}
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # Fit return distribution
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Run simulations
        final_equities = []
        paths = []
        
        for _ in range(num_simulations):
            # Generate random returns
            sim_returns = np.random.normal(mean_return, std_return, num_periods)
            
            # Calculate path
            equity = self.initial_capital
            path = [equity]
            for r in sim_returns:
                equity *= (1 + r)
                path.append(equity)
            
            final_equities.append(equity)
            if len(paths) < 100:  # Store first 100 paths for visualization
                paths.append(path)
        
        final_equities = np.array(final_equities)
        
        return {
            'mean_final': np.mean(final_equities),
            'median_final': np.median(final_equities),
            'std_final': np.std(final_equities),
            'percentile_5': np.percentile(final_equities, 5),
            'percentile_25': np.percentile(final_equities, 25),
            'percentile_75': np.percentile(final_equities, 75),
            'percentile_95': np.percentile(final_equities, 95),
            'prob_profit': (final_equities > self.initial_capital).mean() * 100,
            'prob_double': (final_equities > 2 * self.initial_capital).mean() * 100,
            'prob_ruin': (final_equities < self.initial_capital * 0.5).mean() * 100,
            'sample_paths': paths[:10]  # First 10 paths
        }
    
    def analyze_trade_clusters(self) -> Dict:
        """
        Analyze trade clustering and timing
        
        Returns:
            Dict with clustering analysis
        """
        if not self.trades:
            return {}
        
        # Group by hour of day
        hourly_pnl = {h: [] for h in range(24)}
        hourly_count = {h: 0 for h in range(24)}
        
        # Group by day of week
        daily_pnl = {d: [] for d in range(7)}
        daily_count = {d: 0 for d in range(7)}
        
        for trade in self.trades:
            dt = datetime.fromtimestamp(trade.entry_time)
            
            hourly_pnl[dt.hour].append(trade.pnl)
            hourly_count[dt.hour] += 1
            
            daily_pnl[dt.weekday()].append(trade.pnl)
            daily_count[dt.weekday()] += 1
        
        # Best/worst hours
        hourly_avg = {h: (sum(pnls) / len(pnls) if pnls else 0) for h, pnls in hourly_pnl.items()}
        best_hour = max(hourly_avg, key=hourly_avg.get)
        worst_hour = min(hourly_avg, key=hourly_avg.get)
        
        # Best/worst days
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_avg = {d: (sum(pnls) / len(pnls) if pnls else 0) for d, pnls in daily_pnl.items()}
        best_day = max(daily_avg, key=daily_avg.get)
        worst_day = min(daily_avg, key=daily_avg.get)
        
        # Consecutive winners/losers
        streaks = []
        current_streak = 0
        streak_type = None
        
        for trade in self.trades:
            if trade.is_winner:
                if streak_type == 'win':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append((streak_type, current_streak))
                    current_streak = 1
                    streak_type = 'win'
            else:
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append((streak_type, current_streak))
                    current_streak = 1
                    streak_type = 'loss'
        
        if current_streak > 0:
            streaks.append((streak_type, current_streak))
        
        win_streaks = [s[1] for s in streaks if s[0] == 'win']
        loss_streaks = [s[1] for s in streaks if s[0] == 'loss']
        
        return {
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'best_day': day_names[best_day],
            'worst_day': day_names[worst_day],
            'hourly_counts': hourly_count,
            'hourly_avg_pnl': hourly_avg,
            'daily_counts': daily_count,
            'daily_avg_pnl': {day_names[d]: v for d, v in daily_avg.items()},
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'max_win_streak': max(win_streaks) if win_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0
        }
    
    def generate_report(self) -> str:
        """
        Generate comprehensive performance report
        
        Returns:
            Formatted report string
        """
        monthly = self.get_monthly_returns()
        risk = self.calculate_risk_metrics()
        drawdowns = self.get_drawdown_periods(5.0)
        clusters = self.analyze_trade_clusters()
        
        lines = [
            "=" * 80,
            "COMPREHENSIVE PERFORMANCE REPORT",
            "=" * 80,
            "",
            "SUMMARY STATISTICS",
            "-" * 40,
            f"Initial Capital:     ${self.initial_capital:,.2f}",
            f"Final Equity:        ${self.equity_curve[-1]:,.2f}" if len(self.equity_curve) > 0 else "",
            f"Total Return:        {self.metrics.total_return_pct:.2f}%",
            f"Annualized Return:   {self.metrics.annualized_return:.2f}%",
            "",
            "RISK METRICS",
            "-" * 40,
            f"Sharpe Ratio:        {self.metrics.sharpe_ratio:.2f}",
            f"Sortino Ratio:       {self.metrics.sortino_ratio:.2f}",
            f"Calmar Ratio:        {self.metrics.calmar_ratio:.2f}",
            f"Max Drawdown:        {self.metrics.max_drawdown_pct:.2f}%",
            f"Daily Volatility:    {risk.daily_volatility:.2f}%",
            f"Annual Volatility:   {risk.annualized_volatility:.2f}%",
            f"VaR (95%):           ${risk.var_95:,.2f}",
            f"CVaR (95%):          ${risk.cvar_95:,.2f}",
            f"Ulcer Index:         {risk.ulcer_index:.2f}",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades:        {self.metrics.total_trades}",
            f"Win Rate:            {self.metrics.win_rate:.1f}%",
            f"Profit Factor:       {self.metrics.profit_factor:.2f}",
            f"Average Trade:       ${self.metrics.avg_trade:.2f}",
            f"Average Win:         ${self.metrics.avg_win:.2f}",
            f"Average Loss:        ${self.metrics.avg_loss:.2f}",
            f"Largest Win:         ${self.metrics.largest_win:.2f}",
            f"Largest Loss:        ${self.metrics.largest_loss:.2f}",
            "",
        ]
        
        # Monthly returns table
        if monthly:
            lines.extend([
                "MONTHLY RETURNS",
                "-" * 40,
                f"{'Month':<12} {'Return':>10} {'Trades':>8} {'Win Rate':>10}",
            ])
            for m in monthly[-12:]:  # Last 12 months
                lines.append(
                    f"{m.year}-{m.month:02d}      {m.return_pct:>9.2f}% {m.num_trades:>8} {m.win_rate:>9.1f}%"
                )
            lines.append("")
        
        # Drawdown periods
        if drawdowns:
            lines.extend([
                "SIGNIFICANT DRAWDOWNS",
                "-" * 40,
            ])
            for dd in drawdowns[:5]:  # Top 5
                recovery = f"{dd.recovery_days:.1f}d" if dd.recovery_days else "ongoing"
                lines.append(
                    f"  {dd.start_time.strftime('%Y-%m-%d')}: "
                    f"{dd.drawdown_pct:.1f}% over {dd.duration_days:.1f}d, recovery: {recovery}"
                )
            lines.append("")
        
        # Trade timing
        if clusters:
            lines.extend([
                "TRADE TIMING ANALYSIS",
                "-" * 40,
                f"Best Hour:           {clusters['best_hour']}:00",
                f"Worst Hour:          {clusters['worst_hour']}:00",
                f"Best Day:            {clusters['best_day']}",
                f"Worst Day:           {clusters['worst_day']}",
                f"Avg Win Streak:      {clusters['avg_win_streak']:.1f}",
                f"Avg Loss Streak:     {clusters['avg_loss_streak']:.1f}",
                "",
            ])
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis"""
        if len(data) < 4:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * \
               np.sum(((data - mean) / std) ** 4) - \
               (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))


if __name__ == "__main__":
    # Test performance analyzer
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic trades and equity curve
    np.random.seed(42)
    
    initial = 10000
    equity = [initial]
    trades = []
    
    for i in range(100):
        # Random trade
        is_winner = np.random.random() > 0.45  # 55% win rate
        if is_winner:
            pnl = np.random.uniform(50, 300)
        else:
            pnl = -np.random.uniform(30, 200)
        
        equity.append(equity[-1] + pnl)
        
        trades.append(Trade(
            entry_time=1700000000 + i * 3600,
            exit_time=1700000000 + (i + 1) * 3600,
            entry_price=100000 + np.random.uniform(-1000, 1000),
            exit_price=100000 + np.random.uniform(-1000, 1000),
            size=0.1,
            side='long' if np.random.random() > 0.5 else 'short',
            pnl=pnl,
            pnl_pct=(pnl / initial) * 100,
            exit_reason='signal'
        ))
    
    equity_curve = np.array(equity)
    timestamps = np.array([1700000000 + i * 3600 for i in range(len(equity))])
    
    # Analyze
    analyzer = PerformanceAnalyzer(trades, equity_curve, timestamps, initial)
    
    print(analyzer.generate_report())
    
    # Monte Carlo
    print("\nMONTE CARLO SIMULATION (1000 paths, 252 periods)")
    print("-" * 40)
    mc = analyzer.monte_carlo_simulation(1000, 252)
    print(f"Mean Final:      ${mc['mean_final']:,.2f}")
    print(f"Median Final:    ${mc['median_final']:,.2f}")
    print(f"5th Percentile:  ${mc['percentile_5']:,.2f}")
    print(f"95th Percentile: ${mc['percentile_95']:,.2f}")
    print(f"Prob. Profit:    {mc['prob_profit']:.1f}%")
    print(f"Prob. 2x:        {mc['prob_double']:.1f}%")
    print(f"Prob. Ruin:      {mc['prob_ruin']:.1f}%")
