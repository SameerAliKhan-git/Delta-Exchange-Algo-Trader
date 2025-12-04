"""
Transaction Cost Sensitivity Analysis
=====================================
CRITICAL DELIVERABLE: Understand how costs impact profitability.

Your bot is operating blind to cost drag.
This module runs your backtest with varying slippage/fees (0 to 50 bps)
to reveal the true robustness of your edge.

If profitability collapses after 8-12 bps → Strategy cannot survive
If it survives up to 20-25 bps → You have a robust, scalable edge

This tells you REAL expected returns, not optimistic backtest returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CostSensitivityResult:
    """Result for a single cost level."""
    cost_bps: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    
    # Breakdown
    gross_pnl: float
    total_costs: float
    net_pnl: float
    
    def to_dict(self) -> Dict:
        return {
            'cost_bps': self.cost_bps,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_pnl': self.avg_trade_pnl,
            'gross_pnl': self.gross_pnl,
            'total_costs': self.total_costs,
            'net_pnl': self.net_pnl
        }


@dataclass
class CostSensitivityAnalysis:
    """Complete cost sensitivity analysis results."""
    results: List[CostSensitivityResult]
    
    # Critical thresholds
    breakeven_cost_bps: float  # Cost at which strategy breaks even
    robustness_score: float    # 0-100 score for cost robustness
    
    # Degradation rates
    sharpe_degradation_per_bps: float
    return_degradation_per_bps: float
    
    # Recommendations
    max_viable_cost_bps: float  # Max cost for >1.0 Sharpe
    optimal_cost_assumption_bps: float  # Recommended cost to use in backtest
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    num_cost_levels: int = 0
    strategy_name: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'results': [r.to_dict() for r in self.results],
            'breakeven_cost_bps': self.breakeven_cost_bps,
            'robustness_score': self.robustness_score,
            'sharpe_degradation_per_bps': self.sharpe_degradation_per_bps,
            'return_degradation_per_bps': self.return_degradation_per_bps,
            'max_viable_cost_bps': self.max_viable_cost_bps,
            'optimal_cost_assumption_bps': self.optimal_cost_assumption_bps,
            'timestamp': self.timestamp.isoformat(),
            'num_cost_levels': self.num_cost_levels,
            'strategy_name': self.strategy_name
        }


# =============================================================================
# COST SENSITIVITY ANALYZER
# =============================================================================

class TransactionCostAnalyzer:
    """
    Analyze strategy sensitivity to transaction costs.
    
    This is CRITICAL for understanding true profitability.
    """
    
    def __init__(
        self,
        cost_range: Tuple[float, float] = (0, 50),
        num_levels: int = 21,
        include_spread: bool = True,
        include_slippage: bool = True,
        include_fees: bool = True
    ):
        """
        Initialize cost sensitivity analyzer.
        
        Args:
            cost_range: Min and max costs in bps to test
            num_levels: Number of cost levels to test
            include_spread: Include bid-ask spread in costs
            include_slippage: Include market impact slippage
            include_fees: Include exchange fees
        """
        self.cost_range = cost_range
        self.num_levels = num_levels
        self.include_spread = include_spread
        self.include_slippage = include_slippage
        self.include_fees = include_fees
        
        self.cost_levels = np.linspace(cost_range[0], cost_range[1], num_levels)
    
    def analyze_from_trades(
        self,
        trades: pd.DataFrame,
        capital: float = 100000,
        strategy_name: str = "strategy"
    ) -> CostSensitivityAnalysis:
        """
        Analyze cost sensitivity from trade data.
        
        Args:
            trades: DataFrame with columns:
                - entry_price: Entry price
                - exit_price: Exit price
                - quantity: Trade size
                - side: 'long' or 'short'
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp
            capital: Initial capital
            strategy_name: Name of strategy
        
        Returns:
            CostSensitivityAnalysis with full results
        """
        results = []
        
        for cost_bps in self.cost_levels:
            result = self._calculate_metrics_at_cost(trades, capital, cost_bps)
            results.append(result)
        
        # Calculate key thresholds
        breakeven = self._find_breakeven_cost(results)
        robustness = self._calculate_robustness_score(results)
        sharpe_deg = self._calculate_degradation_rate(results, 'sharpe_ratio')
        return_deg = self._calculate_degradation_rate(results, 'annualized_return')
        max_viable = self._find_max_viable_cost(results, min_sharpe=1.0)
        optimal = self._calculate_optimal_cost_assumption(results)
        
        analysis = CostSensitivityAnalysis(
            results=results,
            breakeven_cost_bps=breakeven,
            robustness_score=robustness,
            sharpe_degradation_per_bps=sharpe_deg,
            return_degradation_per_bps=return_deg,
            max_viable_cost_bps=max_viable,
            optimal_cost_assumption_bps=optimal,
            num_cost_levels=len(results),
            strategy_name=strategy_name
        )
        
        return analysis
    
    def analyze_from_returns(
        self,
        returns: np.ndarray,
        turnover: float,
        capital: float = 100000,
        periods_per_year: int = 252,
        strategy_name: str = "strategy"
    ) -> CostSensitivityAnalysis:
        """
        Analyze cost sensitivity from returns and turnover.
        
        Args:
            returns: Array of strategy returns (before costs)
            turnover: Average daily turnover (e.g., 0.5 = 50% of portfolio)
            capital: Initial capital
            periods_per_year: Number of periods per year
            strategy_name: Name of strategy
        
        Returns:
            CostSensitivityAnalysis
        """
        results = []
        
        for cost_bps in self.cost_levels:
            # Cost per period = turnover * cost_bps / 10000
            cost_per_period = turnover * cost_bps / 10000
            net_returns = returns - cost_per_period
            
            result = self._calculate_metrics_from_returns(
                gross_returns=returns,
                net_returns=net_returns,
                cost_bps=cost_bps,
                capital=capital,
                periods_per_year=periods_per_year
            )
            results.append(result)
        
        # Calculate key thresholds
        breakeven = self._find_breakeven_cost(results)
        robustness = self._calculate_robustness_score(results)
        sharpe_deg = self._calculate_degradation_rate(results, 'sharpe_ratio')
        return_deg = self._calculate_degradation_rate(results, 'annualized_return')
        max_viable = self._find_max_viable_cost(results, min_sharpe=1.0)
        optimal = self._calculate_optimal_cost_assumption(results)
        
        analysis = CostSensitivityAnalysis(
            results=results,
            breakeven_cost_bps=breakeven,
            robustness_score=robustness,
            sharpe_degradation_per_bps=sharpe_deg,
            return_degradation_per_bps=return_deg,
            max_viable_cost_bps=max_viable,
            optimal_cost_assumption_bps=optimal,
            num_cost_levels=len(results),
            strategy_name=strategy_name
        )
        
        return analysis
    
    def _calculate_metrics_at_cost(
        self,
        trades: pd.DataFrame,
        capital: float,
        cost_bps: float
    ) -> CostSensitivityResult:
        """Calculate performance metrics at a specific cost level."""
        
        if len(trades) == 0:
            return CostSensitivityResult(
                cost_bps=cost_bps,
                total_return=0,
                annualized_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                num_trades=0,
                win_rate=0,
                profit_factor=0,
                avg_trade_pnl=0,
                gross_pnl=0,
                total_costs=0,
                net_pnl=0
            )
        
        # Calculate gross P&L per trade
        gross_pnls = []
        for _, trade in trades.iterrows():
            entry = trade['entry_price']
            exit = trade['exit_price']
            qty = trade['quantity']
            side = trade.get('side', 'long')
            
            if side == 'long':
                gross_pnl = (exit - entry) * qty
            else:
                gross_pnl = (entry - exit) * qty
            
            gross_pnls.append(gross_pnl)
        
        gross_pnls = np.array(gross_pnls)
        
        # Calculate costs per trade (entry + exit)
        trade_values = trades['entry_price'] * trades['quantity']
        costs_per_trade = 2 * trade_values * cost_bps / 10000  # Entry + exit
        
        # Net P&L
        net_pnls = gross_pnls - costs_per_trade.values
        
        # Aggregate metrics
        total_gross_pnl = np.sum(gross_pnls)
        total_costs = np.sum(costs_per_trade)
        total_net_pnl = np.sum(net_pnls)
        
        total_return = total_net_pnl / capital
        
        # Annualize based on trade duration
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            duration_days = (trades['exit_time'].max() - trades['entry_time'].min()).days
            if duration_days > 0:
                annualized_return = total_return * 365 / duration_days
            else:
                annualized_return = total_return
        else:
            annualized_return = total_return
        
        # Win rate and profit factor
        winning_trades = net_pnls[net_pnls > 0]
        losing_trades = net_pnls[net_pnls <= 0]
        
        win_rate = len(winning_trades) / len(net_pnls) if len(net_pnls) > 0 else 0
        
        gross_profit = np.sum(winning_trades)
        gross_loss = np.abs(np.sum(losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        if len(net_pnls) > 1:
            sharpe = np.mean(net_pnls) / np.std(net_pnls) * np.sqrt(252 / len(net_pnls) * len(trades))
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(net_pnls) / capital
        running_max = np.maximum.accumulate(cumulative + 1)
        drawdowns = (running_max - (cumulative + 1)) / running_max
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
        
        return CostSensitivityResult(
            cost_bps=cost_bps,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            num_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=np.mean(net_pnls),
            gross_pnl=total_gross_pnl,
            total_costs=total_costs,
            net_pnl=total_net_pnl
        )
    
    def _calculate_metrics_from_returns(
        self,
        gross_returns: np.ndarray,
        net_returns: np.ndarray,
        cost_bps: float,
        capital: float,
        periods_per_year: int
    ) -> CostSensitivityResult:
        """Calculate metrics from return series."""
        
        total_gross = np.prod(1 + gross_returns) - 1
        total_net = np.prod(1 + net_returns) - 1
        total_costs = total_gross - total_net
        
        n_periods = len(net_returns)
        years = n_periods / periods_per_year
        
        annualized = (1 + total_net) ** (1 / years) - 1 if years > 0 else total_net
        
        # Sharpe
        if np.std(net_returns) > 0:
            sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(periods_per_year)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + net_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_dd = float(np.max(drawdowns))
        
        # Win rate
        win_rate = np.mean(net_returns > 0)
        
        # Profit factor
        gains = net_returns[net_returns > 0].sum()
        losses = np.abs(net_returns[net_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        return CostSensitivityResult(
            cost_bps=cost_bps,
            total_return=total_net,
            annualized_return=annualized,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            num_trades=n_periods,  # Using periods as proxy
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=np.mean(net_returns) * capital,
            gross_pnl=total_gross * capital,
            total_costs=total_costs * capital,
            net_pnl=total_net * capital
        )
    
    def _find_breakeven_cost(self, results: List[CostSensitivityResult]) -> float:
        """Find cost level where strategy breaks even."""
        for i, r in enumerate(results):
            if r.total_return <= 0:
                if i == 0:
                    return 0.0
                # Interpolate
                prev = results[i-1]
                slope = (r.total_return - prev.total_return) / (r.cost_bps - prev.cost_bps)
                if slope != 0:
                    breakeven = prev.cost_bps - prev.total_return / slope
                    return max(0, breakeven)
                return prev.cost_bps
        
        # Never breaks even within range
        return self.cost_range[1]
    
    def _find_max_viable_cost(
        self,
        results: List[CostSensitivityResult],
        min_sharpe: float = 1.0
    ) -> float:
        """Find maximum cost where strategy maintains minimum Sharpe."""
        for i, r in enumerate(results):
            if r.sharpe_ratio < min_sharpe:
                if i == 0:
                    return 0.0
                # Interpolate
                prev = results[i-1]
                slope = (r.sharpe_ratio - prev.sharpe_ratio) / (r.cost_bps - prev.cost_bps)
                if slope != 0:
                    max_viable = prev.cost_bps + (min_sharpe - prev.sharpe_ratio) / slope
                    return max(0, max_viable)
                return prev.cost_bps
        
        return self.cost_range[1]
    
    def _calculate_robustness_score(self, results: List[CostSensitivityResult]) -> float:
        """
        Calculate cost robustness score (0-100).
        
        Score based on:
        - Breakeven cost level
        - Sharpe degradation rate
        - Profitability at realistic costs (10-20 bps)
        """
        if not results:
            return 0.0
        
        zero_cost = results[0]
        
        # Find results at realistic cost levels
        result_10bps = next((r for r in results if r.cost_bps >= 10), results[-1])
        result_20bps = next((r for r in results if r.cost_bps >= 20), results[-1])
        
        score = 0.0
        
        # Component 1: Breakeven cost (0-40 points)
        # 50 bps breakeven = 40 points, 0 bps = 0 points
        breakeven = self._find_breakeven_cost(results)
        score += min(40, breakeven * 0.8)
        
        # Component 2: Sharpe at 15 bps (0-30 points)
        # Sharpe >= 2.0 = 30 points, 0 = 0 points
        result_15bps = next((r for r in results if r.cost_bps >= 15), results[-1])
        score += min(30, result_15bps.sharpe_ratio * 15)
        
        # Component 3: Sharpe retention (0-30 points)
        # (Sharpe at 20 bps / Sharpe at 0 bps) * 30
        if zero_cost.sharpe_ratio > 0:
            retention = result_20bps.sharpe_ratio / zero_cost.sharpe_ratio
            score += min(30, retention * 30)
        
        return min(100, score)
    
    def _calculate_degradation_rate(
        self,
        results: List[CostSensitivityResult],
        metric: str
    ) -> float:
        """Calculate rate of metric degradation per bps of cost."""
        if len(results) < 2:
            return 0.0
        
        costs = np.array([r.cost_bps for r in results])
        values = np.array([getattr(r, metric) for r in results])
        
        # Linear regression
        if np.std(costs) > 0:
            slope = np.cov(costs, values)[0, 1] / np.var(costs)
            return float(slope)
        return 0.0
    
    def _calculate_optimal_cost_assumption(
        self,
        results: List[CostSensitivityResult]
    ) -> float:
        """
        Calculate optimal cost assumption for backtesting.
        
        This is the cost level you should use in backtests to get
        realistic expectations. Uses P75 of realistic cost range.
        """
        # Find cost where Sharpe drops to 75% of zero-cost Sharpe
        zero_cost_sharpe = results[0].sharpe_ratio if results else 0
        
        for r in results:
            if r.sharpe_ratio < 0.75 * zero_cost_sharpe:
                return r.cost_bps
        
        # If strategy is very robust, use 15 bps as default
        return 15.0
    
    def generate_report(
        self,
        analysis: CostSensitivityAnalysis,
        output_dir: str = "."
    ) -> str:
        """Generate comprehensive cost sensitivity report."""
        
        report_lines = [
            "=" * 70,
            "TRANSACTION COST SENSITIVITY ANALYSIS",
            "=" * 70,
            f"Strategy: {analysis.strategy_name}",
            f"Generated: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Cost Levels Tested: {analysis.num_cost_levels}",
            "",
            "-" * 70,
            "CRITICAL THRESHOLDS",
            "-" * 70,
            f"Breakeven Cost:              {analysis.breakeven_cost_bps:.1f} bps",
            f"Max Viable Cost (Sharpe>1):  {analysis.max_viable_cost_bps:.1f} bps",
            f"Optimal Backtest Assumption: {analysis.optimal_cost_assumption_bps:.1f} bps",
            f"Cost Robustness Score:       {analysis.robustness_score:.0f}/100",
            "",
            "-" * 70,
            "DEGRADATION RATES",
            "-" * 70,
            f"Sharpe per bps:  {analysis.sharpe_degradation_per_bps:.4f}",
            f"Return per bps:  {analysis.return_degradation_per_bps:.4f}",
            "",
            "-" * 70,
            "PERFORMANCE BY COST LEVEL",
            "-" * 70,
            f"{'Cost(bps)':>10} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'WinRate':>10}",
            "-" * 70
        ]
        
        for r in analysis.results:
            report_lines.append(
                f"{r.cost_bps:>10.1f} {r.annualized_return:>10.2%} "
                f"{r.sharpe_ratio:>10.2f} {r.max_drawdown:>10.2%} {r.win_rate:>10.2%}"
            )
        
        report_lines.extend([
            "",
            "-" * 70,
            "INTERPRETATION",
            "-" * 70,
        ])
        
        # Add interpretation
        if analysis.robustness_score >= 70:
            report_lines.append("✅ ROBUST: Strategy survives realistic transaction costs")
            report_lines.append(f"   Can operate up to {analysis.max_viable_cost_bps:.0f} bps and maintain Sharpe > 1")
        elif analysis.robustness_score >= 40:
            report_lines.append("⚠️  MARGINAL: Strategy is sensitive to transaction costs")
            report_lines.append(f"   Must keep costs below {analysis.max_viable_cost_bps:.0f} bps")
        else:
            report_lines.append("❌ FRAGILE: Strategy does not survive realistic costs")
            report_lines.append("   Edge is likely illusory or requires significant improvement")
        
        report_lines.extend([
            "",
            "-" * 70,
            "RECOMMENDATIONS",
            "-" * 70,
        ])
        
        if analysis.breakeven_cost_bps < 10:
            report_lines.append("• Strategy edge is too thin - consider filtering for higher-conviction trades")
        if analysis.max_viable_cost_bps < 15:
            report_lines.append("• Reduce turnover to minimize cost impact")
            report_lines.append("• Consider limit orders to reduce slippage")
        if analysis.robustness_score < 50:
            report_lines.append("• DO NOT deploy live until robustness improves")
        
        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"cost_sensitivity_{analysis.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        # Save JSON data
        json_file = output_path / f"cost_sensitivity_{analysis.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2, default=str)
        
        return report_text
    
    def plot_sensitivity(
        self,
        analysis: CostSensitivityAnalysis,
        output_file: Optional[str] = None
    ):
        """Plot cost sensitivity curves."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        costs = [r.cost_bps for r in analysis.results]
        
        # Plot 1: Annualized Return vs Cost
        ax1 = axes[0, 0]
        returns = [r.annualized_return * 100 for r in analysis.results]
        ax1.plot(costs, returns, 'b-', linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(x=analysis.breakeven_cost_bps, color='r', linestyle=':', alpha=0.7,
                   label=f'Breakeven: {analysis.breakeven_cost_bps:.1f} bps')
        ax1.fill_between(costs, returns, alpha=0.3)
        ax1.set_xlabel('Transaction Cost (bps)')
        ax1.set_ylabel('Annualized Return (%)')
        ax1.set_title('Return vs Transaction Cost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sharpe Ratio vs Cost
        ax2 = axes[0, 1]
        sharpes = [r.sharpe_ratio for r in analysis.results]
        ax2.plot(costs, sharpes, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        ax2.axvline(x=analysis.max_viable_cost_bps, color='orange', linestyle=':', alpha=0.7,
                   label=f'Max Viable: {analysis.max_viable_cost_bps:.1f} bps')
        ax2.fill_between(costs, sharpes, alpha=0.3, color='green')
        ax2.set_xlabel('Transaction Cost (bps)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio vs Transaction Cost')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: P&L Breakdown
        ax3 = axes[1, 0]
        gross = [r.gross_pnl for r in analysis.results]
        net = [r.net_pnl for r in analysis.results]
        total_costs = [r.total_costs for r in analysis.results]
        
        ax3.fill_between(costs, gross, label='Gross P&L', alpha=0.5, color='blue')
        ax3.fill_between(costs, net, label='Net P&L', alpha=0.5, color='green')
        ax3.plot(costs, total_costs, 'r--', label='Total Costs', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Transaction Cost (bps)')
        ax3.set_ylabel('P&L ($)')
        ax3.set_title('P&L Breakdown by Cost Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Robustness Summary
        ax4 = axes[1, 1]
        
        # Create bar chart of key metrics
        metrics = ['Breakeven\n(bps)', 'Max Viable\n(bps)', 'Robustness\nScore', 'Optimal\nAssumption']
        values = [
            analysis.breakeven_cost_bps,
            analysis.max_viable_cost_bps,
            analysis.robustness_score,
            analysis.optimal_cost_assumption_bps
        ]
        colors = ['red', 'orange', 'blue', 'green']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Value')
        ax4.set_title('Cost Robustness Summary')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()


# =============================================================================
# QUICK ANALYSIS FUNCTIONS
# =============================================================================

def quick_cost_analysis(
    returns: np.ndarray,
    turnover: float = 0.5,
    strategy_name: str = "my_strategy"
) -> CostSensitivityAnalysis:
    """
    Quick cost sensitivity analysis from return series.
    
    Args:
        returns: Daily return series
        turnover: Daily turnover (0.5 = 50% of portfolio traded)
        strategy_name: Name for the strategy
    
    Returns:
        CostSensitivityAnalysis
    
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 252)  # 1 year
        >>> analysis = quick_cost_analysis(returns, turnover=0.3)
        >>> print(f"Breakeven: {analysis.breakeven_cost_bps:.1f} bps")
    """
    analyzer = TransactionCostAnalyzer()
    return analyzer.analyze_from_returns(returns, turnover, strategy_name=strategy_name)


def run_full_cost_analysis(
    backtest_results: Dict,
    output_dir: str = "./cost_analysis"
) -> CostSensitivityAnalysis:
    """
    Run full cost analysis from backtest results.
    
    Args:
        backtest_results: Dict with 'trades' DataFrame or 'returns' array
        output_dir: Directory for output files
    
    Returns:
        CostSensitivityAnalysis
    """
    analyzer = TransactionCostAnalyzer(
        cost_range=(0, 50),
        num_levels=21
    )
    
    if 'trades' in backtest_results:
        analysis = analyzer.analyze_from_trades(
            backtest_results['trades'],
            capital=backtest_results.get('capital', 100000),
            strategy_name=backtest_results.get('name', 'strategy')
        )
    elif 'returns' in backtest_results:
        analysis = analyzer.analyze_from_returns(
            backtest_results['returns'],
            turnover=backtest_results.get('turnover', 0.5),
            capital=backtest_results.get('capital', 100000),
            strategy_name=backtest_results.get('name', 'strategy')
        )
    else:
        raise ValueError("backtest_results must contain 'trades' or 'returns'")
    
    # Generate report
    report = analyzer.generate_report(analysis, output_dir)
    print(report)
    
    # Plot if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        analyzer.plot_sensitivity(
            analysis,
            output_file=f"{output_dir}/cost_sensitivity_plot.png"
        )
    
    return analysis


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate cost sensitivity analysis."""
    
    print("=" * 70)
    print("TRANSACTION COST SENSITIVITY ANALYSIS - DEMO")
    print("=" * 70)
    
    # Generate synthetic backtest returns
    np.random.seed(42)
    n_days = 252  # 1 year
    
    # Strategy with moderate edge
    daily_alpha = 0.0008  # 8 bps daily alpha
    daily_vol = 0.02
    returns = np.random.normal(daily_alpha, daily_vol, n_days)
    
    # Add some positive skew
    returns = returns + np.random.exponential(0.002, n_days) - 0.002
    
    turnover = 0.4  # 40% daily turnover
    
    print(f"\nSynthetic Strategy:")
    print(f"  Daily Alpha: {daily_alpha*100:.2f}%")
    print(f"  Daily Volatility: {daily_vol*100:.1f}%")
    print(f"  Turnover: {turnover*100:.0f}%")
    print(f"  Gross Sharpe: {np.mean(returns)/np.std(returns)*np.sqrt(252):.2f}")
    
    # Run analysis
    analyzer = TransactionCostAnalyzer(cost_range=(0, 50), num_levels=21)
    analysis = analyzer.analyze_from_returns(
        returns,
        turnover=turnover,
        strategy_name="synthetic_strategy"
    )
    
    # Print report
    report = analyzer.generate_report(analysis, "./demo_output")
    print(report)
    
    # Plot
    if MATPLOTLIB_AVAILABLE:
        analyzer.plot_sensitivity(analysis, "./demo_output/cost_sensitivity.png")
    
    return analysis


if __name__ == "__main__":
    demo()
