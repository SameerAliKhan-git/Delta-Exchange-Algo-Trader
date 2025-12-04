"""
E. Sanity-Check & Metric Validation Script
==========================================
Verifies Sharpe, CAGR, Drawdown calculations and ensures no P&L leakage.

Priority: HIGHEST (do first before any other experiments)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeRecord:
    """Single trade record for ledger."""
    trade_id: int
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    holding_bars: int


class MetricValidator:
    """
    Validates trading metrics with multiple calculation methods.
    Detects common errors: look-ahead bias, incorrect annualization, P&L aggregation bugs.
    """
    
    def __init__(self, returns: pd.Series, equity_curve: pd.Series, 
                 trades: List[TradeRecord], freq: str = 'H'):
        """
        Args:
            returns: Period returns (e.g., hourly)
            equity_curve: Portfolio value over time
            trades: List of trade records
            freq: Data frequency ('D', 'H', '1min', etc.)
        """
        self.returns = returns.dropna()
        self.equity = equity_curve.dropna()
        self.trades = trades
        self.freq = freq
        
        # Annualization factors
        self.ann_factors = {
            '1min': 252 * 6.5 * 60,  # ~98,280 minutes/year
            '5min': 252 * 6.5 * 12,   # ~19,656 5-min bars
            '15min': 252 * 6.5 * 4,   # ~6,552 15-min bars
            'H': 252 * 6.5,           # ~1,638 hours
            'D': 252,                 # 252 trading days
            'W': 52,                  # 52 weeks
            'M': 12                   # 12 months
        }
        self.ann_factor = self.ann_factors.get(freq, 252)
    
    def validate_sharpe_ratio(self) -> Dict:
        """
        Calculate Sharpe ratio using multiple methods and compare.
        """
        results = {}
        
        # Method 1: Standard formula
        mean_ret = self.returns.mean()
        std_ret = self.returns.std(ddof=1)
        sharpe_1 = (mean_ret / std_ret) * np.sqrt(self.ann_factor) if std_ret > 0 else 0
        results['standard'] = sharpe_1
        
        # Method 2: Using log returns
        log_returns = np.log1p(self.returns)
        mean_log = log_returns.mean()
        std_log = log_returns.std(ddof=1)
        sharpe_2 = (mean_log / std_log) * np.sqrt(self.ann_factor) if std_log > 0 else 0
        results['log_returns'] = sharpe_2
        
        # Method 3: Downside deviation (Sortino-style but comparable)
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else std_ret
        sharpe_3 = (mean_ret / downside_std) * np.sqrt(self.ann_factor) if downside_std > 0 else 0
        results['sortino_style'] = sharpe_3
        
        # Method 4: Rolling window check (should be consistent)
        rolling_sharpe = []
        window = min(100, len(self.returns) // 4)
        if window > 10:
            for i in range(window, len(self.returns), window):
                chunk = self.returns.iloc[i-window:i]
                if chunk.std() > 0:
                    rolling_sharpe.append((chunk.mean() / chunk.std()) * np.sqrt(self.ann_factor))
        results['rolling_mean'] = np.mean(rolling_sharpe) if rolling_sharpe else sharpe_1
        results['rolling_std'] = np.std(rolling_sharpe) if rolling_sharpe else 0
        
        # Check for consistency
        sharpes = [sharpe_1, sharpe_2]
        results['max_discrepancy'] = max(sharpes) - min(sharpes)
        results['warning'] = results['max_discrepancy'] > 0.5
        
        return results
    
    def validate_cagr(self) -> Dict:
        """
        Calculate CAGR using multiple methods.
        """
        results = {}
        
        if len(self.equity) < 2:
            return {'error': 'Insufficient data'}
        
        start_val = self.equity.iloc[0]
        end_val = self.equity.iloc[-1]
        
        # Calculate time period in years
        if isinstance(self.equity.index, pd.DatetimeIndex):
            years = (self.equity.index[-1] - self.equity.index[0]).days / 365.25
        else:
            years = len(self.equity) / self.ann_factor
        
        years = max(years, 1/365)  # Avoid division by zero
        
        # Method 1: Standard CAGR
        total_return = end_val / start_val - 1
        cagr_1 = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        results['standard'] = cagr_1
        
        # Method 2: From log returns
        log_return = np.log(end_val / start_val)
        cagr_2 = np.exp(log_return / years) - 1 if years > 0 else 0
        results['log_based'] = cagr_2
        
        # Method 3: Annualized from period returns
        cumulative_ret = (1 + self.returns).prod() - 1
        cagr_3 = (1 + cumulative_ret) ** (1/years) - 1 if years > 0 else 0
        results['from_returns'] = cagr_3
        
        # Sanity checks
        results['total_return'] = total_return
        results['years'] = years
        results['max_discrepancy'] = max(cagr_1, cagr_2, cagr_3) - min(cagr_1, cagr_2, cagr_3)
        results['warning'] = abs(cagr_1 - cagr_3) > 0.01
        
        return results
    
    def validate_drawdown(self) -> Dict:
        """
        Calculate drawdown metrics and validate.
        """
        results = {}
        
        # Running maximum
        running_max = self.equity.cummax()
        drawdown = (self.equity - running_max) / running_max
        
        results['max_drawdown'] = drawdown.min()
        results['max_drawdown_idx'] = drawdown.idxmin()
        
        # Find drawdown duration
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            # Count consecutive drawdown periods
            dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
            dd_durations = in_drawdown.groupby(dd_groups).sum()
            results['max_dd_duration'] = dd_durations.max()
            results['avg_dd_duration'] = dd_durations[dd_durations > 0].mean()
        else:
            results['max_dd_duration'] = 0
            results['avg_dd_duration'] = 0
        
        # Calmar ratio check
        cagr = self.validate_cagr().get('standard', 0)
        max_dd = abs(results['max_drawdown'])
        results['calmar_ratio'] = cagr / max_dd if max_dd > 0 else 0
        
        # Drawdown statistics
        results['current_drawdown'] = drawdown.iloc[-1]
        results['avg_drawdown'] = drawdown.mean()
        
        # Warning if equity curve seems wrong
        results['warning'] = results['max_drawdown'] > -0.001 and len(self.equity) > 100
        
        return results
    
    def validate_pnl_chronology(self) -> Dict:
        """
        Check for P&L aggregation bugs and look-ahead bias.
        """
        results = {}
        
        if not self.trades:
            return {'error': 'No trades to analyze'}
        
        # Sort trades by exit time
        sorted_trades = sorted(self.trades, key=lambda t: t.exit_time)
        
        # Check for chronological consistency
        issues = []
        cumulative_pnl = 0
        for i, trade in enumerate(sorted_trades):
            # Check entry before exit
            if trade.entry_time >= trade.exit_time:
                issues.append(f"Trade {trade.trade_id}: entry >= exit time")
            
            # Check P&L calculation
            if trade.direction == 'long':
                expected_gross = (trade.exit_price - trade.entry_price) * trade.size
            else:
                expected_gross = (trade.entry_price - trade.exit_price) * trade.size
            
            if abs(expected_gross - trade.gross_pnl) > 0.01:
                issues.append(f"Trade {trade.trade_id}: P&L mismatch (expected {expected_gross:.2f}, got {trade.gross_pnl:.2f})")
            
            # Check net = gross - costs
            expected_net = trade.gross_pnl - trade.commission - trade.slippage
            if abs(expected_net - trade.net_pnl) > 0.01:
                issues.append(f"Trade {trade.trade_id}: Net P&L mismatch")
            
            cumulative_pnl += trade.net_pnl
        
        results['chronology_issues'] = issues
        results['total_trades'] = len(sorted_trades)
        results['cumulative_pnl_from_trades'] = cumulative_pnl
        
        # Compare with equity curve
        if len(self.equity) > 1:
            equity_pnl = self.equity.iloc[-1] - self.equity.iloc[0]
            results['equity_curve_pnl'] = equity_pnl
            results['pnl_discrepancy'] = abs(cumulative_pnl - equity_pnl)
            results['warning'] = results['pnl_discrepancy'] > 1.0
        
        return results
    
    def generate_trade_ledger(self) -> pd.DataFrame:
        """
        Generate detailed trade ledger for inspection.
        """
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        cumulative_pnl = 0
        
        for trade in sorted(self.trades, key=lambda t: t.exit_time):
            cumulative_pnl += trade.net_pnl
            records.append({
                'trade_id': trade.trade_id,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'gross_pnl': trade.gross_pnl,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'net_pnl': trade.net_pnl,
                'cumulative_pnl': cumulative_pnl,
                'holding_bars': trade.holding_bars,
                'return_pct': (trade.net_pnl / (trade.entry_price * trade.size)) * 100 if trade.size > 0 else 0
            })
        
        return pd.DataFrame(records)
    
    def analyze_worst_trades(self, n: int = 10) -> pd.DataFrame:
        """
        Analyze top losing trades for debugging.
        """
        ledger = self.generate_trade_ledger()
        if ledger.empty:
            return ledger
        
        worst = ledger.nsmallest(n, 'net_pnl')
        return worst
    
    def run_null_model_test(self, n_simulations: int = 100) -> Dict:
        """
        Run null model test: randomize labels and compare metrics.
        If real metrics are similar to null, you're likely overfitting noise.
        """
        if len(self.returns) < 50:
            return {'error': 'Insufficient data for null test'}
        
        real_sharpe = self.validate_sharpe_ratio()['standard']
        real_cagr = self.validate_cagr()['standard']
        
        null_sharpes = []
        null_cagrs = []
        
        for _ in range(n_simulations):
            # Shuffle returns to break any real signal
            shuffled = self.returns.sample(frac=1, replace=False)
            shuffled.index = self.returns.index
            
            # Calculate metrics on shuffled
            mean_ret = shuffled.mean()
            std_ret = shuffled.std(ddof=1)
            null_sharpe = (mean_ret / std_ret) * np.sqrt(self.ann_factor) if std_ret > 0 else 0
            null_sharpes.append(null_sharpe)
            
            # CAGR from shuffled
            cumret = (1 + shuffled).prod() - 1
            years = len(shuffled) / self.ann_factor
            null_cagr = (1 + cumret) ** (1/max(years, 0.01)) - 1
            null_cagrs.append(null_cagr)
        
        results = {
            'real_sharpe': real_sharpe,
            'null_sharpe_mean': np.mean(null_sharpes),
            'null_sharpe_std': np.std(null_sharpes),
            'sharpe_percentile': (np.array(null_sharpes) < real_sharpe).mean() * 100,
            'real_cagr': real_cagr,
            'null_cagr_mean': np.mean(null_cagrs),
            'null_cagr_std': np.std(null_cagrs),
            'cagr_percentile': (np.array(null_cagrs) < real_cagr).mean() * 100,
        }
        
        # Warning if real is not significantly better than null
        results['warning'] = results['sharpe_percentile'] < 95
        results['interpretation'] = (
            "LIKELY OVERFITTING" if results['sharpe_percentile'] < 80 else
            "MARGINAL EDGE" if results['sharpe_percentile'] < 95 else
            "STATISTICALLY SIGNIFICANT"
        )
        
        return results
    
    def compute_turnover_metrics(self) -> Dict:
        """
        Calculate turnover and its impact on returns.
        """
        if not self.trades:
            return {'error': 'No trades'}
        
        total_volume = sum(abs(t.size * t.entry_price) for t in self.trades)
        avg_equity = self.equity.mean()
        
        # Annualized turnover
        if isinstance(self.equity.index, pd.DatetimeIndex):
            days = (self.equity.index[-1] - self.equity.index[0]).days
            years = max(days / 365.25, 1/365)
        else:
            years = len(self.equity) / self.ann_factor
        
        turnover_annual = (total_volume / avg_equity) / years if years > 0 and avg_equity > 0 else 0
        
        # Cost analysis
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        total_costs = total_commission + total_slippage
        
        gross_pnl = sum(t.gross_pnl for t in self.trades)
        net_pnl = sum(t.net_pnl for t in self.trades)
        
        return {
            'total_trades': len(self.trades),
            'total_volume': total_volume,
            'avg_trade_size': total_volume / len(self.trades),
            'turnover_annual': turnover_annual,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_costs': total_costs,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'cost_drag_pct': (total_costs / abs(gross_pnl)) * 100 if gross_pnl != 0 else 0,
            'avg_cost_per_trade': total_costs / len(self.trades)
        }
    
    def full_validation_report(self) -> Dict:
        """
        Run all validations and generate comprehensive report.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(self.returns),
            'frequency': self.freq,
            'annualization_factor': self.ann_factor,
        }
        
        report['sharpe_validation'] = self.validate_sharpe_ratio()
        report['cagr_validation'] = self.validate_cagr()
        report['drawdown_validation'] = self.validate_drawdown()
        report['pnl_validation'] = self.validate_pnl_chronology()
        report['turnover_metrics'] = self.compute_turnover_metrics()
        report['null_model_test'] = self.run_null_model_test()
        
        # Summary warnings
        warnings = []
        if report['sharpe_validation'].get('warning'):
            warnings.append("Sharpe calculation discrepancy detected")
        if report['cagr_validation'].get('warning'):
            warnings.append("CAGR calculation discrepancy detected")
        if report['drawdown_validation'].get('warning'):
            warnings.append("Drawdown calculation may be incorrect")
        if report['pnl_validation'].get('warning'):
            warnings.append("P&L discrepancy between trades and equity curve")
        if report['null_model_test'].get('warning'):
            warnings.append(f"Null model test: {report['null_model_test'].get('interpretation', 'UNKNOWN')}")
        
        report['warnings'] = warnings
        report['has_issues'] = len(warnings) > 0
        
        return report


def print_validation_report(report: Dict):
    """Pretty print validation report."""
    print("\n" + "="*70)
    print("METRIC VALIDATION REPORT")
    print("="*70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Data points: {report['data_points']}")
    print(f"Frequency: {report['frequency']} (Annualization: {report['annualization_factor']})")
    
    # Sharpe
    print("\n" + "-"*70)
    print("SHARPE RATIO VALIDATION")
    print("-"*70)
    sharpe = report['sharpe_validation']
    print(f"  Standard:      {sharpe.get('standard', 'N/A'):.4f}")
    print(f"  Log returns:   {sharpe.get('log_returns', 'N/A'):.4f}")
    print(f"  Sortino-style: {sharpe.get('sortino_style', 'N/A'):.4f}")
    print(f"  Rolling mean:  {sharpe.get('rolling_mean', 'N/A'):.4f} ± {sharpe.get('rolling_std', 0):.4f}")
    print(f"  Max discrepancy: {sharpe.get('max_discrepancy', 0):.4f}")
    if sharpe.get('warning'):
        print("  ⚠️  WARNING: Large discrepancy between methods!")
    
    # CAGR
    print("\n" + "-"*70)
    print("CAGR VALIDATION")
    print("-"*70)
    cagr = report['cagr_validation']
    print(f"  Standard:     {cagr.get('standard', 'N/A')*100:.2f}%")
    print(f"  Log-based:    {cagr.get('log_based', 'N/A')*100:.2f}%")
    print(f"  From returns: {cagr.get('from_returns', 'N/A')*100:.2f}%")
    print(f"  Total return: {cagr.get('total_return', 0)*100:.2f}%")
    print(f"  Time period:  {cagr.get('years', 0):.2f} years")
    if cagr.get('warning'):
        print("  ⚠️  WARNING: CAGR calculation discrepancy!")
    
    # Drawdown
    print("\n" + "-"*70)
    print("DRAWDOWN VALIDATION")
    print("-"*70)
    dd = report['drawdown_validation']
    print(f"  Max drawdown:      {dd.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Max DD duration:   {dd.get('max_dd_duration', 0)} bars")
    print(f"  Avg DD duration:   {dd.get('avg_dd_duration', 0):.1f} bars")
    print(f"  Current drawdown:  {dd.get('current_drawdown', 0)*100:.2f}%")
    print(f"  Calmar ratio:      {dd.get('calmar_ratio', 0):.2f}")
    
    # P&L Validation
    print("\n" + "-"*70)
    print("P&L CHRONOLOGY VALIDATION")
    print("-"*70)
    pnl = report['pnl_validation']
    if 'error' not in pnl:
        print(f"  Total trades:         {pnl.get('total_trades', 0)}")
        print(f"  P&L from trades:      ${pnl.get('cumulative_pnl_from_trades', 0):,.2f}")
        print(f"  P&L from equity:      ${pnl.get('equity_curve_pnl', 0):,.2f}")
        print(f"  Discrepancy:          ${pnl.get('pnl_discrepancy', 0):,.2f}")
        if pnl.get('chronology_issues'):
            print(f"  ⚠️  Issues found: {len(pnl['chronology_issues'])}")
            for issue in pnl['chronology_issues'][:5]:
                print(f"      - {issue}")
    else:
        print(f"  {pnl['error']}")
    
    # Turnover
    print("\n" + "-"*70)
    print("TURNOVER & COST ANALYSIS")
    print("-"*70)
    turn = report['turnover_metrics']
    if 'error' not in turn:
        print(f"  Total trades:       {turn.get('total_trades', 0)}")
        print(f"  Annual turnover:    {turn.get('turnover_annual', 0):.1f}x")
        print(f"  Total commission:   ${turn.get('total_commission', 0):,.2f}")
        print(f"  Total slippage:     ${turn.get('total_slippage', 0):,.2f}")
        print(f"  Gross P&L:          ${turn.get('gross_pnl', 0):,.2f}")
        print(f"  Net P&L:            ${turn.get('net_pnl', 0):,.2f}")
        print(f"  Cost drag:          {turn.get('cost_drag_pct', 0):.1f}%")
    
    # Null Model Test
    print("\n" + "-"*70)
    print("NULL MODEL TEST (Overfitting Detection)")
    print("-"*70)
    null = report['null_model_test']
    if 'error' not in null:
        print(f"  Real Sharpe:        {null.get('real_sharpe', 0):.4f}")
        print(f"  Null Sharpe:        {null.get('null_sharpe_mean', 0):.4f} ± {null.get('null_sharpe_std', 0):.4f}")
        print(f"  Sharpe percentile:  {null.get('sharpe_percentile', 0):.1f}%")
        print(f"  Real CAGR:          {null.get('real_cagr', 0)*100:.2f}%")
        print(f"  Null CAGR:          {null.get('null_cagr_mean', 0)*100:.2f}% ± {null.get('null_cagr_std', 0)*100:.2f}%")
        print(f"  CAGR percentile:    {null.get('cagr_percentile', 0):.1f}%")
        print(f"\n  Interpretation:     {null.get('interpretation', 'UNKNOWN')}")
        if null.get('warning'):
            print("  ⚠️  WARNING: Results may be due to overfitting!")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if report['warnings']:
        print("⚠️  WARNINGS DETECTED:")
        for w in report['warnings']:
            print(f"   - {w}")
    else:
        print("✅ All validations passed!")
    print("="*70)


# =============================================================================
# DEMO / TEST
# =============================================================================

def create_sample_data_for_validation():
    """Create sample data to test the validator."""
    np.random.seed(42)
    n_periods = 2000
    
    # Generate returns with small positive drift
    returns = pd.Series(
        np.random.normal(0.0002, 0.015, n_periods),
        index=pd.date_range('2023-01-01', periods=n_periods, freq='H')
    )
    
    # Generate equity curve
    equity = pd.Series(
        100000 * (1 + returns).cumprod(),
        index=returns.index
    )
    
    # Generate sample trades
    trades = []
    cumulative_pnl = 0
    
    for i in range(50):
        entry_idx = i * 30 + np.random.randint(0, 10)
        exit_idx = entry_idx + np.random.randint(5, 25)
        
        if exit_idx >= len(returns):
            break
        
        entry_time = returns.index[entry_idx]
        exit_time = returns.index[exit_idx]
        
        direction = 'long' if np.random.random() > 0.5 else 'short'
        entry_price = 100 + np.random.randn() * 5
        price_move = np.random.randn() * 2
        exit_price = entry_price + price_move if direction == 'long' else entry_price - price_move
        size = 100
        
        if direction == 'long':
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size
        
        commission = entry_price * size * 0.001 * 2  # 0.1% each way
        slippage = entry_price * size * 0.0005 * 2   # 0.05% each way
        net_pnl = gross_pnl - commission - slippage
        
        trade = TradeRecord(
            trade_id=i+1,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=slippage,
            net_pnl=net_pnl,
            holding_bars=exit_idx - entry_idx
        )
        trades.append(trade)
    
    return returns, equity, trades


def main():
    """Run sanity check validation."""
    print("="*70)
    print("SANITY CHECK: Metric Validation Script")
    print("="*70)
    
    # Create sample data
    print("\nGenerating sample data for validation...")
    returns, equity, trades = create_sample_data_for_validation()
    
    print(f"  Returns: {len(returns)} periods")
    print(f"  Equity range: ${equity.min():,.2f} - ${equity.max():,.2f}")
    print(f"  Trades: {len(trades)}")
    
    # Create validator
    validator = MetricValidator(returns, equity, trades, freq='H')
    
    # Run full validation
    print("\nRunning validation...")
    report = validator.full_validation_report()
    
    # Print report
    print_validation_report(report)
    
    # Generate trade ledger
    print("\n" + "-"*70)
    print("TOP 10 WORST TRADES")
    print("-"*70)
    worst = validator.analyze_worst_trades(10)
    if not worst.empty:
        print(worst[['trade_id', 'direction', 'entry_price', 'exit_price', 
                     'gross_pnl', 'commission', 'net_pnl', 'return_pct']].to_string())
    
    print("\n✅ Sanity check complete!")
    print("\nNext steps:")
    print("  1. Replace sample data with your actual backtest results")
    print("  2. Fix any warnings before proceeding with optimization")
    print("  3. Run this script after every major change to your backtest")


if __name__ == "__main__":
    main()
