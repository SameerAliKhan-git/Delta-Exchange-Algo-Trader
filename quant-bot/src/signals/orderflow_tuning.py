"""
Order-Flow Confirmation Threshold Tuning
========================================

DELIVERABLE D: Tune order-flow confirmation thresholds.

Purpose:
- Grid search across order-flow thresholds
- Measure impact on win rate vs trade frequency
- Find optimal trade-off between signal quality and quantity
- Generate tuning report with recommendations

Thresholds to tune:
- min_trade_score: Minimum composite score to confirm trade
- min_delta_score: Minimum delta imbalance
- min_obi_score: Minimum order book imbalance
- min_liquidity_score: Minimum liquidity quality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
import json
from pathlib import Path


@dataclass
class TuningResult:
    """Result from threshold tuning."""
    # Thresholds
    min_trade_score: float
    min_delta_score: float
    min_obi_score: float
    min_liquidity_score: float
    
    # Performance metrics
    total_signals: int
    confirmed_signals: int
    win_rate_all: float
    win_rate_confirmed: float
    avg_pnl_all: float
    avg_pnl_confirmed: float
    
    # Trade quality
    confirmation_rate: float
    win_rate_lift: float  # % improvement
    pnl_lift: float
    
    # Efficiency
    expected_edge: float  # win_rate * avg_win - (1-win_rate) * avg_loss
    trades_per_day: float


@dataclass
class TuningReport:
    """Complete tuning report."""
    results: List[TuningResult]
    best_by_win_rate: TuningResult
    best_by_pnl: TuningResult
    best_by_edge: TuningResult
    recommended: TuningResult
    summary: str


class OrderFlowThresholdTuner:
    """
    Tune order-flow confirmation thresholds through backtesting.
    """
    
    def __init__(
        self,
        # Grid ranges
        trade_score_range: Tuple[float, float, float] = (0.3, 0.8, 0.1),
        delta_score_range: Tuple[float, float, float] = (0.2, 0.7, 0.1),
        obi_score_range: Tuple[float, float, float] = (0.2, 0.7, 0.1),
        liquidity_score_range: Tuple[float, float, float] = (0.2, 0.6, 0.1),
    ):
        # Generate grid points
        self.trade_scores = np.arange(*trade_score_range)
        self.delta_scores = np.arange(*delta_score_range)
        self.obi_scores = np.arange(*obi_score_range)
        self.liquidity_scores = np.arange(*liquidity_score_range)
        
        # Total combinations
        self.n_combinations = (
            len(self.trade_scores) * 
            len(self.delta_scores) * 
            len(self.obi_scores) * 
            len(self.liquidity_scores)
        )
    
    def _simulate_trade(
        self,
        signal_strength: float,
        orderflow_quality: float,
        base_win_rate: float = 0.52
    ) -> Tuple[bool, float]:
        """
        Simulate a trade outcome.
        
        Higher orderflow quality = higher win rate.
        """
        # Orderflow quality boosts win rate
        adjusted_win_rate = base_win_rate + (orderflow_quality - 0.5) * 0.2
        adjusted_win_rate = np.clip(adjusted_win_rate, 0.3, 0.75)
        
        won = np.random.random() < adjusted_win_rate
        
        # PnL with some variance
        if won:
            pnl = np.random.normal(1.5, 0.5) * signal_strength
        else:
            pnl = np.random.normal(-1.0, 0.3) * signal_strength
        
        return won, pnl
    
    def _generate_synthetic_signals(
        self,
        n_signals: int = 5000
    ) -> pd.DataFrame:
        """Generate synthetic trading signals with orderflow metrics."""
        np.random.seed(42)
        
        signals = []
        for _ in range(n_signals):
            # Generate signal
            signal_direction = np.random.choice([1, -1])
            signal_strength = np.random.uniform(0.3, 1.0)
            
            # Generate orderflow metrics (correlated with eventual outcome quality)
            base_quality = np.random.uniform(0.2, 0.9)
            noise = 0.15
            
            delta_score = np.clip(base_quality + np.random.normal(0, noise), 0, 1)
            obi_score = np.clip(base_quality + np.random.normal(0, noise), 0, 1)
            liquidity_score = np.clip(base_quality + np.random.normal(0, noise * 0.5), 0, 1)
            
            # Composite score
            trade_score = (
                0.4 * delta_score +
                0.3 * obi_score +
                0.2 * liquidity_score +
                0.1 * signal_strength
            )
            
            # Simulate outcome
            won, pnl = self._simulate_trade(signal_strength, trade_score)
            
            signals.append({
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'delta_score': delta_score,
                'obi_score': obi_score,
                'liquidity_score': liquidity_score,
                'trade_score': trade_score,
                'won': won,
                'pnl': pnl
            })
        
        return pd.DataFrame(signals)
    
    def run_grid_search(
        self,
        signals: pd.DataFrame = None,
        n_signals: int = 5000
    ) -> List[TuningResult]:
        """
        Run grid search over threshold combinations.
        """
        if signals is None:
            signals = self._generate_synthetic_signals(n_signals)
        
        results = []
        total_signals = len(signals)
        base_win_rate = signals['won'].mean()
        base_avg_pnl = signals['pnl'].mean()
        
        # Test each combination
        for ts, ds, obs, ls in product(
            self.trade_scores,
            self.delta_scores,
            self.obi_scores,
            self.liquidity_scores
        ):
            # Filter signals
            mask = (
                (signals['trade_score'] >= ts) &
                (signals['delta_score'] >= ds) &
                (signals['obi_score'] >= obs) &
                (signals['liquidity_score'] >= ls)
            )
            
            confirmed = signals[mask]
            n_confirmed = len(confirmed)
            
            if n_confirmed < 10:
                continue
            
            # Calculate metrics
            win_rate = confirmed['won'].mean()
            avg_pnl = confirmed['pnl'].mean()
            
            # Calculate expected edge
            avg_win = confirmed[confirmed['won']]['pnl'].mean() if confirmed['won'].sum() > 0 else 0
            avg_loss = abs(confirmed[~confirmed['won']]['pnl'].mean()) if (~confirmed['won']).sum() > 0 else 0
            expected_edge = win_rate * avg_win - (1 - win_rate) * avg_loss
            
            result = TuningResult(
                min_trade_score=ts,
                min_delta_score=ds,
                min_obi_score=obs,
                min_liquidity_score=ls,
                total_signals=total_signals,
                confirmed_signals=n_confirmed,
                win_rate_all=base_win_rate,
                win_rate_confirmed=win_rate,
                avg_pnl_all=base_avg_pnl,
                avg_pnl_confirmed=avg_pnl,
                confirmation_rate=n_confirmed / total_signals,
                win_rate_lift=(win_rate - base_win_rate) / base_win_rate * 100,
                pnl_lift=(avg_pnl - base_avg_pnl) / abs(base_avg_pnl) * 100 if base_avg_pnl != 0 else 0,
                expected_edge=expected_edge,
                trades_per_day=n_confirmed / 30  # Assuming 30 days of data
            )
            
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[TuningResult]) -> TuningReport:
        """Generate tuning report with recommendations."""
        if not results:
            raise ValueError("No results to analyze")
        
        # Find best by different metrics
        best_win_rate = max(results, key=lambda r: r.win_rate_confirmed)
        best_pnl = max(results, key=lambda r: r.avg_pnl_confirmed)
        best_edge = max(results, key=lambda r: r.expected_edge)
        
        # Find recommended (balance of edge and trade frequency)
        # Score = edge * sqrt(trades_per_day)
        def balanced_score(r: TuningResult) -> float:
            return r.expected_edge * np.sqrt(r.trades_per_day)
        
        recommended = max(results, key=balanced_score)
        
        # Generate summary
        summary_lines = [
            "=" * 60,
            "ORDER-FLOW THRESHOLD TUNING REPORT",
            "=" * 60,
            "",
            f"Total combinations tested: {len(results)}",
            "",
            "BEST BY WIN RATE:",
            f"  Thresholds: trade={best_win_rate.min_trade_score:.2f}, "
            f"delta={best_win_rate.min_delta_score:.2f}, "
            f"obi={best_win_rate.min_obi_score:.2f}, "
            f"liquidity={best_win_rate.min_liquidity_score:.2f}",
            f"  Win Rate: {best_win_rate.win_rate_confirmed:.2%} ({best_win_rate.win_rate_lift:+.1f}% lift)",
            f"  Trades/Day: {best_win_rate.trades_per_day:.1f}",
            f"  Confirmation Rate: {best_win_rate.confirmation_rate:.1%}",
            "",
            "BEST BY PNL:",
            f"  Thresholds: trade={best_pnl.min_trade_score:.2f}, "
            f"delta={best_pnl.min_delta_score:.2f}, "
            f"obi={best_pnl.min_obi_score:.2f}, "
            f"liquidity={best_pnl.min_liquidity_score:.2f}",
            f"  Avg PnL: ${best_pnl.avg_pnl_confirmed:.2f} ({best_pnl.pnl_lift:+.1f}% lift)",
            f"  Trades/Day: {best_pnl.trades_per_day:.1f}",
            "",
            "BEST BY EXPECTED EDGE:",
            f"  Thresholds: trade={best_edge.min_trade_score:.2f}, "
            f"delta={best_edge.min_delta_score:.2f}, "
            f"obi={best_edge.min_obi_score:.2f}, "
            f"liquidity={best_edge.min_liquidity_score:.2f}",
            f"  Expected Edge: ${best_edge.expected_edge:.3f} per trade",
            f"  Win Rate: {best_edge.win_rate_confirmed:.2%}",
            "",
            "=" * 60,
            "RECOMMENDED CONFIGURATION:",
            "=" * 60,
            "",
            f"  min_trade_score: {recommended.min_trade_score:.2f}",
            f"  min_delta_score: {recommended.min_delta_score:.2f}",
            f"  min_obi_score: {recommended.min_obi_score:.2f}",
            f"  min_liquidity_score: {recommended.min_liquidity_score:.2f}",
            "",
            f"  Expected Performance:",
            f"    Win Rate: {recommended.win_rate_confirmed:.2%}",
            f"    Avg PnL: ${recommended.avg_pnl_confirmed:.2f}",
            f"    Expected Edge: ${recommended.expected_edge:.3f}",
            f"    Trades/Day: {recommended.trades_per_day:.1f}",
            f"    Confirmation Rate: {recommended.confirmation_rate:.1%}",
            "",
            "=" * 60
        ]
        
        return TuningReport(
            results=results,
            best_by_win_rate=best_win_rate,
            best_by_pnl=best_pnl,
            best_by_edge=best_edge,
            recommended=recommended,
            summary="\n".join(summary_lines)
        )
    
    def export_results(self, report: TuningReport, filepath: str):
        """Export results to file."""
        results_data = []
        for r in report.results:
            results_data.append({
                'min_trade_score': r.min_trade_score,
                'min_delta_score': r.min_delta_score,
                'min_obi_score': r.min_obi_score,
                'min_liquidity_score': r.min_liquidity_score,
                'confirmed_signals': r.confirmed_signals,
                'win_rate_confirmed': r.win_rate_confirmed,
                'avg_pnl_confirmed': r.avg_pnl_confirmed,
                'expected_edge': r.expected_edge,
                'trades_per_day': r.trades_per_day,
                'confirmation_rate': r.confirmation_rate,
                'win_rate_lift': r.win_rate_lift,
                'pnl_lift': r.pnl_lift
            })
        
        df = pd.DataFrame(results_data)
        df.to_csv(filepath, index=False)
        
        # Also save summary
        summary_path = filepath.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(report.summary)


class LiveThresholdOptimizer:
    """
    Online threshold optimization using recent trade history.
    """
    
    def __init__(
        self,
        initial_thresholds: Dict[str, float] = None,
        learning_rate: float = 0.01,
        window_size: int = 500
    ):
        self.thresholds = initial_thresholds or {
            'min_trade_score': 0.5,
            'min_delta_score': 0.4,
            'min_obi_score': 0.3,
            'min_liquidity_score': 0.3
        }
        self.learning_rate = learning_rate
        self.window_size = window_size
        
        # History
        self._trade_history: List[Dict] = []
        self._threshold_history: List[Dict] = []
    
    def record_trade(
        self,
        trade_score: float,
        delta_score: float,
        obi_score: float,
        liquidity_score: float,
        confirmed: bool,
        won: bool,
        pnl: float
    ):
        """Record a trade and update thresholds."""
        self._trade_history.append({
            'timestamp': datetime.now(),
            'trade_score': trade_score,
            'delta_score': delta_score,
            'obi_score': obi_score,
            'liquidity_score': liquidity_score,
            'confirmed': confirmed,
            'won': won,
            'pnl': pnl
        })
        
        # Optimize periodically
        if len(self._trade_history) % 50 == 0 and len(self._trade_history) >= 100:
            self._optimize_thresholds()
    
    def _optimize_thresholds(self):
        """Optimize thresholds based on recent history."""
        recent = self._trade_history[-self.window_size:]
        
        # Split by outcome
        winners = [t for t in recent if t['won']]
        losers = [t for t in recent if not t['won']]
        
        if len(winners) < 10 or len(losers) < 10:
            return
        
        # Find optimal separation
        for score_type in ['trade_score', 'delta_score', 'obi_score', 'liquidity_score']:
            threshold_key = f'min_{score_type}'
            
            # Average scores for winners vs losers
            avg_winner = np.mean([t[score_type] for t in winners])
            avg_loser = np.mean([t[score_type] for t in losers])
            
            # Move threshold toward separating point
            if avg_winner > avg_loser:
                # Score is predictive - move threshold up
                target = (avg_winner + avg_loser) / 2
                self.thresholds[threshold_key] += self.learning_rate * (target - self.thresholds[threshold_key])
            else:
                # Score is not predictive - relax threshold
                self.thresholds[threshold_key] *= (1 - self.learning_rate)
            
            # Clip to valid range
            self.thresholds[threshold_key] = np.clip(self.thresholds[threshold_key], 0.1, 0.9)
        
        self._threshold_history.append({
            'timestamp': datetime.now(),
            'thresholds': self.thresholds.copy(),
            'window_win_rate': len(winners) / len(recent)
        })
    
    def should_confirm(
        self,
        trade_score: float,
        delta_score: float,
        obi_score: float,
        liquidity_score: float
    ) -> bool:
        """Check if trade should be confirmed."""
        return (
            trade_score >= self.thresholds['min_trade_score'] and
            delta_score >= self.thresholds['min_delta_score'] and
            obi_score >= self.thresholds['min_obi_score'] and
            liquidity_score >= self.thresholds['min_liquidity_score']
        )
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current threshold values."""
        return self.thresholds.copy()
    
    def get_threshold_history(self) -> pd.DataFrame:
        """Get threshold evolution over time."""
        if not self._threshold_history:
            return pd.DataFrame()
        
        records = []
        for h in self._threshold_history:
            record = {'timestamp': h['timestamp'], 'win_rate': h['window_win_rate']}
            record.update(h['thresholds'])
            records.append(record)
        
        return pd.DataFrame(records)


def run_tuning_analysis(output_dir: str = "tuning_results"):
    """Run complete tuning analysis and save results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ORDER-FLOW THRESHOLD TUNING")
    print("=" * 60)
    
    # Create tuner
    tuner = OrderFlowThresholdTuner(
        trade_score_range=(0.3, 0.85, 0.05),
        delta_score_range=(0.2, 0.75, 0.05),
        obi_score_range=(0.2, 0.75, 0.05),
        liquidity_score_range=(0.2, 0.65, 0.05)
    )
    
    print(f"\nGrid search over {tuner.n_combinations} combinations...")
    
    # Run grid search
    results = tuner.run_grid_search(n_signals=10000)
    print(f"Completed: {len(results)} valid configurations")
    
    # Generate report
    report = tuner.generate_report(results)
    print("\n" + report.summary)
    
    # Export
    tuner.export_results(report, str(output_path / "threshold_tuning.csv"))
    print(f"\nResults exported to {output_path}")
    
    return report


if __name__ == "__main__":
    report = run_tuning_analysis()
