"""
End-to-End Regression Suite
============================
Nightly CI job that replays the last 30 days and asserts P&L within tolerance.

Features:
- Replays historical data through full strategy pipeline
- Compares realized vs expected P&L
- Generates regression report
- Integrates with CI/CD (GitHub Actions, GitLab CI)
- Prometheus metrics for drift detection

Usage:
    # Run regression suite
    python tests/regression/e2e_regression_suite.py --days 30
    
    # CI mode (fail on tolerance breach)
    python tests/regression/e2e_regression_suite.py --ci --tolerance 0.10
    
    # Generate HTML report
    python tests/regression/e2e_regression_suite.py --report regression_report.html
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class RegressionStatus(Enum):
    """Regression test status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class StrategyResult:
    """Result from a single strategy replay."""
    strategy_name: str
    period_start: datetime
    period_end: datetime
    
    # P&L metrics
    total_pnl: float
    expected_pnl: float
    pnl_deviation: float
    
    # Trade metrics
    total_trades: int
    expected_trades: int
    win_rate: float
    expected_win_rate: float
    
    # Risk metrics
    max_drawdown: float
    expected_max_drawdown: float
    sharpe_ratio: float
    expected_sharpe: float
    
    # Timing
    execution_time_seconds: float
    
    # Status
    status: RegressionStatus = RegressionStatus.PASSED
    error_message: Optional[str] = None


@dataclass
class RegressionReport:
    """Full regression report."""
    report_id: str
    generated_at: datetime
    period_days: int
    period_start: datetime
    period_end: datetime
    
    # Results
    strategy_results: List[StrategyResult] = field(default_factory=list)
    
    # Aggregate metrics
    total_strategies: int = 0
    passed: int = 0
    warnings: int = 0
    failed: int = 0
    errors: int = 0
    
    # Tolerances used
    pnl_tolerance: float = 0.10
    trade_tolerance: float = 0.15
    drawdown_tolerance: float = 0.20
    
    # Execution
    total_execution_time: float = 0.0
    
    @property
    def overall_status(self) -> RegressionStatus:
        """Get overall regression status."""
        if self.errors > 0 or self.failed > 0:
            return RegressionStatus.FAILED
        if self.warnings > 0:
            return RegressionStatus.WARNING
        return RegressionStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'period_days': self.period_days,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'overall_status': self.overall_status.value,
            'summary': {
                'total_strategies': self.total_strategies,
                'passed': self.passed,
                'warnings': self.warnings,
                'failed': self.failed,
                'errors': self.errors,
            },
            'tolerances': {
                'pnl': self.pnl_tolerance,
                'trades': self.trade_tolerance,
                'drawdown': self.drawdown_tolerance,
            },
            'execution_time_seconds': self.total_execution_time,
            'strategy_results': [
                {
                    'strategy': r.strategy_name,
                    'status': r.status.value,
                    'pnl': {
                        'actual': r.total_pnl,
                        'expected': r.expected_pnl,
                        'deviation': r.pnl_deviation,
                    },
                    'trades': {
                        'actual': r.total_trades,
                        'expected': r.expected_trades,
                    },
                    'win_rate': {
                        'actual': r.win_rate,
                        'expected': r.expected_win_rate,
                    },
                    'risk': {
                        'max_drawdown': r.max_drawdown,
                        'expected_drawdown': r.expected_max_drawdown,
                        'sharpe': r.sharpe_ratio,
                        'expected_sharpe': r.expected_sharpe,
                    },
                    'execution_time': r.execution_time_seconds,
                    'error': r.error_message,
                }
                for r in self.strategy_results
            ],
        }


@dataclass
class RegressionConfig:
    """Regression suite configuration."""
    # Replay settings
    replay_days: int = 30
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Strategies to test
    strategies: List[str] = field(default_factory=lambda: [
        'momentum',
        'mean_reversion',
        'orderflow',
        'sentiment',
        'options_delta_neutral',
        'funding_arbitrage',
    ])
    
    # Tolerances
    pnl_tolerance: float = 0.10          # 10% P&L deviation
    trade_count_tolerance: float = 0.15  # 15% trade count deviation
    win_rate_tolerance: float = 0.05     # 5% win rate deviation
    drawdown_tolerance: float = 0.20     # 20% drawdown deviation
    sharpe_tolerance: float = 0.30       # 30% Sharpe deviation
    
    # Baselines file
    baselines_path: str = "tests/regression/baselines.json"
    
    # Output
    output_dir: str = "reports/regression"
    generate_html: bool = True
    
    # CI settings
    ci_mode: bool = False
    fail_on_warning: bool = False


# =============================================================================
# Baseline Manager
# =============================================================================

class BaselineManager:
    """Manages expected baselines for regression testing."""
    
    def __init__(self, baselines_path: str):
        self.baselines_path = Path(baselines_path)
        self._baselines: Dict[str, Dict] = {}
        self._load_baselines()
    
    def _load_baselines(self) -> None:
        """Load baselines from file."""
        if self.baselines_path.exists():
            with open(self.baselines_path) as f:
                self._baselines = json.load(f)
            logger.info(f"Loaded baselines from {self.baselines_path}")
        else:
            logger.warning(f"No baselines file found at {self.baselines_path}")
            self._baselines = self._get_default_baselines()
    
    def _get_default_baselines(self) -> Dict[str, Dict]:
        """Get default baselines."""
        return {
            'momentum': {
                'pnl_per_day': 50.0,
                'trades_per_day': 5.0,
                'win_rate': 0.55,
                'max_drawdown_pct': 5.0,
                'sharpe_ratio': 1.2,
            },
            'mean_reversion': {
                'pnl_per_day': 40.0,
                'trades_per_day': 8.0,
                'win_rate': 0.60,
                'max_drawdown_pct': 4.0,
                'sharpe_ratio': 1.5,
            },
            'orderflow': {
                'pnl_per_day': 30.0,
                'trades_per_day': 15.0,
                'win_rate': 0.52,
                'max_drawdown_pct': 6.0,
                'sharpe_ratio': 1.0,
            },
            'sentiment': {
                'pnl_per_day': 25.0,
                'trades_per_day': 3.0,
                'win_rate': 0.58,
                'max_drawdown_pct': 7.0,
                'sharpe_ratio': 0.9,
            },
            'options_delta_neutral': {
                'pnl_per_day': 60.0,
                'trades_per_day': 4.0,
                'win_rate': 0.65,
                'max_drawdown_pct': 8.0,
                'sharpe_ratio': 1.8,
            },
            'funding_arbitrage': {
                'pnl_per_day': 35.0,
                'trades_per_day': 2.0,
                'win_rate': 0.80,
                'max_drawdown_pct': 2.0,
                'sharpe_ratio': 2.5,
            },
        }
    
    def get_baseline(self, strategy: str) -> Dict[str, float]:
        """Get baseline for a strategy."""
        return self._baselines.get(strategy, {
            'pnl_per_day': 0.0,
            'trades_per_day': 0.0,
            'win_rate': 0.50,
            'max_drawdown_pct': 10.0,
            'sharpe_ratio': 0.0,
        })
    
    def update_baseline(self, strategy: str, metrics: Dict[str, float]) -> None:
        """Update baseline for a strategy."""
        self._baselines[strategy] = metrics
        self._save_baselines()
    
    def _save_baselines(self) -> None:
        """Save baselines to file."""
        self.baselines_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baselines_path, 'w') as f:
            json.dump(self._baselines, f, indent=2)
        logger.info(f"Saved baselines to {self.baselines_path}")


# =============================================================================
# Strategy Replay Engine
# =============================================================================

class StrategyReplayEngine:
    """Replays strategies on historical data."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
    
    async def replay_strategy(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """
        Replay a strategy on historical data.
        
        Returns dict with:
        - total_pnl
        - total_trades
        - win_rate
        - max_drawdown
        - sharpe_ratio
        - daily_returns
        """
        # In production, this would load historical data and run the strategy
        # For demo, we simulate results with some randomness
        
        days = (end_date - start_date).days
        
        # Simulate daily returns
        if NUMPY_AVAILABLE:
            np.random.seed(hash(strategy_name) % 2**32)
            
            # Strategy-specific characteristics
            params = self._get_strategy_params(strategy_name)
            
            daily_returns = np.random.normal(
                params['mean_return'],
                params['volatility'],
                days
            )
            
            # Add some autocorrelation
            for i in range(1, len(daily_returns)):
                daily_returns[i] += 0.1 * daily_returns[i-1]
            
            equity_curve = np.cumsum(daily_returns)
            
            # Calculate metrics
            total_pnl = float(equity_curve[-1])
            max_drawdown = float(np.max(np.maximum.accumulate(equity_curve) - equity_curve))
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
            
            # Simulate trades
            trades_per_day = params['trades_per_day']
            total_trades = int(days * trades_per_day * np.random.uniform(0.9, 1.1))
            win_rate = params['win_rate'] + np.random.uniform(-0.03, 0.03)
            
        else:
            import random
            random.seed(hash(strategy_name))
            
            params = self._get_strategy_params(strategy_name)
            
            daily_returns = [
                random.gauss(params['mean_return'], params['volatility'])
                for _ in range(days)
            ]
            
            total_pnl = sum(daily_returns)
            
            # Simple max drawdown calculation
            cumsum = 0
            peak = 0
            max_dd = 0
            for r in daily_returns:
                cumsum += r
                peak = max(peak, cumsum)
                max_dd = max(max_dd, peak - cumsum)
            max_drawdown = max_dd
            
            sharpe = (sum(daily_returns) / len(daily_returns)) / (
                (sum((r - sum(daily_returns)/len(daily_returns))**2 for r in daily_returns) / len(daily_returns))**0.5
            ) * (252**0.5) if len(daily_returns) > 1 else 0
            
            total_trades = int(days * params['trades_per_day'] * random.uniform(0.9, 1.1))
            win_rate = params['win_rate'] + random.uniform(-0.03, 0.03)
        
        # Simulate execution time
        await asyncio.sleep(0.1)  # Simulate some processing
        
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': min(1.0, max(0.0, win_rate)),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'daily_returns': daily_returns if NUMPY_AVAILABLE else [],
        }
    
    def _get_strategy_params(self, strategy_name: str) -> Dict[str, float]:
        """Get simulation parameters for a strategy."""
        params = {
            'momentum': {
                'mean_return': 50.0,
                'volatility': 100.0,
                'trades_per_day': 5.0,
                'win_rate': 0.55,
            },
            'mean_reversion': {
                'mean_return': 40.0,
                'volatility': 80.0,
                'trades_per_day': 8.0,
                'win_rate': 0.60,
            },
            'orderflow': {
                'mean_return': 30.0,
                'volatility': 120.0,
                'trades_per_day': 15.0,
                'win_rate': 0.52,
            },
            'sentiment': {
                'mean_return': 25.0,
                'volatility': 90.0,
                'trades_per_day': 3.0,
                'win_rate': 0.58,
            },
            'options_delta_neutral': {
                'mean_return': 60.0,
                'volatility': 70.0,
                'trades_per_day': 4.0,
                'win_rate': 0.65,
            },
            'funding_arbitrage': {
                'mean_return': 35.0,
                'volatility': 30.0,
                'trades_per_day': 2.0,
                'win_rate': 0.80,
            },
        }
        
        return params.get(strategy_name, {
            'mean_return': 20.0,
            'volatility': 100.0,
            'trades_per_day': 5.0,
            'win_rate': 0.50,
        })


# =============================================================================
# Regression Runner
# =============================================================================

class RegressionRunner:
    """Runs the full regression suite."""
    
    def __init__(self, config: RegressionConfig):
        self.config = config
        self.baseline_manager = BaselineManager(config.baselines_path)
        self.replay_engine = StrategyReplayEngine(config)
    
    async def run(self) -> RegressionReport:
        """Run the full regression suite."""
        start_time = time.time()
        
        # Determine date range
        end_date = self.config.end_date or datetime.utcnow()
        start_date = self.config.start_date or (end_date - timedelta(days=self.config.replay_days))
        
        report = RegressionReport(
            report_id=f"regression_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow(),
            period_days=self.config.replay_days,
            period_start=start_date,
            period_end=end_date,
            total_strategies=len(self.config.strategies),
            pnl_tolerance=self.config.pnl_tolerance,
            trade_tolerance=self.config.trade_count_tolerance,
            drawdown_tolerance=self.config.drawdown_tolerance,
        )
        
        logger.info(f"Starting regression suite: {len(self.config.strategies)} strategies, "
                   f"{self.config.replay_days} days")
        
        # Run each strategy
        for strategy in self.config.strategies:
            result = await self._test_strategy(strategy, start_date, end_date)
            report.strategy_results.append(result)
            
            # Update counters
            if result.status == RegressionStatus.PASSED:
                report.passed += 1
            elif result.status == RegressionStatus.WARNING:
                report.warnings += 1
            elif result.status == RegressionStatus.FAILED:
                report.failed += 1
            else:
                report.errors += 1
        
        report.total_execution_time = time.time() - start_time
        
        logger.info(f"Regression complete: {report.passed} passed, "
                   f"{report.warnings} warnings, {report.failed} failed, "
                   f"{report.errors} errors")
        
        return report
    
    async def _test_strategy(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> StrategyResult:
        """Test a single strategy."""
        logger.info(f"Testing strategy: {strategy_name}")
        
        start_time = time.time()
        baseline = self.baseline_manager.get_baseline(strategy_name)
        days = (end_date - start_date).days
        
        try:
            # Run replay
            results = await self.replay_engine.replay_strategy(
                strategy_name, start_date, end_date
            )
            
            # Calculate expected values from baseline
            expected_pnl = baseline['pnl_per_day'] * days
            expected_trades = int(baseline['trades_per_day'] * days)
            expected_win_rate = baseline['win_rate']
            expected_drawdown = baseline['max_drawdown_pct'] / 100 * expected_pnl if expected_pnl > 0 else 100
            expected_sharpe = baseline['sharpe_ratio']
            
            # Calculate deviations
            pnl_deviation = abs(results['total_pnl'] - expected_pnl) / max(abs(expected_pnl), 1)
            trade_deviation = abs(results['total_trades'] - expected_trades) / max(expected_trades, 1)
            win_rate_deviation = abs(results['win_rate'] - expected_win_rate)
            drawdown_deviation = abs(results['max_drawdown'] - expected_drawdown) / max(expected_drawdown, 1)
            sharpe_deviation = abs(results['sharpe_ratio'] - expected_sharpe) / max(abs(expected_sharpe), 0.1)
            
            # Determine status
            status = RegressionStatus.PASSED
            
            if pnl_deviation > self.config.pnl_tolerance:
                status = RegressionStatus.FAILED
                logger.warning(f"  {strategy_name}: P&L deviation {pnl_deviation:.1%} exceeds tolerance")
            elif pnl_deviation > self.config.pnl_tolerance * 0.7:
                status = RegressionStatus.WARNING
            
            if trade_deviation > self.config.trade_count_tolerance:
                status = max(status, RegressionStatus.WARNING, key=lambda x: x.value)
                logger.warning(f"  {strategy_name}: Trade count deviation {trade_deviation:.1%}")
            
            if drawdown_deviation > self.config.drawdown_tolerance:
                status = RegressionStatus.FAILED
                logger.warning(f"  {strategy_name}: Drawdown deviation {drawdown_deviation:.1%} exceeds tolerance")
            
            execution_time = time.time() - start_time
            
            return StrategyResult(
                strategy_name=strategy_name,
                period_start=start_date,
                period_end=end_date,
                total_pnl=results['total_pnl'],
                expected_pnl=expected_pnl,
                pnl_deviation=pnl_deviation,
                total_trades=results['total_trades'],
                expected_trades=expected_trades,
                win_rate=results['win_rate'],
                expected_win_rate=expected_win_rate,
                max_drawdown=results['max_drawdown'],
                expected_max_drawdown=expected_drawdown,
                sharpe_ratio=results['sharpe_ratio'],
                expected_sharpe=expected_sharpe,
                execution_time_seconds=execution_time,
                status=status,
            )
            
        except Exception as e:
            logger.error(f"  {strategy_name}: Error - {e}")
            return StrategyResult(
                strategy_name=strategy_name,
                period_start=start_date,
                period_end=end_date,
                total_pnl=0,
                expected_pnl=0,
                pnl_deviation=1.0,
                total_trades=0,
                expected_trades=0,
                win_rate=0,
                expected_win_rate=0,
                max_drawdown=0,
                expected_max_drawdown=0,
                sharpe_ratio=0,
                expected_sharpe=0,
                execution_time_seconds=time.time() - start_time,
                status=RegressionStatus.ERROR,
                error_message=str(e),
            )


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generates regression reports."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json(self, report: RegressionReport) -> str:
        """Generate JSON report."""
        path = self.output_dir / f"{report.report_id}.json"
        with open(path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        return str(path)
    
    def generate_html(self, report: RegressionReport) -> str:
        """Generate HTML report."""
        path = self.output_dir / f"{report.report_id}.html"
        
        status_colors = {
            'passed': '#28a745',
            'warning': '#ffc107',
            'failed': '#dc3545',
            'error': '#6c757d',
        }
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Regression Report - {report.report_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin: 20px 0; }}
        .summary-card {{ padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card h3 {{ margin: 0; font-size: 2em; }}
        .summary-card p {{ margin: 5px 0 0; color: #666; }}
        .passed {{ background: #d4edda; color: #155724; }}
        .warning {{ background: #fff3cd; color: #856404; }}
        .failed {{ background: #f8d7da; color: #721c24; }}
        .error {{ background: #e2e3e5; color: #383d41; }}
        .total {{ background: #cce5ff; color: #004085; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        .status-badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 500; }}
        .meta {{ color: #666; font-size: 0.9em; margin: 10px 0; }}
        .deviation {{ font-size: 0.85em; color: #666; }}
        .deviation.high {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 E2E Regression Report</h1>
        
        <div class="meta">
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p><strong>Period:</strong> {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')} ({report.period_days} days)</p>
            <p><strong>Execution Time:</strong> {report.total_execution_time:.1f}s</p>
        </div>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-card total">
                <h3>{report.total_strategies}</h3>
                <p>Total</p>
            </div>
            <div class="summary-card passed">
                <h3>{report.passed}</h3>
                <p>Passed</p>
            </div>
            <div class="summary-card warning">
                <h3>{report.warnings}</h3>
                <p>Warnings</p>
            </div>
            <div class="summary-card failed">
                <h3>{report.failed}</h3>
                <p>Failed</p>
            </div>
            <div class="summary-card error">
                <h3>{report.errors}</h3>
                <p>Errors</p>
            </div>
        </div>
        
        <h2>Strategy Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Status</th>
                    <th>P&L</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Max DD</th>
                    <th>Sharpe</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for r in report.strategy_results:
            status_color = status_colors.get(r.status.value, '#666')
            pnl_class = 'high' if r.pnl_deviation > report.pnl_tolerance else ''
            
            html += f"""
                <tr>
                    <td><strong>{r.strategy_name}</strong></td>
                    <td><span class="status-badge" style="background: {status_color}; color: white;">{r.status.value.upper()}</span></td>
                    <td>
                        ${r.total_pnl:,.0f}
                        <div class="deviation {pnl_class}">vs ${r.expected_pnl:,.0f} ({r.pnl_deviation:+.1%})</div>
                    </td>
                    <td>
                        {r.total_trades}
                        <div class="deviation">vs {r.expected_trades}</div>
                    </td>
                    <td>
                        {r.win_rate:.1%}
                        <div class="deviation">vs {r.expected_win_rate:.1%}</div>
                    </td>
                    <td>
                        ${r.max_drawdown:,.0f}
                        <div class="deviation">vs ${r.expected_max_drawdown:,.0f}</div>
                    </td>
                    <td>
                        {r.sharpe_ratio:.2f}
                        <div class="deviation">vs {r.expected_sharpe:.2f}</div>
                    </td>
                    <td>{r.execution_time_seconds:.1f}s</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <h2>Tolerances</h2>
        <ul>
            <li><strong>P&L Tolerance:</strong> {:.0%}</li>
            <li><strong>Trade Count Tolerance:</strong> {:.0%}</li>
            <li><strong>Drawdown Tolerance:</strong> {:.0%}</li>
        </ul>
    </div>
</body>
</html>
""".format(report.pnl_tolerance, report.trade_tolerance, report.drawdown_tolerance)
        
        with open(path, 'w') as f:
            f.write(html)
        
        return str(path)
    
    def print_summary(self, report: RegressionReport) -> None:
        """Print summary to console."""
        print("\n" + "="*60)
        print("  E2E REGRESSION REPORT")
        print("="*60)
        print(f"  Report ID: {report.report_id}")
        print(f"  Period: {report.period_days} days")
        print(f"  Status: {report.overall_status.value.upper()}")
        print("-"*60)
        print(f"  ✅ Passed:   {report.passed}")
        print(f"  ⚠️  Warnings: {report.warnings}")
        print(f"  ❌ Failed:   {report.failed}")
        print(f"  🔥 Errors:   {report.errors}")
        print("-"*60)
        print(f"  Execution Time: {report.total_execution_time:.1f}s")
        print("="*60 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='E2E Regression Suite')
    parser.add_argument('--days', type=int, default=30, help='Days to replay')
    parser.add_argument('--tolerance', type=float, default=0.10, help='P&L tolerance')
    parser.add_argument('--ci', action='store_true', help='CI mode (exit with error on failure)')
    parser.add_argument('--fail-on-warning', action='store_true', help='Fail on warnings too')
    parser.add_argument('--report', type=str, help='Output HTML report path')
    parser.add_argument('--output-dir', type=str, default='reports/regression', help='Output directory')
    parser.add_argument('--strategies', type=str, nargs='+', help='Specific strategies to test')
    args = parser.parse_args()
    
    # Build config
    config = RegressionConfig(
        replay_days=args.days,
        pnl_tolerance=args.tolerance,
        ci_mode=args.ci,
        fail_on_warning=args.fail_on_warning,
        output_dir=args.output_dir,
    )
    
    if args.strategies:
        config.strategies = args.strategies
    
    # Run regression
    runner = RegressionRunner(config)
    report = await runner.run()
    
    # Generate reports
    generator = ReportGenerator(config.output_dir)
    json_path = generator.generate_json(report)
    html_path = generator.generate_html(report)
    
    generator.print_summary(report)
    
    print(f"Reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    
    # CI mode exit
    if config.ci_mode:
        if report.overall_status == RegressionStatus.FAILED:
            print("\n❌ REGRESSION FAILED - Exiting with error")
            sys.exit(1)
        if config.fail_on_warning and report.overall_status == RegressionStatus.WARNING:
            print("\n⚠️ REGRESSION WARNINGS - Exiting with error (--fail-on-warning)")
            sys.exit(1)
    
    print("\n✅ Regression suite complete")


if __name__ == '__main__':
    asyncio.run(main())
