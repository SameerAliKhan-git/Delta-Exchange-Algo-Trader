"""
Canary Deployment Orchestrator
==============================
Staged rollout: 1% → 5% → Full with automated checks.

Progression Rules:
- Canary-1: 1% AUM for 7 days
- Canary-2: 5% AUM for 14 days  
- Full: Scale based on capacity

Acceptance Criteria (each stage):
- Realized P&L within ±10% of simulated
- Slippage ≤ 1.5x simulated
- No critical alerts
- Drawdown < threshold
- Model precision ≥ 90% of baseline
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanaryStage(Enum):
    """Canary deployment stages."""
    SHADOW = "shadow"           # 0% live, 100% paper
    CANARY_1 = "canary_1"       # 1% live
    CANARY_2 = "canary_2"       # 5% live
    PRODUCTION = "production"   # Full allocation
    ROLLBACK = "rollback"       # Failed, rolled back


@dataclass
class StageConfig:
    """Configuration for each canary stage."""
    stage: CanaryStage
    aum_pct: float
    min_days: int
    max_drawdown_pct: float
    pnl_deviation_pct: float  # Max allowed deviation from simulated
    slippage_multiplier: float  # Max allowed vs simulated
    min_precision_ratio: float  # vs baseline


@dataclass
class StageMetrics:
    """Metrics collected during a canary stage."""
    stage: CanaryStage
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # P&L
    simulated_pnl: float = 0.0
    realized_pnl: float = 0.0
    gross_pnl: float = 0.0
    
    # Risk
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Execution
    avg_slippage_realized: float = 0.0
    avg_slippage_simulated: float = 0.0
    fill_rate: float = 1.0
    
    # Model
    model_precision: float = 0.0
    baseline_precision: float = 0.0
    
    # Alerts
    critical_alerts: int = 0
    warning_alerts: int = 0
    
    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'stage': self.stage.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class AcceptanceCriteria:
    """Acceptance criteria check result."""
    criterion: str
    passed: bool
    expected: Any
    actual: Any
    message: str


# =============================================================================
# STAGE CONFIGURATIONS
# =============================================================================

STAGE_CONFIGS = {
    CanaryStage.SHADOW: StageConfig(
        stage=CanaryStage.SHADOW,
        aum_pct=0.0,
        min_days=30,
        max_drawdown_pct=1.0,  # N/A for shadow
        pnl_deviation_pct=1.0,  # N/A
        slippage_multiplier=1.0,
        min_precision_ratio=0.9
    ),
    CanaryStage.CANARY_1: StageConfig(
        stage=CanaryStage.CANARY_1,
        aum_pct=0.01,  # 1%
        min_days=7,
        max_drawdown_pct=0.02,  # 2% max DD at 1% allocation
        pnl_deviation_pct=0.10,  # Within 10% of simulated
        slippage_multiplier=1.5,
        min_precision_ratio=0.9
    ),
    CanaryStage.CANARY_2: StageConfig(
        stage=CanaryStage.CANARY_2,
        aum_pct=0.05,  # 5%
        min_days=14,
        max_drawdown_pct=0.03,  # 3% max DD
        pnl_deviation_pct=0.10,
        slippage_multiplier=1.5,
        min_precision_ratio=0.9
    ),
    CanaryStage.PRODUCTION: StageConfig(
        stage=CanaryStage.PRODUCTION,
        aum_pct=1.0,  # Full (scaled by capacity)
        min_days=0,
        max_drawdown_pct=0.10,  # 10% max DD
        pnl_deviation_pct=0.15,
        slippage_multiplier=2.0,
        min_precision_ratio=0.85
    )
}


# =============================================================================
# CANARY ORCHESTRATOR
# =============================================================================

class CanaryOrchestrator:
    """
    Orchestrate canary deployment with automated checks.
    
    Automatically progresses through stages when criteria are met.
    Automatically rolls back when criteria fail.
    """
    
    def __init__(
        self,
        state_file: str = "./canary_state.json",
        metrics_file: str = "./canary_metrics.json",
        total_aum: float = 100000
    ):
        self.state_file = Path(state_file)
        self.metrics_file = Path(metrics_file)
        self.total_aum = total_aum
        
        # Current state
        self.current_stage = CanaryStage.SHADOW
        self.current_metrics: Optional[StageMetrics] = None
        self.stage_history: List[StageMetrics] = []
        
        # Load saved state
        self._load_state()
    
    def _load_state(self):
        """Load saved canary state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.current_stage = CanaryStage(state.get('current_stage', 'shadow'))
                self.total_aum = state.get('total_aum', self.total_aum)
                logger.info(f"Loaded canary state: {self.current_stage.value}")
            except Exception as e:
                logger.warning(f"Could not load canary state: {e}")
    
    def _save_state(self):
        """Save canary state."""
        state = {
            'current_stage': self.current_stage.value,
            'total_aum': self.total_aum,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_current_allocation(self) -> float:
        """Get current AUM allocation in USD."""
        config = STAGE_CONFIGS[self.current_stage]
        return self.total_aum * config.aum_pct
    
    def start_stage(self, stage: CanaryStage):
        """Start a new canary stage."""
        self.current_stage = stage
        self.current_metrics = StageMetrics(
            stage=stage,
            start_time=datetime.now()
        )
        self._save_state()
        
        config = STAGE_CONFIGS[stage]
        allocation = self.total_aum * config.aum_pct
        
        logger.info(f"""
========================================
CANARY STAGE: {stage.value.upper()}
========================================
AUM Allocation: {config.aum_pct * 100:.1f}% (${allocation:,.0f})
Minimum Duration: {config.min_days} days
Max Drawdown: {config.max_drawdown_pct * 100:.1f}%
P&L Deviation Threshold: ±{config.pnl_deviation_pct * 100:.0f}%
Slippage Multiplier: {config.slippage_multiplier}x
========================================
""")
    
    def update_metrics(
        self,
        simulated_pnl: float = None,
        realized_pnl: float = None,
        slippage_realized: float = None,
        slippage_simulated: float = None,
        drawdown: float = None,
        model_precision: float = None,
        baseline_precision: float = None,
        critical_alerts: int = None,
        trades: int = None,
        wins: int = None
    ):
        """Update current stage metrics."""
        if not self.current_metrics:
            return
        
        m = self.current_metrics
        
        if simulated_pnl is not None:
            m.simulated_pnl = simulated_pnl
        if realized_pnl is not None:
            m.realized_pnl = realized_pnl
        if slippage_realized is not None:
            m.avg_slippage_realized = slippage_realized
        if slippage_simulated is not None:
            m.avg_slippage_simulated = slippage_simulated
        if drawdown is not None:
            m.current_drawdown = drawdown
            m.max_drawdown = max(m.max_drawdown, drawdown)
        if model_precision is not None:
            m.model_precision = model_precision
        if baseline_precision is not None:
            m.baseline_precision = baseline_precision
        if critical_alerts is not None:
            m.critical_alerts = critical_alerts
        if trades is not None:
            m.total_trades = trades
        if wins is not None:
            m.winning_trades = wins
    
    def check_acceptance_criteria(self) -> Tuple[bool, List[AcceptanceCriteria]]:
        """
        Check if current stage passes acceptance criteria.
        
        Returns:
            (all_passed, list of criteria results)
        """
        if not self.current_metrics:
            return False, []
        
        config = STAGE_CONFIGS[self.current_stage]
        m = self.current_metrics
        
        results = []
        
        # 1. Minimum duration
        duration = (datetime.now() - m.start_time).days
        results.append(AcceptanceCriteria(
            criterion="Minimum Duration",
            passed=duration >= config.min_days,
            expected=f"{config.min_days} days",
            actual=f"{duration} days",
            message=f"Stage must run for at least {config.min_days} days"
        ))
        
        # 2. P&L Deviation (skip for shadow)
        if self.current_stage != CanaryStage.SHADOW and m.simulated_pnl != 0:
            deviation = abs(m.realized_pnl - m.simulated_pnl) / abs(m.simulated_pnl)
            results.append(AcceptanceCriteria(
                criterion="P&L Deviation",
                passed=deviation <= config.pnl_deviation_pct,
                expected=f"≤{config.pnl_deviation_pct * 100:.0f}%",
                actual=f"{deviation * 100:.1f}%",
                message="Realized P&L must be within threshold of simulated"
            ))
        
        # 3. Max Drawdown
        results.append(AcceptanceCriteria(
            criterion="Max Drawdown",
            passed=m.max_drawdown <= config.max_drawdown_pct,
            expected=f"≤{config.max_drawdown_pct * 100:.1f}%",
            actual=f"{m.max_drawdown * 100:.2f}%",
            message="Maximum drawdown must not exceed threshold"
        ))
        
        # 4. Slippage
        if m.avg_slippage_simulated > 0:
            slippage_ratio = m.avg_slippage_realized / m.avg_slippage_simulated
            results.append(AcceptanceCriteria(
                criterion="Slippage Ratio",
                passed=slippage_ratio <= config.slippage_multiplier,
                expected=f"≤{config.slippage_multiplier}x",
                actual=f"{slippage_ratio:.2f}x",
                message="Realized slippage must not exceed multiplier of simulated"
            ))
        
        # 5. No Critical Alerts
        results.append(AcceptanceCriteria(
            criterion="Critical Alerts",
            passed=m.critical_alerts == 0,
            expected="0",
            actual=str(m.critical_alerts),
            message="No critical alerts during stage"
        ))
        
        # 6. Model Precision
        if m.baseline_precision > 0:
            precision_ratio = m.model_precision / m.baseline_precision
            results.append(AcceptanceCriteria(
                criterion="Model Precision",
                passed=precision_ratio >= config.min_precision_ratio,
                expected=f"≥{config.min_precision_ratio * 100:.0f}% of baseline",
                actual=f"{precision_ratio * 100:.1f}% of baseline",
                message="Model precision must not degrade significantly"
            ))
        
        all_passed = all(r.passed for r in results)
        return all_passed, results
    
    def can_progress(self) -> Tuple[bool, str]:
        """Check if we can progress to next stage."""
        passed, criteria = self.check_acceptance_criteria()
        
        if not passed:
            failed = [c for c in criteria if not c.passed]
            reasons = [f"{c.criterion}: {c.message} (expected {c.expected}, got {c.actual})" 
                       for c in failed]
            return False, "; ".join(reasons)
        
        return True, "All criteria passed"
    
    def get_next_stage(self) -> Optional[CanaryStage]:
        """Get the next stage in progression."""
        progression = [
            CanaryStage.SHADOW,
            CanaryStage.CANARY_1,
            CanaryStage.CANARY_2,
            CanaryStage.PRODUCTION
        ]
        
        try:
            idx = progression.index(self.current_stage)
            if idx < len(progression) - 1:
                return progression[idx + 1]
        except ValueError:
            pass
        
        return None
    
    async def progress_to_next_stage(self) -> Tuple[bool, str]:
        """Attempt to progress to next stage."""
        can_progress, reason = self.can_progress()
        
        if not can_progress:
            logger.warning(f"Cannot progress: {reason}")
            return False, reason
        
        next_stage = self.get_next_stage()
        if not next_stage:
            logger.info("Already at production stage")
            return True, "Already at production"
        
        # Save current stage metrics
        if self.current_metrics:
            self.current_metrics.end_time = datetime.now()
            self.stage_history.append(self.current_metrics)
        
        # Start next stage
        self.start_stage(next_stage)
        
        logger.info(f"Progressed to {next_stage.value}")
        return True, f"Progressed to {next_stage.value}"
    
    async def rollback(self, reason: str):
        """Rollback to shadow mode."""
        logger.critical(f"CANARY ROLLBACK: {reason}")
        
        if self.current_metrics:
            self.current_metrics.end_time = datetime.now()
            self.stage_history.append(self.current_metrics)
        
        self.current_stage = CanaryStage.ROLLBACK
        self._save_state()
        
        # Trigger automated rollback system
        try:
            from .rollback import RollbackManager
            manager = RollbackManager()
            await manager.execute_rollback(
                reason=f"Canary rollback: {reason}",
                trigger="canary_orchestrator"
            )
        except Exception as e:
            logger.error(f"Failed to trigger rollback system: {e}")
    
    def generate_report(self) -> str:
        """Generate canary progress report."""
        lines = [
            "=" * 60,
            "CANARY DEPLOYMENT REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Current Stage: {self.current_stage.value.upper()}",
            f"Total AUM: ${self.total_aum:,.0f}",
            f"Current Allocation: ${self.get_current_allocation():,.0f}",
            "",
        ]
        
        if self.current_metrics:
            m = self.current_metrics
            duration = (datetime.now() - m.start_time).days
            config = STAGE_CONFIGS[self.current_stage]
            
            lines.extend([
                "-" * 60,
                "CURRENT STAGE METRICS",
                "-" * 60,
                f"Duration: {duration} / {config.min_days} days",
                f"Simulated P&L: ${m.simulated_pnl:,.2f}",
                f"Realized P&L: ${m.realized_pnl:,.2f}",
                f"Max Drawdown: {m.max_drawdown * 100:.2f}%",
                f"Slippage (Realized): {m.avg_slippage_realized:.1f} bps",
                f"Slippage (Simulated): {m.avg_slippage_simulated:.1f} bps",
                f"Model Precision: {m.model_precision * 100:.1f}%",
                f"Critical Alerts: {m.critical_alerts}",
                f"Total Trades: {m.total_trades}",
                f"Win Rate: {m.winning_trades / max(m.total_trades, 1) * 100:.1f}%",
                "",
            ])
        
        # Acceptance criteria
        passed, criteria = self.check_acceptance_criteria()
        lines.extend([
            "-" * 60,
            f"ACCEPTANCE CRITERIA: {'✅ PASSED' if passed else '❌ FAILED'}",
            "-" * 60,
        ])
        
        for c in criteria:
            icon = "✅" if c.passed else "❌"
            lines.append(f"  {icon} {c.criterion}: {c.actual} (expected {c.expected})")
        
        # Next steps
        lines.extend([
            "",
            "-" * 60,
            "NEXT STEPS",
            "-" * 60,
        ])
        
        if passed:
            next_stage = self.get_next_stage()
            if next_stage:
                lines.append(f"  → Ready to progress to {next_stage.value}")
            else:
                lines.append("  → At production stage")
        else:
            failed = [c for c in criteria if not c.passed]
            lines.append("  → Fix the following before progression:")
            for c in failed:
                lines.append(f"    • {c.criterion}: {c.message}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    async def run_check_loop(self, check_interval_minutes: int = 60):
        """Run continuous checking loop."""
        logger.info(f"Starting canary check loop (interval: {check_interval_minutes}m)")
        
        while True:
            try:
                # Check criteria
                passed, criteria = self.check_acceptance_criteria()
                
                # Check for critical failures (immediate rollback)
                if self.current_metrics and self.current_metrics.critical_alerts > 0:
                    await self.rollback("Critical alert triggered")
                    break
                
                config = STAGE_CONFIGS[self.current_stage]
                if self.current_metrics and self.current_metrics.max_drawdown > config.max_drawdown_pct:
                    await self.rollback(f"Max drawdown exceeded: {self.current_metrics.max_drawdown * 100:.1f}%")
                    break
                
                # Check for auto-progression
                can_progress, reason = self.can_progress()
                if can_progress:
                    next_stage = self.get_next_stage()
                    if next_stage:
                        logger.info(f"Auto-progressing to {next_stage.value}")
                        await self.progress_to_next_stage()
                
                # Log status
                logger.info(f"Canary check complete. Stage: {self.current_stage.value}, Passed: {passed}")
                
            except Exception as e:
                logger.error(f"Canary check error: {e}")
            
            await asyncio.sleep(check_interval_minutes * 60)


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Canary Deployment Orchestrator")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--progress", action="store_true", help="Attempt to progress to next stage")
    parser.add_argument("--start", type=str, choices=['shadow', 'canary_1', 'canary_2'], 
                        help="Start a specific stage")
    parser.add_argument("--rollback", action="store_true", help="Rollback to shadow")
    parser.add_argument("--aum", type=float, default=100000, help="Total AUM")
    parser.add_argument("--run", action="store_true", help="Run continuous check loop")
    
    args = parser.parse_args()
    
    orchestrator = CanaryOrchestrator(total_aum=args.aum)
    
    if args.status:
        print(f"\nCurrent Stage: {orchestrator.current_stage.value}")
        print(f"Allocation: ${orchestrator.get_current_allocation():,.0f}")
        return
    
    if args.report:
        print(orchestrator.generate_report())
        return
    
    if args.start:
        stage = CanaryStage(args.start)
        orchestrator.start_stage(stage)
        return
    
    if args.progress:
        success, msg = await orchestrator.progress_to_next_stage()
        print(f"Progress: {'Success' if success else 'Failed'} - {msg}")
        return
    
    if args.rollback:
        await orchestrator.rollback("Manual rollback requested")
        return
    
    if args.run:
        await orchestrator.run_check_loop()
        return
    
    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
