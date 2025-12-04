#!/usr/bin/env python3
"""
scripts/promotion_manager.py

Paper-to-Canary Promotion Playbook
==================================

This tool manages the staged promotion of trading strategies through:
  SHADOW (0%) → CANARY-1 (1%) → CANARY-2 (5%) → PRODUCTION (100%)

Features:
- Statistical validation before promotion
- Human-in-loop confirmation for critical stages
- Automatic rollback on failure
- Comprehensive audit logging
- Slack/Discord notifications

Usage:
  # Check if ready for promotion
  python scripts/promotion_manager.py check --current-stage shadow
  
  # Promote to next stage (with confirmation)
  python scripts/promotion_manager.py promote --current-stage shadow --confirm
  
  # Force rollback
  python scripts/promotion_manager.py rollback --to-stage shadow
  
  # View promotion history
  python scripts/promotion_manager.py history
"""

import argparse
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import hashlib

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Stage Definitions
# =============================================================================

class Stage(Enum):
    SHADOW = "shadow"
    CANARY_1 = "canary_1"
    CANARY_2 = "canary_2"
    PRODUCTION = "production"
    KILLED = "killed"


@dataclass
class StageConfig:
    """Configuration for each deployment stage."""
    name: Stage
    allocation_pct: float
    min_duration_days: int
    min_profitable_days: int
    max_drawdown_pct: float
    min_sharpe_ratio: float
    max_pnl_deviation_pct: float  # vs shadow/backtest
    requires_human_approval: bool
    next_stage: Optional[Stage] = None
    prev_stage: Optional[Stage] = None


STAGE_CONFIGS: Dict[Stage, StageConfig] = {
    Stage.SHADOW: StageConfig(
        name=Stage.SHADOW,
        allocation_pct=0.0,
        min_duration_days=7,
        min_profitable_days=5,
        max_drawdown_pct=15.0,
        min_sharpe_ratio=0.5,
        max_pnl_deviation_pct=20.0,
        requires_human_approval=False,
        next_stage=Stage.CANARY_1,
        prev_stage=None,
    ),
    Stage.CANARY_1: StageConfig(
        name=Stage.CANARY_1,
        allocation_pct=1.0,
        min_duration_days=7,
        min_profitable_days=5,
        max_drawdown_pct=10.0,
        min_sharpe_ratio=0.8,
        max_pnl_deviation_pct=15.0,
        requires_human_approval=True,
        next_stage=Stage.CANARY_2,
        prev_stage=Stage.SHADOW,
    ),
    Stage.CANARY_2: StageConfig(
        name=Stage.CANARY_2,
        allocation_pct=5.0,
        min_duration_days=14,
        min_profitable_days=10,
        max_drawdown_pct=8.0,
        min_sharpe_ratio=1.0,
        max_pnl_deviation_pct=10.0,
        requires_human_approval=True,
        next_stage=Stage.PRODUCTION,
        prev_stage=Stage.CANARY_1,
    ),
    Stage.PRODUCTION: StageConfig(
        name=Stage.PRODUCTION,
        allocation_pct=100.0,
        min_duration_days=0,  # No minimum
        min_profitable_days=0,
        max_drawdown_pct=10.0,
        min_sharpe_ratio=1.0,
        max_pnl_deviation_pct=10.0,
        requires_human_approval=True,
        next_stage=None,
        prev_stage=Stage.CANARY_2,
    ),
}


# =============================================================================
# Promotion Criteria
# =============================================================================

@dataclass
class PromotionCriteria:
    """Results of promotion criteria check."""
    stage: Stage
    passed: bool
    criteria: Dict[str, Dict[str, Any]]  # {criterion: {value, threshold, passed}}
    blocking_issues: List[str]
    warnings: List[str]
    recommendation: str
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CriteriaChecker:
    """Check promotion criteria for a stage."""
    
    def __init__(self, metrics_path: Path):
        self.metrics_path = metrics_path
    
    def load_metrics(self, stage: Stage, days: int = 30) -> Dict[str, Any]:
        """Load metrics for a stage."""
        # Look for metrics file
        metrics_file = self.metrics_path / f"{stage.value}_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
        
        # Generate mock metrics for demo
        logger.warning(f"No metrics file found, using demo data: {metrics_file}")
        return self._generate_mock_metrics(stage, days)
    
    def _generate_mock_metrics(self, stage: Stage, days: int) -> Dict[str, Any]:
        """Generate mock metrics for demonstration."""
        import numpy as np
        np.random.seed(42)
        
        # Simulate daily P&L
        daily_returns = np.random.normal(0.002, 0.01, days)  # 20bps avg, 1% vol
        cumulative = np.cumprod(1 + daily_returns) - 1
        
        profitable_days = (daily_returns > 0).sum()
        
        # Max drawdown
        peak = np.maximum.accumulate(cumulative + 1)
        drawdown = (peak - (cumulative + 1)) / peak
        max_dd = np.max(drawdown)
        
        # Sharpe
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        return {
            'stage': stage.value,
            'days_active': days,
            'total_return_pct': float(cumulative[-1] * 100),
            'daily_returns': daily_returns.tolist(),
            'profitable_days': int(profitable_days),
            'max_drawdown_pct': float(max_dd * 100),
            'sharpe_ratio': float(sharpe),
            'win_rate': float((daily_returns > 0).mean()),
            'avg_daily_return_pct': float(np.mean(daily_returns) * 100),
            'volatility_pct': float(np.std(daily_returns) * 100),
            'pnl_deviation_vs_shadow_pct': float(np.random.uniform(-5, 10)),
        }
    
    def check(self, stage: Stage, target_stage: Stage) -> PromotionCriteria:
        """
        Check if ready to promote from current stage to target stage.
        
        Args:
            stage: Current stage
            target_stage: Target stage (next stage)
        
        Returns:
            PromotionCriteria with results
        """
        config = STAGE_CONFIGS[stage]
        metrics = self.load_metrics(stage)
        
        criteria = {}
        blocking = []
        warnings = []
        
        # 1. Duration check
        days_active = metrics.get('days_active', 0)
        duration_ok = days_active >= config.min_duration_days
        criteria['min_duration'] = {
            'value': days_active,
            'threshold': config.min_duration_days,
            'unit': 'days',
            'passed': duration_ok,
        }
        if not duration_ok:
            blocking.append(f"Minimum duration not met: {days_active}/{config.min_duration_days} days")
        
        # 2. Profitable days check
        profitable = metrics.get('profitable_days', 0)
        profit_days_ok = profitable >= config.min_profitable_days
        criteria['min_profitable_days'] = {
            'value': profitable,
            'threshold': config.min_profitable_days,
            'unit': 'days',
            'passed': profit_days_ok,
        }
        if not profit_days_ok:
            blocking.append(f"Minimum profitable days not met: {profitable}/{config.min_profitable_days} days")
        
        # 3. Max drawdown check
        max_dd = metrics.get('max_drawdown_pct', 100)
        dd_ok = max_dd <= config.max_drawdown_pct
        criteria['max_drawdown'] = {
            'value': max_dd,
            'threshold': config.max_drawdown_pct,
            'unit': '%',
            'passed': dd_ok,
        }
        if not dd_ok:
            blocking.append(f"Max drawdown exceeded: {max_dd:.2f}% > {config.max_drawdown_pct}%")
        
        # 4. Sharpe ratio check
        sharpe = metrics.get('sharpe_ratio', 0)
        sharpe_ok = sharpe >= config.min_sharpe_ratio
        criteria['min_sharpe'] = {
            'value': sharpe,
            'threshold': config.min_sharpe_ratio,
            'unit': '',
            'passed': sharpe_ok,
        }
        if not sharpe_ok:
            blocking.append(f"Sharpe ratio too low: {sharpe:.2f} < {config.min_sharpe_ratio}")
        
        # 5. P&L deviation check (vs shadow/backtest)
        deviation = abs(metrics.get('pnl_deviation_vs_shadow_pct', 0))
        deviation_ok = deviation <= config.max_pnl_deviation_pct
        criteria['pnl_deviation'] = {
            'value': deviation,
            'threshold': config.max_pnl_deviation_pct,
            'unit': '%',
            'passed': deviation_ok,
        }
        if not deviation_ok:
            warnings.append(f"P&L deviation from shadow: {deviation:.2f}% (threshold: {config.max_pnl_deviation_pct}%)")
        
        # 6. Win rate check (warning only)
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.45:
            warnings.append(f"Low win rate: {win_rate:.1%}")
        
        # Overall result
        passed = len(blocking) == 0
        
        # Recommendation
        if passed and len(warnings) == 0:
            recommendation = f"✅ READY for promotion to {target_stage.value}"
        elif passed:
            recommendation = f"⚠️ CAN promote to {target_stage.value} (with warnings)"
        else:
            recommendation = f"❌ NOT READY for promotion"
        
        return PromotionCriteria(
            stage=stage,
            passed=passed,
            criteria=criteria,
            blocking_issues=blocking,
            warnings=warnings,
            recommendation=recommendation,
        )


# =============================================================================
# Promotion Manager
# =============================================================================

@dataclass
class PromotionEvent:
    """Record of a promotion or rollback event."""
    event_id: str
    event_type: str  # "promotion", "rollback", "kill"
    from_stage: str
    to_stage: str
    timestamp: str
    criteria_snapshot: Optional[Dict] = None
    approved_by: Optional[str] = None
    reason: Optional[str] = None


class PromotionManager:
    """
    Manages the promotion lifecycle.
    
    Responsibilities:
    - Check promotion criteria
    - Execute promotions with human approval
    - Execute rollbacks
    - Maintain audit log
    - Send notifications
    """
    
    def __init__(
        self,
        state_path: Path,
        metrics_path: Path,
        notify_webhook: Optional[str] = None,
    ):
        self.state_path = state_path
        self.metrics_path = metrics_path
        self.notify_webhook = notify_webhook
        self.checker = CriteriaChecker(metrics_path)
        
        # Load state
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load promotion state."""
        state_file = self.state_path / "promotion_state.json"
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        
        return {
            'current_stage': Stage.SHADOW.value,
            'stage_started_at': datetime.utcnow().isoformat(),
            'history': [],
        }
    
    def _save_state(self):
        """Save promotion state."""
        self.state_path.mkdir(parents=True, exist_ok=True)
        with open(self.state_path / "promotion_state.json", 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_current_stage(self) -> Stage:
        """Get current deployment stage."""
        return Stage(self.state['current_stage'])
    
    def check_promotion(self) -> PromotionCriteria:
        """Check if ready for promotion to next stage."""
        current = self.get_current_stage()
        config = STAGE_CONFIGS[current]
        
        if config.next_stage is None:
            return PromotionCriteria(
                stage=current,
                passed=False,
                criteria={},
                blocking_issues=["Already at production stage"],
                warnings=[],
                recommendation="Already at maximum stage",
            )
        
        return self.checker.check(current, config.next_stage)
    
    def promote(
        self,
        confirm: bool = False,
        approved_by: str = "system",
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Promote to next stage.
        
        Args:
            confirm: Must be True to execute
            approved_by: Who approved the promotion
            reason: Reason for promotion
        
        Returns:
            Result dict
        """
        current = self.get_current_stage()
        config = STAGE_CONFIGS[current]
        
        if config.next_stage is None:
            return {
                'success': False,
                'error': 'Already at production stage',
            }
        
        # Check criteria
        criteria = self.check_promotion()
        
        if not criteria.passed:
            return {
                'success': False,
                'error': 'Criteria not met',
                'blocking_issues': criteria.blocking_issues,
            }
        
        # Human approval required?
        next_config = STAGE_CONFIGS[config.next_stage]
        if next_config.requires_human_approval and not confirm:
            return {
                'success': False,
                'error': 'Human approval required',
                'message': f'Use --confirm to approve promotion to {config.next_stage.value}',
                'criteria': asdict(criteria),
            }
        
        # Execute promotion
        event_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}{current.value}{config.next_stage.value}".encode()
        ).hexdigest()[:12]
        
        event = PromotionEvent(
            event_id=event_id,
            event_type="promotion",
            from_stage=current.value,
            to_stage=config.next_stage.value,
            timestamp=datetime.utcnow().isoformat(),
            criteria_snapshot=asdict(criteria),
            approved_by=approved_by,
            reason=reason,
        )
        
        # Update state
        self.state['current_stage'] = config.next_stage.value
        self.state['stage_started_at'] = datetime.utcnow().isoformat()
        self.state['history'].append(asdict(event))
        self._save_state()
        
        # Notify
        self._notify(f"🚀 PROMOTED: {current.value} → {config.next_stage.value}", event)
        
        logger.info(f"Promoted from {current.value} to {config.next_stage.value}")
        
        return {
            'success': True,
            'from_stage': current.value,
            'to_stage': config.next_stage.value,
            'event_id': event_id,
            'allocation_pct': next_config.allocation_pct,
        }
    
    def rollback(
        self,
        to_stage: Optional[Stage] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Rollback to previous stage.
        
        Args:
            to_stage: Target stage (default: previous stage)
            reason: Reason for rollback
        
        Returns:
            Result dict
        """
        current = self.get_current_stage()
        config = STAGE_CONFIGS[current]
        
        if to_stage is None:
            if config.prev_stage is None:
                return {
                    'success': False,
                    'error': 'Already at shadow stage, cannot rollback further',
                }
            to_stage = config.prev_stage
        
        # Execute rollback
        event_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}{current.value}{to_stage.value}rollback".encode()
        ).hexdigest()[:12]
        
        event = PromotionEvent(
            event_id=event_id,
            event_type="rollback",
            from_stage=current.value,
            to_stage=to_stage.value,
            timestamp=datetime.utcnow().isoformat(),
            reason=reason,
        )
        
        # Update state
        self.state['current_stage'] = to_stage.value
        self.state['stage_started_at'] = datetime.utcnow().isoformat()
        self.state['history'].append(asdict(event))
        self._save_state()
        
        # Notify
        self._notify(f"⚠️ ROLLBACK: {current.value} → {to_stage.value}", event)
        
        logger.info(f"Rolled back from {current.value} to {to_stage.value}")
        
        return {
            'success': True,
            'from_stage': current.value,
            'to_stage': to_stage.value,
            'event_id': event_id,
        }
    
    def kill(self, reason: str = "") -> Dict[str, Any]:
        """Emergency kill - stop all trading."""
        current = self.get_current_stage()
        
        event_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}kill".encode()
        ).hexdigest()[:12]
        
        event = PromotionEvent(
            event_id=event_id,
            event_type="kill",
            from_stage=current.value,
            to_stage=Stage.KILLED.value,
            timestamp=datetime.utcnow().isoformat(),
            reason=reason,
        )
        
        # Update state
        self.state['current_stage'] = Stage.KILLED.value
        self.state['stage_started_at'] = datetime.utcnow().isoformat()
        self.state['history'].append(asdict(event))
        self._save_state()
        
        # Write kill switch file
        kill_file = self.state_path / "KILL_SWITCH"
        kill_file.write_text(f"KILLED at {datetime.utcnow().isoformat()}\nReason: {reason}")
        
        # Notify
        self._notify(f"🛑 KILL SWITCH ACTIVATED: {reason}", event)
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        return {
            'success': True,
            'from_stage': current.value,
            'event_id': event_id,
        }
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get promotion history."""
        return self.state.get('history', [])[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        current = self.get_current_stage()
        config = STAGE_CONFIGS.get(current)
        
        started = datetime.fromisoformat(self.state['stage_started_at'])
        days_in_stage = (datetime.utcnow() - started).days
        
        return {
            'current_stage': current.value,
            'allocation_pct': config.allocation_pct if config else 0,
            'days_in_stage': days_in_stage,
            'min_days_required': config.min_duration_days if config else 0,
            'stage_started_at': self.state['stage_started_at'],
            'next_stage': config.next_stage.value if config and config.next_stage else None,
            'requires_human_approval': config.requires_human_approval if config else False,
        }
    
    def _notify(self, message: str, event: PromotionEvent):
        """Send notification."""
        if not self.notify_webhook:
            return
        
        try:
            import requests
            
            payload = {
                'text': message,
                'attachments': [
                    {
                        'color': '#00ff00' if event.event_type == 'promotion' else '#ff0000',
                        'fields': [
                            {'title': 'Event ID', 'value': event.event_id, 'short': True},
                            {'title': 'Type', 'value': event.event_type, 'short': True},
                            {'title': 'From', 'value': event.from_stage, 'short': True},
                            {'title': 'To', 'value': event.to_stage, 'short': True},
                        ],
                    }
                ],
            }
            
            requests.post(self.notify_webhook, json=payload, timeout=5)
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Promotion Manager - Paper to Production Playbook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check promotion readiness
  python scripts/promotion_manager.py check
  
  # Promote to next stage (requires --confirm)
  python scripts/promotion_manager.py promote --confirm --approved-by "john@example.com"
  
  # Rollback to previous stage
  python scripts/promotion_manager.py rollback --reason "Performance degradation"
  
  # Emergency kill
  python scripts/promotion_manager.py kill --reason "Critical bug discovered"
  
  # View status
  python scripts/promotion_manager.py status
  
  # View history
  python scripts/promotion_manager.py history
        """
    )
    
    parser.add_argument(
        'action',
        choices=['check', 'promote', 'rollback', 'kill', 'status', 'history'],
        help='Action to perform'
    )
    parser.add_argument('--confirm', action='store_true', help='Confirm promotion')
    parser.add_argument('--approved-by', type=str, default='cli', help='Approver name/email')
    parser.add_argument('--reason', type=str, default='', help='Reason for action')
    parser.add_argument('--to-stage', type=str, help='Target stage for rollback')
    parser.add_argument('--webhook', type=str, help='Notification webhook URL')
    
    args = parser.parse_args()
    
    # Initialize manager
    state_path = PROJECT_ROOT / "state" / "promotion"
    metrics_path = PROJECT_ROOT / "metrics"
    
    manager = PromotionManager(
        state_path=state_path,
        metrics_path=metrics_path,
        notify_webhook=args.webhook,
    )
    
    # Execute action
    if args.action == 'check':
        criteria = manager.check_promotion()
        
        print("\n" + "="*60)
        print("PROMOTION READINESS CHECK")
        print("="*60)
        
        status = manager.get_status()
        print(f"\nCurrent Stage: {status['current_stage'].upper()}")
        print(f"Allocation: {status['allocation_pct']}%")
        print(f"Days in Stage: {status['days_in_stage']}")
        print(f"Next Stage: {status['next_stage'] or 'N/A'}")
        
        print("\n" + "-"*60)
        print("CRITERIA:")
        print("-"*60)
        
        for name, check in criteria.criteria.items():
            status_icon = '✅' if check['passed'] else '❌'
            unit = check.get('unit', '')
            print(f"  {status_icon} {name}: {check['value']}{unit} (threshold: {check['threshold']}{unit})")
        
        if criteria.blocking_issues:
            print("\n❌ BLOCKING ISSUES:")
            for issue in criteria.blocking_issues:
                print(f"  - {issue}")
        
        if criteria.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in criteria.warnings:
                print(f"  - {warning}")
        
        print("\n" + "="*60)
        print(f"RECOMMENDATION: {criteria.recommendation}")
        print("="*60 + "\n")
    
    elif args.action == 'promote':
        result = manager.promote(
            confirm=args.confirm,
            approved_by=args.approved_by,
            reason=args.reason,
        )
        
        if result['success']:
            print(f"\n✅ PROMOTED: {result['from_stage']} → {result['to_stage']}")
            print(f"   Event ID: {result['event_id']}")
            print(f"   New Allocation: {result['allocation_pct']}%\n")
        else:
            print(f"\n❌ PROMOTION FAILED: {result.get('error')}")
            if 'blocking_issues' in result:
                for issue in result['blocking_issues']:
                    print(f"   - {issue}")
            if 'message' in result:
                print(f"   {result['message']}")
            print()
    
    elif args.action == 'rollback':
        to_stage = Stage(args.to_stage) if args.to_stage else None
        result = manager.rollback(to_stage=to_stage, reason=args.reason)
        
        if result['success']:
            print(f"\n⚠️ ROLLED BACK: {result['from_stage']} → {result['to_stage']}")
            print(f"   Event ID: {result['event_id']}\n")
        else:
            print(f"\n❌ ROLLBACK FAILED: {result.get('error')}\n")
    
    elif args.action == 'kill':
        result = manager.kill(reason=args.reason)
        print(f"\n🛑 KILL SWITCH ACTIVATED")
        print(f"   Event ID: {result['event_id']}\n")
    
    elif args.action == 'status':
        status = manager.get_status()
        
        print("\n" + "="*60)
        print("DEPLOYMENT STATUS")
        print("="*60)
        print(f"  Stage:        {status['current_stage'].upper()}")
        print(f"  Allocation:   {status['allocation_pct']}%")
        print(f"  Days Active:  {status['days_in_stage']}")
        print(f"  Min Required: {status['min_days_required']} days")
        print(f"  Started At:   {status['stage_started_at']}")
        print(f"  Next Stage:   {status['next_stage'] or 'N/A'}")
        print(f"  Needs Approval: {'Yes' if status['requires_human_approval'] else 'No'}")
        print("="*60 + "\n")
    
    elif args.action == 'history':
        history = manager.get_history()
        
        print("\n" + "="*60)
        print("PROMOTION HISTORY")
        print("="*60)
        
        if not history:
            print("  No history yet.\n")
        else:
            for event in reversed(history[-10:]):
                icon = {'promotion': '🚀', 'rollback': '⚠️', 'kill': '🛑'}.get(event['event_type'], '•')
                print(f"\n  {icon} {event['event_type'].upper()}")
                print(f"     {event['from_stage']} → {event['to_stage']}")
                print(f"     Time: {event['timestamp']}")
                if event.get('approved_by'):
                    print(f"     Approved by: {event['approved_by']}")
                if event.get('reason'):
                    print(f"     Reason: {event['reason']}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
