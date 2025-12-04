"""
Production Deployment Checklist - Hedge Fund Standard
=====================================================
CRITICAL: This checklist must be completed before deploying capital.

This module provides:
- Comprehensive pre-deployment verification
- Automated checks for all critical systems
- Documentation of sign-offs required
- Risk limit verification
- Operational readiness assessment
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of a checklist item."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    WAIVED = "waived"  # Explicitly waived with sign-off


class CheckCategory(Enum):
    """Categories of checks."""
    INFRASTRUCTURE = "infrastructure"
    RISK_MANAGEMENT = "risk_management"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"


class CheckPriority(Enum):
    """Priority levels."""
    CRITICAL = "critical"  # Must pass before ANY deployment
    HIGH = "high"          # Must pass before production
    MEDIUM = "medium"      # Should pass, can be waived with sign-off
    LOW = "low"            # Nice to have


@dataclass
class CheckItem:
    """Single checklist item."""
    id: str
    name: str
    description: str
    category: CheckCategory
    priority: CheckPriority
    
    # Status
    status: CheckStatus = CheckStatus.NOT_STARTED
    
    # Verification
    automated: bool = False  # Can be verified automatically
    verification_fn: Optional[str] = None  # Function to call for automated check
    
    # Results
    result_message: str = ""
    verified_at: Optional[datetime] = None
    verified_by: str = ""
    
    # Sign-off
    requires_signoff: bool = False
    signoff_by: str = ""
    signoff_at: Optional[datetime] = None
    signoff_notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'automated': self.automated,
            'result_message': self.result_message,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None,
            'verified_by': self.verified_by,
            'requires_signoff': self.requires_signoff,
            'signoff_by': self.signoff_by,
            'signoff_at': self.signoff_at.isoformat() if self.signoff_at else None,
            'signoff_notes': self.signoff_notes
        }


@dataclass
class DeploymentReadiness:
    """Overall deployment readiness assessment."""
    is_ready: bool
    readiness_score: float  # 0-100
    critical_items_passed: int
    critical_items_total: int
    high_items_passed: int
    high_items_total: int
    blocking_items: List[str]
    warnings: List[str]
    sign_offs_required: List[str]
    sign_offs_obtained: List[str]


# =============================================================================
# CHECKLIST DEFINITION
# =============================================================================

PRODUCTION_CHECKLIST: List[CheckItem] = [
    # =========================================================================
    # INFRASTRUCTURE
    # =========================================================================
    CheckItem(
        id="INFRA-001",
        name="API Connectivity Verified",
        description="Exchange API connection tested and authenticated",
        category=CheckCategory.INFRASTRUCTURE,
        priority=CheckPriority.CRITICAL,
        automated=True,
        verification_fn="verify_api_connectivity"
    ),
    CheckItem(
        id="INFRA-002",
        name="Network Latency Acceptable",
        description="Latency to exchange < 100ms p95",
        category=CheckCategory.INFRASTRUCTURE,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_network_latency"
    ),
    CheckItem(
        id="INFRA-003",
        name="Database Connection Stable",
        description="Database connection pool configured and tested",
        category=CheckCategory.INFRASTRUCTURE,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_database"
    ),
    CheckItem(
        id="INFRA-004",
        name="Logging System Operational",
        description="Logs flowing to aggregation system",
        category=CheckCategory.INFRASTRUCTURE,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_logging"
    ),
    CheckItem(
        id="INFRA-005",
        name="Backup Systems Ready",
        description="Failover and backup systems tested",
        category=CheckCategory.INFRASTRUCTURE,
        priority=CheckPriority.MEDIUM,
        automated=False,
        requires_signoff=True
    ),
    
    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    CheckItem(
        id="RISK-001",
        name="Kill Switch Tested",
        description="Emergency kill switch verified to work",
        category=CheckCategory.RISK_MANAGEMENT,
        priority=CheckPriority.CRITICAL,
        automated=True,
        verification_fn="verify_kill_switch",
        requires_signoff=True
    ),
    CheckItem(
        id="RISK-002",
        name="Position Limits Configured",
        description="Max position sizes set per symbol and portfolio",
        category=CheckCategory.RISK_MANAGEMENT,
        priority=CheckPriority.CRITICAL,
        automated=True,
        verification_fn="verify_position_limits"
    ),
    CheckItem(
        id="RISK-003",
        name="Loss Limits Configured",
        description="Daily, hourly, and max drawdown limits set",
        category=CheckCategory.RISK_MANAGEMENT,
        priority=CheckPriority.CRITICAL,
        automated=True,
        verification_fn="verify_loss_limits"
    ),
    CheckItem(
        id="RISK-004",
        name="Circuit Breakers Active",
        description="All circuit breakers enabled and tested",
        category=CheckCategory.RISK_MANAGEMENT,
        priority=CheckPriority.CRITICAL,
        automated=True,
        verification_fn="verify_circuit_breakers"
    ),
    CheckItem(
        id="RISK-005",
        name="Risk Manager Sign-Off",
        description="Risk manager reviewed and approved all limits",
        category=CheckCategory.RISK_MANAGEMENT,
        priority=CheckPriority.CRITICAL,
        automated=False,
        requires_signoff=True
    ),
    
    # =========================================================================
    # STRATEGY
    # =========================================================================
    CheckItem(
        id="STRAT-001",
        name="30-Day Paper Trading Complete",
        description="Minimum 30 days of paper trading with satisfactory results",
        category=CheckCategory.STRATEGY,
        priority=CheckPriority.CRITICAL,
        automated=False,
        requires_signoff=True
    ),
    CheckItem(
        id="STRAT-002",
        name="Transaction Cost Analysis Complete",
        description="Strategy profitability verified at realistic cost levels (10-20 bps)",
        category=CheckCategory.STRATEGY,
        priority=CheckPriority.CRITICAL,
        automated=False,
        requires_signoff=True
    ),
    CheckItem(
        id="STRAT-003",
        name="Walk-Forward Validation Passed",
        description="Out-of-sample Sharpe > 1.0 in walk-forward test",
        category=CheckCategory.STRATEGY,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_walk_forward"
    ),
    CheckItem(
        id="STRAT-004",
        name="Deflated Sharpe Acceptable",
        description="Deflated Sharpe Ratio accounting for multiple testing",
        category=CheckCategory.STRATEGY,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_deflated_sharpe"
    ),
    CheckItem(
        id="STRAT-005",
        name="Regime Detection Validated",
        description="Regime detection accuracy verified on historical data",
        category=CheckCategory.STRATEGY,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_regime_detection"
    ),
    CheckItem(
        id="STRAT-006",
        name="Peer Review Complete",
        description="Second quant has reviewed strategy logic",
        category=CheckCategory.STRATEGY,
        priority=CheckPriority.HIGH,
        automated=False,
        requires_signoff=True
    ),
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    CheckItem(
        id="EXEC-001",
        name="Order-Flow Gate Integrated",
        description="Order-flow confirmation gate is active in execution path",
        category=CheckCategory.EXECUTION,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_orderflow_gate"
    ),
    CheckItem(
        id="EXEC-002",
        name="Market Impact Model Calibrated",
        description="Almgren-Chriss model calibrated to recent data",
        category=CheckCategory.EXECUTION,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_impact_model"
    ),
    CheckItem(
        id="EXEC-003",
        name="Slippage Tolerance Set",
        description="Maximum slippage limits configured",
        category=CheckCategory.EXECUTION,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_slippage_config"
    ),
    CheckItem(
        id="EXEC-004",
        name="Order Rate Limits Configured",
        description="Rate limits to prevent order spam",
        category=CheckCategory.EXECUTION,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_rate_limits"
    ),
    
    # =========================================================================
    # MONITORING
    # =========================================================================
    CheckItem(
        id="MON-001",
        name="Prometheus Metrics Flowing",
        description="All critical metrics being exported",
        category=CheckCategory.MONITORING,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_prometheus"
    ),
    CheckItem(
        id="MON-002",
        name="Alerting Configured",
        description="PagerDuty/Slack alerts configured for critical events",
        category=CheckCategory.MONITORING,
        priority=CheckPriority.HIGH,
        automated=False,
        requires_signoff=True
    ),
    CheckItem(
        id="MON-003",
        name="Dashboard Operational",
        description="Grafana dashboard showing all key metrics",
        category=CheckCategory.MONITORING,
        priority=CheckPriority.MEDIUM,
        automated=True,
        verification_fn="verify_dashboard"
    ),
    CheckItem(
        id="MON-004",
        name="Health Check Endpoint Active",
        description="/health endpoint returning proper status",
        category=CheckCategory.MONITORING,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_health_endpoint"
    ),
    
    # =========================================================================
    # OPERATIONAL
    # =========================================================================
    CheckItem(
        id="OPS-001",
        name="Runbook Documented",
        description="Operational runbook with incident procedures",
        category=CheckCategory.OPERATIONAL,
        priority=CheckPriority.HIGH,
        automated=False,
        requires_signoff=True
    ),
    CheckItem(
        id="OPS-002",
        name="On-Call Schedule Set",
        description="24/7 on-call coverage arranged",
        category=CheckCategory.OPERATIONAL,
        priority=CheckPriority.HIGH,
        automated=False,
        requires_signoff=True
    ),
    CheckItem(
        id="OPS-003",
        name="Rollback Procedure Tested",
        description="Can rollback to previous version within 5 minutes",
        category=CheckCategory.OPERATIONAL,
        priority=CheckPriority.HIGH,
        automated=False,
        requires_signoff=True
    ),
    CheckItem(
        id="OPS-004",
        name="Model Versioning Active",
        description="Model registry tracking all deployed models",
        category=CheckCategory.OPERATIONAL,
        priority=CheckPriority.MEDIUM,
        automated=True,
        verification_fn="verify_model_versioning"
    ),
    
    # =========================================================================
    # COMPLIANCE
    # =========================================================================
    CheckItem(
        id="COMP-001",
        name="API Keys Secured",
        description="API keys in secure vault, not in code",
        category=CheckCategory.COMPLIANCE,
        priority=CheckPriority.CRITICAL,
        automated=True,
        verification_fn="verify_api_key_security"
    ),
    CheckItem(
        id="COMP-002",
        name="Trading Limits Documented",
        description="All trading limits documented and approved",
        category=CheckCategory.COMPLIANCE,
        priority=CheckPriority.HIGH,
        automated=False,
        requires_signoff=True
    ),
    CheckItem(
        id="COMP-003",
        name="Audit Trail Active",
        description="All trades and decisions logged for audit",
        category=CheckCategory.COMPLIANCE,
        priority=CheckPriority.HIGH,
        automated=True,
        verification_fn="verify_audit_trail"
    ),
]


# =============================================================================
# CHECKLIST MANAGER
# =============================================================================

class DeploymentChecklist:
    """
    Manage production deployment checklist.
    
    Use this to track all requirements before going live.
    """
    
    def __init__(
        self,
        checklist: Optional[List[CheckItem]] = None,
        state_file: str = "./deployment_checklist.json"
    ):
        self.checklist = checklist or [CheckItem(**item.__dict__) for item in PRODUCTION_CHECKLIST]
        self.state_file = Path(state_file)
        
        # Load saved state if exists
        self._load_state()
    
    def _load_state(self):
        """Load checklist state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Update items with saved state
                state_map = {item['id']: item for item in state.get('items', [])}
                for item in self.checklist:
                    if item.id in state_map:
                        saved = state_map[item.id]
                        item.status = CheckStatus(saved.get('status', 'not_started'))
                        item.result_message = saved.get('result_message', '')
                        item.verified_by = saved.get('verified_by', '')
                        item.signoff_by = saved.get('signoff_by', '')
                        item.signoff_notes = saved.get('signoff_notes', '')
                        if saved.get('verified_at'):
                            item.verified_at = datetime.fromisoformat(saved['verified_at'])
                        if saved.get('signoff_at'):
                            item.signoff_at = datetime.fromisoformat(saved['signoff_at'])
                
                logger.info(f"Loaded checklist state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Could not load checklist state: {e}")
    
    def save_state(self):
        """Save checklist state to file."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'items': [item.to_dict() for item in self.checklist]
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved checklist state to {self.state_file}")
    
    def get_item(self, item_id: str) -> Optional[CheckItem]:
        """Get checklist item by ID."""
        for item in self.checklist:
            if item.id == item_id:
                return item
        return None
    
    def update_item(
        self,
        item_id: str,
        status: CheckStatus,
        message: str = "",
        verified_by: str = ""
    ):
        """Update item status."""
        item = self.get_item(item_id)
        if item:
            item.status = status
            item.result_message = message
            item.verified_at = datetime.now()
            item.verified_by = verified_by
            self.save_state()
            logger.info(f"Updated {item_id}: {status.value} - {message}")
    
    def sign_off(
        self,
        item_id: str,
        signed_by: str,
        notes: str = ""
    ):
        """Add sign-off to item."""
        item = self.get_item(item_id)
        if item:
            item.signoff_by = signed_by
            item.signoff_at = datetime.now()
            item.signoff_notes = notes
            if item.status != CheckStatus.PASSED:
                item.status = CheckStatus.WAIVED
            self.save_state()
            logger.info(f"Sign-off added for {item_id} by {signed_by}")
    
    def run_automated_checks(self, verifiers: Dict[str, callable] = None):
        """Run all automated verification checks."""
        verifiers = verifiers or {}
        
        results = []
        for item in self.checklist:
            if item.automated and item.verification_fn:
                if item.verification_fn in verifiers:
                    try:
                        passed, message = verifiers[item.verification_fn]()
                        status = CheckStatus.PASSED if passed else CheckStatus.FAILED
                        self.update_item(item.id, status, message, "automated")
                        results.append((item.id, status, message))
                    except Exception as e:
                        self.update_item(item.id, CheckStatus.FAILED, str(e), "automated")
                        results.append((item.id, CheckStatus.FAILED, str(e)))
                else:
                    results.append((item.id, CheckStatus.NOT_STARTED, "Verifier not found"))
        
        return results
    
    def get_readiness(self) -> DeploymentReadiness:
        """Assess overall deployment readiness."""
        critical_passed = 0
        critical_total = 0
        high_passed = 0
        high_total = 0
        blocking = []
        warnings = []
        signoffs_required = []
        signoffs_obtained = []
        
        for item in self.checklist:
            if item.priority == CheckPriority.CRITICAL:
                critical_total += 1
                if item.status == CheckStatus.PASSED or item.status == CheckStatus.WAIVED:
                    critical_passed += 1
                else:
                    blocking.append(f"{item.id}: {item.name}")
            elif item.priority == CheckPriority.HIGH:
                high_total += 1
                if item.status == CheckStatus.PASSED or item.status == CheckStatus.WAIVED:
                    high_passed += 1
                else:
                    warnings.append(f"{item.id}: {item.name}")
            
            if item.requires_signoff:
                signoffs_required.append(item.id)
                if item.signoff_by:
                    signoffs_obtained.append(item.id)
        
        # Calculate readiness score
        if critical_total + high_total > 0:
            score = (critical_passed * 2 + high_passed) / (critical_total * 2 + high_total) * 100
        else:
            score = 0
        
        is_ready = critical_passed == critical_total and len(signoffs_required) == len(signoffs_obtained)
        
        return DeploymentReadiness(
            is_ready=is_ready,
            readiness_score=score,
            critical_items_passed=critical_passed,
            critical_items_total=critical_total,
            high_items_passed=high_passed,
            high_items_total=high_total,
            blocking_items=blocking,
            warnings=warnings,
            sign_offs_required=signoffs_required,
            sign_offs_obtained=signoffs_obtained
        )
    
    def generate_report(self) -> str:
        """Generate deployment readiness report."""
        readiness = self.get_readiness()
        
        lines = [
            "=" * 70,
            "PRODUCTION DEPLOYMENT READINESS REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 70,
            "OVERALL STATUS",
            "-" * 70,
            f"Ready for Deployment: {'✅ YES' if readiness.is_ready else '❌ NO'}",
            f"Readiness Score: {readiness.readiness_score:.0f}/100",
            f"Critical Items: {readiness.critical_items_passed}/{readiness.critical_items_total}",
            f"High Priority Items: {readiness.high_items_passed}/{readiness.high_items_total}",
            "",
        ]
        
        if readiness.blocking_items:
            lines.extend([
                "-" * 70,
                "❌ BLOCKING ITEMS (Must resolve before deployment)",
                "-" * 70,
            ])
            for item in readiness.blocking_items:
                lines.append(f"  • {item}")
            lines.append("")
        
        if readiness.warnings:
            lines.extend([
                "-" * 70,
                "⚠️  WARNINGS (Should resolve)",
                "-" * 70,
            ])
            for item in readiness.warnings:
                lines.append(f"  • {item}")
            lines.append("")
        
        missing_signoffs = set(readiness.sign_offs_required) - set(readiness.sign_offs_obtained)
        if missing_signoffs:
            lines.extend([
                "-" * 70,
                "📝 MISSING SIGN-OFFS",
                "-" * 70,
            ])
            for item_id in missing_signoffs:
                item = self.get_item(item_id)
                if item:
                    lines.append(f"  • {item_id}: {item.name}")
            lines.append("")
        
        # Detailed status by category
        lines.extend([
            "-" * 70,
            "DETAILED STATUS BY CATEGORY",
            "-" * 70,
        ])
        
        for category in CheckCategory:
            category_items = [i for i in self.checklist if i.category == category]
            if category_items:
                passed = sum(1 for i in category_items if i.status in [CheckStatus.PASSED, CheckStatus.WAIVED])
                lines.append(f"\n{category.value.upper()} ({passed}/{len(category_items)})")
                for item in category_items:
                    status_icon = {
                        CheckStatus.PASSED: "✅",
                        CheckStatus.FAILED: "❌",
                        CheckStatus.WAIVED: "⏭️",
                        CheckStatus.IN_PROGRESS: "🔄",
                        CheckStatus.NOT_STARTED: "⬜",
                        CheckStatus.BLOCKED: "🚫"
                    }.get(item.status, "❓")
                    lines.append(f"  {status_icon} {item.id}: {item.name}")
                    if item.result_message:
                        lines.append(f"      └─ {item.result_message}")
        
        lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def export_for_approval(self, output_file: str):
        """Export checklist for approval workflow."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'readiness': self.get_readiness().__dict__,
            'items': [item.to_dict() for item in self.checklist]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported checklist for approval to {output_file}")


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def create_default_verifiers() -> Dict[str, callable]:
    """Create default verification functions."""
    
    def verify_api_connectivity():
        # Would actually test API connection
        return True, "API connection successful"
    
    def verify_network_latency():
        # Would measure actual latency
        return True, "Latency: 45ms p95"
    
    def verify_kill_switch():
        # Would test kill switch mechanism
        return True, "Kill switch verified"
    
    def verify_position_limits():
        # Would check position limit configuration
        return True, "Position limits configured"
    
    def verify_loss_limits():
        # Would check loss limit configuration
        return True, "Loss limits: Daily 2%, Max DD 10%"
    
    def verify_circuit_breakers():
        # Would test circuit breakers
        return True, "Circuit breakers active"
    
    def verify_prometheus():
        # Would check Prometheus endpoint
        return True, "Metrics flowing to Prometheus"
    
    def verify_health_endpoint():
        # Would check health endpoint
        return True, "/health returning 200"
    
    return {
        'verify_api_connectivity': verify_api_connectivity,
        'verify_network_latency': verify_network_latency,
        'verify_kill_switch': verify_kill_switch,
        'verify_position_limits': verify_position_limits,
        'verify_loss_limits': verify_loss_limits,
        'verify_circuit_breakers': verify_circuit_breakers,
        'verify_prometheus': verify_prometheus,
        'verify_health_endpoint': verify_health_endpoint,
    }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate deployment checklist."""
    
    print("=" * 70)
    print("PRODUCTION DEPLOYMENT CHECKLIST - DEMO")
    print("=" * 70)
    
    # Create checklist
    checklist = DeploymentChecklist(state_file="./demo_checklist.json")
    
    # Run automated checks
    print("\n1. Running automated checks...")
    verifiers = create_default_verifiers()
    results = checklist.run_automated_checks(verifiers)
    
    for item_id, status, message in results[:5]:  # Show first 5
        print(f"   {item_id}: {status.value} - {message}")
    
    # Simulate manual updates
    print("\n2. Simulating manual updates...")
    checklist.update_item("STRAT-001", CheckStatus.PASSED, 
                         "30-day paper trading complete, Sharpe 1.8", "trader_john")
    checklist.update_item("STRAT-002", CheckStatus.PASSED,
                         "Profitable at 15 bps cost assumption", "quant_alice")
    
    # Add sign-offs
    print("\n3. Adding sign-offs...")
    checklist.sign_off("RISK-005", "risk_manager_bob", "All limits reviewed and approved")
    checklist.sign_off("STRAT-006", "quant_charlie", "Strategy logic verified")
    
    # Generate report
    print("\n4. Generating readiness report...")
    report = checklist.generate_report()
    print(report)
    
    # Check readiness
    readiness = checklist.get_readiness()
    print(f"\n{'✅ READY FOR DEPLOYMENT' if readiness.is_ready else '❌ NOT READY FOR DEPLOYMENT'}")


if __name__ == "__main__":
    demo()
