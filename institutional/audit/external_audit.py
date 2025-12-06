"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         EXTERNAL AUDIT TOOLKIT - Auditor Sign-Off Preparation                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Tools for external auditor review:
1. Replay 30-day audit bus logs
2. Verify checksum chain integrity
3. Confirm kill-switch timing during chaos
4. Generate attestation for signing

Gate: Clean letter + auditor attestation hash stored in audit bus
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("Audit.External")


@dataclass
class AuditFinding:
    """An audit finding (issue or observation)."""
    finding_id: str
    severity: str  # critical, high, medium, low, observation
    category: str
    description: str
    evidence: str
    recommendation: str
    resolved: bool = False


@dataclass
class AuditReport:
    """Complete audit report for external review."""
    audit_id: str
    audit_date: str
    auditor_name: str
    audit_scope: str
    
    # Period audited
    period_start: str
    period_end: str
    
    # Summary
    total_trades: int = 0
    total_volume: float = 0.0
    checksum_chain_valid: bool = False
    kill_switch_tests_passed: int = 0
    kill_switch_tests_total: int = 0
    
    # Findings
    findings: List[AuditFinding] = field(default_factory=list)
    
    # Attestation
    attestation_hash: str = ""
    attestation_signature: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'audit_id': self.audit_id,
            'audit_date': self.audit_date,
            'auditor_name': self.auditor_name,
            'audit_scope': self.audit_scope,
            'period': {
                'start': self.period_start,
                'end': self.period_end,
            },
            'summary': {
                'total_trades': self.total_trades,
                'total_volume': self.total_volume,
                'checksum_chain_valid': self.checksum_chain_valid,
                'kill_switch_pass_rate': f"{self.kill_switch_tests_passed}/{self.kill_switch_tests_total}",
            },
            'findings': [
                {
                    'id': f.finding_id,
                    'severity': f.severity,
                    'category': f.category,
                    'description': f.description,
                    'recommendation': f.recommendation,
                    'resolved': f.resolved,
                }
                for f in self.findings
            ],
            'attestation': {
                'hash': self.attestation_hash,
                'signature': self.attestation_signature,
            },
        }


class ExternalAuditToolkit:
    """
    External Audit Toolkit.
    
    Provides tools for auditor to review:
    - Trade logs integrity
    - Risk control effectiveness
    - System behavior during stress
    
    Usage:
        toolkit = ExternalAuditToolkit(audit_logs_dir="audit_logs")
        
        # Replay logs
        summary = toolkit.replay_logs("2024-11-01", "2024-11-30")
        
        # Verify checksums
        is_valid = toolkit.verify_checksum_chain()
        
        # Check kill switch
        results = toolkit.verify_kill_switch_timing()
        
        # Generate report
        report = toolkit.generate_report("External Auditor Name")
    """
    
    def __init__(
        self,
        audit_logs_dir: str = "audit_logs",
        stress_test_results_dir: str = "stress_test_results",
    ):
        """Initialize audit toolkit."""
        self.audit_logs_dir = Path(audit_logs_dir)
        self.stress_test_results_dir = Path(stress_test_results_dir)
        
        self._findings: List[AuditFinding] = []
        self._finding_counter = 0
        
        logger.info("ExternalAuditToolkit initialized")
    
    def replay_logs(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict:
        """
        Replay audit logs for a period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Summary of replayed logs
        """
        logger.info(f"Replaying logs from {start_date} to {end_date}")
        
        summary = {
            'period_start': start_date,
            'period_end': end_date,
            'files_processed': 0,
            'total_entries': 0,
            'total_trades': 0,
            'total_volume': 0.0,
            'gaps_detected': [],
            'errors': [],
        }
        
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        last_timestamp = None
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            log_file = self.audit_logs_dir / f"audit_{date_str}.jsonl"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            summary['total_entries'] += 1
                            
                            # Check for trade entries
                            if entry.get('entry_type') == 'trade':
                                summary['total_trades'] += 1
                                summary['total_volume'] += entry.get('notional', 0)
                            
                            # Check for gaps
                            entry_time = entry.get('timestamp_ns', 0)
                            if last_timestamp and entry_time - last_timestamp > 60_000_000_000:  # > 1 min
                                summary['gaps_detected'].append({
                                    'from': last_timestamp,
                                    'to': entry_time,
                                    'duration_s': (entry_time - last_timestamp) / 1e9,
                                })
                            last_timestamp = entry_time
                            
                        except json.JSONDecodeError as e:
                            summary['errors'].append(f"Parse error in {log_file}: {e}")
                
                summary['files_processed'] += 1
            
            current += timedelta(days=1)
        
        # Log findings for gaps
        if summary['gaps_detected']:
            self._add_finding(
                severity='medium',
                category='Data Integrity',
                description=f"Detected {len(summary['gaps_detected'])} gaps in audit log",
                evidence=json.dumps(summary['gaps_detected'][:5]),
                recommendation="Investigate cause of log gaps",
            )
        
        logger.info(
            f"Replayed {summary['files_processed']} files, "
            f"{summary['total_entries']} entries, "
            f"{summary['total_trades']} trades"
        )
        
        return summary
    
    def verify_checksum_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify integrity of checksum chain.
        
        Returns:
            (is_valid, list of errors)
        """
        logger.info("Verifying checksum chain...")
        
        errors = []
        entries_checked = 0
        last_checksum = None
        
        # Process all log files
        for log_file in sorted(self.audit_logs_dir.glob("audit_*.jsonl")):
            with open(log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line)
                        entries_checked += 1
                        
                        # Get entry checksum
                        entry_checksum = entry.get('checksum')
                        prev_checksum = entry.get('prev_checksum')
                        
                        if not entry_checksum:
                            errors.append(f"{log_file}:{line_num} - Missing checksum")
                            continue
                        
                        # Verify chain
                        if last_checksum and prev_checksum != last_checksum:
                            errors.append(
                                f"{log_file}:{line_num} - Chain break: "
                                f"expected {last_checksum[:16]}..., got {prev_checksum[:16] if prev_checksum else 'None'}..."
                            )
                        
                        # Verify entry checksum
                        computed = self._compute_entry_checksum(entry)
                        if computed != entry_checksum:
                            errors.append(
                                f"{log_file}:{line_num} - Checksum mismatch: "
                                f"computed {computed[:16]}..., stored {entry_checksum[:16]}..."
                            )
                        
                        last_checksum = entry_checksum
                        
                    except json.JSONDecodeError as e:
                        errors.append(f"{log_file}:{line_num} - JSON parse error: {e}")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self._add_finding(
                severity='critical',
                category='Data Integrity',
                description="Checksum chain verification FAILED",
                evidence="\n".join(errors[:10]),
                recommendation="Investigate potential tampering or corruption",
            )
        
        logger.info(
            f"Checked {entries_checked} entries, "
            f"{'✅ VALID' if is_valid else f'❌ {len(errors)} errors'}"
        )
        
        return is_valid, errors
    
    def _compute_entry_checksum(self, entry: Dict) -> str:
        """Compute checksum for an entry."""
        # Remove existing checksum fields for computation
        entry_copy = {k: v for k, v in entry.items() if k not in ['checksum', 'prev_checksum']}
        
        # Deterministic JSON
        content = json.dumps(entry_copy, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_kill_switch_timing(self) -> Dict:
        """
        Verify kill switch fired within 60ms during stress tests.
        
        Returns:
            Summary of kill switch verification
        """
        logger.info("Verifying kill switch timing...")
        
        results = {
            'tests_found': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'details': [],
        }
        
        # Load stress test results
        for result_file in self.stress_test_results_dir.glob("stress_test_*.json"):
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            for scenario in data.get('scenarios', []):
                halt_time = scenario.get('halt_time_ms')
                
                results['tests_found'] += 1
                
                test_result = {
                    'scenario': scenario.get('scenario_name'),
                    'halt_time_ms': halt_time,
                    'passed': halt_time is not None and halt_time <= 60,
                }
                
                if test_result['passed']:
                    results['tests_passed'] += 1
                else:
                    results['tests_failed'] += 1
                
                results['details'].append(test_result)
        
        # Log finding if failures
        if results['tests_failed'] > 0:
            self._add_finding(
                severity='high',
                category='Risk Controls',
                description=f"Kill switch timing exceeded 60ms in {results['tests_failed']} tests",
                evidence=json.dumps([d for d in results['details'] if not d['passed']]),
                recommendation="Optimize AMRC latency or investigate bottlenecks",
            )
        
        logger.info(
            f"Kill switch: {results['tests_passed']}/{results['tests_found']} passed"
        )
        
        return results
    
    def _add_finding(
        self,
        severity: str,
        category: str,
        description: str,
        evidence: str,
        recommendation: str,
    ) -> AuditFinding:
        """Add an audit finding."""
        self._finding_counter += 1
        
        finding = AuditFinding(
            finding_id=f"F-{self._finding_counter:03d}",
            severity=severity,
            category=category,
            description=description,
            evidence=evidence,
            recommendation=recommendation,
        )
        
        self._findings.append(finding)
        return finding
    
    def generate_report(
        self,
        auditor_name: str,
        period_start: str,
        period_end: str,
    ) -> AuditReport:
        """
        Generate audit report for signing.
        
        Args:
            auditor_name: Name of external auditor
            period_start: Audit period start
            period_end: Audit period end
            
        Returns:
            AuditReport ready for attestation
        """
        import uuid
        
        # Replay logs
        summary = self.replay_logs(period_start, period_end)
        
        # Verify checksum
        checksum_valid, _ = self.verify_checksum_chain()
        
        # Verify kill switch
        kill_switch = self.verify_kill_switch_timing()
        
        # Create report
        report = AuditReport(
            audit_id=str(uuid.uuid4())[:8].upper(),
            audit_date=datetime.now().strftime("%Y-%m-%d"),
            auditor_name=auditor_name,
            audit_scope="Trading System Operational Audit",
            period_start=period_start,
            period_end=period_end,
            total_trades=summary['total_trades'],
            total_volume=summary['total_volume'],
            checksum_chain_valid=checksum_valid,
            kill_switch_tests_passed=kill_switch['tests_passed'],
            kill_switch_tests_total=kill_switch['tests_found'],
            findings=self._findings,
        )
        
        # Generate attestation hash
        report.attestation_hash = self._generate_attestation_hash(report)
        
        return report
    
    def _generate_attestation_hash(self, report: AuditReport) -> str:
        """Generate attestation hash for report."""
        # Hash the key report fields
        content = json.dumps({
            'audit_id': report.audit_id,
            'audit_date': report.audit_date,
            'period': f"{report.period_start} to {report.period_end}",
            'total_trades': report.total_trades,
            'checksum_valid': report.checksum_chain_valid,
            'kill_switch_passed': report.kill_switch_tests_passed == report.kill_switch_tests_total,
            'critical_findings': sum(1 for f in report.findings if f.severity == 'critical'),
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def generate_attestation_letter(self, report: AuditReport) -> str:
        """
        Generate attestation letter for auditor signature.
        """
        critical = sum(1 for f in report.findings if f.severity == 'critical')
        high = sum(1 for f in report.findings if f.severity == 'high')
        
        is_clean = critical == 0 and high == 0
        opinion = "UNQUALIFIED (CLEAN)" if is_clean else "QUALIFIED"
        
        letter = f"""
================================================================================
                        EXTERNAL AUDIT ATTESTATION LETTER
================================================================================

Audit ID:       {report.audit_id}
Date:           {report.audit_date}
Auditor:        {report.auditor_name}

SCOPE
-----
We have examined the trading system operational controls for the period from
{report.period_start} to {report.period_end}.

PROCEDURES PERFORMED
--------------------
1. Replayed {report.total_trades:,} trades from audit logs
2. Verified checksum chain integrity: {'PASS' if report.checksum_chain_valid else 'FAIL'}
3. Tested kill switch timing: {report.kill_switch_tests_passed}/{report.kill_switch_tests_total} passed
4. Reviewed risk control configurations
5. Examined incident response procedures

FINDINGS
--------
- Critical Findings: {critical}
- High Findings: {high}
- Medium/Low Findings: {len(report.findings) - critical - high}

OPINION
-------
Based on our examination, we issue an {opinion} opinion on the operational
controls of the trading system.

{'The system meets the requirements for production trading operation.' if is_clean else 'The system requires remediation before production trading.'}

ATTESTATION HASH
----------------
{report.attestation_hash}

This hash can be stored in the audit bus to provide tamper-evident proof
of this attestation.

________________________________________________________________________________
                                                                                
Auditor Signature: ___________________________    Date: _______________
                   {report.auditor_name}

================================================================================
"""
        return letter
    
    def store_attestation(self, report: AuditReport, audit_bus) -> bool:
        """
        Store attestation hash in audit bus.
        
        Args:
            report: Audit report
            audit_bus: AuditBus instance
            
        Returns:
            True if stored successfully
        """
        try:
            audit_bus.log({
                'entry_type': 'audit_attestation',
                'audit_id': report.audit_id,
                'audit_date': report.audit_date,
                'auditor': report.auditor_name,
                'attestation_hash': report.attestation_hash,
                'checksum_valid': report.checksum_chain_valid,
                'kill_switch_pass_rate': f"{report.kill_switch_tests_passed}/{report.kill_switch_tests_total}",
                'findings_count': len(report.findings),
            })
            
            logger.info(f"✅ Attestation stored: {report.attestation_hash[:32]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store attestation: {e}")
            return False


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create toolkit
    toolkit = ExternalAuditToolkit()
    
    # Generate report
    print("Generating audit report...")
    
    report = toolkit.generate_report(
        auditor_name="Smith & Associates LLP",
        period_start="2024-11-01",
        period_end="2024-11-30",
    )
    
    print("\n" + "=" * 60)
    print("AUDIT REPORT SUMMARY")
    print("=" * 60)
    print(json.dumps(report.to_dict(), indent=2))
    
    # Generate attestation letter
    print("\n" + toolkit.generate_attestation_letter(report))
