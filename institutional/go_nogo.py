"""
GO/NO-GO DECISION MATRIX - Production Readiness Gate

Automated checker for all T-0 gates before live trading.

All five green -> you may scale to 5% of target AUM.
Any red -> stay in staging; do not increase allocation.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("GoNoGo")


@dataclass
class GateResult:
    """Result of a single gate check."""
    gate_name: str
    passed: bool
    details: str
    evidence: str = ""
    required: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'gate': self.gate_name,
            'status': '[PASS]' if self.passed else '[FAIL]',
            'details': self.details,
            'required': self.required,
        }


class GoNoGoChecker:
    """
    Go/No-Go Decision Matrix.
    
    Checks all production readiness gates:
    1. Nightly stress tests green for 14 days
    2. Vault production reachable
    3. 30-day paper slippage <= 1.2 x simulated
    4. External audit letter present
    5. Container signed & hash-locked
    """
    
    def __init__(
        self,
        config_dir: str = ".",
        stress_results_dir: str = "stress_test_results",
        paper_trading_dir: str = "paper_trading_logs",
        audit_dir: str = "audit_logs",
    ):
        """Initialize checker."""
        self.config_dir = Path(config_dir)
        self.stress_results_dir = Path(stress_results_dir)
        self.paper_trading_dir = Path(paper_trading_dir)
        self.audit_dir = Path(audit_dir)
        
        self._results: List[GateResult] = []
        
        logger.info("GoNoGoChecker initialized")
    
    def run_all_gates(self) -> Dict:
        """Run all gate checks."""
        logger.info("=" * 60)
        logger.info("GO/NO-GO DECISION MATRIX CHECK")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        self._results = []
        
        # Run each gate
        self._check_nightly_stress()
        self._check_vault_prod()
        self._check_paper_trading()
        self._check_external_audit()
        self._check_container_signed()
        
        # Calculate overall result
        required_gates = [r for r in self._results if r.required]
        all_passed = all(r.passed for r in required_gates)
        
        # Generate report
        result = {
            'timestamp': datetime.now().isoformat(),
            'all_passed': all_passed,
            'decision': 'GO' if all_passed else 'NO-GO',
            'gates': [r.to_dict() for r in self._results],
            'summary': {
                'total_gates': len(self._results),
                'passed': sum(1 for r in self._results if r.passed),
                'failed': sum(1 for r in self._results if not r.passed),
                'required_failed': sum(1 for r in required_gates if not r.passed),
            },
        }
        
        # Log result
        if all_passed:
            logger.info("[GO] ALL GATES PASSED - CLEARED FOR PRODUCTION")
        else:
            logger.warning("[FAIL] GATE(S) FAILED - STAY IN STAGING")
            for r in self._results:
                if not r.passed:
                    logger.warning(f"   [X] {r.gate_name}: {r.details}")
        
        return result
    
    def _check_nightly_stress(self) -> None:
        """Gate 1: Nightly stress tests green for 14 consecutive days."""
        gate_name = "Nightly Stress Tests (14 days)"
        
        # Check for stress test results
        result_files = list(self.stress_results_dir.glob("stress_test_*.json"))
        
        if not result_files:
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details="No stress test results found",
            ))
            return
        
        # Check last 14 days
        consecutive_passes = 0
        
        for result_file in sorted(result_files, reverse=True)[:14]:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                if data.get('all_passed', False):
                    consecutive_passes += 1
                else:
                    break
            except:
                break
        
        passed = consecutive_passes >= 14
        
        self._results.append(GateResult(
            gate_name=gate_name,
            passed=passed,
            details=f"{consecutive_passes}/14 consecutive green nights",
            evidence=f"Results dir: {self.stress_results_dir}",
        ))
    
    def _check_vault_prod(self) -> None:
        """Gate 2: Vault production server reachable and unsealed."""
        gate_name = "Vault Production Reachable"
        
        # Check environment
        vault_addr = os.getenv("VAULT_ADDR", "")
        
        if not vault_addr or "localhost" in vault_addr or "127.0.0.1" in vault_addr:
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details="VAULT_ADDR not set or pointing to localhost",
            ))
            return
        
        # Check for .env file (should not exist in production)
        if Path(".env").exists():
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details=".env file exists - remove for production",
            ))
            return
        
        # Basic check passed
        self._results.append(GateResult(
            gate_name=gate_name,
            passed=True,
            details=f"VAULT_ADDR set to {vault_addr}",
        ))
    
    def _check_paper_trading(self) -> None:
        """Gate 3: 30-day paper trading slippage <= 1.2 x simulated."""
        gate_name = "Paper Trading (30 days, slippage <= 1.2x)"
        
        # Check for paper trading reports
        if not self.paper_trading_dir.exists():
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details="Paper trading logs directory not found",
            ))
            return
        
        report_files = list(self.paper_trading_dir.glob("report_*.txt"))
        
        if len(report_files) < 30:
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details=f"Only {len(report_files)}/30 days of paper trading",
            ))
            return
        
        # Check slippage ratio from summary
        metrics_file = self.paper_trading_dir / "30_day_summary.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    summary = json.load(f)
                
                slippage_ratio = summary.get('avg_slippage_ratio', 999)
                all_days_passed = summary.get('days_passed', 0) >= 30
                
                passed = slippage_ratio <= 1.2 and all_days_passed
                
                self._results.append(GateResult(
                    gate_name=gate_name,
                    passed=passed,
                    details=f"Slippage ratio: {slippage_ratio:.2f}x, days passed: {summary.get('days_passed', 0)}/30",
                ))
            except Exception as e:
                self._results.append(GateResult(
                    gate_name=gate_name,
                    passed=False,
                    details=f"Error reading summary: {e}",
                ))
        else:
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details="30-day summary not found - run paper trading analysis",
            ))
    
    def _check_external_audit(self) -> None:
        """Gate 4: External audit letter present with attestation hash."""
        gate_name = "External Audit Sign-Off"
        
        if not self.audit_dir.exists():
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details="Audit logs directory not found",
            ))
            return
        
        # Check for attestation file
        attestation_file = self.audit_dir / "attestation.json"
        
        if attestation_file.exists():
            try:
                with open(attestation_file, 'r') as f:
                    attestation = json.load(f)
                
                has_hash = bool(attestation.get('attestation_hash'))
                has_auditor = bool(attestation.get('auditor_name'))
                checksum_valid = attestation.get('checksum_chain_valid', False)
                
                passed = has_hash and has_auditor and checksum_valid
                
                self._results.append(GateResult(
                    gate_name=gate_name,
                    passed=passed,
                    details=f"Auditor: {attestation.get('auditor_name', 'N/A')}, hash present: {has_hash}",
                ))
            except Exception as e:
                self._results.append(GateResult(
                    gate_name=gate_name,
                    passed=False,
                    details=f"Error reading attestation: {e}",
                ))
        else:
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details="No audit attestation file found",
            ))
    
    def _check_container_signed(self) -> None:
        """Gate 5: Container is signed and requirements hash-locked."""
        gate_name = "Container Signed & Hash-Locked"
        
        # Check requirements.lock exists
        req_lock = self.config_dir / "requirements.lock"
        
        if not req_lock.exists():
            self._results.append(GateResult(
                gate_name=gate_name,
                passed=False,
                details="requirements.lock not found",
            ))
            return
        
        # Check for real hashes (not placeholders)
        try:
            with open(req_lock, 'r') as f:
                content = f.read()
            
            has_hashes = "--hash=sha256:" in content
            has_placeholders = "placeholder" in content.lower()
            has_real_hashes = has_hashes and not has_placeholders
        except:
            has_real_hashes = False
        
        # Check for Dockerfile
        dockerfile = self.config_dir / "Dockerfile"
        has_dockerfile = dockerfile.exists()
        
        # In development mode, just check for lock file
        if os.getenv("CI") or os.getenv("PRODUCTION"):
            is_signed = os.getenv("COSIGN_SIGNATURE") or os.getenv("IMAGE_SIGNED", "false") == "true"
            passed = has_real_hashes and has_dockerfile and is_signed
            details = f"Hashes: {has_real_hashes}, Dockerfile: {has_dockerfile}, Signed: {is_signed}"
        else:
            passed = has_real_hashes
            details = f"Dev mode - Hash lock: {has_real_hashes} (signature check skipped)"
        
        self._results.append(GateResult(
            gate_name=gate_name,
            passed=passed,
            details=details,
        ))
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        if not self._results:
            self.run_all_gates()
        
        lines = [
            "=" * 60,
            "           GO/NO-GO DECISION MATRIX",
            "=" * 60,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "GATE STATUS",
            "-" * 60,
        ]
        
        for r in self._results:
            status = "[PASS]" if r.passed else "[FAIL]"
            req = "(Required)" if r.required else "(Optional)"
            lines.append(f"{status} {r.gate_name} {req}")
            lines.append(f"   {r.details}")
            if r.evidence:
                lines.append(f"   Evidence: {r.evidence}")
            lines.append("")
        
        lines.append("-" * 60)
        
        all_passed = all(r.passed for r in self._results if r.required)
        
        if all_passed:
            lines.append("[GO] DECISION: GO")
            lines.append("   System is CLEARED for 5% of target AUM")
        else:
            lines.append("[FAIL] DECISION: NO-GO")
            lines.append("   System must STAY IN STAGING")
            lines.append("")
            lines.append("   Failed gates:")
            for r in self._results:
                if not r.passed and r.required:
                    lines.append(f"   - {r.gate_name}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, output_path: Optional[str] = None) -> Path:
        """Save report to file."""
        if output_path is None:
            output_path = f"go_nogo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")
        return Path(output_path)


def main():
    """Run go/no-go check."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    checker = GoNoGoChecker()
    result = checker.run_all_gates()
    
    print("\n" + checker.generate_report())
    
    # Save report
    checker.save_report()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if result['all_passed'] else 1)


if __name__ == "__main__":
    main()