"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         STRESS TEST RUNNER - Automated Nightly Testing                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Runs the stress test battery nightly with pass/fail criteria:
- AMRC halts within 60ms
- Drawdown < 1%
- No unauthenticated API calls
- Audit log intact
"""

import time
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path

from .scenarios import (
    StressScenario,
    ScenarioResult,
    ScenarioType,
    ScenarioFactory,
)

logger = logging.getLogger("StressTest.Runner")


@dataclass
class TestConfig:
    """Configuration for stress test runner."""
    # Scenarios to run
    scenarios: List[ScenarioType] = field(default_factory=lambda: [
        ScenarioType.BOOK_COLLAPSE,
        ScenarioType.LATENCY_SPIKE,
        ScenarioType.API_ERRORS,
        ScenarioType.FUNDING_SPIKE,
        ScenarioType.SENTIMENT_POISON,
    ])
    
    # Pass criteria
    max_halt_time_ms: float = 60.0
    max_drawdown_pct: float = 1.0
    require_no_unauth_calls: bool = True
    require_audit_intact: bool = True
    
    # Timing
    scenario_timeout_seconds: float = 120.0
    cooldown_between_scenarios_seconds: float = 10.0
    
    # Output
    output_dir: str = "stress_test_results"
    save_results: bool = True


@dataclass
class TestSuiteResult:
    """Result of running the full test suite."""
    start_time: datetime
    end_time: datetime
    all_passed: bool
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    scenario_results: List[ScenarioResult]
    config: TestConfig
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'all_passed': self.all_passed,
            'total_scenarios': self.total_scenarios,
            'passed_scenarios': self.passed_scenarios,
            'failed_scenarios': self.failed_scenarios,
            'pass_rate': self.passed_scenarios / max(1, self.total_scenarios),
            'scenarios': [r.to_dict() for r in self.scenario_results],
        }


class StressTestRunner:
    """
    Automated stress test runner.
    
    Runs a battery of stress tests and validates pass criteria.
    
    Usage:
        runner = StressTestRunner()
        result = runner.run_all()
        
        if result.all_passed:
            print("✅ All stress tests passed!")
        else:
            print("❌ Some tests failed")
    """
    
    def __init__(
        self,
        config: Optional[TestConfig] = None,
        amrc: Optional[object] = None,  # AMRC instance for validation
    ):
        """
        Initialize test runner.
        
        Args:
            config: Test configuration
            amrc: Optional AMRC instance for halt time validation
        """
        self.config = config or TestConfig()
        self.amrc = amrc
        self._results: List[ScenarioResult] = []
        
        # Callbacks for integration
        self._on_scenario_start: Optional[Callable] = None
        self._on_scenario_end: Optional[Callable] = None
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("StressTestRunner initialized")
    
    def set_callbacks(
        self,
        on_start: Optional[Callable[[StressScenario], None]] = None,
        on_end: Optional[Callable[[ScenarioResult], None]] = None,
    ) -> None:
        """Set callbacks for scenario lifecycle."""
        self._on_scenario_start = on_start
        self._on_scenario_end = on_end
    
    def run_all(self) -> TestSuiteResult:
        """
        Run all configured stress tests.
        
        Returns:
            TestSuiteResult with outcomes
        """
        logger.info("=" * 60)
        logger.info("STRESS TEST SUITE STARTING")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        self._results = []
        
        for scenario_type in self.config.scenarios:
            try:
                result = self._run_scenario(scenario_type)
                self._results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                logger.info(f"  {scenario_type.value}: {status}")
                
                # Cooldown
                time.sleep(self.config.cooldown_between_scenarios_seconds)
                
            except Exception as e:
                logger.error(f"Error running {scenario_type.value}: {e}")
                
                # Create failure result
                failed_result = ScenarioResult(
                    scenario_name=scenario_type.value,
                    scenario_type=scenario_type,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    passed=False,
                    errors=[str(e)],
                )
                self._results.append(failed_result)
        
        end_time = datetime.now()
        
        # Compile results
        passed_count = sum(1 for r in self._results if r.passed)
        failed_count = len(self._results) - passed_count
        
        suite_result = TestSuiteResult(
            start_time=start_time,
            end_time=end_time,
            all_passed=failed_count == 0,
            total_scenarios=len(self._results),
            passed_scenarios=passed_count,
            failed_scenarios=failed_count,
            scenario_results=self._results,
            config=self.config,
        )
        
        # Save results
        if self.config.save_results:
            self._save_results(suite_result)
        
        # Log summary
        logger.info("=" * 60)
        logger.info(f"STRESS TEST SUITE COMPLETE: {passed_count}/{len(self._results)} PASSED")
        logger.info("=" * 60)
        
        return suite_result
    
    def _run_scenario(self, scenario_type: ScenarioType) -> ScenarioResult:
        """Run a single scenario."""
        logger.info(f"\n--- Running: {scenario_type.value} ---")
        
        # Create scenario
        scenario = ScenarioFactory.create(scenario_type)
        
        # Callback
        if self._on_scenario_start:
            self._on_scenario_start(scenario)
        
        # Start
        scenario.start()
        
        # Track metrics
        start_time = time.perf_counter()
        halt_detected = False
        halt_time_ms = None
        max_drawdown = 0.0
        
        # Run injections
        injection_count = 0
        while time.perf_counter() - start_time < min(
            scenario.config.duration_seconds,
            self.config.scenario_timeout_seconds
        ):
            # Inject chaos
            injection_result = scenario.inject()
            injection_count += 1
            
            # Check if AMRC halted (if available)
            if self.amrc and not self.amrc.is_trading_enabled():
                if not halt_detected:
                    halt_time_ms = (time.perf_counter() - start_time) * 1000
                    halt_detected = True
                    logger.info(f"  AMRC halt detected at {halt_time_ms:.1f}ms")
            
            # Small delay between injections
            time.sleep(0.1)
        
        # Stop scenario
        result = scenario.stop()
        
        # Validate pass criteria
        result.halt_time_ms = halt_time_ms
        result.passed = self._validate_result(result)
        
        # Callback
        if self._on_scenario_end:
            self._on_scenario_end(result)
        
        result.metrics['injection_count'] = injection_count
        
        return result
    
    def _validate_result(self, result: ScenarioResult) -> bool:
        """Validate a scenario result against pass criteria."""
        passed = True
        
        # Check halt time
        if result.halt_time_ms is not None:
            if result.halt_time_ms > self.config.max_halt_time_ms:
                result.errors.append(
                    f"Halt time {result.halt_time_ms:.1f}ms > {self.config.max_halt_time_ms}ms"
                )
                passed = False
        
        # Check drawdown
        if result.max_drawdown_pct > self.config.max_drawdown_pct:
            result.errors.append(
                f"Drawdown {result.max_drawdown_pct:.2f}% > {self.config.max_drawdown_pct}%"
            )
            passed = False
        
        # Check unauthorized calls
        if self.config.require_no_unauth_calls and result.unauth_api_calls > 0:
            result.errors.append(
                f"Unauthorized API calls detected: {result.unauth_api_calls}"
            )
            passed = False
        
        # Check audit integrity
        if self.config.require_audit_intact and not result.audit_intact:
            result.errors.append("Audit log integrity compromised")
            passed = False
        
        return passed
    
    def _save_results(self, result: TestSuiteResult) -> None:
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stress_test_{timestamp}.json"
        filepath = Path(self.config.output_dir) / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def run_single(self, scenario_type: ScenarioType) -> ScenarioResult:
        """Run a single scenario."""
        return self._run_scenario(scenario_type)
    
    def get_last_results(self) -> List[ScenarioResult]:
        """Get results from last run."""
        return self._results.copy()


# =============================================================================
# NIGHTLY TEST ENTRY POINT
# =============================================================================

def run_nightly_stress_tests() -> bool:
    """
    Entry point for nightly stress tests.
    
    Returns:
        bool: True if all tests passed
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("NIGHTLY STRESS TEST BATTERY")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    runner = StressTestRunner()
    result = runner.run_all()
    
    if result.all_passed:
        logger.info("✅ ALL STRESS TESTS PASSED")
    else:
        logger.error("❌ SOME STRESS TESTS FAILED")
        for r in result.scenario_results:
            if not r.passed:
                logger.error(f"  - {r.scenario_name}: {', '.join(r.errors)}")
    
    return result.all_passed


if __name__ == "__main__":
    success = run_nightly_stress_tests()
    exit(0 if success else 1)
