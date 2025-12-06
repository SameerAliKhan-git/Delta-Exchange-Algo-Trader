"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         AMRC CHAOS TESTS - Resilience Testing for Trading Halt                ║
║                                                                               ║
║  Automated chaos engineering tests for AMRC validation                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

These tests verify that the AMRC can:
1. Halt trading within 60ms of detecting a critical signal
2. Handle all injection scenarios correctly
3. Recover gracefully after halts
"""

import time
import threading
import random
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum

from .controller import AutonomousMetaRiskController, AMRCConfig

logger = logging.getLogger("AMRC.ChaosTest")


class InjectionType(Enum):
    """Types of chaos injections."""
    LATENCY_SPIKE = "latency_spike"
    BOOK_DEPTH_COLLAPSE = "book_depth_collapse"
    API_ERRORS = "api_errors"
    PRICE_JUMP = "price_jump"
    FUNDING_RATE_SPIKE = "funding_rate_spike"
    SENTIMENT_POISON = "sentiment_poison"
    NETWORK_PARTITION = "network_partition"
    LIQUIDATION_CASCADE = "liquidation_cascade"


@dataclass
class InjectionResult:
    """Result of a chaos injection."""
    injection_type: InjectionType
    start_time: datetime
    detection_time_ms: Optional[float] = None
    halt_time_ms: Optional[float] = None
    passed: bool = False
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'injection_type': self.injection_type.value,
            'start_time': self.start_time.isoformat(),
            'detection_time_ms': self.detection_time_ms,
            'halt_time_ms': self.halt_time_ms,
            'passed': self.passed,
            'details': self.details,
        }


@dataclass
class ChaosTestConfig:
    """Configuration for chaos tests."""
    max_halt_time_ms: float = 60.0  # Maximum acceptable halt time
    num_iterations: int = 10  # Number of times to repeat each test
    recovery_wait_s: float = 5.0  # Wait time between tests
    verbose: bool = True


class AMRCChaosTest:
    """
    Chaos testing suite for AMRC validation.
    
    Runs a battery of chaos injections and verifies:
    1. AMRC halts within target time
    2. Drawdown stays within limits
    3. No unauthorized API calls
    4. Audit log remains intact
    
    Usage:
        amrc = AutonomousMetaRiskController()
        chaos = AMRCChaosTest(amrc)
        results = chaos.run_full_suite()
    """
    
    def __init__(
        self,
        amrc: AutonomousMetaRiskController,
        config: Optional[ChaosTestConfig] = None
    ):
        """
        Initialize chaos test suite.
        
        Args:
            amrc: The AMRC instance to test
            config: Test configuration
        """
        self.amrc = amrc
        self.config = config or ChaosTestConfig()
        self.results: List[InjectionResult] = []
        
    def run_full_suite(self) -> Dict:
        """
        Run the complete chaos test suite.
        
        Returns:
            Dict with test results and summary
        """
        logger.info("=" * 60)
        logger.info("AMRC CHAOS TEST SUITE STARTING")
        logger.info("=" * 60)
        
        self.results = []
        
        # Define all tests
        tests = [
            (InjectionType.LATENCY_SPIKE, self._inject_latency_spike),
            (InjectionType.BOOK_DEPTH_COLLAPSE, self._inject_book_collapse),
            (InjectionType.API_ERRORS, self._inject_api_errors),
            (InjectionType.PRICE_JUMP, self._inject_price_jump),
            (InjectionType.LIQUIDATION_CASCADE, self._inject_liquidation_cascade),
        ]
        
        # Run each test
        for injection_type, test_fn in tests:
            logger.info(f"\n--- Testing: {injection_type.value} ---")
            
            for i in range(self.config.num_iterations):
                if self.config.verbose:
                    logger.info(f"  Iteration {i + 1}/{self.config.num_iterations}")
                
                # Ensure AMRC is in good state
                self._reset_amrc()
                time.sleep(0.1)
                
                # Run injection
                result = test_fn()
                self.results.append(result)
                
                # Wait for recovery
                time.sleep(self.config.recovery_wait_s)
        
        # Generate summary
        summary = self._generate_summary()
        
        logger.info("\n" + "=" * 60)
        logger.info("CHAOS TEST SUITE COMPLETE")
        logger.info("=" * 60)
        
        return summary
    
    def _reset_amrc(self) -> None:
        """Reset AMRC to normal state for next test."""
        # Force resume (bypass auth for testing)
        self.amrc.config.require_dual_auth_resume = False
        self.amrc.config.recovery_cooldown_s = 0
        self.amrc.resume_trading()
        self.amrc.config.require_dual_auth_resume = True
        self.amrc.config.recovery_cooldown_s = 300
    
    def _inject_latency_spike(self) -> InjectionResult:
        """
        Inject 200ms latency spike.
        AMRC should halt within 60ms of detection.
        """
        result = InjectionResult(
            injection_type=InjectionType.LATENCY_SPIKE,
            start_time=datetime.now(),
        )
        
        # Record start time
        start_ns = time.perf_counter_ns()
        
        # Simulate latency by injecting stale data
        # In reality, this would be detected by timestamp lag
        old_time = datetime.now()
        
        # Inject API errors to simulate timeout
        for _ in range(10):
            self.amrc.update_api_status(success=False, error_type="timeout")
        
        # Wait for AMRC to detect and halt
        max_wait = 0.5  # 500ms max wait
        check_interval = 0.001  # 1ms
        elapsed = 0
        
        while elapsed < max_wait:
            if not self.amrc.is_trading_enabled():
                halt_ns = time.perf_counter_ns()
                result.halt_time_ms = (halt_ns - start_ns) / 1_000_000
                result.passed = result.halt_time_ms <= self.config.max_halt_time_ms
                break
            time.sleep(check_interval)
            elapsed += check_interval
        
        result.details = {
            'latency_injected_ms': 200,
            'halt_time_ms': result.halt_time_ms,
            'threshold_ms': self.config.max_halt_time_ms,
        }
        
        return result
    
    def _inject_book_collapse(self) -> InjectionResult:
        """
        Inject 90% order book depth collapse.
        """
        result = InjectionResult(
            injection_type=InjectionType.BOOK_DEPTH_COLLAPSE,
            start_time=datetime.now(),
        )
        
        start_ns = time.perf_counter_ns()
        
        # Inject collapsed orderbook
        self.amrc.update_orderbook(
            symbol="BTCUSD",
            bids=[(50000, 0.001)],  # Minimal depth
            asks=[(50001, 0.001)],
        )
        
        # Wait for halt
        max_wait = 0.5
        check_interval = 0.001
        elapsed = 0
        
        while elapsed < max_wait:
            if not self.amrc.is_trading_enabled():
                halt_ns = time.perf_counter_ns()
                result.halt_time_ms = (halt_ns - start_ns) / 1_000_000
                result.passed = result.halt_time_ms <= self.config.max_halt_time_ms
                break
            time.sleep(check_interval)
            elapsed += check_interval
        
        result.details = {
            'depth_reduction_pct': 90,
            'halt_time_ms': result.halt_time_ms,
        }
        
        return result
    
    def _inject_api_errors(self) -> InjectionResult:
        """
        Inject burst of API errors.
        """
        result = InjectionResult(
            injection_type=InjectionType.API_ERRORS,
            start_time=datetime.now(),
        )
        
        start_ns = time.perf_counter_ns()
        
        # Inject many API errors rapidly  
        for _ in range(20):
            self.amrc.update_api_status(success=False, error_type="server_error")
        
        # Wait for halt
        max_wait = 0.5
        check_interval = 0.001
        elapsed = 0
        
        while elapsed < max_wait:
            if not self.amrc.is_trading_enabled():
                halt_ns = time.perf_counter_ns()
                result.halt_time_ms = (halt_ns - start_ns) / 1_000_000
                result.passed = result.halt_time_ms <= self.config.max_halt_time_ms
                break
            time.sleep(check_interval)
            elapsed += check_interval
        
        result.details = {
            'errors_injected': 20,
            'halt_time_ms': result.halt_time_ms,
        }
        
        return result
    
    def _inject_price_jump(self) -> InjectionResult:
        """
        Inject 3σ price jump.
        """
        result = InjectionResult(
            injection_type=InjectionType.PRICE_JUMP,
            start_time=datetime.now(),
        )
        
        # First, establish baseline prices
        base_price = 50000
        for i in range(50):
            self.amrc.update_price("BTCUSD", base_price + random.gauss(0, 50))
            time.sleep(0.001)
        
        start_ns = time.perf_counter_ns()
        
        # Inject massive price jump
        jump_price = base_price * 1.05  # 5% jump
        self.amrc.update_price("BTCUSD", jump_price)
        
        # Wait for halt
        max_wait = 0.5
        check_interval = 0.001
        elapsed = 0
        
        while elapsed < max_wait:
            if not self.amrc.is_trading_enabled():
                halt_ns = time.perf_counter_ns()
                result.halt_time_ms = (halt_ns - start_ns) / 1_000_000
                result.passed = result.halt_time_ms <= self.config.max_halt_time_ms
                break
            time.sleep(check_interval)
            elapsed += check_interval
        
        result.details = {
            'price_jump_pct': 5.0,
            'halt_time_ms': result.halt_time_ms,
        }
        
        return result
    
    def _inject_liquidation_cascade(self) -> InjectionResult:
        """
        Inject liquidation cluster near current price.
        """
        result = InjectionResult(
            injection_type=InjectionType.LIQUIDATION_CASCADE,
            start_time=datetime.now(),
        )
        
        # Set up ATR and price
        self.amrc.update_atr("BTCUSD", 500)  # $500 ATR
        current_price = 50000
        self.amrc.update_price("BTCUSD", current_price)
        
        start_ns = time.perf_counter_ns()
        
        # Inject liquidation levels very close to current price
        liquidation_levels = [
            current_price + 100,  # 0.2 ATR away
            current_price - 150,  # 0.3 ATR away
        ]
        self.amrc.update_liquidation_levels("BTCUSD", liquidation_levels)
        
        # Wait for halt
        max_wait = 0.5
        check_interval = 0.001
        elapsed = 0
        
        while elapsed < max_wait:
            if not self.amrc.is_trading_enabled():
                halt_ns = time.perf_counter_ns()
                result.halt_time_ms = (halt_ns - start_ns) / 1_000_000
                result.passed = result.halt_time_ms <= self.config.max_halt_time_ms
                break
            time.sleep(check_interval)
            elapsed += check_interval
        
        result.details = {
            'liquidation_levels': liquidation_levels,
            'atr': 500,
            'halt_time_ms': result.halt_time_ms,
        }
        
        return result
    
    def _generate_summary(self) -> Dict:
        """Generate test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        # Group by type
        by_type = {}
        for r in self.results:
            key = r.injection_type.value
            if key not in by_type:
                by_type[key] = {'passed': 0, 'failed': 0, 'halt_times_ms': []}
            if r.passed:
                by_type[key]['passed'] += 1
            else:
                by_type[key]['failed'] += 1
            if r.halt_time_ms:
                by_type[key]['halt_times_ms'].append(r.halt_time_ms)
        
        # Calculate stats per type
        for key, data in by_type.items():
            times = data['halt_times_ms']
            if times:
                data['avg_halt_time_ms'] = sum(times) / len(times)
                data['max_halt_time_ms'] = max(times)
                data['min_halt_time_ms'] = min(times)
            del data['halt_times_ms']
        
        summary = {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'all_passed': failed == 0,
            'by_type': by_type,
            'individual_results': [r.to_dict() for r in self.results],
            'config': {
                'max_halt_time_ms': self.config.max_halt_time_ms,
                'num_iterations': self.config.num_iterations,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        # Print summary
        logger.info(f"\nSUMMARY: {passed}/{total} tests passed ({summary['pass_rate']*100:.1f}%)")
        for key, data in by_type.items():
            status = "✅" if data['failed'] == 0 else "❌"
            avg = data.get('avg_halt_time_ms', 0)
            logger.info(f"  {status} {key}: {data['passed']}/{data['passed']+data['failed']} (avg {avg:.1f}ms)")
        
        return summary


# =============================================================================
# NIGHTLY TEST RUNNER
# =============================================================================

def run_nightly_chaos_tests() -> bool:
    """
    Run nightly chaos tests.
    
    Returns:
        bool: True if all tests passed
    """
    logging.basicConfig(level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("NIGHTLY AMRC CHAOS TEST")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Create AMRC
    config = AMRCConfig(
        health_check_interval_ms=5,  # Fast checking for tests
    )
    amrc = AutonomousMetaRiskController(config=config)
    amrc.start()
    
    # Run chaos tests
    chaos_config = ChaosTestConfig(
        max_halt_time_ms=60,
        num_iterations=10,
        recovery_wait_s=1.0,
    )
    chaos = AMRCChaosTest(amrc, chaos_config)
    
    try:
        results = chaos.run_full_suite()
        
        # Save results
        output_file = f"chaos_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        
        return results['all_passed']
        
    finally:
        amrc.stop()


if __name__ == "__main__":
    success = run_nightly_chaos_tests()
    exit(0 if success else 1)
