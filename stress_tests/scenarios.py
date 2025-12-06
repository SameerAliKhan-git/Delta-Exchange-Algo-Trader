"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         STRESS TEST SCENARIOS - Chaos Injection Definitions                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Defines stress test scenarios for nightly automated testing:
1. 90% book depth collapse in 200ms
2. 200ms added latency (asymmetric, inbound only)
3. 5% trades return "invalid" for 30 min
4. Funding rate jumps to ±0.5%
5. Static sentiment for 4h (poison test)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger("StressTest.Scenarios")


class ScenarioType(Enum):
    """Types of stress scenarios."""
    BOOK_COLLAPSE = "book_depth_collapse"
    LATENCY_SPIKE = "latency_spike"
    API_ERRORS = "api_errors"
    FUNDING_SPIKE = "funding_rate_spike"
    SENTIMENT_POISON = "sentiment_poison"
    NETWORK_PARTITION = "network_partition"
    PRICE_FLASH_CRASH = "price_flash_crash"
    LIQUIDATION_CASCADE = "liquidation_cascade"


@dataclass
class ScenarioConfig:
    """Configuration for a stress scenario."""
    scenario_type: ScenarioType
    name: str
    description: str
    
    # Timing
    duration_seconds: float = 60.0
    ramp_up_seconds: float = 0.0
    ramp_down_seconds: float = 0.0
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Pass criteria
    max_drawdown_pct: float = 1.0
    max_halt_time_ms: float = 60.0
    require_audit_intact: bool = True
    require_no_unauth_calls: bool = True


@dataclass
class ScenarioResult:
    """Result of running a stress scenario."""
    scenario_name: str
    scenario_type: ScenarioType
    start_time: datetime
    end_time: datetime
    
    # Outcomes
    passed: bool
    halt_time_ms: Optional[float] = None
    max_drawdown_pct: float = 0.0
    unauth_api_calls: int = 0
    audit_intact: bool = True
    
    # Details
    events: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Config reference
    config: Optional[ScenarioConfig] = None
    
    def to_dict(self) -> Dict:
        return {
            'scenario_name': self.scenario_name,
            'scenario_type': self.scenario_type.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'passed': self.passed,
            'halt_time_ms': self.halt_time_ms,
            'max_drawdown_pct': self.max_drawdown_pct,
            'unauth_api_calls': self.unauth_api_calls,
            'audit_intact': self.audit_intact,
            'events_count': len(self.events),
            'errors': self.errors,
            'metrics': self.metrics,
        }


class StressScenario:
    """
    Base class for stress test scenarios.
    
    Defines the interface for implementing chaos injections.
    """
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.is_active = False
        self._start_time: Optional[datetime] = None
        self._events: List[Dict] = []
    
    def start(self) -> None:
        """Start the scenario."""
        self.is_active = True
        self._start_time = datetime.now()
        self._events = []
        self._log_event("scenario_started", {})
        logger.info(f"Starting scenario: {self.config.name}")
    
    def stop(self) -> ScenarioResult:
        """Stop the scenario and return results."""
        self.is_active = False
        end_time = datetime.now()
        self._log_event("scenario_stopped", {})
        
        result = ScenarioResult(
            scenario_name=self.config.name,
            scenario_type=self.config.scenario_type,
            start_time=self._start_time or datetime.now(),
            end_time=end_time,
            passed=True,  # Override in subclass
            events=self._events,
            config=self.config,
        )
        
        logger.info(f"Stopped scenario: {self.config.name}")
        return result
    
    def inject(self) -> Dict[str, Any]:
        """
        Inject chaos into the system.
        
        Returns:
            Dict with injection details
        """
        raise NotImplementedError("Subclasses must implement inject()")
    
    def _log_event(self, event_type: str, details: Dict) -> None:
        """Log an event during the scenario."""
        self._events.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
        })


class BookCollapseScenario(StressScenario):
    """
    Scenario: 90% order book depth collapse in 200ms.
    
    Simulates sudden removal of liquidity from the order book.
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        if config is None:
            config = ScenarioConfig(
                scenario_type=ScenarioType.BOOK_COLLAPSE,
                name="Book Depth Collapse",
                description="90% reduction in order book depth within 200ms",
                duration_seconds=60,
                parameters={
                    'depth_reduction_pct': 90,
                    'collapse_time_ms': 200,
                    'affected_levels': 10,
                },
            )
        super().__init__(config)
    
    def inject(self) -> Dict[str, Any]:
        """Generate collapsed order book."""
        reduction = self.config.parameters.get('depth_reduction_pct', 90) / 100
        
        # Simulate collapsed book
        collapsed_book = {
            'bids': [(50000 - i*10, 0.001 * (1 - reduction)) for i in range(5)],
            'asks': [(50001 + i*10, 0.001 * (1 - reduction)) for i in range(5)],
            'reduction_applied': reduction,
        }
        
        self._log_event("book_collapse_injected", collapsed_book)
        
        return collapsed_book


class LatencySpikeScenario(StressScenario):
    """
    Scenario: 200ms added latency (asymmetric, inbound only).
    
    Simulates network issues affecting incoming data but not outgoing orders.
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        if config is None:
            config = ScenarioConfig(
                scenario_type=ScenarioType.LATENCY_SPIKE,
                name="Asymmetric Latency Spike",
                description="200ms added inbound latency",
                duration_seconds=60,
                parameters={
                    'latency_ms': 200,
                    'direction': 'inbound',
                    'jitter_ms': 50,
                },
            )
        super().__init__(config)
    
    def inject(self) -> Dict[str, Any]:
        """Generate latency parameters."""
        import time
        
        base_latency = self.config.parameters.get('latency_ms', 200)
        jitter = self.config.parameters.get('jitter_ms', 50)
        
        actual_latency = base_latency + np.random.uniform(-jitter, jitter)
        
        # Simulate the delay
        time.sleep(actual_latency / 1000)
        
        result = {
            'base_latency_ms': base_latency,
            'actual_latency_ms': actual_latency,
            'direction': 'inbound',
        }
        
        self._log_event("latency_injected", result)
        
        return result


class APIErrorScenario(StressScenario):
    """
    Scenario: 5% of trades return "invalid" for 30 min.
    
    Simulates intermittent API failures.
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        if config is None:
            config = ScenarioConfig(
                scenario_type=ScenarioType.API_ERRORS,
                name="API Error Injection",
                description="5% of API calls return errors for 30 minutes",
                duration_seconds=1800,  # 30 minutes
                parameters={
                    'error_rate': 0.05,
                    'error_types': ['invalid', 'timeout', 'rate_limited'],
                },
            )
        super().__init__(config)
        self._error_count = 0
        self._total_calls = 0
    
    def inject(self) -> Dict[str, Any]:
        """Determine if this call should error."""
        self._total_calls += 1
        
        error_rate = self.config.parameters.get('error_rate', 0.05)
        error_types = self.config.parameters.get('error_types', ['invalid'])
        
        should_error = np.random.random() < error_rate
        
        if should_error:
            self._error_count += 1
            error_type = np.random.choice(error_types)
            
            result = {
                'success': False,
                'error_type': error_type,
                'error_message': f"Injected {error_type} error",
            }
            
            self._log_event("api_error_injected", result)
        else:
            result = {
                'success': True,
            }
        
        return result


class FundingSpikeScenario(StressScenario):
    """
    Scenario: Funding rate jumps to ±0.5%.
    
    Simulates extreme funding rate conditions.
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        if config is None:
            config = ScenarioConfig(
                scenario_type=ScenarioType.FUNDING_SPIKE,
                name="Funding Rate Spike",
                description="Funding rate jumps to extreme values",
                duration_seconds=3600,  # 1 hour
                parameters={
                    'funding_rate': 0.005,  # 0.5%
                    'direction': 'random',  # 'positive', 'negative', or 'random'
                },
            )
        super().__init__(config)
    
    def inject(self) -> Dict[str, Any]:
        """Generate extreme funding rate."""
        rate = self.config.parameters.get('funding_rate', 0.005)
        direction = self.config.parameters.get('direction', 'random')
        
        if direction == 'random':
            rate = rate if np.random.random() > 0.5 else -rate
        elif direction == 'negative':
            rate = -abs(rate)
        
        result = {
            'funding_rate': rate,
            'annualized_rate': rate * 3 * 365,  # 8-hour funding
        }
        
        self._log_event("funding_spike_injected", result)
        
        return result


class SentimentPoisonScenario(StressScenario):
    """
    Scenario: Static "bullish" sentiment for 4 hours.
    
    Tests if system overly relies on sentiment signals.
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        if config is None:
            config = ScenarioConfig(
                scenario_type=ScenarioType.SENTIMENT_POISON,
                name="Sentiment Poison",
                description="Static bullish sentiment for 4 hours",
                duration_seconds=14400,  # 4 hours
                parameters={
                    'sentiment_value': 0.8,  # Fixed bullish
                    'sentiment_label': 'bullish',
                },
            )
        super().__init__(config)
    
    def inject(self) -> Dict[str, Any]:
        """Generate poisoned sentiment."""
        result = {
            'sentiment_value': self.config.parameters.get('sentiment_value', 0.8),
            'sentiment_label': self.config.parameters.get('sentiment_label', 'bullish'),
            'is_poisoned': True,
        }
        
        self._log_event("sentiment_poison_injected", result)
        
        return result


# =============================================================================
# SCENARIO FACTORY
# =============================================================================

class ScenarioFactory:
    """Factory for creating stress test scenarios."""
    
    _scenarios = {
        ScenarioType.BOOK_COLLAPSE: BookCollapseScenario,
        ScenarioType.LATENCY_SPIKE: LatencySpikeScenario,
        ScenarioType.API_ERRORS: APIErrorScenario,
        ScenarioType.FUNDING_SPIKE: FundingSpikeScenario,
        ScenarioType.SENTIMENT_POISON: SentimentPoisonScenario,
    }
    
    @classmethod
    def create(
        cls,
        scenario_type: ScenarioType,
        config: Optional[ScenarioConfig] = None,
    ) -> StressScenario:
        """Create a scenario instance."""
        if scenario_type not in cls._scenarios:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        return cls._scenarios[scenario_type](config)
    
    @classmethod
    def get_all_types(cls) -> List[ScenarioType]:
        """Get all available scenario types."""
        return list(cls._scenarios.keys())


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create and run a scenario
    scenario = ScenarioFactory.create(ScenarioType.BOOK_COLLAPSE)
    
    print(f"Scenario: {scenario.config.name}")
    print(f"Description: {scenario.config.description}")
    
    # Start
    scenario.start()
    
    # Inject chaos
    for i in range(5):
        result = scenario.inject()
        print(f"  Injection {i+1}: {result}")
    
    # Stop and get results
    result = scenario.stop()
    
    print(f"\n=== Results ===")
    print(json.dumps(result.to_dict(), indent=2))
