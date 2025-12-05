import asyncio
import random
import logging
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass

class FaultType(Enum):
    LATENCY = "latency"
    NETWORK_ERROR = "network_error"
    API_TIMEOUT = "api_timeout"
    DATA_CORRUPTION = "data_corruption"
    SERVICE_UNAVAILABLE = "service_unavailable"

@dataclass
class ChaosConfig:
    enabled: bool = False
    fault_probability: float = 0.1
    max_latency_ms: int = 2000
    allowed_faults: list[FaultType] = None

    def __post_init__(self):
        if self.allowed_faults is None:
            self.allowed_faults = list(FaultType)

class ChaosMonkey:
    """
    Injects faults into the system to test resilience.
    """
    def __init__(self, config: ChaosConfig):
        self.config = config
        self.logger = logging.getLogger("ChaosMonkey")
        
    def should_inject(self) -> bool:
        """Check if we should inject a fault."""
        return self.config.enabled and random.random() < self.config.fault_probability
        
    async def inject_async(self, context: str = ""):
        """
        Inject fault into async operation.
        Usage: await chaos.inject_async("fetch_data")
        """
        if not self.should_inject():
            return
            
        fault = random.choice(self.config.allowed_faults)
        self.logger.warning(f"🐒 CHAOS MONKEY: Injecting {fault.value} into {context}")
        
        if fault == FaultType.LATENCY:
            delay = random.randint(100, self.config.max_latency_ms) / 1000.0
            await asyncio.sleep(delay)
            
        elif fault == FaultType.NETWORK_ERROR:
            raise ConnectionError(f"Simulated network error in {context}")
            
        elif fault == FaultType.API_TIMEOUT:
            raise asyncio.TimeoutError(f"Simulated API timeout in {context}")
            
        elif fault == FaultType.SERVICE_UNAVAILABLE:
            raise RuntimeError(f"Simulated 503 Service Unavailable in {context}")
            
    def corrupt_data(self, data: Any) -> Any:
        """
        Corrupt data payload.
        """
        if not self.should_inject() or FaultType.DATA_CORRUPTION not in self.config.allowed_faults:
            return data
            
        self.logger.warning("🐒 CHAOS MONKEY: Corrupting data")
        
        # Simple corruption logic
        if isinstance(data, dict):
            # Remove a random key
            if data:
                key = random.choice(list(data.keys()))
                del data[key]
        elif isinstance(data, list):
            # Remove a random element
            if data:
                data.pop(random.randint(0, len(data)-1))
        elif isinstance(data, (int, float)):
            # Flip sign or zero out
            return data * -1 if random.random() > 0.5 else 0
            
        return data
