import asyncio
import logging
import sys
import os
import unittest
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.resilience.chaos_monkey import ChaosMonkey, ChaosConfig, FaultType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

class MockService:
    def __init__(self, chaos: ChaosMonkey):
        self.chaos = chaos
        self.logger = logging.getLogger("MockService")
        
    async def fetch_data(self):
        """Simulate data fetch with potential chaos."""
        try:
            await self.chaos.inject_async("fetch_data")
            self.logger.info("Data fetched successfully")
            return {"price": 100, "volume": 500}
        except Exception as e:
            self.logger.error(f"Fetch failed: {e}")
            raise e

class TestResilience(unittest.IsolatedAsyncioTestCase):
    
    async def test_chaos_injection(self):
        print("\nTesting Chaos Monkey...")
        
        # High probability of fault
        config = ChaosConfig(
            enabled=True,
            fault_probability=0.8,
            allowed_faults=[FaultType.NETWORK_ERROR, FaultType.LATENCY]
        )
        chaos = ChaosMonkey(config)
        service = MockService(chaos)
        
        success_count = 0
        fail_count = 0
        
        for i in range(10):
            try:
                await service.fetch_data()
                success_count += 1
            except ConnectionError:
                fail_count += 1
                print(f"Caught expected network error ({i+1}/10)")
            except Exception as e:
                print(f"Caught other error: {e}")
                
        print(f"Success: {success_count}, Failures: {fail_count}")
        
        # We expect some failures
        self.assertGreater(fail_count, 0, "Chaos Monkey failed to inject faults")
        self.assertGreater(success_count, 0, "Chaos Monkey killed everything (unlikely but possible)")

if __name__ == '__main__':
    unittest.main()
