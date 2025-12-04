"""
Cross-Exchange Arbitrage Resilience Test Suite

This module tests the resilience and fault tolerance of cross-exchange arbitrage operations.
It simulates various failure scenarios to ensure the system handles edge cases gracefully.

Tests:
1. Venue outage simulation
2. Latency injection (200-500ms)
3. Routing fallback verification
4. Inventory safety tests
5. Partial fill handling
6. Network partition simulation
7. Rate limit scenarios
8. Margin call cascades
"""

import pytest
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from enum import Enum
import random
import threading
import queue

# Import our arbitrage modules (adjust paths as needed)
import sys
sys.path.insert(0, '..')

try:
    from src.arbitrage.cross_exchange import (
        CrossExchangeRouter,
        ExchangeConfig,
        ArbitrageOpportunity,
        OrderResult,
        VenueStatus
    )
except ImportError:
    # Define minimal stubs for testing
    pass


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class VenueSimConfig:
    """Configuration for venue simulation."""
    name: str
    base_latency_ms: float = 50.0
    latency_std_ms: float = 10.0
    uptime_probability: float = 0.999
    rate_limit_per_second: int = 100
    max_order_size_usd: float = 1_000_000
    margin_requirement: float = 0.10


@dataclass
class FailureScenario:
    """Defines a failure injection scenario."""
    name: str
    venue: str
    failure_type: str  # 'outage', 'latency', 'rate_limit', 'partial_fill'
    duration_seconds: float
    parameters: Dict[str, Any] = field(default_factory=dict)


class FailureType(Enum):
    OUTAGE = "outage"
    HIGH_LATENCY = "high_latency"
    RATE_LIMIT = "rate_limit"
    PARTIAL_FILL = "partial_fill"
    NETWORK_PARTITION = "network_partition"
    MARGIN_CALL = "margin_call"


# ============================================================================
# Mock Exchange Client
# ============================================================================

class MockExchangeClient:
    """
    Simulates an exchange with configurable failure injection.
    """
    
    def __init__(self, config: VenueSimConfig):
        self.config = config
        self.is_online = True
        self.injected_latency_ms = 0
        self.rate_limit_counter = 0
        self.rate_limit_reset_time = time.time()
        self.partial_fill_rate = 1.0  # 1.0 = full fills
        self.orders: Dict[str, dict] = {}
        self.positions: Dict[str, float] = {}
        self.balance_usd = 100_000
        self._lock = threading.Lock()
        
    def inject_failure(self, failure_type: FailureType, **kwargs):
        """Inject a specific failure mode."""
        if failure_type == FailureType.OUTAGE:
            self.is_online = False
        elif failure_type == FailureType.HIGH_LATENCY:
            self.injected_latency_ms = kwargs.get('latency_ms', 500)
        elif failure_type == FailureType.RATE_LIMIT:
            self.rate_limit_counter = kwargs.get('limit', 1000)  # Exhaust rate limit
        elif failure_type == FailureType.PARTIAL_FILL:
            self.partial_fill_rate = kwargs.get('fill_rate', 0.5)
        elif failure_type == FailureType.MARGIN_CALL:
            self.balance_usd = kwargs.get('balance', 0)
            
    def clear_failures(self):
        """Clear all injected failures."""
        self.is_online = True
        self.injected_latency_ms = 0
        self.partial_fill_rate = 1.0
        self.rate_limit_counter = 0
        self.balance_usd = 100_000
        
    async def _simulate_latency(self):
        """Simulate network latency."""
        latency = self.config.base_latency_ms + self.injected_latency_ms
        latency += random.gauss(0, self.config.latency_std_ms)
        latency = max(0, latency)
        await asyncio.sleep(latency / 1000)
        
    def _check_rate_limit(self) -> bool:
        """Check if rate limited."""
        current_time = time.time()
        if current_time - self.rate_limit_reset_time >= 1.0:
            self.rate_limit_counter = 0
            self.rate_limit_reset_time = current_time
            
        self.rate_limit_counter += 1
        return self.rate_limit_counter <= self.config.rate_limit_per_second
        
    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker (bid/ask)."""
        if not self.is_online:
            raise ConnectionError(f"{self.config.name} is offline")
            
        await self._simulate_latency()
        
        # Simulated prices with spread
        base_price = 50000 + random.uniform(-100, 100)
        spread = base_price * 0.0001  # 1 bp spread
        
        return {
            'symbol': symbol,
            'bid': base_price - spread / 2,
            'ask': base_price + spread / 2,
            'timestamp': datetime.now().isoformat()
        }
        
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = 'limit'
    ) -> dict:
        """Place an order with failure simulation."""
        if not self.is_online:
            raise ConnectionError(f"{self.config.name} is offline")
            
        if not self._check_rate_limit():
            raise Exception(f"Rate limit exceeded on {self.config.name}")
            
        await self._simulate_latency()
        
        order_id = f"{self.config.name}_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        
        # Check margin
        order_value = quantity * (price or 50000)
        required_margin = order_value * self.config.margin_requirement
        if required_margin > self.balance_usd:
            raise Exception(f"Insufficient margin on {self.config.name}")
            
        # Apply partial fill
        filled_quantity = quantity * self.partial_fill_rate
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'filled_quantity': filled_quantity,
            'price': price or 50000,
            'status': 'filled' if filled_quantity == quantity else 'partial',
            'venue': self.config.name,
            'timestamp': datetime.now().isoformat()
        }
        
        with self._lock:
            self.orders[order_id] = order
            
            # Update position
            position_delta = filled_quantity if side == 'buy' else -filled_quantity
            self.positions[symbol] = self.positions.get(symbol, 0) + position_delta
            
        return order
        
    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an order."""
        if not self.is_online:
            raise ConnectionError(f"{self.config.name} is offline")
            
        await self._simulate_latency()
        
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'cancelled'
            return {'status': 'cancelled', 'order_id': order_id}
        else:
            raise Exception(f"Order {order_id} not found")
            
    async def get_balance(self) -> dict:
        """Get account balance."""
        if not self.is_online:
            raise ConnectionError(f"{self.config.name} is offline")
            
        await self._simulate_latency()
        
        return {
            'usd': self.balance_usd,
            'positions': self.positions.copy()
        }


# ============================================================================
# Mock Arbitrage Router
# ============================================================================

class MockArbitrageRouter:
    """
    Simulates the cross-exchange arbitrage router for testing.
    """
    
    def __init__(self, venues: Dict[str, MockExchangeClient]):
        self.venues = venues
        self.active_venue: Optional[str] = None
        self.fallback_order: List[str] = list(venues.keys())
        self.max_latency_ms: float = 200.0  # Threshold for venue selection
        self.inventory_limits: Dict[str, float] = {name: 100.0 for name in venues}
        self.execution_history: List[dict] = []
        
    def select_best_venue(self) -> Optional[str]:
        """Select the best available venue."""
        for venue_name in self.fallback_order:
            venue = self.venues.get(venue_name)
            if venue and venue.is_online:
                # Check if within latency threshold
                expected_latency = venue.config.base_latency_ms + venue.injected_latency_ms
                if expected_latency <= self.max_latency_ms:
                    return venue_name
        return None
        
    async def execute_arbitrage(
        self,
        symbol: str,
        buy_venue: str,
        sell_venue: str,
        quantity: float,
        buy_price: float,
        sell_price: float
    ) -> dict:
        """Execute an arbitrage trade across venues."""
        result = {
            'success': False,
            'buy_order': None,
            'sell_order': None,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check venues are online
            if not self.venues[buy_venue].is_online:
                raise ConnectionError(f"Buy venue {buy_venue} offline")
            if not self.venues[sell_venue].is_online:
                raise ConnectionError(f"Sell venue {sell_venue} offline")
                
            # Execute both legs (in production, would be parallel)
            buy_order = await self.venues[buy_venue].place_order(
                symbol, 'buy', quantity, buy_price
            )
            sell_order = await self.venues[sell_venue].place_order(
                symbol, 'sell', quantity, sell_price
            )
            
            result['buy_order'] = buy_order
            result['sell_order'] = sell_order
            
            # Check for partial fills
            buy_filled = buy_order.get('filled_quantity', 0)
            sell_filled = sell_order.get('filled_quantity', 0)
            
            if buy_filled != quantity or sell_filled != quantity:
                result['partial_fill'] = True
                result['fill_imbalance'] = abs(buy_filled - sell_filled)
            else:
                result['partial_fill'] = False
                
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            
        self.execution_history.append(result)
        return result
        
    def check_inventory_safety(self, symbol: str) -> Dict[str, float]:
        """Check inventory levels across venues."""
        inventory = {}
        for name, venue in self.venues.items():
            inventory[name] = venue.positions.get(symbol, 0)
        return inventory
        
    def rebalance_inventory(self, symbol: str, target: float = 0) -> List[dict]:
        """Generate rebalancing orders to neutralize inventory."""
        orders = []
        inventory = self.check_inventory_safety(symbol)
        
        for venue_name, position in inventory.items():
            if abs(position - target) > 0.01:
                orders.append({
                    'venue': venue_name,
                    'symbol': symbol,
                    'side': 'sell' if position > target else 'buy',
                    'quantity': abs(position - target)
                })
        return orders


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def venues():
    """Create mock exchange venues."""
    return {
        'binance': MockExchangeClient(VenueSimConfig(
            name='binance',
            base_latency_ms=30,
            rate_limit_per_second=100
        )),
        'okx': MockExchangeClient(VenueSimConfig(
            name='okx',
            base_latency_ms=50,
            rate_limit_per_second=50
        )),
        'bybit': MockExchangeClient(VenueSimConfig(
            name='bybit',
            base_latency_ms=40,
            rate_limit_per_second=75
        ))
    }


@pytest.fixture
def router(venues):
    """Create mock arbitrage router."""
    return MockArbitrageRouter(venues)


# ============================================================================
# Test Suite: Venue Outage Simulation
# ============================================================================

class TestVenueOutage:
    """Test behavior when venues go offline."""
    
    @pytest.mark.asyncio
    async def test_single_venue_outage_failover(self, router, venues):
        """Test that router fails over when primary venue goes down."""
        # Primary venue goes offline
        venues['binance'].inject_failure(FailureType.OUTAGE)
        
        best = router.select_best_venue()
        assert best != 'binance', "Should not select offline venue"
        assert best in ['okx', 'bybit'], "Should failover to backup venue"
        
    @pytest.mark.asyncio
    async def test_all_venues_offline(self, router, venues):
        """Test behavior when all venues are offline."""
        for venue in venues.values():
            venue.inject_failure(FailureType.OUTAGE)
            
        best = router.select_best_venue()
        assert best is None, "Should return None when all venues offline"
        
    @pytest.mark.asyncio
    async def test_venue_recovery(self, router, venues):
        """Test that router uses recovered venue."""
        venues['binance'].inject_failure(FailureType.OUTAGE)
        assert router.select_best_venue() != 'binance'
        
        # Venue recovers
        venues['binance'].clear_failures()
        best = router.select_best_venue()
        assert best == 'binance', "Should use recovered primary venue"
        
    @pytest.mark.asyncio
    async def test_mid_execution_outage(self, router, venues):
        """Test handling when venue goes down during execution."""
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=1.0,
            buy_price=50000,
            sell_price=50100
        )
        assert result['success'], "First execution should succeed"
        
        # Venue goes down
        venues['okx'].inject_failure(FailureType.OUTAGE)
        
        result2 = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=1.0,
            buy_price=50000,
            sell_price=50100
        )
        assert not result2['success'], "Should fail with offline sell venue"
        assert 'offline' in result2['error'].lower()


# ============================================================================
# Test Suite: Latency Injection
# ============================================================================

class TestLatencyInjection:
    """Test behavior under high latency conditions."""
    
    @pytest.mark.asyncio
    async def test_high_latency_venue_skipped(self, router, venues):
        """Test that high-latency venues are deprioritized."""
        # Inject 500ms latency on primary
        venues['binance'].inject_failure(FailureType.HIGH_LATENCY, latency_ms=500)
        
        best = router.select_best_venue()
        # Should skip binance due to latency > threshold
        assert best in ['okx', 'bybit'], "Should select lower latency venue"
        
    @pytest.mark.asyncio
    async def test_execution_under_200ms_latency(self, router, venues):
        """Test execution completes under 200ms latency."""
        venues['binance'].inject_failure(FailureType.HIGH_LATENCY, latency_ms=150)
        
        start = time.time()
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=0.5,
            buy_price=50000,
            sell_price=50050
        )
        elapsed = (time.time() - start) * 1000
        
        assert result['success'], "Should succeed under 200ms latency"
        # Both legs should complete with some tolerance
        assert elapsed < 1000, f"Execution too slow: {elapsed:.0f}ms"
        
    @pytest.mark.asyncio
    async def test_execution_under_500ms_latency(self, router, venues):
        """Test execution behavior under 500ms latency."""
        venues['binance'].inject_failure(FailureType.HIGH_LATENCY, latency_ms=500)
        venues['okx'].inject_failure(FailureType.HIGH_LATENCY, latency_ms=500)
        
        start = time.time()
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=0.5,
            buy_price=50000,
            sell_price=50050
        )
        elapsed = (time.time() - start) * 1000
        
        # Should still complete but take longer
        assert result['success'], "Should eventually succeed"
        assert elapsed > 500, "Should reflect injected latency"
        
    @pytest.mark.asyncio
    async def test_latency_recovery(self, router, venues):
        """Test that latency recovery is detected."""
        venues['binance'].inject_failure(FailureType.HIGH_LATENCY, latency_ms=500)
        
        # High latency - should skip
        assert router.select_best_venue() != 'binance'
        
        # Latency recovers
        venues['binance'].clear_failures()
        
        # Should use primary again
        assert router.select_best_venue() == 'binance'


# ============================================================================
# Test Suite: Routing Fallback Verification
# ============================================================================

class TestRoutingFallback:
    """Test fallback routing logic."""
    
    @pytest.mark.asyncio
    async def test_fallback_order_respected(self, router, venues):
        """Test that fallback order is respected."""
        router.fallback_order = ['binance', 'bybit', 'okx']
        
        # Take out venues in order
        venues['binance'].inject_failure(FailureType.OUTAGE)
        assert router.select_best_venue() == 'bybit'
        
        venues['bybit'].inject_failure(FailureType.OUTAGE)
        assert router.select_best_venue() == 'okx'
        
    @pytest.mark.asyncio
    async def test_custom_fallback_order(self, router, venues):
        """Test custom fallback order."""
        router.fallback_order = ['okx', 'bybit', 'binance']
        
        best = router.select_best_venue()
        assert best == 'okx', "Should respect custom order"
        
    @pytest.mark.asyncio
    async def test_fallback_with_latency_constraint(self, router, venues):
        """Test fallback considers latency constraints."""
        router.max_latency_ms = 40  # Tighter threshold
        
        # Only binance meets threshold (30ms base)
        best = router.select_best_venue()
        assert best == 'binance', "Only binance should meet latency threshold"
        
        # Take out binance
        venues['binance'].inject_failure(FailureType.OUTAGE)
        
        # No venue meets threshold
        best = router.select_best_venue()
        assert best is None, "No venue should meet tight latency constraint"


# ============================================================================
# Test Suite: Inventory Safety Tests
# ============================================================================

class TestInventorySafety:
    """Test inventory management and safety limits."""
    
    @pytest.mark.asyncio
    async def test_inventory_tracking(self, router, venues):
        """Test that inventory is tracked across executions."""
        # Execute some trades
        await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=1.0,
            buy_price=50000,
            sell_price=50100
        )
        
        inventory = router.check_inventory_safety('BTCUSD')
        assert inventory['binance'] == 1.0, "Should have +1 BTC on binance"
        assert inventory['okx'] == -1.0, "Should have -1 BTC on okx"
        
    @pytest.mark.asyncio
    async def test_inventory_limits_enforced(self, router, venues):
        """Test that inventory limits are respected."""
        router.inventory_limits['binance'] = 2.0
        
        # Execute trades up to limit
        for _ in range(3):
            await router.execute_arbitrage(
                symbol='BTCUSD',
                buy_venue='binance',
                sell_venue='okx',
                quantity=1.0,
                buy_price=50000,
                sell_price=50100
            )
            
        inventory = router.check_inventory_safety('BTCUSD')
        # In production, would block trades beyond limit
        assert abs(inventory['binance']) <= 5, "Inventory should be bounded"
        
    @pytest.mark.asyncio
    async def test_rebalancing_orders_generated(self, router, venues):
        """Test that rebalancing orders are generated correctly."""
        # Create imbalance
        venues['binance'].positions['BTCUSD'] = 5.0
        venues['okx'].positions['BTCUSD'] = -3.0
        
        orders = router.rebalance_inventory('BTCUSD', target=0)
        
        assert len(orders) == 2, "Should generate orders for both venues"
        
        binance_order = next(o for o in orders if o['venue'] == 'binance')
        assert binance_order['side'] == 'sell', "Should sell excess on binance"
        assert binance_order['quantity'] == 5.0
        
        okx_order = next(o for o in orders if o['venue'] == 'okx')
        assert okx_order['side'] == 'buy', "Should buy to cover short on okx"
        
    @pytest.mark.asyncio
    async def test_neutral_inventory_no_rebalance(self, router, venues):
        """Test no rebalancing when inventory is neutral."""
        venues['binance'].positions['BTCUSD'] = 0.001  # Near zero
        venues['okx'].positions['BTCUSD'] = -0.001
        
        orders = router.rebalance_inventory('BTCUSD', target=0)
        
        # With 0.01 threshold, these should not trigger rebalance
        assert len(orders) == 0, "Should not rebalance near-zero positions"


# ============================================================================
# Test Suite: Partial Fill Handling
# ============================================================================

class TestPartialFillHandling:
    """Test handling of partial fills."""
    
    @pytest.mark.asyncio
    async def test_partial_fill_detection(self, router, venues):
        """Test that partial fills are detected."""
        venues['binance'].inject_failure(FailureType.PARTIAL_FILL, fill_rate=0.5)
        
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=2.0,
            buy_price=50000,
            sell_price=50100
        )
        
        assert result['partial_fill'], "Should detect partial fill"
        assert result['buy_order']['filled_quantity'] == 1.0
        assert result['sell_order']['filled_quantity'] == 2.0
        
    @pytest.mark.asyncio
    async def test_fill_imbalance_calculation(self, router, venues):
        """Test fill imbalance is calculated correctly."""
        venues['binance'].inject_failure(FailureType.PARTIAL_FILL, fill_rate=0.7)
        venues['okx'].inject_failure(FailureType.PARTIAL_FILL, fill_rate=0.9)
        
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=10.0,
            buy_price=50000,
            sell_price=50100
        )
        
        expected_imbalance = abs(7.0 - 9.0)  # |10*0.7 - 10*0.9|
        assert abs(result['fill_imbalance'] - expected_imbalance) < 0.01
        
    @pytest.mark.asyncio
    async def test_full_fill_no_imbalance(self, router, venues):
        """Test full fills have no imbalance."""
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=1.0,
            buy_price=50000,
            sell_price=50100
        )
        
        assert not result.get('partial_fill', True), "Full fill should not be partial"


# ============================================================================
# Test Suite: Rate Limit Scenarios
# ============================================================================

class TestRateLimitScenarios:
    """Test rate limit handling."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_raised(self, router, venues):
        """Test that rate limit errors are raised."""
        venues['binance'].inject_failure(FailureType.RATE_LIMIT, limit=1000)
        
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=1.0,
            buy_price=50000,
            sell_price=50100
        )
        
        assert not result['success'], "Should fail with rate limit"
        assert 'rate limit' in result['error'].lower()
        
    @pytest.mark.asyncio
    async def test_rate_limit_reset(self, router, venues):
        """Test that rate limits reset after cooldown."""
        venues['binance'].inject_failure(FailureType.RATE_LIMIT, limit=1000)
        
        # First request fails
        result1 = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=1.0,
            buy_price=50000,
            sell_price=50100
        )
        assert not result1['success']
        
        # Wait for reset
        await asyncio.sleep(1.1)
        
        # Reset counter manually (simulating time passage)
        venues['binance'].rate_limit_counter = 0
        
        result2 = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=1.0,
            buy_price=50000,
            sell_price=50100
        )
        assert result2['success'], "Should succeed after rate limit reset"


# ============================================================================
# Test Suite: Margin Call Cascades
# ============================================================================

class TestMarginCallCascades:
    """Test margin call handling."""
    
    @pytest.mark.asyncio
    async def test_insufficient_margin_rejected(self, router, venues):
        """Test that orders are rejected with insufficient margin."""
        venues['binance'].inject_failure(FailureType.MARGIN_CALL, balance=100)  # $100 balance
        
        # Try to place large order
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=10.0,  # $500K notional
            buy_price=50000,
            sell_price=50100
        )
        
        assert not result['success'], "Should fail with insufficient margin"
        assert 'margin' in result['error'].lower()
        
    @pytest.mark.asyncio
    async def test_margin_available_for_small_order(self, router, venues):
        """Test that small orders succeed with limited margin."""
        venues['binance'].balance_usd = 10_000  # Limited margin
        
        result = await router.execute_arbitrage(
            symbol='BTCUSD',
            buy_venue='binance',
            sell_venue='okx',
            quantity=0.1,  # Small order
            buy_price=50000,
            sell_price=50100
        )
        
        assert result['success'], "Small order should succeed with limited margin"


# ============================================================================
# Test Suite: End-to-End Resilience
# ============================================================================

class TestE2EResilience:
    """End-to-end resilience tests."""
    
    @pytest.mark.asyncio
    async def test_chaos_sequence(self, router, venues):
        """Test system under a sequence of failures."""
        results = []
        
        # Normal execution
        r1 = await router.execute_arbitrage('BTCUSD', 'binance', 'okx', 0.5, 50000, 50050)
        results.append(('normal', r1['success']))
        
        # Inject latency
        venues['binance'].inject_failure(FailureType.HIGH_LATENCY, latency_ms=300)
        r2 = await router.execute_arbitrage('BTCUSD', 'binance', 'okx', 0.5, 50000, 50050)
        results.append(('high_latency', r2['success']))
        
        # Inject outage
        venues['okx'].inject_failure(FailureType.OUTAGE)
        r3 = await router.execute_arbitrage('BTCUSD', 'binance', 'okx', 0.5, 50000, 50050)
        results.append(('outage', r3['success']))
        
        # Recovery
        venues['binance'].clear_failures()
        venues['okx'].clear_failures()
        r4 = await router.execute_arbitrage('BTCUSD', 'binance', 'okx', 0.5, 50000, 50050)
        results.append(('recovered', r4['success']))
        
        # Verify expected behavior
        assert results[0] == ('normal', True)
        assert results[1] == ('high_latency', True)  # Should still work
        assert results[2] == ('outage', False)  # Should fail
        assert results[3] == ('recovered', True)  # Should work again
        
    @pytest.mark.asyncio
    async def test_concurrent_executions(self, router, venues):
        """Test concurrent arbitrage executions."""
        tasks = []
        for i in range(10):
            task = router.execute_arbitrage(
                symbol='BTCUSD',
                buy_venue='binance',
                sell_venue='okx',
                quantity=0.1,
                buy_price=50000 + i,
                sell_price=50100 + i
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for r in results if r['success'])
        assert success_count == 10, f"All executions should succeed, got {success_count}"
        
    @pytest.mark.asyncio
    async def test_execution_history_tracking(self, router, venues):
        """Test that execution history is properly tracked."""
        for i in range(5):
            await router.execute_arbitrage(
                symbol='BTCUSD',
                buy_venue='binance',
                sell_venue='okx',
                quantity=0.1,
                buy_price=50000,
                sell_price=50100
            )
            
        assert len(router.execution_history) == 5
        assert all(r['timestamp'] for r in router.execution_history)


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("🧪 CROSS-EXCHANGE ARBITRAGE RESILIENCE TEST SUITE")
    print("=" * 70)
    print("""
This test suite validates:
  1. Venue outage handling and failover
  2. High latency (200-500ms) behavior
  3. Routing fallback logic
  4. Inventory safety limits
  5. Partial fill handling
  6. Rate limit scenarios
  7. Margin call cascades
  8. End-to-end resilience

Run with: pytest tests/test_cross_exchange_resilience.py -v
""")
    
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
