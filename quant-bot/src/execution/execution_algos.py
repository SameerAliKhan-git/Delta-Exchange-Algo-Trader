"""
Execution Algorithms
====================
Smart order execution for large trades.

This module provides:
- TWAP (Time Weighted Average Price)
- VWAP (Volume Weighted Average Price)
- Implementation Shortfall minimization
- Iceberg orders
- Adaptive execution

Based on:
- Market microstructure theory
- Optimal execution literature (Almgren-Chriss)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# EXECUTION DATA CLASSES
# =============================================================================

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class ChildOrder:
    """Individual slice of a parent order."""
    order_id: str
    parent_id: str
    side: OrderSide
    symbol: str
    quantity: float
    limit_price: Optional[float]
    scheduled_time: datetime
    filled_quantity: float = 0
    fill_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity


@dataclass
class ExecutionReport:
    """Summary of execution performance."""
    parent_order_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    filled_quantity: float
    avg_fill_price: float
    arrival_price: float  # Price when algo started
    vwap_benchmark: float  # Market VWAP during execution
    slippage_bps: float  # Slippage in basis points
    implementation_shortfall: float
    n_child_orders: int
    n_fills: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    fills: List[Dict] = field(default_factory=list)


# =============================================================================
# BASE EXECUTION ALGORITHM
# =============================================================================

class BaseExecutionAlgo(ABC):
    """
    Base class for execution algorithms.
    """
    
    def __init__(self, symbol: str, side: OrderSide,
                 total_quantity: float, duration_minutes: int,
                 order_placer: Optional[Callable] = None):
        """
        Initialize execution algorithm.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            total_quantity: Total quantity to execute
            duration_minutes: Time window for execution
            order_placer: Callback to place orders
        """
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.duration_minutes = duration_minutes
        self.order_placer = order_placer
        
        self.parent_id = f"EXEC_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.child_orders: List[ChildOrder] = []
        self.fills: List[Dict] = []
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.arrival_price: Optional[float] = None
        
        self._order_counter = 0
        self._running = False
    
    def _generate_order_id(self) -> str:
        """Generate unique child order ID."""
        self._order_counter += 1
        return f"{self.parent_id}_{self._order_counter:04d}"
    
    @abstractmethod
    def generate_schedule(self, market_data: Dict) -> List[ChildOrder]:
        """
        Generate execution schedule.
        
        Args:
            market_data: Current market data (price, volume, etc.)
        
        Returns:
            List of scheduled child orders
        """
        pass
    
    def start(self, arrival_price: float, market_data: Optional[Dict] = None):
        """Start execution."""
        self.start_time = datetime.now()
        self.arrival_price = arrival_price
        self._running = True
        
        self.child_orders = self.generate_schedule(market_data or {})
    
    def stop(self):
        """Stop execution."""
        self._running = False
        self.end_time = datetime.now()
    
    def record_fill(self, order_id: str, filled_qty: float, fill_price: float):
        """Record a fill for a child order."""
        for order in self.child_orders:
            if order.order_id == order_id:
                order.filled_quantity += filled_qty
                order.fill_price = fill_price
                
                if order.filled_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                else:
                    order.status = OrderStatus.PARTIAL
                
                self.fills.append({
                    'order_id': order_id,
                    'quantity': filled_qty,
                    'price': fill_price,
                    'timestamp': datetime.now()
                })
                break
    
    @property
    def filled_quantity(self) -> float:
        """Total filled quantity."""
        return sum(f['quantity'] for f in self.fills)
    
    @property
    def avg_fill_price(self) -> float:
        """Average fill price."""
        if not self.fills:
            return 0
        
        total_value = sum(f['quantity'] * f['price'] for f in self.fills)
        total_qty = sum(f['quantity'] for f in self.fills)
        
        return total_value / total_qty if total_qty > 0 else 0
    
    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to execute."""
        return self.total_quantity - self.filled_quantity
    
    def get_report(self, vwap_benchmark: float) -> ExecutionReport:
        """Generate execution report."""
        self.end_time = self.end_time or datetime.now()
        
        avg_price = self.avg_fill_price
        
        # Calculate slippage
        if self.side == OrderSide.BUY:
            slippage = (avg_price - self.arrival_price) / self.arrival_price
        else:
            slippage = (self.arrival_price - avg_price) / self.arrival_price
        
        slippage_bps = slippage * 10000
        
        # Implementation shortfall
        impl_shortfall = abs(avg_price - self.arrival_price) * self.filled_quantity
        
        return ExecutionReport(
            parent_order_id=self.parent_id,
            symbol=self.symbol,
            side=self.side,
            total_quantity=self.total_quantity,
            filled_quantity=self.filled_quantity,
            avg_fill_price=avg_price,
            arrival_price=self.arrival_price,
            vwap_benchmark=vwap_benchmark,
            slippage_bps=slippage_bps,
            implementation_shortfall=impl_shortfall,
            n_child_orders=len(self.child_orders),
            n_fills=len(self.fills),
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=(self.end_time - self.start_time).total_seconds(),
            fills=self.fills
        )


# =============================================================================
# TWAP ALGORITHM
# =============================================================================

class TWAPAlgo(BaseExecutionAlgo):
    """
    Time Weighted Average Price algorithm.
    
    Splits order evenly across time intervals.
    Best for: Low-urgency orders, predictable impact.
    """
    
    def __init__(self, symbol: str, side: OrderSide,
                 total_quantity: float, duration_minutes: int,
                 n_slices: int = 10,
                 randomize: bool = True,
                 order_placer: Optional[Callable] = None):
        """
        Initialize TWAP.
        
        Args:
            n_slices: Number of child orders
            randomize: Add random jitter to timing
        """
        super().__init__(symbol, side, total_quantity, duration_minutes, order_placer)
        self.n_slices = n_slices
        self.randomize = randomize
    
    def generate_schedule(self, market_data: Dict) -> List[ChildOrder]:
        """Generate TWAP schedule."""
        orders = []
        
        # Calculate slice size and interval
        slice_qty = self.total_quantity / self.n_slices
        interval_seconds = (self.duration_minutes * 60) / self.n_slices
        
        for i in range(self.n_slices):
            # Base scheduled time
            base_offset = i * interval_seconds
            
            # Add randomization
            if self.randomize and i < self.n_slices - 1:
                jitter = np.random.uniform(-0.2, 0.2) * interval_seconds
                offset = base_offset + jitter
            else:
                offset = base_offset
            
            scheduled_time = self.start_time + timedelta(seconds=offset)
            
            order = ChildOrder(
                order_id=self._generate_order_id(),
                parent_id=self.parent_id,
                side=self.side,
                symbol=self.symbol,
                quantity=slice_qty,
                limit_price=None,  # Market orders
                scheduled_time=scheduled_time
            )
            orders.append(order)
        
        return orders


# =============================================================================
# VWAP ALGORITHM
# =============================================================================

class VWAPAlgo(BaseExecutionAlgo):
    """
    Volume Weighted Average Price algorithm.
    
    Distributes orders based on historical volume profile.
    Best for: Following market rhythm, minimizing impact.
    """
    
    def __init__(self, symbol: str, side: OrderSide,
                 total_quantity: float, duration_minutes: int,
                 volume_profile: Optional[np.ndarray] = None,
                 n_slices: int = 10,
                 participation_rate: float = 0.10,
                 order_placer: Optional[Callable] = None):
        """
        Initialize VWAP.
        
        Args:
            volume_profile: Historical volume distribution (normalized)
            n_slices: Number of child orders
            participation_rate: Max % of interval volume
        """
        super().__init__(symbol, side, total_quantity, duration_minutes, order_placer)
        self.volume_profile = volume_profile
        self.n_slices = n_slices
        self.participation_rate = participation_rate
    
    def generate_schedule(self, market_data: Dict) -> List[ChildOrder]:
        """Generate VWAP schedule based on volume profile."""
        orders = []
        
        # Use provided volume profile or generate default
        if self.volume_profile is None:
            # U-shaped volume profile (typical intraday)
            x = np.linspace(0, 1, self.n_slices)
            profile = 1 + 0.5 * (np.abs(x - 0.5) ** 2)  # Higher at open/close
            profile = profile / profile.sum()
        else:
            profile = self.volume_profile / self.volume_profile.sum()
            # Resample to n_slices if needed
            if len(profile) != self.n_slices:
                profile = np.interp(
                    np.linspace(0, 1, self.n_slices),
                    np.linspace(0, 1, len(profile)),
                    profile
                )
                profile = profile / profile.sum()
        
        interval_seconds = (self.duration_minutes * 60) / self.n_slices
        
        for i in range(self.n_slices):
            # Quantity proportional to volume
            slice_qty = self.total_quantity * profile[i]
            
            scheduled_time = self.start_time + timedelta(seconds=i * interval_seconds)
            
            order = ChildOrder(
                order_id=self._generate_order_id(),
                parent_id=self.parent_id,
                side=self.side,
                symbol=self.symbol,
                quantity=slice_qty,
                limit_price=None,
                scheduled_time=scheduled_time
            )
            orders.append(order)
        
        return orders


# =============================================================================
# IMPLEMENTATION SHORTFALL ALGORITHM
# =============================================================================

class ImplementationShortfallAlgo(BaseExecutionAlgo):
    """
    Implementation Shortfall (IS) algorithm.
    
    Minimizes expected cost (impact + timing risk).
    Based on Almgren-Chriss framework.
    
    Best for: Balancing urgency vs. market impact.
    """
    
    def __init__(self, symbol: str, side: OrderSide,
                 total_quantity: float, duration_minutes: int,
                 volatility: float = 0.02,
                 risk_aversion: float = 0.01,
                 impact_coeff: float = 0.01,
                 n_slices: int = 10,
                 order_placer: Optional[Callable] = None):
        """
        Initialize IS algo.
        
        Args:
            volatility: Expected price volatility (per minute)
            risk_aversion: Lambda - risk aversion parameter
            impact_coeff: Market impact coefficient
            n_slices: Number of child orders
        """
        super().__init__(symbol, side, total_quantity, duration_minutes, order_placer)
        self.volatility = volatility
        self.risk_aversion = risk_aversion
        self.impact_coeff = impact_coeff
        self.n_slices = n_slices
    
    def _optimal_trajectory(self) -> np.ndarray:
        """
        Calculate optimal execution trajectory using Almgren-Chriss.
        
        Returns remaining inventory at each time step.
        """
        T = self.duration_minutes
        n = self.n_slices
        sigma = self.volatility
        lam = self.risk_aversion
        eta = self.impact_coeff
        X = self.total_quantity
        
        # Time step
        tau = T / n
        
        # Kappa (urgency parameter)
        kappa = np.sqrt(lam * sigma ** 2 / eta)
        
        # Calculate trajectory
        t = np.linspace(0, T, n + 1)
        
        if kappa * T < 1e-6:
            # Linear trajectory for small kappa
            trajectory = X * (1 - t / T)
        else:
            trajectory = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        
        return trajectory
    
    def generate_schedule(self, market_data: Dict) -> List[ChildOrder]:
        """Generate IS schedule."""
        orders = []
        
        trajectory = self._optimal_trajectory()
        
        # Trade sizes are changes in trajectory
        trade_sizes = -np.diff(trajectory)
        
        interval_seconds = (self.duration_minutes * 60) / self.n_slices
        
        for i in range(self.n_slices):
            scheduled_time = self.start_time + timedelta(seconds=i * interval_seconds)
            
            order = ChildOrder(
                order_id=self._generate_order_id(),
                parent_id=self.parent_id,
                side=self.side,
                symbol=self.symbol,
                quantity=trade_sizes[i],
                limit_price=None,
                scheduled_time=scheduled_time
            )
            orders.append(order)
        
        return orders


# =============================================================================
# ICEBERG ALGORITHM
# =============================================================================

class IcebergAlgo(BaseExecutionAlgo):
    """
    Iceberg order algorithm.
    
    Shows only a small portion of total order to hide size.
    Best for: Large orders, avoiding front-running.
    """
    
    def __init__(self, symbol: str, side: OrderSide,
                 total_quantity: float, duration_minutes: int,
                 display_qty: float,
                 price_tolerance: float = 0.001,
                 order_placer: Optional[Callable] = None):
        """
        Initialize Iceberg.
        
        Args:
            display_qty: Visible quantity per slice
            price_tolerance: Price tolerance for limit orders
        """
        super().__init__(symbol, side, total_quantity, duration_minutes, order_placer)
        self.display_qty = display_qty
        self.price_tolerance = price_tolerance
    
    def generate_schedule(self, market_data: Dict) -> List[ChildOrder]:
        """Generate iceberg schedule."""
        orders = []
        
        n_slices = int(np.ceil(self.total_quantity / self.display_qty))
        interval_seconds = (self.duration_minutes * 60) / n_slices
        
        remaining = self.total_quantity
        
        for i in range(n_slices):
            slice_qty = min(self.display_qty, remaining)
            remaining -= slice_qty
            
            scheduled_time = self.start_time + timedelta(seconds=i * interval_seconds)
            
            order = ChildOrder(
                order_id=self._generate_order_id(),
                parent_id=self.parent_id,
                side=self.side,
                symbol=self.symbol,
                quantity=slice_qty,
                limit_price=None,  # Will be set based on market
                scheduled_time=scheduled_time
            )
            orders.append(order)
        
        return orders


# =============================================================================
# ADAPTIVE EXECUTION ALGORITHM
# =============================================================================

class AdaptiveAlgo(BaseExecutionAlgo):
    """
    Adaptive execution algorithm.
    
    Adjusts execution based on real-time market conditions.
    Combines elements of TWAP, VWAP, and IS.
    """
    
    def __init__(self, symbol: str, side: OrderSide,
                 total_quantity: float, duration_minutes: int,
                 aggression: float = 0.5,
                 participation_limit: float = 0.15,
                 n_slices: int = 20,
                 order_placer: Optional[Callable] = None):
        """
        Initialize Adaptive algo.
        
        Args:
            aggression: 0-1, higher = faster execution
            participation_limit: Max % of volume
            n_slices: Number of evaluation points
        """
        super().__init__(symbol, side, total_quantity, duration_minutes, order_placer)
        self.aggression = aggression
        self.participation_limit = participation_limit
        self.n_slices = n_slices
    
    def generate_schedule(self, market_data: Dict) -> List[ChildOrder]:
        """Generate initial schedule (will be adapted during execution)."""
        orders = []
        
        # Base quantity per slice
        base_qty = self.total_quantity / self.n_slices
        
        interval_seconds = (self.duration_minutes * 60) / self.n_slices
        
        for i in range(self.n_slices):
            # Adjust quantity based on aggression
            # Higher aggression = more front-loaded
            time_weight = 1 - (i / self.n_slices)
            aggression_factor = 1 + self.aggression * (time_weight - 0.5)
            slice_qty = base_qty * aggression_factor
            
            scheduled_time = self.start_time + timedelta(seconds=i * interval_seconds)
            
            order = ChildOrder(
                order_id=self._generate_order_id(),
                parent_id=self.parent_id,
                side=self.side,
                symbol=self.symbol,
                quantity=slice_qty,
                limit_price=None,
                scheduled_time=scheduled_time
            )
            orders.append(order)
        
        # Normalize to total quantity
        total_scheduled = sum(o.quantity for o in orders)
        for order in orders:
            order.quantity *= self.total_quantity / total_scheduled
        
        return orders
    
    def adapt_to_market(self, current_price: float, volume: float,
                        spread: float, urgency_increase: float = 0):
        """
        Adapt remaining schedule based on market conditions.
        
        Args:
            current_price: Current market price
            volume: Recent volume
            spread: Current bid-ask spread
            urgency_increase: Additional urgency factor
        """
        pending_orders = [o for o in self.child_orders if not o.is_complete]
        
        if not pending_orders:
            return
        
        remaining_qty = sum(o.quantity for o in pending_orders)
        
        # Adjust based on conditions
        if spread > 0.002:  # Wide spread - slow down
            adjustment = 0.8
        elif spread < 0.0005:  # Tight spread - speed up
            adjustment = 1.2
        else:
            adjustment = 1.0
        
        # Add urgency
        adjustment += urgency_increase
        
        # Redistribute remaining quantity
        n_remaining = len(pending_orders)
        new_base_qty = remaining_qty / n_remaining * adjustment
        
        # Ensure we don't exceed original remaining
        total_new = 0
        for i, order in enumerate(pending_orders[:-1]):
            order.quantity = min(new_base_qty, remaining_qty - total_new)
            total_new += order.quantity
        
        # Last order gets remainder
        if pending_orders:
            pending_orders[-1].quantity = remaining_qty - total_new


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Manages execution algorithms and order flow.
    """
    
    def __init__(self, order_placer: Optional[Callable] = None):
        """
        Initialize execution engine.
        
        Args:
            order_placer: Async function to place orders
        """
        self.order_placer = order_placer
        self.active_algos: Dict[str, BaseExecutionAlgo] = {}
        self.completed_algos: List[BaseExecutionAlgo] = []
    
    def create_algo(self, algo_type: str, symbol: str, side: str,
                   quantity: float, duration_minutes: int,
                   **kwargs) -> BaseExecutionAlgo:
        """
        Create an execution algorithm.
        
        Args:
            algo_type: 'twap', 'vwap', 'is', 'iceberg', 'adaptive'
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Total quantity
            duration_minutes: Execution window
            **kwargs: Algorithm-specific parameters
        
        Returns:
            Execution algorithm instance
        """
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        
        algo_classes = {
            'twap': TWAPAlgo,
            'vwap': VWAPAlgo,
            'is': ImplementationShortfallAlgo,
            'iceberg': IcebergAlgo,
            'adaptive': AdaptiveAlgo
        }
        
        algo_class = algo_classes.get(algo_type.lower())
        if algo_class is None:
            raise ValueError(f"Unknown algo type: {algo_type}")
        
        algo = algo_class(
            symbol=symbol,
            side=order_side,
            total_quantity=quantity,
            duration_minutes=duration_minutes,
            order_placer=self.order_placer,
            **kwargs
        )
        
        return algo
    
    def start_algo(self, algo: BaseExecutionAlgo, arrival_price: float,
                  market_data: Optional[Dict] = None):
        """Start an execution algorithm."""
        algo.start(arrival_price, market_data)
        self.active_algos[algo.parent_id] = algo
    
    def stop_algo(self, parent_id: str):
        """Stop an execution algorithm."""
        if parent_id in self.active_algos:
            algo = self.active_algos.pop(parent_id)
            algo.stop()
            self.completed_algos.append(algo)
    
    def get_pending_orders(self, parent_id: Optional[str] = None) -> List[ChildOrder]:
        """Get pending child orders."""
        orders = []
        
        algos = [self.active_algos[parent_id]] if parent_id else list(self.active_algos.values())
        
        for algo in algos:
            for order in algo.child_orders:
                if order.status == OrderStatus.PENDING:
                    if order.scheduled_time <= datetime.now():
                        orders.append(order)
        
        return orders
    
    def record_fill(self, parent_id: str, order_id: str,
                   filled_qty: float, fill_price: float):
        """Record a fill."""
        if parent_id in self.active_algos:
            self.active_algos[parent_id].record_fill(order_id, filled_qty, fill_price)
    
    def get_execution_reports(self, vwap_benchmarks: Dict[str, float]) -> List[ExecutionReport]:
        """Get execution reports for completed algorithms."""
        reports = []
        
        for algo in self.completed_algos:
            vwap = vwap_benchmarks.get(algo.symbol, algo.arrival_price)
            reports.append(algo.get_report(vwap))
        
        return reports


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXECUTION ALGORITHMS")
    print("="*70)
    
    # Simulation parameters
    symbol = "BTC-USDT"
    quantity = 10.0  # BTC
    duration = 30  # minutes
    arrival_price = 45000.0
    
    # 1. TWAP Algorithm
    print("\n1. TWAP (Time Weighted Average Price)")
    print("-" * 50)
    
    twap = TWAPAlgo(
        symbol=symbol,
        side=OrderSide.BUY,
        total_quantity=quantity,
        duration_minutes=duration,
        n_slices=10,
        randomize=True
    )
    
    twap.start(arrival_price)
    
    print(f"   Total quantity: {quantity} BTC")
    print(f"   Duration: {duration} minutes")
    print(f"   Slices: {len(twap.child_orders)}")
    
    for order in twap.child_orders[:3]:
        print(f"     Order {order.order_id[-4:]}: {order.quantity:.4f} BTC at {order.scheduled_time.strftime('%H:%M:%S')}")
    print("     ...")
    
    # 2. VWAP Algorithm
    print("\n2. VWAP (Volume Weighted Average Price)")
    print("-" * 50)
    
    # U-shaped volume profile
    volume_profile = np.array([1.5, 1.2, 0.8, 0.6, 0.5, 0.5, 0.6, 0.8, 1.2, 1.5])
    
    vwap = VWAPAlgo(
        symbol=symbol,
        side=OrderSide.BUY,
        total_quantity=quantity,
        duration_minutes=duration,
        volume_profile=volume_profile,
        n_slices=10
    )
    
    vwap.start(arrival_price)
    
    print(f"   Volume profile: U-shaped (higher at start/end)")
    print(f"   Slice quantities:")
    
    for i, order in enumerate(vwap.child_orders):
        bar = "█" * int(order.quantity / quantity * 50)
        print(f"     Slice {i+1:2d}: {order.quantity:.4f} BTC {bar}")
    
    # 3. Implementation Shortfall
    print("\n3. Implementation Shortfall (Almgren-Chriss)")
    print("-" * 50)
    
    is_algo = ImplementationShortfallAlgo(
        symbol=symbol,
        side=OrderSide.BUY,
        total_quantity=quantity,
        duration_minutes=duration,
        volatility=0.02,
        risk_aversion=0.01,
        impact_coeff=0.01,
        n_slices=10
    )
    
    is_algo.start(arrival_price)
    
    print(f"   Risk aversion: 0.01 (balanced)")
    print(f"   Optimal trajectory (front-loaded for urgency):")
    
    for i, order in enumerate(is_algo.child_orders):
        bar = "█" * int(order.quantity / quantity * 50)
        print(f"     Slice {i+1:2d}: {order.quantity:.4f} BTC {bar}")
    
    # 4. Iceberg Order
    print("\n4. Iceberg Order")
    print("-" * 50)
    
    iceberg = IcebergAlgo(
        symbol=symbol,
        side=OrderSide.BUY,
        total_quantity=quantity,
        duration_minutes=duration,
        display_qty=1.0  # Show only 1 BTC at a time
    )
    
    iceberg.start(arrival_price)
    
    print(f"   Total quantity: {quantity} BTC (hidden)")
    print(f"   Display quantity: 1.0 BTC (visible)")
    print(f"   Number of slices: {len(iceberg.child_orders)}")
    
    # 5. Execution Report (simulated)
    print("\n5. Simulated Execution Report")
    print("-" * 50)
    
    # Simulate fills for TWAP
    for i, order in enumerate(twap.child_orders):
        # Simulate some slippage
        fill_price = arrival_price * (1 + np.random.uniform(-0.001, 0.002))
        twap.record_fill(order.order_id, order.quantity, fill_price)
    
    twap.stop()
    
    # Generate report
    vwap_benchmark = arrival_price * 1.001  # Simulated market VWAP
    report = twap.get_report(vwap_benchmark)
    
    print(f"   Order ID: {report.parent_order_id}")
    print(f"   Symbol: {report.symbol}")
    print(f"   Side: {report.side.value.upper()}")
    print(f"   Filled: {report.filled_quantity:.4f} / {report.total_quantity:.4f}")
    print(f"   Avg Fill Price: ${report.avg_fill_price:,.2f}")
    print(f"   Arrival Price: ${report.arrival_price:,.2f}")
    print(f"   VWAP Benchmark: ${report.vwap_benchmark:,.2f}")
    print(f"   Slippage: {report.slippage_bps:.2f} bps")
    print(f"   Impl. Shortfall: ${report.implementation_shortfall:,.2f}")
    print(f"   Duration: {report.duration_seconds:.1f} seconds")
    
    # 6. Execution Engine
    print("\n6. Execution Engine Usage")
    print("-" * 50)
    
    engine = ExecutionEngine()
    
    # Create multiple algos
    algo1 = engine.create_algo('twap', 'ETH-USDT', 'buy', 50.0, 20, n_slices=10)
    algo2 = engine.create_algo('vwap', 'BTC-USDT', 'sell', 5.0, 30, n_slices=15)
    algo3 = engine.create_algo('is', 'SOL-USDT', 'buy', 100.0, 15, risk_aversion=0.02)
    
    print(f"   Created algos: TWAP (ETH), VWAP (BTC), IS (SOL)")
    
    # Start algos
    engine.start_algo(algo1, 3000.0)
    engine.start_algo(algo2, 45000.0)
    engine.start_algo(algo3, 100.0)
    
    print(f"   Active algos: {len(engine.active_algos)}")
    
    # Get pending orders
    pending = engine.get_pending_orders()
    print(f"   Pending child orders: {len(pending)}")
    
    print("\n" + "="*70)
    print("INTEGRATION GUIDE")
    print("="*70)
    print("""
1. Basic TWAP execution:
   from execution_algos import ExecutionEngine
   
   engine = ExecutionEngine()
   algo = engine.create_algo('twap', 'BTC-USDT', 'buy', 
                             quantity=10.0, duration_minutes=30)
   engine.start_algo(algo, arrival_price=45000)
   
   # In your execution loop
   for order in engine.get_pending_orders():
       fill = place_order(order)
       engine.record_fill(order.parent_id, order.order_id, 
                         fill['quantity'], fill['price'])

2. Choose algorithm based on need:
   - TWAP: Low urgency, even execution
   - VWAP: Match market rhythm, reduce impact
   - IS: Optimal balance of urgency vs impact
   - Iceberg: Hide large order size
   - Adaptive: Dynamic market response

3. Execution quality metrics:
   - Slippage (bps): Cost vs arrival price
   - Implementation Shortfall: Total execution cost
   - VWAP deviation: Performance vs market benchmark
   
4. Best practices:
   - Use participation limits (10-15% of volume)
   - Monitor spread and adjust
   - Cancel on extreme volatility
   - Log all fills for analysis
""")
