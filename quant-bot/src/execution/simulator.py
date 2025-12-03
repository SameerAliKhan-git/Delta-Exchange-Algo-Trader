"""
Order Execution Simulator

Simulates realistic order execution for paper trading and backtesting:
- Market impact modeling
- Latency simulation
- Partial fills
- Order book dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import PriorityQueue
from loguru import logger


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionMode(Enum):
    """Execution simulation mode."""
    INSTANT = "instant"  # Immediate fill at current price
    REALISTIC = "realistic"  # With slippage and delays
    ORDER_BOOK = "order_book"  # Full order book simulation


@dataclass
class ExecutionOrder:
    """Order for execution."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit', 'stop'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fills: List[Tuple[float, float, datetime]] = field(default_factory=list)  # (price, qty, time)
    commission: float = 0.0
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def remaining_quantity(self) -> float:
        """Remaining unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is fully filled or terminal."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]


@dataclass
class ExecutionConfig:
    """Execution simulator configuration."""
    mode: ExecutionMode = ExecutionMode.REALISTIC
    
    # Latency settings
    min_latency_ms: int = 10
    max_latency_ms: int = 100
    
    # Slippage settings
    base_slippage_bps: float = 5.0  # Basis points
    impact_factor: float = 0.1  # Market impact multiplier
    
    # Fill probability
    market_fill_prob: float = 0.99
    limit_fill_prob: float = 0.80
    
    # Commission
    maker_fee_pct: float = 0.001  # 0.1%
    taker_fee_pct: float = 0.002  # 0.2%
    
    # Partial fills
    allow_partial: bool = True
    min_fill_pct: float = 0.1  # Minimum 10% fill


@dataclass
class MarketSnapshot:
    """Current market state snapshot."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    volume: float = 0.0
    
    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid) * 10000 if self.mid > 0 else 0


class ExecutionSimulator:
    """
    Realistic order execution simulator.
    
    Features:
    - Latency modeling
    - Slippage based on order size and market conditions
    - Partial fills
    - Market impact
    
    Example:
        simulator = ExecutionSimulator()
        order = ExecutionOrder(
            order_id="001",
            symbol="BTC/USD",
            side="buy",
            quantity=1.0,
            order_type="market"
        )
        filled_order = simulator.submit_order(order, market_snapshot)
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize simulator."""
        self.config = config or ExecutionConfig()
        
        # Order tracking
        self.pending_orders: Dict[str, ExecutionOrder] = {}
        self.completed_orders: Dict[str, ExecutionOrder] = {}
        self.order_history: List[ExecutionOrder] = []
        
        # Execution statistics
        self.total_volume = 0.0
        self.total_slippage = 0.0
        self.total_commission = 0.0
        
        # Order ID counter
        self._order_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD{self._order_counter:06d}"
    
    def _simulate_latency(self) -> timedelta:
        """Simulate network/processing latency."""
        latency_ms = np.random.uniform(
            self.config.min_latency_ms,
            self.config.max_latency_ms
        )
        return timedelta(milliseconds=latency_ms)
    
    def _calculate_slippage(
        self,
        order: ExecutionOrder,
        market: MarketSnapshot
    ) -> float:
        """
        Calculate execution slippage.
        
        Components:
        1. Base slippage (bid-ask spread)
        2. Market impact (order size vs liquidity)
        3. Volatility adjustment
        
        Args:
            order: Order to execute
            market: Current market state
            
        Returns:
            Slippage in price units
        """
        # Base slippage from spread
        half_spread = market.spread / 2
        
        # Market impact based on order size
        if market.volume > 0:
            participation_rate = (order.quantity * market.last) / market.volume
            impact = market.last * self.config.impact_factor * np.sqrt(participation_rate)
        else:
            impact = market.last * self.config.base_slippage_bps / 10000
        
        total_slippage = half_spread + impact
        
        # Direction-dependent slippage
        if order.side == 'buy':
            return total_slippage
        else:
            return -total_slippage
    
    def _calculate_fill_price(
        self,
        order: ExecutionOrder,
        market: MarketSnapshot
    ) -> float:
        """
        Calculate actual fill price.
        
        Args:
            order: Order to execute
            market: Current market state
            
        Returns:
            Fill price
        """
        if order.order_type == 'market':
            # Market order fills at best available price + slippage
            if order.side == 'buy':
                base_price = market.ask
            else:
                base_price = market.bid
            
            slippage = self._calculate_slippage(order, market)
            return base_price + slippage
        
        elif order.order_type == 'limit':
            # Limit order fills at limit price or better
            if order.side == 'buy':
                # Buy limit fills if price <= limit
                if market.ask <= order.limit_price:
                    return min(market.ask, order.limit_price)
            else:
                # Sell limit fills if price >= limit
                if market.bid >= order.limit_price:
                    return max(market.bid, order.limit_price)
            
            # Limit not reached
            return None
        
        elif order.order_type == 'stop':
            # Stop order triggers when price crosses stop level
            if order.side == 'buy':
                if market.ask >= order.stop_price:
                    # Triggered - execute as market order
                    slippage = self._calculate_slippage(order, market)
                    return market.ask + slippage
            else:
                if market.bid <= order.stop_price:
                    slippage = self._calculate_slippage(order, market)
                    return market.bid + slippage
            
            # Stop not triggered
            return None
        
        return None
    
    def _calculate_fill_quantity(
        self,
        order: ExecutionOrder,
        market: MarketSnapshot
    ) -> float:
        """
        Calculate fill quantity (may be partial).
        
        Args:
            order: Order to execute
            market: Current market state
            
        Returns:
            Fill quantity
        """
        if not self.config.allow_partial:
            # All or nothing
            return order.remaining_quantity
        
        # Available liquidity
        if order.side == 'buy':
            available = market.ask_size
        else:
            available = market.bid_size
        
        # Random partial fill
        if available > 0 and available < order.remaining_quantity:
            fill_pct = np.random.uniform(
                self.config.min_fill_pct,
                min(1.0, available / order.remaining_quantity)
            )
            return order.remaining_quantity * fill_pct
        
        return order.remaining_quantity
    
    def _calculate_commission(
        self,
        order: ExecutionOrder,
        fill_price: float,
        fill_quantity: float
    ) -> float:
        """Calculate commission for fill."""
        notional = fill_price * fill_quantity
        
        if order.order_type == 'limit':
            # Maker fee for limit orders
            return notional * self.config.maker_fee_pct
        else:
            # Taker fee for market/stop orders
            return notional * self.config.taker_fee_pct
    
    def _should_fill(
        self,
        order: ExecutionOrder,
        market: MarketSnapshot
    ) -> bool:
        """Determine if order should be filled (probabilistic)."""
        if order.order_type == 'market':
            return np.random.random() < self.config.market_fill_prob
        else:
            return np.random.random() < self.config.limit_fill_prob
    
    def submit_order(
        self,
        order: ExecutionOrder,
        market: MarketSnapshot
    ) -> ExecutionOrder:
        """
        Submit and potentially execute an order.
        
        Args:
            order: Order to submit
            market: Current market state
            
        Returns:
            Updated order with fills
        """
        # Generate ID if not set
        if not order.order_id:
            order.order_id = self._generate_order_id()
        
        # Simulate latency
        latency = self._simulate_latency()
        execution_time = market.timestamp + latency
        
        # Update status
        order.status = OrderStatus.SUBMITTED
        
        # Check if should fill
        if not self._should_fill(order, market):
            if order.order_type == 'market':
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order {order.order_id} rejected (no fill)")
            else:
                # Limit/stop order pending
                self.pending_orders[order.order_id] = order
                logger.debug(f"Order {order.order_id} pending")
            return order
        
        # Calculate fill price
        fill_price = self._calculate_fill_price(order, market)
        
        if fill_price is None:
            # Limit/stop conditions not met
            self.pending_orders[order.order_id] = order
            return order
        
        # Calculate fill quantity
        fill_quantity = self._calculate_fill_quantity(order, market)
        
        # Calculate commission
        commission = self._calculate_commission(order, fill_price, fill_quantity)
        
        # Record fill
        order.fills.append((fill_price, fill_quantity, execution_time))
        order.filled_quantity += fill_quantity
        order.commission += commission
        
        # Update average fill price
        total_value = sum(p * q for p, q, _ in order.fills)
        total_qty = sum(q for _, q, _ in order.fills)
        order.average_fill_price = total_value / total_qty if total_qty > 0 else 0
        
        # Update status
        if order.filled_quantity >= order.quantity * 0.999:  # Allow small rounding
            order.status = OrderStatus.FILLED
            self.completed_orders[order.order_id] = order
        else:
            order.status = OrderStatus.PARTIAL
            self.pending_orders[order.order_id] = order
        
        # Update statistics
        self.total_volume += fill_quantity * fill_price
        self.total_commission += commission
        
        # Calculate and track slippage
        if order.side == 'buy':
            expected_price = market.mid
            slippage = (fill_price - expected_price) * fill_quantity
        else:
            expected_price = market.mid
            slippage = (expected_price - fill_price) * fill_quantity
        self.total_slippage += slippage
        
        logger.debug(
            f"Order {order.order_id} {order.status.value}: "
            f"{fill_quantity:.4f} @ {fill_price:.4f}"
        )
        
        self.order_history.append(order)
        
        return order
    
    def update_pending_orders(
        self,
        market: MarketSnapshot
    ) -> List[ExecutionOrder]:
        """
        Update pending limit/stop orders with new market data.
        
        Args:
            market: Current market state
            
        Returns:
            List of orders that were filled/updated
        """
        updated = []
        
        for order_id, order in list(self.pending_orders.items()):
            if order.symbol != market.symbol:
                continue
            
            # Check if conditions met
            fill_price = self._calculate_fill_price(order, market)
            
            if fill_price is not None and self._should_fill(order, market):
                # Execute fill
                fill_quantity = self._calculate_fill_quantity(order, market)
                commission = self._calculate_commission(order, fill_price, fill_quantity)
                
                order.fills.append((fill_price, fill_quantity, market.timestamp))
                order.filled_quantity += fill_quantity
                order.commission += commission
                
                total_value = sum(p * q for p, q, _ in order.fills)
                total_qty = sum(q for _, q, _ in order.fills)
                order.average_fill_price = total_value / total_qty if total_qty > 0 else 0
                
                if order.filled_quantity >= order.quantity * 0.999:
                    order.status = OrderStatus.FILLED
                    del self.pending_orders[order_id]
                    self.completed_orders[order_id] = order
                else:
                    order.status = OrderStatus.PARTIAL
                
                self.total_volume += fill_quantity * fill_price
                self.total_commission += commission
                
                updated.append(order)
        
        return updated
    
    def cancel_order(self, order_id: str) -> Optional[ExecutionOrder]:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancelled order or None
        """
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            self.completed_orders[order_id] = order
            logger.debug(f"Order {order_id} cancelled")
            return order
        
        return None
    
    def cancel_all(self, symbol: Optional[str] = None) -> List[ExecutionOrder]:
        """Cancel all pending orders."""
        cancelled = []
        
        for order_id, order in list(self.pending_orders.items()):
            if symbol is None or order.symbol == symbol:
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order_id]
                self.completed_orders[order_id] = order
                cancelled.append(order)
        
        return cancelled
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        return {
            'total_orders': len(self.order_history),
            'pending_orders': len(self.pending_orders),
            'completed_orders': len(self.completed_orders),
            'total_volume': self.total_volume,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'avg_slippage_bps': (
                (self.total_slippage / self.total_volume * 10000)
                if self.total_volume > 0 else 0
            )
        }


class PaperTradingEngine:
    """
    Paper trading engine for live simulation.
    
    Tracks positions and PnL in real-time with simulated execution.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        config: Optional[ExecutionConfig] = None
    ):
        """Initialize paper trading engine."""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.simulator = ExecutionSimulator(config)
        
        # Positions
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.position_values: Dict[str, float] = {}  # symbol -> entry value
        
        # PnL tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.peak_equity = initial_capital
    
    def submit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market: MarketSnapshot
    ) -> ExecutionOrder:
        """Submit a market order."""
        order = ExecutionOrder(
            order_id=self.simulator._generate_order_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="market"
        )
        
        filled_order = self.simulator.submit_order(order, market)
        
        if filled_order.status == OrderStatus.FILLED:
            self._update_position(filled_order)
        
        return filled_order
    
    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        market: MarketSnapshot
    ) -> ExecutionOrder:
        """Submit a limit order."""
        order = ExecutionOrder(
            order_id=self.simulator._generate_order_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="limit",
            limit_price=limit_price
        )
        
        filled_order = self.simulator.submit_order(order, market)
        
        if filled_order.status == OrderStatus.FILLED:
            self._update_position(filled_order)
        
        return filled_order
    
    def _update_position(self, order: ExecutionOrder) -> None:
        """Update position after fill."""
        symbol = order.symbol
        fill_value = order.average_fill_price * order.filled_quantity
        
        current_position = self.positions.get(symbol, 0.0)
        
        if order.side == 'buy':
            # Add to position
            if current_position >= 0:
                # Adding to long or opening long
                self.positions[symbol] = current_position + order.filled_quantity
                self.position_values[symbol] = (
                    self.position_values.get(symbol, 0.0) + fill_value
                )
            else:
                # Reducing short
                if order.filled_quantity >= abs(current_position):
                    # Close short and open long
                    close_qty = abs(current_position)
                    pnl = self._calculate_close_pnl(symbol, close_qty, order.average_fill_price)
                    self.realized_pnl += pnl
                    self.capital += pnl
                    
                    remaining = order.filled_quantity - close_qty
                    self.positions[symbol] = remaining
                    self.position_values[symbol] = remaining * order.average_fill_price
                else:
                    # Partially close short
                    pnl = self._calculate_close_pnl(symbol, order.filled_quantity, order.average_fill_price)
                    self.realized_pnl += pnl
                    self.capital += pnl
                    self.positions[symbol] = current_position + order.filled_quantity
            
            self.capital -= fill_value + order.commission
        
        else:  # sell
            if current_position <= 0:
                # Adding to short or opening short
                self.positions[symbol] = current_position - order.filled_quantity
                self.position_values[symbol] = (
                    self.position_values.get(symbol, 0.0) + fill_value
                )
            else:
                # Reducing long
                if order.filled_quantity >= current_position:
                    # Close long and open short
                    pnl = self._calculate_close_pnl(symbol, current_position, order.average_fill_price)
                    self.realized_pnl += pnl
                    self.capital += pnl
                    
                    remaining = order.filled_quantity - current_position
                    self.positions[symbol] = -remaining
                    self.position_values[symbol] = remaining * order.average_fill_price
                else:
                    # Partially close long
                    pnl = self._calculate_close_pnl(symbol, order.filled_quantity, order.average_fill_price)
                    self.realized_pnl += pnl
                    self.capital += pnl
                    self.positions[symbol] = current_position - order.filled_quantity
            
            self.capital += fill_value - order.commission
    
    def _calculate_close_pnl(
        self,
        symbol: str,
        quantity: float,
        close_price: float
    ) -> float:
        """Calculate PnL for closing position."""
        if symbol not in self.positions or self.positions[symbol] == 0:
            return 0.0
        
        position_size = abs(self.positions[symbol])
        avg_entry = self.position_values.get(symbol, 0) / position_size if position_size > 0 else 0
        
        if self.positions[symbol] > 0:  # Long position
            pnl = (close_price - avg_entry) * quantity
        else:  # Short position
            pnl = (avg_entry - close_price) * quantity
        
        return pnl
    
    def update_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Update unrealized PnL with current prices."""
        self.unrealized_pnl = 0.0
        
        for symbol, quantity in self.positions.items():
            if quantity == 0 or symbol not in prices:
                continue
            
            current_price = prices[symbol]
            entry_value = self.position_values.get(symbol, 0)
            
            if abs(quantity) > 0:
                avg_entry = entry_value / abs(quantity)
                
                if quantity > 0:  # Long
                    pnl = (current_price - avg_entry) * quantity
                else:  # Short
                    pnl = (avg_entry - current_price) * abs(quantity)
                
                self.unrealized_pnl += pnl
        
        return self.unrealized_pnl
    
    def get_equity(self) -> float:
        """Get total equity."""
        return self.capital + self.unrealized_pnl
    
    def get_summary(self) -> Dict:
        """Get account summary."""
        equity = self.get_equity()
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        return {
            'capital': self.capital,
            'equity': equity,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_return': (equity / self.initial_capital) - 1,
            'drawdown': drawdown,
            'positions': self.positions.copy(),
            'execution_stats': self.simulator.get_statistics()
        }


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("EXECUTION SIMULATOR DEMO")
    print("=" * 60)
    
    # Create simulator
    config = ExecutionConfig(
        mode=ExecutionMode.REALISTIC,
        base_slippage_bps=5.0,
        impact_factor=0.1
    )
    
    simulator = ExecutionSimulator(config)
    
    # Create market snapshot
    market = MarketSnapshot(
        symbol="BTC/USD",
        timestamp=datetime.now(),
        bid=50000.0,
        ask=50010.0,
        last=50005.0,
        bid_size=10.0,
        ask_size=8.0,
        volume=1000.0
    )
    
    print(f"\nMarket: {market.symbol}")
    print(f"Bid: ${market.bid:.2f}, Ask: ${market.ask:.2f}")
    print(f"Spread: ${market.spread:.2f} ({market.spread_bps:.1f} bps)")
    
    # Submit market order
    order = ExecutionOrder(
        order_id="",
        symbol="BTC/USD",
        side="buy",
        quantity=0.5,
        order_type="market"
    )
    
    filled_order = simulator.submit_order(order, market)
    
    print(f"\nOrder: {filled_order.order_id}")
    print(f"Status: {filled_order.status.value}")
    print(f"Filled: {filled_order.filled_quantity:.4f} @ ${filled_order.average_fill_price:.2f}")
    print(f"Commission: ${filled_order.commission:.2f}")
    
    # Get statistics
    stats = simulator.get_statistics()
    print(f"\nExecution Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
