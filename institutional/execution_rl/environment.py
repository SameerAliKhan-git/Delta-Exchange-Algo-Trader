"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         LOB ENVIRONMENT - Limit Order Book Simulation                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Gym-compatible environment for training execution agents.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger("ExecutionRL.Environment")


@dataclass
class LOBState:
    """Limit Order Book state."""
    # Bids: [(price, size), ...] sorted descending by price
    bids: List[Tuple[float, float]]
    # Asks: [(price, size), ...] sorted ascending by price
    asks: List[Tuple[float, float]]
    
    # Trade tape
    last_trade_price: float
    last_trade_size: float
    last_trade_side: str
    
    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0][0] - self.bids[0][0]
    
    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid <= 0:
            return 0.0
        return (self.spread / mid) * 10000
    
    def bid_depth(self, levels: int = 10) -> float:
        return sum(size for _, size in self.bids[:levels])
    
    def ask_depth(self, levels: int = 10) -> float:
        return sum(size for _, size in self.asks[:levels])
    
    def imbalance(self, levels: int = 5) -> float:
        bid_vol = self.bid_depth(levels)
        ask_vol = self.ask_depth(levels)
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total


@dataclass
class Order:
    """An order in the LOB."""
    order_id: str
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    filled: float = 0.0
    status: str = 'open'  # open, filled, cancelled
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def remaining(self) -> float:
        return self.size - self.filled


class LOBEnvironment:
    """
    Limit Order Book Environment for RL training.
    
    Simulates a limit order book with:
    - Order matching
    - Queue position tracking
    - Market impact simulation
    
    Usage:
        env = LOBEnvironment()
        state = env.reset()
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
    """
    
    def __init__(
        self,
        initial_mid: float = 50000.0,
        spread_bps: float = 2.0,
        volatility: float = 0.02,
        depth_per_level: float = 1.0,
        n_levels: int = 10,
        tick_size: float = 0.5,
        time_horizon_steps: int = 100,
    ):
        """
        Initialize LOB environment.
        
        Args:
            initial_mid: Initial mid price
            spread_bps: Initial spread in basis points
            volatility: Price volatility
            depth_per_level: Average depth per level
            n_levels: Number of price levels to simulate
            tick_size: Minimum price increment
            time_horizon_steps: Steps per episode
        """
        self.initial_mid = initial_mid
        self.spread_bps = spread_bps
        self.volatility = volatility
        self.depth_per_level = depth_per_level
        self.n_levels = n_levels
        self.tick_size = tick_size
        self.time_horizon_steps = time_horizon_steps
        
        # State
        self._lob: Optional[LOBState] = None
        self._our_order: Optional[Order] = None
        self._target_qty: float = 1.0
        self._filled_qty: float = 0.0
        self._step_count: int = 0
        self._total_slippage: float = 0.0
        
        # History
        self._price_history: List[float] = []
        self._fill_history: List[Dict] = []
        
        logger.info("LOBEnvironment initialized")
    
    def reset(
        self,
        target_qty: float = 1.0,
        side: str = 'buy',
    ) -> LOBState:
        """
        Reset environment for new episode.
        
        Args:
            target_qty: Target quantity to fill
            side: 'buy' or 'sell'
            
        Returns:
            Initial LOB state
        """
        self._target_qty = target_qty
        self._filled_qty = 0.0
        self._step_count = 0
        self._total_slippage = 0.0
        self._our_order = None
        self._price_history = []
        self._fill_history = []
        
        # Generate initial LOB
        self._lob = self._generate_lob(self.initial_mid)
        
        return self._lob
    
    def _generate_lob(self, mid_price: float) -> LOBState:
        """Generate LOB around mid price."""
        spread = mid_price * (self.spread_bps / 10000)
        half_spread = spread / 2
        
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # Generate bids
        bids = []
        for i in range(self.n_levels):
            price = best_bid - i * self.tick_size
            size = self.depth_per_level * (1 + 0.3 * np.random.randn())
            size = max(0.1, size)
            bids.append((price, size))
        
        # Generate asks
        asks = []
        for i in range(self.n_levels):
            price = best_ask + i * self.tick_size
            size = self.depth_per_level * (1 + 0.3 * np.random.randn())
            size = max(0.1, size)
            asks.append((price, size))
        
        return LOBState(
            bids=bids,
            asks=asks,
            last_trade_price=mid_price,
            last_trade_size=0.1,
            last_trade_side='buy',
        )
    
    def step(
        self,
        action: int,
    ) -> Tuple[LOBState, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0-6)
            
        Returns:
            (next_state, reward, done, info)
        """
        self._step_count += 1
        
        # Execute action
        fill, slippage = self._execute_action(action)
        self._filled_qty += fill
        self._total_slippage += slippage
        
        if fill > 0:
            self._fill_history.append({
                'step': self._step_count,
                'fill': fill,
                'slippage': slippage,
            })
        
        # Update LOB (price moves, depth changes)
        self._update_lob()
        
        # Calculate reward
        reward = self._calculate_reward(fill, slippage, action)
        
        # Check done
        done = (
            self._filled_qty >= self._target_qty or
            self._step_count >= self.time_horizon_steps
        )
        
        # Info
        info = {
            'fill': fill,
            'slippage': slippage,
            'total_filled': self._filled_qty,
            'target': self._target_qty,
            'fill_rate': self._filled_qty / self._target_qty,
            'total_slippage': self._total_slippage,
        }
        
        return self._lob, reward, done, info
    
    def _execute_action(self, action: int) -> Tuple[float, float]:
        """
        Execute action and return (fill_qty, slippage_bps).
        """
        if action == 0:  # CANCEL
            if self._our_order:
                self._our_order.status = 'cancelled'
                self._our_order = None
            return 0.0, 0.0
        
        if action == 6:  # HOLD
            # Check if existing order fills
            if self._our_order and self._our_order.status == 'open':
                return self._check_passive_fill()
            return 0.0, 0.0
        
        remaining = self._target_qty - self._filled_qty
        mid = self._lob.mid_price
        
        # Determine price based on action
        if action == 1:  # PASSIVE_5TH
            if len(self._lob.bids) > 4:
                price = self._lob.bids[4][0]
            else:
                price = mid * 0.999
            fill_prob = 0.1  # Low fill probability
        elif action == 2:  # PASSIVE_3RD
            if len(self._lob.bids) > 2:
                price = self._lob.bids[2][0]
            else:
                price = mid * 0.9995
            fill_prob = 0.3
        elif action == 3:  # MID
            price = mid
            fill_prob = 0.5
        elif action == 4:  # CROSS_1
            if len(self._lob.asks) > 0:
                price = self._lob.asks[0][0]
            else:
                price = mid * 1.001
            fill_prob = 0.9
        elif action == 5:  # CROSS_3
            if len(self._lob.asks) > 2:
                price = self._lob.asks[2][0]
            else:
                price = mid * 1.003
            fill_prob = 1.0
        else:
            price = mid
            fill_prob = 0.5
        
        # Simulate fill
        if np.random.random() < fill_prob:
            fill = min(remaining, 0.5)  # Partial fills
            slippage = ((price - mid) / mid) * 10000  # bps
            return fill, slippage
        else:
            # Create passive order
            self._our_order = Order(
                order_id=f"ord_{self._step_count}",
                side='buy',
                price=price,
                size=remaining,
            )
            return 0.0, 0.0
    
    def _check_passive_fill(self) -> Tuple[float, float]:
        """Check if passive order fills."""
        if not self._our_order:
            return 0.0, 0.0
        
        # Simple fill probability based on queue position
        fill_prob = 0.15  # Base probability per step
        
        if np.random.random() < fill_prob:
            fill = min(self._our_order.remaining, 0.3)
            self._our_order.filled += fill
            
            mid = self._lob.mid_price
            slippage = ((self._our_order.price - mid) / mid) * 10000
            
            if self._our_order.filled >= self._our_order.size:
                self._our_order.status = 'filled'
            
            return fill, slippage
        
        return 0.0, 0.0
    
    def _update_lob(self) -> None:
        """Update LOB state (price moves, depth changes)."""
        # Random price movement
        old_mid = self._lob.mid_price
        price_change = np.random.randn() * old_mid * self.volatility / 100
        new_mid = old_mid + price_change
        
        self._price_history.append(new_mid)
        
        # Regenerate LOB around new mid
        self._lob = self._generate_lob(new_mid)
    
    def _calculate_reward(
        self,
        fill: float,
        slippage: float,
        action: int,
    ) -> float:
        """Calculate reward."""
        reward = -slippage  # Penalize slippage
        
        if action == 0:  # Cancel penalty
            reward -= 0.1
        
        if fill > 0:
            reward += 0.1  # Small bonus for filling
        
        # Time pressure penalty
        time_remaining = 1 - (self._step_count / self.time_horizon_steps)
        fill_rate = self._filled_qty / self._target_qty
        
        if fill_rate < time_remaining:
            reward -= 0.05  # Penalty for falling behind
        
        return reward
    
    def get_observation(self) -> np.ndarray:
        """Get observation vector for agent."""
        if not self._lob:
            return np.zeros(51)
        
        bid_prices = np.array([b[0] for b in self._lob.bids])
        bid_sizes = np.array([b[1] for b in self._lob.bids])
        ask_prices = np.array([a[0] for a in self._lob.asks])
        ask_sizes = np.array([a[1] for a in self._lob.asks])
        
        return np.concatenate([
            bid_prices[:10],
            bid_sizes[:10],
            ask_prices[:10],
            ask_sizes[:10],
            np.array([
                float(self._our_order is not None),
                self._our_order.price if self._our_order else 0,
                self._target_qty,
                self._filled_qty,
                self._filled_qty / self._target_qty,
                1 - self._step_count / self.time_horizon_steps,
                self._lob.mid_price,
                self._lob.spread_bps,
                self._lob.imbalance(),
                float(len(self._price_history)),
                np.std(self._price_history[-20:]) if len(self._price_history) > 1 else 0,
            ])
        ])


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create environment
    env = LOBEnvironment(
        initial_mid=50000,
        spread_bps=2.0,
        time_horizon_steps=50,
    )
    
    # Run episode with random actions
    state = env.reset(target_qty=2.0)
    print(f"Initial mid: {state.mid_price:.2f}, spread: {state.spread_bps:.2f} bps")
    
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        action = np.random.randint(0, 7)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}: action={action}, fill_rate={info['fill_rate']:.1%}, "
                  f"slippage={info['total_slippage']:.2f}bps")
    
    print(f"\nEpisode complete:")
    print(f"  Steps: {step}")
    print(f"  Filled: {info['total_filled']:.2f} / {info['target']:.2f}")
    print(f"  Total slippage: {info['total_slippage']:.2f} bps")
    print(f"  Total reward: {total_reward:.2f}")
