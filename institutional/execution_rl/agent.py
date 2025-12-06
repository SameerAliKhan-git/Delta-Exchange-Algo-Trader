"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         EXECUTION RL AGENT - Quote-Level Reinforcement Learning              ║
║                                                                               ║
║  LOB-aware execution for optimal order placement                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

State:
  s_t = (LOB_10levels, queue_position, time_to_horizon, target_fill, current_fill)

Action:
  a_t ∈ {cancel, passive_5th, passive_3rd, mid, cross_1, cross_3}

Reward:
  r_t = (−1) × realized slippage (bps) − 0.1 × cancel_penalty

Training:
  PPO with 8-hour replay from production fills
  Weekly retrain, A/B deploy with 5% traffic
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger("ExecutionRL.Agent")


class ExecutionAction(Enum):
    """Available execution actions."""
    CANCEL = 0          # Cancel existing order
    PASSIVE_5TH = 1     # Post at 5th level (very passive)
    PASSIVE_3RD = 2     # Post at 3rd level (passive)
    MID = 3             # Post at mid price
    CROSS_1 = 4         # Cross the spread (1 level)
    CROSS_3 = 5         # Aggressive crossing (3 levels)
    HOLD = 6            # Keep current order


@dataclass
class ExecutionState:
    """State for execution agent."""
    # LOB features (10 levels each side)
    bid_prices: np.ndarray   # [10]
    bid_sizes: np.ndarray    # [10]
    ask_prices: np.ndarray   # [10]
    ask_sizes: np.ndarray    # [10]
    
    # Order state
    queue_position: float    # 0-1, position in queue
    has_order: bool
    order_price: float
    order_side: str          # 'buy' or 'sell'
    
    # Execution targets
    target_fill: float       # Target quantity to fill
    current_fill: float      # Already filled
    time_remaining_pct: float  # 0-1, time left in horizon
    
    # Market state
    mid_price: float
    spread_bps: float
    volatility: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for model."""
        return np.concatenate([
            self.bid_prices,
            self.bid_sizes,
            self.ask_prices,
            self.ask_sizes,
            np.array([
                self.queue_position,
                float(self.has_order),
                self.order_price / self.mid_price if self.mid_price > 0 else 0,
                1.0 if self.order_side == 'buy' else 0.0,
                self.target_fill,
                self.current_fill,
                self.current_fill / (self.target_fill + 1e-10),  # Fill rate
                self.time_remaining_pct,
                self.mid_price,
                self.spread_bps,
                self.volatility,
            ])
        ])


@dataclass
class ExecutionResult:
    """Result of an execution action."""
    action: ExecutionAction
    order_placed: bool
    order_price: float
    order_size: float
    filled: float
    slippage_bps: float
    execution_time_ms: float


class ExecutionRLAgent:
    """
    RL Agent for optimal order execution.
    
    Uses trained policy to decide optimal order placement
    to minimize execution slippage.
    
    Usage:
        agent = ExecutionRLAgent()
        
        # Get action for current state
        state = ExecutionState(...)
        action = agent.get_action(state)
        
        # Execute and report reward
        result = execute_order(action)
        agent.update(state, action, reward, next_state)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        exploration_rate: float = 0.1,
        slippage_penalty: float = 1.0,
        cancel_penalty: float = 0.1,
    ):
        """
        Initialize execution agent.
        
        Args:
            model_path: Path to trained model weights
            exploration_rate: Epsilon for exploration
            slippage_penalty: Penalty multiplier for slippage
            cancel_penalty: Penalty for canceling orders
        """
        self.exploration_rate = exploration_rate
        self.slippage_penalty = slippage_penalty
        self.cancel_penalty = cancel_penalty
        
        # Model (in production, load from file)
        self._model = None
        if model_path:
            self._load_model(model_path)
        
        # Experience buffer for training
        self._experience_buffer: List[Dict] = []
        self._max_buffer_size = 10000
        
        # Statistics
        self._stats = {
            'actions_taken': 0,
            'total_slippage_bps': 0.0,
            'fills': 0,
            'cancels': 0,
        }
        
        logger.info("ExecutionRLAgent initialized")
    
    def _load_model(self, path: str) -> None:
        """Load trained model."""
        # In production, load PyTorch/TensorFlow model
        logger.info(f"Loading model from {path}")
        pass
    
    def get_action(
        self,
        state: ExecutionState,
        deterministic: bool = False,
    ) -> ExecutionAction:
        """
        Get optimal action for current state.
        
        Args:
            state: Current execution state
            deterministic: If True, no exploration
            
        Returns:
            ExecutionAction to take
        """
        self._stats['actions_taken'] += 1
        
        # Exploration
        if not deterministic and np.random.random() < self.exploration_rate:
            return np.random.choice(list(ExecutionAction))
        
        # If we have a trained model, use it
        if self._model is not None:
            return self._model_predict(state)
        
        # Otherwise, use heuristic policy
        return self._heuristic_policy(state)
    
    def _model_predict(self, state: ExecutionState) -> ExecutionAction:
        """Get action from trained model."""
        # In production, run inference
        state_vec = state.to_vector()
        
        # Placeholder: would run through neural network
        action_probs = np.ones(len(ExecutionAction)) / len(ExecutionAction)
        action_idx = np.argmax(action_probs)
        
        return ExecutionAction(action_idx)
    
    def _heuristic_policy(self, state: ExecutionState) -> ExecutionAction:
        """
        Rule-based heuristic policy.
        
        Used when no trained model is available.
        """
        fill_rate = state.current_fill / (state.target_fill + 1e-10)
        time_left = state.time_remaining_pct
        
        # If almost done, be patient
        if fill_rate > 0.9:
            if state.has_order:
                return ExecutionAction.HOLD
            return ExecutionAction.PASSIVE_3RD
        
        # If running out of time, be aggressive
        if time_left < 0.1:
            return ExecutionAction.CROSS_3
        
        if time_left < 0.3:
            return ExecutionAction.CROSS_1
        
        # Normal operation: based on spread
        if state.spread_bps < 2:
            # Tight spread: can be passive
            return ExecutionAction.PASSIVE_3RD
        elif state.spread_bps < 5:
            return ExecutionAction.MID
        else:
            # Wide spread: be more aggressive
            return ExecutionAction.CROSS_1
    
    def calculate_reward(
        self,
        slippage_bps: float,
        cancelled: bool = False,
        filled: bool = True,
        urgency_met: bool = True,
    ) -> float:
        """
        Calculate reward for an execution.
        
        reward = (-1) × realized_slippage - cancel_penalty × cancelled
        """
        reward = -self.slippage_penalty * slippage_bps
        
        if cancelled:
            reward -= self.cancel_penalty
        
        if not filled:
            reward -= 0.5  # Penalty for not filling
        
        if not urgency_met:
            reward -= 1.0  # Penalty for missing time target
        
        return reward
    
    def update(
        self,
        state: ExecutionState,
        action: ExecutionAction,
        reward: float,
        next_state: ExecutionState,
        done: bool = False,
    ) -> None:
        """
        Update agent with experience.
        
        Args:
            state: State when action was taken
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode done
        """
        # Store experience
        experience = {
            'state': state.to_vector(),
            'action': action.value,
            'reward': reward,
            'next_state': next_state.to_vector(),
            'done': done,
            'timestamp': datetime.now().isoformat(),
        }
        
        self._experience_buffer.append(experience)
        
        # Trim buffer
        if len(self._experience_buffer) > self._max_buffer_size:
            self._experience_buffer = self._experience_buffer[-self._max_buffer_size:]
        
        # Update statistics
        if action == ExecutionAction.CANCEL:
            self._stats['cancels'] += 1
        self._stats['total_slippage_bps'] += abs(reward)
        self._stats['fills'] += 1 if done else 0
    
    def get_order_params(
        self,
        action: ExecutionAction,
        state: ExecutionState,
    ) -> Dict:
        """
        Convert action to order parameters.
        
        Args:
            action: Action to convert
            state: Current state
            
        Returns:
            Dict with order parameters
        """
        if action == ExecutionAction.CANCEL:
            return {'cancel': True}
        
        if action == ExecutionAction.HOLD:
            return {'hold': True}
        
        remaining = state.target_fill - state.current_fill
        
        if state.order_side == 'buy':
            if action == ExecutionAction.PASSIVE_5TH:
                price = state.bid_prices[4] if len(state.bid_prices) > 4 else state.mid_price * 0.999
            elif action == ExecutionAction.PASSIVE_3RD:
                price = state.bid_prices[2] if len(state.bid_prices) > 2 else state.mid_price * 0.9995
            elif action == ExecutionAction.MID:
                price = state.mid_price
            elif action == ExecutionAction.CROSS_1:
                price = state.ask_prices[0] if len(state.ask_prices) > 0 else state.mid_price * 1.001
            elif action == ExecutionAction.CROSS_3:
                price = state.ask_prices[2] if len(state.ask_prices) > 2 else state.mid_price * 1.003
            else:
                price = state.mid_price
        else:
            # Sell side (mirror)
            if action == ExecutionAction.PASSIVE_5TH:
                price = state.ask_prices[4] if len(state.ask_prices) > 4 else state.mid_price * 1.001
            elif action == ExecutionAction.PASSIVE_3RD:
                price = state.ask_prices[2] if len(state.ask_prices) > 2 else state.mid_price * 1.0005
            elif action == ExecutionAction.MID:
                price = state.mid_price
            elif action == ExecutionAction.CROSS_1:
                price = state.bid_prices[0] if len(state.bid_prices) > 0 else state.mid_price * 0.999
            elif action == ExecutionAction.CROSS_3:
                price = state.bid_prices[2] if len(state.bid_prices) > 2 else state.mid_price * 0.997
            else:
                price = state.mid_price
        
        return {
            'price': price,
            'size': remaining,
            'side': state.order_side,
            'type': 'limit' if action in [ExecutionAction.PASSIVE_5TH, ExecutionAction.PASSIVE_3RD, ExecutionAction.MID] else 'market',
        }
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            **self._stats,
            'buffer_size': len(self._experience_buffer),
            'avg_slippage_bps': (
                self._stats['total_slippage_bps'] / max(1, self._stats['fills'])
            ),
        }
    
    def save_experience(self, path: str) -> None:
        """Save experience buffer for training."""
        import json
        with open(path, 'w') as f:
            json.dump(self._experience_buffer, f)
        logger.info(f"Saved {len(self._experience_buffer)} experiences to {path}")
    
    def load_experience(self, path: str) -> None:
        """Load experience buffer."""
        import json
        with open(path, 'r') as f:
            self._experience_buffer = json.load(f)
        logger.info(f"Loaded {len(self._experience_buffer)} experiences from {path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create agent
    agent = ExecutionRLAgent(exploration_rate=0.2)
    
    # Create example state
    state = ExecutionState(
        bid_prices=np.array([50000 - i*5 for i in range(10)]),
        bid_sizes=np.array([0.5, 0.8, 1.0, 1.2, 0.3, 0.5, 0.7, 0.2, 0.1, 0.1]),
        ask_prices=np.array([50001 + i*5 for i in range(10)]),
        ask_sizes=np.array([0.4, 0.6, 0.9, 1.1, 0.4, 0.6, 0.5, 0.3, 0.2, 0.1]),
        queue_position=0.3,
        has_order=False,
        order_price=0,
        order_side='buy',
        target_fill=1.0,
        current_fill=0.0,
        time_remaining_pct=0.8,
        mid_price=50000.5,
        spread_bps=2.0,
        volatility=0.02,
    )
    
    # Get action
    action = agent.get_action(state, deterministic=True)
    print(f"Action: {action.name}")
    
    # Get order params
    params = agent.get_order_params(action, state)
    print(f"Order params: {params}")
    
    # Simulate execution and reward
    slippage = 1.5  # bps
    reward = agent.calculate_reward(slippage)
    print(f"Reward: {reward:.3f}")
    
    # Update
    next_state = ExecutionState(
        bid_prices=state.bid_prices,
        bid_sizes=state.bid_sizes,
        ask_prices=state.ask_prices,
        ask_sizes=state.ask_sizes,
        queue_position=0.2,
        has_order=True,
        order_price=params.get('price', 50000),
        order_side='buy',
        target_fill=1.0,
        current_fill=0.3,
        time_remaining_pct=0.7,
        mid_price=50001,
        spread_bps=2.1,
        volatility=0.02,
    )
    
    agent.update(state, action, reward, next_state)
    
    print(f"\nStats: {agent.get_stats()}")
