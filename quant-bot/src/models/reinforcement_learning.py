"""
Reinforcement Learning for Trading
==================================
RL agents that learn optimal trading policies:
- DQN (Deep Q-Network)
- Policy Gradient
- Actor-Critic
- Trading Environment (Gym-style)

Key concepts:
- State: Market features + position
- Action: Buy/Sell/Hold + size
- Reward: Risk-adjusted returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
import random
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================

@dataclass
class TradingState:
    """Current state of the trading environment."""
    step: int
    position: float  # -1 to 1 (short to long)
    entry_price: float
    cash: float
    portfolio_value: float
    features: np.ndarray
    price: float


@dataclass
class StepResult:
    """Result from environment step."""
    next_state: np.ndarray
    reward: float
    done: bool
    info: Dict


class TradingEnvironment:
    """
    OpenAI Gym-style trading environment.
    
    State: [features..., position, unrealized_pnl, time_in_position]
    Action: 
        - Discrete: 0=Hold, 1=Buy, 2=Sell
        - Continuous: Position target [-1, 1]
    Reward: Risk-adjusted P&L
    """
    
    def __init__(self, 
                 prices: np.ndarray,
                 features: np.ndarray,
                 initial_cash: float = 10000,
                 transaction_cost: float = 0.001,
                 max_position: float = 1.0,
                 reward_scaling: float = 1000,
                 use_sharpe_reward: bool = True):
        """
        Initialize environment.
        
        Args:
            prices: Price series
            features: Feature matrix (n_steps, n_features)
            initial_cash: Starting capital
            transaction_cost: Cost per trade (fraction)
            max_position: Maximum position size
            reward_scaling: Scale factor for reward
            use_sharpe_reward: Use differential Sharpe as reward
        """
        self.prices = prices
        self.features = features
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.use_sharpe_reward = use_sharpe_reward
        
        self.n_steps = len(prices)
        self.n_features = features.shape[1] if features.ndim > 1 else 1
        
        # State dimension: features + position info
        self.state_dim = self.n_features + 3  # features + position + unrealized_pnl + time_in_position
        
        # Action space
        self.action_dim = 3  # Hold, Buy, Sell
        
        # Reset environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.cash = self.initial_cash
        self.time_in_position = 0
        
        # Track for Sharpe calculation
        self.returns_history = []
        self.portfolio_values = [self.initial_cash]
        
        # Track trades
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        features = self.features[self.current_step]
        if features.ndim == 0:
            features = np.array([features])
        
        # Unrealized P&L
        current_price = self.prices[self.current_step]
        if self.position != 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0
        
        # Combine into state
        state = np.concatenate([
            features,
            [self.position, unrealized_pnl, self.time_in_position / 100]
        ])
        
        return state
    
    def _calculate_reward(self, prev_value: float, curr_value: float, 
                          action_cost: float) -> float:
        """
        Calculate reward.
        
        Options:
        1. Simple return
        2. Differential Sharpe ratio
        3. Risk-adjusted return
        """
        # Portfolio return
        ret = (curr_value - prev_value) / prev_value - action_cost
        
        if not self.use_sharpe_reward:
            return ret * self.reward_scaling
        
        # Differential Sharpe ratio (Moody & Saffell, 1998)
        self.returns_history.append(ret)
        
        if len(self.returns_history) < 2:
            return ret * self.reward_scaling
        
        # Online Sharpe approximation
        returns = np.array(self.returns_history[-100:])  # Rolling window
        mean_ret = returns.mean()
        std_ret = returns.std() + 1e-8
        
        # Reward is improvement in Sharpe
        reward = mean_ret / std_ret * self.reward_scaling
        
        return reward
    
    def step(self, action: int) -> StepResult:
        """
        Take action and advance environment.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
        
        Returns:
            StepResult with next_state, reward, done, info
        """
        current_price = self.prices[self.current_step]
        prev_portfolio_value = self.cash + self.position * self.initial_cash * current_price / self.prices[0]
        
        action_cost = 0.0
        
        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            # Close short if any
            if self.position < 0:
                action_cost += abs(self.position) * self.transaction_cost
                self.cash -= abs(self.position) * self.initial_cash * current_price / self.prices[0]
            
            # Open long
            self.position = self.max_position
            self.entry_price = current_price
            self.time_in_position = 0
            action_cost += self.max_position * self.transaction_cost
            
            self.trades.append({
                'step': self.current_step,
                'action': 'buy',
                'price': current_price,
                'position': self.position
            })
        
        elif action == 2 and self.position >= 0:  # Sell
            # Close long if any
            if self.position > 0:
                action_cost += abs(self.position) * self.transaction_cost
                self.cash += abs(self.position) * self.initial_cash * current_price / self.prices[0]
            
            # Open short
            self.position = -self.max_position
            self.entry_price = current_price
            self.time_in_position = 0
            action_cost += self.max_position * self.transaction_cost
            
            self.trades.append({
                'step': self.current_step,
                'action': 'sell',
                'price': current_price,
                'position': self.position
            })
        
        else:  # Hold
            if self.position != 0:
                self.time_in_position += 1
        
        # Advance time
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # New state
        if not done:
            new_price = self.prices[self.current_step]
            curr_portfolio_value = self.cash + self.position * self.initial_cash * new_price / self.prices[0]
        else:
            # Close position at end
            new_price = self.prices[self.current_step]
            self.cash += self.position * self.initial_cash * new_price / self.prices[0]
            self.position = 0
            curr_portfolio_value = self.cash
        
        self.portfolio_values.append(curr_portfolio_value)
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, curr_portfolio_value, action_cost)
        
        next_state = self._get_state() if not done else np.zeros(self.state_dim)
        
        info = {
            'portfolio_value': curr_portfolio_value,
            'position': self.position,
            'price': new_price,
            'total_return': (curr_portfolio_value - self.initial_cash) / self.initial_cash,
            'n_trades': len(self.trades)
        }
        
        return StepResult(next_state, reward, done, info)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for episode."""
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        total_return = (values[-1] - values[0]) / values[0]
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_dd = drawdown.max()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'n_trades': len(self.trades),
            'final_value': values[-1]
        }


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DQN AGENT
# =============================================================================

class DQNAgent:
    """
    Deep Q-Network Agent.
    
    Uses neural network to approximate Q(s, a).
    
    Key techniques:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update: int = 100):
        """
        Initialize DQN agent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.learning_rate = learning_rate
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks (simple MLP)
        self.q_network = self._build_network(hidden_dim)
        self.target_network = self._build_network(hidden_dim)
        self._copy_weights()
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.steps = 0
    
    def _build_network(self, hidden_dim: int) -> Dict:
        """Build Q-network weights."""
        scale = np.sqrt(2.0 / self.state_dim)
        return {
            'W1': np.random.randn(self.state_dim, hidden_dim) * scale,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, hidden_dim) * scale,
            'b2': np.zeros(hidden_dim),
            'W3': np.random.randn(hidden_dim, self.action_dim) * scale,
            'b3': np.zeros(self.action_dim)
        }
    
    def _copy_weights(self):
        """Copy Q-network weights to target network."""
        for key in self.q_network:
            self.target_network[key] = self.q_network[key].copy()
    
    def _forward(self, state: np.ndarray, network: Dict) -> np.ndarray:
        """Forward pass through network."""
        x = state
        
        # Hidden layer 1
        x = np.maximum(0, x @ network['W1'] + network['b1'])  # ReLU
        
        # Hidden layer 2
        x = np.maximum(0, x @ network['W2'] + network['b2'])
        
        # Output layer
        q_values = x @ network['W3'] + network['b3']
        
        return q_values
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        q_values = self._forward(state.reshape(1, -1), self.q_network)
        return np.argmax(q_values[0])
    
    def train_step(self) -> float:
        """
        Perform one training step.
        
        Returns loss.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Compute targets
        next_q_values = self._forward(next_states, self.target_network)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Current Q-values
        current_q = self._forward(states, self.q_network)
        
        # Compute loss and gradients (simplified)
        batch_indices = np.arange(self.batch_size)
        q_for_actions = current_q[batch_indices, actions]
        
        loss = np.mean((q_for_actions - targets) ** 2)
        
        # Gradient descent (simplified update)
        error = 2 * (q_for_actions - targets) / self.batch_size
        
        # Update weights (simplified gradient computation)
        # In practice, use automatic differentiation
        self._update_weights(states, actions, error)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self._copy_weights()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def _update_weights(self, states: np.ndarray, actions: np.ndarray, error: np.ndarray):
        """Simplified weight update."""
        # This is a simplified gradient update
        # In production, use PyTorch/TensorFlow autograd
        
        # Forward pass to get activations
        h1 = np.maximum(0, states @ self.q_network['W1'] + self.q_network['b1'])
        h2 = np.maximum(0, h1 @ self.q_network['W2'] + self.q_network['b2'])
        
        # Backward pass (simplified)
        d_out = np.zeros((self.batch_size, self.action_dim))
        d_out[np.arange(self.batch_size), actions] = error
        
        # Output layer gradients
        dW3 = h2.T @ d_out
        db3 = d_out.sum(axis=0)
        
        # Update output layer
        self.q_network['W3'] -= self.learning_rate * dW3
        self.q_network['b3'] -= self.learning_rate * db3
        
        # Hidden layer 2
        d_h2 = d_out @ self.q_network['W3'].T
        d_h2 = d_h2 * (h2 > 0)  # ReLU derivative
        
        dW2 = h1.T @ d_h2
        db2 = d_h2.sum(axis=0)
        
        self.q_network['W2'] -= self.learning_rate * dW2
        self.q_network['b2'] -= self.learning_rate * db2
        
        # Hidden layer 1
        d_h1 = d_h2 @ self.q_network['W2'].T
        d_h1 = d_h1 * (h1 > 0)
        
        dW1 = states.T @ d_h1
        db1 = d_h1.sum(axis=0)
        
        self.q_network['W1'] -= self.learning_rate * dW1
        self.q_network['b1'] -= self.learning_rate * db1


# =============================================================================
# POLICY GRADIENT AGENT
# =============================================================================

class PolicyGradientAgent:
    """
    REINFORCE Policy Gradient Agent.
    
    Directly learns policy π(a|s) without Q-function.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99):
        """Initialize policy gradient agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Policy network
        scale = np.sqrt(2.0 / state_dim)
        self.policy = {
            'W1': np.random.randn(state_dim, hidden_dim) * scale,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, action_dim) * scale,
            'b2': np.zeros(action_dim)
        }
        
        # Episode buffer
        self.states = []
        self.actions = []
        self.rewards = []
    
    def _forward(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities."""
        h = np.maximum(0, state @ self.policy['W1'] + self.policy['b1'])
        logits = h @ self.policy['W2'] + self.policy['b2']
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        return probs
    
    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy."""
        probs = self._forward(state)
        action = np.random.choice(self.action_dim, p=probs)
        return action
    
    def store_transition(self, state, action, reward):
        """Store transition for episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train_episode(self) -> float:
        """
        Train on collected episode.
        
        Returns average loss.
        """
        if len(self.states) == 0:
            return 0.0
        
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        
        # Normalize returns
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        total_loss = 0
        for state, action, G in zip(self.states, self.actions, returns):
            state = np.array(state).reshape(1, -1)
            
            # Forward pass
            h = np.maximum(0, state @ self.policy['W1'] + self.policy['b1'])
            logits = h @ self.policy['W2'] + self.policy['b2']
            
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            # Loss: -log(π(a|s)) * G
            log_prob = np.log(probs[0, action] + 1e-10)
            loss = -log_prob * G
            total_loss += loss
            
            # Gradient of softmax cross-entropy
            d_logits = probs.copy()
            d_logits[0, action] -= 1
            d_logits *= -G
            
            # Backprop
            dW2 = h.T @ d_logits
            db2 = d_logits.sum(axis=0)
            
            d_h = d_logits @ self.policy['W2'].T
            d_h = d_h * (h > 0)
            
            dW1 = state.T @ d_h
            db1 = d_h.sum(axis=0)
            
            # Update
            self.policy['W2'] -= self.learning_rate * dW2
            self.policy['b2'] -= self.learning_rate * db2
            self.policy['W1'] -= self.learning_rate * dW1
            self.policy['b1'] -= self.learning_rate * db1
        
        # Clear episode buffer
        self.states = []
        self.actions = []
        self.rewards = []
        
        return total_loss / len(returns)


# =============================================================================
# RL TRAINER
# =============================================================================

class RLTrainer:
    """
    Train RL agents on trading environments.
    """
    
    def __init__(self, env: TradingEnvironment, agent_type: str = 'dqn', **kwargs):
        """
        Initialize trainer.
        
        Args:
            env: Trading environment
            agent_type: 'dqn' or 'policy_gradient'
            **kwargs: Agent-specific parameters
        """
        self.env = env
        
        if agent_type == 'dqn':
            self.agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                **kwargs
            )
        else:
            self.agent = PolicyGradientAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                **kwargs
            )
        
        self.agent_type = agent_type
        self.training_history = []
    
    def train(self, n_episodes: int = 100, verbose: bool = True) -> List[Dict]:
        """
        Train agent for n_episodes.
        
        Returns list of episode metrics.
        """
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            losses = []
            
            while True:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Environment step
                result = self.env.step(action)
                
                # Store transition
                if self.agent_type == 'dqn':
                    self.agent.buffer.push(
                        state, action, result.reward, 
                        result.next_state, result.done
                    )
                    loss = self.agent.train_step()
                    losses.append(loss)
                else:
                    self.agent.store_transition(state, action, result.reward)
                
                episode_reward += result.reward
                state = result.next_state
                
                if result.done:
                    break
            
            # Train policy gradient at end of episode
            if self.agent_type == 'policy_gradient':
                loss = self.agent.train_episode()
                losses = [loss]
            
            # Get performance metrics
            metrics = self.env.get_performance_metrics()
            metrics['episode'] = episode
            metrics['episode_reward'] = episode_reward
            metrics['avg_loss'] = np.mean(losses) if losses else 0
            metrics['epsilon'] = self.agent.epsilon if hasattr(self.agent, 'epsilon') else 0
            
            self.training_history.append(metrics)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}: "
                      f"Return={metrics['total_return']:.2%}, "
                      f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                      f"Trades={metrics['n_trades']}")
        
        return self.training_history
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """
        Evaluate trained agent.
        
        Returns average performance metrics.
        """
        all_metrics = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            
            while True:
                action = self.agent.select_action(state, training=False)
                result = self.env.step(action)
                state = result.next_state
                
                if result.done:
                    break
            
            all_metrics.append(self.env.get_performance_metrics())
        
        # Average metrics
        return {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("REINFORCEMENT LEARNING FOR TRADING")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate sample market data
    n_steps = 1000
    
    # Price with trend and noise
    trend = np.cumsum(np.random.randn(n_steps) * 0.01 + 0.0001)
    noise = np.random.randn(n_steps) * 0.02
    prices = 100 * np.exp(trend + noise)
    
    # Features: returns, momentum, volatility
    returns = np.diff(prices, prepend=prices[0]) / prices
    momentum = pd.Series(prices).rolling(10).mean().fillna(prices[0]).values / prices - 1
    volatility = pd.Series(returns).rolling(10).std().fillna(0.01).values
    
    features = np.column_stack([returns, momentum, volatility])
    
    print(f"\n1. Environment Setup")
    print(f"   Steps: {n_steps}")
    print(f"   Features: {features.shape[1]}")
    print(f"   Price range: [{prices.min():.2f}, {prices.max():.2f}]")
    
    # Create environment
    env = TradingEnvironment(
        prices=prices,
        features=features,
        initial_cash=10000,
        transaction_cost=0.001,
        use_sharpe_reward=True
    )
    
    print(f"   State dim: {env.state_dim}")
    print(f"   Action dim: {env.action_dim}")
    
    # Train DQN agent
    print(f"\n2. Training DQN Agent (50 episodes)")
    print("-" * 50)
    
    trainer = RLTrainer(env, agent_type='dqn', 
                        hidden_dim=32,
                        learning_rate=0.001,
                        buffer_size=5000,
                        batch_size=32)
    
    history = trainer.train(n_episodes=50, verbose=True)
    
    # Evaluate
    print(f"\n3. Evaluation")
    print("-" * 50)
    
    eval_metrics = trainer.evaluate(n_episodes=5)
    
    print(f"   Average Return: {eval_metrics['total_return']:.2%}")
    print(f"   Average Sharpe: {eval_metrics['sharpe_ratio']:.2f}")
    print(f"   Average Max DD: {eval_metrics['max_drawdown']:.2%}")
    print(f"   Average Trades: {eval_metrics['n_trades']:.0f}")
    
    # Compare with buy-and-hold
    print(f"\n4. Comparison with Buy-and-Hold")
    print("-" * 50)
    
    bnh_return = (prices[-1] - prices[0]) / prices[0]
    bnh_returns = np.diff(prices) / prices[:-1]
    bnh_sharpe = bnh_returns.mean() / bnh_returns.std() * np.sqrt(252)
    
    print(f"   Buy-and-Hold Return: {bnh_return:.2%}")
    print(f"   Buy-and-Hold Sharpe: {bnh_sharpe:.2f}")
    print(f"   RL Agent Return:     {eval_metrics['total_return']:.2%}")
    print(f"   RL Agent Sharpe:     {eval_metrics['sharpe_ratio']:.2f}")
    
    print("\n" + "="*70)
    print("PRODUCTION NOTES")
    print("="*70)
    print("""
This is a NumPy-only educational implementation.
For production RL trading:

1. Use PyTorch/TensorFlow:
   - GPU acceleration
   - Automatic differentiation
   - Pre-built optimizers

2. Better Algorithms:
   - PPO (Proximal Policy Optimization)
   - SAC (Soft Actor-Critic)
   - TD3 (Twin Delayed DDPG)

3. Key Improvements:
   - Prioritized Experience Replay
   - Dueling Networks
   - Multi-step returns
   - Distributional RL

4. Reward Engineering:
   - Differential Sharpe ratio
   - Risk-adjusted returns
   - Drawdown penalties
   - Transaction cost awareness

5. State Design:
   - Include position info
   - Market regime features
   - Order book features
   - Multi-timeframe

6. Installation for production:
   pip install stable-baselines3
   pip install gym
""")
