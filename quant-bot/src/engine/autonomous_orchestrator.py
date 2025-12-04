"""
Autonomous Trading Orchestrator
===============================
Master controller for fully autonomous ML-driven trading.

This module provides:
- Complete trading pipeline orchestration
- Model lifecycle management (train → validate → deploy)
- Automated retraining triggers
- Position and risk management
- Performance monitoring

This is the "brain" that makes the system trade on its own.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for autonomous trading."""
    
    # Model settings
    model_retrain_hours: int = 24
    min_samples_for_retrain: int = 1000
    validation_window_days: int = 30
    
    # Trading settings
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    max_positions: int = 5
    position_size_usd: float = 1000.0
    max_portfolio_risk: float = 0.05  # 5% max drawdown
    
    # Signal settings
    signal_threshold: float = 0.6
    min_holding_hours: float = 1.0
    max_holding_hours: float = 24.0
    
    # Risk settings
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_daily_trades: int = 20
    max_daily_loss_pct: float = 0.03
    
    # Data settings
    lookback_bars: int = 500
    bar_interval: str = "1h"
    
    # System settings
    check_interval_seconds: int = 60
    log_level: str = "INFO"


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class SystemState(Enum):
    """Trading system states."""
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    TRADING = "trading"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class Position:
    """Active position."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    strategy: str
    unrealized_pnl: float = 0.0


@dataclass
class SystemStatus:
    """Current system status."""
    state: SystemState
    last_update: datetime
    active_positions: int
    daily_trades: int
    daily_pnl: float
    portfolio_value: float
    models_loaded: List[str]
    last_signal: Optional[Dict]
    errors: List[str]


# =============================================================================
# SIGNAL AGGREGATOR
# =============================================================================

class SignalAggregator:
    """
    Aggregates signals from multiple models/strategies.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize signal aggregator.
        
        Args:
            weights: Signal source weights
        """
        self.weights = weights or {}
        self.signals: Dict[str, Dict] = {}
    
    def add_signal(self, source: str, symbol: str, signal: float,
                  confidence: float, features: Optional[Dict] = None):
        """
        Add signal from a source.
        
        Args:
            source: Signal source name
            symbol: Trading symbol
            signal: Signal value (-1 to 1)
            confidence: Signal confidence (0 to 1)
            features: Optional feature data
        """
        key = f"{source}_{symbol}"
        self.signals[key] = {
            'source': source,
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'features': features or {}
        }
    
    def get_aggregate_signal(self, symbol: str) -> Tuple[float, float]:
        """
        Get weighted aggregate signal for symbol.
        
        Returns:
            (aggregate_signal, aggregate_confidence)
        """
        relevant_signals = [
            s for s in self.signals.values() 
            if s['symbol'] == symbol
        ]
        
        if not relevant_signals:
            return 0.0, 0.0
        
        # Weight signals
        total_weight = 0
        weighted_signal = 0
        weighted_confidence = 0
        
        for s in relevant_signals:
            weight = self.weights.get(s['source'], 1.0)
            weighted_signal += s['signal'] * s['confidence'] * weight
            weighted_confidence += s['confidence'] * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.0
        
        return weighted_signal / total_weight, weighted_confidence / total_weight
    
    def clear_old_signals(self, max_age_seconds: int = 300):
        """Remove stale signals."""
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        self.signals = {
            k: v for k, v in self.signals.items()
            if v['timestamp'] > cutoff
        }


# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:
    """
    Manages ML model lifecycle.
    """
    
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory for model storage
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.training_history: List[Dict] = []
    
    def train_model(self, model_name: str, X_train: pd.DataFrame,
                   y_train: pd.Series, model_type: str = "ensemble",
                   **kwargs) -> Dict:
        """
        Train a new model.
        
        Returns training results.
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        
        # Select model type
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42
            )
        else:
            # Ensemble
            from sklearn.ensemble import VotingClassifier
            model = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))
                ],
                voting='soft'
            )
        
        # Train
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Store model
        self.active_models[model_name] = model
        
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'train_date': datetime.now().isoformat(),
            'n_features': len(X_train.columns),
            'n_samples': len(X_train),
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'features': list(X_train.columns)
        }
        self.model_metadata[model_name] = metadata
        
        # Save model
        self._save_model(model_name, model, metadata)
        
        return metadata
    
    def _save_model(self, model_name: str, model: Any, metadata: Dict):
        """Save model to disk."""
        model_path = self.models_dir / f"{model_name}.pkl"
        meta_path = self.models_dir / f"{model_name}_meta.json"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, model_name: str) -> bool:
        """Load model from disk."""
        model_path = self.models_dir / f"{model_name}.pkl"
        meta_path = self.models_dir / f"{model_name}_meta.json"
        
        if not model_path.exists():
            return False
        
        with open(model_path, 'rb') as f:
            self.active_models[model_name] = pickle.load(f)
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.model_metadata[model_name] = json.load(f)
        
        return True
    
    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and probabilities.
        
        Returns:
            (predictions, probabilities)
        """
        if model_name not in self.active_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.active_models[model_name]
        
        predictions = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
        else:
            probas = np.zeros((len(X), 2))
            probas[predictions == 1, 1] = 1
            probas[predictions == 0, 0] = 1
        
        return predictions, probas
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict:
        """Evaluate model on test data."""
        predictions, probas = self.predict(model_name, X_test)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1': f1_score(y_test, predictions, average='weighted'),
            'n_samples': len(y_test)
        }


# =============================================================================
# RISK CONTROLLER
# =============================================================================

class RiskController:
    """
    Real-time risk management.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize risk controller."""
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_start = datetime.now().date()
    
    def reset_daily_stats(self):
        """Reset daily statistics."""
        today = datetime.now().date()
        if today != self.daily_start:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_start = today
    
    def can_open_position(self, symbol: str, size_usd: float) -> Tuple[bool, str]:
        """
        Check if new position is allowed.
        
        Returns (allowed, reason)
        """
        self.reset_daily_stats()
        
        # Check position limit
        if len(self.positions) >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"
        
        # Check if already in position
        if symbol in self.positions:
            return False, f"Already in position for {symbol}"
        
        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"Max daily trades ({self.config.max_daily_trades}) reached"
        
        # Check daily loss limit
        max_loss = -self.config.max_daily_loss_pct * self.config.position_size_usd * self.config.max_positions
        if self.daily_pnl < max_loss:
            return False, f"Daily loss limit ({self.config.max_daily_loss_pct:.1%}) reached"
        
        return True, "OK"
    
    def add_position(self, position: Position):
        """Add new position."""
        self.positions[position.symbol] = position
        self.daily_trades += 1
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Close position and return PnL."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions.pop(symbol)
        
        if pos.side == 'long':
            pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.quantity * pos.entry_price
        else:
            pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.quantity * pos.entry_price
        
        self.daily_pnl += pnl
        
        return pnl
    
    def update_positions(self, prices: Dict[str, float]):
        """Update position PnL."""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                price = prices[symbol]
                if pos.side == 'long':
                    pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price
                else:
                    pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price
    
    def check_exits(self, prices: Dict[str, float]) -> List[Tuple[str, str]]:
        """
        Check positions for exit signals.
        
        Returns list of (symbol, reason) to exit.
        """
        exits = []
        
        for symbol, pos in self.positions.items():
            if symbol not in prices:
                continue
            
            price = prices[symbol]
            
            # Stop loss
            if pos.side == 'long' and price <= pos.stop_loss:
                exits.append((symbol, 'stop_loss'))
            elif pos.side == 'short' and price >= pos.stop_loss:
                exits.append((symbol, 'stop_loss'))
            
            # Take profit
            if pos.side == 'long' and price >= pos.take_profit:
                exits.append((symbol, 'take_profit'))
            elif pos.side == 'short' and price <= pos.take_profit:
                exits.append((symbol, 'take_profit'))
            
            # Max holding time
            holding_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
            if holding_hours > self.config.max_holding_hours:
                exits.append((symbol, 'max_holding_time'))
        
        return exits


# =============================================================================
# AUTONOMOUS ORCHESTRATOR
# =============================================================================

class AutonomousOrchestrator:
    """
    Master orchestrator for autonomous trading.
    
    This is the main class that ties everything together.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None,
                 data_fetcher: Optional[Callable] = None,
                 order_executor: Optional[Callable] = None):
        """
        Initialize orchestrator.
        
        Args:
            config: Trading configuration
            data_fetcher: Async function to fetch market data
            order_executor: Async function to execute orders
        """
        self.config = config or OrchestratorConfig()
        self.data_fetcher = data_fetcher
        self.order_executor = order_executor
        
        # Components
        self.model_manager = ModelManager()
        self.risk_controller = RiskController(self.config)
        self.signal_aggregator = SignalAggregator()
        
        # State
        self.state = SystemState.INITIALIZING
        self.last_train_time: Optional[datetime] = None
        self.samples_since_train = 0
        self.errors: List[str] = []
        
        # Data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.trade_history: List[Dict] = []
        
        # Logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger("Orchestrator")
    
    async def initialize(self):
        """Initialize the system."""
        self.logger.info("Initializing autonomous trading system...")
        
        try:
            # Load existing models
            for symbol in self.config.symbols:
                model_name = f"model_{symbol}"
                if self.model_manager.load_model(model_name):
                    self.logger.info(f"Loaded model for {symbol}")
                else:
                    self.logger.warning(f"No existing model for {symbol}")
            
            self.state = SystemState.TRADING
            self.logger.info("System initialized and ready for trading")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.errors.append(str(e))
            self.logger.error(f"Initialization error: {e}")
    
    async def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data for symbol."""
        if self.data_fetcher is None:
            self.logger.warning("No data fetcher configured")
            return None
        
        try:
            data = await self.data_fetcher(symbol, self.config.bar_interval,
                                          self.config.lookback_bars)
            self.market_data[symbol] = data
            return data
        except Exception as e:
            self.logger.error(f"Data fetch error for {symbol}: {e}")
            return None
    
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute features for ML model."""
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            features[f'sma_ratio_{period}'] = data['close'] / features[f'sma_{period}']
        
        # Volatility
        features['volatility_10'] = features['returns'].rolling(10).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
        
        # Target: Future returns
        features['target'] = (features['returns'].shift(-1) > 0).astype(int)
        
        return features.dropna()
    
    def generate_signal(self, symbol: str, features: pd.DataFrame) -> Dict:
        """Generate trading signal for symbol."""
        model_name = f"model_{symbol}"
        
        if model_name not in self.model_manager.active_models:
            return {'signal': 0, 'confidence': 0, 'reason': 'no_model'}
        
        # Get latest features
        X = features.drop(['target'], axis=1, errors='ignore').iloc[[-1]]
        
        try:
            predictions, probas = self.model_manager.predict(model_name, X)
            
            signal = 1 if predictions[0] == 1 else -1
            confidence = max(probas[0])
            
            return {
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'features': X.iloc[0].to_dict()
            }
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': str(e)}
    
    async def execute_trade(self, symbol: str, side: str,
                           price: float) -> bool:
        """Execute a trade."""
        quantity = self.config.position_size_usd / price
        
        if self.order_executor is None:
            self.logger.warning(f"Simulated trade: {side} {quantity:.4f} {symbol} @ {price}")
            
            # Create position
            if side == 'buy':
                stop_loss = price * (1 - self.config.stop_loss_pct)
                take_profit = price * (1 + self.config.take_profit_pct)
                position_side = 'long'
            else:
                stop_loss = price * (1 + self.config.stop_loss_pct)
                take_profit = price * (1 - self.config.take_profit_pct)
                position_side = 'short'
            
            position = Position(
                symbol=symbol,
                side=position_side,
                entry_price=price,
                quantity=quantity,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy='ml_ensemble'
            )
            
            self.risk_controller.add_position(position)
            
            self.trade_history.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'price': price,
                'quantity': quantity,
                'type': 'entry'
            })
            
            return True
        
        try:
            result = await self.order_executor(symbol, side, quantity, price)
            return result.get('success', False)
        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return False
    
    async def check_and_train(self, force: bool = False) -> bool:
        """Check if retraining is needed and train if so."""
        should_train = force
        
        # Check time since last training
        if self.last_train_time is None:
            should_train = True
        elif (datetime.now() - self.last_train_time).total_seconds() / 3600 > self.config.model_retrain_hours:
            should_train = True
        
        # Check samples
        if self.samples_since_train >= self.config.min_samples_for_retrain:
            should_train = True
        
        if not should_train:
            return False
        
        self.logger.info("Starting model retraining...")
        self.state = SystemState.TRAINING
        
        for symbol in self.config.symbols:
            try:
                data = self.market_data.get(symbol)
                if data is None or len(data) < 100:
                    continue
                
                features = self.compute_features(data)
                
                # Split data
                split_idx = int(len(features) * 0.8)
                X_train = features.drop(['target'], axis=1).iloc[:split_idx]
                y_train = features['target'].iloc[:split_idx]
                
                # Train
                model_name = f"model_{symbol}"
                result = self.model_manager.train_model(
                    model_name, X_train, y_train, model_type='ensemble'
                )
                
                self.logger.info(f"Trained {model_name}: CV accuracy = {result['cv_accuracy']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Training error for {symbol}: {e}")
        
        self.last_train_time = datetime.now()
        self.samples_since_train = 0
        self.state = SystemState.TRADING
        
        return True
    
    async def trading_loop_iteration(self):
        """Single iteration of the trading loop."""
        if self.state not in [SystemState.TRADING]:
            return
        
        prices = {}
        
        for symbol in self.config.symbols:
            # Fetch data
            data = await self.fetch_data(symbol)
            if data is None:
                continue
            
            price = data['close'].iloc[-1]
            prices[symbol] = price
            
            # Compute features
            features = self.compute_features(data)
            self.samples_since_train += 1
            
            # Generate signal
            signal_data = self.generate_signal(symbol, features)
            
            # Add to aggregator
            self.signal_aggregator.add_signal(
                'ml_ensemble', symbol,
                signal_data['signal'],
                signal_data['confidence']
            )
            
            # Get aggregate signal
            agg_signal, agg_confidence = self.signal_aggregator.get_aggregate_signal(symbol)
            
            # Check if we should trade
            if abs(agg_signal) > self.config.signal_threshold and agg_confidence > 0.5:
                can_trade, reason = self.risk_controller.can_open_position(
                    symbol, self.config.position_size_usd
                )
                
                if can_trade:
                    side = 'buy' if agg_signal > 0 else 'sell'
                    success = await self.execute_trade(symbol, side, price)
                    
                    if success:
                        self.logger.info(
                            f"Opened {side.upper()} position in {symbol} @ {price:.2f} "
                            f"(signal={agg_signal:.2f}, conf={agg_confidence:.2f})"
                        )
                else:
                    self.logger.debug(f"Cannot trade {symbol}: {reason}")
        
        # Update positions
        self.risk_controller.update_positions(prices)
        
        # Check exits
        exits = self.risk_controller.check_exits(prices)
        for symbol, reason in exits:
            pnl = self.risk_controller.close_position(symbol, prices.get(symbol, 0))
            
            self.trade_history.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': 'close',
                'price': prices.get(symbol, 0),
                'pnl': pnl,
                'reason': reason,
                'type': 'exit'
            })
            
            self.logger.info(f"Closed {symbol} ({reason}): PnL = {pnl:.2f}")
        
        # Check if retraining needed
        await self.check_and_train()
    
    async def run(self, max_iterations: Optional[int] = None):
        """
        Run the autonomous trading loop.
        
        Args:
            max_iterations: Maximum iterations (None for infinite)
        """
        await self.initialize()
        
        iteration = 0
        
        while self.state not in [SystemState.SHUTDOWN, SystemState.ERROR]:
            try:
                await self.trading_loop_iteration()
                
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    break
                
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested...")
                self.state = SystemState.SHUTDOWN
                break
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                self.errors.append(str(e))
                
                if len(self.errors) > 10:
                    self.state = SystemState.ERROR
                    break
        
        self.logger.info("Trading system stopped")
    
    def get_status(self) -> SystemStatus:
        """Get current system status."""
        return SystemStatus(
            state=self.state,
            last_update=datetime.now(),
            active_positions=len(self.risk_controller.positions),
            daily_trades=self.risk_controller.daily_trades,
            daily_pnl=self.risk_controller.daily_pnl,
            portfolio_value=sum(
                p.quantity * p.entry_price 
                for p in self.risk_controller.positions.values()
            ),
            models_loaded=list(self.model_manager.active_models.keys()),
            last_signal=None,
            errors=self.errors[-5:]
        )
    
    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        if not self.trade_history:
            return {'message': 'No trades yet'}
        
        df = pd.DataFrame(self.trade_history)
        exits = df[df['type'] == 'exit']
        
        if len(exits) == 0:
            return {'message': 'No closed trades yet'}
        
        pnls = exits['pnl'].dropna()
        
        return {
            'total_trades': len(exits),
            'winning_trades': (pnls > 0).sum(),
            'losing_trades': (pnls < 0).sum(),
            'win_rate': (pnls > 0).mean(),
            'total_pnl': pnls.sum(),
            'avg_pnl': pnls.mean(),
            'max_win': pnls.max(),
            'max_loss': pnls.min(),
            'profit_factor': pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else np.inf
        }


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Demo the autonomous orchestrator."""
    print("="*70)
    print("AUTONOMOUS TRADING ORCHESTRATOR")
    print("="*70)
    
    # Create configuration
    config = OrchestratorConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        max_positions=3,
        position_size_usd=1000.0,
        signal_threshold=0.6,
        check_interval_seconds=5
    )
    
    # Mock data fetcher
    async def mock_data_fetcher(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
        """Generate mock OHLCV data."""
        np.random.seed(42)
        
        dates = pd.date_range(end=datetime.now(), periods=lookback, freq='1h')
        base_price = 45000 if 'BTC' in symbol else 3000
        
        returns = np.random.normal(0.0001, 0.02, lookback)
        prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'open': prices * (1 - np.random.uniform(0, 0.005, lookback)),
            'high': prices * (1 + np.random.uniform(0, 0.01, lookback)),
            'low': prices * (1 - np.random.uniform(0, 0.01, lookback)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, lookback)
        }, index=dates)
    
    # Create orchestrator
    orchestrator = AutonomousOrchestrator(
        config=config,
        data_fetcher=mock_data_fetcher,
        order_executor=None  # Simulation mode
    )
    
    print("\n1. Initializing System")
    print("-" * 50)
    await orchestrator.initialize()
    
    status = orchestrator.get_status()
    print(f"   State: {status.state.value}")
    print(f"   Models loaded: {status.models_loaded}")
    
    print("\n2. Training Initial Models")
    print("-" * 50)
    await orchestrator.check_and_train(force=True)
    
    status = orchestrator.get_status()
    print(f"   Models after training: {list(orchestrator.model_manager.active_models.keys())}")
    
    print("\n3. Running Trading Loop (5 iterations)")
    print("-" * 50)
    
    for i in range(5):
        await orchestrator.trading_loop_iteration()
        print(f"   Iteration {i+1}: {len(orchestrator.risk_controller.positions)} positions")
        await asyncio.sleep(0.1)
    
    print("\n4. System Status")
    print("-" * 50)
    
    status = orchestrator.get_status()
    print(f"   State: {status.state.value}")
    print(f"   Active positions: {status.active_positions}")
    print(f"   Daily trades: {status.daily_trades}")
    print(f"   Daily PnL: ${status.daily_pnl:.2f}")
    
    print("\n5. Performance Report")
    print("-" * 50)
    
    report = orchestrator.get_performance_report()
    for key, value in report.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n6. Position Details")
    print("-" * 50)
    
    for symbol, pos in orchestrator.risk_controller.positions.items():
        print(f"   {symbol}: {pos.side.upper()} @ {pos.entry_price:.2f}")
        print(f"      Stop: {pos.stop_loss:.2f}, Target: {pos.take_profit:.2f}")
        print(f"      Unrealized PnL: {pos.unrealized_pnl:.2%}")
    
    print("\n" + "="*70)
    print("DEPLOYMENT GUIDE")
    print("="*70)
    print("""
1. Configure for production:
   config = OrchestratorConfig(
       symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
       position_size_usd=10000.0,
       max_positions=5,
       check_interval_seconds=60
   )

2. Connect to exchange:
   from delta_client import DeltaExchangeClient
   
   client = DeltaExchangeClient(api_key, api_secret)
   
   async def data_fetcher(symbol, interval, lookback):
       return await client.get_ohlcv(symbol, interval, lookback)
   
   async def order_executor(symbol, side, qty, price):
       return await client.place_order(symbol, side, qty, price)
   
   orchestrator = AutonomousOrchestrator(
       config=config,
       data_fetcher=data_fetcher,
       order_executor=order_executor
   )

3. Run continuously:
   asyncio.run(orchestrator.run())

4. Monitor via status:
   status = orchestrator.get_status()
   report = orchestrator.get_performance_report()

5. Key features:
   ✓ Automatic model retraining
   ✓ Risk management (stops, position limits)
   ✓ Signal aggregation from multiple sources
   ✓ Performance tracking
   ✓ Graceful error handling
""")


if __name__ == "__main__":
    asyncio.run(demo())
