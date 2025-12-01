"""
Strategy Base Class - Jesse-like lifecycle hooks and developer ergonomics

Provides a clean, intuitive API for strategy development:
- Lifecycle hooks: on_start, on_tick, on_candle, on_exit
- Data fetching: self.fetch_ohlc(), self.fetch_orderbook()
- Position management: self.buy(), self.sell(), self.close()
- State access: self.position, self.balance, self.equity
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import numpy as np


class StrategyState(Enum):
    """Strategy lifecycle states"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def ohlc(self) -> tuple:
        return (self.open, self.high, self.low, self.close)
    
    @property
    def hlc3(self) -> float:
        return (self.high + self.low + self.close) / 3
    
    @property
    def ohlc4(self) -> float:
        return (self.open + self.high + self.low + self.close) / 4
    
    @property
    def hl2(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class Tick:
    """Real-time tick data"""
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'


@dataclass 
class Position:
    """Current position state"""
    symbol: str
    side: str  # 'long', 'short', or 'flat'
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.size > 0
    
    @property
    def is_long(self) -> bool:
        return self.side == 'long' and self.size > 0
    
    @property
    def is_short(self) -> bool:
        return self.side == 'short' and self.size > 0


@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    side: str
    size: float
    order_type: str
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategyContext:
    """
    Environment context passed to strategy
    Provides access to market data, execution, and account state
    """
    symbol: str
    timeframe: str = "1m"
    
    # Data providers (injected by orchestrator)
    fetch_ohlc: Callable = None
    fetch_ticker: Callable = None
    fetch_orderbook: Callable = None
    fetch_trades: Callable = None
    
    # Execution providers
    place_order: Callable = None
    cancel_order: Callable = None
    get_position: Callable = None
    get_orders: Callable = None
    
    # Account providers
    get_balance: Callable = None
    get_equity: Callable = None
    
    # Signal providers (for multi-modal)
    get_sentiment: Callable = None
    get_news_score: Callable = None
    
    # Risk providers
    check_can_trade: Callable = None
    compute_size: Callable = None


class StrategyBase(ABC):
    """
    Base class for all trading strategies
    
    Jesse-like lifecycle:
    1. __init__() - Strategy instantiation
    2. on_start() - Called once when strategy starts
    3. on_candle() - Called on each new candle
    4. on_tick() - Called on each tick (optional, for scalping)
    5. on_exit() - Called when strategy stops
    
    Usage:
        class MyStrategy(StrategyBase):
            def on_candle(self, candle: Candle):
                if self.should_buy():
                    self.buy(size=0.1)
    """
    
    # Strategy metadata (override in subclass)
    name: str = "base_strategy"
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    
    # Supported timeframes
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    def __init__(self, ctx: StrategyContext):
        """
        Initialize strategy with context
        
        Args:
            ctx: StrategyContext providing data and execution access
        """
        self.ctx = ctx
        self.symbol = ctx.symbol
        self.timeframe = ctx.timeframe
        
        # Internal state
        self._state = StrategyState.INITIALIZED
        self._candles: List[Candle] = []
        self._position: Optional[Position] = None
        self._open_orders: List[Order] = []
        self._vars: Dict[str, Any] = {}  # User-defined variables
        
        # Performance tracking
        self._trade_count = 0
        self._win_count = 0
        self._total_pnl = 0.0
    
    # ==================== Lifecycle Hooks ====================
    
    def on_start(self):
        """
        Called once when strategy starts
        Override to initialize indicators, state, etc.
        """
        pass
    
    @abstractmethod
    def on_candle(self, candle: Candle):
        """
        Called on each new candle
        Main strategy logic goes here
        
        Args:
            candle: The new OHLCV candle
        """
        pass
    
    def on_tick(self, tick: Tick):
        """
        Called on each tick (optional)
        Override for high-frequency strategies
        
        Args:
            tick: Real-time tick data
        """
        pass
    
    def on_order_filled(self, order: Order):
        """
        Called when an order is filled
        Override to handle fill events
        
        Args:
            order: The filled order
        """
        pass
    
    def on_position_opened(self, position: Position):
        """
        Called when a new position is opened
        """
        pass
    
    def on_position_closed(self, position: Position, pnl: float):
        """
        Called when a position is closed
        
        Args:
            position: The closed position
            pnl: Realized PnL
        """
        pass
    
    def on_exit(self):
        """
        Called when strategy stops
        Override for cleanup
        """
        pass
    
    # ==================== Data Access ====================
    
    @property
    def candles(self) -> List[Candle]:
        """Get all candles"""
        return self._candles
    
    @property
    def close(self) -> np.ndarray:
        """Get close prices as numpy array"""
        return np.array([c.close for c in self._candles])
    
    @property
    def open(self) -> np.ndarray:
        """Get open prices as numpy array"""
        return np.array([c.open for c in self._candles])
    
    @property
    def high(self) -> np.ndarray:
        """Get high prices as numpy array"""
        return np.array([c.high for c in self._candles])
    
    @property
    def low(self) -> np.ndarray:
        """Get low prices as numpy array"""
        return np.array([c.low for c in self._candles])
    
    @property
    def volume(self) -> np.ndarray:
        """Get volumes as numpy array"""
        return np.array([c.volume for c in self._candles])
    
    @property
    def current_candle(self) -> Optional[Candle]:
        """Get the most recent candle"""
        return self._candles[-1] if self._candles else None
    
    @property
    def price(self) -> float:
        """Get current price (last close)"""
        return self._candles[-1].close if self._candles else 0.0
    
    def fetch_ohlc(
        self, 
        symbol: str = None, 
        timeframe: str = None,
        limit: int = 100
    ) -> List[Candle]:
        """
        Fetch OHLCV data
        
        Args:
            symbol: Trading pair (default: strategy symbol)
            timeframe: Timeframe (default: strategy timeframe)
            limit: Number of candles
        
        Returns:
            List of Candle objects
        """
        if self.ctx.fetch_ohlc:
            return self.ctx.fetch_ohlc(
                symbol or self.symbol,
                timeframe or self.timeframe,
                limit
            )
        return []
    
    def fetch_orderbook(self, symbol: str = None, depth: int = 10) -> Dict:
        """Fetch orderbook"""
        if self.ctx.fetch_orderbook:
            return self.ctx.fetch_orderbook(symbol or self.symbol, depth)
        return {"bids": [], "asks": []}
    
    def fetch_ticker(self, symbol: str = None) -> Dict:
        """Fetch ticker"""
        if self.ctx.fetch_ticker:
            return self.ctx.fetch_ticker(symbol or self.symbol)
        return {}
    
    # ==================== Position & Account ====================
    
    @property
    def position(self) -> Position:
        """Get current position"""
        if self._position is None:
            self._position = Position(symbol=self.symbol, side='flat')
        return self._position
    
    @property
    def is_long(self) -> bool:
        """Check if currently long"""
        return self.position.is_long
    
    @property
    def is_short(self) -> bool:
        """Check if currently short"""
        return self.position.is_short
    
    @property
    def is_flat(self) -> bool:
        """Check if no position"""
        return not self.position.is_open
    
    @property
    def balance(self) -> float:
        """Get available balance"""
        if self.ctx.get_balance:
            return self.ctx.get_balance()
        return 0.0
    
    @property
    def equity(self) -> float:
        """Get total equity"""
        if self.ctx.get_equity:
            return self.ctx.get_equity()
        return self.balance
    
    # ==================== Order Execution ====================
    
    def buy(
        self,
        size: float = None,
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[Order]:
        """
        Place a buy order
        
        Args:
            size: Position size (computed if None)
            price: Limit price (market if None)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Order object or None
        """
        # Compute size if not provided
        if size is None and self.ctx.compute_size:
            size = self.ctx.compute_size(
                entry_price=price or self.price,
                stop_price=stop_loss
            )
        
        if not size or size <= 0:
            return None
        
        # Check if can trade
        if self.ctx.check_can_trade:
            result = self.ctx.check_can_trade()
            if not result.get('allowed', True):
                return None
        
        # Place order
        if self.ctx.place_order:
            order = self.ctx.place_order(
                symbol=self.symbol,
                side='buy',
                size=size,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            if order:
                self._open_orders.append(order)
            return order
        
        return None
    
    def sell(
        self,
        size: float = None,
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[Order]:
        """
        Place a sell order
        
        Args:
            size: Position size (computed if None)
            price: Limit price (market if None)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Order object or None
        """
        if size is None and self.ctx.compute_size:
            size = self.ctx.compute_size(
                entry_price=price or self.price,
                stop_price=stop_loss
            )
        
        if not size or size <= 0:
            return None
        
        if self.ctx.check_can_trade:
            result = self.ctx.check_can_trade()
            if not result.get('allowed', True):
                return None
        
        if self.ctx.place_order:
            order = self.ctx.place_order(
                symbol=self.symbol,
                side='sell',
                size=size,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            if order:
                self._open_orders.append(order)
            return order
        
        return None
    
    def close(self, price: float = None) -> Optional[Order]:
        """
        Close current position
        
        Args:
            price: Limit price (market if None)
        
        Returns:
            Order object or None
        """
        if not self.position.is_open:
            return None
        
        if self.position.is_long:
            return self.sell(size=self.position.size, price=price)
        else:
            return self.buy(size=self.position.size, price=price)
    
    def cancel_all(self):
        """Cancel all open orders"""
        if self.ctx.cancel_order:
            for order in self._open_orders:
                if order.status == 'pending':
                    self.ctx.cancel_order(order.id)
        self._open_orders = []
    
    # ==================== Signals & Indicators ====================
    
    def get_sentiment(self) -> float:
        """Get current sentiment score (-1 to 1)"""
        if self.ctx.get_sentiment:
            return self.ctx.get_sentiment()
        return 0.0
    
    def get_news_score(self) -> float:
        """Get news impact score (-1 to 1)"""
        if self.ctx.get_news_score:
            return self.ctx.get_news_score()
        return 0.0
    
    # ==================== State Management ====================
    
    @property
    def state(self) -> StrategyState:
        """Get strategy state"""
        return self._state
    
    @property
    def vars(self) -> Dict[str, Any]:
        """Get user-defined variables"""
        return self._vars
    
    def set_var(self, key: str, value: Any):
        """Set a user-defined variable"""
        self._vars[key] = value
    
    def get_var(self, key: str, default: Any = None) -> Any:
        """Get a user-defined variable"""
        return self._vars.get(key, default)
    
    # ==================== Internal Methods ====================
    
    def _add_candle(self, candle: Candle):
        """Add a candle to history"""
        self._candles.append(candle)
    
    def _update_position(self, position: Position):
        """Update position state"""
        self._position = position
    
    def _set_state(self, state: StrategyState):
        """Set strategy state"""
        self._state = state
    
    # ==================== Performance ====================
    
    @property
    def trade_count(self) -> int:
        return self._trade_count
    
    @property
    def win_rate(self) -> float:
        if self._trade_count == 0:
            return 0.0
        return self._win_count / self._trade_count
    
    @property
    def total_pnl(self) -> float:
        return self._total_pnl
    
    def _record_trade(self, pnl: float):
        """Record a completed trade"""
        self._trade_count += 1
        self._total_pnl += pnl
        if pnl > 0:
            self._win_count += 1
    
    # ==================== Utility ====================
    
    def log(self, message: str, **kwargs):
        """Log a message (override for custom logging)"""
        print(f"[{self.name}] {message}", kwargs if kwargs else "")
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(symbol={self.symbol}, state={self._state.value})>"
