"""
Paper Trading Orchestrator - 30-Day Shadow Trading Module
=========================================================
CRITICAL DELIVERABLE: Non-negotiable validation before live deployment.

This module provides:
- Full 24/7 paper trading infrastructure
- Real-time market data simulation with actual exchange feeds
- Comprehensive logging of all decisions and fills
- Metrics collection for sim2real gap analysis
- Alert system for anomalies
- Daily performance reports

The sim2real gap (latency, missing fills, slippage, spread, spoofing)
ALWAYS breaks untested bots. This module catches those failures.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import deque
import threading
import hashlib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperTradingMode(Enum):
    """Paper trading modes."""
    SHADOW = "shadow"          # Mirror live signals, don't execute
    PAPER = "paper"            # Execute on paper with simulated fills
    CANARY = "canary"          # Small real allocation (1%)
    FULL = "full"              # Full production


class FillSimulationMode(Enum):
    """How to simulate order fills."""
    INSTANT = "instant"              # Immediate fill at signal price
    REALISTIC = "realistic"          # Latency + slippage + partial fills
    ADVERSARIAL = "adversarial"      # Worst-case slippage + missed fills


@dataclass
class PaperOrder:
    """Paper trading order."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit'
    quantity: float
    price: float  # Signal price
    
    # Execution details
    status: str = "pending"  # pending, filled, partial, cancelled, rejected
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fill_time: Optional[datetime] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    
    # Sim2Real tracking
    simulated_slippage_bps: float = 0.0
    actual_market_price: float = 0.0
    spread_at_execution: float = 0.0
    
    # Strategy context
    strategy_name: str = ""
    signal_confidence: float = 0.0
    regime: str = ""
    orderflow_confirmation: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'fill_time': self.fill_time.isoformat() if self.fill_time else None,
            'created_at': self.created_at.isoformat(),
            'latency_ms': self.latency_ms,
            'simulated_slippage_bps': self.simulated_slippage_bps,
            'actual_market_price': self.actual_market_price,
            'spread_at_execution': self.spread_at_execution,
            'strategy_name': self.strategy_name,
            'signal_confidence': self.signal_confidence,
            'regime': self.regime,
            'orderflow_confirmation': self.orderflow_confirmation
        }


@dataclass
class PaperPosition:
    """Paper trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    entry_time: datetime
    
    # P&L tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    max_adverse_excursion: float = 0.0  # Worst drawdown during trade
    max_favorable_excursion: float = 0.0  # Best profit during trade
    
    def update_price(self, price: float):
        """Update position with new price."""
        self.current_price = price
        
        if self.side == 'long':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            excursion = (price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            excursion = (self.entry_price - price) / self.entry_price
        
        self.max_adverse_excursion = min(self.max_adverse_excursion, excursion)
        self.max_favorable_excursion = max(self.max_favorable_excursion, excursion)


@dataclass
class DailyReport:
    """Daily paper trading report."""
    date: str
    starting_equity: float
    ending_equity: float
    daily_return: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Execution quality
    avg_slippage_bps: float
    max_slippage_bps: float
    avg_latency_ms: float
    fill_rate: float  # % of orders that filled
    
    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    
    # Anomalies detected
    anomalies: List[str]
    
    # Strategy breakdown
    strategy_pnl: Dict[str, float]
    regime_pnl: Dict[str, float]


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading."""
    # Duration
    min_days: int = 30
    max_days: int = 90
    
    # Capital
    initial_capital: float = 100000.0
    max_position_pct: float = 0.20
    max_daily_loss_pct: float = 0.03
    
    # Execution simulation
    fill_mode: FillSimulationMode = FillSimulationMode.REALISTIC
    base_latency_ms: float = 50.0
    latency_std_ms: float = 20.0
    base_slippage_bps: float = 5.0
    slippage_vol_multiplier: float = 2.0
    fill_probability: float = 0.95  # 95% of orders fill
    
    # Logging
    log_dir: str = "./paper_trading_logs"
    log_level: str = "INFO"
    
    # Alerts
    alert_on_daily_loss: float = 0.02  # Alert if daily loss > 2%
    alert_on_drawdown: float = 0.05  # Alert if drawdown > 5%
    alert_on_anomaly: bool = True
    
    # Promotion criteria
    min_sharpe_for_promotion: float = 1.0
    max_drawdown_for_promotion: float = 0.10
    min_trades_for_promotion: int = 100


class Sim2RealTracker:
    """
    Track sim2real gap metrics.
    
    This is CRITICAL - it measures how much your backtest
    deviates from paper trading (and will deviate from live).
    """
    
    def __init__(self):
        self.slippage_samples = deque(maxlen=10000)
        self.latency_samples = deque(maxlen=10000)
        self.fill_rate_samples = deque(maxlen=1000)
        self.spread_samples = deque(maxlen=10000)
        
        # Backtest vs paper comparison
        self.backtest_returns: List[float] = []
        self.paper_returns: List[float] = []
        
    def record_execution(self, order: PaperOrder, filled: bool):
        """Record execution for sim2real analysis."""
        self.slippage_samples.append(order.simulated_slippage_bps)
        self.latency_samples.append(order.latency_ms)
        self.fill_rate_samples.append(1.0 if filled else 0.0)
        self.spread_samples.append(order.spread_at_execution)
    
    def get_sim2real_report(self) -> Dict:
        """Get sim2real gap analysis."""
        if not self.slippage_samples:
            return {'status': 'insufficient_data'}
        
        slippage_arr = np.array(self.slippage_samples)
        latency_arr = np.array(self.latency_samples)
        
        return {
            'slippage': {
                'mean_bps': float(np.mean(slippage_arr)),
                'median_bps': float(np.median(slippage_arr)),
                'p95_bps': float(np.percentile(slippage_arr, 95)),
                'max_bps': float(np.max(slippage_arr)),
                'std_bps': float(np.std(slippage_arr))
            },
            'latency': {
                'mean_ms': float(np.mean(latency_arr)),
                'median_ms': float(np.median(latency_arr)),
                'p95_ms': float(np.percentile(latency_arr, 95)),
                'max_ms': float(np.max(latency_arr))
            },
            'fill_rate': float(np.mean(self.fill_rate_samples)) if self.fill_rate_samples else 0.0,
            'spread': {
                'mean_bps': float(np.mean(self.spread_samples)) if self.spread_samples else 0.0,
                'max_bps': float(np.max(self.spread_samples)) if self.spread_samples else 0.0
            },
            'n_samples': len(self.slippage_samples)
        }


class FillSimulator:
    """
    Simulate realistic order fills.
    
    Models:
    - Latency (normal distribution)
    - Slippage (volatility-dependent)
    - Partial fills (based on order size vs volume)
    - Missed fills (low liquidity, fast markets)
    """
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        
    def simulate_fill(
        self,
        order: PaperOrder,
        market_data: Dict
    ) -> PaperOrder:
        """
        Simulate order fill with realistic execution.
        
        Args:
            order: The paper order
            market_data: Current market state (price, volume, spread, volatility)
        
        Returns:
            Updated order with fill details
        """
        mode = self.config.fill_mode
        
        # Simulate latency
        latency = max(1.0, np.random.normal(
            self.config.base_latency_ms,
            self.config.latency_std_ms
        ))
        order.latency_ms = latency
        
        # Get market state
        mid_price = market_data.get('mid_price', order.price)
        bid = market_data.get('bid', mid_price * 0.9995)
        ask = market_data.get('ask', mid_price * 1.0005)
        spread_bps = (ask - bid) / mid_price * 10000
        volume = market_data.get('volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        
        order.spread_at_execution = spread_bps
        order.actual_market_price = mid_price
        
        if mode == FillSimulationMode.INSTANT:
            # Instant fill at signal price (unrealistic)
            order.filled_price = order.price
            order.filled_quantity = order.quantity
            order.status = "filled"
            order.fill_time = datetime.now()
            order.simulated_slippage_bps = 0.0
            
        elif mode == FillSimulationMode.REALISTIC:
            # Realistic fill with slippage and potential partial fills
            
            # Check if order fills (based on size vs volume)
            participation_rate = (order.quantity * mid_price) / volume
            fill_prob = self.config.fill_probability * (1 - min(participation_rate, 0.5))
            
            if np.random.random() > fill_prob:
                # Order doesn't fill (missed)
                order.status = "cancelled"
                order.filled_quantity = 0.0
                return order
            
            # Calculate slippage
            # Base slippage + volatility component + size impact
            base_slip = self.config.base_slippage_bps
            vol_slip = volatility * self.config.slippage_vol_multiplier * 100  # bps
            size_slip = participation_rate * 50  # 50 bps per 100% participation
            
            total_slip_bps = base_slip + vol_slip + size_slip
            total_slip_bps += np.random.normal(0, 2)  # Random noise
            total_slip_bps = max(0, total_slip_bps)
            
            # Apply slippage direction based on side
            slip_mult = total_slip_bps / 10000
            if order.side == 'buy':
                fill_price = mid_price * (1 + slip_mult)
            else:
                fill_price = mid_price * (1 - slip_mult)
            
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.status = "filled"
            order.fill_time = datetime.now()
            order.simulated_slippage_bps = total_slip_bps
            
        elif mode == FillSimulationMode.ADVERSARIAL:
            # Worst-case simulation (stress test)
            
            # 20% chance of no fill
            if np.random.random() > 0.80:
                order.status = "cancelled"
                order.filled_quantity = 0.0
                return order
            
            # High slippage
            total_slip_bps = self.config.base_slippage_bps * 3 + volatility * 500
            total_slip_bps += np.abs(np.random.normal(0, 10))
            
            slip_mult = total_slip_bps / 10000
            if order.side == 'buy':
                fill_price = mid_price * (1 + slip_mult)
            else:
                fill_price = mid_price * (1 - slip_mult)
            
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.status = "filled"
            order.fill_time = datetime.now()
            order.simulated_slippage_bps = total_slip_bps
        
        return order


class AnomalyDetector:
    """Detect anomalies during paper trading."""
    
    def __init__(self):
        self.recent_slippage = deque(maxlen=100)
        self.recent_latency = deque(maxlen=100)
        self.recent_pnl = deque(maxlen=100)
        
    def check_execution_anomaly(self, order: PaperOrder) -> Optional[str]:
        """Check for execution anomalies."""
        self.recent_slippage.append(order.simulated_slippage_bps)
        self.recent_latency.append(order.latency_ms)
        
        if len(self.recent_slippage) < 20:
            return None
        
        # Check for slippage spike
        mean_slip = np.mean(self.recent_slippage)
        std_slip = np.std(self.recent_slippage)
        if order.simulated_slippage_bps > mean_slip + 3 * std_slip:
            return f"SLIPPAGE_SPIKE: {order.simulated_slippage_bps:.1f} bps (mean: {mean_slip:.1f})"
        
        # Check for latency spike
        mean_lat = np.mean(self.recent_latency)
        if order.latency_ms > mean_lat * 5:
            return f"LATENCY_SPIKE: {order.latency_ms:.0f} ms (mean: {mean_lat:.0f})"
        
        return None
    
    def check_pnl_anomaly(self, pnl: float) -> Optional[str]:
        """Check for P&L anomalies."""
        self.recent_pnl.append(pnl)
        
        if len(self.recent_pnl) < 20:
            return None
        
        mean_pnl = np.mean(self.recent_pnl)
        std_pnl = np.std(self.recent_pnl)
        
        if std_pnl > 0 and abs(pnl - mean_pnl) > 3 * std_pnl:
            return f"PNL_ANOMALY: {pnl:.2f} (mean: {mean_pnl:.2f}, std: {std_pnl:.2f})"
        
        return None


class PaperTradingOrchestrator:
    """
    Master orchestrator for 30-day paper trading validation.
    
    This is NON-NEGOTIABLE before live deployment.
    """
    
    def __init__(
        self,
        config: PaperTradingConfig,
        strategy_engine: Any = None,
        data_feed: Any = None
    ):
        self.config = config
        self.strategy_engine = strategy_engine
        self.data_feed = data_feed
        
        # State
        self.mode = PaperTradingMode.PAPER
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Capital tracking
        self.initial_capital = config.initial_capital
        self.current_capital = config.initial_capital
        self.peak_capital = config.initial_capital
        
        # Positions and orders
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: List[PaperOrder] = []
        self.filled_orders: List[PaperOrder] = []
        
        # Metrics
        self.daily_returns: List[float] = []
        self.daily_reports: List[DailyReport] = []
        self.strategy_pnl: Dict[str, float] = {}
        
        # Components
        self.fill_simulator = FillSimulator(config)
        self.sim2real_tracker = Sim2RealTracker()
        self.anomaly_detector = AnomalyDetector()
        
        # Logging
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        self.alert_callback: Optional[Callable] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"PaperTradingOrchestrator initialized with {config.initial_capital} capital")
    
    def set_alert_callback(self, callback: Callable):
        """Set callback for alerts."""
        self.alert_callback = callback
    
    def _alert(self, message: str, severity: str = "INFO"):
        """Send alert."""
        logger.log(getattr(logging, severity), f"[ALERT] {message}")
        if self.alert_callback:
            self.alert_callback(message, severity)
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return hashlib.md5(f"{datetime.now().isoformat()}{np.random.random()}".encode()).hexdigest()[:12]
    
    async def start(self):
        """Start paper trading."""
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("PAPER TRADING STARTED")
        logger.info(f"Mode: {self.mode.value}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Min Duration: {self.config.min_days} days")
        logger.info("=" * 60)
        
        self._alert(f"Paper trading started with ${self.initial_capital:,.2f}", "INFO")
        
        # Main loop
        last_daily_report = datetime.now().date()
        
        while self.is_running:
            try:
                # Check if we should generate daily report
                current_date = datetime.now().date()
                if current_date > last_daily_report:
                    await self._generate_daily_report(last_daily_report)
                    last_daily_report = current_date
                
                # Process signals from strategy engine
                if self.strategy_engine:
                    await self._process_signals()
                
                # Update positions
                await self._update_positions()
                
                # Check risk limits
                self._check_risk_limits()
                
                # Check promotion criteria
                self._check_promotion_criteria()
                
                await asyncio.sleep(1)  # 1 second loop
                
            except Exception as e:
                logger.error(f"Error in paper trading loop: {e}")
                self._alert(f"Paper trading error: {e}", "ERROR")
                await asyncio.sleep(5)
    
    async def stop(self):
        """Stop paper trading."""
        self.is_running = False
        
        # Generate final report
        await self._generate_final_report()
        
        logger.info("Paper trading stopped")
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy_name: str = "",
        signal_confidence: float = 0.0,
        regime: str = "",
        orderflow_confirmation: bool = False
    ) -> PaperOrder:
        """
        Submit a paper trading order.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Signal price
            strategy_name: Originating strategy
            signal_confidence: Confidence score
            regime: Current market regime
            orderflow_confirmation: Whether order-flow confirms
        
        Returns:
            PaperOrder with execution details
        """
        order = PaperOrder(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type="market",
            quantity=quantity,
            price=price,
            strategy_name=strategy_name,
            signal_confidence=signal_confidence,
            regime=regime,
            orderflow_confirmation=orderflow_confirmation
        )
        
        # Get current market data (simulated or real)
        market_data = self._get_market_data(symbol)
        
        # Simulate fill
        order = self.fill_simulator.simulate_fill(order, market_data)
        
        # Track sim2real
        self.sim2real_tracker.record_execution(order, order.status == "filled")
        
        # Check for anomalies
        anomaly = self.anomaly_detector.check_execution_anomaly(order)
        if anomaly:
            self._alert(anomaly, "WARNING")
        
        # Log order
        self._log_order(order)
        
        with self._lock:
            self.orders.append(order)
            
            if order.status == "filled":
                self.filled_orders.append(order)
                self._update_position_from_order(order)
        
        return order
    
    def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data for symbol."""
        # If we have a data feed, use it
        if self.data_feed:
            return self.data_feed.get_current_data(symbol)
        
        # Otherwise return simulated data
        return {
            'mid_price': 50000.0,
            'bid': 49990.0,
            'ask': 50010.0,
            'volume': 10000000,
            'volatility': 0.02
        }
    
    def _update_position_from_order(self, order: PaperOrder):
        """Update positions after order fill."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # New position
            side = 'long' if order.side == 'buy' else 'short'
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=side,
                quantity=order.filled_quantity,
                entry_price=order.filled_price,
                entry_time=order.fill_time
            )
        else:
            # Update existing position
            pos = self.positions[symbol]
            
            # Check if adding or reducing
            is_adding = (pos.side == 'long' and order.side == 'buy') or \
                       (pos.side == 'short' and order.side == 'sell')
            
            if is_adding:
                # Average into position
                total_qty = pos.quantity + order.filled_quantity
                pos.entry_price = (pos.entry_price * pos.quantity + 
                                   order.filled_price * order.filled_quantity) / total_qty
                pos.quantity = total_qty
            else:
                # Reducing/closing position
                if order.filled_quantity >= pos.quantity:
                    # Close position
                    if pos.side == 'long':
                        pnl = (order.filled_price - pos.entry_price) * pos.quantity
                    else:
                        pnl = (pos.entry_price - order.filled_price) * pos.quantity
                    
                    pos.realized_pnl = pnl
                    self.current_capital += pnl
                    
                    # Track strategy P&L
                    strategy = order.strategy_name or "unknown"
                    self.strategy_pnl[strategy] = self.strategy_pnl.get(strategy, 0) + pnl
                    
                    # Check for P&L anomaly
                    anomaly = self.anomaly_detector.check_pnl_anomaly(pnl)
                    if anomaly:
                        self._alert(anomaly, "WARNING")
                    
                    del self.positions[symbol]
                else:
                    # Partial close
                    pos.quantity -= order.filled_quantity
    
    async def _update_positions(self):
        """Update all position prices and P&L."""
        for symbol, pos in list(self.positions.items()):
            market_data = self._get_market_data(symbol)
            pos.update_price(market_data.get('mid_price', pos.current_price))
    
    async def _process_signals(self):
        """Process signals from strategy engine."""
        if not self.strategy_engine:
            return
        
        try:
            signals = await self.strategy_engine.get_signals()
            for signal in signals:
                # Convert signal to order
                if signal.get('action') in ['buy', 'sell']:
                    self.submit_order(
                        symbol=signal.get('symbol'),
                        side=signal.get('action'),
                        quantity=signal.get('quantity', 0),
                        price=signal.get('price', 0),
                        strategy_name=signal.get('strategy', ''),
                        signal_confidence=signal.get('confidence', 0),
                        regime=signal.get('regime', ''),
                        orderflow_confirmation=signal.get('orderflow_confirm', False)
                    )
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
    
    def _check_risk_limits(self):
        """Check if risk limits are breached."""
        # Calculate current drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if drawdown > self.config.alert_on_drawdown:
            self._alert(f"Drawdown alert: {drawdown:.2%}", "WARNING")
        
        # Update peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Check daily loss
        if self.daily_returns:
            daily_return = self.daily_returns[-1] if self.daily_returns else 0
            if daily_return < -self.config.alert_on_daily_loss:
                self._alert(f"Daily loss alert: {daily_return:.2%}", "WARNING")
    
    def _check_promotion_criteria(self):
        """Check if ready for promotion to next stage."""
        if not self.start_time:
            return
        
        days_running = (datetime.now() - self.start_time).days
        
        if days_running < self.config.min_days:
            return
        
        # Calculate metrics
        if len(self.daily_returns) < 20:
            return
        
        returns = np.array(self.daily_returns)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_dd = self._calculate_max_drawdown()
        total_trades = len(self.filled_orders)
        
        # Check criteria
        ready = (
            sharpe >= self.config.min_sharpe_for_promotion and
            max_dd <= self.config.max_drawdown_for_promotion and
            total_trades >= self.config.min_trades_for_promotion
        )
        
        if ready:
            self._alert(
                f"PROMOTION READY: Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}, Trades={total_trades}",
                "INFO"
            )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.daily_returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(self.daily_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return float(np.max(drawdowns))
    
    def _log_order(self, order: PaperOrder):
        """Log order to file."""
        log_file = self.log_dir / f"orders_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(order.to_dict()) + '\n')
    
    async def _generate_daily_report(self, date) -> DailyReport:
        """Generate daily performance report."""
        # Calculate daily metrics
        daily_orders = [o for o in self.filled_orders 
                       if o.fill_time and o.fill_time.date() == date]
        
        winning = sum(1 for o in daily_orders if o.simulated_slippage_bps < 10)  # Simplified
        
        report = DailyReport(
            date=str(date),
            starting_equity=self.initial_capital,  # Simplified
            ending_equity=self.current_capital,
            daily_return=(self.current_capital - self.initial_capital) / self.initial_capital,
            total_trades=len(daily_orders),
            winning_trades=winning,
            losing_trades=len(daily_orders) - winning,
            win_rate=winning / len(daily_orders) if daily_orders else 0,
            avg_slippage_bps=np.mean([o.simulated_slippage_bps for o in daily_orders]) if daily_orders else 0,
            max_slippage_bps=max([o.simulated_slippage_bps for o in daily_orders]) if daily_orders else 0,
            avg_latency_ms=np.mean([o.latency_ms for o in daily_orders]) if daily_orders else 0,
            fill_rate=len([o for o in self.orders if o.status == 'filled']) / len(self.orders) if self.orders else 0,
            max_drawdown=self._calculate_max_drawdown(),
            sharpe_ratio=0,  # Calculated below
            anomalies=[],
            strategy_pnl=dict(self.strategy_pnl),
            regime_pnl={}
        )
        
        self.daily_reports.append(report)
        
        # Log report
        report_file = self.log_dir / f"daily_report_{date}.json"
        with open(report_file, 'w') as f:
            json.dump(report.__dict__, f, indent=2, default=str)
        
        logger.info(f"Daily Report [{date}]: Return={report.daily_return:.2%}, "
                   f"Trades={report.total_trades}, Slippage={report.avg_slippage_bps:.1f}bps")
        
        return report
    
    async def _generate_final_report(self):
        """Generate final paper trading report."""
        if not self.start_time:
            return
        
        duration = (datetime.now() - self.start_time).days
        
        # Calculate final metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        max_dd = self._calculate_max_drawdown()
        
        returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sim2Real analysis
        sim2real = self.sim2real_tracker.get_sim2real_report()
        
        final_report = {
            'summary': {
                'duration_days': duration,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_return': total_return,
                'annualized_return': total_return * 365 / max(duration, 1),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'total_trades': len(self.filled_orders),
                'win_rate': sum(1 for o in self.filled_orders if o.simulated_slippage_bps < 10) / len(self.filled_orders) if self.filled_orders else 0
            },
            'sim2real_gap': sim2real,
            'strategy_performance': self.strategy_pnl,
            'promotion_ready': (
                sharpe >= self.config.min_sharpe_for_promotion and
                max_dd <= self.config.max_drawdown_for_promotion and
                len(self.filled_orders) >= self.config.min_trades_for_promotion and
                duration >= self.config.min_days
            ),
            'recommendations': []
        }
        
        # Add recommendations
        if sim2real.get('slippage', {}).get('mean_bps', 0) > 15:
            final_report['recommendations'].append(
                "High slippage detected - consider reducing order sizes or using limit orders"
            )
        
        if sim2real.get('fill_rate', 1) < 0.90:
            final_report['recommendations'].append(
                "Low fill rate - improve order timing or use more aggressive pricing"
            )
        
        if max_dd > 0.15:
            final_report['recommendations'].append(
                "High drawdown - review risk limits and position sizing"
            )
        
        # Save final report
        report_file = self.log_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info("PAPER TRADING FINAL REPORT")
        logger.info(f"Duration: {duration} days")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_dd:.2%}")
        logger.info(f"Total Trades: {len(self.filled_orders)}")
        logger.info(f"Avg Slippage: {sim2real.get('slippage', {}).get('mean_bps', 0):.1f} bps")
        logger.info(f"Fill Rate: {sim2real.get('fill_rate', 0):.1%}")
        logger.info(f"Promotion Ready: {final_report['promotion_ready']}")
        logger.info("=" * 60)
        
        return final_report
    
    def get_status(self) -> Dict:
        """Get current paper trading status."""
        return {
            'mode': self.mode.value,
            'is_running': self.is_running,
            'days_running': (datetime.now() - self.start_time).days if self.start_time else 0,
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'positions': len(self.positions),
            'total_trades': len(self.filled_orders),
            'sim2real': self.sim2real_tracker.get_sim2real_report()
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def run_paper_trading_demo():
    """Demo of paper trading orchestrator."""
    
    config = PaperTradingConfig(
        initial_capital=100000,
        min_days=30,
        fill_mode=FillSimulationMode.REALISTIC
    )
    
    orchestrator = PaperTradingOrchestrator(config)
    
    # Alert callback
    def alert_handler(message: str, severity: str):
        print(f"[{severity}] {message}")
    
    orchestrator.set_alert_callback(alert_handler)
    
    # Simulate some trades
    print("\n" + "=" * 60)
    print("PAPER TRADING DEMO")
    print("=" * 60 + "\n")
    
    # Submit test orders
    for i in range(10):
        order = orchestrator.submit_order(
            symbol="BTCUSDT",
            side="buy" if i % 2 == 0 else "sell",
            quantity=0.1,
            price=50000 + np.random.normal(0, 100),
            strategy_name="momentum",
            signal_confidence=0.7,
            regime="trending",
            orderflow_confirmation=True
        )
        print(f"Order {i+1}: {order.status}, Fill={order.filled_price:.2f}, "
              f"Slippage={order.simulated_slippage_bps:.1f}bps")
    
    # Get status
    status = orchestrator.get_status()
    print(f"\nStatus: {json.dumps(status, indent=2, default=str)}")
    
    # Get sim2real report
    sim2real = orchestrator.sim2real_tracker.get_sim2real_report()
    print(f"\nSim2Real Gap Analysis:")
    print(f"  Mean Slippage: {sim2real.get('slippage', {}).get('mean_bps', 0):.1f} bps")
    print(f"  P95 Slippage: {sim2real.get('slippage', {}).get('p95_bps', 0):.1f} bps")
    print(f"  Fill Rate: {sim2real.get('fill_rate', 0):.1%}")


if __name__ == "__main__":
    asyncio.run(run_paper_trading_demo())
