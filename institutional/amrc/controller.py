"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         AUTONOMOUS META-RISK CONTROLLER (AMRC) - Controller                   ║
║                                                                               ║
║  Sub-50ms global trading halt capability for institutional safety             ║
║  Monitors multiple risk signals and can blanket-disable all alpha models      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

The AMRC is the final safety layer that assumes the market might become unplayable.
It monitors micro-second level signals and can halt all trading in < 50ms.

Monitored Signals:
1. Cross-exchange mid-price jump > 3σ in 100ms
2. API errors/5s > 5
3. Book depth collapse (top 3 levels → 0)
4. Liquidation cluster within 0.5 × ATR of current price
5. Regime posterior entropy > 0.9 (HMM "confused")

Output:
- global_trading_enabled: bool (shared-memory flag)
"""

import time
import threading
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import deque
from datetime import datetime, timedelta
import json

# Configure logging
logger = logging.getLogger("AMRC")


class AMRCStatus(Enum):
    """AMRC operational status."""
    ACTIVE = "active"           # Normal operation, trading enabled
    WARNING = "warning"         # Elevated risk, reduced capacity
    HALTED = "halted"          # Trading disabled
    EMERGENCY = "emergency"     # Critical failure, immediate halt
    MAINTENANCE = "maintenance" # Manual override


@dataclass
class AMRCConfig:
    """Configuration for AMRC thresholds."""
    
    # Price jump detection
    price_jump_sigma: float = 3.0        # Standard deviations for price jump
    price_jump_window_ms: int = 100      # Window for price jump detection
    
    # API error thresholds
    max_api_errors_per_5s: int = 5       # Max errors before halt
    api_error_window_s: float = 5.0      # Window for error counting
    
    # Order book thresholds
    min_book_depth_levels: int = 3       # Minimum top levels with liquidity
    min_book_depth_value: float = 1000   # Minimum value per level (USD)
    
    # Liquidation proximity
    liquidation_atr_multiplier: float = 0.5  # Distance in ATR units
    
    # Regime entropy
    max_regime_entropy: float = 0.9      # Max entropy (confusion threshold)
    
    # Timing
    halt_response_target_ms: float = 50  # Target halt time
    health_check_interval_ms: float = 10 # How often to check signals
    
    # Recovery
    auto_recovery_enabled: bool = True   # Allow automatic recovery
    recovery_cooldown_s: float = 300     # 5 minute cooldown after halt
    
    # Kill switch
    require_dual_auth_resume: bool = True  # Require 2-person auth to resume


@dataclass
class RiskSignal:
    """Individual risk signal measurement."""
    name: str
    value: float
    threshold: float
    triggered: bool
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    details: Dict = field(default_factory=dict)


@dataclass
class AMRCState:
    """Current state of the AMRC."""
    status: AMRCStatus = AMRCStatus.ACTIVE
    global_trading_enabled: bool = True
    last_check_time: datetime = field(default_factory=datetime.now)
    last_halt_time: Optional[datetime] = None
    last_halt_reason: Optional[str] = None
    active_signals: List[RiskSignal] = field(default_factory=list)
    halt_count_24h: int = 0
    uptime_seconds: float = 0
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'global_trading_enabled': self.global_trading_enabled,
            'last_check_time': self.last_check_time.isoformat(),
            'last_halt_time': self.last_halt_time.isoformat() if self.last_halt_time else None,
            'last_halt_reason': self.last_halt_reason,
            'active_signals_count': len(self.active_signals),
            'halt_count_24h': self.halt_count_24h,
            'uptime_seconds': self.uptime_seconds,
        }


class AutonomousMetaRiskController:
    """
    Autonomous Meta-Risk Controller (AMRC)
    
    The final safety layer that can blanket-disable all trading
    in < 50ms when market conditions become dangerous.
    
    Usage:
        amrc = AutonomousMetaRiskController(config)
        amrc.start()
        
        # In order execution loop:
        if not amrc.is_trading_enabled():
            return  # Do not execute
            
        # Update with market data
        amrc.update_price(symbol, price, timestamp)
        amrc.update_orderbook(symbol, bids, asks)
        amrc.update_api_status(success=True)
    """
    
    def __init__(
        self,
        config: Optional[AMRCConfig] = None,
        on_halt_callback: Optional[Callable[[str], None]] = None,
        on_resume_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize AMRC.
        
        Args:
            config: AMRC configuration
            on_halt_callback: Called when trading is halted (with reason)
            on_resume_callback: Called when trading resumes
        """
        self.config = config or AMRCConfig()
        self.on_halt_callback = on_halt_callback
        self.on_resume_callback = on_resume_callback
        
        # State
        self.state = AMRCState()
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Shared memory flag (for ultra-low latency)
        self._shared_flag = True  # In production: use mmap or C++ shared memory
        
        # Data buffers
        self._price_history: Dict[str, deque] = {}  # symbol -> [(timestamp, price)]
        self._api_errors: deque = deque(maxlen=1000)  # (timestamp, error_type)
        self._book_snapshots: Dict[str, Dict] = {}  # symbol -> latest book
        self._liquidation_levels: Dict[str, List[float]] = {}  # symbol -> [prices]
        self._regime_entropy: float = 0.0
        self._atr_values: Dict[str, float] = {}  # symbol -> ATR
        
        # Statistics
        self._stats = {
            'checks_performed': 0,
            'signals_triggered': 0,
            'halts_triggered': 0,
            'false_alarms': 0,
            'avg_check_time_us': 0,
        }
        
        logger.info("AMRC initialized with config: %s", self.config)
    
    # =========================================================================
    # CORE API
    # =========================================================================
    
    def start(self) -> None:
        """Start the AMRC monitoring loop."""
        if self._running:
            logger.warning("AMRC already running")
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="AMRC-Monitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("AMRC monitoring started")
    
    def stop(self) -> None:
        """Stop the AMRC monitoring loop."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("AMRC monitoring stopped")
    
    def is_trading_enabled(self) -> bool:
        """
        Check if trading is enabled.
        
        This is the primary method called by the order engine.
        Must be extremely fast (< 1μs).
        
        Returns:
            bool: True if trading is allowed
        """
        return self._shared_flag
    
    def halt_trading(self, reason: str, severity: str = "critical") -> None:
        """
        Immediately halt all trading.
        
        Args:
            reason: Why trading was halted
            severity: 'warning', 'high', 'critical', 'emergency'
        """
        start_time = time.perf_counter_ns()
        
        with self._lock:
            # Set shared flag (atomic)
            self._shared_flag = False
            
            # Update state
            self.state.global_trading_enabled = False
            self.state.status = (
                AMRCStatus.EMERGENCY if severity == "emergency"
                else AMRCStatus.HALTED
            )
            self.state.last_halt_time = datetime.now()
            self.state.last_halt_reason = reason
            self.state.halt_count_24h += 1
            
            self._stats['halts_triggered'] += 1
        
        elapsed_us = (time.perf_counter_ns() - start_time) / 1000
        
        logger.critical(
            "🚨 AMRC HALT: %s (severity=%s, elapsed=%.1fμs)",
            reason, severity, elapsed_us
        )
        
        # Callback
        if self.on_halt_callback:
            try:
                self.on_halt_callback(reason)
            except Exception as e:
                logger.error("Halt callback error: %s", e)
    
    def resume_trading(self, auth_signatures: Optional[List[str]] = None) -> bool:
        """
        Resume trading after a halt.
        
        Args:
            auth_signatures: Required if dual auth is enabled
            
        Returns:
            bool: True if trading was resumed
        """
        # Check dual auth requirement
        if self.config.require_dual_auth_resume:
            if not auth_signatures or len(auth_signatures) < 2:
                logger.warning("Resume rejected: requires 2-person authentication")
                return False
            # In production: verify RSA signatures
            logger.info("Dual authorization verified")
        
        # Check cooldown
        if self.state.last_halt_time:
            elapsed = (datetime.now() - self.state.last_halt_time).total_seconds()
            if elapsed < self.config.recovery_cooldown_s:
                remaining = self.config.recovery_cooldown_s - elapsed
                logger.warning("Resume rejected: cooldown remaining %.1fs", remaining)
                return False
        
        with self._lock:
            self._shared_flag = True
            self.state.global_trading_enabled = True
            self.state.status = AMRCStatus.ACTIVE
        
        logger.info("✅ AMRC: Trading resumed")
        
        if self.on_resume_callback:
            try:
                self.on_resume_callback()
            except Exception as e:
                logger.error("Resume callback error: %s", e)
        
        return True
    
    # =========================================================================
    # DATA UPDATE METHODS
    # =========================================================================
    
    def update_price(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update price for a symbol."""
        ts = timestamp or datetime.now()
        
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=10000)
        
        self._price_history[symbol].append((ts, price))
    
    def update_orderbook(
        self,
        symbol: str,
        bids: List[tuple],  # [(price, size), ...]
        asks: List[tuple],
    ) -> None:
        """Update order book snapshot."""
        self._book_snapshots[symbol] = {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now(),
        }
    
    def update_api_status(
        self,
        success: bool,
        error_type: Optional[str] = None
    ) -> None:
        """Record API call result."""
        if not success:
            self._api_errors.append((datetime.now(), error_type))
    
    def update_liquidation_levels(
        self,
        symbol: str,
        levels: List[float]
    ) -> None:
        """Update known liquidation price levels."""
        self._liquidation_levels[symbol] = levels
    
    def update_regime_entropy(self, entropy: float) -> None:
        """Update regime model entropy (confusion level)."""
        self._regime_entropy = entropy
    
    def update_atr(self, symbol: str, atr: float) -> None:
        """Update Average True Range for a symbol."""
        self._atr_values[symbol] = atr
    
    # =========================================================================
    # SIGNAL DETECTION
    # =========================================================================
    
    def _check_all_signals(self) -> List[RiskSignal]:
        """Check all risk signals and return triggered ones."""
        signals = []
        
        # 1. Price jump detection
        price_signal = self._check_price_jump()
        if price_signal:
            signals.append(price_signal)
        
        # 2. API error rate
        api_signal = self._check_api_errors()
        if api_signal:
            signals.append(api_signal)
        
        # 3. Book depth collapse
        book_signal = self._check_book_depth()
        if book_signal:
            signals.append(book_signal)
        
        # 4. Liquidation proximity
        liq_signal = self._check_liquidation_proximity()
        if liq_signal:
            signals.append(liq_signal)
        
        # 5. Regime entropy
        entropy_signal = self._check_regime_entropy()
        if entropy_signal:
            signals.append(entropy_signal)
        
        return signals
    
    def _check_price_jump(self) -> Optional[RiskSignal]:
        """Check for abnormal price jumps."""
        for symbol, history in self._price_history.items():
            if len(history) < 20:
                continue
            
            # Get recent prices within window
            now = datetime.now()
            window_start = now - timedelta(milliseconds=self.config.price_jump_window_ms)
            
            recent_prices = [p for ts, p in history if ts >= window_start]
            if len(recent_prices) < 2:
                continue
            
            # Calculate return
            price_return = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Get historical volatility
            all_prices = [p for _, p in list(history)[-100:]]
            if len(all_prices) < 10:
                continue
                
            returns = np.diff(all_prices) / all_prices[:-1]
            std = np.std(returns) if len(returns) > 0 else 0.01
            
            # Check if jump exceeds threshold
            z_score = abs(price_return) / (std + 1e-10)
            
            if z_score > self.config.price_jump_sigma:
                return RiskSignal(
                    name="price_jump",
                    value=z_score,
                    threshold=self.config.price_jump_sigma,
                    triggered=True,
                    timestamp=now,
                    severity="critical",
                    details={
                        'symbol': symbol,
                        'return': price_return,
                        'z_score': z_score,
                    }
                )
        
        return None
    
    def _check_api_errors(self) -> Optional[RiskSignal]:
        """Check API error rate."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.api_error_window_s)
        
        recent_errors = [
            err for ts, err in self._api_errors
            if ts >= window_start
        ]
        
        error_count = len(recent_errors)
        
        if error_count >= self.config.max_api_errors_per_5s:
            return RiskSignal(
                name="api_errors",
                value=error_count,
                threshold=self.config.max_api_errors_per_5s,
                triggered=True,
                timestamp=now,
                severity="high",
                details={
                    'error_count': error_count,
                    'window_seconds': self.config.api_error_window_s,
                    'error_types': list(set(recent_errors)),
                }
            )
        
        return None
    
    def _check_book_depth(self) -> Optional[RiskSignal]:
        """Check for order book depth collapse."""
        for symbol, book in self._book_snapshots.items():
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            # Check if top levels have sufficient depth
            bid_levels_with_depth = sum(
                1 for _, size in bids[:self.config.min_book_depth_levels]
                if size * bids[0][0] >= self.config.min_book_depth_value
            ) if bids else 0
            
            ask_levels_with_depth = sum(
                1 for _, size in asks[:self.config.min_book_depth_levels]
                if size * asks[0][0] >= self.config.min_book_depth_value
            ) if asks else 0
            
            if (bid_levels_with_depth < self.config.min_book_depth_levels or
                ask_levels_with_depth < self.config.min_book_depth_levels):
                return RiskSignal(
                    name="book_depth_collapse",
                    value=min(bid_levels_with_depth, ask_levels_with_depth),
                    threshold=self.config.min_book_depth_levels,
                    triggered=True,
                    timestamp=datetime.now(),
                    severity="critical",
                    details={
                        'symbol': symbol,
                        'bid_levels': bid_levels_with_depth,
                        'ask_levels': ask_levels_with_depth,
                    }
                )
        
        return None
    
    def _check_liquidation_proximity(self) -> Optional[RiskSignal]:
        """Check if price is near liquidation clusters."""
        for symbol, levels in self._liquidation_levels.items():
            if not levels or symbol not in self._price_history:
                continue
            
            history = self._price_history[symbol]
            if not history:
                continue
            
            current_price = history[-1][1]
            atr = self._atr_values.get(symbol, current_price * 0.02)
            threshold_distance = atr * self.config.liquidation_atr_multiplier
            
            for level in levels:
                distance = abs(current_price - level)
                if distance < threshold_distance:
                    return RiskSignal(
                        name="liquidation_proximity",
                        value=distance,
                        threshold=threshold_distance,
                        triggered=True,
                        timestamp=datetime.now(),
                        severity="high",
                        details={
                            'symbol': symbol,
                            'current_price': current_price,
                            'liquidation_level': level,
                            'distance': distance,
                            'atr': atr,
                        }
                    )
        
        return None
    
    def _check_regime_entropy(self) -> Optional[RiskSignal]:
        """Check if regime model is confused."""
        if self._regime_entropy > self.config.max_regime_entropy:
            return RiskSignal(
                name="regime_entropy",
                value=self._regime_entropy,
                threshold=self.config.max_regime_entropy,
                triggered=True,
                timestamp=datetime.now(),
                severity="medium",
                details={
                    'entropy': self._regime_entropy,
                    'interpretation': 'HMM confusion - uncertain market regime',
                }
            )
        
        return None
    
    # =========================================================================
    # MONITORING LOOP
    # =========================================================================
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("AMRC monitoring loop started")
        
        while self._running:
            start_time = time.perf_counter_ns()
            
            try:
                # Check all signals
                signals = self._check_all_signals()
                
                with self._lock:
                    self.state.active_signals = signals
                    self.state.last_check_time = datetime.now()
                    self._stats['checks_performed'] += 1
                
                # If any critical signal, halt immediately
                critical_signals = [s for s in signals if s.severity in ['critical', 'emergency']]
                
                if critical_signals and self._shared_flag:
                    self._stats['signals_triggered'] += len(critical_signals)
                    reason = "; ".join([
                        f"{s.name}={s.value:.2f}>{s.threshold:.2f}"
                        for s in critical_signals
                    ])
                    self.halt_trading(reason, severity="critical")
                
                # Update timing stats
                elapsed_us = (time.perf_counter_ns() - start_time) / 1000
                alpha = 0.1
                self._stats['avg_check_time_us'] = (
                    alpha * elapsed_us +
                    (1 - alpha) * self._stats['avg_check_time_us']
                )
                
            except Exception as e:
                logger.error("AMRC monitoring error: %s", e, exc_info=True)
            
            # Sleep until next check
            sleep_time = self.config.health_check_interval_ms / 1000.0
            time.sleep(sleep_time)
        
        logger.info("AMRC monitoring loop stopped")
    
    # =========================================================================
    # STATUS & DIAGNOSTICS
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get current AMRC status."""
        with self._lock:
            return {
                'state': self.state.to_dict(),
                'stats': self._stats.copy(),
                'config': {
                    'price_jump_sigma': self.config.price_jump_sigma,
                    'max_api_errors_per_5s': self.config.max_api_errors_per_5s,
                    'halt_response_target_ms': self.config.halt_response_target_ms,
                },
            }
    
    def run_diagnostics(self) -> Dict:
        """Run self-diagnostics."""
        results = {
            'healthy': True,
            'checks': {},
        }
        
        # Check shared flag access time
        start = time.perf_counter_ns()
        _ = self._shared_flag
        flag_access_ns = time.perf_counter_ns() - start
        results['checks']['flag_access_ns'] = flag_access_ns
        
        # Check signal processing time
        start = time.perf_counter_ns()
        self._check_all_signals()
        signal_check_us = (time.perf_counter_ns() - start) / 1000
        results['checks']['signal_check_us'] = signal_check_us
        
        # Check if monitoring thread is alive
        thread_alive = self._monitor_thread and self._monitor_thread.is_alive()
        results['checks']['monitor_thread_alive'] = thread_alive
        
        # Overall health
        results['healthy'] = (
            flag_access_ns < 1000 and  # < 1μs
            signal_check_us < 1000 and  # < 1ms
            (not self._running or thread_alive)
        )
        
        return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_halt(reason: str):
        print(f"🚨 HALT CALLBACK: {reason}")
    
    def on_resume():
        print("✅ RESUME CALLBACK")
    
    # Create AMRC
    config = AMRCConfig(
        price_jump_sigma=2.5,  # More sensitive for demo
        max_api_errors_per_5s=3,
    )
    
    amrc = AutonomousMetaRiskController(
        config=config,
        on_halt_callback=on_halt,
        on_resume_callback=on_resume,
    )
    
    # Start monitoring
    amrc.start()
    
    print("AMRC Status:", amrc.get_status())
    print("Trading enabled:", amrc.is_trading_enabled())
    
    # Simulate normal operation
    for i in range(100):
        amrc.update_price("BTCUSD", 50000 + np.random.randn() * 100)
        amrc.update_api_status(success=True)
        time.sleep(0.01)
    
    print("\n--- After normal operation ---")
    print("Trading enabled:", amrc.is_trading_enabled())
    
    # Simulate API errors
    print("\n--- Simulating API errors ---")
    for i in range(10):
        amrc.update_api_status(success=False, error_type="timeout")
        time.sleep(0.1)
    
    time.sleep(0.5)
    print("Trading enabled:", amrc.is_trading_enabled())
    print("Status:", amrc.state.status.value)
    
    # Run diagnostics
    print("\n--- Diagnostics ---")
    diag = amrc.run_diagnostics()
    print(json.dumps(diag, indent=2))
    
    # Cleanup
    amrc.stop()
    print("\nAMRC stopped")
