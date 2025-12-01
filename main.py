"""
Main Orchestrator for Delta Exchange Algo Trading Bot v2.0
Production-ready entry point with full lifecycle management
Enhanced with automatic instrument discovery, options trading, and capital-aware sizing
"""

import os
import sys
import signal
import time
import threading
from datetime import datetime, date
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from config import get_config, ensure_directories
from logger import setup_logging, get_logger, get_audit_logger
from delta_client import get_delta_client
from data_ingest import get_data_ingestor
from strategy import get_strategy_manager, SignalDirection
from risk_manager import get_risk_manager
from order_manager import get_order_manager
from monitoring import (
    get_monitoring_server, get_metrics, get_alert_manager
)
from product_discovery import get_product_discovery
from instrument_selector import get_instrument_selector, InstrumentChoice
from options import get_options_scanner


class TradingBot:
    """
    Main trading bot orchestrator
    
    Coordinates all components and manages the trading lifecycle
    """
    
    def __init__(self):
        # Initialize configuration
        ensure_directories()
        self.config = get_config()
        
        # Initialize logging
        self.logger = setup_logging(
            log_level=self.config.logging.log_level,
            log_file=self.config.logging.log_file,
            json_logging=self.config.logging.json_logging
        )
        self.audit = get_audit_logger()
        
        # Initialize components
        self.client = get_delta_client()
        self.data_ingestor = get_data_ingestor()
        self.strategy_manager = get_strategy_manager()
        self.risk_manager = get_risk_manager()
        self.order_manager = get_order_manager()
        
        # Initialize monitoring
        self.metrics = get_metrics()
        self.alerts = get_alert_manager()
        self.monitoring_server = get_monitoring_server()
        
        # Initialize v2 components
        self.product_discovery = get_product_discovery()
        self.instrument_selector = get_instrument_selector()
        self.options_scanner = get_options_scanner()
        
        # State
        self._running = False
        self._shutdown_event = threading.Event()
        self._last_health_check = datetime.min
        self._last_day = date.today()
        self._tradable_underlyings: List[str] = []
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        self.logger.info(
            "Trading bot initialized",
            product=self.config.trading.product_symbol,
            mode=self.config.trading.trading_mode,
            is_testnet=self.client.is_testnet,
            version="2.0"
        )
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Shutdown signal received", signal=signum)
        self.stop()
    
    def _check_kill_switch(self) -> bool:
        """Check if kill switch is active"""
        if self.risk_manager.is_kill_switch_active:
            self.logger.critical("KILL SWITCH ACTIVE - Trading disabled")
            return True
        return False
    
    def _check_new_day(self):
        """Check for new trading day and reset daily stats"""
        today = date.today()
        if today != self._last_day:
            self.logger.info("New trading day detected", date=today)
            self.risk_manager.reset_daily_stats()
            self._last_day = today
    
    def _run_health_check(self):
        """Run periodic health checks"""
        now = datetime.now()
        interval = self.config.system.health_check_interval
        
        if (now - self._last_health_check).total_seconds() < interval:
            return
        
        self._last_health_check = now
        
        # Check API connectivity
        try:
            is_healthy = self.client.health_check()
            self.metrics.health_status.set(1 if is_healthy else 0)
            
            if not is_healthy:
                self.alerts.alert_error("API_HEALTH", "Delta Exchange API health check failed")
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            self.metrics.health_status.set(0)
        
        # Sync positions and orders
        try:
            self.order_manager.sync_positions()
            self.order_manager.sync_orders()
        except Exception as e:
            self.logger.error("Sync failed", error=str(e))
        
        # Refresh product discovery
        try:
            self.product_discovery.discover_all()
            self._tradable_underlyings = self.instrument_selector.get_tradable_underlyings()
        except Exception as e:
            self.logger.error("Product discovery failed", error=str(e))
        
        # Update capital metrics
        try:
            capital_metrics = self.risk_manager.get_capital_metrics()
            self.logger.debug(
                "Capital status",
                available=capital_metrics['available_balance'],
                equity=capital_metrics['equity']
            )
        except Exception as e:
            self.logger.error("Capital metrics failed", error=str(e))
        
        # Update metrics
        self.metrics.update_uptime()
        self.metrics.kill_switch_active.set(1 if self.risk_manager.is_kill_switch_active else 0)
        
        risk_metrics = self.risk_manager.get_risk_metrics()
        self.metrics.pnl_daily.set(risk_metrics['daily_pnl'])
        self.metrics.daily_loss_remaining.set(risk_metrics['daily_loss_remaining'])
        self.metrics.open_positions.set(risk_metrics['open_positions'])
        self.metrics.win_rate.set(risk_metrics['win_rate'])
    
    def _execute_signal(self, signal, underlying: str = None):
        """Execute a trading signal using instrument selector for optimal instrument choice"""
        if not signal.is_actionable:
            return
        
        # Use default underlying if not specified
        underlying = underlying or self.config.trading.product_symbol.replace("USD", "").replace("PERP", "")
        
        # Build signals dict for instrument selector
        signals = {
            "momentum": signal.indicators.get("momentum", 0),
            "orderbook_imbalance": signal.indicators.get("orderbook_imbalance", 0),
            "sentiment": signal.indicators.get("sentiment", 0),
            "news_event": signal.indicators.get("news_event", 0)
        }
        
        # Get current market state for context
        state = self.data_ingestor.get_current_state()
        market_data = {
            "implied_volatility": 0.4,  # Will be updated if options available
            "trend_strength": abs(signals.get("momentum", 0))
        }
        
        # Get recommendation from instrument selector
        recommendation = self.instrument_selector.select_instrument(
            underlying=underlying,
            signals=signals,
            market_data=market_data
        )
        
        # Log the recommendation
        self.logger.info(
            "Instrument selector recommendation",
            underlying=underlying,
            choice=recommendation.instrument_choice.value,
            direction=recommendation.direction,
            confidence=recommendation.confidence,
            reasons=recommendation.reasons[:3]  # First 3 reasons
        )
        
        # Check if recommendation is valid
        if not recommendation.is_valid:
            self.logger.debug(
                "No valid trade recommendation",
                reasons=recommendation.reasons
            )
            return
        
        # Execute based on instrument choice
        if recommendation.instrument_choice in (InstrumentChoice.PERPETUAL, InstrumentChoice.FUTURES):
            self._execute_futures_trade(recommendation, state)
        elif recommendation.instrument_choice in (InstrumentChoice.CALL_OPTION, InstrumentChoice.PUT_OPTION):
            self._execute_options_trade(recommendation, state)
    
    def _execute_futures_trade(self, recommendation, state):
        """Execute a futures/perpetual trade"""
        side = 'buy' if recommendation.direction == 'long' else 'sell'
        
        # Use recommendation sizing
        size = recommendation.size
        stop_price = recommendation.stop_price
        take_profit_price = recommendation.take_profit
        
        if size <= 0:
            self.logger.warning("Invalid futures position size, skipping trade")
            return
        
        # Validate affordability
        affordability = self.risk_manager.validate_trade_affordability(
            instrument_type='futures',
            size=size,
            price=recommendation.entry_price
        )
        
        if not affordability.allowed:
            self.logger.warning(
                "Trade not affordable",
                message=affordability.message
            )
            if affordability.adjusted_size:
                size = affordability.adjusted_size
            else:
                return
        
        self.logger.info(
            "Executing futures signal",
            direction=recommendation.direction,
            confidence=recommendation.confidence,
            side=side,
            size=size,
            stop_price=stop_price,
            take_profit=take_profit_price
        )
        
        # Check if paper trading
        if self.config.trading.is_paper:
            self.logger.info(
                "[PAPER] Would place futures order",
                side=side,
                size=size,
                price=recommendation.entry_price
            )
            self.metrics.record_order("market", side, "paper")
            return
        
        # Get product ID
        product_id = recommendation.instrument.product_id if recommendation.instrument else self.config.trading.product_id
        
        # Place order
        order = self.order_manager.place_market_order(
            product_id=product_id,
            side=side,
            size=size,
            attach_stop=True,
            stop_price=stop_price,
            take_profit_price=take_profit_price
        )
        
        if order:
            self.metrics.record_order("market", side, "placed")
            self.alerts.alert_order_placed({
                "instrument": "futures",
                "symbol": recommendation.instrument.symbol if recommendation.instrument else "N/A",
                "side": side,
                "size": size,
                "price": recommendation.entry_price,
                "stop": stop_price,
                "take_profit": take_profit_price
            })
        else:
            self.metrics.record_error("order_placement")
    
    def _execute_options_trade(self, recommendation, state):
        """Execute an options trade"""
        if not recommendation.option_trade:
            self.logger.warning("No option trade details, skipping")
            return
        
        option_trade = recommendation.option_trade
        side = 'buy'  # We only buy options for directional trades
        size = option_trade.contracts
        
        if size <= 0:
            self.logger.warning("Invalid option position size, skipping trade")
            return
        
        # Validate affordability
        affordability = self.risk_manager.validate_trade_affordability(
            instrument_type='options',
            size=size,
            price=option_trade.premium,
            premium=option_trade.premium
        )
        
        if not affordability.allowed:
            self.logger.warning(
                "Option trade not affordable",
                message=affordability.message
            )
            if affordability.adjusted_size:
                size = int(affordability.adjusted_size)
            else:
                return
        
        self.logger.info(
            "Executing options signal",
            direction=recommendation.direction,
            option_type=option_trade.direction,
            symbol=option_trade.instrument.symbol,
            strike=option_trade.strike,
            contracts=size,
            premium=option_trade.premium,
            delta=option_trade.delta,
            expiry=option_trade.expiry
        )
        
        # Check if paper trading
        if self.config.trading.is_paper:
            self.logger.info(
                "[PAPER] Would place options order",
                symbol=option_trade.instrument.symbol,
                contracts=size,
                premium=option_trade.premium
            )
            self.metrics.record_order("limit", side, "paper")
            return
        
        # Place limit order for options (at ask price)
        order = self.order_manager.place_limit_order(
            product_id=option_trade.instrument.product_id,
            side=side,
            size=float(size),
            price=option_trade.premium
        )
        
        if order:
            self.metrics.record_order("limit", side, "placed")
            self.alerts.alert_order_placed({
                "instrument": "option",
                "symbol": option_trade.instrument.symbol,
                "type": option_trade.direction,
                "strike": option_trade.strike,
                "expiry": option_trade.expiry.isoformat() if option_trade.expiry else "N/A",
                "contracts": size,
                "premium": option_trade.premium,
                "max_loss": option_trade.max_loss
            })
        else:
            self.metrics.record_error("order_placement")
    
    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting trading loop")
        poll_interval = self.config.data_ingestion.ticker_poll_interval
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # Check for new day
                self._check_new_day()
                
                # Run health check
                self._run_health_check()
                
                # Check kill switch
                if self._check_kill_switch():
                    time.sleep(poll_interval)
                    continue
                
                # Get current market state
                state = self.data_ingestor.get_current_state()
                
                # Update market metrics
                self.metrics.current_price.labels(
                    symbol=self.config.trading.product_symbol
                ).set(state.price)
                
                if state.orderbook:
                    self.metrics.orderbook_imbalance.labels(
                        symbol=self.config.trading.product_symbol
                    ).set(state.orderbook.imbalance)
                    self.metrics.spread.labels(
                        symbol=self.config.trading.product_symbol
                    ).set(state.orderbook.spread)
                
                self.metrics.sentiment_score.set(state.sentiment_score)
                
                # Check if warmed up
                if not self.data_ingestor.is_warmed_up():
                    self.logger.debug(
                        "Warming up",
                        collected=len(state.prices),
                        required=self.config.strategy.warmup_period
                    )
                    time.sleep(poll_interval)
                    continue
                
                # Evaluate strategy
                signal = self.strategy_manager.evaluate(state)
                
                # Update signal metrics
                self.metrics.signal_strength.labels(
                    direction=signal.direction.value
                ).set(signal.strength)
                self.metrics.composite_score.set(
                    signal.indicators.get('composite', 0)
                )
                
                # Check if we can trade
                risk_check = self.risk_manager.check_can_trade()
                
                if signal.is_actionable and risk_check.allowed:
                    # Check if we already have a position
                    if not self.order_manager.has_position(self.config.trading.product_id):
                        self._execute_signal(signal)
                    else:
                        self.logger.debug("Already have position, skipping signal")
                elif not risk_check.allowed:
                    self.logger.debug(
                        "Trade blocked by risk manager",
                        reason=risk_check.message
                    )
                else:
                    self.logger.debug(
                        "No actionable signal",
                        direction=signal.direction.value,
                        strength=signal.strength
                    )
                
                # Sleep until next iteration
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.exception("Error in trading loop", error=str(e))
                self.metrics.record_error("trading_loop")
                self.alerts.alert_error("TRADING_LOOP", str(e))
                time.sleep(poll_interval)
    
    def start(self):
        """Start the trading bot"""
        if self._running:
            self.logger.warning("Bot already running")
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Initial product discovery
        self.logger.info("Discovering tradable instruments...")
        try:
            instruments = self.product_discovery.discover_all(force=True)
            summary = self.product_discovery.get_tradable_summary()
            self._tradable_underlyings = self.instrument_selector.get_tradable_underlyings()
            
            self.logger.info(
                "Product discovery complete",
                total_instruments=summary['total_instruments'],
                active_instruments=summary['active_instruments'],
                tradable_underlyings=self._tradable_underlyings[:5]  # First 5
            )
        except Exception as e:
            self.logger.error("Initial product discovery failed", error=str(e))
        
        # Fetch initial account state
        try:
            account = self.risk_manager.fetch_account_state(force_refresh=True)
            self.logger.info(
                "Account state loaded",
                available=account.available_balance,
                equity=account.equity,
                currency=account.currency
            )
        except Exception as e:
            self.logger.warning("Could not fetch account state", error=str(e))
        
        # Log startup
        self.logger.log_system_event(
            event_type="startup",
            details={
                "version": "2.0",
                "product": self.config.trading.product_symbol,
                "mode": self.config.trading.trading_mode,
                "is_testnet": self.client.is_testnet,
                "risk_per_trade": self.config.risk.risk_per_trade_inr,
                "max_daily_loss": self.config.risk.max_daily_loss_inr,
                "exposure_pct": self.config.risk.exposure_pct,
                "confidence_threshold": self.config.risk.confidence_threshold,
                "tradable_underlyings": len(self._tradable_underlyings)
            }
        )
        
        # Start monitoring server
        if self.config.monitoring.enable_metrics:
            self.monitoring_server.set_health_checker(self.client.health_check)
            self.monitoring_server.set_bot_instance(self)
            self.monitoring_server.start()
        
        # Send startup alert
        self.alerts.alert_startup({
            "product": self.config.trading.product_symbol,
            "mode": self.config.trading.trading_mode,
            "testnet": self.client.is_testnet
        })
        
        # Run trading loop
        try:
            self._trading_loop()
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot gracefully"""
        if not self._running:
            return
        
        self.logger.info("Stopping trading bot...")
        self._running = False
        self._shutdown_event.set()
        
        # Close all positions if configured
        try:
            # Cancel all open orders
            self.order_manager.cancel_all_orders()
            
            # Optionally close positions (uncomment if desired)
            # self.order_manager.close_all_positions()
        except Exception as e:
            self.logger.error("Error during shutdown cleanup", error=str(e))
        
        # Stop monitoring server
        if self.monitoring_server:
            self.monitoring_server.stop()
        
        # Send shutdown alert
        self.alerts.alert_shutdown("Graceful shutdown")
        
        # Log shutdown
        self.logger.log_system_event(
            event_type="shutdown",
            details={"reason": "graceful"}
        )
        
        self.logger.info("Trading bot stopped")
    
    def emergency_stop(self, reason: str = "Emergency stop"):
        """Emergency stop - close all positions and halt trading"""
        self.logger.critical("EMERGENCY STOP", reason=reason)
        
        # Activate kill switch
        self.risk_manager.activate_kill_switch(reason)
        
        # Cancel all orders
        try:
            self.order_manager.cancel_all_orders()
        except Exception as e:
            self.logger.error("Failed to cancel orders", error=str(e))
        
        # Close all positions
        try:
            self.order_manager.close_all_positions()
        except Exception as e:
            self.logger.error("Failed to close positions", error=str(e))
        
        # Alert
        self.alerts.alert_kill_switch(reason)
        
        # Stop bot
        self.stop()


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         Delta Exchange Algo Trading Bot v2.0                  ║
    ║         Enhanced with Options & Instrument Discovery          ║
    ║         ⚠️  USE AT YOUR OWN RISK - START ON TESTNET  ⚠️        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for required environment variables
    config = get_config()
    
    if not config.delta.api_key or not config.delta.api_secret:
        print("ERROR: DELTA_API_KEY and DELTA_API_SECRET must be set in .env file")
        print("Copy .env.example to .env and fill in your credentials")
        sys.exit(1)
    
    if not config.delta.is_testnet:
        print("\n⚠️  WARNING: You are connecting to PRODUCTION (not testnet)")
        print("    Make sure you know what you're doing!")
        response = input("    Type 'YES' to continue: ")
        if response != 'YES':
            print("Aborted.")
            sys.exit(0)
    
    # Create and start bot
    bot = TradingBot()
    
    try:
        bot.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        bot.emergency_stop(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
