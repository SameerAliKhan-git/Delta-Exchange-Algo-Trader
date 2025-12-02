#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗                  ║
║    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                  ║
║    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║                  ║
║    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║                  ║
║    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║                  ║
║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝                  ║
║                                                                           ║
║                      📝 PAPER TRADING EDITION 📝                          ║
║                                                                           ║
║   • Real market data from Delta Exchange                                  ║
║   • SIMULATED trades - NO real money at risk                              ║
║   • Track strategy performance before going live                          ║
║   • Learn what works without losing capital                               ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import threading
import requests
import websocket
from datetime import datetime
from collections import deque

# =============================================================================
# CONFIGURATION
# =============================================================================

# Starting paper capital (simulate with $1000)
PAPER_CAPITAL = 1000.00

# Assets to trade
ASSETS = {
    'BTCUSD': {'contract_size': 0.001, 'leverage': 100, 'product_id': 139},
    'ETHUSD': {'contract_size': 0.01, 'leverage': 100, 'product_id': 140},
    'SOLUSD': {'contract_size': 0.1, 'leverage': 50, 'product_id': 146},
    'XRPUSD': {'contract_size': 5, 'leverage': 50, 'product_id': 185},
}

# Trading fees (same as real)
MAKER_FEE = 0.0002  # 0.02%
TAKER_FEE = 0.0005  # 0.05%

# Strategy parameters
MIN_PROFIT_TARGET = 0.005   # 0.5% minimum target
MAX_PROFIT_TARGET = 0.02    # 2.0% maximum target
STOP_LOSS = 0.003           # 0.3% stop loss
BREAKEVEN_TRIGGER = 0.003   # Move stop to breakeven at +0.3%
TRAILING_TRIGGER = 0.005    # Start trailing at +0.5%
TRAILING_DISTANCE = 0.002   # Trail by 0.2%

# Technical analysis
TREND_PERIOD = 20           # Candles for trend
MOMENTUM_PERIOD = 10        # Candles for momentum
MIN_TREND_STRENGTH = 0.3    # Minimum trend strength to trade

# Position sizing
RISK_PER_TRADE = 0.02       # Risk 2% per trade
MAX_POSITIONS = 2           # Maximum concurrent positions

# WebSocket
WS_URL = "wss://socket.india.delta.exchange"

# =============================================================================
# PRICE ENGINE - Real Market Data
# =============================================================================

class PriceEngine:
    """Streams real prices from Delta Exchange"""
    
    def __init__(self, on_price_update):
        self.prices = {}
        self.price_history = {symbol: deque(maxlen=100) for symbol in ASSETS}
        self.on_price_update = on_price_update
        self.ws = None
        self.running = False
        
    def start(self):
        self.running = True
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()
            
    def _run(self):
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.run_forever(ping_interval=25)
            except Exception as e:
                print(f"⚠️ WebSocket error: {e}")
                time.sleep(1)
                
    def _on_open(self, ws):
        print("📡 Connected to Delta Exchange (Real Market Data)")
        # Subscribe to all assets
        for symbol, info in ASSETS.items():
            msg = {
                "type": "subscribe",
                "payload": {
                    "channels": [
                        {"name": "v2/ticker", "symbols": [symbol]}
                    ]
                }
            }
            ws.send(json.dumps(msg))
            
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'v2/ticker':
                symbol = data.get('symbol')
                mark_price = float(data.get('mark_price', 0))
                
                if symbol and mark_price > 0:
                    old_price = self.prices.get(symbol, mark_price)
                    self.prices[symbol] = mark_price
                    self.price_history[symbol].append({
                        'price': mark_price,
                        'time': time.time()
                    })
                    
                    # Callback
                    self.on_price_update(symbol, mark_price, old_price)
                    
        except Exception as e:
            pass
            
    def _on_error(self, ws, error):
        pass
        
    def _on_close(self, ws, close_code, close_msg):
        if self.running:
            print("📡 Reconnecting...")
            
    def get_trend(self, symbol):
        """Calculate trend direction and strength"""
        history = list(self.price_history[symbol])
        if len(history) < TREND_PERIOD:
            return 0, 0
            
        prices = [h['price'] for h in history[-TREND_PERIOD:]]
        
        # Simple trend: compare first half vs second half
        first_half = sum(prices[:TREND_PERIOD//2]) / (TREND_PERIOD//2)
        second_half = sum(prices[TREND_PERIOD//2:]) / (TREND_PERIOD//2)
        
        change = (second_half - first_half) / first_half
        direction = 1 if change > 0 else -1
        strength = min(abs(change) * 100, 1.0)  # Normalize to 0-1
        
        return direction, strength
        
    def get_momentum(self, symbol):
        """Calculate short-term momentum"""
        history = list(self.price_history[symbol])
        if len(history) < MOMENTUM_PERIOD:
            return 0
            
        prices = [h['price'] for h in history[-MOMENTUM_PERIOD:]]
        change = (prices[-1] - prices[0]) / prices[0]
        
        return change

# =============================================================================
# PAPER TRADING ENGINE
# =============================================================================

class PaperPosition:
    """Represents a simulated position"""
    
    def __init__(self, symbol, side, entry_price, contracts, value):
        self.symbol = symbol
        self.side = side  # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.contracts = contracts
        self.value = value
        self.entry_time = time.time()
        
        # Targets
        self.target_price = None
        self.stop_price = None
        self.breakeven_activated = False
        self.trailing_activated = False
        self.trailing_stop = None
        
        # Tracking
        self.peak_pnl_pct = 0
        self.current_pnl_pct = 0
        
    def update(self, current_price):
        """Update position with current price"""
        if self.side == 'LONG':
            self.current_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            self.current_pnl_pct = (self.entry_price - current_price) / self.entry_price
            
        # Track peak
        if self.current_pnl_pct > self.peak_pnl_pct:
            self.peak_pnl_pct = self.current_pnl_pct
            
        # Breakeven logic
        if not self.breakeven_activated and self.current_pnl_pct >= BREAKEVEN_TRIGGER:
            self.breakeven_activated = True
            self.stop_price = self.entry_price
            
        # Trailing logic
        if self.current_pnl_pct >= TRAILING_TRIGGER:
            self.trailing_activated = True
            if self.side == 'LONG':
                new_trail = current_price * (1 - TRAILING_DISTANCE)
                if self.trailing_stop is None or new_trail > self.trailing_stop:
                    self.trailing_stop = new_trail
            else:
                new_trail = current_price * (1 + TRAILING_DISTANCE)
                if self.trailing_stop is None or new_trail < self.trailing_stop:
                    self.trailing_stop = new_trail
                    
    def should_exit(self, current_price):
        """Check if position should be closed"""
        # Hit target
        if self.side == 'LONG' and self.target_price and current_price >= self.target_price:
            return 'TARGET', current_price
        if self.side == 'SHORT' and self.target_price and current_price <= self.target_price:
            return 'TARGET', current_price
            
        # Hit stop
        if self.side == 'LONG' and self.stop_price and current_price <= self.stop_price:
            return 'STOP', current_price
        if self.side == 'SHORT' and self.stop_price and current_price >= self.stop_price:
            return 'STOP', current_price
            
        # Hit trailing stop
        if self.trailing_stop:
            if self.side == 'LONG' and current_price <= self.trailing_stop:
                return 'TRAIL', current_price
            if self.side == 'SHORT' and current_price >= self.trailing_stop:
                return 'TRAIL', current_price
                
        return None, None

# =============================================================================
# MAIN PAPER TRADING BOT
# =============================================================================

class AladdinPaper:
    """Paper trading bot with real market data"""
    
    def __init__(self):
        self.capital = PAPER_CAPITAL
        self.starting_capital = PAPER_CAPITAL
        self.positions = {}
        self.trade_history = []
        self.price_engine = PriceEngine(self.on_price_update)
        
        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.total_fees = 0
        self.best_trade = 0
        self.worst_trade = 0
        
        # State
        self.running = False
        self.last_signal_check = 0
        self.cooldown = {}  # Symbol -> last trade time
        
    def start(self):
        """Start the paper trading bot"""
        self.running = True
        self.print_banner()
        
        print("\n📡 Connecting to real market data...")
        self.price_engine.start()
        
        # Wait for prices
        time.sleep(3)
        
        print("\n" + "="*60)
        print("📝 PAPER TRADING ACTIVE")
        print("   Using REAL market data, SIMULATED money")
        print("   NO real trades will be placed!")
        print("="*60 + "\n")
        
        try:
            while self.running:
                self.check_signals()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Stop the bot and show summary"""
        self.running = False
        self.price_engine.stop()
        self.print_summary()
        
    def print_banner(self):
        """Print startup banner"""
        print("\n" + "╔" + "═"*70 + "╗")
        print("║" + " "*70 + "║")
        print("║" + "     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗".center(70) + "║")
        print("║" + "    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║".center(70) + "║")
        print("║" + "    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║".center(70) + "║")
        print("║" + "    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║".center(70) + "║")
        print("║" + "    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║".center(70) + "║")
        print("║" + "    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝".center(70) + "║")
        print("║" + " "*70 + "║")
        print("║" + "📝 PAPER TRADING MODE 📝".center(70) + "║")
        print("║" + " "*70 + "║")
        print("╠" + "═"*70 + "╣")
        print("║" + " "*70 + "║")
        print("║" + f"  💰 PAPER CAPITAL: ${self.capital:,.2f}".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("║" + "  📊 TRADEABLE ASSETS:".ljust(69) + "║")
        for symbol in ASSETS:
            print("║" + f"     • {symbol}".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("║" + f"  🎯 Strategy: Trend Following with Momentum".ljust(69) + "║")
        print("║" + f"  📈 Target: +{MIN_PROFIT_TARGET*100:.1f}% to +{MAX_PROFIT_TARGET*100:.1f}%".ljust(69) + "║")
        print("║" + f"  🛑 Stop: -{STOP_LOSS*100:.1f}%".ljust(69) + "║")
        print("║" + f"  💼 Risk per trade: {RISK_PER_TRADE*100:.0f}%".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("║" + "  ⚠️ NO REAL MONEY AT RISK - SIMULATION ONLY".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("║" + "  Press Ctrl+C to stop".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("╚" + "═"*70 + "╝")
        
    def on_price_update(self, symbol, price, old_price):
        """Handle price updates"""
        if symbol not in self.positions:
            return
            
        pos = self.positions[symbol]
        pos.update(price)
        
        # Check for exit
        exit_reason, exit_price = pos.should_exit(price)
        if exit_reason:
            self.close_position(symbol, exit_price, exit_reason)
            return
            
        # Display position status
        pnl_color = "🟢" if pos.current_pnl_pct > 0 else "🔴" if pos.current_pnl_pct < -0.001 else "🟡"
        hold_time = int(time.time() - pos.entry_time)
        
        # Calculate $ P&L
        gross_pnl = pos.value * pos.current_pnl_pct
        fees = pos.value * TAKER_FEE * 2  # Entry + exit fees
        net_pnl = gross_pnl - fees
        
        status = f"{pnl_color} {symbol} {pos.side} | ${price:.4f} | "
        status += f"Net: {pos.current_pnl_pct*100:+.2f}% (${net_pnl:+.4f}) | "
        status += f"Peak: {pos.peak_pnl_pct*100:+.2f}% | "
        status += f"⏱️ {hold_time}s"
        
        if pos.trailing_activated:
            status += " | 📈 TRAILING"
        elif pos.breakeven_activated:
            status += " | 🔒 BREAKEVEN"
            
        print(f"\r{status}".ljust(100), end='', flush=True)
        
    def check_signals(self):
        """Check for new trade signals"""
        if len(self.positions) >= MAX_POSITIONS:
            return
            
        now = time.time()
        if now - self.last_signal_check < 5:  # Check every 5 seconds
            return
        self.last_signal_check = now
        
        for symbol in ASSETS:
            # Skip if already in position or in cooldown
            if symbol in self.positions:
                continue
            if symbol in self.cooldown and now - self.cooldown[symbol] < 60:
                continue
                
            # Get price
            price = self.price_engine.prices.get(symbol)
            if not price:
                continue
                
            # Get trend and momentum
            direction, strength = self.price_engine.get_trend(symbol)
            momentum = self.price_engine.get_momentum(symbol)
            
            # Need strong trend
            if strength < MIN_TREND_STRENGTH:
                continue
                
            # Momentum should confirm trend
            if direction > 0 and momentum < 0.001:
                continue
            if direction < 0 and momentum > -0.001:
                continue
                
            # Calculate signal strength
            signal_strength = strength * (1 + abs(momentum) * 10)
            
            if signal_strength > 0.4:
                side = 'LONG' if direction > 0 else 'SHORT'
                self.open_position(symbol, side, price, signal_strength)
                
    def open_position(self, symbol, side, price, signal_strength):
        """Open a paper position"""
        info = ASSETS[symbol]
        
        # Position sizing based on risk
        risk_amount = self.capital * RISK_PER_TRADE
        contract_value = price * info['contract_size']
        max_loss_per_contract = contract_value * STOP_LOSS
        
        contracts = max(1, int(risk_amount / max_loss_per_contract))
        position_value = contracts * contract_value
        
        # Check if we have enough capital
        margin_required = position_value / info['leverage']
        if margin_required > self.capital * 0.5:
            contracts = 1
            position_value = contract_value
            
        # Create position
        pos = PaperPosition(symbol, side, price, contracts, position_value)
        
        # Set targets
        if side == 'LONG':
            pos.target_price = price * (1 + MIN_PROFIT_TARGET)
            pos.stop_price = price * (1 - STOP_LOSS)
        else:
            pos.target_price = price * (1 - MIN_PROFIT_TARGET)
            pos.stop_price = price * (1 + STOP_LOSS)
            
        self.positions[symbol] = pos
        
        # Entry fee
        entry_fee = position_value * TAKER_FEE
        self.total_fees += entry_fee
        
        print("\n")
        print("⚡" * 30)
        print(f"📝 PAPER TRADE OPENED")
        print()
        print(f"   📊 Asset:       {symbol}")
        print(f"   🎯 Direction:   {side}")
        print(f"   📈 Signal:      {signal_strength:.2f}")
        print()
        print(f"   💰 Position:")
        print(f"      Entry:      ${price:.4f}")
        print(f"      Contracts:  {contracts}")
        print(f"      Value:      ${position_value:.2f}")
        print()
        print(f"   🎯 Targets:")
        print(f"      Target:     ${pos.target_price:.4f} (+{MIN_PROFIT_TARGET*100:.1f}%)")
        print(f"      Stop:       ${pos.stop_price:.4f} (-{STOP_LOSS*100:.1f}%)")
        print()
        print(f"   💸 Entry Fee:  ${entry_fee:.4f}")
        print("⚡" * 30)
        print()
        
    def close_position(self, symbol, exit_price, reason):
        """Close a paper position"""
        pos = self.positions.pop(symbol)
        
        # Calculate P&L
        if pos.side == 'LONG':
            gross_pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.value
        else:
            gross_pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.value
            
        # Fees
        exit_fee = pos.value * TAKER_FEE
        total_fees = pos.value * TAKER_FEE * 2  # Entry + exit
        self.total_fees += exit_fee
        
        # Net P&L
        net_pnl = gross_pnl - total_fees
        pnl_pct = net_pnl / pos.value * 100
        
        # Update stats
        self.total_trades += 1
        self.total_pnl += net_pnl
        self.capital += net_pnl
        
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        if net_pnl > self.best_trade:
            self.best_trade = net_pnl
        if net_pnl < self.worst_trade:
            self.worst_trade = net_pnl
            
        # Record trade
        hold_time = time.time() - pos.entry_time
        self.trade_history.append({
            'symbol': symbol,
            'side': pos.side,
            'entry': pos.entry_price,
            'exit': exit_price,
            'contracts': pos.contracts,
            'gross_pnl': gross_pnl,
            'fees': total_fees,
            'net_pnl': net_pnl,
            'reason': reason,
            'hold_time': hold_time,
            'peak_pnl': pos.peak_pnl_pct
        })
        
        # Cooldown
        self.cooldown[symbol] = time.time()
        
        # Display
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        emoji = "🎉" if net_pnl > 0 else "💔"
        
        print("\n")
        print("=" * 60)
        print(f"   {emoji} {reason} - POSITION CLOSED")
        print()
        print(f"   {symbol} {pos.side} x{pos.contracts}")
        print(f"   Entry: ${pos.entry_price:.4f} → Exit: ${exit_price:.4f}")
        print(f"   Hold:  {hold_time:.0f}s ({hold_time/60:.1f} min)")
        print()
        print(f"   Peak P&L:  {pos.peak_pnl_pct*100:+.2f}%")
        print(f"   Exit P&L:  {pnl_pct:+.2f}%")
        print()
        print(f"   Gross:     ${gross_pnl:+.4f}")
        print(f"   Fees:      ${total_fees:.4f}")
        print(f"   NET:       ${net_pnl:+.4f}")
        print("=" * 60)
        print()
        print(f"📊 SESSION: Trades: {self.total_trades} | W:{self.winning_trades} L:{self.losing_trades} ({win_rate:.0f}%) | Total P&L: ${self.total_pnl:+.2f}")
        print()
        
    def print_summary(self):
        """Print session summary"""
        runtime = time.time() - self.price_engine.price_history[list(ASSETS.keys())[0]][0]['time'] if self.price_engine.price_history[list(ASSETS.keys())[0]] else 0
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        growth = ((self.capital - self.starting_capital) / self.starting_capital * 100)
        
        print("\n")
        print("╔" + "═"*70 + "╗")
        print("║" + "📊 PAPER TRADING SESSION SUMMARY".center(70) + "║")
        print("╠" + "═"*70 + "╣")
        print("║" + " "*70 + "║")
        print("║" + f"  ⏱️ RUNTIME: {runtime:.0f}s ({runtime/60:.1f} min)".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("║" + "  💰 CAPITAL:".ljust(69) + "║")
        print("║" + f"     Starting:     ${self.starting_capital:,.2f}".ljust(69) + "║")
        print("║" + f"     Current:      ${self.capital:,.2f}".ljust(69) + "║")
        print("║" + f"     Growth:       {growth:+.1f}%".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("║" + "  📈 TRADES:".ljust(69) + "║")
        print("║" + f"     Total:        {self.total_trades}".ljust(69) + "║")
        print("║" + f"     Wins:         {self.winning_trades}".ljust(69) + "║")
        print("║" + f"     Losses:       {self.losing_trades}".ljust(69) + "║")
        print("║" + f"     Win Rate:     {win_rate:.0f}%".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("║" + "  💵 P&L:".ljust(69) + "║")
        print("║" + f"     Total Fees:   ${self.total_fees:.2f}".ljust(69) + "║")
        print("║" + f"     Net P&L:      ${self.total_pnl:+.2f}".ljust(69) + "║")
        print("║" + f"     Best Trade:   ${self.best_trade:+.2f}".ljust(69) + "║")
        print("║" + f"     Worst Trade:  ${self.worst_trade:+.2f}".ljust(69) + "║")
        print("║" + " "*70 + "║")
        
        if self.trade_history:
            print("║" + "  📜 TRADE HISTORY:".ljust(69) + "║")
            for i, trade in enumerate(self.trade_history[-10:], 1):
                emoji = "✅" if trade['net_pnl'] > 0 else "❌"
                line = f"     {emoji} {trade['symbol']} {trade['side']}: ${trade['net_pnl']:+.2f} ({trade['reason']})"
                print("║" + line.ljust(69) + "║")
            print("║" + " "*70 + "║")
            
        print("║" + "  ⚠️ Remember: This was PAPER TRADING - no real money involved!".ljust(69) + "║")
        print("║" + " "*70 + "║")
        print("╚" + "═"*70 + "╝")
        
        # Strategy assessment
        print("\n")
        if self.total_trades >= 5:
            if win_rate >= 50 and self.total_pnl > 0:
                print("✅ STRATEGY ASSESSMENT: PROFITABLE")
                print("   This strategy shows promise! Consider paper trading longer")
                print("   or starting with small real capital when comfortable.")
            elif win_rate >= 40:
                print("🟡 STRATEGY ASSESSMENT: BREAKEVEN")
                print("   Strategy needs refinement. Adjust parameters and retest.")
            else:
                print("❌ STRATEGY ASSESSMENT: LOSING")
                print("   Current parameters are not profitable.")
                print("   DO NOT use real money with these settings!")
        else:
            print("⏳ NEED MORE DATA")
            print("   Run for longer to get meaningful statistics.")
            print("   Aim for at least 20-30 trades for reliable assessment.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n📝 ALADDIN PAPER TRADING BOT")
    print("   • Real market data from Delta Exchange")
    print("   • SIMULATED trades - NO real money")
    print("   • Test strategies safely before going live")
    print()
    print("   Press Ctrl+C to stop\n")
    
    bot = AladdinPaper()
    bot.start()
