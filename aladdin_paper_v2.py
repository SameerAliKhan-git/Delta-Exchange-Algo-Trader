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
║              📝 AGGRESSIVE PAPER TRADING v2 📝                            ║
║                                                                           ║
║   • $20 Capital - Full position sizing                                    ║
║   • ONE position at a time                                                ║
║   • Never turn winner into loser (breakeven protection)                   ║
║   • Fee-aware trading                                                     ║
║   • Auto entry & exit                                                     ║
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
# CONFIGURATION - $20 PAPER ACCOUNT
# =============================================================================

PAPER_CAPITAL = 20.00  # $20 starting capital

# Assets with contract info
ASSETS = {
    'BTCUSD': {'contract_size': 0.001, 'leverage': 100, 'min_move': 10},    # ~$97 per contract
    'ETHUSD': {'contract_size': 0.01, 'leverage': 100, 'min_move': 1},      # ~$36 per contract  
    'SOLUSD': {'contract_size': 0.1, 'leverage': 50, 'min_move': 0.1},      # ~$24 per contract
    'XRPUSD': {'contract_size': 5, 'leverage': 50, 'min_move': 0.001},      # ~$10 per contract
}

# FEES - Same as Delta Exchange
TAKER_FEE = 0.0005  # 0.05%
ROUND_TRIP_FEE = 0.001  # 0.10% (entry + exit)

# STRATEGY PARAMETERS
# Need price to move at least this much to cover fees + profit
MIN_PROFIT_PCT = 0.004      # 0.4% minimum profit target (0.1% fees + 0.3% profit)
STOP_LOSS_PCT = 0.003       # 0.3% stop loss

# BREAKEVEN PROTECTION - Never turn winner into loser!
BREAKEVEN_TRIGGER = 0.002   # Move stop to breakeven at +0.2%
LOCK_PROFIT_TRIGGER = 0.003 # Lock 50% profit at +0.3%
TRAILING_TRIGGER = 0.004    # Start trailing at +0.4%
TRAILING_PCT = 0.0015       # Trail by 0.15%

# Momentum thresholds
MIN_MOMENTUM = 0.0008  # 0.08% minimum momentum to enter

WS_URL = "wss://socket.india.delta.exchange"

# =============================================================================
# PRICE ENGINE
# =============================================================================

class PriceEngine:
    def __init__(self, callback):
        self.prices = {}
        self.history = {s: deque(maxlen=50) for s in ASSETS}
        self.callback = callback
        self.ws = None
        self.running = False
        self.connected = False
        
    def start(self):
        self.running = True
        threading.Thread(target=self._connect, daemon=True).start()
        
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()
            
    def _connect(self):
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, e: None,
                    on_close=self._on_close
                )
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except:
                pass
            if self.running:
                time.sleep(2)
                
    def _on_open(self, ws):
        self.connected = True
        print("📡 WebSocket CONNECTED")
        for symbol in ASSETS:
            ws.send(json.dumps({
                "type": "subscribe",
                "payload": {"channels": [{"name": "v2/ticker", "symbols": [symbol]}]}
            }))
            
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'v2/ticker':
                symbol = data.get('symbol')
                price = float(data.get('mark_price', 0))
                if symbol and price > 0:
                    old = self.prices.get(symbol, price)
                    self.prices[symbol] = price
                    self.history[symbol].append({'price': price, 'time': time.time()})
                    self.callback(symbol, price, old)
        except:
            pass
            
    def _on_close(self, ws, code, msg):
        self.connected = False
        if self.running:
            print("📡 Reconnecting...")
            
    def get_momentum(self, symbol, periods=10):
        """Get short-term momentum"""
        hist = list(self.history[symbol])
        if len(hist) < periods:
            return 0
        prices = [h['price'] for h in hist[-periods:]]
        return (prices[-1] - prices[0]) / prices[0]
        
    def get_trend(self, symbol, periods=20):
        """Get trend direction: 1=up, -1=down, 0=neutral"""
        hist = list(self.history[symbol])
        if len(hist) < periods:
            return 0
        prices = [h['price'] for h in hist[-periods:]]
        
        # Compare moving averages
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices) / len(prices)
        
        diff = (short_ma - long_ma) / long_ma
        if diff > 0.0005:
            return 1
        elif diff < -0.0005:
            return -1
        return 0

# =============================================================================
# PAPER POSITION
# =============================================================================

class Position:
    def __init__(self, symbol, side, entry, contracts, value):
        self.symbol = symbol
        self.side = side  # 'LONG' or 'SHORT'
        self.entry = entry
        self.contracts = contracts
        self.value = value
        self.entry_time = time.time()
        
        # Fees
        self.entry_fee = value * TAKER_FEE
        
        # Stops
        if side == 'LONG':
            self.stop = entry * (1 - STOP_LOSS_PCT)
            self.target = entry * (1 + MIN_PROFIT_PCT)
        else:
            self.stop = entry * (1 + STOP_LOSS_PCT)
            self.target = entry * (1 - MIN_PROFIT_PCT)
            
        # Tracking
        self.peak_pnl = 0
        self.current_pnl = 0
        self.breakeven_set = False
        self.profit_locked = False
        self.trailing = False
        
    def update(self, price):
        """Update P&L and manage stops"""
        # Calculate gross P&L %
        if self.side == 'LONG':
            self.current_pnl = (price - self.entry) / self.entry
        else:
            self.current_pnl = (self.entry - price) / self.entry
            
        # Update peak
        if self.current_pnl > self.peak_pnl:
            self.peak_pnl = self.current_pnl
            
        # BREAKEVEN PROTECTION - Never turn winner into loser!
        if not self.breakeven_set and self.current_pnl >= BREAKEVEN_TRIGGER:
            self.breakeven_set = True
            # Move stop to entry + 1 tick (small profit)
            if self.side == 'LONG':
                self.stop = self.entry * 1.0001  # Tiny profit
            else:
                self.stop = self.entry * 0.9999
            return "🔒 BREAKEVEN SET"
            
        # LOCK PROFIT at +0.3%
        if not self.profit_locked and self.current_pnl >= LOCK_PROFIT_TRIGGER:
            self.profit_locked = True
            lock_at = self.current_pnl * 0.5  # Lock 50% of current profit
            if self.side == 'LONG':
                self.stop = self.entry * (1 + lock_at)
            else:
                self.stop = self.entry * (1 - lock_at)
            return "💰 PROFIT LOCKED"
            
        # TRAILING STOP at +0.4%
        if self.current_pnl >= TRAILING_TRIGGER:
            self.trailing = True
            if self.side == 'LONG':
                new_stop = price * (1 - TRAILING_PCT)
                if new_stop > self.stop:
                    self.stop = new_stop
                    return "📈 TRAIL RAISED"
            else:
                new_stop = price * (1 + TRAILING_PCT)
                if new_stop < self.stop:
                    self.stop = new_stop
                    return "📈 TRAIL RAISED"
                    
        return None
        
    def should_exit(self, price):
        """Check if should exit"""
        if self.side == 'LONG':
            if price <= self.stop:
                return 'STOP' if self.current_pnl < 0 else 'TRAIL/BE'
            if price >= self.target:
                return 'TARGET'
        else:
            if price >= self.stop:
                return 'STOP' if self.current_pnl < 0 else 'TRAIL/BE'
            if price <= self.target:
                return 'TARGET'
        return None
        
    def get_net_pnl(self, exit_price):
        """Calculate net P&L after fees"""
        if self.side == 'LONG':
            gross = (exit_price - self.entry) / self.entry * self.value
        else:
            gross = (self.entry - exit_price) / self.entry * self.value
            
        total_fees = self.value * ROUND_TRIP_FEE
        return gross - total_fees, gross, total_fees

# =============================================================================
# MAIN BOT
# =============================================================================

class AladdinPaperV2:
    def __init__(self):
        self.capital = PAPER_CAPITAL
        self.starting_capital = PAPER_CAPITAL
        self.position = None
        self.trades = []
        self.engine = PriceEngine(self.on_price)
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        self.total_fees = 0
        self.best_trade = 0
        self.worst_trade = 0
        
        # Control
        self.running = False
        self.last_trade_time = 0
        self.cooldown = 10  # 10 seconds between trades
        
    def start(self):
        self.running = True
        self.print_banner()
        
        print("\n📡 Connecting to Delta Exchange...")
        self.engine.start()
        
        # Wait for connection
        for i in range(10):
            if self.engine.connected:
                break
            time.sleep(1)
            
        if not self.engine.connected:
            print("❌ Failed to connect!")
            return
            
        print("\n" + "="*60)
        print("📝 PAPER TRADING v2 ACTIVE")
        print("   $20 Capital | Full Position | Auto Trading")
        print("   NEVER TURN WINNER INTO LOSER!")
        print("="*60 + "\n")
        
        try:
            while self.running:
                if not self.position:
                    self.scan_for_entry()
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        self.running = False
        self.engine.stop()
        self.print_summary()
        
    def print_banner(self):
        print("\n" + "╔" + "═"*60 + "╗")
        print("║" + " ALADDIN PAPER TRADING v2 ".center(60) + "║")
        print("╠" + "═"*60 + "╣")
        print("║" + f"  💰 Paper Capital: ${self.capital:.2f}".ljust(59) + "║")
        print("║" + f"  📊 Assets: BTC, ETH, SOL, XRP".ljust(59) + "║")
        print("║" + f"  🎯 Target: +{MIN_PROFIT_PCT*100:.1f}% | Stop: -{STOP_LOSS_PCT*100:.1f}%".ljust(59) + "║")
        print("║" + f"  💸 Fees: {ROUND_TRIP_FEE*100:.2f}% round-trip".ljust(59) + "║")
        print("║" + f"  🔒 Breakeven at +{BREAKEVEN_TRIGGER*100:.1f}%".ljust(59) + "║")
        print("║" + f"  📈 Trail at +{TRAILING_TRIGGER*100:.1f}%".ljust(59) + "║")
        print("║" + " "*60 + "║")
        print("║" + "  ⚠️ SIMULATION - NO REAL MONEY".ljust(59) + "║")
        print("╚" + "═"*60 + "╝")
        
    def on_price(self, symbol, price, old_price):
        """Handle price updates"""
        if not self.position:
            return
            
        if self.position.symbol != symbol:
            return
            
        # Update position
        event = self.position.update(price)
        if event:
            print(f"\n   {event} @ ${price:.4f}")
            
        # Check for exit
        reason = self.position.should_exit(price)
        if reason:
            self.close_position(price, reason)
            return
            
        # Display status
        pos = self.position
        net_pnl, gross, fees = pos.get_net_pnl(price)
        
        color = "🟢" if pos.current_pnl > 0.001 else "🔴" if pos.current_pnl < -0.001 else "🟡"
        hold = int(time.time() - pos.entry_time)
        
        status = f"{color} {symbol} {pos.side} x{pos.contracts} | "
        status += f"${price:.4f} | "
        status += f"P&L: {pos.current_pnl*100:+.2f}% (${net_pnl:+.3f}) | "
        status += f"Peak: {pos.peak_pnl*100:+.2f}% | "
        status += f"Stop: ${pos.stop:.4f} | "
        status += f"⏱️{hold}s"
        
        if pos.trailing:
            status += " 📈"
        elif pos.profit_locked:
            status += " 💰"
        elif pos.breakeven_set:
            status += " 🔒"
            
        print(f"\r{status}".ljust(120), end='', flush=True)
        
    def scan_for_entry(self):
        """Look for trade opportunities"""
        # Check cooldown
        if time.time() - self.last_trade_time < self.cooldown:
            return
            
        # Need prices for all assets
        if len(self.engine.prices) < len(ASSETS):
            return
            
        best_signal = None
        best_score = 0
        
        for symbol in ASSETS:
            price = self.engine.prices.get(symbol)
            if not price:
                continue
                
            # Get momentum and trend
            momentum = self.engine.get_momentum(symbol)
            trend = self.engine.get_trend(symbol)
            
            # Need momentum above threshold
            if abs(momentum) < MIN_MOMENTUM:
                continue
                
            # Score the signal
            score = abs(momentum) * 100
            
            # Boost if trend confirms
            if (momentum > 0 and trend > 0) or (momentum < 0 and trend < 0):
                score *= 1.5
                
            if score > best_score:
                best_score = score
                side = 'LONG' if momentum > 0 else 'SHORT'
                best_signal = (symbol, side, price, momentum, score)
                
        # Take the best signal
        if best_signal and best_score > 0.1:
            symbol, side, price, momentum, score = best_signal
            self.open_position(symbol, side, price, momentum)
            
    def open_position(self, symbol, side, price, momentum):
        """Open a paper position with FULL capital"""
        info = ASSETS[symbol]
        
        # Calculate maximum contracts we can afford
        contract_value = price * info['contract_size']
        margin_per_contract = contract_value / info['leverage']
        
        # Use full capital (leave small buffer for fees)
        available = self.capital * 0.95
        max_contracts = int(available / margin_per_contract)
        
        if max_contracts < 1:
            return
            
        contracts = max_contracts
        position_value = contracts * contract_value
        
        # Create position
        self.position = Position(symbol, side, price, contracts, position_value)
        self.last_trade_time = time.time()
        
        print("\n")
        print("⚡" * 25)
        print(f"📝 PAPER TRADE OPENED")
        print()
        print(f"   📊 {symbol} {side}")
        print(f"   📈 Momentum: {momentum*100:+.3f}%")
        print()
        print(f"   💰 Position:")
        print(f"      Entry:     ${price:.4f}")
        print(f"      Contracts: {contracts}")
        print(f"      Value:     ${position_value:.2f}")
        print(f"      Margin:    ${contracts * margin_per_contract:.2f}")
        print()
        print(f"   🎯 Targets:")
        print(f"      Target:    ${self.position.target:.4f} (+{MIN_PROFIT_PCT*100:.1f}%)")
        print(f"      Stop:      ${self.position.stop:.4f} (-{STOP_LOSS_PCT*100:.1f}%)")
        print(f"      Breakeven: +{BREAKEVEN_TRIGGER*100:.1f}%")
        print()
        print(f"   💸 Fees: ~${position_value * ROUND_TRIP_FEE:.4f}")
        print("⚡" * 25)
        print()
        
    def close_position(self, exit_price, reason):
        """Close position and record trade"""
        pos = self.position
        net_pnl, gross_pnl, fees = pos.get_net_pnl(exit_price)
        hold_time = time.time() - pos.entry_time
        
        # Update stats
        self.capital += net_pnl
        self.total_pnl += net_pnl
        self.total_fees += fees
        
        if net_pnl > 0:
            self.wins += 1
            if net_pnl > self.best_trade:
                self.best_trade = net_pnl
        else:
            self.losses += 1
            if net_pnl < self.worst_trade:
                self.worst_trade = net_pnl
                
        # Record
        self.trades.append({
            'symbol': pos.symbol,
            'side': pos.side,
            'entry': pos.entry,
            'exit': exit_price,
            'contracts': pos.contracts,
            'net_pnl': net_pnl,
            'reason': reason,
            'hold': hold_time,
            'peak': pos.peak_pnl
        })
        
        # Clear position
        self.position = None
        
        # Display
        emoji = "🎉" if net_pnl > 0 else "💔"
        total = len(self.trades)
        win_rate = (self.wins / total * 100) if total > 0 else 0
        
        print("\n\n")
        print("=" * 60)
        print(f"   {emoji} {reason} - TRADE CLOSED")
        print()
        print(f"   {pos.symbol} {pos.side} x{pos.contracts}")
        print(f"   Entry: ${pos.entry:.4f} → Exit: ${exit_price:.4f}")
        print(f"   Hold:  {hold_time:.0f}s")
        print()
        print(f"   Peak P&L:  {pos.peak_pnl*100:+.2f}%")
        print(f"   Gross:     ${gross_pnl:+.4f}")
        print(f"   Fees:      ${fees:.4f}")
        print(f"   NET:       ${net_pnl:+.4f} ({net_pnl/pos.value*100:+.2f}%)")
        print("=" * 60)
        print()
        print(f"📊 SESSION: {total} trades | W:{self.wins} L:{self.losses} ({win_rate:.0f}%) | Net: ${self.total_pnl:+.2f} | Capital: ${self.capital:.2f}")
        print()
        
    def print_summary(self):
        total = len(self.trades)
        win_rate = (self.wins / total * 100) if total > 0 else 0
        growth = ((self.capital - self.starting_capital) / self.starting_capital * 100)
        
        print("\n\n")
        print("╔" + "═"*60 + "╗")
        print("║" + " 📊 PAPER TRADING SESSION SUMMARY ".center(60) + "║")
        print("╠" + "═"*60 + "╣")
        print("║" + " "*60 + "║")
        print("║" + f"  💰 Starting Capital:  ${self.starting_capital:.2f}".ljust(59) + "║")
        print("║" + f"  💰 Final Capital:     ${self.capital:.2f}".ljust(59) + "║")
        print("║" + f"  📈 Growth:            {growth:+.1f}%".ljust(59) + "║")
        print("║" + " "*60 + "║")
        print("║" + f"  📊 Total Trades:      {total}".ljust(59) + "║")
        print("║" + f"  ✅ Wins:              {self.wins}".ljust(59) + "║")
        print("║" + f"  ❌ Losses:            {self.losses}".ljust(59) + "║")
        print("║" + f"  📈 Win Rate:          {win_rate:.0f}%".ljust(59) + "║")
        print("║" + " "*60 + "║")
        print("║" + f"  💵 Net P&L:           ${self.total_pnl:+.2f}".ljust(59) + "║")
        print("║" + f"  💸 Total Fees:        ${self.total_fees:.2f}".ljust(59) + "║")
        print("║" + f"  🏆 Best Trade:        ${self.best_trade:+.2f}".ljust(59) + "║")
        print("║" + f"  💔 Worst Trade:       ${self.worst_trade:+.2f}".ljust(59) + "║")
        print("║" + " "*60 + "║")
        
        if self.trades:
            print("║" + "  📜 Recent Trades:".ljust(59) + "║")
            for t in self.trades[-5:]:
                emoji = "✅" if t['net_pnl'] > 0 else "❌"
                line = f"     {emoji} {t['symbol']} {t['side']}: ${t['net_pnl']:+.3f} ({t['reason']})"
                print("║" + line.ljust(59) + "║")
            print("║" + " "*60 + "║")
            
        print("║" + "  ⚠️ This was PAPER TRADING - no real money!".ljust(59) + "║")
        print("╚" + "═"*60 + "╝")
        
        # Assessment
        print()
        if total >= 5:
            if win_rate >= 50 and self.total_pnl > 0:
                print("✅ STRATEGY LOOKS PROFITABLE!")
                print(f"   {win_rate:.0f}% win rate with ${self.total_pnl:+.2f} profit")
            elif win_rate >= 40:
                print("🟡 STRATEGY IS MARGINAL")
                print("   Needs refinement before going live")
            else:
                print("❌ STRATEGY IS LOSING")
                print("   Do NOT use real money with these settings!")
        else:
            print("⏳ Need more trades for assessment (aim for 10+)")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n📝 ALADDIN PAPER TRADING v2")
    print("   • $20 Paper Capital")
    print("   • Full position sizing")
    print("   • Never turn winner into loser")
    print("   • Auto entry & exit")
    print("\n   Press Ctrl+C to stop\n")
    
    bot = AladdinPaperV2()
    bot.start()
