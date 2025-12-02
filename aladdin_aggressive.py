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
║                    🚀 AGGRESSIVE PAPER TRADER 🚀                          ║
║                                                                           ║
║   • $20 Capital - Uses FULL capital                                       ║
║   • ONE position at a time                                                ║
║   • IMMEDIATE entry on first signal                                       ║
║   • Never turn winner into loser                                          ║
║   • Fee-aware profit targets                                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import threading
import requests
import websocket
from collections import deque

# =============================================================================
# $20 PAPER ACCOUNT
# =============================================================================

PAPER_CAPITAL = 20.00

ASSETS = {
    'XRPUSD': {'contract_size': 5, 'leverage': 50},      # Best for $20 - ~$10/contract
    'SOLUSD': {'contract_size': 0.1, 'leverage': 50},    # ~$24/contract
    'ETHUSD': {'contract_size': 0.01, 'leverage': 100},  # ~$36/contract
    'BTCUSD': {'contract_size': 0.001, 'leverage': 100}, # ~$97/contract
}

# FEES
TAKER_FEE = 0.0005  # 0.05%
ROUND_TRIP = 0.001  # 0.10%

# STRATEGY - Need profit > fees
TARGET_PCT = 0.004      # 0.4% target (covers 0.1% fees + 0.3% profit)
STOP_PCT = 0.003        # 0.3% stop
BREAKEVEN_AT = 0.0015   # Move to breakeven at +0.15%
LOCK_PROFIT_AT = 0.0025 # Lock profit at +0.25%
TRAIL_AT = 0.0035       # Trail at +0.35%
TRAIL_DISTANCE = 0.001  # 0.1% trail distance

WS_URL = "wss://socket.india.delta.exchange"

# =============================================================================
# GLOBALS
# =============================================================================

prices = {}
price_history = {s: deque(maxlen=30) for s in ASSETS}
position = None
stats = {
    'capital': PAPER_CAPITAL,
    'trades': 0,
    'wins': 0,
    'losses': 0,
    'pnl': 0,
    'fees': 0,
    'best': 0,
    'worst': 0
}
ws_connected = False
running = True

# =============================================================================
# POSITION CLASS
# =============================================================================

class Position:
    def __init__(self, symbol, side, entry, contracts, value):
        self.symbol = symbol
        self.side = side
        self.entry = entry
        self.contracts = contracts
        self.value = value
        self.start = time.time()
        
        # Set initial stops
        if side == 'LONG':
            self.stop = entry * (1 - STOP_PCT)
            self.target = entry * (1 + TARGET_PCT)
        else:
            self.stop = entry * (1 + STOP_PCT)
            self.target = entry * (1 - TARGET_PCT)
            
        self.peak = 0
        self.pnl = 0
        self.breakeven = False
        self.locked = False
        self.trailing = False
        
    def update(self, price):
        # Calculate P&L
        if self.side == 'LONG':
            self.pnl = (price - self.entry) / self.entry
        else:
            self.pnl = (self.entry - price) / self.entry
            
        # Track peak
        if self.pnl > self.peak:
            self.peak = self.pnl
            
        # NEVER TURN WINNER INTO LOSER!
        if not self.breakeven and self.pnl >= BREAKEVEN_AT:
            self.breakeven = True
            if self.side == 'LONG':
                self.stop = self.entry * 1.0001
            else:
                self.stop = self.entry * 0.9999
            print(f"\n   🔒 BREAKEVEN SET @ ${self.stop:.4f}")
            
        # Lock profit
        if not self.locked and self.pnl >= LOCK_PROFIT_AT:
            self.locked = True
            lock = self.pnl * 0.4  # Lock 40%
            if self.side == 'LONG':
                self.stop = self.entry * (1 + lock)
            else:
                self.stop = self.entry * (1 - lock)
            print(f"\n   💰 PROFIT LOCKED @ ${self.stop:.4f}")
            
        # Trailing
        if self.pnl >= TRAIL_AT:
            self.trailing = True
            if self.side == 'LONG':
                new_stop = price * (1 - TRAIL_DISTANCE)
                if new_stop > self.stop:
                    self.stop = new_stop
            else:
                new_stop = price * (1 + TRAIL_DISTANCE)
                if new_stop < self.stop:
                    self.stop = new_stop
                    
    def check_exit(self, price):
        if self.side == 'LONG':
            if price <= self.stop:
                return 'STOP' if self.pnl < 0 else 'PROTECTED'
            if price >= self.target:
                return 'TARGET'
        else:
            if price >= self.stop:
                return 'STOP' if self.pnl < 0 else 'PROTECTED'
            if price <= self.target:
                return 'TARGET'
        return None
        
    def get_pnl(self, exit_price):
        if self.side == 'LONG':
            gross = (exit_price - self.entry) / self.entry * self.value
        else:
            gross = (self.entry - exit_price) / self.entry * self.value
        fees = self.value * ROUND_TRIP
        return gross - fees, gross, fees

# =============================================================================
# WEBSOCKET
# =============================================================================

def on_open(ws):
    global ws_connected
    ws_connected = True
    print("📡 CONNECTED to Delta Exchange")
    for symbol in ASSETS:
        ws.send(json.dumps({
            "type": "subscribe",
            "payload": {"channels": [{"name": "v2/ticker", "symbols": [symbol]}]}
        }))

def on_message(ws, message):
    global position
    try:
        data = json.loads(message)
        if data.get('type') == 'v2/ticker':
            symbol = data.get('symbol')
            price = float(data.get('mark_price', 0))
            if symbol and price > 0:
                prices[symbol] = price
                price_history[symbol].append(price)
                
                # Handle position
                if position and position.symbol == symbol:
                    handle_position(price)
                    
    except Exception as e:
        pass

def on_close(ws, code, msg):
    global ws_connected
    ws_connected = False
    if running:
        print("📡 Reconnecting...")

def on_error(ws, error):
    pass

# =============================================================================
# TRADING LOGIC
# =============================================================================

def handle_position(price):
    global position, stats
    
    pos = position
    pos.update(price)
    
    # Check exit
    reason = pos.check_exit(price)
    if reason:
        close_position(price, reason)
        return
        
    # Display
    net, _, _ = pos.get_pnl(price)
    hold = int(time.time() - pos.start)
    
    color = "🟢" if pos.pnl > 0.001 else "🔴" if pos.pnl < -0.001 else "🟡"
    
    status = f"{color} {pos.symbol} {pos.side} x{pos.contracts} | "
    status += f"${price:.4f} | "
    status += f"P&L: {pos.pnl*100:+.2f}% (${net:+.3f}) | "
    status += f"Peak: {pos.peak*100:+.2f}% | "
    status += f"Stop: ${pos.stop:.4f} | "
    status += f"⏱️{hold}s"
    
    if pos.trailing:
        status += " 📈"
    elif pos.locked:
        status += " 💰"
    elif pos.breakeven:
        status += " 🔒"
        
    print(f"\r{status}".ljust(110), end='', flush=True)

def open_position(symbol, side, price):
    global position, stats
    
    info = ASSETS[symbol]
    contract_value = price * info['contract_size']
    margin_per = contract_value / info['leverage']
    
    # Use full capital
    available = stats['capital'] * 0.9
    contracts = max(1, int(available / margin_per))
    value = contracts * contract_value
    
    position = Position(symbol, side, price, contracts, value)
    
    print("\n")
    print("⚡" * 20)
    print(f"📝 OPENED: {symbol} {side}")
    print(f"   Entry: ${price:.4f} | Contracts: {contracts} | Value: ${value:.2f}")
    print(f"   Target: ${position.target:.4f} (+{TARGET_PCT*100:.1f}%)")
    print(f"   Stop: ${position.stop:.4f} (-{STOP_PCT*100:.1f}%)")
    print(f"   Fees: ~${value * ROUND_TRIP:.3f}")
    print("⚡" * 20)
    print()

def close_position(price, reason):
    global position, stats
    
    pos = position
    net, gross, fees = pos.get_pnl(price)
    hold = time.time() - pos.start
    
    stats['capital'] += net
    stats['trades'] += 1
    stats['pnl'] += net
    stats['fees'] += fees
    
    if net > 0:
        stats['wins'] += 1
        if net > stats['best']:
            stats['best'] = net
    else:
        stats['losses'] += 1
        if net < stats['worst']:
            stats['worst'] = net
            
    position = None
    
    emoji = "🎉" if net > 0 else "💔"
    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    
    print("\n\n")
    print("=" * 55)
    print(f"   {emoji} {reason}")
    print(f"   {pos.symbol} {pos.side}: ${pos.entry:.4f} → ${price:.4f}")
    print(f"   Hold: {hold:.0f}s | Peak: {pos.peak*100:+.2f}%")
    print(f"   Gross: ${gross:+.3f} | Fees: ${fees:.3f} | NET: ${net:+.3f}")
    print("=" * 55)
    print(f"📊 Trades: {stats['trades']} | W:{stats['wins']} L:{stats['losses']} ({wr:.0f}%) | P&L: ${stats['pnl']:+.2f} | Capital: ${stats['capital']:.2f}")
    print()

def find_entry():
    """Find immediate entry opportunity"""
    if position:
        return
        
    # Need at least 5 price points
    best = None
    best_score = 0
    
    for symbol in ASSETS:
        if symbol not in prices:
            continue
        hist = list(price_history[symbol])
        if len(hist) < 5:
            continue
            
        price = prices[symbol]
        
        # Calculate momentum
        momentum = (hist[-1] - hist[0]) / hist[0]
        
        # Simple trend: are we moving?
        recent = (hist[-1] - hist[-3]) / hist[-3] if len(hist) >= 3 else 0
        
        # Score
        score = abs(momentum) + abs(recent) * 2
        
        if score > best_score:
            best_score = score
            side = 'LONG' if momentum > 0 else 'SHORT'
            best = (symbol, side, price, momentum)
            
    # Take any signal with movement
    if best and best_score > 0.0001:
        symbol, side, price, mom = best
        print(f"\n🎯 SIGNAL: {symbol} {side} (momentum: {mom*100:+.3f}%)")
        open_position(symbol, side, price)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global running
    
    print("\n" + "╔" + "═"*55 + "╗")
    print("║" + " 🚀 AGGRESSIVE PAPER TRADER 🚀 ".center(55) + "║")
    print("╠" + "═"*55 + "╣")
    print("║" + f"  💰 Capital: ${PAPER_CAPITAL:.2f}".ljust(54) + "║")
    print("║" + f"  🎯 Target: +{TARGET_PCT*100:.1f}% | Stop: -{STOP_PCT*100:.1f}%".ljust(54) + "║")
    print("║" + f"  🔒 Breakeven at +{BREAKEVEN_AT*100:.2f}%".ljust(54) + "║")
    print("║" + f"  💸 Fees: {ROUND_TRIP*100:.2f}%".ljust(54) + "║")
    print("║" + "  ⚠️ PAPER TRADING - NO REAL MONEY".ljust(54) + "║")
    print("╚" + "═"*55 + "╝")
    print("\n📡 Connecting...\n")
    
    # Start WebSocket
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error
    )
    
    ws_thread = threading.Thread(target=lambda: ws.run_forever(ping_interval=20), daemon=True)
    ws_thread.start()
    
    # Wait for connection
    for _ in range(10):
        if ws_connected:
            break
        time.sleep(1)
        
    if not ws_connected:
        print("❌ Connection failed!")
        return
        
    print("\n" + "="*55)
    print("📝 PAPER TRADING ACTIVE - FULL CAPITAL MODE")
    print("   Will enter trade on first momentum signal!")
    print("="*55 + "\n")
    
    # Wait for prices
    print("⏳ Waiting for price data...\n")
    time.sleep(3)
    
    try:
        while running:
            find_entry()
            time.sleep(0.3)
    except KeyboardInterrupt:
        running = False
        
    # Summary
    print("\n\n" + "╔" + "═"*55 + "╗")
    print("║" + " 📊 SESSION SUMMARY ".center(55) + "║")
    print("╠" + "═"*55 + "╣")
    print("║" + f"  💰 Final Capital: ${stats['capital']:.2f}".ljust(54) + "║")
    growth = ((stats['capital'] - PAPER_CAPITAL) / PAPER_CAPITAL * 100)
    print("║" + f"  📈 Growth: {growth:+.1f}%".ljust(54) + "║")
    print("║" + f"  📊 Trades: {stats['trades']}".ljust(54) + "║")
    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    print("║" + f"  ✅ Wins: {stats['wins']} | ❌ Losses: {stats['losses']} ({wr:.0f}%)".ljust(54) + "║")
    print("║" + f"  💵 Net P&L: ${stats['pnl']:+.2f}".ljust(54) + "║")
    print("║" + f"  💸 Fees Paid: ${stats['fees']:.2f}".ljust(54) + "║")
    print("╚" + "═"*55 + "╝")

if __name__ == "__main__":
    main()
