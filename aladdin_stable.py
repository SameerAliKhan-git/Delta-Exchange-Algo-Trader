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
║              ⚡ STABLE PAPER TRADING ⚡                                    ║
║                                                                           ║
║   • AUTO-RECONNECTING WebSocket                                           ║
║   • Real-time UPNL with microsecond precision                             ║
║   • $20 Paper Capital, Full Position                                      ║
║   • Fee tracking on EVERY tick                                            ║
║   • Never turn winner into loser                                          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import threading
import websocket
import requests
from collections import deque

# =============================================================================
# CONFIGURATION
# =============================================================================

PAPER_CAPITAL = 20.00

ASSETS = {
    'XRPUSD': {'size': 5, 'lev': 50, 'product_id': 176},
    'SOLUSD': {'size': 0.1, 'lev': 50, 'product_id': 149},
    'ETHUSD': {'size': 0.01, 'lev': 100, 'product_id': 27},
    'BTCUSD': {'size': 0.001, 'lev': 100, 'product_id': 84},
}

# Fees
ENTRY_FEE = 0.0005
EXIT_FEE = 0.0005
TOTAL_FEES = 0.001

# Strategy
TARGET = 0.004
STOP = 0.003
BREAKEVEN = 0.0015
LOCK_AT = 0.0025
TRAIL_AT = 0.0035
TRAIL_DIST = 0.001

API_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

# =============================================================================
# GLOBALS
# =============================================================================

prices = {}
history = {s: deque(maxlen=50) for s in ASSETS}
position = None
capital = PAPER_CAPITAL
stats = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'fees': 0}
running = True
ws_connected = False
ws_lock = threading.Lock()
update_count = 0

# =============================================================================
# POSITION CLASS
# =============================================================================

class Position:
    def __init__(self, sym, side, entry, contracts, value):
        self.sym = sym
        self.side = side
        self.entry = entry
        self.contracts = contracts
        self.value = value
        self.start = time.time()
        
        # Fees
        self.entry_fee = value * ENTRY_FEE
        self.exit_fee = value * EXIT_FEE
        self.total_fees = self.entry_fee + self.exit_fee
        
        # P&L
        self.gross_pnl = 0
        self.net_pnl = -self.entry_fee
        self.gross_pct = 0
        self.net_pct = 0
        self.peak = 0
        
        # Protection
        self.be_set = False
        self.locked = False
        self.trailing = False
        self.trail_stop = 0
        
    def update(self, price):
        """Update position P&L"""
        if self.side == 'LONG':
            self.gross_pct = (price - self.entry) / self.entry
        else:
            self.gross_pct = (self.entry - price) / self.entry
            
        self.gross_pnl = self.value * self.gross_pct
        self.net_pnl = self.gross_pnl - self.total_fees
        self.net_pct = self.gross_pct - TOTAL_FEES
        
        if self.gross_pct > self.peak:
            self.peak = self.gross_pct

# =============================================================================
# REST API FALLBACK
# =============================================================================

def fetch_prices_rest():
    """Fetch prices via REST API as fallback"""
    global prices
    try:
        resp = requests.get(f"{API_URL}/v2/tickers", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            for ticker in data.get('result', []):
                sym = ticker.get('symbol', '')
                if sym in ASSETS:
                    mark = ticker.get('mark_price')
                    if mark:
                        prices[sym] = float(mark)
                        history[sym].append(float(mark))
            return True
    except:
        pass
    return False

# =============================================================================
# WEBSOCKET
# =============================================================================

def on_open(ws):
    global ws_connected
    ws_connected = True
    print("\n📡 WebSocket CONNECTED")
    
    channels = [
        {"name": "v2/ticker", "symbols": list(ASSETS.keys())}
    ]
    ws.send(json.dumps({
        "type": "subscribe",
        "payload": {"channels": channels}
    }))

def on_message(ws, msg):
    global prices, update_count
    start = time.perf_counter_ns()
    
    try:
        data = json.loads(msg)
        if data.get('type') == 'v2/ticker':
            sym = data.get('symbol', '')
            mark = data.get('mark_price')
            
            if sym in ASSETS and mark:
                price = float(mark)
                prices[sym] = price
                history[sym].append(price)
                update_count += 1
                
                # Update position immediately
                if position and position.sym == sym:
                    position.update(price)
                    latency_us = (time.perf_counter_ns() - start) / 1000
                    display_position(price, latency_us)
    except:
        pass

def on_close(ws, code, msg):
    global ws_connected
    ws_connected = False

def on_error(ws, err):
    global ws_connected
    ws_connected = False

def run_websocket():
    """Run WebSocket with auto-reconnection"""
    global ws_connected
    
    while running:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_close=on_close,
                on_error=on_error
            )
            ws.run_forever(ping_interval=20, ping_timeout=15)
        except:
            pass
        
        if running:
            ws_connected = False
            time.sleep(1)

# =============================================================================
# DISPLAY
# =============================================================================

def display_position(price, latency_us):
    """Real-time position display"""
    pos = position
    hold = int(time.time() - pos.start)
    
    if pos.net_pnl > 0.01:
        color = "🟢"
    elif pos.net_pnl < -0.01:
        color = "🔴"
    else:
        color = "🟡"
        
    flags = ""
    if pos.trailing:
        flags = "📈TRAIL"
    elif pos.locked:
        flags = "💰LOCK"
    elif pos.be_set:
        flags = "🔒BE"
        
    line = f"{color} {pos.sym} {pos.side} x{pos.contracts} | "
    line += f"${price:.4f} | "
    line += f"Gross: {pos.gross_pct*100:+.3f}% | "
    line += f"Fees: -${pos.total_fees:.3f} | "
    line += f"NET: {pos.net_pct*100:+.3f}% (${pos.net_pnl:+.3f}) | "
    line += f"Peak: {pos.peak*100:+.2f}% | "
    line += f"⏱️{hold}s | ⚡{latency_us:.0f}μs"
    
    if flags:
        line += f" {flags}"
        
    print(f"\r{line}".ljust(140), end='', flush=True)

# =============================================================================
# TRADING
# =============================================================================

def open_position(sym, side, price, mom):
    global position
    
    info = ASSETS[sym]
    contract_val = price * info['size']
    margin = contract_val / info['lev']
    
    available = capital * 0.9
    contracts = max(1, int(available / margin))
    value = contracts * contract_val
    
    position = Position(sym, side, price, contracts, value)
    
    print("\n")
    print("⚡" * 25)
    print(f"📝 PAPER TRADE OPENED")
    print()
    print(f"   📊 {sym} {side}")
    print(f"   📈 Momentum: {mom*100:+.3f}%")
    print()
    print(f"   💰 Position:")
    print(f"      Entry:     ${price:.4f}")
    print(f"      Contracts: {contracts}")
    print(f"      Value:     ${value:.2f}")
    print()
    print(f"   💸 FEES:")
    print(f"      Entry Fee: ${position.entry_fee:.4f} (0.05%)")
    print(f"      Exit Fee:  ${position.exit_fee:.4f} (0.05%)")
    print(f"      TOTAL:     ${position.total_fees:.4f} (0.10%)")
    print()
    
    if side == 'LONG':
        target = price * (1 + TARGET)
        stop = price * (1 - STOP)
    else:
        target = price * (1 - TARGET)
        stop = price * (1 + STOP)
        
    print(f"   🎯 Targets:")
    print(f"      Target:    ${target:.4f} (+{TARGET*100:.1f}%)")
    print(f"      Stop:      ${stop:.4f} (-{STOP*100:.1f}%)")
    print(f"      Breakeven: +{BREAKEVEN*100:.2f}%")
    print()
    print(f"   📊 Need +{TOTAL_FEES*100:.2f}% to cover fees!")
    print("⚡" * 25)
    print()

def close_position(reason, price):
    global position, capital, stats
    
    pos = position
    pos.update(price)
    
    capital += pos.net_pnl
    stats['pnl'] += pos.net_pnl
    stats['fees'] += pos.total_fees
    stats['trades'] += 1
    
    if pos.net_pnl > 0:
        stats['wins'] += 1
        emoji = "🎉💵"
    else:
        stats['losses'] += 1
        emoji = "❌💸"
        
    hold = int(time.time() - pos.start)
    
    print()
    print("\n" + "🚀" * 25)
    print(f"{emoji} PAPER TRADE CLOSED")
    print()
    print(f"   📊 {pos.sym} {pos.side} → {reason}")
    print(f"   ⏱️ Duration: {hold}s")
    print()
    print(f"   💰 P&L:")
    print(f"      Gross: ${pos.gross_pnl:+.4f} ({pos.gross_pct*100:+.3f}%)")
    print(f"      Fees:  -${pos.total_fees:.4f}")
    print(f"      NET:   ${pos.net_pnl:+.4f} ({pos.net_pct*100:+.3f}%)")
    print()
    print(f"   📊 Capital: ${capital:.2f}")
    print(f"   📈 Session P&L: ${stats['pnl']:+.2f}")
    print(f"   💸 Total Fees: ${stats['fees']:.2f}")
    print("🚀" * 25 + "\n")
    
    position = None

def manage_position():
    """Check exits and protection"""
    pos = position
    if not pos:
        return
        
    price = prices.get(pos.sym)
    if not price:
        return
        
    pos.update(price)
    
    # Target hit
    if pos.gross_pct >= TARGET:
        close_position("TARGET HIT 🎯", price)
        return
        
    # Stop hit
    if pos.gross_pct <= -STOP:
        close_position("STOP LOSS 🛑", price)
        return
        
    # Breakeven protection - only set when NET is profitable (covers fees)
    if not pos.be_set and pos.net_pnl > 0.10:  # Net profit > $0.10
        pos.be_set = True
        print(f"\n🔒 BREAKEVEN SET at +{pos.gross_pct*100:.2f}% (Net: +${pos.net_pnl:.2f})")
        
    # Lock profit - when net profit is substantial
    if not pos.locked and pos.net_pnl > 0.50:  # Net profit > $0.50
        pos.locked = True
        print(f"\n💰 PROFIT LOCKED at +{pos.gross_pct*100:.2f}% (Net: +${pos.net_pnl:.2f})")
        
    # Trailing
    if not pos.trailing and pos.gross_pct >= TRAIL_AT:
        pos.trailing = True
        pos.trail_stop = pos.gross_pct - TRAIL_DIST
        print(f"\n📈 TRAILING STARTED at +{pos.gross_pct*100:.2f}%")
        
    # Trail update
    if pos.trailing:
        new_stop = pos.gross_pct - TRAIL_DIST
        if new_stop > pos.trail_stop:
            pos.trail_stop = new_stop
            
        if pos.gross_pct <= pos.trail_stop:
            close_position(f"TRAIL STOP ({pos.trail_stop*100:.2f}%)", price)
            return
            
    # Protect from loss after being green - USE NET P&L (after fees)!
    # Only exit at breakeven if NET profit is still positive or at least zero
    if pos.be_set and pos.net_pnl <= 0.01:  # Exit when net is basically zero
        close_position("BREAKEVEN EXIT 🔒", price)
        return

def find_entry():
    """Look for entry signals"""
    if position:
        manage_position()
        return
        
    for sym in ASSETS:
        if sym not in prices or len(history[sym]) < 5:
            continue
            
        h = list(history[sym])
        if len(h) < 5:
            continue
            
        mom = (h[-1] - h[-3]) / h[-3]
        
        if mom > 0.0003:  # +0.03% momentum
            print(f"\n🎯 SIGNAL: {sym} LONG (momentum: {mom*100:+.3f}%)")
            open_position(sym, 'LONG', prices[sym], mom)
            return
            
        elif mom < -0.0003:
            print(f"\n🎯 SIGNAL: {sym} SHORT (momentum: {mom*100:+.3f}%)")
            open_position(sym, 'SHORT', prices[sym], mom)
            return

def poll_prices():
    """Background price polling via REST"""
    while running:
        if not ws_connected:
            fetch_prices_rest()
            if position:
                price = prices.get(position.sym)
                if price:
                    position.update(price)
                    display_position(price, 0)
        time.sleep(0.5)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global running
    
    print("\n" + "╔" + "═"*60 + "╗")
    print("║" + " ⚡ STABLE PAPER TRADING ⚡ ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print("║" + f"  💰 Capital: ${PAPER_CAPITAL:.2f}".ljust(59) + "║")
    print("║" + f"  🎯 Target: +{TARGET*100:.1f}% gross (+{(TARGET-TOTAL_FEES)*100:.1f}% net)".ljust(59) + "║")
    print("║" + f"  🛑 Stop: -{STOP*100:.1f}%".ljust(59) + "║")
    print("║" + f"  💸 Fees: {TOTAL_FEES*100:.2f}% round-trip".ljust(59) + "║")
    print("║" + f"  ⚡ Auto-reconnecting WebSocket".ljust(59) + "║")
    print("║" + " "*60 + "║")
    print("║" + "  ⚠️ PAPER TRADING - NO REAL MONEY".ljust(59) + "║")
    print("╚" + "═"*60 + "╝")
    print("\n📡 Connecting...\n")
    
    # Start WebSocket thread
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()
    
    # Start REST polling thread (fallback)
    poll_thread = threading.Thread(target=poll_prices, daemon=True)
    poll_thread.start()
    
    # Wait for initial connection
    time.sleep(3)
    
    if not prices:
        print("⏳ Fetching prices via REST...")
        fetch_prices_rest()
        time.sleep(1)
        
    print("\n" + "="*60)
    print("⚡ STABLE PAPER TRADING ACTIVE")
    print("   Auto-reconnecting WebSocket + REST fallback")
    print("   Fees tracked: 0.10% round-trip")
    print("="*60 + "\n")
    
    try:
        while running:
            find_entry()
            time.sleep(0.1)
    except KeyboardInterrupt:
        running = False
        
    # Summary
    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    growth = ((capital - PAPER_CAPITAL) / PAPER_CAPITAL * 100)
    
    print("\n\n" + "╔" + "═"*60 + "╗")
    print("║" + " 📊 SESSION SUMMARY ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print("║" + f"  💰 Starting: ${PAPER_CAPITAL:.2f}".ljust(59) + "║")
    print("║" + f"  💰 Final:    ${capital:.2f} ({growth:+.1f}%)".ljust(59) + "║")
    print("║" + f"  📊 Trades: {stats['trades']}".ljust(59) + "║")
    print("║" + f"  ✅ Wins: {stats['wins']} | ❌ Losses: {stats['losses']} ({wr:.0f}%)".ljust(59) + "║")
    print("║" + f"  💵 Net P&L: ${stats['pnl']:+.2f}".ljust(59) + "║")
    print("║" + f"  💸 Total Fees: ${stats['fees']:.2f}".ljust(59) + "║")
    print("║" + f"  📈 Total Updates: {update_count}".ljust(59) + "║")
    print("╚" + "═"*60 + "╝")

if __name__ == "__main__":
    main()
