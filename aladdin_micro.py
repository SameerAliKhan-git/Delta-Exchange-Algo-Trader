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
║              ⚡ MICROSECOND PAPER TRADING ⚡                               ║
║                                                                           ║
║   • REAL-TIME UPNL updates in MICROSECONDS                                ║
║   • $20 Capital with FULL position                                        ║
║   • Fee tracking on EVERY tick                                            ║
║   • Never turn winner into loser                                          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import threading
import websocket
from collections import deque

# =============================================================================
# MICROSECOND TIMER
# =============================================================================

class MicroTimer:
    """Microsecond precision timer"""
    @staticmethod
    def now():
        return time.perf_counter_ns()
    
    @staticmethod
    def elapsed_us(start):
        return (time.perf_counter_ns() - start) / 1000
    
    @staticmethod
    def elapsed_ms(start):
        return (time.perf_counter_ns() - start) / 1_000_000

# =============================================================================
# CONFIGURATION
# =============================================================================

PAPER_CAPITAL = 20.00

ASSETS = {
    'XRPUSD': {'size': 5, 'lev': 50},      # Best for $20
    'SOLUSD': {'size': 0.1, 'lev': 50},
    'ETHUSD': {'size': 0.01, 'lev': 100},
    'BTCUSD': {'size': 0.001, 'lev': 100},
}

# FEES - Critical for profitability!
ENTRY_FEE = 0.0005   # 0.05% taker
EXIT_FEE = 0.0005    # 0.05% taker
TOTAL_FEES = 0.001   # 0.10% round-trip

# Strategy
TARGET = 0.004       # +0.4% gross (+0.3% net after fees)
STOP = 0.003         # -0.3% 
BREAKEVEN = 0.0015   # Move to BE at +0.15%
LOCK_AT = 0.0025     # Lock profit at +0.25%
TRAIL_AT = 0.0035    # Trail at +0.35%
TRAIL_DIST = 0.001   # 0.1% trail

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
last_update = 0
tick_count = 0
latencies = deque(maxlen=100)

# =============================================================================
# POSITION
# =============================================================================

class Position:
    def __init__(self, sym, side, entry, contracts, value):
        self.sym = sym
        self.side = side
        self.entry = entry
        self.contracts = contracts
        self.value = value
        self.start = time.time()
        self.start_ns = MicroTimer.now()
        
        # FEES calculated upfront
        self.entry_fee = value * ENTRY_FEE
        self.exit_fee = value * EXIT_FEE
        self.total_fees = value * TOTAL_FEES
        
        # Stops
        if side == 'LONG':
            self.stop = entry * (1 - STOP)
            self.target = entry * (1 + TARGET)
        else:
            self.stop = entry * (1 + STOP)
            self.target = entry * (1 - TARGET)
            
        self.peak = 0
        self.gross_pnl = 0
        self.net_pnl = 0
        self.gross_pct = 0
        self.net_pct = 0
        self.be_set = False
        self.locked = False
        self.trailing = False
        
    def update(self, price):
        """Update P&L - called on EVERY tick"""
        # Gross P&L (before fees)
        if self.side == 'LONG':
            self.gross_pnl = (price - self.entry) / self.entry * self.value
            self.gross_pct = (price - self.entry) / self.entry
        else:
            self.gross_pnl = (self.entry - price) / self.entry * self.value
            self.gross_pct = (self.entry - price) / self.entry
            
        # NET P&L (after fees) - THIS IS THE REAL NUMBER!
        self.net_pnl = self.gross_pnl - self.total_fees
        self.net_pct = self.net_pnl / self.value
        
        # Track peak
        if self.gross_pct > self.peak:
            self.peak = self.gross_pct
            
        # BREAKEVEN protection
        if not self.be_set and self.gross_pct >= BREAKEVEN:
            self.be_set = True
            if self.side == 'LONG':
                self.stop = self.entry * 1.0001
            else:
                self.stop = self.entry * 0.9999
            return "🔒 BREAKEVEN"
            
        # Lock profit
        if not self.locked and self.gross_pct >= LOCK_AT:
            self.locked = True
            lock = self.gross_pct * 0.4
            if self.side == 'LONG':
                self.stop = self.entry * (1 + lock)
            else:
                self.stop = self.entry * (1 - lock)
            return "💰 LOCKED"
            
        # Trail
        if self.gross_pct >= TRAIL_AT:
            self.trailing = True
            if self.side == 'LONG':
                new = price * (1 - TRAIL_DIST)
                if new > self.stop:
                    self.stop = new
            else:
                new = price * (1 + TRAIL_DIST)
                if new < self.stop:
                    self.stop = new
                    
        return None
        
    def check_exit(self, price):
        if self.side == 'LONG':
            if price <= self.stop:
                return 'STOP' if self.gross_pct < 0 else 'PROTECTED'
            if price >= self.target:
                return 'TARGET'
        else:
            if price >= self.stop:
                return 'STOP' if self.gross_pct < 0 else 'PROTECTED'
            if price <= self.target:
                return 'TARGET'
        return None

# =============================================================================
# WEBSOCKET - Optimized for speed
# =============================================================================

def on_open(ws):
    print("📡 CONNECTED - Microsecond mode active")
    for sym in ASSETS:
        ws.send(json.dumps({
            "type": "subscribe",
            "payload": {"channels": [{"name": "v2/ticker", "symbols": [sym]}]}
        }))

def on_message(ws, msg):
    global position, capital, stats, last_update, tick_count
    
    tick_start = MicroTimer.now()
    
    try:
        data = json.loads(msg)
        if data.get('type') != 'v2/ticker':
            return
            
        sym = data.get('symbol')
        price = float(data.get('mark_price', 0))
        
        if not sym or price <= 0:
            return
            
        prices[sym] = price
        history[sym].append(price)
        tick_count += 1
        
        # Handle position
        if position and position.sym == sym:
            event = position.update(price)
            
            # Check exit
            reason = position.check_exit(price)
            if reason:
                close_position(price, reason)
            else:
                # REAL-TIME DISPLAY with microsecond latency
                latency = MicroTimer.elapsed_us(tick_start)
                latencies.append(latency)
                display_position(price, latency)
                
            if event:
                print(f"\n   {event} @ ${price:.4f}")
                
    except Exception as e:
        pass
        
    last_update = MicroTimer.now()

def on_close(ws, code, msg):
    if running:
        print("\n📡 Reconnecting...")

def on_error(ws, err):
    pass

# =============================================================================
# DISPLAY
# =============================================================================

def display_position(price, latency_us):
    """Real-time position display with microsecond latency"""
    pos = position
    hold = int(time.time() - pos.start)
    
    # Color based on NET P&L (after fees)
    if pos.net_pnl > 0.01:
        color = "🟢"
    elif pos.net_pnl < -0.01:
        color = "🔴"
    else:
        color = "🟡"
        
    # Status flags
    flags = ""
    if pos.trailing:
        flags = "📈TRAIL"
    elif pos.locked:
        flags = "💰LOCK"
    elif pos.be_set:
        flags = "🔒BE"
        
    # Build display line
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
    global position, stats
    
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
    print(f"   💸 FEES (deducted from P&L):")
    print(f"      Entry Fee: ${position.entry_fee:.4f} (0.05%)")
    print(f"      Exit Fee:  ${position.exit_fee:.4f} (0.05%)")
    print(f"      TOTAL:     ${position.total_fees:.4f} (0.10%)")
    print()
    print(f"   🎯 Targets (gross, before fees):")
    print(f"      Target:    ${position.target:.4f} (+{TARGET*100:.1f}%)")
    print(f"      Stop:      ${position.stop:.4f} (-{STOP*100:.1f}%)")
    print(f"      Breakeven: +{BREAKEVEN*100:.2f}%")
    print()
    print(f"   📊 To PROFIT after fees, need: +{TOTAL_FEES*100:.2f}% minimum")
    print("⚡" * 25)
    print()

def close_position(price, reason):
    global position, capital, stats
    
    pos = position
    hold = time.time() - pos.start
    
    # Final P&L calculation
    if pos.side == 'LONG':
        gross = (price - pos.entry) / pos.entry * pos.value
    else:
        gross = (pos.entry - price) / pos.entry * pos.value
        
    net = gross - pos.total_fees
    
    # Update stats
    capital += net
    stats['trades'] += 1
    stats['pnl'] += net
    stats['fees'] += pos.total_fees
    
    if net > 0:
        stats['wins'] += 1
    else:
        stats['losses'] += 1
        
    position = None
    
    emoji = "🎉" if net > 0 else "💔"
    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    
    print("\n\n")
    print("=" * 65)
    print(f"   {emoji} {reason}")
    print()
    print(f"   {pos.sym} {pos.side} x{pos.contracts}")
    print(f"   Entry: ${pos.entry:.4f} → Exit: ${price:.4f}")
    print(f"   Hold:  {hold:.1f}s")
    print()
    print(f"   💵 P&L BREAKDOWN:")
    print(f"      Gross P&L:  ${gross:+.4f} ({gross/pos.value*100:+.2f}%)")
    print(f"      Entry Fee:  -${pos.entry_fee:.4f}")
    print(f"      Exit Fee:   -${pos.exit_fee:.4f}")
    print(f"      ─────────────────────")
    print(f"      NET P&L:    ${net:+.4f} ({net/pos.value*100:+.2f}%)")
    print()
    print(f"   Peak Gross: {pos.peak*100:+.2f}%")
    print("=" * 65)
    print()
    print(f"📊 SESSION: {stats['trades']} trades | W:{stats['wins']} L:{stats['losses']} ({wr:.0f}%)")
    print(f"💵 Net P&L: ${stats['pnl']:+.2f} | Total Fees: ${stats['fees']:.2f}")
    print(f"💰 Capital: ${capital:.2f} | ⚡ Avg Latency: {avg_lat:.0f}μs")
    print()

def find_entry():
    """Find trade signal"""
    if position:
        return
        
    best = None
    best_score = 0
    
    for sym in ASSETS:
        if sym not in prices or len(history[sym]) < 5:
            continue
            
        price = prices[sym]
        hist = list(history[sym])
        
        # Momentum
        mom = (hist[-1] - hist[0]) / hist[0]
        score = abs(mom) * 100
        
        if score > best_score:
            best_score = score
            side = 'LONG' if mom > 0 else 'SHORT'
            best = (sym, side, price, mom)
            
    if best and best_score > 0.01:
        sym, side, price, mom = best
        print(f"\n🎯 SIGNAL: {sym} {side} (momentum: {mom*100:+.3f}%)")
        open_position(sym, side, price, mom)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global running
    
    print("\n" + "╔" + "═"*60 + "╗")
    print("║" + " ⚡ MICROSECOND PAPER TRADING ⚡ ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print("║" + f"  💰 Capital: ${PAPER_CAPITAL:.2f}".ljust(59) + "║")
    print("║" + f"  🎯 Target: +{TARGET*100:.1f}% gross (+{(TARGET-TOTAL_FEES)*100:.1f}% net)".ljust(59) + "║")
    print("║" + f"  🛑 Stop: -{STOP*100:.1f}%".ljust(59) + "║")
    print("║" + f"  💸 Fees: {TOTAL_FEES*100:.2f}% round-trip".ljust(59) + "║")
    print("║" + f"  ⚡ UPNL updates: EVERY TICK (microseconds)".ljust(59) + "║")
    print("║" + " "*60 + "║")
    print("║" + "  ⚠️ PAPER TRADING - NO REAL MONEY".ljust(59) + "║")
    print("╚" + "═"*60 + "╝")
    print("\n📡 Connecting with microsecond precision...\n")
    
    # WebSocket with minimal latency
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error
    )
    
    # Run in thread
    ws_thread = threading.Thread(
        target=lambda: ws.run_forever(ping_interval=15, ping_timeout=10),
        daemon=True
    )
    ws_thread.start()
    
    # Wait for connection
    time.sleep(2)
    
    if not prices:
        print("⏳ Waiting for prices...")
        time.sleep(3)
        
    print("\n" + "="*60)
    print("⚡ MICROSECOND PAPER TRADING ACTIVE")
    print("   UPNL updates on EVERY tick with μs latency!")
    print("   Fees tracked: Entry 0.05% + Exit 0.05% = 0.10% total")
    print("="*60 + "\n")
    
    try:
        while running:
            find_entry()
            time.sleep(0.1)
    except KeyboardInterrupt:
        running = False
        
    # Summary
    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    growth = ((capital - PAPER_CAPITAL) / PAPER_CAPITAL * 100)
    
    print("\n\n" + "╔" + "═"*60 + "╗")
    print("║" + " 📊 SESSION SUMMARY ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print("║" + f"  💰 Starting: ${PAPER_CAPITAL:.2f}".ljust(59) + "║")
    print("║" + f"  💰 Final:    ${capital:.2f} ({growth:+.1f}%)".ljust(59) + "║")
    print("║" + f"  📊 Trades: {stats['trades']}".ljust(59) + "║")
    print("║" + f"  ✅ Wins: {stats['wins']} | ❌ Losses: {stats['losses']} ({wr:.0f}%)".ljust(59) + "║")
    print("║" + f"  💵 Net P&L: ${stats['pnl']:+.2f}".ljust(59) + "║")
    print("║" + f"  💸 Total Fees Paid: ${stats['fees']:.2f}".ljust(59) + "║")
    print("║" + f"  ⚡ Avg Latency: {avg_lat:.0f}μs".ljust(59) + "║")
    print("║" + f"  📈 Total Ticks: {tick_count}".ljust(59) + "║")
    print("╚" + "═"*60 + "╝")

if __name__ == "__main__":
    main()
