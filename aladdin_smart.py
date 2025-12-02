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
║              🧠 SMART SENTIMENT TRADING 🧠                                ║
║                                                                           ║
║   • Market Sentiment Analysis (Order Book Imbalance)                      ║
║   • Crypto Fear & Greed Index                                             ║
║   • Multi-Timeframe Trend Confirmation                                    ║
║   • Only trades when HIGH PROBABILITY setup exists                        ║
║   • PATIENT - Waits for the RIGHT moment                                  ║
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
# CONFIGURATION
# =============================================================================

API_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

PAPER_CAPITAL = 20.00

# Only trade BTC and ETH (most liquid, best for sentiment analysis)
ASSETS = {
    'BTCUSD': {'size': 0.001, 'lev': 100, 'product_id': 84},
    'ETHUSD': {'size': 0.01, 'lev': 100, 'product_id': 27},
}

# Fees
TOTAL_FEES = 0.001  # 0.10%

# Conservative Strategy - Only trade when HIGH CONFIDENCE
TARGET = 0.006       # +0.6% gross (+0.5% net)
STOP = 0.003         # -0.3% (2:1 reward ratio)
MIN_CONFIDENCE = 65  # Only trade above 65% confidence

# =============================================================================
# GLOBALS
# =============================================================================

prices = {}
orderbooks = {}
history = {s: deque(maxlen=200) for s in ASSETS}
position = None
capital = PAPER_CAPITAL
stats = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'fees': 0}
running = True
market_sentiment = {'fear_greed': 50, 'btc_trend': 'neutral', 'last_update': 0}
ws_connected = False

# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

def get_fear_greed_index():
    """Get Crypto Fear & Greed Index"""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            value = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            return value, classification
    except:
        pass
    return 50, "Neutral"

def get_btc_trend():
    """Analyze BTC trend from price history"""
    try:
        if 'BTCUSD' in history and len(history['BTCUSD']) > 20:
            prices_list = list(history['BTCUSD'])
            
            # Calculate EMAs
            ema_short = sum(prices_list[-10:]) / 10
            ema_long = sum(prices_list[-20:]) / 20
            current = prices_list[-1]
            
            # Trend determination
            if current > ema_short > ema_long:
                return 'bullish', (current - ema_long) / ema_long * 100
            elif current < ema_short < ema_long:
                return 'bearish', (ema_long - current) / ema_long * 100
            else:
                return 'neutral', 0
    except:
        pass
    return 'neutral', 0

def analyze_orderbook(symbol):
    """Analyze order book for buy/sell pressure"""
    if symbol not in orderbooks:
        return 50, 'balanced'
    
    ob = orderbooks[symbol]
    if not ob.get('bids') or not ob.get('asks'):
        return 50, 'balanced'
    
    try:
        bid_volume = sum(float(b[1]) for b in ob['bids'][:10])
        ask_volume = sum(float(a[1]) for a in ob['asks'][:10])
        
        total = bid_volume + ask_volume
        if total == 0:
            return 50, 'balanced'
        
        buy_pressure = (bid_volume / total) * 100
        
        if buy_pressure > 58:
            return buy_pressure, 'bullish'
        elif buy_pressure < 42:
            return buy_pressure, 'bearish'
        else:
            return buy_pressure, 'balanced'
    except:
        return 50, 'balanced'

def calculate_momentum(symbol):
    """Calculate price momentum"""
    if symbol not in history or len(history[symbol]) < 20:
        return 0, 'neutral'
    
    prices_list = list(history[symbol])
    
    # Short-term momentum (5 ticks)
    short_mom = (prices_list[-1] - prices_list[-5]) / prices_list[-5] * 100
    
    # Medium-term momentum (20 ticks)
    med_mom = (prices_list[-1] - prices_list[-20]) / prices_list[-20] * 100
    
    # Combined momentum score
    score = (short_mom * 0.6) + (med_mom * 0.4)
    
    if score > 0.08:
        return score, 'bullish'
    elif score < -0.08:
        return score, 'bearish'
    else:
        return score, 'neutral'

def get_market_sentiment():
    """Comprehensive market sentiment analysis"""
    global market_sentiment
    
    # Update Fear & Greed (every 5 minutes)
    if time.time() - market_sentiment['last_update'] > 300:
        fg_value, fg_class = get_fear_greed_index()
        market_sentiment['fear_greed'] = fg_value
        market_sentiment['fear_greed_class'] = fg_class
        market_sentiment['last_update'] = time.time()
        print(f"\n📊 Fear & Greed Update: {fg_value} ({fg_class})")
    
    return market_sentiment

def calculate_trade_confidence(symbol):
    """
    Calculate confidence score for a trade
    Returns: (confidence 0-100, direction 'LONG'/'SHORT'/None, reasons)
    """
    reasons = []
    bullish_score = 0
    bearish_score = 0
    
    # 1. Order Book Analysis (30 points max)
    ob_pressure, ob_sentiment = analyze_orderbook(symbol)
    if ob_sentiment == 'bullish':
        points = 30 * (ob_pressure - 50) / 50
        bullish_score += points
        reasons.append(f"📗 Order book: {ob_pressure:.0f}% buy pressure (+{points:.0f})")
    elif ob_sentiment == 'bearish':
        points = 30 * (50 - ob_pressure) / 50
        bearish_score += points
        reasons.append(f"📕 Order book: {ob_pressure:.0f}% buy pressure (+{points:.0f})")
    
    # 2. Price Momentum (25 points max)
    mom_score, mom_dir = calculate_momentum(symbol)
    if mom_dir == 'bullish':
        points = min(25, abs(mom_score) * 15)
        bullish_score += points
        reasons.append(f"📈 Momentum: +{mom_score:.3f}% (+{points:.0f})")
    elif mom_dir == 'bearish':
        points = min(25, abs(mom_score) * 15)
        bearish_score += points
        reasons.append(f"📉 Momentum: {mom_score:.3f}% (+{points:.0f})")
    
    # 3. BTC Trend (25 points max) - BTC leads the market
    btc_trend, btc_strength = get_btc_trend()
    if btc_trend == 'bullish':
        points = min(25, btc_strength * 8)
        bullish_score += points
        reasons.append(f"₿ BTC trend: UP +{btc_strength:.2f}% (+{points:.0f})")
    elif btc_trend == 'bearish':
        points = min(25, btc_strength * 8)
        bearish_score += points
        reasons.append(f"₿ BTC trend: DOWN -{btc_strength:.2f}% (+{points:.0f})")
    
    # 4. Fear & Greed (10 points max)
    sentiment = get_market_sentiment()
    fg = sentiment.get('fear_greed', 50)
    if fg > 55:  # Greed
        points = (fg - 50) * 0.2
        bullish_score += points
        reasons.append(f"😊 Fear/Greed: {fg} (Greed +{points:.0f})")
    elif fg < 45:  # Fear
        points = (50 - fg) * 0.2
        bearish_score += points
        reasons.append(f"😰 Fear/Greed: {fg} (Fear +{points:.0f})")
    
    # 5. Trend Alignment (10 points max)
    if symbol in history and len(history[symbol]) > 50:
        prices_list = list(history[symbol])
        trend_short = prices_list[-1] > prices_list[-10]
        trend_med = prices_list[-1] > prices_list[-30]
        trend_long = prices_list[-1] > prices_list[-50]
        
        aligned = sum([trend_short, trend_med, trend_long])
        if aligned == 3:
            bullish_score += 10
            reasons.append("📊 All timeframes UP (+10)")
        elif aligned == 0:
            bearish_score += 10
            reasons.append("📊 All timeframes DOWN (+10)")
    
    # Determine direction and confidence
    if bullish_score > bearish_score + 10:  # Need clear winner
        return min(100, bullish_score), 'LONG', reasons
    elif bearish_score > bullish_score + 10:
        return min(100, bearish_score), 'SHORT', reasons
    else:
        return max(bullish_score, bearish_score), None, ["⏳ No clear signal - waiting"]

# =============================================================================
# POSITION
# =============================================================================

class Position:
    def __init__(self, sym, side, entry, contracts, value, confidence, reasons):
        self.sym = sym
        self.side = side
        self.entry = entry
        self.contracts = contracts
        self.value = value
        self.confidence = confidence
        self.reasons = reasons
        self.start = time.time()
        
        self.total_fees = value * TOTAL_FEES
        self.gross_pnl = 0
        self.net_pnl = -self.total_fees / 2
        self.gross_pct = 0
        self.net_pct = 0
        self.peak = 0
        
    def update(self, price):
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
# WEBSOCKET
# =============================================================================

def on_open(ws):
    global ws_connected
    ws_connected = True
    print("\n📡 WebSocket CONNECTED")
    
    channels = [
        {"name": "v2/ticker", "symbols": list(ASSETS.keys())},
        {"name": "l2_orderbook", "symbols": list(ASSETS.keys())}
    ]
    ws.send(json.dumps({
        "type": "subscribe",
        "payload": {"channels": channels}
    }))

def on_message(ws, msg):
    global prices, orderbooks
    
    try:
        data = json.loads(msg)
        msg_type = data.get('type', '')
        
        if msg_type == 'v2/ticker':
            sym = data.get('symbol', '')
            mark = data.get('mark_price')
            
            if sym in ASSETS and mark:
                price = float(mark)
                prices[sym] = price
                history[sym].append(price)
                
                if position and position.sym == sym:
                    position.update(price)
                    display_position(price)
                    
        elif msg_type == 'l2_orderbook':
            sym = data.get('symbol', '')
            if sym in ASSETS:
                orderbooks[sym] = {
                    'bids': data.get('buy', []),
                    'asks': data.get('sell', [])
                }
    except:
        pass

def on_close(ws, code, msg):
    global ws_connected
    ws_connected = False

def on_error(ws, err):
    global ws_connected
    ws_connected = False

def run_websocket():
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
            time.sleep(2)

# =============================================================================
# DISPLAY
# =============================================================================

def display_position(price):
    pos = position
    hold = int(time.time() - pos.start)
    
    if pos.net_pnl > 0.05:
        color = "🟢"
    elif pos.net_pnl < -0.05:
        color = "🔴"
    else:
        color = "🟡"
    
    line = f"{color} {pos.sym} {pos.side} (Conf:{pos.confidence:.0f}%) | "
    line += f"${price:.2f} | "
    line += f"Gross: {pos.gross_pct*100:+.3f}% | "
    line += f"NET: ${pos.net_pnl:+.2f} | "
    line += f"Peak: {pos.peak*100:+.2f}% | "
    line += f"⏱️{hold}s"
    
    print(f"\r{line}".ljust(100), end='', flush=True)

# =============================================================================
# TRADING
# =============================================================================

def open_position(sym, side, price, confidence, reasons):
    global position
    
    info = ASSETS[sym]
    contract_val = price * info['size']
    margin = contract_val / info['lev']
    
    available = capital * 0.9
    contracts = max(1, int(available / margin))
    value = contracts * contract_val
    
    position = Position(sym, side, price, contracts, value, confidence, reasons)
    
    print("\n")
    print("🧠" * 25)
    print(f"📝 SMART TRADE OPENED")
    print()
    print(f"   📊 {sym} {side}")
    print(f"   🎯 Confidence: {confidence:.0f}%")
    print()
    print(f"   📈 Analysis:")
    for r in reasons:
        print(f"      • {r}")
    print()
    print(f"   💰 Position: {contracts} contracts @ ${price:.2f}")
    print(f"   💸 Value: ${value:.2f} | Fees: ${position.total_fees:.2f}")
    print()
    
    if side == 'LONG':
        target = price * (1 + TARGET)
        stop = price * (1 - STOP)
    else:
        target = price * (1 - TARGET)
        stop = price * (1 + STOP)
        
    print(f"   🎯 Target: ${target:.2f} (+{TARGET*100:.1f}%)")
    print(f"   🛑 Stop:   ${stop:.2f} (-{STOP*100:.1f}%)")
    print("🧠" * 25)
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
    
    print()
    print("\n" + "🚀" * 25)
    print(f"{emoji} TRADE CLOSED - {reason}")
    print()
    print(f"   📊 {pos.sym} {pos.side} (Entry Conf: {pos.confidence:.0f}%)")
    print(f"   ⏱️ Duration: {int(time.time() - pos.start)}s")
    print()
    print(f"   💰 P&L:")
    print(f"      Gross: ${pos.gross_pnl:+.2f} ({pos.gross_pct*100:+.3f}%)")
    print(f"      Fees:  -${pos.total_fees:.2f}")
    print(f"      NET:   ${pos.net_pnl:+.2f}")
    print()
    print(f"   📊 Capital: ${capital:.2f}")
    print(f"   📈 Session: ${stats['pnl']:+.2f}")
    print("🚀" * 25 + "\n")
    
    position = None

def manage_position():
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
    
    # Stop loss
    if pos.gross_pct <= -STOP:
        close_position("STOP LOSS 🛑", price)
        return
    
    # Trailing stop after 0.4%
    if pos.peak >= 0.004 and pos.gross_pct <= pos.peak - 0.002:
        close_position(f"TRAIL STOP (Peak: {pos.peak*100:.2f}%)", price)
        return

def find_entry():
    if position:
        manage_position()
        return
    
    for sym in ASSETS:
        if sym not in prices or len(history[sym]) < 50:
            continue
        
        confidence, direction, reasons = calculate_trade_confidence(sym)
        
        if direction and confidence >= MIN_CONFIDENCE:
            print(f"\n\n🎯 HIGH CONFIDENCE SIGNAL!")
            open_position(sym, direction, prices[sym], confidence, reasons)
            return

# =============================================================================
# MAIN
# =============================================================================

def main():
    global running
    
    print("\n" + "╔" + "═"*60 + "╗")
    print("║" + " 🧠 SMART SENTIMENT TRADING 🧠 ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print("║" + f"  💰 Capital: ${PAPER_CAPITAL:.2f}".ljust(59) + "║")
    print("║" + f"  🎯 Target: +{TARGET*100:.1f}% | Stop: -{STOP*100:.1f}%".ljust(59) + "║")
    print("║" + f"  📊 Min Confidence: {MIN_CONFIDENCE}%".ljust(59) + "║")
    print("║" + " "*60 + "║")
    print("║" + "  🔍 Sentiment Analysis:".ljust(59) + "║")
    print("║" + "     • Order Book Imbalance".ljust(59) + "║")
    print("║" + "     • Price Momentum".ljust(59) + "║")
    print("║" + "     • BTC Market Trend".ljust(59) + "║")
    print("║" + "     • Fear & Greed Index".ljust(59) + "║")
    print("║" + "     • Multi-Timeframe Alignment".ljust(59) + "║")
    print("║" + " "*60 + "║")
    print("║" + "  ⚠️ PAPER TRADING - NO REAL MONEY".ljust(59) + "║")
    print("╚" + "═"*60 + "╝")
    
    # Get initial sentiment
    print("\n📊 Fetching market sentiment...")
    fg_value, fg_class = get_fear_greed_index()
    print(f"   Fear & Greed Index: {fg_value} ({fg_class})")
    market_sentiment['fear_greed'] = fg_value
    market_sentiment['fear_greed_class'] = fg_class
    market_sentiment['last_update'] = time.time()
    
    # Start WebSocket
    print("\n📡 Connecting...")
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()
    
    time.sleep(5)
    
    print("\n" + "="*60)
    print("🧠 SMART TRADING ACTIVE")
    print("   Analyzing market conditions...")
    print("   Will trade only when confidence > 65%")
    print("="*60 + "\n")
    
    last_scan = 0
    last_status = 0
    
    try:
        while running:
            now = time.time()
            
            # Scan for opportunities every 3 seconds
            if now - last_scan > 3:
                find_entry()
                last_scan = now
            
            # Print status every 10 seconds if no position
            if not position and now - last_status > 10:
                status_parts = []
                for sym in ASSETS:
                    if sym in prices and len(history[sym]) > 50:
                        conf, dir, _ = calculate_trade_confidence(sym)
                        arrow = "↑" if dir == 'LONG' else "↓" if dir == 'SHORT' else "→"
                        status_parts.append(f"{sym}: ${prices[sym]:.0f} {arrow}{conf:.0f}%")
                
                if status_parts:
                    print(f"\r⏳ {' | '.join(status_parts)} | Waiting for {MIN_CONFIDENCE}%+ signal...".ljust(100), end='', flush=True)
                last_status = now
            
            time.sleep(0.5)
            
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
    print("╚" + "═"*60 + "╝")

if __name__ == "__main__":
    main()
