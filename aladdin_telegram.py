#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗                      ║
║    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                      ║
║    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║                      ║
║    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║                      ║
║    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║                      ║
║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝                      ║
║                                                                               ║
║        💎💎💎 WEALTH BUILDER + TELEGRAM CONTROL 💎💎💎                        ║
║                                                                               ║
║   ════════════════════════════════════════════════════════════════════════    ║
║                                                                               ║
║   📱 TELEGRAM COMMANDS:                                                       ║
║      /start - Start trading                                                   ║
║      /stop - Stop trading                                                     ║
║      /status - Check balance & positions                                      ║
║      /trades - View trade history                                             ║
║      /settings - View current settings                                        ║
║      /kill - Shutdown bot completely                                          ║
║                                                                               ║
║   🔔 AUTO NOTIFICATIONS:                                                      ║
║      • Trade opened alerts                                                    ║
║      • Trade closed alerts with P&L                                           ║
║      • Daily summary reports                                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import hashlib
import hmac
import threading
import requests
import websocket
from datetime import datetime
from collections import deque
from statistics import mean, stdev
import sys
import os

# Fast output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# TELEGRAM CONFIGURATION
# =============================================================================

TELEGRAM_TOKEN = "8231064948:AAHh2bLGNriAv_z6YI4U1gt8T2xONX8LV-Y"
TELEGRAM_CHAT_ID = None  # Will be set when user sends first message
TELEGRAM_ENABLED = True

# =============================================================================
# API CONFIGURATION - SAMMY SUB-ACCOUNT
# =============================================================================

API_KEY = "vu55c1iSzUMwQSZwmPEMgHUVfJGpXo"
API_SECRET = "gXzg0GMwqLTOhyzuSKf9G4OlOF1sX8RSJrcuPy0A98YofCWGcKEh8DoForAF"
API_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

# =============================================================================
# ASSET CONFIGURATION (Sorted by minimum margin required)
# =============================================================================

ASSETS = {
    'XRPUSD': {'product_id': 185, 'contract_size': 10, 'leverage': 50, 'tick': 0.0001, 'min_margin': 0.05},
    'SOLUSD': {'product_id': 146, 'contract_size': 0.1, 'leverage': 50, 'tick': 0.01, 'min_margin': 0.50},
    'ETHUSD': {'product_id': 27, 'contract_size': 0.01, 'leverage': 100, 'tick': 0.05, 'min_margin': 0.40},
    'BTCUSD': {'product_id': 84, 'contract_size': 0.001, 'leverage': 100, 'tick': 0.5, 'min_margin': 1.00},
}

# =============================================================================
# STRATEGY CONFIGURATION
# =============================================================================

MAKER_FEE = 0.0002
TAKER_FEE = 0.0005
TOTAL_FEE = 0.001

MAX_RISK_PER_TRADE = 0.03
MAX_DAILY_LOSS = 0.08
MAX_DRAWDOWN = 0.05
MIN_CONFIDENCE = 65           # LOWERED to 65% for more trades
MAX_POSITIONS = 2
SCAN_INTERVAL = 1

BASE_TARGET = 0.008
BASE_STOP = 0.004
BREAKEVEN_TRIGGER = 0.003
TRAIL_TRIGGER = 0.005
TRAIL_DISTANCE = 0.002

# =============================================================================
# GLOBAL STATE
# =============================================================================

prices = {}
orderbooks = {}
price_history = {s: deque(maxlen=500) for s in ASSETS}
positions = {}

account_balance = 0
starting_balance = 0
daily_pnl = 0
daily_start_balance = 0

market_sentiment = {
    'fear_greed': 50,
    'fear_greed_class': 'Neutral',
    'news_score': 0,
    'global_trend': 'neutral',
    'volatility': 'normal',
    'last_update': 0
}

stats = {
    'trades': 0, 'wins': 0, 'losses': 0,
    'gross_pnl': 0, 'fees_paid': 0, 'net_pnl': 0,
    'peak_balance': 0, 'max_drawdown': 0,
    'best_trade': 0, 'worst_trade': 0,
    'strategies': {}
}

trade_history = []

running = True
trading_active = True  # Can be toggled via Telegram
ws_connected = False
last_telegram_update = 0

# =============================================================================
# TELEGRAM FUNCTIONS
# =============================================================================

def telegram_send(message):
    """Send message to Telegram"""
    global TELEGRAM_CHAT_ID
    
    if not TELEGRAM_ENABLED or not TELEGRAM_CHAT_ID:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, json=data, timeout=10)
        return resp.status_code == 200
    except:
        return False

def telegram_get_updates():
    """Get updates from Telegram"""
    global last_telegram_update, TELEGRAM_CHAT_ID, trading_active, running
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        params = {"offset": last_telegram_update + 1, "timeout": 1}
        resp = requests.get(url, params=params, timeout=5)
        
        if resp.status_code != 200:
            return
        
        data = resp.json()
        if not data.get('ok'):
            return
        
        for update in data.get('result', []):
            last_telegram_update = update['update_id']
            
            message = update.get('message', {})
            chat_id = message.get('chat', {}).get('id')
            text = message.get('text', '').strip().lower()
            
            if chat_id:
                # Set chat ID if not set
                if not TELEGRAM_CHAT_ID:
                    TELEGRAM_CHAT_ID = chat_id
                    telegram_send("🤖 <b>ALADDIN BOT CONNECTED!</b>\n\nI'm now linked to this chat.\n\nCommands:\n/start - Start trading\n/stop - Stop trading\n/status - Check status\n/trades - Trade history\n/settings - View settings\n/kill - Shutdown bot")
                    print(f"📱 Telegram connected! Chat ID: {chat_id}")
                
                # Process commands
                if text == '/start':
                    trading_active = True
                    telegram_send("🚀 <b>TRADING STARTED!</b>\n\nBot is now actively looking for trades.")
                    print("📱 Telegram: Trading STARTED")
                    
                elif text == '/stop':
                    trading_active = False
                    telegram_send("🛑 <b>TRADING STOPPED!</b>\n\nBot is paused. Existing positions will be managed.")
                    print("📱 Telegram: Trading STOPPED")
                    
                elif text == '/status':
                    send_status_message()
                    
                elif text == '/trades':
                    send_trades_message()
                    
                elif text == '/settings':
                    send_settings_message()
                    
                elif text == '/kill':
                    telegram_send("💀 <b>SHUTTING DOWN BOT...</b>\n\nGoodbye!")
                    print("📱 Telegram: KILL command received")
                    running = False
                    
                elif text.startswith('/'):
                    telegram_send("❓ Unknown command.\n\nAvailable:\n/start /stop /status /trades /settings /kill")
    except:
        pass

def send_status_message():
    """Send status to Telegram"""
    wr = stats['wins'] / max(1, stats['trades']) * 100
    
    pos_text = ""
    if positions:
        for sym, pos in positions.items():
            pos_text += f"\n   • {sym} {pos.side}: ${pos.net_pnl:+.4f}"
    else:
        pos_text = "\n   No open positions"
    
    affordable = get_affordable_assets()
    assets_str = ', '.join([f"{a['symbol']}" for a in affordable]) if affordable else "None"
    
    msg = f"""📊 <b>STATUS REPORT</b>

💰 <b>Account:</b>
   Balance: ${account_balance:.4f}
   Daily P&L: ${daily_pnl:+.4f}
   Total P&L: ${stats['net_pnl']:+.4f}

📈 <b>Performance:</b>
   Trades: {stats['trades']}
   Win Rate: {wr:.1f}%
   Wins: {stats['wins']} | Losses: {stats['losses']}

📍 <b>Positions:</b>{pos_text}

🌍 <b>Market:</b>
   Fear & Greed: {market_sentiment['fear_greed']} ({market_sentiment['fear_greed_class']})
   Trend: {market_sentiment['global_trend']}
   Tradeable: {assets_str}

🤖 <b>Bot Status:</b> {'🟢 ACTIVE' if trading_active else '🔴 PAUSED'}"""
    
    telegram_send(msg)

def send_trades_message():
    """Send recent trades to Telegram"""
    if not trade_history:
        telegram_send("📜 <b>TRADE HISTORY</b>\n\nNo trades yet.")
        return
    
    recent = trade_history[-10:]  # Last 10 trades
    
    msg = "📜 <b>RECENT TRADES</b>\n"
    for t in reversed(recent):
        emoji = "✅" if t['pnl'] > 0 else "❌"
        msg += f"\n{emoji} {t['symbol']} {t['side']}: ${t['pnl']:+.4f} ({t['strategy']})"
    
    telegram_send(msg)

def send_settings_message():
    """Send settings to Telegram"""
    msg = f"""⚙️ <b>BOT SETTINGS</b>

🎯 <b>Trading:</b>
   Min Confidence: {MIN_CONFIDENCE}%
   Max Positions: {MAX_POSITIONS}
   Scan Interval: {SCAN_INTERVAL}s

💰 <b>Risk:</b>
   Max Risk/Trade: {MAX_RISK_PER_TRADE*100}%
   Max Daily Loss: {MAX_DAILY_LOSS*100}%
   Max Drawdown: {MAX_DRAWDOWN*100}%

📊 <b>Targets:</b>
   Base Target: {BASE_TARGET*100}%
   Base Stop: {BASE_STOP*100}%
   Breakeven: {BREAKEVEN_TRIGGER*100}%
   Trail Start: {TRAIL_TRIGGER*100}%"""
    
    telegram_send(msg)

def send_trade_alert(action, symbol, side, price, pnl=None, confidence=None, strategy=None, reasons=None):
    """Send trade alert to Telegram"""
    if action == 'OPEN':
        reasons_text = "\n".join([f"   • {r}" for r in (reasons or [])])
        msg = f"""🚀 <b>TRADE OPENED</b>

📊 {symbol} <b>{side}</b>
💰 Entry: ${price:.4f}
🎯 Confidence: {confidence}%
📈 Strategy: {strategy}

<b>Analysis:</b>
{reasons_text}"""
    
    elif action == 'CLOSE':
        emoji = "🎉" if pnl > 0 else "❌"
        msg = f"""{emoji} <b>TRADE CLOSED</b>

📊 {symbol} {side}
💰 Exit: ${price:.4f}
💵 P&L: <b>${pnl:+.4f}</b>
📈 Strategy: {strategy}"""
    
    telegram_send(msg)

def run_telegram():
    """Telegram polling thread"""
    while running:
        try:
            telegram_get_updates()
        except:
            pass
        time.sleep(2)

# =============================================================================
# API FUNCTIONS
# =============================================================================

def sign_request(method, endpoint, payload=""):
    timestamp = str(int(time.time()))
    message = method + timestamp + endpoint + payload
    signature = hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    return {
        'api-key': API_KEY,
        'timestamp': timestamp,
        'signature': signature,
        'Content-Type': 'application/json'
    }

def api_get(endpoint):
    try:
        resp = requests.get(API_URL + endpoint, headers=sign_request('GET', endpoint), timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def api_post(endpoint, data):
    try:
        payload = json.dumps(data)
        resp = requests.post(API_URL + endpoint, headers=sign_request('POST', endpoint, payload), data=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def get_account_balance():
    global account_balance, starting_balance, daily_start_balance
    
    resp = api_get('/v2/wallet/balances')
    if resp and resp.get('success'):
        for asset in resp.get('result', []):
            if asset.get('asset_symbol') in ['USDT', 'USD']:
                account_balance = float(asset.get('available_balance', 0))
                if starting_balance == 0:
                    starting_balance = account_balance
                    daily_start_balance = account_balance
                    stats['peak_balance'] = account_balance
                return account_balance
    return 0

# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

def fetch_fear_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return int(data['data'][0]['value']), data['data'][0]['value_classification']
    except:
        pass
    return 50, "Neutral"

def fetch_news_sentiment():
    bullish_keywords = ['bullish', 'rally', 'surge', 'buy', 'breakout', 'adoption', 'etf', 'institutional', 'higher']
    bearish_keywords = ['bearish', 'crash', 'dump', 'sell', 'breakdown', 'ban', 'hack', 'fraud', 'lower']
    
    score = 0
    try:
        resp = requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC,ETH", timeout=5)
        if resp.status_code == 200:
            news = resp.json().get('Data', [])[:15]
            for item in news:
                text = (item.get('title', '') + ' ' + item.get('body', '')[:200]).lower()
                bull = sum(1 for kw in bullish_keywords if kw in text)
                bear = sum(1 for kw in bearish_keywords if kw in text)
                if bull + bear > 0:
                    score += (bull - bear) / (bull + bear)
    except:
        pass
    return score / 3 if score else 0

def update_market_sentiment():
    global market_sentiment
    
    now = time.time()
    if now - market_sentiment['last_update'] < 180:
        return
    
    print("\n📊 Updating market intelligence...")
    
    fg, fg_class = fetch_fear_greed()
    market_sentiment['fear_greed'] = fg
    market_sentiment['fear_greed_class'] = fg_class
    
    news = fetch_news_sentiment()
    market_sentiment['news_score'] = news
    
    if 'BTCUSD' in price_history and len(price_history['BTCUSD']) > 50:
        btc = list(price_history['BTCUSD'])
        ema20 = sum(btc[-20:]) / 20
        ema50 = sum(btc[-50:]) / 50
        if btc[-1] > ema20 > ema50:
            market_sentiment['global_trend'] = 'bullish'
        elif btc[-1] < ema20 < ema50:
            market_sentiment['global_trend'] = 'bearish'
        else:
            market_sentiment['global_trend'] = 'neutral'
    
    market_sentiment['last_update'] = now
    
    print(f"   😰 Fear & Greed: {fg} ({fg_class})")
    print(f"   📰 News Score: {news:+.2f}")
    print(f"   📈 BTC Trend: {market_sentiment['global_trend']}")

# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

def ema(prices_list, period):
    if len(prices_list) < period:
        return None
    mult = 2 / (period + 1)
    result = sum(prices_list[:period]) / period
    for p in prices_list[period:]:
        result = (p - result) * mult + result
    return result

def rsi(prices_list, period=14):
    if len(prices_list) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(-period, 0):
        diff = prices_list[i] - prices_list[i-1]
        gains.append(max(0, diff))
        losses.append(max(0, -diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(prices_list, period=20):
    if len(prices_list) < period:
        return None, None, None
    recent = list(prices_list)[-period:]
    mid = mean(recent)
    std = stdev(recent) if len(recent) > 1 else 0
    return mid + 2*std, mid, mid - 2*std

def atr(symbol, period=14):
    if len(price_history[symbol]) < period + 1:
        return None
    prices_list = list(price_history[symbol])
    trs = []
    for i in range(-period, 0):
        tr = abs(prices_list[i] - prices_list[i-1])
        trs.append(tr)
    return mean(trs)

def orderbook_pressure(symbol):
    if symbol not in orderbooks:
        return 0, False, False
    
    ob = orderbooks[symbol]
    bids = ob.get('bids', [])[:20]
    asks = ob.get('asks', [])[:20]
    
    if not bids or not asks:
        return 0, False, False
    
    try:
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total = bid_vol + ask_vol
        
        if total == 0:
            return 0, False, False
        
        imbalance = (bid_vol - ask_vol) / total * 100
        
        avg_bid = bid_vol / len(bids)
        avg_ask = ask_vol / len(asks)
        buy_wall = any(float(b[1]) > avg_bid * 4 for b in bids[:5])
        sell_wall = any(float(a[1]) > avg_ask * 4 for a in asks[:5])
        
        return imbalance, buy_wall, sell_wall
    except:
        return 0, False, False

def momentum_score(symbol):
    if len(price_history[symbol]) < 100:
        return 0, 'neutral'
    
    prices_list = list(price_history[symbol])
    curr = prices_list[-1]
    
    short_mom = (curr - prices_list[-10]) / prices_list[-10] * 100
    med_mom = (curr - prices_list[-30]) / prices_list[-30] * 100
    long_mom = (curr - prices_list[-100]) / prices_list[-100] * 100
    
    score = short_mom * 0.5 + med_mom * 0.3 + long_mom * 0.2
    
    if score > 0.08:  # LOWERED threshold
        return score, 'bullish'
    elif score < -0.08:  # LOWERED threshold
        return score, 'bearish'
    return score, 'neutral'

# =============================================================================
# 5 WINNING STRATEGIES - IMPROVED FOR MORE SIGNALS
# =============================================================================

def strategy_trend_following(symbol):
    """Strategy 1: Trend Following - IMPROVED"""
    if len(price_history[symbol]) < 100:
        return None, 0, []
    
    prices_list = list(price_history[symbol])
    curr = prices_list[-1]
    
    ema9 = ema(prices_list, 9)
    ema21 = ema(prices_list, 21)
    ema50 = ema(prices_list, 50)
    
    if not all([ema9, ema21, ema50]):
        return None, 0, []
    
    reasons = []
    conf = 0
    direction = None
    
    # LONG: EMAs aligned bullish
    if ema9 > ema21 > ema50 and curr > ema9:
        direction = 'LONG'
        conf = 45  # Increased base
        reasons.append(f"📈 Bullish trend (EMA9 > EMA21 > EMA50)")
        
        mom, _ = momentum_score(symbol)
        if mom > 0.05:  # Lowered
            conf += 12
            reasons.append(f"🚀 Momentum +{mom:.2f}%")
        
        imb, wall, _ = orderbook_pressure(symbol)
        if imb > 10:  # Lowered
            conf += 10
            reasons.append(f"📗 Buy pressure {imb:.0f}%")
        if wall:
            conf += 5
            reasons.append("🧱 Buy wall")
    
    # SHORT: EMAs aligned bearish
    elif ema9 < ema21 < ema50 and curr < ema9:
        direction = 'SHORT'
        conf = 45  # Increased base
        reasons.append(f"📉 Bearish trend (EMA9 < EMA21 < EMA50)")
        
        mom, _ = momentum_score(symbol)
        if mom < -0.05:  # Lowered
            conf += 12
            reasons.append(f"💥 Momentum {mom:.2f}%")
        
        imb, _, wall = orderbook_pressure(symbol)
        if imb < -10:  # Lowered
            conf += 10
            reasons.append(f"📕 Sell pressure {imb:.0f}%")
        if wall:
            conf += 5
            reasons.append("🧱 Sell wall")
    
    # PARTIAL TREND - Still tradeable
    elif ema9 > ema21 and curr > ema9:
        direction = 'LONG'
        conf = 35
        reasons.append(f"📈 Short-term bullish (EMA9 > EMA21)")
        
        mom, _ = momentum_score(symbol)
        if mom > 0.05:
            conf += 15
            reasons.append(f"🚀 Momentum +{mom:.2f}%")
        
        imb, _, _ = orderbook_pressure(symbol)
        if imb > 15:
            conf += 12
            reasons.append(f"📗 Buy pressure {imb:.0f}%")
    
    elif ema9 < ema21 and curr < ema9:
        direction = 'SHORT'
        conf = 35
        reasons.append(f"📉 Short-term bearish (EMA9 < EMA21)")
        
        mom, _ = momentum_score(symbol)
        if mom < -0.05:
            conf += 15
            reasons.append(f"💥 Momentum {mom:.2f}%")
        
        imb, _, _ = orderbook_pressure(symbol)
        if imb < -15:
            conf += 12
            reasons.append(f"📕 Sell pressure {imb:.0f}%")
    
    return direction, conf, reasons

def strategy_mean_reversion(symbol):
    """Strategy 2: Mean Reversion - IMPROVED"""
    if len(price_history[symbol]) < 50:
        return None, 0, []
    
    prices_list = list(price_history[symbol])
    curr = prices_list[-1]
    
    upper, mid, lower = bollinger_bands(prices_list)
    if not all([upper, mid, lower]):
        return None, 0, []
    
    rsi_val = rsi(prices_list)
    
    reasons = []
    conf = 0
    direction = None
    
    # OVERSOLD - LONG (relaxed conditions)
    if curr <= lower * 1.002 and rsi_val < 35:  # Relaxed
        direction = 'LONG'
        conf = 45
        reasons.append(f"📉 Near lower BB ${curr:.2f}")
        reasons.append(f"😰 RSI: {rsi_val:.0f}")
        
        if rsi_val < 25:
            conf += 15
            reasons.append("🔥 Oversold!")
        
        if prices_list[-1] > prices_list[-2]:
            conf += 8
            reasons.append("📊 Reversal candle")
        
        if market_sentiment['fear_greed'] < 35:
            conf += 8
            reasons.append(f"😰 Market fear")
    
    # OVERBOUGHT - SHORT (relaxed conditions)
    elif curr >= upper * 0.998 and rsi_val > 65:  # Relaxed
        direction = 'SHORT'
        conf = 45
        reasons.append(f"📈 Near upper BB ${curr:.2f}")
        reasons.append(f"😊 RSI: {rsi_val:.0f}")
        
        if rsi_val > 75:
            conf += 15
            reasons.append("🔥 Overbought!")
        
        if prices_list[-1] < prices_list[-2]:
            conf += 8
            reasons.append("📊 Reversal candle")
        
        if market_sentiment['fear_greed'] > 65:
            conf += 8
            reasons.append(f"😊 Market greed")
    
    return direction, conf, reasons

def strategy_sentiment_divergence(symbol):
    """Strategy 3: Sentiment Divergence - IMPROVED"""
    if len(price_history[symbol]) < 50:
        return None, 0, []
    
    prices_list = list(price_history[symbol])
    fg = market_sentiment['fear_greed']
    
    reasons = []
    conf = 0
    direction = None
    
    mom, mom_dir = momentum_score(symbol)
    
    # FEAR + price recovering = BUY
    if fg < 30:  # Relaxed from 25
        if mom > 0.02:  # Relaxed
            direction = 'LONG'
            conf = 50
            reasons.append(f"😰 Fear ({fg}) + Price up")
            reasons.append("📈 CONTRARIAN BUY")
            
            imb, _, _ = orderbook_pressure(symbol)
            if imb > 5:
                conf += 10
                reasons.append("🐋 Accumulation")
            
            rsi_val = rsi(prices_list)
            if rsi_val < 45:
                conf += 8
                reasons.append(f"📊 RSI low: {rsi_val:.0f}")
    
    # GREED + price weakening = SELL
    elif fg > 70:  # Relaxed from 75
        if mom < -0.02:  # Relaxed
            direction = 'SHORT'
            conf = 50
            reasons.append(f"😊 Greed ({fg}) + Price down")
            reasons.append("📉 CONTRARIAN SELL")
            
            imb, _, _ = orderbook_pressure(symbol)
            if imb < -5:
                conf += 10
                reasons.append("🐋 Distribution")
            
            rsi_val = rsi(prices_list)
            if rsi_val > 55:
                conf += 8
                reasons.append(f"📊 RSI high: {rsi_val:.0f}")
    
    return direction, conf, reasons

def strategy_smart_money(symbol):
    """Strategy 4: Smart Money - IMPROVED"""
    if len(price_history[symbol]) < 30:
        return None, 0, []
    
    imb, buy_wall, sell_wall = orderbook_pressure(symbol)
    
    reasons = []
    conf = 0
    direction = None
    
    mom, _ = momentum_score(symbol)
    
    # Strong accumulation (relaxed)
    if imb > 25:  # Lowered from 35
        direction = 'LONG'
        conf = 45
        reasons.append(f"🐋 Accumulation ({imb:.0f}% buy)")
        
        if buy_wall:
            conf += 10
            reasons.append("🧱 Buy wall")
        
        if mom >= -0.02:  # Relaxed
            conf += 15
            reasons.append("📈 Price stable/up")
        
        if market_sentiment['fear_greed'] < 50:
            conf += 8
            reasons.append("😰 Buying fear")
    
    # Strong distribution (relaxed)
    elif imb < -25:  # Lowered from -35
        direction = 'SHORT'
        conf = 45
        reasons.append(f"🐋 Distribution ({imb:.0f}% sell)")
        
        if sell_wall:
            conf += 10
            reasons.append("🧱 Sell wall")
        
        if mom <= 0.02:  # Relaxed
            conf += 15
            reasons.append("📉 Price stable/down")
        
        if market_sentiment['fear_greed'] > 50:
            conf += 8
            reasons.append("😊 Selling greed")
    
    return direction, conf, reasons

def strategy_breakout(symbol):
    """Strategy 5: Breakout - IMPROVED"""
    if len(price_history[symbol]) < 100:
        return None, 0, []
    
    prices_list = list(price_history[symbol])
    curr = prices_list[-1]
    
    recent = prices_list[-60:]
    high = max(recent)
    low = min(recent)
    
    reasons = []
    conf = 0
    direction = None
    
    # Breakout UP (relaxed)
    if curr > high * 1.0005:  # Lowered from 1.0008
        direction = 'LONG'
        conf = 42
        reasons.append(f"🚀 Breakout above ${high:.2f}")
        
        mom, _ = momentum_score(symbol)
        if mom > 0.08:
            conf += 15
            reasons.append(f"💨 Momentum +{mom:.2f}%")
        
        imb, _, _ = orderbook_pressure(symbol)
        if imb > 10:
            conf += 10
            reasons.append("📗 Volume support")
    
    # Breakdown DOWN (relaxed)
    elif curr < low * 0.9995:  # Lowered from 0.9992
        direction = 'SHORT'
        conf = 42
        reasons.append(f"💥 Breakdown below ${low:.2f}")
        
        mom, _ = momentum_score(symbol)
        if mom < -0.08:
            conf += 15
            reasons.append(f"💨 Momentum {mom:.2f}%")
        
        imb, _, _ = orderbook_pressure(symbol)
        if imb < -10:
            conf += 10
            reasons.append("📕 Volume support")
    
    return direction, conf, reasons

def strategy_momentum_only(symbol):
    """Strategy 6: Pure Momentum - NEW for more trades"""
    if len(price_history[symbol]) < 100:
        return None, 0, []
    
    prices_list = list(price_history[symbol])
    mom, mom_dir = momentum_score(symbol)
    
    reasons = []
    conf = 0
    direction = None
    
    if mom_dir == 'bullish' and mom > 0.10:
        direction = 'LONG'
        conf = 40
        reasons.append(f"🚀 Strong momentum +{mom:.2f}%")
        
        imb, _, _ = orderbook_pressure(symbol)
        if imb > 10:
            conf += 15
            reasons.append(f"📗 Order book supports")
        
        rsi_val = rsi(prices_list)
        if rsi_val < 70:
            conf += 10
            reasons.append(f"📊 RSI OK: {rsi_val:.0f}")
    
    elif mom_dir == 'bearish' and mom < -0.10:
        direction = 'SHORT'
        conf = 40
        reasons.append(f"💥 Strong momentum {mom:.2f}%")
        
        imb, _, _ = orderbook_pressure(symbol)
        if imb < -10:
            conf += 15
            reasons.append(f"📕 Order book supports")
        
        rsi_val = rsi(prices_list)
        if rsi_val > 30:
            conf += 10
            reasons.append(f"📊 RSI OK: {rsi_val:.0f}")
    
    return direction, conf, reasons

# =============================================================================
# POSITION CLASS
# =============================================================================

class Position:
    def __init__(self, symbol, side, entry, contracts, value, confidence, strategy, reasons):
        self.symbol = symbol
        self.side = side
        self.entry = entry
        self.contracts = contracts
        self.value = value
        self.confidence = confidence
        self.strategy = strategy
        self.reasons = reasons
        self.start_time = time.time()
        
        self.entry_fee = value * TAKER_FEE
        self.total_fee = value * TOTAL_FEE
        
        self.current_price = entry
        self.gross_pnl = 0
        self.net_pnl = -self.entry_fee
        self.gross_pct = 0
        self.peak_pct = 0
        
        sym_atr = atr(symbol) or (entry * 0.005)
        atr_pct = sym_atr / entry
        
        self.target_pct = max(0.006, atr_pct * 2)
        self.stop_pct = max(0.003, atr_pct * 1)
        
        self.breakeven_active = False
        self.trailing_active = False
        self.current_stop_pct = self.stop_pct
        
    def update(self, price):
        self.current_price = price
        
        if self.side == 'LONG':
            self.gross_pct = (price - self.entry) / self.entry
        else:
            self.gross_pct = (self.entry - price) / self.entry
        
        self.gross_pnl = self.value * self.gross_pct
        self.net_pnl = self.gross_pnl - self.total_fee
        
        if self.gross_pct > self.peak_pct:
            self.peak_pct = self.gross_pct
        
        if self.gross_pct >= 0.003 and not self.breakeven_active:
            self.breakeven_active = True
            self.current_stop_pct = -0.0002
        
        if self.gross_pct >= 0.005 and not self.trailing_active:
            self.trailing_active = True
        
        if self.trailing_active:
            new_stop = self.peak_pct - 0.0015
            if new_stop > self.current_stop_pct:
                self.current_stop_pct = new_stop

# =============================================================================
# TRADE EXECUTION
# =============================================================================

def calculate_max_position(symbol, price):
    global account_balance
    
    asset = ASSETS[symbol]
    leverage = asset['leverage']
    contract_size = asset['contract_size']
    
    contract_value = price * contract_size
    margin_per_contract = contract_value / leverage
    
    used_margin = sum(pos.value / ASSETS[pos.symbol]['leverage'] for pos in positions.values())
    available = max(0, account_balance - used_margin) * 0.85
    
    if available < margin_per_contract:
        return 0, 0
    
    max_contracts = int(available / margin_per_contract)
    risk_cap = int((account_balance * MAX_RISK_PER_TRADE * 10) / margin_per_contract)
    
    contracts = max(1, min(max_contracts, risk_cap))
    position_value = contracts * contract_value
    
    return contracts, position_value

def get_affordable_assets():
    global account_balance
    
    used_margin = sum(pos.value / ASSETS[pos.symbol]['leverage'] for pos in positions.values())
    available = max(0, account_balance - used_margin)
    
    affordable = []
    
    for symbol, asset in ASSETS.items():
        if symbol in positions:
            continue
        if symbol not in prices:
            continue
            
        price = prices[symbol]
        contract_value = price * asset['contract_size']
        margin_per_contract = contract_value / asset['leverage']
        
        if available >= margin_per_contract:
            max_lots = int(available / margin_per_contract)
            affordable.append({
                'symbol': symbol,
                'margin_per_lot': margin_per_contract,
                'max_lots': max_lots,
                'price': price
            })
    
    affordable.sort(key=lambda x: x['margin_per_lot'])
    return affordable

def can_trade_multiple_assets():
    affordable = get_affordable_assets()
    
    if len(affordable) < 2:
        return False
    
    total_min_margin = affordable[0]['margin_per_lot'] + affordable[1]['margin_per_lot']
    
    used_margin = sum(pos.value / ASSETS[pos.symbol]['leverage'] for pos in positions.values())
    available = max(0, account_balance - used_margin)
    
    return available >= total_min_margin * 1.2

def open_position(symbol, side, price, confidence, strategy, reasons):
    global positions
    
    if symbol in positions:
        return False
    
    contracts, value = calculate_max_position(symbol, price)
    if contracts < 1:
        return False
    
    pos = Position(symbol, side, price, contracts, value, confidence, strategy, reasons)
    positions[symbol] = pos
    
    # Send Telegram alert
    send_trade_alert('OPEN', symbol, side, price, confidence=confidence, strategy=strategy, reasons=reasons)
    
    print("\n" + "💎"*25)
    print(f"\n🚀 TRADE OPENED - {strategy}")
    print(f"\n   📊 {symbol} {side}")
    print(f"   🎯 Confidence: {confidence}%")
    print()
    for r in reasons:
        print(f"      • {r}")
    print()
    print(f"   💰 Entry: ${price:.4f}")
    print(f"   📦 Contracts: {contracts}")
    print(f"   💵 Value: ${value:.4f}")
    print("\n" + "💎"*25)
    
    return True

def close_position(symbol, reason, price):
    global positions, daily_pnl, stats, trade_history
    
    if symbol not in positions:
        return
    
    pos = positions[symbol]
    pos.update(price)
    
    stats['trades'] += 1
    stats['gross_pnl'] += pos.gross_pnl
    stats['fees_paid'] += pos.total_fee
    stats['net_pnl'] += pos.net_pnl
    daily_pnl += pos.net_pnl
    
    if pos.strategy not in stats['strategies']:
        stats['strategies'][pos.strategy] = {'trades': 0, 'wins': 0, 'pnl': 0}
    stats['strategies'][pos.strategy]['trades'] += 1
    stats['strategies'][pos.strategy]['pnl'] += pos.net_pnl
    
    if pos.net_pnl > 0:
        stats['wins'] += 1
        stats['strategies'][pos.strategy]['wins'] += 1
        emoji = "🎉💵"
        if pos.net_pnl > stats['best_trade']:
            stats['best_trade'] = pos.net_pnl
    else:
        stats['losses'] += 1
        emoji = "❌💸"
        if pos.net_pnl < stats['worst_trade']:
            stats['worst_trade'] = pos.net_pnl
    
    # Log trade
    trade_history.append({
        'symbol': symbol,
        'side': pos.side,
        'entry': pos.entry,
        'exit': price,
        'pnl': pos.net_pnl,
        'strategy': pos.strategy,
        'time': datetime.now().isoformat()
    })
    
    # Send Telegram alert
    send_trade_alert('CLOSE', symbol, pos.side, price, pnl=pos.net_pnl, strategy=pos.strategy)
    
    duration = int(time.time() - pos.start_time)
    
    print("\n" + "🚀"*25)
    print(f"\n{emoji} TRADE CLOSED - {reason}")
    print(f"\n   📊 {symbol} {pos.side} ({pos.strategy})")
    print(f"   ⏱️ Duration: {duration}s")
    print(f"   💰 NET: ${pos.net_pnl:+.4f}")
    print(f"\n   📊 Session: {stats['trades']} trades, {stats['wins']} wins")
    print("\n" + "🚀"*25)
    
    del positions[symbol]

def manage_positions():
    for symbol in list(positions.keys()):
        pos = positions[symbol]
        price = prices.get(symbol)
        if not price:
            continue
        
        pos.update(price)
        
        should_exit = False
        reason = ""
        
        if pos.gross_pct >= pos.target_pct:
            should_exit = True
            reason = "🎯 TARGET HIT"
        
        elif pos.gross_pct <= -pos.current_stop_pct:
            if pos.trailing_active:
                reason = f"📈 TRAILING STOP"
            elif pos.breakeven_active:
                reason = "🔒 BREAKEVEN EXIT"
            else:
                reason = "🛑 STOP LOSS"
            should_exit = True
        
        elif time.time() - pos.start_time > 300:
            if pos.net_pnl > 0:
                should_exit = True
                reason = "⏰ TIME EXIT (profit)"
            elif pos.gross_pct > -0.001:
                should_exit = True
                reason = "⏰ TIME EXIT (breakeven)"
        
        if should_exit:
            close_position(symbol, reason, price)

# =============================================================================
# SIGNAL FINDER - IMPROVED
# =============================================================================

def find_best_signal():
    """Find best signal - IMPROVED to trade both directions"""
    if not trading_active:
        return None
    
    if len(positions) >= MAX_POSITIONS:
        return None
    
    if daily_pnl < -account_balance * MAX_DAILY_LOSS:
        return None
    
    affordable = get_affordable_assets()
    if not affordable:
        return None
    
    # Get current direction if we have positions
    current_direction = None
    if positions:
        for sym, pos in positions.items():
            current_direction = pos.side
            break
    
    best = None
    best_conf = 0
    
    # 6 strategies now
    strategies = [
        ('Trend Following', strategy_trend_following),
        ('Mean Reversion', strategy_mean_reversion),
        ('Sentiment Divergence', strategy_sentiment_divergence),
        ('Smart Money', strategy_smart_money),
        ('Breakout', strategy_breakout),
        ('Momentum', strategy_momentum_only),
    ]
    
    tradeable_symbols = [a['symbol'] for a in affordable]
    
    for symbol in tradeable_symbols:
        if symbol in positions:
            continue
        if len(price_history[symbol]) < 100:
            continue
        
        for name, func in strategies:
            direction, conf, reasons = func(symbol)
            
            if not direction:
                continue
            
            # Same direction enforcement for multiple positions
            if current_direction and direction != current_direction:
                continue
            
            # Apply sentiment adjustments
            fg = market_sentiment['fear_greed']
            
            # Boost LONG in fear, SHORT in greed
            if direction == 'LONG' and fg < 35:
                conf += 8
            elif direction == 'SHORT' and fg > 65:
                conf += 8
            # Penalize opposite
            elif direction == 'LONG' and fg > 75:
                conf -= 5
            elif direction == 'SHORT' and fg < 25:
                conf -= 5
            
            if conf > best_conf:
                best_conf = conf
                best = {
                    'symbol': symbol,
                    'direction': direction,
                    'confidence': conf,
                    'strategy': name,
                    'reasons': reasons,
                    'price': prices[symbol]
                }
    
    if best and best['confidence'] >= MIN_CONFIDENCE:
        return best
    return None

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
                price_history[sym].append(price)
                
                for s, pos in positions.items():
                    if s == sym:
                        pos.update(price)
                    
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

def display_position(pos):
    hold = int(time.time() - pos.start_time)
    net_pct = pos.gross_pct - TOTAL_FEE
    
    if pos.net_pnl > 0.001:
        color = "🟢"
    elif pos.net_pnl < -0.001:
        color = "🔴"
    else:
        color = "🟡"
    
    protection = "📈" if pos.trailing_active else "🔒" if pos.breakeven_active else "🔴"
    
    line = f"{color} {pos.symbol} {pos.side} | "
    line += f"${pos.current_price:.2f} | "
    line += f"NET: ${pos.net_pnl:+.4f} ({net_pct*100:+.2f}%) | "
    line += f"{protection} | ⏱️{hold}s"
    
    return line

def display_all_positions():
    lines = []
    total_upnl = 0
    
    for sym, pos in positions.items():
        lines.append(display_position(pos))
        total_upnl += pos.net_pnl
    
    output = " | ".join(lines)
    output += f" | TOTAL: ${total_upnl:+.4f}"
    
    sys.stdout.write(f"\r{output}".ljust(150))
    sys.stdout.flush()

def display_status():
    print("\n" + "─"*70)
    wr = stats['wins'] / max(1, stats['trades']) * 100
    status = "🟢 ACTIVE" if trading_active else "🔴 PAUSED"
    print(f"📊 {status} | Balance: ${account_balance:.4f} | Net: ${stats['net_pnl']:+.4f} | WR: {wr:.0f}%")
    print("─"*70)
    
    affordable = get_affordable_assets()
    
    if affordable:
        print(f"💰 Tradeable: {', '.join([a['symbol'] for a in affordable])}")
    
    print("─"*70)
    
    # Show all signals for each asset
    for sym in ASSETS:
        if sym not in prices or len(price_history[sym]) < 50:
            continue
        
        price = prices[sym]
        mom, mom_dir = momentum_score(sym)
        rsi_val = rsi(list(price_history[sym]))
        imb, _, _ = orderbook_pressure(sym)
        
        # Get best signal
        best_conf = 0
        best_dir = ""
        for name, func in [
            ('TF', strategy_trend_following),
            ('MR', strategy_mean_reversion),
            ('SD', strategy_sentiment_divergence),
            ('SM', strategy_smart_money),
            ('BO', strategy_breakout),
            ('MO', strategy_momentum_only),
        ]:
            direction, conf, _ = func(sym)
            if conf > best_conf:
                best_conf = conf
                best_dir = direction or ""
        
        arrow = "↑" if mom_dir == 'bullish' else "↓" if mom_dir == 'bearish' else "→"
        
        is_affordable = any(a['symbol'] == sym for a in affordable)
        
        if best_conf >= MIN_CONFIDENCE and is_affordable:
            status = f"🟢 {best_dir}"
        elif best_conf >= 50:
            status = f"🟡 {best_dir} ({best_conf}%)"
        else:
            status = "⏳"
        
        print(f"   {sym}: ${price:,.2f} {arrow} | RSI:{rsi_val:.0f} | OB:{imb:+.0f}% | {status}")
    
    print(f"\n   😰 F&G: {market_sentiment['fear_greed']} | 📈 Trend: {market_sentiment['global_trend']}")
    print(f"   📱 Telegram: {'Connected' if TELEGRAM_CHAT_ID else 'Waiting for /start'}")
    print("─"*70)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global running, account_balance
    
    print("\n" + "╔" + "═"*70 + "╗")
    print("║" + " 💎 ALADDIN WEALTH BUILDER + TELEGRAM 💎 ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print("║" + " ".ljust(70) + "║")
    print("║" + "   📱 TELEGRAM CONTROL ENABLED".ljust(70) + "║")
    print("║" + "   Commands: /start /stop /status /trades /settings /kill".ljust(70) + "║")
    print("║" + " ".ljust(70) + "║")
    print("║" + "   💎 6 TRADING STRATEGIES:".ljust(70) + "║")
    print("║" + "      1. Trend Following".ljust(70) + "║")
    print("║" + "      2. Mean Reversion".ljust(70) + "║")
    print("║" + "      3. Sentiment Divergence".ljust(70) + "║")
    print("║" + "      4. Smart Money Detection".ljust(70) + "║")
    print("║" + "      5. Breakout Momentum".ljust(70) + "║")
    print("║" + "      6. Pure Momentum".ljust(70) + "║")
    print("║" + " ".ljust(70) + "║")
    print("║" + f"   🎯 MIN CONFIDENCE: {MIN_CONFIDENCE}%".ljust(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Get balance
    print("\n💰 Fetching account balance...")
    balance = get_account_balance()
    if balance > 0:
        print(f"   ✅ Balance: ${balance:.4f}")
    else:
        print("   ⚠️ Could not fetch balance, using $20 paper mode")
        account_balance = 20.00
        starting_balance = 20.00
        stats['peak_balance'] = 20.00
    
    # Initial sentiment
    print("\n📊 Analyzing market...")
    update_market_sentiment()
    
    # Start Telegram thread
    print("\n📱 Starting Telegram bot...")
    print("   Send /start to @CryptoDEIbot to connect!")
    telegram_thread = threading.Thread(target=run_telegram, daemon=True)
    telegram_thread.start()
    
    # Start WebSocket
    print("\n📡 Connecting to market...")
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()
    
    time.sleep(5)
    
    print("\n" + "═"*70)
    print("💎 WEALTH BUILDER ACTIVE")
    print("   Scanning for setups... (Min confidence: 65%)")
    print("═"*70 + "\n")
    
    last_scan = 0
    last_status = 0
    last_sentiment = 0
    
    try:
        while running:
            now = time.time()
            
            if now - last_sentiment > 180:
                update_market_sentiment()
                last_sentiment = now
            
            manage_positions()
            
            if now - last_scan >= SCAN_INTERVAL:
                signal = find_best_signal()
                if signal:
                    open_position(
                        signal['symbol'],
                        signal['direction'],
                        signal['price'],
                        signal['confidence'],
                        signal['strategy'],
                        signal['reasons']
                    )
                last_scan = now
            
            if positions:
                display_all_positions()
            elif now - last_status > 15:
                display_status()
                last_status = now
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        running = False
    
    # Summary
    wr = stats['wins'] / max(1, stats['trades']) * 100
    final_balance = account_balance if account_balance > 0 else starting_balance
    roi = (stats['net_pnl'] / max(0.01, starting_balance) * 100)
    
    summary = f"""
📊 FINAL SUMMARY

💰 Starting: ${starting_balance:.4f}
💰 Final: ${final_balance + stats['net_pnl']:.4f} ({roi:+.1f}%)

📊 Trades: {stats['trades']}
✅ Wins: {stats['wins']} | ❌ Losses: {stats['losses']}
🎯 Win Rate: {wr:.1f}%

💎 Net P&L: ${stats['net_pnl']:+.4f}
"""
    
    print(summary)
    telegram_send(summary.replace('\n', '\n'))

if __name__ == "__main__":
    main()
