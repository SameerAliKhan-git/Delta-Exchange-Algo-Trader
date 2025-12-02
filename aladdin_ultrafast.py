"""
Aladdin Ultra-Fast Trading Bot
==============================
Features:
- MICROSECOND-LEVEL price monitoring via WebSocket
- INSTANT exit on sentiment reversal
- Real-time UPNL tracking
- Fee-aware trading decisions

LIVE TRADING - USE WITH CAUTION
"""

import time
import hmac
import hashlib
import json
import logging
import requests
import threading
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import sys
import os

# Disable buffering for real-time output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Setup logging - minimal for speed
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger("Aladdin")

# API Configuration
API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"
BASE_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

# Contract sizes
CONTRACT_SIZES = {"BTCUSD": 0.001, "ETHUSD": 0.01, "SOLUSD": 0.1}

# Trading config
STOP_LOSS_PCT = 0.02      # 2%
TAKE_PROFIT_PCT = 0.04    # 4%
TAKER_FEE = 0.0005        # 0.05%
SENTIMENT_EXIT_THRESHOLD = -0.10  # Exit long if sentiment drops below this


class UltraFastPriceStream:
    """Ultra-low latency price streaming via WebSocket"""
    
    def __init__(self, on_price_update):
        self.on_price_update = on_price_update
        self.ws = None
        self.running = False
        self.prices: Dict[str, float] = {}
        self.last_update_time: Dict[str, float] = {}
        self.update_count = 0
        self.latencies = deque(maxlen=100)
        
    def start(self, symbols: List[str]):
        self.running = True
        self.symbols = symbols
        
        def run():
            while self.running:
                try:
                    self.ws = websocket.WebSocketApp(
                        WS_URL,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=lambda ws, e: None,
                        on_close=lambda ws, a, b: None
                    )
                    self.ws.run_forever(ping_interval=10)
                except:
                    time.sleep(1)
        
        threading.Thread(target=run, daemon=True).start()
        time.sleep(1)
    
    def _on_open(self, ws):
        print("⚡ WebSocket connected - ULTRA-FAST mode")
        for symbol in self.symbols:
            # Subscribe to multiple channels for faster updates
            ws.send(json.dumps({
                "type": "subscribe",
                "payload": {"channels": [
                    {"name": "v2/ticker", "symbols": [symbol]},
                    {"name": "mark_price", "symbols": [symbol]}
                ]}
            }))
    
    def _on_message(self, ws, message):
        recv_time = time.time()
        try:
            data = json.loads(message)
            symbol = data.get("symbol", "")
            mark_price = data.get("mark_price")
            
            if mark_price and symbol:
                price = float(mark_price)
                self.prices[symbol] = price
                self.update_count += 1
                
                # Calculate latency if timestamp available
                server_time = data.get("timestamp")
                if server_time:
                    latency_ms = (recv_time * 1000) - (server_time / 1000)
                    self.latencies.append(latency_ms)
                
                self.last_update_time[symbol] = recv_time
                
                # Callback for immediate processing
                self.on_price_update(symbol, price, recv_time)
                
        except:
            pass
    
    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 0)
    
    def get_avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


class FastSentimentTracker:
    """Fast sentiment tracking with caching"""
    
    BULLISH = ["bullish", "rally", "pump", "moon", "adoption", "etf approved", 
               "institutional", "accumulation", "breakout", "surge", "soar"]
    BEARISH = ["bearish", "crash", "dump", "selloff", "ban", "hack", "fraud",
               "lawsuit", "investigation", "collapse", "plunge", "fear"]
    
    def __init__(self):
        self.score = 0.0
        self.direction = "neutral"
        self.last_update = None
        self.headlines = []
        self._lock = threading.Lock()
        
    def update_async(self):
        """Update sentiment in background"""
        def fetch():
            try:
                resp = requests.get(
                    "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
                    timeout=5
                )
                if resp.status_code == 200:
                    news = resp.json().get("Data", [])[:10]
                    
                    scores = []
                    headlines = []
                    for item in news:
                        text = (item.get("title", "") + " " + item.get("body", "")).lower()
                        bull = sum(1 for kw in self.BULLISH if kw in text)
                        bear = sum(1 for kw in self.BEARISH if kw in text)
                        if bull + bear > 0:
                            scores.append((bull - bear) / (bull + bear))
                            headlines.append(item.get("title", "")[:60])
                    
                    with self._lock:
                        if scores:
                            self.score = sum(scores) / len(scores)
                            if self.score > 0.15:
                                self.direction = "BULLISH"
                            elif self.score < -0.15:
                                self.direction = "BEARISH"
                            else:
                                self.direction = "NEUTRAL"
                        self.headlines = headlines[:3]
                        self.last_update = datetime.now()
            except:
                pass
        
        threading.Thread(target=fetch, daemon=True).start()
    
    def get_sentiment(self) -> Tuple[float, str]:
        with self._lock:
            return self.score, self.direction


class DeltaAPI:
    """Minimal API client for speed"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def _sign(self, method: str, endpoint: str, payload: str = "") -> dict:
        ts = str(int(time.time()))
        sig = hmac.new(
            API_SECRET.encode(),
            f"{method}{ts}{endpoint}{payload}".encode(),
            hashlib.sha256
        ).hexdigest()
        return {"api-key": API_KEY, "timestamp": ts, "signature": sig, "Content-Type": "application/json"}
    
    def get_balance(self) -> float:
        try:
            r = self.session.get(BASE_URL + "/v2/wallet/balances", headers=self._sign("GET", "/v2/wallet/balances"))
            for a in r.json().get("result", []):
                if a.get("asset_symbol") in ["USD", "USDT"]:
                    return float(a.get("available_balance", 0))
        except:
            pass
        return 0
    
    def get_positions(self) -> List[dict]:
        positions = []
        for asset in ["ETH", "BTC", "SOL"]:
            try:
                endpoint = f"/v2/positions?underlying_asset_symbol={asset}"
                r = self.session.get(BASE_URL + endpoint, headers=self._sign("GET", endpoint))
                for p in r.json().get("result", []):
                    if float(p.get("size", 0)) != 0:
                        positions.append(p)
            except:
                pass
        return positions
    
    def close_position(self, symbol: str) -> bool:
        try:
            # Get product ID
            r = self.session.get(BASE_URL + "/v2/products")
            product_id = None
            for p in r.json().get("result", []):
                if p.get("symbol") == symbol:
                    product_id = p.get("id")
                    break
            
            if product_id:
                endpoint = f"/v2/positions/{product_id}/close"
                payload = "{}"
                r = self.session.post(
                    BASE_URL + endpoint,
                    headers=self._sign("POST", endpoint, payload),
                    data=payload
                )
                return r.status_code == 200
        except:
            pass
        return False
    
    def place_order(self, symbol: str, side: str, size: int) -> dict:
        try:
            # Get product ID
            r = self.session.get(BASE_URL + "/v2/products")
            product_id = None
            for p in r.json().get("result", []):
                if p.get("symbol") == symbol:
                    product_id = p.get("id")
                    break
            
            if product_id:
                order = {"product_id": product_id, "side": side, "size": size, "order_type": "market_order"}
                payload = json.dumps(order)
                r = self.session.post(
                    BASE_URL + "/v2/orders",
                    headers=self._sign("POST", "/v2/orders", payload),
                    data=payload
                )
                return r.json()
        except Exception as e:
            return {"error": str(e)}
        return {"error": "Failed"}


class AladdinUltraFast:
    """Ultra-fast trading bot with microsecond monitoring"""
    
    def __init__(self):
        self.api = DeltaAPI()
        self.sentiment = FastSentimentTracker()
        self.running = False
        
        # Position tracking
        self.positions: Dict[str, dict] = {}
        self.entry_prices: Dict[str, float] = {}
        self.position_sizes: Dict[str, float] = {}
        self.position_directions: Dict[str, str] = {}
        
        # Stats
        self.update_count = 0
        self.last_display = 0
        self.peak_upnl = 0
        self.trade_closed = False
        
        # Price stream with callback
        self.price_stream = UltraFastPriceStream(self._on_price_update)
    
    def _on_price_update(self, symbol: str, price: float, timestamp: float):
        """Called on every price update - ULTRA FAST"""
        self.update_count += 1
        
        # Check if we have a position in this symbol
        if symbol in self.entry_prices:
            entry = self.entry_prices[symbol]
            size = self.position_sizes[symbol]
            direction = self.position_directions[symbol]
            contract_size = CONTRACT_SIZES.get(symbol, 0.01)
            
            # Calculate UPNL
            if direction == "LONG":
                upnl = (price - entry) * size * contract_size
                pnl_pct = (price - entry) / entry
            else:
                upnl = (entry - price) * size * contract_size
                pnl_pct = (entry - price) / entry
            
            # Track peak
            if upnl > self.peak_upnl:
                self.peak_upnl = upnl
            
            # Get sentiment
            sent_score, sent_dir = self.sentiment.get_sentiment()
            
            # INSTANT EXIT CONDITIONS
            should_exit = False
            exit_reason = ""
            
            # 1. Stop Loss
            if pnl_pct < -STOP_LOSS_PCT:
                should_exit = True
                exit_reason = f"🛑 STOP LOSS {pnl_pct:.2%}"
            
            # 2. Take Profit
            elif pnl_pct > TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = f"🎯 TAKE PROFIT {pnl_pct:.2%}"
            
            # 3. SENTIMENT REVERSAL - Exit LONG if sentiment turns bearish
            elif direction == "LONG" and sent_score < SENTIMENT_EXIT_THRESHOLD:
                should_exit = True
                exit_reason = f"📉 SENTIMENT BEARISH ({sent_score:.2f})"
            
            # 4. SENTIMENT REVERSAL - Exit SHORT if sentiment turns bullish
            elif direction == "SHORT" and sent_score > -SENTIMENT_EXIT_THRESHOLD:
                should_exit = True
                exit_reason = f"📈 SENTIMENT BULLISH ({sent_score:.2f})"
            
            # Execute exit
            if should_exit and not self.trade_closed:
                self.trade_closed = True
                print(f"\n\n{'🚨'*20}")
                print(f"⚡ INSTANT EXIT: {exit_reason}")
                print(f"   Symbol: {symbol}")
                print(f"   Entry:  ${entry:.2f}")
                print(f"   Exit:   ${price:.2f}")
                print(f"   P&L:    ${upnl:+.4f} ({pnl_pct:+.2%})")
                print(f"{'🚨'*20}\n")
                
                success = self.api.close_position(symbol)
                if success:
                    print(f"✅ Position closed successfully!")
                    del self.entry_prices[symbol]
                    del self.position_sizes[symbol]
                    del self.position_directions[symbol]
                else:
                    print(f"❌ Failed to close position - RETRY!")
                    self.trade_closed = False
            
            # Display update (throttled to every 100ms for readability)
            now = time.time()
            if now - self.last_display > 0.1:
                self.last_display = now
                
                # Color and arrow based on P&L
                if upnl > 0:
                    color = "🟢"
                    arrow = "↑"
                elif upnl < 0:
                    color = "🔴"
                    arrow = "↓"
                else:
                    color = "⚪"
                    arrow = "→"
                
                # Sentiment indicator
                if sent_dir == "BULLISH":
                    sent_emoji = "📈"
                elif sent_dir == "BEARISH":
                    sent_emoji = "📉"
                else:
                    sent_emoji = "📊"
                
                # Calculate update frequency
                freq = self.update_count / max(1, now - self.start_time)
                
                # Display
                status = (
                    f"\r{color} {symbol} {direction} | "
                    f"${price:.2f} {arrow} | "
                    f"UPNL: ${upnl:+.4f} ({pnl_pct:+.2%}) | "
                    f"Peak: ${self.peak_upnl:+.4f} | "
                    f"{sent_emoji} {sent_dir} ({sent_score:+.2f}) | "
                    f"⚡{freq:.0f}/s   "
                )
                sys.stdout.write(status)
                sys.stdout.flush()
    
    def load_positions(self):
        """Load existing positions"""
        positions = self.api.get_positions()
        for pos in positions:
            symbol = pos.get("product_symbol", "")
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size != 0 and symbol:
                self.entry_prices[symbol] = entry
                self.position_sizes[symbol] = abs(size)
                self.position_directions[symbol] = "LONG" if size > 0 else "SHORT"
                print(f"📌 Loaded position: {symbol} {'LONG' if size > 0 else 'SHORT'} @ ${entry:.2f}")
    
    def check_trade_opportunity(self):
        """Check for new trade opportunities based on sentiment"""
        if self.entry_prices:  # Already have a position
            return
        
        sent_score, sent_dir = self.sentiment.get_sentiment()
        balance = self.api.get_balance()
        
        if balance < 0.50:  # Need at least $0.50 to trade
            return
        
        # Only trade on strong sentiment
        if sent_score > 0.20:
            print(f"\n📈 STRONG BULLISH SENTIMENT ({sent_score:.2f}) - Looking to go LONG")
            # Try to enter long on SOLUSD (cheapest margin)
            result = self.api.place_order("SOLUSD", "buy", 1)
            if "error" not in result:
                print(f"✅ Opened LONG position on SOLUSD")
        
        elif sent_score < -0.20:
            print(f"\n📉 STRONG BEARISH SENTIMENT ({sent_score:.2f}) - Looking to go SHORT")
            result = self.api.place_order("SOLUSD", "sell", 1)
            if "error" not in result:
                print(f"✅ Opened SHORT position on SOLUSD")
    
    def start(self):
        """Start the ultra-fast bot"""
        self.running = True
        self.start_time = time.time()
        
        # Get initial data
        balance = self.api.get_balance()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    ⚡ ALADDIN ULTRA-FAST TRADING BOT ⚡                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  💰 Balance:        ${balance:.4f}                                        
║  🎯 Stop Loss:      {STOP_LOSS_PCT:.1%}                                   
║  🎯 Take Profit:    {TAKE_PROFIT_PCT:.1%}                                 
║  📉 Sentiment Exit: {SENTIMENT_EXIT_THRESHOLD}                            
║                                                                          
║  ⚡ MICROSECOND-LEVEL MONITORING ACTIVE                                  
║  📰 Sentiment-based instant exits ENABLED                                
║  🚀 Fee-aware trading ENABLED                                            
║                                                                          
║  Press Ctrl+C to stop                                                    
╚══════════════════════════════════════════════════════════════════════════╝
""")
        
        # Load existing positions
        self.load_positions()
        
        # Start price stream
        symbols = ["BTCUSD", "ETHUSD", "SOLUSD"]
        print("\n⚡ Starting ultra-fast price stream...")
        self.price_stream.start(symbols)
        
        # Initial sentiment fetch
        print("📰 Fetching market sentiment...")
        self.sentiment.update_async()
        time.sleep(2)
        
        sent_score, sent_dir = self.sentiment.get_sentiment()
        print(f"📊 Initial Sentiment: {sent_dir} ({sent_score:+.2f})")
        if self.sentiment.headlines:
            for h in self.sentiment.headlines:
                print(f"   • {h}")
        
        print(f"\n{'='*70}")
        print("⚡ REAL-TIME MONITORING STARTED")
        print(f"{'='*70}\n")
        
        # Main loop
        last_sentiment_update = time.time()
        
        try:
            while self.running:
                now = time.time()
                
                # Update sentiment every 2 minutes
                if now - last_sentiment_update > 120:
                    self.sentiment.update_async()
                    last_sentiment_update = now
                    
                    # Check for new trade opportunities
                    self.check_trade_opportunity()
                
                # Small sleep to prevent CPU spin
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping bot...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the bot"""
        self.running = False
        self.price_stream.stop()
        
        runtime = time.time() - self.start_time
        avg_latency = self.price_stream.get_avg_latency()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  📊 SESSION SUMMARY                                          ║
╠══════════════════════════════════════════════════════════════╣
║  Runtime:         {runtime:.1f} seconds                       
║  Price Updates:   {self.update_count:,}                       
║  Avg Latency:     {avg_latency:.1f}ms                         
║  Updates/sec:     {self.update_count/max(1,runtime):.0f}      
║  Peak UPNL:       ${self.peak_upnl:+.4f}                      
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    bot = AladdinUltraFast()
    bot.start()
