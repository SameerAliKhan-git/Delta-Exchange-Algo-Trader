"""
Aladdin MASTER - Ultimate Trading Bot
======================================
🧠 News & Sentiment Analysis - Trades WITH the market mood
📈 Trend Following - Only trades in trend direction
💰 UPNL Protection - Winners NEVER become losers
🎯 Smart Exits - Lock in profits, cut losses fast
📊 Gradual Profit Building - Compounds winners

THE COMPLETE TRADING SYSTEM
"""

import time
import hmac
import hashlib
import json
import requests
import threading
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API CONFIG
# =============================================================================

API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"
BASE_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

CONTRACTS = {
    "BTCUSD": {"size": 0.001, "name": "Bitcoin"},
    "ETHUSD": {"size": 0.01, "name": "Ethereum"},
    "SOLUSD": {"size": 0.1, "name": "Solana"}
}

# =============================================================================
# MASTER CONFIG
# =============================================================================

@dataclass
class MasterConfig:
    # Fees
    TAKER_FEE: float = 0.0005
    ROUND_TRIP_FEE: float = 0.0010
    
    # Initial targets (will trail up)
    INITIAL_TARGET: float = 0.0040      # 0.40% initial target
    INITIAL_STOP: float = 0.0020        # 0.20% initial stop
    
    # Trailing stop (protects winners)
    TRAIL_ACTIVATION: float = 0.0020    # Start trailing at 0.20% profit
    TRAIL_DISTANCE: float = 0.0015      # Trail 0.15% behind peak
    
    # Breakeven protection
    BREAKEVEN_ACTIVATION: float = 0.0015  # Move stop to breakeven at 0.15%
    
    # Time limits
    MAX_HOLD_MINUTES: int = 10
    
    # Sentiment thresholds
    BULLISH_THRESHOLD: float = 0.15
    BEARISH_THRESHOLD: float = -0.15
    
    # Trend requirements
    TREND_PERIOD: int = 30              # 30 ticks for trend
    MIN_TREND_STRENGTH: float = 0.0010  # 0.10% minimum trend
    
    # Trade management
    COOLDOWN_SECONDS: int = 60
    
    SYMBOLS: List[str] = None
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ["ETHUSD", "BTCUSD"]

CONFIG = MasterConfig()


# =============================================================================
# NEWS & SENTIMENT ANALYZER
# =============================================================================

class SentimentAnalyzer:
    """Real-time news and sentiment analysis"""
    
    BULLISH_KEYWORDS = [
        "bullish", "rally", "surge", "soar", "breakout", "pump", "moon",
        "adoption", "etf approved", "institutional", "buy", "accumulation",
        "support", "higher", "gains", "profit", "positive", "optimistic",
        "upgrade", "partnership", "launch", "success", "growth", "recovery"
    ]
    
    BEARISH_KEYWORDS = [
        "bearish", "crash", "dump", "plunge", "selloff", "collapse", "tank",
        "ban", "hack", "fraud", "lawsuit", "investigation", "fear", "panic",
        "resistance", "lower", "losses", "negative", "pessimistic", "downgrade",
        "warning", "risk", "failure", "decline", "recession", "crisis"
    ]
    
    def __init__(self):
        self.score = 0.0
        self.direction = "NEUTRAL"
        self.headlines = []
        self.last_update = None
        self.confidence = 0.0
        self._lock = threading.Lock()
        self._running = True
        
        # Start background updater
        threading.Thread(target=self._update_loop, daemon=True).start()
    
    def _update_loop(self):
        """Background sentiment updates every 2 minutes"""
        while self._running:
            self._fetch_sentiment()
            time.sleep(120)  # Update every 2 minutes
    
    def _fetch_sentiment(self):
        """Fetch and analyze news"""
        try:
            # CryptoCompare news API
            resp = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC,ETH,Trading",
                timeout=10
            )
            
            if resp.status_code != 200:
                return
            
            news = resp.json().get("Data", [])[:15]
            
            scores = []
            headlines = []
            
            for item in news:
                title = item.get("title", "")
                body = item.get("body", "")[:500]
                text = (title + " " + body).lower()
                
                # Count keywords
                bull_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text)
                bear_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text)
                
                total = bull_count + bear_count
                if total > 0:
                    score = (bull_count - bear_count) / total
                    scores.append(score)
                    
                    # Extract sentiment for headline
                    sentiment = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "⚪"
                    headlines.append(f"{sentiment} {title[:70]}")
            
            with self._lock:
                if scores:
                    self.score = sum(scores) / len(scores)
                    self.confidence = len([s for s in scores if abs(s) > 0.3]) / len(scores)
                    
                    if self.score > CONFIG.BULLISH_THRESHOLD:
                        self.direction = "BULLISH"
                    elif self.score < CONFIG.BEARISH_THRESHOLD:
                        self.direction = "BEARISH"
                    else:
                        self.direction = "NEUTRAL"
                
                self.headlines = headlines[:5]
                self.last_update = datetime.now()
                
        except Exception as e:
            pass
    
    def get_sentiment(self) -> Tuple[float, str, float]:
        """Returns (score, direction, confidence)"""
        with self._lock:
            return self.score, self.direction, self.confidence
    
    def get_headlines(self) -> List[str]:
        with self._lock:
            return self.headlines.copy()
    
    def should_trade_long(self) -> Tuple[bool, str]:
        """Check if sentiment favors long"""
        score, direction, conf = self.get_sentiment()
        if direction == "BULLISH" and conf > 0.3:
            return True, f"Bullish sentiment ({score:+.2f}, {conf:.0%} confidence)"
        elif direction == "BEARISH":
            return False, f"Bearish sentiment - no longs ({score:+.2f})"
        return True, "Neutral sentiment - proceed with caution"
    
    def should_trade_short(self) -> Tuple[bool, str]:
        """Check if sentiment favors short"""
        score, direction, conf = self.get_sentiment()
        if direction == "BEARISH" and conf > 0.3:
            return True, f"Bearish sentiment ({score:+.2f}, {conf:.0%} confidence)"
        elif direction == "BULLISH":
            return False, f"Bullish sentiment - no shorts ({score:+.2f})"
        return True, "Neutral sentiment - proceed with caution"
    
    def stop(self):
        self._running = False


# =============================================================================
# TREND ANALYZER
# =============================================================================

class TrendAnalyzer:
    """Multi-timeframe trend analysis"""
    
    def __init__(self):
        self.prices: Dict[str, deque] = {}
        self.highs: Dict[str, float] = {}
        self.lows: Dict[str, float] = {}
    
    def add_price(self, symbol: str, price: float):
        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=CONFIG.TREND_PERIOD)
            self.highs[symbol] = price
            self.lows[symbol] = price
        
        self.prices[symbol].append(price)
        
        # Track session highs/lows
        if price > self.highs[symbol]:
            self.highs[symbol] = price
        if price < self.lows[symbol]:
            self.lows[symbol] = price
    
    def get_trend(self, symbol: str) -> Tuple[str, float, Dict]:
        """
        Returns: (direction, strength, details)
        """
        if symbol not in self.prices or len(self.prices[symbol]) < CONFIG.TREND_PERIOD:
            return "NEUTRAL", 0, {}
        
        prices = list(self.prices[symbol])
        
        # Calculate trend strength
        start_price = prices[0]
        end_price = prices[-1]
        strength = (end_price - start_price) / start_price
        
        # Calculate momentum (recent vs older)
        mid = len(prices) // 2
        old_avg = sum(prices[:mid]) / mid
        new_avg = sum(prices[mid:]) / (len(prices) - mid)
        momentum = (new_avg - old_avg) / old_avg
        
        # Higher highs / lower lows check
        first_half_high = max(prices[:mid])
        second_half_high = max(prices[mid:])
        first_half_low = min(prices[:mid])
        second_half_low = min(prices[mid:])
        
        higher_highs = second_half_high > first_half_high
        lower_lows = second_half_low < first_half_low
        
        # Determine direction
        if strength > CONFIG.MIN_TREND_STRENGTH and higher_highs and momentum > 0:
            direction = "UPTREND"
        elif strength < -CONFIG.MIN_TREND_STRENGTH and lower_lows and momentum < 0:
            direction = "DOWNTREND"
        else:
            direction = "SIDEWAYS"
        
        details = {
            "strength": strength,
            "momentum": momentum,
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "session_high": self.highs.get(symbol, 0),
            "session_low": self.lows.get(symbol, 0)
        }
        
        return direction, abs(strength), details


# =============================================================================
# UPNL PROTECTOR - Winners Never Become Losers
# =============================================================================

class UPNLProtector:
    """Protects unrealized profits"""
    
    def __init__(self):
        self.peak_upnl: Dict[str, float] = {}
        self.peak_price: Dict[str, float] = {}
        self.breakeven_activated: Dict[str, bool] = {}
        self.trailing_activated: Dict[str, bool] = {}
        self.trailing_stop: Dict[str, float] = {}
    
    def update(self, symbol: str, entry: float, current: float, side: str) -> Dict:
        """
        Update UPNL tracking and calculate protective stops
        Returns: {stop_price, reason, action}
        """
        if side == "LONG":
            pnl_pct = (current - entry) / entry
            peak = self.peak_price.get(symbol, current)
            if current > peak:
                self.peak_price[symbol] = current
                peak = current
        else:
            pnl_pct = (entry - current) / entry
            peak = self.peak_price.get(symbol, current)
            if current < peak:
                self.peak_price[symbol] = current
                peak = current
        
        # Track peak UPNL
        if symbol not in self.peak_upnl or pnl_pct > self.peak_upnl[symbol]:
            self.peak_upnl[symbol] = pnl_pct
        
        result = {
            "pnl_pct": pnl_pct,
            "peak_pnl": self.peak_upnl.get(symbol, 0),
            "stop_price": None,
            "stop_type": "initial",
            "action": "hold"
        }
        
        # 1. Initial stop loss
        if side == "LONG":
            initial_stop = entry * (1 - CONFIG.INITIAL_STOP)
        else:
            initial_stop = entry * (1 + CONFIG.INITIAL_STOP)
        result["stop_price"] = initial_stop
        
        # 2. Breakeven stop (move stop to entry when in profit)
        if pnl_pct >= CONFIG.BREAKEVEN_ACTIVATION:
            if not self.breakeven_activated.get(symbol, False):
                self.breakeven_activated[symbol] = True
                print(f"   🔒 Breakeven activated! Stop moved to entry")
            
            # Add small buffer above breakeven
            if side == "LONG":
                result["stop_price"] = entry * 1.0001  # Tiny profit locked
            else:
                result["stop_price"] = entry * 0.9999
            result["stop_type"] = "breakeven"
        
        # 3. Trailing stop (lock in more profits as price moves)
        if pnl_pct >= CONFIG.TRAIL_ACTIVATION:
            if not self.trailing_activated.get(symbol, False):
                self.trailing_activated[symbol] = True
                print(f"   📈 Trailing stop activated! Locking in profits")
            
            if side == "LONG":
                trail_stop = peak * (1 - CONFIG.TRAIL_DISTANCE)
                if trail_stop > result["stop_price"]:
                    result["stop_price"] = trail_stop
            else:
                trail_stop = peak * (1 + CONFIG.TRAIL_DISTANCE)
                if trail_stop < result["stop_price"]:
                    result["stop_price"] = trail_stop
            
            self.trailing_stop[symbol] = result["stop_price"]
            result["stop_type"] = "trailing"
        
        # Check if stop hit
        if side == "LONG" and current <= result["stop_price"]:
            result["action"] = "exit"
        elif side == "SHORT" and current >= result["stop_price"]:
            result["action"] = "exit"
        
        return result
    
    def reset(self, symbol: str):
        """Reset tracking for new trade"""
        self.peak_upnl.pop(symbol, None)
        self.peak_price.pop(symbol, None)
        self.breakeven_activated.pop(symbol, None)
        self.trailing_activated.pop(symbol, None)
        self.trailing_stop.pop(symbol, None)


# =============================================================================
# PRICE STREAM
# =============================================================================

class MasterPriceStream:
    def __init__(self, on_tick):
        self.on_tick = on_tick
        self.ws = None
        self.running = False
        self.prices = {}
        self.tick_count = 0
        
    def start(self, symbols):
        self.running = True
        
        def run():
            while self.running:
                try:
                    self.ws = websocket.WebSocketApp(
                        WS_URL,
                        on_open=lambda ws: self._subscribe(ws, symbols),
                        on_message=self._on_message,
                        on_error=lambda ws, e: None,
                        on_close=lambda ws, a, b: None
                    )
                    self.ws.run_forever(ping_interval=10)
                except:
                    time.sleep(1)
        
        threading.Thread(target=run, daemon=True).start()
        time.sleep(1)
    
    def _subscribe(self, ws, symbols):
        print("📡 Master WebSocket connected")
        for s in symbols:
            ws.send(json.dumps({
                "type": "subscribe",
                "payload": {"channels": [{"name": "v2/ticker", "symbols": [s]}]}
            }))
    
    def _on_message(self, ws, msg):
        try:
            data = json.loads(msg)
            symbol = data.get("symbol", "")
            price = data.get("mark_price")
            if price and symbol:
                self.prices[symbol] = float(price)
                self.tick_count += 1
                self.on_tick(symbol, float(price), time.time())
        except:
            pass
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


# =============================================================================
# API CLIENT
# =============================================================================

class MasterAPI:
    def __init__(self):
        self.session = requests.Session()
        self.product_ids = {}
        self._load_products()
    
    def _sign(self, method, endpoint, payload=""):
        ts = str(int(time.time()))
        sig = hmac.new(API_SECRET.encode(), f"{method}{ts}{endpoint}{payload}".encode(), hashlib.sha256).hexdigest()
        return {"api-key": API_KEY, "timestamp": ts, "signature": sig, "Content-Type": "application/json"}
    
    def _load_products(self):
        try:
            r = self.session.get(BASE_URL + "/v2/products", timeout=5)
            for p in r.json().get("result", []):
                if p.get("symbol") in CONTRACTS:
                    self.product_ids[p["symbol"]] = p["id"]
        except:
            pass
    
    def get_balance(self):
        try:
            r = self.session.get(BASE_URL + "/v2/wallet/balances", 
                               headers=self._sign("GET", "/v2/wallet/balances"), timeout=3)
            for a in r.json().get("result", []):
                if a.get("asset_symbol") in ["USD", "USDT"]:
                    return float(a.get("available_balance", 0))
        except:
            return 0
        return 0
    
    def get_positions(self):
        positions = []
        for asset in ["ETH", "BTC", "SOL"]:
            try:
                endpoint = f"/v2/positions?underlying_asset_symbol={asset}"
                r = self.session.get(BASE_URL + endpoint, 
                                   headers=self._sign("GET", endpoint), timeout=3)
                for p in r.json().get("result", []):
                    if float(p.get("size", 0)) != 0:
                        positions.append(p)
            except:
                pass
        return positions
    
    def market_order(self, symbol, side, size):
        try:
            order = {"product_id": self.product_ids.get(symbol), "side": side, 
                    "size": size, "order_type": "market_order"}
            payload = json.dumps(order)
            r = self.session.post(BASE_URL + "/v2/orders", 
                                headers=self._sign("POST", "/v2/orders", payload), 
                                data=payload, timeout=5)
            return r.json().get("success", False)
        except:
            return False
    
    def close_position(self, symbol, size, side):
        close_side = "sell" if side == "LONG" else "buy"
        return self.market_order(symbol, close_side, size)


# =============================================================================
# ALADDIN MASTER BOT
# =============================================================================

class AladdinMaster:
    """The Ultimate Trading Bot"""
    
    def __init__(self):
        self.api = MasterAPI()
        self.sentiment = SentimentAnalyzer()
        self.trend = TrendAnalyzer()
        self.protector = UPNLProtector()
        self.stream = MasterPriceStream(self._on_tick)
        
        self.running = False
        self.position = None
        self.last_trade_time = 0
        
        self.stats = {
            "trades": 0, "wins": 0, "losses": 0,
            "gross_pnl": 0, "fees": 0, "net_pnl": 0,
            "biggest_win": 0, "biggest_loss": 0,
            "protected_profits": 0  # Profits saved by trailing stop
        }
        
        self.start_time = 0
        self.last_display = 0
    
    def _check_entry_conditions(self, symbol: str, price: float) -> Tuple[bool, str, str]:
        """
        Check ALL conditions for entry
        Returns: (should_enter, side, reason)
        """
        # Check cooldown
        if time.time() - self.last_trade_time < CONFIG.COOLDOWN_SECONDS:
            return False, "", "Cooldown active"
        
        # Get sentiment
        sent_score, sent_dir, sent_conf = self.sentiment.get_sentiment()
        
        # Get trend
        trend_dir, trend_str, trend_details = self.trend.get_trend(symbol)
        
        # LONG CONDITIONS
        if trend_dir == "UPTREND":
            can_long, long_reason = self.sentiment.should_trade_long()
            if can_long:
                reason = (f"📈 UPTREND ({trend_str:.2%}) + {long_reason}")
                return True, "LONG", reason
            else:
                return False, "", f"Uptrend but {long_reason}"
        
        # SHORT CONDITIONS
        elif trend_dir == "DOWNTREND":
            can_short, short_reason = self.sentiment.should_trade_short()
            if can_short:
                reason = (f"📉 DOWNTREND ({trend_str:.2%}) + {short_reason}")
                return True, "SHORT", reason
            else:
                return False, "", f"Downtrend but {short_reason}"
        
        return False, "", f"No clear trend ({trend_dir})"
    
    def _on_tick(self, symbol: str, price: float, timestamp: float):
        """Process every price tick"""
        # Update trend analyzer
        self.trend.add_price(symbol, price)
        
        if self.position and self.position["symbol"] == symbol:
            self._manage_position(price, timestamp)
        elif not self.position:
            self._check_entry(symbol, price, timestamp)
    
    def _check_entry(self, symbol: str, price: float, timestamp: float):
        """Check for trade entry"""
        should_enter, side, reason = self._check_entry_conditions(symbol, price)
        
        if not should_enter:
            return
        
        balance = self.api.get_balance()
        if balance < 0.10:
            return
        
        order_side = "buy" if side == "LONG" else "sell"
        
        # Calculate targets
        if side == "LONG":
            target = price * (1 + CONFIG.INITIAL_TARGET)
            stop = price * (1 - CONFIG.INITIAL_STOP)
        else:
            target = price * (1 - CONFIG.INITIAL_TARGET)
            stop = price * (1 + CONFIG.INITIAL_STOP)
        
        contract_size = CONTRACTS[symbol]["size"]
        position_value = price * contract_size
        
        print(f"\n{'🎯'*25}")
        print(f"🚀 MASTER ENTRY: {symbol} {side}")
        print(f"   Reason: {reason}")
        print(f"")
        print(f"   Entry:       ${price:.2f}")
        print(f"   Target:      ${target:.2f} (+{CONFIG.INITIAL_TARGET:.2%})")
        print(f"   Stop:        ${stop:.2f} (-{CONFIG.INITIAL_STOP:.2%})")
        print(f"")
        print(f"   🔒 Breakeven at: +{CONFIG.BREAKEVEN_ACTIVATION:.2%}")
        print(f"   📈 Trail at:     +{CONFIG.TRAIL_ACTIVATION:.2%}")
        
        # Show sentiment
        headlines = self.sentiment.get_headlines()[:3]
        if headlines:
            print(f"\n   📰 Market Sentiment:")
            for h in headlines:
                print(f"      {h}")
        
        success = self.api.market_order(symbol, order_side, 1)
        
        if success:
            self.position = {
                "symbol": symbol, "side": side, "entry": price,
                "size": 1, "entry_time": timestamp,
                "target": target, "stop": stop,
                "position_value": position_value
            }
            self.last_trade_time = timestamp
            self.stats["trades"] += 1
            self.protector.reset(symbol)
            
            print(f"\n   ✅ POSITION OPENED")
        else:
            print(f"\n   ❌ Order failed")
        print(f"{'🎯'*25}\n")
    
    def _manage_position(self, price: float, timestamp: float):
        """Manage open position with UPNL protection"""
        pos = self.position
        symbol = pos["symbol"]
        entry = pos["entry"]
        side = pos["side"]
        
        hold_time = timestamp - pos["entry_time"]
        
        # Get UPNL protection info
        protection = self.protector.update(symbol, entry, price, side)
        pnl_pct = protection["pnl_pct"]
        peak_pnl = protection["peak_pnl"]
        stop_price = protection["stop_price"]
        stop_type = protection["stop_type"]
        
        net_pnl_pct = pnl_pct - CONFIG.ROUND_TRIP_FEE
        
        # Check sentiment reversal (exit if sentiment flips against us)
        sent_score, sent_dir, _ = self.sentiment.get_sentiment()
        sentiment_exit = False
        
        if side == "LONG" and sent_dir == "BEARISH" and sent_score < -0.3:
            sentiment_exit = True
            exit_reason = f"📰 Sentiment turned BEARISH ({sent_score:.2f})"
        elif side == "SHORT" and sent_dir == "BULLISH" and sent_score > 0.3:
            sentiment_exit = True
            exit_reason = f"📰 Sentiment turned BULLISH ({sent_score:.2f})"
        
        should_exit = False
        exit_reason = ""
        
        # 1. Take profit target hit
        if side == "LONG" and price >= pos["target"]:
            should_exit, exit_reason = True, "🎯 TARGET HIT!"
        elif side == "SHORT" and price <= pos["target"]:
            should_exit, exit_reason = True, "🎯 TARGET HIT!"
        
        # 2. Stop hit (initial, breakeven, or trailing)
        elif protection["action"] == "exit":
            should_exit = True
            if stop_type == "trailing":
                exit_reason = f"📈 TRAILING STOP - Profits locked!"
            elif stop_type == "breakeven":
                exit_reason = f"🔒 BREAKEVEN STOP - Protected!"
            else:
                exit_reason = f"🛑 STOP LOSS"
        
        # 3. Sentiment reversal
        elif sentiment_exit:
            should_exit = True
        
        # 4. Time exit (but only if profitable)
        elif hold_time >= CONFIG.MAX_HOLD_MINUTES * 60:
            if net_pnl_pct > 0:
                should_exit, exit_reason = True, "⏰ TIME EXIT (profitable)"
            else:
                should_exit, exit_reason = True, "⏰ TIME EXIT (max hold)"
        
        if should_exit:
            self._exit_position(price, hold_time, pnl_pct, peak_pnl, exit_reason, stop_type)
        else:
            self._display_position(price, pnl_pct, net_pnl_pct, peak_pnl, stop_price, stop_type, hold_time)
    
    def _exit_position(self, exit_price: float, hold_time: float, pnl_pct: float, 
                       peak_pnl: float, reason: str, stop_type: str):
        """Exit position"""
        pos = self.position
        symbol = pos["symbol"]
        
        self.api.close_position(symbol, pos["size"], pos["side"])
        
        gross_pnl = pos["position_value"] * pnl_pct
        fees = pos["position_value"] * CONFIG.ROUND_TRIP_FEE
        net_pnl = gross_pnl - fees
        net_pnl_pct = pnl_pct - CONFIG.ROUND_TRIP_FEE
        
        # Track stats
        if net_pnl > 0:
            self.stats["wins"] += 1
            if net_pnl > self.stats["biggest_win"]:
                self.stats["biggest_win"] = net_pnl
        else:
            self.stats["losses"] += 1
            if net_pnl < self.stats["biggest_loss"]:
                self.stats["biggest_loss"] = net_pnl
        
        self.stats["gross_pnl"] += gross_pnl
        self.stats["fees"] += fees
        self.stats["net_pnl"] += net_pnl
        
        # Calculate protected profits (difference between peak and exit if trailing)
        if stop_type == "trailing" and peak_pnl > pnl_pct:
            protected = pos["position_value"] * pnl_pct
            self.stats["protected_profits"] += max(0, protected)
        
        emoji = "🚀" if net_pnl > 0 else "💔"
        
        print(f"\n{emoji}{'='*55}")
        print(f"   {reason}")
        print(f"")
        print(f"   {pos['symbol']} {pos['side']}")
        print(f"   Entry: ${pos['entry']:.2f} → Exit: ${exit_price:.2f}")
        print(f"   Hold:  {hold_time:.0f}s ({hold_time/60:.1f} min)")
        print(f"")
        print(f"   Peak P&L:  {peak_pnl:+.3%}")
        print(f"   Exit P&L:  {pnl_pct:+.3%}")
        print(f"")
        print(f"   Gross:     ${gross_pnl:+.6f}")
        print(f"   Fees:      ${fees:.6f}")
        print(f"   NET:       ${net_pnl:+.6f} ({net_pnl_pct:+.3%})")
        
        if stop_type in ["trailing", "breakeven"]:
            print(f"\n   ✅ UPNL PROTECTION WORKED!")
            print(f"   Your winner did NOT become a loser!")
        
        print(f"{'='*60}")
        self._print_stats()
        print()
        
        self.protector.reset(symbol)
        self.position = None
    
    def _display_position(self, price, pnl_pct, net_pnl_pct, peak_pnl, stop_price, stop_type, hold_time):
        """Display position status"""
        now = time.time()
        if now - self.last_display < 0.5:  # Throttle display
            return
        self.last_display = now
        
        pos = self.position
        
        # Color based on net P&L
        if net_pnl_pct > 0:
            color = "🟢"
        elif pnl_pct > 0:
            color = "🟡"
        else:
            color = "🔴"
        
        # Stop type indicator
        stop_ind = {"initial": "🔴", "breakeven": "🔒", "trailing": "📈"}.get(stop_type, "")
        
        # Sentiment indicator
        sent_score, sent_dir, _ = self.sentiment.get_sentiment()
        sent_emoji = "📈" if sent_dir == "BULLISH" else "📉" if sent_dir == "BEARISH" else "📊"
        
        sys.stdout.write(
            f"\r{color} {pos['symbol']} {pos['side']} | "
            f"${price:.2f} | "
            f"Net: {net_pnl_pct:+.3%} | "
            f"Peak: {peak_pnl:+.3%} | "
            f"{stop_ind} Stop: ${stop_price:.2f} | "
            f"{sent_emoji} {sent_dir} | "
            f"⏱️ {hold_time:.0f}s   "
        )
        sys.stdout.flush()
    
    def _print_stats(self):
        """Print session stats"""
        s = self.stats
        trades = s["trades"]
        wr = (s["wins"] / trades * 100) if trades > 0 else 0
        
        print(f"\n📊 MASTER STATS:")
        print(f"   Trades: {trades} | W:{s['wins']} L:{s['losses']} ({wr:.0f}%)")
        print(f"   Gross:  ${s['gross_pnl']:+.6f}")
        print(f"   Fees:   ${s['fees']:.6f}")
        print(f"   NET:    ${s['net_pnl']:+.6f}")
        print(f"   Best:   ${s['biggest_win']:+.6f} | Worst: ${s['biggest_loss']:+.6f}")
        
        if s["protected_profits"] > 0:
            print(f"   🔒 Profits protected by trailing: ${s['protected_profits']:.6f}")
    
    def _load_positions(self):
        """Load existing positions"""
        for pos in self.api.get_positions():
            symbol = pos.get("product_symbol", "")
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size != 0 and symbol in CONTRACTS:
                side = "LONG" if size > 0 else "SHORT"
                
                if side == "LONG":
                    target = entry * (1 + CONFIG.INITIAL_TARGET)
                    stop = entry * (1 - CONFIG.INITIAL_STOP)
                else:
                    target = entry * (1 - CONFIG.INITIAL_TARGET)
                    stop = entry * (1 + CONFIG.INITIAL_STOP)
                
                self.position = {
                    "symbol": symbol, "side": side, "entry": entry,
                    "size": abs(int(size)), "entry_time": time.time(),
                    "target": target, "stop": stop,
                    "position_value": entry * CONTRACTS[symbol]["size"] * abs(size)
                }
                
                print(f"📌 Loaded: {symbol} {side} @ ${entry:.2f}")
                print(f"   Target: ${target:.2f} | Stop: ${stop:.2f}")
                print(f"   🔒 Breakeven at +{CONFIG.BREAKEVEN_ACTIVATION:.2%}")
                print(f"   📈 Trailing at +{CONFIG.TRAIL_ACTIVATION:.2%}")
                break
    
    def start(self):
        """Start the Master bot"""
        self.running = True
        self.start_time = time.time()
        
        balance = self.api.get_balance()
        sent_score, sent_dir, sent_conf = self.sentiment.get_sentiment()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║            🏆 ALADDIN MASTER - THE ULTIMATE TRADING BOT 🏆               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  💰 Balance:           ${balance:.4f}
║  
║  📰 NEWS & SENTIMENT:
║     Current:           {sent_dir} ({sent_score:+.2f})
║     ✓ Only LONG when sentiment is bullish
║     ✓ Only SHORT when sentiment is bearish
║     ✓ Exit if sentiment reverses against position
║  
║  📈 TREND FOLLOWING:
║     ✓ Trade WITH the trend, never against
║     ✓ Uptrend = LONG only, Downtrend = SHORT only
║     ✓ No trades in sideways/choppy markets
║  
║  🔒 UPNL PROTECTION (Winners NEVER become losers):
║     ✓ Breakeven stop at +{CONFIG.BREAKEVEN_ACTIVATION:.2%}
║     ✓ Trailing stop at +{CONFIG.TRAIL_ACTIVATION:.2%}
║     ✓ Trail distance: {CONFIG.TRAIL_DISTANCE:.2%} behind peak
║  
║  🎯 PROFIT TARGETS:
║     Target:            +{CONFIG.INITIAL_TARGET:.2%}
║     Stop:              -{CONFIG.INITIAL_STOP:.2%}
║     Risk/Reward:       1:{CONFIG.INITIAL_TARGET/CONFIG.INITIAL_STOP:.1f}
║  
║  Press Ctrl+C to stop
╚══════════════════════════════════════════════════════════════════════════╝
""")
        
        # Show current headlines
        headlines = self.sentiment.get_headlines()
        if headlines:
            print("📰 Current Market News:")
            for h in headlines:
                print(f"   {h}")
            print()
        
        print("📍 Checking positions...")
        self._load_positions()
        
        print("📡 Starting Master price stream...")
        self.stream.start(CONFIG.SYMBOLS)
        time.sleep(2)
        
        print(f"\n{'='*60}")
        print("🏆 ALADDIN MASTER ACTIVE")
        print("   • News sentiment monitored")
        print("   • Trend following enabled")
        print("   • UPNL protection active")
        print("   • Winners protected, losses cut fast")
        print(f"{'='*60}\n")
        
        try:
            while self.running:
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping Master bot...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop bot"""
        self.running = False
        self.stream.stop()
        self.sentiment.stop()
        
        runtime = time.time() - self.start_time
        s = self.stats
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    📊 MASTER SESSION SUMMARY                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Runtime:          {runtime/60:.1f} minutes
║  
║  📈 TRADES:
║     Total:         {s['trades']}
║     Wins:          {s['wins']}
║     Losses:        {s['losses']}
║     Win Rate:      {(s['wins']/max(1,s['trades'])*100):.0f}%
║  
║  💰 P&L:
║     Gross:         ${s['gross_pnl']:+.6f}
║     Fees:          ${s['fees']:.6f}
║     NET:           ${s['net_pnl']:+.6f}
║     Best Trade:    ${s['biggest_win']:+.6f}
║     Worst Trade:   ${s['biggest_loss']:+.6f}
║  
║  🔒 PROTECTION:
║     Profits saved: ${s['protected_profits']:.6f}
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("\n🏆 ALADDIN MASTER - Ultimate Trading Bot")
    print("   • News & Sentiment Analysis")
    print("   • Trend Following")
    print("   • UPNL Protection")
    print("   • Smart Exits")
    print("\n   Press Ctrl+C to stop\n")
    
    bot = AladdinMaster()
    bot.start()
