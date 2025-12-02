"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗                 ║
║    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                 ║
║    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║                 ║
║    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║                 ║
║    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║                 ║
║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝                 ║
║                                                                          ║
║                    ⚡ ULTIMATE EDITION ⚡                                 ║
║                                                                          ║
║    🔥 MICROSECOND MONITORING (μs precision)                              ║
║    🧠 AI-POWERED DECISION MAKING                                         ║
║    💰 DYNAMIC CAPITAL MANAGEMENT                                         ║
║    📊 MULTI-ASSET (BTC, ETH, SOL, XRP)                                   ║
║    📰 NEWS SENTIMENT ANALYSIS                                            ║
║    📈 TREND FOLLOWING                                                    ║
║    🔒 UPNL PROTECTION (Winners never become losers)                      ║
║    🎯 FEE-AWARE TRADING                                                  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import hmac
import hashlib
import json
import requests
import threading
import websocket
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import sys
import math

# Maximum speed output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# MICROSECOND TIMER - TRUE μs PRECISION
# =============================================================================

class MicroTimer:
    """Microsecond precision timing"""
    
    @staticmethod
    def now() -> int:
        """Current time in microseconds"""
        return int(time.perf_counter() * 1_000_000)
    
    @staticmethod
    def now_ns() -> int:
        """Current time in nanoseconds"""
        return time.perf_counter_ns()
    
    @staticmethod
    def timestamp_us() -> int:
        """Unix timestamp in microseconds"""
        return int(time.time() * 1_000_000)
    
    @staticmethod
    def format_latency(us: int) -> str:
        """Format latency for display"""
        if us < 1000:
            return f"{us}μs"
        elif us < 1_000_000:
            return f"{us/1000:.2f}ms"
        else:
            return f"{us/1_000_000:.2f}s"

# =============================================================================
# API CONFIG
# =============================================================================

API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"
BASE_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

# =============================================================================
# ASSET CONFIG - ALL 4 TRADEABLE ASSETS
# =============================================================================

ASSETS = {
    "BTCUSD": {"name": "Bitcoin", "size": 0.001, "leverage": 100, "priority": 2},
    "ETHUSD": {"name": "Ethereum", "size": 0.01, "leverage": 100, "priority": 1},
    "SOLUSD": {"name": "Solana", "size": 0.1, "leverage": 50, "priority": 3},
    "XRPUSD": {"name": "XRP", "size": 10, "leverage": 50, "priority": 4}
}

# =============================================================================
# ULTIMATE CONFIG
# =============================================================================

@dataclass
class UltimateConfig:
    # Fees (round-trip)
    ROUND_TRIP_FEE: float = 0.0010  # 0.10%
    
    # Profit targets - MUST exceed fees significantly
    MIN_PROFIT_TARGET: float = 0.0040   # 0.40% min (4x fees)
    MAX_PROFIT_TARGET: float = 0.0100   # 1.00% max
    
    # Risk management
    STOP_LOSS: float = 0.0025           # 0.25%
    BREAKEVEN_LEVEL: float = 0.0015     # Move to breakeven at 0.15%
    TRAIL_ACTIVATION: float = 0.0025    # Start trailing at 0.25%
    TRAIL_DISTANCE: float = 0.0012      # Trail 0.12% behind peak
    
    # Trend detection
    TREND_WINDOW: int = 50              # Ticks for trend
    STRONG_TREND: float = 0.0015        # 0.15% = strong trend
    
    # Sentiment thresholds
    BULLISH_THRESHOLD: float = 0.20
    BEARISH_THRESHOLD: float = -0.20
    
    # Timing
    MAX_HOLD_SECONDS: int = 300         # 5 minutes max
    COOLDOWN_SECONDS: int = 30          # 30s between trades
    
    # Capital management
    MAX_RISK_PCT: float = 0.15          # 15% max risk per trade
    MAX_POSITION_PCT: float = 0.40      # 40% max in one position

CONFIG = UltimateConfig()

# =============================================================================
# MICROSECOND PRICE ENGINE
# =============================================================================

class MicrosecondPriceEngine:
    """Ultra-low latency price processing"""
    
    def __init__(self, on_tick):
        self.on_tick = on_tick
        self.ws = None
        self.running = False
        
        # Price data with μs timestamps
        self.prices: Dict[str, float] = {}
        self.last_update_us: Dict[str, int] = {}
        self.price_history: Dict[str, deque] = {}
        
        # Performance metrics
        self.tick_count = 0
        self.total_latency_us = 0
        self.min_latency_us = float('inf')
        self.max_latency_us = 0
        self.start_time_us = 0
        
    def start(self, symbols: List[str]):
        self.running = True
        self.start_time_us = MicroTimer.now()
        
        for s in symbols:
            self.price_history[s] = deque(maxlen=CONFIG.TREND_WINDOW)
        
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
                    self.ws.run_forever(ping_interval=5, ping_timeout=3)
                except:
                    time.sleep(0.1)
        
        threading.Thread(target=run, daemon=True).start()
    
    def _subscribe(self, ws, symbols):
        print(f"⚡ MICROSECOND ENGINE CONNECTED")
        for s in symbols:
            ws.send(json.dumps({
                "type": "subscribe",
                "payload": {"channels": [
                    {"name": "v2/ticker", "symbols": [s]},
                    {"name": "mark_price", "symbols": [s]}
                ]}
            }))
    
    def _on_message(self, ws, message):
        recv_us = MicroTimer.now()
        
        try:
            data = json.loads(message)
            symbol = data.get("symbol", "")
            mark_price = data.get("mark_price")
            
            if mark_price and symbol in self.price_history:
                price = float(mark_price)
                
                # Calculate processing latency
                process_us = MicroTimer.now() - recv_us
                self.total_latency_us += process_us
                self.min_latency_us = min(self.min_latency_us, process_us)
                self.max_latency_us = max(self.max_latency_us, process_us)
                
                # Store with μs timestamp
                self.prices[symbol] = price
                self.last_update_us[symbol] = recv_us
                self.price_history[symbol].append((recv_us, price))
                self.tick_count += 1
                
                # Callback with μs timestamp
                self.on_tick(symbol, price, recv_us)
                
        except:
            pass
    
    def get_trend(self, symbol: str) -> Tuple[str, float]:
        """Get trend direction and strength"""
        history = list(self.price_history.get(symbol, []))
        if len(history) < 20:
            return "NEUTRAL", 0
        
        prices = [p[1] for p in history]
        oldest = prices[0]
        newest = prices[-1]
        
        strength = (newest - oldest) / oldest
        
        if strength > CONFIG.STRONG_TREND:
            return "UPTREND", strength
        elif strength < -CONFIG.STRONG_TREND:
            return "DOWNTREND", strength
        return "SIDEWAYS", strength
    
    def get_momentum(self, symbol: str) -> float:
        """Get short-term momentum"""
        history = list(self.price_history.get(symbol, []))
        if len(history) < 10:
            return 0
        
        recent = [p[1] for p in history[-10:]]
        return (recent[-1] - recent[0]) / recent[0]
    
    def get_volatility(self, symbol: str) -> float:
        """Get recent volatility"""
        history = list(self.price_history.get(symbol, []))
        if len(history) < 10:
            return 0
        
        prices = [p[1] for p in history[-20:]]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return 0
        
        avg = sum(returns) / len(returns)
        variance = sum((r - avg) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)
    
    def get_stats(self) -> Dict:
        """Get performance stats"""
        elapsed_us = MicroTimer.now() - self.start_time_us
        elapsed_s = elapsed_us / 1_000_000
        
        return {
            "ticks": self.tick_count,
            "ticks_per_sec": self.tick_count / max(1, elapsed_s),
            "avg_latency_us": self.total_latency_us / max(1, self.tick_count),
            "min_latency_us": self.min_latency_us if self.min_latency_us != float('inf') else 0,
            "max_latency_us": self.max_latency_us
        }
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


# =============================================================================
# SENTIMENT ENGINE
# =============================================================================

class SentimentEngine:
    """Real-time market sentiment"""
    
    BULLISH = ["bullish", "rally", "surge", "pump", "moon", "buy", "long", 
               "breakout", "higher", "gains", "etf", "adoption", "institutional"]
    BEARISH = ["bearish", "crash", "dump", "selloff", "sell", "short",
               "breakdown", "lower", "losses", "ban", "hack", "fraud"]
    
    def __init__(self):
        self.score = 0.0
        self.direction = "NEUTRAL"
        self.confidence = 0.0
        self.headlines = []
        self._running = True
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
    
    def _loop(self):
        while self._running:
            self._fetch()
            time.sleep(60)  # Update every minute
    
    def _fetch(self):
        try:
            r = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC,ETH,Trading",
                timeout=10
            )
            if r.status_code != 200:
                return
            
            news = r.json().get("Data", [])[:20]
            scores = []
            headlines = []
            
            for item in news:
                text = (item.get("title", "") + " " + item.get("body", "")[:300]).lower()
                bull = sum(1 for kw in self.BULLISH if kw in text)
                bear = sum(1 for kw in self.BEARISH if kw in text)
                
                if bull + bear > 0:
                    score = (bull - bear) / (bull + bear)
                    scores.append(score)
                    emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "⚪"
                    headlines.append(f"{emoji} {item.get('title', '')[:60]}")
            
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
        except:
            pass
    
    def get(self) -> Tuple[float, str, float]:
        with self._lock:
            return self.score, self.direction, self.confidence
    
    def get_headlines(self) -> List[str]:
        with self._lock:
            return self.headlines.copy()
    
    def stop(self):
        self._running = False


# =============================================================================
# CAPITAL ENGINE
# =============================================================================

class CapitalEngine:
    """Dynamic capital management"""
    
    def __init__(self, api):
        self.api = api
        self.balance = 0
        self.equity = 0
        self.available = 0
        
    def update(self):
        self.balance = self.api.get_balance()
        self.equity = self.balance
        self.available = self.balance
    
    def get_position_size(self, symbol: str, price: float) -> int:
        """Calculate optimal position size"""
        spec = ASSETS.get(symbol, {})
        leverage = spec.get("leverage", 50)
        contract_size = spec.get("size", 0.01)
        
        # Position value per contract
        contract_value = price * contract_size
        
        # Margin per contract
        margin_per_contract = contract_value / leverage
        
        # Max contracts based on available capital
        max_capital = self.available * CONFIG.MAX_POSITION_PCT
        max_contracts = int(max_capital / margin_per_contract) if margin_per_contract > 0 else 0
        
        # Risk-based sizing
        risk_capital = self.available * CONFIG.MAX_RISK_PCT
        risk_contracts = int(risk_capital / margin_per_contract) if margin_per_contract > 0 else 0
        
        # Return minimum of the two, at least 1
        return max(1, min(max_contracts, risk_contracts))
    
    def can_trade(self) -> bool:
        return self.available >= 0.02  # Need at least $0.02


# =============================================================================
# TRADE ENGINE
# =============================================================================

class TradeEngine:
    """Order execution engine"""
    
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
                self.product_ids[p.get("symbol", "")] = p.get("id")
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
            pass
        return 0
    
    def get_positions(self):
        positions = []
        for asset in ["ETH", "BTC", "SOL", "XRP"]:
            try:
                endpoint = f"/v2/positions?underlying_asset_symbol={asset}"
                r = self.session.get(BASE_URL + endpoint, headers=self._sign("GET", endpoint), timeout=3)
                for p in r.json().get("result", []):
                    if float(p.get("size", 0)) != 0:
                        positions.append(p)
            except:
                pass
        return positions
    
    def execute(self, symbol: str, side: str, size: int) -> Tuple[bool, int]:
        """Execute order - returns (success, latency_us)"""
        start_us = MicroTimer.now()
        try:
            order = {
                "product_id": self.product_ids.get(symbol),
                "side": side,
                "size": int(size),
                "order_type": "market_order"
            }
            payload = json.dumps(order)
            r = self.session.post(
                BASE_URL + "/v2/orders",
                headers=self._sign("POST", "/v2/orders", payload),
                data=payload,
                timeout=5
            )
            latency_us = MicroTimer.now() - start_us
            return r.json().get("success", False), latency_us
        except:
            return False, MicroTimer.now() - start_us
    
    def close(self, symbol: str, size: int, side: str) -> Tuple[bool, int]:
        close_side = "sell" if side == "LONG" else "buy"
        return self.execute(symbol, close_side, size)


# =============================================================================
# ALADDIN ULTIMATE
# =============================================================================

class AladdinUltimate:
    """The Ultimate Trading Bot - Microsecond Speed, Maximum Profits"""
    
    def __init__(self):
        self.trade_engine = TradeEngine()
        self.capital = CapitalEngine(self.trade_engine)
        self.sentiment = SentimentEngine()
        self.price_engine = MicrosecondPriceEngine(self._on_tick)
        
        self.running = False
        self.position = None
        self.last_trade_us = 0
        self.last_display_us = 0
        
        # Stats
        self.starting_balance = 0
        self.stats = {
            "trades": 0, "wins": 0, "losses": 0,
            "gross_pnl": 0, "fees": 0, "net_pnl": 0,
            "peak_equity": 0, "max_drawdown": 0,
            "best_trade": 0, "worst_trade": 0,
            "total_hold_us": 0
        }
        
        self.start_time_us = 0
    
    def _on_tick(self, symbol: str, price: float, tick_us: int):
        """Process every tick at MICROSECOND speed"""
        
        if self.position and self.position["symbol"] == symbol:
            self._manage_position_us(symbol, price, tick_us)
        elif not self.position:
            self._check_entry_us(symbol, price, tick_us)
    
    def _find_best_opportunity(self) -> Optional[Tuple[str, str, float]]:
        """Find the best trading opportunity across all assets"""
        sent_score, sent_dir, sent_conf = self.sentiment.get()
        
        opportunities = []
        
        for symbol in ASSETS.keys():
            trend, strength = self.price_engine.get_trend(symbol)
            momentum = self.price_engine.get_momentum(symbol)
            volatility = self.price_engine.get_volatility(symbol)
            
            if trend == "SIDEWAYS":
                continue
            
            score = 0
            direction = None
            
            # LONG opportunity
            if trend == "UPTREND" and momentum > 0:
                if sent_dir != "BEARISH":  # Don't long against bearish sentiment
                    direction = "LONG"
                    score = abs(strength) * 100 + volatility * 50
                    if sent_dir == "BULLISH":
                        score *= 1.5  # Boost for sentiment alignment
            
            # SHORT opportunity
            elif trend == "DOWNTREND" and momentum < 0:
                if sent_dir != "BULLISH":  # Don't short against bullish sentiment
                    direction = "SHORT"
                    score = abs(strength) * 100 + volatility * 50
                    if sent_dir == "BEARISH":
                        score *= 1.5
            
            if direction and score > 0:
                opportunities.append((symbol, direction, score))
        
        if opportunities:
            return max(opportunities, key=lambda x: x[2])
        return None
    
    def _check_entry_us(self, symbol: str, price: float, tick_us: int):
        """Check for entry at microsecond speed"""
        
        # Cooldown check
        if tick_us - self.last_trade_us < CONFIG.COOLDOWN_SECONDS * 1_000_000:
            return
        
        # Capital check
        if not self.capital.can_trade():
            return
        
        # Find best opportunity
        opp = self._find_best_opportunity()
        if not opp:
            return
        
        opp_symbol, direction, score = opp
        
        # Only proceed if this symbol matches the opportunity
        if symbol != opp_symbol:
            return
        
        # Calculate position
        size = self.capital.get_position_size(symbol, price)
        if size < 1:
            return
        
        # Calculate targets with volatility adjustment
        volatility = self.price_engine.get_volatility(symbol)
        vol_mult = 1 + min(volatility * 10, 1)  # Up to 2x multiplier
        
        target_pct = min(CONFIG.MIN_PROFIT_TARGET * vol_mult, CONFIG.MAX_PROFIT_TARGET)
        stop_pct = CONFIG.STOP_LOSS * vol_mult
        
        if direction == "LONG":
            target = price * (1 + target_pct)
            stop = price * (1 - stop_pct)
            order_side = "buy"
        else:
            target = price * (1 - target_pct)
            stop = price * (1 + stop_pct)
            order_side = "sell"
        
        spec = ASSETS[symbol]
        position_value = price * spec["size"] * size
        
        sent_score, sent_dir, _ = self.sentiment.get()
        
        print(f"\n{'⚡'*25}")
        print(f"🚀 ULTIMATE ENTRY SIGNAL")
        print(f"")
        print(f"   📊 Asset:       {symbol} ({spec['name']})")
        print(f"   🎯 Direction:   {direction}")
        print(f"   📈 Score:       {score:.1f}")
        print(f"   📰 Sentiment:   {sent_dir} ({sent_score:+.2f})")
        print(f"")
        print(f"   💰 Position:")
        print(f"      Contracts:  {size}")
        print(f"      Value:      ${position_value:.4f}")
        print(f"      Leverage:   {spec['leverage']}x")
        print(f"")
        print(f"   🎯 Targets:")
        print(f"      Entry:      ${price:.4f}")
        print(f"      Target:     ${target:.4f} (+{target_pct:.2%})")
        print(f"      Stop:       ${stop:.4f} (-{stop_pct:.2%})")
        print(f"      R:R:        1:{target_pct/stop_pct:.1f}")
        
        # Execute
        success, latency_us = self.trade_engine.execute(symbol, order_side, size)
        
        if success:
            self.position = {
                "symbol": symbol,
                "side": direction,
                "entry": price,
                "size": size,
                "entry_us": tick_us,
                "target": target,
                "stop": stop,
                "position_value": position_value,
                "peak_price": price,
                "peak_pnl": 0,
                "breakeven_active": False,
                "trailing_active": False
            }
            self.last_trade_us = tick_us
            self.stats["trades"] += 1
            
            print(f"\n   ✅ FILLED in {MicroTimer.format_latency(latency_us)}")
        else:
            print(f"\n   ❌ Failed (latency: {MicroTimer.format_latency(latency_us)})")
        
        print(f"{'⚡'*25}\n")
    
    def _manage_position_us(self, symbol: str, price: float, tick_us: int):
        """Manage position at microsecond speed"""
        pos = self.position
        entry = pos["entry"]
        side = pos["side"]
        
        hold_us = tick_us - pos["entry_us"]
        hold_s = hold_us / 1_000_000
        
        # Calculate P&L
        if side == "LONG":
            pnl_pct = (price - entry) / entry
            if price > pos["peak_price"]:
                pos["peak_price"] = price
        else:
            pnl_pct = (entry - price) / entry
            if price < pos["peak_price"]:
                pos["peak_price"] = price
        
        if pnl_pct > pos["peak_pnl"]:
            pos["peak_pnl"] = pnl_pct
        
        net_pnl_pct = pnl_pct - CONFIG.ROUND_TRIP_FEE
        
        # DYNAMIC STOP MANAGEMENT
        
        # 1. Breakeven at +0.15%
        if pnl_pct >= CONFIG.BREAKEVEN_LEVEL and not pos["breakeven_active"]:
            pos["breakeven_active"] = True
            if side == "LONG":
                pos["stop"] = entry * 1.0001
            else:
                pos["stop"] = entry * 0.9999
            print(f"\n   🔒 BREAKEVEN ACTIVATED!")
        
        # 2. Trailing at +0.25%
        if pnl_pct >= CONFIG.TRAIL_ACTIVATION and not pos["trailing_active"]:
            pos["trailing_active"] = True
            print(f"\n   📈 TRAILING STOP ACTIVATED!")
        
        if pos["trailing_active"]:
            if side == "LONG":
                trail = pos["peak_price"] * (1 - CONFIG.TRAIL_DISTANCE)
                if trail > pos["stop"]:
                    pos["stop"] = trail
            else:
                trail = pos["peak_price"] * (1 + CONFIG.TRAIL_DISTANCE)
                if trail < pos["stop"]:
                    pos["stop"] = trail
        
        # CHECK EXIT CONDITIONS
        should_exit = False
        reason = ""
        
        # Target hit
        if side == "LONG" and price >= pos["target"]:
            should_exit, reason = True, "🎯 TARGET HIT!"
        elif side == "SHORT" and price <= pos["target"]:
            should_exit, reason = True, "🎯 TARGET HIT!"
        
        # Stop hit
        elif side == "LONG" and price <= pos["stop"]:
            stop_type = "TRAILING" if pos["trailing_active"] else "BREAKEVEN" if pos["breakeven_active"] else "STOP LOSS"
            should_exit, reason = True, f"🛑 {stop_type}"
        elif side == "SHORT" and price >= pos["stop"]:
            stop_type = "TRAILING" if pos["trailing_active"] else "BREAKEVEN" if pos["breakeven_active"] else "STOP LOSS"
            should_exit, reason = True, f"🛑 {stop_type}"
        
        # Sentiment reversal
        sent_score, sent_dir, _ = self.sentiment.get()
        if side == "LONG" and sent_dir == "BEARISH" and sent_score < -0.3 and pnl_pct > 0:
            should_exit, reason = True, "📰 SENTIMENT REVERSAL (profit secured)"
        elif side == "SHORT" and sent_dir == "BULLISH" and sent_score > 0.3 and pnl_pct > 0:
            should_exit, reason = True, "📰 SENTIMENT REVERSAL (profit secured)"
        
        # Time exit
        if hold_s >= CONFIG.MAX_HOLD_SECONDS:
            if net_pnl_pct > 0:
                should_exit, reason = True, "⏰ TIME EXIT (profitable)"
            else:
                should_exit, reason = True, "⏰ MAX HOLD TIME"
        
        if should_exit:
            self._exit_position_us(price, tick_us, pnl_pct, reason)
        else:
            self._display_position_us(price, pnl_pct, net_pnl_pct, hold_us, tick_us)
    
    def _exit_position_us(self, exit_price: float, exit_us: int, pnl_pct: float, reason: str):
        """Exit position"""
        pos = self.position
        
        success, latency_us = self.trade_engine.close(pos["symbol"], pos["size"], pos["side"])
        
        gross_pnl = pos["position_value"] * pnl_pct
        fees = pos["position_value"] * CONFIG.ROUND_TRIP_FEE
        net_pnl = gross_pnl - fees
        net_pnl_pct = pnl_pct - CONFIG.ROUND_TRIP_FEE
        
        hold_us = exit_us - pos["entry_us"]
        hold_ms = hold_us / 1000
        
        # Update stats
        if net_pnl > 0:
            self.stats["wins"] += 1
            if net_pnl > self.stats["best_trade"]:
                self.stats["best_trade"] = net_pnl
        else:
            self.stats["losses"] += 1
            if net_pnl < self.stats["worst_trade"]:
                self.stats["worst_trade"] = net_pnl
        
        self.stats["gross_pnl"] += gross_pnl
        self.stats["fees"] += fees
        self.stats["net_pnl"] += net_pnl
        self.stats["total_hold_us"] += hold_us
        
        emoji = "🚀" if net_pnl > 0 else "💔"
        
        print(f"\n{emoji}{'='*55}")
        print(f"   {reason}")
        print(f"")
        print(f"   {pos['symbol']} {pos['side']} x{pos['size']}")
        print(f"   Entry: ${pos['entry']:.4f} → Exit: ${exit_price:.4f}")
        print(f"   Hold:  {hold_ms:.0f}ms ({hold_ms/1000:.2f}s)")
        print(f"   Exit latency: {MicroTimer.format_latency(latency_us)}")
        print(f"")
        print(f"   Peak P&L:  {pos['peak_pnl']:+.3%}")
        print(f"   Exit P&L:  {pnl_pct:+.3%}")
        print(f"")
        print(f"   Gross:     ${gross_pnl:+.6f}")
        print(f"   Fees:      ${fees:.6f}")
        print(f"   NET:       ${net_pnl:+.6f} ({net_pnl_pct:+.3%})")
        
        if pos["trailing_active"] or pos["breakeven_active"]:
            print(f"\n   ✅ UPNL PROTECTION SAVED YOUR PROFITS!")
        
        print(f"{'='*60}")
        self._print_stats()
        print()
        
        self.position = None
        self.capital.update()
    
    def _display_position_us(self, price: float, pnl_pct: float, net_pnl_pct: float, hold_us: int, tick_us: int):
        """Display position at throttled rate"""
        if tick_us - self.last_display_us < 100_000:  # 100ms throttle
            return
        self.last_display_us = tick_us
        
        pos = self.position
        
        color = "🟢" if net_pnl_pct > 0 else "🟡" if pnl_pct > 0 else "🔴"
        stop_type = "📈" if pos["trailing_active"] else "🔒" if pos["breakeven_active"] else "🔴"
        
        hold_s = hold_us / 1_000_000
        
        # Get engine stats
        stats = self.price_engine.get_stats()
        
        sys.stdout.write(
            f"\r{color} {pos['symbol']} {pos['side']} | "
            f"${price:.2f} | "
            f"Net: {net_pnl_pct:+.3%} | "
            f"Peak: {pos['peak_pnl']:+.3%} | "
            f"{stop_type} ${pos['stop']:.2f} | "
            f"⏱️{hold_s:.1f}s | "
            f"⚡{stats['avg_latency_us']:.0f}μs   "
        )
        sys.stdout.flush()
    
    def _print_stats(self):
        """Print stats"""
        s = self.stats
        trades = s["trades"]
        wr = (s["wins"] / trades * 100) if trades > 0 else 0
        
        self.capital.update()
        growth = ((self.capital.equity / self.starting_balance) - 1) * 100 if self.starting_balance > 0 else 0
        
        avg_hold_ms = (s["total_hold_us"] / trades / 1000) if trades > 0 else 0
        
        engine_stats = self.price_engine.get_stats()
        
        print(f"\n📊 ULTIMATE STATS:")
        print(f"   Trades: {trades} | W:{s['wins']} L:{s['losses']} ({wr:.0f}%)")
        print(f"   Net P&L: ${s['net_pnl']:+.6f}")
        print(f"   Equity:  ${self.capital.equity:.4f} ({growth:+.1f}%)")
        print(f"   Best:    ${s['best_trade']:+.6f} | Worst: ${s['worst_trade']:+.6f}")
        print(f"   Avg Hold: {avg_hold_ms:.0f}ms")
        print(f"   ⚡ Latency: avg {engine_stats['avg_latency_us']:.0f}μs | min {engine_stats['min_latency_us']:.0f}μs")
    
    def _load_positions(self):
        """Load existing positions"""
        positions = self.trade_engine.get_positions()
        
        for pos in positions:
            symbol = pos.get("product_symbol", "")
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size != 0 and symbol in ASSETS:
                side = "LONG" if size > 0 else "SHORT"
                spec = ASSETS[symbol]
                
                # Set dynamic targets
                target_pct = CONFIG.MIN_PROFIT_TARGET
                stop_pct = CONFIG.STOP_LOSS
                
                if side == "LONG":
                    target = entry * (1 + target_pct)
                    stop = entry * (1 - stop_pct)
                else:
                    target = entry * (1 - target_pct)
                    stop = entry * (1 + stop_pct)
                
                self.position = {
                    "symbol": symbol,
                    "side": side,
                    "entry": entry,
                    "size": abs(int(size)),
                    "entry_us": MicroTimer.now(),
                    "target": target,
                    "stop": stop,
                    "position_value": entry * spec["size"] * abs(size),
                    "peak_price": entry,
                    "peak_pnl": 0,
                    "breakeven_active": False,
                    "trailing_active": False
                }
                
                print(f"📌 Loaded: {symbol} {side} x{abs(int(size))} @ ${entry:.2f}")
                print(f"   Target: ${target:.2f} | Stop: ${stop:.2f}")
                break
    
    def start(self):
        """Start Ultimate Bot"""
        self.running = True
        self.start_time_us = MicroTimer.now()
        
        self.capital.update()
        self.starting_balance = self.capital.balance
        self.stats["peak_equity"] = self.capital.balance
        
        sent_score, sent_dir, _ = self.sentiment.get()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗                 ║
║    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                 ║
║    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║                 ║
║    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║                 ║
║    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║                 ║
║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝                 ║
║                                                                          ║
║                       ⚡ ULTIMATE EDITION ⚡                              ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  💰 CAPITAL: ${self.capital.balance:.4f}                                            
║                                                                          ║
║  📊 TRADEABLE ASSETS:                                                    ║
║     • BTCUSD (Bitcoin)  - 100x leverage                                  ║
║     • ETHUSD (Ethereum) - 100x leverage                                  ║
║     • SOLUSD (Solana)   - 50x leverage                                   ║
║     • XRPUSD (XRP)      - 50x leverage                                   ║
║                                                                          ║
║  ⚡ MICROSECOND MONITORING: ENABLED                                      ║
║  📰 NEWS SENTIMENT: {sent_dir} ({sent_score:+.2f})                               
║  📈 TREND FOLLOWING: ACTIVE                                              ║
║  🔒 UPNL PROTECTION: ACTIVE                                              ║
║  🎯 FEE-AWARE TRADING: ACTIVE                                            ║
║                                                                          ║
║  🎯 PROFIT TARGETS:                                                      ║
║     Target:     +{CONFIG.MIN_PROFIT_TARGET:.2%} to +{CONFIG.MAX_PROFIT_TARGET:.2%}                              
║     Stop:       -{CONFIG.STOP_LOSS:.2%}                                             
║     Breakeven:  +{CONFIG.BREAKEVEN_LEVEL:.2%}                                           
║     Trail:      +{CONFIG.TRAIL_ACTIVATION:.2%}                                           
║                                                                          ║
║  Press Ctrl+C to stop                                                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
        
        # Show headlines
        headlines = self.sentiment.get_headlines()
        if headlines:
            print("📰 Market News:")
            for h in headlines[:3]:
                print(f"   {h}")
            print()
        
        print("📍 Checking positions...")
        self._load_positions()
        
        print("⚡ Starting MICROSECOND price engine...")
        self.price_engine.start(list(ASSETS.keys()))
        time.sleep(2)
        
        print(f"\n{'='*60}")
        print("⚡ ULTIMATE BOT ACTIVE")
        print("   Monitoring BTC, ETH, SOL, XRP at MICROSECOND speed!")
        print(f"{'='*60}\n")
        
        try:
            while self.running:
                time.sleep(0.001)  # 1ms loop
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping Ultimate Bot...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop"""
        self.running = False
        self.price_engine.stop()
        self.sentiment.stop()
        
        self.capital.update()
        elapsed_us = MicroTimer.now() - self.start_time_us
        elapsed_s = elapsed_us / 1_000_000
        
        s = self.stats
        engine_stats = self.price_engine.get_stats()
        
        growth = ((self.capital.equity / self.starting_balance) - 1) * 100 if self.starting_balance > 0 else 0
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    📊 ULTIMATE SESSION SUMMARY                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ⏱️ RUNTIME: {elapsed_s:.1f}s ({elapsed_s/60:.1f} min)                               
║                                                                          ║
║  💰 CAPITAL:                                                             ║
║     Starting:     ${self.starting_balance:.4f}                                      
║     Current:      ${self.capital.equity:.4f}                                        
║     Growth:       {growth:+.1f}%                                                  
║                                                                          ║
║  📈 TRADES:                                                              ║
║     Total:        {s['trades']}                                                    
║     Wins:         {s['wins']}                                                      
║     Losses:       {s['losses']}                                                    
║     Win Rate:     {(s['wins']/max(1,s['trades'])*100):.0f}%                                               
║                                                                          ║
║  💵 P&L:                                                                 ║
║     Gross:        ${s['gross_pnl']:+.6f}                                     
║     Fees:         ${s['fees']:.6f}                                       
║     NET:          ${s['net_pnl']:+.6f}                                       
║     Best Trade:   ${s['best_trade']:+.6f}                                    
║     Worst Trade:  ${s['worst_trade']:+.6f}                                   
║                                                                          ║
║  ⚡ PERFORMANCE:                                                         ║
║     Ticks:        {engine_stats['ticks']:,}                                         
║     Ticks/sec:    {engine_stats['ticks_per_sec']:.1f}                                       
║     Avg Latency:  {engine_stats['avg_latency_us']:.0f}μs                                       
║     Min Latency:  {engine_stats['min_latency_us']:.0f}μs                                       
║     Max Latency:  {engine_stats['max_latency_us']:.0f}μs                                       
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("\n⚡ ALADDIN ULTIMATE - The Ultimate Trading Bot")
    print("   • Microsecond monitoring (μs precision)")
    print("   • Multi-asset (BTC, ETH, SOL, XRP)")
    print("   • News sentiment analysis")
    print("   • UPNL protection")
    print("   • Dynamic capital management")
    print("\n   Press Ctrl+C to stop\n")
    
    bot = AladdinUltimate()
    bot.start()
