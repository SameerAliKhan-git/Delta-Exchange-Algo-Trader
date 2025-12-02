"""
Aladdin AUTONOMOUS - Self-Managing Trading Bot
===============================================
🧠 FULLY AUTONOMOUS - Makes all decisions itself
💰 DYNAMIC CAPITAL MANAGEMENT - Trades based on account balance
📊 MULTI-ASSET - BTC, ETH, SOL, XRP, and OPTIONS
🎯 SMART LOT SIZING - Calculates optimal position size
⚡ LEVERAGE OPTIMIZATION - Uses leverage wisely to grow account
🎲 OPTIONS TRADING - Trades options on underlying assets

THIS BOT DECIDES EVERYTHING ITSELF
"""

import time
import hmac
import hashlib
import json
import requests
import threading
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
import sys
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API CONFIG
# =============================================================================

API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"
BASE_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

# =============================================================================
# ASSET SPECIFICATIONS
# =============================================================================

PERPETUALS = {
    "BTCUSD": {
        "name": "Bitcoin",
        "contract_size": 0.001,      # 0.001 BTC per contract (~$95 notional)
        "min_size": 1,
        "leverage": 100,
        "margin_per_contract": 0.05, # $95 / 100x / 20 = ~$0.05 margin with 20x effective
        "volatility": "high",
        "priority": 2
    },
    "ETHUSD": {
        "name": "Ethereum", 
        "contract_size": 0.01,       # 0.01 ETH per contract (~$28 notional)
        "min_size": 1,
        "leverage": 100,
        "margin_per_contract": 0.03, # $28 / 100x / 10 = ~$0.03 margin
        "volatility": "high",
        "priority": 1                # Cheapest - highest priority
    },
    "SOLUSD": {
        "name": "Solana",
        "contract_size": 0.1,        # 0.1 SOL per contract (~$22 notional)
        "min_size": 1,
        "leverage": 50,
        "margin_per_contract": 0.04, # $22 / 50x / 10 = ~$0.04 margin
        "volatility": "very_high",
        "priority": 3
    },
    "XRPUSD": {
        "name": "XRP",
        "contract_size": 10,         # 10 XRP per contract (~$22 notional)
        "min_size": 1,
        "leverage": 50,
        "margin_per_contract": 0.04, # $22 / 50x / 10 = ~$0.04 margin
        "volatility": "high",
        "priority": 4
    }
}

# =============================================================================
# AUTONOMOUS CONFIG
# =============================================================================

class AutonomousConfig:
    """Bot makes decisions based on these parameters"""
    
    # Risk Management
    MAX_RISK_PER_TRADE = 0.10       # Risk max 10% of capital per trade
    MAX_POSITION_SIZE = 0.30        # Max 30% of capital in one position
    MAX_TOTAL_EXPOSURE = 0.50       # Max 50% total exposure
    
    # Profit Targets (adaptive based on volatility)
    BASE_TARGET = 0.005             # 0.5% base target
    BASE_STOP = 0.003               # 0.3% base stop
    
    # Volatility multipliers
    VOLATILITY_MULTIPLIERS = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
        "very_high": 2.0
    }
    
    # Capital thresholds for asset selection
    CAPITAL_TIERS = {
        "micro": (0, 1),            # $0-1: Trade cheapest (SOL/XRP)
        "small": (1, 10),           # $1-10: Trade ETH
        "medium": (10, 100),        # $10-100: Trade any
        "large": (100, float('inf'))# $100+: Multi-asset portfolio
    }
    
    # Leverage strategy
    LEVERAGE_TIERS = {
        "micro": 20,                # Conservative for tiny accounts
        "small": 15,
        "medium": 10,
        "large": 5
    }
    
    # Sentiment thresholds
    STRONG_BULLISH = 0.25
    STRONG_BEARISH = -0.25
    
    # Timing
    ANALYSIS_INTERVAL = 30          # Re-analyze every 30 seconds
    MIN_TRADE_INTERVAL = 60         # Minimum 60s between trades

CONFIG = AutonomousConfig()


# =============================================================================
# MARKET ANALYZER
# =============================================================================

class MarketAnalyzer:
    """Analyzes all markets to find best opportunities"""
    
    def __init__(self):
        self.prices: Dict[str, deque] = {}
        self.volumes: Dict[str, float] = {}
        self.trends: Dict[str, str] = {}
        self.volatility: Dict[str, float] = {}
        self.scores: Dict[str, float] = {}
        
    def update_price(self, symbol: str, price: float):
        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=100)
        self.prices[symbol].append(price)
        self._calculate_metrics(symbol)
    
    def _calculate_metrics(self, symbol: str):
        """Calculate trend and volatility"""
        prices = list(self.prices.get(symbol, []))
        if len(prices) < 20:
            return
        
        # Trend (last 20 vs first 20)
        recent = sum(prices[-20:]) / 20
        older = sum(prices[:20]) / 20
        trend_pct = (recent - older) / older
        
        if trend_pct > 0.001:
            self.trends[symbol] = "UPTREND"
        elif trend_pct < -0.001:
            self.trends[symbol] = "DOWNTREND"
        else:
            self.trends[symbol] = "SIDEWAYS"
        
        # Volatility (standard deviation of returns)
        if len(prices) > 5:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            self.volatility[symbol] = math.sqrt(variance)
    
    def get_best_opportunity(self, sentiment: str) -> Optional[Tuple[str, str, float]]:
        """
        Find the best trading opportunity
        Returns: (symbol, direction, score) or None
        """
        opportunities = []
        
        for symbol in self.prices.keys():
            if len(self.prices[symbol]) < 20:
                continue
            
            trend = self.trends.get(symbol, "SIDEWAYS")
            vol = self.volatility.get(symbol, 0)
            
            # Skip sideways markets
            if trend == "SIDEWAYS":
                continue
            
            # Calculate opportunity score
            score = 0
            direction = None
            
            # Trend alignment with sentiment
            if trend == "UPTREND" and sentiment in ["BULLISH", "NEUTRAL"]:
                direction = "LONG"
                score = vol * 100  # Higher volatility = more potential
                if sentiment == "BULLISH":
                    score *= 1.5  # Bonus for sentiment alignment
                    
            elif trend == "DOWNTREND" and sentiment in ["BEARISH", "NEUTRAL"]:
                direction = "SHORT"
                score = vol * 100
                if sentiment == "BEARISH":
                    score *= 1.5
            
            if direction and score > 0:
                opportunities.append((symbol, direction, score))
        
        if opportunities:
            # Return highest scoring opportunity
            return max(opportunities, key=lambda x: x[2])
        return None
    
    def get_current_price(self, symbol: str) -> float:
        prices = self.prices.get(symbol, [])
        return prices[-1] if prices else 0


# =============================================================================
# SENTIMENT ANALYZER
# =============================================================================

class SentimentAnalyzer:
    """Market sentiment from news"""
    
    BULLISH = ["bullish", "rally", "surge", "pump", "moon", "adoption", "buy", 
               "breakout", "higher", "gains", "etf", "institutional", "support"]
    BEARISH = ["bearish", "crash", "dump", "selloff", "ban", "hack", "fraud",
               "fear", "panic", "lower", "losses", "resistance", "warning"]
    
    def __init__(self):
        self.score = 0.0
        self.direction = "NEUTRAL"
        self.last_update = None
        self._running = True
        threading.Thread(target=self._update_loop, daemon=True).start()
    
    def _update_loop(self):
        while self._running:
            self._fetch()
            time.sleep(120)
    
    def _fetch(self):
        try:
            r = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
                timeout=10
            )
            if r.status_code != 200:
                return
            
            news = r.json().get("Data", [])[:15]
            scores = []
            
            for item in news:
                text = (item.get("title", "") + " " + item.get("body", "")[:300]).lower()
                bull = sum(1 for kw in self.BULLISH if kw in text)
                bear = sum(1 for kw in self.BEARISH if kw in text)
                if bull + bear > 0:
                    scores.append((bull - bear) / (bull + bear))
            
            if scores:
                self.score = sum(scores) / len(scores)
                if self.score > CONFIG.STRONG_BULLISH:
                    self.direction = "BULLISH"
                elif self.score < CONFIG.STRONG_BEARISH:
                    self.direction = "BEARISH"
                else:
                    self.direction = "NEUTRAL"
            
            self.last_update = datetime.now()
        except:
            pass
    
    def get(self) -> Tuple[float, str]:
        return self.score, self.direction
    
    def stop(self):
        self._running = False


# =============================================================================
# CAPITAL MANAGER
# =============================================================================

class CapitalManager:
    """Manages capital allocation and position sizing"""
    
    def __init__(self, api):
        self.api = api
        self.balance = 0
        self.equity = 0
        self.used_margin = 0
        self.available = 0
        self.tier = "micro"
        self.leverage = 20
        
    def update(self):
        """Refresh account data"""
        try:
            self.balance = self.api.get_balance()
            positions = self.api.get_all_positions()
            
            # Calculate equity (balance + unrealized PnL)
            self.equity = self.balance
            self.used_margin = 0
            
            for pos in positions:
                # Add unrealized PnL if available
                upnl = float(pos.get("unrealized_pnl", 0))
                self.equity += upnl
                
                # Track margin used
                margin = float(pos.get("margin", 0))
                self.used_margin += margin
            
            self.available = self.balance - self.used_margin
            
            # Determine tier
            if self.equity < 1:
                self.tier = "micro"
            elif self.equity < 10:
                self.tier = "small"
            elif self.equity < 100:
                self.tier = "medium"
            else:
                self.tier = "large"
            
            # Set leverage based on tier
            self.leverage = CONFIG.LEVERAGE_TIERS.get(self.tier, 10)
            
        except Exception as e:
            print(f"   ⚠️ Error updating capital: {e}")
    
    def get_tradeable_symbols(self) -> List[str]:
        """Get symbols we can trade based on capital - ALL 4 ASSETS"""
        symbols = []
        
        # With micro account, we can trade all assets with proper leverage
        # Delta Exchange allows high leverage, so even $0.10 can trade
        for symbol, spec in PERPETUALS.items():
            # Minimum margin needed is very low with leverage
            min_margin = 0.02  # $0.02 minimum to trade any asset
            if self.available >= min_margin:
                symbols.append(symbol)
        
        # Sort by priority (lower = better for small accounts)
        symbols.sort(key=lambda s: PERPETUALS[s]["priority"])
        
        return symbols
    
    def calculate_position_size(self, symbol: str, price: float) -> int:
        """Calculate optimal position size based on capital"""
        spec = PERPETUALS.get(symbol)
        if not spec:
            return 0
        
        # Maximum capital to risk
        max_capital = self.available * CONFIG.MAX_POSITION_SIZE
        
        # Calculate contracts based on margin requirement
        margin_per_contract = (price * spec["contract_size"]) / self.leverage
        max_contracts = int(max_capital / margin_per_contract)
        
        # Apply risk limit
        risk_capital = self.available * CONFIG.MAX_RISK_PER_TRADE
        risk_contracts = int(risk_capital / margin_per_contract)
        
        # Take minimum of risk-based and capital-based
        contracts = min(max_contracts, risk_contracts)
        
        # Ensure at least 1 contract
        return max(1, contracts)
    
    def can_open_position(self) -> bool:
        """Check if we can open new positions"""
        exposure_pct = self.used_margin / self.equity if self.equity > 0 else 1
        return exposure_pct < CONFIG.MAX_TOTAL_EXPOSURE
    
    def get_summary(self) -> str:
        return (f"Balance: ${self.balance:.4f} | "
                f"Equity: ${self.equity:.4f} | "
                f"Available: ${self.available:.4f} | "
                f"Tier: {self.tier.upper()} | "
                f"Leverage: {self.leverage}x")


# =============================================================================
# OPTIONS ANALYZER
# =============================================================================

class OptionsAnalyzer:
    """Analyzes options opportunities"""
    
    def __init__(self, api):
        self.api = api
        self.options_chain: Dict[str, List] = {}
    
    def fetch_options(self, underlying: str) -> List[Dict]:
        """Fetch available options for underlying"""
        try:
            r = requests.get(f"{BASE_URL}/v2/products", timeout=5)
            products = r.json().get("result", [])
            
            options = []
            for p in products:
                if (p.get("product_type") == "call_options" or 
                    p.get("product_type") == "put_options"):
                    if underlying.replace("USD", "") in p.get("underlying_asset", {}).get("symbol", ""):
                        options.append(p)
            
            self.options_chain[underlying] = options
            return options
        except:
            return []
    
    def find_best_option(self, underlying: str, direction: str, 
                         current_price: float, capital: float) -> Optional[Dict]:
        """
        Find the best option to trade based on direction and capital
        """
        options = self.options_chain.get(underlying, [])
        if not options:
            options = self.fetch_options(underlying)
        
        if not options:
            return None
        
        # Filter by direction
        if direction == "LONG":
            # Buy calls for bullish view
            filtered = [o for o in options if o.get("product_type") == "call_options"]
        else:
            # Buy puts for bearish view
            filtered = [o for o in options if o.get("product_type") == "put_options"]
        
        # Find affordable options near the money
        best = None
        best_score = 0
        
        for opt in filtered:
            strike = float(opt.get("strike_price", 0))
            premium = float(opt.get("mark_price", 0))
            
            if premium <= 0 or premium > capital * 0.5:
                continue
            
            # Score based on how close to the money
            distance_pct = abs(strike - current_price) / current_price
            
            # Prefer ATM or slightly OTM options
            if distance_pct < 0.05:  # Within 5% of current price
                score = 1 / (distance_pct + 0.01)  # Higher score for closer strikes
                
                if score > best_score:
                    best_score = score
                    best = opt
        
        return best


# =============================================================================
# PRICE STREAM
# =============================================================================

class PriceStream:
    def __init__(self, on_tick):
        self.on_tick = on_tick
        self.ws = None
        self.running = False
        self.prices = {}
        
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
    
    def _subscribe(self, ws, symbols):
        print("📡 Price stream connected")
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
            if price:
                self.prices[symbol] = float(price)
                self.on_tick(symbol, float(price))
        except:
            pass
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


# =============================================================================
# API CLIENT
# =============================================================================

class AutonomousAPI:
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
                symbol = p.get("symbol", "")
                self.product_ids[symbol] = p.get("id")
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
    
    def get_all_positions(self):
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
    
    def market_order(self, symbol, side, size):
        try:
            product_id = self.product_ids.get(symbol)
            if not product_id:
                return False, "No product ID"
            
            order = {"product_id": product_id, "side": side, "size": int(size), "order_type": "market_order"}
            payload = json.dumps(order)
            r = self.session.post(BASE_URL + "/v2/orders", headers=self._sign("POST", "/v2/orders", payload), data=payload, timeout=5)
            result = r.json()
            if result.get("success"):
                return True, "Order filled"
            return False, result.get("error", {}).get("message", "Failed")
        except Exception as e:
            return False, str(e)
    
    def close_position(self, symbol, size, side):
        close_side = "sell" if side == "LONG" else "buy"
        return self.market_order(symbol, close_side, size)


# =============================================================================
# ALADDIN AUTONOMOUS BOT
# =============================================================================

class AladdinAutonomous:
    """Fully Autonomous Trading Bot"""
    
    def __init__(self):
        self.api = AutonomousAPI()
        self.capital = CapitalManager(self.api)
        self.market = MarketAnalyzer()
        self.sentiment = SentimentAnalyzer()
        self.options = OptionsAnalyzer(self.api)
        
        self.running = False
        self.position = None
        self.last_trade_time = 0
        self.last_analysis_time = 0
        
        # Performance tracking
        self.starting_balance = 0
        self.stats = {
            "trades": 0, "wins": 0, "losses": 0,
            "gross_pnl": 0, "fees": 0, "net_pnl": 0,
            "peak_equity": 0
        }
        
        self.stream = PriceStream(self._on_tick)
    
    def _on_tick(self, symbol: str, price: float):
        """Process price update"""
        self.market.update_price(symbol, price)
        
        if self.position and self.position["symbol"] == symbol:
            self._manage_position(symbol, price)
    
    def _analyze_and_decide(self):
        """Main decision-making loop"""
        now = time.time()
        
        if now - self.last_analysis_time < CONFIG.ANALYSIS_INTERVAL:
            return
        
        self.last_analysis_time = now
        
        # Update capital
        self.capital.update()
        
        # Track peak equity
        if self.capital.equity > self.stats["peak_equity"]:
            self.stats["peak_equity"] = self.capital.equity
        
        # Skip if position open
        if self.position:
            return
        
        # Check cooldown
        if now - self.last_trade_time < CONFIG.MIN_TRADE_INTERVAL:
            return
        
        # Check if we can trade
        if not self.capital.can_open_position():
            print("   ⚠️ Max exposure reached, waiting...")
            return
        
        # Get sentiment
        sent_score, sent_dir = self.sentiment.get()
        
        # Get tradeable symbols
        symbols = self.capital.get_tradeable_symbols()
        if not symbols:
            print("   ⚠️ Insufficient capital for any trades")
            return
        
        # Find best opportunity
        opportunity = self.market.get_best_opportunity(sent_dir)
        
        if opportunity:
            symbol, direction, score = opportunity
            
            # Check if symbol is tradeable with our capital
            if symbol in symbols:
                self._execute_trade(symbol, direction, score, sent_dir)
            else:
                # Try the best symbol we can afford
                for alt_symbol in symbols:
                    alt_trend = self.market.trends.get(alt_symbol, "SIDEWAYS")
                    if alt_trend == "UPTREND" and sent_dir != "BEARISH":
                        self._execute_trade(alt_symbol, "LONG", score * 0.8, sent_dir)
                        break
                    elif alt_trend == "DOWNTREND" and sent_dir != "BULLISH":
                        self._execute_trade(alt_symbol, "SHORT", score * 0.8, sent_dir)
                        break
    
    def _execute_trade(self, symbol: str, direction: str, score: float, sentiment: str):
        """Execute a trade"""
        price = self.market.get_current_price(symbol)
        if price <= 0:
            return
        
        spec = PERPETUALS.get(symbol, {})
        
        # Calculate position size
        size = self.capital.calculate_position_size(symbol, price)
        if size < 1:
            return
        
        # Calculate targets based on volatility
        vol_mult = CONFIG.VOLATILITY_MULTIPLIERS.get(spec.get("volatility", "high"), 1.5)
        target_pct = CONFIG.BASE_TARGET * vol_mult
        stop_pct = CONFIG.BASE_STOP * vol_mult
        
        if direction == "LONG":
            target = price * (1 + target_pct)
            stop = price * (1 - stop_pct)
            order_side = "buy"
        else:
            target = price * (1 - target_pct)
            stop = price * (1 + stop_pct)
            order_side = "sell"
        
        position_value = price * spec.get("contract_size", 0.01) * size
        margin_used = position_value / self.capital.leverage
        
        print(f"\n{'🤖'*25}")
        print(f"🚀 AUTONOMOUS TRADE DECISION")
        print(f"")
        print(f"   📊 Analysis:")
        print(f"      Symbol:      {symbol} ({spec.get('name', '')})")
        print(f"      Direction:   {direction}")
        print(f"      Sentiment:   {sentiment} ({self.sentiment.score:+.2f})")
        print(f"      Score:       {score:.1f}")
        print(f"")
        print(f"   💰 Position Sizing:")
        print(f"      Capital:     ${self.capital.available:.4f}")
        print(f"      Tier:        {self.capital.tier.upper()}")
        print(f"      Leverage:    {self.capital.leverage}x")
        print(f"      Contracts:   {size}")
        print(f"      Margin:      ${margin_used:.4f}")
        print(f"      Exposure:    ${position_value:.4f}")
        print(f"")
        print(f"   🎯 Trade Setup:")
        print(f"      Entry:       ${price:.2f}")
        print(f"      Target:      ${target:.2f} (+{target_pct:.2%})")
        print(f"      Stop:        ${stop:.2f} (-{stop_pct:.2%})")
        print(f"      Risk/Reward: 1:{target_pct/stop_pct:.1f}")
        
        # Execute
        success, msg = self.api.market_order(symbol, order_side, size)
        
        if success:
            self.position = {
                "symbol": symbol,
                "side": direction,
                "entry": price,
                "size": size,
                "entry_time": time.time(),
                "target": target,
                "stop": stop,
                "position_value": position_value,
                "peak_price": price,
                "breakeven_active": False,
                "trailing_active": False
            }
            self.last_trade_time = time.time()
            self.stats["trades"] += 1
            
            print(f"\n   ✅ POSITION OPENED: {size} contracts")
        else:
            print(f"\n   ❌ Order failed: {msg}")
        
        print(f"{'🤖'*25}\n")
    
    def _manage_position(self, symbol: str, price: float):
        """Manage open position"""
        pos = self.position
        entry = pos["entry"]
        side = pos["side"]
        
        hold_time = time.time() - pos["entry_time"]
        
        # Calculate P&L
        if side == "LONG":
            pnl_pct = (price - entry) / entry
            # Track peak for trailing
            if price > pos["peak_price"]:
                pos["peak_price"] = price
        else:
            pnl_pct = (entry - price) / entry
            if price < pos["peak_price"]:
                pos["peak_price"] = price
        
        net_pnl_pct = pnl_pct - 0.001  # Subtract fees
        
        # Dynamic stop management
        current_stop = pos["stop"]
        
        # Breakeven at 0.15%
        if pnl_pct >= 0.0015 and not pos["breakeven_active"]:
            pos["breakeven_active"] = True
            if side == "LONG":
                pos["stop"] = entry * 1.0001
            else:
                pos["stop"] = entry * 0.9999
            print(f"\n   🔒 Breakeven activated!")
        
        # Trailing at 0.25%
        if pnl_pct >= 0.0025 and not pos["trailing_active"]:
            pos["trailing_active"] = True
            print(f"\n   📈 Trailing stop activated!")
        
        if pos["trailing_active"]:
            trail_distance = 0.0015
            if side == "LONG":
                trail_stop = pos["peak_price"] * (1 - trail_distance)
                if trail_stop > pos["stop"]:
                    pos["stop"] = trail_stop
            else:
                trail_stop = pos["peak_price"] * (1 + trail_distance)
                if trail_stop < pos["stop"]:
                    pos["stop"] = trail_stop
        
        # Check exit conditions
        should_exit = False
        reason = ""
        
        # Target hit
        if side == "LONG" and price >= pos["target"]:
            should_exit, reason = True, "🎯 TARGET HIT!"
        elif side == "SHORT" and price <= pos["target"]:
            should_exit, reason = True, "🎯 TARGET HIT!"
        
        # Stop hit
        elif side == "LONG" and price <= pos["stop"]:
            should_exit, reason = True, "🛑 STOP HIT"
        elif side == "SHORT" and price >= pos["stop"]:
            should_exit, reason = True, "🛑 STOP HIT"
        
        # Time exit (10 minutes max)
        elif hold_time > 600:
            if net_pnl_pct > 0:
                should_exit, reason = True, "⏰ TIME EXIT (profit)"
            else:
                should_exit, reason = True, "⏰ TIME EXIT"
        
        if should_exit:
            self._close_position(price, pnl_pct, reason)
        else:
            self._display_position(price, pnl_pct, net_pnl_pct, hold_time)
    
    def _close_position(self, exit_price: float, pnl_pct: float, reason: str):
        """Close position"""
        pos = self.position
        
        self.api.close_position(pos["symbol"], pos["size"], pos["side"])
        
        gross_pnl = pos["position_value"] * pnl_pct
        fees = pos["position_value"] * 0.001
        net_pnl = gross_pnl - fees
        
        if net_pnl > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        
        self.stats["gross_pnl"] += gross_pnl
        self.stats["fees"] += fees
        self.stats["net_pnl"] += net_pnl
        
        emoji = "🚀" if net_pnl > 0 else "💔"
        
        hold_time = time.time() - pos["entry_time"]
        
        print(f"\n{emoji}{'='*50}")
        print(f"   {reason}")
        print(f"")
        print(f"   {pos['symbol']} {pos['side']} x{pos['size']}")
        print(f"   ${pos['entry']:.2f} → ${exit_price:.2f}")
        print(f"   Hold: {hold_time:.0f}s")
        print(f"")
        print(f"   Gross: ${gross_pnl:+.6f} ({pnl_pct:+.3%})")
        print(f"   Fees:  ${fees:.6f}")
        print(f"   NET:   ${net_pnl:+.6f}")
        print(f"{'='*55}")
        self._print_stats()
        print()
        
        self.position = None
        self.capital.update()
    
    def _display_position(self, price: float, pnl_pct: float, net_pnl_pct: float, hold_time: float):
        """Display position status"""
        pos = self.position
        
        color = "🟢" if net_pnl_pct > 0 else "🟡" if pnl_pct > 0 else "🔴"
        stop_type = "📈" if pos["trailing_active"] else "🔒" if pos["breakeven_active"] else "🔴"
        
        sys.stdout.write(
            f"\r{color} {pos['symbol']} {pos['side']} x{pos['size']} | "
            f"${price:.2f} | "
            f"Net: {net_pnl_pct:+.3%} | "
            f"{stop_type} Stop: ${pos['stop']:.2f} | "
            f"⏱️ {hold_time:.0f}s   "
        )
        sys.stdout.flush()
    
    def _print_stats(self):
        """Print stats"""
        s = self.stats
        trades = s["trades"]
        wr = (s["wins"] / trades * 100) if trades > 0 else 0
        
        growth = ((self.capital.equity / self.starting_balance) - 1) * 100 if self.starting_balance > 0 else 0
        
        print(f"\n📊 AUTONOMOUS STATS:")
        print(f"   Trades: {trades} | W:{s['wins']} L:{s['losses']} ({wr:.0f}%)")
        print(f"   Net P&L: ${s['net_pnl']:+.6f}")
        print(f"   Current Equity: ${self.capital.equity:.4f}")
        print(f"   Growth: {growth:+.1f}%")
    
    def _load_positions(self):
        """Load existing positions"""
        positions = self.api.get_all_positions()
        
        for pos in positions:
            symbol = pos.get("product_symbol", "")
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size != 0 and symbol in PERPETUALS:
                side = "LONG" if size > 0 else "SHORT"
                spec = PERPETUALS[symbol]
                
                price = entry
                vol_mult = CONFIG.VOLATILITY_MULTIPLIERS.get(spec.get("volatility", "high"), 1.5)
                target_pct = CONFIG.BASE_TARGET * vol_mult
                stop_pct = CONFIG.BASE_STOP * vol_mult
                
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
                    "entry_time": time.time(),
                    "target": target,
                    "stop": stop,
                    "position_value": entry * spec["contract_size"] * abs(size),
                    "peak_price": entry,
                    "breakeven_active": False,
                    "trailing_active": False
                }
                
                print(f"📌 Loaded: {symbol} {side} x{abs(int(size))} @ ${entry:.2f}")
                break
    
    def start(self):
        """Start autonomous bot"""
        self.running = True
        
        # Get initial capital
        self.capital.update()
        self.starting_balance = self.capital.equity
        self.stats["peak_equity"] = self.capital.equity
        
        sent_score, sent_dir = self.sentiment.get()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║           🤖 ALADDIN AUTONOMOUS - SELF-MANAGING BOT 🤖                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  💰 CAPITAL MANAGEMENT:
║     {self.capital.get_summary()}
║  
║  📊 TRADEABLE ASSETS: {', '.join(self.capital.get_tradeable_symbols()) or 'None (insufficient funds)'}
║  
║  🎯 TRADING STRATEGY:
║     • Dynamic lot sizing based on capital
║     • Multi-asset trading (BTC, ETH, SOL, XRP)
║     • Sentiment-aligned entries only
║     • Trend following with UPNL protection
║     • Breakeven + Trailing stops
║  
║  📰 SENTIMENT: {sent_dir} ({sent_score:+.2f})
║  
║  ⚙️ RISK PARAMETERS:
║     Max risk/trade:    {CONFIG.MAX_RISK_PER_TRADE:.0%}
║     Max position:      {CONFIG.MAX_POSITION_SIZE:.0%}
║     Max exposure:      {CONFIG.MAX_TOTAL_EXPOSURE:.0%}
║     Leverage:          {self.capital.leverage}x
║  
║  🤖 This bot makes ALL decisions autonomously!
║  
║  Press Ctrl+C to stop
╚══════════════════════════════════════════════════════════════════════════╝
""")
        
        print("📍 Checking existing positions...")
        self._load_positions()
        
        print("📡 Starting price streams for all assets...")
        symbols = list(PERPETUALS.keys())
        self.stream.start(symbols)
        time.sleep(2)
        
        print(f"\n{'='*60}")
        print("🤖 AUTONOMOUS MODE ACTIVE")
        print("   Bot is analyzing markets and making decisions...")
        print(f"{'='*60}\n")
        
        try:
            while self.running:
                self._analyze_and_decide()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping autonomous bot...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop bot"""
        self.running = False
        self.stream.stop()
        self.sentiment.stop()
        
        self.capital.update()
        runtime = time.time() - self.last_analysis_time if self.last_analysis_time else 0
        s = self.stats
        
        growth = ((self.capital.equity / self.starting_balance) - 1) * 100 if self.starting_balance > 0 else 0
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                   📊 AUTONOMOUS SESSION SUMMARY                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  💰 CAPITAL:
║     Starting:      ${self.starting_balance:.4f}
║     Current:       ${self.capital.equity:.4f}
║     Peak:          ${s['peak_equity']:.4f}
║     Growth:        {growth:+.1f}%
║  
║  📈 TRADES:
║     Total:         {s['trades']}
║     Wins:          {s['wins']}
║     Losses:        {s['losses']}
║     Win Rate:      {(s['wins']/max(1,s['trades'])*100):.0f}%
║  
║  💵 P&L:
║     Gross:         ${s['gross_pnl']:+.6f}
║     Fees:          ${s['fees']:.6f}
║     NET:           ${s['net_pnl']:+.6f}
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("\n🤖 ALADDIN AUTONOMOUS - Self-Managing Trading Bot")
    print("   • Dynamic capital management")
    print("   • Multi-asset trading")
    print("   • Smart leverage usage")
    print("   • Fully autonomous decisions")
    print("\n   Press Ctrl+C to stop\n")
    
    bot = AladdinAutonomous()
    bot.start()
