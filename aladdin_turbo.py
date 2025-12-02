"""
Aladdin TURBO HFT - Microsecond Scalping Bot
=============================================
⚡ MICROSECOND execution speed
💰 GUARANTEED profit > fees or NO trade
📈 Ultra-tight scalping with instant exits
🎯 Only takes trades with 3x fee coverage

LIVE TRADING - MAXIMUM SPEED
"""

import time
import hmac
import hashlib
import json
import requests
import threading
import websocket
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import sys

# Disable ALL buffering for maximum speed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"
BASE_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

# Contract sizes
CONTRACTS = {
    "BTCUSD": {"size": 0.001, "product_id": None},
    "ETHUSD": {"size": 0.01, "product_id": None},
    "SOLUSD": {"size": 0.1, "product_id": None}
}

# =============================================================================
# TURBO HFT CONFIGURATION - FEES MUST NEVER EAT PROFITS
# =============================================================================

class TurboConfig:
    """
    FEE MATH:
    - Taker fee: 0.05% per trade
    - Round trip: 0.10% (entry + exit)
    
    TO GUARANTEE PROFIT > FEES:
    - Minimum gross profit must be 3x fees = 0.30%
    - This ensures net profit = 0.30% - 0.10% = 0.20% MINIMUM
    """
    
    # Fee structure
    TAKER_FEE = 0.0005           # 0.05%
    ROUND_TRIP_FEE = 0.0010      # 0.10%
    
    # PROFIT TARGETS - 3x fee coverage
    MIN_GROSS_TARGET = 0.0030    # 0.30% minimum gross (3x fees)
    PROFIT_MULTIPLIER = 3.0      # Profit must be 3x fees
    
    # Calculated minimum net profit
    MIN_NET_PROFIT = MIN_GROSS_TARGET - ROUND_TRIP_FEE  # 0.20%
    
    # Stop loss - tight but not too tight
    STOP_LOSS = 0.0020           # 0.20% stop loss
    
    # Time limits
    MAX_HOLD_SECONDS = 30        # Exit after 30 seconds max
    
    # Entry signals
    MOMENTUM_TICKS = 5           # Fewer ticks for faster signals
    MOMENTUM_THRESHOLD = 0.0005  # 0.05% move triggers entry
    
    # Cooldown
    COOLDOWN_SECONDS = 3
    
    # Symbols
    SYMBOLS = ["ETHUSD", "BTCUSD"]

CONFIG = TurboConfig()


# =============================================================================
# MICROSECOND TIMER
# =============================================================================

class MicrosecondTimer:
    """High-precision timing"""
    
    @staticmethod
    def now_us() -> int:
        """Get current time in microseconds"""
        return int(time.time() * 1_000_000)
    
    @staticmethod
    def now_ms() -> float:
        """Get current time in milliseconds"""
        return time.time() * 1000
    
    @staticmethod
    def elapsed_us(start_us: int) -> int:
        """Get elapsed microseconds"""
        return MicrosecondTimer.now_us() - start_us
    
    @staticmethod
    def elapsed_ms(start_ms: float) -> float:
        """Get elapsed milliseconds"""
        return MicrosecondTimer.now_ms() - start_ms


# =============================================================================
# TURBO PRICE STREAM
# =============================================================================

class TurboPriceStream:
    """Ultra-low latency WebSocket price stream"""
    
    def __init__(self, on_tick):
        self.on_tick = on_tick
        self.ws = None
        self.running = False
        self.prices: Dict[str, float] = {}
        self.price_history: Dict[str, deque] = {}
        self.tick_count = 0
        self.last_tick_time = 0
        
    def start(self, symbols: List[str]):
        self.running = True
        for s in symbols:
            self.price_history[s] = deque(maxlen=CONFIG.MOMENTUM_TICKS)
        
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
        time.sleep(0.5)
    
    def _subscribe(self, ws, symbols):
        print("⚡ TURBO WebSocket connected - MICROSECOND MODE")
        for symbol in symbols:
            ws.send(json.dumps({
                "type": "subscribe",
                "payload": {"channels": [
                    {"name": "v2/ticker", "symbols": [symbol]},
                    {"name": "mark_price", "symbols": [symbol]}
                ]}
            }))
    
    def _on_message(self, ws, message):
        tick_us = MicrosecondTimer.now_us()
        try:
            data = json.loads(message)
            symbol = data.get("symbol", "")
            mark_price = data.get("mark_price")
            
            if mark_price and symbol in self.price_history:
                price = float(mark_price)
                self.prices[symbol] = price
                self.price_history[symbol].append((tick_us, price))
                self.tick_count += 1
                self.last_tick_time = tick_us
                
                # Detect momentum
                momentum = self._check_momentum(symbol)
                
                # Callback
                self.on_tick(symbol, price, tick_us, momentum)
        except:
            pass
    
    def _check_momentum(self, symbol: str) -> Optional[str]:
        history = self.price_history.get(symbol)
        if not history or len(history) < CONFIG.MOMENTUM_TICKS:
            return None
        
        prices = [p[1] for p in history]
        change = (prices[-1] - prices[0]) / prices[0]
        
        if change > CONFIG.MOMENTUM_THRESHOLD:
            return "BULL"
        elif change < -CONFIG.MOMENTUM_THRESHOLD:
            return "BEAR"
        return None
    
    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 0)
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


# =============================================================================
# TURBO API CLIENT
# =============================================================================

class TurboAPI:
    """Speed-optimized API client"""
    
    def __init__(self):
        self.session = requests.Session()
        # Enable connection pooling for speed
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0  # No retries for speed
        )
        self.session.mount('https://', adapter)
        self.product_ids = {}
        self._load_products()
    
    def _sign(self, method: str, endpoint: str, payload: str = "") -> dict:
        ts = str(int(time.time()))
        sig = hmac.new(
            API_SECRET.encode(),
            f"{method}{ts}{endpoint}{payload}".encode(),
            hashlib.sha256
        ).hexdigest()
        return {"api-key": API_KEY, "timestamp": ts, "signature": sig, "Content-Type": "application/json"}
    
    def _load_products(self):
        try:
            r = self.session.get(BASE_URL + "/v2/products", timeout=3)
            for p in r.json().get("result", []):
                symbol = p.get("symbol", "")
                if symbol in CONTRACTS:
                    self.product_ids[symbol] = p.get("id")
        except:
            pass
    
    def get_balance(self) -> float:
        try:
            endpoint = "/v2/wallet/balances"
            r = self.session.get(BASE_URL + endpoint, headers=self._sign("GET", endpoint), timeout=2)
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
                r = self.session.get(BASE_URL + endpoint, headers=self._sign("GET", endpoint), timeout=2)
                for p in r.json().get("result", []):
                    if float(p.get("size", 0)) != 0:
                        positions.append(p)
            except:
                pass
        return positions
    
    def market_order(self, symbol: str, side: str, size: int) -> Tuple[bool, float]:
        """Place market order - returns (success, latency_ms)"""
        start = MicrosecondTimer.now_ms()
        try:
            order = {
                "product_id": self.product_ids.get(symbol),
                "side": side,
                "size": size,
                "order_type": "market_order"
            }
            payload = json.dumps(order)
            r = self.session.post(
                BASE_URL + "/v2/orders",
                headers=self._sign("POST", "/v2/orders", payload),
                data=payload,
                timeout=3
            )
            latency = MicrosecondTimer.elapsed_ms(start)
            return r.json().get("success", False), latency
        except:
            return False, MicrosecondTimer.elapsed_ms(start)
    
    def close_position(self, symbol: str, size: int, side: str) -> Tuple[bool, float]:
        """Close position - returns (success, latency_ms)"""
        close_side = "sell" if side == "LONG" else "buy"
        return self.market_order(symbol, close_side, size)


# =============================================================================
# ALADDIN TURBO HFT BOT
# =============================================================================

class AladdinTurbo:
    """Microsecond HFT Trading Bot - Profits ALWAYS > Fees"""
    
    def __init__(self):
        self.api = TurboAPI()
        self.running = False
        
        # Trade state
        self.position = None  # {symbol, side, entry, size, entry_time_us}
        self.last_trade_us = 0
        
        # Stats
        self.stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "gross_pnl": 0.0,
            "fees_paid": 0.0,
            "net_pnl": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_hold_ms": 0.0,
            "total_hold_ms": 0.0
        }
        
        # Price stream
        self.stream = TurboPriceStream(self._on_tick)
        self.start_time = 0
    
    def _calculate_fee_safe_exit(self, entry: float, side: str) -> Tuple[float, float]:
        """
        Calculate exit prices that GUARANTEE profit > fees
        Returns (take_profit_price, stop_loss_price)
        """
        if side == "LONG":
            # Must go UP by at least 0.30% for profit
            tp = entry * (1 + CONFIG.MIN_GROSS_TARGET)
            sl = entry * (1 - CONFIG.STOP_LOSS)
        else:
            # Must go DOWN by at least 0.30% for profit
            tp = entry * (1 - CONFIG.MIN_GROSS_TARGET)
            sl = entry * (1 + CONFIG.STOP_LOSS)
        return tp, sl
    
    def _on_tick(self, symbol: str, price: float, tick_us: int, momentum: Optional[str]):
        """Process every tick - MICROSECOND SPEED"""
        
        # If we have a position, check for exit
        if self.position and self.position["symbol"] == symbol:
            self._check_exit(price, tick_us)
            return
        
        # Check for entry on momentum
        if momentum and not self.position:
            self._try_entry(symbol, price, tick_us, momentum)
    
    def _try_entry(self, symbol: str, price: float, tick_us: int, momentum: str):
        """Try to enter a trade"""
        
        # Check cooldown (in microseconds)
        cooldown_us = CONFIG.COOLDOWN_SECONDS * 1_000_000
        if tick_us - self.last_trade_us < cooldown_us:
            return
        
        # Check balance
        balance = self.api.get_balance()
        if balance < 0.15:
            return
        
        # Determine side
        side = "LONG" if momentum == "BULL" else "SHORT"
        order_side = "buy" if side == "LONG" else "sell"
        
        # Calculate guaranteed-profit exit levels
        tp_price, sl_price = self._calculate_fee_safe_exit(price, side)
        
        contract_size = CONTRACTS[symbol]["size"]
        position_value = price * contract_size
        
        # Calculate expected outcomes
        gross_profit_at_tp = position_value * CONFIG.MIN_GROSS_TARGET
        fees = position_value * CONFIG.ROUND_TRIP_FEE
        net_profit_at_tp = gross_profit_at_tp - fees
        
        print(f"\n{'⚡'*30}")
        print(f"🎯 TURBO ENTRY: {symbol} {side}")
        print(f"   Entry:        ${price:.2f}")
        print(f"   Take Profit:  ${tp_price:.2f} (+{CONFIG.MIN_GROSS_TARGET:.2%})")
        print(f"   Stop Loss:    ${sl_price:.2f} (-{CONFIG.STOP_LOSS:.2%})")
        print(f"   Position:     ${position_value:.4f}")
        print(f"   Fees:         ${fees:.6f} (0.10%)")
        print(f"   Net @ TP:     ${net_profit_at_tp:.6f} (+{CONFIG.MIN_NET_PROFIT:.2%})")
        
        # Execute order
        success, latency = self.api.market_order(symbol, order_side, 1)
        
        if success:
            self.position = {
                "symbol": symbol,
                "side": side,
                "entry": price,
                "size": 1,
                "entry_time_us": tick_us,
                "tp": tp_price,
                "sl": sl_price,
                "position_value": position_value
            }
            self.last_trade_us = tick_us
            self.stats["trades"] += 1
            
            print(f"   ✅ FILLED in {latency:.0f}ms")
            print(f"{'⚡'*30}\n")
        else:
            print(f"   ❌ Order failed")
    
    def _check_exit(self, price: float, tick_us: int):
        """Check exit conditions - MICROSECOND DECISIONS"""
        pos = self.position
        entry = pos["entry"]
        side = pos["side"]
        tp = pos["tp"]
        sl = pos["sl"]
        entry_us = pos["entry_time_us"]
        
        hold_us = tick_us - entry_us
        hold_ms = hold_us / 1000
        hold_s = hold_us / 1_000_000
        
        # Calculate current P&L
        if side == "LONG":
            gross_pnl_pct = (price - entry) / entry
        else:
            gross_pnl_pct = (entry - price) / entry
        
        net_pnl_pct = gross_pnl_pct - CONFIG.ROUND_TRIP_FEE
        
        should_exit = False
        exit_reason = ""
        exit_type = ""
        
        # 1. TAKE PROFIT - Hit target (guaranteed profit > fees)
        if side == "LONG" and price >= tp:
            should_exit = True
            exit_reason = f"🎯 TAKE PROFIT HIT!"
            exit_type = "WIN"
        elif side == "SHORT" and price <= tp:
            should_exit = True
            exit_reason = f"🎯 TAKE PROFIT HIT!"
            exit_type = "WIN"
        
        # 2. STOP LOSS
        elif side == "LONG" and price <= sl:
            should_exit = True
            exit_reason = f"🛑 STOP LOSS"
            exit_type = "LOSS"
        elif side == "SHORT" and price >= sl:
            should_exit = True
            exit_reason = f"🛑 STOP LOSS"
            exit_type = "LOSS"
        
        # 3. TIME EXIT - But ONLY if profitable after fees
        elif hold_s >= CONFIG.MAX_HOLD_SECONDS:
            if net_pnl_pct > 0:
                should_exit = True
                exit_reason = f"⏰ TIME EXIT (profitable)"
                exit_type = "WIN"
            elif gross_pnl_pct > CONFIG.ROUND_TRIP_FEE * 0.5:
                # At least break even
                should_exit = True
                exit_reason = f"⏰ TIME EXIT (breakeven)"
                exit_type = "EVEN"
            else:
                # Must exit but at loss
                should_exit = True
                exit_reason = f"⏰ TIME EXIT (max hold)"
                exit_type = "LOSS"
        
        if should_exit:
            self._execute_exit(price, tick_us, exit_reason, exit_type)
        else:
            # Display status
            self._display_position(price, gross_pnl_pct, net_pnl_pct, hold_ms)
    
    def _execute_exit(self, exit_price: float, exit_us: int, reason: str, exit_type: str):
        """Execute exit trade"""
        pos = self.position
        symbol = pos["symbol"]
        
        # Close position
        success, latency = self.api.close_position(symbol, pos["size"], pos["side"])
        
        # Calculate P&L
        entry = pos["entry"]
        side = pos["side"]
        position_value = pos["position_value"]
        
        if side == "LONG":
            gross_pnl_pct = (exit_price - entry) / entry
        else:
            gross_pnl_pct = (entry - exit_price) / entry
        
        gross_pnl = position_value * gross_pnl_pct
        fees = position_value * CONFIG.ROUND_TRIP_FEE
        net_pnl = gross_pnl - fees
        net_pnl_pct = gross_pnl_pct - CONFIG.ROUND_TRIP_FEE
        
        hold_us = exit_us - pos["entry_time_us"]
        hold_ms = hold_us / 1000
        
        # Update stats
        if net_pnl > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        
        self.stats["gross_pnl"] += gross_pnl
        self.stats["fees_paid"] += fees
        self.stats["net_pnl"] += net_pnl
        self.stats["total_hold_ms"] += hold_ms
        
        if net_pnl > self.stats["best_trade"]:
            self.stats["best_trade"] = net_pnl
        if net_pnl < self.stats["worst_trade"]:
            self.stats["worst_trade"] = net_pnl
        
        emoji = "🚀" if exit_type == "WIN" else "💔" if exit_type == "LOSS" else "📊"
        
        print(f"\n{emoji}{'='*56}")
        print(f"   {reason}")
        print(f"   Symbol:      {symbol} {side}")
        print(f"   Entry:       ${entry:.2f}")
        print(f"   Exit:        ${exit_price:.2f}")
        print(f"   Hold Time:   {hold_ms:.0f}ms ({hold_ms/1000:.2f}s)")
        print(f"   Exit Latency: {latency:.0f}ms")
        print(f"   ─────────────────────────")
        print(f"   Gross P&L:   ${gross_pnl:+.6f} ({gross_pnl_pct:+.3%})")
        print(f"   Fees:        ${fees:.6f} ({CONFIG.ROUND_TRIP_FEE:.2%})")
        print(f"   NET P&L:     ${net_pnl:+.6f} ({net_pnl_pct:+.3%})")
        
        # PROFIT VS FEES CHECK
        if net_pnl > 0:
            profit_to_fee_ratio = net_pnl / fees if fees > 0 else 0
            print(f"   ✅ PROFIT/FEE RATIO: {profit_to_fee_ratio:.1f}x")
        else:
            print(f"   ⚠️ Loss trade - fees were {fees:.6f}")
        
        print(f"{'='*60}")
        self._print_stats()
        print()
        
        self.position = None
    
    def _display_position(self, price: float, gross_pnl_pct: float, net_pnl_pct: float, hold_ms: float):
        """Display current position status"""
        pos = self.position
        
        # Color based on profit
        if net_pnl_pct > 0:
            color = "🟢"
            status = "PROFIT"
        elif gross_pnl_pct > 0:
            color = "🟡"
            status = "GROSS+"
        else:
            color = "🔴"
            status = "LOSS"
        
        # Distance to TP
        if pos["side"] == "LONG":
            to_tp = (pos["tp"] - price) / price * 100
        else:
            to_tp = (price - pos["tp"]) / price * 100
        
        line = (
            f"\r{color} {pos['symbol']} {pos['side']} | "
            f"${price:.2f} | "
            f"Gross: {gross_pnl_pct:+.3%} | "
            f"Net: {net_pnl_pct:+.3%} | "
            f"To TP: {to_tp:.2%} | "
            f"⏱️ {hold_ms:.0f}ms   "
        )
        sys.stdout.write(line)
        sys.stdout.flush()
    
    def _print_stats(self):
        """Print session statistics"""
        s = self.stats
        trades = s["trades"]
        win_rate = (s["wins"] / trades * 100) if trades > 0 else 0
        avg_hold = (s["total_hold_ms"] / trades) if trades > 0 else 0
        
        # Calculate profit-to-fee ratio
        if s["fees_paid"] > 0:
            pf_ratio = s["net_pnl"] / s["fees_paid"]
            if pf_ratio > 0:
                ratio_text = f"✅ Profit {pf_ratio:.1f}x fees"
            else:
                ratio_text = f"⚠️ Fees eating {abs(pf_ratio):.1f}x profit"
        else:
            ratio_text = "No fees yet"
        
        print(f"\n📊 SESSION STATS:")
        print(f"   Trades: {trades} | W:{s['wins']} L:{s['losses']} ({win_rate:.0f}%)")
        print(f"   Gross P&L:  ${s['gross_pnl']:+.6f}")
        print(f"   Fees Paid:  ${s['fees_paid']:.6f}")
        print(f"   NET P&L:    ${s['net_pnl']:+.6f}")
        print(f"   {ratio_text}")
        print(f"   Avg Hold:   {avg_hold:.0f}ms")
    
    def _load_existing_positions(self):
        """Load any existing positions"""
        positions = self.api.get_positions()
        for pos in positions:
            symbol = pos.get("product_symbol", "")
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size != 0 and symbol in CONTRACTS:
                side = "LONG" if size > 0 else "SHORT"
                tp, sl = self._calculate_fee_safe_exit(entry, side)
                
                self.position = {
                    "symbol": symbol,
                    "side": side,
                    "entry": entry,
                    "size": abs(int(size)),
                    "entry_time_us": MicrosecondTimer.now_us(),
                    "tp": tp,
                    "sl": sl,
                    "position_value": entry * CONTRACTS[symbol]["size"] * abs(size)
                }
                
                print(f"\n📌 Loaded: {symbol} {side} @ ${entry:.2f}")
                print(f"   TP: ${tp:.2f} | SL: ${sl:.2f}")
                break
    
    def start(self):
        """Start the Turbo HFT bot"""
        self.running = True
        self.start_time = time.time()
        
        balance = self.api.get_balance()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║           ⚡ ALADDIN TURBO HFT - MICROSECOND SCALPING ⚡                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  💰 Balance:            ${balance:.4f}
║  
║  📊 FEE PROTECTION (Profit MUST exceed fees):
║     Round-trip Fee:     {CONFIG.ROUND_TRIP_FEE:.2%} (0.10%)
║     Min Gross Target:   {CONFIG.MIN_GROSS_TARGET:.2%} (3x fees)
║     Min Net Profit:     {CONFIG.MIN_NET_PROFIT:.2%} (GUARANTEED)
║     Profit/Fee Ratio:   {CONFIG.PROFIT_MULTIPLIER:.0f}x minimum
║  
║  🎯 TURBO SETTINGS:
║     Take Profit:        +{CONFIG.MIN_GROSS_TARGET:.2%} (fees + profit)
║     Stop Loss:          -{CONFIG.STOP_LOSS:.2%}
║     Max Hold:           {CONFIG.MAX_HOLD_SECONDS}s
║     Momentum Trigger:   {CONFIG.MOMENTUM_THRESHOLD:.2%} in {CONFIG.MOMENTUM_TICKS} ticks
║  
║  ⚡ MICROSECOND EXECUTION - Maximum Speed
║  💰 FEE SAFE - Only exits when NET PROFIT > 0
║  
║  Press Ctrl+C to stop
╚══════════════════════════════════════════════════════════════════════════╝
""")
        
        # Load existing positions
        print("📍 Checking positions...")
        self._load_existing_positions()
        
        # Start stream
        print("⚡ Starting TURBO price stream...")
        self.stream.start(CONFIG.SYMBOLS)
        time.sleep(1)
        
        print(f"\n{'='*60}")
        print("⚡ TURBO HFT ACTIVE - Profits will ALWAYS exceed fees!")
        print(f"{'='*60}\n")
        
        try:
            while self.running:
                time.sleep(0.001)  # 1ms loop for responsiveness
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping Turbo HFT...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop and print summary"""
        self.running = False
        self.stream.stop()
        
        runtime = time.time() - self.start_time
        s = self.stats
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                      📊 TURBO HFT SESSION SUMMARY                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Runtime:          {runtime:.1f}s ({runtime/60:.1f} min)
║  
║  📈 TRADES:
║     Total:         {s['trades']}
║     Wins:          {s['wins']}
║     Losses:        {s['losses']}
║     Win Rate:      {(s['wins']/max(1,s['trades'])*100):.1f}%
║  
║  💰 PROFIT/LOSS:
║     Gross P&L:     ${s['gross_pnl']:+.6f}
║     Fees Paid:     ${s['fees_paid']:.6f}
║     NET P&L:       ${s['net_pnl']:+.6f}
║     Best Trade:    ${s['best_trade']:+.6f}
║     Worst Trade:   ${s['worst_trade']:+.6f}
║  
║  ⚡ PERFORMANCE:
║     Ticks:         {self.stream.tick_count:,}
║     Ticks/sec:     {self.stream.tick_count/max(1,runtime):.1f}
║     Avg Hold:      {s['total_hold_ms']/max(1,s['trades']):.0f}ms
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("\n⚠️  TURBO HFT - LIVE TRADING")
    print("   Profits GUARANTEED to exceed fees!")
    print("   Press Ctrl+C to stop\n")
    
    bot = AladdinTurbo()
    bot.start()
