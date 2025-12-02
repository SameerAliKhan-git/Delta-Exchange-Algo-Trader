"""
Aladdin HFT - High Frequency Scalping Bot
==========================================
⚡ Takes profits in SECONDS
💰 Fee-aware: Only trades when profit > fees
📈 Momentum-based micro-scalping
🎯 Tight stops, quick exits

LIVE TRADING - HIGH RISK
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
from dataclasses import dataclass
import sys

# Disable buffering
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"
BASE_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

# Contract specifications
CONTRACTS = {
    "BTCUSD": {"size": 0.001, "min_move": 0.5, "product_id": None},
    "ETHUSD": {"size": 0.01, "min_move": 0.01, "product_id": None},
    "SOLUSD": {"size": 0.1, "min_move": 0.001, "product_id": None}
}

# =============================================================================
# HFT CONFIGURATION - CRITICAL SETTINGS
# =============================================================================

@dataclass
class HFTConfig:
    # Fee structure (Delta Exchange India)
    MAKER_FEE: float = 0.0002      # 0.02%
    TAKER_FEE: float = 0.0005      # 0.05%
    
    # Since we use market orders (taker), round-trip cost = 2 * 0.05% = 0.10%
    ROUND_TRIP_FEE: float = 0.0010  # 0.10%
    
    # Minimum profit target AFTER fees (must be > round_trip_fee)
    MIN_PROFIT_AFTER_FEES: float = 0.0015  # 0.15% net profit minimum
    
    # Therefore, gross target = 0.10% (fees) + 0.15% (profit) = 0.25%
    SCALP_TARGET: float = 0.0025   # 0.25% gross profit target
    
    # Tight stop loss for HFT
    STOP_LOSS: float = 0.0015      # 0.15% stop (small loss better than big)
    
    # Maximum hold time (seconds) - exit if no profit in this time
    MAX_HOLD_TIME: int = 60        # 60 seconds max per trade
    
    # Momentum detection settings
    MOMENTUM_WINDOW: int = 10      # Number of price ticks to analyze
    MOMENTUM_THRESHOLD: float = 0.0003  # 0.03% move triggers entry
    
    # Trade cooldown (seconds between trades)
    COOLDOWN: int = 5
    
    # Symbols to trade (start with most liquid)
    SYMBOLS: List[str] = None
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ["ETHUSD", "BTCUSD"]
    
    def calculate_breakeven(self) -> float:
        """Price must move this % just to break even"""
        return self.ROUND_TRIP_FEE
    
    def calculate_net_profit(self, gross_pnl_pct: float) -> float:
        """Calculate net profit after fees"""
        return gross_pnl_pct - self.ROUND_TRIP_FEE
    
    def is_profitable_trade(self, gross_pnl_pct: float) -> bool:
        """Check if trade would be profitable after fees"""
        return self.calculate_net_profit(gross_pnl_pct) >= self.MIN_PROFIT_AFTER_FEES

CONFIG = HFTConfig()


# =============================================================================
# ULTRA-LOW LATENCY PRICE STREAM
# =============================================================================

class HFTPriceStream:
    """High-frequency price stream with momentum detection"""
    
    def __init__(self, on_tick, on_momentum):
        self.on_tick = on_tick
        self.on_momentum = on_momentum
        self.ws = None
        self.running = False
        
        # Price data
        self.prices: Dict[str, float] = {}
        self.price_history: Dict[str, deque] = {}
        self.tick_count = 0
        
    def start(self, symbols: List[str]):
        self.running = True
        
        for symbol in symbols:
            self.price_history[symbol] = deque(maxlen=CONFIG.MOMENTUM_WINDOW)
        
        def run():
            while self.running:
                try:
                    self.ws = websocket.WebSocketApp(
                        WS_URL,
                        on_open=lambda ws: self._on_open(ws, symbols),
                        on_message=self._on_message,
                        on_error=lambda ws, e: None,
                        on_close=lambda ws, a, b: None
                    )
                    self.ws.run_forever(ping_interval=5)
                except:
                    time.sleep(0.5)
        
        threading.Thread(target=run, daemon=True).start()
        time.sleep(1)
    
    def _on_open(self, ws, symbols):
        print("⚡ HFT WebSocket connected")
        for symbol in symbols:
            ws.send(json.dumps({
                "type": "subscribe",
                "payload": {"channels": [
                    {"name": "v2/ticker", "symbols": [symbol]},
                    {"name": "mark_price", "symbols": [symbol]}
                ]}
            }))
    
    def _on_message(self, ws, message):
        tick_time = time.time()
        try:
            data = json.loads(message)
            symbol = data.get("symbol", "")
            mark_price = data.get("mark_price")
            
            if mark_price and symbol in self.price_history:
                price = float(mark_price)
                old_price = self.prices.get(symbol, price)
                self.prices[symbol] = price
                self.tick_count += 1
                
                # Store tick with timestamp
                self.price_history[symbol].append((tick_time, price))
                
                # Callback for every tick
                self.on_tick(symbol, price, tick_time)
                
                # Check for momentum
                momentum = self._detect_momentum(symbol)
                if momentum:
                    self.on_momentum(symbol, price, momentum, tick_time)
                    
        except:
            pass
    
    def _detect_momentum(self, symbol: str) -> Optional[str]:
        """Detect price momentum for entry signals"""
        history = self.price_history.get(symbol, deque())
        
        if len(history) < CONFIG.MOMENTUM_WINDOW:
            return None
        
        prices = [p[1] for p in history]
        oldest = prices[0]
        newest = prices[-1]
        
        change_pct = (newest - oldest) / oldest
        
        if change_pct > CONFIG.MOMENTUM_THRESHOLD:
            return "BULLISH"
        elif change_pct < -CONFIG.MOMENTUM_THRESHOLD:
            return "BEARISH"
        
        return None
    
    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 0)
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


# =============================================================================
# DELTA API CLIENT
# =============================================================================

class DeltaHFTClient:
    """Optimized API client for HFT"""
    
    def __init__(self):
        self.session = requests.Session()
        self.product_ids = {}
        self._load_products()
    
    def _sign(self, method: str, endpoint: str, payload: str = "") -> dict:
        ts = str(int(time.time()))
        sig = hmac.new(
            API_SECRET.encode(),
            f"{method}{ts}{endpoint}{payload}".encode(),
            hashlib.sha256
        ).hexdigest()
        return {
            "api-key": API_KEY,
            "timestamp": ts,
            "signature": sig,
            "Content-Type": "application/json"
        }
    
    def _load_products(self):
        """Load product IDs"""
        try:
            r = self.session.get(BASE_URL + "/v2/products", timeout=5)
            for p in r.json().get("result", []):
                symbol = p.get("symbol", "")
                if symbol in CONTRACTS:
                    self.product_ids[symbol] = p.get("id")
                    CONTRACTS[symbol]["product_id"] = p.get("id")
        except:
            pass
    
    def get_balance(self) -> float:
        try:
            endpoint = "/v2/wallet/balances"
            r = self.session.get(BASE_URL + endpoint, headers=self._sign("GET", endpoint), timeout=3)
            for a in r.json().get("result", []):
                if a.get("asset_symbol") in ["USD", "USDT"]:
                    return float(a.get("available_balance", 0))
        except:
            pass
        return 0
    
    def get_position(self, symbol: str) -> Optional[dict]:
        """Get specific position"""
        asset = symbol[:3]  # ETH from ETHUSD
        try:
            endpoint = f"/v2/positions?underlying_asset_symbol={asset}"
            r = self.session.get(BASE_URL + endpoint, headers=self._sign("GET", endpoint), timeout=3)
            for p in r.json().get("result", []):
                if p.get("product_symbol") == symbol and float(p.get("size", 0)) != 0:
                    return p
        except:
            pass
        return None
    
    def get_all_positions(self) -> List[dict]:
        """Get all open positions"""
        positions = []
        for asset in ["ETH", "BTC", "SOL"]:
            try:
                endpoint = f"/v2/positions?underlying_asset_symbol={asset}"
                r = self.session.get(BASE_URL + endpoint, headers=self._sign("GET", endpoint), timeout=3)
                for p in r.json().get("result", []):
                    if float(p.get("size", 0)) != 0:
                        positions.append(p)
            except:
                pass
        return positions
    
    def market_order(self, symbol: str, side: str, size: int) -> Tuple[bool, str]:
        """Place market order - returns (success, message)"""
        try:
            product_id = self.product_ids.get(symbol)
            if not product_id:
                return False, "No product ID"
            
            order = {
                "product_id": product_id,
                "side": side,
                "size": size,
                "order_type": "market_order"
            }
            payload = json.dumps(order)
            endpoint = "/v2/orders"
            
            r = self.session.post(
                BASE_URL + endpoint,
                headers=self._sign("POST", endpoint, payload),
                data=payload,
                timeout=5
            )
            
            result = r.json()
            if result.get("success"):
                return True, f"Order filled"
            else:
                return False, result.get("error", {}).get("message", "Failed")
                
        except Exception as e:
            return False, str(e)
    
    def close_position(self, symbol: str) -> Tuple[bool, str]:
        """Close position via market order"""
        pos = self.get_position(symbol)
        if not pos:
            return False, "No position"
        
        size = abs(int(float(pos.get("size", 0))))
        side = "sell" if float(pos.get("size", 0)) > 0 else "buy"
        
        return self.market_order(symbol, side, size)


# =============================================================================
# HFT SCALPING BOT
# =============================================================================

class AladdinHFT:
    """High Frequency Trading Scalping Bot"""
    
    def __init__(self):
        self.api = DeltaHFTClient()
        self.running = False
        
        # Active trade tracking
        self.active_trade = None  # {symbol, side, entry_price, entry_time, size}
        self.last_trade_time = 0
        
        # Statistics
        self.stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "total_gross_pnl": 0,
            "total_fees": 0,
            "total_net_pnl": 0
        }
        
        # Price stream
        self.price_stream = HFTPriceStream(
            on_tick=self._on_tick,
            on_momentum=self._on_momentum
        )
        
        self.start_time = None
    
    def _calculate_trade_size(self, symbol: str, balance: float) -> int:
        """Calculate position size - use minimum for HFT"""
        # For HFT, we use minimum size to reduce risk
        return 1  # 1 contract minimum
    
    def _on_tick(self, symbol: str, price: float, timestamp: float):
        """Process every price tick - CHECK FOR EXIT"""
        if not self.active_trade or self.active_trade["symbol"] != symbol:
            return
        
        trade = self.active_trade
        entry = trade["entry_price"]
        side = trade["side"]
        entry_time = trade["entry_time"]
        hold_time = timestamp - entry_time
        
        # Calculate P&L
        if side == "LONG":
            gross_pnl_pct = (price - entry) / entry
        else:
            gross_pnl_pct = (entry - price) / entry
        
        net_pnl_pct = CONFIG.calculate_net_profit(gross_pnl_pct)
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # 1. TAKE PROFIT - Target reached
        if gross_pnl_pct >= CONFIG.SCALP_TARGET:
            should_exit = True
            exit_reason = f"🎯 TARGET HIT! +{gross_pnl_pct:.3%} gross, +{net_pnl_pct:.3%} net"
        
        # 2. STOP LOSS
        elif gross_pnl_pct <= -CONFIG.STOP_LOSS:
            should_exit = True
            exit_reason = f"🛑 STOP LOSS! {gross_pnl_pct:.3%}"
        
        # 3. TIME EXIT - Max hold time exceeded
        elif hold_time >= CONFIG.MAX_HOLD_TIME:
            should_exit = True
            exit_reason = f"⏰ TIME EXIT ({hold_time:.0f}s) @ {gross_pnl_pct:+.3%}"
        
        # 4. BREAKEVEN EXIT - If past 30s and barely profitable, exit
        elif hold_time > 30 and gross_pnl_pct > CONFIG.ROUND_TRIP_FEE:
            should_exit = True
            exit_reason = f"📊 BREAKEVEN+ EXIT @ {gross_pnl_pct:+.3%}"
        
        if should_exit:
            self._exit_trade(price, gross_pnl_pct, net_pnl_pct, exit_reason, hold_time)
        else:
            # Display current status
            self._display_trade_status(symbol, side, entry, price, gross_pnl_pct, net_pnl_pct, hold_time)
    
    def _on_momentum(self, symbol: str, price: float, direction: str, timestamp: float):
        """Momentum detected - CHECK FOR ENTRY"""
        
        # Skip if we have an active trade
        if self.active_trade:
            return
        
        # Check cooldown
        if timestamp - self.last_trade_time < CONFIG.COOLDOWN:
            return
        
        # Check balance
        balance = self.api.get_balance()
        if balance < 0.20:  # Need at least $0.20
            return
        
        # Determine trade direction
        if direction == "BULLISH":
            side = "buy"
            trade_side = "LONG"
        else:
            side = "sell"
            trade_side = "SHORT"
        
        # Calculate expected fees
        contract_size = CONTRACTS[symbol]["size"]
        position_value = price * contract_size
        total_fees = position_value * CONFIG.ROUND_TRIP_FEE
        expected_profit = position_value * CONFIG.MIN_PROFIT_AFTER_FEES
        
        print(f"\n{'='*60}")
        print(f"⚡ MOMENTUM DETECTED: {symbol} {direction}")
        print(f"   Price: ${price:.2f}")
        print(f"   Position Value: ${position_value:.4f}")
        print(f"   Round-trip Fees: ${total_fees:.6f} ({CONFIG.ROUND_TRIP_FEE:.2%})")
        print(f"   Target Profit:   ${expected_profit:.6f} ({CONFIG.SCALP_TARGET:.2%} gross)")
        print(f"   Net Profit:      ${expected_profit - total_fees:.6f} ({CONFIG.MIN_PROFIT_AFTER_FEES:.2%})")
        
        # Place order
        success, msg = self.api.market_order(symbol, side, 1)
        
        if success:
            self.active_trade = {
                "symbol": symbol,
                "side": trade_side,
                "entry_price": price,
                "entry_time": timestamp,
                "size": 1,
                "position_value": position_value
            }
            self.last_trade_time = timestamp
            self.stats["trades"] += 1
            
            print(f"✅ ENTERED {trade_side} @ ${price:.2f}")
            print(f"   Stop Loss:    ${price * (1 - CONFIG.STOP_LOSS) if trade_side == 'LONG' else price * (1 + CONFIG.STOP_LOSS):.2f} ({CONFIG.STOP_LOSS:.2%})")
            print(f"   Take Profit:  ${price * (1 + CONFIG.SCALP_TARGET) if trade_side == 'LONG' else price * (1 - CONFIG.SCALP_TARGET):.2f} ({CONFIG.SCALP_TARGET:.2%})")
            print(f"   Max Hold:     {CONFIG.MAX_HOLD_TIME}s")
            print(f"{'='*60}\n")
        else:
            print(f"❌ Entry failed: {msg}")
    
    def _exit_trade(self, exit_price: float, gross_pnl_pct: float, net_pnl_pct: float, reason: str, hold_time: float):
        """Exit current trade"""
        trade = self.active_trade
        symbol = trade["symbol"]
        
        # Close position
        success, msg = self.api.close_position(symbol)
        
        # Calculate actual P&L
        contract_size = CONTRACTS[symbol]["size"]
        position_value = trade["position_value"]
        gross_pnl = position_value * gross_pnl_pct
        fees = position_value * CONFIG.ROUND_TRIP_FEE
        net_pnl = gross_pnl - fees
        
        # Update stats
        if net_pnl > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        
        self.stats["total_gross_pnl"] += gross_pnl
        self.stats["total_fees"] += fees
        self.stats["total_net_pnl"] += net_pnl
        
        print(f"\n{'🚀' if net_pnl > 0 else '💔'}{'='*56}")
        print(f"   {reason}")
        print(f"   Symbol:     {symbol} {trade['side']}")
        print(f"   Entry:      ${trade['entry_price']:.2f}")
        print(f"   Exit:       ${exit_price:.2f}")
        print(f"   Hold Time:  {hold_time:.1f} seconds")
        print(f"   Gross P&L:  ${gross_pnl:+.6f} ({gross_pnl_pct:+.3%})")
        print(f"   Fees:       ${fees:.6f} ({CONFIG.ROUND_TRIP_FEE:.2%})")
        print(f"   NET P&L:    ${net_pnl:+.6f} ({net_pnl_pct:+.3%})")
        print(f"{'='*60}")
        self._print_session_stats()
        print()
        
        self.active_trade = None
    
    def _display_trade_status(self, symbol: str, side: str, entry: float, current: float, 
                              gross_pnl_pct: float, net_pnl_pct: float, hold_time: float):
        """Display current trade status"""
        color = "🟢" if gross_pnl_pct > 0 else "🔴" if gross_pnl_pct < 0 else "⚪"
        breakeven = "📈" if gross_pnl_pct > CONFIG.ROUND_TRIP_FEE else "📉"
        
        status = (
            f"\r{color} {symbol} {side} | "
            f"${current:.2f} | "
            f"Gross: {gross_pnl_pct:+.3%} | "
            f"Net: {net_pnl_pct:+.3%} {breakeven} | "
            f"⏱️ {hold_time:.1f}s/{CONFIG.MAX_HOLD_TIME}s   "
        )
        sys.stdout.write(status)
        sys.stdout.flush()
    
    def _print_session_stats(self):
        """Print session statistics"""
        win_rate = (self.stats["wins"] / self.stats["trades"] * 100) if self.stats["trades"] > 0 else 0
        print(f"\n📊 SESSION: {self.stats['trades']} trades | "
              f"W:{self.stats['wins']} L:{self.stats['losses']} ({win_rate:.0f}%) | "
              f"Net P&L: ${self.stats['total_net_pnl']:+.6f} | "
              f"Fees Paid: ${self.stats['total_fees']:.6f}")
    
    def _handle_existing_position(self):
        """Check and manage existing positions"""
        positions = self.api.get_all_positions()
        
        for pos in positions:
            symbol = pos.get("product_symbol", "")
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size != 0:
                side = "LONG" if size > 0 else "SHORT"
                print(f"\n📌 Found existing position: {symbol} {side} @ ${entry:.2f}")
                
                # Track this position
                self.active_trade = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry,
                    "entry_time": time.time(),  # Use current time
                    "size": abs(int(size)),
                    "position_value": entry * CONTRACTS.get(symbol, {}).get("size", 0.01) * abs(size)
                }
                break
    
    def start(self):
        """Start the HFT bot"""
        self.running = True
        self.start_time = time.time()
        
        balance = self.api.get_balance()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║              ⚡ ALADDIN HFT - HIGH FREQUENCY SCALPING BOT ⚡              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  💰 Balance:           ${balance:.4f}
║  
║  📊 FEE STRUCTURE:
║     Taker Fee:         {CONFIG.TAKER_FEE:.2%} per trade
║     Round-trip Cost:   {CONFIG.ROUND_TRIP_FEE:.2%}
║     Breakeven:         Price must move {CONFIG.ROUND_TRIP_FEE:.2%} to break even
║  
║  🎯 HFT SETTINGS:
║     Scalp Target:      {CONFIG.SCALP_TARGET:.2%} (gross)
║     Net Profit Target: {CONFIG.MIN_PROFIT_AFTER_FEES:.2%} (after fees)
║     Stop Loss:         {CONFIG.STOP_LOSS:.2%}
║     Max Hold Time:     {CONFIG.MAX_HOLD_TIME} seconds
║     Momentum Trigger:  {CONFIG.MOMENTUM_THRESHOLD:.2%} move in {CONFIG.MOMENTUM_WINDOW} ticks
║  
║  ⚡ ULTRA-FAST: Exits in SECONDS when target hit
║  💰 FEE-AWARE: Only trades when profit > fees
║  
║  Press Ctrl+C to stop
╚══════════════════════════════════════════════════════════════════════════╝
""")
        
        # Check for existing positions
        print("📍 Checking for existing positions...")
        self._handle_existing_position()
        
        # Start price stream
        print("⚡ Starting HFT price stream...")
        self.price_stream.start(CONFIG.SYMBOLS)
        time.sleep(2)
        
        print(f"\n{'='*60}")
        print("⚡ HFT SCALPING ACTIVE - Waiting for momentum signals...")
        print(f"{'='*60}\n")
        
        try:
            while self.running:
                time.sleep(0.01)  # 10ms loop
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping HFT bot...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the bot and print final stats"""
        self.running = False
        self.price_stream.stop()
        
        runtime = time.time() - self.start_time if self.start_time else 0
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         📊 HFT SESSION SUMMARY                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Runtime:           {runtime:.1f} seconds ({runtime/60:.1f} minutes)
║  
║  📈 TRADES:
║     Total:          {self.stats['trades']}
║     Wins:           {self.stats['wins']}
║     Losses:         {self.stats['losses']}
║     Win Rate:       {(self.stats['wins']/max(1,self.stats['trades'])*100):.1f}%
║  
║  💰 PROFIT/LOSS:
║     Gross P&L:      ${self.stats['total_gross_pnl']:+.6f}
║     Total Fees:     ${self.stats['total_fees']:.6f}
║     NET P&L:        ${self.stats['total_net_pnl']:+.6f}
║  
║  ⚡ EFFICIENCY:
║     Ticks Processed: {self.price_stream.tick_count:,}
║     Ticks/Second:   {self.price_stream.tick_count/max(1,runtime):.1f}
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("\n⚠️  WARNING: HFT LIVE TRADING MODE")
    print("    This bot will execute real trades!")
    print("    Press Ctrl+C at any time to stop\n")
    
    bot = AladdinHFT()
    bot.start()
