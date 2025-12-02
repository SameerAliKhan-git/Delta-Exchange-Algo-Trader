"""
Aladdin Real-Time Trading Bot
=============================
Features:
- REAL-TIME UPNL updates via WebSocket
- Fee-aware profit calculations before entry
- Automatic position management
- Sentiment-based exits

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
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AladdinBot")

# API Configuration
API_KEY = "TcYH2ep58nMawSd15Ff7aOtsXUARO7"
API_SECRET = "VRQkpWBR8Y3ioeNB3HaTNklcpjpYhfOJ2LSZVcLKHrNnCbSbukcQtlH7mG8n"
BASE_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Trading configuration with fee awareness"""
    # Delta Exchange India fees
    MAKER_FEE: float = 0.0002  # 0.02% maker fee
    TAKER_FEE: float = 0.0005  # 0.05% taker fee
    
    # MINIMUM PROFIT REQUIREMENTS
    MIN_PROFIT_AFTER_FEES_PCT: float = 0.003  # Need at least 0.3% profit after fees
    MIN_EXPECTED_MOVE_PCT: float = 0.01  # Expect at least 1% move to enter
    
    # Risk management
    MAX_POSITION_SIZE_PCT: float = 0.90  # 90% of balance for micro accounts
    STOP_LOSS_PCT: float = 0.02  # 2% stop loss
    TAKE_PROFIT_PCT: float = 0.04  # 4% take profit
    MAX_OPEN_POSITIONS: int = 1
    DEFAULT_LEVERAGE: int = 20
    
    # Trading pairs
    TRADING_SYMBOLS: List[str] = None
    
    def __post_init__(self):
        if self.TRADING_SYMBOLS is None:
            self.TRADING_SYMBOLS = ["SOLUSD", "ETHUSD", "BTCUSD"]
    
    def calculate_round_trip_fees(self, position_value: float) -> float:
        """Calculate total fees for entering and exiting a position"""
        entry_fee = position_value * self.TAKER_FEE
        exit_fee = position_value * self.TAKER_FEE
        return entry_fee + exit_fee
    
    def calculate_breakeven_move(self) -> float:
        """Calculate minimum price move needed to break even after fees"""
        # Round trip = 2 * taker fee
        return 2 * self.TAKER_FEE
    
    def is_trade_profitable(self, expected_move_pct: float, position_value: float) -> Tuple[bool, float, float]:
        """
        Check if a trade is expected to be profitable after fees
        
        Returns:
            Tuple of (is_profitable, expected_profit, total_fees)
        """
        total_fees = self.calculate_round_trip_fees(position_value)
        expected_profit = position_value * expected_move_pct
        net_profit = expected_profit - total_fees
        
        is_profitable = net_profit > (position_value * self.MIN_PROFIT_AFTER_FEES_PCT)
        
        return is_profitable, net_profit, total_fees


# Contract sizes for Delta Exchange
CONTRACT_SIZES = {
    "BTCUSD": 0.001,  # 1 contract = 0.001 BTC
    "ETHUSD": 0.01,   # 1 contract = 0.01 ETH
    "SOLUSD": 0.1     # 1 contract = 0.1 SOL
}


# =============================================================================
# REAL-TIME PRICE TRACKER (WebSocket)
# =============================================================================

class RealTimePriceTracker:
    """Tracks prices in real-time via WebSocket"""
    
    def __init__(self):
        self.prices: Dict[str, float] = {}
        self.ws = None
        self.running = False
        self.connected = False
        self.last_update: Dict[str, datetime] = {}
        
    def start(self, symbols: List[str]):
        """Start WebSocket connection for real-time prices"""
        self.running = True
        self.symbols = symbols
        
        def run_websocket():
            while self.running:
                try:
                    self.ws = websocket.WebSocketApp(
                        WS_URL,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close
                    )
                    self.ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    time.sleep(5)
        
        self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        self.ws_thread.start()
        
        # Wait for connection
        for _ in range(10):
            if self.connected:
                break
            time.sleep(0.5)
    
    def _on_open(self, ws):
        """Subscribe to ticker channels"""
        self.connected = True
        logger.info("📡 WebSocket connected - subscribing to real-time prices")
        
        for symbol in self.symbols:
            subscribe_msg = {
                "type": "subscribe",
                "payload": {
                    "channels": [
                        {"name": "v2/ticker", "symbols": [symbol]}
                    ]
                }
            }
            ws.send(json.dumps(subscribe_msg))
    
    def _on_message(self, ws, message):
        """Handle incoming price updates"""
        try:
            data = json.loads(message)
            
            if data.get("type") == "v2/ticker":
                symbol = data.get("symbol", "")
                mark_price = data.get("mark_price")
                
                if mark_price:
                    self.prices[symbol] = float(mark_price)
                    self.last_update[symbol] = datetime.now()
                    
        except Exception as e:
            pass  # Ignore parse errors
    
    def _on_error(self, ws, error):
        logger.warning(f"WebSocket error: {error}")
        self.connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket closed, reconnecting...")
        self.connected = False
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.prices.get(symbol)
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()


# =============================================================================
# DELTA EXCHANGE API CLIENT
# =============================================================================

class DeltaExchangeClient:
    """Delta Exchange India API Client"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BASE_URL
        self.session = requests.Session()
    
    def _generate_signature(self, method: str, endpoint: str, payload: str, timestamp: str) -> str:
        message = f"{method}{timestamp}{endpoint}{payload}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method: str, endpoint: str, data: dict = None) -> Dict:
        timestamp = str(int(time.time()))
        payload = json.dumps(data) if data else ""
        signature = self._generate_signature(method, endpoint, payload, timestamp)
        
        headers = {
            "api-key": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            "Content-Type": "application/json"
        }
        
        url = self.base_url + endpoint
        
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, timeout=30)
            elif method == "POST":
                response = self.session.post(url, headers=headers, data=payload, timeout=30)
            elif method == "DELETE":
                response = self.session.delete(url, timeout=30)
            
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_code = error_data.get('error', {}).get('code', 'unknown')
                    error_context = error_data.get('error', {}).get('context', {})
                    
                    if error_code == 'insufficient_margin':
                        available = error_context.get('available_balance', 'N/A')
                        required = error_context.get('required_additional_balance', 'N/A')
                        logger.error(f"💰 INSUFFICIENT MARGIN: Have ${available}, Need ${required} more")
                    else:
                        logger.error(f"API Error: {error_code}")
                except:
                    logger.error(f"API Error: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}
    
    def get_available_balance(self) -> float:
        result = self._request("GET", "/v2/wallet/balances")
        if "result" in result:
            for asset in result["result"]:
                if asset.get("asset_symbol") in ["USDT", "USD", "USDC"]:
                    return float(asset.get("available_balance", 0))
        return 0.0
    
    def get_positions(self) -> List[Dict]:
        all_positions = []
        for asset in ["BTC", "ETH", "SOL"]:
            result = self._request("GET", f"/v2/positions?underlying_asset_symbol={asset}")
            if "result" in result:
                positions = [p for p in result["result"] if float(p.get("size", 0)) != 0]
                all_positions.extend(positions)
        return all_positions
    
    def get_ticker(self, symbol: str) -> Dict:
        try:
            response = self.session.get(f"{self.base_url}/v2/tickers/{symbol}", timeout=10)
            if response.status_code == 200:
                return response.json().get("result", {})
        except:
            pass
        return {}
    
    def _get_product_id(self, symbol: str) -> Optional[int]:
        try:
            response = self.session.get(f"{self.base_url}/v2/products", timeout=10)
            if response.status_code == 200:
                for p in response.json().get("result", []):
                    if p.get("symbol") == symbol:
                        return p.get("id")
        except:
            pass
        return None
    
    def place_order(self, symbol: str, side: str, size: int, stop_loss: float = None, take_profit: float = None) -> Dict:
        product_id = self._get_product_id(symbol)
        if not product_id:
            return {"error": f"Product not found: {symbol}"}
        
        order_data = {
            "product_id": product_id,
            "side": side,
            "size": size,
            "order_type": "market_order"
        }
        
        if stop_loss:
            order_data["stop_loss_order"] = {
                "order_type": "market_order",
                "stop_price": str(stop_loss)
            }
        
        if take_profit:
            order_data["take_profit_order"] = {
                "order_type": "market_order",
                "stop_price": str(take_profit)
            }
        
        return self._request("POST", "/v2/orders", order_data)
    
    def close_position(self, symbol: str) -> Dict:
        product_id = self._get_product_id(symbol)
        if not product_id:
            return {"error": f"Product not found: {symbol}"}
        return self._request("POST", f"/v2/positions/{product_id}/close", {})


# =============================================================================
# NEWS SENTIMENT ANALYZER
# =============================================================================

class NewsSentimentAnalyzer:
    """Analyzes crypto news for trading signals"""
    
    BULLISH_KEYWORDS = [
        "bitcoin etf approved", "crypto adoption", "institutional buying",
        "rate cut", "fed dovish", "inflation falling", "stimulus",
        "bitcoin halving", "ethereum upgrade", "bullish", "rally",
        "accumulation", "whale buying", "record inflows", "adoption",
        "partnership", "integration", "approval", "pump", "moon", "ath"
    ]
    
    BEARISH_KEYWORDS = [
        "crypto ban", "regulation crackdown", "sec lawsuit", "hack",
        "exchange collapse", "rate hike", "fed hawkish", "inflation rising",
        "recession", "crash", "selloff", "liquidation", "outflows",
        "whale selling", "fraud", "investigation", "bearish", "dump"
    ]
    
    def __init__(self):
        self.news_cache = []
        self.last_fetch = None
        self.sentiment_score = 0.0
    
    def fetch_news(self) -> List[Dict]:
        news = []
        
        # CryptoCompare
        try:
            resp = requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN", timeout=10)
            if resp.status_code == 200:
                for item in resp.json().get("Data", [])[:15]:
                    news.append({"title": item.get("title", ""), "body": item.get("body", "")})
        except:
            pass
        
        self.news_cache = news
        self.last_fetch = datetime.now()
        return news
    
    def get_sentiment(self) -> Tuple[float, str]:
        """Get market sentiment from news"""
        if not self.last_fetch or (datetime.now() - self.last_fetch) > timedelta(minutes=10):
            self.fetch_news()
        
        if not self.news_cache:
            return 0.0, "neutral"
        
        scores = []
        for news in self.news_cache:
            text = (news.get("title", "") + " " + news.get("body", "")).lower()
            bull = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text)
            bear = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text)
            if bull + bear > 0:
                scores.append((bull - bear) / (bull + bear))
        
        if not scores:
            return 0.0, "neutral"
        
        avg = sum(scores) / len(scores)
        self.sentiment_score = avg
        
        if avg > 0.15:
            return avg, "bullish"
        elif avg < -0.15:
            return avg, "bearish"
        return avg, "neutral"


# =============================================================================
# REAL-TIME TRADING BOT
# =============================================================================

class AladdinRealTimeBot:
    """Real-time trading bot with live UPNL and fee-aware trading"""
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.client = DeltaExchangeClient(API_KEY, API_SECRET)
        self.price_tracker = RealTimePriceTracker()
        self.sentiment = NewsSentimentAnalyzer()
        
        self.running = False
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.last_upnl_display = None
        
        # Stats
        self.trades_today = 0
        self.daily_pnl = 0.0
    
    def calculate_upnl(self, symbol: str, entry_price: float, size: float, current_price: float) -> Tuple[float, float]:
        """Calculate unrealized PnL"""
        contract_size = CONTRACT_SIZES.get(symbol, 0.01)
        upnl = (current_price - entry_price) * size * contract_size
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        return upnl, pnl_pct
    
    def should_enter_trade(self, symbol: str, direction: str, balance: float) -> Tuple[bool, str, Dict]:
        """
        Check if trade should be entered based on:
        1. Sentiment strength
        2. Expected profit after fees
        """
        # Get current price
        ticker = self.client.get_ticker(symbol)
        if not ticker:
            return False, "No price data", {}
        
        current_price = float(ticker.get("mark_price", 0))
        if current_price == 0:
            return False, "Invalid price", {}
        
        # Calculate position details
        contract_size = CONTRACT_SIZES.get(symbol, 0.01)
        leverage = self.config.DEFAULT_LEVERAGE
        
        # Calculate position value for 1 contract
        position_value = contract_size * current_price
        margin_required = position_value / leverage
        
        # Check if we have enough margin
        if margin_required > balance * self.config.MAX_POSITION_SIZE_PCT:
            return False, f"Insufficient margin: need ${margin_required:.4f}", {}
        
        # Calculate fees for round trip
        total_fees = self.config.calculate_round_trip_fees(position_value)
        
        # Expected move based on sentiment and volatility
        sentiment_score, sentiment_dir = self.sentiment.get_sentiment()
        
        # Expected profit based on take profit target
        expected_profit = position_value * self.config.TAKE_PROFIT_PCT
        
        # Net profit after fees
        net_profit = expected_profit - total_fees
        
        # Check if profitable
        is_profitable = net_profit > (position_value * self.config.MIN_PROFIT_AFTER_FEES_PCT)
        
        # Breakeven price move needed
        breakeven_pct = self.config.calculate_breakeven_move() * 100
        
        analysis = {
            "symbol": symbol,
            "current_price": current_price,
            "position_value": position_value,
            "margin_required": margin_required,
            "total_fees": total_fees,
            "expected_profit": expected_profit,
            "net_profit": net_profit,
            "breakeven_move_pct": breakeven_pct,
            "sentiment_score": sentiment_score,
            "is_profitable": is_profitable
        }
        
        if not is_profitable:
            return False, f"Trade not profitable after fees. Net: ${net_profit:.4f}, Fees: ${total_fees:.4f}", analysis
        
        # Check sentiment alignment
        if direction == "long" and sentiment_score < 0.05:
            return False, f"Sentiment not bullish enough: {sentiment_score:.2f}", analysis
        if direction == "short" and sentiment_score > -0.05:
            return False, f"Sentiment not bearish enough: {sentiment_score:.2f}", analysis
        
        return True, "Trade approved", analysis
    
    def display_realtime_upnl(self):
        """Display real-time UPNL for all positions"""
        positions = self.client.get_positions()
        
        if not positions:
            return
        
        now = datetime.now()
        
        for pos in positions:
            symbol = pos.get("product_symbol", pos.get("symbol", ""))
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size == 0:
                continue
            
            # Get real-time price (WebSocket or fallback to API)
            current_price = self.price_tracker.get_price(symbol)
            if not current_price:
                ticker = self.client.get_ticker(symbol)
                current_price = float(ticker.get("mark_price", entry)) if ticker else entry
            
            upnl, pnl_pct = self.calculate_upnl(symbol, entry, size, current_price)
            
            # Store position info
            self.positions[symbol] = {
                "size": size,
                "entry": entry,
                "current": current_price,
                "upnl": upnl,
                "pnl_pct": pnl_pct,
                "direction": "LONG" if size > 0 else "SHORT"
            }
            
            # Display
            emoji = "🟢" if upnl >= 0 else "🔴"
            arrow = "↑" if upnl > 0 else "↓" if upnl < 0 else "→"
            
            # Clear line and print
            sys.stdout.write(f"\r{emoji} {symbol}: {self.positions[symbol]['direction']} | Entry: ${entry:.2f} | Now: ${current_price:.2f} {arrow} | UPNL: ${upnl:+.4f} ({pnl_pct:+.2%})   ")
            sys.stdout.flush()
            
            # Check for exit conditions
            self.check_exit_conditions(symbol, upnl, pnl_pct)
    
    def check_exit_conditions(self, symbol: str, upnl: float, pnl_pct: float):
        """Check if position should be closed"""
        pos = self.positions.get(symbol, {})
        if not pos:
            return
        
        is_long = pos["direction"] == "LONG"
        sentiment_score = self.sentiment.sentiment_score
        
        should_exit = False
        reason = ""
        
        # Stop Loss
        if pnl_pct < -self.config.STOP_LOSS_PCT:
            should_exit = True
            reason = f"🛑 STOP LOSS: {pnl_pct:.2%}"
        
        # Take Profit
        elif pnl_pct > self.config.TAKE_PROFIT_PCT:
            should_exit = True
            reason = f"🎯 TAKE PROFIT: {pnl_pct:.2%}"
        
        # Sentiment Reversal
        elif is_long and sentiment_score < -0.15:
            should_exit = True
            reason = f"📉 SENTIMENT BEARISH: {sentiment_score:.2f}"
        elif not is_long and sentiment_score > 0.15:
            should_exit = True
            reason = f"📈 SENTIMENT BULLISH: {sentiment_score:.2f}"
        
        if should_exit:
            print(f"\n⚠️ CLOSING {symbol}: {reason}")
            result = self.client.close_position(symbol)
            if "error" not in result:
                print(f"✅ Position closed. Final P&L: ${upnl:+.4f}")
                self.daily_pnl += upnl
                del self.positions[symbol]
            else:
                print(f"❌ Failed to close: {result}")
    
    def execute_trade(self, symbol: str, direction: str) -> bool:
        """Execute a trade with fee analysis"""
        balance = self.client.get_available_balance()
        
        # Check if trade is profitable after fees
        should_trade, reason, analysis = self.should_enter_trade(symbol, direction, balance)
        
        print(f"\n{'='*60}")
        print(f"📊 TRADE ANALYSIS: {symbol}")
        print(f"{'='*60}")
        print(f"   Direction:      {direction.upper()}")
        print(f"   Current Price:  ${analysis.get('current_price', 0):.2f}")
        print(f"   Position Value: ${analysis.get('position_value', 0):.4f}")
        print(f"   Margin Needed:  ${analysis.get('margin_required', 0):.4f}")
        print(f"   Available:      ${balance:.4f}")
        print(f"   ")
        print(f"   📌 FEE ANALYSIS:")
        print(f"   Total Fees:     ${analysis.get('total_fees', 0):.6f}")
        print(f"   Breakeven Move: {analysis.get('breakeven_move_pct', 0):.3f}%")
        print(f"   Expected Profit:${analysis.get('expected_profit', 0):.4f}")
        print(f"   Net After Fees: ${analysis.get('net_profit', 0):.4f}")
        print(f"   Profitable:     {'✅ YES' if analysis.get('is_profitable') else '❌ NO'}")
        print(f"   ")
        print(f"   Sentiment:      {analysis.get('sentiment_score', 0):+.2f}")
        print(f"{'='*60}")
        
        if not should_trade:
            print(f"❌ Trade rejected: {reason}")
            return False
        
        print(f"✅ Trade approved: {reason}")
        
        # Place order
        ticker = self.client.get_ticker(symbol)
        current_price = float(ticker.get("mark_price", 0))
        
        if direction == "long":
            stop_loss = current_price * (1 - self.config.STOP_LOSS_PCT)
            take_profit = current_price * (1 + self.config.TAKE_PROFIT_PCT)
        else:
            stop_loss = current_price * (1 + self.config.STOP_LOSS_PCT)
            take_profit = current_price * (1 - self.config.TAKE_PROFIT_PCT)
        
        side = "buy" if direction == "long" else "sell"
        result = self.client.place_order(symbol, side, 1, stop_loss, take_profit)
        
        if "error" not in result:
            print(f"✅ Order placed! ID: {result.get('result', {}).get('id', 'N/A')}")
            self.trades_today += 1
            return True
        else:
            print(f"❌ Order failed: {result}")
            return False
    
    def start(self):
        """Start the real-time trading bot"""
        self.running = True
        
        # Start price tracker
        print("📡 Connecting to real-time price feed...")
        self.price_tracker.start(self.config.TRADING_SYMBOLS)
        time.sleep(2)
        
        # Get initial data
        balance = self.client.get_available_balance()
        positions = self.client.get_positions()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    ALADDIN REAL-TIME TRADING BOT                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  💰 Balance:     ${balance:.4f}                                       
║  📊 Positions:   {len(positions)}                                              
║  🎯 Symbols:     {', '.join(self.config.TRADING_SYMBOLS)}                       
║  ⚡ Leverage:    {self.config.DEFAULT_LEVERAGE}x                                
║  💸 Taker Fee:   {self.config.TAKER_FEE:.3%}                                    
║  🛑 Stop Loss:   {self.config.STOP_LOSS_PCT:.1%}                                
║  🎯 Take Profit: {self.config.TAKE_PROFIT_PCT:.1%}                              
╠══════════════════════════════════════════════════════════════════════╣
║  ⚡ REAL-TIME UPNL MONITORING ACTIVE                                 ║
║  📰 Fee-aware trading enabled                                        ║
║  Press Ctrl+C to stop                                                ║
╚══════════════════════════════════════════════════════════════════════╝
""")
        
        last_sentiment_check = None
        
        try:
            while self.running:
                # Real-time UPNL display (every 2 seconds)
                self.display_realtime_upnl()
                
                # Check sentiment and trading opportunities every 5 minutes
                now = datetime.now()
                if not last_sentiment_check or (now - last_sentiment_check) > timedelta(minutes=5):
                    print(f"\n\n{'━'*60}")
                    print(f"📰 Checking market sentiment... {now.strftime('%H:%M:%S')}")
                    
                    sentiment_score, sentiment_dir = self.sentiment.get_sentiment()
                    print(f"   Sentiment: {sentiment_dir.upper()} ({sentiment_score:+.2f})")
                    
                    # Check if we should trade
                    if len(self.positions) < self.config.MAX_OPEN_POSITIONS:
                        if sentiment_score > 0.1:
                            for symbol in self.config.TRADING_SYMBOLS:
                                if symbol not in self.positions:
                                    self.execute_trade(symbol, "long")
                                    break
                        elif sentiment_score < -0.1:
                            for symbol in self.config.TRADING_SYMBOLS:
                                if symbol not in self.positions:
                                    self.execute_trade(symbol, "short")
                                    break
                    
                    print(f"{'━'*60}\n")
                    last_sentiment_check = now
                
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping bot...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the bot"""
        self.running = False
        self.price_tracker.stop()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  📊 SESSION SUMMARY                                          ║
╠══════════════════════════════════════════════════════════════╣
║  Trades Today:  {self.trades_today}
║  Daily P&L:     ${self.daily_pnl:+.4f}
╚══════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config = TradingConfig(
        STOP_LOSS_PCT=0.02,
        TAKE_PROFIT_PCT=0.04,
        DEFAULT_LEVERAGE=20,
        MAX_OPEN_POSITIONS=1
    )
    
    bot = AladdinRealTimeBot(config)
    bot.start()
