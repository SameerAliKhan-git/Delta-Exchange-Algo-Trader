"""
Aladdin Autonomous Trading Bot
==============================
Trades autonomously based on:
- Account balance from Delta Exchange
- Global news & economic sentiment
- Fee-aware profit calculations
- Automatic position management

LIVE TRADING - USE WITH CAUTION
"""

import time
import hmac
import hashlib
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AladdinBot")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Trading configuration with fee awareness"""
    # Delta Exchange India fees
    MAKER_FEE: float = 0.0002  # 0.02% maker fee
    TAKER_FEE: float = 0.0005  # 0.05% taker fee
    
    # Risk management - Aggressive for small accounts
    MAX_POSITION_SIZE_PCT: float = 0.80  # Max 80% of balance per trade for micro accounts
    MIN_PROFIT_AFTER_FEES: float = 0.002  # Min 0.2% profit after fees to enter
    MAX_DAILY_LOSS_PCT: float = 0.20  # Max 20% daily loss (aggressive)
    MAX_DRAWDOWN_PCT: float = 0.25  # Max 25% drawdown
    
    # Position management
    STOP_LOSS_PCT: float = 0.02  # 2% stop loss
    TAKE_PROFIT_PCT: float = 0.04  # 4% take profit
    MAX_OPEN_POSITIONS: int = 1  # Only 1 position for small accounts
    
    # Trading pairs - SOLUSD first (cheapest margin requirement)
    TRADING_SYMBOLS: List[str] = None
    
    # Minimum trade values - LOWERED FOR SMALL ACCOUNTS
    MIN_TRADE_VALUE_USD: float = 0.01  # Very low for testing
    
    # Leverage - 10x for small accounts
    DEFAULT_LEVERAGE: int = 10
    
    def __post_init__(self):
        if self.TRADING_SYMBOLS is None:
            # SOLUSD first - requires least margin (~$1.26 per contract at 10x)
            self.TRADING_SYMBOLS = ["SOLUSD", "ETHUSD", "BTCUSD"]
    
    def calculate_fee_cost(self, position_size: float, entry_price: float, exit_price: float, is_maker: bool = False) -> float:
        """Calculate total fees for a round-trip trade"""
        fee_rate = self.MAKER_FEE if is_maker else self.TAKER_FEE
        entry_fee = position_size * entry_price * fee_rate
        exit_fee = position_size * exit_price * fee_rate
        return entry_fee + exit_fee
    
    def min_move_for_profit(self, entry_price: float, is_maker: bool = False) -> float:
        """Calculate minimum price move needed for profit after fees"""
        fee_rate = self.MAKER_FEE if is_maker else self.TAKER_FEE
        # Round trip fee + minimum profit margin
        total_cost = (2 * fee_rate) + self.MIN_PROFIT_AFTER_FEES
        return entry_price * total_cost


# =============================================================================
# DELTA EXCHANGE API CLIENT
# =============================================================================

class DeltaExchangeClient:
    """Delta Exchange India API Client with full trading capabilities"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://cdn-ind.testnet.deltaex.org" if testnet else "https://api.india.delta.exchange"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _generate_signature(self, method: str, endpoint: str, payload: str = "") -> Dict[str, str]:
        """Generate authentication headers"""
        timestamp = str(int(time.time()))
        signature_data = method + timestamp + endpoint + payload
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'api-key': self.api_key,
            'signature': signature,
            'timestamp': timestamp
        }
    
    def _request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make authenticated API request"""
        url = f"{self.base_url}{endpoint}"
        payload = json.dumps(data) if data else ""
        
        headers = self._generate_signature(method, endpoint, payload)
        self.session.headers.update(headers)
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=30)
            elif method == "POST":
                response = self.session.post(url, data=payload, timeout=30)
            elif method == "DELETE":
                response = self.session.delete(url, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Log response for debugging
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_code = error_data.get('error', {}).get('code', 'unknown')
                    error_context = error_data.get('error', {}).get('context', {})
                    
                    if error_code == 'insufficient_margin':
                        available = error_context.get('available_balance', 'N/A')
                        required = error_context.get('required_additional_balance', 'N/A')
                        logger.error(f"💰 INSUFFICIENT MARGIN")
                        logger.error(f"   Available: ${available}")
                        logger.error(f"   Required Additional: ${required}")
                        logger.error(f"   ➡️ Deposit at least ${float(required) + 0.05:.2f} more to trade")
                    else:
                        logger.error(f"API Error: {error_code} - {error_context}")
                except:
                    logger.error(f"API Error Response: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}
    
    def get_wallet_balance(self) -> Dict:
        """Get account wallet balances"""
        result = self._request("GET", "/v2/wallet/balances")
        return result
    
    def get_available_balance(self) -> float:
        """Get total available balance in USD"""
        result = self.get_wallet_balance()
        if "result" in result:
            total = 0.0
            for asset in result["result"]:
                # Available balance (not in positions)
                available = float(asset.get("available_balance", 0))
                # For USDT/USD assets
                if asset.get("asset_symbol") in ["USDT", "USD", "USDC"]:
                    total += available
            return total
        return 0.0
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        # Need to specify underlying_asset_symbol for Delta Exchange India
        all_positions = []
        for asset in ["BTC", "ETH", "SOL"]:
            result = self._request("GET", f"/v2/positions?underlying_asset_symbol={asset}")
            if "result" in result:
                positions = [p for p in result["result"] if float(p.get("size", 0)) != 0]
                all_positions.extend(positions)
        return all_positions
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker for symbol"""
        try:
            response = self.session.get(
                f"{self.base_url}/v2/tickers/{symbol}",
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("result", {})
        except Exception as e:
            logger.error(f"Failed to get ticker: {e}")
        return {}
    
    def get_orderbook(self, symbol: str, depth: int = 5) -> Dict:
        """Get orderbook for symbol"""
        try:
            response = self.session.get(
                f"{self.base_url}/v2/l2orderbook/{symbol}",
                params={"depth": depth},
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("result", {})
        except Exception as e:
            logger.error(f"Failed to get orderbook: {e}")
        return {}
    
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market_order",
        limit_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        leverage: int = 5
    ) -> Dict:
        """Place an order"""
        
        # Get product ID for symbol
        product_id = self._get_product_id(symbol)
        if not product_id:
            return {"error": f"Product not found: {symbol}"}
        
        # Delta Exchange requires INTEGER size (number of contracts)
        # Minimum is 1 contract
        size_int = max(1, int(round(size)))
        
        order_data = {
            "product_id": product_id,
            "side": side,
            "size": size_int,  # Must be integer!
            "order_type": order_type
        }
        
        if order_type == "limit_order" and limit_price:
            order_data["limit_price"] = str(limit_price)
        
        # Add bracket orders for stop loss and take profit
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
        
        result = self._request("POST", "/v2/orders", order_data)
        return result
    
    def close_position(self, symbol: str) -> Dict:
        """Close entire position for symbol"""
        product_id = self._get_product_id(symbol)
        if not product_id:
            return {"error": f"Product not found: {symbol}"}
        
        result = self._request("POST", f"/v2/positions/{product_id}/close", {})
        return result
    
    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """Cancel all open orders"""
        if symbol:
            product_id = self._get_product_id(symbol)
            return self._request("DELETE", f"/v2/orders?product_id={product_id}")
        return self._request("DELETE", "/v2/orders")
    
    def _get_product_id(self, symbol: str) -> Optional[int]:
        """Get product ID from symbol"""
        try:
            response = self.session.get(f"{self.base_url}/v2/products", timeout=10)
            if response.status_code == 200:
                products = response.json().get("result", [])
                for p in products:
                    if p.get("symbol") == symbol:
                        return p.get("id")
        except Exception as e:
            logger.error(f"Failed to get product ID: {e}")
        return None
    
    def get_product_info(self, symbol: str) -> Dict:
        """Get product details"""
        try:
            response = self.session.get(f"{self.base_url}/v2/products", timeout=10)
            if response.status_code == 200:
                products = response.json().get("result", [])
                for p in products:
                    if p.get("symbol") == symbol:
                        return p
        except Exception as e:
            logger.error(f"Failed to get product info: {e}")
        return {}


# =============================================================================
# NEWS & SENTIMENT ANALYZER
# =============================================================================

class NewsSentimentAnalyzer:
    """Analyzes global news for crypto trading signals"""
    
    # Keywords that indicate bullish sentiment for crypto
    BULLISH_KEYWORDS = [
        "bitcoin etf approved", "crypto adoption", "institutional buying",
        "rate cut", "fed dovish", "inflation falling", "stimulus",
        "bitcoin halving", "ethereum upgrade", "bullish", "rally",
        "accumulation", "whale buying", "record inflows", "adoption",
        "partnership", "integration", "approval", "regulations positive"
    ]
    
    # Keywords that indicate bearish sentiment
    BEARISH_KEYWORDS = [
        "crypto ban", "regulation crackdown", "sec lawsuit", "hack",
        "exchange collapse", "rate hike", "fed hawkish", "inflation rising",
        "recession", "crash", "selloff", "liquidation", "outflows",
        "whale selling", "fraud", "investigation", "bearish", "dump"
    ]
    
    # Economic events calendar awareness
    HIGH_IMPACT_EVENTS = [
        "FOMC", "CPI", "NFP", "GDP", "unemployment", "retail sales"
    ]
    
    def __init__(self):
        self.news_cache: List[Dict] = []
        self.last_fetch = None
        self.sentiment_score = 0.0
    
    def fetch_crypto_news(self) -> List[Dict]:
        """Fetch news from multiple free sources"""
        news_items = []
        
        # CryptoCompare News (free)
        try:
            response = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                for item in data.get("Data", [])[:20]:
                    news_items.append({
                        "title": item.get("title", ""),
                        "body": item.get("body", ""),
                        "source": item.get("source", ""),
                        "published": item.get("published_on", 0),
                        "url": item.get("url", "")
                    })
        except Exception as e:
            logger.warning(f"CryptoCompare news fetch failed: {e}")
        
        # CoinGecko trending (for market sentiment)
        try:
            response = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                # Trending coins indicate market interest
                trending = [c["item"]["name"] for c in data.get("coins", [])[:5]]
                if trending:
                    news_items.append({
                        "title": f"Trending: {', '.join(trending)}",
                        "body": "High market interest in these coins",
                        "source": "CoinGecko",
                        "published": int(time.time())
                    })
        except Exception as e:
            logger.warning(f"CoinGecko trending fetch failed: {e}")
        
        self.news_cache = news_items
        self.last_fetch = datetime.now()
        return news_items
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment (-1 to 1)"""
        text_lower = text.lower()
        
        bullish_score = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_score = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_score + bearish_score
        if total == 0:
            return 0.0
        
        return (bullish_score - bearish_score) / total
    
    def get_market_sentiment(self) -> Tuple[float, str, List[str]]:
        """
        Get overall market sentiment
        
        Returns:
            Tuple of (sentiment_score, direction, key_news)
        """
        # Fetch fresh news if cache is old
        if not self.last_fetch or (datetime.now() - self.last_fetch) > timedelta(minutes=15):
            self.fetch_crypto_news()
        
        if not self.news_cache:
            return 0.0, "neutral", []
        
        # Analyze all news
        sentiments = []
        key_headlines = []
        
        for news in self.news_cache[:15]:  # Top 15 news
            title = news.get("title", "")
            body = news.get("body", "")
            
            sentiment = self.analyze_sentiment(title + " " + body)
            sentiments.append(sentiment)
            
            if abs(sentiment) > 0.3:
                key_headlines.append(f"{'📈' if sentiment > 0 else '📉'} {title[:80]}")
        
        if not sentiments:
            return 0.0, "neutral", []
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        if avg_sentiment > 0.2:
            direction = "bullish"
        elif avg_sentiment < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"
        
        self.sentiment_score = avg_sentiment
        return avg_sentiment, direction, key_headlines[:5]
    
    def should_trade(self) -> Tuple[bool, str, float]:
        """
        Determine if conditions are favorable for trading
        
        Returns:
            Tuple of (should_trade, direction, confidence)
        """
        sentiment, direction, _ = self.get_market_sentiment()
        
        # Lower threshold for more trades - 0.05 instead of 0.15
        if abs(sentiment) < 0.05:
            return False, "neutral", 0.0
        
        confidence = min(abs(sentiment) * 3, 1.0)  # Scale to 0-1 more aggressively
        trade_direction = "long" if sentiment > 0 else "short"
        
        return True, trade_direction, confidence


# =============================================================================
# AUTONOMOUS TRADING BOT
# =============================================================================

class AladdinTradingBot:
    """
    Autonomous trading bot that:
    - Monitors account balance
    - Analyzes news sentiment
    - Executes trades with fee awareness
    - Manages risk and positions
    """
    
    def __init__(self, api_key: str, api_secret: str, config: TradingConfig = None):
        self.client = DeltaExchangeClient(api_key, api_secret)
        self.config = config or TradingConfig()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        
        # State tracking
        self.running = False
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.trades_today = 0
        self.last_trade_time = None
        
        # Performance tracking
        self.total_profit = 0.0
        self.total_fees_paid = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info("🤖 Aladdin Trading Bot initialized")
    
    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary with live UPNL"""
        balance = self.client.get_available_balance()
        positions = self.client.get_positions()
        
        total_position_value = 0
        total_unrealized_pnl = 0
        
        # Contract sizes for each product
        contract_sizes = {"BTCUSD": 0.001, "ETHUSD": 0.01, "SOLUSD": 0.1}
        
        for pos in positions:
            symbol = pos.get("product_symbol", pos.get("symbol", ""))
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size == 0:
                continue
            
            # Fetch current mark price from ticker
            ticker = self.client.get_ticker(symbol)
            mark = float(ticker.get("mark_price", entry)) if ticker else entry
            
            # Calculate position value and PnL
            contract_size = contract_sizes.get(symbol, 0.01)
            position_value = abs(size) * contract_size * mark
            
            # UPNL = (mark - entry) * size * contract_size
            pnl = (mark - entry) * size * contract_size
            
            total_position_value += position_value
            total_unrealized_pnl += pnl
        
        return {
            "available_balance": balance,
            "positions": len(positions),
            "total_position_value": total_position_value,
            "unrealized_pnl": total_unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "trades_today": self.trades_today
        }
    
    def calculate_position_size(self, symbol: str, balance: float, direction: str) -> Tuple[int, float, float]:
        """
        Calculate optimal position size in CONTRACTS (not crypto amount).
        Delta Exchange uses integer contracts:
        - BTCUSD: 1 contract = 0.001 BTC
        - ETHUSD: 1 contract = 0.01 ETH  
        - SOLUSD: 1 contract = 0.1 SOL
        
        Returns:
            Tuple of (size_contracts, stop_loss, take_profit)
        """
        ticker = self.client.get_ticker(symbol)
        if not ticker:
            return 0, 0, 0
        
        current_price = float(ticker.get("mark_price", ticker.get("last_price", 0)))
        if current_price == 0:
            return 0, 0, 0
        
        # Contract values (USD value per 1 contract)
        contract_values = {
            "BTCUSD": 0.001 * current_price,  # 0.001 BTC per contract
            "ETHUSD": 0.01 * current_price,    # 0.01 ETH per contract
            "SOLUSD": 0.1 * current_price      # 0.1 SOL per contract
        }
        
        contract_value = contract_values.get(symbol, 1)
        leverage = self.config.DEFAULT_LEVERAGE
        
        # Calculate margin required per contract
        margin_per_contract = contract_value / leverage
        
        # How much margin we're willing to use
        available_margin = balance * self.config.MAX_POSITION_SIZE_PCT
        
        # Calculate number of contracts we can afford
        num_contracts = int(available_margin / margin_per_contract)
        
        # Minimum 1 contract
        num_contracts = max(1, num_contracts)
        
        # Calculate required margin for the trade
        required_margin = num_contracts * margin_per_contract
        
        logger.info(f"💰 Balance: ${balance:.4f}")
        logger.info(f"   Contract Value: ${contract_value:.2f} ({symbol})")
        logger.info(f"   Margin/Contract: ${margin_per_contract:.4f} at {leverage}x")
        logger.info(f"   Contracts: {num_contracts}")
        logger.info(f"   Required Margin: ${required_margin:.4f}")
        
        # Check if we have enough
        if required_margin > balance:
            logger.warning(f"⚠️ Insufficient margin: need ${required_margin:.4f}, have ${balance:.4f}")
            # Still return 1 contract - API will reject if not enough
        
        # Calculate stop loss and take profit
        if direction == "long":
            stop_loss = current_price * (1 - self.config.STOP_LOSS_PCT)
            take_profit = current_price * (1 + self.config.TAKE_PROFIT_PCT)
        else:
            stop_loss = current_price * (1 + self.config.STOP_LOSS_PCT)
            take_profit = current_price * (1 - self.config.TAKE_PROFIT_PCT)
        
        return num_contracts, round(stop_loss, 2), round(take_profit, 2)
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits allow trading"""
        summary = self.get_account_summary()
        
        # Check daily loss limit
        if self.daily_start_balance > 0:
            daily_loss_pct = -self.daily_pnl / self.daily_start_balance
            if daily_loss_pct > self.config.MAX_DAILY_LOSS_PCT:
                logger.warning(f"⚠️ Daily loss limit reached: {daily_loss_pct:.1%}")
                return False
        
        # Check max positions
        if summary["positions"] >= self.config.MAX_OPEN_POSITIONS:
            logger.info(f"Max positions reached: {summary['positions']}")
            return False
        
        return True
    
    def execute_trade(self, symbol: str, direction: str, confidence: float) -> bool:
        """Execute a trade based on signal"""
        
        # Get balance
        balance = self.client.get_available_balance()
        
        # For micro accounts (under $1), skip balance check and try anyway
        if balance <= 0:
            logger.error(f"Zero balance: ${balance:.4f}")
            return False
        
        logger.info(f"💵 Available balance: ${balance:.4f}")
        
        # Calculate position size (returns integer contracts)
        num_contracts, stop_loss, take_profit = self.calculate_position_size(symbol, balance, direction)
        
        if num_contracts == 0:
            return False
        
        # Get current price for logging
        ticker = self.client.get_ticker(symbol)
        current_price = float(ticker.get("mark_price", 0))
        
        # Calculate expected fees (based on contract value)
        contract_sizes = {"BTCUSD": 0.001, "ETHUSD": 0.01, "SOLUSD": 0.1}
        contract_size = contract_sizes.get(symbol, 0.01)
        position_value = num_contracts * contract_size * current_price
        expected_fee = position_value * self.config.TAKER_FEE * 2  # Round trip
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║  📊 EXECUTING TRADE                                          ║
╠══════════════════════════════════════════════════════════════╣
║  Symbol:      {symbol:<20}                       ║
║  Direction:   {'🟢 LONG' if direction == 'long' else '🔴 SHORT':<20}                       ║
║  Contracts:   {num_contracts:<20}                       ║
║  Price:       ${current_price:,.2f}                               
║  Position:    ${position_value:.4f}
║  Stop Loss:   ${stop_loss:,.2f}                               
║  Take Profit: ${take_profit:,.2f}                               
║  Est. Fees:   ${expected_fee:.6f}                               
║  Confidence:  {confidence:.1%}                                 
╚══════════════════════════════════════════════════════════════╝
""")
        
        # Place order
        side = "buy" if direction == "long" else "sell"
        result = self.client.place_order(
            symbol=symbol,
            side=side,
            size=num_contracts,  # Integer contracts!
            order_type="market_order",
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.config.DEFAULT_LEVERAGE
        )
        
        if "error" in result:
            logger.error(f"❌ Order failed: {result['error']}")
            return False
        
        logger.info(f"✅ Order placed successfully: {result.get('result', {}).get('id', 'N/A')}")
        
        self.trades_today += 1
        self.last_trade_time = datetime.now()
        self.total_fees_paid += expected_fee
        
        return True
    
    def manage_positions(self):
        """
        Monitor and manage open positions
        - Exit if sentiment reverses against position
        - Check stop loss / take profit
        """
        positions = self.client.get_positions()
        
        if not positions:
            return
        
        # Get current sentiment (use cached to avoid too many API calls)
        sentiment = self.sentiment_analyzer.sentiment_score
        
        # Contract sizes for each product
        contract_sizes = {"BTCUSD": 0.001, "ETHUSD": 0.01, "SOLUSD": 0.1}
        
        for pos in positions:
            symbol = pos.get("product_symbol", pos.get("symbol", ""))
            size = float(pos.get("size", 0))
            entry = float(pos.get("entry_price", 0))
            
            if size == 0:
                continue
            
            # FETCH LIVE MARK PRICE from ticker
            ticker = self.client.get_ticker(symbol)
            if not ticker:
                logger.warning(f"Could not fetch ticker for {symbol}")
                continue
                
            mark = float(ticker.get("mark_price", entry))
            contract_size = contract_sizes.get(symbol, 0.01)
            
            # Calculate UPNL correctly
            # UPNL = (mark_price - entry_price) * size * contract_size
            unrealized_pnl = (mark - entry) * size * contract_size
            
            # Calculate PnL percentage
            pnl_pct = (mark - entry) / entry if entry > 0 else 0
            if size < 0:  # Short position
                pnl_pct = -pnl_pct
            
            # Determine position direction
            is_long = size > 0
            position_dir = "LONG" if is_long else "SHORT"
            
            # Log position status with LIVE data
            emoji = "🟢" if unrealized_pnl > 0 else "🔴"
            logger.info(f"{emoji} {symbol}: {position_dir} Size={abs(size)} | Entry=${entry:.2f} | Mark=${mark:.2f} | P&L=${unrealized_pnl:.4f} ({pnl_pct:+.2%})")
            
            # CHECK FOR SENTIMENT REVERSAL - AUTO EXIT
            should_exit = False
            exit_reason = ""
            
            # If LONG and sentiment turns BEARISH strongly
            if is_long and sentiment < -0.15:
                should_exit = True
                exit_reason = f"Sentiment reversed to BEARISH ({sentiment:.2f})"
            
            # If SHORT and sentiment turns BULLISH strongly  
            elif not is_long and sentiment > 0.15:
                should_exit = True
                exit_reason = f"Sentiment reversed to BULLISH ({sentiment:.2f})"
            
            # Also exit on significant loss (beyond stop loss)
            if pnl_pct < -self.config.STOP_LOSS_PCT:
                should_exit = True
                exit_reason = f"Stop loss triggered ({pnl_pct:.2%})"
            
            # Take profit
            if pnl_pct > self.config.TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = f"Take profit reached ({pnl_pct:.2%})"
            
            if should_exit:
                logger.warning(f"⚠️ CLOSING POSITION: {symbol} - {exit_reason}")
                result = self.client.close_position(symbol)
                if "error" not in result:
                    logger.info(f"✅ Position closed successfully")
                    if unrealized_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    self.daily_pnl += unrealized_pnl
                else:
                    logger.error(f"❌ Failed to close: {result}")
    
    def trading_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting autonomous trading loop...")
        
        # Initialize daily tracking
        self.daily_start_balance = self.client.get_available_balance()
        self.daily_pnl = 0.0
        self.trades_today = 0
        
        last_analysis_time = None
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Analyze sentiment every 5 minutes
                if not last_analysis_time or (current_time - last_analysis_time) > timedelta(minutes=5):
                    
                    # Get account status
                    summary = self.get_account_summary()
                    logger.info(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 ACCOUNT STATUS
   Balance: ${summary['available_balance']:,.4f}
   Positions: {summary['positions']}
   Unrealized P&L: ${summary['unrealized_pnl']:,.4f}
   Trades Today: {summary['trades_today']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
                    
                    # Analyze sentiment
                    sentiment, direction, headlines = self.sentiment_analyzer.get_market_sentiment()
                    
                    logger.info(f"📰 Market Sentiment: {direction.upper()} ({sentiment:+.2f})")
                    for headline in headlines[:3]:
                        logger.info(f"   {headline}")
                    
                    # Check if we should trade
                    should_trade, trade_dir, confidence = self.sentiment_analyzer.should_trade()
                    
                    # FORCE TRADE MODE for testing with micro accounts
                    force_first_trade = (self.trades_today == 0 and summary['positions'] == 0)
                    
                    if (should_trade or force_first_trade) and self.check_risk_limits():
                        # If forcing trade, use sentiment direction or default to long
                        if not should_trade and force_first_trade:
                            logger.info("⚡ FORCE TRADE MODE: Executing first trade for testing")
                            trade_dir = direction if direction != "neutral" else "long"
                            confidence = 0.5  # Medium confidence for forced trade
                        
                        # Select best symbol based on liquidity
                        for symbol in self.config.TRADING_SYMBOLS:
                            ticker = self.client.get_ticker(symbol)
                            if ticker:
                                volume = float(ticker.get("volume", 0))
                                if volume > 0:
                                    success = self.execute_trade(symbol, trade_dir, confidence)
                                    if success:
                                        break
                    else:
                        logger.info("📊 No trade signal or risk limits reached")
                    
                    last_analysis_time = current_time
                
                # Manage existing positions
                self.manage_positions()
                
                # Sleep before next iteration
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("⏹️ Stopping bot...")
                self.running = False
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(30)
    
    def start(self):
        """Start the trading bot"""
        self.running = True
        
        # Print startup banner
        balance = self.client.get_available_balance()
        positions = self.client.get_positions()
        
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗            ║
║    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║            ║
║    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║            ║
║    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║            ║
║    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║            ║
║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝            ║
║                                                                      ║
║              🤖 AUTONOMOUS TRADING BOT v2.0                          ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
""")
        print(f"║  💰 Account Balance:  ${balance:,.4f}")
        print(f"║  📊 Open Positions:   {len(positions)}")
        print(f"║  🎯 Trading Symbols:  {', '.join(self.config.TRADING_SYMBOLS)}")
        print(f"║  ⚡ Leverage:         {self.config.DEFAULT_LEVERAGE}x")
        print(f"║  🛡️  Max Position:     {self.config.MAX_POSITION_SIZE_PCT:.0%} of balance")
        print(f"║  🛑 Stop Loss:        {self.config.STOP_LOSS_PCT:.0%}")
        print(f"║  🎯 Take Profit:      {self.config.TAKE_PROFIT_PCT:.0%}")
        print(f"║  💸 Taker Fee:        {self.config.TAKER_FEE:.3%}")
        
        # Check if balance is too low
        MIN_BALANCE_FOR_TRADING = 0.20  # ~₹17 minimum for 1 contract
        if balance < MIN_BALANCE_FOR_TRADING:
            print(f"""║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠️  WARNING: BALANCE TOO LOW FOR TRADING                            ║
║                                                                      ║
║  Current Balance:   ${balance:.4f} (₹{balance*85:.2f})
║  Minimum Required:  ${MIN_BALANCE_FOR_TRADING:.2f} (₹{MIN_BALANCE_FOR_TRADING*85:.2f})
║  Please Deposit:    ${MIN_BALANCE_FOR_TRADING - balance:.2f} (₹{(MIN_BALANCE_FOR_TRADING - balance)*85:.2f})
║                                                                      ║
║  The bot will monitor markets but cannot execute trades.             ║
╚══════════════════════════════════════════════════════════════════════╝
""")
        else:
            print("""║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
        
        # Start trading loop
        self.trading_loop()
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        logger.info("🛑 Bot stopped")
        
        # Print final summary
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  📊 FINAL SESSION SUMMARY                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Total Trades:     {self.trades_today}
║  Winning Trades:   {self.winning_trades}
║  Losing Trades:    {self.losing_trades}
║  Total Fees Paid:  ${self.total_fees_paid:.2f}
║  Daily P&L:        ${self.daily_pnl:+,.2f}
╚══════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from config.credentials import API_KEY, API_SECRET
    
    # Create trading config
    config = TradingConfig(
        MAX_POSITION_SIZE_PCT=0.90,  # 90% of balance for micro accounts
        STOP_LOSS_PCT=0.02,          # 2% stop loss
        TAKE_PROFIT_PCT=0.04,        # 4% take profit
        MAX_DAILY_LOSS_PCT=0.20,     # 20% max daily loss (aggressive for small)
        MAX_OPEN_POSITIONS=1,
        DEFAULT_LEVERAGE=20          # 20x leverage for micro accounts
    )
    
    # Initialize and start bot
    bot = AladdinTradingBot(API_KEY, API_SECRET, config)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
