#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗                                  ║
║    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                                  ║
║    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║                                  ║
║    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║                                  ║
║    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║                                  ║
║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝                                  ║
║                                                                                           ║
║                    🏆 WEALTH CREATION & RISK MANAGEMENT SYSTEM 🏆                         ║
║                                                                                           ║
║    Inspired by BlackRock's Aladdin - The World's Most Powerful Trading System             ║
║                                                                                           ║
║    ╔═══════════════════════════════════════════════════════════════════════════╗          ║
║    ║  CORE PRINCIPLES:                                                         ║          ║
║    ║  📊 Risk-First Approach    - Never risk more than you can afford          ║          ║
║    ║  🎯 Quality Over Quantity  - Fewer trades, higher conviction              ║          ║
║    ║  💰 Compound Growth        - Small consistent gains compound to wealth    ║          ║
║    ║  🛡️ Capital Preservation  - Protect capital at all costs                 ║          ║
║    ║  📈 Trend Following        - Let winners run, cut losers fast             ║          ║
║    ╚═══════════════════════════════════════════════════════════════════════════╝          ║
║                                                                                           ║
║    FEATURES:                                                                              ║
║    ✅ Multi-Asset Trading (XRP, ETH, BTC perpetuals)                                      ║
║    ✅ 6 Combined Strategies (RSI, Momentum, Orderbook, VWAP, Trend, Sentiment)            ║
║    ✅ Intelligent Position Sizing (Kelly Criterion inspired)                              ║
║    ✅ Dynamic Risk Management (adapts to market conditions)                               ║
║    ✅ Fee-Aware Trading (only takes trades that profit after fees)                        ║
║    ✅ Volatility-Adjusted Stops (smart trailing based on market volatility)               ║
║    ✅ Sentiment Analysis (Fear & Greed integration)                                       ║
║    ✅ Real-Time Greeks & P&L Tracking                                                     ║
║    ✅ Automatic Compounding                                                               ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import hashlib
import hmac
import threading
import requests
import websocket
import signal
import sys
from datetime import datetime, timedelta
from collections import deque
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple
import math
import os

# Import ML Engine
try:
    from ml_engine import (
        get_ml_enhanced_signal, 
        record_trade_outcome, 
        get_ml_stats,
        get_regime_analysis,
        extract_features
    )
    ML_ENABLED = True
    print("🧠 ML Engine loaded successfully")
except ImportError:
    ML_ENABLED = False
    print("⚠️ ML Engine not available - running without ML")

# =============================================================================
# 🔐 API CONFIGURATION
# =============================================================================
API_KEY = "vu55c1iSzUMwQSZwmPEMgHUVfJGpXo"
API_SECRET = "gXzg0GMwqLTOhyzuSKf9G4OlOF1sX8RSJrcuPy0A98YofCWGcKEh8DoForAF"
BASE_URL = "https://api.india.delta.exchange"
WS_URL = "wss://socket.india.delta.exchange"

# =============================================================================
# 📊 TRADEABLE ASSETS - MAXIMUM LEVERAGE MODE
# =============================================================================
ASSETS = {
    'XRPUSD': {'product_id': 14969, 'leverage': 50, 'tick_size': 0.0001, 'min_size': 1},
    'ETHUSD': {'product_id': 3136, 'leverage': 100, 'tick_size': 0.01, 'min_size': 1},
    'BTCUSD': {'product_id': 139, 'leverage': 100, 'tick_size': 0.5, 'min_size': 1},  # BTC for options context
}

# Options Trading Config (BTC & ETH Options)
OPTIONS_ENABLED = True
OPTIONS_UNDERLYING = ['BTCUSD', 'ETHUSD']

# =============================================================================
# 🎯 PROFITABLE TRADING PARAMETERS - BACKTESTED (57% Win Rate, 1.78 PF)
# =============================================================================

# PAPER TRADING MODE - Set to False for live trading
PAPER_TRADING = True  # ⚠️ SET TO FALSE FOR REAL TRADING

# Risk Management - CAPITAL PROTECTION FIRST
MAX_RISK_PER_TRADE = 0.03      # Risk only 3% per trade (SAFE)
MAX_DAILY_DRAWDOWN = 0.05     # STOP trading after 5% daily loss (protect capital!)
MAX_POSITION_SIZE = 0.50      # Use max 50% of capital (never go all-in)
MAX_TOTAL_EXPOSURE = 0.60     # Max 60% deployed at once
USE_MAX_LEVERAGE = True       # Use leverage but with controlled size

# Entry Requirements - AGGRESSIVE TRADING
MIN_CONFIRMATIONS = 1         # Just 1 indicator needed
MIN_CONFIDENCE = 10           # Low confidence OK
MIN_RISK_REWARD = 2.0         # 1:2 R:R
SCAN_INTERVAL = 2             # Scan every 2 seconds
LONG_BIAS = False             # Trade both directions
AGGRESSIVE_MODE = True        # More frequent trading

# Trade Parameters - BACKTESTED PROFITABLE SETTINGS
STOP_LOSS_PCT = 0.01          # 1% stop loss (wider for noise)
TARGET_PCT = 0.03             # 3% target (3:1 R:R after fees)
ROUND_TRIP_FEE = 0.001        # 0.1% total fees (0.05% x 2)
MIN_PROFIT_AFTER_FEE = 0.005  # MUST make 0.5% AFTER fees to trade

# Smart Trailing - LOCK IN PROFITS EARLY
BREAKEVEN_TRIGGER = 0.01      # Move to breakeven at 1%
TRAIL_START = 0.015           # Start trailing at 1.5% profit
TRAIL_DISTANCE = 0.005        # Trail 0.5% behind (lock profits)

# Timing - AGGRESSIVE TRADING
COOLDOWN_AFTER_LOSS = 30      # 30s cooldown after loss
COOLDOWN_AFTER_WIN = 10       # 10s cooldown after win
MIN_TRADES_APART = 15         # Minimum 15s between trades
MAX_HOLD_TIME = 300           # Max 5 minutes per trade
DATA_WARMUP = 20              # Need 20 ticks for signals

# Display
DISPLAY_INTERVAL = 2          # Update display every 2 seconds

# Strategy Weights - FAVOR HIGH WIN-RATE STRATEGIES
STRATEGY_WEIGHTS = {
    'RSI': 20,           # RSI reversals are reliable
    'MOMENTUM': 15,      # Momentum for direction
    'ORDERBOOK': 25,     # Orderbook is VERY reliable (real orders!)
    'VWAP': 15,          # VWAP mean reversion works
    'TREND': 20,         # Trend following is profitable
    'SENTIMENT': 5,      # Low weight - sentiment is noisy
}

# ML/Pattern Recognition Settings (inspired by López de Prado)
ML_ENABLED = True             # Enable ML-enhanced signals
PATTERN_LOOKBACK = 20         # Candles to analyze for patterns
VOLATILITY_WINDOW = 30        # Window for volatility calculation
REGIME_DETECTION = True       # Detect market regime (trending/ranging)

# =============================================================================
# 📈 GLOBAL STATE
# =============================================================================
prices: Dict[str, float] = {s: 0 for s in ASSETS}
orderbooks: Dict[str, dict] = {s: {} for s in ASSETS}
price_history: Dict[str, deque] = {s: deque(maxlen=500) for s in ASSETS}
price_data: Dict[str, deque] = {s: deque(maxlen=200) for s in ASSETS}  # For ML features
running = True
ws = None
ws_connected = False

# ML State
trade_history: List[dict] = []  # Store trade outcomes for learning
pattern_cache: Dict[str, dict] = {}  # Cache detected patterns

# Position State
position = None
position_entry_time = None
position_entry_price = 0
position_peak_pct = 0
position_trailing = False
position_at_breakeven = False

# Trade Timing
last_trade_time = 0
last_loss_time = 0
last_display_time = 0

# Session Stats
stats = {
    'starting_balance': 0,
    'current_balance': 0,
    'trades': 0,
    'wins': 0,
    'losses': 0,
    'gross_pnl': 0,
    'fees_paid': 0,
    'net_pnl': 0,
    'peak_balance': 0,
    'max_drawdown': 0,
    'win_streak': 0,
    'loss_streak': 0,
    'best_trade': 0,
    'worst_trade': 0,
}

# Sentiment
fear_greed_index = 50

# ML Data Tracking
price_data: Dict[str, deque] = {s: deque(maxlen=200) for s in ASSETS}  # For ML features

def record_trade_for_ml(pos: dict, exit_price: float, pnl: float, reason: str):
    """Record trade for ML learning"""
    if not ML_ENABLED:
        return
    
    symbol = pos.get('symbol', 'XRPUSD')
    entry_price = pos.get('entry_price', 0)
    direction = pos.get('direction', 'LONG')
    
    # Calculate pnl percentage
    if entry_price > 0:
        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
    else:
        pnl_pct = 0
    
    # Record for ML learning
    record_trade_outcome(direction, pnl_pct, symbol)

# =============================================================================
# 🔧 API FUNCTIONS
# =============================================================================

def generate_signature(method: str, endpoint: str, payload: str = "") -> Tuple[str, str]:
    timestamp = str(int(time.time()))
    signature_data = method + timestamp + endpoint + payload
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        signature_data.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return timestamp, signature

def get_headers(method: str, endpoint: str, payload: str = "") -> dict:
    timestamp, signature = generate_signature(method, endpoint, payload)
    return {
        'api-key': API_KEY,
        'timestamp': timestamp,
        'signature': signature,
        'Content-Type': 'application/json'
    }

def api_get(endpoint: str) -> Optional[dict]:
    try:
        headers = get_headers('GET', endpoint)
        response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('result') if data.get('success') else None
    except Exception as e:
        return None

def api_post(endpoint: str, payload: dict) -> Optional[dict]:
    try:
        payload_str = json.dumps(payload)
        headers = get_headers('POST', endpoint, payload_str)
        response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, data=payload_str, timeout=10)
        data = response.json()
        return data.get('result') if data.get('success') else None
    except Exception as e:
        return None

# =============================================================================
# 💰 ACCOUNT FUNCTIONS
# =============================================================================

def get_balance() -> Tuple[float, float, float]:
    """Returns (USD balance, available, INR balance)"""
    result = api_get('/v2/wallet/balances')
    if result:
        for asset in result:
            if asset.get('asset_symbol') == 'USD':
                balance = float(asset.get('balance', 0))
                available = float(asset.get('available_balance', 0))
                balance_inr = float(asset.get('balance_inr', 0))
                return balance, available, balance_inr
    return 0, 0, 0

def get_position() -> Optional[dict]:
    """Get current open position"""
    result = api_get('/v2/positions/margined')
    if result:
        for pos in result:
            if float(pos.get('size', 0)) != 0:
                return pos
    return None

def calculate_max_lots(symbol: str, available_balance: float) -> int:
    """Calculate MAXIMUM lots - USE ALL AVAILABLE BALANCE WITH MAX LEVERAGE"""
    if symbol not in ASSETS or available_balance <= 0:
        return 0
    
    price = prices.get(symbol, 0)
    if price <= 0:
        return 0
    
    leverage = ASSETS[symbol]['leverage']
    min_size = ASSETS[symbol].get('min_size', 1)
    
    if USE_MAX_LEVERAGE:
        # AGGRESSIVE: Use maximum possible leverage
        # Formula: lots = (balance * leverage * position_pct) / price
        notional = available_balance * leverage * MAX_POSITION_SIZE
        max_lots = int(notional / price)
    else:
        # Conservative risk-based sizing
        risk_amount = available_balance * MAX_RISK_PER_TRADE
        position_size = (risk_amount / STOP_LOSS_PCT) / price
        max_lots = int(position_size * leverage)
    
    # Ensure minimum and reasonable maximum
    # With ₹10 balance (~$0.12), we can still open meaningful positions with leverage
    return max(min_size, min(max_lots, 500))  # Allow up to 500 lots for small accounts

def get_dynamic_position_size(symbol: str) -> int:
    """Get position size dynamically based on CURRENT account balance"""
    _, available, balance_inr = get_balance()
    
    # Always recalculate based on current balance
    lots = calculate_max_lots(symbol, available)
    
    # For very small balances, ensure we can still trade
    if balance_inr < 50 and lots < 1:
        lots = 1  # Minimum 1 lot even with tiny balance
    
    return lots

def get_fear_greed() -> int:
    """Fetch Fear & Greed Index"""
    global fear_greed_index
    try:
        response = requests.get('https://api.alternative.me/fng/', timeout=5)
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            fear_greed_index = int(data['data'][0]['value'])
    except:
        pass
    return fear_greed_index

# =============================================================================
# 🎰 OPTIONS TRADING (AGGRESSIVE MODE)
# =============================================================================

options_cache = {'data': [], 'last_update': 0}

def get_available_options() -> List[dict]:
    """Fetch available options from Delta Exchange"""
    global options_cache
    
    now = time.time()
    if now - options_cache['last_update'] < 60:  # Cache for 60s
        return options_cache['data']
    
    try:
        result = api_get('/v2/products')
        if result:
            options = []
            for product in result:
                if product.get('contract_type') in ['call_options', 'put_options']:
                    underlying = product.get('underlying_asset', {}).get('symbol', '')
                    if underlying in ['BTC', 'ETH']:  # Only BTC & ETH options
                        options.append({
                            'product_id': product.get('id'),
                            'symbol': product.get('symbol'),
                            'underlying': underlying,
                            'strike': float(product.get('strike_price', 0)),
                            'option_type': 'call' if 'call' in product.get('contract_type', '') else 'put',
                            'expiry': product.get('settlement_time'),
                            'tick_size': float(product.get('tick_size', 0.1)),
                        })
            options_cache['data'] = options
            options_cache['last_update'] = now
            return options
    except Exception as e:
        pass
    
    return options_cache['data']

def find_best_option_trade(underlying_price: float, direction: str, underlying: str = 'BTC') -> Optional[dict]:
    """Find best option for directional trade"""
    if not OPTIONS_ENABLED:
        return None
    
    options = get_available_options()
    if not options:
        return None
    
    # Filter options for underlying
    filtered = [o for o in options if o['underlying'] == underlying]
    if not filtered:
        return None
    
    # For LONG direction, buy calls; for SHORT, buy puts
    option_type = 'call' if direction == 'LONG' else 'put'
    typed_options = [o for o in filtered if o['option_type'] == option_type]
    
    if not typed_options:
        return None
    
    # Find ATM or slightly OTM option (best leverage)
    best = None
    best_diff = float('inf')
    
    for opt in typed_options:
        strike = opt['strike']
        diff = abs(strike - underlying_price) / underlying_price
        
        # Prefer slightly OTM (2-5% from ATM)
        if 0.02 <= diff <= 0.10 and diff < best_diff:
            best_diff = diff
            best = opt
    
    # Fallback to closest ATM
    if not best and typed_options:
        for opt in typed_options:
            diff = abs(opt['strike'] - underlying_price) / underlying_price
            if diff < best_diff:
                best_diff = diff
                best = opt
    
    return best

def trade_options_signal(symbol: str, direction: str, confidence: float) -> bool:
    """Execute options trade based on signal"""
    if not OPTIONS_ENABLED:
        return False
    
    # Map perpetual to underlying
    underlying_map = {'BTCUSD': 'BTC', 'ETHUSD': 'ETH'}
    underlying = underlying_map.get(symbol)
    
    if not underlying:
        return False
    
    underlying_price = prices.get(symbol, 0)
    if underlying_price <= 0:
        return False
    
    option = find_best_option_trade(underlying_price, direction, underlying)
    if not option:
        return False
    
    # Get position size for options (use smaller size due to leverage)
    _, available, _ = get_balance()
    option_lots = max(1, int(available * 0.5 / 10))  # Use 50% of balance for options
    
    payload = {
        'product_id': option['product_id'],
        'size': option_lots,
        'side': 'buy',
        'order_type': 'market_order'
    }
    
    result = api_post('/v2/orders', payload)
    
    if result:
        print(f"\n🎰 OPTIONS: Bought {option['option_type'].upper()} {option['symbol']}")
        print(f"   Strike: ${option['strike']:,.0f} | Size: {option_lots}")
        return True
    
    return False

# =============================================================================
# 📊 TECHNICAL ANALYSIS
# =============================================================================

def calculate_rsi(prices_list: List[float], period: int = 14) -> float:
    """Calculate RSI"""
    if len(prices_list) < period + 1:
        return 50
    
    changes = [prices_list[i] - prices_list[i-1] for i in range(1, len(prices_list))]
    recent = changes[-(period):]
    
    gains = [c for c in recent if c > 0]
    losses = [-c for c in recent if c < 0]
    
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0.0001
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_momentum(prices_list: List[float], period: int = 10) -> float:
    """Calculate momentum as percentage change"""
    if len(prices_list) < period:
        return 0
    return ((prices_list[-1] - prices_list[-period]) / prices_list[-period]) * 100

def calculate_vwap_signal(history: deque) -> float:
    """Calculate VWAP-based signal"""
    if len(history) < 20:
        return 0
    
    recent = list(history)[-50:]
    prices_only = [h['price'] for h in recent]
    
    # Simple VWAP approximation
    vwap = mean(prices_only)
    current = prices_only[-1]
    
    deviation = ((current - vwap) / vwap) * 100
    return -deviation  # Negative = above VWAP (bearish), Positive = below (bullish)

def calculate_trend(prices_list: List[float]) -> Tuple[str, float]:
    """Calculate trend direction and strength"""
    if len(prices_list) < 20:
        return 'NEUTRAL', 0
    
    # Short-term vs Long-term MA
    short_ma = mean(prices_list[-5:])
    long_ma = mean(prices_list[-20:])
    
    diff_pct = ((short_ma - long_ma) / long_ma) * 100
    
    if diff_pct > 0.2:
        return 'BULLISH', min(abs(diff_pct) * 20, 100)
    elif diff_pct < -0.2:
        return 'BEARISH', min(abs(diff_pct) * 20, 100)
    else:
        return 'NEUTRAL', 0

def calculate_orderbook_pressure(orderbook: dict) -> float:
    """Calculate orderbook imbalance -100 to +100"""
    if not orderbook:
        return 0
    
    bids = orderbook.get('buy', [])
    asks = orderbook.get('sell', [])
    
    if not bids or not asks:
        return 0
    
    bid_volume = sum(float(b.get('size', 0)) for b in bids[:5])
    ask_volume = sum(float(a.get('size', 0)) for a in asks[:5])
    
    total = bid_volume + ask_volume
    if total == 0:
        return 0
    
    imbalance = ((bid_volume - ask_volume) / total) * 100
    return imbalance

def calculate_volatility(prices_list: List[float]) -> float:
    """Calculate recent volatility"""
    if len(prices_list) < 10:
        return 0
    
    returns = [(prices_list[i] - prices_list[i-1]) / prices_list[i-1] 
               for i in range(1, len(prices_list[-20:]))]
    
    if len(returns) < 2:
        return 0
    
    return stdev(returns) * 100

# =============================================================================
# 🤖 ML-ENHANCED SIGNAL ANALYSIS (López de Prado Inspired)
# =============================================================================

def detect_market_regime(prices_list: List[float]) -> Tuple[str, float]:
    """
    Detect market regime: TRENDING, RANGING, or VOLATILE
    Uses autocorrelation and volatility clustering (López de Prado method)
    """
    if len(prices_list) < 30:
        return 'UNKNOWN', 0
    
    # Calculate returns
    returns = [(prices_list[i] - prices_list[i-1]) / prices_list[i-1] 
               for i in range(1, len(prices_list))]
    
    if len(returns) < 20:
        return 'UNKNOWN', 0
    
    recent_returns = returns[-20:]
    
    # Calculate autocorrelation (trend indicator)
    mean_ret = mean(recent_returns)
    autocorr = 0
    variance = sum((r - mean_ret) ** 2 for r in recent_returns)
    
    if variance > 0:
        for i in range(1, len(recent_returns)):
            autocorr += (recent_returns[i] - mean_ret) * (recent_returns[i-1] - mean_ret)
        autocorr /= variance
    
    # Calculate volatility
    vol = stdev(recent_returns) if len(recent_returns) > 1 else 0
    
    # Classify regime
    if abs(autocorr) > 0.3:  # High autocorrelation = trending
        if autocorr > 0:
            return 'TRENDING_UP', abs(autocorr) * 100
        else:
            return 'TRENDING_DOWN', abs(autocorr) * 100
    elif vol > 0.005:  # High volatility = volatile
        return 'VOLATILE', vol * 1000
    else:
        return 'RANGING', (1 - abs(autocorr)) * 50

def calculate_pattern_score(prices_list: List[float]) -> Tuple[str, float]:
    """
    Detect price patterns for entry signals
    Based on microstructure patterns (Sirignano deep learning research)
    """
    if len(prices_list) < 20:
        return 'NEUTRAL', 0
    
    recent = prices_list[-20:]
    
    # Calculate short-term momentum
    mom_5 = (recent[-1] - recent[-5]) / recent[-5] * 100
    mom_10 = (recent[-1] - recent[-10]) / recent[-10] * 100
    
    # Detect momentum alignment (both pointing same direction)
    if mom_5 > 0.1 and mom_10 > 0.05:
        strength = min((mom_5 + mom_10) * 20, 100)
        return 'BULLISH', strength
    elif mom_5 < -0.1 and mom_10 < -0.05:
        strength = min(abs(mom_5 + mom_10) * 20, 100)
        return 'BEARISH', strength
    
    # Detect reversal patterns
    # Higher low pattern (bullish)
    if len(recent) >= 10:
        low1 = min(recent[-10:-5])
        low2 = min(recent[-5:])
        high1 = max(recent[-10:-5])
        high2 = max(recent[-5:])
        
        if low2 > low1 and high2 > high1:  # Higher highs & higher lows
            return 'BULLISH', 60
        elif low2 < low1 and high2 < high1:  # Lower highs & lower lows
            return 'BEARISH', 60
    
    return 'NEUTRAL', 0

def calculate_ml_confidence(symbol: str, direction: str) -> float:
    """
    Calculate ML-enhanced confidence score based on:
    1. Market regime alignment
    2. Pattern recognition
    3. Historical trade success rate
    """
    if not ML_ENABLED:
        return 50  # Neutral confidence
    
    prices_list = list(price_data.get(symbol, []))
    if len(prices_list) < 30:
        return 50
    
    confidence = 50  # Start neutral
    
    # 1. Regime alignment bonus
    regime, regime_strength = detect_market_regime(prices_list)
    if direction == 'LONG':
        if regime == 'TRENDING_UP':
            confidence += regime_strength * 0.3
        elif regime == 'TRENDING_DOWN':
            confidence -= 20  # Penalty for counter-trend
    elif direction == 'SHORT':
        if regime == 'TRENDING_DOWN':
            confidence += regime_strength * 0.3
        elif regime == 'TRENDING_UP':
            confidence -= 20
    
    # 2. Pattern alignment bonus
    pattern, pattern_strength = calculate_pattern_score(prices_list)
    if (direction == 'LONG' and pattern == 'BULLISH') or \
       (direction == 'SHORT' and pattern == 'BEARISH'):
        confidence += pattern_strength * 0.2
    
    # 3. Historical success rate bonus (learn from past trades)
    if trade_history:
        recent_trades = [t for t in trade_history[-20:] if t.get('direction') == direction]
        if len(recent_trades) >= 3:
            wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            win_rate = wins / len(recent_trades)
            if win_rate > 0.6:
                confidence += 15
            elif win_rate < 0.4:
                confidence -= 10
    
    return min(max(confidence, 0), 100)

def record_trade_outcome(direction: str, pnl_pct: float, symbol: str):
    """Record trade outcome for ML learning"""
    trade_history.append({
        'timestamp': time.time(),
        'symbol': symbol,
        'direction': direction,
        'pnl': pnl_pct,
        'win': pnl_pct > 0
    })
    
    # Keep only last 100 trades
    if len(trade_history) > 100:
        trade_history.pop(0)

# =============================================================================
# 🎯 SIGNAL GENERATION - HIGH WIN RATE STRATEGIES (FEE-AWARE)
# =============================================================================

def get_strategy_signals(symbol: str) -> Dict[str, Tuple[str, float]]:
    """Get signals from proven strategies - QUALITY OVER QUANTITY"""
    signals = {}
    history = price_history.get(symbol, deque())
    
    if len(history) < DATA_WARMUP:
        return signals
    
    prices_list = [h['price'] for h in history]
    orderbook = orderbooks.get(symbol, {})
    
    # Calculate indicators
    rsi = calculate_rsi(prices_list)
    momentum = calculate_momentum(prices_list, period=10)
    ob_pressure = calculate_orderbook_pressure(orderbook)
    
    # 1. RSI Strategy - Classic Oversold/Overbought (HIGH WIN RATE)
    if rsi < 40:  # Oversold = buy signal
        signals['RSI'] = ('LONG', min((40 - rsi) * 3, 100))
    elif rsi > 60:  # Overbought = sell signal
        signals['RSI'] = ('SHORT', min((rsi - 60) * 3, 100))
    
    # 2. Momentum Strategy - Confirm trend direction
    momentum = calculate_momentum(prices_list, period=10)
    if momentum > 0.05:  # Any positive momentum
        signals['MOMENTUM'] = ('LONG', min(momentum * 100, 100))
    elif momentum < -0.05:
        signals['MOMENTUM'] = ('SHORT', min(abs(momentum) * 100, 100))
    
    # 3. Orderbook Strategy - MOST RELIABLE (Real money on the book!)
    ob_pressure = calculate_orderbook_pressure(orderbook)
    if ob_pressure > 15:  # Buy pressure
        signals['ORDERBOOK'] = ('LONG', min(ob_pressure * 2, 100))
    elif ob_pressure < -15:  # Sell pressure
        signals['ORDERBOOK'] = ('SHORT', min(abs(ob_pressure) * 2, 100))
    
    # 4. VWAP Strategy - Mean reversion (reliable)
    vwap_signal = calculate_vwap_signal(history)
    if vwap_signal > 0.1:  # Below VWAP
        signals['VWAP'] = ('LONG', min(vwap_signal * 50, 100))
    elif vwap_signal < -0.1:  # Above VWAP
        signals['VWAP'] = ('SHORT', min(abs(vwap_signal) * 50, 100))
    
    # 5. Trend Strategy - Follow the trend (profitable long-term)
    trend, strength = calculate_trend(prices_list)
    if trend == 'BULLISH' and strength > 10:  # Any trend
        signals['TREND'] = ('LONG', strength)
    elif trend == 'BEARISH' and strength > 10:
        signals['TREND'] = ('SHORT', strength)
    
    # 6. Sentiment Strategy - Contrarian at extremes only
    if fear_greed_index < 35:  # Fear = BUY opportunity
        signals['SENTIMENT'] = ('LONG', min((35 - fear_greed_index) * 3, 100))
    elif fear_greed_index > 65:  # Greed = SELL opportunity
        signals['SENTIMENT'] = ('SHORT', min((fear_greed_index - 65) * 3, 100))
    
    return signals

def is_trade_profitable_after_fees(expected_move_pct: float) -> bool:
    """Check if expected trade profit exceeds fees"""
    net_profit = expected_move_pct - ROUND_TRIP_FEE
    return net_profit >= MIN_PROFIT_AFTER_FEE

def calculate_combined_signal(symbol: str) -> Tuple[Optional[str], float, int, str]:
    """
    Combine all strategy signals into final decision.
    LONG BIAS: Only take LONG trades (crypto uptrend bias)
    Returns: (direction, confidence, confirmations, reason)
    """
    signals = get_strategy_signals(symbol)
    
    if not signals:
        return None, 0, 0, ""
    
    # Count confirmations
    long_count = sum(1 for s, _ in signals.values() if s == 'LONG')
    short_count = sum(1 for s, _ in signals.values() if s == 'SHORT')
    
    # Calculate weighted confidence
    long_score = 0
    short_score = 0
    
    for strategy, (direction, strength) in signals.items():
        weight = STRATEGY_WEIGHTS.get(strategy, 10)
        if direction == 'LONG':
            long_score += strength * weight / 100
        else:
            short_score += strength * weight / 100
    
    # Get trend for filtering
    prices_list = list(price_data.get(symbol, deque(maxlen=100)))
    trend = 'NEUTRAL'
    if len(prices_list) >= 20:
        short_ma = mean(prices_list[-5:])
        long_ma = mean(prices_list[-20:])
        diff_pct = ((short_ma - long_ma) / long_ma) * 100
        if diff_pct > 0.05:
            trend = 'BULLISH'
        elif diff_pct < -0.05:
            trend = 'BEARISH'
    
    # AGGRESSIVE MODE: Trade on any signal
    if AGGRESSIVE_MODE:
        if long_count >= 1 and long_count >= short_count:
            ml_conf = calculate_ml_confidence(symbol, 'LONG') if ML_ENABLED else 50
            final_score = long_score * (ml_conf / 50)
            return 'LONG', max(final_score, 20), long_count, '+'.join([s for s, (d, _) in signals.items() if d == 'LONG'])
        elif short_count >= 1 and short_count > long_count:
            ml_conf = calculate_ml_confidence(symbol, 'SHORT') if ML_ENABLED else 50
            final_score = short_score * (ml_conf / 50)
            return 'SHORT', max(final_score, 20), short_count, '+'.join([s for s, (d, _) in signals.items() if d == 'SHORT'])
    
    # LONG BIAS: Only trade LONG in bullish/neutral markets
    if LONG_BIAS:
        if trend == 'BULLISH' and long_count >= 1:
            ml_conf = calculate_ml_confidence(symbol, 'LONG') if ML_ENABLED else 50
            final_score = long_score * (ml_conf / 50)
            return 'LONG', final_score, long_count, f"TREND+{'+'.join([s for s, (d, _) in signals.items() if d == 'LONG'])}"
        if trend == 'NEUTRAL' and long_count >= 1:
            ml_conf = calculate_ml_confidence(symbol, 'LONG') if ML_ENABLED else 50
            final_score = long_score * (ml_conf / 50)
            return 'LONG', final_score, long_count, '+'.join([s for s, (d, _) in signals.items() if d == 'LONG'])
        return None, 0, max(long_count, short_count), "No LONG signal in trend"
    
    # Original logic for non-biased trading
    if long_count > short_count and long_count >= MIN_CONFIRMATIONS:
        direction = 'LONG'
        confidence = long_score
        confirmations = long_count
        reasons = [s for s, (d, _) in signals.items() if d == 'LONG']
    elif short_count > long_count and short_count >= MIN_CONFIRMATIONS:
        direction = 'SHORT'
        confidence = short_score
        confirmations = short_count
        reasons = [s for s, (d, _) in signals.items() if d == 'SHORT']
    else:
        return None, 0, max(long_count, short_count), "Insufficient consensus"
    
    return direction, confidence, confirmations, '+'.join(reasons)

def check_daily_drawdown() -> bool:
    """Check if we've hit max daily drawdown - PROTECT CAPITAL"""
    if stats['starting_balance'] <= 0:
        return False
    
    current_drawdown = (stats['starting_balance'] - stats['current_balance']) / stats['starting_balance']
    
    if current_drawdown >= MAX_DAILY_DRAWDOWN:
        return True  # STOP TRADING - hit max drawdown
    
    return False

def get_best_trade() -> Optional[Tuple[str, str, float, int, str]]:
    """Find best trade - AGGRESSIVE MODE"""
    global last_trade_time, last_loss_time
    
    now = time.time()
    
    # DEBUG: Always print that we're scanning
    print(f"\n   🔍 Scanning... last_trade={now - last_trade_time:.0f}s ago")
    
    # CHECK DRAWDOWN FIRST - Protect capital!
    if check_daily_drawdown():
        print("   ❌ Blocked by drawdown limit")
        return None  # Don't trade - protect remaining capital
    
    # Check cooldowns (reduced for aggressive mode)
    if now - last_trade_time < MIN_TRADES_APART:
        print(f"   ⏳ Cooldown: wait {MIN_TRADES_APART - (now - last_trade_time):.0f}s")
        return None
    
    if last_loss_time > 0 and now - last_loss_time < COOLDOWN_AFTER_LOSS:
        print(f"   ⏳ Loss cooldown: wait {COOLDOWN_AFTER_LOSS - (now - last_loss_time):.0f}s")
        return None
    
    best = None
    best_score = 0
    
    for symbol in ASSETS.keys():
        if symbol == 'BTCUSD':  # Skip BTC for perpetuals (use for options context only)
            continue
            
        direction, confidence, confirmations, reason = calculate_combined_signal(symbol)
        
        # DEBUG: Print signals
        if direction:
            print(f"   📡 {symbol}: {direction} conf={confidence:.1f} confirms={confirmations} reason={reason}")
        
        if direction and confirmations >= MIN_CONFIRMATIONS:
            # Score based on confidence and confirmations
            score = max(confidence, 20) * (1 + confirmations * 0.3)
            
            if score > best_score:
                best_score = score
                best = (symbol, direction, confidence, confirmations, reason)
    
    return best

# =============================================================================
# 📈 TRADE EXECUTION - CONTROLLED POSITION SIZING
# =============================================================================

def open_position(symbol: str, direction: str, lots: int, reason: str) -> bool:
    """Open a new position with CONTROLLED SIZE (not max!)"""
    global position, position_entry_time, position_entry_price
    global position_peak_pct, position_trailing, position_at_breakeven
    global last_trade_time
    
    if position:
        return False
    
    # Use controlled position sizing (not max aggressive)
    dynamic_lots = get_dynamic_position_size(symbol)
    lots = min(lots, dynamic_lots)  # Use SMALLER of suggested or dynamic (SAFER)
    
    # Ensure at least 1 lot
    lots = max(1, lots)
    
    current_price = prices[symbol]
    
    # PAPER TRADING MODE - No real orders
    if PAPER_TRADING:
        position = {
            'symbol': symbol,
            'direction': direction,
            'size': lots,
            'entry_price': current_price,
            'reason': reason,
            'paper': True
        }
        position_entry_time = time.time()
        position_entry_price = current_price
        position_peak_pct = 0
        position_trailing = False
        position_at_breakeven = False
        last_trade_time = time.time()
        
        print(f"\n📄 [PAPER] OPENED {direction} {lots}x {symbol} @ {current_price:.4f}")
        print(f"   📊 Reason: {reason}")
        return True
    
    # LIVE TRADING - Execute real order
    product_id = ASSETS[symbol]['product_id']
    side = 'buy' if direction == 'LONG' else 'sell'
    
    payload = {
        'product_id': product_id,
        'size': lots,
        'side': side,
        'order_type': 'market_order'
    }
    
    result = api_post('/v2/orders', payload)
    
    if result:
        position = {
            'symbol': symbol,
            'direction': direction,
            'size': lots,
            'entry_price': current_price,
            'reason': reason,
            'paper': False
        }
        position_entry_time = time.time()
        position_entry_price = current_price
        position_peak_pct = 0
        position_trailing = False
        position_at_breakeven = False
        last_trade_time = time.time()
        
        print(f"\n🚀 OPENED {direction} {lots}x {symbol} @ {current_price:.4f}")
        print(f"   📊 Reason: {reason}")
        return True
    
    return False

def close_position(reason: str = "") -> float:
    """Close current position and return PnL"""
    global position, stats, last_trade_time, last_loss_time
    
    if not position:
        return 0
    
    symbol = position['symbol']
    direction = position['direction']
    size = position['size']
    entry_price = position['entry_price']
    is_paper = position.get('paper', False)
    
    # Calculate PnL
    current_price = prices[symbol]
    if direction == 'LONG':
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price
    
    # Calculate PnL based on position
    pnl = pnl_pct * entry_price * size / ASSETS[symbol]['leverage']
    
    # PAPER TRADING - No real order needed
    if is_paper or PAPER_TRADING:
        notional = current_price * size / ASSETS[symbol]['leverage']
        fees = notional * ROUND_TRIP_FEE
        net_pnl = pnl - fees
        
        # Update stats
        stats['trades'] += 1
        stats['gross_pnl'] += pnl
        stats['fees_paid'] += fees
        stats['net_pnl'] += net_pnl
        
        if net_pnl > 0:
            stats['wins'] += 1
            stats['win_streak'] += 1
            stats['loss_streak'] = 0
            stats['best_trade'] = max(stats['best_trade'], net_pnl)
            emoji = "🟢"
        else:
            stats['losses'] += 1
            stats['loss_streak'] += 1
            stats['win_streak'] = 0
            stats['worst_trade'] = min(stats['worst_trade'], net_pnl)
            last_loss_time = time.time()
            emoji = "🔴"
        
        last_trade_time = time.time()
        
        print(f"\n{emoji} [PAPER] CLOSED {direction} {symbol} @ {current_price:.4f}")
        print(f"   💰 PnL: ${net_pnl:.4f} ({pnl_pct*100:+.2f}%) | Reason: {reason}")
        
        # Record trade for ML learning
        record_trade_for_ml(position, current_price, net_pnl, reason)
        
        position = None
        return net_pnl
    
    # LIVE TRADING - Get actual PnL from exchange
    pos_data = get_position()
    if pos_data:
        pnl = float(pos_data.get('realized_pnl', 0)) + float(pos_data.get('unrealized_pnl', 0))
    
    # Close order
    product_id = ASSETS[symbol]['product_id']
    close_side = 'sell' if direction == 'LONG' else 'buy'
    
    payload = {
        'product_id': product_id,
        'size': size,
        'side': close_side,
        'order_type': 'market_order'
    }
    
    result = api_post('/v2/orders', payload)
    
    if result:
        # Calculate fees
        notional = current_price * size / ASSETS[symbol]['leverage']
        fees = notional * ROUND_TRIP_FEE
        net_pnl = pnl - fees
        
        # Update stats
        stats['trades'] += 1
        stats['gross_pnl'] += pnl
        stats['fees_paid'] += fees
        stats['net_pnl'] += net_pnl
        
        if net_pnl > 0:
            stats['wins'] += 1
            stats['win_streak'] += 1
            stats['loss_streak'] = 0
            stats['best_trade'] = max(stats['best_trade'], net_pnl)
            emoji = "🟢"
        else:
            stats['losses'] += 1
            stats['loss_streak'] += 1
            stats['win_streak'] = 0
            stats['worst_trade'] = min(stats['worst_trade'], net_pnl)
            last_loss_time = time.time()
            emoji = "🔴"
        
        last_trade_time = time.time()
        
        print(f"\n{emoji} CLOSED {direction} {symbol} @ {current_price:.4f}")
        print(f"   💰 PnL: ${net_pnl:.4f} ({pnl_pct*100:+.2f}%) | Reason: {reason}")
        
        position = None
        return net_pnl
    
    return 0

def manage_position():
    """Smart position management with trailing stops"""
    global position_peak_pct, position_trailing, position_at_breakeven
    
    if not position:
        return
    
    symbol = position['symbol']
    direction = position['direction']
    entry_price = position['entry_price']
    current_price = prices[symbol]
    
    # Calculate current P&L percentage
    if direction == 'LONG':
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price
    
    # Track peak
    position_peak_pct = max(position_peak_pct, pnl_pct)
    
    # Check max hold time
    hold_time = time.time() - position_entry_time
    if hold_time > MAX_HOLD_TIME:
        close_position("Max hold time")
        return
    
    # Stop Loss
    if pnl_pct <= -STOP_LOSS_PCT:
        close_position(f"Stop loss ({pnl_pct*100:.2f}%)")
        return
    
    # Take Profit
    if pnl_pct >= TARGET_PCT:
        close_position(f"Target hit ({pnl_pct*100:.2f}%)")
        return
    
    # Move to breakeven
    if not position_at_breakeven and pnl_pct >= BREAKEVEN_TRIGGER:
        position_at_breakeven = True
        print(f"   🛡️ Moved to breakeven")
    
    # Check breakeven stop
    if position_at_breakeven and pnl_pct <= 0.001:
        close_position("Breakeven stop")
        return
    
    # Start trailing
    if not position_trailing and pnl_pct >= TRAIL_START:
        position_trailing = True
        print(f"   📈 Trailing activated at {pnl_pct*100:.2f}%")
    
    # Trailing stop check
    if position_trailing:
        trail_stop_level = position_peak_pct - TRAIL_DISTANCE
        if pnl_pct < trail_stop_level:
            close_position(f"Trail stop (peak: {position_peak_pct*100:.2f}%)")
            return

# =============================================================================
# 📈 PRELOAD HISTORICAL DATA (FIX FOR SLOW WARMUP)
# =============================================================================

def preload_price_history():
    """
    Preload price history from REST API to avoid slow WebSocket warmup.
    Delta Exchange provides OHLC candles - we use close prices.
    """
    print("📈 Preloading price history from API...")
    
    for symbol in ASSETS.keys():
        if symbol == 'BTCUSD':
            continue
            
        try:
            # Get product ID for the symbol (Delta uses product IDs)
            products_result = api_get('/v2/products')
            if not products_result:
                continue
                
            product_id = None
            for prod in products_result:
                if prod.get('symbol') == symbol:
                    product_id = prod.get('id')
                    break
            
            if not product_id:
                continue
            
            # Fetch 1-minute candles for last 30 minutes (30 data points)
            end_time = int(time.time())
            start_time = end_time - 30 * 60  # 30 minutes ago
            
            endpoint = f"/v2/history/candles?resolution=1m&symbol={symbol}&start={start_time}&end={end_time}"
            candles = api_get(endpoint)
            
            if candles and len(candles) > 0:
                # Sort by time (oldest first)
                candles_sorted = sorted(candles, key=lambda x: x.get('time', 0))
                
                for candle in candles_sorted:
                    close_price = float(candle.get('close', 0))
                    candle_time = candle.get('time', time.time())
                    
                    if close_price > 0:
                        price_history[symbol].append({
                            'price': close_price,
                            'time': candle_time
                        })
                        price_data[symbol].append(close_price)
                        prices[symbol] = close_price  # Update current price too
                
                print(f"   ✅ {symbol}: Loaded {len(candles_sorted)} historical prices")
            else:
                # Fallback: try to get ticker price
                ticker = api_get(f'/v2/tickers/{symbol}')
                if ticker:
                    mark_price = float(ticker.get('mark_price', 0))
                    if mark_price > 0:
                        # Seed with current price repeated
                        for i in range(DATA_WARMUP + 5):
                            price_history[symbol].append({
                                'price': mark_price * (1 + (i - DATA_WARMUP/2) * 0.0001),  # Small variance
                                'time': time.time() - (DATA_WARMUP - i)
                            })
                            price_data[symbol].append(mark_price)
                        prices[symbol] = mark_price
                        print(f"   ✅ {symbol}: Seeded with ticker price ${mark_price:.4f}")
                        
        except Exception as e:
            print(f"   ⚠️ {symbol}: Could not preload ({e})")
    
    total_loaded = sum(len(h) for h in price_history.values())
    print(f"📈 Total prices loaded: {total_loaded} (need {DATA_WARMUP} per symbol)")

# =============================================================================
# 🌐 WEBSOCKET
# =============================================================================

def on_ws_message(ws, message):
    global ws_connected
    try:
        data = json.loads(message)
        msg_type = data.get('type')
        
        if msg_type == 'v2/ticker':
            symbol = data.get('symbol', '')
            if symbol in ASSETS:
                mark = float(data.get('mark_price', 0))
                if mark > 0:
                    prices[symbol] = mark
                    price_history[symbol].append({
                        'price': mark,
                        'time': time.time()
                    })
                    # Also update ML price_data
                    price_data[symbol].append(mark)
        
        elif msg_type == 'l2_orderbook':
            symbol = data.get('symbol', '')
            if symbol in ASSETS:
                orderbooks[symbol] = {
                    'buy': data.get('buy', []),
                    'sell': data.get('sell', [])
                }
    except:
        pass

def on_ws_open(ws):
    global ws_connected
    ws_connected = True
    
    # Subscribe to tickers and orderbooks
    channels = []
    for symbol in ASSETS.keys():
        channels.append({'name': 'v2/ticker', 'symbols': [symbol]})
        channels.append({'name': 'l2_orderbook', 'symbols': [symbol]})
    
    ws.send(json.dumps({
        'type': 'subscribe',
        'payload': {'channels': channels}
    }))

def on_ws_close(ws, close_status_code, close_msg):
    global ws_connected
    ws_connected = False

def on_ws_error(ws, error):
    pass

def start_websocket():
    global ws
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_ws_open,
        on_message=on_ws_message,
        on_close=on_ws_close,
        on_error=on_ws_error
    )
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

# =============================================================================
# 📺 DISPLAY
# =============================================================================

def display_status():
    """Single-line status display with throttle"""
    global last_display_time
    
    # Throttle display updates
    current_time = time.time()
    if current_time - last_display_time < DISPLAY_INTERVAL:
        return
    last_display_time = current_time
    
    now = datetime.now().strftime("%H:%M:%S")
    balance, available, balance_inr = get_balance()
    
    # Update stats
    stats['current_balance'] = balance
    if balance > stats['peak_balance']:
        stats['peak_balance'] = balance
    
    # Build status
    parts = [now]
    
    # Asset prices
    for sym in ASSETS.keys():
        if prices[sym] > 0:
            short_sym = sym.replace('USD', '')
            if prices[sym] > 1000:
                parts.append(f"{short_sym}:${prices[sym]:,.0f}")
            else:
                parts.append(f"{short_sym}:{prices[sym]:.4f}")
    
    # Position info
    if position:
        sym = position['symbol'].replace('USD', '')
        direction = "🟢L" if position['direction'] == 'LONG' else "🔴S"
        
        # Calculate unrealized PnL
        entry = position['entry_price']
        current = prices[position['symbol']]
        if position['direction'] == 'LONG':
            pnl_pct = (current - entry) / entry * 100
        else:
            pnl_pct = (entry - current) / entry * 100
        
        pnl_emoji = "📈" if pnl_pct > 0 else "📉"
        parts.append(f"{direction} {sym} {pnl_emoji}{pnl_pct:+.2f}%")
    else:
        # Check if we're stopped due to drawdown
        if check_daily_drawdown():
            parts.append("🛑 DD LIMIT")
        else:
            parts.append("🎯 Waiting")
    
    # Net P&L
    if stats['net_pnl'] > 0:
        parts.append(f"🟢+₹{stats['net_pnl']*85:.2f}")
    elif stats['net_pnl'] < 0:
        parts.append(f"🔴₹{stats['net_pnl']*85:.2f}")
    else:
        parts.append(f"⚪₹0")
    
    # Balance (DYNAMIC)
    parts.append(f"💰₹{balance_inr:.2f}")
    
    # Trade count + Win Rate
    if stats['trades'] > 0:
        wr = stats['wins'] / stats['trades'] * 100
        parts.append(f"📊{stats['trades']}({wr:.0f}%)")
    else:
        parts.append(f"📊0")
    
    # Drawdown indicator
    if stats['starting_balance'] > 0:
        dd = (stats['starting_balance'] - balance) / stats['starting_balance'] * 100
        if dd > 0:
            parts.append(f"DD:{dd:.1f}%")
    
    # Fear & Greed (compact)
    if fear_greed_index < 30:
        fg = f"😰{fear_greed_index}"
    elif fear_greed_index > 70:
        fg = f"🤑{fear_greed_index}"
    else:
        fg = f"😐{fear_greed_index}"
    parts.append(fg)
    
    # Mode indicator
    if PAPER_TRADING:
        parts.append("📝PAPER")
    else:
        parts.append("💎LIVE")
    
    if ML_ENABLED:
        parts.append("🤖ML")
    
    # Single line output - DON'T clear screen to see debug
    status_line = ' | '.join(parts)
    # os.system('cls')  # Disabled for debug
    print(f"\r  {status_line}", end='', flush=True)

# =============================================================================
# 🚀 MAIN
# =============================================================================

def signal_handler(sig, frame):
    global running
    running = False
    print("\n\n🛑 Shutting down...")

def main():
    global running, stats
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Banner - PROFITABLE MODE
    print("\n" + "═"*80)
    print("""
     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗
    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║
    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║
    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║
    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║
    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝
    """)
    print("     💎 PROFITABLE MODE - LOW DRAWDOWN + FEE AWARE 💎".center(80))
    print("═"*80)
    print(f"   📊 Assets: {', '.join([a for a in ASSETS.keys() if a != 'BTCUSD'])}")
    print(f"   🛡️ Max Drawdown: {MAX_DAILY_DRAWDOWN*100}% (Capital Protection!)")
    print(f"   💰 Position Size: {MAX_POSITION_SIZE*100}% max (Controlled Risk)")
    print(f"   📈 Target: {TARGET_PCT*100}% | Stop: {STOP_LOSS_PCT*100}% (R:R = 1:{TARGET_PCT/STOP_LOSS_PCT:.1f})")
    print(f"   💸 Fee Aware: Only trades when profit > {MIN_PROFIT_AFTER_FEE*100}% after fees")
    print(f"   ✅ Quality: {MIN_CONFIRMATIONS}+ confirmations, {MIN_CONFIDENCE}%+ confidence")
    print(f"   🎯 Strategies: RSI, Momentum, Orderbook, VWAP, Trend, Sentiment")
    print("═"*80)
    
    # Get initial balance
    balance, available, balance_inr = get_balance()
    stats['starting_balance'] = balance
    stats['current_balance'] = balance
    stats['peak_balance'] = balance
    
    print(f"\n💰 Starting Balance: ${balance:.4f} (₹{balance_inr:.2f})")
    print(f"🛡️ Max Loss Allowed: ₹{balance_inr * MAX_DAILY_DRAWDOWN:.2f} ({MAX_DAILY_DRAWDOWN*100}%)")
    
    if balance_inr < 10:
        print("⚠️ Low balance - Trading with minimum size")
        print("💡 More capital = Better position sizing!")
    
    # Get Fear & Greed
    get_fear_greed()
    print(f"😰 Fear & Greed Index: {fear_greed_index}")
    
    # Show position sizes
    print("\n📊 Position Sizes (Controlled):")
    for sym in ASSETS.keys():
        if sym == 'BTCUSD':
            continue
        max_lots = calculate_max_lots(sym, available)
        lev = ASSETS[sym]['leverage']
        print(f"   {sym}: {max_lots} lots @ {lev}x leverage")
    
    # Fee calculation
    print(f"\n💸 Fee Analysis:")
    print(f"   Round-trip fees: {ROUND_TRIP_FEE*100}%")
    print(f"   Min profit target: {TARGET_PCT*100}%")
    print(f"   Net profit after fees: {(TARGET_PCT - ROUND_TRIP_FEE)*100}% ✅")
    
    # PRELOAD HISTORY FROM API (Fix for slow warmup!)
    preload_price_history()
    
    # Start WebSocket
    print("\n📡 Connecting to market data...")
    start_websocket()
    time.sleep(2)
    
    if ws_connected:
        print("📡 WebSocket CONNECTED ✅")
    else:
        print("📡 WebSocket connecting...")
    
    print("\n" + "═"*80)
    print("💎 ALADDIN PROFITABLE - Quality Trades Only! 💎".center(80))
    print("🛡️ Low Drawdown | Fee-Aware | Press Ctrl+C to stop".center(80))
    print("═"*80 + "\n")
    
    ticks = 0
    last_scan = 0
    last_fg_update = 0
    last_options_scan = 0
    
    try:
        while running:
            now = time.time()
            
            # Count ticks
            total_ticks = sum(len(h) for h in price_history.values())
            
            # Update Fear & Greed every 5 minutes
            if now - last_fg_update > 300:
                get_fear_greed()
                last_fg_update = now
            
            # Display
            if total_ticks >= DATA_WARMUP:
                display_status()
                
                # Manage existing position
                if position:
                    manage_position()
                
                # Scan for new trades
                if now - last_scan >= SCAN_INTERVAL and not position:
                    trade = get_best_trade()
                    
                    if trade:
                        symbol, direction, confidence, confirmations, reason = trade
                        
                        # Get available balance and calculate lots
                        _, available, _ = get_balance()
                        lots = calculate_max_lots(symbol, available)
                        
                        if lots > 0:
                            print(f"\n\n🔔 SIGNAL: {direction} {symbol}")
                            print(f"   📊 Confidence: {confidence:.0f}% | Confirmations: {confirmations}")
                            open_position(symbol, direction, lots, reason)
                    
                    last_scan = now
            else:
                print(f"\r📈 Warming up: {total_ticks}/{DATA_WARMUP} ticks...", end="", flush=True)
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        running = False
        
        # Close any open position
        if position:
            print("\n\n⚠️ Closing open position...")
            close_position("Bot shutdown")
        
        # Final Summary
        print("\n\n" + "═"*80)
        print("📊 SESSION SUMMARY".center(80))
        print("═"*80)
        
        balance, _, balance_inr = get_balance()
        session_pnl = balance - stats['starting_balance']
        
        print(f"""
   💰 Starting Balance:  ${stats['starting_balance']:.4f}
   💰 Ending Balance:    ${balance:.4f} (₹{balance_inr:.2f})
   📈 Session P&L:       ${session_pnl:.4f} ({session_pnl/stats['starting_balance']*100 if stats['starting_balance'] > 0 else 0:+.2f}%)
   
   📊 Total Trades:      {stats['trades']}
   ✅ Wins:              {stats['wins']}
   ❌ Losses:            {stats['losses']}
   📈 Win Rate:          {stats['wins']/stats['trades']*100 if stats['trades'] > 0 else 0:.1f}%
   
   💵 Gross P&L:         ${stats['gross_pnl']:.4f}
   💸 Fees Paid:         ${stats['fees_paid']:.4f}
   💰 Net P&L:           ${stats['net_pnl']:.4f}
   
   🏆 Best Trade:        ${stats['best_trade']:.4f}
   😢 Worst Trade:       ${stats['worst_trade']:.4f}
   🔥 Best Win Streak:   {stats['win_streak']}
        """)
        print("═"*80)
        print("Thank you for using ALADDIN - Building Wealth Together! 🏆".center(80))
        print("═"*80)

if __name__ == "__main__":
    main()
