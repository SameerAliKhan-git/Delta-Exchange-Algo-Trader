"""
TradingView Integration for Delta Exchange Algo Trader
=======================================================
This module provides:
1. Webhook receiver for TradingView alerts
2. Signal export to TradingView-compatible format
3. Pine Script generator for your strategies

TradingView Charts: https://www.tradingview.com/chart/
Delta Exchange Charts: https://www.delta.exchange/app/trade
"""

import json
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Callable
from flask import Flask, request, jsonify
import threading


class TradingViewWebhook:
    """
    Receives webhook alerts from TradingView and executes trades.
    
    Setup in TradingView:
    1. Create an alert on your chart
    2. Set webhook URL to: http://YOUR_IP:5555/webhook
    3. Set message format (see generate_alert_message)
    """
    
    def __init__(self, secret_key: str = "your-secret-key", port: int = 5555):
        self.secret_key = secret_key
        self.port = port
        self.app = Flask(__name__)
        self.callbacks: Dict[str, Callable] = {}
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/webhook', methods=['POST'])
        def webhook():
            try:
                data = request.json
                
                # Validate secret
                if data.get('secret') != self.secret_key:
                    return jsonify({"error": "Invalid secret"}), 401
                
                # Parse signal
                signal = {
                    "symbol": data.get("symbol", "BTCUSD"),
                    "action": data.get("action", "").upper(),  # BUY, SELL, CLOSE
                    "price": float(data.get("price", 0)),
                    "quantity": float(data.get("quantity", 1)),
                    "strategy": data.get("strategy", "TradingView"),
                    "timestamp": datetime.now().isoformat(),
                    "message": data.get("message", ""),
                }
                
                print(f"📡 TradingView Signal: {signal['action']} {signal['symbol']} @ ${signal['price']}")
                
                # Execute callback
                if signal['action'] in self.callbacks:
                    self.callbacks[signal['action']](signal)
                elif 'ALL' in self.callbacks:
                    self.callbacks['ALL'](signal)
                
                return jsonify({"success": True, "signal": signal})
                
            except Exception as e:
                print(f"❌ Webhook error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "online", "timestamp": datetime.now().isoformat()})
    
    def on_signal(self, action: str, callback: Callable):
        """Register callback for specific action (BUY, SELL, CLOSE, or ALL)"""
        self.callbacks[action.upper()] = callback
    
    def start(self, threaded: bool = True):
        """Start webhook server"""
        print(f"🚀 TradingView Webhook Server starting on port {self.port}")
        print(f"   Webhook URL: http://YOUR_IP:{self.port}/webhook")
        print()
        
        if threaded:
            thread = threading.Thread(
                target=lambda: self.app.run(host='0.0.0.0', port=self.port, debug=False),
                daemon=True
            )
            thread.start()
        else:
            self.app.run(host='0.0.0.0', port=self.port, debug=False)
    
    @staticmethod
    def generate_alert_message(secret: str, symbol: str = "BTCUSD") -> str:
        """Generate TradingView alert message template"""
        return f'''{{
    "secret": "{secret}",
    "symbol": "{symbol}",
    "action": "{{{{strategy.order.action}}}}",
    "price": {{{{close}}}},
    "quantity": {{{{strategy.order.contracts}}}},
    "strategy": "{{{{strategy.order.id}}}}",
    "message": "{{{{strategy.order.comment}}}}"
}}'''


class PineScriptGenerator:
    """
    Generates TradingView Pine Script for your strategies.
    Use these scripts directly on TradingView charts.
    """
    
    @staticmethod
    def ema_crossover(fast: int = 20, slow: int = 50) -> str:
        """EMA Crossover Strategy Pine Script"""
        return f'''
//@version=5
strategy("EMA Crossover [{fast}/{slow}]", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// Inputs
fastLen = input.int({fast}, "Fast EMA")
slowLen = input.int({slow}, "Slow EMA")

// EMAs
fastEMA = ta.ema(close, fastLen)
slowEMA = ta.ema(close, slowLen)

// Plot
plot(fastEMA, "Fast EMA", color=color.blue, linewidth=2)
plot(slowEMA, "Slow EMA", color=color.purple, linewidth=2)

// Signals
longCondition = ta.crossover(fastEMA, slowEMA)
shortCondition = ta.crossunder(fastEMA, slowEMA)

// Strategy
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)

// Alerts for webhook
alertcondition(longCondition, title="Buy Signal", message="BUY")
alertcondition(shortCondition, title="Sell Signal", message="SELL")
'''

    @staticmethod
    def rsi_strategy(period: int = 14, oversold: int = 30, overbought: int = 70) -> str:
        """RSI Strategy Pine Script"""
        return f'''
//@version=5
strategy("RSI Strategy [{period}]", overlay=false, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// Inputs
rsiPeriod = input.int({period}, "RSI Period")
oversold = input.int({oversold}, "Oversold Level")
overbought = input.int({overbought}, "Overbought Level")

// RSI
rsi = ta.rsi(close, rsiPeriod)

// Plot
plot(rsi, "RSI", color=color.orange, linewidth=2)
hline(oversold, "Oversold", color=color.green)
hline(overbought, "Overbought", color=color.red)
hline(50, "Mid", color=color.gray)

// Signals
longCondition = ta.crossover(rsi, oversold)
shortCondition = ta.crossunder(rsi, overbought)

// Strategy
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)

// Alerts
alertcondition(longCondition, title="RSI Buy", message="BUY - RSI Oversold")
alertcondition(shortCondition, title="RSI Sell", message="SELL - RSI Overbought")
'''

    @staticmethod
    def macd_strategy(fast: int = 12, slow: int = 26, signal: int = 9) -> str:
        """MACD Strategy Pine Script"""
        return f'''
//@version=5
strategy("MACD Strategy [{fast},{slow},{signal}]", overlay=false, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// Inputs  
fastLen = input.int({fast}, "Fast Length")
slowLen = input.int({slow}, "Slow Length")
signalLen = input.int({signal}, "Signal Length")

// MACD
[macdLine, signalLine, histLine] = ta.macd(close, fastLen, slowLen, signalLen)

// Plot
plot(macdLine, "MACD", color=color.blue, linewidth=2)
plot(signalLine, "Signal", color=color.orange, linewidth=2)
plot(histLine, "Histogram", style=plot.style_histogram, color=histLine >= 0 ? color.green : color.red)

// Signals
longCondition = ta.crossover(macdLine, signalLine)
shortCondition = ta.crossunder(macdLine, signalLine)

// Strategy
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)

// Alerts
alertcondition(longCondition, title="MACD Buy", message="BUY - MACD Crossover")
alertcondition(shortCondition, title="MACD Sell", message="SELL - MACD Crossunder")
'''

    @staticmethod
    def supertrend_strategy(period: int = 10, multiplier: float = 3.0) -> str:
        """SuperTrend Strategy Pine Script"""
        return f'''
//@version=5
strategy("SuperTrend Strategy", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// Inputs
atrPeriod = input.int({period}, "ATR Period")
factor = input.float({multiplier}, "Factor")

// SuperTrend
[supertrend, direction] = ta.supertrend(factor, atrPeriod)

// Plot
plot(supertrend, "SuperTrend", color=direction < 0 ? color.green : color.red, linewidth=2)

// Signals
longCondition = ta.crossover(close, supertrend)
shortCondition = ta.crossunder(close, supertrend)

// Strategy
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition  
    strategy.entry("Short", strategy.short)

// Alerts
alertcondition(longCondition, title="ST Buy", message="BUY - SuperTrend")
alertcondition(shortCondition, title="ST Sell", message="SELL - SuperTrend")
'''

    @staticmethod
    def medallion_strategy() -> str:
        """Renaissance Medallion-style Multi-Factor Strategy"""
        return '''
//@version=5
strategy("Medallion Multi-Factor", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// ============ Trend Indicators ============
ema20 = ta.ema(close, 20)
ema50 = ta.ema(close, 50)
ema200 = ta.ema(close, 200)

// ============ Momentum ============
rsi = ta.rsi(close, 14)
[macdLine, signalLine, _] = ta.macd(close, 12, 26, 9)

// ============ Volatility ============
atr = ta.atr(14)
bbBasis = ta.sma(close, 20)
bbDev = ta.stdev(close, 20) * 2
bbUpper = bbBasis + bbDev
bbLower = bbBasis - bbDev

// ============ Volume ============
volumeMA = ta.sma(volume, 20)
volumeSpike = volume > volumeMA * 1.5

// ============ Multi-Factor Score ============
trendScore = (ema20 > ema50 ? 1 : -1) + (ema50 > ema200 ? 1 : -1) + (close > ema20 ? 1 : -1)
momentumScore = (rsi > 50 ? 1 : -1) + (macdLine > signalLine ? 1 : -1)
volatilityScore = close < bbLower ? 1 : close > bbUpper ? -1 : 0

totalScore = trendScore + momentumScore + volatilityScore

// ============ Plot ============
plot(ema20, "EMA 20", color=color.blue)
plot(ema50, "EMA 50", color=color.purple)
plot(ema200, "EMA 200", color=color.gray)

// Score visualization
bgcolor(totalScore >= 3 ? color.new(color.green, 90) : totalScore <= -3 ? color.new(color.red, 90) : na)

// ============ Signals ============
strongBuy = totalScore >= 3 and volumeSpike
strongSell = totalScore <= -3 and volumeSpike

// Strategy
if strongBuy
    strategy.entry("Long", strategy.long)
if strongSell
    strategy.entry("Short", strategy.short)

// Stop Loss & Take Profit
strategy.exit("Exit Long", "Long", stop=close - atr * 2, limit=close + atr * 4)
strategy.exit("Exit Short", "Short", stop=close + atr * 2, limit=close - atr * 4)

// Alerts
alertcondition(strongBuy, title="Strong Buy", message="STRONG BUY - Multi-Factor")
alertcondition(strongSell, title="Strong Sell", message="STRONG SELL - Multi-Factor")
'''


def print_setup_instructions():
    """Print setup instructions for TradingView integration"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRADINGVIEW + DELTA EXCHANGE SETUP                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 OPTION 1: Use Delta Exchange Charts Directly                             ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  1. Go to: https://www.delta.exchange/app/trade/BTCUSD                       ║
║  2. Click on the chart (TradingView powered)                                 ║
║  3. Add indicators: EMA, RSI, MACD, etc.                                     ║
║  4. Your algo bot runs separately and executes trades                        ║
║                                                                              ║
║  📈 OPTION 2: Use TradingView Charts + Webhooks                              ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  1. Go to: https://www.tradingview.com/chart/                                ║
║  2. Search for: DELTA:BTCUSD or DELTA:ETHUSD                                 ║
║  3. Add Pine Script strategy (generated below)                               ║
║  4. Create Alert → Set webhook URL → Get signals in your bot                 ║
║                                                                              ║
║  🔗 OPTION 3: Export Signals to TradingView                                  ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Your bot generates signals → Displayed on TradingView via webhooks          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_setup_instructions()
    
    print("\n" + "="*60)
    print("📜 PINE SCRIPTS FOR TRADINGVIEW")
    print("="*60)
    
    print("\n--- EMA Crossover Strategy ---")
    print(PineScriptGenerator.ema_crossover())
    
    print("\n--- Medallion Multi-Factor Strategy ---")
    print(PineScriptGenerator.medallion_strategy())
