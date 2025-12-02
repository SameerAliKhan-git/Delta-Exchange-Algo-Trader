"""
Delta Exchange Algo Trader - External Charting Setup
=====================================================
Quick commands to view charts and run your algo bot.
"""

import webbrowser
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def open_delta_charts():
    """Open Delta Exchange trading charts in browser"""
    urls = {
        "BTCUSD": "https://www.delta.exchange/app/trade/BTCUSD",
        "ETHUSD": "https://www.delta.exchange/app/trade/ETHUSD", 
        "SOLUSD": "https://www.delta.exchange/app/trade/SOLUSD",
    }
    
    print("🚀 Opening Delta Exchange Charts...")
    print()
    
    for symbol, url in urls.items():
        print(f"   📊 {symbol}: {url}")
    
    # Open BTCUSD by default
    webbrowser.open(urls["BTCUSD"])
    print()
    print("✅ Delta Exchange chart opened in your browser!")
    print("   The chart is powered by TradingView - add any indicators you want.")


def open_tradingview():
    """Open TradingView with Delta Exchange symbols"""
    url = "https://www.tradingview.com/chart/?symbol=DELTA%3ABTCUSD"
    
    print("🚀 Opening TradingView...")
    print(f"   📈 URL: {url}")
    
    webbrowser.open(url)
    print()
    print("✅ TradingView opened!")
    print("   Search for DELTA:BTCUSD, DELTA:ETHUSD, etc.")


def print_pine_scripts():
    """Print Pine Scripts for TradingView"""
    from integrations.tradingview import PineScriptGenerator
    
    print()
    print("="*70)
    print("📜 PINE SCRIPTS - Copy to TradingView")
    print("="*70)
    
    strategies = {
        "EMA Crossover (Simple)": PineScriptGenerator.ema_crossover(),
        "RSI Strategy": PineScriptGenerator.rsi_strategy(),
        "MACD Strategy": PineScriptGenerator.macd_strategy(),
        "SuperTrend": PineScriptGenerator.supertrend_strategy(),
        "Medallion Multi-Factor (Advanced)": PineScriptGenerator.medallion_strategy(),
    }
    
    for name, script in strategies.items():
        print(f"\n{'─'*70}")
        print(f"📊 {name}")
        print(f"{'─'*70}")
        print(script)


def start_webhook_server():
    """Start TradingView webhook receiver"""
    from integrations.tradingview import TradingViewWebhook
    
    SECRET = "delta-algo-2024"  # Change this!
    
    webhook = TradingViewWebhook(secret_key=SECRET, port=5555)
    
    # Register signal handlers
    def on_buy(signal):
        print(f"🟢 BUY SIGNAL: {signal['symbol']} @ ${signal['price']}")
        # Here you would call your order execution
    
    def on_sell(signal):
        print(f"🔴 SELL SIGNAL: {signal['symbol']} @ ${signal['price']}")
    
    webhook.on_signal('BUY', on_buy)
    webhook.on_signal('SELL', on_sell)
    
    print()
    print("="*60)
    print("🎯 TRADINGVIEW WEBHOOK SERVER")
    print("="*60)
    print()
    print(f"   Secret Key: {SECRET}")
    print(f"   Webhook URL: http://YOUR_PUBLIC_IP:5555/webhook")
    print()
    print("   TradingView Alert Message Format:")
    print(TradingViewWebhook.generate_alert_message(SECRET))
    print()
    print("="*60)
    
    webhook.start(threaded=False)


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       DELTA EXCHANGE ALGO TRADER - CHARTING OPTIONS         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║                                                              ║")
    print("║  1. Open Delta Exchange Charts (Recommended)                 ║")
    print("║  2. Open TradingView Charts                                  ║")
    print("║  3. Print Pine Scripts for TradingView                       ║")
    print("║  4. Start TradingView Webhook Server                         ║")
    print("║  5. Exit                                                     ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    choice = input("Select option (1-5): ").strip()
    
    if choice == "1":
        open_delta_charts()
    elif choice == "2":
        open_tradingview()
    elif choice == "3":
        print_pine_scripts()
    elif choice == "4":
        start_webhook_server()
    elif choice == "5":
        print("Goodbye!")
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
