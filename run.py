"""
Delta Exchange Algo Trader - Main Entry Point
==============================================
Use external charting (Delta Exchange or TradingView)
Run this file to start the trading bot.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       ██████╗ ███████╗██╗  ████████╗ █████╗      █████╗ ██╗      ██████╗     ║
║       ██╔══██╗██╔════╝██║  ╚══██╔══╝██╔══██╗    ██╔══██╗██║     ██╔════╝     ║
║       ██║  ██║█████╗  ██║     ██║   ███████║    ███████║██║     ██║  ███╗    ║
║       ██║  ██║██╔══╝  ██║     ██║   ██╔══██║    ██╔══██║██║     ██║   ██║    ║
║       ██████╔╝███████╗███████╗██║   ██║  ██║    ██║  ██║███████╗╚██████╔╝    ║
║       ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝     ║
║                                                                              ║
║                    RENAISSANCE-LEVEL ALGO TRADING                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   📊 CHARTING OPTIONS:                                                       ║
║                                                                              ║
║   • Delta Exchange: https://www.delta.exchange/app/trade/BTCUSD              ║
║   • TradingView:    https://www.tradingview.com/chart/?symbol=DELTA:BTCUSD   ║
║                                                                              ║
║   Charts are powered by TradingView with full indicator support.             ║
║   Your algo bot runs independently and executes trades automatically.        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def show_menu():
    print("""
┌──────────────────────────────────────────────────────────────┐
│                      MAIN MENU                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   1. 📈 Open Charts (Delta Exchange / TradingView)           │
│   2. 🤖 Start Trading Bot                                    │
│   3. 📊 View Market Data                                     │
│   4. 📜 Get Pine Scripts for TradingView                     │
│   5. 🔗 Start TradingView Webhook Server                     │
│   6. 📋 Run Backtest                                         │
│   7. ⚙️  Check API Connection                                │
│   8. ❌ Exit                                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
""")
    return input("Select option (1-8): ").strip()


def open_charts():
    import webbrowser
    print("\n📊 CHARTING OPTIONS:")
    print("   1. Delta Exchange (Recommended)")
    print("   2. TradingView")
    
    choice = input("\nSelect (1-2): ").strip()
    
    if choice == "1":
        url = "https://www.delta.exchange/app/trade/BTCUSD"
        print(f"\n🚀 Opening Delta Exchange: {url}")
        webbrowser.open(url)
    else:
        # url = "https://www.tradingview.com/chart/?symbol=DELTA%3ABTCUSD"
        url = "https://www.tradingview.com/chart/?symbol=DELTA%3ABTCUSD"
        print(f"\n🚀 Opening TradingView: {url}")
        webbrowser.open(url)
    
    print("✅ Chart opened in your browser!")


def start_trading_bot():
    try:
        from execution.client import create_client
        from config.credentials import API_KEY, API_SECRET, TESTNET
        
        print("\n🤖 STARTING TRADING BOT...")
        print("─" * 50)
        
        client = create_client(API_KEY, API_SECRET, testnet=TESTNET)
        
        # Get current ticker
        response = client.get_ticker("BTCUSD")
        if response and response.get('success'):
            # Find BTCUSD in results
            tickers = response.get('result', [])
            for t in tickers:
                if t.get('symbol') == 'BTCUSD':
                    mark_price = t.get('mark_price', 'N/A')
                    print(f"   📊 BTC Price: ${mark_price}")
                    break
        
        print()
        print("   ⚠️  Trading bot would run here.")
        print("   Currently in PAPER mode for safety.")
        print()
        print("   To enable live trading:")
        print("   1. Set up risk parameters")
        print("   2. Configure position sizing") 
        print("   3. Enable in settings")
        print()
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please ensure all modules are set up correctly.")


def view_market_data():
    try:
        from execution.client import create_client
        from config.credentials import API_KEY, API_SECRET, TESTNET
        
        print("\n📊 FETCHING MARKET DATA...")
        print("─" * 50)
        
        client = create_client(API_KEY, API_SECRET, testnet=TESTNET)
        
        symbols = ["BTCUSD", "ETHUSD"]
        
        # Get all tickers
        response = client.get_tickers()
        if response and response.get('success'):
            tickers = response.get('result', [])
            ticker_map = {t['symbol']: t for t in tickers}
            
            for symbol in symbols:
                if symbol in ticker_map:
                    t = ticker_map[symbol]
                    mark = t.get('mark_price', 'N/A')
                    high = t.get('high', 'N/A')
                    low = t.get('low', 'N/A')
                    volume = t.get('volume', 'N/A')
                    print(f"\n   {symbol}:")
                    print(f"   ├─ Price: ${mark}")
                    print(f"   ├─ 24h High: ${high}")
                    print(f"   ├─ 24h Low: ${low}")
                    print(f"   └─ Volume: {volume}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def get_pine_scripts():
    from integrations.tradingview import PineScriptGenerator
    
    print("\n📜 PINE SCRIPTS FOR TRADINGVIEW")
    print("─" * 50)
    print("\nCopy any of these to TradingView's Pine Editor:\n")
    
    strategies = [
        ("EMA Crossover", PineScriptGenerator.ema_crossover),
        ("RSI Strategy", PineScriptGenerator.rsi_strategy),
        ("MACD Strategy", PineScriptGenerator.macd_strategy),
        ("SuperTrend", PineScriptGenerator.supertrend_strategy),
        ("Medallion Multi-Factor", PineScriptGenerator.medallion_strategy),
    ]
    
    for i, (name, _) in enumerate(strategies, 1):
        print(f"   {i}. {name}")
    
    choice = input("\nSelect strategy to view (1-5): ").strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(strategies):
            name, func = strategies[idx]
            print(f"\n{'='*60}")
            print(f"📊 {name}")
            print(f"{'='*60}")
            print(func())
            print(f"{'='*60}")
    except (ValueError, IndexError):
        print("Invalid selection")


def start_webhook_server():
    from integrations.tradingview import TradingViewWebhook
    from execution.client import create_client
    from config.credentials import API_KEY, API_SECRET, TESTNET
    
    SECRET = "delta-algo-2024"
    
    print("\n🔗 TRADINGVIEW WEBHOOK SERVER")
    print("─" * 50)
    
    webhook = TradingViewWebhook(secret_key=SECRET, port=5555)
    client = create_client(API_KEY, API_SECRET, testnet=TESTNET)
    
    def execute_signal(signal):
        action = signal['action']
        symbol = signal['symbol']
        price = signal['price']
        
        if action == 'BUY':
            print(f"🟢 BUY SIGNAL: {symbol} @ ${price}")
            # client.place_order(symbol, 'buy', ...)
        elif action == 'SELL':
            print(f"🔴 SELL SIGNAL: {symbol} @ ${price}")
            # client.place_order(symbol, 'sell', ...)
    
    webhook.on_signal('BUY', execute_signal)
    webhook.on_signal('SELL', execute_signal)
    webhook.on_signal('ALL', execute_signal)
    
    print(f"\n   Secret: {SECRET}")
    print(f"   URL: http://YOUR_PUBLIC_IP:5555/webhook")
    print()
    print("   TradingView Alert Message Format:")
    print(TradingViewWebhook.generate_alert_message(SECRET))
    print()
    print("   Press Ctrl+C to stop")
    print()
    
    webhook.start(threaded=False)


def run_backtest():
    print("\n📋 BACKTEST")
    print("─" * 50)
    print("\n   Backtest module available in backtest/")
    print("   Run: python backtest/run_backtest.py")
    print()


def check_api():
    try:
        from execution.client import create_client
        from config.credentials import API_KEY, API_SECRET, TESTNET
        
        print("\n⚙️  CHECKING API CONNECTION...")
        print("─" * 50)
        
        client = create_client(API_KEY, API_SECRET, testnet=TESTNET)
        
        # Get all tickers 
        response = client.get_tickers()
        
        if response and response.get('success'):
            tickers = response.get('result', [])
            for t in tickers:
                if t.get('symbol') == 'BTCUSD':
                    print(f"\n   ✅ API Connection: SUCCESSFUL")
                    print(f"   ✅ Market Data: WORKING")
                    print(f"   📊 BTC Price: ${t.get('mark_price', 'N/A')}")
                    break
        else:
            print(f"\n   ❌ API Connection: FAILED")
            print(f"   Response: {response}")
        
        print()
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        print()


def main():
    print_banner()
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            open_charts()
        elif choice == "2":
            start_trading_bot()
        elif choice == "3":
            view_market_data()
        elif choice == "4":
            get_pine_scripts()
        elif choice == "5":
            start_webhook_server()
        elif choice == "6":
            run_backtest()
        elif choice == "7":
            check_api()
        elif choice == "8":
            print("\n👋 Goodbye!\n")
            break
        else:
            print("Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
