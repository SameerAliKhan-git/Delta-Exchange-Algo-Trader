"""
ALADDIN AI - Autonomous Trading Bot Launcher
=============================================
Launch the complete autonomous trading system with chart visualization.

This bot:
1. Opens Delta Exchange charts in your browser
2. Analyzes market conditions in real-time
3. Monitors global news and sentiment
4. Makes autonomous trading decisions
5. Executes trades automatically
6. Manages risk continuously
"""

import sys
import os
import time
import webbrowser
import threading
import logging
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AladdinLauncher")


def print_banner():
    """Print startup banner"""
    banner = """
    
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗                  ║
    ║    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║                  ║
    ║    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║                  ║
    ║    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║                  ║
    ║    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║                  ║
    ║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝                  ║
    ║                                                                           ║
    ║                 AUTONOMOUS AI TRADING SYSTEM                              ║
    ║                 Inspired by BlackRock's Aladdin                           ║
    ║                                                                           ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║    FEATURES:                                                              ║
    ║    - Real-time market analysis across multiple timeframes                 ║
    ║    - Global news & sentiment monitoring                                   ║
    ║    - Autonomous trading decisions                                         ║
    ║    - Dynamic risk management                                              ║
    ║    - Multi-strategy execution                                             ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def open_delta_exchange_charts():
    """Open Delta Exchange trading charts in browser"""
    print("\n[CHARTS] Opening Delta Exchange in your browser...")
    
    charts = [
        ("BTCUSD", "https://www.delta.exchange/app/trade/BTCUSD"),
        ("ETHUSD", "https://www.delta.exchange/app/trade/ETHUSD"),
    ]
    
    # Open primary chart
    webbrowser.open(charts[0][1])
    
    print(f"[CHARTS] Opened: {charts[0][0]} - {charts[0][1]}")
    print("[CHARTS] You can monitor the charts while the bot trades autonomously")
    print()
    
    return True


def run_market_analysis_demo():
    """Run a demo of the market analysis"""
    print("\n" + "=" * 70)
    print("MARKET ANALYSIS")
    print("=" * 70)
    
    try:
        from aladdin.market_analyzer import MarketAnalyzer
        analyzer = MarketAnalyzer()
        
        for symbol in ["BTCUSD", "ETHUSD"]:
            analysis = analyzer.analyze(symbol)
            price = analysis.get("price", 0)
            trend = analysis.get("trend", 0)
            momentum = analysis.get("momentum", 0)
            volatility = analysis.get("volatility", 0)
            
            trend_str = "BULLISH" if trend > 0.2 else "BEARISH" if trend < -0.2 else "NEUTRAL"
            
            print(f"\n  {symbol}:")
            print(f"    Price:      ${price:,.2f}")
            print(f"    Trend:      {trend_str} ({trend:+.2f})")
            print(f"    Momentum:   {momentum:+.2f}")
            print(f"    Volatility: {volatility:.2f}")
            
    except Exception as e:
        print(f"  Error in analysis: {e}")


def run_sentiment_analysis_demo():
    """Run a demo of the sentiment analysis"""
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS")
    print("=" * 70)
    
    try:
        from aladdin.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        
        sentiment = analyzer.get_sentiment("BTC")
        
        print(f"\n  Fear & Greed Index: {sentiment['fear_greed']:.0f}")
        print(f"  Market Mood:        {analyzer.get_market_mood()}")
        print(f"  News Sentiment:     {sentiment['news_sentiment']:+.2f}")
        print(f"  Social Sentiment:   {sentiment['social_sentiment']:+.2f}")
        print(f"  Economic Outlook:   {sentiment['economic_outlook']}")
        
        bias, confidence = analyzer.get_trading_bias()
        print(f"\n  Trading Bias:       {bias} (Confidence: {confidence:.1%})")
        
        print("\n  Recent Headlines:")
        for news in analyzer.get_news_summary(3):
            icon = "[+]" if news["sentiment"] > 0 else "[-]" if news["sentiment"] < 0 else "[ ]"
            print(f"    {icon} {news['title'][:55]}...")
            
    except Exception as e:
        print(f"  Error in sentiment analysis: {e}")


def run_autonomous_bot(paper_mode: bool = True):
    """Run the autonomous trading bot"""
    print("\n" + "=" * 70)
    print("AUTONOMOUS TRADING BOT")
    print("=" * 70)
    print(f"\n  Mode: {'PAPER TRADING (Simulated)' if paper_mode else 'LIVE TRADING'}")
    print("  Symbols: BTCUSD, ETHUSD")
    print("  Analysis Interval: 60 seconds")
    print("\n  Press Ctrl+C to stop the bot")
    print()
    
    try:
        from aladdin.core import AladdinAI
        
        # Initialize with config
        config = {
            "trading": {
                "symbols": ["BTCUSD", "ETHUSD"],
                "max_positions": 3,
                "max_position_size_pct": 10,
                "min_signal_strength": 3,
                "min_confidence": 0.6,
            },
            "risk": {
                "max_daily_loss_pct": 5,
                "max_drawdown_pct": 15,
                "risk_per_trade_pct": 2,
                "max_leverage": 10,
            },
            "analysis": {
                "update_interval_seconds": 60,
                "timeframes": ["5m", "15m", "1h", "4h"],
                "sentiment_weight": 0.3,
                "technical_weight": 0.5,
                "regime_weight": 0.2,
            }
        }
        
        aladdin = AladdinAI(config)
        
        # Set paper mode on executor
        aladdin.executor.paper_mode = paper_mode
        
        cycle = 0
        
        while True:
            cycle += 1
            print(f"\n{'─' * 70}")
            print(f"CYCLE {cycle} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'─' * 70}")
            
            for symbol in config["trading"]["symbols"]:
                print(f"\n  Analyzing {symbol}...")
                
                # Make decision
                signal = aladdin.make_decision(symbol)
                
                if signal:
                    print(f"\n  SIGNAL GENERATED:")
                    print(f"    Direction:  {signal.direction.upper()}")
                    print(f"    Entry:      ${signal.entry_price:,.2f}")
                    print(f"    Stop Loss:  ${signal.stop_loss:,.2f}")
                    print(f"    Take Profit: ${signal.take_profit:,.2f}")
                    print(f"    Strategy:   {signal.strategy}")
                    print(f"    Confidence: {signal.confidence:.1%}")
                    
                    # Execute (paper or live)
                    result = aladdin.execute_signal(signal)
                    
                    if result:
                        print(f"    Status:     EXECUTED {'(PAPER)' if paper_mode else '(LIVE)'}")
                else:
                    print(f"    No signal - Market conditions not favorable")
            
            # Status update
            status = aladdin.get_status()
            print(f"\n  Portfolio:")
            print(f"    Equity:     ${status['portfolio']['equity']:,.2f}")
            print(f"    Positions:  {status['portfolio']['open_positions']}")
            print(f"    Daily P&L:  ${status['portfolio']['daily_pnl']:+,.2f}")
            
            print(f"\n  Waiting {config['analysis']['update_interval_seconds']}s until next analysis...")
            time.sleep(config["analysis"]["update_interval_seconds"])
            
    except KeyboardInterrupt:
        print("\n\n  Bot stopped by user")
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


def show_menu():
    """Show main menu"""
    print("\n" + "=" * 70)
    print("MAIN MENU")
    print("=" * 70)
    print("""
    1. Open Delta Exchange Charts (Browser)
    2. Run Market Analysis
    3. Run Sentiment Analysis  
    4. Start Autonomous Bot (Paper Mode)
    5. Start Autonomous Bot (LIVE MODE) [Caution!]
    6. Full Demo (Charts + Analysis + Bot)
    7. Exit
    """)
    return input("Select option (1-7): ").strip()


def run_full_demo():
    """Run full demonstration"""
    print("\n[DEMO] Starting full Aladdin AI demonstration...")
    
    # Open charts
    open_delta_exchange_charts()
    time.sleep(2)
    
    # Run analysis
    run_market_analysis_demo()
    time.sleep(1)
    
    run_sentiment_analysis_demo()
    time.sleep(1)
    
    # Start bot
    print("\n[DEMO] Starting autonomous trading bot in 5 seconds...")
    print("[DEMO] Press Ctrl+C at any time to stop")
    time.sleep(5)
    
    run_autonomous_bot(paper_mode=True)


def main():
    """Main entry point"""
    print_banner()
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            open_delta_exchange_charts()
        elif choice == "2":
            run_market_analysis_demo()
        elif choice == "3":
            run_sentiment_analysis_demo()
        elif choice == "4":
            run_autonomous_bot(paper_mode=True)
        elif choice == "5":
            print("\n  WARNING: LIVE TRADING MODE")
            print("  This will execute real trades with real money!")
            confirm = input("  Type 'YES' to confirm: ").strip()
            if confirm == "YES":
                run_autonomous_bot(paper_mode=False)
            else:
                print("  Live trading cancelled")
        elif choice == "6":
            run_full_demo()
        elif choice == "7":
            print("\n  Goodbye!")
            break
        else:
            print("\n  Invalid option")
        
        input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()
