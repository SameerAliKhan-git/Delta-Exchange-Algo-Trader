"""
Aladdin Bot 24/7 Runner
=======================
Runs the trading bot continuously with:
- Auto-restart on crash
- Profit tracking
- Session logging
- Graceful shutdown

To stop: Press Ctrl+C twice
"""

import subprocess
import sys
import time
import os
from datetime import datetime
import signal

# Configuration
MAX_RESTARTS = 100  # Max restarts before stopping
RESTART_DELAY = 30  # Seconds between restarts
LOG_DIR = "logs"

# Track state
restart_count = 0
total_runtime = 0
start_time = datetime.now()
running = True

def signal_handler(sig, frame):
    global running
    print("\n⚠️ Shutdown signal received. Press Ctrl+C again to force quit.")
    running = False

signal.signal(signal.SIGINT, signal_handler)

def create_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def run_bot():
    """Run the trading bot and capture output"""
    global restart_count, running
    
    log_file = os.path.join(LOG_DIR, f"aladdin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     ALADDIN 24/7 TRADING DAEMON                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Started:   {start_time.strftime('%Y-%m-%d %H:%M:%S'):<30}              ║
║  Restarts:  {restart_count:<30}              ║
║  Log File:  {log_file:<30}
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    try:
        # Run the bot and stream output
        process = subprocess.Popen(
            [sys.executable, "aladdin_autonomous.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Stream output to console and log file
        with open(log_file, "w", encoding="utf-8") as f:
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                f.flush()
        
        process.wait()
        return process.returncode
        
    except Exception as e:
        print(f"❌ Error running bot: {e}")
        return 1

def main():
    global restart_count, running
    
    create_log_dir()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     █████╗ ██╗      █████╗ ██████╗ ██████╗ ██╗███╗   ██╗         ║
    ║    ██╔══██╗██║     ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║         ║
    ║    ███████║██║     ███████║██║  ██║██║  ██║██║██╔██╗ ██║         ║
    ║    ██╔══██║██║     ██╔══██║██║  ██║██║  ██║██║██║╚██╗██║         ║
    ║    ██║  ██║███████╗██║  ██║██████╔╝██████╔╝██║██║ ╚████║         ║
    ║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝         ║
    ║                                                                   ║
    ║              🚀 24/7 AUTONOMOUS TRADING DAEMON 🚀                 ║
    ║                                                                   ║
    ║  This bot trades based on:                                       ║
    ║  • Global crypto news sentiment                                  ║
    ║  • Economic news (Fed, inflation, etc.)                          ║
    ║  • Market trends from CryptoCompare & CoinGecko                  ║
    ║                                                                   ║
    ║  Auto-features:                                                  ║
    ║  • Exits trades when sentiment reverses                          ║
    ║  • Auto-restart on crash                                         ║
    ║  • Stop loss & take profit                                       ║
    ║                                                                   ║
    ║  Press Ctrl+C to stop                                            ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    while running and restart_count < MAX_RESTARTS:
        print(f"\n🚀 Starting Aladdin Bot (Attempt #{restart_count + 1})...\n")
        
        exit_code = run_bot()
        
        if not running:
            print("\n✅ Bot stopped by user")
            break
        
        restart_count += 1
        
        if exit_code != 0:
            print(f"\n⚠️ Bot exited with code {exit_code}")
            print(f"🔄 Restarting in {RESTART_DELAY} seconds...")
            
            for i in range(RESTART_DELAY, 0, -1):
                if not running:
                    break
                print(f"   {i}...", end="\r")
                time.sleep(1)
        else:
            print("\n✅ Bot exited normally")
            break
    
    # Final summary
    runtime = datetime.now() - start_time
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     SESSION COMPLETE                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total Runtime:  {str(runtime).split('.')[0]:<30}           ║
║  Restarts:       {restart_count:<30}           ║
║  Logs saved in:  {LOG_DIR}/ folder                                      
╚══════════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
