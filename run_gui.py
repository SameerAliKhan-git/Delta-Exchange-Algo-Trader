"""
Launch the Delta Exchange Algo Trader GUI
==========================================
Professional trading interface with REAL-TIME market data,
candlestick charts, order book visualization, and trading controls.

Data is fetched LIVE from Delta Exchange API.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the trading GUI"""
    print("=" * 60)
    print("🤖 DELTA EXCHANGE ALGO TRADER v4.0")
    print("   Renaissance Edition - Real-Time Trading Interface")
    print("=" * 60)
    print()
    
    # Check dependencies
    try:
        import tkinter
        print("✅ Tkinter available")
    except ImportError:
        print("❌ Tkinter not available. Please install Python with Tkinter support.")
        sys.exit(1)
    
    try:
        import matplotlib
        print("✅ Matplotlib available")
    except ImportError:
        print("⚠️  Matplotlib not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib', '-q'])
        print("✅ Matplotlib installed")
    
    try:
        import numpy
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available. Please install: pip install numpy")
        sys.exit(1)
    
    try:
        import requests
        print("✅ Requests available (for live API)")
    except ImportError:
        print("⚠️  Requests not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests', '-q'])
        print("✅ Requests installed")
    
    print()
    print("📡 Connecting to Delta Exchange LIVE API...")
    print("🚀 Launching Real-Time Trading GUI...")
    print()
    
    from gui import launch_gui
    launch_gui(testnet=False)  # Use LIVE API for real market data


if __name__ == '__main__':
    main()
