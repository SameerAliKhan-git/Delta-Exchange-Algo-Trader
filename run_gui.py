"""
Launch the Delta Exchange Algo Trader GUI
==========================================
Professional trading interface with candlestick charts,
order book visualization, and strategy controls.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the trading GUI"""
    print("=" * 60)
    print("🤖 DELTA EXCHANGE ALGO TRADER v4.0")
    print("   Renaissance Edition - Professional Trading Interface")
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
    
    print()
    print("🚀 Launching GUI...")
    print()
    
    from gui import launch_gui
    launch_gui()


if __name__ == '__main__':
    main()
