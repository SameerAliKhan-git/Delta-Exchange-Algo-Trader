"""
GUI Module - Professional Real-Time Trading Interface
======================================================
Live market data, candlestick charts, order book, and trading controls.
"""

from .main_window import TradingGUI, run_app

def launch_gui(testnet: bool = False):
    """Launch the trading GUI"""
    run_app(testnet=testnet)

__all__ = ['TradingGUI', 'run_app', 'launch_gui']
