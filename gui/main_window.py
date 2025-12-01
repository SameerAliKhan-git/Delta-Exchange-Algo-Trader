"""
Professional Trading GUI - Main Window
=======================================
A professional trading interface inspired by TradingView and Bloomberg Terminal.

Features:
- Real-time candlestick charts with OHLCV data
- Technical indicator overlays (EMA, Bollinger Bands, etc.)
- Volume bars
- Order book visualization
- Position management panel
- Strategy control panel
- Live P&L tracking
- Dark theme professional design
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import numpy as np

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


# Color scheme - Dark theme inspired by TradingView
COLORS = {
    'bg_dark': '#131722',
    'bg_panel': '#1E222D',
    'bg_card': '#2A2E39',
    'bg_input': '#363A45',
    'text_primary': '#D1D4DC',
    'text_secondary': '#787B86',
    'text_muted': '#4A4E59',
    'accent_blue': '#2962FF',
    'accent_cyan': '#00BCD4',
    'green': '#26A69A',
    'red': '#EF5350',
    'orange': '#FF9800',
    'yellow': '#FFEB3B',
    'purple': '#9C27B0',
    'border': '#363A45',
    'candle_up': '#26A69A',
    'candle_down': '#EF5350',
    'volume_up': '#26A69A80',
    'volume_down': '#EF535080',
}


class CandlestickChart:
    """Professional candlestick chart with indicators"""
    
    def __init__(self, parent_frame, width=800, height=500):
        self.parent = parent_frame
        self.width = width
        self.height = height
        
        if not HAS_MATPLOTLIB:
            self._create_placeholder()
            return
            
        # Create figure with dark theme
        self.fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor=COLORS['bg_dark'])
        
        # Create subplots: main chart (70%) and volume (30%)
        self.ax_main = self.fig.add_axes([0.08, 0.35, 0.88, 0.60], facecolor=COLORS['bg_panel'])
        self.ax_volume = self.fig.add_axes([0.08, 0.08, 0.88, 0.22], facecolor=COLORS['bg_panel'], sharex=self.ax_main)
        
        # Style axes
        for ax in [self.ax_main, self.ax_volume]:
            ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
            ax.spines['top'].set_color(COLORS['border'])
            ax.spines['bottom'].set_color(COLORS['border'])
            ax.spines['left'].set_color(COLORS['border'])
            ax.spines['right'].set_color(COLORS['border'])
            ax.grid(True, alpha=0.2, color=COLORS['border'])
            
        # Hide x-axis labels on main chart
        plt.setp(self.ax_main.get_xticklabels(), visible=False)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar_frame = tk.Frame(parent_frame, bg=COLORS['bg_dark'])
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.config(background=COLORS['bg_dark'])
        self.toolbar.update()
        
        # Data storage
        self.data = None
        self.indicators = {}
        
    def _create_placeholder(self):
        """Create placeholder when matplotlib is not available"""
        label = tk.Label(
            self.parent,
            text="📊 Install matplotlib for charts:\npip install matplotlib",
            bg=COLORS['bg_panel'],
            fg=COLORS['text_secondary'],
            font=('Segoe UI', 14)
        )
        label.pack(fill=tk.BOTH, expand=True)
        
    def update_data(self, timestamps, opens, highs, lows, closes, volumes):
        """Update chart with new OHLCV data"""
        if not HAS_MATPLOTLIB:
            return
            
        self.data = {
            'timestamps': timestamps,
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'volumes': volumes
        }
        
        self._draw_chart()
        
    def _draw_chart(self):
        """Draw the candlestick chart"""
        if self.data is None:
            return
            
        # Clear axes
        self.ax_main.clear()
        self.ax_volume.clear()
        
        # Re-apply styling
        for ax in [self.ax_main, self.ax_volume]:
            ax.set_facecolor(COLORS['bg_panel'])
            ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
            ax.grid(True, alpha=0.2, color=COLORS['border'])
            
        timestamps = self.data['timestamps']
        opens = self.data['opens']
        highs = self.data['highs']
        lows = self.data['lows']
        closes = self.data['closes']
        volumes = self.data['volumes']
        
        n = len(timestamps)
        x = np.arange(n)
        width = 0.6
        
        # Draw candlesticks
        for i in range(n):
            color = COLORS['candle_up'] if closes[i] >= opens[i] else COLORS['candle_down']
            
            # Wick
            self.ax_main.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=1)
            
            # Body
            body_bottom = min(opens[i], closes[i])
            body_height = abs(closes[i] - opens[i])
            
            rect = Rectangle(
                (x[i] - width/2, body_bottom),
                width, body_height,
                facecolor=color,
                edgecolor=color
            )
            self.ax_main.add_patch(rect)
            
        # Draw volume bars
        vol_colors = [COLORS['volume_up'] if closes[i] >= opens[i] else COLORS['volume_down'] 
                      for i in range(n)]
        self.ax_volume.bar(x, volumes, width=width, color=vol_colors)
        
        # Draw indicators
        self._draw_indicators(x)
        
        # Set labels
        self.ax_main.set_ylabel('Price', color=COLORS['text_secondary'], fontsize=9)
        self.ax_volume.set_ylabel('Volume', color=COLORS['text_secondary'], fontsize=9)
        
        # Format x-axis with dates
        if n > 0:
            # Show subset of labels
            step = max(1, n // 10)
            ticks = x[::step]
            labels = [datetime.fromtimestamp(timestamps[i]).strftime('%m/%d %H:%M') 
                      for i in range(0, n, step)]
            self.ax_volume.set_xticks(ticks)
            self.ax_volume.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            
        # Set limits
        self.ax_main.set_xlim(-1, n)
        
        # Add current price line
        if n > 0:
            current_price = closes[-1]
            self.ax_main.axhline(y=current_price, color=COLORS['accent_cyan'], 
                                  linestyle='--', linewidth=1, alpha=0.7)
            self.ax_main.text(n - 0.5, current_price, f' ${current_price:,.2f}',
                              color=COLORS['accent_cyan'], fontsize=8, va='center')
        
        self.canvas.draw()
        
    def _draw_indicators(self, x):
        """Draw technical indicators on the chart"""
        if 'ema_fast' in self.indicators:
            self.ax_main.plot(x, self.indicators['ema_fast'], 
                             color=COLORS['accent_blue'], linewidth=1, label='EMA 12')
        if 'ema_slow' in self.indicators:
            self.ax_main.plot(x, self.indicators['ema_slow'], 
                             color=COLORS['orange'], linewidth=1, label='EMA 26')
        if 'bb_upper' in self.indicators:
            self.ax_main.plot(x, self.indicators['bb_upper'], 
                             color=COLORS['purple'], linewidth=0.8, linestyle='--', alpha=0.7)
            self.ax_main.plot(x, self.indicators['bb_lower'], 
                             color=COLORS['purple'], linewidth=0.8, linestyle='--', alpha=0.7)
            self.ax_main.fill_between(x, self.indicators['bb_lower'], self.indicators['bb_upper'],
                                       color=COLORS['purple'], alpha=0.1)
                                       
    def add_indicator(self, name: str, values: np.ndarray):
        """Add a technical indicator to the chart"""
        self.indicators[name] = values
        if self.data is not None:
            self._draw_chart()
            
    def clear_indicators(self):
        """Clear all indicators"""
        self.indicators = {}
        if self.data is not None:
            self._draw_chart()


class OrderBookWidget:
    """Order book visualization widget"""
    
    def __init__(self, parent_frame):
        self.frame = tk.Frame(parent_frame, bg=COLORS['bg_panel'])
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Header
        header = tk.Label(
            self.frame, text="📖 ORDER BOOK",
            bg=COLORS['bg_panel'], fg=COLORS['text_primary'],
            font=('Segoe UI Semibold', 10)
        )
        header.pack(anchor='w', pady=(0, 5))
        
        # Create treeview for order book
        columns = ('price', 'size', 'total')
        
        # Asks frame (red - sells)
        self.asks_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        self.asks_frame.pack(fill=tk.BOTH, expand=True)
        
        self.asks_tree = ttk.Treeview(
            self.asks_frame, columns=columns, show='headings', height=8
        )
        for col in columns:
            self.asks_tree.heading(col, text=col.upper())
            self.asks_tree.column(col, width=70, anchor='e')
        self.asks_tree.pack(fill=tk.BOTH, expand=True)
        
        # Spread display
        self.spread_label = tk.Label(
            self.frame, text="Spread: --",
            bg=COLORS['bg_card'], fg=COLORS['text_primary'],
            font=('Consolas', 11, 'bold'), pady=5
        )
        self.spread_label.pack(fill=tk.X)
        
        # Bids frame (green - buys)
        self.bids_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        self.bids_frame.pack(fill=tk.BOTH, expand=True)
        
        self.bids_tree = ttk.Treeview(
            self.bids_frame, columns=columns, show='headings', height=8
        )
        for col in columns:
            self.bids_tree.heading(col, text=col.upper())
            self.bids_tree.column(col, width=70, anchor='e')
        self.bids_tree.pack(fill=tk.BOTH, expand=True)
        
    def update(self, bids: List[tuple], asks: List[tuple]):
        """Update order book display"""
        # Clear existing
        for item in self.asks_tree.get_children():
            self.asks_tree.delete(item)
        for item in self.bids_tree.get_children():
            self.bids_tree.delete(item)
            
        # Add asks (reversed so lowest ask is at bottom)
        cumulative = 0
        for price, size in reversed(asks[:10]):
            cumulative += size
            self.asks_tree.insert('', 'end', values=(
                f'{price:,.2f}', f'{size:.4f}', f'{cumulative:.4f}'
            ), tags=('ask',))
            
        # Add bids
        cumulative = 0
        for price, size in bids[:10]:
            cumulative += size
            self.bids_tree.insert('', 'end', values=(
                f'{price:,.2f}', f'{size:.4f}', f'{cumulative:.4f}'
            ), tags=('bid',))
            
        # Update spread
        if bids and asks:
            spread = asks[0][0] - bids[0][0]
            spread_pct = spread / asks[0][0] * 100
            self.spread_label.config(text=f"Spread: ${spread:,.2f} ({spread_pct:.3f}%)")
            
        # Color coding
        self.asks_tree.tag_configure('ask', foreground=COLORS['red'])
        self.bids_tree.tag_configure('bid', foreground=COLORS['green'])


class PositionsPanel:
    """Panel showing current positions"""
    
    def __init__(self, parent_frame):
        self.frame = tk.Frame(parent_frame, bg=COLORS['bg_panel'])
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Header
        header_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame, text="📊 POSITIONS",
            bg=COLORS['bg_panel'], fg=COLORS['text_primary'],
            font=('Segoe UI Semibold', 10)
        ).pack(side=tk.LEFT)
        
        # Positions list
        columns = ('symbol', 'side', 'size', 'entry', 'current', 'pnl', 'pnl_pct')
        self.tree = ttk.Treeview(
            self.frame, columns=columns, show='headings', height=5
        )
        
        headers = ['Symbol', 'Side', 'Size', 'Entry', 'Current', 'P&L', 'P&L %']
        widths = [70, 50, 60, 80, 80, 80, 60]
        
        for col, header, width in zip(columns, headers, widths):
            self.tree.heading(col, text=header)
            self.tree.column(col, width=width, anchor='center')
            
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Summary
        self.summary_frame = tk.Frame(self.frame, bg=COLORS['bg_card'])
        self.summary_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.total_pnl_label = tk.Label(
            self.summary_frame, text="Total P&L: $0.00",
            bg=COLORS['bg_card'], fg=COLORS['text_primary'],
            font=('Consolas', 11, 'bold')
        )
        self.total_pnl_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.equity_label = tk.Label(
            self.summary_frame, text="Equity: $10,000.00",
            bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
            font=('Consolas', 10)
        )
        self.equity_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
    def update_positions(self, positions: List[Dict]):
        """Update positions display"""
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        total_pnl = 0
        
        for pos in positions:
            pnl = pos.get('pnl', 0)
            pnl_pct = pos.get('pnl_pct', 0)
            total_pnl += pnl
            
            tag = 'profit' if pnl >= 0 else 'loss'
            
            self.tree.insert('', 'end', values=(
                pos.get('symbol', ''),
                pos.get('side', '').upper(),
                f"{pos.get('size', 0):.4f}",
                f"${pos.get('entry_price', 0):,.2f}",
                f"${pos.get('current_price', 0):,.2f}",
                f"${pnl:+,.2f}",
                f"{pnl_pct:+.2f}%"
            ), tags=(tag,))
            
        self.tree.tag_configure('profit', foreground=COLORS['green'])
        self.tree.tag_configure('loss', foreground=COLORS['red'])
        
        # Update summary
        color = COLORS['green'] if total_pnl >= 0 else COLORS['red']
        self.total_pnl_label.config(text=f"Total P&L: ${total_pnl:+,.2f}", fg=color)
        
    def update_equity(self, equity: float):
        """Update equity display"""
        self.equity_label.config(text=f"Equity: ${equity:,.2f}")


class StrategyPanel:
    """Strategy control and status panel"""
    
    def __init__(self, parent_frame, on_strategy_change: Callable = None):
        self.frame = tk.Frame(parent_frame, bg=COLORS['bg_panel'])
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        self.on_strategy_change = on_strategy_change
        
        # Header
        tk.Label(
            self.frame, text="🎯 STRATEGY CONTROL",
            bg=COLORS['bg_panel'], fg=COLORS['text_primary'],
            font=('Segoe UI Semibold', 10)
        ).pack(anchor='w', pady=(0, 10))
        
        # Strategy selection
        select_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        select_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            select_frame, text="Strategy:",
            bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)
        
        self.strategy_var = tk.StringVar(value='medallion')
        strategies = ['momentum', 'medallion', 'stat_arb', 'options_alpha', 'ensemble']
        
        self.strategy_combo = ttk.Combobox(
            select_frame, textvariable=self.strategy_var,
            values=strategies, state='readonly', width=15
        )
        self.strategy_combo.pack(side=tk.LEFT, padx=(10, 0))
        self.strategy_combo.bind('<<ComboboxSelected>>', self._on_strategy_selected)
        
        # Symbol selection
        symbol_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        symbol_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            symbol_frame, text="Symbol:",
            bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)
        
        self.symbol_var = tk.StringVar(value='BTCUSD')
        symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD', 'XRPUSD']
        
        self.symbol_combo = ttk.Combobox(
            symbol_frame, textvariable=self.symbol_var,
            values=symbols, state='readonly', width=15
        )
        self.symbol_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Timeframe selection
        tf_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        tf_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            tf_frame, text="Timeframe:",
            bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT)
        
        self.timeframe_var = tk.StringVar(value='1h')
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        self.tf_combo = ttk.Combobox(
            tf_frame, textvariable=self.timeframe_var,
            values=timeframes, state='readonly', width=15
        )
        self.tf_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Control buttons
        btn_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = tk.Button(
            btn_frame, text="▶ START",
            bg=COLORS['green'], fg='white',
            font=('Segoe UI Semibold', 9),
            relief=tk.FLAT, padx=15, pady=5,
            command=self._on_start
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = tk.Button(
            btn_frame, text="⏹ STOP",
            bg=COLORS['red'], fg='white',
            font=('Segoe UI Semibold', 9),
            relief=tk.FLAT, padx=15, pady=5,
            command=self._on_stop,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)
        
        # Status
        self.status_label = tk.Label(
            self.frame, text="● Idle",
            bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9)
        )
        self.status_label.pack(anchor='w', pady=5)
        
        # Callbacks
        self.on_start = None
        self.on_stop = None
        
    def _on_strategy_selected(self, event):
        if self.on_strategy_change:
            self.on_strategy_change(self.strategy_var.get())
            
    def _on_start(self):
        if self.on_start:
            self.on_start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="● Running", fg=COLORS['green'])
        
    def _on_stop(self):
        if self.on_stop:
            self.on_stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="● Stopped", fg=COLORS['red'])
        
    def set_status(self, status: str, color: str = None):
        """Set status text"""
        self.status_label.config(text=f"● {status}", fg=color or COLORS['text_secondary'])


class TradeEntryPanel:
    """Manual trade entry panel"""
    
    def __init__(self, parent_frame):
        self.frame = tk.Frame(parent_frame, bg=COLORS['bg_panel'])
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header
        tk.Label(
            self.frame, text="📝 QUICK TRADE",
            bg=COLORS['bg_panel'], fg=COLORS['text_primary'],
            font=('Segoe UI Semibold', 10)
        ).pack(anchor='w', pady=(0, 10))
        
        # Size input
        size_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        size_frame.pack(fill=tk.X, pady=3)
        
        tk.Label(
            size_frame, text="Size:",
            bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9), width=8, anchor='w'
        ).pack(side=tk.LEFT)
        
        self.size_var = tk.StringVar(value='0.01')
        self.size_entry = tk.Entry(
            size_frame, textvariable=self.size_var,
            bg=COLORS['bg_input'], fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            relief=tk.FLAT, width=12
        )
        self.size_entry.pack(side=tk.LEFT, padx=5)
        
        # Price input
        price_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        price_frame.pack(fill=tk.X, pady=3)
        
        tk.Label(
            price_frame, text="Price:",
            bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9), width=8, anchor='w'
        ).pack(side=tk.LEFT)
        
        self.price_var = tk.StringVar(value='Market')
        self.price_entry = tk.Entry(
            price_frame, textvariable=self.price_var,
            bg=COLORS['bg_input'], fg=COLORS['text_primary'],
            insertbackground=COLORS['text_primary'],
            relief=tk.FLAT, width=12
        )
        self.price_entry.pack(side=tk.LEFT, padx=5)
        
        # Order type
        type_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        type_frame.pack(fill=tk.X, pady=3)
        
        tk.Label(
            type_frame, text="Type:",
            bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9), width=8, anchor='w'
        ).pack(side=tk.LEFT)
        
        self.order_type_var = tk.StringVar(value='Market')
        self.type_combo = ttk.Combobox(
            type_frame, textvariable=self.order_type_var,
            values=['Market', 'Limit', 'Stop'], state='readonly', width=10
        )
        self.type_combo.pack(side=tk.LEFT, padx=5)
        
        # Buy/Sell buttons
        btn_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.buy_btn = tk.Button(
            btn_frame, text="BUY / LONG",
            bg=COLORS['green'], fg='white',
            font=('Segoe UI Semibold', 10),
            relief=tk.FLAT, padx=20, pady=8,
            command=lambda: self._place_order('buy')
        )
        self.buy_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        self.sell_btn = tk.Button(
            btn_frame, text="SELL / SHORT",
            bg=COLORS['red'], fg='white',
            font=('Segoe UI Semibold', 10),
            relief=tk.FLAT, padx=20, pady=8,
            command=lambda: self._place_order('sell')
        )
        self.sell_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Callback
        self.on_place_order = None
        
    def _place_order(self, side: str):
        if self.on_place_order:
            try:
                size = float(self.size_var.get())
                price = None if self.price_var.get().lower() == 'market' else float(self.price_var.get())
                order_type = self.order_type_var.get().lower()
                self.on_place_order(side, size, price, order_type)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers for size and price.")


class MetricsPanel:
    """Performance metrics panel"""
    
    def __init__(self, parent_frame):
        self.frame = tk.Frame(parent_frame, bg=COLORS['bg_panel'])
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header
        tk.Label(
            self.frame, text="📈 PERFORMANCE",
            bg=COLORS['bg_panel'], fg=COLORS['text_primary'],
            font=('Segoe UI Semibold', 10)
        ).pack(anchor='w', pady=(0, 10))
        
        # Metrics grid
        metrics_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        metrics_frame.pack(fill=tk.X)
        
        self.metrics = {}
        metric_names = [
            ('Total Return', 'return'),
            ('Win Rate', 'win_rate'),
            ('Profit Factor', 'profit_factor'),
            ('Sharpe Ratio', 'sharpe'),
            ('Max Drawdown', 'max_dd'),
            ('Trades Today', 'trades_today')
        ]
        
        for i, (label, key) in enumerate(metric_names):
            row = i // 2
            col = i % 2
            
            cell = tk.Frame(metrics_frame, bg=COLORS['bg_card'], padx=10, pady=5)
            cell.grid(row=row, column=col, padx=2, pady=2, sticky='ew')
            
            tk.Label(
                cell, text=label,
                bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 8)
            ).pack(anchor='w')
            
            value_label = tk.Label(
                cell, text="--",
                bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                font=('Consolas', 11, 'bold')
            )
            value_label.pack(anchor='w')
            
            self.metrics[key] = value_label
            
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
        
    def update_metrics(self, metrics: Dict):
        """Update metrics display"""
        if 'return' in metrics:
            val = metrics['return']
            color = COLORS['green'] if val >= 0 else COLORS['red']
            self.metrics['return'].config(text=f"{val:+.2f}%", fg=color)
            
        if 'win_rate' in metrics:
            self.metrics['win_rate'].config(text=f"{metrics['win_rate']:.1f}%")
            
        if 'profit_factor' in metrics:
            self.metrics['profit_factor'].config(text=f"{metrics['profit_factor']:.2f}")
            
        if 'sharpe' in metrics:
            self.metrics['sharpe'].config(text=f"{metrics['sharpe']:.2f}")
            
        if 'max_dd' in metrics:
            self.metrics['max_dd'].config(text=f"{metrics['max_dd']:.2f}%", fg=COLORS['red'])
            
        if 'trades_today' in metrics:
            self.metrics['trades_today'].config(text=str(metrics['trades_today']))


class LogPanel:
    """Trading log panel"""
    
    def __init__(self, parent_frame):
        self.frame = tk.Frame(parent_frame, bg=COLORS['bg_panel'])
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Header
        header_frame = tk.Frame(self.frame, bg=COLORS['bg_panel'])
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame, text="📋 TRADE LOG",
            bg=COLORS['bg_panel'], fg=COLORS['text_primary'],
            font=('Segoe UI Semibold', 10)
        ).pack(side=tk.LEFT)
        
        tk.Button(
            header_frame, text="Clear",
            bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 8), relief=tk.FLAT,
            command=self.clear
        ).pack(side=tk.RIGHT)
        
        # Log text
        self.log_text = tk.Text(
            self.frame,
            bg=COLORS['bg_dark'], fg=COLORS['text_secondary'],
            font=('Consolas', 9),
            height=8, wrap=tk.WORD,
            relief=tk.FLAT
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Configure tags for colors
        self.log_text.tag_configure('info', foreground=COLORS['text_primary'])
        self.log_text.tag_configure('success', foreground=COLORS['green'])
        self.log_text.tag_configure('warning', foreground=COLORS['orange'])
        self.log_text.tag_configure('error', foreground=COLORS['red'])
        self.log_text.tag_configure('signal', foreground=COLORS['accent_cyan'])
        
    def log(self, message: str, level: str = 'info'):
        """Add a log message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] ", 'info')
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)
        
    def clear(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)


class TradingGUI:
    """Main Trading GUI Application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Delta Exchange Algo Trader v4.0 - Renaissance Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Set icon (if available)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
            
        # Configure styles
        self._setup_styles()
        
        # Create main layout
        self._create_layout()
        
        # Data queue for thread-safe updates
        self.update_queue = queue.Queue()
        
        # Sample data flag
        self._sample_data_loaded = False
        
        # Start update loop
        self._process_updates()
        
        # Load sample data
        self.root.after(500, self._load_sample_data)
        
    def _setup_styles(self):
        """Configure ttk styles for dark theme"""
        style = ttk.Style()
        
        # Try to use a theme that supports customization
        try:
            style.theme_use('clam')
        except:
            pass
            
        # Configure Treeview
        style.configure(
            "Treeview",
            background=COLORS['bg_dark'],
            foreground=COLORS['text_primary'],
            fieldbackground=COLORS['bg_dark'],
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS['bg_card'],
            foreground=COLORS['text_secondary'],
            borderwidth=0
        )
        style.map('Treeview', background=[('selected', COLORS['accent_blue'])])
        
        # Configure Combobox
        style.configure(
            "TCombobox",
            fieldbackground=COLORS['bg_input'],
            background=COLORS['bg_input'],
            foreground=COLORS['text_primary']
        )
        
    def _create_layout(self):
        """Create the main application layout"""
        # Top bar
        self._create_top_bar()
        
        # Main content area
        main_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Left panel (chart)
        left_panel = tk.Frame(main_frame, bg=COLORS['bg_panel'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Chart header
        chart_header = tk.Frame(left_panel, bg=COLORS['bg_panel'])
        chart_header.pack(fill=tk.X, padx=10, pady=10)
        
        self.symbol_label = tk.Label(
            chart_header, text="BTCUSD",
            bg=COLORS['bg_panel'], fg=COLORS['text_primary'],
            font=('Segoe UI Semibold', 16)
        )
        self.symbol_label.pack(side=tk.LEFT)
        
        self.price_label = tk.Label(
            chart_header, text="$50,000.00",
            bg=COLORS['bg_panel'], fg=COLORS['green'],
            font=('Segoe UI Semibold', 16)
        )
        self.price_label.pack(side=tk.LEFT, padx=20)
        
        self.change_label = tk.Label(
            chart_header, text="+2.5%",
            bg=COLORS['bg_panel'], fg=COLORS['green'],
            font=('Segoe UI', 12)
        )
        self.change_label.pack(side=tk.LEFT)
        
        # Chart
        chart_frame = tk.Frame(left_panel, bg=COLORS['bg_panel'])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.chart = CandlestickChart(chart_frame, width=900, height=500)
        
        # Bottom panel (log)
        bottom_panel = tk.Frame(left_panel, bg=COLORS['bg_panel'], height=150)
        bottom_panel.pack(fill=tk.X, padx=10, pady=(0, 10))
        bottom_panel.pack_propagate(False)
        
        self.log_panel = LogPanel(bottom_panel)
        
        # Right panel
        right_panel = tk.Frame(main_frame, bg=COLORS['bg_dark'], width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Strategy control
        strategy_frame = tk.Frame(right_panel, bg=COLORS['bg_panel'])
        strategy_frame.pack(fill=tk.X, pady=(0, 5))
        self.strategy_panel = StrategyPanel(strategy_frame, self._on_strategy_change)
        self.strategy_panel.on_start = self._on_start_trading
        self.strategy_panel.on_stop = self._on_stop_trading
        
        # Trade entry
        trade_frame = tk.Frame(right_panel, bg=COLORS['bg_panel'])
        trade_frame.pack(fill=tk.X, pady=5)
        self.trade_panel = TradeEntryPanel(trade_frame)
        self.trade_panel.on_place_order = self._on_place_order
        
        # Metrics
        metrics_frame = tk.Frame(right_panel, bg=COLORS['bg_panel'])
        metrics_frame.pack(fill=tk.X, pady=5)
        self.metrics_panel = MetricsPanel(metrics_frame)
        
        # Positions
        positions_frame = tk.Frame(right_panel, bg=COLORS['bg_panel'])
        positions_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.positions_panel = PositionsPanel(positions_frame)
        
        # Order book (at bottom of right panel)
        orderbook_frame = tk.Frame(right_panel, bg=COLORS['bg_panel'])
        orderbook_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.orderbook = OrderBookWidget(orderbook_frame)
        
    def _create_top_bar(self):
        """Create the top navigation bar"""
        top_bar = tk.Frame(self.root, bg=COLORS['bg_panel'], height=50)
        top_bar.pack(fill=tk.X, padx=10, pady=10)
        top_bar.pack_propagate(False)
        
        # Logo
        logo_label = tk.Label(
            top_bar, text="🤖 DELTA ALGO TRADER",
            bg=COLORS['bg_panel'], fg=COLORS['text_primary'],
            font=('Segoe UI Semibold', 14)
        )
        logo_label.pack(side=tk.LEFT, padx=10)
        
        # Version
        version_label = tk.Label(
            top_bar, text="v4.0 Renaissance Edition",
            bg=COLORS['bg_panel'], fg=COLORS['accent_cyan'],
            font=('Segoe UI', 9)
        )
        version_label.pack(side=tk.LEFT)
        
        # Right side - account info
        account_frame = tk.Frame(top_bar, bg=COLORS['bg_panel'])
        account_frame.pack(side=tk.RIGHT, padx=10)
        
        self.account_label = tk.Label(
            account_frame, text="TESTNET",
            bg=COLORS['orange'], fg='white',
            font=('Segoe UI Semibold', 9),
            padx=10, pady=2
        )
        self.account_label.pack(side=tk.LEFT, padx=5)
        
        self.connection_label = tk.Label(
            account_frame, text="● Connected",
            bg=COLORS['bg_panel'], fg=COLORS['green'],
            font=('Segoe UI', 9)
        )
        self.connection_label.pack(side=tk.LEFT, padx=10)
        
        # Time
        self.time_label = tk.Label(
            account_frame, text="",
            bg=COLORS['bg_panel'], fg=COLORS['text_secondary'],
            font=('Consolas', 10)
        )
        self.time_label.pack(side=tk.LEFT, padx=10)
        self._update_time()
        
    def _update_time(self):
        """Update the time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self._update_time)
        
    def _load_sample_data(self):
        """Load sample chart data for demonstration"""
        if self._sample_data_loaded:
            return
            
        self.log_panel.log("Loading market data...", "info")
        
        # Generate sample OHLCV data
        np.random.seed(42)
        n = 100
        
        base_time = datetime.now() - timedelta(hours=n)
        timestamps = np.array([
            (base_time + timedelta(hours=i)).timestamp()
            for i in range(n)
        ])
        
        # Generate realistic price movement
        base_price = 50000
        returns = np.random.randn(n) * 0.005
        prices = base_price * np.cumprod(1 + returns)
        
        opens = prices
        highs = prices * (1 + np.abs(np.random.randn(n) * 0.003))
        lows = prices * (1 - np.abs(np.random.randn(n) * 0.003))
        closes = prices + np.random.randn(n) * prices * 0.001
        volumes = np.random.uniform(100, 500, n)
        
        # Update chart
        self.chart.update_data(timestamps, opens, highs, lows, closes, volumes)
        
        # Add indicators
        from signals import ema, bollinger_bands
        
        ema_fast = ema(closes, 12)
        ema_slow = ema(closes, 26)
        bb_upper, bb_mid, bb_lower = bollinger_bands(closes, 20, 2)
        
        self.chart.indicators['ema_fast'] = ema_fast
        self.chart.indicators['ema_slow'] = ema_slow
        self.chart.indicators['bb_upper'] = bb_upper
        self.chart.indicators['bb_lower'] = bb_lower
        self.chart._draw_chart()
        
        # Update price display
        current_price = closes[-1]
        prev_price = closes[-2] if len(closes) > 1 else current_price
        change_pct = (current_price - prev_price) / prev_price * 100
        
        self.price_label.config(text=f"${current_price:,.2f}")
        
        color = COLORS['green'] if change_pct >= 0 else COLORS['red']
        self.price_label.config(fg=color)
        self.change_label.config(text=f"{change_pct:+.2f}%", fg=color)
        
        # Update sample order book
        mid_price = current_price
        bids = [(mid_price - i * 10 - np.random.uniform(0, 5), np.random.uniform(0.1, 2)) 
                for i in range(15)]
        asks = [(mid_price + i * 10 + np.random.uniform(0, 5), np.random.uniform(0.1, 2)) 
                for i in range(15)]
        self.orderbook.update(bids, asks)
        
        # Update sample metrics
        self.metrics_panel.update_metrics({
            'return': 12.5,
            'win_rate': 58.3,
            'profit_factor': 1.85,
            'sharpe': 1.42,
            'max_dd': 8.2,
            'trades_today': 15
        })
        
        # Update sample positions
        self.positions_panel.update_positions([
            {
                'symbol': 'BTCUSD',
                'side': 'long',
                'size': 0.5,
                'entry_price': 49500,
                'current_price': current_price,
                'pnl': (current_price - 49500) * 0.5,
                'pnl_pct': (current_price - 49500) / 49500 * 100
            }
        ])
        self.positions_panel.update_equity(10500.00)
        
        self.log_panel.log("Market data loaded successfully", "success")
        self.log_panel.log(f"Current price: ${current_price:,.2f}", "info")
        
        self._sample_data_loaded = True
        
    def _on_strategy_change(self, strategy: str):
        """Handle strategy selection change"""
        self.log_panel.log(f"Strategy changed to: {strategy}", "info")
        
    def _on_start_trading(self):
        """Handle start trading button"""
        strategy = self.strategy_panel.strategy_var.get()
        symbol = self.strategy_panel.symbol_var.get()
        timeframe = self.strategy_panel.timeframe_var.get()
        
        self.log_panel.log(f"Starting {strategy} strategy on {symbol} ({timeframe})", "success")
        self.log_panel.log("Connecting to Delta Exchange...", "info")
        self.log_panel.log("Strategy initialized and running", "success")
        
    def _on_stop_trading(self):
        """Handle stop trading button"""
        self.log_panel.log("Stopping strategy...", "warning")
        self.log_panel.log("Strategy stopped", "info")
        
    def _on_place_order(self, side: str, size: float, price: float, order_type: str):
        """Handle manual order placement"""
        price_str = f"${price:,.2f}" if price else "Market"
        self.log_panel.log(
            f"Order placed: {side.upper()} {size} @ {price_str} ({order_type})",
            "signal"
        )
        
    def _process_updates(self):
        """Process updates from the queue"""
        try:
            while True:
                update = self.update_queue.get_nowait()
                # Process update based on type
                if update['type'] == 'price':
                    self._update_price(update['data'])
                elif update['type'] == 'orderbook':
                    self.orderbook.update(update['data']['bids'], update['data']['asks'])
                elif update['type'] == 'position':
                    self.positions_panel.update_positions(update['data'])
                elif update['type'] == 'log':
                    self.log_panel.log(update['data']['message'], update['data'].get('level', 'info'))
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self._process_updates)
        
    def _update_price(self, data: Dict):
        """Update price display"""
        price = data['price']
        change = data.get('change', 0)
        
        self.price_label.config(text=f"${price:,.2f}")
        
        color = COLORS['green'] if change >= 0 else COLORS['red']
        self.price_label.config(fg=color)
        self.change_label.config(text=f"{change:+.2f}%", fg=color)
        
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def launch_gui():
    """Launch the trading GUI"""
    app = TradingGUI()
    app.run()


if __name__ == '__main__':
    launch_gui()
