"""
Professional Real-Time Trading GUI for Delta Exchange
Connects to live market data via Delta Exchange API
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ==================== Color Theme ====================
class Theme:
    """Professional dark trading terminal theme"""
    BG_PRIMARY = "#0d1117"
    BG_SECONDARY = "#161b22"
    BG_TERTIARY = "#21262d"
    TEXT_PRIMARY = "#e6edf3"
    TEXT_SECONDARY = "#8b949e"
    TEXT_MUTED = "#6e7681"
    ACCENT = "#58a6ff"
    SUCCESS = "#3fb950"
    DANGER = "#f85149"
    WARNING = "#d29922"
    BORDER = "#30363d"
    CANDLE_UP = "#00C853"
    CANDLE_DOWN = "#FF1744"
    VOLUME_UP = "#00C85355"
    VOLUME_DOWN = "#FF174455"


# ==================== Real-Time Data Fetcher ====================
class RealTimeDataFetcher:
    """Fetches real market data from Delta Exchange API"""
    
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        if testnet:
            self.base_url = "https://cdn-ind.testnet.deltaex.org"
        else:
            self.base_url = "https://api.india.delta.exchange"
        
        self.session = requests.Session() if HAS_REQUESTS else None
        self.symbols_cache = {}
        self.product_id_map = {}
    
    def get_products(self) -> List[Dict]:
        """Get all available products"""
        try:
            resp = self.session.get(f"{self.base_url}/v2/products", timeout=10)
            data = resp.json()
            if "result" in data:
                products = data["result"]
                # Cache symbol to product ID mapping
                for p in products:
                    self.product_id_map[p.get("symbol", "")] = p.get("id")
                return products
            return []
        except Exception as e:
            print(f"Error fetching products: {e}")
            return []
    
    def get_tickers(self) -> Dict[str, Dict]:
        """Get all tickers with current prices"""
        try:
            resp = self.session.get(f"{self.base_url}/v2/tickers", timeout=10)
            data = resp.json()
            if "result" in data:
                return {t["symbol"]: t for t in data["result"]}
            return {}
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker for specific symbol"""
        try:
            resp = self.session.get(
                f"{self.base_url}/v2/tickers/{symbol}",
                timeout=10
            )
            data = resp.json()
            if "result" in data:
                return data["result"]
            return None
        except Exception as e:
            print(f"Error fetching ticker {symbol}: {e}")
            return None
    
    def get_candles(
        self,
        symbol: str,
        resolution: int = 60,
        limit: int = 200
    ) -> List[Dict]:
        """
        Get OHLCV candles
        
        Args:
            symbol: Trading pair symbol
            resolution: Candle resolution in seconds (60=1m, 300=5m, 900=15m, 3600=1h)
            limit: Number of candles to fetch
        """
        try:
            end_time = int(time.time())
            start_time = end_time - (resolution * limit)
            
            resp = self.session.get(
                f"{self.base_url}/v2/history/candles",
                params={
                    "symbol": symbol,
                    "resolution": resolution,
                    "start": start_time,
                    "end": end_time
                },
                timeout=15
            )
            data = resp.json()
            
            if "result" in data and data["result"]:
                candles = []
                for c in data["result"]:
                    candles.append({
                        "time": c.get("time", c.get("t", 0)),
                        "open": float(c.get("open", c.get("o", 0))),
                        "high": float(c.get("high", c.get("h", 0))),
                        "low": float(c.get("low", c.get("l", 0))),
                        "close": float(c.get("close", c.get("c", 0))),
                        "volume": float(c.get("volume", c.get("v", 0)))
                    })
                # Sort by time
                candles.sort(key=lambda x: x["time"])
                return candles
            return []
        except Exception as e:
            print(f"Error fetching candles for {symbol}: {e}")
            return []
    
    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """Get order book"""
        try:
            product_id = self.product_id_map.get(symbol)
            if not product_id:
                # Try to find product ID
                products = self.get_products()
                for p in products:
                    if p.get("symbol") == symbol:
                        product_id = p.get("id")
                        break
            
            if not product_id:
                return {"buy": [], "sell": []}
            
            resp = self.session.get(
                f"{self.base_url}/v2/l2orderbook/{product_id}",
                params={"depth": depth},
                timeout=10
            )
            data = resp.json()
            
            if "result" in data:
                result = data["result"]
                return {
                    "buy": result.get("buy", []),
                    "sell": result.get("sell", [])
                }
            return {"buy": [], "sell": []}
        except Exception as e:
            print(f"Error fetching orderbook for {symbol}: {e}")
            return {"buy": [], "sell": []}
    
    def get_recent_trades(self, symbol: str) -> List[Dict]:
        """Get recent trades"""
        try:
            product_id = self.product_id_map.get(symbol)
            if not product_id:
                return []
            
            resp = self.session.get(
                f"{self.base_url}/v2/trades/{product_id}",
                timeout=10
            )
            data = resp.json()
            
            if "result" in data:
                return data["result"]
            return []
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {e}")
            return []


# ==================== Candlestick Chart ====================
class RealTimeCandlestickChart:
    """Professional candlestick chart with real-time updates"""
    
    def __init__(self, parent, width: int = 800, height: int = 500):
        self.parent = parent
        self.width = width
        self.height = height
        
        self.frame = tk.Frame(parent, bg=Theme.BG_PRIMARY)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        if not HAS_MATPLOTLIB:
            label = tk.Label(
                self.frame,
                text="Matplotlib required for charts",
                bg=Theme.BG_PRIMARY,
                fg=Theme.TEXT_PRIMARY,
                font=("Consolas", 14)
            )
            label.pack(expand=True)
            return
        
        # Create figure with dark theme
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor=Theme.BG_PRIMARY)
        
        # Main price chart (80% height)
        self.ax_price = self.fig.add_axes([0.08, 0.25, 0.88, 0.70])
        # Volume chart (20% height)
        self.ax_volume = self.fig.add_axes([0.08, 0.05, 0.88, 0.18])
        
        self._setup_axes()
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Data storage
        self.candles = []
        self.symbol = ""
        self.last_price = 0.0
    
    def _setup_axes(self):
        """Configure chart axes"""
        for ax in [self.ax_price, self.ax_volume]:
            ax.set_facecolor(Theme.BG_PRIMARY)
            ax.tick_params(colors=Theme.TEXT_SECONDARY, labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(Theme.BORDER)
            ax.spines['left'].set_color(Theme.BORDER)
            ax.grid(True, alpha=0.2, color=Theme.BORDER, linestyle='--')
        
        self.ax_volume.set_xlabel('')
        self.ax_price.set_ylabel('Price', color=Theme.TEXT_SECONDARY, fontsize=9)
        self.ax_volume.set_ylabel('Vol', color=Theme.TEXT_SECONDARY, fontsize=9)
    
    def update(self, candles: List[Dict], symbol: str = ""):
        """Update chart with new candle data"""
        if not HAS_MATPLOTLIB or not candles:
            return
        
        self.candles = candles
        self.symbol = symbol
        
        # Clear axes
        self.ax_price.clear()
        self.ax_volume.clear()
        self._setup_axes()
        
        # Prepare data
        n = len(candles)
        x = list(range(n))
        
        opens = [c["open"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]
        
        if closes:
            self.last_price = closes[-1]
        
        # Draw candlesticks
        for i in range(n):
            color = Theme.CANDLE_UP if closes[i] >= opens[i] else Theme.CANDLE_DOWN
            
            # Wick
            self.ax_price.plot(
                [i, i], [lows[i], highs[i]],
                color=color, linewidth=1
            )
            
            # Body
            body_bottom = min(opens[i], closes[i])
            body_height = abs(closes[i] - opens[i])
            if body_height < 0.001:
                body_height = 0.001
            
            rect = Rectangle(
                (i - 0.3, body_bottom),
                0.6, body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=1
            )
            self.ax_price.add_patch(rect)
            
            # Volume bar
            vol_color = Theme.VOLUME_UP if closes[i] >= opens[i] else Theme.VOLUME_DOWN
            self.ax_volume.bar(i, volumes[i], color=vol_color, width=0.6)
        
        # Add moving averages
        if n >= 20:
            ma20 = self._calculate_ma(closes, 20)
            self.ax_price.plot(
                range(19, n), ma20[19:],
                color=Theme.ACCENT, linewidth=1.2, alpha=0.8,
                label='MA20'
            )
        
        if n >= 50:
            ma50 = self._calculate_ma(closes, 50)
            self.ax_price.plot(
                range(49, n), ma50[49:],
                color=Theme.WARNING, linewidth=1.2, alpha=0.8,
                label='MA50'
            )
        
        # Add current price line
        if closes:
            self.ax_price.axhline(
                y=closes[-1],
                color=Theme.SUCCESS if closes[-1] >= opens[-1] else Theme.DANGER,
                linestyle='--', linewidth=1, alpha=0.7
            )
            
            # Price label
            self.ax_price.annotate(
                f'${closes[-1]:,.2f}',
                xy=(n - 1, closes[-1]),
                xytext=(n + 1, closes[-1]),
                fontsize=9,
                color=Theme.TEXT_PRIMARY,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=Theme.BG_TERTIARY,
                    edgecolor=Theme.BORDER
                )
            )
        
        # Set limits
        if n > 0:
            self.ax_price.set_xlim(-1, n + 5)
            self.ax_volume.set_xlim(-1, n + 5)
            
            price_range = max(highs) - min(lows)
            self.ax_price.set_ylim(
                min(lows) - price_range * 0.05,
                max(highs) + price_range * 0.1
            )
        
        # Title
        if symbol:
            change = 0
            if len(closes) >= 2:
                change = ((closes[-1] - closes[0]) / closes[0]) * 100
            
            title_color = Theme.SUCCESS if change >= 0 else Theme.DANGER
            self.ax_price.set_title(
                f'{symbol}  ${closes[-1]:,.2f}  ({change:+.2f}%)',
                color=title_color, fontsize=12, fontweight='bold',
                loc='left', pad=10
            )
        
        # Legend
        self.ax_price.legend(loc='upper left', fontsize=8, framealpha=0.5)
        
        # Hide x-axis labels on price chart
        self.ax_price.set_xticklabels([])
        
        # Format volume y-axis
        self.ax_volume.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}')
        )
        
        self.canvas.draw()
    
    def _calculate_ma(self, data: List[float], period: int) -> List[float]:
        """Calculate moving average"""
        ma = []
        for i in range(len(data)):
            if i < period - 1:
                ma.append(data[i])
            else:
                ma.append(sum(data[i-period+1:i+1]) / period)
        return ma


# ==================== Order Book Panel ====================
class OrderBookPanel:
    """Real-time order book display"""
    
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=Theme.BG_SECONDARY)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title
        title = tk.Label(
            self.frame,
            text="📊 ORDER BOOK",
            bg=Theme.BG_SECONDARY,
            fg=Theme.ACCENT,
            font=("Consolas", 11, "bold")
        )
        title.pack(pady=(5, 10))
        
        # Headers
        header_frame = tk.Frame(self.frame, bg=Theme.BG_SECONDARY)
        header_frame.pack(fill=tk.X, padx=5)
        
        tk.Label(
            header_frame, text="Price", bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY, font=("Consolas", 9), width=12, anchor='e'
        ).pack(side=tk.LEFT)
        
        tk.Label(
            header_frame, text="Size", bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY, font=("Consolas", 9), width=10, anchor='e'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            header_frame, text="Total", bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY, font=("Consolas", 9), width=12, anchor='e'
        ).pack(side=tk.LEFT)
        
        # Asks (sells) - displayed above
        self.asks_frame = tk.Frame(self.frame, bg=Theme.BG_SECONDARY)
        self.asks_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Spread label
        self.spread_label = tk.Label(
            self.frame,
            text="Spread: --",
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            font=("Consolas", 10, "bold"),
            pady=5
        )
        self.spread_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Bids (buys) - displayed below
        self.bids_frame = tk.Frame(self.frame, bg=Theme.BG_SECONDARY)
        self.bids_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        self.ask_labels = []
        self.bid_labels = []
        self._create_rows()
    
    def _create_rows(self, num_rows: int = 10):
        """Create order book rows"""
        # Clear existing
        for widget in self.asks_frame.winfo_children():
            widget.destroy()
        for widget in self.bids_frame.winfo_children():
            widget.destroy()
        
        self.ask_labels = []
        self.bid_labels = []
        
        # Create ask rows (reversed - highest at top)
        for i in range(num_rows):
            row = tk.Frame(self.asks_frame, bg=Theme.BG_SECONDARY)
            row.pack(fill=tk.X, pady=1)
            
            price_lbl = tk.Label(
                row, text="--", bg=Theme.BG_SECONDARY,
                fg=Theme.DANGER, font=("Consolas", 9), width=12, anchor='e'
            )
            price_lbl.pack(side=tk.LEFT)
            
            size_lbl = tk.Label(
                row, text="--", bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_PRIMARY, font=("Consolas", 9), width=10, anchor='e'
            )
            size_lbl.pack(side=tk.LEFT, padx=5)
            
            total_lbl = tk.Label(
                row, text="--", bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_SECONDARY, font=("Consolas", 9), width=12, anchor='e'
            )
            total_lbl.pack(side=tk.LEFT)
            
            self.ask_labels.append((price_lbl, size_lbl, total_lbl))
        
        # Create bid rows
        for i in range(num_rows):
            row = tk.Frame(self.bids_frame, bg=Theme.BG_SECONDARY)
            row.pack(fill=tk.X, pady=1)
            
            price_lbl = tk.Label(
                row, text="--", bg=Theme.BG_SECONDARY,
                fg=Theme.SUCCESS, font=("Consolas", 9), width=12, anchor='e'
            )
            price_lbl.pack(side=tk.LEFT)
            
            size_lbl = tk.Label(
                row, text="--", bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_PRIMARY, font=("Consolas", 9), width=10, anchor='e'
            )
            size_lbl.pack(side=tk.LEFT, padx=5)
            
            total_lbl = tk.Label(
                row, text="--", bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_SECONDARY, font=("Consolas", 9), width=12, anchor='e'
            )
            total_lbl.pack(side=tk.LEFT)
            
            self.bid_labels.append((price_lbl, size_lbl, total_lbl))
    
    def update(self, orderbook: Dict):
        """Update order book display"""
        bids = orderbook.get("buy", [])
        asks = orderbook.get("sell", [])
        
        # Sort: bids descending, asks ascending
        bids = sorted(bids, key=lambda x: float(x.get("price", 0)), reverse=True)[:10]
        asks = sorted(asks, key=lambda x: float(x.get("price", 0)))[:10]
        
        # Update asks (reversed for display)
        asks_reversed = list(reversed(asks))
        for i, labels in enumerate(self.ask_labels):
            if i < len(asks_reversed):
                order = asks_reversed[i]
                price = float(order.get("price", 0))
                size = float(order.get("size", 0))
                labels[0].config(text=f"${price:,.2f}")
                labels[1].config(text=f"{size:,.4f}")
                labels[2].config(text=f"${price * size:,.2f}")
            else:
                labels[0].config(text="--")
                labels[1].config(text="--")
                labels[2].config(text="--")
        
        # Update bids
        for i, labels in enumerate(self.bid_labels):
            if i < len(bids):
                order = bids[i]
                price = float(order.get("price", 0))
                size = float(order.get("size", 0))
                labels[0].config(text=f"${price:,.2f}")
                labels[1].config(text=f"{size:,.4f}")
                labels[2].config(text=f"${price * size:,.2f}")
            else:
                labels[0].config(text="--")
                labels[1].config(text="--")
                labels[2].config(text="--")
        
        # Update spread
        if bids and asks:
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 0))
            spread = best_ask - best_bid
            spread_pct = (spread / best_ask) * 100 if best_ask > 0 else 0
            self.spread_label.config(
                text=f"Spread: ${spread:.2f} ({spread_pct:.3f}%)"
            )


# ==================== Market Ticker Panel ====================
class MarketTickerPanel:
    """Live market ticker display"""
    
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=Theme.BG_SECONDARY)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title
        title = tk.Label(
            self.frame,
            text="📈 LIVE MARKETS",
            bg=Theme.BG_SECONDARY,
            fg=Theme.ACCENT,
            font=("Consolas", 11, "bold")
        )
        title.pack(pady=(5, 10))
        
        # Ticker list with scrollbar
        list_frame = tk.Frame(self.frame, bg=Theme.BG_SECONDARY)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.ticker_listbox = tk.Listbox(
            list_frame,
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            font=("Consolas", 9),
            selectbackground=Theme.ACCENT,
            selectforeground=Theme.TEXT_PRIMARY,
            highlightthickness=0,
            bd=0,
            yscrollcommand=scrollbar.set,
            height=15
        )
        self.ticker_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.ticker_listbox.yview)
        
        self.tickers = {}
        self.on_select_callback = None
    
    def set_on_select(self, callback):
        """Set callback for ticker selection"""
        self.on_select_callback = callback
        self.ticker_listbox.bind('<<ListboxSelect>>', self._on_select)
    
    def _on_select(self, event):
        """Handle ticker selection"""
        selection = self.ticker_listbox.curselection()
        if selection and self.on_select_callback:
            index = selection[0]
            items = list(self.tickers.keys())
            if index < len(items):
                symbol = items[index]
                self.on_select_callback(symbol)
    
    def update(self, tickers: Dict[str, Dict]):
        """Update ticker display"""
        self.tickers = tickers
        self.ticker_listbox.delete(0, tk.END)
        
        # Sort by volume
        sorted_tickers = sorted(
            tickers.items(),
            key=lambda x: float(x[1].get("volume", 0)),
            reverse=True
        )
        
        for symbol, data in sorted_tickers[:30]:
            price = float(data.get("close", data.get("mark_price", 0)))
            change = float(data.get("price_change_percent_24h", 0))
            
            # Format display
            if price >= 1000:
                price_str = f"${price:,.0f}"
            elif price >= 1:
                price_str = f"${price:,.2f}"
            else:
                price_str = f"${price:.6f}"
            
            change_str = f"{change:+.2f}%"
            
            # Color indicator
            indicator = "🟢" if change >= 0 else "🔴"
            
            line = f"{indicator} {symbol:<15} {price_str:>12} {change_str:>8}"
            self.ticker_listbox.insert(tk.END, line)


# ==================== Trade Panel ====================
class TradePanel:
    """Quick trade panel"""
    
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=Theme.BG_SECONDARY)
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Title
        title = tk.Label(
            self.frame,
            text="⚡ QUICK TRADE",
            bg=Theme.BG_SECONDARY,
            fg=Theme.ACCENT,
            font=("Consolas", 11, "bold")
        )
        title.pack(pady=(5, 10))
        
        # Symbol display
        self.symbol_label = tk.Label(
            self.frame,
            text="BTCUSD",
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            font=("Consolas", 14, "bold"),
            pady=10
        )
        self.symbol_label.pack(fill=tk.X, padx=10)
        
        # Price display
        self.price_label = tk.Label(
            self.frame,
            text="$0.00",
            bg=Theme.BG_SECONDARY,
            fg=Theme.SUCCESS,
            font=("Consolas", 20, "bold")
        )
        self.price_label.pack(pady=10)
        
        # Amount input
        input_frame = tk.Frame(self.frame, bg=Theme.BG_SECONDARY)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            input_frame, text="Amount:", bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY, font=("Consolas", 10)
        ).pack(side=tk.LEFT)
        
        self.amount_entry = tk.Entry(
            input_frame,
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            font=("Consolas", 12),
            insertbackground=Theme.TEXT_PRIMARY,
            width=15
        )
        self.amount_entry.pack(side=tk.LEFT, padx=5)
        self.amount_entry.insert(0, "0.01")
        
        # Buy/Sell buttons
        btn_frame = tk.Frame(self.frame, bg=Theme.BG_SECONDARY)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.buy_btn = tk.Button(
            btn_frame,
            text="🟢 BUY / LONG",
            bg=Theme.SUCCESS,
            fg="white",
            font=("Consolas", 11, "bold"),
            activebackground="#2ea043",
            cursor="hand2",
            width=15
        )
        self.buy_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.sell_btn = tk.Button(
            btn_frame,
            text="🔴 SELL / SHORT",
            bg=Theme.DANGER,
            fg="white",
            font=("Consolas", 11, "bold"),
            activebackground="#da3633",
            cursor="hand2",
            width=15
        )
        self.sell_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
    
    def update_symbol(self, symbol: str):
        """Update symbol display"""
        self.symbol_label.config(text=symbol)
    
    def update_price(self, price: float, change: float = 0):
        """Update price display"""
        color = Theme.SUCCESS if change >= 0 else Theme.DANGER
        self.price_label.config(text=f"${price:,.2f}", fg=color)


# ==================== Status Bar ====================
class StatusBar:
    """Bottom status bar"""
    
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=Theme.BG_TERTIARY, height=25)
        self.frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.frame.pack_propagate(False)
        
        # Connection status
        self.conn_label = tk.Label(
            self.frame,
            text="● DISCONNECTED",
            bg=Theme.BG_TERTIARY,
            fg=Theme.DANGER,
            font=("Consolas", 9)
        )
        self.conn_label.pack(side=tk.LEFT, padx=10)
        
        # Last update
        self.update_label = tk.Label(
            self.frame,
            text="Last update: --",
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_SECONDARY,
            font=("Consolas", 9)
        )
        self.update_label.pack(side=tk.LEFT, padx=20)
        
        # API mode
        self.api_label = tk.Label(
            self.frame,
            text="API: Live",
            bg=Theme.BG_TERTIARY,
            fg=Theme.ACCENT,
            font=("Consolas", 9)
        )
        self.api_label.pack(side=tk.RIGHT, padx=10)
    
    def set_connected(self, connected: bool):
        """Update connection status"""
        if connected:
            self.conn_label.config(text="● CONNECTED", fg=Theme.SUCCESS)
        else:
            self.conn_label.config(text="● DISCONNECTED", fg=Theme.DANGER)
    
    def set_last_update(self, timestamp: datetime = None):
        """Update last update time"""
        if timestamp is None:
            timestamp = datetime.now()
        self.update_label.config(
            text=f"Last update: {timestamp.strftime('%H:%M:%S')}"
        )


# ==================== Main Trading GUI ====================
class TradingGUI:
    """
    Professional Real-Time Trading GUI for Delta Exchange
    """
    
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self.root = tk.Tk()
        self.root.title("Delta Exchange Algo Trader - Real-Time Terminal")
        self.root.geometry("1400x900")
        self.root.configure(bg=Theme.BG_PRIMARY)
        self.root.minsize(1200, 700)
        
        # Data fetcher
        self.data_fetcher = RealTimeDataFetcher(testnet=testnet)
        
        # Current symbol
        self.current_symbol = "BTCUSD"
        self.current_resolution = 60  # 1 minute candles
        
        # Update flags
        self.running = True
        self.update_queue = queue.Queue()
        
        # Build UI
        self._build_ui()
        
        # Start data threads
        self._start_data_threads()
        
        # Process queue
        self._process_queue()
    
    def _build_ui(self):
        """Build the main UI"""
        # Top toolbar
        self._build_toolbar()
        
        # Main content area
        main_frame = tk.Frame(self.root, bg=Theme.BG_PRIMARY)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (ticker list)
        left_panel = tk.Frame(main_frame, bg=Theme.BG_SECONDARY, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.ticker_panel = MarketTickerPanel(left_panel)
        self.ticker_panel.set_on_select(self._on_symbol_select)
        
        # Center panel (chart)
        center_panel = tk.Frame(main_frame, bg=Theme.BG_SECONDARY)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Chart controls
        chart_controls = tk.Frame(center_panel, bg=Theme.BG_SECONDARY)
        chart_controls.pack(fill=tk.X, pady=5)
        
        # Timeframe buttons
        for tf, res in [("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14400), ("1D", 86400)]:
            btn = tk.Button(
                chart_controls,
                text=tf,
                bg=Theme.BG_TERTIARY if res != self.current_resolution else Theme.ACCENT,
                fg=Theme.TEXT_PRIMARY,
                font=("Consolas", 9),
                width=4,
                cursor="hand2",
                command=lambda r=res: self._change_resolution(r)
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # Refresh button
        refresh_btn = tk.Button(
            chart_controls,
            text="🔄 Refresh",
            bg=Theme.BG_TERTIARY,
            fg=Theme.ACCENT,
            font=("Consolas", 9),
            cursor="hand2",
            command=self._force_refresh
        )
        refresh_btn.pack(side=tk.RIGHT, padx=5)
        
        # Candlestick chart
        self.chart = RealTimeCandlestickChart(center_panel, width=800, height=450)
        
        # Right panel (order book + trade)
        right_panel = tk.Frame(main_frame, bg=Theme.BG_SECONDARY, width=320)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Trade panel
        self.trade_panel = TradePanel(right_panel)
        
        # Separator
        ttk.Separator(right_panel, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Order book
        self.orderbook_panel = OrderBookPanel(right_panel)
        
        # Status bar
        self.status_bar = StatusBar(self.root)
    
    def _build_toolbar(self):
        """Build top toolbar"""
        toolbar = tk.Frame(self.root, bg=Theme.BG_TERTIARY, height=40)
        toolbar.pack(fill=tk.X)
        toolbar.pack_propagate(False)
        
        # Logo
        logo = tk.Label(
            toolbar,
            text="🚀 DELTA EXCHANGE ALGO TRADER",
            bg=Theme.BG_TERTIARY,
            fg=Theme.ACCENT,
            font=("Consolas", 12, "bold")
        )
        logo.pack(side=tk.LEFT, padx=15)
        
        # Symbol search
        search_frame = tk.Frame(toolbar, bg=Theme.BG_TERTIARY)
        search_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            search_frame, text="Symbol:", bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_SECONDARY, font=("Consolas", 10)
        ).pack(side=tk.LEFT)
        
        self.symbol_entry = tk.Entry(
            search_frame,
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
            font=("Consolas", 11),
            insertbackground=Theme.TEXT_PRIMARY,
            width=15
        )
        self.symbol_entry.pack(side=tk.LEFT, padx=5)
        self.symbol_entry.insert(0, self.current_symbol)
        self.symbol_entry.bind('<Return>', lambda e: self._search_symbol())
        
        search_btn = tk.Button(
            search_frame,
            text="Go",
            bg=Theme.ACCENT,
            fg="white",
            font=("Consolas", 10),
            cursor="hand2",
            command=self._search_symbol
        )
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # API Mode indicator
        mode_label = tk.Label(
            toolbar,
            text="🟢 LIVE" if not self.testnet else "🟡 TESTNET",
            bg=Theme.BG_TERTIARY,
            fg=Theme.SUCCESS if not self.testnet else Theme.WARNING,
            font=("Consolas", 10, "bold")
        )
        mode_label.pack(side=tk.RIGHT, padx=15)
        
        # Time
        self.time_label = tk.Label(
            toolbar,
            text="",
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            font=("Consolas", 10)
        )
        self.time_label.pack(side=tk.RIGHT, padx=15)
        self._update_time()
    
    def _update_time(self):
        """Update time display"""
        now = datetime.now()
        self.time_label.config(text=now.strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self._update_time)
    
    def _start_data_threads(self):
        """Start background data fetching threads"""
        # Ticker update thread
        def ticker_thread():
            while self.running:
                try:
                    tickers = self.data_fetcher.get_tickers()
                    if tickers:
                        self.update_queue.put(("tickers", tickers))
                        self.update_queue.put(("connected", True))
                except Exception as e:
                    self.update_queue.put(("connected", False))
                time.sleep(5)  # Update every 5 seconds
        
        # Candle update thread
        def candle_thread():
            while self.running:
                try:
                    candles = self.data_fetcher.get_candles(
                        self.current_symbol,
                        self.current_resolution,
                        limit=150
                    )
                    if candles:
                        self.update_queue.put(("candles", candles))
                except Exception as e:
                    print(f"Candle fetch error: {e}")
                time.sleep(3)  # Update every 3 seconds
        
        # Order book update thread
        def orderbook_thread():
            while self.running:
                try:
                    orderbook = self.data_fetcher.get_orderbook(
                        self.current_symbol,
                        depth=15
                    )
                    if orderbook:
                        self.update_queue.put(("orderbook", orderbook))
                except Exception as e:
                    print(f"Orderbook fetch error: {e}")
                time.sleep(2)  # Update every 2 seconds
        
        # Start threads
        threading.Thread(target=ticker_thread, daemon=True).start()
        threading.Thread(target=candle_thread, daemon=True).start()
        threading.Thread(target=orderbook_thread, daemon=True).start()
    
    def _process_queue(self):
        """Process update queue in main thread"""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()
                
                if update_type == "tickers":
                    self.ticker_panel.update(data)
                    # Update trade panel price
                    if self.current_symbol in data:
                        ticker = data[self.current_symbol]
                        price = float(ticker.get("close", ticker.get("mark_price", 0)))
                        change = float(ticker.get("price_change_percent_24h", 0))
                        self.trade_panel.update_price(price, change)
                
                elif update_type == "candles":
                    self.chart.update(data, self.current_symbol)
                    self.status_bar.set_last_update()
                
                elif update_type == "orderbook":
                    self.orderbook_panel.update(data)
                
                elif update_type == "connected":
                    self.status_bar.set_connected(data)
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._process_queue)
    
    def _on_symbol_select(self, symbol: str):
        """Handle symbol selection from ticker panel"""
        self.current_symbol = symbol
        self.symbol_entry.delete(0, tk.END)
        self.symbol_entry.insert(0, symbol)
        self.trade_panel.update_symbol(symbol)
        self._force_refresh()
    
    def _search_symbol(self):
        """Search for symbol"""
        symbol = self.symbol_entry.get().upper().strip()
        if symbol:
            self.current_symbol = symbol
            self.trade_panel.update_symbol(symbol)
            self._force_refresh()
    
    def _change_resolution(self, resolution: int):
        """Change chart resolution"""
        self.current_resolution = resolution
        self._force_refresh()
    
    def _force_refresh(self):
        """Force immediate data refresh"""
        def refresh():
            try:
                candles = self.data_fetcher.get_candles(
                    self.current_symbol,
                    self.current_resolution,
                    limit=150
                )
                if candles:
                    self.update_queue.put(("candles", candles))
                
                orderbook = self.data_fetcher.get_orderbook(
                    self.current_symbol,
                    depth=15
                )
                if orderbook:
                    self.update_queue.put(("orderbook", orderbook))
                
            except Exception as e:
                print(f"Refresh error: {e}")
        
        threading.Thread(target=refresh, daemon=True).start()
    
    def run(self):
        """Run the GUI"""
        try:
            # Initialize products
            self.data_fetcher.get_products()
            self.root.mainloop()
        finally:
            self.running = False
    
    def quit(self):
        """Quit the application"""
        self.running = False
        self.root.quit()


def run_app(testnet: bool = False):
    """Run the trading application"""
    app = TradingGUI(testnet=testnet)
    app.run()


if __name__ == "__main__":
    run_app()
