"""
Advanced Bar Types (AFML Ch. 2)
==============================
Alternative sampling methods that produce more stationary series than time bars.

Bar Types:
- Tick Bars: Sample every N ticks
- Volume Bars: Sample every N units of volume
- Dollar Bars: Sample every N units of dollar volume
- Imbalance Bars: Sample when buy/sell imbalance exceeds threshold
- Runs Bars: Sample based on sequential trade direction runs

Reference: Advances in Financial Machine Learning, Ch. 2
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings


class BarType(Enum):
    """Enumeration of bar types."""
    TIME = "time"
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"
    TICK_IMBALANCE = "tick_imbalance"
    VOLUME_IMBALANCE = "volume_imbalance"
    TICK_RUNS = "tick_runs"
    VOLUME_RUNS = "volume_runs"


@dataclass
class TickData:
    """Single tick/trade record."""
    timestamp: pd.Timestamp
    price: float
    volume: float
    side: int  # 1 for buy, -1 for sell


class BarSampler:
    """
    Advanced bar sampling from tick data.
    
    Produces bars that are more stationary and informative than time bars.
    """
    
    def __init__(self, initial_threshold: Optional[float] = None):
        """
        Initialize sampler.
        
        Args:
            initial_threshold: Initial threshold for adaptive methods
        """
        self.initial_threshold = initial_threshold
        self._ewma_span = 100  # For adaptive threshold
        
    # =========================================================================
    # TICK BARS
    # =========================================================================
    
    def tick_bars(self, ticks: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
        """
        Create tick bars - sample every N ticks.
        
        More robust to periods of low activity than time bars.
        
        Args:
            ticks: DataFrame with columns ['timestamp', 'price', 'volume']
            threshold: Number of ticks per bar
        
        Returns:
            DataFrame with OHLCV bars
        """
        bars = []
        n = len(ticks)
        
        for i in range(0, n, threshold):
            chunk = ticks.iloc[i:min(i + threshold, n)]
            
            if len(chunk) == 0:
                continue
                
            bar = {
                'timestamp': chunk['timestamp'].iloc[-1],
                'open': chunk['price'].iloc[0],
                'high': chunk['price'].max(),
                'low': chunk['price'].min(),
                'close': chunk['price'].iloc[-1],
                'volume': chunk['volume'].sum(),
                'tick_count': len(chunk),
                'vwap': (chunk['price'] * chunk['volume']).sum() / chunk['volume'].sum()
                        if chunk['volume'].sum() > 0 else chunk['price'].mean()
            }
            bars.append(bar)
        
        return pd.DataFrame(bars).set_index('timestamp')
    
    # =========================================================================
    # VOLUME BARS
    # =========================================================================
    
    def volume_bars(self, ticks: pd.DataFrame, threshold: float = 1000) -> pd.DataFrame:
        """
        Create volume bars - sample every N units of volume.
        
        Samples more during active periods, less during quiet periods.
        
        Args:
            ticks: DataFrame with columns ['timestamp', 'price', 'volume']
            threshold: Volume per bar
        
        Returns:
            DataFrame with OHLCV bars
        """
        bars = []
        cum_volume = 0
        bar_ticks = []
        
        for _, tick in ticks.iterrows():
            bar_ticks.append(tick)
            cum_volume += tick['volume']
            
            if cum_volume >= threshold:
                prices = [t['price'] for t in bar_ticks]
                volumes = [t['volume'] for t in bar_ticks]
                
                bar = {
                    'timestamp': bar_ticks[-1]['timestamp'],
                    'open': prices[0],
                    'high': max(prices),
                    'low': min(prices),
                    'close': prices[-1],
                    'volume': sum(volumes),
                    'tick_count': len(bar_ticks),
                    'vwap': sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
                }
                bars.append(bar)
                
                # Reset for next bar
                cum_volume = 0
                bar_ticks = []
        
        return pd.DataFrame(bars).set_index('timestamp') if bars else pd.DataFrame()
    
    # =========================================================================
    # DOLLAR BARS
    # =========================================================================
    
    def dollar_bars(self, ticks: pd.DataFrame, threshold: float = 100000) -> pd.DataFrame:
        """
        Create dollar bars - sample every N units of dollar volume.
        
        Most stable across different price levels.
        
        Args:
            ticks: DataFrame with columns ['timestamp', 'price', 'volume']
            threshold: Dollar volume per bar
        
        Returns:
            DataFrame with OHLCV bars
        """
        bars = []
        cum_dollar = 0
        bar_ticks = []
        
        for _, tick in ticks.iterrows():
            bar_ticks.append(tick)
            cum_dollar += tick['price'] * tick['volume']
            
            if cum_dollar >= threshold:
                prices = [t['price'] for t in bar_ticks]
                volumes = [t['volume'] for t in bar_ticks]
                dollar_volumes = [t['price'] * t['volume'] for t in bar_ticks]
                
                bar = {
                    'timestamp': bar_ticks[-1]['timestamp'],
                    'open': prices[0],
                    'high': max(prices),
                    'low': min(prices),
                    'close': prices[-1],
                    'volume': sum(volumes),
                    'dollar_volume': sum(dollar_volumes),
                    'tick_count': len(bar_ticks),
                    'vwap': sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
                }
                bars.append(bar)
                
                cum_dollar = 0
                bar_ticks = []
        
        return pd.DataFrame(bars).set_index('timestamp') if bars else pd.DataFrame()
    
    # =========================================================================
    # TICK IMBALANCE BARS (TIBs)
    # =========================================================================
    
    def _classify_tick(self, price: float, prev_price: float) -> int:
        """Classify tick as buy (+1) or sell (-1) using tick rule."""
        if price > prev_price:
            return 1
        elif price < prev_price:
            return -1
        else:
            return 0  # No change
    
    def tick_imbalance_bars(self, ticks: pd.DataFrame, 
                            initial_threshold: Optional[float] = None,
                            expected_ticks: int = 100) -> pd.DataFrame:
        """
        Create tick imbalance bars (TIBs) - AFML Ch. 2.
        
        Samples when cumulative tick imbalance exceeds a dynamic threshold.
        Information-driven sampling.
        
        Args:
            ticks: DataFrame with columns ['timestamp', 'price', 'volume']
            initial_threshold: Initial imbalance threshold (auto-calculated if None)
            expected_ticks: Expected number of ticks per bar for threshold calculation
        
        Returns:
            DataFrame with OHLCV bars
        """
        bars = []
        
        # Classify all ticks
        ticks = ticks.copy()
        ticks['b_t'] = 0  # Buy/sell classification
        
        prev_price = ticks['price'].iloc[0]
        prev_b = 1  # Previous tick direction
        
        for i in range(len(ticks)):
            price = ticks['price'].iloc[i]
            if price > prev_price:
                ticks.iloc[i, ticks.columns.get_loc('b_t')] = 1
                prev_b = 1
            elif price < prev_price:
                ticks.iloc[i, ticks.columns.get_loc('b_t')] = -1
                prev_b = -1
            else:
                ticks.iloc[i, ticks.columns.get_loc('b_t')] = prev_b
            prev_price = price
        
        # Calculate initial expected imbalance
        P_plus = (ticks['b_t'] == 1).mean()  # Probability of buy
        P_minus = (ticks['b_t'] == -1).mean()  # Probability of sell
        
        # Expected imbalance = E[T] * |2P+ - 1|
        if initial_threshold is None:
            initial_threshold = expected_ticks * abs(2 * P_plus - 1)
        
        threshold = initial_threshold
        theta_t = []  # Track thresholds for EWMA
        
        # Generate bars
        cum_theta = 0
        bar_ticks = []
        
        for i, (_, tick) in enumerate(ticks.iterrows()):
            bar_ticks.append(tick)
            cum_theta += tick['b_t']
            
            if abs(cum_theta) >= threshold:
                prices = [t['price'] for t in bar_ticks]
                volumes = [t['volume'] for t in bar_ticks]
                b_values = [t['b_t'] for t in bar_ticks]
                
                bar = {
                    'timestamp': bar_ticks[-1]['timestamp'],
                    'open': prices[0],
                    'high': max(prices),
                    'low': min(prices),
                    'close': prices[-1],
                    'volume': sum(volumes),
                    'tick_count': len(bar_ticks),
                    'imbalance': cum_theta,
                    'buy_count': sum(1 for b in b_values if b == 1),
                    'sell_count': sum(1 for b in b_values if b == -1)
                }
                bars.append(bar)
                
                # Update threshold using EWMA
                theta_t.append(abs(cum_theta))
                if len(theta_t) > 1:
                    threshold = pd.Series(theta_t).ewm(span=self._ewma_span).mean().iloc[-1]
                
                cum_theta = 0
                bar_ticks = []
        
        return pd.DataFrame(bars).set_index('timestamp') if bars else pd.DataFrame()
    
    # =========================================================================
    # VOLUME IMBALANCE BARS (VIBs)
    # =========================================================================
    
    def volume_imbalance_bars(self, ticks: pd.DataFrame,
                               initial_threshold: Optional[float] = None,
                               expected_volume: float = 10000) -> pd.DataFrame:
        """
        Create volume imbalance bars (VIBs) - AFML Ch. 2.
        
        Like TIBs but weights by volume.
        
        Args:
            ticks: DataFrame with columns ['timestamp', 'price', 'volume', 'side']
                   If 'side' not present, uses tick rule
            initial_threshold: Initial threshold
            expected_volume: Expected volume per bar
        
        Returns:
            DataFrame with OHLCV bars
        """
        bars = []
        ticks = ticks.copy()
        
        # Classify ticks if side not provided
        if 'side' not in ticks.columns:
            ticks['side'] = 0
            prev_price = ticks['price'].iloc[0]
            prev_side = 1
            
            for i in range(len(ticks)):
                price = ticks['price'].iloc[i]
                if price > prev_price:
                    ticks.iloc[i, ticks.columns.get_loc('side')] = 1
                    prev_side = 1
                elif price < prev_price:
                    ticks.iloc[i, ticks.columns.get_loc('side')] = -1
                    prev_side = -1
                else:
                    ticks.iloc[i, ticks.columns.get_loc('side')] = prev_side
                prev_price = price
        
        # Calculate signed volume
        ticks['signed_volume'] = ticks['volume'] * ticks['side']
        
        # Initial threshold
        if initial_threshold is None:
            buy_vol = ticks[ticks['side'] == 1]['volume'].sum()
            sell_vol = ticks[ticks['side'] == -1]['volume'].sum()
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                v_plus = buy_vol / total_vol
                v_minus = sell_vol / total_vol
                initial_threshold = expected_volume * abs(v_plus - v_minus)
            else:
                initial_threshold = expected_volume * 0.1
        
        threshold = initial_threshold
        theta_t = []
        
        # Generate bars
        cum_theta = 0
        bar_ticks = []
        
        for _, tick in ticks.iterrows():
            bar_ticks.append(tick)
            cum_theta += tick['signed_volume']
            
            if abs(cum_theta) >= threshold:
                prices = [t['price'] for t in bar_ticks]
                volumes = [t['volume'] for t in bar_ticks]
                
                bar = {
                    'timestamp': bar_ticks[-1]['timestamp'],
                    'open': prices[0],
                    'high': max(prices),
                    'low': min(prices),
                    'close': prices[-1],
                    'volume': sum(volumes),
                    'tick_count': len(bar_ticks),
                    'volume_imbalance': cum_theta,
                    'buy_volume': sum(t['volume'] for t in bar_ticks if t['side'] == 1),
                    'sell_volume': sum(t['volume'] for t in bar_ticks if t['side'] == -1)
                }
                bars.append(bar)
                
                theta_t.append(abs(cum_theta))
                if len(theta_t) > 1:
                    threshold = pd.Series(theta_t).ewm(span=self._ewma_span).mean().iloc[-1]
                
                cum_theta = 0
                bar_ticks = []
        
        return pd.DataFrame(bars).set_index('timestamp') if bars else pd.DataFrame()
    
    # =========================================================================
    # TICK RUNS BARS
    # =========================================================================
    
    def tick_runs_bars(self, ticks: pd.DataFrame,
                       initial_threshold: Optional[float] = None,
                       expected_ticks: int = 100) -> pd.DataFrame:
        """
        Create tick runs bars - AFML Ch. 2.
        
        Samples based on the longest run of consecutive buys or sells.
        Detects momentum/trend exhaustion.
        
        Args:
            ticks: DataFrame with tick data
            initial_threshold: Initial runs threshold
            expected_ticks: Expected ticks per bar
        
        Returns:
            DataFrame with bars
        """
        bars = []
        ticks = ticks.copy()
        
        # Classify ticks
        ticks['b_t'] = 0
        prev_price = ticks['price'].iloc[0]
        prev_b = 1
        
        for i in range(len(ticks)):
            price = ticks['price'].iloc[i]
            if price > prev_price:
                ticks.iloc[i, ticks.columns.get_loc('b_t')] = 1
                prev_b = 1
            elif price < prev_price:
                ticks.iloc[i, ticks.columns.get_loc('b_t')] = -1
                prev_b = -1
            else:
                ticks.iloc[i, ticks.columns.get_loc('b_t')] = prev_b
            prev_price = price
        
        # Calculate expected runs
        P_plus = (ticks['b_t'] == 1).mean()
        P_minus = 1 - P_plus
        
        if initial_threshold is None:
            # Expected max run length
            initial_threshold = expected_ticks * max(P_plus, P_minus)
        
        threshold = initial_threshold
        theta_t = []
        
        # Generate bars
        buy_run = 0
        sell_run = 0
        bar_ticks = []
        
        for _, tick in ticks.iterrows():
            bar_ticks.append(tick)
            
            if tick['b_t'] == 1:
                buy_run += 1
                sell_run = 0
            else:
                sell_run += 1
                buy_run = 0
            
            max_run = max(buy_run, sell_run)
            
            if max_run >= threshold:
                prices = [t['price'] for t in bar_ticks]
                volumes = [t['volume'] for t in bar_ticks]
                
                bar = {
                    'timestamp': bar_ticks[-1]['timestamp'],
                    'open': prices[0],
                    'high': max(prices),
                    'low': min(prices),
                    'close': prices[-1],
                    'volume': sum(volumes),
                    'tick_count': len(bar_ticks),
                    'max_run': max_run,
                    'run_direction': 1 if buy_run > sell_run else -1
                }
                bars.append(bar)
                
                theta_t.append(max_run)
                if len(theta_t) > 1:
                    threshold = pd.Series(theta_t).ewm(span=self._ewma_span).mean().iloc[-1]
                
                buy_run = 0
                sell_run = 0
                bar_ticks = []
        
        return pd.DataFrame(bars).set_index('timestamp') if bars else pd.DataFrame()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def simulate_tick_data(n_ticks: int = 10000, 
                       start_price: float = 100.0,
                       volatility: float = 0.001,
                       seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic tick data for testing.
    
    Args:
        n_ticks: Number of ticks to generate
        start_price: Starting price
        volatility: Price volatility per tick
        seed: Random seed
    
    Returns:
        DataFrame with tick data
    """
    np.random.seed(seed)
    
    # Generate prices with random walk
    returns = np.random.randn(n_ticks) * volatility
    prices = start_price * np.cumprod(1 + returns)
    
    # Generate volumes (log-normal)
    volumes = np.exp(np.random.randn(n_ticks) * 0.5 + 3)
    
    # Generate timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_ticks, freq='100ms')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes
    })


def compare_bar_types(ticks: pd.DataFrame) -> dict:
    """
    Compare statistics across different bar types.
    
    Useful for selecting the best bar type for a given dataset.
    """
    sampler = BarSampler()
    
    results = {}
    
    # Time bars (1 minute)
    time_bars = ticks.set_index('timestamp').resample('1min').agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    time_bars.columns = ['open', 'high', 'low', 'close', 'volume']
    time_bars = time_bars.dropna()
    
    # Various bar types
    bar_types = {
        'time_1min': time_bars,
        'tick_100': sampler.tick_bars(ticks, threshold=100),
        'volume_1000': sampler.volume_bars(ticks, threshold=1000),
        'dollar_10000': sampler.dollar_bars(ticks, threshold=10000),
        'tick_imbalance': sampler.tick_imbalance_bars(ticks),
        'volume_imbalance': sampler.volume_imbalance_bars(ticks)
    }
    
    for name, bars in bar_types.items():
        if len(bars) > 10:
            returns = bars['close'].pct_change().dropna()
            results[name] = {
                'n_bars': len(bars),
                'return_mean': returns.mean(),
                'return_std': returns.std(),
                'return_skew': returns.skew(),
                'return_kurt': returns.kurtosis(),
                'autocorr_1': returns.autocorr(1) if len(returns) > 1 else 0,
                'normality_stat': abs(returns.skew()) + abs(returns.kurtosis() - 3)  # Closer to 0 = more normal
            }
    
    return results


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ALTERNATIVE BAR TYPES DEMO (AFML Ch. 2)")
    print("="*70)
    
    # Generate sample data
    print("\n1. Generating synthetic tick data...")
    ticks = simulate_tick_data(n_ticks=50000)
    print(f"   Generated {len(ticks)} ticks")
    
    # Create different bar types
    sampler = BarSampler()
    
    print("\n2. Creating different bar types...")
    
    tick_bars = sampler.tick_bars(ticks, threshold=100)
    print(f"   Tick Bars (100): {len(tick_bars)} bars")
    
    volume_bars = sampler.volume_bars(ticks, threshold=1000)
    print(f"   Volume Bars (1000): {len(volume_bars)} bars")
    
    dollar_bars = sampler.dollar_bars(ticks, threshold=50000)
    print(f"   Dollar Bars (50000): {len(dollar_bars)} bars")
    
    tib = sampler.tick_imbalance_bars(ticks)
    print(f"   Tick Imbalance Bars: {len(tib)} bars")
    
    vib = sampler.volume_imbalance_bars(ticks)
    print(f"   Volume Imbalance Bars: {len(vib)} bars")
    
    trb = sampler.tick_runs_bars(ticks)
    print(f"   Tick Runs Bars: {len(trb)} bars")
    
    # Compare statistics
    print("\n3. Comparing bar type statistics...")
    stats = compare_bar_types(ticks)
    
    print(f"\n{'Bar Type':<20} {'N Bars':>8} {'Std':>10} {'Skew':>8} {'Kurt':>8} {'Norm':>8}")
    print("-" * 65)
    
    for name, s in stats.items():
        print(f"{name:<20} {s['n_bars']:>8} {s['return_std']:>10.6f} "
              f"{s['return_skew']:>8.3f} {s['return_kurt']:>8.3f} {s['normality_stat']:>8.3f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
- Lower 'Norm' score = more Gaussian returns (better for ML)
- Imbalance bars capture information arrival
- Dollar bars are most stable across price levels
- Volume bars sample more during active periods

RECOMMENDATION: Start with dollar bars for most ML applications.
Use imbalance bars for detecting informed trading.
""")
