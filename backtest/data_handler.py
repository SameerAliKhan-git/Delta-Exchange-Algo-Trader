"""
Data Handler - Historical data loading and replay for backtesting

Provides:
- CSV/Parquet data loading
- Data normalization
- Historical data replay
- Multiple timeframe support
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Generator, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("Aladdin.DataHandler")


@dataclass
class HistoricalBar:
    """Single OHLCV bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = "1h"
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HistoricalBar':
        """Create from dictionary"""
        return cls(
            timestamp=data.get('timestamp', datetime.now()),
            open=float(data.get('open', 0)),
            high=float(data.get('high', 0)),
            low=float(data.get('low', 0)),
            close=float(data.get('close', 0)),
            volume=float(data.get('volume', 0)),
            symbol=data.get('symbol', ''),
            timeframe=data.get('timeframe', '1h')
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol,
            'timeframe': self.timeframe
        }


@dataclass
class DataSeries:
    """
    Time series data container with numpy arrays for fast access
    """
    symbol: str
    timeframe: str
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    opens: np.ndarray = field(default_factory=lambda: np.array([]))
    highs: np.ndarray = field(default_factory=lambda: np.array([]))
    lows: np.ndarray = field(default_factory=lambda: np.array([]))
    closes: np.ndarray = field(default_factory=lambda: np.array([]))
    volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __len__(self) -> int:
        return len(self.timestamps)
    
    def __getitem__(self, idx) -> HistoricalBar:
        return HistoricalBar(
            timestamp=datetime.fromtimestamp(self.timestamps[idx]),
            open=self.opens[idx],
            high=self.highs[idx],
            low=self.lows[idx],
            close=self.closes[idx],
            volume=self.volumes[idx],
            symbol=self.symbol,
            timeframe=self.timeframe
        )
    
    def get_slice(self, start: int, end: int) -> 'DataSeries':
        """Get slice of data"""
        return DataSeries(
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamps=self.timestamps[start:end],
            opens=self.opens[start:end],
            highs=self.highs[start:end],
            lows=self.lows[start:end],
            closes=self.closes[start:end],
            volumes=self.volumes[start:end]
        )
    
    def get_lookback(self, idx: int, periods: int) -> 'DataSeries':
        """Get lookback window ending at idx"""
        start = max(0, idx - periods + 1)
        return self.get_slice(start, idx + 1)
    
    @classmethod
    def from_bars(cls, bars: List[HistoricalBar]) -> 'DataSeries':
        """Create from list of bars"""
        if not bars:
            return cls(symbol="", timeframe="")
        
        return cls(
            symbol=bars[0].symbol,
            timeframe=bars[0].timeframe,
            timestamps=np.array([b.timestamp.timestamp() for b in bars]),
            opens=np.array([b.open for b in bars]),
            highs=np.array([b.high for b in bars]),
            lows=np.array([b.low for b in bars]),
            closes=np.array([b.close for b in bars]),
            volumes=np.array([b.volume for b in bars])
        )


class DataHandler:
    """
    Historical data handler for backtesting
    
    Features:
    - Load from CSV/Parquet
    - Generate synthetic data
    - Multi-symbol support
    - Data replay iteration
    """
    
    def __init__(self):
        """Initialize data handler"""
        self._data: Dict[str, Dict[str, DataSeries]] = {}  # symbol -> timeframe -> data
        self._current_idx: Dict[str, int] = {}
        
    def load_csv(
        self,
        filepath: str,
        symbol: str,
        timeframe: str = "1h",
        date_column: str = "timestamp",
        date_format: str = "%Y-%m-%d %H:%M:%S"
    ) -> DataSeries:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            symbol: Symbol name
            timeframe: Timeframe (1m, 5m, 1h, 4h, 1d)
            date_column: Name of date column
            date_format: Date format string
        
        Returns:
            DataSeries
        """
        try:
            import pandas as pd
            
            df = pd.read_csv(filepath)
            
            # Parse dates
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            df = df.sort_values(date_column)
            
            # Create data series
            data = DataSeries(
                symbol=symbol,
                timeframe=timeframe,
                timestamps=df[date_column].values.astype('datetime64[s]').astype('float64'),
                opens=df['open'].values.astype('float64'),
                highs=df['high'].values.astype('float64'),
                lows=df['low'].values.astype('float64'),
                closes=df['close'].values.astype('float64'),
                volumes=df['volume'].values.astype('float64')
            )
            
            self._store_data(symbol, timeframe, data)
            logger.info(f"Loaded {len(data)} bars for {symbol} {timeframe} from {filepath}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def generate_synthetic(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        num_bars: int = 1000,
        start_date: datetime = None
    ) -> DataSeries:
        """
        Generate synthetic OHLCV data using geometric Brownian motion
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            start_price: Starting price
            volatility: Price volatility (std dev per period)
            drift: Price drift (mean return per period)
            num_bars: Number of bars to generate
            start_date: Starting date
        
        Returns:
            DataSeries
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(hours=num_bars)
        
        # Timeframe to seconds
        tf_seconds = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        period = tf_seconds.get(timeframe, 3600)
        
        # Generate timestamps
        timestamps = np.array([
            (start_date + timedelta(seconds=i * period)).timestamp()
            for i in range(num_bars)
        ])
        
        # Generate prices using GBM
        np.random.seed(42)  # Reproducible
        returns = np.random.normal(drift, volatility, num_bars)
        log_prices = np.log(start_price) + np.cumsum(returns)
        closes = np.exp(log_prices)
        
        # Generate OHLV from close
        opens = np.roll(closes, 1)
        opens[0] = start_price
        
        # High/Low based on volatility
        intra_vol = volatility * 0.5
        highs = closes * (1 + np.abs(np.random.normal(0, intra_vol, num_bars)))
        lows = closes * (1 - np.abs(np.random.normal(0, intra_vol, num_bars)))
        
        # Ensure OHLC constraints
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # Volume (random with some correlation to price movement)
        base_volume = 1000
        volumes = base_volume * (1 + np.abs(returns) * 10) * np.random.uniform(0.5, 1.5, num_bars)
        
        data = DataSeries(
            symbol=symbol,
            timeframe=timeframe,
            timestamps=timestamps,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes
        )
        
        self._store_data(symbol, timeframe, data)
        logger.info(f"Generated {num_bars} synthetic bars for {symbol} {timeframe}")
        
        return data
    
    def _store_data(self, symbol: str, timeframe: str, data: DataSeries) -> None:
        """Store data internally"""
        if symbol not in self._data:
            self._data[symbol] = {}
        self._data[symbol][timeframe] = data
        self._current_idx[f"{symbol}_{timeframe}"] = 0
    
    def get_data(self, symbol: str, timeframe: str = "1h") -> Optional[DataSeries]:
        """Get stored data series"""
        return self._data.get(symbol, {}).get(timeframe)
    
    def iter_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> Generator[Tuple[int, HistoricalBar], None, None]:
        """
        Iterate through bars for replay
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            start_idx: Starting index
            end_idx: Ending index
        
        Yields:
            Tuple of (index, HistoricalBar)
        """
        data = self.get_data(symbol, timeframe)
        if data is None:
            return
        
        if end_idx is None:
            end_idx = len(data)
        
        for i in range(start_idx, end_idx):
            yield i, data[i]
    
    def get_bar_at(
        self,
        symbol: str,
        timeframe: str,
        idx: int
    ) -> Optional[HistoricalBar]:
        """Get bar at specific index"""
        data = self.get_data(symbol, timeframe)
        if data is None or idx >= len(data):
            return None
        return data[idx]
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        return list(self._data.keys())
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """Get available timeframes for symbol"""
        return list(self._data.get(symbol, {}).keys())
    
    def get_date_range(
        self,
        symbol: str,
        timeframe: str = "1h"
    ) -> Optional[Tuple[datetime, datetime]]:
        """Get date range for symbol"""
        data = self.get_data(symbol, timeframe)
        if data is None or len(data) == 0:
            return None
        
        return (
            datetime.fromtimestamp(data.timestamps[0]),
            datetime.fromtimestamp(data.timestamps[-1])
        )


class MultiSymbolHandler:
    """
    Handler for synchronized multi-symbol data replay
    """
    
    def __init__(self, handler: DataHandler):
        """
        Initialize multi-symbol handler
        
        Args:
            handler: DataHandler with loaded data
        """
        self.handler = handler
        self._symbols: List[str] = []
        self._timeframe: str = "1h"
        self._current_time: float = 0
        self._indices: Dict[str, int] = {}
    
    def set_symbols(self, symbols: List[str], timeframe: str = "1h") -> None:
        """Set symbols to replay"""
        self._symbols = symbols
        self._timeframe = timeframe
        
        # Find common start time
        start_times = []
        for symbol in symbols:
            data = self.handler.get_data(symbol, timeframe)
            if data and len(data) > 0:
                start_times.append(data.timestamps[0])
                self._indices[symbol] = 0
        
        if start_times:
            self._current_time = max(start_times)
    
    def iter_synced(self) -> Generator[Tuple[float, Dict[str, HistoricalBar]], None, None]:
        """
        Iterate through synchronized bars
        
        Yields:
            Tuple of (timestamp, dict of symbol -> bar)
        """
        # Get all timestamps
        all_timestamps = set()
        for symbol in self._symbols:
            data = self.handler.get_data(symbol, self._timeframe)
            if data:
                all_timestamps.update(data.timestamps)
        
        # Sort and iterate
        for ts in sorted(all_timestamps):
            bars = {}
            for symbol in self._symbols:
                data = self.handler.get_data(symbol, self._timeframe)
                if data is None:
                    continue
                
                idx = self._indices.get(symbol, 0)
                while idx < len(data) and data.timestamps[idx] < ts:
                    idx += 1
                
                if idx < len(data) and data.timestamps[idx] == ts:
                    bars[symbol] = data[idx]
                    self._indices[symbol] = idx + 1
            
            if bars:
                yield ts, bars


if __name__ == "__main__":
    # Test data handler
    logging.basicConfig(level=logging.INFO)
    
    handler = DataHandler()
    
    # Generate synthetic data
    btc_data = handler.generate_synthetic(
        symbol="BTCUSD",
        timeframe="1h",
        start_price=100000,
        volatility=0.02,
        num_bars=500
    )
    
    eth_data = handler.generate_synthetic(
        symbol="ETHUSD",
        timeframe="1h",
        start_price=3000,
        volatility=0.025,
        num_bars=500
    )
    
    print("\n" + "=" * 60)
    print("DATA HANDLER TEST")
    print("=" * 60)
    
    print(f"\nAvailable symbols: {handler.get_available_symbols()}")
    print(f"BTC date range: {handler.get_date_range('BTCUSD', '1h')}")
    print(f"ETH date range: {handler.get_date_range('ETHUSD', '1h')}")
    
    # Sample data
    print("\nFirst 5 BTC bars:")
    for i, bar in handler.iter_bars("BTCUSD", "1h", 0, 5):
        print(f"  {bar.timestamp}: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f}")
    
    # Multi-symbol sync
    multi = MultiSymbolHandler(handler)
    multi.set_symbols(["BTCUSD", "ETHUSD"], "1h")
    
    print("\nFirst 3 synchronized bars:")
    count = 0
    for ts, bars in multi.iter_synced():
        if count >= 3:
            break
        print(f"  {datetime.fromtimestamp(ts)}")
        for sym, bar in bars.items():
            print(f"    {sym}: {bar.close:.2f}")
        count += 1
    
    print("\n✅ Data handler test complete!")
