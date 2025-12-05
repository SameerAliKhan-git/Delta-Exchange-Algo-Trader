"""
Data Loader - Load and cache OHLCV data from Delta Exchange

Provides:
- Historical candle data fetching
- Local file caching
- Data validation and preprocessing
- Multiple timeframe support
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CandleData:
    """Container for OHLCV candle data"""
    symbol: str
    resolution: str
    timestamps: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    
    def __len__(self) -> int:
        return len(self.timestamps)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame({
            'timestamp': pd.to_datetime(self.timestamps, unit='s'),
            'open': self.opens,
            'high': self.highs,
            'low': self.lows,
            'close': self.closes,
            'volume': self.volumes
        })
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, symbol: str, resolution: str) -> "CandleData":
        """Create from pandas DataFrame"""
        return cls(
            symbol=symbol,
            resolution=resolution,
            timestamps=df['timestamp'].astype(np.int64) // 10**9 if pd.api.types.is_datetime64_any_dtype(df['timestamp']) else df['timestamp'].values,
            opens=df['open'].values.astype(np.float64),
            highs=df['high'].values.astype(np.float64),
            lows=df['low'].values.astype(np.float64),
            closes=df['close'].values.astype(np.float64),
            volumes=df['volume'].values.astype(np.float64)
        )


class DataLoader:
    """
    Load and cache historical data from Delta Exchange
    """
    
    RESOLUTIONS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400
    }
    
    def __init__(
        self,
        delta_client=None,
        cache_dir: str = "./data_cache",
        use_cache: bool = True
    ):
        """
        Initialize data loader
        
        Args:
            delta_client: Delta Exchange client instance
            cache_dir: Directory for cached data
            use_cache: Whether to use file caching
        """
        self.client = delta_client
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_candles(
        self,
        symbol: str,
        resolution: str = "1h",
        start: datetime = None,
        end: datetime = None,
        limit: int = 1000
    ) -> CandleData:
        """
        Load candle data for symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            resolution: Candle resolution ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            start: Start datetime
            end: End datetime
            limit: Maximum candles to fetch
        
        Returns:
            CandleData object
        """
        # Try cache first
        if self.use_cache and start and end:
            cached = self._load_from_cache(symbol, resolution, start, end)
            if cached is not None:
                return cached
        
        # Fetch from API
        candles = self._fetch_candles(symbol, resolution, start, end, limit)
        
        # Cache the data
        if self.use_cache and candles is not None and len(candles) > 0:
            self._save_to_cache(candles)
        
        return candles
    
    def _fetch_candles(
        self,
        symbol: str,
        resolution: str,
        start: datetime = None,
        end: datetime = None,
        limit: int = 1000
    ) -> CandleData:
        """Fetch candles from Delta Exchange API"""
        if self.client is None:
            raise ValueError("Delta client not configured")
        
        # Convert resolution to seconds
        resolution_secs = self.RESOLUTIONS.get(resolution, 3600)
        
        # Calculate timestamps
        end_ts = int(end.timestamp()) if end else int(time.time())
        start_ts = int(start.timestamp()) if start else end_ts - (limit * resolution_secs)
        
        try:
            # Call Delta API
            response = self.client.get_candles(
                symbol=symbol,
                resolution=resolution_secs,
                start=start_ts,
                end=end_ts
            )
            
            if not response or 'result' not in response:
                return self._empty_candles(symbol, resolution)
            
            candles = response['result']
            
            if not candles:
                return self._empty_candles(symbol, resolution)
            
            # Parse candles
            timestamps = np.array([c['time'] for c in candles], dtype=np.float64)
            opens = np.array([float(c['open']) for c in candles], dtype=np.float64)
            highs = np.array([float(c['high']) for c in candles], dtype=np.float64)
            lows = np.array([float(c['low']) for c in candles], dtype=np.float64)
            closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
            volumes = np.array([float(c.get('volume', 0)) for c in candles], dtype=np.float64)
            
            return CandleData(
                symbol=symbol,
                resolution=resolution,
                timestamps=timestamps,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes
            )
            
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return self._empty_candles(symbol, resolution)
    
    def _empty_candles(self, symbol: str, resolution: str) -> CandleData:
        """Return empty CandleData"""
        return CandleData(
            symbol=symbol,
            resolution=resolution,
            timestamps=np.array([], dtype=np.float64),
            opens=np.array([], dtype=np.float64),
            highs=np.array([], dtype=np.float64),
            lows=np.array([], dtype=np.float64),
            closes=np.array([], dtype=np.float64),
            volumes=np.array([], dtype=np.float64)
        )
    
    def _cache_path(self, symbol: str, resolution: str, date: datetime) -> Path:
        """Get cache file path for date"""
        date_str = date.strftime("%Y%m%d")
        return self.cache_dir / f"{symbol}_{resolution}_{date_str}.parquet"
    
    def _load_from_cache(
        self,
        symbol: str,
        resolution: str,
        start: datetime,
        end: datetime
    ) -> Optional[CandleData]:
        """Load data from cache"""
        try:
            # Check for cached file
            cache_file = self.cache_dir / f"{symbol}_{resolution}.parquet"
            
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                
                # Filter to requested range
                if 'timestamp' in df.columns:
                    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
                    df = df[mask]
                
                if len(df) > 0:
                    return CandleData.from_dataframe(df, symbol, resolution)
            
        except Exception:
            pass
        
        return None
    
    def _save_to_cache(self, data: CandleData) -> None:
        """Save data to cache"""
        try:
            cache_file = self.cache_dir / f"{data.symbol}_{data.resolution}.parquet"
            
            df = data.to_dataframe()
            
            # Append to existing if present
            if cache_file.exists():
                existing = pd.read_parquet(cache_file)
                df = pd.concat([existing, df]).drop_duplicates(subset=['timestamp'])
                df = df.sort_values('timestamp')
            
            df.to_parquet(cache_file, index=False)
            
        except Exception as e:
            print(f"Error saving to cache: {e}")


def load_candles(
    symbol: str,
    resolution: str = "1h",
    days: int = 30,
    delta_client=None
) -> CandleData:
    """
    Convenience function to load candles
    
    Args:
        symbol: Trading symbol
        resolution: Candle resolution
        days: Number of days of history
        delta_client: Delta Exchange client
    
    Returns:
        CandleData
    """
    loader = DataLoader(delta_client=delta_client)
    
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    
    return loader.load_candles(symbol, resolution, start, end)


def load_candles_range(
    symbol: str,
    start: datetime,
    end: datetime,
    resolution: str = "1h",
    delta_client=None
) -> CandleData:
    """
    Load candles for specific date range
    
    Args:
        symbol: Trading symbol
        start: Start datetime
        end: End datetime
        resolution: Candle resolution
        delta_client: Delta Exchange client
    
    Returns:
        CandleData
    """
    loader = DataLoader(delta_client=delta_client)
    return loader.load_candles(symbol, resolution, start, end)


def get_available_symbols(delta_client=None) -> List[str]:
    """
    Get list of available trading symbols
    
    Args:
        delta_client: Delta Exchange client
    
    Returns:
        List of symbol strings
    """
    if delta_client is None:
        return []
    
    try:
        products = delta_client.get_products()
        if products and 'result' in products:
            return [p['symbol'] for p in products['result']]
    except Exception:
        pass
    
    return []


def preprocess_candles(data: CandleData) -> CandleData:
    """
    Preprocess and validate candle data
    
    - Remove duplicates
    - Fill gaps
    - Handle missing values
    
    Args:
        data: Raw CandleData
    
    Returns:
        Cleaned CandleData
    """
    if len(data) == 0:
        return data
    
    # Convert to DataFrame for easier manipulation
    df = data.to_dataframe()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Forward fill missing values
    df = df.ffill()
    
    # Validate OHLC relationships
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    # Ensure positive prices
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].clip(lower=0)
    
    # Ensure non-negative volume
    df['volume'] = df['volume'].clip(lower=0)
    
    return CandleData.from_dataframe(df, data.symbol, data.resolution)


def resample_candles(
    data: CandleData,
    target_resolution: str
) -> CandleData:
    """
    Resample candles to different timeframe
    
    Args:
        data: Input CandleData
        target_resolution: Target resolution (must be larger than source)
    
    Returns:
        Resampled CandleData
    """
    if len(data) == 0:
        return data
    
    df = data.to_dataframe()
    df.set_index('timestamp', inplace=True)
    
    # Map resolution to pandas offset
    offset_map = {
        "1m": "1T",
        "5m": "5T",
        "15m": "15T",
        "30m": "30T",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D"
    }
    
    offset = offset_map.get(target_resolution, "1H")
    
    resampled = df.resample(offset).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled = resampled.reset_index()
    
    return CandleData.from_dataframe(resampled, data.symbol, target_resolution)
