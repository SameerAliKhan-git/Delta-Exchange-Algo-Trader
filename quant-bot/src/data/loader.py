"""
Data Loader Module

Handles data ingestion from various sources:
- CSV files
- APIs (CCXT, Yahoo Finance)
- Database connections
- Caching layer for performance
"""

import os
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from loguru import logger

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import ccxt
except ImportError:
    ccxt = None


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load OHLCV data."""
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate loaded data."""
        pass


class CSVDataLoader(BaseDataLoader):
    """
    Load OHLCV data from CSV files.
    
    Expected columns: timestamp/date, open, high, low, close, volume
    
    Example:
        loader = CSVDataLoader("data/btcusd.csv")
        df = loader.load(start="2020-01-01", end="2023-12-31")
    """
    
    REQUIRED_COLUMNS = {'open', 'high', 'low', 'close', 'volume'}
    DATE_COLUMNS = ['timestamp', 'date', 'datetime', 'time']
    
    def __init__(
        self,
        filepath: str,
        datetime_col: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize CSV loader.
        
        Args:
            filepath: Path to CSV file
            datetime_col: Name of datetime column (auto-detected if None)
            cache_dir: Directory for caching processed data
        """
        self.filepath = Path(filepath)
        self.datetime_col = datetime_col
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    def _detect_datetime_col(self, df: pd.DataFrame) -> str:
        """Auto-detect datetime column."""
        columns_lower = {col.lower(): col for col in df.columns}
        
        for date_col in self.DATE_COLUMNS:
            if date_col in columns_lower:
                return columns_lower[date_col]
        
        # Try to find column with dates
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head())
                return col
            except (ValueError, TypeError):
                continue
        
        raise ValueError("Could not detect datetime column")
    
    def _get_cache_key(self, start: str, end: str) -> str:
        """Generate cache key."""
        key = f"{self.filepath}_{start}_{end}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load from cache if available."""
        if not self.cache_dir:
            return None
        
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_key: str) -> None:
        """Save to cache."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        df.to_parquet(cache_path)
        logger.debug(f"Saved to cache: {cache_path}")
    
    def load(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        resample: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            resample: Resample frequency (e.g., '1H', '4H', '1D')
            
        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        # Check cache
        if start and end:
            cache_key = self._get_cache_key(start, end)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Load CSV
        logger.info(f"Loading CSV: {self.filepath}")
        df = pd.read_csv(self.filepath, **kwargs)
        
        # Detect and set datetime index
        datetime_col = self.datetime_col or self._detect_datetime_col(df)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
        df.index.name = 'timestamp'
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Validate required columns
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Select only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Filter by date range
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        
        # Sort by datetime
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Resample if requested
        if resample:
            df = self._resample(df, resample)
        
        # Forward fill missing values (limited)
        df = df.ffill(limit=5)
        
        # Drop remaining NaN rows
        df = df.dropna()
        
        # Validate
        if not self.validate(df):
            logger.warning("Data validation failed")
        
        # Cache result
        if start and end:
            self._save_to_cache(df, cache_key)
        
        logger.info(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
        return df
    
    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLCV data to different timeframe."""
        return df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data integrity.
        
        Checks:
        - High >= Low
        - High >= Open, Close
        - Low <= Open, Close
        - Volume >= 0
        - No extreme outliers
        """
        issues = []
        
        # Check high >= low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"{invalid_hl} rows where high < low")
        
        # Check high >= open/close
        invalid_ho = (df['high'] < df['open']).sum()
        invalid_hc = (df['high'] < df['close']).sum()
        if invalid_ho > 0 or invalid_hc > 0:
            issues.append(f"{invalid_ho + invalid_hc} rows where high < open or close")
        
        # Check low <= open/close
        invalid_lo = (df['low'] > df['open']).sum()
        invalid_lc = (df['low'] > df['close']).sum()
        if invalid_lo > 0 or invalid_lc > 0:
            issues.append(f"{invalid_lo + invalid_lc} rows where low > open or close")
        
        # Check non-negative volume
        neg_vol = (df['volume'] < 0).sum()
        if neg_vol > 0:
            issues.append(f"{neg_vol} rows with negative volume")
        
        # Check for extreme price changes (>50% in single bar)
        pct_change = df['close'].pct_change().abs()
        extreme = (pct_change > 0.5).sum()
        if extreme > 0:
            issues.append(f"{extreme} rows with >50% price change")
        
        if issues:
            logger.warning(f"Data validation issues: {issues}")
            return False
        
        return True


class CCXTDataLoader(BaseDataLoader):
    """
    Load OHLCV data from crypto exchanges via CCXT.
    
    Example:
        loader = CCXTDataLoader("binance")
        df = loader.load(symbol="BTC/USDT", timeframe="1h")
    """
    
    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize CCXT loader.
        
        Args:
            exchange_id: CCXT exchange ID (e.g., 'binance', 'ftx')
            api_key: Optional API key
            api_secret: Optional API secret
            cache_dir: Directory for caching
        """
        if ccxt is None:
            raise ImportError("ccxt not installed. Run: pip install ccxt")
        
        self.exchange_id = exchange_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        config = {}
        if api_key:
            config['apiKey'] = api_key
        if api_secret:
            config['secret'] = api_secret
        
        self.exchange = exchange_class(config)
        self.exchange.load_markets()
        
        logger.info(f"Initialized {exchange_id} with {len(self.exchange.markets)} markets")
    
    def load(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load OHLCV data from exchange.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1m", "1h", "1d")
            start: Start datetime
            end: End datetime
            limit: Max candles per request
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {symbol} {timeframe} from {self.exchange_id}")
        
        # Convert start/end to timestamps
        since = None
        if start:
            since = int(pd.Timestamp(start).timestamp() * 1000)
        
        end_ts = None
        if end:
            end_ts = int(pd.Timestamp(end).timestamp() * 1000)
        
        # Fetch data in chunks
        all_ohlcv = []
        
        while True:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Get last timestamp for next iteration
            last_ts = ohlcv[-1][0]
            
            # Check if we've reached the end
            if end_ts and last_ts >= end_ts:
                break
            
            # Check if we got less than limit (no more data)
            if len(ohlcv) < limit:
                break
            
            # Move to next batch
            since = last_ts + 1
            
            logger.debug(f"Fetched {len(all_ohlcv)} candles so far...")
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Filter by end date if specified
        if end:
            df = df[df.index <= end]
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate CCXT data."""
        if df.empty:
            return False
        
        # Check for required columns
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            return False
        
        return True


class YahooDataLoader(BaseDataLoader):
    """
    Load OHLCV data from Yahoo Finance.
    
    Example:
        loader = YahooDataLoader()
        df = loader.load(symbol="AAPL", start="2020-01-01")
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize Yahoo Finance loader."""
        if yf is None:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
    
    def load(
        self,
        symbol: str = "SPY",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol (e.g., "AAPL", "SPY")
            start: Start date
            end: End date
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {symbol} from Yahoo Finance")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=end,
            interval=interval
        )
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Select OHLCV columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        
        # Set index name
        df.index.name = 'timestamp'
        
        logger.info(f"Loaded {len(df)} rows for {symbol}")
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate Yahoo data."""
        return not df.empty and 'close' in df.columns


class DataPipeline:
    """
    Pipeline for loading, validating, and preprocessing data.
    
    Example:
        pipeline = DataPipeline()
        pipeline.add_loader(CSVDataLoader("data/btc.csv"))
        df = pipeline.execute(start="2020-01-01", end="2023-12-31")
    """
    
    def __init__(self):
        """Initialize pipeline."""
        self.loaders: List[BaseDataLoader] = []
        self.transformers: List[callable] = []
    
    def add_loader(self, loader: BaseDataLoader) -> "DataPipeline":
        """Add data loader to pipeline."""
        self.loaders.append(loader)
        return self
    
    def add_transformer(self, func: callable) -> "DataPipeline":
        """Add transformation function to pipeline."""
        self.transformers.append(func)
        return self
    
    def execute(self, **kwargs) -> pd.DataFrame:
        """Execute pipeline and return combined data."""
        if not self.loaders:
            raise ValueError("No loaders configured")
        
        # Load data from all sources
        dfs = []
        for loader in self.loaders:
            try:
                df = loader.load(**kwargs)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Loader failed: {e}")
        
        if not dfs:
            raise ValueError("No data loaded from any source")
        
        # Combine if multiple loaders
        if len(dfs) > 1:
            df = pd.concat(dfs, axis=0)
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
        else:
            df = dfs[0]
        
        # Apply transformers
        for transformer in self.transformers:
            df = transformer(df)
        
        return df


def create_sample_data(
    filepath: str = "data/sample_ohlcv.csv",
    n_rows: int = 10000,
    start_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing.
    
    Args:
        filepath: Output path
        n_rows: Number of rows to generate
        start_price: Starting price
        volatility: Daily volatility
        
    Returns:
        Generated DataFrame
    """
    np.random.seed(42)
    
    # Generate timestamps (hourly)
    start = datetime(2020, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(n_rows)]
    
    # Generate price with random walk
    returns = np.random.normal(0.0001, volatility, n_rows)
    prices = start_price * np.cumprod(1 + returns)
    
    # Generate OHLCV
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close
        high_mult = 1 + np.random.uniform(0, volatility)
        low_mult = 1 - np.random.uniform(0, volatility)
        
        if i == 0:
            open_price = start_price
        else:
            open_price = data[-1]['close']
        
        high = max(open_price, close) * high_mult
        low = min(open_price, close) * low_mult
        
        # Volume with some patterns
        base_volume = 1000000
        volume = base_volume * (1 + np.random.exponential(0.5))
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Save to file
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Created sample data: {filepath} ({n_rows} rows)")
    
    return df


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("DATA LOADER DEMO")
    print("=" * 60)
    
    # Create sample data
    sample_file = "data/sample_ohlcv.csv"
    create_sample_data(sample_file)
    
    # Load with CSV loader
    loader = CSVDataLoader(sample_file)
    df = loader.load(start="2020-06-01", end="2021-06-01")
    
    print(f"\nLoaded {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nSample data:")
    print(df.head())
    print(f"\nStatistics:")
    print(df.describe())
