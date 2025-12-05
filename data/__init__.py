"""
Data Module - Historical data loading and storage

Provides:
- OHLCV data loading from Delta Exchange
- File-based data caching
- Data preprocessing and validation
"""

from .loader import (
    DataLoader,
    CandleData,
    load_candles,
    load_candles_range,
    get_available_symbols,
    preprocess_candles
)
from .catalog import (
    ProductCatalog,
    get_futures_catalog,
    get_options_catalog,
    filter_by_liquidity
)

__all__ = [
    # Loader
    "DataLoader",
    "CandleData",
    "load_candles",
    "load_candles_range",
    "get_available_symbols",
    "preprocess_candles",
    
    # Catalog
    "ProductCatalog",
    "get_futures_catalog",
    "get_options_catalog",
    "filter_by_liquidity",
]
