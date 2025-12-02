"""
MARKET ANALYZER - Technical Analysis & Market Regime Detection
===============================================================
Comprehensive technical analysis engine for autonomous trading.

Features:
- Multi-timeframe analysis
- Pattern recognition
- Market regime detection
- Volatility analysis
- Support/Resistance levels
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
except ImportError:
    np = None

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("AladdinAI.MarketAnalyzer")


@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TechnicalIndicators:
    """Technical indicator values"""
    # Trend
    ema_20: float = 0
    ema_50: float = 0
    ema_200: float = 0
    sma_20: float = 0
    
    # Momentum
    rsi: float = 50
    macd: float = 0
    macd_signal: float = 0
    macd_histogram: float = 0
    stochastic_k: float = 50
    stochastic_d: float = 50
    
    # Volatility
    atr: float = 0
    bb_upper: float = 0
    bb_middle: float = 0
    bb_lower: float = 0
    
    # Volume
    volume_sma: float = 0
    obv: float = 0
    
    # Support/Resistance
    pivot: float = 0
    support_1: float = 0
    support_2: float = 0
    resistance_1: float = 0
    resistance_2: float = 0
    
    # Price Levels
    high_20: float = 0
    low_20: float = 0


class MarketAnalyzer:
    """
    Comprehensive market analysis engine
    """
    
    # Resolution mapping for Delta Exchange
    RESOLUTIONS = {
        "1m": "1m",
        "5m": "5m", 
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d"
    }
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._price_cache: Dict[str, List[OHLCV]] = {}
        self._indicator_cache: Dict[str, TechnicalIndicators] = {}
        self._last_fetch: Dict[str, datetime] = {}
        
        # API base URL
        self.base_url = "https://api.india.delta.exchange"
        
        logger.info("Market Analyzer initialized")
    
    def analyze(self, symbol: str, timeframe: str = "1h") -> Dict:
        """
        Perform comprehensive technical analysis
        
        Returns:
            Dict with analysis results
        """
        # Fetch latest data
        candles = self._fetch_candles(symbol, timeframe, limit=200)
        
        if not candles or len(candles) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return self._default_analysis()
        
        # Calculate indicators
        indicators = self._calculate_indicators(candles)
        
        # Analyze components
        trend = self._analyze_trend(candles, indicators)
        momentum = self._analyze_momentum(indicators)
        volatility = self._analyze_volatility(candles, indicators)
        volume_profile = self._analyze_volume(candles, indicators)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": candles[-1].close,
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility,
            "volume_profile": volume_profile,
            "indicators": indicators,
            "timestamp": datetime.now().isoformat()
        }
    
    def _default_analysis(self) -> Dict:
        """Default analysis when data unavailable"""
        return {
            "trend": 0,
            "momentum": 0,
            "volatility": 0.5,
            "volume_profile": "normal",
            "indicators": TechnicalIndicators()
        }
    
    def get_current_data(self, symbol: str) -> Dict:
        """Get current market data with indicators"""
        analysis = self.analyze(symbol)
        indicators = analysis.get("indicators", TechnicalIndicators())
        
        if isinstance(indicators, TechnicalIndicators):
            return {
                "price": analysis.get("price", 0),
                "rsi": indicators.rsi,
                "macd": indicators.macd,
                "macd_signal": indicators.macd_signal,
                "bb_upper": indicators.bb_upper,
                "bb_middle": indicators.bb_middle,
                "bb_lower": indicators.bb_lower,
                "atr": indicators.atr,
                "ema_20": indicators.ema_20,
                "ema_50": indicators.ema_50,
                "high_20": indicators.high_20,
                "low_20": indicators.low_20,
                "volume": indicators.volume_sma,
                "avg_volume": indicators.volume_sma
            }
        return {"price": analysis.get("price", 0)}
    
    def _fetch_candles(self, symbol: str, resolution: str, limit: int = 200) -> List[OHLCV]:
        """Fetch OHLCV data from Delta Exchange"""
        cache_key = f"{symbol}_{resolution}"
        
        # Check cache (5 minute expiry for 1h+ timeframes)
        if cache_key in self._last_fetch:
            age = (datetime.now() - self._last_fetch[cache_key]).seconds
            cache_expiry = 60 if resolution in ["1m", "5m"] else 300
            if age < cache_expiry and cache_key in self._price_cache:
                return self._price_cache[cache_key]
        
        try:
            end = int(time.time())
            start = end - (limit * self._resolution_to_seconds(resolution))
            
            url = f"{self.base_url}/v2/history/candles"
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "start": start,
                "end": end
            }
            
            if requests:
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("success") and data.get("result"):
                        candles = []
                        for c in data["result"]:
                            candles.append(OHLCV(
                                timestamp=datetime.fromtimestamp(c["time"]),
                                open=float(c["open"]),
                                high=float(c["high"]),
                                low=float(c["low"]),
                                close=float(c["close"]),
                                volume=float(c.get("volume", 0))
                            ))
                        
                        # Sort by timestamp
                        candles.sort(key=lambda x: x.timestamp)
                        
                        self._price_cache[cache_key] = candles
                        self._last_fetch[cache_key] = datetime.now()
                        
                        logger.debug(f"Fetched {len(candles)} candles for {symbol}")
                        return candles
                        
        except Exception as e:
            logger.warning(f"Failed to fetch candles: {e}")
        
        return self._price_cache.get(cache_key, [])
    
    def _resolution_to_seconds(self, resolution: str) -> int:
        """Convert resolution to seconds"""
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "1d": 86400
        }
        return mapping.get(resolution, 3600)
    
    def _calculate_indicators(self, candles: List[OHLCV]) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        if not candles or len(candles) < 20:
            return TechnicalIndicators()
        
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        
        indicators = TechnicalIndicators()
        
        # EMAs
        indicators.ema_20 = self._ema(closes, 20)
        indicators.ema_50 = self._ema(closes, 50) if len(closes) >= 50 else indicators.ema_20
        indicators.ema_200 = self._ema(closes, 200) if len(closes) >= 200 else indicators.ema_50
        indicators.sma_20 = self._sma(closes, 20)
        
        # RSI
        indicators.rsi = self._rsi(closes, 14)
        
        # MACD
        macd_line, signal_line, histogram = self._macd(closes)
        indicators.macd = macd_line
        indicators.macd_signal = signal_line
        indicators.macd_histogram = histogram
        
        # Stochastic
        indicators.stochastic_k, indicators.stochastic_d = self._stochastic(highs, lows, closes)
        
        # ATR
        indicators.atr = self._atr(highs, lows, closes, 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(closes, 20, 2)
        indicators.bb_upper = bb_upper
        indicators.bb_middle = bb_middle
        indicators.bb_lower = bb_lower
        
        # Volume
        indicators.volume_sma = self._sma(volumes, 20)
        
        # Support/Resistance (Pivot Points)
        last_high = highs[-1]
        last_low = lows[-1]
        last_close = closes[-1]
        
        indicators.pivot = (last_high + last_low + last_close) / 3
        indicators.support_1 = 2 * indicators.pivot - last_high
        indicators.support_2 = indicators.pivot - (last_high - last_low)
        indicators.resistance_1 = 2 * indicators.pivot - last_low
        indicators.resistance_2 = indicators.pivot + (last_high - last_low)
        
        # 20-period high/low
        indicators.high_20 = max(highs[-20:])
        indicators.low_20 = min(lows[-20:])
        
        return indicators
    
    def _sma(self, data: List[float], period: int) -> float:
        """Simple Moving Average"""
        if len(data) < period:
            return data[-1] if data else 0
        return sum(data[-period:]) / period
    
    def _ema(self, data: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(data) < period:
            return data[-1] if data else 0
        
        multiplier = 2 / (period + 1)
        ema = self._sma(data[:period], period)
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _rsi(self, data: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(data)):
            change = data[i] - data[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        
        avg_gain = self._sma(gains[-period:], period)
        avg_loss = self._sma(losses[-period:], period)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _macd(self, data: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD Indicator"""
        if len(data) < slow:
            return 0, 0, 0
        
        ema_fast = self._ema(data, fast)
        ema_slow = self._ema(data, slow)
        macd_line = ema_fast - ema_slow
        
        # Calculate MACD history for signal line
        macd_history = []
        for i in range(slow, len(data) + 1):
            subset = data[:i]
            ema_f = self._ema(subset, fast)
            ema_s = self._ema(subset, slow)
            macd_history.append(ema_f - ema_s)
        
        signal_line = self._ema(macd_history, signal) if len(macd_history) >= signal else macd_line
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                    k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Stochastic Oscillator"""
        if len(closes) < k_period:
            return 50, 50
        
        highest_high = max(highs[-k_period:])
        lowest_low = min(lows[-k_period:])
        
        if highest_high == lowest_low:
            k = 50
        else:
            k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (SMA of %K)
        k_values = []
        for i in range(d_period):
            if len(closes) >= k_period + i:
                idx = -(i + 1)
                hh = max(highs[idx - k_period:idx] if idx != -1 else highs[-k_period:])
                ll = min(lows[idx - k_period:idx] if idx != -1 else lows[-k_period:])
                if hh != ll:
                    k_values.append(((closes[idx] - ll) / (hh - ll)) * 100)
        
        d = sum(k_values) / len(k_values) if k_values else k
        
        return k, d
    
    def _atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Average True Range"""
        if len(closes) < 2:
            return 0
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        return self._sma(true_ranges[-period:], period)
    
    def _bollinger_bands(self, data: List[float], period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Bollinger Bands"""
        if len(data) < period:
            price = data[-1] if data else 0
            return price * 1.02, price, price * 0.98
        
        middle = self._sma(data, period)
        
        # Calculate standard deviation
        subset = data[-period:]
        variance = sum((x - middle) ** 2 for x in subset) / period
        std = variance ** 0.5
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def _analyze_trend(self, candles: List[OHLCV], indicators: TechnicalIndicators) -> float:
        """
        Analyze trend direction and strength
        Returns: -1 (strong bearish) to 1 (strong bullish)
        """
        if not candles:
            return 0
        
        price = candles[-1].close
        
        # EMA alignment score
        ema_score = 0
        if price > indicators.ema_20:
            ema_score += 0.2
        if price > indicators.ema_50:
            ema_score += 0.2
        if indicators.ema_20 > indicators.ema_50:
            ema_score += 0.2
        if indicators.ema_50 > indicators.ema_200:
            ema_score += 0.2
        
        # Invert for bearish
        if price < indicators.ema_20:
            ema_score -= 0.2
        if price < indicators.ema_50:
            ema_score -= 0.2
        if indicators.ema_20 < indicators.ema_50:
            ema_score -= 0.2
        if indicators.ema_50 < indicators.ema_200:
            ema_score -= 0.2
        
        # MACD trend
        macd_score = 0
        if indicators.macd > indicators.macd_signal:
            macd_score = 0.2
        elif indicators.macd < indicators.macd_signal:
            macd_score = -0.2
        
        # Price momentum (rate of change)
        if len(candles) >= 20:
            roc = (price - candles[-20].close) / candles[-20].close
            roc_score = max(-0.2, min(0.2, roc * 2))
        else:
            roc_score = 0
        
        trend = ema_score + macd_score + roc_score
        return max(-1, min(1, trend))
    
    def _analyze_momentum(self, indicators: TechnicalIndicators) -> float:
        """
        Analyze momentum
        Returns: -1 (bearish momentum) to 1 (bullish momentum)
        """
        # RSI component
        rsi = indicators.rsi
        if rsi > 70:
            rsi_score = 0.3
        elif rsi > 50:
            rsi_score = (rsi - 50) / 50 * 0.3
        elif rsi < 30:
            rsi_score = -0.3
        else:
            rsi_score = (rsi - 50) / 50 * 0.3
        
        # MACD histogram
        hist = indicators.macd_histogram
        if hist > 0:
            macd_score = min(0.3, hist / 100)
        else:
            macd_score = max(-0.3, hist / 100)
        
        # Stochastic
        stoch = indicators.stochastic_k
        if stoch > 80:
            stoch_score = 0.2
        elif stoch < 20:
            stoch_score = -0.2
        else:
            stoch_score = (stoch - 50) / 50 * 0.2
        
        momentum = rsi_score + macd_score + stoch_score
        return max(-1, min(1, momentum))
    
    def _analyze_volatility(self, candles: List[OHLCV], indicators: TechnicalIndicators) -> float:
        """
        Analyze volatility
        Returns: 0 (low) to 1 (high)
        """
        if not candles or indicators.atr == 0:
            return 0.5
        
        price = candles[-1].close
        atr_pct = indicators.atr / price
        
        # Normalize ATR percentage (typical range 0.5% to 5%)
        volatility = min(1, atr_pct / 0.05)
        
        # Bollinger Band width
        bb_width = (indicators.bb_upper - indicators.bb_lower) / indicators.bb_middle
        bb_volatility = min(1, bb_width / 0.1)
        
        return (volatility + bb_volatility) / 2
    
    def _analyze_volume(self, candles: List[OHLCV], indicators: TechnicalIndicators) -> str:
        """Analyze volume profile"""
        if not candles or indicators.volume_sma == 0:
            return "normal"
        
        current_volume = candles[-1].volume
        avg_volume = indicators.volume_sma
        
        ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if ratio > 2:
            return "very_high"
        elif ratio > 1.5:
            return "high"
        elif ratio < 0.5:
            return "low"
        else:
            return "normal"
    
    def get_support_resistance(self, symbol: str) -> Dict:
        """Get support and resistance levels"""
        analysis = self.analyze(symbol)
        indicators = analysis.get("indicators", TechnicalIndicators())
        
        if isinstance(indicators, TechnicalIndicators):
            return {
                "pivot": indicators.pivot,
                "support_1": indicators.support_1,
                "support_2": indicators.support_2,
                "resistance_1": indicators.resistance_1,
                "resistance_2": indicators.resistance_2,
                "high_20": indicators.high_20,
                "low_20": indicators.low_20
            }
        return {}
    
    def get_multi_timeframe_analysis(self, symbol: str) -> Dict:
        """Analyze across multiple timeframes"""
        timeframes = ["15m", "1h", "4h", "1d"]
        results = {}
        
        for tf in timeframes:
            analysis = self.analyze(symbol, tf)
            results[tf] = {
                "trend": analysis.get("trend", 0),
                "momentum": analysis.get("momentum", 0),
                "volatility": analysis.get("volatility", 0)
            }
        
        # Aggregate trend across timeframes
        trends = [r["trend"] for r in results.values()]
        avg_trend = sum(trends) / len(trends)
        
        # Higher timeframes have more weight
        weighted_trend = (
            results.get("15m", {}).get("trend", 0) * 0.1 +
            results.get("1h", {}).get("trend", 0) * 0.2 +
            results.get("4h", {}).get("trend", 0) * 0.3 +
            results.get("1d", {}).get("trend", 0) * 0.4
        )
        
        return {
            "timeframes": results,
            "aggregate_trend": avg_trend,
            "weighted_trend": weighted_trend,
            "alignment": "aligned" if all(t > 0 for t in trends) or all(t < 0 for t in trends) else "mixed"
        }
    
    def print_analysis(self, symbol: str):
        """Print formatted analysis"""
        analysis = self.analyze(symbol)
        indicators = analysis.get("indicators", TechnicalIndicators())
        
        print("\n" + "=" * 60)
        print(f"TECHNICAL ANALYSIS - {symbol}")
        print("=" * 60)
        print(f"\n  Price: ${analysis.get('price', 0):,.2f}")
        print(f"  Trend: {analysis.get('trend', 0):.2f}")
        print(f"  Momentum: {analysis.get('momentum', 0):.2f}")
        print(f"  Volatility: {analysis.get('volatility', 0):.2f}")
        print(f"  Volume: {analysis.get('volume_profile', 'normal')}")
        
        if isinstance(indicators, TechnicalIndicators):
            print(f"\n  Indicators:")
            print(f"    RSI: {indicators.rsi:.1f}")
            print(f"    MACD: {indicators.macd:.2f} / Signal: {indicators.macd_signal:.2f}")
            print(f"    EMA 20/50: {indicators.ema_20:.2f} / {indicators.ema_50:.2f}")
            print(f"    ATR: {indicators.atr:.2f}")
            print(f"    BB: {indicators.bb_lower:.2f} - {indicators.bb_upper:.2f}")
        
        print("=" * 60)
