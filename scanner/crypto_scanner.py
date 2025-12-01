"""
Crypto Scanner - Analyze and rank all available cryptocurrencies

Provides:
- Real-time market scanning
- Multi-factor ranking (volume, volatility, momentum, liquidity)
- Opportunity detection
- Automatic selection of optimal trading pairs
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time


class OpportunityType(Enum):
    """Types of trading opportunities"""
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLUME_SURGE = "volume_surge"
    WHALE_ACTIVITY = "whale_activity"
    ARBITRAGE = "arbitrage"


@dataclass
class CryptoMetrics:
    """Comprehensive metrics for a cryptocurrency"""
    symbol: str
    price: float
    volume_24h: float
    market_cap: float = 0.0
    
    # Technical metrics
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    momentum_1h: float = 0.0
    momentum_24h: float = 0.0
    rsi_14: float = 50.0
    
    # Volume metrics
    volume_ratio: float = 1.0  # Current vs average
    buy_volume_pct: float = 0.5
    
    # Liquidity metrics
    spread_pct: float = 0.0
    depth_ratio: float = 1.0  # Bid depth / Ask depth
    
    # Trend metrics
    trend_strength: float = 0.0  # -1 to 1
    trend_duration_hours: int = 0
    
    # Opportunity score (0-100)
    opportunity_score: float = 0.0
    opportunity_type: Optional[OpportunityType] = None
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScanResult:
    """Result of market scan"""
    timestamp: datetime
    total_symbols: int
    opportunities: List[CryptoMetrics]
    top_momentum: List[CryptoMetrics]
    top_volume: List[CryptoMetrics]
    top_volatility: List[CryptoMetrics]
    mean_reversion_candidates: List[CryptoMetrics]


class CryptoScanner:
    """
    Advanced cryptocurrency market scanner
    
    Analyzes all available trading pairs and identifies
    optimal opportunities based on multiple factors.
    """
    
    def __init__(self, delta_client=None):
        """
        Initialize scanner
        
        Args:
            delta_client: Delta Exchange API client
        """
        self.client = delta_client
        self._cache: Dict[str, CryptoMetrics] = {}
        self._history: Dict[str, List[float]] = {}  # Price history
        self._last_scan: datetime = None
        self._scan_interval = 60  # Seconds between scans
    
    def scan_all(self, force: bool = False) -> ScanResult:
        """
        Scan all available cryptocurrencies
        
        Args:
            force: Force scan even if within interval
        
        Returns:
            ScanResult with ranked opportunities
        """
        now = datetime.utcnow()
        
        # Check scan interval
        if not force and self._last_scan:
            elapsed = (now - self._last_scan).total_seconds()
            if elapsed < self._scan_interval:
                return self._build_result_from_cache()
        
        # Get all products
        products = self._get_products()
        
        metrics_list = []
        for product in products:
            try:
                metrics = self._analyze_product(product)
                if metrics:
                    self._cache[metrics.symbol] = metrics
                    metrics_list.append(metrics)
            except Exception as e:
                continue
        
        self._last_scan = now
        
        return self._build_result(metrics_list)
    
    def get_top_opportunities(
        self,
        n: int = 5,
        min_volume: float = 100000,
        max_spread: float = 0.005
    ) -> List[CryptoMetrics]:
        """
        Get top trading opportunities
        
        Args:
            n: Number of opportunities to return
            min_volume: Minimum 24h volume
            max_spread: Maximum spread percentage
        
        Returns:
            List of top opportunities
        """
        if not self._cache:
            self.scan_all()
        
        candidates = [
            m for m in self._cache.values()
            if m.volume_24h >= min_volume and m.spread_pct <= max_spread
        ]
        
        # Sort by opportunity score
        candidates.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        return candidates[:n]
    
    def get_momentum_leaders(self, n: int = 5) -> List[CryptoMetrics]:
        """Get top momentum stocks"""
        candidates = list(self._cache.values())
        candidates.sort(key=lambda x: x.momentum_24h, reverse=True)
        return candidates[:n]
    
    def get_volatility_plays(self, n: int = 5) -> List[CryptoMetrics]:
        """Get high volatility opportunities"""
        candidates = list(self._cache.values())
        candidates.sort(key=lambda x: x.volatility_24h, reverse=True)
        return candidates[:n]
    
    def get_mean_reversion_candidates(self, n: int = 5) -> List[CryptoMetrics]:
        """Get oversold/overbought candidates for mean reversion"""
        candidates = []
        
        for m in self._cache.values():
            # Oversold
            if m.rsi_14 < 30:
                candidates.append((m, 30 - m.rsi_14))
            # Overbought
            elif m.rsi_14 > 70:
                candidates.append((m, m.rsi_14 - 70))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:n]]
    
    def detect_volume_surge(
        self,
        threshold: float = 3.0
    ) -> List[CryptoMetrics]:
        """Detect unusual volume activity"""
        return [
            m for m in self._cache.values()
            if m.volume_ratio >= threshold
        ]
    
    def _get_products(self) -> List[Dict]:
        """Get all available products from exchange"""
        if self.client:
            try:
                return self.client.get_products()
            except Exception:
                pass
        
        # Demo products for testing
        return [
            {'symbol': 'BTCUSD', 'product_id': 1},
            {'symbol': 'ETHUSD', 'product_id': 2},
            {'symbol': 'SOLUSD', 'product_id': 3},
            {'symbol': 'XRPUSD', 'product_id': 4},
            {'symbol': 'ADAUSD', 'product_id': 5},
            {'symbol': 'AVAXUSD', 'product_id': 6},
            {'symbol': 'DOTUSD', 'product_id': 7},
            {'symbol': 'LINKUSD', 'product_id': 8},
            {'symbol': 'MATICUSD', 'product_id': 9},
            {'symbol': 'ATOMUSD', 'product_id': 10},
        ]
    
    def _analyze_product(self, product: Dict) -> Optional[CryptoMetrics]:
        """Analyze a single product"""
        symbol = product.get('symbol', '')
        
        # Get market data
        ticker = self._get_ticker(symbol)
        if not ticker:
            return None
        
        # Get historical data for analysis
        candles = self._get_candles(symbol, '1h', 100)
        
        # Calculate metrics
        metrics = CryptoMetrics(
            symbol=symbol,
            price=ticker.get('last_price', 0),
            volume_24h=ticker.get('volume_24h', 0)
        )
        
        if candles:
            closes = np.array([c['close'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            volumes = np.array([c['volume'] for c in candles])
            
            # Volatility
            returns = np.diff(closes) / closes[:-1]
            metrics.volatility_1h = np.std(returns[-1:]) * 100 if len(returns) >= 1 else 0
            metrics.volatility_24h = np.std(returns[-24:]) * 100 if len(returns) >= 24 else 0
            
            # Momentum
            if len(closes) >= 24:
                metrics.momentum_1h = (closes[-1] / closes[-2] - 1) * 100
                metrics.momentum_24h = (closes[-1] / closes[-24] - 1) * 100
            
            # RSI
            metrics.rsi_14 = self._calculate_rsi(closes, 14)
            
            # Volume ratio
            avg_volume = np.mean(volumes[:-1])
            metrics.volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # Trend strength
            metrics.trend_strength = self._calculate_trend_strength(closes)
        
        # Get orderbook metrics
        orderbook = self._get_orderbook(symbol)
        if orderbook:
            metrics.spread_pct = self._calculate_spread(orderbook)
            metrics.depth_ratio = self._calculate_depth_ratio(orderbook)
        
        # Calculate opportunity score
        metrics.opportunity_score = self._calculate_opportunity_score(metrics)
        metrics.opportunity_type = self._classify_opportunity(metrics)
        
        return metrics
    
    def _get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker data"""
        if self.client:
            try:
                return self.client.get_ticker(symbol)
            except Exception:
                pass
        
        # Demo data
        np.random.seed(hash(symbol) % 2**32)
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        return {
            'last_price': base_price * (1 + np.random.randn() * 0.01),
            'volume_24h': np.random.uniform(1e6, 1e9)
        }
    
    def _get_candles(self, symbol: str, resolution: str, limit: int) -> List[Dict]:
        """Get historical candles"""
        if self.client:
            try:
                data = self.client.get_candles(symbol, resolution, limit)
                return data
            except Exception:
                pass
        
        # Generate demo candles
        np.random.seed(hash(symbol) % 2**32)
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        candles = []
        price = base_price
        for i in range(limit):
            change = np.random.randn() * 0.01
            open_price = price
            close_price = price * (1 + change)
            high = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.003))
            low = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.003))
            volume = np.random.uniform(100, 1000)
            
            candles.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
            price = close_price
        
        return candles
    
    def _get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get orderbook data"""
        if self.client:
            try:
                return self.client.get_orderbook(symbol)
            except Exception:
                pass
        
        # Demo orderbook
        return {
            'bids': [[99.9, 10], [99.8, 20], [99.7, 30]],
            'asks': [[100.1, 10], [100.2, 20], [100.3, 30]]
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength (-1 to 1)"""
        if len(prices) < 20:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize by price range
        price_range = prices.max() - prices.min()
        if price_range == 0:
            return 0.0
        
        normalized_slope = slope * len(prices) / price_range
        return np.clip(normalized_slope, -1, 1)
    
    def _calculate_spread(self, orderbook: Dict) -> float:
        """Calculate bid-ask spread percentage"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return 0.01
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        
        return (best_ask - best_bid) / mid if mid > 0 else 0.01
    
    def _calculate_depth_ratio(self, orderbook: Dict) -> float:
        """Calculate bid/ask depth ratio"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        bid_depth = sum(b[1] for b in bids[:10])
        ask_depth = sum(a[1] for a in asks[:10])
        
        return bid_depth / ask_depth if ask_depth > 0 else 1.0
    
    def _calculate_opportunity_score(self, metrics: CryptoMetrics) -> float:
        """Calculate composite opportunity score (0-100)"""
        score = 0.0
        
        # Volume score (0-25)
        if metrics.volume_ratio > 2:
            score += min(metrics.volume_ratio * 5, 25)
        else:
            score += metrics.volume_ratio * 5
        
        # Momentum score (0-25)
        abs_momentum = abs(metrics.momentum_24h)
        if abs_momentum > 5:
            score += min(abs_momentum * 2, 25)
        
        # Volatility score (0-20)
        if 1 < metrics.volatility_24h < 10:
            score += 20
        elif metrics.volatility_24h >= 10:
            score += 15
        
        # RSI extremes score (0-15)
        if metrics.rsi_14 < 25 or metrics.rsi_14 > 75:
            score += 15
        elif metrics.rsi_14 < 30 or metrics.rsi_14 > 70:
            score += 10
        
        # Trend strength score (0-15)
        score += abs(metrics.trend_strength) * 15
        
        return min(score, 100)
    
    def _classify_opportunity(self, metrics: CryptoMetrics) -> OpportunityType:
        """Classify the type of opportunity"""
        # Volume surge
        if metrics.volume_ratio > 3:
            return OpportunityType.VOLUME_SURGE
        
        # Mean reversion
        if metrics.rsi_14 < 25 or metrics.rsi_14 > 75:
            return OpportunityType.MEAN_REVERSION
        
        # Momentum
        if abs(metrics.momentum_24h) > 5:
            return OpportunityType.MOMENTUM
        
        # Volatility expansion
        if metrics.volatility_24h > 5:
            return OpportunityType.VOLATILITY_EXPANSION
        
        # Breakout/Breakdown
        if metrics.trend_strength > 0.7:
            return OpportunityType.BREAKOUT
        elif metrics.trend_strength < -0.7:
            return OpportunityType.BREAKDOWN
        
        return OpportunityType.MOMENTUM
    
    def _build_result(self, metrics_list: List[CryptoMetrics]) -> ScanResult:
        """Build scan result from metrics"""
        # Sort by various criteria
        by_score = sorted(metrics_list, key=lambda x: x.opportunity_score, reverse=True)
        by_momentum = sorted(metrics_list, key=lambda x: abs(x.momentum_24h), reverse=True)
        by_volume = sorted(metrics_list, key=lambda x: x.volume_24h, reverse=True)
        by_volatility = sorted(metrics_list, key=lambda x: x.volatility_24h, reverse=True)
        
        mean_reversion = [
            m for m in metrics_list
            if m.rsi_14 < 30 or m.rsi_14 > 70
        ]
        
        return ScanResult(
            timestamp=datetime.utcnow(),
            total_symbols=len(metrics_list),
            opportunities=by_score[:10],
            top_momentum=by_momentum[:5],
            top_volume=by_volume[:5],
            top_volatility=by_volatility[:5],
            mean_reversion_candidates=mean_reversion[:5]
        )
    
    def _build_result_from_cache(self) -> ScanResult:
        """Build result from cached data"""
        return self._build_result(list(self._cache.values()))
