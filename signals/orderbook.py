"""
Orderbook Signals - Analyze order book for trading signals

Provides:
- Bid/ask imbalance detection
- Depth ratio analysis
- Wall detection (large orders)
- VWAP deviation
- Microstructure analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OrderbookLevel:
    """Single orderbook level"""
    price: float
    size: float


@dataclass
class OrderbookSnapshot:
    """Complete orderbook snapshot"""
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    timestamp: float
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid
    
    @property
    def spread_pct(self) -> float:
        if self.mid_price > 0:
            return self.spread / self.mid_price
        return 0.0


def parse_orderbook(data: Dict) -> OrderbookSnapshot:
    """
    Parse orderbook from API response
    
    Args:
        data: Dict with 'bids' and 'asks' arrays
    
    Returns:
        OrderbookSnapshot
    """
    bids = []
    asks = []
    
    for bid in data.get('bids', []):
        if isinstance(bid, (list, tuple)) and len(bid) >= 2:
            bids.append(OrderbookLevel(float(bid[0]), float(bid[1])))
        elif isinstance(bid, dict):
            bids.append(OrderbookLevel(float(bid.get('price', 0)), float(bid.get('size', 0))))
    
    for ask in data.get('asks', []):
        if isinstance(ask, (list, tuple)) and len(ask) >= 2:
            asks.append(OrderbookLevel(float(ask[0]), float(ask[1])))
        elif isinstance(ask, dict):
            asks.append(OrderbookLevel(float(ask.get('price', 0)), float(ask.get('size', 0))))
    
    # Sort bids descending, asks ascending
    bids.sort(key=lambda x: x.price, reverse=True)
    asks.sort(key=lambda x: x.price)
    
    return OrderbookSnapshot(
        bids=bids,
        asks=asks,
        timestamp=data.get('timestamp', 0)
    )


def compute_imbalance(orderbook: Dict, depth: int = 10) -> float:
    """
    Compute bid/ask volume imbalance
    
    Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    Positive: More buying pressure
    Negative: More selling pressure
    
    Args:
        orderbook: Orderbook dict with 'bids' and 'asks'
        depth: Number of levels to consider
    
    Returns:
        Imbalance ratio (-1 to 1)
    """
    ob = parse_orderbook(orderbook) if isinstance(orderbook, dict) else orderbook
    
    # Sum volumes up to depth
    bid_volume = sum(level.size for level in ob.bids[:depth])
    ask_volume = sum(level.size for level in ob.asks[:depth])
    
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    
    return (bid_volume - ask_volume) / total


def compute_depth_ratio(orderbook: Dict, price_range_pct: float = 0.01) -> float:
    """
    Compute depth ratio within price range
    
    Args:
        orderbook: Orderbook data
        price_range_pct: Price range as percentage of mid price
    
    Returns:
        Depth ratio (bid_depth / ask_depth)
    """
    ob = parse_orderbook(orderbook) if isinstance(orderbook, dict) else orderbook
    
    mid = ob.mid_price
    if mid == 0:
        return 1.0
    
    price_range = mid * price_range_pct
    
    # Sum volumes within range
    bid_depth = sum(
        level.size for level in ob.bids 
        if mid - price_range <= level.price <= mid
    )
    ask_depth = sum(
        level.size for level in ob.asks
        if mid <= level.price <= mid + price_range
    )
    
    if ask_depth == 0:
        return 10.0 if bid_depth > 0 else 1.0
    
    return bid_depth / ask_depth


def detect_walls(
    orderbook: Dict, 
    threshold_multiplier: float = 3.0,
    min_levels: int = 5
) -> Tuple[List[OrderbookLevel], List[OrderbookLevel]]:
    """
    Detect large orders (walls) in orderbook
    
    A wall is a level with significantly higher volume than average
    
    Args:
        orderbook: Orderbook data
        threshold_multiplier: Multiple of average to be considered a wall
        min_levels: Minimum levels for average calculation
    
    Returns:
        (bid_walls, ask_walls)
    """
    ob = parse_orderbook(orderbook) if isinstance(orderbook, dict) else orderbook
    
    bid_walls = []
    ask_walls = []
    
    # Calculate average sizes
    if len(ob.bids) >= min_levels:
        avg_bid = np.mean([level.size for level in ob.bids[:min_levels]])
        threshold_bid = avg_bid * threshold_multiplier
        
        bid_walls = [level for level in ob.bids if level.size >= threshold_bid]
    
    if len(ob.asks) >= min_levels:
        avg_ask = np.mean([level.size for level in ob.asks[:min_levels]])
        threshold_ask = avg_ask * threshold_multiplier
        
        ask_walls = [level for level in ob.asks if level.size >= threshold_ask]
    
    return bid_walls, ask_walls


def compute_weighted_mid(orderbook: Dict) -> float:
    """
    Compute volume-weighted mid price
    
    Weighted more towards side with less volume (more aggressive)
    
    Args:
        orderbook: Orderbook data
    
    Returns:
        Weighted mid price
    """
    ob = parse_orderbook(orderbook) if isinstance(orderbook, dict) else orderbook
    
    if not ob.bids or not ob.asks:
        return ob.mid_price
    
    best_bid = ob.bids[0]
    best_ask = ob.asks[0]
    
    total = best_bid.size + best_ask.size
    if total == 0:
        return ob.mid_price
    
    # Weight towards aggressive side (less volume)
    return (best_bid.price * best_ask.size + best_ask.price * best_bid.size) / total


def compute_vwap_deviation(orderbook: Dict, vwap: float) -> float:
    """
    Compute deviation of mid price from VWAP
    
    Args:
        orderbook: Orderbook data
        vwap: Current VWAP
    
    Returns:
        Deviation as percentage
    """
    ob = parse_orderbook(orderbook) if isinstance(orderbook, dict) else orderbook
    
    if vwap == 0:
        return 0.0
    
    return (ob.mid_price - vwap) / vwap


def compute_book_pressure(orderbook: Dict, levels: int = 5) -> Tuple[float, float]:
    """
    Compute buying and selling pressure from orderbook
    
    Args:
        orderbook: Orderbook data
        levels: Number of levels to analyze
    
    Returns:
        (buy_pressure, sell_pressure) as 0-1 normalized values
    """
    ob = parse_orderbook(orderbook) if isinstance(orderbook, dict) else orderbook
    
    bid_sizes = [level.size for level in ob.bids[:levels]]
    ask_sizes = [level.size for level in ob.asks[:levels]]
    
    # Weight by distance from best price
    weights = [1.0 / (i + 1) for i in range(levels)]
    
    bid_pressure = sum(s * w for s, w in zip(bid_sizes, weights)) if bid_sizes else 0
    ask_pressure = sum(s * w for s, w in zip(ask_sizes, weights)) if ask_sizes else 0
    
    max_pressure = max(bid_pressure, ask_pressure, 1)
    
    return bid_pressure / max_pressure, ask_pressure / max_pressure


def compute_spread_metrics(orderbook: Dict) -> Dict[str, float]:
    """
    Compute various spread metrics
    
    Args:
        orderbook: Orderbook data
    
    Returns:
        Dict with spread metrics
    """
    ob = parse_orderbook(orderbook) if isinstance(orderbook, dict) else orderbook
    
    return {
        "spread_absolute": ob.spread,
        "spread_pct": ob.spread_pct,
        "mid_price": ob.mid_price,
        "best_bid": ob.best_bid,
        "best_ask": ob.best_ask,
        "weighted_mid": compute_weighted_mid(ob)
    }


def compute_order_flow_imbalance(
    trades: List[Dict],
    lookback: int = 100
) -> float:
    """
    Compute order flow imbalance from recent trades
    
    Args:
        trades: List of trade dicts with 'side' and 'size'
        lookback: Number of trades to consider
    
    Returns:
        Imbalance (-1 to 1)
    """
    if not trades:
        return 0.0
    
    recent = trades[-lookback:] if len(trades) > lookback else trades
    
    buy_volume = sum(t.get('size', 0) for t in recent if t.get('side') == 'buy')
    sell_volume = sum(t.get('size', 0) for t in recent if t.get('side') == 'sell')
    
    total = buy_volume + sell_volume
    if total == 0:
        return 0.0
    
    return (buy_volume - sell_volume) / total


def get_orderbook_signal(orderbook: Dict, threshold: float = 0.05) -> int:
    """
    Get simple orderbook signal
    
    Args:
        orderbook: Orderbook data
        threshold: Imbalance threshold
    
    Returns:
        1 (bullish), -1 (bearish), or 0 (neutral)
    """
    imbalance = compute_imbalance(orderbook)
    
    if imbalance > threshold:
        return 1
    elif imbalance < -threshold:
        return -1
    else:
        return 0
