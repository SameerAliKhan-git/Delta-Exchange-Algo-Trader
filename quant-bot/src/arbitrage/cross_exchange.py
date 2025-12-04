"""
Cross-Exchange Arbitrage Router
================================
Multi-exchange price arbitrage with latency-normalized execution.

Author: Quant Bot
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging
import asyncio
from collections import deque
import heapq

logger = logging.getLogger(__name__)


class Exchange(Enum):
    BINANCE = "binance"
    DELTA = "delta"
    DERIBIT = "deribit"
    OKX = "okx"
    BYBIT = "bybit"
    FTX = "ftx"
    HUOBI = "huobi"
    KRAKEN = "kraken"
    COINBASE = "coinbase"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class ArbitrageType(Enum):
    SPATIAL = "spatial"           # Same asset, different exchanges
    TRIANGULAR = "triangular"     # A→B→C→A cycle
    CROSS_ASSET = "cross_asset"   # Synthetic vs direct


@dataclass
class ExchangeConfig:
    """Configuration for an exchange."""
    exchange: Exchange
    maker_fee_bps: float
    taker_fee_bps: float
    withdrawal_fee: float
    min_order_size: float
    latency_ms: float              # Typical latency
    latency_std_ms: float          # Latency standard deviation
    api_rate_limit: int            # Requests per second
    supports_websocket: bool = True
    margin_available: bool = True


@dataclass
class OrderBookLevel:
    """Single price level in orderbook."""
    price: float
    quantity: float
    exchange: Exchange


@dataclass
class AggregatedBook:
    """Aggregated orderbook across exchanges."""
    symbol: str
    bids: List[OrderBookLevel]     # Sorted by price descending
    asks: List[OrderBookLevel]     # Sorted by price ascending
    timestamp: datetime
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None


@dataclass
class ArbitrageOpportunity:
    """Detected cross-exchange arbitrage opportunity."""
    arb_type: ArbitrageType
    symbol: str
    buy_exchange: Exchange
    sell_exchange: Exchange
    buy_price: float
    sell_price: float
    spread: float
    spread_bps: float
    max_quantity: float
    expected_profit: float
    fees_total: float
    net_profit: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # For triangular
    path: Optional[List[Tuple[Exchange, str, OrderSide]]] = None
    
    @property
    def is_profitable(self) -> bool:
        return self.net_profit > 0
    
    @property
    def profit_after_slippage(self) -> float:
        # Assume 20% slippage on expected
        return self.net_profit * 0.8


@dataclass
class ArbitrageExecution:
    """Execution result for an arbitrage trade."""
    opportunity: ArbitrageOpportunity
    executed_quantity: float
    buy_fill_price: float
    sell_fill_price: float
    actual_spread: float
    fees_paid: float
    realized_profit: float
    execution_time_ms: float
    slippage_bps: float
    success: bool
    error_message: Optional[str] = None


class OrderBookAggregator:
    """
    Aggregate orderbooks from multiple exchanges.
    """
    
    def __init__(self):
        self.books: Dict[str, Dict[Exchange, Tuple[List, List, datetime]]] = {}
        self.latencies: Dict[Exchange, deque] = {}
        
        logger.info("OrderBookAggregator initialized")
    
    def update_book(
        self,
        symbol: str,
        exchange: Exchange,
        bids: List[Tuple[float, float]],  # [(price, qty), ...]
        asks: List[Tuple[float, float]],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update orderbook for a symbol/exchange."""
        timestamp = timestamp or datetime.now()
        
        if symbol not in self.books:
            self.books[symbol] = {}
        
        self.books[symbol][exchange] = (bids, asks, timestamp)
    
    def get_aggregated_book(
        self,
        symbol: str,
        max_age_ms: float = 1000
    ) -> Optional[AggregatedBook]:
        """
        Get aggregated orderbook across all exchanges.
        
        Only includes books newer than max_age_ms.
        """
        if symbol not in self.books:
            return None
        
        cutoff = datetime.now() - timedelta(milliseconds=max_age_ms)
        
        all_bids = []
        all_asks = []
        
        for exchange, (bids, asks, ts) in self.books[symbol].items():
            if ts < cutoff:
                continue
            
            for price, qty in bids:
                all_bids.append(OrderBookLevel(price, qty, exchange))
            
            for price, qty in asks:
                all_asks.append(OrderBookLevel(price, qty, exchange))
        
        if not all_bids or not all_asks:
            return None
        
        # Sort bids descending, asks ascending
        all_bids.sort(key=lambda x: -x.price)
        all_asks.sort(key=lambda x: x.price)
        
        return AggregatedBook(
            symbol=symbol,
            bids=all_bids,
            asks=all_asks,
            timestamp=datetime.now()
        )
    
    def get_best_prices(
        self,
        symbol: str
    ) -> Dict[Exchange, Tuple[float, float]]:
        """
        Get best bid/ask per exchange.
        
        Returns dict of exchange -> (best_bid, best_ask).
        """
        if symbol not in self.books:
            return {}
        
        result = {}
        
        for exchange, (bids, asks, _) in self.books[symbol].items():
            if bids and asks:
                result[exchange] = (bids[0][0], asks[0][0])
        
        return result


class LatencyNormalizer:
    """
    Normalize prices for latency differences between exchanges.
    """
    
    def __init__(self):
        self.latency_history: Dict[Exchange, deque] = {}
        self.price_velocity: Dict[str, float] = {}  # symbol -> price change per ms
        
        logger.info("LatencyNormalizer initialized")
    
    def update_latency(
        self,
        exchange: Exchange,
        latency_ms: float
    ) -> None:
        """Record latency observation."""
        if exchange not in self.latency_history:
            self.latency_history[exchange] = deque(maxlen=1000)
        self.latency_history[exchange].append(latency_ms)
    
    def get_expected_latency(self, exchange: Exchange) -> float:
        """Get expected latency for exchange."""
        if exchange not in self.latency_history:
            return 100  # Default 100ms
        
        return np.mean(list(self.latency_history[exchange]))
    
    def update_price_velocity(
        self,
        symbol: str,
        prices: List[Tuple[datetime, float]]
    ) -> None:
        """
        Calculate price velocity (change per ms).
        """
        if len(prices) < 2:
            return
        
        velocities = []
        for i in range(1, len(prices)):
            dt = (prices[i][0] - prices[i-1][0]).total_seconds() * 1000
            dp = abs(prices[i][1] - prices[i-1][1])
            if dt > 0:
                velocities.append(dp / dt)
        
        self.price_velocity[symbol] = np.mean(velocities) if velocities else 0
    
    def adjust_price_for_latency(
        self,
        symbol: str,
        price: float,
        exchange: Exchange,
        is_bid: bool
    ) -> float:
        """
        Adjust price for expected latency.
        
        If we're slower, the price may have moved against us.
        """
        latency = self.get_expected_latency(exchange)
        velocity = self.price_velocity.get(symbol, 0)
        
        # Expected price movement during our latency
        expected_move = velocity * latency
        
        # Conservative adjustment (assume price moves against us)
        if is_bid:
            # If we're buying, price might have gone up
            return price + expected_move
        else:
            # If we're selling, price might have gone down
            return price - expected_move


class CrossExchangeArbitrageEngine:
    """
    Production-grade cross-exchange arbitrage engine.
    
    Features:
    - Multi-exchange orderbook aggregation
    - Latency-normalized price comparison
    - Fee and transfer cost modeling
    - Execution feasibility checking
    - Triangular arbitrage detection
    """
    
    def __init__(
        self,
        exchange_configs: Optional[Dict[Exchange, ExchangeConfig]] = None,
        min_profit_bps: float = 10,
        max_position_usd: float = 10000,
        execution_buffer_bps: float = 5
    ):
        self.book_aggregator = OrderBookAggregator()
        self.latency_normalizer = LatencyNormalizer()
        
        self.exchange_configs = exchange_configs or self._default_configs()
        self.min_profit_bps = min_profit_bps
        self.max_position_usd = max_position_usd
        self.execution_buffer_bps = execution_buffer_bps
        
        self.opportunities_history: deque = deque(maxlen=10000)
        self.executions: List[ArbitrageExecution] = []
        
        logger.info("CrossExchangeArbitrageEngine initialized")
    
    def _default_configs(self) -> Dict[Exchange, ExchangeConfig]:
        """Default exchange configurations."""
        return {
            Exchange.BINANCE: ExchangeConfig(
                Exchange.BINANCE, 10, 10, 0.0005, 10, 50, 20, 1200
            ),
            Exchange.DELTA: ExchangeConfig(
                Exchange.DELTA, 5, 7.5, 0.0003, 10, 80, 30, 300
            ),
            Exchange.DERIBIT: ExchangeConfig(
                Exchange.DERIBIT, 3, 5, 0.0002, 10, 60, 25, 500
            ),
            Exchange.OKX: ExchangeConfig(
                Exchange.OKX, 8, 10, 0.0004, 10, 70, 25, 600
            ),
            Exchange.BYBIT: ExchangeConfig(
                Exchange.BYBIT, 10, 10, 0.0005, 10, 60, 25, 500
            ),
        }
    
    def update_orderbook(
        self,
        symbol: str,
        exchange: Exchange,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        latency_ms: Optional[float] = None
    ) -> None:
        """Update orderbook from exchange."""
        self.book_aggregator.update_book(symbol, exchange, bids, asks)
        
        if latency_ms is not None:
            self.latency_normalizer.update_latency(exchange, latency_ms)
    
    def detect_spatial_arbitrage(
        self,
        symbol: str
    ) -> List[ArbitrageOpportunity]:
        """
        Detect spatial arbitrage (same asset, different exchanges).
        """
        opportunities = []
        
        # Get best prices per exchange
        prices = self.book_aggregator.get_best_prices(symbol)
        
        if len(prices) < 2:
            return []
        
        exchanges = list(prices.keys())
        
        # Compare all pairs
        for i, buy_ex in enumerate(exchanges):
            for sell_ex in exchanges[i+1:]:
                buy_bid, buy_ask = prices[buy_ex]
                sell_bid, sell_ask = prices[sell_ex]
                
                # Check both directions
                
                # Direction 1: Buy at buy_ex, sell at sell_ex
                spread1 = sell_bid - buy_ask
                if spread1 > 0:
                    opp = self._evaluate_opportunity(
                        symbol, buy_ex, sell_ex, buy_ask, sell_bid
                    )
                    if opp and opp.is_profitable:
                        opportunities.append(opp)
                
                # Direction 2: Buy at sell_ex, sell at buy_ex
                spread2 = buy_bid - sell_ask
                if spread2 > 0:
                    opp = self._evaluate_opportunity(
                        symbol, sell_ex, buy_ex, sell_ask, buy_bid
                    )
                    if opp and opp.is_profitable:
                        opportunities.append(opp)
        
        return sorted(opportunities, key=lambda x: -x.net_profit)
    
    def _evaluate_opportunity(
        self,
        symbol: str,
        buy_exchange: Exchange,
        sell_exchange: Exchange,
        buy_price: float,
        sell_price: float
    ) -> Optional[ArbitrageOpportunity]:
        """Evaluate a potential arbitrage opportunity."""
        # Adjust for latency
        adj_buy_price = self.latency_normalizer.adjust_price_for_latency(
            symbol, buy_price, buy_exchange, is_bid=False
        )
        adj_sell_price = self.latency_normalizer.adjust_price_for_latency(
            symbol, sell_price, sell_exchange, is_bid=True
        )
        
        spread = adj_sell_price - adj_buy_price
        
        if spread <= 0:
            return None
        
        spread_bps = (spread / adj_buy_price) * 10000
        
        # Calculate fees
        buy_config = self.exchange_configs.get(buy_exchange)
        sell_config = self.exchange_configs.get(sell_exchange)
        
        if not buy_config or not sell_config:
            return None
        
        # Assume taker fees
        buy_fee_bps = buy_config.taker_fee_bps
        sell_fee_bps = sell_config.taker_fee_bps
        
        total_fees_bps = buy_fee_bps + sell_fee_bps + self.execution_buffer_bps
        
        net_spread_bps = spread_bps - total_fees_bps
        
        if net_spread_bps < self.min_profit_bps:
            return None
        
        # Calculate max quantity
        book = self.book_aggregator.get_aggregated_book(symbol)
        if not book:
            return None
        
        # Find available liquidity
        buy_liquidity = sum(
            level.quantity for level in book.asks
            if level.exchange == buy_exchange and level.price <= adj_buy_price * 1.001
        )
        sell_liquidity = sum(
            level.quantity for level in book.bids
            if level.exchange == sell_exchange and level.price >= adj_sell_price * 0.999
        )
        
        max_qty = min(buy_liquidity, sell_liquidity)
        max_qty_usd = min(max_qty * adj_buy_price, self.max_position_usd)
        max_qty = max_qty_usd / adj_buy_price
        
        # Expected profit
        fees_total = max_qty_usd * (total_fees_bps / 10000)
        expected_profit = max_qty * spread
        net_profit = expected_profit - fees_total
        
        # Confidence based on spread stability and liquidity
        confidence = min(
            spread_bps / (self.min_profit_bps * 2),  # Spread confidence
            min(buy_liquidity, sell_liquidity) / (max_qty * 2),  # Liquidity confidence
            1.0
        )
        
        return ArbitrageOpportunity(
            arb_type=ArbitrageType.SPATIAL,
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=adj_buy_price,
            sell_price=adj_sell_price,
            spread=spread,
            spread_bps=spread_bps,
            max_quantity=max_qty,
            expected_profit=expected_profit,
            fees_total=fees_total,
            net_profit=net_profit,
            confidence=confidence
        )
    
    def detect_triangular_arbitrage(
        self,
        base_asset: str,
        quote_assets: List[str]
    ) -> List[ArbitrageOpportunity]:
        """
        Detect triangular arbitrage opportunities.
        
        Example: BTC → ETH → USDT → BTC
        
        Args:
            base_asset: Starting asset (e.g., 'BTC')
            quote_assets: Intermediate assets (e.g., ['ETH', 'USDT'])
        
        Returns:
            List of triangular arbitrage opportunities
        """
        opportunities = []
        
        # Generate all possible triangular paths
        for q1 in quote_assets:
            for q2 in quote_assets:
                if q1 == q2:
                    continue
                
                # Path: base → q1 → q2 → base
                path = [
                    (f"{base_asset}{q1}", OrderSide.SELL),   # Sell base for q1
                    (f"{q1}{q2}", OrderSide.SELL),           # Sell q1 for q2
                    (f"{base_asset}{q2}", OrderSide.BUY),    # Buy base with q2
                ]
                
                profit = self._calculate_triangular_profit(path)
                
                if profit is not None and profit > self.min_profit_bps:
                    opportunities.append(
                        ArbitrageOpportunity(
                            arb_type=ArbitrageType.TRIANGULAR,
                            symbol=base_asset,
                            buy_exchange=Exchange.BINANCE,  # Simplified
                            sell_exchange=Exchange.BINANCE,
                            buy_price=0,
                            sell_price=0,
                            spread=profit / 10000,
                            spread_bps=profit,
                            max_quantity=0,
                            expected_profit=0,
                            fees_total=0,
                            net_profit=profit,
                            confidence=0.5,
                            path=[(Exchange.BINANCE, p[0], p[1]) for p in path]
                        )
                    )
        
        return opportunities
    
    def _calculate_triangular_profit(
        self,
        path: List[Tuple[str, OrderSide]]
    ) -> Optional[float]:
        """
        Calculate profit from triangular path.
        
        Returns profit in basis points or None if not feasible.
        """
        # Start with 1 unit
        amount = 1.0
        
        for symbol, side in path:
            book = self.book_aggregator.get_aggregated_book(symbol)
            if not book:
                return None
            
            if side == OrderSide.BUY:
                if not book.best_ask:
                    return None
                # Buying: amount / price
                amount = amount / book.best_ask.price
            else:
                if not book.best_bid:
                    return None
                # Selling: amount * price
                amount = amount * book.best_bid.price
        
        # Profit = ending amount - 1 (started with 1)
        profit_pct = (amount - 1) * 100
        profit_bps = profit_pct * 100
        
        # Subtract fees (3 trades)
        fees_bps = 30  # Rough estimate: 10bps per trade
        
        return profit_bps - fees_bps
    
    def check_execution_feasibility(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Tuple[bool, str]:
        """
        Check if opportunity can be executed.
        
        Returns:
            Tuple of (is_feasible, reason)
        """
        # Check minimum size
        min_usd = 10
        if opportunity.max_quantity * opportunity.buy_price < min_usd:
            return False, f"Size too small: ${opportunity.max_quantity * opportunity.buy_price:.2f}"
        
        # Check if exchanges are configured
        if opportunity.buy_exchange not in self.exchange_configs:
            return False, f"Buy exchange {opportunity.buy_exchange} not configured"
        if opportunity.sell_exchange not in self.exchange_configs:
            return False, f"Sell exchange {opportunity.sell_exchange} not configured"
        
        # Check combined latency
        buy_latency = self.latency_normalizer.get_expected_latency(opportunity.buy_exchange)
        sell_latency = self.latency_normalizer.get_expected_latency(opportunity.sell_exchange)
        total_latency = buy_latency + sell_latency
        
        if total_latency > 500:  # 500ms max
            return False, f"Total latency too high: {total_latency:.0f}ms"
        
        # Check profit confidence
        if opportunity.confidence < 0.5:
            return False, f"Low confidence: {opportunity.confidence:.1%}"
        
        return True, "Feasible"
    
    async def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity,
        quantity: Optional[float] = None
    ) -> ArbitrageExecution:
        """
        Execute arbitrage opportunity (simulation).
        
        In production, this would place actual orders.
        """
        start_time = datetime.now()
        
        quantity = quantity or opportunity.max_quantity
        quantity = min(quantity, opportunity.max_quantity)
        
        # Simulate execution
        # Add some slippage
        slippage_pct = np.random.uniform(0.0001, 0.001)
        
        buy_fill = opportunity.buy_price * (1 + slippage_pct)
        sell_fill = opportunity.sell_price * (1 - slippage_pct)
        
        actual_spread = sell_fill - buy_fill
        
        # Calculate fees
        buy_config = self.exchange_configs[opportunity.buy_exchange]
        sell_config = self.exchange_configs[opportunity.sell_exchange]
        
        notional = quantity * (buy_fill + sell_fill) / 2
        fees = notional * (buy_config.taker_fee_bps + sell_config.taker_fee_bps) / 10000
        
        # Calculate profit
        gross_profit = quantity * actual_spread
        realized_profit = gross_profit - fees
        
        # Execution time (simulated)
        exec_time = np.random.uniform(
            buy_config.latency_ms,
            buy_config.latency_ms + sell_config.latency_ms
        )
        
        # Slippage in bps
        expected_spread = opportunity.sell_price - opportunity.buy_price
        slippage_bps = ((expected_spread - actual_spread) / opportunity.buy_price) * 10000
        
        execution = ArbitrageExecution(
            opportunity=opportunity,
            executed_quantity=quantity,
            buy_fill_price=buy_fill,
            sell_fill_price=sell_fill,
            actual_spread=actual_spread,
            fees_paid=fees,
            realized_profit=realized_profit,
            execution_time_ms=exec_time,
            slippage_bps=slippage_bps,
            success=realized_profit > 0
        )
        
        self.executions.append(execution)
        
        logger.info(
            f"Executed arb: {opportunity.symbol} "
            f"buy@{buy_fill:.2f} sell@{sell_fill:.2f}, "
            f"profit=${realized_profit:.2f}, time={exec_time:.0f}ms"
        )
        
        return execution
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of arbitrage executions."""
        if not self.executions:
            return {'total_executions': 0}
        
        profits = [e.realized_profit for e in self.executions]
        slippages = [e.slippage_bps for e in self.executions]
        exec_times = [e.execution_time_ms for e in self.executions]
        
        return {
            'total_executions': len(self.executions),
            'successful': sum(1 for e in self.executions if e.success),
            'total_profit': sum(profits),
            'avg_profit': np.mean(profits),
            'win_rate': np.mean([p > 0 for p in profits]),
            'avg_slippage_bps': np.mean(slippages),
            'avg_exec_time_ms': np.mean(exec_times),
            'by_exchange_pair': self._aggregate_by_exchange()
        }
    
    def _aggregate_by_exchange(self) -> Dict:
        """Aggregate stats by exchange pair."""
        stats = {}
        
        for e in self.executions:
            key = f"{e.opportunity.buy_exchange.value}->{e.opportunity.sell_exchange.value}"
            if key not in stats:
                stats[key] = {'count': 0, 'profit': 0}
            stats[key]['count'] += 1
            stats[key]['profit'] += e.realized_profit
        
        return stats


class TransferTimeModel:
    """
    Model transfer times between exchanges for capital rebalancing.
    """
    
    def __init__(self):
        # Estimated transfer times in minutes
        self.transfer_times = {
            ('binance', 'okx'): 10,
            ('binance', 'bybit'): 10,
            ('okx', 'bybit'): 10,
            ('binance', 'deribit'): 20,
            ('deribit', 'okx'): 20,
        }
        
        # Blockchain confirmation times (minutes)
        self.blockchain_times = {
            'BTC': 30,   # ~3 confirmations
            'ETH': 5,    # ~12 confirmations
            'USDT': 5,   # ERC-20
            'USDC': 5,   # ERC-20
        }
    
    def estimate_transfer_time(
        self,
        from_exchange: Exchange,
        to_exchange: Exchange,
        asset: str
    ) -> float:
        """Estimate transfer time in minutes."""
        key = (from_exchange.value, to_exchange.value)
        reverse_key = (to_exchange.value, from_exchange.value)
        
        base_time = self.transfer_times.get(
            key,
            self.transfer_times.get(reverse_key, 30)
        )
        
        # Add blockchain time
        chain_time = self.blockchain_times.get(asset, 10)
        
        return base_time + chain_time


class FeesReconciler:
    """
    Track and reconcile fees across exchanges.
    """
    
    def __init__(self):
        self.fee_history: List[Dict] = []
    
    def record_fee(
        self,
        exchange: Exchange,
        fee_type: str,  # 'trading', 'withdrawal', 'funding'
        amount: float,
        asset: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a fee payment."""
        self.fee_history.append({
            'exchange': exchange.value,
            'fee_type': fee_type,
            'amount': amount,
            'asset': asset,
            'timestamp': timestamp or datetime.now()
        })
    
    def get_total_fees(
        self,
        exchange: Optional[Exchange] = None,
        fee_type: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> float:
        """Get total fees with optional filters."""
        fees = self.fee_history
        
        if exchange:
            fees = [f for f in fees if f['exchange'] == exchange.value]
        if fee_type:
            fees = [f for f in fees if f['fee_type'] == fee_type]
        if start_date:
            fees = [f for f in fees if f['timestamp'] >= start_date]
        
        return sum(f['amount'] for f in fees)
    
    def get_fee_summary(self) -> Dict:
        """Get fee summary by exchange and type."""
        summary = {}
        
        for fee in self.fee_history:
            ex = fee['exchange']
            ft = fee['fee_type']
            
            if ex not in summary:
                summary[ex] = {}
            if ft not in summary[ex]:
                summary[ex][ft] = 0
            
            summary[ex][ft] += fee['amount']
        
        return summary


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Cross-Exchange Arbitrage Engine")
    parser.add_argument("--action", type=str, required=True,
                       choices=['detect', 'simulate'],
                       help="Action to perform")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    
    args = parser.parse_args()
    
    engine = CrossExchangeArbitrageEngine()
    
    # Simulate orderbook data
    base_price = 100000
    
    # Add orderbooks with slight price differences
    engine.update_orderbook(
        args.symbol,
        Exchange.BINANCE,
        [(base_price - 10, 1.0), (base_price - 20, 2.0)],  # bids
        [(base_price + 10, 1.0), (base_price + 20, 2.0)],  # asks
        latency_ms=50
    )
    
    engine.update_orderbook(
        args.symbol,
        Exchange.OKX,
        [(base_price - 5, 1.0), (base_price - 15, 2.0)],   # Higher bids
        [(base_price + 15, 1.0), (base_price + 25, 2.0)],  # Higher asks
        latency_ms=70
    )
    
    engine.update_orderbook(
        args.symbol,
        Exchange.BYBIT,
        [(base_price - 15, 1.0), (base_price - 25, 2.0)],  # Lower bids
        [(base_price + 5, 1.0), (base_price + 15, 2.0)],   # Lower asks
        latency_ms=60
    )
    
    if args.action == 'detect':
        opportunities = engine.detect_spatial_arbitrage(args.symbol)
        
        print(f"\n{'='*60}")
        print(f"ARBITRAGE OPPORTUNITIES: {args.symbol}")
        print(f"{'='*60}")
        
        if not opportunities:
            print("No profitable opportunities found")
        else:
            for opp in opportunities[:5]:
                feasible, reason = engine.check_execution_feasibility(opp)
                
                print(f"\n{opp.buy_exchange.value} → {opp.sell_exchange.value}")
                print(f"  Buy:      ${opp.buy_price:.2f}")
                print(f"  Sell:     ${opp.sell_price:.2f}")
                print(f"  Spread:   {opp.spread_bps:.1f} bps")
                print(f"  Net Profit: ${opp.net_profit:.2f}")
                print(f"  Confidence: {opp.confidence:.1%}")
                print(f"  Feasible: {'✅' if feasible else '❌'} {reason}")
    
    elif args.action == 'simulate':
        opportunities = engine.detect_spatial_arbitrage(args.symbol)
        
        if opportunities:
            opp = opportunities[0]
            
            # Run async execution
            import asyncio
            execution = asyncio.run(engine.execute_arbitrage(opp))
            
            print(f"\n{'='*60}")
            print("EXECUTION RESULT")
            print(f"{'='*60}")
            print(f"Quantity:       {execution.executed_quantity:.4f}")
            print(f"Buy Fill:       ${execution.buy_fill_price:.2f}")
            print(f"Sell Fill:      ${execution.sell_fill_price:.2f}")
            print(f"Actual Spread:  ${execution.actual_spread:.2f}")
            print(f"Fees:           ${execution.fees_paid:.2f}")
            print(f"Realized P&L:   ${execution.realized_profit:.2f}")
            print(f"Execution Time: {execution.execution_time_ms:.0f}ms")
            print(f"Slippage:       {execution.slippage_bps:.1f} bps")
            print(f"Success:        {'✅' if execution.success else '❌'}")
