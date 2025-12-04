"""
Spot-Perpetual Funding Arbitrage Module
=========================================
High-yield funding rate arbitrage between spot and perpetual futures.
Most profitable arbitrage strategy in crypto markets.

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

logger = logging.getLogger(__name__)


class FundingDirection(Enum):
    POSITIVE = "positive"  # Longs pay shorts
    NEGATIVE = "negative"  # Shorts pay longs
    NEUTRAL = "neutral"


class PositionSide(Enum):
    CASH_AND_CARRY = "cash_and_carry"      # Long spot, short perp (collect positive funding)
    REVERSE_CARRY = "reverse_carry"         # Short spot, long perp (collect negative funding)


@dataclass
class FundingRate:
    """Funding rate snapshot."""
    symbol: str
    rate: float                    # Per funding period (typically 8h)
    annualized: float              # Annualized rate
    next_funding_time: datetime
    predicted_rate: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def direction(self) -> FundingDirection:
        if self.rate > 0.0001:
            return FundingDirection.POSITIVE
        elif self.rate < -0.0001:
            return FundingDirection.NEGATIVE
        return FundingDirection.NEUTRAL
    
    @property
    def hours_to_funding(self) -> float:
        delta = self.next_funding_time - datetime.now()
        return max(0, delta.total_seconds() / 3600)


@dataclass
class BasisSpread:
    """Spot-Perp basis spread."""
    symbol: str
    spot_price: float
    perp_price: float
    basis: float                   # Absolute basis (perp - spot)
    basis_pct: float               # Basis as percentage of spot
    annualized_basis: float        # Annualized basis yield
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_contango(self) -> bool:
        return self.basis > 0
    
    @property
    def is_backwardation(self) -> bool:
        return self.basis < 0


@dataclass
class ArbitragePosition:
    """Active arbitrage position."""
    symbol: str
    side: PositionSide
    spot_quantity: float
    perp_quantity: float
    spot_entry_price: float
    perp_entry_price: float
    entry_basis: float
    entry_time: datetime
    
    # P&L tracking
    funding_collected: float = 0.0
    basis_pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Risk metrics
    margin_used: float = 0.0
    liquidation_price: Optional[float] = None
    
    @property
    def net_pnl(self) -> float:
        return self.funding_collected + self.basis_pnl - self.fees_paid
    
    @property
    def notional(self) -> float:
        return abs(self.spot_quantity * self.spot_entry_price)
    
    @property
    def yield_annualized(self) -> float:
        if self.notional == 0:
            return 0
        days_held = max(1, (datetime.now() - self.entry_time).days)
        return (self.net_pnl / self.notional) * (365 / days_held)


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    symbol: str
    side: PositionSide
    funding_rate: FundingRate
    basis: BasisSpread
    expected_yield_8h: float       # Expected yield per funding period
    expected_yield_annual: float   # Annualized expected yield
    confidence: float              # 0-1 confidence score
    max_position_size: float       # Max size based on liquidity
    risk_score: float              # 0-1 risk score
    
    @property
    def is_attractive(self) -> bool:
        return (
            self.expected_yield_annual > 0.10 and  # >10% APY
            self.confidence > 0.7 and
            self.risk_score < 0.5
        )


class FundingRateEngine:
    """
    Engine for tracking and analyzing funding rates.
    """
    
    FUNDING_PERIODS_PER_DAY = 3  # Most exchanges: 8-hour funding
    
    def __init__(
        self,
        history_length: int = 1000,
        min_attractive_rate: float = 0.0001  # 0.01% per 8h = ~11% APY
    ):
        self.history_length = history_length
        self.min_attractive_rate = min_attractive_rate
        
        self.funding_history: Dict[str, deque] = {}
        self.current_rates: Dict[str, FundingRate] = {}
        
        logger.info("FundingRateEngine initialized")
    
    def update_funding_rate(
        self,
        symbol: str,
        rate: float,
        next_funding_time: datetime,
        predicted_rate: Optional[float] = None
    ) -> FundingRate:
        """Update funding rate for a symbol."""
        annualized = rate * self.FUNDING_PERIODS_PER_DAY * 365
        
        funding = FundingRate(
            symbol=symbol,
            rate=rate,
            annualized=annualized,
            next_funding_time=next_funding_time,
            predicted_rate=predicted_rate
        )
        
        self.current_rates[symbol] = funding
        
        if symbol not in self.funding_history:
            self.funding_history[symbol] = deque(maxlen=self.history_length)
        self.funding_history[symbol].append(funding)
        
        return funding
    
    def get_funding_metrics(self, symbol: str) -> Dict:
        """Get funding rate statistics."""
        if symbol not in self.funding_history:
            return {}
        
        history = list(self.funding_history[symbol])
        if len(history) < 2:
            return {}
        
        rates = [f.rate for f in history]
        
        return {
            'current_rate': rates[-1],
            'mean_rate': np.mean(rates),
            'std_rate': np.std(rates),
            'median_rate': np.median(rates),
            'max_rate': np.max(rates),
            'min_rate': np.min(rates),
            'positive_pct': np.mean([r > 0 for r in rates]),
            'mean_annual_yield': np.mean(rates) * self.FUNDING_PERIODS_PER_DAY * 365,
            'observations': len(rates)
        }
    
    def predict_next_funding(self, symbol: str) -> Optional[float]:
        """
        Predict next funding rate using EMA.
        """
        if symbol not in self.funding_history:
            return None
        
        history = list(self.funding_history[symbol])
        if len(history) < 10:
            return history[-1].rate if history else None
        
        rates = [f.rate for f in history[-20:]]  # Last 20 observations
        
        # EMA with decay
        alpha = 0.3
        ema = rates[0]
        for r in rates[1:]:
            ema = alpha * r + (1 - alpha) * ema
        
        return ema
    
    def find_opportunities(
        self,
        min_yield: float = 0.10  # 10% APY minimum
    ) -> List[Tuple[str, float, str]]:
        """
        Find symbols with attractive funding rates.
        
        Returns list of (symbol, annualized_yield, direction).
        """
        opportunities = []
        
        for symbol, funding in self.current_rates.items():
            yield_annual = abs(funding.annualized)
            
            if yield_annual >= min_yield:
                direction = "CASH_CARRY" if funding.rate > 0 else "REVERSE_CARRY"
                opportunities.append((symbol, yield_annual, direction))
        
        return sorted(opportunities, key=lambda x: x[1], reverse=True)


class BasisCalculator:
    """
    Calculate and track spot-perp basis.
    """
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.basis_history: Dict[str, deque] = {}
        
        logger.info("BasisCalculator initialized")
    
    def calculate_basis(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        days_to_funding: float = 1/3  # Assume 8h to next funding
    ) -> BasisSpread:
        """Calculate current basis spread."""
        basis = perp_price - spot_price
        basis_pct = basis / spot_price
        
        # Annualize: (basis / days_to_funding) * 365
        annualized = (basis_pct / days_to_funding) * 365 if days_to_funding > 0 else 0
        
        spread = BasisSpread(
            symbol=symbol,
            spot_price=spot_price,
            perp_price=perp_price,
            basis=basis,
            basis_pct=basis_pct,
            annualized_basis=annualized
        )
        
        if symbol not in self.basis_history:
            self.basis_history[symbol] = deque(maxlen=self.history_length)
        self.basis_history[symbol].append(spread)
        
        return spread
    
    def get_basis_metrics(self, symbol: str) -> Dict:
        """Get basis statistics."""
        if symbol not in self.basis_history:
            return {}
        
        history = list(self.basis_history[symbol])
        if len(history) < 2:
            return {}
        
        basis_pcts = [b.basis_pct for b in history]
        
        return {
            'current_basis_pct': basis_pcts[-1],
            'mean_basis_pct': np.mean(basis_pcts),
            'std_basis_pct': np.std(basis_pcts),
            'z_score': (basis_pcts[-1] - np.mean(basis_pcts)) / np.std(basis_pcts) if np.std(basis_pcts) > 0 else 0,
            'contango_pct': np.mean([b > 0 for b in basis_pcts]),
            'observations': len(basis_pcts)
        }


class FundingArbitrageEngine:
    """
    Production-grade funding rate arbitrage engine.
    
    Features:
    - Automatic opportunity detection
    - Position sizing with risk controls
    - Liquidation monitoring
    - P&L tracking and yield calculation
    """
    
    def __init__(
        self,
        funding_engine: Optional[FundingRateEngine] = None,
        basis_calculator: Optional[BasisCalculator] = None,
        max_leverage: float = 3.0,
        min_yield_threshold: float = 0.10,  # 10% APY
        max_position_pct: float = 0.20,      # Max 20% of capital per position
        liquidation_buffer: float = 0.20     # 20% buffer from liquidation
    ):
        self.funding_engine = funding_engine or FundingRateEngine()
        self.basis_calculator = basis_calculator or BasisCalculator()
        
        self.max_leverage = max_leverage
        self.min_yield_threshold = min_yield_threshold
        self.max_position_pct = max_position_pct
        self.liquidation_buffer = liquidation_buffer
        
        self.positions: Dict[str, ArbitragePosition] = {}
        self.closed_positions: List[ArbitragePosition] = []
        self.total_capital: float = 0
        
        logger.info("FundingArbitrageEngine initialized")
    
    def set_capital(self, capital: float) -> None:
        """Set available capital for arbitrage."""
        self.total_capital = capital
        logger.info(f"Capital set to ${capital:,.2f}")
    
    def detect_opportunity(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        funding_rate: float,
        next_funding_time: datetime,
        spot_liquidity: float,
        perp_liquidity: float
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect and evaluate arbitrage opportunity.
        
        Args:
            symbol: Trading symbol
            spot_price: Current spot price
            perp_price: Current perp price
            funding_rate: Current/predicted funding rate
            next_funding_time: Next funding timestamp
            spot_liquidity: Available spot liquidity (USD)
            perp_liquidity: Available perp liquidity (USD)
        
        Returns:
            ArbitrageOpportunity if attractive, None otherwise
        """
        # Update engines
        funding = self.funding_engine.update_funding_rate(
            symbol, funding_rate, next_funding_time
        )
        basis = self.basis_calculator.calculate_basis(
            symbol, spot_price, perp_price
        )
        
        # Determine direction
        if funding.rate > self.funding_engine.min_attractive_rate:
            side = PositionSide.CASH_AND_CARRY
            expected_yield = funding.rate  # Collect from longs
        elif funding.rate < -self.funding_engine.min_attractive_rate:
            side = PositionSide.REVERSE_CARRY
            expected_yield = abs(funding.rate)  # Collect from shorts
        else:
            return None
        
        # Expected yields
        yield_8h = expected_yield
        yield_annual = expected_yield * 3 * 365
        
        # Check minimum threshold
        if yield_annual < self.min_yield_threshold:
            return None
        
        # Calculate confidence based on historical consistency
        funding_metrics = self.funding_engine.get_funding_metrics(symbol)
        if funding_metrics:
            # Higher confidence if funding has been consistently positive/negative
            if side == PositionSide.CASH_AND_CARRY:
                confidence = funding_metrics.get('positive_pct', 0.5)
            else:
                confidence = 1 - funding_metrics.get('positive_pct', 0.5)
            
            # Adjust by volatility
            std = funding_metrics.get('std_rate', 0)
            mean = abs(funding_metrics.get('mean_rate', funding.rate))
            if mean > 0:
                confidence *= (1 - min(std / mean, 0.5))
        else:
            confidence = 0.5
        
        # Calculate max position size
        liquidity_limit = min(spot_liquidity, perp_liquidity) * 0.1  # 10% of book
        capital_limit = self.total_capital * self.max_position_pct
        max_size = min(liquidity_limit, capital_limit)
        
        # Risk score
        risk_score = self._calculate_risk_score(symbol, basis, funding)
        
        return ArbitrageOpportunity(
            symbol=symbol,
            side=side,
            funding_rate=funding,
            basis=basis,
            expected_yield_8h=yield_8h,
            expected_yield_annual=yield_annual,
            confidence=confidence,
            max_position_size=max_size,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(
        self,
        symbol: str,
        basis: BasisSpread,
        funding: FundingRate
    ) -> float:
        """Calculate risk score (0=low risk, 1=high risk)."""
        risk = 0.0
        
        # Basis risk (large basis = higher risk of mean reversion)
        basis_metrics = self.basis_calculator.get_basis_metrics(symbol)
        if basis_metrics:
            z_score = abs(basis_metrics.get('z_score', 0))
            risk += min(z_score / 3, 0.3)  # Max 0.3 from basis
        
        # Funding volatility risk
        funding_metrics = self.funding_engine.get_funding_metrics(symbol)
        if funding_metrics:
            std = funding_metrics.get('std_rate', 0)
            mean = abs(funding_metrics.get('mean_rate', 0.0001))
            if mean > 0:
                risk += min(std / mean, 0.3)  # Max 0.3 from volatility
        
        # Time to funding risk (higher risk if close to funding)
        if funding.hours_to_funding < 1:
            risk += 0.1  # Rush risk
        
        # Leverage risk
        risk += (self.max_leverage - 1) / 10  # Higher leverage = more risk
        
        return min(risk, 1.0)
    
    def open_position(
        self,
        opportunity: ArbitrageOpportunity,
        size_usd: float,
        spot_fee_bps: float = 10,
        perp_fee_bps: float = 5
    ) -> Optional[ArbitragePosition]:
        """
        Open arbitrage position.
        
        Args:
            opportunity: Detected opportunity
            size_usd: Position size in USD
            spot_fee_bps: Spot trading fee in basis points
            perp_fee_bps: Perp trading fee in basis points
        
        Returns:
            ArbitragePosition if opened successfully
        """
        symbol = opportunity.symbol
        
        # Check if already have position
        if symbol in self.positions:
            logger.warning(f"Already have position in {symbol}")
            return None
        
        # Enforce size limits
        size_usd = min(size_usd, opportunity.max_position_size)
        
        # Calculate quantities
        spot_price = opportunity.basis.spot_price
        perp_price = opportunity.basis.perp_price
        
        if opportunity.side == PositionSide.CASH_AND_CARRY:
            # Long spot, short perp
            spot_qty = size_usd / spot_price
            perp_qty = -size_usd / perp_price
        else:
            # Short spot (margin), long perp
            spot_qty = -size_usd / spot_price
            perp_qty = size_usd / perp_price
        
        # Calculate fees
        total_fees = (
            abs(size_usd) * (spot_fee_bps + perp_fee_bps) / 10000
        )
        
        # Calculate margin and liquidation
        margin_used = size_usd / self.max_leverage
        
        position = ArbitragePosition(
            symbol=symbol,
            side=opportunity.side,
            spot_quantity=spot_qty,
            perp_quantity=perp_qty,
            spot_entry_price=spot_price,
            perp_entry_price=perp_price,
            entry_basis=opportunity.basis.basis,
            entry_time=datetime.now(),
            fees_paid=total_fees,
            margin_used=margin_used
        )
        
        # Calculate liquidation price
        position.liquidation_price = self._calculate_liquidation_price(position)
        
        self.positions[symbol] = position
        
        logger.info(
            f"Opened {opportunity.side.value} position: {symbol} "
            f"size=${size_usd:,.2f}, expected yield={opportunity.expected_yield_annual:.1%}"
        )
        
        return position
    
    def _calculate_liquidation_price(self, position: ArbitragePosition) -> float:
        """Calculate liquidation price for the position."""
        # Simplified: liquidation when margin is depleted
        # For cash-and-carry: long spot, short perp
        # Liquidation happens when perp goes against us significantly
        
        margin = position.margin_used
        notional = position.notional
        
        # Price move that would wipe out margin
        max_loss_pct = (margin / notional) * (1 - self.liquidation_buffer)
        
        if position.side == PositionSide.CASH_AND_CARRY:
            # Short perp: liquidation if perp price goes up
            return position.perp_entry_price * (1 + max_loss_pct)
        else:
            # Long perp: liquidation if perp price goes down
            return position.perp_entry_price * (1 - max_loss_pct)
    
    def process_funding_payment(
        self,
        symbol: str,
        funding_rate: float
    ) -> Optional[float]:
        """
        Process funding payment for a position.
        
        Args:
            symbol: Symbol
            funding_rate: Realized funding rate
        
        Returns:
            Funding payment amount (positive = received)
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate funding payment
        # For perp: notional * funding_rate
        # Positive funding: longs pay shorts
        perp_notional = abs(position.perp_quantity * position.perp_entry_price)
        
        if position.side == PositionSide.CASH_AND_CARRY:
            # Short perp: receive positive funding, pay negative
            payment = perp_notional * funding_rate
        else:
            # Long perp: pay positive funding, receive negative
            payment = -perp_notional * funding_rate
        
        position.funding_collected += payment
        
        logger.info(
            f"Funding payment {symbol}: {'+' if payment > 0 else ''}{payment:.4f} "
            f"(total: {position.funding_collected:.4f})"
        )
        
        return payment
    
    def update_basis_pnl(
        self,
        symbol: str,
        current_spot: float,
        current_perp: float
    ) -> Optional[float]:
        """
        Update basis P&L for a position.
        
        Args:
            symbol: Symbol
            current_spot: Current spot price
            current_perp: Current perp price
        
        Returns:
            Current basis P&L
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate unrealized P&L
        spot_pnl = position.spot_quantity * (current_spot - position.spot_entry_price)
        perp_pnl = position.perp_quantity * (current_perp - position.perp_entry_price)
        
        position.basis_pnl = spot_pnl + perp_pnl
        
        return position.basis_pnl
    
    def check_liquidation_risk(
        self,
        symbol: str,
        current_perp: float
    ) -> Tuple[bool, float]:
        """
        Check if position is at liquidation risk.
        
        Returns:
            Tuple of (is_at_risk, distance_to_liquidation_pct)
        """
        if symbol not in self.positions:
            return False, 1.0
        
        position = self.positions[symbol]
        
        if position.liquidation_price is None:
            return False, 1.0
        
        if position.side == PositionSide.CASH_AND_CARRY:
            # Short perp: liquidation above
            distance = (position.liquidation_price - current_perp) / current_perp
        else:
            # Long perp: liquidation below
            distance = (current_perp - position.liquidation_price) / current_perp
        
        is_at_risk = distance < self.liquidation_buffer
        
        return is_at_risk, distance
    
    def close_position(
        self,
        symbol: str,
        current_spot: float,
        current_perp: float,
        spot_fee_bps: float = 10,
        perp_fee_bps: float = 5
    ) -> Optional[ArbitragePosition]:
        """
        Close arbitrage position.
        
        Returns closed position with final P&L.
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Update final basis P&L
        self.update_basis_pnl(symbol, current_spot, current_perp)
        
        # Add closing fees
        notional = abs(position.spot_quantity * current_spot)
        closing_fees = notional * (spot_fee_bps + perp_fee_bps) / 10000
        position.fees_paid += closing_fees
        
        # Move to closed positions
        del self.positions[symbol]
        self.closed_positions.append(position)
        
        logger.info(
            f"Closed position {symbol}: funding=${position.funding_collected:.2f}, "
            f"basis=${position.basis_pnl:.2f}, fees=${position.fees_paid:.2f}, "
            f"net=${position.net_pnl:.2f}, yield={position.yield_annualized:.1%}"
        )
        
        return position
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary of all positions."""
        total_funding = sum(p.funding_collected for p in self.positions.values())
        total_basis_pnl = sum(p.basis_pnl for p in self.positions.values())
        total_fees = sum(p.fees_paid for p in self.positions.values())
        total_notional = sum(p.notional for p in self.positions.values())
        total_margin = sum(p.margin_used for p in self.positions.values())
        
        # Include closed positions
        total_funding += sum(p.funding_collected for p in self.closed_positions)
        total_basis_pnl += sum(p.basis_pnl for p in self.closed_positions)
        total_fees += sum(p.fees_paid for p in self.closed_positions)
        
        return {
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'total_notional': total_notional,
            'total_margin_used': total_margin,
            'margin_utilization': total_margin / self.total_capital if self.total_capital > 0 else 0,
            'total_funding_collected': total_funding,
            'total_basis_pnl': total_basis_pnl,
            'total_fees_paid': total_fees,
            'net_pnl': total_funding + total_basis_pnl - total_fees,
            'positions': {
                sym: {
                    'side': pos.side.value,
                    'notional': pos.notional,
                    'funding': pos.funding_collected,
                    'basis_pnl': pos.basis_pnl,
                    'net_pnl': pos.net_pnl,
                    'days_held': (datetime.now() - pos.entry_time).days
                }
                for sym, pos in self.positions.items()
            }
        }
    
    def get_yield_analytics(self) -> Dict:
        """Get yield analytics across all positions."""
        all_positions = list(self.positions.values()) + self.closed_positions
        
        if not all_positions:
            return {}
        
        yields = [p.yield_annualized for p in all_positions if p.notional > 0]
        funding_per_day = [
            p.funding_collected / max(1, (datetime.now() - p.entry_time).days)
            for p in all_positions
        ]
        
        return {
            'avg_yield_annualized': np.mean(yields) if yields else 0,
            'median_yield_annualized': np.median(yields) if yields else 0,
            'best_yield': max(yields) if yields else 0,
            'worst_yield': min(yields) if yields else 0,
            'avg_daily_funding': np.mean(funding_per_day) if funding_per_day else 0,
            'total_positions': len(all_positions)
        }


class CarryYieldTracker:
    """
    Track and forecast carry yields over time.
    """
    
    def __init__(self):
        self.daily_yields: Dict[str, List[Tuple[datetime, float]]] = {}
        
    def record_daily_yield(
        self,
        symbol: str,
        date: datetime,
        yield_value: float
    ) -> None:
        """Record daily yield."""
        if symbol not in self.daily_yields:
            self.daily_yields[symbol] = []
        self.daily_yields[symbol].append((date, yield_value))
    
    def get_rolling_yield(
        self,
        symbol: str,
        days: int = 30
    ) -> Optional[float]:
        """Get rolling average yield."""
        if symbol not in self.daily_yields:
            return None
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = [y for d, y in self.daily_yields[symbol] if d >= cutoff]
        
        return np.mean(recent) if recent else None
    
    def forecast_yield(
        self,
        symbol: str,
        horizon_days: int = 7
    ) -> Optional[float]:
        """Forecast yield for next N days using EMA."""
        if symbol not in self.daily_yields or len(self.daily_yields[symbol]) < 5:
            return None
        
        yields = [y for _, y in self.daily_yields[symbol][-30:]]
        
        # EMA forecast
        alpha = 0.2
        ema = yields[0]
        for y in yields[1:]:
            ema = alpha * y + (1 - alpha) * ema
        
        return ema * horizon_days


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Funding Arbitrage Engine")
    parser.add_argument("--action", type=str, required=True,
                       choices=['detect', 'simulate', 'status'],
                       help="Action to perform")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--spot-price", type=float, default=100000)
    parser.add_argument("--perp-price", type=float, default=100050)
    parser.add_argument("--funding-rate", type=float, default=0.0003)  # 0.03% = 33% APY
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--size", type=float, default=10000)
    
    args = parser.parse_args()
    
    engine = FundingArbitrageEngine()
    engine.set_capital(args.capital)
    
    if args.action == 'detect':
        next_funding = datetime.now() + timedelta(hours=4)
        
        opportunity = engine.detect_opportunity(
            symbol=args.symbol,
            spot_price=args.spot_price,
            perp_price=args.perp_price,
            funding_rate=args.funding_rate,
            next_funding_time=next_funding,
            spot_liquidity=1000000,
            perp_liquidity=5000000
        )
        
        if opportunity:
            print(f"\n{'='*60}")
            print(f"ARBITRAGE OPPORTUNITY DETECTED")
            print(f"{'='*60}")
            print(f"Symbol:           {opportunity.symbol}")
            print(f"Strategy:         {opportunity.side.value}")
            print(f"Funding Rate:     {opportunity.funding_rate.rate:.4%}")
            print(f"Expected Yield:   {opportunity.expected_yield_annual:.1%} APY")
            print(f"Confidence:       {opportunity.confidence:.1%}")
            print(f"Risk Score:       {opportunity.risk_score:.2f}")
            print(f"Max Position:     ${opportunity.max_position_size:,.2f}")
            print(f"Is Attractive:    {'✅ YES' if opportunity.is_attractive else '❌ NO'}")
        else:
            print("No attractive opportunity found")
    
    elif args.action == 'simulate':
        next_funding = datetime.now() + timedelta(hours=4)
        
        # Detect and open
        opportunity = engine.detect_opportunity(
            args.symbol, args.spot_price, args.perp_price,
            args.funding_rate, next_funding, 1000000, 5000000
        )
        
        if opportunity and opportunity.is_attractive:
            position = engine.open_position(opportunity, args.size)
            
            if position:
                # Simulate 3 funding payments
                for i in range(3):
                    payment = engine.process_funding_payment(args.symbol, args.funding_rate)
                
                # Update basis P&L (assume prices unchanged)
                engine.update_basis_pnl(args.symbol, args.spot_price, args.perp_price)
                
                # Close position
                closed = engine.close_position(
                    args.symbol, args.spot_price, args.perp_price
                )
                
                print(f"\n{'='*60}")
                print(f"SIMULATION RESULTS")
                print(f"{'='*60}")
                print(f"Funding Collected:  ${closed.funding_collected:.2f}")
                print(f"Basis P&L:          ${closed.basis_pnl:.2f}")
                print(f"Fees Paid:          ${closed.fees_paid:.2f}")
                print(f"Net P&L:            ${closed.net_pnl:.2f}")
                print(f"Annualized Yield:   {closed.yield_annualized:.1%}")
    
    elif args.action == 'status':
        summary = engine.get_portfolio_summary()
        print(json.dumps(summary, indent=2, default=str))
