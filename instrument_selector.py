"""
Instrument Selector for Delta Exchange Algo Trading Bot
Decides between futures and options based on market conditions and confidence
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from config import get_config
from logger import get_logger
from product_discovery import (
    get_product_discovery,
    Instrument,
    InstrumentType
)
from options import get_options_scanner, OptionTrade
from risk_manager import get_risk_manager


class InstrumentChoice(Enum):
    """Types of instrument choices"""
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    CALL_OPTION = "call_option"
    PUT_OPTION = "put_option"
    NO_TRADE = "no_trade"


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class MarketConditions:
    """Current market conditions for an underlying"""
    underlying: str
    underlying_price: float
    
    # Momentum indicators
    momentum_score: float = 0.0  # -1 to 1
    trend_strength: float = 0.0  # 0 to 1
    
    # Volatility
    realized_volatility: float = 0.0
    implied_volatility: float = 0.0
    iv_rank: float = 0.0  # 0-100 percentile
    
    # Sentiment
    sentiment_score: float = 0.0  # -1 to 1
    news_impact: float = 0.0  # -1 to 1
    
    # Liquidity
    futures_liquidity: float = 0.0
    options_liquidity: float = 0.0
    
    # Put-call metrics
    put_call_ratio: float = 0.0
    
    # Regime
    regime: MarketRegime = MarketRegime.RANGING
    
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeRecommendation:
    """Trade recommendation from the selector"""
    instrument_choice: InstrumentChoice
    underlying: str
    direction: str  # 'long' or 'short'
    confidence: float  # 0-1
    
    # Instrument details
    instrument: Optional[Instrument] = None
    option_trade: Optional[OptionTrade] = None
    
    # Sizing
    size: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_profit: float = 0.0
    
    # Risk metrics
    max_loss: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Decision reasoning
    reasons: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return (
            self.instrument_choice != InstrumentChoice.NO_TRADE
            and self.confidence >= 0.65
            and (self.instrument is not None or self.option_trade is not None)
        )


class InstrumentSelector:
    """
    Selects optimal instrument (futures vs options) based on market conditions
    
    Decision Logic:
    - Strong momentum → Futures/Perpetual
    - High IV + directional expectation → Options
    - Ranging market → Options for premium collection (advanced)
    - Low confidence → No trade
    
    Confidence threshold: >= 0.65
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.discovery = get_product_discovery()
        self.options_scanner = get_options_scanner()
        self.risk_mgr = get_risk_manager()
        
        # Thresholds
        self.confidence_threshold = getattr(self.config.risk, 'confidence_threshold', 0.65)
        self.min_liquidity_usd = getattr(self.config.risk, 'min_liquidity_usd', 1000)
        
        # Momentum thresholds
        self.strong_momentum_threshold = 0.6
        self.weak_momentum_threshold = 0.3
        
        # IV thresholds
        self.high_iv_threshold = 0.60  # 60% IV
        self.low_iv_threshold = 0.30   # 30% IV
        
        # Signal weights for composite confidence
        self.weights = {
            "momentum": 0.40,
            "orderbook_imbalance": 0.20,
            "sentiment": 0.25,
            "news_event": 0.15
        }
        
        self.logger.info(
            "Instrument selector initialized",
            confidence_threshold=self.confidence_threshold,
            min_liquidity=self.min_liquidity_usd
        )
    
    def compute_composite_confidence(
        self,
        signals: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Compute composite confidence score from multiple signals
        
        Args:
            signals: Dict of signal_name -> signal_value (-1 to 1)
        
        Returns:
            (confidence_score, direction)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for signal_name, weight in self.weights.items():
            if signal_name in signals:
                value = signals[signal_name]
                weighted_sum += value * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0, "neutral"
        
        # Normalize to -1 to 1
        normalized = weighted_sum / total_weight
        
        # Convert to confidence (0-1) and direction
        confidence = abs(normalized)
        direction = "long" if normalized > 0 else "short" if normalized < 0 else "neutral"
        
        return confidence, direction
    
    def classify_market_regime(
        self,
        momentum: float,
        volatility: float,
        trend_strength: float
    ) -> MarketRegime:
        """Classify the current market regime"""
        # High volatility overrides other conditions
        if volatility > self.high_iv_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.low_iv_threshold:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend classification
        if trend_strength > 0.6:
            if momentum > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        return MarketRegime.RANGING
    
    def select_instrument(
        self,
        underlying: str,
        signals: Dict[str, float],
        market_data: Optional[Dict[str, Any]] = None
    ) -> TradeRecommendation:
        """
        Select optimal instrument based on market conditions
        
        Args:
            underlying: Underlying asset (e.g., 'BTC', 'ETH')
            signals: Dict of signal values (momentum, sentiment, etc.)
            market_data: Additional market data (price, volatility, etc.)
        
        Returns:
            TradeRecommendation with instrument choice and sizing
        """
        reasons = []
        
        # Compute composite confidence
        confidence, direction = self.compute_composite_confidence(signals)
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return TradeRecommendation(
                instrument_choice=InstrumentChoice.NO_TRADE,
                underlying=underlying,
                direction=direction,
                confidence=confidence,
                reasons=[f"Confidence {confidence:.2%} below threshold {self.confidence_threshold:.0%}"]
            )
        
        reasons.append(f"Composite confidence: {confidence:.2%}")
        reasons.append(f"Direction: {direction}")
        
        # Get market data
        perpetual = self.discovery.get_perpetual(underlying)
        if not perpetual:
            return TradeRecommendation(
                instrument_choice=InstrumentChoice.NO_TRADE,
                underlying=underlying,
                direction=direction,
                confidence=confidence,
                reasons=["No perpetual found for underlying"]
            )
        
        # Update market data
        self.discovery.update_market_data([perpetual.product_id])
        underlying_price = perpetual.mark_price or perpetual.last_price
        
        if not underlying_price:
            return TradeRecommendation(
                instrument_choice=InstrumentChoice.NO_TRADE,
                underlying=underlying,
                direction=direction,
                confidence=confidence,
                reasons=["Could not get underlying price"]
            )
        
        # Get additional metrics
        momentum = signals.get("momentum", 0)
        iv = market_data.get("implied_volatility", 0.4) if market_data else 0.4
        trend_strength = market_data.get("trend_strength", abs(momentum)) if market_data else abs(momentum)
        
        # Classify market regime
        regime = self.classify_market_regime(momentum, iv, trend_strength)
        reasons.append(f"Market regime: {regime.value}")
        
        # Decision logic
        instrument_choice, instrument_reasons = self._decide_instrument(
            underlying=underlying,
            direction=direction,
            confidence=confidence,
            momentum=momentum,
            iv=iv,
            regime=regime,
            perpetual=perpetual,
            underlying_price=underlying_price
        )
        
        reasons.extend(instrument_reasons)
        
        # Build recommendation based on choice
        if instrument_choice == InstrumentChoice.NO_TRADE:
            return TradeRecommendation(
                instrument_choice=instrument_choice,
                underlying=underlying,
                direction=direction,
                confidence=confidence,
                reasons=reasons
            )
        
        elif instrument_choice in (InstrumentChoice.FUTURES, InstrumentChoice.PERPETUAL):
            return self._build_futures_recommendation(
                underlying=underlying,
                direction=direction,
                confidence=confidence,
                perpetual=perpetual,
                underlying_price=underlying_price,
                momentum=momentum,
                reasons=reasons
            )
        
        else:  # Options
            option_direction = "call" if direction == "long" else "put"
            return self._build_options_recommendation(
                underlying=underlying,
                direction=direction,
                option_direction=option_direction,
                confidence=confidence,
                underlying_price=underlying_price,
                reasons=reasons
            )
    
    def _decide_instrument(
        self,
        underlying: str,
        direction: str,
        confidence: float,
        momentum: float,
        iv: float,
        regime: MarketRegime,
        perpetual: Instrument,
        underlying_price: float
    ) -> Tuple[InstrumentChoice, List[str]]:
        """
        Core decision logic for instrument selection
        
        Returns:
            (InstrumentChoice, list of reasons)
        """
        reasons = []
        abs_momentum = abs(momentum)
        
        # Check futures liquidity
        futures_liquid = perpetual.meets_liquidity_threshold(self.min_liquidity_usd)
        
        # Check if options available and liquid
        options_analysis = self.options_scanner.analyze_option_chain(underlying)
        options_available = "error" not in options_analysis
        options_liquid = False
        
        if options_available:
            options_volume = options_analysis.get("call_volume", 0) + options_analysis.get("put_volume", 0)
            options_liquid = options_volume >= self.min_liquidity_usd
            options_iv = (options_analysis.get("avg_call_iv", 0) + options_analysis.get("avg_put_iv", 0)) / 2
            iv = options_iv if options_iv > 0 else iv
        
        # Decision tree
        
        # Case 1: Strong momentum -> Futures
        if abs_momentum >= self.strong_momentum_threshold:
            reasons.append(f"Strong momentum ({momentum:.2f}) favors futures")
            
            if futures_liquid:
                reasons.append("Futures liquidity adequate")
                return InstrumentChoice.PERPETUAL, reasons
            elif options_available and options_liquid:
                reasons.append("Futures illiquid, using options instead")
                return (InstrumentChoice.CALL_OPTION if direction == "long" 
                       else InstrumentChoice.PUT_OPTION), reasons
            else:
                reasons.append("Insufficient liquidity in both instruments")
                return InstrumentChoice.NO_TRADE, reasons
        
        # Case 2: High IV + directional -> Options
        if iv >= self.high_iv_threshold and confidence >= 0.70:
            reasons.append(f"High IV ({iv:.1%}) with strong confidence")
            
            if options_available and options_liquid:
                reasons.append("Using options for defined risk")
                return (InstrumentChoice.CALL_OPTION if direction == "long" 
                       else InstrumentChoice.PUT_OPTION), reasons
            elif futures_liquid:
                reasons.append("Options unavailable, using futures with tight stops")
                return InstrumentChoice.PERPETUAL, reasons
            else:
                reasons.append("No liquid instruments available")
                return InstrumentChoice.NO_TRADE, reasons
        
        # Case 3: Moderate momentum -> Prefer futures if liquid
        if abs_momentum >= self.weak_momentum_threshold:
            reasons.append(f"Moderate momentum ({momentum:.2f})")
            
            if futures_liquid:
                reasons.append("Using perpetual for leverage efficiency")
                return InstrumentChoice.PERPETUAL, reasons
            elif options_available and options_liquid:
                reasons.append("Using options as futures backup")
                return (InstrumentChoice.CALL_OPTION if direction == "long" 
                       else InstrumentChoice.PUT_OPTION), reasons
        
        # Case 4: Low volatility -> Options can be cheap
        if regime == MarketRegime.LOW_VOLATILITY and options_available and options_liquid:
            reasons.append(f"Low IV ({iv:.1%}) - options relatively cheap")
            return (InstrumentChoice.CALL_OPTION if direction == "long" 
                   else InstrumentChoice.PUT_OPTION), reasons
        
        # Case 5: Ranging market with weak signals
        if regime == MarketRegime.RANGING or abs_momentum < self.weak_momentum_threshold:
            reasons.append("Ranging market with weak signals")
            return InstrumentChoice.NO_TRADE, reasons
        
        # Default: Use perpetual if liquid
        if futures_liquid:
            reasons.append("Default selection: perpetual")
            return InstrumentChoice.PERPETUAL, reasons
        
        reasons.append("No suitable instrument found")
        return InstrumentChoice.NO_TRADE, reasons
    
    def _build_futures_recommendation(
        self,
        underlying: str,
        direction: str,
        confidence: float,
        perpetual: Instrument,
        underlying_price: float,
        momentum: float,
        reasons: List[str]
    ) -> TradeRecommendation:
        """Build recommendation for futures/perpetual trade"""
        
        # Calculate entry and stops
        entry_price = underlying_price
        
        # ATR-based stop or percentage
        atr_pct = 0.02  # 2% default
        stop_distance = underlying_price * atr_pct
        
        if direction == "long":
            stop_price = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 2)  # 2:1 R/R
        else:
            stop_price = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 2)
        
        # Calculate size
        size, sizing_details = self.risk_mgr.compute_futures_size(
            entry_price=entry_price,
            stop_price=stop_price
        )
        
        if size <= 0:
            reasons.append("Could not compute valid position size")
            return TradeRecommendation(
                instrument_choice=InstrumentChoice.NO_TRADE,
                underlying=underlying,
                direction=direction,
                confidence=confidence,
                reasons=reasons
            )
        
        # Risk metrics
        risk_per_unit = abs(entry_price - stop_price)
        max_loss = size * risk_per_unit
        reward = size * abs(take_profit - entry_price)
        risk_reward_ratio = reward / max_loss if max_loss > 0 else 0
        
        reasons.append(f"Size: {size} ({sizing_details.get('limiting_factor', 'N/A')} limited)")
        reasons.append(f"Stop: {stop_price:.2f}, TP: {take_profit:.2f}")
        
        return TradeRecommendation(
            instrument_choice=InstrumentChoice.PERPETUAL,
            underlying=underlying,
            direction=direction,
            confidence=confidence,
            instrument=perpetual,
            size=size,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit=take_profit,
            max_loss=max_loss,
            risk_reward_ratio=risk_reward_ratio,
            reasons=reasons
        )
    
    def _build_options_recommendation(
        self,
        underlying: str,
        direction: str,
        option_direction: str,
        confidence: float,
        underlying_price: float,
        reasons: List[str]
    ) -> TradeRecommendation:
        """Build recommendation for options trade"""
        
        # Select option using scanner
        option_trade = self.options_scanner.select_directional_option(
            underlying=underlying,
            direction=option_direction
        )
        
        if not option_trade:
            reasons.append("No suitable option found")
            return TradeRecommendation(
                instrument_choice=InstrumentChoice.NO_TRADE,
                underlying=underlying,
                direction=direction,
                confidence=confidence,
                reasons=reasons
            )
        
        # Risk metrics for options
        max_loss = option_trade.max_loss
        
        # Estimated reward (conservative - assume 50% of premium * delta move)
        estimated_reward = option_trade.contracts * option_trade.premium * 2
        risk_reward_ratio = estimated_reward / max_loss if max_loss > 0 else 0
        
        reasons.append(f"Selected {option_trade.instrument.symbol}")
        reasons.append(f"Strike: {option_trade.strike}, Delta: {option_trade.delta:.2f}")
        reasons.append(f"Contracts: {option_trade.contracts}, Premium: {option_trade.premium:.4f}")
        
        instrument_choice = (InstrumentChoice.CALL_OPTION if option_direction == "call" 
                           else InstrumentChoice.PUT_OPTION)
        
        return TradeRecommendation(
            instrument_choice=instrument_choice,
            underlying=underlying,
            direction=direction,
            confidence=confidence,
            instrument=option_trade.instrument,
            option_trade=option_trade,
            size=float(option_trade.contracts),
            entry_price=option_trade.premium,
            stop_price=0,  # Max loss is premium for bought options
            take_profit=option_trade.premium * 2,  # Target 100% gain
            max_loss=max_loss,
            risk_reward_ratio=risk_reward_ratio,
            reasons=reasons
        )
    
    def get_tradable_underlyings(self) -> List[str]:
        """Get list of underlyings that have tradable instruments"""
        underlyings = self.discovery.get_all_underlyings()
        
        tradable = []
        for underlying in underlyings:
            perpetual = self.discovery.get_perpetual(underlying)
            if perpetual and perpetual.is_active:
                tradable.append(underlying)
        
        return tradable
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of selector state and parameters"""
        return {
            "confidence_threshold": self.confidence_threshold,
            "min_liquidity_usd": self.min_liquidity_usd,
            "strong_momentum_threshold": self.strong_momentum_threshold,
            "high_iv_threshold": self.high_iv_threshold,
            "signal_weights": self.weights,
            "tradable_underlyings": self.get_tradable_underlyings()
        }


# Singleton instance
_instrument_selector: Optional[InstrumentSelector] = None


def get_instrument_selector() -> InstrumentSelector:
    """Get or create the global instrument selector"""
    global _instrument_selector
    if _instrument_selector is None:
        _instrument_selector = InstrumentSelector()
    return _instrument_selector


if __name__ == "__main__":
    # Test instrument selector
    selector = get_instrument_selector()
    
    print("Testing Instrument Selector")
    print("=" * 50)
    
    # Get summary
    summary = selector.get_selection_summary()
    print("\nSelector Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test with bullish signals
    bullish_signals = {
        "momentum": 0.7,
        "orderbook_imbalance": 0.3,
        "sentiment": 0.5,
        "news_event": 0.2
    }
    
    print("\n\nTesting with bullish signals:")
    print(f"Signals: {bullish_signals}")
    
    recommendation = selector.select_instrument("BTC", bullish_signals)
    print(f"\nRecommendation:")
    print(f"  Choice: {recommendation.instrument_choice.value}")
    print(f"  Direction: {recommendation.direction}")
    print(f"  Confidence: {recommendation.confidence:.2%}")
    print(f"  Valid: {recommendation.is_valid}")
    print(f"  Reasons:")
    for reason in recommendation.reasons:
        print(f"    - {reason}")
    
    if recommendation.is_valid:
        print(f"\n  Size: {recommendation.size}")
        print(f"  Entry: {recommendation.entry_price}")
        print(f"  Stop: {recommendation.stop_price}")
        print(f"  Max Loss: {recommendation.max_loss}")
    
    # Test with weak signals
    weak_signals = {
        "momentum": 0.2,
        "orderbook_imbalance": 0.1,
        "sentiment": 0.15,
        "news_event": 0.0
    }
    
    print("\n\nTesting with weak signals:")
    print(f"Signals: {weak_signals}")
    
    recommendation = selector.select_instrument("BTC", weak_signals)
    print(f"\nRecommendation:")
    print(f"  Choice: {recommendation.instrument_choice.value}")
    print(f"  Confidence: {recommendation.confidence:.2%}")
    print(f"  Reasons:")
    for reason in recommendation.reasons:
        print(f"    - {reason}")
