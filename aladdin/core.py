"""
ALADDIN AI CORE - The Brain of the Trading System
==================================================
Coordinates all components for autonomous trading decisions.

Key Features:
- Multi-timeframe analysis
- News & sentiment integration
- Dynamic strategy selection
- Risk-adjusted position sizing
- Self-learning optimization
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AladdinAI")


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class SignalStrength(Enum):
    """Trading signal strength"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1


@dataclass
class MarketState:
    """Current market state"""
    regime: MarketRegime = MarketRegime.NEUTRAL
    trend_direction: float = 0.0  # -1 to 1
    volatility: float = 0.0
    momentum: float = 0.0
    volume_profile: str = "normal"
    fear_greed_index: float = 50.0
    news_sentiment: float = 0.0  # -1 to 1
    economic_outlook: str = "neutral"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    symbol: str
    direction: str  # "long" or "short"
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    strategy: str
    confidence: float  # 0 to 1
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_equity: float = 10000.0
    available_margin: float = 10000.0
    open_positions: Dict = field(default_factory=dict)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0


class AladdinAI:
    """
    Autonomous AI Trading System
    
    The brain that coordinates:
    1. Market Analysis
    2. Sentiment Analysis
    3. Strategy Selection
    4. Risk Management
    5. Trade Execution
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.market_state = MarketState()
        self.portfolio = PortfolioState()
        self.active_signals: List[TradingSignal] = []
        self.trade_history: List[Dict] = []
        
        # Components (initialized lazily)
        self._sentiment_analyzer = None
        self._market_analyzer = None
        self._risk_engine = None
        self._executor = None
        
        # State
        self.is_running = False
        self._thread = None
        self._last_analysis_time = None
        
        logger.info("Aladdin AI initialized")
    
    def _default_config(self) -> Dict:
        return {
            "trading": {
                "symbols": ["BTCUSD", "ETHUSD"],
                "max_positions": 3,
                "max_position_size_pct": 10,  # % of portfolio
                "min_signal_strength": SignalStrength.MODERATE.value,
                "min_confidence": 0.6,
            },
            "risk": {
                "max_daily_loss_pct": 5,
                "max_drawdown_pct": 15,
                "risk_per_trade_pct": 2,
                "max_leverage": 10,
                "max_positions": 5,
                "min_risk_reward": 1.5,
                "max_correlation_exposure": 0.7,
            },
            "analysis": {
                "update_interval_seconds": 60,
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "sentiment_weight": 0.3,
                "technical_weight": 0.5,
                "regime_weight": 0.2,
            },
            "strategies": {
                "enabled": [
                    "trend_following",
                    "mean_reversion",
                    "momentum",
                    "breakout",
                    "sentiment_based",
                ],
                "auto_select": True,
            }
        }
    
    # ==================== Component Accessors ====================
    
    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            from aladdin.sentiment import SentimentAnalyzer
            self._sentiment_analyzer = SentimentAnalyzer()
        return self._sentiment_analyzer
    
    @property
    def market_analyzer(self):
        if self._market_analyzer is None:
            from aladdin.market_analyzer import MarketAnalyzer
            self._market_analyzer = MarketAnalyzer()
        return self._market_analyzer
    
    @property
    def risk_engine(self):
        if self._risk_engine is None:
            from aladdin.risk_engine import RiskEngine
            self._risk_engine = RiskEngine(self.config["risk"])
        return self._risk_engine
    
    @property
    def executor(self):
        if self._executor is None:
            from aladdin.executor import TradeExecutor
            self._executor = TradeExecutor()
        return self._executor
    
    # ==================== Core Analysis ====================
    
    def analyze_market(self, symbol: str) -> MarketState:
        """
        Comprehensive market analysis combining all factors
        """
        logger.info(f"Analyzing market for {symbol}...")
        
        # 1. Technical Analysis
        technical = self.market_analyzer.analyze(symbol)
        
        # 2. Sentiment Analysis
        sentiment = self.sentiment_analyzer.get_sentiment(symbol)
        
        # 3. Update Market State
        self.market_state = MarketState(
            regime=self._determine_regime(technical),
            trend_direction=technical.get("trend", 0),
            volatility=technical.get("volatility", 0),
            momentum=technical.get("momentum", 0),
            volume_profile=technical.get("volume_profile", "normal"),
            fear_greed_index=sentiment.get("fear_greed", 50),
            news_sentiment=sentiment.get("news_sentiment", 0),
            economic_outlook=sentiment.get("economic_outlook", "neutral"),
            timestamp=datetime.now()
        )
        
        logger.info(f"Market State: {self.market_state.regime.value}, "
                   f"Trend: {self.market_state.trend_direction:.2f}, "
                   f"Sentiment: {self.market_state.news_sentiment:.2f}")
        
        return self.market_state
    
    def _determine_regime(self, technical: Dict) -> MarketRegime:
        """Determine market regime from technical indicators"""
        trend = technical.get("trend", 0)
        volatility = technical.get("volatility", 0)
        
        # High volatility regime takes precedence
        if volatility > 0.8:
            return MarketRegime.HIGH_VOLATILITY
        if volatility < 0.2:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend-based regime
        if trend > 0.7:
            return MarketRegime.STRONG_BULL
        elif trend > 0.3:
            return MarketRegime.BULL
        elif trend < -0.7:
            return MarketRegime.STRONG_BEAR
        elif trend < -0.3:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL
    
    # ==================== Signal Generation ====================
    
    def generate_signals(self, symbol: str) -> List[TradingSignal]:
        """
        Generate trading signals using multiple strategies
        """
        signals = []
        
        # Get current market data
        market_data = self.market_analyzer.get_current_data(symbol)
        if not market_data:
            return signals
        
        current_price = market_data.get("price", 0)
        
        # Strategy 1: Trend Following
        if "trend_following" in self.config["strategies"]["enabled"]:
            signal = self._trend_following_signal(symbol, market_data)
            if signal:
                signals.append(signal)
        
        # Strategy 2: Mean Reversion
        if "mean_reversion" in self.config["strategies"]["enabled"]:
            signal = self._mean_reversion_signal(symbol, market_data)
            if signal:
                signals.append(signal)
        
        # Strategy 3: Momentum
        if "momentum" in self.config["strategies"]["enabled"]:
            signal = self._momentum_signal(symbol, market_data)
            if signal:
                signals.append(signal)
        
        # Strategy 4: Breakout
        if "breakout" in self.config["strategies"]["enabled"]:
            signal = self._breakout_signal(symbol, market_data)
            if signal:
                signals.append(signal)
        
        # Strategy 5: Sentiment Based
        if "sentiment_based" in self.config["strategies"]["enabled"]:
            signal = self._sentiment_signal(symbol, market_data)
            if signal:
                signals.append(signal)
        
        # Filter and rank signals
        valid_signals = self._filter_signals(signals)
        
        logger.info(f"Generated {len(valid_signals)} valid signals for {symbol}")
        return valid_signals
    
    def _trend_following_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Trend following strategy"""
        trend = self.market_state.trend_direction
        price = data.get("price", 0)
        atr = data.get("atr", price * 0.02)
        
        if abs(trend) < 0.3:
            return None  # No clear trend
        
        direction = "long" if trend > 0 else "short"
        strength = SignalStrength.STRONG if abs(trend) > 0.6 else SignalStrength.MODERATE
        
        # Calculate levels
        if direction == "long":
            stop_loss = price - (atr * 2)
            take_profit = price + (atr * 4)
        else:
            stop_loss = price + (atr * 2)
            take_profit = price - (atr * 4)
        
        return TradingSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(price, stop_loss),
            strategy="trend_following",
            confidence=0.5 + abs(trend) * 0.4,
            reasons=[
                f"Strong {direction} trend detected",
                f"Trend strength: {abs(trend):.2f}",
                f"Regime: {self.market_state.regime.value}"
            ]
        )
    
    def _mean_reversion_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Mean reversion strategy"""
        price = data.get("price", 0)
        bb_upper = data.get("bb_upper", price * 1.02)
        bb_lower = data.get("bb_lower", price * 0.98)
        bb_middle = data.get("bb_middle", price)
        rsi = data.get("rsi", 50)
        atr = data.get("atr", price * 0.02)
        
        signal = None
        
        # Oversold condition
        if price < bb_lower and rsi < 30:
            signal = TradingSignal(
                symbol=symbol,
                direction="long",
                strength=SignalStrength.STRONG if rsi < 20 else SignalStrength.MODERATE,
                entry_price=price,
                stop_loss=price - (atr * 1.5),
                take_profit=bb_middle,
                position_size=self._calculate_position_size(price, price - atr * 1.5),
                strategy="mean_reversion",
                confidence=0.6 + (30 - rsi) / 100,
                reasons=[
                    "Price below lower Bollinger Band",
                    f"RSI oversold at {rsi:.1f}",
                    "Mean reversion expected"
                ]
            )
        
        # Overbought condition
        elif price > bb_upper and rsi > 70:
            signal = TradingSignal(
                symbol=symbol,
                direction="short",
                strength=SignalStrength.STRONG if rsi > 80 else SignalStrength.MODERATE,
                entry_price=price,
                stop_loss=price + (atr * 1.5),
                take_profit=bb_middle,
                position_size=self._calculate_position_size(price, price + atr * 1.5),
                strategy="mean_reversion",
                confidence=0.6 + (rsi - 70) / 100,
                reasons=[
                    "Price above upper Bollinger Band",
                    f"RSI overbought at {rsi:.1f}",
                    "Mean reversion expected"
                ]
            )
        
        return signal
    
    def _momentum_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Momentum strategy"""
        momentum = self.market_state.momentum
        price = data.get("price", 0)
        macd = data.get("macd", 0)
        macd_signal = data.get("macd_signal", 0)
        atr = data.get("atr", price * 0.02)
        
        if abs(momentum) < 0.3:
            return None
        
        macd_bullish = macd > macd_signal
        direction = "long" if momentum > 0 and macd_bullish else "short" if momentum < 0 and not macd_bullish else None
        
        if not direction:
            return None
        
        if direction == "long":
            stop_loss = price - (atr * 2)
            take_profit = price + (atr * 3)
        else:
            stop_loss = price + (atr * 2)
            take_profit = price - (atr * 3)
        
        return TradingSignal(
            symbol=symbol,
            direction=direction,
            strength=SignalStrength.STRONG if abs(momentum) > 0.6 else SignalStrength.MODERATE,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(price, stop_loss),
            strategy="momentum",
            confidence=0.5 + abs(momentum) * 0.3,
            reasons=[
                f"Strong momentum: {momentum:.2f}",
                f"MACD {'bullish' if macd_bullish else 'bearish'} crossover",
            ]
        )
    
    def _breakout_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Breakout strategy"""
        price = data.get("price", 0)
        high_20 = data.get("high_20", price * 1.05)
        low_20 = data.get("low_20", price * 0.95)
        volume = data.get("volume", 0)
        avg_volume = data.get("avg_volume", volume)
        atr = data.get("atr", price * 0.02)
        
        volume_surge = volume > avg_volume * 1.5
        
        if price > high_20 and volume_surge:
            return TradingSignal(
                symbol=symbol,
                direction="long",
                strength=SignalStrength.STRONG,
                entry_price=price,
                stop_loss=low_20,
                take_profit=price + (high_20 - low_20),
                position_size=self._calculate_position_size(price, low_20),
                strategy="breakout",
                confidence=0.7,
                reasons=[
                    "Price broke 20-day high",
                    "Volume surge confirmed",
                    "Breakout momentum strong"
                ]
            )
        
        elif price < low_20 and volume_surge:
            return TradingSignal(
                symbol=symbol,
                direction="short",
                strength=SignalStrength.STRONG,
                entry_price=price,
                stop_loss=high_20,
                take_profit=price - (high_20 - low_20),
                position_size=self._calculate_position_size(price, high_20),
                strategy="breakout",
                confidence=0.7,
                reasons=[
                    "Price broke 20-day low",
                    "Volume surge confirmed",
                    "Breakdown momentum strong"
                ]
            )
        
        return None
    
    def _sentiment_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """Sentiment-based strategy"""
        news_sentiment = self.market_state.news_sentiment
        fear_greed = self.market_state.fear_greed_index
        price = data.get("price", 0)
        atr = data.get("atr", price * 0.02)
        
        # Strong bullish sentiment
        if news_sentiment > 0.5 and fear_greed > 60:
            return TradingSignal(
                symbol=symbol,
                direction="long",
                strength=SignalStrength.MODERATE,
                entry_price=price,
                stop_loss=price - (atr * 2),
                take_profit=price + (atr * 3),
                position_size=self._calculate_position_size(price, price - atr * 2),
                strategy="sentiment_based",
                confidence=0.4 + news_sentiment * 0.3,
                reasons=[
                    f"Bullish news sentiment: {news_sentiment:.2f}",
                    f"Fear/Greed Index: {fear_greed:.0f} (Greed)",
                    "Market sentiment favorable"
                ]
            )
        
        # Strong bearish sentiment
        elif news_sentiment < -0.5 and fear_greed < 40:
            return TradingSignal(
                symbol=symbol,
                direction="short",
                strength=SignalStrength.MODERATE,
                entry_price=price,
                stop_loss=price + (atr * 2),
                take_profit=price - (atr * 3),
                position_size=self._calculate_position_size(price, price + atr * 2),
                strategy="sentiment_based",
                confidence=0.4 + abs(news_sentiment) * 0.3,
                reasons=[
                    f"Bearish news sentiment: {news_sentiment:.2f}",
                    f"Fear/Greed Index: {fear_greed:.0f} (Fear)",
                    "Market sentiment negative"
                ]
            )
        
        return None
    
    def _calculate_position_size(self, entry: float, stop_loss: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.portfolio.total_equity * (self.config["risk"]["risk_per_trade_pct"] / 100)
        risk_per_unit = abs(entry - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        # Cap at max position size
        max_size = self.portfolio.total_equity * (self.config["trading"]["max_position_size_pct"] / 100) / entry
        return min(position_size, max_size)
    
    def _filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter and rank signals"""
        min_strength = self.config["trading"]["min_signal_strength"]
        min_confidence = self.config["trading"]["min_confidence"]
        
        valid = [
            s for s in signals
            if s.strength.value >= min_strength and s.confidence >= min_confidence
        ]
        
        # Sort by confidence * strength
        valid.sort(key=lambda s: s.confidence * s.strength.value, reverse=True)
        
        return valid
    
    # ==================== Decision Making ====================
    
    def make_decision(self, symbol: str) -> Optional[TradingSignal]:
        """
        Make autonomous trading decision
        """
        # 1. Analyze market
        self.analyze_market(symbol)
        
        # 2. Check if trading is allowed
        if not self.risk_engine.can_trade(self.portfolio):
            logger.warning("Trading restricted by risk engine")
            return None
        
        # 3. Generate signals
        signals = self.generate_signals(symbol)
        
        if not signals:
            logger.info(f"No valid signals for {symbol}")
            return None
        
        # 4. Select best signal
        best_signal = signals[0]
        
        # 5. Risk check on specific signal
        if not self.risk_engine.approve_signal(best_signal, self.portfolio):
            logger.warning(f"Signal rejected by risk engine: {best_signal.strategy}")
            return None
        
        logger.info(f"DECISION: {best_signal.direction.upper()} {symbol} "
                   f"@ {best_signal.entry_price:.2f} "
                   f"[{best_signal.strategy}] "
                   f"Confidence: {best_signal.confidence:.2%}")
        
        return best_signal
    
    # ==================== Execution ====================
    
    def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute trading signal"""
        logger.info(f"Executing {signal.direction} {signal.symbol}...")
        
        try:
            result = self.executor.execute(signal)
            
            if result.get("success"):
                # Update portfolio
                self.portfolio.open_positions[signal.symbol] = {
                    "direction": signal.direction,
                    "entry_price": signal.entry_price,
                    "size": signal.position_size,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "strategy": signal.strategy,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Order executed successfully: {result}")
                return True
            else:
                logger.error(f"Order failed: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
    
    # ==================== Autonomous Loop ====================
    
    def start(self):
        """Start autonomous trading"""
        if self.is_running:
            logger.warning("Aladdin AI is already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info("=" * 60)
        logger.info("ALADDIN AI STARTED - Autonomous Trading Active")
        logger.info("=" * 60)
    
    def stop(self):
        """Stop autonomous trading"""
        self.is_running = False
        logger.info("Aladdin AI stopped")
    
    def _run_loop(self):
        """Main autonomous trading loop"""
        while self.is_running:
            try:
                for symbol in self.config["trading"]["symbols"]:
                    # Make decision
                    signal = self.make_decision(symbol)
                    
                    if signal:
                        # Execute trade
                        self.execute_signal(signal)
                    
                    time.sleep(1)  # Brief pause between symbols
                
                # Wait for next analysis cycle
                interval = self.config["analysis"]["update_interval_seconds"]
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
    
    # ==================== Status & Reporting ====================
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "is_running": self.is_running,
            "market_state": {
                "regime": self.market_state.regime.value,
                "trend": self.market_state.trend_direction,
                "volatility": self.market_state.volatility,
                "sentiment": self.market_state.news_sentiment,
                "fear_greed": self.market_state.fear_greed_index
            },
            "portfolio": {
                "equity": self.portfolio.total_equity,
                "open_positions": len(self.portfolio.open_positions),
                "daily_pnl": self.portfolio.daily_pnl,
                "total_pnl": self.portfolio.total_pnl
            },
            "last_analysis": self._last_analysis_time
        }
    
    def print_status(self):
        """Print formatted status"""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("ALADDIN AI STATUS")
        print("=" * 60)
        print(f"Running: {'YES' if status['is_running'] else 'NO'}")
        print(f"\nMarket State:")
        print(f"  Regime: {status['market_state']['regime']}")
        print(f"  Trend: {status['market_state']['trend']:.2f}")
        print(f"  Volatility: {status['market_state']['volatility']:.2f}")
        print(f"  Sentiment: {status['market_state']['sentiment']:.2f}")
        print(f"  Fear/Greed: {status['market_state']['fear_greed']:.0f}")
        print(f"\nPortfolio:")
        print(f"  Equity: ${status['portfolio']['equity']:,.2f}")
        print(f"  Open Positions: {status['portfolio']['open_positions']}")
        print(f"  Daily P&L: ${status['portfolio']['daily_pnl']:,.2f}")
        print(f"  Total P&L: ${status['portfolio']['total_pnl']:,.2f}")
        print("=" * 60)
