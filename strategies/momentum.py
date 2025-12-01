"""
Momentum Strategy - Multi-modal signal fusion for directional trades

This strategy combines:
- EMA crossover for trend direction
- Price momentum over lookback period
- Orderbook imbalance for buying/selling pressure
- Sentiment score from social/news data

Trade when signals align with sufficient confidence.
"""

import numpy as np
from typing import Optional, Dict, Any

from strategies.base import StrategyBase, StrategyContext, Candle, Order
from signals.technical import ema, sma, atr, momentum as calc_momentum, rsi
from signals.orderbook import compute_imbalance
from signals.sentiment import get_composite_sentiment


class MomentumStrategy(StrategyBase):
    """
    Multi-modal momentum strategy
    
    Entry conditions (Long):
    - Price > EMA(fast) > EMA(slow) [trend alignment]
    - Momentum > threshold [directional strength]
    - OB imbalance > 0 [buying pressure]
    - Sentiment > bull_threshold OR neutral
    - Composite confidence >= min_confidence
    
    Exit conditions:
    - Stop loss hit
    - Take profit hit
    - Trend reversal (EMA cross back)
    """
    
    name = "momentum"
    version = "2.0.0"
    author = "Delta Algo Bot"
    description = "Multi-modal momentum strategy with EMA, orderbook, and sentiment"
    
    # Default parameters (can be overridden)
    params = {
        # EMA settings
        "ema_fast": 20,
        "ema_slow": 50,
        
        # Momentum settings
        "momentum_period": 20,
        "momentum_threshold": 0.002,  # 0.2%
        
        # Orderbook settings
        "ob_imbalance_threshold": 0.05,
        
        # Sentiment settings
        "sentiment_bull_threshold": 0.3,
        "sentiment_bear_threshold": -0.3,
        
        # Confidence settings
        "min_confidence": 0.65,
        "signal_weights": {
            "momentum": 0.40,
            "orderbook": 0.20,
            "sentiment": 0.25,
            "news": 0.15
        },
        
        # Risk settings
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "risk_reward_ratio": 2.0,
        
        # Warmup
        "warmup_period": 60
    }
    
    def __init__(self, ctx: StrategyContext, **kwargs):
        super().__init__(ctx)
        
        # Override default params with kwargs
        self.params = {**self.params, **kwargs}
        
        # Cache for computed indicators
        self._indicators: Dict[str, Any] = {}
    
    def on_start(self):
        """Initialize strategy"""
        self.log("Strategy started", params=self.params)
    
    def on_candle(self, candle: Candle):
        """Main strategy logic on each candle"""
        # Add candle to history
        self._add_candle(candle)
        
        # Check warmup
        if len(self._candles) < self.params["warmup_period"]:
            return
        
        # Compute indicators
        self._compute_indicators()
        
        # Generate signal
        signal, confidence, direction = self._generate_signal()
        
        # Check if we should trade
        if confidence >= self.params["min_confidence"] and signal != 0:
            self._execute_signal(signal, confidence, direction)
        
        # Manage existing position
        self._manage_position()
    
    def _compute_indicators(self):
        """Compute all technical indicators"""
        closes = self.close
        highs = self.high
        lows = self.low
        
        # EMAs
        self._indicators["ema_fast"] = ema(closes, self.params["ema_fast"])
        self._indicators["ema_slow"] = ema(closes, self.params["ema_slow"])
        
        # Momentum
        self._indicators["momentum"] = calc_momentum(closes, self.params["momentum_period"])
        
        # ATR for stops
        self._indicators["atr"] = atr(highs, lows, closes, self.params["atr_period"])
        
        # RSI (optional filter)
        self._indicators["rsi"] = rsi(closes, 14)
    
    def _generate_signal(self) -> tuple:
        """
        Generate trading signal from multiple sources
        
        Returns:
            (signal, confidence, direction)
            signal: 1 for long, -1 for short, 0 for no trade
            confidence: 0-1 score
            direction: 'long', 'short', or 'neutral'
        """
        signals = {}
        
        # 1. Momentum signal
        ema_fast = self._indicators["ema_fast"][-1]
        ema_slow = self._indicators["ema_slow"][-1]
        price = self.price
        momentum = self._indicators["momentum"][-1]
        
        # Trend alignment
        if price > ema_fast > ema_slow:
            trend_signal = 1.0
        elif price < ema_fast < ema_slow:
            trend_signal = -1.0
        else:
            trend_signal = 0.0
        
        # Momentum strength
        if abs(momentum) > self.params["momentum_threshold"]:
            momentum_signal = np.sign(momentum) * min(abs(momentum) / 0.01, 1.0)
        else:
            momentum_signal = 0.0
        
        # Combined momentum score
        signals["momentum"] = (trend_signal * 0.6 + momentum_signal * 0.4)
        
        # 2. Orderbook signal
        try:
            orderbook = self.fetch_orderbook()
            ob_imbalance = compute_imbalance(orderbook)
            
            if abs(ob_imbalance) > self.params["ob_imbalance_threshold"]:
                signals["orderbook"] = np.clip(ob_imbalance * 5, -1, 1)
            else:
                signals["orderbook"] = 0.0
        except Exception:
            signals["orderbook"] = 0.0
        
        # 3. Sentiment signal
        try:
            sentiment = self.get_sentiment()
            if sentiment > self.params["sentiment_bull_threshold"]:
                signals["sentiment"] = sentiment
            elif sentiment < self.params["sentiment_bear_threshold"]:
                signals["sentiment"] = sentiment
            else:
                signals["sentiment"] = 0.0
        except Exception:
            signals["sentiment"] = 0.0
        
        # 4. News signal
        try:
            news_score = self.get_news_score()
            signals["news"] = news_score
        except Exception:
            signals["news"] = 0.0
        
        # Compute weighted composite
        weights = self.params["signal_weights"]
        composite = 0.0
        total_weight = 0.0
        
        for signal_name, weight in weights.items():
            if signal_name in signals:
                composite += signals[signal_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            composite /= total_weight
        
        # Determine signal and confidence
        confidence = abs(composite)
        
        if composite > 0:
            return 1, confidence, "long"
        elif composite < 0:
            return -1, confidence, "short"
        else:
            return 0, 0.0, "neutral"
    
    def _execute_signal(self, signal: int, confidence: float, direction: str):
        """Execute a trading signal"""
        # Don't trade if already in position
        if self.position.is_open:
            return
        
        # Calculate stop and take profit
        atr_value = self._indicators["atr"][-1]
        stop_distance = atr_value * self.params["atr_multiplier"]
        tp_distance = stop_distance * self.params["risk_reward_ratio"]
        
        if signal == 1:  # Long
            stop_loss = self.price - stop_distance
            take_profit = self.price + tp_distance
            
            self.log(
                f"LONG signal",
                confidence=f"{confidence:.2%}",
                entry=self.price,
                stop=stop_loss,
                tp=take_profit
            )
            
            self.buy(
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        elif signal == -1:  # Short
            stop_loss = self.price + stop_distance
            take_profit = self.price - tp_distance
            
            self.log(
                f"SHORT signal",
                confidence=f"{confidence:.2%}",
                entry=self.price,
                stop=stop_loss,
                tp=take_profit
            )
            
            self.sell(
                stop_loss=stop_loss,
                take_profit=take_profit
            )
    
    def _manage_position(self):
        """Manage existing position (trailing stop, etc.)"""
        if not self.position.is_open:
            return
        
        # Check for trend reversal exit
        ema_fast = self._indicators["ema_fast"][-1]
        ema_slow = self._indicators["ema_slow"][-1]
        
        if self.position.is_long and ema_fast < ema_slow:
            self.log("Closing long - EMA crossover down")
            self.close()
        elif self.position.is_short and ema_fast > ema_slow:
            self.log("Closing short - EMA crossover up")
            self.close()
    
    def on_position_closed(self, position, pnl: float):
        """Track closed trades"""
        self._record_trade(pnl)
        self.log(f"Position closed", pnl=pnl, total_pnl=self.total_pnl)
    
    def on_exit(self):
        """Cleanup on exit"""
        self.log(
            "Strategy stopped",
            trades=self.trade_count,
            win_rate=f"{self.win_rate:.1%}",
            total_pnl=self.total_pnl
        )
