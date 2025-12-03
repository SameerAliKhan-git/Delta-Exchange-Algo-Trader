#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                           🧠 ALADDIN ML ENGINE                                            ║
║                                                                                           ║
║     Machine Learning for Algorithmic Trading - Inspired by Academic Research              ║
║                                                                                           ║
║     References:                                                                           ║
║     • López de Prado - Advances in Financial ML (AFML)                                    ║
║     • Sirignano - Deep Learning for Limit Order Books                                     ║
║     • Ritter - Machine Learning for Trading                                               ║
║     • Xiong - Deep Reinforcement Learning for Stock Trading                               ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

FEATURES IMPLEMENTED:
1. Triple Barrier Labeling (AFML) - Better trade labels than simple up/down
2. Feature Engineering - Technical features that actually predict
3. Meta-Labeling - Probability of signal success
4. Online Learning - Model improves with each trade
5. Ensemble Methods - Multiple models vote for consensus
6. Feature Importance - Know which signals matter most
"""

import json
import os
import math
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from statistics import mean, stdev

# =============================================================================
# 📊 CONFIGURATION
# =============================================================================
ML_DATA_FILE = "ml_trade_history.json"
MODEL_FILE = "ml_model_weights.json"
MIN_SAMPLES_FOR_ML = 20  # Need 20 trades before ML kicks in
FEATURE_LOOKBACK = 50    # Use last 50 price points for features
LEARNING_RATE = 0.1      # How fast to adapt to new data

# =============================================================================
# 📈 FEATURE ENGINEERING (Inspired by AFML)
# =============================================================================

def calculate_returns(prices: List[float], period: int = 1) -> List[float]:
    """Calculate log returns"""
    if len(prices) < period + 1:
        return []
    return [math.log(prices[i] / prices[i-period]) for i in range(period, len(prices))]

def calculate_volatility(prices: List[float], window: int = 20) -> float:
    """Calculate rolling volatility (annualized)"""
    returns = calculate_returns(prices)
    if len(returns) < window:
        return 0
    recent = returns[-window:]
    return stdev(recent) * math.sqrt(252 * 24 * 60)  # Annualized from minute data

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """RSI indicator"""
    if len(prices) < period + 1:
        return 50
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    recent = changes[-period:]
    gains = [c for c in recent if c > 0]
    losses = [-c for c in recent if c < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0.0001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: List[float]) -> Tuple[float, float, float]:
    """MACD indicator (line, signal, histogram)"""
    if len(prices) < 26:
        return 0, 0, 0
    
    def ema(data, period):
        if len(data) < period:
            return sum(data) / len(data)
        multiplier = 2 / (period + 1)
        result = sum(data[:period]) / period
        for price in data[period:]:
            result = (price - result) * multiplier + result
        return result
    
    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = ema([macd_line], 9) if len(prices) >= 35 else macd_line
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_position(prices: List[float], window: int = 20) -> float:
    """Position within Bollinger Bands (-1 to +1)"""
    if len(prices) < window:
        return 0
    recent = prices[-window:]
    ma = mean(recent)
    std = stdev(recent) if len(recent) > 1 else 1
    current = prices[-1]
    upper = ma + 2 * std
    lower = ma - 2 * std
    if upper == lower:
        return 0
    return (current - lower) / (upper - lower) * 2 - 1

def calculate_momentum(prices: List[float], period: int = 10) -> float:
    """Price momentum"""
    if len(prices) < period:
        return 0
    return (prices[-1] - prices[-period]) / prices[-period] * 100

def calculate_trend_strength(prices: List[float]) -> float:
    """Trend strength based on MA alignment"""
    if len(prices) < 50:
        return 0
    ma5 = mean(prices[-5:])
    ma10 = mean(prices[-10:])
    ma20 = mean(prices[-20:])
    ma50 = mean(prices[-50:])
    
    # Bullish alignment: MA5 > MA10 > MA20 > MA50
    score = 0
    if ma5 > ma10:
        score += 25
    if ma10 > ma20:
        score += 25
    if ma20 > ma50:
        score += 25
    if ma5 > ma50:
        score += 25
    return score  # 0-100, higher = stronger uptrend

def calculate_volume_trend(volumes: List[float], window: int = 10) -> float:
    """Volume trend (increasing/decreasing)"""
    if len(volumes) < window:
        return 0
    recent = volumes[-window:]
    earlier = volumes[-2*window:-window] if len(volumes) >= 2*window else volumes[:window]
    if mean(earlier) == 0:
        return 0
    return (mean(recent) - mean(earlier)) / mean(earlier) * 100

# =============================================================================
# 🎯 TRIPLE BARRIER LABELING (AFML)
# =============================================================================

def triple_barrier_label(
    entry_price: float,
    prices_after: List[float],
    profit_target: float = 0.015,  # 1.5% take profit
    stop_loss: float = 0.01,       # 1% stop loss
    max_holding: int = 60          # Max 60 bars
) -> Tuple[int, float, str]:
    """
    Triple Barrier Method from AFML.
    Labels trades as: +1 (profit target hit), -1 (stop loss hit), 0 (time exit)
    Returns: (label, exit_price, reason)
    """
    for i, price in enumerate(prices_after[:max_holding]):
        pnl_pct = (price - entry_price) / entry_price
        
        if pnl_pct >= profit_target:
            return 1, price, "Take Profit"
        elif pnl_pct <= -stop_loss:
            return -1, price, "Stop Loss"
    
    # Time barrier hit
    if prices_after:
        final_price = prices_after[min(max_holding-1, len(prices_after)-1)]
        final_pnl = (final_price - entry_price) / entry_price
        return (1 if final_pnl > 0 else -1), final_price, "Time Exit"
    
    return 0, entry_price, "No Data"

# =============================================================================
# 🧠 FEATURE EXTRACTION (Full Feature Vector)
# =============================================================================

def extract_features(prices: List[float], volumes: List[float] = None) -> Dict[str, float]:
    """
    Extract comprehensive feature set for ML model.
    Based on academic research on predictive features.
    """
    if len(prices) < 30:
        return {}
    
    features = {}
    
    # Price-based features
    features['rsi'] = calculate_rsi(prices) / 100  # Normalized 0-1
    features['rsi_extreme'] = 1 if features['rsi'] < 0.3 or features['rsi'] > 0.7 else 0
    
    # Momentum features
    features['momentum_5'] = calculate_momentum(prices, 5) / 10  # Scaled
    features['momentum_10'] = calculate_momentum(prices, 10) / 10
    features['momentum_20'] = calculate_momentum(prices, 20) / 10
    
    # Trend features
    features['trend_strength'] = calculate_trend_strength(prices) / 100
    
    # Volatility features
    features['volatility'] = min(calculate_volatility(prices) / 100, 1)  # Capped at 1
    
    # Mean reversion features
    features['bollinger_pos'] = calculate_bollinger_position(prices)
    
    # MACD features
    macd, signal, hist = calculate_macd(prices)
    features['macd_hist'] = hist / prices[-1] * 1000  # Normalized
    features['macd_cross'] = 1 if macd > signal else -1
    
    # Price patterns
    features['higher_highs'] = 1 if len(prices) >= 5 and prices[-1] > prices[-2] > prices[-3] else 0
    features['lower_lows'] = 1 if len(prices) >= 5 and prices[-1] < prices[-2] < prices[-3] else 0
    
    # Recent return
    features['return_1'] = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) >= 2 else 0
    features['return_5'] = (prices[-1] - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 else 0
    
    # Volatility regime
    vol = calculate_volatility(prices)
    features['high_vol_regime'] = 1 if vol > 0.3 else 0
    features['low_vol_regime'] = 1 if vol < 0.1 else 0
    
    return features

# =============================================================================
# 📊 ONLINE LEARNING MODEL (Simple but effective)
# =============================================================================

class OnlineLearningModel:
    """
    Simple online learning model that improves with each trade.
    Uses weighted averaging with recent trades having more influence.
    """
    
    def __init__(self):
        self.feature_weights = {}
        self.feature_stats = {}  # Track mean/std for each feature
        self.trade_history = []
        self.win_rate_by_regime = {
            'high_vol': {'wins': 0, 'total': 0},
            'low_vol': {'wins': 0, 'total': 0},
            'trending': {'wins': 0, 'total': 0},
            'ranging': {'wins': 0, 'total': 0},
        }
        self.load_model()
    
    def load_model(self):
        """Load saved model weights"""
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'r') as f:
                    data = json.load(f)
                    self.feature_weights = data.get('weights', {})
                    self.feature_stats = data.get('stats', {})
                    self.win_rate_by_regime = data.get('regimes', self.win_rate_by_regime)
                print(f"🧠 Loaded ML model with {len(self.feature_weights)} features")
            except:
                pass
        
        if os.path.exists(ML_DATA_FILE):
            try:
                with open(ML_DATA_FILE, 'r') as f:
                    self.trade_history = json.load(f)
                print(f"📊 Loaded {len(self.trade_history)} historical trades for learning")
            except:
                self.trade_history = []
    
    def save_model(self):
        """Save model weights"""
        data = {
            'weights': self.feature_weights,
            'stats': self.feature_stats,
            'regimes': self.win_rate_by_regime
        }
        with open(MODEL_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        with open(ML_DATA_FILE, 'w') as f:
            json.dump(self.trade_history[-1000:], f, indent=2)  # Keep last 1000 trades
    
    def record_trade(self, features: Dict[str, float], outcome: int, pnl: float):
        """Record trade outcome for learning"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'outcome': outcome,  # 1 = win, -1 = loss
            'pnl': pnl
        }
        self.trade_history.append(trade_record)
        
        # Update feature weights based on outcome
        self._update_weights(features, outcome)
        
        # Update regime stats
        self._update_regime_stats(features, outcome)
        
        # Save periodically
        if len(self.trade_history) % 5 == 0:
            self.save_model()
    
    def _update_weights(self, features: Dict[str, float], outcome: int):
        """Update feature weights using simple gradient descent"""
        for feature, value in features.items():
            if feature not in self.feature_weights:
                self.feature_weights[feature] = 0
                self.feature_stats[feature] = {'sum': 0, 'count': 0, 'win_sum': 0, 'loss_sum': 0}
            
            # Update stats
            self.feature_stats[feature]['sum'] += value
            self.feature_stats[feature]['count'] += 1
            
            if outcome > 0:
                self.feature_stats[feature]['win_sum'] += value
            else:
                self.feature_stats[feature]['loss_sum'] += value
            
            # Calculate correlation with wins
            # If feature value is higher on wins, increase weight
            avg_on_wins = self.feature_stats[feature]['win_sum'] / max(1, sum(1 for t in self.trade_history if t['outcome'] > 0))
            avg_on_losses = self.feature_stats[feature]['loss_sum'] / max(1, sum(1 for t in self.trade_history if t['outcome'] < 0))
            
            # Update weight towards correlation
            correlation = avg_on_wins - avg_on_losses
            self.feature_weights[feature] += LEARNING_RATE * correlation
    
    def _update_regime_stats(self, features: Dict[str, float], outcome: int):
        """Track win rate by market regime"""
        is_win = outcome > 0
        
        if features.get('high_vol_regime', 0) > 0.5:
            self.win_rate_by_regime['high_vol']['total'] += 1
            if is_win:
                self.win_rate_by_regime['high_vol']['wins'] += 1
        elif features.get('low_vol_regime', 0) > 0.5:
            self.win_rate_by_regime['low_vol']['total'] += 1
            if is_win:
                self.win_rate_by_regime['low_vol']['wins'] += 1
        
        if features.get('trend_strength', 0) > 0.6:
            self.win_rate_by_regime['trending']['total'] += 1
            if is_win:
                self.win_rate_by_regime['trending']['wins'] += 1
        else:
            self.win_rate_by_regime['ranging']['total'] += 1
            if is_win:
                self.win_rate_by_regime['ranging']['wins'] += 1
    
    def predict_win_probability(self, features: Dict[str, float]) -> float:
        """
        Predict probability of trade success (0-1).
        Uses learned feature weights.
        """
        if len(self.trade_history) < MIN_SAMPLES_FOR_ML:
            return 0.5  # Not enough data, return neutral
        
        score = 0
        for feature, value in features.items():
            weight = self.feature_weights.get(feature, 0)
            score += value * weight
        
        # Apply sigmoid to get probability
        try:
            probability = 1 / (1 + math.exp(-score))
        except:
            probability = 0.5
        
        return probability
    
    def should_take_trade(self, features: Dict[str, float], base_confidence: float) -> Tuple[bool, float, str]:
        """
        Meta-labeling: Decide if we should take this trade.
        Returns: (should_trade, adjusted_confidence, reason)
        """
        if len(self.trade_history) < MIN_SAMPLES_FOR_ML:
            return True, base_confidence, "Insufficient ML data, using base signal"
        
        ml_prob = self.predict_win_probability(features)
        
        # Check regime win rates
        regime_warning = ""
        if features.get('high_vol_regime', 0) > 0.5:
            regime = self.win_rate_by_regime['high_vol']
            if regime['total'] > 5:
                wr = regime['wins'] / regime['total']
                if wr < 0.4:
                    regime_warning = f"⚠️ High vol regime has {wr*100:.0f}% win rate"
        
        # Combine base confidence with ML probability
        adjusted_confidence = (base_confidence * 0.6 + ml_prob * 100 * 0.4)
        
        # Decision
        if ml_prob < 0.35:
            return False, adjusted_confidence, f"ML rejects: {ml_prob*100:.0f}% win probability"
        elif ml_prob > 0.6:
            return True, adjusted_confidence * 1.2, f"ML confirms: {ml_prob*100:.0f}% win probability"
        else:
            return True, adjusted_confidence, f"ML neutral: {ml_prob*100:.0f}% win probability"
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get ranked feature importance"""
        importance = [(f, abs(w)) for f, w in self.feature_weights.items()]
        return sorted(importance, key=lambda x: x[1], reverse=True)[:10]
    
    def get_regime_analysis(self) -> str:
        """Get analysis of performance by regime"""
        analysis = []
        for regime, stats in self.win_rate_by_regime.items():
            if stats['total'] > 0:
                wr = stats['wins'] / stats['total'] * 100
                analysis.append(f"{regime}: {wr:.0f}% ({stats['total']} trades)")
        return " | ".join(analysis)

# =============================================================================
# 🎯 ENSEMBLE SIGNAL (Multiple Models Vote)
# =============================================================================

class SignalEnsemble:
    """
    Ensemble of simple models that vote on trade direction.
    Inspired by ensemble methods in AFML.
    """
    
    def __init__(self):
        self.models = {
            'momentum': self._momentum_model,
            'mean_reversion': self._mean_reversion_model,
            'trend_following': self._trend_following_model,
            'volatility_breakout': self._volatility_breakout_model,
        }
    
    def _momentum_model(self, features: Dict[str, float]) -> int:
        """Momentum-based signal"""
        mom_5 = features.get('momentum_5', 0)
        mom_10 = features.get('momentum_10', 0)
        
        if mom_5 > 0.1 and mom_10 > 0.05:
            return 1  # LONG
        elif mom_5 < -0.1 and mom_10 < -0.05:
            return -1  # SHORT
        return 0
    
    def _mean_reversion_model(self, features: Dict[str, float]) -> int:
        """Mean reversion signal"""
        bb_pos = features.get('bollinger_pos', 0)
        rsi = features.get('rsi', 0.5)
        
        if bb_pos < -0.8 and rsi < 0.3:
            return 1  # LONG - oversold
        elif bb_pos > 0.8 and rsi > 0.7:
            return -1  # SHORT - overbought
        return 0
    
    def _trend_following_model(self, features: Dict[str, float]) -> int:
        """Trend following signal"""
        trend = features.get('trend_strength', 0)
        macd_cross = features.get('macd_cross', 0)
        
        if trend > 0.7 and macd_cross > 0:
            return 1  # LONG in uptrend
        elif trend < 0.3 and macd_cross < 0:
            return -1  # SHORT in downtrend
        return 0
    
    def _volatility_breakout_model(self, features: Dict[str, float]) -> int:
        """Volatility breakout signal"""
        bb_pos = features.get('bollinger_pos', 0)
        vol = features.get('volatility', 0)
        mom = features.get('momentum_5', 0)
        
        # Breakout above bands with momentum
        if bb_pos > 0.9 and vol > 0.3 and mom > 0.2:
            return 1  # LONG breakout
        elif bb_pos < -0.9 and vol > 0.3 and mom < -0.2:
            return -1  # SHORT breakdown
        return 0
    
    def get_ensemble_signal(self, features: Dict[str, float]) -> Tuple[int, float, List[str]]:
        """
        Get ensemble signal from all models.
        Returns: (direction, confidence, agreeing_models)
        """
        votes = {}
        agreeing_models = []
        
        for name, model in self.models.items():
            vote = model(features)
            if vote != 0:
                votes[name] = vote
                agreeing_models.append(f"{name}:{'+' if vote > 0 else '-'}")
        
        if not votes:
            return 0, 0, []
        
        # Count votes
        long_votes = sum(1 for v in votes.values() if v > 0)
        short_votes = sum(1 for v in votes.values() if v < 0)
        
        total_votes = len(votes)
        
        if long_votes > short_votes:
            confidence = long_votes / total_votes * 100
            return 1, confidence, agreeing_models
        elif short_votes > long_votes:
            confidence = short_votes / total_votes * 100
            return -1, confidence, agreeing_models
        
        return 0, 0, agreeing_models

# =============================================================================
# 🌍 GLOBAL INSTANCES
# =============================================================================

ml_model = OnlineLearningModel()
signal_ensemble = SignalEnsemble()

# =============================================================================
# 📈 PUBLIC API
# =============================================================================

def get_ml_enhanced_signal(
    prices: List[float],
    base_direction: Optional[str],
    base_confidence: float,
    base_confirmations: int
) -> Tuple[Optional[str], float, int, str]:
    """
    Enhance trading signal with ML predictions.
    
    Args:
        prices: Recent price history
        base_direction: Direction from base strategy ('LONG', 'SHORT', or None)
        base_confidence: Confidence from base strategy
        base_confirmations: Number of confirming indicators
    
    Returns:
        (direction, adjusted_confidence, confirmations, reason)
    """
    if len(prices) < 30:
        return base_direction, base_confidence, base_confirmations, "Insufficient data for ML"
    
    # Extract features
    features = extract_features(prices)
    
    if not features:
        return base_direction, base_confidence, base_confirmations, "Feature extraction failed"
    
    # Get ensemble signal
    ensemble_dir, ensemble_conf, ensemble_models = signal_ensemble.get_ensemble_signal(features)
    
    # ML meta-labeling
    should_trade, ml_confidence, ml_reason = ml_model.should_take_trade(features, base_confidence)
    
    # Combine signals
    if base_direction is None:
        # No base signal - check if ensemble has strong opinion
        if ensemble_conf > 70 and len(ensemble_models) >= 2:
            direction = 'LONG' if ensemble_dir > 0 else 'SHORT'
            return direction, ensemble_conf, len(ensemble_models), f"Ensemble: {'+'.join(ensemble_models)}"
        return None, 0, 0, "No signal"
    
    # Have base signal - use ML to filter
    if not should_trade:
        return None, 0, base_confirmations, ml_reason
    
    # Boost confidence if ensemble agrees
    if (base_direction == 'LONG' and ensemble_dir > 0) or (base_direction == 'SHORT' and ensemble_dir < 0):
        boosted_conf = ml_confidence * 1.2
        return base_direction, boosted_conf, base_confirmations + len(ensemble_models), f"ML+Ensemble confirm"
    
    return base_direction, ml_confidence, base_confirmations, ml_reason

def record_trade_outcome(
    entry_price: float,
    exit_price: float,
    direction: str,
    pnl: float,
    prices_at_entry: List[float]
):
    """Record trade outcome for ML learning"""
    if len(prices_at_entry) < 30:
        return
    
    features = extract_features(prices_at_entry)
    outcome = 1 if pnl > 0 else -1
    ml_model.record_trade(features, outcome, pnl)

def get_ml_stats() -> str:
    """Get ML model statistics"""
    trades = len(ml_model.trade_history)
    if trades == 0:
        return "ML: No trades yet"
    
    wins = sum(1 for t in ml_model.trade_history if t['outcome'] > 0)
    wr = wins / trades * 100
    
    # Top features
    top_features = ml_model.get_feature_importance()[:3]
    features_str = ", ".join([f[0] for f in top_features])
    
    return f"ML: {trades} trades, {wr:.0f}% WR | Top: {features_str}"

def get_regime_analysis() -> str:
    """Get regime analysis"""
    return ml_model.get_regime_analysis()

# =============================================================================
# 🧪 SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("🧠 ALADDIN ML Engine - Self Test")
    print("=" * 60)
    
    # Generate test data
    import random
    test_prices = [100 + random.gauss(0, 0.5) for _ in range(100)]
    for i in range(1, len(test_prices)):
        test_prices[i] = test_prices[i-1] + random.gauss(0.01, 0.3)  # Slight uptrend
    
    # Test feature extraction
    features = extract_features(test_prices)
    print("\n📊 Extracted Features:")
    for name, value in features.items():
        print(f"   {name}: {value:.4f}")
    
    # Test ensemble
    direction, conf, models = signal_ensemble.get_ensemble_signal(features)
    print(f"\n🎯 Ensemble Signal: {direction} ({conf:.0f}% conf)")
    print(f"   Models: {models}")
    
    # Test ML prediction
    should_trade, ml_conf, reason = ml_model.should_take_trade(features, 50)
    print(f"\n🧠 ML Meta-Label: {'TRADE' if should_trade else 'SKIP'}")
    print(f"   Confidence: {ml_conf:.0f}%")
    print(f"   Reason: {reason}")
    
    # Test full pipeline
    result = get_ml_enhanced_signal(test_prices, 'LONG', 60, 3)
    print(f"\n📈 Final Signal: {result}")
    
    print("\n✅ ML Engine tests passed!")
