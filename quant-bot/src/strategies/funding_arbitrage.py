"""
Strategy 4: Funding Rate / Perpetual Futures Arbitrage
======================================================

Why it works in crypto (LOW RISK!):
- Perpetual futures have funding payments every 8 hours
- Funding tends to be: Predictable, Mean-reverting, Overreacting
- One of the safest quant strategies in crypto

Concepts:
- Funding rate harvest
- Basis trade (spot vs perp)
- Index–perp spread arbitrage
- Predicting funding spikes
- Beta-neutral portfolios
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from .base import (
    BaseStrategy, StrategyConfig, Signal, SignalType,
    TechnicalIndicators
)


@dataclass
class FundingConfig(StrategyConfig):
    """Configuration for funding arbitrage strategies."""
    name: str = "funding_arbitrage"
    
    # Funding thresholds
    funding_entry_threshold: float = 0.0003  # 0.03% = 3 bps
    funding_exit_threshold: float = 0.0001  # 0.01% = 1 bp
    extreme_funding_threshold: float = 0.001  # 0.1% = 10 bps
    
    # Basis trade
    basis_entry_threshold: float = 0.003  # 0.3% annualized
    basis_exit_threshold: float = 0.001  # 0.1%
    
    # Prediction
    prediction_lookback: int = 100  # Historical funding rates
    
    # Risk
    max_position_per_pair: float = 0.2  # 20% of capital
    hedge_ratio: float = 1.0  # 1:1 hedge
    max_drawdown_pct: float = 0.02  # 2% max loss before exit
    
    # Timing
    funding_frequency_hours: int = 8
    entry_before_funding_hours: float = 4  # Enter 4 hours before
    exit_after_funding_hours: float = 1  # Exit 1 hour after
    
    # Regime
    allowed_regimes: List[str] = field(default_factory=lambda: ["all"])


@dataclass
class FundingSnapshot:
    """Funding rate data point."""
    timestamp: datetime
    symbol: str
    funding_rate: float  # Current rate
    predicted_rate: float  # Next predicted
    time_to_funding: timedelta
    open_interest: float
    mark_price: float
    index_price: float


class FundingRateHarvester:
    """
    Funding rate arbitrage - collect funding payments.
    
    Strategy:
    1. Monitor funding rates across perpetual pairs
    2. Short pairs with high positive funding (get paid)
    3. Long pairs with high negative funding (get paid)
    4. Delta-hedge with spot to be market-neutral
    
    Returns: Funding rate (typically 0.01-0.3% per 8h) with minimal risk
    """
    
    def __init__(self, config: FundingConfig):
        """
        Initialize funding harvester.
        
        Args:
            config: Funding configuration
        """
        self.config = config
        
        self._funding_history: Dict[str, deque] = {}  # symbol -> history
        self._positions: Dict[str, Dict] = {}  # symbol -> position info
        self._pnl_history: List[float] = []
    
    def update_funding(self, snapshot: FundingSnapshot):
        """Update with new funding snapshot."""
        symbol = snapshot.symbol
        
        if symbol not in self._funding_history:
            self._funding_history[symbol] = deque(maxlen=self.config.prediction_lookback)
        
        self._funding_history[symbol].append({
            'timestamp': snapshot.timestamp,
            'rate': snapshot.funding_rate,
            'oi': snapshot.open_interest
        })
    
    def get_opportunities(self, snapshots: List[FundingSnapshot]) -> List[Dict]:
        """
        Find funding arbitrage opportunities.
        
        Returns list of opportunities sorted by expected return.
        """
        opportunities = []
        
        for snap in snapshots:
            # High positive funding = Short perp, Long spot
            if snap.funding_rate > self.config.funding_entry_threshold:
                expected_return = snap.funding_rate - 0.0001  # Minus fees
                opportunities.append({
                    'symbol': snap.symbol,
                    'direction': 'short',
                    'funding_rate': snap.funding_rate,
                    'expected_return': expected_return,
                    'action': 'Short perp, Long spot',
                    'confidence': self._calculate_confidence(snap),
                    'time_to_funding': snap.time_to_funding
                })
            
            # High negative funding = Long perp, Short spot
            elif snap.funding_rate < -self.config.funding_entry_threshold:
                expected_return = abs(snap.funding_rate) - 0.0001
                opportunities.append({
                    'symbol': snap.symbol,
                    'direction': 'long',
                    'funding_rate': snap.funding_rate,
                    'expected_return': expected_return,
                    'action': 'Long perp, Short spot',
                    'confidence': self._calculate_confidence(snap),
                    'time_to_funding': snap.time_to_funding
                })
        
        # Sort by expected return
        opportunities.sort(key=lambda x: x['expected_return'], reverse=True)
        
        return opportunities
    
    def _calculate_confidence(self, snap: FundingSnapshot) -> float:
        """Calculate confidence in funding trade."""
        confidence = 0.5
        
        # Higher funding = more confident
        if abs(snap.funding_rate) > self.config.extreme_funding_threshold:
            confidence += 0.2
        elif abs(snap.funding_rate) > self.config.funding_entry_threshold * 2:
            confidence += 0.1
        
        # Check historical consistency
        if snap.symbol in self._funding_history:
            history = list(self._funding_history[snap.symbol])
            if len(history) > 10:
                recent_rates = [h['rate'] for h in history[-10:]]
                
                # Consistent direction = more confident
                if all(r > 0 for r in recent_rates) or all(r < 0 for r in recent_rates):
                    confidence += 0.15
                
                # Mean-reversion tendency
                avg_rate = np.mean(recent_rates)
                if abs(snap.funding_rate) > abs(avg_rate) * 1.5:
                    confidence += 0.1  # Extreme = likely to revert
        
        return min(0.9, confidence)
    
    def predict_funding(self, symbol: str) -> Dict[str, float]:
        """
        Predict next funding rate.
        
        Uses historical patterns to forecast.
        """
        if symbol not in self._funding_history:
            return {'predicted_rate': 0, 'confidence': 0}
        
        history = list(self._funding_history[symbol])
        if len(history) < 20:
            return {'predicted_rate': 0, 'confidence': 0}
        
        rates = np.array([h['rate'] for h in history])
        
        # Simple mean reversion model
        mean_rate = np.mean(rates)
        current_rate = rates[-1]
        
        # Funding tends to mean-revert
        reversion_factor = 0.3
        predicted = current_rate + reversion_factor * (mean_rate - current_rate)
        
        # Momentum component
        momentum = np.mean(rates[-5:]) - np.mean(rates[-20:-5])
        predicted += momentum * 0.2
        
        # Confidence based on prediction accuracy
        if len(rates) > 30:
            predictions = []
            actuals = []
            for i in range(30, len(rates)):
                pred = rates[i-1] + reversion_factor * (np.mean(rates[:i]) - rates[i-1])
                predictions.append(pred)
                actuals.append(rates[i])
            
            mse = np.mean((np.array(predictions) - np.array(actuals))**2)
            confidence = max(0.3, 1 - mse * 1000)
        else:
            confidence = 0.5
        
        return {
            'predicted_rate': predicted,
            'confidence': confidence,
            'mean_rate': mean_rate,
            'momentum': momentum
        }
    
    def calculate_position_pnl(self, symbol: str, entry_funding: float,
                               current_funding: float, periods: int) -> float:
        """Calculate PnL from funding collection."""
        if symbol not in self._positions:
            return 0.0
        
        pos = self._positions[symbol]
        direction = 1 if pos['direction'] == 'long' else -1
        
        # Funding collected (opposite sign = you pay, same sign = you receive)
        funding_pnl = -direction * current_funding * periods
        
        return funding_pnl


class BasisTrader:
    """
    Basis trade (Cash and Carry Arbitrage).
    
    When perp trades at premium to spot:
    - Short perp
    - Long spot
    - Earn the basis as it converges + funding
    
    This is a near risk-free return if you can hold to convergence.
    """
    
    def __init__(self, config: FundingConfig):
        """
        Initialize basis trader.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        
        self._basis_history: Dict[str, deque] = {}
        self._positions: Dict[str, Dict] = {}
    
    def calculate_basis(self, perp_price: float, spot_price: float,
                       days_to_expiry: float = 30) -> Dict[str, float]:
        """
        Calculate basis and annualized return.
        
        Args:
            perp_price: Perpetual contract price
            spot_price: Spot price
            days_to_expiry: Days for annualization
        
        Returns basis metrics.
        """
        raw_basis = perp_price - spot_price
        basis_pct = raw_basis / spot_price
        
        # Annualize
        annualized_basis = basis_pct * (365 / days_to_expiry)
        
        return {
            'raw_basis': raw_basis,
            'basis_pct': basis_pct,
            'annualized_basis': annualized_basis,
            'perp_premium': perp_price > spot_price,
            'perp_discount': perp_price < spot_price
        }
    
    def find_opportunities(self, pairs: List[Dict]) -> List[Dict]:
        """
        Find basis trade opportunities.
        
        Args:
            pairs: List of {symbol, perp_price, spot_price}
        
        Returns sorted opportunities.
        """
        opportunities = []
        
        for pair in pairs:
            basis = self.calculate_basis(
                pair['perp_price'],
                pair['spot_price']
            )
            
            # Positive basis = Short perp, Long spot
            if basis['annualized_basis'] > self.config.basis_entry_threshold:
                opportunities.append({
                    'symbol': pair['symbol'],
                    'trade': 'Short perp, Long spot',
                    'direction': 'convergence_short',
                    'basis': basis['basis_pct'],
                    'annualized_return': basis['annualized_basis'],
                    'perp_price': pair['perp_price'],
                    'spot_price': pair['spot_price']
                })
            
            # Negative basis = Long perp, Short spot
            elif basis['annualized_basis'] < -self.config.basis_entry_threshold:
                opportunities.append({
                    'symbol': pair['symbol'],
                    'trade': 'Long perp, Short spot',
                    'direction': 'convergence_long',
                    'basis': basis['basis_pct'],
                    'annualized_return': abs(basis['annualized_basis']),
                    'perp_price': pair['perp_price'],
                    'spot_price': pair['spot_price']
                })
        
        opportunities.sort(key=lambda x: x['annualized_return'], reverse=True)
        
        return opportunities
    
    def update(self, symbol: str, perp_price: float, spot_price: float):
        """Update basis tracking."""
        if symbol not in self._basis_history:
            self._basis_history[symbol] = deque(maxlen=1000)
        
        basis = self.calculate_basis(perp_price, spot_price)
        self._basis_history[symbol].append({
            'timestamp': datetime.now(),
            'basis': basis['basis_pct'],
            'perp': perp_price,
            'spot': spot_price
        })
    
    def analyze_basis_dynamics(self, symbol: str) -> Dict:
        """Analyze basis behavior for a symbol."""
        if symbol not in self._basis_history:
            return {}
        
        history = list(self._basis_history[symbol])
        if len(history) < 20:
            return {}
        
        basis_series = np.array([h['basis'] for h in history])
        
        return {
            'current_basis': basis_series[-1],
            'mean_basis': np.mean(basis_series),
            'std_basis': np.std(basis_series),
            'zscore': (basis_series[-1] - np.mean(basis_series)) / (np.std(basis_series) + 1e-10),
            'max_basis': np.max(basis_series),
            'min_basis': np.min(basis_series),
            'trend': np.polyfit(range(min(50, len(basis_series))), basis_series[-50:], 1)[0]
        }


class FundingPredictor:
    """
    ML-based funding rate prediction.
    
    Predicts funding rates using:
    - Historical funding patterns
    - Open interest
    - Long/short ratio
    - Price momentum
    """
    
    def __init__(self, lookback: int = 100):
        """
        Initialize funding predictor.
        
        Args:
            lookback: Historical periods to use
        """
        self.lookback = lookback
        
        self._data: Dict[str, List[Dict]] = {}
        self._model = None
    
    def add_observation(self, symbol: str, funding_rate: float,
                       open_interest: float, long_short_ratio: float,
                       price_change_24h: float):
        """Add new observation for training."""
        if symbol not in self._data:
            self._data[symbol] = []
        
        self._data[symbol].append({
            'timestamp': datetime.now(),
            'funding_rate': funding_rate,
            'open_interest': open_interest,
            'long_short_ratio': long_short_ratio,
            'price_change': price_change_24h
        })
    
    def train(self, symbol: str):
        """Train prediction model for symbol."""
        if symbol not in self._data or len(self._data[symbol]) < 50:
            return False
        
        data = self._data[symbol]
        
        # Create features
        X = []
        y = []
        
        for i in range(20, len(data) - 1):
            # Features
            recent = data[i-20:i]
            
            features = [
                np.mean([d['funding_rate'] for d in recent]),
                np.std([d['funding_rate'] for d in recent]),
                data[i]['open_interest'] / np.mean([d['open_interest'] for d in recent]),
                data[i]['long_short_ratio'],
                data[i]['price_change']
            ]
            
            X.append(features)
            y.append(data[i + 1]['funding_rate'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Simple linear regression (in production use more sophisticated model)
        try:
            from sklearn.linear_model import Ridge
            self._model = Ridge(alpha=0.1)
            self._model.fit(X, y)
            return True
        except ImportError:
            # Fallback to mean reversion model
            self._model = None
            return False
    
    def predict(self, symbol: str) -> Dict[str, float]:
        """Predict next funding rate."""
        if symbol not in self._data or len(self._data[symbol]) < 20:
            return {'predicted': 0, 'confidence': 0}
        
        data = self._data[symbol]
        recent = data[-20:]
        
        # Create features
        features = np.array([[
            np.mean([d['funding_rate'] for d in recent]),
            np.std([d['funding_rate'] for d in recent]),
            data[-1]['open_interest'] / np.mean([d['open_interest'] for d in recent]),
            data[-1]['long_short_ratio'],
            data[-1]['price_change']
        ]])
        
        if self._model is not None:
            predicted = self._model.predict(features)[0]
            # R² as confidence proxy
            confidence = 0.6
        else:
            # Mean reversion fallback
            mean_rate = np.mean([d['funding_rate'] for d in recent])
            current = data[-1]['funding_rate']
            predicted = current + 0.3 * (mean_rate - current)
            confidence = 0.4
        
        return {
            'predicted': predicted,
            'confidence': confidence,
            'mean_rate': np.mean([d['funding_rate'] for d in recent]),
            'current_rate': data[-1]['funding_rate']
        }


class FundingArbitrageStrategy(BaseStrategy):
    """
    Complete funding rate arbitrage strategy.
    
    Combines:
    1. Funding rate harvesting
    2. Basis trading
    3. Funding prediction
    
    For market-neutral returns from funding.
    """
    
    def __init__(self, config: Optional[FundingConfig] = None):
        super().__init__(config or FundingConfig())
        self.config: FundingConfig = self.config
        
        self.funding_harvester = FundingRateHarvester(self.config)
        self.basis_trader = BasisTrader(self.config)
        self.funding_predictor = FundingPredictor()
        
        self._last_funding_data: Dict[str, FundingSnapshot] = {}
    
    def update(self, data: pd.DataFrame):
        """Update with price data."""
        self.current_bar = len(data) - 1
        
        # In production, this would be replaced with real funding data
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        # Simulate funding data (in production use exchange API)
        simulated_snapshot = FundingSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            funding_rate=np.random.normal(0.0001, 0.0005),  # Simulate
            predicted_rate=0,
            time_to_funding=timedelta(hours=np.random.uniform(0, 8)),
            open_interest=np.random.uniform(100000000, 500000000),
            mark_price=close,
            index_price=close * (1 + np.random.uniform(-0.001, 0.001))
        )
        
        self._last_funding_data[symbol] = simulated_snapshot
        self.funding_harvester.update_funding(simulated_snapshot)
        
        # Update basis trader
        self.basis_trader.update(
            symbol,
            simulated_snapshot.mark_price,
            simulated_snapshot.index_price
        )
        
        # Update predictor
        self.funding_predictor.add_observation(
            symbol,
            simulated_snapshot.funding_rate,
            simulated_snapshot.open_interest,
            1.0,  # Long/short ratio (would come from API)
            data['close'].pct_change(24).iloc[-1] if len(data) > 24 else 0
        )
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate funding arbitrage signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        close = data['close'].iloc[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        if symbol not in self._last_funding_data:
            return None
        
        snap = self._last_funding_data[symbol]
        
        signal_type = None
        reason = ""
        strength = 0.0
        confidence = 0.0
        
        # Check for funding opportunity
        opportunities = self.funding_harvester.get_opportunities([snap])
        
        if opportunities:
            best = opportunities[0]
            
            if best['expected_return'] > self.config.funding_entry_threshold:
                if best['direction'] == 'short':
                    signal_type = SignalType.SHORT  # Short perp (will long spot to hedge)
                    reason = f"Funding arb: {best['funding_rate']*100:.3f}% funding, {best['action']}"
                else:
                    signal_type = SignalType.LONG  # Long perp (will short spot to hedge)
                    reason = f"Funding arb: {best['funding_rate']*100:.3f}% funding, {best['action']}"
                
                strength = min(1.0, best['expected_return'] / 0.001)
                confidence = best['confidence']
        
        # Check basis opportunity if no funding signal
        if signal_type is None:
            basis_opps = self.basis_trader.find_opportunities([{
                'symbol': symbol,
                'perp_price': snap.mark_price,
                'spot_price': snap.index_price
            }])
            
            if basis_opps:
                best_basis = basis_opps[0]
                
                if best_basis['annualized_return'] > self.config.basis_entry_threshold:
                    if best_basis['direction'] == 'convergence_short':
                        signal_type = SignalType.SHORT
                    else:
                        signal_type = SignalType.LONG
                    
                    reason = f"Basis arb: {best_basis['basis']*100:.3f}% basis, {best_basis['annualized_return']*100:.1f}% ann."
                    strength = min(1.0, best_basis['annualized_return'] / 0.1)
                    confidence = 0.7
        
        if signal_type is None:
            return None
        
        if confidence < self.config.min_confidence:
            return None
        
        return self._create_signal(
            signal_type=signal_type,
            price=close,
            symbol=symbol,
            reason=reason,
            strength=strength,
            confidence=confidence,
            atr=close * 0.001,  # Tight stops for arb
            funding_rate=snap.funding_rate,
            is_arbitrage=True,
            hedge_required=True
        )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FUNDING RATE ARBITRAGE STRATEGIES")
    print("="*70)
    
    config = FundingConfig()
    
    print("\n1. FUNDING RATE HARVESTING")
    print("-" * 50)
    
    harvester = FundingRateHarvester(config)
    
    # Simulate funding snapshots
    snapshots = []
    for i, symbol in enumerate(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']):
        # Random funding rates with some high
        rate = np.random.normal(0.0001, 0.0003)
        if np.random.random() > 0.7:  # 30% chance of high funding
            rate = np.random.uniform(0.0005, 0.002) * (1 if np.random.random() > 0.5 else -1)
        
        snap = FundingSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            funding_rate=rate,
            predicted_rate=0,
            time_to_funding=timedelta(hours=np.random.uniform(1, 8)),
            open_interest=np.random.uniform(100e6, 500e6),
            mark_price=50000 if 'BTC' in symbol else 3000,
            index_price=50000 if 'BTC' in symbol else 3000
        )
        snapshots.append(snap)
        harvester.update_funding(snap)
    
    opportunities = harvester.get_opportunities(snapshots)
    
    print("   Current opportunities:")
    for opp in opportunities[:3]:
        print(f"   {opp['symbol']}: {opp['funding_rate']*100:.4f}% → {opp['action']}")
        print(f"      Expected return: {opp['expected_return']*100:.4f}%")
    
    print("\n2. BASIS TRADING")
    print("-" * 50)
    
    basis_trader = BasisTrader(config)
    
    pairs = [
        {'symbol': 'BTCUSDT', 'perp_price': 50100, 'spot_price': 50000},
        {'symbol': 'ETHUSDT', 'perp_price': 3015, 'spot_price': 3000},
        {'symbol': 'SOLUSDT', 'perp_price': 98, 'spot_price': 100},  # Discount
    ]
    
    basis_opps = basis_trader.find_opportunities(pairs)
    
    print("   Basis opportunities:")
    for opp in basis_opps:
        print(f"   {opp['symbol']}: Basis={opp['basis']*100:.3f}%")
        print(f"      {opp['trade']}")
        print(f"      Annualized: {opp['annualized_return']*100:.1f}%")
    
    print("\n3. FUNDING PREDICTION")
    print("-" * 50)
    
    predictor = FundingPredictor()
    
    # Add historical data
    for i in range(100):
        predictor.add_observation(
            'BTCUSDT',
            funding_rate=0.0001 + np.sin(i/10) * 0.0003 + np.random.normal(0, 0.0001),
            open_interest=200e9 + np.random.normal(0, 10e9),
            long_short_ratio=1.0 + np.random.normal(0, 0.1),
            price_change_24h=np.random.normal(0, 0.02)
        )
    
    prediction = predictor.predict('BTCUSDT')
    print(f"   BTCUSDT Funding Prediction:")
    print(f"      Current: {prediction['current_rate']*100:.4f}%")
    print(f"      Predicted: {prediction['predicted']*100:.4f}%")
    print(f"      Mean: {prediction['mean_rate']*100:.4f}%")
    print(f"      Confidence: {prediction['confidence']:.2f}")
    
    print("\n4. COMPLETE STRATEGY")
    print("-" * 50)
    
    np.random.seed(42)
    n = 100
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    
    data = pd.DataFrame({
        'open': prices - 10,
        'high': prices + 50,
        'low': prices - 50,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n),
        'symbol': 'BTCUSDT'
    })
    
    strat = FundingArbitrageStrategy(config)
    
    signals = []
    for i in range(20, len(data)):
        strat.current_bar = i
        strat.update(data.iloc[:i+1])
        signal = strat.generate_signal(data.iloc[:i+1])
        if signal:
            signals.append(signal)
    
    print(f"   Signals generated: {len(signals)}")
    if signals:
        print(f"   Last signal: {signals[-1].reason}")
    
    print("\n" + "="*70)
    print("FUNDING ARBITRAGE KEY INSIGHTS")
    print("="*70)
    print("""
1. THIS IS ONE OF THE SAFEST STRATEGIES IN CRYPTO
   - Market neutral when properly hedged
   - Collect funding regardless of price direction
   - Typical returns: 10-50% APY with low risk
   
2. HOW FUNDING WORKS
   - Every 8 hours, longs pay shorts (or vice versa)
   - Rate based on premium/discount to index
   - High positive rate = perp at premium = shorts get paid
   - High negative rate = perp at discount = longs get paid
   
3. THE TRADE EXECUTION
   Positive Funding (common in bull markets):
   - Short perp (receive funding)
   - Long spot (hedge)
   - Net position: Market neutral, positive carry
   
   Negative Funding (common in bear markets):
   - Long perp (receive funding)
   - Short spot (hedge)
   - Net position: Market neutral, positive carry
   
4. KEY METRICS TO MONITOR
   - Funding rate (>0.03% = attractive)
   - Basis (perp vs spot price difference)
   - Open interest (higher = more reliable funding)
   - Time to funding (enter 2-4 hours before)
   
5. RISKS
   - Execution risk (slippage when hedging)
   - Margin requirements
   - Funding can flip direction
   - Liquidation risk if unhedged
   
6. BEST PAIRS FOR FUNDING ARB
   - BTCUSDT (most liquid, stable funding)
   - ETHUSDT (second most liquid)
   - Popular alts in bull markets (high funding)
   
7. REALISTIC RETURNS
   - Conservative: 15-25% APY
   - Aggressive: 30-50% APY
   - With leverage (risky): 50-100%+ APY
""")
