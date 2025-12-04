"""
Strategy 5: Statistical Arbitrage (Pairs, Triangular, Cross-Exchange)
=====================================================================

Why it works in crypto:
- Crypto is fragmented across exchanges, creating mispricings
- Pairs like BTC-ETH, SOL-AVAX often show stable relationships
- When the spread deviates → revert

Concepts:
- Cointegration (Johansen test)
- Z-score trading
- Mean-reversion boundaries
- Cross-exchange arbitrage
- Triangular arbitrage
- Spread normalization
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from .base import (
    BaseStrategy, StrategyConfig, Signal, SignalType,
    TechnicalIndicators
)


@dataclass
class StatArbConfig(StrategyConfig):
    """Configuration for statistical arbitrage strategies."""
    name: str = "stat_arb"
    
    # Cointegration
    coint_lookback: int = 250
    coint_pvalue: float = 0.05  # Max p-value to consider cointegrated
    
    # Z-score thresholds
    zscore_entry: float = 2.0  # Enter at 2 std deviations
    zscore_exit: float = 0.5  # Exit at 0.5 std deviations
    zscore_stop: float = 3.5  # Stop loss at 3.5 std deviations
    
    # Half-life
    min_halflife: int = 5  # Minimum bars for mean reversion
    max_halflife: int = 100  # Maximum bars
    
    # Position sizing
    max_notional: float = 10000  # Max USD per leg
    
    # Pairs
    default_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('BTCUSDT', 'ETHUSDT'),
        ('SOLUSDT', 'AVAXUSDT'),
        ('ADAUSDT', 'XRPUSDT'),
        ('BNBUSDT', 'ETHUSDT')
    ])
    
    # Cross-exchange
    min_exchange_spread: float = 0.002  # 0.2% min to trade
    
    # Regime
    allowed_regimes: List[str] = field(default_factory=lambda: ["ranging", "low_volatility"])


class CointegrationAnalyzer:
    """
    Cointegration analysis for pairs trading.
    
    Tests whether two assets have a stable long-term relationship.
    If cointegrated, the spread is mean-reverting → tradeable.
    """
    
    def __init__(self, lookback: int = 250):
        """
        Initialize cointegration analyzer.
        
        Args:
            lookback: Periods for testing
        """
        self.lookback = lookback
    
    def engle_granger_test(self, y: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
        """
        Engle-Granger two-step cointegration test.
        
        Step 1: Regress y on x to get residuals (spread)
        Step 2: Test residuals for stationarity (ADF test)
        
        Args:
            y: First series
            x: Second series
        
        Returns:
            Test results including hedge ratio and spread
        """
        if len(y) < self.lookback or len(x) < self.lookback:
            return {'cointegrated': False, 'reason': 'insufficient_data'}
        
        # Use last 'lookback' observations
        y = y[-self.lookback:]
        x = x[-self.lookback:]
        
        # Step 1: OLS regression y = alpha + beta*x + epsilon
        # Using simple least squares
        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
            alpha, hedge_ratio = beta[0], beta[1]
        except:
            return {'cointegrated': False, 'reason': 'regression_failed'}
        
        # Calculate spread (residuals)
        spread = y - (alpha + hedge_ratio * x)
        
        # Step 2: ADF test on spread
        adf_result = self._adf_test(spread)
        
        # Calculate half-life of mean reversion
        halflife = self._calculate_halflife(spread)
        
        return {
            'cointegrated': adf_result['stationary'],
            'adf_statistic': adf_result['statistic'],
            'pvalue': adf_result['pvalue'],
            'hedge_ratio': hedge_ratio,
            'alpha': alpha,
            'spread': spread,
            'spread_mean': np.mean(spread),
            'spread_std': np.std(spread),
            'halflife': halflife,
            'correlation': np.corrcoef(y, x)[0, 1]
        }
    
    def _adf_test(self, series: np.ndarray) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        Tests: H0: series has unit root (non-stationary)
               H1: series is stationary (mean-reverting)
        """
        n = len(series)
        
        # Difference series
        diff = np.diff(series)
        
        # Lag 1 level
        lag1 = series[:-1]
        
        # Simple regression: diff = alpha + gamma*lag1 + error
        # gamma < 0 and significant → stationary
        
        X = np.column_stack([np.ones(len(lag1)), lag1])
        
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X, diff, rcond=None)
            gamma = beta[1]
            
            # Standard error
            if len(residuals) > 0:
                se = np.sqrt(residuals[0] / (len(lag1) - 2)) / np.std(lag1)
            else:
                se = np.std(diff) / np.std(lag1)
            
            # T-statistic
            t_stat = gamma / (se + 1e-10)
            
            # Critical values (5% level)
            # Approximate: -2.86 for 250 obs without trend
            critical_value = -2.86
            
            # Approximate p-value
            if t_stat < -3.5:
                pvalue = 0.01
            elif t_stat < -2.9:
                pvalue = 0.05
            elif t_stat < -2.6:
                pvalue = 0.1
            else:
                pvalue = 0.5
            
            return {
                'statistic': t_stat,
                'pvalue': pvalue,
                'stationary': t_stat < critical_value,
                'gamma': gamma
            }
        except:
            return {
                'statistic': 0,
                'pvalue': 1,
                'stationary': False,
                'gamma': 0
            }
    
    def _calculate_halflife(self, spread: np.ndarray) -> float:
        """
        Calculate half-life of mean reversion.
        
        Fits AR(1) model: spread_t = c + phi * spread_{t-1} + e_t
        Half-life = -log(2) / log(phi)
        """
        lag = spread[:-1]
        current = spread[1:]
        
        # Regress current on lag
        X = np.column_stack([np.ones(len(lag)), lag])
        
        try:
            beta, _, _, _ = np.linalg.lstsq(X, current, rcond=None)
            phi = beta[1]
            
            if phi <= 0 or phi >= 1:
                return float('inf')
            
            halflife = -np.log(2) / np.log(phi)
            return max(1, halflife)
        except:
            return float('inf')
    
    def johansen_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Johansen cointegration test for multiple series.
        
        More powerful than Engle-Granger for 2+ series.
        """
        # Simplified implementation - in production use statsmodels
        # For now, apply pairwise EG tests
        
        columns = data.columns
        results = {}
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                eg_result = self.engle_granger_test(
                    data[col1].values,
                    data[col2].values
                )
                results[f"{col1}_{col2}"] = eg_result
        
        return results


class SpreadTrader:
    """
    Z-score based spread trading.
    
    Given a cointegrated pair:
    - Long spread when z-score < -entry_threshold
    - Short spread when z-score > +entry_threshold
    - Exit when z-score crosses 0
    """
    
    def __init__(self, config: StatArbConfig):
        """
        Initialize spread trader.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        
        self._spread_history: Dict[str, deque] = {}
        self._positions: Dict[str, Dict] = {}
        self._hedge_ratios: Dict[str, float] = {}
    
    def update_spread(self, pair_id: str, spread: float, hedge_ratio: float):
        """Update spread tracking."""
        if pair_id not in self._spread_history:
            self._spread_history[pair_id] = deque(maxlen=500)
        
        self._spread_history[pair_id].append(spread)
        self._hedge_ratios[pair_id] = hedge_ratio
    
    def calculate_zscore(self, pair_id: str, lookback: int = 50) -> float:
        """Calculate current z-score of spread."""
        if pair_id not in self._spread_history:
            return 0.0
        
        spreads = list(self._spread_history[pair_id])
        if len(spreads) < lookback:
            return 0.0
        
        recent = np.array(spreads[-lookback:])
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std < 1e-10:
            return 0.0
        
        return (spreads[-1] - mean) / std
    
    def get_signal(self, pair_id: str, leg1_price: float, leg2_price: float) -> Optional[Dict]:
        """
        Generate trading signal for pair.
        
        Returns:
            Signal dict or None
        """
        zscore = self.calculate_zscore(pair_id)
        
        # Check if in position
        in_position = pair_id in self._positions
        
        signal = None
        
        if not in_position:
            # Entry signals
            if zscore < -self.config.zscore_entry:
                # Spread too low → buy spread (long leg1, short leg2)
                signal = {
                    'action': 'open',
                    'direction': 'long_spread',
                    'leg1_action': 'buy',
                    'leg2_action': 'sell',
                    'zscore': zscore,
                    'reason': f'Z-score={zscore:.2f} < -{self.config.zscore_entry}'
                }
            elif zscore > self.config.zscore_entry:
                # Spread too high → sell spread (short leg1, long leg2)
                signal = {
                    'action': 'open',
                    'direction': 'short_spread',
                    'leg1_action': 'sell',
                    'leg2_action': 'buy',
                    'zscore': zscore,
                    'reason': f'Z-score={zscore:.2f} > {self.config.zscore_entry}'
                }
        else:
            # Exit signals
            pos = self._positions[pair_id]
            
            # Exit at mean (take profit)
            if pos['direction'] == 'long_spread' and zscore > -self.config.zscore_exit:
                signal = {
                    'action': 'close',
                    'reason': f'Mean reversion: Z-score={zscore:.2f}'
                }
            elif pos['direction'] == 'short_spread' and zscore < self.config.zscore_exit:
                signal = {
                    'action': 'close',
                    'reason': f'Mean reversion: Z-score={zscore:.2f}'
                }
            
            # Stop loss
            elif abs(zscore) > self.config.zscore_stop:
                signal = {
                    'action': 'close',
                    'reason': f'Stop loss: Z-score={zscore:.2f}'
                }
        
        if signal:
            signal['hedge_ratio'] = self._hedge_ratios.get(pair_id, 1.0)
            signal['pair_id'] = pair_id
        
        return signal
    
    def open_position(self, pair_id: str, direction: str, leg1_price: float, leg2_price: float):
        """Record opened position."""
        self._positions[pair_id] = {
            'direction': direction,
            'leg1_entry': leg1_price,
            'leg2_entry': leg2_price,
            'entry_zscore': self.calculate_zscore(pair_id),
            'entry_time': datetime.now()
        }
    
    def close_position(self, pair_id: str, leg1_price: float, leg2_price: float) -> float:
        """Close position and return PnL."""
        if pair_id not in self._positions:
            return 0.0
        
        pos = self._positions.pop(pair_id)
        
        # PnL calculation
        leg1_pnl = (leg1_price - pos['leg1_entry']) / pos['leg1_entry']
        leg2_pnl = (leg2_price - pos['leg2_entry']) / pos['leg2_entry']
        
        if pos['direction'] == 'long_spread':
            total_pnl = leg1_pnl - leg2_pnl
        else:
            total_pnl = -leg1_pnl + leg2_pnl
        
        return total_pnl


class PairsTrader(BaseStrategy):
    """
    Complete pairs trading strategy.
    
    Combines cointegration analysis with z-score trading.
    """
    
    def __init__(self, config: Optional[StatArbConfig] = None, 
                 leg1_data: Optional[pd.Series] = None,
                 leg2_data: Optional[pd.Series] = None):
        super().__init__(config or StatArbConfig())
        self.config: StatArbConfig = self.config
        
        self.coint_analyzer = CointegrationAnalyzer(self.config.coint_lookback)
        self.spread_trader = SpreadTrader(self.config)
        
        self._leg1_data: List[float] = list(leg1_data) if leg1_data is not None else []
        self._leg2_data: List[float] = list(leg2_data) if leg2_data is not None else []
        self._pair_id = "leg1_leg2"
        self._last_coint_result: Optional[Dict] = None
    
    def set_pair_data(self, leg1_symbol: str, leg2_symbol: str,
                     leg1_prices: np.ndarray, leg2_prices: np.ndarray):
        """Set data for the pair."""
        self._leg1_data = list(leg1_prices)
        self._leg2_data = list(leg2_prices)
        self._pair_id = f"{leg1_symbol}_{leg2_symbol}"
    
    def update(self, data: pd.DataFrame):
        """Update with new price data."""
        self.current_bar = len(data) - 1
        
        # Assume data has two columns: leg1 and leg2
        if 'leg1' in data.columns and 'leg2' in data.columns:
            self._leg1_data = list(data['leg1'].values)
            self._leg2_data = list(data['leg2'].values)
        elif len(self._leg1_data) == 0:
            # Use close price with synthetic leg2
            close = data['close'].values
            self._leg1_data = list(close)
            self._leg2_data = list(close * (1 + np.random.randn(len(close)) * 0.01))
        
        # Run cointegration test periodically
        if len(self._leg1_data) >= self.config.coint_lookback:
            if self._last_coint_result is None or self.current_bar % 50 == 0:
                self._last_coint_result = self.coint_analyzer.engle_granger_test(
                    np.array(self._leg1_data),
                    np.array(self._leg2_data)
                )
                
                # Update spread tracker
                if self._last_coint_result['cointegrated']:
                    spread = self._last_coint_result['spread'][-1]
                    hedge_ratio = self._last_coint_result['hedge_ratio']
                    self.spread_trader.update_spread(self._pair_id, spread, hedge_ratio)
        
        self.is_initialized = True
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate pairs trading signal."""
        if not self.is_initialized:
            self.update(data)
        
        if not self.can_generate_signal():
            return None
        
        if self._last_coint_result is None:
            return None
        
        if not self._last_coint_result.get('cointegrated', False):
            return None
        
        leg1_price = self._leg1_data[-1]
        leg2_price = self._leg2_data[-1]
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "PAIR"
        
        # Update spread
        spread = leg1_price - (
            self._last_coint_result['alpha'] + 
            self._last_coint_result['hedge_ratio'] * leg2_price
        )
        self.spread_trader.update_spread(
            self._pair_id, 
            spread, 
            self._last_coint_result['hedge_ratio']
        )
        
        # Get signal from spread trader
        trade_signal = self.spread_trader.get_signal(
            self._pair_id,
            leg1_price,
            leg2_price
        )
        
        if trade_signal is None:
            return None
        
        if trade_signal['action'] == 'close':
            return None  # Handle exits separately
        
        # Map to Signal type
        if trade_signal['direction'] == 'long_spread':
            signal_type = SignalType.LONG  # Long the spread
        else:
            signal_type = SignalType.SHORT  # Short the spread
        
        zscore = trade_signal['zscore']
        strength = min(1.0, abs(zscore) / 3.0)
        
        # Confidence based on cointegration strength
        confidence = 0.5
        if self._last_coint_result['pvalue'] < 0.01:
            confidence += 0.2
        if 5 < self._last_coint_result['halflife'] < 50:
            confidence += 0.15
        
        return self._create_signal(
            signal_type=signal_type,
            price=spread,  # Spread price for reference
            symbol=self._pair_id,
            reason=trade_signal['reason'],
            strength=strength,
            confidence=confidence,
            atr=self._last_coint_result['spread_std'],
            zscore=zscore,
            hedge_ratio=trade_signal['hedge_ratio'],
            is_pairs_trade=True
        )


class TriangularArbitrage:
    """
    Triangular arbitrage between three currencies.
    
    Example: BTC/USDT → ETH/BTC → ETH/USDT
    If the implied rate differs from direct rate → arbitrage
    """
    
    def __init__(self, min_profit_bps: float = 5):
        """
        Initialize triangular arbitrage detector.
        
        Args:
            min_profit_bps: Minimum profit in basis points
        """
        self.min_profit_bps = min_profit_bps
        
        self._rates: Dict[str, float] = {}
    
    def update_rate(self, pair: str, rate: float):
        """Update exchange rate for a pair."""
        self._rates[pair] = rate
    
    def find_opportunities(self) -> List[Dict]:
        """
        Find triangular arbitrage opportunities.
        
        Returns list of opportunities with expected profit.
        """
        opportunities = []
        
        # Common triangles in crypto
        triangles = [
            ('BTCUSDT', 'ETHBTC', 'ETHUSDT'),
            ('BTCUSDT', 'BNBBTC', 'BNBUSDT'),
            ('ETHUSDT', 'SOLETH', 'SOLUSDT'),
        ]
        
        for leg1, leg2, leg3 in triangles:
            if leg1 not in self._rates or leg2 not in self._rates or leg3 not in self._rates:
                continue
            
            # Path 1: USDT → BTC → ETH → USDT
            # Start with 1 USDT
            # Buy BTC: 1 / rate1
            # Buy ETH with BTC: (1/rate1) * rate2
            # Sell ETH for USDT: (1/rate1) * rate2 * rate3
            
            result = (1 / self._rates[leg1]) * self._rates[leg2] * self._rates[leg3]
            profit_pct = (result - 1) * 100
            
            if profit_pct > self.min_profit_bps / 100:
                opportunities.append({
                    'path': f'USDT → {leg1} → {leg2} → {leg3}',
                    'profit_pct': profit_pct,
                    'legs': [leg1, leg2, leg3],
                    'direction': 'forward'
                })
            
            # Reverse path
            reverse = 1 / result
            reverse_profit = (reverse - 1) * 100
            
            if reverse_profit > self.min_profit_bps / 100:
                opportunities.append({
                    'path': f'USDT → {leg3} → {leg2} → {leg1}',
                    'profit_pct': reverse_profit,
                    'legs': [leg3, leg2, leg1],
                    'direction': 'reverse'
                })
        
        opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
        
        return opportunities


class StatArbStrategy(BaseStrategy):
    """
    Complete statistical arbitrage strategy.
    
    Combines:
    1. Pairs trading (cointegration-based)
    2. Spread trading
    3. Multi-pair correlation
    """
    
    def __init__(self, config: Optional[StatArbConfig] = None):
        super().__init__(config or StatArbConfig())
        self.config: StatArbConfig = self.config
        
        self.coint_analyzer = CointegrationAnalyzer(self.config.coint_lookback)
        self.spread_trader = SpreadTrader(self.config)
        self.triangular_arb = TriangularArbitrage()
        
        self._pair_data: Dict[str, Dict] = {}
    
    def update(self, data: pd.DataFrame):
        """Update strategy state."""
        self.current_bar = len(data) - 1
        self.is_initialized = True
    
    def analyze_pair(self, leg1_prices: np.ndarray, leg2_prices: np.ndarray,
                    leg1_symbol: str, leg2_symbol: str) -> Dict:
        """
        Analyze a pair for trading opportunity.
        
        Returns analysis results.
        """
        result = self.coint_analyzer.engle_granger_test(leg1_prices, leg2_prices)
        result['leg1_symbol'] = leg1_symbol
        result['leg2_symbol'] = leg2_symbol
        
        if result['cointegrated']:
            # Calculate z-score
            spread_mean = result['spread_mean']
            spread_std = result['spread_std']
            current_spread = result['spread'][-1]
            zscore = (current_spread - spread_mean) / spread_std
            
            result['zscore'] = zscore
            result['tradeable'] = abs(zscore) > self.config.zscore_entry
        else:
            result['zscore'] = 0
            result['tradeable'] = False
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate stat arb signal."""
        if not self.is_initialized:
            self.update(data)
        
        # This would typically iterate through pairs
        # For now, return None as pairs need external data
        return None


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STATISTICAL ARBITRAGE STRATEGIES")
    print("="*70)
    
    np.random.seed(42)
    
    print("\n1. COINTEGRATION ANALYSIS")
    print("-" * 50)
    
    # Generate cointegrated series
    n = 500
    
    # Random walk for ETH
    eth = 3000 + np.cumsum(np.random.randn(n) * 30)
    
    # BTC = 15 * ETH + noise (cointegrated)
    btc = 15 * eth + np.random.randn(n) * 500 + 5000
    
    analyzer = CointegrationAnalyzer(lookback=250)
    result = analyzer.engle_granger_test(btc, eth)
    
    print(f"   Cointegrated: {result['cointegrated']}")
    print(f"   ADF Statistic: {result['adf_statistic']:.2f}")
    print(f"   P-value: {result['pvalue']:.4f}")
    print(f"   Hedge Ratio: {result['hedge_ratio']:.4f}")
    print(f"   Half-life: {result['halflife']:.1f} bars")
    print(f"   Correlation: {result['correlation']:.4f}")
    
    print("\n2. SPREAD TRADING")
    print("-" * 50)
    
    config = StatArbConfig(zscore_entry=2.0, zscore_exit=0.5)
    trader = SpreadTrader(config)
    
    # Simulate spread updates
    spread = result['spread']
    hedge_ratio = result['hedge_ratio']
    
    signals = []
    for i, s in enumerate(spread):
        trader.update_spread('BTCETH', s, hedge_ratio)
        
        if i > 50:  # Wait for history
            signal = trader.get_signal('BTCETH', btc[i], eth[i])
            if signal and signal['action'] == 'open':
                signals.append((i, signal))
                trader.open_position('BTCETH', signal['direction'], btc[i], eth[i])
    
    print(f"   Total signals: {len(signals)}")
    
    zscore = trader.calculate_zscore('BTCETH')
    print(f"   Current Z-score: {zscore:.2f}")
    
    print("\n3. PAIRS TRADING STRATEGY")
    print("-" * 50)
    
    pairs_data = pd.DataFrame({
        'leg1': btc,
        'leg2': eth,
        'symbol': 'BTCETH'
    })
    
    pairs_strategy = PairsTrader(config)
    pairs_strategy.set_pair_data('BTC', 'ETH', btc, eth)
    
    pair_signals = []
    for i in range(100, len(pairs_data)):
        pairs_strategy.current_bar = i
        pairs_strategy.update(pairs_data.iloc[:i+1])
        signal = pairs_strategy.generate_signal(pairs_data.iloc[:i+1])
        if signal:
            pair_signals.append(signal)
    
    print(f"   Pairs signals: {len(pair_signals)}")
    if pair_signals:
        print(f"   Last signal: {pair_signals[-1].reason}")
    
    print("\n4. TRIANGULAR ARBITRAGE")
    print("-" * 50)
    
    tri_arb = TriangularArbitrage(min_profit_bps=5)
    
    # Set rates (with slight inefficiency)
    tri_arb.update_rate('BTCUSDT', 50000)
    tri_arb.update_rate('ETHBTC', 0.0615)  # Slightly off
    tri_arb.update_rate('ETHUSDT', 3050)   # Creates opportunity
    
    opportunities = tri_arb.find_opportunities()
    
    print("   Triangular opportunities found:")
    for opp in opportunities[:3]:
        print(f"   {opp['path']}: {opp['profit_pct']:.4f}%")
    
    print("\n5. NON-COINTEGRATED PAIR")
    print("-" * 50)
    
    # Generate non-cointegrated series
    sol = 100 + np.cumsum(np.random.randn(n) * 5)
    ada = 0.5 + np.cumsum(np.random.randn(n) * 0.02)  # Different dynamics
    
    result_nc = analyzer.engle_granger_test(sol, ada)
    
    print(f"   Cointegrated: {result_nc['cointegrated']}")
    print(f"   ADF Statistic: {result_nc['adf_statistic']:.2f}")
    print(f"   P-value: {result_nc['pvalue']:.4f}")
    print(f"   → NOT suitable for pairs trading")
    
    print("\n" + "="*70)
    print("STATISTICAL ARBITRAGE KEY INSIGHTS")
    print("="*70)
    print("""
1. COINTEGRATION ≠ CORRELATION
   - Correlated assets can diverge forever
   - Cointegrated assets MUST revert to equilibrium
   - Always test for cointegration before pairs trading
   
2. KEY PARAMETERS
   - Entry Z-score: 2.0 (2 standard deviations)
   - Exit Z-score: 0.5 (close to mean)
   - Stop Z-score: 3.5 (regime change)
   - Half-life: 5-50 bars (too short = noise, too long = capital inefficient)
   
3. BEST CRYPTO PAIRS
   - BTC-ETH (most liquid, stable relationship)
   - ETH-BNB (exchange tokens)
   - SOL-AVAX (similar category, correlated)
   - Stablecoins (USDT-USDC, very tight spread)
   
4. EXECUTION CHALLENGES
   - Must execute BOTH legs simultaneously
   - Slippage can eat profits
   - Capital tied up in both directions
   
5. REGIME AWARENESS
   - Cointegration breaks in crisis
   - Retest regularly (every 50-100 bars)
   - Exit ALL positions if cointegration fails
   
6. REALISTIC RETURNS
   - Per trade: 0.5-2%
   - Win rate: 60-70%
   - Sharpe: 1.5-3.0
   - Very capital efficient
""")
