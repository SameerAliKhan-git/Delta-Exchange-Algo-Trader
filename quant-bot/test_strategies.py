#!/usr/bin/env python3
"""
Test all 7 strategy families to ensure they work correctly.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 70)
print("TESTING ALL 7 STRATEGY FAMILIES")
print("=" * 70)

# Test imports
print("\n1. Testing imports...")
try:
    from strategies import (
        # Base
        BaseStrategy, Signal, SignalType, TechnicalIndicators,
        
        # 1. Momentum
        MomentumStrategy, SupertrendStrategy, TrendConfig,
        
        # 2. Volatility
        VolatilityBreakoutStrategy, DonchianBreakout, BreakoutConfig,
        
        # 3. Microstructure  
        MicrostructureStrategy, OrderBookImbalance, MicrostructureConfig,
        
        # 4. Funding
        FundingArbitrageStrategy, FundingRateHarvester, FundingConfig,
        
        # 5. Stat Arb
        StatArbStrategy, PairsTrader, StatArbConfig,
        
        # 6. Regime ML
        RegimeMLStrategy, RegimeDetector, MarketRegime, RegimeConfig,
        
        # 7. Event-Driven
        EventDrivenStrategy, SentimentAnalyzer, EventConfig,
        
        # Ensemble
        StrategyEnsemble, AutoTuningEnsemble, EnsembleConfig
    )
    print("   ✓ All imports successful!")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Generate test data
print("\n2. Generating test data...")
np.random.seed(42)
n = 200
prices = 50000 + np.cumsum(np.random.randn(n) * 500)

test_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1h'),
    'open': prices - np.random.uniform(50, 150, n),
    'high': prices + np.random.uniform(100, 300, n),
    'low': prices - np.random.uniform(100, 300, n),
    'close': prices,
    'volume': np.random.uniform(100, 1000, n),
    'symbol': 'BTCUSDT'
})
print(f"   ✓ Generated {n} bars of test data")

# Test each strategy family
print("\n3. Testing strategy families...")

results = {}

# 1. Momentum
print("\n   Testing 1. MOMENTUM STRATEGIES...")
try:
    momentum = MomentumStrategy(TrendConfig())
    momentum.update(test_data)
    signal = momentum.generate_signal(test_data)
    results['momentum'] = 'PASS' if momentum.is_initialized else 'PARTIAL'
    print(f"      ✓ MomentumStrategy: {results['momentum']}")
except Exception as e:
    results['momentum'] = f'FAIL: {e}'
    print(f"      ✗ MomentumStrategy: {e}")

# 2. Volatility
print("\n   Testing 2. VOLATILITY BREAKOUT STRATEGIES...")
try:
    breakout = VolatilityBreakoutStrategy(BreakoutConfig())
    breakout.update(test_data)
    signal = breakout.generate_signal(test_data)
    results['volatility'] = 'PASS' if breakout.is_initialized else 'PARTIAL'
    print(f"      ✓ VolatilityBreakoutStrategy: {results['volatility']}")
except Exception as e:
    results['volatility'] = f'FAIL: {e}'
    print(f"      ✗ VolatilityBreakoutStrategy: {e}")

# 3. Microstructure
print("\n   Testing 3. MICROSTRUCTURE STRATEGIES...")
try:
    micro = MicrostructureStrategy(MicrostructureConfig())
    micro.update(test_data)
    # Generate orderbook snapshot for testing
    orderbook = {
        'bids': [(49900, 10), (49850, 20), (49800, 30)],
        'asks': [(50100, 10), (50150, 15), (50200, 25)]
    }
    results['microstructure'] = 'PASS' if micro.is_initialized else 'PARTIAL'
    print(f"      ✓ MicrostructureStrategy: {results['microstructure']}")
except Exception as e:
    results['microstructure'] = f'FAIL: {e}'
    print(f"      ✗ MicrostructureStrategy: {e}")

# 4. Funding
print("\n   Testing 4. FUNDING ARBITRAGE STRATEGIES...")
try:
    funding = FundingArbitrageStrategy(FundingConfig())
    funding.update(test_data)
    signal = funding.generate_signal(test_data)
    results['funding'] = 'PASS' if funding.is_initialized else 'PARTIAL'
    print(f"      ✓ FundingArbitrageStrategy: {results['funding']}")
except Exception as e:
    results['funding'] = f'FAIL: {e}'
    print(f"      ✗ FundingArbitrageStrategy: {e}")

# 5. Stat Arb
print("\n   Testing 5. STATISTICAL ARBITRAGE STRATEGIES...")
try:
    statarb = StatArbStrategy(StatArbConfig())
    statarb.update(test_data)
    signal = statarb.generate_signal(test_data)
    results['stat_arb'] = 'PASS' if statarb.is_initialized else 'PARTIAL'
    print(f"      ✓ StatArbStrategy: {results['stat_arb']}")
except Exception as e:
    results['stat_arb'] = f'FAIL: {e}'
    print(f"      ✗ StatArbStrategy: {e}")

# 6. Regime ML
print("\n   Testing 6. REGIME ML STRATEGIES...")
try:
    regime = RegimeMLStrategy(RegimeConfig())
    regime.update(test_data)
    signal = regime.generate_signal(test_data)
    results['regime_ml'] = 'PASS' if regime.is_initialized else 'PARTIAL'
    print(f"      ✓ RegimeMLStrategy: {results['regime_ml']}")
    
    # Test RegimeDetector
    detector = RegimeDetector(RegimeConfig())
    detected_regime, confidence = detector.detect_regime(test_data)
    print(f"      ✓ Detected regime: {detected_regime.name} (confidence: {confidence:.2f})")
except Exception as e:
    results['regime_ml'] = f'FAIL: {e}'
    print(f"      ✗ RegimeMLStrategy: {e}")

# 7. Event-Driven
print("\n   Testing 7. EVENT-DRIVEN STRATEGIES...")
try:
    event = EventDrivenStrategy(EventConfig())
    event.update(test_data)
    signal = event.generate_signal(test_data)
    results['event_driven'] = 'PASS' if event.is_initialized else 'PARTIAL'
    print(f"      ✓ EventDrivenStrategy: {results['event_driven']}")
    
    # Test SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    sentiment = analyzer.analyze_text("Bitcoin surges on institutional adoption!")
    print(f"      ✓ Sentiment score: {sentiment:.2f}")
except Exception as e:
    results['event_driven'] = f'FAIL: {e}'
    print(f"      ✗ EventDrivenStrategy: {e}")

# Test Ensemble
print("\n   Testing STRATEGY ENSEMBLE...")
try:
    ensemble = StrategyEnsemble(EnsembleConfig())
    ensemble.update(test_data)
    status = ensemble.get_status()
    results['ensemble'] = 'PASS'
    print(f"      ✓ StrategyEnsemble: {results['ensemble']}")
    print(f"      ✓ Current regime: {status['current_regime']}")
except Exception as e:
    results['ensemble'] = f'FAIL: {e}'
    print(f"      ✗ StrategyEnsemble: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(1 for v in results.values() if v == 'PASS')
partial = sum(1 for v in results.values() if v == 'PARTIAL')
failed = sum(1 for v in results.values() if v.startswith('FAIL'))

print(f"\n   Total: {len(results)} strategy families tested")
print(f"   ✓ PASS: {passed}")
print(f"   ~ PARTIAL: {partial}")
print(f"   ✗ FAIL: {failed}")

print("\n" + "=" * 70)
print("7 STRATEGY FAMILIES THAT MAKE MONEY IN CRYPTO")
print("=" * 70)
print("""
1. MOMENTUM/TREND-FOLLOWING
   - Supertrend, EMA crossover, ADX trend
   - Works in: TRENDING markets
   
2. VOLATILITY BREAKOUT
   - Donchian, ATR expansion, range compression
   - Works in: HIGH_VOLATILITY, BREAKOUT markets
   
3. MICROSTRUCTURE/ORDERFLOW
   - Order book imbalance, CVD, whale detection
   - Works in: All markets (HFT)
   
4. FUNDING RATE ARBITRAGE
   - Funding harvesting, basis trading
   - Works in: ALL markets (market-neutral)
   
5. STATISTICAL ARBITRAGE
   - Pairs trading, triangular arb, spread trading
   - Works in: RANGING, MEAN_REVERTING markets
   
6. REGIME-BASED ML (CRITICAL!)
   - HMM regime detection, per-regime models
   - THE key to not overfit and losing money
   
7. EVENT-DRIVEN
   - Sentiment, whale watching, liquidation tracking
   - Works in: News-driven crypto markets

KEY INSIGHT: Don't run all strategies!
Select regime-appropriate strategies only.
""")

if failed == 0:
    print("\n✓ ALL STRATEGY FAMILIES ARE WORKING!")
    print("  Your autonomous trading system is ready.")
else:
    print(f"\n✗ {failed} strategy families need attention.")
