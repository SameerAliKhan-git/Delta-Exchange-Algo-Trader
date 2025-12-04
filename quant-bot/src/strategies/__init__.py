"""
Crypto Strategy Families - Proven Profitable Approaches
========================================================

7 Strategy Families That Actually Make Money in Crypto:
1. Momentum / Trend-Following
2. Volatility Breakout
3. Market Microstructure / Orderflow
4. Funding Rate Arbitrage
5. Statistical Arbitrage
6. Regime-Based ML (THE CRITICAL MISSING PIECE)
7. Event-Driven Trading

Each strategy is designed for crypto's unique characteristics:
- High volatility
- 24/7 markets
- Strong trending behavior
- Retail-driven momentum
- Weak mean-reversion

KEY INSIGHT: Trade regime-appropriate strategies!
Train separate models per regime, only trade when regime matches model's skill.
"""

# Base classes
from .base import (
    BaseStrategy,
    StrategyConfig,
    Signal,
    SignalType,
    TechnicalIndicators
)

# Strategy 1: Momentum / Trend-Following
from .momentum import (
    MomentumStrategy,
    SupertrendStrategy,
    EMAcrossoverStrategy,
    ADXTrendStrategy,
    TrendConfig
)

# Strategy 2: Volatility Breakout
from .volatility_breakout import (
    VolatilityBreakoutStrategy,
    DonchianBreakout,
    ATRBreakout,
    RangeExpansion,
    BreakoutConfig
)

# Strategy 3: Market Microstructure / Orderflow
from .microstructure import (
    MicrostructureStrategy,
    OrderBookImbalance,
    CVDAnalyzer,
    WhaleDetector,
    SpreadAnalyzer,
    MicrostructureConfig
)

# Strategy 4: Funding Rate / Perpetual Arbitrage
from .funding_arbitrage import (
    FundingArbitrageStrategy,
    FundingRateHarvester,
    BasisTrader,
    FundingPredictor,
    FundingConfig
)

# Strategy 5: Statistical Arbitrage
from .stat_arb import (
    StatArbStrategy,
    CointegrationAnalyzer,
    SpreadTrader,
    PairsTrader,
    TriangularArbitrage,
    StatArbConfig
)

# Strategy 6: Regime-Based ML (MOST CRITICAL)
from .regime_ml import (
    RegimeDetector,
    RegimeClassifier,
    RegimeAwareModel,
    RegimeMLStrategy,
    RegimeConfig,
    MarketRegime
)

# Strategy 7: Event-Driven Trading
from .event_driven import (
    SentimentAnalyzer,
    WhaleWatcher,
    LiquidationTracker,
    EventAggregator,
    SentimentTrader,
    EventDrivenStrategy,
    EventConfig,
    EventType,
    Event
)

# Master Ensemble
from .strategy_ensemble import (
    StrategyEnsemble,
    AutoTuningEnsemble,
    StrategySelector,
    SignalAggregator,
    StrategyPerformance,
    EnsembleConfig
)

__all__ = [
    # Base
    'BaseStrategy',
    'StrategyConfig', 
    'Signal',
    'SignalType',
    'TechnicalIndicators',
    
    # Strategy 1: Momentum
    'MomentumStrategy',
    'SupertrendStrategy',
    'EMAcrossoverStrategy',
    'ADXTrendStrategy',
    'TrendConfig',
    
    # Strategy 2: Volatility Breakout
    'VolatilityBreakoutStrategy',
    'DonchianBreakout',
    'ATRBreakout',
    'RangeExpansion',
    'BreakoutConfig',
    
    # Strategy 3: Microstructure
    'MicrostructureStrategy',
    'OrderBookImbalance',
    'CVDAnalyzer',
    'WhaleDetector',
    'SpreadAnalyzer',
    'MicrostructureConfig',
    
    # Strategy 4: Funding Arbitrage
    'FundingArbitrageStrategy',
    'FundingRateHarvester',
    'BasisTrader',
    'FundingPredictor',
    'FundingConfig',
    
    # Strategy 5: Statistical Arbitrage
    'StatArbStrategy',
    'CointegrationAnalyzer',
    'SpreadTrader',
    'PairsTrader',
    'TriangularArbitrage',
    'StatArbConfig',
    
    # Strategy 6: Regime ML
    'RegimeDetector',
    'RegimeClassifier',
    'RegimeAwareModel',
    'RegimeMLStrategy',
    'RegimeConfig',
    'MarketRegime',
    
    # Strategy 7: Event-Driven
    'SentimentAnalyzer',
    'WhaleWatcher',
    'LiquidationTracker',
    'EventAggregator',
    'SentimentTrader',
    'EventDrivenStrategy',
    'EventConfig',
    'EventType',
    'Event',
    
    # Ensemble
    'StrategyEnsemble',
    'AutoTuningEnsemble',
    'StrategySelector',
    'SignalAggregator',
    'StrategyPerformance',
    'EnsembleConfig',
]
