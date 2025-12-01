"""
End-to-End Tests for Delta Exchange Algo Trader

Tests all major modules and their integrations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta


def test_signals_technical():
    """Test technical indicator signals"""
    print("\n[TEST] Technical Signals")
    print("-" * 40)
    
    from signals import technical
    
    # Generate test data
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.randn(100) * 0.01)
    
    # Test EMA
    ema_result = technical.ema(prices, 20)
    assert len(ema_result) == len(prices), "EMA length mismatch"
    assert not np.isnan(ema_result[-1]), "EMA has NaN at end"
    print(f"  ✓ EMA: {ema_result[-1]:.2f}")
    
    # Test SMA
    sma_result = technical.sma(prices, 20)
    assert len(sma_result) == len(prices), "SMA length mismatch"
    print(f"  ✓ SMA: {sma_result[-1]:.2f}")
    
    # Test RSI
    rsi_result = technical.rsi(prices, 14)
    assert 0 <= rsi_result[-1] <= 100, f"RSI out of range: {rsi_result[-1]}"
    print(f"  ✓ RSI: {rsi_result[-1]:.2f}")
    
    # Test ATR
    highs = prices * 1.01
    lows = prices * 0.99
    atr_result = technical.atr(highs, lows, prices, 14)
    assert atr_result[-1] > 0, "ATR should be positive"
    print(f"  ✓ ATR: {atr_result[-1]:.2f}")
    
    # Test MACD
    macd_line, signal_line, histogram = technical.macd(prices)
    assert len(macd_line) == len(prices), "MACD length mismatch"
    print(f"  ✓ MACD: {macd_line[-1]:.4f}")
    
    # Test Bollinger Bands
    upper, middle, lower = technical.bollinger_bands(prices, 20)
    assert upper[-1] > middle[-1] > lower[-1], "Bollinger band order incorrect"
    print(f"  ✓ Bollinger: Upper={upper[-1]:.2f}, Lower={lower[-1]:.2f}")
    
    print("  [PASS] Technical Signals")
    return True


def test_signals_orderbook():
    """Test orderbook signal analysis"""
    print("\n[TEST] Orderbook Signals")
    print("-" * 40)
    
    from signals.orderbook import parse_orderbook, compute_imbalance, detect_walls, compute_depth_ratio
    
    # Create mock orderbook as dict (API format)
    orderbook_data = {
        'bids': [(100.0, 10.0), (99.5, 15.0), (99.0, 20.0), (98.5, 25.0), (98.0, 30.0)],
        'asks': [(100.5, 8.0), (101.0, 12.0), (101.5, 5.0), (102.0, 50.0), (102.5, 10.0)],
        'timestamp': datetime.now().timestamp()
    }
    
    # Parse orderbook
    snapshot = parse_orderbook(orderbook_data)
    print(f"  ✓ Parsed orderbook: {len(snapshot.bids)} bids, {len(snapshot.asks)} asks")
    
    # Test imbalance
    imbalance = compute_imbalance(orderbook_data)
    assert -1 <= imbalance <= 1, f"Imbalance out of range: {imbalance}"
    print(f"  ✓ Imbalance: {imbalance:.3f}")
    
    # Test wall detection
    bid_walls, ask_walls = detect_walls(orderbook_data, threshold_multiplier=2.0)
    print(f"  ✓ Walls - Bids: {len(bid_walls)}, Asks: {len(ask_walls)}")
    
    # Test depth ratio
    ratio = compute_depth_ratio(orderbook_data, price_range_pct=0.05)
    assert ratio > 0, "Depth ratio should be positive"
    print(f"  ✓ Depth Ratio: {ratio:.3f}")
    
    print("  [PASS] Orderbook Signals")
    return True


def test_signals_sentiment():
    """Test sentiment analysis"""
    print("\n[TEST] Sentiment Signals")
    print("-" * 40)
    
    from signals.sentiment import lexicon_sentiment, analyze_text_sentiment, get_composite_sentiment
    
    # Test with bullish text
    bullish_text = "Bitcoin is surging to new highs! Great bullish momentum."
    bullish_score, bullish_conf = lexicon_sentiment(bullish_text)
    print(f"  ✓ Bullish text score: {bullish_score:.3f} (conf: {bullish_conf:.3f})")
    
    # Test with bearish text
    bearish_text = "Crypto market crashes, massive selloff continues."
    bearish_score, bearish_conf = lexicon_sentiment(bearish_text)
    print(f"  ✓ Bearish text score: {bearish_score:.3f} (conf: {bearish_conf:.3f})")
    
    # Test analyze_text_sentiment
    result = analyze_text_sentiment("Bitcoin rally continues, bulls in control")
    print(f"  ✓ Text analysis: score={result.score:.3f}, source={result.source}")
    
    # Test composite sentiment
    composite = get_composite_sentiment("BTC")
    assert -1 <= composite.score <= 1, f"Composite out of range: {composite.score}"
    print(f"  ✓ Composite sentiment: {composite.score:.3f}")
    
    print("  [PASS] Sentiment Signals")
    return True


def test_data_module():
    """Test data loading and candle structures"""
    print("\n[TEST] Data Module")
    print("-" * 40)
    
    from data import CandleData
    
    # Create test candle data
    n = 100
    timestamps = np.array([datetime.now().timestamp() + i * 3600 for i in range(n)])
    opens = np.random.uniform(50000, 51000, n)
    highs = opens * 1.01
    lows = opens * 0.99
    closes = opens + np.random.uniform(-100, 100, n)
    volumes = np.random.uniform(1000, 5000, n)
    
    data = CandleData(
        symbol="BTCUSD",
        resolution="1h",
        timestamps=timestamps,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes
    )
    
    assert len(data) == n, f"Length mismatch: {len(data)} != {n}"
    print(f"  ✓ CandleData created: {len(data)} candles")
    print(f"  ✓ Symbol: {data.symbol}, Resolution: {data.resolution}")
    
    print("  [PASS] Data Module")
    return True


def test_risk_manager():
    """Test risk management"""
    print("\n[TEST] Risk Manager")
    print("-" * 40)
    
    from risk.manager import RiskManager, RiskConfig
    
    config = RiskConfig(
        max_capital_per_trade_pct=0.02,
        max_daily_loss_pct=0.05,
        max_leverage=5
    )
    
    manager = RiskManager(config, initial_capital=10000)
    
    # Test position sizing
    size_result = manager.compute_futures_size(
        price=50000,
        atr=500
    )
    assert size_result.size >= 0, "Position size should be non-negative"
    print(f"  ✓ Futures size: {size_result.size:.6f} (contracts: {size_result.contracts})")
    
    # Test option contracts
    option_result = manager.compute_option_contracts(premium=100)
    print(f"  ✓ Option contracts: {option_result.contracts}")
    
    # Test risk check
    allowed, reason = manager.can_open_position("BTCUSD")
    print(f"  ✓ Can open position: allowed={allowed}, reason='{reason}'")
    
    # Test PnL recording
    manager.on_position_opened("BTCUSD", 0.1, 50000)
    manager.on_position_closed("BTCUSD", 50)
    print(f"  ✓ Daily trades: {manager.trade_count_today}")
    
    # Test kill switch
    assert manager.can_trade(), "Should be able to trade"
    print(f"  ✓ Can trade: {manager.can_trade()}")
    
    print("  [PASS] Risk Manager")
    return True


def test_backtest_runner():
    """Test backtest engine"""
    print("\n[TEST] Backtest Runner")
    print("-" * 40)
    
    from backtest import BacktestRunner, BacktestConfig
    from data import CandleData
    from signals import technical
    
    # Create test data
    n = 500
    np.random.seed(42)
    base_price = 50000
    timestamps = np.array([datetime.now().timestamp() + i * 3600 for i in range(n)])
    prices = base_price * np.cumprod(1 + np.random.randn(n) * 0.01)
    
    data = CandleData(
        symbol="BTCUSD",
        resolution="1h",
        timestamps=timestamps,
        opens=prices,
        highs=prices * 1.005,
        lows=prices * 0.995,
        closes=prices * (1 + np.random.randn(n) * 0.001),
        volumes=np.random.uniform(100, 500, n)
    )
    
    # Simple strategy for testing
    class SimpleTestStrategy:
        def __init__(self):
            self.closes = []
            
        def on_start(self):
            pass
            
        def on_exit(self):
            pass
            
        def on_candle(self, candle):
            self.closes.append(candle.close)
            if len(self.closes) < 30:
                return 0
            
            arr = np.array(self.closes)
            ema12 = technical.ema(arr, 12)[-1]
            ema26 = technical.ema(arr, 26)[-1]
            
            if np.isnan(ema12) or np.isnan(ema26):
                return 0
            
            if len(self.closes) > 30:
                prev_arr = np.array(self.closes[:-1])
                prev_ema12 = technical.ema(prev_arr, 12)[-1]
                prev_ema26 = technical.ema(prev_arr, 26)[-1]
                
                if not np.isnan(prev_ema12) and not np.isnan(prev_ema26):
                    if ema12 > ema26 and prev_ema12 <= prev_ema26:
                        return 1
                    elif ema12 < ema26 and prev_ema12 >= prev_ema26:
                        return -1
            return 0
    
    config = BacktestConfig(
        initial_capital=10000,
        commission_rate=0.0006,
        leverage=1
    )
    
    strategy = SimpleTestStrategy()
    runner = BacktestRunner(strategy, config)
    result = runner.run(data)
    
    print(f"  ✓ Final equity: ${result.final_equity:,.2f}")
    print(f"  ✓ Total return: {result.total_return_pct:.2f}%")
    print(f"  ✓ Total trades: {result.metrics.total_trades}")
    print(f"  ✓ Win rate: {result.metrics.win_rate:.1f}%")
    print(f"  ✓ Sharpe ratio: {result.metrics.sharpe_ratio:.2f}")
    
    print("  [PASS] Backtest Runner")
    return True


def test_config_settings():
    """Test configuration and settings"""
    print("\n[TEST] Config Settings")
    print("-" * 40)
    
    from config import get_settings
    
    settings = get_settings()
    
    print(f"  ✓ API Testnet: {settings.api.testnet}")
    print(f"  ✓ Log level: {settings.monitoring.log_level}")
    print(f"  ✓ Max leverage: {settings.trading.max_leverage}")
    print(f"  ✓ Max daily loss: {settings.risk.max_daily_loss_pct * 100:.1f}%")
    print(f"  ✓ Default strategy: {settings.trading.strategy}")
    
    print("  [PASS] Config Settings")
    return True


def test_metrics_calculation():
    """Test performance metrics calculation"""
    print("\n[TEST] Performance Metrics")
    print("-" * 40)
    
    from backtest.metrics import Trade, calculate_metrics
    import numpy as np
    
    # Create sample trades
    trades = [
        Trade(entry_time=0, exit_time=1, entry_price=100, exit_price=105, 
              size=1, side="long", pnl=5, pnl_pct=5, exit_reason="signal"),
        Trade(entry_time=2, exit_time=3, entry_price=105, exit_price=102, 
              size=1, side="long", pnl=-3, pnl_pct=-2.86, exit_reason="stop_loss"),
        Trade(entry_time=4, exit_time=5, entry_price=102, exit_price=98, 
              size=1, side="short", pnl=4, pnl_pct=3.92, exit_reason="signal"),
        Trade(entry_time=6, exit_time=7, entry_price=98, exit_price=100, 
              size=1, side="short", pnl=-2, pnl_pct=-2.04, exit_reason="stop_loss"),
    ]
    
    # Create equity curve
    equity = np.array([10000, 10005, 10002, 10006, 10004])
    
    metrics = calculate_metrics(trades, equity, initial_capital=10000)
    
    print(f"  ✓ Total trades: {metrics.total_trades}")
    print(f"  ✓ Win rate: {metrics.win_rate:.1f}%")
    print(f"  ✓ Profit factor: {metrics.profit_factor:.2f}")
    print(f"  ✓ Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  ✓ Max drawdown: {metrics.max_drawdown_pct:.2f}%")
    
    assert metrics.total_trades == 4, "Trade count mismatch"
    assert metrics.winning_trades == 2, "Winning trades mismatch"
    assert metrics.losing_trades == 2, "Losing trades mismatch"
    
    print("  [PASS] Performance Metrics")
    return True


def run_all_tests():
    """Run all end-to-end tests"""
    print("=" * 60)
    print("Delta Exchange Algo Trader - End-to-End Tests")
    print("=" * 60)
    
    tests = [
        ("Technical Signals", test_signals_technical),
        ("Orderbook Signals", test_signals_orderbook),
        ("Sentiment Signals", test_signals_sentiment),
        ("Data Module", test_data_module),
        ("Risk Manager", test_risk_manager),
        ("Backtest Runner", test_backtest_runner),
        ("Config Settings", test_config_settings),
        ("Performance Metrics", test_metrics_calculation),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n  [FAIL] {name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    for name, success, error in results:
        status = "✓ PASS" if success else f"✗ FAIL: {error}"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if failed > 0:
        print(f"\n⚠️  {failed} test(s) failed!")
        return False
    else:
        print(f"\n✅ All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
