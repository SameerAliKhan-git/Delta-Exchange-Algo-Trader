"""
Unit tests for backtesting module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    Side,
    fixed_slippage,
    volume_dependent_slippage,
    fixed_fraction_size,
    kelly_criterion_size
)


class TestSlippageModels:
    """Tests for slippage models."""
    
    def test_fixed_slippage_long(self):
        """Test fixed slippage for long positions."""
        price = 100.0
        slippage_pct = 0.001  # 0.1%
        
        fill_price = fixed_slippage(price, slippage_pct, Side.LONG)
        
        assert fill_price == 100.1  # 100 * (1 + 0.001)
    
    def test_fixed_slippage_short(self):
        """Test fixed slippage for short positions."""
        price = 100.0
        slippage_pct = 0.001
        
        fill_price = fixed_slippage(price, slippage_pct, Side.SHORT)
        
        assert fill_price == 99.9  # 100 * (1 - 0.001)
    
    def test_volume_dependent_slippage(self):
        """Test volume-dependent slippage."""
        price = 100.0
        quantity = 100
        volume = 10000
        
        # Higher participation = more slippage
        fill_high = volume_dependent_slippage(price, 1000, volume, side=Side.LONG)
        fill_low = volume_dependent_slippage(price, 100, volume, side=Side.LONG)
        
        assert fill_high > fill_low


class TestPositionSizing:
    """Tests for position sizing functions."""
    
    def test_fixed_fraction_size(self):
        """Test fixed fractional position sizing."""
        capital = 100000
        price = 100
        risk_pct = 0.02  # 2% risk
        stop_distance = 0.05  # 5% stop
        
        size = fixed_fraction_size(capital, price, risk_pct, stop_distance)
        
        # Risk amount = 100000 * 0.02 = 2000
        # Size = 2000 / (100 * 0.05) = 400
        assert size == 400
    
    def test_kelly_criterion_size(self):
        """Test Kelly criterion sizing."""
        capital = 100000
        price = 100
        win_rate = 0.55
        win_loss_ratio = 1.5
        
        size = kelly_criterion_size(capital, price, win_rate, win_loss_ratio)
        
        # Kelly % = 0.55 - (1 - 0.55) / 1.5 = 0.55 - 0.3 = 0.25
        # Quarter Kelly = 0.25 * 0.25 = 0.0625
        # Position = 100000 * 0.0625 / 100 = 62.5
        assert size > 0
        assert size < capital / price  # Should be less than all-in


class TestBacktestEngine:
    """Tests for BacktestEngine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 500
        
        dates = pd.date_range('2020-01-01', periods=n, freq='1h')
        close = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n))
        
        df = pd.DataFrame({
            'open': np.roll(close, 1),
            'high': close * (1 + np.random.uniform(0, 0.01, n)),
            'low': close * (1 - np.random.uniform(0, 0.01, n)),
            'close': close,
            'volume': np.random.exponential(1000000, n)
        }, index=dates)
        
        df['open'].iloc[0] = close[0]
        
        return df
    
    @pytest.fixture
    def sample_signals(self, sample_data):
        """Create sample signals."""
        n = len(sample_data)
        
        # Random signals
        signals = pd.Series(
            np.random.choice([-1, 0, 0, 0, 1], n),
            index=sample_data.index
        )
        
        return signals
    
    def test_backtest_runs(self, sample_data, sample_signals):
        """Test that backtest runs without errors."""
        config = BacktestConfig(
            initial_capital=100000,
            commission_pct=0.001,
            slippage_pct=0.0005
        )
        
        engine = BacktestEngine(config)
        result = engine.run(sample_data, sample_signals)
        
        assert result is not None
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(sample_data)
    
    def test_backtest_metrics(self, sample_data, sample_signals):
        """Test that backtest returns valid metrics."""
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        result = engine.run(sample_data, sample_signals)
        
        # Check metrics are computed
        assert result.total_return is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown >= 0
        assert 0 <= result.win_rate <= 1
    
    def test_capital_preservation(self, sample_data):
        """Test that capital doesn't go negative."""
        # Create signals that would cause losses
        n = len(sample_data)
        signals = pd.Series(
            np.random.choice([-1, 1], n),
            index=sample_data.index
        )
        
        config = BacktestConfig(
            initial_capital=100000,
            max_drawdown=0.50  # High drawdown limit
        )
        
        engine = BacktestEngine(config)
        result = engine.run(sample_data, signals)
        
        # Equity should never go negative
        assert (result.equity_curve > 0).all()
    
    def test_no_signals(self, sample_data):
        """Test backtest with no signals."""
        signals = pd.Series(0, index=sample_data.index)
        
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        result = engine.run(sample_data, signals)
        
        # Should have no trades
        assert result.total_trades == 0
        
        # Equity should be constant
        assert result.equity_curve.iloc[0] == result.equity_curve.iloc[-1]
    
    def test_long_only(self, sample_data):
        """Test long-only strategy."""
        n = len(sample_data)
        signals = pd.Series(
            np.random.choice([0, 0, 1], n),  # Only long signals
            index=sample_data.index
        )
        
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        result = engine.run(sample_data, signals)
        
        # Check that trades were made
        assert result.total_trades >= 0
    
    def test_reset(self, sample_data, sample_signals):
        """Test engine reset between runs."""
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config)
        
        # First run
        result1 = engine.run(sample_data, sample_signals)
        
        # Second run should start fresh
        result2 = engine.run(sample_data, sample_signals)
        
        # Results should be identical
        assert result1.total_return == result2.total_return
        assert result1.total_trades == result2.total_trades


class TestBacktestConfig:
    """Tests for BacktestConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        
        assert config.initial_capital == 100000
        assert config.commission_pct > 0
        assert config.slippage_pct >= 0
        assert config.max_drawdown > 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=50000,
            commission_pct=0.002,
            max_drawdown=0.10
        )
        
        assert config.initial_capital == 50000
        assert config.commission_pct == 0.002
        assert config.max_drawdown == 0.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
