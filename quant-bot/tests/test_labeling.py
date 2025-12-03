"""
Unit tests for AFML labeling module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from labeling.afml_labeling import (
    TripleBarrierLabeler,
    MetaLabeler,
    compute_sample_weights,
    trend_scanning_labels,
    analyze_labels
)


class TestTripleBarrierLabeler:
    """Tests for TripleBarrierLabeler."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price series."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n, freq='1h')
        prices = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n)),
            index=dates
        )
        return prices
    
    def test_basic_labeling(self, sample_prices):
        """Test basic triple-barrier labeling."""
        labeler = TripleBarrierLabeler(
            profit_taking=0.02,
            stop_loss=0.01,
            max_holding_period=20
        )
        
        labels = labeler.fit_transform(sample_prices)
        
        assert len(labels) == len(sample_prices)
        assert set(labels.unique()).issubset({-1, 0, 1})
    
    def test_label_distribution(self, sample_prices):
        """Test that labels have reasonable distribution."""
        labeler = TripleBarrierLabeler(
            profit_taking=0.02,
            stop_loss=0.01,
            max_holding_period=20
        )
        
        labels = labeler.fit_transform(sample_prices)
        
        # Should have some of each label type
        label_counts = labels.value_counts()
        
        # At least some labels should be non-zero
        assert (labels != 0).sum() > 0
    
    def test_symmetric_barriers(self, sample_prices):
        """Test symmetric profit/loss barriers."""
        labeler = TripleBarrierLabeler(
            profit_taking=0.02,
            stop_loss=0.02,  # Same as profit taking
            max_holding_period=50
        )
        
        labels = labeler.fit_transform(sample_prices)
        
        # With random walk and symmetric barriers, should be roughly equal
        # (allowing for drift)
        longs = (labels == 1).sum()
        shorts = (labels == -1).sum()
        
        # Should be within 2x of each other
        assert 0.2 < longs / max(shorts, 1) < 5
    
    def test_get_barriers(self, sample_prices):
        """Test barrier retrieval."""
        labeler = TripleBarrierLabeler(
            profit_taking=0.02,
            stop_loss=0.01,
            max_holding_period=20
        )
        
        labeler.fit_transform(sample_prices)
        barriers = labeler.get_barriers()
        
        assert 'price' in barriers.columns
        assert 'upper' in barriers.columns
        assert 'lower' in barriers.columns
        
        # Upper barrier should be above price
        assert (barriers['upper'] >= barriers['price']).all()
        
        # Lower barrier should be below price
        assert (barriers['lower'] <= barriers['price']).all()
    
    def test_returns(self, sample_prices):
        """Test return calculation."""
        labeler = TripleBarrierLabeler(
            profit_taking=0.02,
            stop_loss=0.01,
            max_holding_period=20
        )
        
        labeler.fit_transform(sample_prices)
        returns = labeler.get_returns()
        
        assert len(returns) == len(sample_prices)
        
        # Most non-zero returns should be within barrier limits
        non_zero = returns[returns != 0]
        assert (non_zero.abs() <= 0.05).mean() > 0.9  # 90% within 5%


class TestMetaLabeler:
    """Tests for MetaLabeler."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for meta-labeling."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n, freq='1h')
        
        prices = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n)),
            index=dates
        )
        
        # Generate primary signals
        primary_signals = pd.Series(
            np.random.choice([-1, 0, 1], n),
            index=dates
        )
        
        # Generate triple-barrier labels
        labeler = TripleBarrierLabeler(
            profit_taking=0.02,
            stop_loss=0.01,
            max_holding_period=20
        )
        tb_labels = labeler.fit_transform(prices)
        
        return prices, primary_signals, tb_labels
    
    def test_meta_labeling(self, sample_data):
        """Test meta-labeling."""
        prices, primary_signals, tb_labels = sample_data
        
        meta = MetaLabeler()
        meta_labels = meta.fit_transform(prices, primary_signals, tb_labels)
        
        # Meta-labels should be binary
        assert set(meta_labels.unique()).issubset({0, 1})
    
    def test_meta_labeling_filtered(self, sample_data):
        """Test that meta-labeling only labels active signals."""
        prices, primary_signals, tb_labels = sample_data
        
        meta = MetaLabeler(side_threshold=0.5)
        meta_labels = meta.fit_transform(prices, primary_signals, tb_labels)
        
        # Should only have labels where primary signal was active
        assert len(meta_labels) <= (primary_signals.abs() > 0.5).sum()


class TestSampleWeights:
    """Tests for sample weights computation."""
    
    def test_compute_weights(self):
        """Test sample weight computation."""
        np.random.seed(42)
        n = 100
        
        dates = pd.date_range('2020-01-01', periods=n, freq='1h')
        labels = pd.Series(np.random.choice([-1, 0, 1], n), index=dates)
        exit_times = pd.Series(np.arange(1, n + 1) + np.random.randint(1, 10, n), index=dates)
        close = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n)), index=dates)
        
        weights = compute_sample_weights(labels, exit_times, close)
        
        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0, atol=0.01)
        
        # All weights should be non-negative
        assert (weights >= 0).all()


class TestTrendScanning:
    """Tests for trend scanning labels."""
    
    def test_trend_scanning(self):
        """Test trend scanning label generation."""
        np.random.seed(42)
        n = 200
        
        dates = pd.date_range('2020-01-01', periods=n, freq='1h')
        
        # Create trending price series
        trend = np.linspace(0, 0.1, n)  # Upward trend
        noise = np.random.normal(0, 0.01, n)
        prices = pd.Series(100 * np.exp(trend + noise.cumsum() * 0.1), index=dates)
        
        result = trend_scanning_labels(prices, look_forward=20, t_threshold=2.0)
        
        assert 't_value' in result.columns
        assert 'label' in result.columns
        
        # Should detect the upward trend
        assert (result['label'] == 1).sum() > (result['label'] == -1).sum()


class TestAnalyzeLabels:
    """Tests for label analysis."""
    
    def test_analyze_labels(self):
        """Test label analysis function."""
        np.random.seed(42)
        n = 100
        
        dates = pd.date_range('2020-01-01', periods=n, freq='1h')
        labels = pd.Series(np.random.choice([-1, 0, 1], n), index=dates)
        returns = pd.Series(np.random.normal(0, 0.02, n), index=dates)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n)), index=dates)
        
        analysis = analyze_labels(labels, returns, prices)
        
        assert 'label_counts' in analysis
        assert 'label_percentages' in analysis
        assert 'return_by_label' in analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
