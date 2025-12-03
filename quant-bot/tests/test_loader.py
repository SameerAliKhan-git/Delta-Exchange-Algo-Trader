"""
Unit tests for data loader module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.loader import CSVDataLoader, DataPipeline, create_sample_data


class TestCSVDataLoader:
    """Tests for CSVDataLoader."""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing."""
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='1h')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, n),
            'high': np.random.uniform(110, 120, n),
            'low': np.random.uniform(90, 100, n),
            'close': np.random.uniform(100, 110, n),
            'volume': np.random.uniform(1000, 10000, n)
        })
        
        # Ensure high >= open, close and low <= open, close
        df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.01
        df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.99
        
        filepath = tmp_path / "test_data.csv"
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def test_load_basic(self, sample_csv):
        """Test basic CSV loading."""
        loader = CSVDataLoader(str(sample_csv))
        df = loader.load()
        
        assert len(df) == 100
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_load_date_filter(self, sample_csv):
        """Test loading with date filters."""
        loader = CSVDataLoader(str(sample_csv))
        df = loader.load(start='2020-01-01', end='2020-01-02')
        
        assert len(df) < 100
        assert df.index.min() >= pd.Timestamp('2020-01-01')
    
    def test_validate_data(self, sample_csv):
        """Test data validation."""
        loader = CSVDataLoader(str(sample_csv))
        df = loader.load()
        
        assert loader.validate(df) == True
    
    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            CSVDataLoader("nonexistent.csv")
    
    def test_resample(self, sample_csv):
        """Test resampling functionality."""
        loader = CSVDataLoader(str(sample_csv))
        df = loader.load(resample='4h')
        
        # Should have fewer rows after resampling
        assert len(df) <= 100 // 4 + 1


class TestCreateSampleData:
    """Tests for sample data creation."""
    
    def test_create_sample_data(self, tmp_path):
        """Test sample data generation."""
        filepath = tmp_path / "sample.csv"
        df = create_sample_data(str(filepath), n_rows=100)
        
        assert len(df) == 100
        assert filepath.exists()
        
        # Verify OHLCV columns
        assert 'timestamp' in df.columns
        assert 'open' in df.columns
        assert 'close' in df.columns


class TestDataPipeline:
    """Tests for DataPipeline."""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create sample CSV."""
        filepath = tmp_path / "pipeline_test.csv"
        create_sample_data(str(filepath), n_rows=100)
        return filepath
    
    def test_pipeline_single_loader(self, sample_csv):
        """Test pipeline with single loader."""
        pipeline = DataPipeline()
        pipeline.add_loader(CSVDataLoader(str(sample_csv)))
        
        df = pipeline.execute()
        
        assert len(df) == 100
    
    def test_pipeline_with_transformer(self, sample_csv):
        """Test pipeline with transformer."""
        pipeline = DataPipeline()
        pipeline.add_loader(CSVDataLoader(str(sample_csv)))
        
        # Add transformer that adds a column
        pipeline.add_transformer(lambda df: df.assign(returns=df['close'].pct_change()))
        
        df = pipeline.execute()
        
        assert 'returns' in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
