import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
import asyncio

# Add project root to path
# walk_forward.py is in quant-bot/src/validation
# We need to reach quant-bot/ to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# We also need Delta Exchange Algo/ for legacy imports if any
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.engine.backtest_engine import OrchestratorBacktester
from src.engine.autonomous_orchestrator import OrchestratorConfig

# Fallback Data Loader
class HistoricalDataLoader:
    def generate_synthetic_data(self, start_date, end_date, volatility=0.02):
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        n = len(dates)
        prices = 10000 * np.cumprod(1 + np.random.normal(0, volatility/np.sqrt(1440), n))
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, n)
        })
        return df

class WalkForwardValidator:
    """
    Performs Walk-Forward Validation to test strategy robustness.
    Prevents overfitting by testing on out-of-sample data.
    """
    
    def __init__(self, data: pd.DataFrame, config: OrchestratorConfig):
        self.data = data
        self.config = config
        self.logger = logging.getLogger("WalkForward")
        self.results: List[Dict] = []
        
    async def run(self, train_window_days: int = 90, test_window_days: int = 30, step_days: int = 30):
        """
        Run sliding window validation.
        """
        start_date = self.data['timestamp'].min()
        end_date = self.data['timestamp'].max()
        
        current_date = start_date
        
        self.logger.info(f"Starting Walk-Forward Validation: {start_date} to {end_date}")
        
        window_count = 0
        
        while current_date + timedelta(days=train_window_days + test_window_days) <= end_date:
            # Define windows
            train_start = current_date
            train_end = current_date + timedelta(days=train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_window_days)
            
            self.logger.info(f"Window {window_count+1}: Test[{test_start.date()} -> {test_end.date()}]")
            
            # Slice data
            test_data = self.data[(self.data['timestamp'] >= test_start) & (self.data['timestamp'] < test_end)]
            
            if len(test_data) < 10:
                break
                
            # Run Backtest on Test Data
            backtester = OrchestratorBacktester(test_data, self.config)
            window_result = await backtester.run()
            
            # Record metrics (Assuming window_result returns dict)
            self.results.append({
                'window': window_count + 1,
                'test_start': test_start,
                'test_end': test_end,
                'metrics': window_result
            })
            
            # Step forward
            current_date += timedelta(days=step_days)
            window_count += 1
            
        return self._aggregate_results()

    def _aggregate_results(self):
        print(f"Completed {len(self.results)} windows.")
        return self.results

if __name__ == "__main__":
    print("Starting Walk-Forward Validation Script...")
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("Generating synthetic data...")
        loader = HistoricalDataLoader()
        data = loader.generate_synthetic_data(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 1), # Shorter for test
            volatility=0.02
        )
        print(f"Generated {len(data)} rows.")
        
        config = OrchestratorConfig(symbols=["BTCUSD"])
        validator = WalkForwardValidator(data, config)
        
        print("Running validation loop...")
        asyncio.run(validator.run(train_window_days=30, test_window_days=7, step_days=7))
        print("Validation complete.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
