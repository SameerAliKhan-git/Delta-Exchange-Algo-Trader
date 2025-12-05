import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

from src.engine.backtest_engine import OrchestratorBacktester
from src.engine.autonomous_orchestrator import OrchestratorConfig
from src.validation.walk_forward import HistoricalDataLoader


class Autotuner:
    """
    Hyper-parameter optimization using Bayesian Optimization (Gaussian Processes).
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.logger = logging.getLogger("Autotuner")
        
        if not HAS_SKOPT:
            self.logger.warning("scikit-optimize not found. Autotuning disabled.")
            
    def optimize(self, n_calls: int = 20):
        """Run optimization loop."""
        if not HAS_SKOPT:
            return None
            
        # Define search space
        space = [
            Real(0.005, 0.05, name='stop_loss_pct'),
            Real(0.01, 0.10, name='take_profit_pct'),
            Real(0.60, 0.95, name='signal_threshold'),
            Integer(5, 20, name='max_daily_trades')
        ]
        
        @use_named_args(space)
        def objective(**params):
            # Create config with params
            config = OrchestratorConfig(
                stop_loss_pct=params['stop_loss_pct'],
                take_profit_pct=params['take_profit_pct'],
                signal_threshold=params['signal_threshold'],
                max_daily_trades=params['max_daily_trades']
            )
            
            # Run backtest
            # We use a shorter window for speed during optimization
            # Ideally we use Walk-Forward here, but for speed we'll use a simple backtest
            # on the last 30 days of data.
            backtester = OrchestratorBacktester(self.data, config)
            
            # Since backtester.run is async, we need to run it synchronously here
            # skopt expects a synchronous function
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(backtester.run())
                loop.close()
                
                # Metric to maximize: Sharpe Ratio
                # We return negative because gp_minimize minimizes
                sharpe = results.get('sharpe_ratio', 0.0)
                
                # Penalty for low trades (overfitting to nothing)
                if results.get('total_trades', 0) < 5:
                    return 0.0
                    
                return -sharpe
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                return 0.0

        self.logger.info(f"Starting Bayesian Optimization ({n_calls} calls)...")
        res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
        
        self.logger.info(f"Best Score: {-res.fun:.4f}")
        self.logger.info(f"Best Params: {res.x}")
        
        return {
            'stop_loss_pct': res.x[0],
            'take_profit_pct': res.x[1],
            'signal_threshold': res.x[2],
            'max_daily_trades': res.x[3]
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    loader = HistoricalDataLoader()
    data = loader.generate_synthetic_data(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 1),
        volatility=0.02
    )
    
    tuner = Autotuner(data)
    best_params = tuner.optimize(n_calls=10)
    print("Optimization Result:", best_params)
