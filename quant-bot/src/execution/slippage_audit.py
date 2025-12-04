"""
Realized Slippage Audit & Calibration
=====================================

DELIVERABLE A: Analyze historical fills vs simulated fills and calibrate execution model.

This module:
1. Compares simulated fills vs actual fills
2. Calculates realized slippage by venue, symbol, order size
3. Calibrates a two-tier slippage model (normal vs stressed)
4. Outputs CSV reports and calibrated parameters

Impact: VERY HIGH — execution kills simulated alpha.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path


class MarketCondition(Enum):
    """Market condition for tiered slippage."""
    NORMAL = "normal"
    STRESSED = "stressed"
    EXTREME = "extreme"


@dataclass
class SlippageRecord:
    """Record of a single fill's slippage."""
    timestamp: datetime
    symbol: str
    venue: str
    side: str  # 'buy' or 'sell'
    
    # Prices
    expected_price: float      # What we expected (mid or signal price)
    limit_price: float         # Our limit price
    fill_price: float          # Actual fill price
    
    # Sizes
    order_size: float
    filled_size: float
    
    # Market context
    spread_bps: float          # Spread at order time in bps
    book_depth_usd: float      # Depth at our price level
    volatility: float          # Recent volatility
    
    # Calculated
    slippage_bps: float = 0.0
    market_condition: MarketCondition = MarketCondition.NORMAL
    
    def __post_init__(self):
        """Calculate slippage after initialization."""
        if self.expected_price > 0:
            if self.side == 'buy':
                self.slippage_bps = (self.fill_price - self.expected_price) / self.expected_price * 10000
            else:
                self.slippage_bps = (self.expected_price - self.fill_price) / self.expected_price * 10000


@dataclass
class SlippageModelParams:
    """Calibrated slippage model parameters."""
    # Base slippage (constant)
    base_slippage_bps: float = 1.0
    
    # Size impact: slippage = base + size_coefficient * (order_size / depth)^size_exponent
    size_coefficient: float = 5.0
    size_exponent: float = 1.5
    
    # Spread component
    spread_multiplier: float = 0.5  # Fraction of spread to add
    
    # Volatility component
    volatility_coefficient: float = 2.0
    
    # Stressed market multipliers
    stressed_multiplier: float = 2.0
    extreme_multiplier: float = 4.0
    
    # Thresholds
    stressed_volatility_threshold: float = 0.03  # 3% daily vol
    extreme_volatility_threshold: float = 0.06   # 6% daily vol
    stressed_spread_threshold_bps: float = 20    # 20 bps spread
    
    def to_dict(self) -> Dict:
        return {
            'base_slippage_bps': self.base_slippage_bps,
            'size_coefficient': self.size_coefficient,
            'size_exponent': self.size_exponent,
            'spread_multiplier': self.spread_multiplier,
            'volatility_coefficient': self.volatility_coefficient,
            'stressed_multiplier': self.stressed_multiplier,
            'extreme_multiplier': self.extreme_multiplier,
            'stressed_volatility_threshold': self.stressed_volatility_threshold,
            'extreme_volatility_threshold': self.extreme_volatility_threshold,
            'stressed_spread_threshold_bps': self.stressed_spread_threshold_bps
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'SlippageModelParams':
        return cls(**d)


class SlippageModel:
    """
    Two-tier slippage model with normal and stressed conditions.
    
    slippage = condition_multiplier * (
        base_slippage +
        size_coefficient * (order_pov)^size_exponent +
        spread_multiplier * spread +
        volatility_coefficient * volatility
    )
    """
    
    def __init__(self, params: Optional[SlippageModelParams] = None):
        self.params = params or SlippageModelParams()
    
    def classify_condition(
        self,
        volatility: float,
        spread_bps: float
    ) -> MarketCondition:
        """Classify current market condition."""
        if (volatility > self.params.extreme_volatility_threshold or 
            spread_bps > self.params.stressed_spread_threshold_bps * 2):
            return MarketCondition.EXTREME
        elif (volatility > self.params.stressed_volatility_threshold or
              spread_bps > self.params.stressed_spread_threshold_bps):
            return MarketCondition.STRESSED
        return MarketCondition.NORMAL
    
    def estimate_slippage(
        self,
        order_size_usd: float,
        book_depth_usd: float,
        spread_bps: float,
        volatility: float,
        condition: Optional[MarketCondition] = None
    ) -> float:
        """
        Estimate slippage in basis points.
        
        Args:
            order_size_usd: Order size in USD
            book_depth_usd: Available depth at price level
            spread_bps: Current spread in basis points
            volatility: Recent volatility (daily)
            condition: Market condition override
        
        Returns:
            Estimated slippage in basis points
        """
        if condition is None:
            condition = self.classify_condition(volatility, spread_bps)
        
        # Calculate participation rate (POV)
        pov = order_size_usd / max(book_depth_usd, 1)
        
        # Base slippage components
        base = self.params.base_slippage_bps
        size_impact = self.params.size_coefficient * (pov ** self.params.size_exponent)
        spread_impact = self.params.spread_multiplier * spread_bps
        vol_impact = self.params.volatility_coefficient * volatility * 10000  # Convert to bps
        
        slippage = base + size_impact + spread_impact + vol_impact
        
        # Apply condition multiplier
        if condition == MarketCondition.STRESSED:
            slippage *= self.params.stressed_multiplier
        elif condition == MarketCondition.EXTREME:
            slippage *= self.params.extreme_multiplier
        
        return slippage


class SlippageAuditor:
    """
    Audit and analyze realized vs simulated slippage.
    """
    
    def __init__(self, model: Optional[SlippageModel] = None):
        self.model = model or SlippageModel()
        self.records: List[SlippageRecord] = []
    
    def add_record(self, record: SlippageRecord):
        """Add a slippage record."""
        self.records.append(record)
    
    def add_fill(
        self,
        timestamp: datetime,
        symbol: str,
        venue: str,
        side: str,
        expected_price: float,
        limit_price: float,
        fill_price: float,
        order_size: float,
        filled_size: float,
        spread_bps: float,
        book_depth_usd: float,
        volatility: float
    ):
        """Add a fill event for slippage tracking."""
        condition = self.model.classify_condition(volatility, spread_bps)
        
        record = SlippageRecord(
            timestamp=timestamp,
            symbol=symbol,
            venue=venue,
            side=side,
            expected_price=expected_price,
            limit_price=limit_price,
            fill_price=fill_price,
            order_size=order_size,
            filled_size=filled_size,
            spread_bps=spread_bps,
            book_depth_usd=book_depth_usd,
            volatility=volatility,
            market_condition=condition
        )
        self.records.append(record)
    
    def load_from_csv(self, filepath: str):
        """Load historical fills from CSV."""
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        
        for _, row in df.iterrows():
            self.add_fill(
                timestamp=row['timestamp'],
                symbol=row['symbol'],
                venue=row.get('venue', 'unknown'),
                side=row['side'],
                expected_price=row['expected_price'],
                limit_price=row.get('limit_price', row['expected_price']),
                fill_price=row['fill_price'],
                order_size=row['order_size'],
                filled_size=row.get('filled_size', row['order_size']),
                spread_bps=row.get('spread_bps', 5),
                book_depth_usd=row.get('book_depth_usd', 100000),
                volatility=row.get('volatility', 0.02)
            )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to DataFrame."""
        if not self.records:
            return pd.DataFrame()
        
        data = []
        for r in self.records:
            data.append({
                'timestamp': r.timestamp,
                'symbol': r.symbol,
                'venue': r.venue,
                'side': r.side,
                'expected_price': r.expected_price,
                'limit_price': r.limit_price,
                'fill_price': r.fill_price,
                'order_size': r.order_size,
                'filled_size': r.filled_size,
                'spread_bps': r.spread_bps,
                'book_depth_usd': r.book_depth_usd,
                'volatility': r.volatility,
                'realized_slippage_bps': r.slippage_bps,
                'market_condition': r.market_condition.value
            })
        
        return pd.DataFrame(data)
    
    def calculate_simulated_slippage(self) -> pd.DataFrame:
        """Calculate simulated slippage for comparison."""
        df = self.to_dataframe()
        
        if df.empty:
            return df
        
        simulated = []
        for _, row in df.iterrows():
            sim_slip = self.model.estimate_slippage(
                order_size_usd=row['order_size'] * row['expected_price'],
                book_depth_usd=row['book_depth_usd'],
                spread_bps=row['spread_bps'],
                volatility=row['volatility']
            )
            simulated.append(sim_slip)
        
        df['simulated_slippage_bps'] = simulated
        df['slippage_error_bps'] = df['realized_slippage_bps'] - df['simulated_slippage_bps']
        df['slippage_ratio'] = df['realized_slippage_bps'] / df['simulated_slippage_bps'].replace(0, np.nan)
        
        return df
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        df = self.calculate_simulated_slippage()
        
        if df.empty:
            return {}
        
        stats = {
            'total_fills': len(df),
            'total_volume_usd': (df['filled_size'] * df['fill_price']).sum(),
            
            # Realized slippage
            'realized_slippage_mean_bps': df['realized_slippage_bps'].mean(),
            'realized_slippage_median_bps': df['realized_slippage_bps'].median(),
            'realized_slippage_std_bps': df['realized_slippage_bps'].std(),
            'realized_slippage_p95_bps': df['realized_slippage_bps'].quantile(0.95),
            
            # Simulated slippage
            'simulated_slippage_mean_bps': df['simulated_slippage_bps'].mean(),
            'simulated_slippage_median_bps': df['simulated_slippage_bps'].median(),
            
            # Error
            'slippage_error_mean_bps': df['slippage_error_bps'].mean(),
            'slippage_error_std_bps': df['slippage_error_bps'].std(),
            'slippage_ratio_mean': df['slippage_ratio'].mean(),
            
            # By condition
            'fills_normal': len(df[df['market_condition'] == 'normal']),
            'fills_stressed': len(df[df['market_condition'] == 'stressed']),
            'fills_extreme': len(df[df['market_condition'] == 'extreme']),
        }
        
        # By condition breakdown
        for condition in ['normal', 'stressed', 'extreme']:
            cond_df = df[df['market_condition'] == condition]
            if len(cond_df) > 0:
                stats[f'{condition}_realized_mean_bps'] = cond_df['realized_slippage_bps'].mean()
                stats[f'{condition}_simulated_mean_bps'] = cond_df['simulated_slippage_bps'].mean()
        
        return stats
    
    def get_by_symbol(self) -> pd.DataFrame:
        """Get slippage breakdown by symbol."""
        df = self.calculate_simulated_slippage()
        
        if df.empty:
            return pd.DataFrame()
        
        return df.groupby('symbol').agg({
            'realized_slippage_bps': ['mean', 'std', 'count'],
            'simulated_slippage_bps': ['mean'],
            'slippage_error_bps': ['mean'],
            'filled_size': 'sum'
        }).round(3)
    
    def get_by_venue(self) -> pd.DataFrame:
        """Get slippage breakdown by venue."""
        df = self.calculate_simulated_slippage()
        
        if df.empty:
            return pd.DataFrame()
        
        return df.groupby('venue').agg({
            'realized_slippage_bps': ['mean', 'std', 'count'],
            'simulated_slippage_bps': ['mean'],
            'slippage_error_bps': ['mean'],
            'filled_size': 'sum'
        }).round(3)
    
    def get_by_size_bucket(self, buckets: List[float] = None) -> pd.DataFrame:
        """Get slippage by order size bucket."""
        df = self.calculate_simulated_slippage()
        
        if df.empty:
            return pd.DataFrame()
        
        if buckets is None:
            buckets = [0, 1000, 5000, 10000, 50000, 100000, float('inf')]
        
        df['size_usd'] = df['order_size'] * df['expected_price']
        df['size_bucket'] = pd.cut(df['size_usd'], bins=buckets)
        
        return df.groupby('size_bucket').agg({
            'realized_slippage_bps': ['mean', 'std', 'count'],
            'simulated_slippage_bps': ['mean'],
            'slippage_error_bps': ['mean']
        }).round(3)


class SlippageCalibrator:
    """
    Calibrate slippage model parameters from historical data.
    """
    
    def __init__(self, auditor: SlippageAuditor):
        self.auditor = auditor
    
    def calibrate(self, method: str = 'least_squares') -> SlippageModelParams:
        """
        Calibrate slippage model parameters.
        
        Args:
            method: Calibration method ('least_squares', 'quantile')
        
        Returns:
            Calibrated SlippageModelParams
        """
        df = self.auditor.to_dataframe()
        
        if df.empty or len(df) < 10:
            print("Insufficient data for calibration, using defaults")
            return SlippageModelParams()
        
        # Calculate features
        df['size_usd'] = df['order_size'] * df['expected_price']
        df['pov'] = df['size_usd'] / df['book_depth_usd']
        df['vol_bps'] = df['volatility'] * 10000
        
        if method == 'least_squares':
            return self._calibrate_least_squares(df)
        elif method == 'quantile':
            return self._calibrate_quantile(df)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _calibrate_least_squares(self, df: pd.DataFrame) -> SlippageModelParams:
        """Calibrate using least squares regression."""
        from scipy.optimize import minimize
        
        # Split by condition
        normal_df = df[df['market_condition'] == 'normal']
        stressed_df = df[df['market_condition'] == 'stressed']
        
        def objective(params):
            base, size_coef, size_exp, spread_mult, vol_coef = params
            
            # Predict for normal conditions
            pred = (base + 
                   size_coef * (normal_df['pov'] ** size_exp) +
                   spread_mult * normal_df['spread_bps'] +
                   vol_coef * normal_df['vol_bps'])
            
            mse = ((normal_df['realized_slippage_bps'] - pred) ** 2).mean()
            
            # Regularization
            reg = 0.01 * (base**2 + size_coef**2 + vol_coef**2)
            
            return mse + reg
        
        # Initial guess
        x0 = [1.0, 5.0, 1.5, 0.5, 0.002]
        bounds = [(0, 10), (0, 50), (1, 3), (0, 1), (0, 0.01)]
        
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        base, size_coef, size_exp, spread_mult, vol_coef = result.x
        
        # Calibrate stressed multiplier
        if len(stressed_df) > 5 and len(normal_df) > 5:
            normal_mean = normal_df['realized_slippage_bps'].mean()
            stressed_mean = stressed_df['realized_slippage_bps'].mean()
            stressed_mult = stressed_mean / max(normal_mean, 1)
        else:
            stressed_mult = 2.0
        
        return SlippageModelParams(
            base_slippage_bps=base,
            size_coefficient=size_coef,
            size_exponent=size_exp,
            spread_multiplier=spread_mult,
            volatility_coefficient=vol_coef,
            stressed_multiplier=max(1.5, min(4.0, stressed_mult))
        )
    
    def _calibrate_quantile(self, df: pd.DataFrame) -> SlippageModelParams:
        """Calibrate using quantile-based approach (more robust)."""
        normal_df = df[df['market_condition'] == 'normal']
        stressed_df = df[df['market_condition'] == 'stressed']
        
        # Use median for robustness
        base = normal_df['realized_slippage_bps'].median()
        
        # Size impact from correlation
        size_impact = normal_df.groupby(
            pd.qcut(normal_df['pov'], 5, duplicates='drop')
        )['realized_slippage_bps'].median()
        
        if len(size_impact) >= 2:
            size_coef = (size_impact.iloc[-1] - size_impact.iloc[0]) / 0.1
        else:
            size_coef = 5.0
        
        # Spread impact
        spread_corr = normal_df['realized_slippage_bps'].corr(normal_df['spread_bps'])
        spread_mult = max(0, min(1, spread_corr))
        
        # Volatility impact
        vol_corr = normal_df['realized_slippage_bps'].corr(normal_df['vol_bps'])
        vol_coef = max(0, min(0.01, vol_corr * 0.005))
        
        # Stressed multiplier
        if len(stressed_df) > 5 and len(normal_df) > 5:
            stressed_mult = (stressed_df['realized_slippage_bps'].median() / 
                          max(normal_df['realized_slippage_bps'].median(), 1))
        else:
            stressed_mult = 2.0
        
        return SlippageModelParams(
            base_slippage_bps=max(0.5, base),
            size_coefficient=max(1, size_coef),
            size_exponent=1.5,
            spread_multiplier=spread_mult,
            volatility_coefficient=vol_coef,
            stressed_multiplier=max(1.5, min(4.0, stressed_mult))
        )


def generate_sample_fills(n_fills: int = 1000) -> List[Dict]:
    """Generate sample fill data for testing."""
    np.random.seed(42)
    
    fills = []
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    venues = ['binance', 'bybit', 'okx']
    
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(n_fills):
        symbol = np.random.choice(symbols)
        venue = np.random.choice(venues)
        side = np.random.choice(['buy', 'sell'])
        
        # Base price depends on symbol
        if symbol == 'BTCUSDT':
            price = 50000 + np.random.randn() * 2000
        elif symbol == 'ETHUSDT':
            price = 3000 + np.random.randn() * 200
        elif symbol == 'SOLUSDT':
            price = 100 + np.random.randn() * 10
        else:
            price = 400 + np.random.randn() * 20
        
        # Order size (heavier tail)
        order_size_usd = np.random.exponential(5000) + 100
        order_size = order_size_usd / price
        
        # Market conditions
        volatility = np.random.exponential(0.02) + 0.005
        spread_bps = np.random.exponential(5) + 1
        book_depth_usd = np.random.exponential(500000) + 50000
        
        # Realized slippage (realistic model)
        pov = order_size_usd / book_depth_usd
        base_slip = 1 + 5 * (pov ** 1.5) + 0.5 * spread_bps + 200 * volatility
        
        # Add noise and occasional large slippage
        noise = np.random.randn() * 2
        if np.random.random() < 0.05:  # 5% chance of large slippage
            noise += np.random.exponential(5)
        
        slippage_bps = max(0, base_slip + noise)
        
        # Calculate fill price
        if side == 'buy':
            fill_price = price * (1 + slippage_bps / 10000)
        else:
            fill_price = price * (1 - slippage_bps / 10000)
        
        fills.append({
            'timestamp': base_time + timedelta(minutes=i*5),
            'symbol': symbol,
            'venue': venue,
            'side': side,
            'expected_price': price,
            'limit_price': price * (1.001 if side == 'buy' else 0.999),
            'fill_price': fill_price,
            'order_size': order_size,
            'filled_size': order_size * np.random.uniform(0.95, 1.0),
            'spread_bps': spread_bps,
            'book_depth_usd': book_depth_usd,
            'volatility': volatility
        })
    
    return fills


def run_slippage_audit(
    fills: Optional[List[Dict]] = None,
    output_dir: str = './reports'
) -> Dict:
    """
    Run complete slippage audit and calibration.
    
    Args:
        fills: List of fill dictionaries (or None to generate samples)
        output_dir: Directory for output files
    
    Returns:
        Audit results dictionary
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create auditor
    auditor = SlippageAuditor()
    
    # Load or generate data
    if fills is None:
        print("Generating sample fill data...")
        fills = generate_sample_fills(1000)
    
    # Add fills
    for fill in fills:
        auditor.add_fill(**fill)
    
    print(f"Loaded {len(auditor.records)} fills")
    
    # Calculate comparison
    df = auditor.calculate_simulated_slippage()
    
    # Get summary stats
    stats = auditor.get_summary_stats()
    
    print("\n" + "="*60)
    print("SLIPPAGE AUDIT SUMMARY")
    print("="*60)
    print(f"\nTotal fills: {stats['total_fills']}")
    print(f"Total volume: ${stats['total_volume_usd']:,.0f}")
    print(f"\nRealized slippage:")
    print(f"  Mean: {stats['realized_slippage_mean_bps']:.2f} bps")
    print(f"  Median: {stats['realized_slippage_median_bps']:.2f} bps")
    print(f"  Std: {stats['realized_slippage_std_bps']:.2f} bps")
    print(f"  95th percentile: {stats['realized_slippage_p95_bps']:.2f} bps")
    print(f"\nSimulated slippage:")
    print(f"  Mean: {stats['simulated_slippage_mean_bps']:.2f} bps")
    print(f"\nModel error:")
    print(f"  Mean error: {stats['slippage_error_mean_bps']:.2f} bps")
    print(f"  Ratio (realized/sim): {stats['slippage_ratio_mean']:.2f}x")
    
    # Calibrate model
    print("\n" + "="*60)
    print("CALIBRATING SLIPPAGE MODEL")
    print("="*60)
    
    calibrator = SlippageCalibrator(auditor)
    calibrated_params = calibrator.calibrate(method='least_squares')
    
    print("\nCalibrated parameters:")
    for k, v in calibrated_params.to_dict().items():
        print(f"  {k}: {v:.4f}")
    
    # Test calibrated model
    calibrated_model = SlippageModel(calibrated_params)
    auditor_cal = SlippageAuditor(calibrated_model)
    for fill in fills:
        auditor_cal.add_fill(**fill)
    
    df_cal = auditor_cal.calculate_simulated_slippage()
    
    print("\nCalibrated model performance:")
    print(f"  Mean error: {df_cal['slippage_error_bps'].mean():.2f} bps (was {stats['slippage_error_mean_bps']:.2f})")
    print(f"  Std error: {df_cal['slippage_error_bps'].std():.2f} bps (was {stats['slippage_error_std_bps']:.2f})")
    
    # Save outputs
    df.to_csv(f"{output_dir}/slippage_audit.csv", index=False)
    
    by_symbol = auditor.get_by_symbol()
    by_symbol.to_csv(f"{output_dir}/slippage_by_symbol.csv")
    
    by_venue = auditor.get_by_venue()
    by_venue.to_csv(f"{output_dir}/slippage_by_venue.csv")
    
    by_size = auditor.get_by_size_bucket()
    by_size.to_csv(f"{output_dir}/slippage_by_size.csv")
    
    # Save calibrated params
    with open(f"{output_dir}/calibrated_slippage_params.json", 'w') as f:
        json.dump(calibrated_params.to_dict(), f, indent=2)
    
    print(f"\nReports saved to {output_dir}/")
    
    return {
        'stats': stats,
        'calibrated_params': calibrated_params,
        'by_symbol': by_symbol,
        'by_venue': by_venue,
        'by_size': by_size
    }


if __name__ == "__main__":
    results = run_slippage_audit()
