"""
Quant-Bot Demo Script

Demonstrates the full ML trading pipeline:
1. Load data
2. Generate features
3. Create labels (triple-barrier)
4. Train model
5. Backtest strategy
6. Generate performance report
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from loguru import logger

from data.loader import CSVDataLoader, create_sample_data
from features.feature_engineer import FeaturePipeline
from labeling.afml_labeling import TripleBarrierLabeler, compute_sample_weights
from models.train import ModelTrainer
from backtest.engine import BacktestEngine, BacktestConfig
from utils.metrics import generate_performance_report, print_performance_report


def main():
    """Run the demo pipeline."""
    print("=" * 70)
    print("            QUANT-BOT: FINANCIAL ML TRADING DEMO")
    print("=" * 70)
    
    # ==========================================================================
    # STEP 1: Load or Create Data
    # ==========================================================================
    print("\n📊 STEP 1: Loading Data...")
    
    data_path = Path(__file__).parent.parent / "data" / "sample_ohlcv.csv"
    
    if not data_path.exists():
        print("Creating sample data...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        create_sample_data(str(data_path), n_rows=5000)
    
    loader = CSVDataLoader(str(data_path))
    df = loader.load()
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # ==========================================================================
    # STEP 2: Feature Engineering
    # ==========================================================================
    print("\n🔧 STEP 2: Generating Features...")
    
    feature_pipeline = FeaturePipeline()
    features = feature_pipeline.fit_transform(df)
    
    print(f"✅ Generated {len(feature_pipeline.get_feature_names())} features")
    print(f"   Sample features: {feature_pipeline.get_feature_names()[:5]}")
    
    # ==========================================================================
    # STEP 3: Create Labels (Triple-Barrier Method)
    # ==========================================================================
    print("\n🏷️ STEP 3: Creating Triple-Barrier Labels...")
    
    labeler = TripleBarrierLabeler(
        profit_taking=0.02,  # 2% profit target
        stop_loss=0.01,      # 1% stop loss
        max_holding_period=20,  # 20 bars max
        volatility_scaling=False
    )
    
    labels = labeler.fit_transform(df['close'])
    
    print(f"✅ Label distribution:")
    label_counts = labels.value_counts()
    for label, count in label_counts.items():
        label_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}[label]
        print(f"   {label_name}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Compute sample weights
    weights = compute_sample_weights(
        labels,
        labeler.exit_times_,
        df['close']
    )
    
    # ==========================================================================
    # STEP 4: Prepare Training Data
    # ==========================================================================
    print("\n📚 STEP 4: Preparing Training Data...")
    
    # Align features and labels
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx].dropna()
    y = labels.loc[X.index]
    sample_weights = weights.loc[X.index] if weights is not None else None
    
    # Convert to binary classification (long vs not-long)
    y_binary = (y == 1).astype(int)
    
    print(f"✅ Training samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Positive class: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
    
    # ==========================================================================
    # STEP 5: Train Model
    # ==========================================================================
    print("\n🤖 STEP 5: Training Model...")
    
    trainer = ModelTrainer(
        model_type='xgboost',
        n_splits=5,
        purge_gap=10,
        embargo_pct=0.01
    )
    
    results = trainer.train(
        X, y_binary,
        sample_weights=sample_weights,
        optimize=False  # Set True for hyperparameter optimization
    )
    
    print(f"✅ Training complete!")
    print(f"   CV Score: {results.mean_score:.4f} (+/- {results.std_score:.4f})")
    print(f"   Top features:")
    
    # Sort and display top features
    sorted_importance = sorted(
        results.feature_importance.items(),
        key=lambda x: -x[1]
    )[:5]
    
    for feat, imp in sorted_importance:
        print(f"   - {feat}: {imp:.4f}")
    
    # ==========================================================================
    # STEP 6: Generate Signals
    # ==========================================================================
    print("\n📡 STEP 6: Generating Trading Signals...")
    
    # Get probability predictions
    proba = trainer.predict_proba(X)[:, 1]
    
    # Convert to signals
    # Buy when probability > 0.6, sell when < 0.4, flat otherwise
    signals = pd.Series(
        np.where(proba > 0.6, 1, np.where(proba < 0.4, -1, 0)),
        index=X.index
    )
    
    print(f"✅ Signal distribution:")
    signal_counts = signals.value_counts()
    for signal, count in sorted(signal_counts.items()):
        signal_name = {-1: 'Short', 0: 'Flat', 1: 'Long'}[signal]
        print(f"   {signal_name}: {count} ({count/len(signals)*100:.1f}%)")
    
    # ==========================================================================
    # STEP 7: Backtest Strategy
    # ==========================================================================
    print("\n📈 STEP 7: Running Backtest...")
    
    # Get data for backtest period
    backtest_data = df.loc[signals.index]
    
    config = BacktestConfig(
        initial_capital=100000,
        commission_pct=0.001,  # 0.1% commission
        slippage_pct=0.0005,   # 0.05% slippage
        risk_per_trade=0.02,   # 2% risk per trade
        max_position_pct=0.20, # 20% max position
        max_drawdown=0.15      # 15% max drawdown
    )
    
    engine = BacktestEngine(config)
    backtest_result = engine.run(backtest_data, signals)
    
    print(f"✅ Backtest complete!")
    print(backtest_result.summary())
    
    # ==========================================================================
    # STEP 8: Detailed Performance Report
    # ==========================================================================
    print("\n📋 STEP 8: Generating Performance Report...")
    
    # Generate trade returns for additional metrics
    trade_returns = pd.Series([t.pnl_pct for t in backtest_result.trades]) if backtest_result.trades else None
    
    report = generate_performance_report(
        backtest_result.returns,
        trade_returns,
        periods_per_year=252 * 24  # Hourly data
    )
    
    print_performance_report(report)
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("                        SUMMARY")
    print("=" * 70)
    print(f"""
📊 Data:           {len(df)} bars
🔧 Features:       {X.shape[1]} features
🏷️ Labels:         Triple-barrier (2% TP, 1% SL, 20 bar max)
🤖 Model:          XGBoost with purged K-fold CV
📡 Signals:        {len(signals)} generated
📈 Trades:         {backtest_result.total_trades} executed
💰 Return:         {backtest_result.total_return:.2%}
📉 Max Drawdown:   {backtest_result.max_drawdown:.2%}
⚡ Sharpe:         {backtest_result.sharpe_ratio:.2f}
🎯 Win Rate:       {backtest_result.win_rate:.2%}
""")
    
    print("✅ Demo complete! See the notebooks/ directory for more examples.")
    
    return backtest_result


if __name__ == "__main__":
    result = main()
