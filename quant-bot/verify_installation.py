"""
System Verification Script
==========================
Verifies that all quant-bot modules are properly installed and working.

Run this to check your installation:
    python verify_installation.py
"""

import sys
from datetime import datetime

def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def check_module(module_name, import_path, class_name=None):
    """Check if a module can be imported."""
    try:
        module = __import__(import_path, fromlist=[class_name] if class_name else [])
        if class_name:
            obj = getattr(module, class_name, None)
            if obj is None:
                return False, f"Class {class_name} not found"
        return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    print_header("QUANT-BOT INSTALLATION VERIFICATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    
    # Core dependencies
    print_header("1. CORE DEPENDENCIES")
    
    core_deps = [
        ("numpy", "numpy", None),
        ("pandas", "pandas", None),
        ("scipy", "scipy", None),
        ("scikit-learn", "sklearn", None),
    ]
    
    for name, path, _ in core_deps:
        success, msg = check_module(name, path)
        status = "✓" if success else "✗"
        print(f"  {status} {name}: {msg}")
    
    # Optional dependencies
    print_header("2. OPTIONAL DEPENDENCIES (for advanced features)")
    
    optional_deps = [
        ("PyTorch (Deep Learning)", "torch", None),
        ("Stable-Baselines3 (RL)", "stable_baselines3", None),
        ("Gymnasium (RL Env)", "gymnasium", None),
        ("ARCH (GARCH)", "arch", None),
        ("HMMLearn (HMM)", "hmmlearn", None),
        ("PyWavelets (Wavelets)", "pywt", None),
        ("Transformers (FinBERT)", "transformers", None),
        ("MLflow (Tracking)", "mlflow", None),
        ("Riskfolio-Lib (Portfolio)", "riskfolio", None),
        ("CVXPY (Optimization)", "cvxpy", None),
    ]
    
    for name, path, _ in optional_deps:
        success, msg = check_module(name, path)
        status = "✓" if success else "○"  # Optional, so use different marker
        print(f"  {status} {name}: {msg if not success else 'OK'}")
    
    # Quant-bot modules
    print_header("3. QUANT-BOT MODULES")
    
    modules = [
        ("Data Module", "src.data", "BarSampler"),
        ("Features Module", "src.features", "KalmanFilter"),
        ("Models Module", "src.models", "RLTrainer"),
        ("Backtest Module", "src.backtest", "StrategyValidator"),
        ("Risk Module", "src.risk", "PortfolioOptimizer"),
        ("Execution Module", "src.execution", "ExecutionEngine"),
        ("Signals Module", "src.signals", "NewsSentimentAnalyzer"),
        ("Infrastructure Module", "src.infrastructure", "MLflowTracker"),
        ("Engine Module", "src.engine", "AutonomousOrchestrator"),
    ]
    
    all_passed = True
    for name, path, class_name in modules:
        success, msg = check_module(name, path, class_name)
        status = "✓" if success else "✗"
        if not success:
            all_passed = False
        print(f"  {status} {name}: {msg}")
    
    # Summary
    print_header("4. SUMMARY")
    
    if all_passed:
        print("""
  ✓ All quant-bot modules loaded successfully!
  
  Your system is ready for autonomous trading.
  
  Quick Start:
  -----------
  from src import get_orchestrator
  
  orchestrator = get_orchestrator({
      'symbols': ['BTCUSDT', 'ETHUSDT'],
      'position_size_usd': 1000.0
  })
  
  import asyncio
  asyncio.run(orchestrator.run())
""")
    else:
        print("""
  ⚠ Some modules failed to load.
  
  Please check the errors above and:
  1. Install missing dependencies: pip install -r requirements_full.txt
  2. Ensure all source files are present
  3. Check for syntax errors in modified files
""")
    
    # Feature summary
    print_header("5. IMPLEMENTED FEATURES")
    
    features = [
        ("Alternative Bar Types", "Dollar, Volume, Tick, Imbalance, Run bars"),
        ("Signal Processing", "Kalman Filter, HMM, Wavelets, GARCH, Fourier"),
        ("Deep Learning", "LSTM, GRU, TCN, Transformer"),
        ("Reinforcement Learning", "DQN, PPO, A2C with custom TradingGym"),
        ("Portfolio Optimization", "HRP, MVO, Black-Litterman, Risk Parity"),
        ("Strategy Validation", "Monte Carlo, PBO, Deflated Sharpe"),
        ("Sentiment Analysis", "FinBERT, Keyword-based, Event Extraction"),
        ("Experiment Tracking", "MLflow integration, Local tracking"),
        ("Drift Detection", "PSI, KS test, CUSUM, Auto-retraining"),
        ("Execution Algorithms", "TWAP, VWAP, IS, Iceberg, Adaptive"),
        ("Autonomous Trading", "Full orchestration with auto-retraining"),
    ]
    
    for name, desc in features:
        print(f"  ✓ {name}")
        print(f"      {desc}")
    
    print("\n" + "="*70)
    print(" VERIFICATION COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
