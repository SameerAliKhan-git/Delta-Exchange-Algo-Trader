"""
MLflow Experiment Tracking
==========================
ML lifecycle management for trading models.

This module provides:
- Experiment tracking
- Model versioning
- Parameter logging
- Artifact management
- Model registry

Requires: pip install mlflow
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExperimentRun:
    """Single experiment run record."""
    run_id: str
    experiment_name: str
    model_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # 'running', 'finished', 'failed'
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    artifacts: List[str]
    

@dataclass
class ModelVersion:
    """Model version record."""
    model_name: str
    version: int
    run_id: str
    stage: str  # 'staging', 'production', 'archived'
    created_at: datetime
    description: str
    metrics: Dict[str, float]


# =============================================================================
# LOCAL EXPERIMENT TRACKER (NO MLFLOW DEPENDENCY)
# =============================================================================

class LocalExperimentTracker:
    """
    Local experiment tracking without MLflow dependency.
    
    Stores experiments in JSON files for simplicity.
    Good for development and when MLflow server not available.
    """
    
    def __init__(self, tracking_dir: str = "./mlruns"):
        """
        Initialize local tracker.
        
        Args:
            tracking_dir: Directory to store experiment data
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment: Optional[str] = None
        self.current_run: Optional[ExperimentRun] = None
        
        # Initialize registry
        self.registry_file = self.tracking_dir / "model_registry.json"
        if not self.registry_file.exists():
            self._save_registry({})
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]
    
    def _get_experiment_dir(self, experiment_name: str) -> Path:
        """Get experiment directory."""
        exp_dir = self.tracking_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def _load_registry(self) -> Dict:
        """Load model registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self, registry: Dict):
        """Save model registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def set_experiment(self, experiment_name: str):
        """Set current experiment."""
        self.current_experiment = experiment_name
        self._get_experiment_dir(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new experiment run.
        
        Returns run ID.
        """
        if self.current_experiment is None:
            self.current_experiment = "default"
        
        run_id = self._generate_run_id()
        
        self.current_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.current_experiment,
            model_name=run_name or f"run_{run_id}",
            start_time=datetime.now(),
            end_time=None,
            status='running',
            parameters={},
            metrics={},
            tags=tags or {},
            artifacts=[]
        )
        
        return run_id
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run first.")
        self.current_run.parameters[key] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run first.")
        
        if step is not None:
            key = f"{key}_step_{step}"
        
        self.current_run.metrics[key] = value
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run first.")
        self.current_run.tags[key] = value
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run first.")
        
        # Copy file to artifacts directory
        exp_dir = self._get_experiment_dir(self.current_run.experiment_name)
        artifacts_dir = exp_dir / self.current_run.run_id / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        src_path = Path(local_path)
        if artifact_path:
            dest_path = artifacts_dir / artifact_path
        else:
            dest_path = artifacts_dir / src_path.name
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        import shutil
        shutil.copy2(src_path, dest_path)
        
        self.current_run.artifacts.append(str(dest_path))
    
    def log_model(self, model: Any, artifact_name: str = "model"):
        """Log a model as pickle file."""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run first.")
        
        exp_dir = self._get_experiment_dir(self.current_run.experiment_name)
        model_dir = exp_dir / self.current_run.run_id / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{artifact_name}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.current_run.artifacts.append(str(model_path))
        
        return str(model_path)
    
    def end_run(self, status: str = 'finished'):
        """End the current run."""
        if self.current_run is None:
            return
        
        self.current_run.end_time = datetime.now()
        self.current_run.status = status
        
        # Save run data
        exp_dir = self._get_experiment_dir(self.current_run.experiment_name)
        run_dir = exp_dir / self.current_run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        run_file = run_dir / "run.json"
        
        run_data = asdict(self.current_run)
        run_data['start_time'] = self.current_run.start_time.isoformat()
        run_data['end_time'] = self.current_run.end_time.isoformat() if self.current_run.end_time else None
        
        with open(run_file, 'w') as f:
            json.dump(run_data, f, indent=2, default=str)
        
        self.current_run = None
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get run data by ID."""
        for exp_name in os.listdir(self.tracking_dir):
            exp_dir = self.tracking_dir / exp_name
            if not exp_dir.is_dir():
                continue
            
            run_dir = exp_dir / run_id
            if run_dir.exists():
                run_file = run_dir / "run.json"
                if run_file.exists():
                    with open(run_file, 'r') as f:
                        return json.load(f)
        
        return None
    
    def search_runs(self, experiment_name: Optional[str] = None,
                   filter_string: Optional[str] = None,
                   order_by: Optional[str] = None) -> List[Dict]:
        """
        Search for runs.
        
        Args:
            experiment_name: Filter by experiment
            filter_string: Simple filter like "metrics.sharpe > 1.5"
            order_by: Sort by metric like "metrics.sharpe DESC"
        """
        runs = []
        
        # Get experiments to search
        if experiment_name:
            exp_dirs = [self.tracking_dir / experiment_name]
        else:
            exp_dirs = [d for d in self.tracking_dir.iterdir() if d.is_dir()]
        
        # Collect all runs
        for exp_dir in exp_dirs:
            if not exp_dir.exists():
                continue
            
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                run_file = run_dir / "run.json"
                if run_file.exists():
                    with open(run_file, 'r') as f:
                        run_data = json.load(f)
                        runs.append(run_data)
        
        # Apply filter
        if filter_string:
            filtered_runs = []
            for run in runs:
                try:
                    # Parse simple filter
                    if '>' in filter_string:
                        parts = filter_string.split('>')
                        key = parts[0].strip()
                        value = float(parts[1].strip())
                        
                        if key.startswith('metrics.'):
                            metric_key = key[8:]
                            if run['metrics'].get(metric_key, float('-inf')) > value:
                                filtered_runs.append(run)
                    elif '<' in filter_string:
                        parts = filter_string.split('<')
                        key = parts[0].strip()
                        value = float(parts[1].strip())
                        
                        if key.startswith('metrics.'):
                            metric_key = key[8:]
                            if run['metrics'].get(metric_key, float('inf')) < value:
                                filtered_runs.append(run)
                    else:
                        filtered_runs.append(run)
                except Exception:
                    filtered_runs.append(run)
            
            runs = filtered_runs
        
        # Sort
        if order_by:
            try:
                parts = order_by.split()
                key = parts[0]
                descending = len(parts) > 1 and parts[1].upper() == 'DESC'
                
                if key.startswith('metrics.'):
                    metric_key = key[8:]
                    runs.sort(
                        key=lambda r: r['metrics'].get(metric_key, float('-inf')),
                        reverse=descending
                    )
            except Exception:
                pass
        
        return runs
    
    def register_model(self, model_name: str, run_id: str,
                      stage: str = 'staging',
                      description: str = '') -> int:
        """
        Register a model version.
        
        Returns version number.
        """
        registry = self._load_registry()
        
        if model_name not in registry:
            registry[model_name] = {'versions': [], 'latest_version': 0}
        
        version = registry[model_name]['latest_version'] + 1
        registry[model_name]['latest_version'] = version
        
        # Get run metrics
        run_data = self.get_run(run_id)
        metrics = run_data['metrics'] if run_data else {}
        
        version_data = {
            'version': version,
            'run_id': run_id,
            'stage': stage,
            'created_at': datetime.now().isoformat(),
            'description': description,
            'metrics': metrics
        }
        
        registry[model_name]['versions'].append(version_data)
        self._save_registry(registry)
        
        return version
    
    def transition_model_stage(self, model_name: str, version: int,
                               stage: str):
        """Transition model to new stage."""
        registry = self._load_registry()
        
        if model_name not in registry:
            raise ValueError(f"Model {model_name} not found")
        
        for v in registry[model_name]['versions']:
            if v['version'] == version:
                v['stage'] = stage
                break
        
        self._save_registry(registry)
    
    def get_model_version(self, model_name: str, version: Optional[int] = None,
                         stage: Optional[str] = None) -> Optional[Dict]:
        """Get model version."""
        registry = self._load_registry()
        
        if model_name not in registry:
            return None
        
        versions = registry[model_name]['versions']
        
        if version is not None:
            for v in versions:
                if v['version'] == version:
                    return v
        elif stage is not None:
            for v in reversed(versions):
                if v['stage'] == stage:
                    return v
        else:
            return versions[-1] if versions else None
        
        return None
    
    def load_model(self, model_name: str, version: Optional[int] = None,
                   stage: Optional[str] = None) -> Any:
        """Load a registered model."""
        version_data = self.get_model_version(model_name, version, stage)
        
        if version_data is None:
            raise ValueError(f"Model version not found")
        
        run_id = version_data['run_id']
        run_data = self.get_run(run_id)
        
        if run_data is None:
            raise ValueError(f"Run {run_id} not found")
        
        # Find model artifact
        for artifact in run_data['artifacts']:
            if artifact.endswith('.pkl') and 'model' in artifact:
                with open(artifact, 'rb') as f:
                    return pickle.load(f)
        
        raise ValueError("No model artifact found")


# =============================================================================
# MLFLOW WRAPPER (WHEN MLFLOW AVAILABLE)
# =============================================================================

class MLflowTracker:
    """
    MLflow-based experiment tracking.
    
    Wraps MLflow for consistent API.
    Requires: pip install mlflow
    """
    
    def __init__(self, tracking_uri: Optional[str] = None,
                 experiment_name: str = "trading_models"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow_available = False
        self._mlflow = None
        
        # Fallback tracker
        self._local_tracker = LocalExperimentTracker()
        
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow if available."""
        try:
            import mlflow
            self._mlflow = mlflow
            
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_available = True
            
        except ImportError:
            print("MLflow not installed. Using local tracker.")
            self._mlflow_available = False
    
    def set_experiment(self, experiment_name: str):
        """Set current experiment."""
        self.experiment_name = experiment_name
        
        if self._mlflow_available:
            self._mlflow.set_experiment(experiment_name)
        else:
            self._local_tracker.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new run."""
        if self._mlflow_available:
            run = self._mlflow.start_run(run_name=run_name, tags=tags)
            return run.info.run_id
        else:
            return self._local_tracker.start_run(run_name, tags)
    
    def log_param(self, key: str, value: Any):
        """Log parameter."""
        if self._mlflow_available:
            self._mlflow.log_param(key, value)
        else:
            self._local_tracker.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        if self._mlflow_available:
            self._mlflow.log_params(params)
        else:
            self._local_tracker.log_params(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log metric."""
        if self._mlflow_available:
            self._mlflow.log_metric(key, value, step=step)
        else:
            self._local_tracker.log_metric(key, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        if self._mlflow_available:
            self._mlflow.log_metrics(metrics, step=step)
        else:
            self._local_tracker.log_metrics(metrics, step)
    
    def set_tag(self, key: str, value: str):
        """Set tag."""
        if self._mlflow_available:
            self._mlflow.set_tag(key, value)
        else:
            self._local_tracker.set_tag(key, value)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact."""
        if self._mlflow_available:
            self._mlflow.log_artifact(local_path, artifact_path)
        else:
            self._local_tracker.log_artifact(local_path, artifact_path)
    
    def log_model(self, model: Any, artifact_name: str = "model",
                  conda_env: Optional[Dict] = None):
        """Log model."""
        if self._mlflow_available:
            # Determine model type and use appropriate flavor
            try:
                if hasattr(model, 'predict'):
                    self._mlflow.sklearn.log_model(model, artifact_name)
                else:
                    # Fallback to pickle
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                        pickle.dump(model, f)
                        self._mlflow.log_artifact(f.name, artifact_name)
            except Exception as e:
                print(f"Model logging error: {e}")
                # Fallback
                self._local_tracker.log_model(model, artifact_name)
        else:
            return self._local_tracker.log_model(model, artifact_name)
    
    def end_run(self, status: str = 'FINISHED'):
        """End run."""
        if self._mlflow_available:
            self._mlflow.end_run(status=status)
        else:
            self._local_tracker.end_run(status.lower())
    
    def search_runs(self, experiment_name: Optional[str] = None,
                   filter_string: Optional[str] = None,
                   order_by: Optional[str] = None) -> pd.DataFrame:
        """Search runs."""
        if self._mlflow_available:
            exp = experiment_name or self.experiment_name
            experiment = self._mlflow.get_experiment_by_name(exp)
            
            if experiment is None:
                return pd.DataFrame()
            
            order_by_list = [order_by] if order_by else None
            
            return self._mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=order_by_list
            )
        else:
            runs = self._local_tracker.search_runs(experiment_name, filter_string, order_by)
            return pd.DataFrame(runs)
    
    def register_model(self, model_name: str, run_id: str,
                      stage: str = 'Staging') -> int:
        """Register model."""
        if self._mlflow_available:
            model_uri = f"runs:/{run_id}/model"
            result = self._mlflow.register_model(model_uri, model_name)
            
            # Transition to stage
            client = self._mlflow.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage=stage
            )
            
            return result.version
        else:
            return self._local_tracker.register_model(
                model_name, run_id, stage.lower()
            )


# =============================================================================
# TRADING MODEL TRACKER
# =============================================================================

class TradingModelTracker:
    """
    Specialized tracker for trading models.
    
    Adds trading-specific metrics and validations.
    """
    
    def __init__(self, tracker: Optional[Union[LocalExperimentTracker, MLflowTracker]] = None):
        """
        Initialize trading model tracker.
        
        Args:
            tracker: Base experiment tracker (defaults to local)
        """
        self.tracker = tracker or LocalExperimentTracker()
    
    def log_backtest_results(self, results: Dict[str, Any]):
        """
        Log backtest results.
        
        Expected keys: sharpe, total_return, max_drawdown, win_rate, etc.
        """
        # Standard metrics
        standard_metrics = [
            'sharpe', 'sharpe_ratio', 'total_return', 'annualized_return',
            'max_drawdown', 'calmar_ratio', 'sortino_ratio', 'win_rate',
            'profit_factor', 'avg_trade', 'n_trades', 'volatility'
        ]
        
        metrics = {}
        for key in standard_metrics:
            if key in results:
                metrics[key] = float(results[key])
        
        self.tracker.log_metrics(metrics)
        
        # Log additional info as params
        params = {}
        for key, value in results.items():
            if key not in standard_metrics:
                if isinstance(value, (int, float, str, bool)):
                    params[key] = value
        
        if params:
            self.tracker.log_params(params)
    
    def log_strategy_config(self, config: Dict[str, Any]):
        """Log strategy configuration."""
        self.tracker.log_params(config)
    
    def log_equity_curve(self, equity_curve: pd.Series, filename: str = "equity_curve.csv"):
        """Log equity curve as artifact."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            equity_curve.to_csv(f.name)
            self.tracker.log_artifact(f.name, filename)
    
    def log_trades(self, trades: pd.DataFrame, filename: str = "trades.csv"):
        """Log trades as artifact."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            trades.to_csv(f.name)
            self.tracker.log_artifact(f.name, filename)
    
    def compare_models(self, model_names: List[str],
                      metric: str = 'sharpe') -> pd.DataFrame:
        """Compare registered models by metric."""
        comparison = []
        
        for model_name in model_names:
            try:
                version_data = self.tracker._local_tracker.get_model_version(
                    model_name, stage='production'
                )
                if version_data is None:
                    version_data = self.tracker._local_tracker.get_model_version(
                        model_name, stage='staging'
                    )
                
                if version_data:
                    comparison.append({
                        'model': model_name,
                        'version': version_data['version'],
                        'stage': version_data['stage'],
                        metric: version_data['metrics'].get(metric, np.nan)
                    })
            except Exception as e:
                print(f"Error comparing {model_name}: {e}")
        
        return pd.DataFrame(comparison)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MLFLOW EXPERIMENT TRACKING")
    print("="*70)
    
    # 1. Local Tracker Demo
    print("\n1. Local Experiment Tracker")
    print("-" * 50)
    
    tracker = LocalExperimentTracker("./mlruns_demo")
    
    # Create experiment
    tracker.set_experiment("momentum_strategy")
    
    # Start run
    run_id = tracker.start_run(
        run_name="momentum_v1",
        tags={"author": "quant-bot", "version": "1.0"}
    )
    print(f"   Started run: {run_id}")
    
    # Log parameters
    params = {
        "lookback": 20,
        "threshold": 0.02,
        "stop_loss": 0.05,
        "take_profit": 0.10
    }
    tracker.log_params(params)
    print(f"   Logged parameters: {list(params.keys())}")
    
    # Log metrics
    metrics = {
        "sharpe": 1.85,
        "total_return": 0.42,
        "max_drawdown": 0.15,
        "win_rate": 0.58,
        "n_trades": 150
    }
    tracker.log_metrics(metrics)
    print(f"   Logged metrics: {list(metrics.keys())}")
    
    # Log a simple "model"
    simple_model = {"type": "momentum", "params": params}
    model_path = tracker.log_model(simple_model, "momentum_model")
    print(f"   Logged model to: {model_path}")
    
    # End run
    tracker.end_run()
    print("   Run completed")
    
    # 2. Search runs
    print("\n2. Search Runs")
    print("-" * 50)
    
    # Create more runs
    for i in range(3):
        tracker.start_run(run_name=f"momentum_v{i+2}")
        tracker.log_params({"lookback": 20 + i*5, "version": i+2})
        tracker.log_metrics({
            "sharpe": 1.5 + np.random.rand() * 0.5,
            "max_drawdown": 0.1 + np.random.rand() * 0.1
        })
        tracker.log_model({"version": i+2}, f"model_v{i+2}")
        tracker.end_run()
    
    # Search for high Sharpe runs
    runs = tracker.search_runs(
        experiment_name="momentum_strategy",
        filter_string="metrics.sharpe > 1.6",
        order_by="metrics.sharpe DESC"
    )
    
    print(f"   Found {len(runs)} runs with Sharpe > 1.6")
    for run in runs[:3]:
        print(f"     Run {run['run_id'][:8]}: Sharpe={run['metrics'].get('sharpe', 'N/A'):.2f}")
    
    # 3. Model Registry
    print("\n3. Model Registry")
    print("-" * 50)
    
    if runs:
        best_run = runs[0]
        
        # Register best model
        version = tracker.register_model(
            model_name="momentum_best",
            run_id=best_run['run_id'],
            stage="staging",
            description="Best momentum strategy"
        )
        print(f"   Registered momentum_best version {version} (staging)")
        
        # Promote to production
        tracker.transition_model_stage("momentum_best", version, "production")
        print(f"   Promoted to production")
        
        # Get production model
        prod_model = tracker.get_model_version("momentum_best", stage="production")
        print(f"   Production model: version {prod_model['version']}, "
              f"Sharpe={prod_model['metrics'].get('sharpe', 'N/A'):.2f}")
    
    # 4. Trading Model Tracker
    print("\n4. Trading Model Tracker")
    print("-" * 50)
    
    trading_tracker = TradingModelTracker(LocalExperimentTracker("./mlruns_trading"))
    trading_tracker.tracker.set_experiment("stat_arb_btc")
    trading_tracker.tracker.start_run(run_name="stat_arb_v1")
    
    # Log strategy config
    config = {
        "pair": "BTC/USDT",
        "z_score_entry": 2.0,
        "z_score_exit": 0.5,
        "lookback_hours": 24
    }
    trading_tracker.log_strategy_config(config)
    print(f"   Logged strategy config")
    
    # Log backtest results
    backtest_results = {
        "sharpe": 2.1,
        "total_return": 0.65,
        "max_drawdown": 0.12,
        "win_rate": 0.62,
        "profit_factor": 1.8,
        "n_trades": 89,
        "avg_holding_hours": 4.5
    }
    trading_tracker.log_backtest_results(backtest_results)
    print(f"   Logged backtest results")
    
    trading_tracker.tracker.end_run()
    
    print("\n" + "="*70)
    print("PRODUCTION SETUP")
    print("="*70)
    print("""
For production MLflow tracking:

1. Install MLflow:
   pip install mlflow

2. Start MLflow Server:
   mlflow server --backend-store-uri sqlite:///mlflow.db \\
                 --default-artifact-root ./mlartifacts \\
                 --host 0.0.0.0 --port 5000

3. Connect from code:
   tracker = MLflowTracker(
       tracking_uri="http://localhost:5000",
       experiment_name="my_strategy"
   )

4. Model Registry Workflow:
   - Log best model from backtests
   - Register as 'staging'
   - Validate with walk-forward
   - Promote to 'production'
   - Load in live trading

5. Integration with strategies:
   from mlflow_tracking import TradingModelTracker
   
   tracker = TradingModelTracker()
   tracker.tracker.start_run(run_name="my_strategy")
   
   # After backtest
   tracker.log_backtest_results(backtest_results)
   tracker.log_equity_curve(equity)
   tracker.log_trades(trades)
   
   tracker.tracker.end_run()
""")
    
    # Cleanup demo directories
    import shutil
    try:
        shutil.rmtree("./mlruns_demo")
        shutil.rmtree("./mlruns_trading")
    except Exception:
        pass
