"""
Online Learning Guardrails - Safe Model Updates
================================================
CRITICAL DELIVERABLE: Prevent model degradation during live trading.

Right now your online-learning loop updates even when performance worsens.
This is EXTREMELY DANGEROUS.

This module provides:
1. Performance validation gate - Only update if model improves
2. "Hold last good model" fallback - Always keep a working model
3. Drift-aware update throttle - Don't update during regime shifts
4. Rollback mechanism - Revert to previous model on failure

Without these guardrails, your bot can:
- Forget profitable behavior
- Overfit to market noise
- Drift into loss spirals
- Reinforce bad trades
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import json
import pickle
import hashlib
import copy
import threading
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelCheckpoint:
    """Checkpoint of a model state."""
    checkpoint_id: str
    timestamp: datetime
    model_state: Any  # Serialized model weights/parameters
    
    # Performance at checkpoint
    sharpe_ratio: float
    accuracy: float
    total_return: float
    max_drawdown: float
    
    # Validation metrics
    validation_sharpe: float = 0.0
    validation_accuracy: float = 0.0
    
    # Metadata
    n_training_samples: int = 0
    regime: str = ""
    drift_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp.isoformat(),
            'sharpe_ratio': self.sharpe_ratio,
            'accuracy': self.accuracy,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'validation_sharpe': self.validation_sharpe,
            'validation_accuracy': self.validation_accuracy,
            'n_training_samples': self.n_training_samples,
            'regime': self.regime,
            'drift_score': self.drift_score
        }


@dataclass
class UpdateDecision:
    """Decision about whether to apply an update."""
    should_update: bool
    reason: str
    confidence: float  # 0-1 confidence in decision
    
    # Metrics comparison
    current_performance: Dict[str, float]
    proposed_performance: Dict[str, float]
    
    # Risk assessment
    risk_level: str  # 'low', 'medium', 'high'
    rollback_available: bool


@dataclass
class GuardrailConfig:
    """Configuration for online learning guardrails."""
    
    # Performance thresholds
    min_sharpe_for_update: float = 0.5  # Don't update if new Sharpe < this
    max_sharpe_degradation: float = 0.3  # Don't update if Sharpe drops more than this
    min_accuracy_for_update: float = 0.52  # Need better than random
    max_drawdown_for_update: float = 0.15  # Don't update if DD > 15%
    
    # Validation requirements
    min_validation_samples: int = 100
    validation_lookback_days: int = 14
    require_oos_improvement: bool = True  # Must improve on out-of-sample
    
    # Throttling
    min_update_interval_hours: float = 1.0
    max_updates_per_day: int = 24
    cooldown_after_rollback_hours: float = 6.0
    
    # Drift controls
    max_drift_for_update: float = 0.3  # PSI threshold
    pause_during_regime_change: bool = True
    
    # Rollback
    max_checkpoints: int = 10
    auto_rollback_on_drawdown: float = 0.10  # Rollback if DD > 10%
    
    # Safety
    require_manual_approval: bool = False
    alert_on_degradation: bool = True


class PerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.returns = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Metrics history
        self.sharpe_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        self.drawdown_history = deque(maxlen=100)
    
    def record(self, prediction: float, actual: float, 
               return_val: float, timestamp: datetime = None):
        """Record a prediction result."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.returns.append(return_val)
        self.timestamps.append(timestamp or datetime.now())
    
    def get_metrics(self, lookback: int = None) -> Dict[str, float]:
        """Calculate current performance metrics."""
        if len(self.predictions) < 10:
            return {
                'sharpe_ratio': 0.0,
                'accuracy': 0.5,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'n_samples': len(self.predictions)
            }
        
        lookback = lookback or len(self.predictions)
        
        preds = np.array(list(self.predictions)[-lookback:])
        acts = np.array(list(self.actuals)[-lookback:])
        rets = np.array(list(self.returns)[-lookback:])
        
        # Accuracy (for classification) or correlation (for regression)
        if np.std(preds) > 0 and np.std(acts) > 0:
            if set(acts).issubset({-1, 0, 1}):
                # Classification
                accuracy = np.mean(np.sign(preds) == np.sign(acts))
            else:
                # Regression - use correlation
                accuracy = np.corrcoef(preds, acts)[0, 1]
        else:
            accuracy = 0.5
        
        # Sharpe ratio
        if np.std(rets) > 0:
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Total return
        total_return = np.sum(rets)
        
        # Max drawdown
        cumulative = np.cumsum(rets)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        metrics = {
            'sharpe_ratio': float(sharpe),
            'accuracy': float(accuracy),
            'total_return': float(total_return),
            'max_drawdown': float(max_dd),
            'n_samples': len(preds)
        }
        
        # Update history
        self.sharpe_history.append(sharpe)
        self.accuracy_history.append(accuracy)
        self.drawdown_history.append(max_dd)
        
        return metrics
    
    def get_trend(self, metric: str, lookback: int = 20) -> float:
        """Get trend of a metric (positive = improving)."""
        if metric == 'sharpe_ratio':
            history = list(self.sharpe_history)
        elif metric == 'accuracy':
            history = list(self.accuracy_history)
        elif metric == 'max_drawdown':
            history = list(self.drawdown_history)
        else:
            return 0.0
        
        if len(history) < lookback:
            return 0.0
        
        recent = history[-lookback:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        # For drawdown, negative slope is good
        if metric == 'max_drawdown':
            slope = -slope
        
        return float(slope)


class ValidationGate:
    """
    Gate that determines if model update should be applied.
    
    This is the CRITICAL safety mechanism.
    """
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.last_update_time: Optional[datetime] = None
        self.updates_today = 0
        self.last_rollback_time: Optional[datetime] = None
        self.current_day = datetime.now().date()
        
    def evaluate_update(
        self,
        current_metrics: Dict[str, float],
        proposed_metrics: Dict[str, float],
        drift_score: float = 0.0,
        regime_changed: bool = False
    ) -> UpdateDecision:
        """
        Evaluate whether a model update should be applied.
        
        Args:
            current_metrics: Performance of current model
            proposed_metrics: Performance of proposed new model
            drift_score: Current drift score (PSI)
            regime_changed: Whether regime has changed recently
        
        Returns:
            UpdateDecision with recommendation
        """
        reasons = []
        should_update = True
        risk_level = "low"
        
        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self.current_day:
            self.current_day = today
            self.updates_today = 0
        
        # Check 1: Update interval
        if self.last_update_time:
            hours_since = (datetime.now() - self.last_update_time).total_seconds() / 3600
            if hours_since < self.config.min_update_interval_hours:
                should_update = False
                reasons.append(f"Too soon since last update ({hours_since:.1f}h < {self.config.min_update_interval_hours}h)")
        
        # Check 2: Daily limit
        if self.updates_today >= self.config.max_updates_per_day:
            should_update = False
            reasons.append(f"Daily update limit reached ({self.updates_today})")
        
        # Check 3: Rollback cooldown
        if self.last_rollback_time:
            hours_since_rollback = (datetime.now() - self.last_rollback_time).total_seconds() / 3600
            if hours_since_rollback < self.config.cooldown_after_rollback_hours:
                should_update = False
                reasons.append(f"In rollback cooldown ({hours_since_rollback:.1f}h)")
                risk_level = "medium"
        
        # Check 4: Minimum Sharpe
        proposed_sharpe = proposed_metrics.get('sharpe_ratio', 0)
        if proposed_sharpe < self.config.min_sharpe_for_update:
            should_update = False
            reasons.append(f"Proposed Sharpe too low ({proposed_sharpe:.2f} < {self.config.min_sharpe_for_update})")
            risk_level = "high"
        
        # Check 5: Sharpe degradation
        current_sharpe = current_metrics.get('sharpe_ratio', 0)
        if current_sharpe > 0:
            sharpe_change = (proposed_sharpe - current_sharpe) / current_sharpe
            if sharpe_change < -self.config.max_sharpe_degradation:
                should_update = False
                reasons.append(f"Sharpe degradation too high ({sharpe_change:.1%})")
                risk_level = "high"
        
        # Check 6: Minimum accuracy
        proposed_accuracy = proposed_metrics.get('accuracy', 0.5)
        if proposed_accuracy < self.config.min_accuracy_for_update:
            should_update = False
            reasons.append(f"Accuracy too low ({proposed_accuracy:.2%})")
        
        # Check 7: Maximum drawdown
        proposed_dd = proposed_metrics.get('max_drawdown', 0)
        if proposed_dd > self.config.max_drawdown_for_update:
            should_update = False
            reasons.append(f"Drawdown too high ({proposed_dd:.1%})")
            risk_level = "high"
        
        # Check 8: Drift threshold
        if drift_score > self.config.max_drift_for_update:
            should_update = False
            reasons.append(f"Drift too high ({drift_score:.2f} > {self.config.max_drift_for_update})")
            risk_level = "medium"
        
        # Check 9: Regime change pause
        if regime_changed and self.config.pause_during_regime_change:
            should_update = False
            reasons.append("Paused during regime change")
            risk_level = "medium"
        
        # Check 10: Out-of-sample improvement
        if self.config.require_oos_improvement:
            val_sharpe_current = current_metrics.get('validation_sharpe', 0)
            val_sharpe_proposed = proposed_metrics.get('validation_sharpe', 0)
            if val_sharpe_proposed <= val_sharpe_current:
                should_update = False
                reasons.append(f"No OOS improvement ({val_sharpe_proposed:.2f} <= {val_sharpe_current:.2f})")
        
        # Check 11: Minimum validation samples
        n_samples = proposed_metrics.get('n_samples', 0)
        if n_samples < self.config.min_validation_samples:
            should_update = False
            reasons.append(f"Insufficient validation samples ({n_samples})")
        
        # Calculate confidence
        confidence = 1.0
        if not should_update:
            confidence = 0.0
        else:
            # Reduce confidence based on risk factors
            if drift_score > 0.1:
                confidence -= 0.2
            if proposed_sharpe < 1.0:
                confidence -= 0.2
            if proposed_dd > 0.05:
                confidence -= 0.2
            confidence = max(0.1, confidence)
        
        if not reasons:
            reasons.append("All validation checks passed")
        
        return UpdateDecision(
            should_update=should_update,
            reason="; ".join(reasons),
            confidence=confidence,
            current_performance=current_metrics,
            proposed_performance=proposed_metrics,
            risk_level=risk_level,
            rollback_available=True
        )
    
    def record_update(self):
        """Record that an update was applied."""
        self.last_update_time = datetime.now()
        self.updates_today += 1
    
    def record_rollback(self):
        """Record that a rollback occurred."""
        self.last_rollback_time = datetime.now()


class CheckpointManager:
    """
    Manage model checkpoints for rollback capability.
    
    Always maintains a "last known good" model.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./model_checkpoints",
        max_checkpoints: int = 10
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        self.checkpoints: List[ModelCheckpoint] = []
        self.best_checkpoint: Optional[ModelCheckpoint] = None
        self._lock = threading.Lock()
        
        # Load existing checkpoints
        self._load_checkpoints()
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID."""
        return hashlib.md5(
            f"{datetime.now().isoformat()}{np.random.random()}".encode()
        ).hexdigest()[:12]
    
    def _load_checkpoints(self):
        """Load existing checkpoints from disk."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        for f in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime)[-self.max_checkpoints:]:
            try:
                with open(f, 'rb') as file:
                    checkpoint = pickle.load(file)
                    self.checkpoints.append(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {f}: {e}")
        
        # Find best checkpoint
        if self.checkpoints:
            self.best_checkpoint = max(
                self.checkpoints,
                key=lambda c: c.sharpe_ratio
            )
    
    def save_checkpoint(
        self,
        model_state: Any,
        metrics: Dict[str, float],
        regime: str = "",
        drift_score: float = 0.0
    ) -> ModelCheckpoint:
        """
        Save a model checkpoint.
        
        Args:
            model_state: Model weights/parameters to save
            metrics: Current performance metrics
            regime: Current market regime
            drift_score: Current drift score
        
        Returns:
            ModelCheckpoint that was saved
        """
        checkpoint = ModelCheckpoint(
            checkpoint_id=self._generate_checkpoint_id(),
            timestamp=datetime.now(),
            model_state=copy.deepcopy(model_state),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            accuracy=metrics.get('accuracy', 0.5),
            total_return=metrics.get('total_return', 0),
            max_drawdown=metrics.get('max_drawdown', 0),
            validation_sharpe=metrics.get('validation_sharpe', 0),
            validation_accuracy=metrics.get('validation_accuracy', 0.5),
            n_training_samples=metrics.get('n_samples', 0),
            regime=regime,
            drift_score=drift_score
        )
        
        with self._lock:
            # Add to list
            self.checkpoints.append(checkpoint)
            
            # Update best if better
            if (self.best_checkpoint is None or 
                checkpoint.sharpe_ratio > self.best_checkpoint.sharpe_ratio):
                self.best_checkpoint = checkpoint
            
            # Save to disk
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint.checkpoint_id}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Prune old checkpoints (but always keep best)
            while len(self.checkpoints) > self.max_checkpoints:
                oldest = self.checkpoints[0]
                if oldest.checkpoint_id != self.best_checkpoint.checkpoint_id:
                    self.checkpoints.pop(0)
                    # Delete from disk
                    old_file = self.checkpoint_dir / f"checkpoint_{oldest.checkpoint_id}.pkl"
                    if old_file.exists():
                        old_file.unlink()
                else:
                    # Don't delete best, remove second oldest
                    if len(self.checkpoints) > 1:
                        self.checkpoints.pop(1)
        
        logger.info(f"Saved checkpoint {checkpoint.checkpoint_id} (Sharpe: {checkpoint.sharpe_ratio:.2f})")
        
        return checkpoint
    
    def get_latest(self) -> Optional[ModelCheckpoint]:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None
    
    def get_best(self) -> Optional[ModelCheckpoint]:
        """Get the best performing checkpoint."""
        return self.best_checkpoint
    
    def get_previous(self, n: int = 1) -> Optional[ModelCheckpoint]:
        """Get the nth previous checkpoint."""
        if len(self.checkpoints) > n:
            return self.checkpoints[-(n+1)]
        return None
    
    def rollback_to(self, checkpoint_id: str) -> Optional[Any]:
        """
        Rollback to a specific checkpoint.
        
        Returns the model state from that checkpoint.
        """
        for cp in self.checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                logger.info(f"Rolling back to checkpoint {checkpoint_id}")
                return cp.model_state
        
        logger.error(f"Checkpoint {checkpoint_id} not found")
        return None
    
    def rollback_to_best(self) -> Optional[Any]:
        """Rollback to the best performing checkpoint."""
        if self.best_checkpoint:
            logger.info(f"Rolling back to best checkpoint {self.best_checkpoint.checkpoint_id}")
            return self.best_checkpoint.model_state
        return None


class OnlineLearningGuardrail:
    """
    Master guardrail class for safe online learning.
    
    Use this to wrap your online learning model updates.
    """
    
    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
        checkpoint_dir: str = "./model_checkpoints",
        alert_callback: Optional[Callable] = None
    ):
        self.config = config or GuardrailConfig()
        
        # Components
        self.performance_tracker = PerformanceTracker()
        self.validation_gate = ValidationGate(self.config)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=self.config.max_checkpoints
        )
        
        self.alert_callback = alert_callback
        
        # State
        self.current_model_state: Any = None
        self.is_paused = False
        self.pause_reason: str = ""
        
        # Statistics
        self.total_updates_proposed = 0
        self.total_updates_applied = 0
        self.total_rollbacks = 0
        
        logger.info("OnlineLearningGuardrail initialized")
    
    def _alert(self, message: str, severity: str = "INFO"):
        """Send alert."""
        logger.log(getattr(logging, severity), message)
        if self.alert_callback:
            self.alert_callback(message, severity)
    
    def record_prediction(
        self,
        prediction: float,
        actual: float,
        return_val: float
    ):
        """Record a prediction and its outcome."""
        self.performance_tracker.record(prediction, actual, return_val)
        
        # Check for auto-rollback trigger
        metrics = self.performance_tracker.get_metrics(lookback=50)
        if metrics['max_drawdown'] > self.config.auto_rollback_on_drawdown:
            self._alert(
                f"Auto-rollback triggered: Drawdown {metrics['max_drawdown']:.1%}",
                "WARNING"
            )
            self._trigger_auto_rollback()
    
    def propose_update(
        self,
        new_model_state: Any,
        validation_metrics: Dict[str, float],
        drift_score: float = 0.0,
        regime: str = "",
        regime_changed: bool = False
    ) -> UpdateDecision:
        """
        Propose a model update.
        
        Args:
            new_model_state: Proposed new model state
            validation_metrics: Performance metrics on validation data
            drift_score: Current drift score
            regime: Current market regime
            regime_changed: Whether regime has changed
        
        Returns:
            UpdateDecision with recommendation
        """
        self.total_updates_proposed += 1
        
        if self.is_paused:
            return UpdateDecision(
                should_update=False,
                reason=f"Updates paused: {self.pause_reason}",
                confidence=0.0,
                current_performance={},
                proposed_performance=validation_metrics,
                risk_level="high",
                rollback_available=True
            )
        
        # Get current performance
        current_metrics = self.performance_tracker.get_metrics()
        
        # Add validation metrics to proposed
        validation_metrics['validation_sharpe'] = validation_metrics.get('sharpe_ratio', 0)
        validation_metrics['validation_accuracy'] = validation_metrics.get('accuracy', 0.5)
        
        # Evaluate update
        decision = self.validation_gate.evaluate_update(
            current_metrics=current_metrics,
            proposed_metrics=validation_metrics,
            drift_score=drift_score,
            regime_changed=regime_changed
        )
        
        logger.info(
            f"Update proposal: should_update={decision.should_update}, "
            f"confidence={decision.confidence:.2f}, reason={decision.reason}"
        )
        
        # If update should be applied, save checkpoint first
        if decision.should_update:
            if self.current_model_state is not None:
                self.checkpoint_manager.save_checkpoint(
                    self.current_model_state,
                    current_metrics,
                    regime=regime,
                    drift_score=drift_score
                )
        
        return decision
    
    def apply_update(
        self,
        new_model_state: Any,
        metrics: Dict[str, float],
        regime: str = ""
    ):
        """
        Apply a model update (after validation passed).
        
        Call this after propose_update returns should_update=True.
        """
        self.current_model_state = copy.deepcopy(new_model_state)
        self.validation_gate.record_update()
        self.total_updates_applied += 1
        
        # Save new checkpoint
        self.checkpoint_manager.save_checkpoint(
            new_model_state,
            metrics,
            regime=regime
        )
        
        logger.info(f"Model update applied (total: {self.total_updates_applied})")
    
    def rollback(self, to_best: bool = True) -> Optional[Any]:
        """
        Rollback to a previous model state.
        
        Args:
            to_best: If True, rollback to best checkpoint. Otherwise previous.
        
        Returns:
            Model state to restore
        """
        self.total_rollbacks += 1
        self.validation_gate.record_rollback()
        
        if to_best:
            model_state = self.checkpoint_manager.rollback_to_best()
        else:
            prev = self.checkpoint_manager.get_previous()
            model_state = prev.model_state if prev else None
        
        if model_state:
            self.current_model_state = model_state
            self._alert(f"Rollback executed (total: {self.total_rollbacks})", "WARNING")
        
        return model_state
    
    def _trigger_auto_rollback(self):
        """Trigger automatic rollback due to performance degradation."""
        model_state = self.rollback(to_best=True)
        
        if model_state is None:
            # No checkpoint to rollback to, pause updates
            self.pause_updates("No checkpoint available for rollback")
    
    def pause_updates(self, reason: str):
        """Pause all model updates."""
        self.is_paused = True
        self.pause_reason = reason
        self._alert(f"Updates paused: {reason}", "WARNING")
    
    def resume_updates(self):
        """Resume model updates."""
        self.is_paused = False
        self.pause_reason = ""
        logger.info("Updates resumed")
    
    def get_status(self) -> Dict:
        """Get current guardrail status."""
        return {
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'total_updates_proposed': self.total_updates_proposed,
            'total_updates_applied': self.total_updates_applied,
            'total_rollbacks': self.total_rollbacks,
            'update_rate': self.total_updates_applied / max(1, self.total_updates_proposed),
            'current_metrics': self.performance_tracker.get_metrics(),
            'best_checkpoint': self.checkpoint_manager.get_best().to_dict() if self.checkpoint_manager.get_best() else None,
            'n_checkpoints': len(self.checkpoint_manager.checkpoints)
        }
    
    def get_performance_report(self) -> Dict:
        """Get performance report."""
        metrics = self.performance_tracker.get_metrics()
        
        return {
            'current_metrics': metrics,
            'sharpe_trend': self.performance_tracker.get_trend('sharpe_ratio'),
            'accuracy_trend': self.performance_tracker.get_trend('accuracy'),
            'drawdown_trend': self.performance_tracker.get_trend('max_drawdown'),
            'update_statistics': {
                'proposed': self.total_updates_proposed,
                'applied': self.total_updates_applied,
                'rejected': self.total_updates_proposed - self.total_updates_applied,
                'rollbacks': self.total_rollbacks
            },
            'checkpoint_status': {
                'total': len(self.checkpoint_manager.checkpoints),
                'best_sharpe': self.checkpoint_manager.best_checkpoint.sharpe_ratio if self.checkpoint_manager.best_checkpoint else 0
            }
        }


# =============================================================================
# SAFE ONLINE LEARNER WRAPPER
# =============================================================================

class SafeOnlineLearner:
    """
    Wrapper that adds guardrails to any online learning model.
    
    Usage:
        model = YourOnlineModel()
        safe_model = SafeOnlineLearner(model, guardrail_config)
        
        # Training loop
        for data in stream:
            prediction = safe_model.predict(data)
            actual = get_actual(data)
            safe_model.update(data, actual)  # Guardrails applied automatically
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[GuardrailConfig] = None,
        get_state_fn: Optional[Callable] = None,
        set_state_fn: Optional[Callable] = None,
        checkpoint_dir: str = "./model_checkpoints"
    ):
        """
        Initialize safe online learner.
        
        Args:
            model: The underlying online learning model
            config: Guardrail configuration
            get_state_fn: Function to get model state (default: model.get_params())
            set_state_fn: Function to set model state (default: model.set_params())
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.config = config or GuardrailConfig()
        
        # State management functions
        self.get_state_fn = get_state_fn or self._default_get_state
        self.set_state_fn = set_state_fn or self._default_set_state
        
        # Guardrail
        self.guardrail = OnlineLearningGuardrail(
            config=self.config,
            checkpoint_dir=checkpoint_dir
        )
        
        # Initialize with current model state
        self.guardrail.current_model_state = self.get_state_fn(self.model)
        
        # Buffer for validation
        self.update_buffer = deque(maxlen=self.config.min_validation_samples * 2)
        
        logger.info("SafeOnlineLearner initialized")
    
    def _default_get_state(self, model) -> Any:
        """Default state getter."""
        if hasattr(model, 'get_params'):
            return model.get_params()
        elif hasattr(model, 'coef_'):
            return {'coef': model.coef_.copy(), 'intercept': model.intercept_.copy()}
        else:
            return copy.deepcopy(model)
    
    def _default_set_state(self, model, state) -> Any:
        """Default state setter."""
        if hasattr(model, 'set_params'):
            model.set_params(**state)
        elif hasattr(model, 'coef_'):
            model.coef_ = state['coef'].copy()
            model.intercept_ = state['intercept'].copy()
        return model
    
    def predict(self, X) -> Any:
        """Make prediction using current model."""
        return self.model.predict(X)
    
    def update(
        self,
        X,
        y,
        return_val: float = 0.0,
        drift_score: float = 0.0,
        regime: str = "",
        regime_changed: bool = False
    ) -> bool:
        """
        Propose and potentially apply model update.
        
        Args:
            X: Training features
            y: Training labels
            return_val: Return for this prediction (for tracking)
            drift_score: Current drift score
            regime: Current market regime
            regime_changed: Whether regime has changed
        
        Returns:
            True if update was applied, False otherwise
        """
        # Record for tracking
        pred = self.predict(X)
        if hasattr(pred, '__len__'):
            pred = pred[0] if len(pred) > 0 else 0
        if hasattr(y, '__len__'):
            y_val = y[0] if len(y) > 0 else y
        else:
            y_val = y
        
        self.guardrail.record_prediction(float(pred), float(y_val), return_val)
        
        # Buffer the update
        self.update_buffer.append((X, y, return_val))
        
        # Check if we have enough data for validation
        if len(self.update_buffer) < self.config.min_validation_samples:
            return False
        
        # Create a copy of model and train on buffered data
        proposed_model = copy.deepcopy(self.model)
        for X_buf, y_buf, _ in self.update_buffer:
            if hasattr(proposed_model, 'partial_fit'):
                proposed_model.partial_fit(X_buf.reshape(1, -1) if len(X_buf.shape) == 1 else X_buf, 
                                          np.array([y_buf]) if not hasattr(y_buf, '__len__') else y_buf)
            elif hasattr(proposed_model, 'fit'):
                # For non-incremental models, refit on all data
                X_all = np.vstack([x for x, _, _ in self.update_buffer])
                y_all = np.array([y for _, y, _ in self.update_buffer])
                proposed_model.fit(X_all, y_all)
                break
        
        # Get proposed model state
        proposed_state = self.get_state_fn(proposed_model)
        
        # Calculate validation metrics on held-out portion
        n_val = len(self.update_buffer) // 4
        val_data = list(self.update_buffer)[-n_val:]
        
        val_preds = []
        val_actuals = []
        val_returns = []
        
        for X_val, y_val, ret_val in val_data:
            pred = proposed_model.predict(X_val.reshape(1, -1) if len(X_val.shape) == 1 else X_val)
            val_preds.append(float(pred[0]) if hasattr(pred, '__len__') else float(pred))
            val_actuals.append(float(y_val[0]) if hasattr(y_val, '__len__') else float(y_val))
            val_returns.append(ret_val)
        
        val_preds = np.array(val_preds)
        val_actuals = np.array(val_actuals)
        val_returns = np.array(val_returns)
        
        # Calculate validation metrics
        if np.std(val_returns) > 0:
            val_sharpe = np.mean(val_returns) / np.std(val_returns) * np.sqrt(252)
        else:
            val_sharpe = 0
        
        if len(set(val_actuals)) == 2:  # Binary classification
            val_accuracy = np.mean(np.sign(val_preds) == np.sign(val_actuals))
        else:
            val_accuracy = np.corrcoef(val_preds, val_actuals)[0, 1] if np.std(val_preds) > 0 else 0
        
        validation_metrics = {
            'sharpe_ratio': val_sharpe,
            'accuracy': val_accuracy,
            'total_return': np.sum(val_returns),
            'max_drawdown': np.max(np.maximum.accumulate(np.cumsum(val_returns)) - np.cumsum(val_returns)),
            'n_samples': len(val_data)
        }
        
        # Propose update through guardrail
        decision = self.guardrail.propose_update(
            new_model_state=proposed_state,
            validation_metrics=validation_metrics,
            drift_score=drift_score,
            regime=regime,
            regime_changed=regime_changed
        )
        
        if decision.should_update:
            # Apply update
            self.model = proposed_model
            self.guardrail.apply_update(proposed_state, validation_metrics, regime)
            
            # Clear buffer after successful update
            self.update_buffer.clear()
            
            return True
        
        return False
    
    def force_rollback(self, to_best: bool = True) -> bool:
        """Force a rollback to previous model state."""
        state = self.guardrail.rollback(to_best=to_best)
        
        if state:
            self.model = self.set_state_fn(self.model, state)
            return True
        
        return False
    
    def get_status(self) -> Dict:
        """Get guardrail status."""
        return self.guardrail.get_status()


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate online learning guardrails."""
    
    print("=" * 70)
    print("ONLINE LEARNING GUARDRAILS - DEMO")
    print("=" * 70)
    
    # Create guardrail with config
    config = GuardrailConfig(
        min_sharpe_for_update=0.5,
        max_sharpe_degradation=0.3,
        min_validation_samples=50,
        auto_rollback_on_drawdown=0.10
    )
    
    guardrail = OnlineLearningGuardrail(config=config)
    
    print("\n1. Recording predictions...")
    np.random.seed(42)
    for i in range(100):
        pred = np.random.choice([-1, 1])
        actual = np.random.choice([-1, 1], p=[0.45, 0.55])  # Slightly biased
        ret = pred * actual * 0.01  # 1% per correct prediction
        guardrail.record_prediction(pred, actual, ret)
    
    metrics = guardrail.performance_tracker.get_metrics()
    print(f"   Current metrics: Sharpe={metrics['sharpe_ratio']:.2f}, "
          f"Accuracy={metrics['accuracy']:.2%}")
    
    print("\n2. Proposing model update (should pass)...")
    decision = guardrail.propose_update(
        new_model_state={'weights': [1, 2, 3]},
        validation_metrics={
            'sharpe_ratio': 1.5,
            'accuracy': 0.58,
            'total_return': 0.15,
            'max_drawdown': 0.05,
            'n_samples': 100
        },
        drift_score=0.05
    )
    print(f"   Decision: {decision.should_update}, Reason: {decision.reason}")
    
    if decision.should_update:
        guardrail.apply_update({'weights': [1, 2, 3]}, decision.proposed_performance)
    
    print("\n3. Proposing bad update (should fail)...")
    decision = guardrail.propose_update(
        new_model_state={'weights': [0, 0, 0]},
        validation_metrics={
            'sharpe_ratio': 0.2,  # Too low
            'accuracy': 0.48,    # Below random
            'total_return': -0.10,
            'max_drawdown': 0.20,  # Too high
            'n_samples': 100
        },
        drift_score=0.4  # High drift
    )
    print(f"   Decision: {decision.should_update}, Reason: {decision.reason}")
    
    print("\n4. Testing rollback...")
    model_state = guardrail.rollback(to_best=True)
    print(f"   Rolled back to: {model_state}")
    
    print("\n5. Status report...")
    status = guardrail.get_status()
    print(f"   Updates proposed: {status['total_updates_proposed']}")
    print(f"   Updates applied: {status['total_updates_applied']}")
    print(f"   Rollbacks: {status['total_rollbacks']}")
    print(f"   Checkpoints: {status['n_checkpoints']}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
