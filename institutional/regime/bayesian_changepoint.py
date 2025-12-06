"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         BAYESIAN CHANGEPOINT DETECTOR                                         ║
║                                                                               ║
║  Detects regime shifts with false-alarm penalty                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Problem: HMM can appear to change when only noise increases.

Solution: Bayesian Change-Point detection with:
- Negative Binomial-Inverse Gamma prior
- Bayes Factor threshold with false-alarm penalty
- Freeze-then-retrain workflow

Trigger:
  log BF_{τ, τ−1} > 5 AND out-of-sample log-loss increases > 5%
  
Action:
  1. Freeze weights for 30 min
  2. Retrain on post-τ data only
  3. If new regime persists > 4h, promote to primary model
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
from scipy import stats
import logging

logger = logging.getLogger("Regime.Changepoint")


@dataclass
class ChangePointResult:
    """Result of changepoint detection."""
    detected: bool
    changepoint_idx: Optional[int] = None
    bayes_factor: float = 0.0
    confidence: float = 0.0
    timestamp: Optional[datetime] = None
    description: str = ""
    
    # Pre/post statistics
    pre_mean: Optional[float] = None
    post_mean: Optional[float] = None
    pre_std: Optional[float] = None
    post_std: Optional[float] = None


@dataclass
class RegimeState:
    """Current regime state."""
    regime_id: int
    regime_probability: float
    entropy: float
    start_time: datetime
    observations: int = 0
    mean_return: float = 0.0
    volatility: float = 0.0


class BayesianChangepoint:
    """
    Bayesian Changepoint Detector.
    
    Uses Bayesian inference to detect structural breaks in time series.
    Includes false-alarm penalty to avoid spurious regime changes.
    
    Usage:
        detector = BayesianChangepoint()
        
        # Add observations
        for value in returns:
            detector.add_observation(value)
        
        # Check for changepoint
        result = detector.detect()
        if result.detected:
            print(f"Changepoint at {result.changepoint_idx}")
    """
    
    def __init__(
        self,
        min_segment_length: int = 50,
        bf_threshold: float = 5.0,  # log Bayes Factor threshold
        oos_loss_threshold: float = 0.05,  # 5% OOS loss increase
        hazard_rate: float = 1/250,  # Expected changepoint every 250 obs
    ):
        """
        Initialize changepoint detector.
        
        Args:
            min_segment_length: Minimum observations before/after changepoint
            bf_threshold: Log Bayes Factor threshold for detection
            oos_loss_threshold: OOS loss increase threshold
            hazard_rate: Prior probability of changepoint at each step
        """
        self.min_segment_length = min_segment_length
        self.bf_threshold = bf_threshold
        self.oos_loss_threshold = oos_loss_threshold
        self.hazard_rate = hazard_rate
        
        # Data
        self._data: deque = deque(maxlen=10000)
        self._timestamps: deque = deque(maxlen=10000)
        
        # Prior parameters (Normal-Inverse-Gamma)
        self._mu0 = 0.0      # Prior mean
        self._kappa0 = 1.0   # Prior precision weight
        self._alpha0 = 1.0   # Shape
        self._beta0 = 1.0    # Scale
        
        # State
        self._last_changepoint: Optional[int] = None
        self._pending_changepoint: Optional[ChangePointResult] = None
        self._regime_history: List[RegimeState] = []
        self._current_regime: Optional[RegimeState] = None
        
        logger.info("BayesianChangepoint detector initialized")
    
    def add_observation(
        self,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add an observation."""
        self._data.append(value)
        self._timestamps.append(timestamp or datetime.now())
    
    def add_observations(self, values: np.ndarray) -> None:
        """Add multiple observations."""
        for v in values:
            self.add_observation(v)
    
    def detect(self) -> ChangePointResult:
        """
        Detect changepoint in current data.
        
        Returns:
            ChangePointResult with detection outcome
        """
        data = np.array(self._data)
        n = len(data)
        
        if n < 2 * self.min_segment_length:
            return ChangePointResult(
                detected=False,
                description="Insufficient data"
            )
        
        # Find most likely changepoint
        best_bf = -np.inf
        best_idx = None
        
        for tau in range(self.min_segment_length, n - self.min_segment_length):
            bf = self._compute_bayes_factor(data, tau)
            if bf > best_bf:
                best_bf = bf
                best_idx = tau
        
        # Check threshold
        if best_bf > self.bf_threshold:
            # Additional check: OOS loss
            oos_check = self._check_oos_loss(data, best_idx)
            
            if oos_check:
                # Calculate statistics
                pre_data = data[:best_idx]
                post_data = data[best_idx:]
                
                result = ChangePointResult(
                    detected=True,
                    changepoint_idx=best_idx,
                    bayes_factor=best_bf,
                    confidence=1 - np.exp(-best_bf),
                    timestamp=self._timestamps[best_idx] if self._timestamps else None,
                    description=f"Changepoint detected at index {best_idx}",
                    pre_mean=np.mean(pre_data),
                    post_mean=np.mean(post_data),
                    pre_std=np.std(pre_data),
                    post_std=np.std(post_data),
                )
                
                logger.info(
                    f"Changepoint detected: idx={best_idx}, BF={best_bf:.2f}, "
                    f"pre_mean={result.pre_mean:.4f}, post_mean={result.post_mean:.4f}"
                )
                
                return result
        
        return ChangePointResult(
            detected=False,
            bayes_factor=best_bf,
            description="No changepoint detected"
        )
    
    def _compute_bayes_factor(self, data: np.ndarray, tau: int) -> float:
        """
        Compute log Bayes Factor for changepoint at tau.
        
        BF = P(data | changepoint at tau) / P(data | no changepoint)
        """
        n = len(data)
        
        # Marginal likelihood under H0: no changepoint
        log_ml_h0 = self._marginal_likelihood(data)
        
        # Marginal likelihood under H1: changepoint at tau
        log_ml_h1_pre = self._marginal_likelihood(data[:tau])
        log_ml_h1_post = self._marginal_likelihood(data[tau:])
        log_ml_h1 = log_ml_h1_pre + log_ml_h1_post
        
        # Add prior on changepoint location
        log_prior = np.log(self.hazard_rate) + np.log(1 - self.hazard_rate) * (tau - 1)
        
        # Log Bayes Factor
        log_bf = log_ml_h1 - log_ml_h0 + log_prior
        
        return log_bf
    
    def _marginal_likelihood(self, data: np.ndarray) -> float:
        """
        Compute log marginal likelihood under Normal-Inverse-Gamma prior.
        """
        n = len(data)
        if n == 0:
            return 0.0
        
        # Sufficient statistics
        x_bar = np.mean(data)
        s2 = np.var(data)
        
        # Posterior parameters
        kappa_n = self._kappa0 + n
        mu_n = (self._kappa0 * self._mu0 + n * x_bar) / kappa_n
        alpha_n = self._alpha0 + n / 2
        beta_n = (
            self._beta0 + 
            0.5 * n * s2 + 
            0.5 * (self._kappa0 * n / kappa_n) * (x_bar - self._mu0)**2
        )
        
        # Log marginal likelihood
        log_ml = (
            -n/2 * np.log(2 * np.pi) +
            0.5 * np.log(self._kappa0 / kappa_n) +
            self._alpha0 * np.log(self._beta0) - alpha_n * np.log(beta_n) +
            np.math.lgamma(alpha_n) - np.math.lgamma(self._alpha0)
        )
        
        return log_ml
    
    def _check_oos_loss(self, data: np.ndarray, tau: int) -> bool:
        """
        Check if out-of-sample loss increases after changepoint.
        
        This helps avoid false alarms from noise increases.
        """
        pre_data = data[:tau]
        post_data = data[tau:]
        
        # Fit simple model on pre-changepoint data
        pre_mean = np.mean(pre_data)
        pre_std = np.std(pre_data)
        
        # OOS loss: negative log likelihood on post-changepoint data
        if pre_std < 1e-10:
            return True
        
        # Log loss under pre-changepoint model
        pre_model_loss = -np.mean(stats.norm.logpdf(post_data, pre_mean, pre_std))
        
        # Log loss under post-changepoint model
        post_mean = np.mean(post_data)
        post_std = np.std(post_data)
        if post_std < 1e-10:
            post_std = pre_std
        post_model_loss = -np.mean(stats.norm.logpdf(post_data, post_mean, post_std))
        
        # Loss increase
        loss_increase = (pre_model_loss - post_model_loss) / (abs(pre_model_loss) + 1e-10)
        
        return loss_increase > self.oos_loss_threshold
    
    def online_detect(self, value: float) -> Optional[ChangePointResult]:
        """
        Online changepoint detection.
        
        Args:
            value: New observation
            
        Returns:
            ChangePointResult if changepoint detected, None otherwise
        """
        self.add_observation(value)
        
        # Only check periodically
        if len(self._data) % 10 == 0:
            return self.detect()
        
        return None
    
    def get_regime_probability(self) -> np.ndarray:
        """
        Get probability distribution over changepoint locations.
        
        Returns:
            Array of probabilities for each index
        """
        data = np.array(self._data)
        n = len(data)
        
        if n < 2 * self.min_segment_length:
            return np.zeros(n)
        
        log_probs = []
        
        for tau in range(n):
            if tau < self.min_segment_length or tau > n - self.min_segment_length:
                log_probs.append(-np.inf)
            else:
                log_probs.append(self._compute_bayes_factor(data, tau))
        
        # Convert to probabilities
        log_probs = np.array(log_probs)
        max_log_prob = np.max(log_probs[np.isfinite(log_probs)])
        probs = np.exp(log_probs - max_log_prob)
        probs = probs / np.sum(probs)
        
        return probs
    
    def get_entropy(self) -> float:
        """Get entropy of changepoint distribution (high = uncertain)."""
        probs = self.get_regime_probability()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
    
    def reset(self) -> None:
        """Reset detector state."""
        self._data.clear()
        self._timestamps.clear()
        self._last_changepoint = None
        self._pending_changepoint = None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = BayesianChangepoint(
        min_segment_length=30,
        bf_threshold=3.0,
    )
    
    # Generate data with changepoint
    np.random.seed(42)
    pre_change = np.random.normal(0.001, 0.02, 100)  # Low vol, positive drift
    post_change = np.random.normal(-0.002, 0.04, 100)  # High vol, negative drift
    data = np.concatenate([pre_change, post_change])
    
    print(f"Generated {len(data)} observations with changepoint at 100")
    print(f"Pre-change: mean={np.mean(pre_change):.4f}, std={np.std(pre_change):.4f}")
    print(f"Post-change: mean={np.mean(post_change):.4f}, std={np.std(post_change):.4f}")
    
    # Add data
    detector.add_observations(data)
    
    # Detect
    result = detector.detect()
    
    print(f"\n=== Detection Result ===")
    print(f"Detected: {result.detected}")
    print(f"Changepoint Index: {result.changepoint_idx}")
    print(f"Bayes Factor: {result.bayes_factor:.2f}")
    print(f"Confidence: {result.confidence:.2%}")
    
    if result.detected:
        print(f"\nPre-changepoint: mean={result.pre_mean:.4f}, std={result.pre_std:.4f}")
        print(f"Post-changepoint: mean={result.post_mean:.4f}, std={result.post_std:.4f}")
    
    # Check entropy
    entropy = detector.get_entropy()
    print(f"\nEntropy: {entropy:.4f}")
