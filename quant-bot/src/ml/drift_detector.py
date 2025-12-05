"""
Drift Detector Module
=====================
Detects distribution drift in model features using Population Stability Index (PSI).
Critical for preventing model decay in production.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects feature drift using Population Stability Index (PSI).
    """
    
    def __init__(self, bin_edges: Optional[np.ndarray] = None, training_distribution: Optional[np.ndarray] = None):
        """
        Initialize DriftDetector.
        
        Args:
            bin_edges: Edges of bins from training data histogram
            training_distribution: Normalized frequency of training data in each bin
        """
        self.bin_edges = bin_edges
        self.training_distribution = training_distribution
        
    def fit(self, training_features: np.ndarray, n_bins: int = 10):
        """
        Fit the detector to training data.
        
        Args:
            training_features: Array of feature values from training set
            n_bins: Number of bins for histogram
        """
        # Calculate histogram and normalize
        counts, self.bin_edges = np.histogram(training_features, bins=n_bins)
        self.training_distribution = counts / len(training_features)
        
        # Avoid zero probabilities for log calculation
        self.training_distribution = np.maximum(self.training_distribution, 1e-6)
        
        logger.info(f"DriftDetector fitted with {len(training_features)} samples")

    def psi(self, current_features: np.ndarray) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            current_features: Array of recent feature values
            
        Returns:
            PSI value
        """
        if self.bin_edges is None or self.training_distribution is None:
            raise ValueError("DriftDetector must be fitted before calculating PSI")
            
        # Calculate histogram using same bins as training
        current_dist, _ = np.histogram(current_features, bins=self.bin_edges)
        current_dist = current_dist / len(current_features)
        
        # Avoid zero probabilities
        current_dist = np.maximum(current_dist, 1e-6)
        
        psi_val = 0.0
        for i in range(len(self.training_distribution)):
            if self.training_distribution[i] > 0 and current_dist[i] > 0:
                psi_val += (current_dist[i] - self.training_distribution[i]) * np.log(current_dist[i] / self.training_distribution[i])
        
        return psi_val
    
    def check_drift(self, recent_features: np.ndarray, threshold: float = 0.25) -> Tuple[str, float]:
        """
        Check for drift status.
        
        Args:
            recent_features: Recent feature data
            threshold: PSI threshold for high drift
            
        Returns:
            Tuple of (status_string, psi_score)
        """
        try:
            psi_score = self.psi(recent_features)
            
            if psi_score > threshold:
                logger.warning(f"HIGH DRIFT DETECTED: PSI={psi_score:.4f}")
                return "HIGH_DRIFT", psi_score
            elif psi_score > 0.15:
                logger.info(f"Medium drift detected: PSI={psi_score:.4f}")
                return "MEDIUM_DRIFT", psi_score
            
            return "STABLE", psi_score
            
        except Exception as e:
            logger.error(f"Error checking drift: {e}")
            return "ERROR", 0.0
