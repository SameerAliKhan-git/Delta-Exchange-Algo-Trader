"""
Signal Processing Module
========================
Advanced signal processing techniques for trading:
- Kalman Filter (dynamic estimation)
- Hidden Markov Models (regime detection)
- GARCH (volatility forecasting)
- Wavelets (multi-scale decomposition)

These methods extract cleaner signals from noisy financial data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    from scipy.signal import butter, filtfilt
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# KALMAN FILTER
# =============================================================================

class KalmanFilter:
    """
    Kalman Filter for dynamic estimation.
    
    Applications:
    - Smoothing price/volatility estimates
    - Dynamic hedge ratios (pairs trading)
    - Trend extraction
    - Noise filtering
    
    State space model:
        x_t = F * x_{t-1} + w_t  (state transition)
        z_t = H * x_t + v_t      (observation)
    """
    
    def __init__(self, 
                 dim_state: int = 2,
                 dim_obs: int = 1,
                 transition_covariance: float = 0.01,
                 observation_covariance: float = 1.0):
        """
        Initialize Kalman Filter.
        
        Args:
            dim_state: Dimension of state vector (e.g., 2 for position + velocity)
            dim_obs: Dimension of observation
            transition_covariance: Process noise (Q)
            observation_covariance: Measurement noise (R)
        """
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        
        # State transition matrix (random walk for price)
        self.F = np.eye(dim_state)
        if dim_state == 2:
            self.F[0, 1] = 1  # Position updated by velocity
        
        # Observation matrix
        self.H = np.zeros((dim_obs, dim_state))
        self.H[0, 0] = 1  # We observe the first state (price)
        
        # Process noise covariance
        self.Q = np.eye(dim_state) * transition_covariance
        
        # Observation noise covariance
        self.R = np.eye(dim_obs) * observation_covariance
        
        # Initial state
        self.x = np.zeros(dim_state)
        self.P = np.eye(dim_state) * 1000  # Large initial uncertainty
        
        self.is_initialized = False
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict step: propagate state forward.
        
        Returns:
            Predicted state and covariance
        """
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred
    
    def update(self, z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step: incorporate new observation.
        
        Args:
            z: New observation
        
        Returns:
            Updated state and covariance
        """
        # Predict
        x_pred, P_pred = self.predict()
        
        # Innovation
        y = np.array([z]) - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Update
        self.x = x_pred + K @ y
        self.P = (np.eye(self.dim_state) - K @ self.H) @ P_pred
        
        return self.x.copy(), self.P.copy()
    
    def filter(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run filter over entire series.
        
        Args:
            observations: Array of observations
        
        Returns:
            Dictionary with filtered states, predictions, and variances
        """
        n = len(observations)
        
        states = np.zeros((n, self.dim_state))
        predictions = np.zeros((n, self.dim_state))
        variances = np.zeros((n, self.dim_state, self.dim_state))
        
        # Initialize with first observation
        self.x[0] = observations[0]
        self.is_initialized = True
        
        for t in range(n):
            # Predict
            x_pred, P_pred = self.predict()
            predictions[t] = x_pred
            
            # Update
            self.x, self.P = self.update(observations[t])
            states[t] = self.x
            variances[t] = self.P
        
        return {
            'filtered_state': states,
            'predicted_state': predictions,
            'variance': variances,
            'filtered_price': states[:, 0],
            'filtered_velocity': states[:, 1] if self.dim_state > 1 else None
        }
    
    def smooth(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Rauch-Tung-Striebel smoother for offline processing.
        
        Provides better estimates by using both past and future data.
        """
        # Forward pass
        forward = self.filter(observations)
        
        n = len(observations)
        smoothed = np.zeros((n, self.dim_state))
        smoothed[-1] = forward['filtered_state'][-1]
        
        # Backward pass
        for t in range(n - 2, -1, -1):
            x_t = forward['filtered_state'][t]
            P_t = forward['variance'][t]
            
            x_pred = self.F @ x_t
            P_pred = self.F @ P_t @ self.F.T + self.Q
            
            J = P_t @ self.F.T @ np.linalg.inv(P_pred)
            smoothed[t] = x_t + J @ (smoothed[t + 1] - x_pred)
        
        return {
            'smoothed_state': smoothed,
            'smoothed_price': smoothed[:, 0]
        }


class DynamicHedgeRatio(KalmanFilter):
    """
    Kalman Filter for dynamic hedge ratio estimation.
    
    Used in pairs trading to estimate time-varying beta.
    
    Model: y_t = alpha + beta * x_t + e_t
    State: [alpha, beta]
    """
    
    def __init__(self, delta: float = 0.0001):
        """
        Initialize dynamic hedge ratio filter.
        
        Args:
            delta: Controls how fast beta can change
        """
        super().__init__(dim_state=2, dim_obs=1)
        
        # State is [intercept, slope]
        self.F = np.eye(2)
        self.Q = np.eye(2) * delta
        self.R = np.array([[1.0]])
        
        self.x = np.array([0.0, 1.0])  # Initial: alpha=0, beta=1
        self.P = np.eye(2)
    
    def update_pair(self, y: float, x: float) -> Tuple[float, float, float]:
        """
        Update with new pair observation.
        
        Args:
            y: Dependent variable (e.g., stock price)
            x: Independent variable (e.g., ETF price)
        
        Returns:
            (alpha, beta, spread)
        """
        # Observation matrix depends on x
        self.H = np.array([[1.0, x]])
        
        # Update
        self.update(y)
        
        alpha, beta = self.x
        spread = y - (alpha + beta * x)
        
        return alpha, beta, spread
    
    def fit(self, y: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        """
        Fit to series of paired observations.
        
        Returns DataFrame with alpha, beta, and spread over time.
        """
        n = len(y)
        results = {
            'alpha': np.zeros(n),
            'beta': np.zeros(n),
            'spread': np.zeros(n),
            'y': y,
            'x': x
        }
        
        for t in range(n):
            alpha, beta, spread = self.update_pair(y[t], x[t])
            results['alpha'][t] = alpha
            results['beta'][t] = beta
            results['spread'][t] = spread
        
        return pd.DataFrame(results)


# =============================================================================
# HIDDEN MARKOV MODEL
# =============================================================================

class GaussianHMM:
    """
    Gaussian Hidden Markov Model for regime detection.
    
    Assumes observations come from one of K hidden states,
    each with its own Gaussian distribution.
    
    Applications:
    - Market regime detection (bull/bear/sideways)
    - Volatility regime switching
    - Trend/mean-reversion classification
    """
    
    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-4):
        """
        Initialize HMM.
        
        Args:
            n_states: Number of hidden states
            n_iter: Maximum EM iterations
            tol: Convergence tolerance
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        
        # Model parameters (to be estimated)
        self.pi = None  # Initial state distribution
        self.A = None   # Transition matrix
        self.means = None  # State means
        self.stds = None   # State standard deviations
        
        self.is_fitted = False
    
    def _init_params(self, X: np.ndarray):
        """Initialize parameters using k-means-like heuristic."""
        n = len(X)
        k = self.n_states
        
        # Initial distribution: uniform
        self.pi = np.ones(k) / k
        
        # Transition matrix: slight persistence
        self.A = np.ones((k, k)) * 0.1 / (k - 1)
        np.fill_diagonal(self.A, 0.9)
        
        # Means: spread across data range
        percentiles = np.linspace(10, 90, k)
        self.means = np.percentile(X, percentiles)
        
        # Stds: based on data spread
        self.stds = np.ones(k) * X.std() / k
    
    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """Gaussian probability density."""
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward algorithm: compute P(z_t | x_1:t)."""
        n = len(X)
        k = self.n_states
        
        alpha = np.zeros((n, k))
        scale = np.zeros(n)
        
        # Initial
        for j in range(k):
            alpha[0, j] = self.pi[j] * self._gaussian_pdf(X[0], self.means[j], self.stds[j])
        
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        
        # Recursion
        for t in range(1, n):
            for j in range(k):
                alpha[t, j] = sum(alpha[t-1, i] * self.A[i, j] for i in range(k))
                alpha[t, j] *= self._gaussian_pdf(X[t], self.means[j], self.stds[j])
            
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
        
        return alpha, scale
    
    def _backward(self, X: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Backward algorithm: compute P(x_{t+1:T} | z_t)."""
        n = len(X)
        k = self.n_states
        
        beta = np.zeros((n, k))
        beta[-1] = 1
        
        for t in range(n - 2, -1, -1):
            for i in range(k):
                for j in range(k):
                    beta[t, i] += self.A[i, j] * self._gaussian_pdf(X[t+1], self.means[j], self.stds[j]) * beta[t+1, j]
            
            if scale[t+1] > 0:
                beta[t] /= scale[t+1]
        
        return beta
    
    def fit(self, X: np.ndarray) -> 'GaussianHMM':
        """
        Fit HMM using Baum-Welch (EM) algorithm.
        
        Args:
            X: Observation sequence
        
        Returns:
            self
        """
        X = np.asarray(X).flatten()
        n = len(X)
        k = self.n_states
        
        self._init_params(X)
        
        prev_ll = -np.inf
        
        for iteration in range(self.n_iter):
            # E-step
            alpha, scale = self._forward(X)
            beta = self._backward(X, scale)
            
            # Posterior: P(z_t = j | X)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10
            
            # Joint: P(z_t = i, z_{t+1} = j | X)
            xi = np.zeros((n - 1, k, k))
            for t in range(n - 1):
                for i in range(k):
                    for j in range(k):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                       self._gaussian_pdf(X[t+1], self.means[j], self.stds[j]) * 
                                       beta[t+1, j])
                xi[t] /= xi[t].sum() + 1e-10
            
            # M-step
            self.pi = gamma[0] / (gamma[0].sum() + 1e-10)
            
            for i in range(k):
                for j in range(k):
                    self.A[i, j] = xi[:, i, j].sum() / (gamma[:-1, i].sum() + 1e-10)
            
            for j in range(k):
                denom = gamma[:, j].sum() + 1e-10
                self.means[j] = (gamma[:, j] * X).sum() / denom
                self.stds[j] = np.sqrt((gamma[:, j] * (X - self.means[j])**2).sum() / denom + 1e-6)
            
            # Log-likelihood
            ll = np.sum(np.log(scale + 1e-10))
            
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence (Viterbi algorithm).
        
        Args:
            X: Observation sequence
        
        Returns:
            Array of state indices
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X).flatten()
        n = len(X)
        k = self.n_states
        
        # Viterbi
        delta = np.zeros((n, k))
        psi = np.zeros((n, k), dtype=int)
        
        # Initial
        for j in range(k):
            delta[0, j] = np.log(self.pi[j] + 1e-10) + np.log(self._gaussian_pdf(X[0], self.means[j], self.stds[j]) + 1e-10)
        
        # Recursion
        for t in range(1, n):
            for j in range(k):
                probs = delta[t-1] + np.log(self.A[:, j] + 1e-10)
                psi[t, j] = np.argmax(probs)
                delta[t, j] = probs[psi[t, j]] + np.log(self._gaussian_pdf(X[t], self.means[j], self.stds[j]) + 1e-10)
        
        # Backtrack
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        
        return states
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities.
        
        Returns:
            Array of shape (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = np.asarray(X).flatten()
        alpha, scale = self._forward(X)
        beta = self._backward(X, scale)
        
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10
        
        return gamma
    
    def get_regime_stats(self) -> pd.DataFrame:
        """Get summary statistics for each regime."""
        return pd.DataFrame({
            'state': range(self.n_states),
            'mean': self.means,
            'std': self.stds,
            'initial_prob': self.pi
        })


# =============================================================================
# GARCH VOLATILITY MODEL
# =============================================================================

class GARCH:
    """
    GARCH(1,1) model for volatility forecasting.
    
    sigma^2_t = omega + alpha * e^2_{t-1} + beta * sigma^2_{t-1}
    
    Where:
    - omega: Long-term variance weight
    - alpha: Shock impact
    - beta: Persistence
    """
    
    def __init__(self, omega: float = 0.01, alpha: float = 0.1, beta: float = 0.85):
        """
        Initialize GARCH model.
        
        Defaults are typical for daily financial data.
        Constraint: alpha + beta < 1 for stationarity.
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        
        self.long_run_var = omega / (1 - alpha - beta) if alpha + beta < 1 else omega
        self.is_fitted = False
    
    def _log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Negative log-likelihood for optimization."""
        omega, alpha, beta = params
        
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        n = len(returns)
        var = np.zeros(n)
        var[0] = returns.var()
        
        ll = 0
        for t in range(1, n):
            var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
            if var[t] <= 0:
                return 1e10
            ll += -0.5 * (np.log(var[t]) + returns[t]**2 / var[t])
        
        return -ll
    
    def fit(self, returns: np.ndarray) -> 'GARCH':
        """
        Fit GARCH model using MLE.
        
        Args:
            returns: Return series
        
        Returns:
            self
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for GARCH fitting")
        
        returns = np.asarray(returns).flatten()
        
        # Initial guess
        var = returns.var()
        x0 = [var * 0.01, 0.1, 0.85]
        
        # Bounds
        bounds = [(1e-6, None), (0, 0.5), (0, 0.99)]
        
        # Optimize
        result = minimize(
            self._log_likelihood,
            x0,
            args=(returns,),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        self.omega, self.alpha, self.beta = result.x
        self.long_run_var = self.omega / (1 - self.alpha - self.beta)
        self.is_fitted = True
        
        return self
    
    def forecast(self, returns: np.ndarray, horizon: int = 1) -> Dict[str, np.ndarray]:
        """
        Forecast volatility.
        
        Args:
            returns: Historical returns
            horizon: Forecast horizon
        
        Returns:
            Dictionary with conditional variance and volatility forecasts
        """
        returns = np.asarray(returns).flatten()
        n = len(returns)
        
        # Historical conditional variance
        var = np.zeros(n)
        var[0] = returns.var()
        
        for t in range(1, n):
            var[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * var[t-1]
        
        # Multi-step forecast
        forecast_var = np.zeros(horizon)
        forecast_var[0] = self.omega + self.alpha * returns[-1]**2 + self.beta * var[-1]
        
        for h in range(1, horizon):
            forecast_var[h] = self.omega + (self.alpha + self.beta) * forecast_var[h-1]
        
        return {
            'conditional_variance': var,
            'conditional_volatility': np.sqrt(var),
            'forecast_variance': forecast_var,
            'forecast_volatility': np.sqrt(forecast_var),
            'long_run_volatility': np.sqrt(self.long_run_var)
        }


# =============================================================================
# WAVELETS (Simple DWT)
# =============================================================================

class WaveletDecomposition:
    """
    Discrete Wavelet Transform for multi-scale analysis.
    
    Decomposes signal into:
    - Approximation (low-frequency trend)
    - Details (high-frequency noise)
    
    Applications:
    - Denoising price series
    - Multi-timeframe feature extraction
    - Trend/cycle separation
    """
    
    def __init__(self, wavelet: str = 'haar', levels: int = 4):
        """
        Initialize wavelet decomposition.
        
        Args:
            wavelet: Wavelet type ('haar', 'db4', etc.)
            levels: Decomposition levels
        """
        self.wavelet = wavelet
        self.levels = levels
        
        # Haar wavelet filters
        self.lo_d = np.array([1, 1]) / np.sqrt(2)  # Low-pass decomposition
        self.hi_d = np.array([1, -1]) / np.sqrt(2)  # High-pass decomposition
        self.lo_r = np.array([1, 1]) / np.sqrt(2)  # Low-pass reconstruction
        self.hi_r = np.array([1, -1]) / np.sqrt(2)  # High-pass reconstruction
    
    def _convolve_downsample(self, signal: np.ndarray, filter: np.ndarray) -> np.ndarray:
        """Convolve and downsample by 2."""
        result = np.convolve(signal, filter, mode='full')
        return result[::2]
    
    def _upsample_convolve(self, signal: np.ndarray, filter: np.ndarray, length: int) -> np.ndarray:
        """Upsample by 2 and convolve."""
        upsampled = np.zeros(len(signal) * 2)
        upsampled[::2] = signal
        result = np.convolve(upsampled, filter, mode='full')
        return result[:length]
    
    def decompose(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform wavelet decomposition.
        
        Args:
            signal: Input signal
        
        Returns:
            Dictionary with approximation and detail coefficients
        """
        signal = np.asarray(signal).flatten()
        
        coeffs = {'approximation': [], 'details': []}
        current = signal
        
        for level in range(self.levels):
            # Decompose
            approx = self._convolve_downsample(current, self.lo_d)
            detail = self._convolve_downsample(current, self.hi_d)
            
            coeffs['details'].insert(0, detail)
            current = approx
        
        coeffs['approximation'] = current
        return coeffs
    
    def reconstruct(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reconstruct signal from coefficients.
        """
        current = coeffs['approximation']
        
        for detail in coeffs['details']:
            length = len(detail) * 2
            approx_up = self._upsample_convolve(current, self.lo_r, length)
            detail_up = self._upsample_convolve(detail, self.hi_r, length)
            current = approx_up + detail_up
        
        return current
    
    def denoise(self, signal: np.ndarray, threshold_pct: float = 0.5) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.
        
        Args:
            signal: Noisy signal
            threshold_pct: Threshold percentile for detail coefficients
        
        Returns:
            Denoised signal
        """
        coeffs = self.decompose(signal)
        
        # Threshold detail coefficients
        thresholded_details = []
        for detail in coeffs['details']:
            threshold = np.percentile(np.abs(detail), threshold_pct * 100)
            thresholded = np.where(np.abs(detail) > threshold, detail, 0)
            thresholded_details.append(thresholded)
        
        coeffs['details'] = thresholded_details
        
        return self.reconstruct(coeffs)[:len(signal)]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SIGNAL PROCESSING MODULE DEMO")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate sample data with regime changes
    n = 1000
    regime = np.concatenate([
        np.zeros(300),   # Low vol
        np.ones(400),    # High vol
        np.zeros(300)    # Low vol again
    ]).astype(int)
    
    returns = np.where(
        regime == 0,
        np.random.randn(n) * 0.01,
        np.random.randn(n) * 0.03
    )
    
    prices = 100 * np.cumprod(1 + returns)
    
    # 1. Kalman Filter
    print("\n1. KALMAN FILTER")
    print("-" * 40)
    kf = KalmanFilter(dim_state=2, transition_covariance=0.001)
    result = kf.filter(prices)
    
    print(f"   Raw price range: [{prices.min():.2f}, {prices.max():.2f}]")
    print(f"   Filtered price range: [{result['filtered_price'].min():.2f}, {result['filtered_price'].max():.2f}]")
    print(f"   Smoothing reduced noise by: {1 - np.std(np.diff(result['filtered_price'])) / np.std(np.diff(prices)):.1%}")
    
    # 2. HMM Regime Detection
    print("\n2. HIDDEN MARKOV MODEL")
    print("-" * 40)
    hmm = GaussianHMM(n_states=2)
    hmm.fit(returns)
    
    predicted_regime = hmm.predict(returns)
    accuracy = (predicted_regime == regime).mean()
    
    print(f"   Regime detection accuracy: {accuracy:.1%}")
    print(f"   Regime statistics:")
    print(hmm.get_regime_stats().to_string())
    
    # 3. GARCH Volatility
    print("\n3. GARCH VOLATILITY MODEL")
    print("-" * 40)
    garch = GARCH()
    garch.fit(returns)
    
    print(f"   Estimated parameters:")
    print(f"     omega: {garch.omega:.6f}")
    print(f"     alpha: {garch.alpha:.4f}")
    print(f"     beta:  {garch.beta:.4f}")
    print(f"     Long-run vol: {np.sqrt(garch.long_run_var) * 100:.2f}%")
    
    forecast = garch.forecast(returns, horizon=10)
    print(f"   10-step volatility forecast: {forecast['forecast_volatility'][-1] * 100:.2f}%")
    
    # 4. Wavelet Denoising
    print("\n4. WAVELET DECOMPOSITION")
    print("-" * 40)
    wavelet = WaveletDecomposition(levels=4)
    
    noisy_signal = prices + np.random.randn(n) * 2  # Add noise
    denoised = wavelet.denoise(noisy_signal, threshold_pct=0.7)
    
    print(f"   Original signal std: {np.std(np.diff(prices)):.4f}")
    print(f"   Noisy signal std:    {np.std(np.diff(noisy_signal)):.4f}")
    print(f"   Denoised signal std: {np.std(np.diff(denoised)):.4f}")
    
    print("\n" + "="*70)
    print("TRADING APPLICATIONS")
    print("="*70)
    print("""
1. KALMAN FILTER:
   - Smooth price for cleaner trend detection
   - Dynamic hedge ratios in pairs trading
   - Filter noise from indicators

2. HIDDEN MARKOV MODEL:
   - Detect bull/bear/sideways regimes
   - Adjust strategy parameters by regime
   - Risk management (reduce size in high-vol regime)

3. GARCH:
   - Forecast volatility for options pricing
   - Dynamic position sizing (inverse volatility)
   - Set stop-losses based on expected vol

4. WAVELETS:
   - Denoise price for pattern recognition
   - Multi-timeframe features
   - Cycle detection
""")
