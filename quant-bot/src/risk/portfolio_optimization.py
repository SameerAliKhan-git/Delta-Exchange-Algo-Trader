"""
Portfolio Optimization Module
=============================
Advanced portfolio construction and allocation:
- Mean-Variance Optimization (Markowitz)
- Hierarchical Risk Parity (HRP) - López de Prado
- Black-Litterman Model
- Risk Parity
- Nested Clustered Optimization (NCO)

Reference: AFML Ch. 16, Riskfolio-lib patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform, pdist
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioResult:
    """Result from portfolio optimization."""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str
    metadata: Dict


class OptimizationMethod(Enum):
    """Available optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    HRP = "hrp"
    BLACK_LITTERMAN = "black_litterman"
    NCO = "nco"


# =============================================================================
# COVARIANCE ESTIMATION
# =============================================================================

class CovarianceEstimator:
    """
    Robust covariance estimation methods.
    
    Raw sample covariance is noisy for small samples.
    Shrinkage and denoising improve estimation.
    """
    
    @staticmethod
    def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
        """Standard sample covariance."""
        return returns.cov()
    
    @staticmethod
    def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator.
        
        Shrinks sample covariance towards identity matrix.
        Optimal shrinkage intensity minimizes MSE.
        """
        X = returns.values
        n, p = X.shape
        
        # Sample covariance
        sample_cov = np.cov(X, rowvar=False)
        
        # Shrinkage target: scaled identity
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)
        
        # Compute optimal shrinkage intensity
        X_centered = X - X.mean(axis=0)
        
        # Sum of squared off-diagonal elements
        delta = sample_cov - target
        delta_sq_sum = (delta ** 2).sum()
        
        # Estimate asymptotic variance
        y = X_centered ** 2
        phi_mat = (y.T @ y) / n - sample_cov ** 2
        phi = phi_mat.sum()
        
        # Optimal shrinkage
        kappa = (phi - delta_sq_sum / n) / ((n - 1) * delta_sq_sum)
        shrinkage = max(0, min(1, kappa))
        
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
        
        return pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)
    
    @staticmethod
    def exponential_weighted(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
        """
        Exponentially weighted covariance.
        
        More weight on recent observations.
        """
        return returns.ewm(span=span).cov().iloc[-len(returns.columns):]
    
    @staticmethod
    def denoise_covariance(cov: pd.DataFrame, 
                           n_observations: int,
                           method: str = 'constant') -> pd.DataFrame:
        """
        Denoise covariance matrix using Random Matrix Theory.
        
        Identifies and removes noise eigenvalues.
        Reference: AFML Ch. 2
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov.values)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Marcenko-Pastur bounds
        q = len(cov) / n_observations
        lambda_plus = (1 + np.sqrt(q)) ** 2
        lambda_minus = (1 - np.sqrt(q)) ** 2
        
        # Identify signal vs noise eigenvalues
        variance = eigenvalues.mean()
        lambda_scaled = eigenvalues / variance
        
        signal_mask = (lambda_scaled > lambda_plus) | (lambda_scaled < lambda_minus)
        
        if method == 'constant':
            # Replace noise eigenvalues with their average
            noise_eigenvalues = eigenvalues[~signal_mask]
            if len(noise_eigenvalues) > 0:
                eigenvalues[~signal_mask] = noise_eigenvalues.mean()
        
        # Reconstruct covariance
        denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return pd.DataFrame(denoised, index=cov.index, columns=cov.columns)


# =============================================================================
# MEAN-VARIANCE OPTIMIZATION
# =============================================================================

class MeanVarianceOptimizer:
    """
    Classic Markowitz Mean-Variance Optimization.
    """
    
    def __init__(self, returns: pd.DataFrame, cov_matrix: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize optimizer.
        
        Args:
            returns: Historical returns (T x N)
            cov_matrix: Covariance matrix (N x N), computed if None
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.expected_returns = returns.mean() * 252  # Annualized
        
        if cov_matrix is None:
            self.cov_matrix = CovarianceEstimator.ledoit_wolf_shrinkage(returns) * 252
        else:
            self.cov_matrix = cov_matrix
        
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.assets = returns.columns.tolist()
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Portfolio expected return."""
        return np.dot(weights, self.expected_returns)
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Portfolio volatility."""
        return np.sqrt(weights @ self.cov_matrix.values @ weights)
    
    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio (for minimization)."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return -(ret - self.risk_free_rate) / (vol + 1e-10)
    
    def optimize(self, method: str = 'max_sharpe',
                 constraints: Optional[Dict] = None) -> PortfolioResult:
        """
        Run optimization.
        
        Args:
            method: 'max_sharpe', 'min_variance', or 'target_return'
            constraints: Additional constraints
        
        Returns:
            PortfolioResult with optimal weights
        """
        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Constraints: weights sum to 1
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds: long-only by default
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        if method == 'max_sharpe':
            objective = self._negative_sharpe
        elif method == 'min_variance':
            objective = self._portfolio_volatility
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        weights = pd.Series(result.x, index=self.assets)
        weights = weights / weights.sum()  # Normalize
        
        port_return = self._portfolio_return(weights.values)
        port_vol = self._portfolio_volatility(weights.values)
        sharpe = (port_return - self.risk_free_rate) / port_vol
        
        return PortfolioResult(
            weights=weights,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            method=method,
            metadata={'converged': result.success}
        )
    
    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Compute efficient frontier.
        
        Returns DataFrame with return, volatility, Sharpe for each point.
        """
        # Range of target returns
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        
        for target_ret in target_returns:
            # Minimize variance subject to target return
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: self._portfolio_return(w) - target_ret}
            ]
            
            x0 = np.ones(self.n_assets) / self.n_assets
            bounds = [(0, 1) for _ in range(self.n_assets)]
            
            result = minimize(
                self._portfolio_volatility,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                vol = self._portfolio_volatility(result.x)
                sharpe = (target_ret - self.risk_free_rate) / vol
                frontier.append({
                    'return': target_ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': result.x
                })
        
        return pd.DataFrame(frontier)


# =============================================================================
# HIERARCHICAL RISK PARITY (HRP)
# =============================================================================

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) - López de Prado.
    
    Advantages over Markowitz:
    - No need to invert covariance matrix
    - More stable weights
    - Better out-of-sample performance
    - Works with singular covariance matrices
    
    Reference: AFML Ch. 16
    """
    
    def __init__(self, returns: pd.DataFrame, cov_matrix: Optional[pd.DataFrame] = None):
        """
        Initialize HRP.
        
        Args:
            returns: Historical returns
            cov_matrix: Covariance matrix (computed if None)
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        
        if cov_matrix is None:
            self.cov_matrix = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
        else:
            self.cov_matrix = cov_matrix
        
        # Correlation matrix
        std = np.sqrt(np.diag(self.cov_matrix.values))
        self.corr_matrix = self.cov_matrix.values / np.outer(std, std)
    
    def _tree_clustering(self) -> np.ndarray:
        """
        Step 1: Tree Clustering.
        
        Build hierarchical tree from correlation distance.
        """
        # Distance matrix from correlation
        dist = np.sqrt(0.5 * (1 - self.corr_matrix))
        
        # Hierarchical clustering
        dist_condensed = squareform(dist)
        link = linkage(dist_condensed, method='single')
        
        return link
    
    def _quasi_diagonalize(self, link: np.ndarray) -> List[int]:
        """
        Step 2: Quasi-Diagonalization.
        
        Reorder assets so similar assets are close together.
        """
        n = len(self.assets)
        sort_idx = pd.Series([n + i for i in range(n - 1)])
        
        def _get_cluster_items(link: np.ndarray, idx: int, n: int) -> List[int]:
            """Recursively get items in cluster."""
            if idx < n:
                return [idx]
            
            left = int(link[idx - n, 0])
            right = int(link[idx - n, 1])
            
            return _get_cluster_items(link, left, n) + _get_cluster_items(link, right, n)
        
        return _get_cluster_items(link, 2 * n - 2, n)
    
    def _recursive_bisection(self, sort_idx: List[int]) -> np.ndarray:
        """
        Step 3: Recursive Bisection.
        
        Allocate weights using inverse-variance within clusters.
        """
        weights = pd.Series(1.0, index=sort_idx)
        clusters = [sort_idx]
        
        while len(clusters) > 0:
            # Split clusters in two
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]
                    
                    # Variance of each sub-cluster
                    cov_left = self.cov_matrix.iloc[left, left].values
                    cov_right = self.cov_matrix.iloc[right, right].values
                    
                    var_left = self._cluster_variance(cov_left)
                    var_right = self._cluster_variance(cov_right)
                    
                    # Allocate based on inverse variance
                    alpha = 1 - var_left / (var_left + var_right + 1e-10)
                    
                    weights[left] *= alpha
                    weights[right] *= 1 - alpha
                    
                    new_clusters.append(left)
                    new_clusters.append(right)
            
            clusters = new_clusters
        
        return weights.values
    
    def _cluster_variance(self, cov: np.ndarray) -> float:
        """Compute cluster variance using inverse-variance portfolio."""
        inv_var = 1 / (np.diag(cov) + 1e-10)
        weights = inv_var / inv_var.sum()
        return weights @ cov @ weights
    
    def optimize(self) -> PortfolioResult:
        """
        Run HRP optimization.
        
        Returns:
            PortfolioResult with HRP weights
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for HRP")
        
        # Step 1: Tree clustering
        link = self._tree_clustering()
        
        # Step 2: Quasi-diagonalization
        sort_idx = self._quasi_diagonalize(link)
        
        # Step 3: Recursive bisection
        raw_weights = self._recursive_bisection(sort_idx)
        
        # Map back to original order
        weights = pd.Series(index=self.assets, dtype=float)
        for i, idx in enumerate(sort_idx):
            weights.iloc[idx] = raw_weights[i]
        
        # Normalize
        weights = weights / weights.sum()
        
        # Compute portfolio metrics
        expected_returns = self.returns.mean() * 252
        port_return = (weights * expected_returns).sum()
        port_vol = np.sqrt(weights.values @ (self.cov_matrix.values * 252) @ weights.values)
        sharpe = port_return / port_vol
        
        return PortfolioResult(
            weights=weights,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            method='HRP',
            metadata={'linkage': link, 'sort_idx': sort_idx}
        )


# =============================================================================
# RISK PARITY
# =============================================================================

class RiskParity:
    """
    Risk Parity: Equal risk contribution from each asset.
    
    Each asset contributes equally to portfolio variance.
    """
    
    def __init__(self, returns: pd.DataFrame, cov_matrix: Optional[pd.DataFrame] = None):
        """Initialize Risk Parity optimizer."""
        self.returns = returns
        self.assets = returns.columns.tolist()
        
        if cov_matrix is None:
            self.cov_matrix = CovarianceEstimator.ledoit_wolf_shrinkage(returns) * 252
        else:
            self.cov_matrix = cov_matrix
    
    def _risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """Compute marginal risk contribution for each asset."""
        port_var = weights @ self.cov_matrix.values @ weights
        marginal_contrib = self.cov_matrix.values @ weights
        risk_contrib = weights * marginal_contrib / np.sqrt(port_var)
        return risk_contrib
    
    def _risk_parity_objective(self, weights: np.ndarray) -> float:
        """
        Objective: Minimize deviation from equal risk contribution.
        """
        risk_contrib = self._risk_contribution(weights)
        target_contrib = np.sqrt(weights @ self.cov_matrix.values @ weights) / len(weights)
        
        return np.sum((risk_contrib - target_contrib) ** 2)
    
    def optimize(self) -> PortfolioResult:
        """Run Risk Parity optimization."""
        n = len(self.assets)
        x0 = np.ones(n) / n
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 1) for _ in range(n)]
        
        result = minimize(
            self._risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = pd.Series(result.x, index=self.assets)
        weights = weights / weights.sum()
        
        expected_returns = self.returns.mean() * 252
        port_return = (weights * expected_returns).sum()
        port_vol = np.sqrt(weights.values @ self.cov_matrix.values @ weights.values)
        
        return PortfolioResult(
            weights=weights,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=port_return / port_vol,
            method='Risk Parity',
            metadata={'risk_contributions': self._risk_contribution(weights.values)}
        )


# =============================================================================
# BLACK-LITTERMAN
# =============================================================================

class BlackLitterman:
    """
    Black-Litterman Model.
    
    Combines market equilibrium returns with investor views.
    More intuitive than raw expected returns.
    """
    
    def __init__(self, returns: pd.DataFrame, market_caps: pd.Series,
                 cov_matrix: Optional[pd.DataFrame] = None,
                 risk_aversion: float = 2.5,
                 tau: float = 0.05):
        """
        Initialize Black-Litterman model.
        
        Args:
            returns: Historical returns
            market_caps: Market capitalizations for each asset
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion coefficient
            tau: Uncertainty in equilibrium returns
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.market_caps = market_caps[self.assets]
        
        if cov_matrix is None:
            self.cov_matrix = CovarianceEstimator.ledoit_wolf_shrinkage(returns) * 252
        else:
            self.cov_matrix = cov_matrix
        
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # Market weights
        self.market_weights = self.market_caps / self.market_caps.sum()
        
        # Equilibrium returns (reverse optimization)
        self.equilibrium_returns = self._compute_equilibrium_returns()
    
    def _compute_equilibrium_returns(self) -> pd.Series:
        """Compute implied equilibrium returns from market weights."""
        return pd.Series(
            self.risk_aversion * self.cov_matrix.values @ self.market_weights.values,
            index=self.assets
        )
    
    def add_views(self, P: np.ndarray, Q: np.ndarray, 
                  omega: Optional[np.ndarray] = None) -> pd.Series:
        """
        Incorporate investor views.
        
        Args:
            P: Pick matrix (K x N) - which assets are involved in each view
            Q: View vector (K x 1) - expected returns for each view
            omega: Uncertainty matrix (K x K) - confidence in views
        
        Returns:
            Blended expected returns
        """
        n = len(self.assets)
        k = len(Q)
        
        # Default omega: proportional to variance
        if omega is None:
            omega = np.diag(np.diag(P @ (self.tau * self.cov_matrix.values) @ P.T))
        
        # Prior
        pi = self.equilibrium_returns.values
        sigma = self.cov_matrix.values
        tau_sigma = self.tau * sigma
        
        # Posterior (Black-Litterman formula)
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega)
        
        posterior_cov = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
        posterior_mean = posterior_cov @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)
        
        return pd.Series(posterior_mean, index=self.assets)
    
    def optimize(self, P: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None) -> PortfolioResult:
        """
        Optimize portfolio with optional views.
        
        If no views provided, uses equilibrium returns.
        """
        if P is not None and Q is not None:
            expected_returns = self.add_views(P, Q)
        else:
            expected_returns = self.equilibrium_returns
        
        # Mean-variance optimization with BL returns
        optimizer = MeanVarianceOptimizer(
            self.returns,
            self.cov_matrix
        )
        optimizer.expected_returns = expected_returns
        
        result = optimizer.optimize(method='max_sharpe')
        result.method = 'Black-Litterman'
        result.metadata['equilibrium_returns'] = self.equilibrium_returns
        result.metadata['bl_returns'] = expected_returns
        
        return result


# =============================================================================
# PORTFOLIO OPTIMIZER FACTORY
# =============================================================================

class PortfolioOptimizer:
    """
    Unified interface for portfolio optimization.
    """
    
    def __init__(self, returns: pd.DataFrame, 
                 cov_estimator: str = 'ledoit_wolf'):
        """
        Initialize optimizer.
        
        Args:
            returns: Historical returns DataFrame
            cov_estimator: 'sample', 'ledoit_wolf', or 'exponential'
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        
        # Estimate covariance
        if cov_estimator == 'ledoit_wolf':
            self.cov_matrix = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
        elif cov_estimator == 'exponential':
            self.cov_matrix = CovarianceEstimator.exponential_weighted(returns)
        else:
            self.cov_matrix = CovarianceEstimator.sample_covariance(returns)
    
    def optimize(self, method: str = 'hrp', **kwargs) -> PortfolioResult:
        """
        Run portfolio optimization.
        
        Args:
            method: 'hrp', 'mean_variance', 'risk_parity', 'min_variance'
            **kwargs: Method-specific parameters
        
        Returns:
            PortfolioResult
        """
        if method == 'hrp':
            optimizer = HierarchicalRiskParity(self.returns, self.cov_matrix)
            return optimizer.optimize()
        
        elif method == 'risk_parity':
            optimizer = RiskParity(self.returns, self.cov_matrix * 252)
            return optimizer.optimize()
        
        elif method in ['mean_variance', 'max_sharpe', 'min_variance']:
            optimizer = MeanVarianceOptimizer(self.returns, self.cov_matrix * 252)
            return optimizer.optimize(method=method.replace('mean_variance', 'max_sharpe'))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compare_methods(self) -> pd.DataFrame:
        """Compare all optimization methods."""
        results = []
        
        for method in ['hrp', 'risk_parity', 'max_sharpe', 'min_variance']:
            try:
                result = self.optimize(method)
                results.append({
                    'method': method,
                    'expected_return': result.expected_return,
                    'volatility': result.volatility,
                    'sharpe': result.sharpe_ratio,
                    'max_weight': result.weights.max(),
                    'min_weight': result.weights.min(),
                    'n_assets': (result.weights > 0.01).sum()
                })
            except Exception as e:
                print(f"Method {method} failed: {e}")
        
        return pd.DataFrame(results)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PORTFOLIO OPTIMIZATION MODULE")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate sample returns
    n_assets = 10
    n_days = 500
    
    asset_names = [f'Asset_{i}' for i in range(n_assets)]
    
    # Correlated returns with different characteristics
    base_returns = np.random.randn(n_days, n_assets) * 0.02
    
    # Add some correlation
    common_factor = np.random.randn(n_days, 1) * 0.01
    returns = base_returns + common_factor * np.random.rand(1, n_assets)
    
    # Make some assets have higher returns
    returns[:, :3] += 0.0003  # Higher expected return
    returns[:, 7:] -= 0.0001  # Lower expected return
    
    returns_df = pd.DataFrame(
        returns,
        columns=asset_names,
        index=pd.date_range('2022-01-01', periods=n_days, freq='D')
    )
    
    print(f"\n1. Sample Data")
    print(f"   Assets: {n_assets}")
    print(f"   Observations: {n_days}")
    print(f"   Annualized returns range: [{returns_df.mean().min()*252:.2%}, {returns_df.mean().max()*252:.2%}]")
    
    # Compare optimization methods
    print("\n2. Comparing Optimization Methods")
    print("-" * 60)
    
    optimizer = PortfolioOptimizer(returns_df, cov_estimator='ledoit_wolf')
    comparison = optimizer.compare_methods()
    
    print(comparison.to_string(index=False))
    
    # Detailed HRP
    print("\n3. HRP Detailed Results")
    print("-" * 60)
    
    hrp = HierarchicalRiskParity(returns_df)
    result = hrp.optimize()
    
    print(f"   Expected Return: {result.expected_return:.2%}")
    print(f"   Volatility:      {result.volatility:.2%}")
    print(f"   Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"\n   Weights:")
    for asset, weight in result.weights.sort_values(ascending=False).items():
        print(f"     {asset}: {weight:.2%}")
    
    # Efficient Frontier
    print("\n4. Efficient Frontier (Mean-Variance)")
    print("-" * 60)
    
    mv = MeanVarianceOptimizer(returns_df)
    frontier = mv.efficient_frontier(n_points=10)
    
    print(f"   {'Return':>10} {'Vol':>10} {'Sharpe':>10}")
    for _, row in frontier.iterrows():
        print(f"   {row['return']:>10.2%} {row['volatility']:>10.2%} {row['sharpe']:>10.2f}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
1. HRP (Hierarchical Risk Parity):
   - Best for out-of-sample performance
   - No matrix inversion required
   - Handles correlated assets well
   - USE THIS AS DEFAULT

2. Risk Parity:
   - Equal risk contribution
   - Good for diversification
   - Tends to overweight low-vol assets

3. Mean-Variance (Max Sharpe):
   - Classic optimization
   - Sensitive to estimation errors
   - Use with shrinkage covariance

4. Covariance Estimation:
   - Always use Ledoit-Wolf shrinkage
   - Consider exponential weighting for regime changes
   - Denoise for small sample sizes

5. Production Tips:
   - Rebalance monthly or quarterly
   - Set minimum weight constraints (e.g., 1%)
   - Monitor turnover and transaction costs
""")
