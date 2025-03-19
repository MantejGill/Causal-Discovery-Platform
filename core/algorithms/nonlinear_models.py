# core/algorithms/nonlinear_models.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import optimize, stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostNonlinearModel:
    """
    Implementation of Post-Nonlinear (PNL) causal model: Y = f2(f1(X) + E)
    
    PNL models capture more complex causal relationships where both
    the cause-effect relationship (f1) and the effect's distortion mechanism (f2)
    can be nonlinear.
    """
    def __init__(self, 
                f1_degree: int = 3, 
                f2_degree: int = 3,
                independence_test: str = 'hsic'):
        """
        Initialize PNL model
        
        Args:
            f1_degree: Degree of polynomial for f1 function
            f2_degree: Degree of polynomial for f2 function
            independence_test: Test to use for independence ('hsic', 'pearson')
        """
        self.f1_degree = f1_degree
        self.f2_degree = f2_degree
        self.independence_test = independence_test
        self.f1_params = None
        self.f2_params = None
        self.residuals = None
        
    def _fit_f1(self, x: np.ndarray) -> callable:
        """Fit f1 function (polynomial basis)"""
        X_poly = np.column_stack([x**i for i in range(1, self.f1_degree + 1)])
        # Add a column of ones for intercept
        X_poly = np.column_stack([np.ones(len(x)), X_poly])
        
        # Initialize parameters
        self.f1_params = np.random.randn(self.f1_degree + 1)
        
        def f1(x_new, params=None):
            params = params if params is not None else self.f1_params
            x_poly = np.column_stack([x_new**i for i in range(1, self.f1_degree + 1)])
            x_poly = np.column_stack([np.ones(len(x_new)), x_poly])
            return x_poly @ params
        
        return f1
    
    def _fit_f2_inverse(self, y: np.ndarray, f1_x: np.ndarray) -> callable:
        """Fit f2 inverse function (polynomial basis)"""
        # We actually fit the inverse of f2 to recover the residuals
        Y_poly = np.column_stack([y**i for i in range(1, self.f2_degree + 1)])
        Y_poly = np.column_stack([np.ones(len(y)), Y_poly])
        
        # Initialize parameters 
        self.f2_params = np.random.randn(self.f2_degree + 1)
        
        def f2_inverse(y_new, params=None):
            params = params if params is not None else self.f2_params
            y_poly = np.column_stack([y_new**i for i in range(1, self.f2_degree + 1)])
            y_poly = np.column_stack([np.ones(len(y_new)), y_poly])
            return y_poly @ params
        
        return f2_inverse
    
    def _independence_score(self, x: np.ndarray, residuals: np.ndarray) -> float:
        """Test independence between x and residuals"""
        if self.independence_test == 'hsic':
            from sklearn.metrics.pairwise import rbf_kernel
            
            # Normalize data
            x_std = (x - x.mean()) / x.std()
            e_std = (residuals - residuals.mean()) / residuals.std()
            
            # Compute kernel matrices
            K = rbf_kernel(x_std.reshape(-1, 1))
            L = rbf_kernel(e_std.reshape(-1, 1))
            
            # Center kernel matrices
            n = len(x)
            H = np.eye(n) - np.ones((n, n)) / n
            K_centered = H @ K @ H
            L_centered = H @ L @ H
            
            # Compute HSIC statistic
            hsic = np.trace(K_centered @ L_centered) / (n * n)
            
            # Return independence score (lower HSIC means more independent)
            return 1.0 - hsic
        
        elif self.independence_test == 'pearson':
            # Pearson correlation (absolute value)
            corr, _ = stats.pearsonr(x, residuals)
            # Convert to independence score (1 - |corr|)
            return 1.0 - abs(corr)
        
        else:
            raise ValueError(f"Unknown independence test: {self.independence_test}")
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Fit PNL model to data
        
        Args:
            x: Cause variable values
            y: Effect variable values
            
        Returns:
            Independence score (higher means more independent)
        """
        # Ensure 1D arrays
        x = x.flatten()
        y = y.flatten()
        
        # Scale data for stability
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        x_scaled = self.x_scaler.fit_transform(x.reshape(-1, 1)).flatten()
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Get initial function estimates
        f1_func = self._fit_f1(x_scaled)
        f1_x = f1_func(x_scaled)
        f2_inv_func = self._fit_f2_inverse(y_scaled, f1_x)
        
        # Define the objective function to minimize
        def objective(params):
            # Split params into f1 and f2 parameters
            f1_params = params[:self.f1_degree + 1]
            f2_params = params[self.f1_degree + 1:]
            
            # Apply functions
            f1_x = f1_func(x_scaled, f1_params)
            residuals = f2_inv_func(y_scaled, f2_params) - f1_x
            
            # Calculate negative independence score (we want to maximize independence)
            neg_indep_score = -self._independence_score(x_scaled, residuals)
            
            return neg_indep_score
        
        # Initial parameter vector
        initial_params = np.concatenate([self.f1_params, self.f2_params])
        
        # Optimize to find the best parameters
        try:
            result = optimize.minimize(
                objective, 
                initial_params, 
                method='BFGS',
                options={'maxiter': 100}
            )
            
            # Update model parameters
            self.f1_params = result.x[:self.f1_degree + 1]
            self.f2_params = result.x[self.f1_degree + 1:]
            
            # Calculate final residuals
            f1_x = f1_func(x_scaled, self.f1_params)
            self.residuals = f2_inv_func(y_scaled, self.f2_params) - f1_x
            
            # Return final independence score
            return -result.fun
        
        except Exception as e:
            logger.error(f"Error in PNL optimization: {str(e)}")
            # Return a default low score
            return 0.0
    
    def test_direction(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Test causal direction X→Y vs Y→X
        
        Args:
            x: First variable values
            y: Second variable values
            
        Returns:
            Dictionary with test results
        """
        # Ensure 1D arrays
        x = x.flatten()
        y = y.flatten()
        
        # Test X→Y
        forward_score = self.fit(x, y)
        
        # Create a new model instance for backward direction
        backward_model = PostNonlinearModel(
            f1_degree=self.f1_degree,
            f2_degree=self.f2_degree,
            independence_test=self.independence_test
        )
        
        # Test Y→X
        backward_score = backward_model.fit(y, x)
        
        # Determine the direction based on which has higher independence score
        if forward_score > backward_score:
            direction = "0->1"  # X→Y
            confidence = forward_score / (forward_score + backward_score)
        else:
            direction = "1->0"  # Y→X
            confidence = backward_score / (forward_score + backward_score)
        
        return {
            "direction": direction,
            "forward_score": forward_score,
            "backward_score": backward_score,
            "confidence": confidence
        }


class AdditiveNoiseModel:
    """
    Implementation of Additive Noise Model (ANM): Y = f(X) + E
    
    ANM assumes that the effect is a (possibly nonlinear) function of 
    the cause plus independent noise.
    """
    def __init__(self, regression_method: str = 'gp'):
        """
        Initialize ANM
        
        Args:
            regression_method: Method for regression ('gp' for Gaussian Process)
        """
        self.regression_method = regression_method
        self.model = None
        self.residuals = None
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Fit ANM to data
        
        Args:
            x: Cause variable values
            y: Effect variable values
            
        Returns:
            Independence score (higher means more independent)
        """
        # Ensure 1D arrays and proper shapes
        x = x.flatten()
        y = y.flatten()
        X = x.reshape(-1, 1)
        
        # Scale data for stability
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        if self.regression_method == 'gp':
            # Define GP kernel
            kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            
            # Create and fit GP model
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            self.model.fit(X_scaled, y_scaled)
            
            # Get predictions and calculate residuals
            y_pred = self.model.predict(X_scaled)
            self.residuals = y_scaled - y_pred
            
            # Calculate independence score
            indep_score = self._test_independence(X_scaled.flatten(), self.residuals)
            
            return indep_score
        
        else:
            raise ValueError(f"Unknown regression method: {self.regression_method}")
    
    def _test_independence(self, x: np.ndarray, residuals: np.ndarray) -> float:
        """
        Test independence between x and residuals
        
        Args:
            x: Cause variable values
            residuals: Residuals from regression
            
        Returns:
            Independence score (higher means more independent)
        """
        # Use HSIC as independence measure
        from sklearn.metrics.pairwise import rbf_kernel
        
        # Compute kernel matrices
        K = rbf_kernel(x.reshape(-1, 1))
        L = rbf_kernel(residuals.reshape(-1, 1))
        
        # Center kernel matrices
        n = len(x)
        H = np.eye(n) - np.ones((n, n)) / n
        K_centered = H @ K @ H
        L_centered = H @ L @ H
        
        # Compute HSIC statistic
        hsic = np.trace(K_centered @ L_centered) / (n * n)
        
        # Return independence score (lower HSIC means more independent)
        return 1.0 - hsic
    
    def test_direction(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Test causal direction using ANM
        
        Args:
            x: First variable values
            y: Second variable values
            
        Returns:
            Dictionary with test results
        """
        # Ensure 1D arrays
        x = x.flatten()
        y = y.flatten()
        
        # Test X→Y
        forward_score = self.fit(x, y)
        
        # Create a new model instance for backward direction
        backward_model = AdditiveNoiseModel(regression_method=self.regression_method)
        
        # Test Y→X
        backward_score = backward_model.fit(y, x)
        
        # Determine direction based on independence scores
        if forward_score > backward_score:
            direction = "0->1"  # X→Y
            confidence = forward_score / (forward_score + backward_score)
        else:
            direction = "1->0"  # Y→X
            confidence = backward_score / (forward_score + backward_score)
        
        return {
            "direction": direction,
            "forward_score": forward_score,
            "backward_score": backward_score,
            "confidence": confidence
        }


class InformationGeometricCausalInference:
    """
    Implementation of Information Geometric Causal Inference (IGCI)
    
    IGCI infers causality from the relative complexity of conditional probability distributions
    and is particularly useful for deterministic relationships with no or little noise.
    """
    def __init__(self, method: str = 'entropy'):
        """
        Initialize IGCI
        
        Args:
            method: Method for estimating complexity ('entropy' or 'slope')
        """
        self.method = method
    
    def _uniform_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Transform data to have uniform distribution on [0, 1]
        """
        return stats.rankdata(x) / len(x)
    
    def _entropy_complexity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate complexity using differential entropy
        """
        # Transform to uniform marginals
        x_uniform = self._uniform_sample(x)
        y_uniform = self._uniform_sample(y)
        
        # Estimate k-nearest neighbor distances
        k = min(10, len(x) // 2)
        from sklearn.neighbors import NearestNeighbors
        
        nn_x = NearestNeighbors(n_neighbors=k).fit(x_uniform.reshape(-1, 1))
        nn_y = NearestNeighbors(n_neighbors=k).fit(y_uniform.reshape(-1, 1))
        
        # Get distances
        x_dists = nn_x.kneighbors()[0]
        y_dists = nn_y.kneighbors()[0]
        
        # Compute average log distances (related to differential entropy)
        x_entropy = np.mean(np.log(x_dists[:, -1]))
        y_entropy = np.mean(np.log(y_dists[:, -1]))
        
        # Return difference in entropies
        return y_entropy - x_entropy
    
    def _slope_complexity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate complexity using slope-based method
        """
        # Sort data
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        
        # Compute slopes
        x_diffs = np.diff(x_sorted)
        y_diffs = np.diff(y_sorted)
        
        # Avoid division by zero
        x_diffs[x_diffs == 0] = np.min(x_diffs[x_diffs > 0]) if np.any(x_diffs > 0) else 1e-10
        
        # Compute log absolute slopes
        log_slopes = np.log(np.abs(y_diffs / x_diffs))
        
        # Return average log slope
        return np.mean(log_slopes)
    
    def test_direction(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Test causal direction using IGCI
        
        Args:
            x: First variable values
            y: Second variable values
            
        Returns:
            Dictionary with test results
        """
        # Ensure 1D arrays
        x = x.flatten()
        y = y.flatten()
        
        # Scale data to [0, 1]
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        # Avoid extreme values
        eps = 1e-10
        x_scaled = x_scaled * (1 - 2*eps) + eps
        y_scaled = y_scaled * (1 - 2*eps) + eps
        
        # Calculate complexity score
        if self.method == 'entropy':
            score = self._entropy_complexity(x_scaled, y_scaled)
        elif self.method == 'slope':
            score = self._slope_complexity(x_scaled, y_scaled)
        else:
            raise ValueError(f"Unknown IGCI method: {self.method}")
        
        # IGCI assumes Y=f(X) if score < 0
        if score < 0:
            direction = "0->1"  # X→Y
            confidence = min(1.0, abs(score))
        else:
            direction = "1->0"  # Y→X
            confidence = min(1.0, abs(score))
        
        return {
            "direction": direction,
            "score": score,
            "confidence": confidence
        }