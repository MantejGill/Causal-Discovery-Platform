# core/algorithms/kernel_methods.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KernelCausalDiscovery:
    """
    Implementation of kernel-based causal discovery methods
    """
    def __init__(self, kernel_type: str = 'rbf', kernel_params: Optional[Dict[str, Any]] = None):
        """
        Initialize kernel-based causal discovery
        
        Args:
            kernel_type: Type of kernel ('rbf', 'polynomial', 'laplacian')
            kernel_params: Parameters for the kernel
        """
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {}
        
    def kernel_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix
        
        Args:
            x: Input data
            
        Returns:
            Kernel matrix
        """
        # Ensure data is properly shaped
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        if self.kernel_type == 'rbf':
            return rbf_kernel(x, **self.kernel_params)
        elif self.kernel_type == 'polynomial':
            return polynomial_kernel(x, **self.kernel_params)
        elif self.kernel_type == 'laplacian':
            return laplacian_kernel(x, **self.kernel_params)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def center_kernel_matrix(self, K: np.ndarray) -> np.ndarray:
        """
        Center a kernel matrix
        
        Args:
            K: Kernel matrix
            
        Returns:
            Centered kernel matrix
        """
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    
    def hsic_test(self, x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Hilbert-Schmidt Independence Criterion test
        
        Args:
            x: First variable
            y: Second variable
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Ensure 1D arrays and reshape for kernel computation
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n = x.shape[0]
        
        # Compute kernel matrices
        K = self.kernel_matrix(x)
        L = self.kernel_matrix(y)
        
        # Center kernel matrices
        K_centered = self.center_kernel_matrix(K)
        L_centered = self.center_kernel_matrix(L)
        
        # Compute HSIC statistic
        hsic = np.trace(K_centered @ L_centered) / (n * n)
        
        # Bootstrap for threshold
        thresholds = []
        for _ in range(500):
            perm = np.random.permutation(n)
            L_perm = L[perm][:, perm]
            L_perm_centered = self.center_kernel_matrix(L_perm)
            hsic_perm = np.trace(K_centered @ L_perm_centered) / (n * n)
            thresholds.append(hsic_perm)
        
        threshold = np.percentile(thresholds, (1-alpha)*100)
        independent = hsic <= threshold
        p_value = np.mean(np.array(thresholds) >= hsic)
        
        return {
            "independent": independent,
            "statistic": hsic,
            "threshold": threshold,
            "p_value": p_value
        }
    
    def conditional_hsic_test(self, 
                             x: np.ndarray, 
                             y: np.ndarray, 
                             z: np.ndarray, 
                             alpha: float = 0.05) -> Dict[str, Any]:
        """
        Conditional Hilbert-Schmidt Independence Criterion test
        
        Args:
            x: First variable
            y: Second variable
            z: Conditioning variable(s)
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Ensure proper dimensions
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
            
        n = x.shape[0]
        
        # Compute kernel matrices
        K_x = self.kernel_matrix(x)
        K_y = self.kernel_matrix(y)
        K_z = self.kernel_matrix(z)
        
        # Center kernel matrices
        K_x_centered = self.center_kernel_matrix(K_x)
        K_y_centered = self.center_kernel_matrix(K_y)
        K_z_centered = self.center_kernel_matrix(K_z)
        
        # Compute residuals by projecting out Z's effect
        # This is a simplified method - more complex methods exist
        K_z_inv = np.linalg.pinv(K_z_centered + 1e-10 * np.eye(n))
        
        K_x_res = K_x_centered - K_x_centered @ K_z_inv @ K_z_centered
        K_y_res = K_y_centered - K_y_centered @ K_z_inv @ K_z_centered
        
        # Compute conditional HSIC statistic
        chsic = np.trace(K_x_res @ K_y_res) / (n * n)
        
        # Bootstrap for threshold
        thresholds = []
        for _ in range(500):
            perm = np.random.permutation(n)
            K_y_perm = K_y[perm][:, perm]
            K_y_perm_centered = self.center_kernel_matrix(K_y_perm)
            K_y_perm_res = K_y_perm_centered - K_y_perm_centered @ K_z_inv @ K_z_centered
            
            chsic_perm = np.trace(K_x_res @ K_y_perm_res) / (n * n)
            thresholds.append(chsic_perm)
        
        threshold = np.percentile(thresholds, (1-alpha)*100)
        independent = chsic <= threshold
        p_value = np.mean(np.array(thresholds) >= chsic)
        
        return {
            "independent": independent,
            "statistic": chsic,
            "threshold": threshold,
            "p_value": p_value
        }
    
    def kernel_gaussian_process_regression(self, 
                                         x: np.ndarray, 
                                         y: np.ndarray) -> Dict[str, Any]:
        """
        Kernel regression using Gaussian Process
        
        Args:
            x: Predictor variable
            y: Response variable
            
        Returns:
            Dictionary with regression results
        """
        # Ensure proper dimensions
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n = x.shape[0]
        
        # Compute kernel matrix
        K = self.kernel_matrix(x)
        
        # Add small noise to diagonal for numerical stability
        K_reg = K + 1e-5 * np.eye(n)
        
        # Compute weights
        try:
            weights = np.linalg.solve(K_reg, y)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            weights = np.linalg.pinv(K_reg) @ y
        
        # Compute predictions
        y_pred = K @ weights
        
        # Compute residuals
        residuals = y - y_pred
        
        return {
            "weights": weights,
            "y_pred": y_pred,
            "residuals": residuals,
            "mse": np.mean(residuals**2)
        }
    
    def kernel_pc_independence_test(self, 
                                  x: np.ndarray, 
                                  y: np.ndarray, 
                                  z: Optional[np.ndarray] = None, 
                                  alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Kernel-based independence test for PC algorithm
        
        Args:
            x: First variable
            y: Second variable
            z: Conditioning set (optional)
            alpha: Significance level
            
        Returns:
            Tuple of (independent, p_value)
        """
        if z is None or z.size == 0:
            # Unconditional test
            result = self.hsic_test(x, y, alpha)
        else:
            # Conditional test
            result = self.conditional_hsic_test(x, y, z, alpha)
        
        return result["independent"], result["p_value"]