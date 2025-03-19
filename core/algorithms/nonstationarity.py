# core/algorithms/nonstationarity.py
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy.stats import ks_2samp, chi2_contingency
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NonStationaryCausalDiscovery:
    """
    Causal discovery methods that leverage nonstationarity or heterogeneity in data
    
    These methods are based on the principle that changes in causal mechanisms
    can be exploited to identify causal direction.
    """
    def __init__(self):
        """Initialize non-stationary causal discovery"""
        pass
        
    def detect_distribution_changes(self, 
                                  data: np.ndarray, 
                                  time_index: np.ndarray,
                                  alpha: float = 0.05) -> Dict[int, Dict[str, Any]]:
        """
        Detect changes in variable distributions across time/domains
        
        Args:
            data: Data matrix (samples x variables)
            time_index: Indices marking different time periods or domains
            alpha: Significance level
        
        Returns:
            Dictionary mapping variable indices to change detection results
        """
        n_vars = data.shape[1]
        changes = {}
        
        # Get unique time periods/domains
        unique_times = np.unique(time_index)
        n_periods = len(unique_times)
        
        if n_periods <= 1:
            logger.warning("Only one time period/domain found. Cannot detect changes.")
            return changes
        
        # For each variable
        for i in range(n_vars):
            # Test for distribution changes across time points
            p_vals = []
            change_strengths = []
            
            # Pairwise tests between time points
            for t1 in range(len(unique_times)):
                for t2 in range(t1+1, len(unique_times)):
                    time1 = unique_times[t1]
                    time2 = unique_times[t2]
                    
                    data1 = data[time_index == time1, i]
                    data2 = data[time_index == time2, i]
                    
                    # Check if there are enough samples
                    if len(data1) < 5 or len(data2) < 5:
                        continue
                    
                    # Kolmogorov-Smirnov test for distribution change
                    statistic, p_val = ks_2samp(data1, data2)
                    p_vals.append(p_val)
                    change_strengths.append(statistic)
            
            # If significant changes detected
            if p_vals and np.min(p_vals) < alpha:
                changes[i] = {
                    "changing": True, 
                    "p_value": np.min(p_vals),
                    "strength": np.max(change_strengths),
                    "n_significant_changes": sum(p < alpha for p in p_vals),
                    "total_tests": len(p_vals)
                }
            else:
                changes[i] = {
                    "changing": False, 
                    "p_value": np.min(p_vals) if p_vals else 1.0,
                    "strength": np.max(change_strengths) if change_strengths else 0.0,
                    "n_significant_changes": sum(p < alpha for p in p_vals) if p_vals else 0,
                    "total_tests": len(p_vals) if p_vals else 0
                }
                
        return changes
    
    def detect_relation_changes(self, 
                              data: np.ndarray, 
                              time_index: np.ndarray,
                              alpha: float = 0.05) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Detect changes in relationships between variables across time/domains
        
        Args:
            data: Data matrix (samples x variables)
            time_index: Indices marking different time periods or domains
            alpha: Significance level
        
        Returns:
            Dictionary mapping variable pairs to change detection results
        """
        n_vars = data.shape[1]
        changes = {}
        
        # Get unique time periods/domains
        unique_times = np.unique(time_index)
        n_periods = len(unique_times)
        
        if n_periods <= 1:
            logger.warning("Only one time period/domain found. Cannot detect changes.")
            return changes
        
        # For each pair of variables
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                # Test for relationship changes across time points
                p_vals = []
                change_strengths = []
                corrs = []
                
                # For each time period, compute correlation
                period_corrs = {}
                for t in range(len(unique_times)):
                    time_period = unique_times[t]
                    mask = time_index == time_period
                    
                    if np.sum(mask) < 10:  # Need enough samples
                        continue
                        
                    # Get data for this period
                    data_period = data[mask, :]
                    
                    # Compute correlation for this period
                    corr = np.corrcoef(data_period[:, i], data_period[:, j])[0, 1]
                    period_corrs[time_period] = corr
                    corrs.append(corr)
                
                # Test if correlations are significantly different
                if len(corrs) >= 2:
                    # Fisher's Z transform for correlation differences
                    z_values = 0.5 * np.log((1 + np.array(corrs)) / (1 - np.array(corrs)))
                    
                    # Compute variance of differences
                    var_z = 1.0 / (len(corrs) - 3)
                    
                    # Compute test statistic
                    z_statistic = np.var(z_values) / var_z
                    
                    # p-value from chi-square distribution
                    p_val = 1 - chi2.cdf(z_statistic, len(z_values) - 1)
                    
                    # Measure of change strength: variance of correlations
                    strength = np.var(corrs)
                    
                    changes[(i, j)] = {
                        "changing": p_val < alpha,
                        "p_value": p_val,
                        "strength": strength,
                        "correlations": period_corrs
                    }
        
        return changes
    
    def causal_discovery_nonstationary(self, 
                                     data: np.ndarray, 
                                     time_index: np.ndarray,
                                     alpha: float = 0.05) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Perform causal discovery exploiting nonstationarity
        
        Args:
            data: Data matrix (samples x variables)
            time_index: Indices marking different time periods or domains
            alpha: Significance level
        
        Returns:
            Tuple of (NetworkX DiGraph, additional info)
        """
        n_vars = data.shape[1]
        G = nx.DiGraph()
        G.add_nodes_from(range(n_vars))
        
        # 1. Detect changing modules
        var_changes = self.detect_distribution_changes(data, time_index, alpha)
        
        # 2. Detect changing relationships
        rel_changes = self.detect_relation_changes(data, time_index, alpha)
        
        # 3. Apply CD-NOD principle: If X's mechanism changes but Y doesn't,
        #    and their relationship changes, then X -> Y is more likely
        for (i, j), rel_info in rel_changes.items():
            if rel_info["changing"]:
                # Check if one variable's distribution changes and the other doesn't
                i_changes = var_changes.get(i, {"changing": False})
                j_changes = var_changes.get(j, {"changing": False})
                
                if i_changes["changing"] and not j_changes["changing"]:
                    # Evidence that i -> j
                    G.add_edge(i, j, 
                             weight=rel_info["strength"],
                             nonstationary_evidence=True,
                             p_value=rel_info["p_value"])
                
                elif j_changes["changing"] and not i_changes["changing"]:
                    # Evidence that j -> i
                    G.add_edge(j, i, 
                             weight=rel_info["strength"],
                             nonstationary_evidence=True,
                             p_value=rel_info["p_value"])
        
        # 4. For each changing variable, identify potential effects
        for i, info in var_changes.items():
            if info["changing"]:
                # Calculate correlations with all other variables
                # Higher correlation suggests potential causal relationship
                corrs = []
                for j in range(n_vars):
                    if i != j:
                        corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                        corrs.append((j, abs(corr)))
                
                # Sort by correlation strength
                corrs.sort(key=lambda x: x[1], reverse=True)
                
                # Add edges to top correlated variables
                for j, corr in corrs[:3]:  # Arbitrary: take top 3
                    if corr > 0.3 and not G.has_edge(j, i):  # Only if edge doesn't exist in opposite direction
                        G.add_edge(i, j, 
                                 weight=corr,
                                 nonstationary_evidence=True,
                                 correlation=corr)
        
        additional_info = {
            "variable_changes": var_changes,
            "relationship_changes": rel_changes
        }
        
        return G, additional_info


class DomainAdaptationCausalDiscovery:
    """
    Causal discovery methods that exploit multiple related domains or datasets
    
    These methods use invariance principles to identify causal relationships
    that are consistent across domains.
    """
    def __init__(self):
        """Initialize domain adaptation causal discovery"""
        pass
    
    def detect_invariant_relationships(self, 
                                     datasets: List[np.ndarray],
                                     domain_labels: Optional[List[str]] = None,
                                     alpha: float = 0.05) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Detect relationships that are invariant across domains/datasets
        
        Args:
            datasets: List of data matrices with the same variables
            domain_labels: Optional labels for each dataset
            alpha: Significance level
        
        Returns:
            Dictionary mapping variable pairs to invariance test results
        """
        if not datasets:
            raise ValueError("At least one dataset is required")
        
        n_domains = len(datasets)
        if n_domains < 2:
            logger.warning("At least two domains are required to detect invariance")
            return {}
        
        n_vars = datasets[0].shape[1]
        
        # Check that all datasets have the same number of variables
        for i, data in enumerate(datasets):
            if data.shape[1] != n_vars:
                raise ValueError(f"Dataset {i} has {data.shape[1]} variables, but expected {n_vars}")
        
        # If no domain labels provided, create default ones
        if domain_labels is None:
            domain_labels = [f"Domain{i}" for i in range(n_domains)]
        
        # Dictionary to store invariance results
        invariance_results = {}
        
        # For each pair of variables
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                # Calculate regressions for each domain
                slopes = []
                intercepts = []
                r_squares = []
                
                for domain_idx, data in enumerate(datasets):
                    x = data[:, i]
                    y = data[:, j]
                    
                    # Simple linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    slopes.append(slope)
                    intercepts.append(intercept)
                    r_squares.append(r_value ** 2)
                
                # Test invariance: are slopes consistent across domains?
                # Lower variance in slopes suggests invariance
                slope_var = np.var(slopes)
                
                # Apply an F-test for equality of regression coefficients
                # This is a simplified version of the test
                overall_x = np.concatenate([data[:, i] for data in datasets])
                overall_y = np.concatenate([data[:, j] for data in datasets])
                overall_slope, overall_intercept, r_value, p_value, std_err = stats.linregress(overall_x, overall_y)
                
                # Residual sum of squares for pooled model
                residuals_pooled = overall_y - (overall_slope * overall_x + overall_intercept)
                rss_pooled = np.sum(residuals_pooled ** 2)
                
                # Residual sum of squares for separate models
                rss_separate = 0
                for domain_idx, data in enumerate(datasets):
                    x = data[:, i]
                    y = data[:, j]
                    predicted = slopes[domain_idx] * x + intercepts[domain_idx]
                    residuals = y - predicted
                    rss_separate += np.sum(residuals ** 2)
                
                # Calculate F-statistic
                n_total = len(overall_x)
                k = 2  # Number of parameters in each model (slope, intercept)
                df1 = (n_domains - 1) * k
                df2 = n_total - n_domains * k
                
                if df2 <= 0:
                    # Not enough degrees of freedom
                    invariance_results[(i, j)] = {
                        "invariant": False,
                        "p_value": 1.0,
                        "slopes": slopes,
                        "intercepts": intercepts,
                        "r_squares": r_squares,
                        "error": "Not enough degrees of freedom"
                    }
                    continue
                
                f_stat = ((rss_pooled - rss_separate) / df1) / (rss_separate / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                
                # For invariance, we want high p-value (fail to reject null hypothesis)
                invariant = p_value > alpha
                
                invariance_results[(i, j)] = {
                    "invariant": invariant,
                    "p_value": p_value,
                    "f_statistic": f_stat,
                    "slope_variance": slope_var,
                    "slopes": slopes,
                    "intercepts": intercepts,
                    "r_squares": r_squares,
                    "domains": domain_labels
                }
        
        return invariance_results
    
    def causal_discovery_invariant(self, 
                                 datasets: List[np.ndarray],
                                 domain_labels: Optional[List[str]] = None,
                                 alpha: float = 0.05) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Perform causal discovery using invariance across domains
        
        Args:
            datasets: List of data matrices with the same variables
            domain_labels: Optional labels for each dataset
            alpha: Significance level
        
        Returns:
            Tuple of (NetworkX DiGraph, additional info)
        """
        n_vars = datasets[0].shape[1]
        G = nx.DiGraph()
        G.add_nodes_from(range(n_vars))
        
        # 1. Detect invariant relationships
        invariance_results = self.detect_invariant_relationships(datasets, domain_labels, alpha)
        
        # 2. Combine datasets for overall correlation
        combined_data = np.vstack(datasets)
        
        # 3. For each invariant relationship, determine likely causal direction
        for (i, j), inv_info in invariance_results.items():
            if inv_info["invariant"]:
                # Invariant relationships are more likely to be causal
                # We need to determine the direction
                
                # Calculate correlation
                corr = np.corrcoef(combined_data[:, i], combined_data[:, j])[0, 1]
                
                # To determine direction, we could use various heuristics:
                # - Non-Gaussian methods (if applicable)
                # - Domain-specific knowledge
                # - Time ordering (if available)
                
                # Here we'll use a simple heuristic: stronger R² suggests cause → effect
                r2_x_y = inv_info["r_squares"]  # R² of x predicting y in each domain
                
                # Calculate R² for the reverse direction
                r2_y_x = []
                for domain_idx, data in enumerate(datasets):
                    y = data[:, i]  # Reversed
                    x = data[:, j]  # Reversed
                    _, _, r_value, _, _ = stats.linregress(x, y)
                    r2_y_x.append(r_value ** 2)
                
                # Compare average R²
                avg_r2_x_y = np.mean(r2_x_y)
                avg_r2_y_x = np.mean(r2_y_x)
                
                if avg_r2_x_y > avg_r2_y_x:
                    # Evidence that i → j
                    G.add_edge(i, j, 
                             weight=avg_r2_x_y,
                             invariant=True,
                             p_value=inv_info["p_value"],
                             correlation=corr)
                else:
                    # Evidence that j → i
                    G.add_edge(j, i, 
                             weight=avg_r2_y_x,
                             invariant=True,
                             p_value=inv_info["p_value"],
                             correlation=corr)
        
        additional_info = {
            "invariance_results": invariance_results
        }
        
        return G, additional_info