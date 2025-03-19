# core/algorithms/timeseries.py
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesCausalDiscovery:
    """
    Advanced causal discovery methods for time series data
    """
    def __init__(self, method: str = 'grangervar'):
        """
        Initialize time series causal discovery
        
        Args:
            method: Method to use ('grangervar', 'granger_pairwise', 'transfer_entropy')
        """
        self.method = method
    
    def discover_causal_graph(self, 
                            data: Union[np.ndarray, pd.DataFrame], 
                            lags: Optional[int] = None,
                            var_names: Optional[List[str]] = None,
                            alpha: float = 0.05) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Discover causal graph from time series data
        
        Args:
            data: Time series data (samples x variables)
            lags: Maximum lag to consider (auto-determined if None)
            var_names: Variable names (optional)
            alpha: Significance level for tests
            
        Returns:
            Tuple of (NetworkX DiGraph, additional info)
        """
        # Convert to numpy array if DataFrame
        if isinstance(data, pd.DataFrame):
            if var_names is None:
                var_names = data.columns.tolist()
            data_np = data.values
        else:
            data_np = data
            if var_names is None:
                var_names = [f"var{i}" for i in range(data_np.shape[1])]
        
        n_vars = data_np.shape[1]
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n_vars):
            G.add_node(i, name=var_names[i])
        
        # Auto-determine lags if not provided
        if lags is None:
            # Heuristic: min of 10 or 1/5 of series length
            lags = min(10, int(data_np.shape[0] / 5))
        
        additional_info = {"lags": lags, "method": self.method}
        
        # Choose method
        if self.method == 'grangervar':
            try:
                from statsmodels.tsa.api import VAR
                
                # Fit VAR model
                model = VAR(data_np)
                results = model.fit(lags)
                additional_info["var_results"] = results
                
                # Extract Granger causality relationships from VAR coefficients
                coefs = results.coefs  # Shape: (lags, n_vars, n_vars)
                
                # Aggregate coefficients across lags
                agg_coefs = np.zeros((n_vars, n_vars))
                for l in range(lags):
                    agg_coefs += np.abs(coefs[l])
                
                # Add edges based on coefficient magnitude
                for i in range(n_vars):  # effect
                    for j in range(n_vars):  # cause
                        if i != j:
                            # Test if any lag coefficients are significant
                            for lag in range(lags):
                                coef = coefs[lag, i, j]
                                # Add edge if coefficient exceeds threshold
                                if np.abs(coef) > 0.1:  # Simple threshold
                                    G.add_edge(j, i, 
                                             weight=np.abs(agg_coefs[i, j]),
                                             time_lag=lag+1, 
                                             coefficient=coef)
                                    break
                
            except Exception as e:
                logger.error(f"Error in VAR model fitting: {str(e)}")
                # Fall back to simpler method if statsmodels not available
                self.method = 'granger_pairwise'
        
        if self.method == 'granger_pairwise':
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                
                # Perform pairwise Granger causality tests
                test_results = {}
                
                for i in range(n_vars):  # effect
                    for j in range(n_vars):  # cause
                        if i != j:
                            # Extract the two time series
                            y = data_np[:, i]
                            x = data_np[:, j]
                            
                            # Stack the two variables
                            test_data = np.column_stack((y, x))
                            
                            # Perform Granger causality test
                            try:
                                test = grangercausalitytests(test_data, maxlag=lags, verbose=False)
                                
                                # Extract p-values (using ssr F-test)
                                p_values = [test[lag+1][0]['ssr_ftest'][1] for lag in range(lags)]
                                min_p_value = min(p_values)
                                min_p_lag = p_values.index(min_p_value) + 1
                                
                                test_results[(j, i)] = {
                                    "p_values": p_values,
                                    "min_p_value": min_p_value,
                                    "min_p_lag": min_p_lag
                                }
                                
                                # Add edge if significant
                                if min_p_value < alpha:
                                    G.add_edge(j, i, 
                                             weight=1.0 - min_p_value,  # Higher weight for lower p-value
                                             p_value=min_p_value,
                                             time_lag=min_p_lag)
                            
                            except Exception as e:
                                logger.warning(f"Granger test failed for variables {j}->{i}: {str(e)}")
                
                additional_info["test_results"] = test_results
                
            except Exception as e:
                logger.error(f"Error in Granger pairwise tests: {str(e)}")
        
        elif self.method == 'transfer_entropy':
            # Implementation of transfer entropy (information-theoretic measure)
            # This is more complex and requires additional libraries
            logger.warning("Transfer entropy method not fully implemented")
            
            # Placeholder for transfer entropy calculation
            additional_info["warning"] = "Transfer entropy method not fully implemented"
        
        return G, additional_info
    
    def detect_instantaneous_effects(self, 
                                   data: Union[np.ndarray, pd.DataFrame],
                                   significance: float = 0.05) -> nx.DiGraph:
        """
        Detect instantaneous causal effects in time series
        
        Args:
            data: Time series data
            significance: Significance level
            
        Returns:
            NetworkX DiGraph of instantaneous effects
        """
        # Convert to numpy array if DataFrame
        if isinstance(data, pd.DataFrame):
            var_names = data.columns.tolist()
            data_np = data.values
        else:
            data_np = data
            var_names = [f"var{i}" for i in range(data_np.shape[1])]
        
        n_vars = data_np.shape[1]
        G_inst = nx.DiGraph()
        
        # Add nodes
        for i in range(n_vars):
            G_inst.add_node(i, name=var_names[i])
        
        # Detect instantaneous effects using residuals from time lagged models
        # This approach requires fitting VAR models first to account for time-lagged effects
        try:
            from statsmodels.tsa.api import VAR
            
            # Determine reasonable lag based on data
            max_lag = min(10, int(data_np.shape[0] / 5))
            
            # Fit VAR model
            model = VAR(data_np)
            results = model.fit(max_lag)
            
            # Get residuals (these should only contain instantaneous effects)
            residuals = results.resid
            
            # Compute correlation matrix of residuals
            corr_matrix = np.corrcoef(residuals.T)
            
            # Simple heuristic: use correlation of residuals as evidence of instantaneous effects
            # In a more sophisticated approach, one would use structural learning on the residuals
            
            for i in range(n_vars):
                for j in range(i+1, n_vars):  # Upper triangular to avoid duplicates
                    corr = corr_matrix[i, j]
                    
                    if abs(corr) > 0.3:  # Arbitrary threshold
                        # For instantaneous effects, we create bidirectional edges
                        # Since we can't determine direction from correlation alone
                        G_inst.add_edge(i, j, weight=abs(corr), instantaneous=True)
                        G_inst.add_edge(j, i, weight=abs(corr), instantaneous=True)
            
            return G_inst
            
        except Exception as e:
            logger.error(f"Error detecting instantaneous effects: {str(e)}")
            return G_inst
    
    def combine_temporal_instantaneous_graph(self, 
                                          temporal_graph: nx.DiGraph,
                                          instantaneous_graph: nx.DiGraph) -> nx.DiGraph:
        """
        Combine temporal and instantaneous causal graphs
        
        Args:
            temporal_graph: Graph with temporal causal relations
            instantaneous_graph: Graph with instantaneous effects
            
        Returns:
            Combined causal graph
        """
        # Create a copy of the temporal graph
        combined_graph = temporal_graph.copy()
        
        # Add instantaneous edges
        for u, v, data in instantaneous_graph.edges(data=True):
            if combined_graph.has_edge(u, v):
                # Edge already exists in temporal graph, update attributes
                combined_graph[u][v]['instantaneous'] = True
                if 'weight' in data:
                    # Take max of weights
                    combined_graph[u][v]['weight'] = max(
                        combined_graph[u][v].get('weight', 0),
                        data['weight']
                    )
            else:
                # Add new edge
                combined_graph.add_edge(u, v, **data)
        
        return combined_graph


class VARLiNGAM:
    """
    Vector Autoregressive Linear Non-Gaussian Acyclic Model
    
    Combines VAR modeling with LiNGAM to handle both time-lagged and
    contemporaneous causal effects.
    """
    def __init__(self, lags: int = 1):
        """
        Initialize VAR-LiNGAM
        
        Args:
            lags: Number of time lags to include
        """
        self.lags = lags
        self.ar_coefs = None  # Autoregressive coefficients
        self.lingam_coefs = None  # LiNGAM contemporaneous coefficients
        self.residuals = None
    
    def fit(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Fit VAR-LiNGAM model
        
        Args:
            data: Time series data (samples x variables)
            
        Returns:
            Dictionary with fitting results
        """
        n_samples, n_vars = data.shape
        
        # Step 1: Fit VAR model to capture time-lagged effects
        try:
            from statsmodels.tsa.api import VAR
            
            # Fit VAR model
            var_model = VAR(data)
            var_results = var_model.fit(self.lags)
            
            # Extract coefficients and residuals
            self.ar_coefs = var_results.coefs  # Shape: (lags, n_vars, n_vars)
            residuals = var_results.resid
            
            # Step 2: Fit LiNGAM on residuals to capture contemporaneous effects
            try:
                # Try to use causallearn's LiNGAM
                from causallearn.search.FCMBased.lingam import DirectLiNGAM
                
                lingam_model = DirectLiNGAM()
                lingam_model.fit(residuals)
                
                # Extract results
                self.lingam_coefs = lingam_model.adjacency_matrix_
                self.residuals = residuals @ (np.eye(n_vars) - self.lingam_coefs)
                
                return {
                    "success": True,
                    "ar_coefs": self.ar_coefs,
                    "lingam_coefs": self.lingam_coefs,
                    "residuals": self.residuals,
                    "var_results": var_results
                }
                
            except Exception as e:
                logger.warning(f"Error fitting LiNGAM: {str(e)}. Using simplified approach.")
                
                # Simplified approach: use correlation structure of residuals
                corr_matrix = np.corrcoef(residuals.T)
                
                # Create a simple adjacency matrix based on correlations
                # This is a very simplified approach
                adj_matrix = np.zeros((n_vars, n_vars))
                for i in range(n_vars):
                    for j in range(n_vars):
                        if i != j and abs(corr_matrix[i, j]) > 0.3:
                            adj_matrix[j, i] = corr_matrix[i, j]
                
                self.lingam_coefs = adj_matrix
                self.residuals = residuals
                
                return {
                    "success": True,
                    "ar_coefs": self.ar_coefs,
                    "lingam_coefs": self.lingam_coefs,
                    "residuals": self.residuals,
                    "var_results": var_results,
                    "warning": "Used correlation-based approximation instead of LiNGAM"
                }
                
        except Exception as e:
            logger.error(f"Error fitting VAR-LiNGAM: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def to_networkx_graph(self, var_names: Optional[List[str]] = None) -> nx.DiGraph:
        """
        Convert VAR-LiNGAM model to NetworkX graph
        
        Args:
            var_names: Variable names
            
        Returns:
            NetworkX DiGraph of causal relationships
        """
        if self.ar_coefs is None or self.lingam_coefs is None:
            raise ValueError("Model has not been fitted yet")
        
        n_vars = self.lingam_coefs.shape[0]
        
        if var_names is None:
            var_names = [f"var{i}" for i in range(n_vars)]
        
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n_vars):
            G.add_node(i, name=var_names[i])
        
        # Add contemporaneous edges (from LiNGAM)
        for i in range(n_vars):
            for j in range(n_vars):
                if self.lingam_coefs[i, j] != 0:
                    G.add_edge(j, i, 
                             weight=abs(self.lingam_coefs[i, j]),
                             contemporaneous=True,
                             coefficient=self.lingam_coefs[i, j])
        
        # Add time-lagged edges (from VAR)
        for lag in range(self.lags):
            for i in range(n_vars):
                for j in range(n_vars):
                    if self.ar_coefs[lag, i, j] != 0:
                        # If the edge already exists (from contemporaneous effects),
                        # we could update it, but here we'll create separate lagged nodes
                        lagged_node = f"{j}_lag{lag+1}"
                        if lagged_node not in G:
                            G.add_node(lagged_node, name=f"{var_names[j]}(t-{lag+1})", lagged=True)
                        
                        G.add_edge(lagged_node, i,
                                 weight=abs(self.ar_coefs[lag, i, j]),
                                 time_lag=lag+1,
                                 coefficient=self.ar_coefs[lag, i, j])
        
        return G