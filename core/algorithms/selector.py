# core/algorithms/selector.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

class DataType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MIXED = "mixed"

class Distribution(Enum):
    GAUSSIAN = "gaussian"
    NON_GAUSSIAN = "non_gaussian"
    NONLINEAR = "nonlinear"
    UNKNOWN = "unknown"

class AlgorithmSelector:
    """
    Selects appropriate causal discovery algorithms based on data characteristics.
    Implements the comprehensive algorithm selection framework described in the technical proposal.
    """
    
    def __init__(self):
        self.algorithms = {
            # Constraint-based methods
            "pc_fisherz": "PC algorithm with Fisher's Z test (for continuous, Gaussian data)",
            "pc_chisq": "PC algorithm with Chi-square test (for discrete data)",
            "pc_gsq": "PC algorithm with G-square test (for discrete data)",
            "pc_kci": "PC algorithm with Kernel CI test (for nonlinear dependencies)",
            "fci_fisherz": "FCI algorithm with Fisher's Z test (with latent confounders, continuous data)",
            "fci_chisq": "FCI algorithm with Chi-square test (with latent confounders, discrete data)",
            "fci_kci": "FCI algorithm with Kernel CI test (with latent confounders, nonlinear)",
            "cdnod": "CD-NOD algorithm for heterogeneous/nonstationary data",
            
            # Score-based methods
            "ges_bic": "GES algorithm with BIC score (for continuous, Gaussian data)",
            "ges_bdeu": "GES algorithm with BDeu score (for discrete data)",
            "ges_cv": "GES algorithm with CV score (for nonlinear relationships)",
            "grasp": "GRaSP algorithm (permutation-based)",
            "boss": "BOSS algorithm (permutation-based)",
            "exact_dp": "Exact search with dynamic programming",
            "exact_astar": "Exact search with A* algorithm",
            
            # FCM-based methods
            "lingam_ica": "ICA-based LiNGAM (for linear non-Gaussian acyclic models)",
            "lingam_direct": "DirectLiNGAM (for linear non-Gaussian acyclic models)",
            "lingam_var": "VAR-LiNGAM (for time series data)",
            "lingam_rcd": "RCD (for linear non-Gaussian with latent confounders)",
            "lingam_camuv": "CAM-UV (for causal additive models with unobserved variables)",
            "anm": "Additive Noise Model (for nonlinear relationships)",
            "pnl": "Post-Nonlinear causal model (for nonlinear relationships)",
            
            # Hidden causal methods
            "gin": "GIN (for linear non-Gaussian latent variable models)",
            
            # Granger causality
            "granger_test": "Linear Granger causality test (for time series, 2 variables)",
            "granger_lasso": "Linear Granger causality with Lasso (for multivariate time series)"
        }
        
        # Map of method groups
        self.method_groups = {
            "constraint_based": ["pc_fisherz", "pc_chisq", "pc_gsq", "pc_kci", "fci_fisherz", 
                                "fci_chisq", "fci_kci", "cdnod"],
            "score_based": ["ges_bic", "ges_bdeu", "ges_cv", "grasp", "boss", "exact_dp", "exact_astar"],
            "fcm_based": ["lingam_ica", "lingam_direct", "lingam_var", "lingam_rcd", "lingam_camuv", "anm", "pnl"],
            "hidden_causal": ["gin"],
            "granger": ["granger_test", "granger_lasso"]
        }
    
    def get_all_algorithms(self) -> Dict[str, str]:
        """Return all available algorithms with descriptions"""
        return self.algorithms
    
    def get_algorithms_by_group(self, group: str) -> Dict[str, str]:
        """Return algorithms by group name"""
        if group not in self.method_groups:
            raise ValueError(f"Unknown group: {group}. Available groups: {list(self.method_groups.keys())}")
        return {alg: self.algorithms[alg] for alg in self.method_groups[group]}
    
    def analyze_data_properties(self, data: pd.DataFrame, 
                               sample_threshold: int = 500) -> Dict[str, Any]:
        """
        Analyze data properties to determine characteristics for algorithm selection
        
        Args:
            data: DataFrame containing the data
            sample_threshold: Threshold for determining if the sample size is sufficient
            
        Returns:
            Dictionary of data properties
        """
        n_samples, n_features = data.shape
        
        # Determine data type for each column
        dtypes = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Check if actually discrete (integer-like values)
                if data[col].dropna().apply(lambda x: float(x).is_integer()).all():
                    unique_ratio = data[col].nunique() / len(data[col].dropna())
                    dtypes[col] = DataType.DISCRETE if unique_ratio < 0.05 else DataType.CONTINUOUS
                else:
                    dtypes[col] = DataType.CONTINUOUS
            else:
                dtypes[col] = DataType.DISCRETE
        
        # Overall data type
        discrete_count = sum(1 for t in dtypes.values() if t == DataType.DISCRETE)
        continuous_count = sum(1 for t in dtypes.values() if t == DataType.CONTINUOUS)
        
        if discrete_count == n_features:
            overall_type = DataType.DISCRETE
        elif continuous_count == n_features:
            overall_type = DataType.CONTINUOUS
        else:
            overall_type = DataType.MIXED
        
        # Check for Gaussian distribution on continuous variables
        from scipy import stats
        distributions = {}
        for col in data.columns:
            if dtypes[col] == DataType.CONTINUOUS:
                # Apply normality test
                _, p_value = stats.shapiro(data[col].dropna())
                distributions[col] = Distribution.GAUSSIAN if p_value > 0.05 else Distribution.NON_GAUSSIAN
            else:
                distributions[col] = Distribution.UNKNOWN
        
        # Overall distribution characteristic
        if overall_type == DataType.CONTINUOUS:
            gaussian_count = sum(1 for d in distributions.values() if d == Distribution.GAUSSIAN)
            if gaussian_count == n_features:
                overall_distribution = Distribution.GAUSSIAN
            else:
                overall_distribution = Distribution.NON_GAUSSIAN
        else:
            overall_distribution = Distribution.UNKNOWN
            
        # Check if sample size is sufficient
        is_large_sample = n_samples >= sample_threshold
        
        # TODO: Implement checks for time series data and nonlinear relationships
        # This would require more sophisticated tests
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "column_dtypes": dtypes,
            "overall_type": overall_type,
            "distributions": distributions,
            "overall_distribution": overall_distribution,
            "is_large_sample": is_large_sample
        }
        
    def select_algorithms(self, data_properties: Dict[str, Any], 
                          with_latent_confounders: bool = False,
                          is_timeseries: bool = False,
                          nonlinear_relationships: bool = False) -> List[str]:
        """
        Select appropriate algorithms based on data properties and user specifications
        
        Args:
            data_properties: Dictionary of data properties from analyze_data_properties
            with_latent_confounders: If True, consider algorithms that handle latent confounders
            is_timeseries: If True, consider time series algorithms
            nonlinear_relationships: If True, consider algorithms for nonlinear relationships
            
        Returns:
            List of recommended algorithm IDs
        """
        recommended = []
        
        overall_type = data_properties["overall_type"]
        overall_distribution = data_properties["overall_distribution"]
        is_large_sample = data_properties["is_large_sample"]
        n_features = data_properties["n_features"]
        
        # Handle time series data first
        if is_timeseries:
            recommended.append("lingam_var")
            
            if n_features == 2:
                recommended.append("granger_test")
            else:
                recommended.append("granger_lasso")
            
            # Add CD-NOD for nonstationary time series
            recommended.append("cdnod")
            return recommended
        
        # CASE 1: WITH LATENT CONFOUNDERS
        if with_latent_confounders:
            # Add FCI variants
            if overall_type == DataType.CONTINUOUS:
                if overall_distribution == Distribution.GAUSSIAN:
                    recommended.append("fci_fisherz")
                else:
                    recommended.append("fci_fisherz")  # Still works but suboptimal
                    if is_large_sample:
                        recommended.append("lingam_rcd")  # RCD for non-Gaussian with latents
            
            elif overall_type == DataType.DISCRETE:
                recommended.append("fci_chisq")
            
            else:  # MIXED type
                recommended.append("fci_fisherz")  # May work with limitations
            
            # Add for nonlinear relationships
            if nonlinear_relationships and is_large_sample:
                recommended.append("fci_kci")
                recommended.append("lingam_camuv")
            
            # Add GIN for linear, non-Gaussian latent models
            if (overall_type == DataType.CONTINUOUS and 
                overall_distribution == Distribution.NON_GAUSSIAN):
                recommended.append("gin")
            
            return recommended
        
        # CASE 2: WITHOUT LATENT CONFOUNDERS
        
        # Constraint-based methods
        if overall_type == DataType.CONTINUOUS:
            if overall_distribution == Distribution.GAUSSIAN:
                recommended.append("pc_fisherz")
            else:
                recommended.append("pc_fisherz")  # Still works but not optimal
                
                # LiNGAM variants for non-Gaussian continuous
                recommended.append("lingam_ica")
                recommended.append("lingam_direct")
        
        elif overall_type == DataType.DISCRETE:
            recommended.append("pc_chisq")
            recommended.append("pc_gsq")
        
        else:  # MIXED type
            recommended.append("pc_fisherz")  # May work with limitations for mixed
        
        # Add for nonlinear relationships
        if nonlinear_relationships and is_large_sample:
            recommended.append("pc_kci")
            recommended.append("anm")
            recommended.append("pnl")
        
        # Score-based methods
        if overall_type == DataType.CONTINUOUS:
            recommended.append("ges_bic")
        elif overall_type == DataType.DISCRETE:
            recommended.append("ges_bdeu")
        
        # For nonlinear relationships, add GES with CV score
        if nonlinear_relationships:
            recommended.append("ges_cv")
        
        # Add permutation-based methods
        recommended.append("grasp")
        
        # Only add exact search for small problems
        if n_features <= 20:
            recommended.append("boss")
            
        if n_features <= 10:
            recommended.append("exact_astar")
        
        return recommended
    
    def suggest_algorithms(self, profile: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Suggest appropriate causal discovery algorithms based on data profile
        
        Args:
            profile: Data profile from profile_data method
            
        Returns:
            Dictionary of algorithm groups and specific algorithms
        """
        suggestions = {
            "primary": [],
            "secondary": [],
            "not_recommended": []
        }
        
        # Determine if we're dealing with time series
        is_time_series = profile["judgments"]["may_be_time_series"]
        
        # Determine if we should consider latent confounders
        consider_latents = profile["judgments"]["may_have_latent_confounders"]
        
        # Check if we have sufficient sample size
        has_sufficient_samples = profile["sufficient_sample_size"]["structure_learning"]
        
        # Check distribution type
        is_gaussian = profile["overall_distribution"] == "gaussian"
        
        # Check data type
        is_continuous = profile["overall_type"] == "continuous"
        is_discrete = profile["overall_type"] == "discrete"
        is_mixed = profile["overall_type"] == "mixed"
        
        # Check if suitable for nonlinear methods
        can_use_nonlinear = profile["sufficient_sample_size"]["nonlinear_methods"]
        
        # Set for time series data
        if is_time_series:
            suggestions["primary"].append("lingam_var")
            suggestions["primary"].append("granger_lasso")
            suggestions["secondary"].append("cdnod")
            
            # Not recommended for time series
            suggestions["not_recommended"].extend([
                "pc_fisherz", "pc_chisq", "pc_kci", 
                "ges_bic", "ges_bdeu", "exact_astar", "exact_dp"
            ])
            
            return suggestions
        
        # Consider latent confounders
        if consider_latents:
            if is_continuous:
                suggestions["primary"].append("fci_fisherz")
                
                if not is_gaussian:
                    suggestions["primary"].append("lingam_rcd")
                    suggestions["secondary"].append("gin")
                
                if can_use_nonlinear:
                    suggestions["secondary"].append("fci_kci")
                    suggestions["secondary"].append("lingam_camuv")
            
            elif is_discrete:
                suggestions["primary"].append("fci_chisq")
            
            else:  # Mixed
                suggestions["primary"].append("fci_fisherz")  # With limitations
                suggestions["secondary"].append("fci_chisq")  # With limitations
        
        # Without latent confounders
        else:
            # Constraint-based methods
            if is_continuous:
                suggestions["primary"].append("pc_fisherz")
                
                if not is_gaussian:
                    suggestions["primary"].append("lingam_direct")
                    suggestions["secondary"].append("lingam_ica")
            
            elif is_discrete:
                suggestions["primary"].append("pc_chisq")
                suggestions["secondary"].append("pc_gsq")
            
            else:  # Mixed
                suggestions["primary"].append("pc_fisherz")  # With limitations
            
            # Score-based methods
            if is_continuous:
                suggestions["primary"].append("ges_bic")
            elif is_discrete:
                suggestions["primary"].append("ges_bdeu")
            
            # For small graphs, exact methods are feasible
            if profile["n_features"] <= 10:
                suggestions["secondary"].append("exact_astar")
            
            if profile["n_features"] <= 20:
                suggestions["secondary"].append("boss")
            
            # Permutation-based methods
            suggestions["secondary"].append("grasp")
            
            # For nonlinear relationships, if sample size is sufficient
            if can_use_nonlinear:
                suggestions["secondary"].append("pc_kci")
                
                # Pairwise methods (only if few variables)
                if profile["n_features"] <= 5:
                    suggestions["secondary"].append("anm")
                    suggestions["secondary"].append("pnl")
        
        # Methods that generally don't work well with small sample sizes
        if not has_sufficient_samples:
            not_recommended_small_sample = [
                "pc_kci", "fci_kci", "anm", "pnl", 
                "lingam_camuv", "lingam_rcd", "gin"
            ]
            
            # Move any of these from primary/secondary to not_recommended
            for method in not_recommended_small_sample:
                if method in suggestions["primary"]:
                    suggestions["primary"].remove(method)
                    if method not in suggestions["not_recommended"]:
                        suggestions["not_recommended"].append(method)
                
                if method in suggestions["secondary"]:
                    suggestions["secondary"].remove(method)
                    if method not in suggestions["not_recommended"]:
                        suggestions["not_recommended"].append(method)
        
        return suggestions