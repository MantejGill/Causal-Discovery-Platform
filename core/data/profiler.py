# core/data/profiler.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from scipy import stats
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProfiler:
    """
    Analyzes datasets to determine variable types and characteristics
    for causal discovery algorithm selection.
    """
    
    def __init__(self):
        pass
    
    def profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data to determine characteristics relevant for causal discovery
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            Dictionary of data properties and characteristics
        """
        if data is None or data.empty:
            raise ValueError("Data is empty or None")
        
        n_samples, n_features = data.shape
        
        # Basic dataset properties
        profile = {
            "n_samples": n_samples,
            "n_features": n_features,
            "column_names": list(data.columns),
            "missing_values": data.isna().sum().to_dict(),
            "has_missing_values": data.isna().any().any()
        }
        
        # Determine variable types
        var_types = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Check if actually discrete (integer-like values)
                if data[col].dropna().apply(lambda x: float(x).is_integer()).all():
                    unique_count = data[col].nunique()
                    unique_ratio = unique_count / len(data[col].dropna()) if len(data[col].dropna()) > 0 else 0
                    
                    if unique_count <= 2:
                        var_types[col] = "binary"
                    elif unique_count <= 10 or unique_ratio < 0.05:
                        var_types[col] = "discrete"
                    else:
                        var_types[col] = "continuous"
                else:
                    var_types[col] = "continuous"
            else:
                var_types[col] = "categorical"
        
        profile["variable_types"] = var_types
        
        # Count variable types
        type_counts = {}
        for v_type in var_types.values():
            type_counts[v_type] = type_counts.get(v_type, 0) + 1
        
        profile["type_counts"] = type_counts
        
        # Determine overall data type
        if type_counts.get("continuous", 0) == n_features:
            profile["overall_type"] = "continuous"
        elif type_counts.get("discrete", 0) + type_counts.get("binary", 0) + type_counts.get("categorical", 0) == n_features:
            profile["overall_type"] = "discrete"
        else:
            profile["overall_type"] = "mixed"
        
        # Check for Gaussian distribution for continuous variables
        distributions = {}
        for col in data.columns:
            if var_types[col] == "continuous":
                col_data = data[col].dropna()
                if len(col_data) < 8:  # Minimum size for Shapiro-Wilk test
                    distributions[col] = "unknown"
                    continue
                
                # Apply normality test
                try:
                    _, p_value = stats.shapiro(col_data)
                    distributions[col] = "gaussian" if p_value > 0.05 else "non_gaussian"
                except Exception:
                    distributions[col] = "unknown"
            else:
                distributions[col] = "non_applicable"
        
        profile["distributions"] = distributions
        
        # Determine overall distribution characteristic
        if profile["overall_type"] == "continuous":
            gaussian_count = sum(1 for d in distributions.values() if d == "gaussian")
            if gaussian_count == n_features:
                profile["overall_distribution"] = "gaussian"
            else:
                profile["overall_distribution"] = "non_gaussian"
        else:
            profile["overall_distribution"] = "non_applicable"
        
        # Check for temporal ordering (very basic check)
        has_time_col = any(col.lower() in ["time", "date", "timestamp", "datetime", "year", "month", "day"] 
                          for col in data.columns)
        profile["possible_time_series"] = has_time_col
        
        # Check if sample size is sufficient for various algorithms
        profile["sufficient_sample_size"] = {
            "parametric": n_samples >= 30,  # Rule of thumb for parametric tests
            "nonparametric": n_samples >= 100,  # More samples needed for nonparametric methods
            "structure_learning": n_samples >= n_features * 10,  # Heuristic: 10x samples per feature
            "nonlinear_methods": n_samples >= 200  # Nonlinear methods typically need more data
        }
        
        # Additional statistical properties
        column_stats = {}
        for col in data.columns:
            if var_types[col] in ["continuous", "discrete"]:
                col_data = data[col].dropna()
                column_stats[col] = {
                    "mean": col_data.mean() if len(col_data) > 0 else None,
                    "median": col_data.median() if len(col_data) > 0 else None,
                    "std": col_data.std() if len(col_data) > 0 else None,
                    "min": col_data.min() if len(col_data) > 0 else None,
                    "max": col_data.max() if len(col_data) > 0 else None,
                    "skewness": stats.skew(col_data) if len(col_data) > 0 else None,
                    "kurtosis": stats.kurtosis(col_data) if len(col_data) > 0 else None
                }
            elif var_types[col] in ["categorical", "binary"]:
                col_data = data[col].dropna()
                column_stats[col] = {
                    "unique_values": col_data.nunique() if len(col_data) > 0 else None,
                    "most_common": col_data.value_counts().index[0] if len(col_data) > 0 and col_data.nunique() > 0 else None,
                    "most_common_percentage": col_data.value_counts(normalize=True).max() * 100 if len(col_data) > 0 and col_data.nunique() > 0 else None
                }
        
        profile["column_stats"] = column_stats
        
        # Check for correlations between variables
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = data[numeric_cols].corr('pearson')
                profile["correlation_matrix"] = corr_matrix.to_dict()
                
                # Find strongly correlated pairs
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) >= 0.7:  # Strong correlation threshold
                            corr_pairs.append({
                                "var1": col1,
                                "var2": col2,
                                "correlation": corr
                            })
                
                profile["strong_correlations"] = corr_pairs
        except Exception as e:
            logger.warning(f"Could not compute correlation matrix: {str(e)}")
            profile["correlation_matrix"] = None
            profile["strong_correlations"] = []
        
        # Summary judgments for algorithm selection
        profile["judgments"] = {
            "prefer_nonparametric": profile["overall_distribution"] == "non_gaussian",
            "may_have_latent_confounders": True,  # Default assumption for safety
            "prefer_constraint_based": n_samples < n_features * 15,  # Heuristic
            "prefer_score_based": n_samples >= n_features * 15,  # Heuristic
            "suitable_for_lingam": profile["overall_distribution"] == "non_gaussian" and profile["overall_type"] == "continuous",
            "suitable_for_nonlinear_methods": n_samples >= 200,
            "may_be_time_series": profile["possible_time_series"]
        }
        
        return profile
    
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