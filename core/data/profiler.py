import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy import stats
import logging
import warnings
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProfiler:
    """
    Analyzes datasets to determine variable types and characteristics
    for causal discovery algorithm selection.
    Provides comprehensive data profiling, statistical analyses,
    and algorithm recommendations based on data characteristics.
    """
    
    def __init__(self, sample_max_rows: int = 10000):
        """
        Initialize DataProfiler
        
        Args:
            sample_max_rows: Maximum number of rows to use for computationally 
                            intensive profiling operations (for performance)
        """
        self.sample_max_rows = sample_max_rows
    
    def profile_data(self, data: pd.DataFrame, detailed: bool = True) -> Dict[str, Any]:
        """
        Analyze data to determine characteristics relevant for causal discovery
        
        Args:
            data: DataFrame containing the data
            detailed: Whether to perform detailed profiling (may be slower for large datasets)
            
        Returns:
            Dictionary of data properties and characteristics
        """
        if data is None or data.empty:
            raise ValueError("Data is empty or None")
        
        n_samples, n_features = data.shape
        
        # For very large datasets, work with a sample for performance
        if n_samples > self.sample_max_rows and detailed:
            logger.info(f"Dataset has {n_samples} samples. Using {self.sample_max_rows} rows for detailed profiling.")
            sample_data = data.sample(self.sample_max_rows, random_state=42)
        else:
            sample_data = data
        
        # Basic dataset properties
        profile = {
            "n_samples": n_samples,
            "n_features": n_features,
            "column_names": list(data.columns),
            "missing_values": data.isna().sum().to_dict(),
            "has_missing_values": data.isna().any().any(),
            "missing_percentage": data.isna().sum().sum() / (n_samples * n_features) * 100
        }
        
        # Determine variable types with improved detection
        var_types = self._determine_variable_types(data)
        profile["variable_types"] = var_types
        
        # Count variable types
        type_counts = {}
        for v_type in var_types.values():
            type_counts[v_type] = type_counts.get(v_type, 0) + 1
        
        profile["type_counts"] = type_counts
        
        # Determine overall data type
        profile["overall_type"] = self._determine_overall_type(type_counts, n_features)
        
        # Check for distribution characteristics
        distributions = self._analyze_distributions(sample_data, var_types)
        profile["distributions"] = distributions
        
        # Determine overall distribution characteristic
        profile["overall_distribution"] = self._determine_overall_distribution(
            profile["overall_type"], 
            distributions, 
            n_features
        )
        
        # Check for temporal patterns (enhanced time series detection)
        temporal_info = self._detect_temporal_patterns(data)
        profile.update(temporal_info)
        
        # Check if sample size is sufficient for various algorithms
        profile["sufficient_sample_size"] = self._evaluate_sample_sufficiency(n_samples, n_features)
        
        # Statistical properties by column
        profile["column_stats"] = self._compute_column_stats(sample_data, var_types)
        
        # Analyze correlations and relationships
        if detailed:
            correlation_info = self._analyze_correlations(sample_data, var_types)
            profile.update(correlation_info)
            
            # Detect potential outliers
            profile["outliers"] = self._detect_outliers(sample_data, var_types)
            
            # Check for multicollinearity
            profile["multicollinearity"] = self._check_multicollinearity(sample_data, var_types)
        
        # Additional dataset characteristics for algorithm selection
        profile["dataset_characteristics"] = self._derive_dataset_characteristics(
            sample_data, profile["overall_type"], profile["overall_distribution"]
        )
        
        # Summary judgments for algorithm selection
        profile["judgments"] = self._make_algorithm_judgments(profile)
        
        return profile
    
    def _determine_variable_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Determine the type of each variable using improved detection rules
        
        Args:
            data: The input DataFrame
            
        Returns:
            Dictionary mapping column names to variable types
        """
        var_types = {}
        
        for col in data.columns:
            # Check if datetime
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                var_types[col] = "datetime"
                continue
            
            # Check if numeric
            if pd.api.types.is_numeric_dtype(data[col]):
                col_data = data[col].dropna()
                
                # Empty column case
                if len(col_data) == 0:
                    var_types[col] = "unknown"
                    continue
                
                # Check if binary (only two unique values)
                unique_values = col_data.unique()
                if len(unique_values) <= 2:
                    var_types[col] = "binary"
                    continue
                
                # Check if actually discrete (integer-like values)
                try:
                    if col_data.apply(lambda x: float(x).is_integer()).all():
                        unique_count = col_data.nunique()
                        unique_ratio = unique_count / len(col_data)
                        
                        if unique_count <= 10 or unique_ratio < 0.05:
                            var_types[col] = "discrete"
                        else:
                            var_types[col] = "continuous"
                    else:
                        var_types[col] = "continuous"
                except:
                    # If any conversion issues, default to continuous
                    var_types[col] = "continuous"
            
            # Check if categorical/string
            elif pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                unique_count = data[col].nunique()
                if unique_count <= 2:
                    var_types[col] = "binary"
                else:
                    var_types[col] = "categorical"
            
            # Check if boolean
            elif pd.api.types.is_bool_dtype(data[col]):
                var_types[col] = "binary"
            
            # Default to categorical for other types
            else:
                var_types[col] = "categorical"
        
        return var_types
    
    def _determine_overall_type(self, type_counts: Dict[str, int], n_features: int) -> str:
        """
        Determine the overall data type based on variable type counts
        
        Args:
            type_counts: Dictionary of variable type counts
            n_features: Total number of features
            
        Returns:
            Overall data type
        """
        if type_counts.get("continuous", 0) == n_features:
            return "continuous"
        elif type_counts.get("continuous", 0) / n_features >= 0.8:
            return "mostly_continuous"
        elif (type_counts.get("discrete", 0) + type_counts.get("binary", 0) + 
              type_counts.get("categorical", 0)) == n_features:
            return "discrete"
        elif (type_counts.get("discrete", 0) + type_counts.get("binary", 0) + 
              type_counts.get("categorical", 0)) / n_features >= 0.8:
            return "mostly_discrete"
        else:
            return "mixed"
    
    def _analyze_distributions(self, data: pd.DataFrame, var_types: Dict[str, str]) -> Dict[str, str]:
        """
        Analyze the distribution of continuous variables using multiple tests
        
        Args:
            data: The input DataFrame
            var_types: Dictionary mapping column names to variable types
            
        Returns:
            Dictionary mapping column names to distribution types
        """
        distributions = {}
        
        for col in data.columns:
            if var_types[col] == "continuous":
                col_data = data[col].dropna()
                
                # Skip if too few samples
                if len(col_data) < 8:
                    distributions[col] = "unknown"
                    continue
                
                # Apply normality tests
                try:
                    # Shapiro-Wilk test (best for n < 2000)
                    if len(col_data) < 2000:
                        _, p_shapiro = stats.shapiro(col_data)
                        is_normal_shapiro = p_shapiro > 0.05
                    else:
                        is_normal_shapiro = None
                    
                    # D'Agostino-Pearson test (combines skew and kurtosis)
                    _, p_dagostino = stats.normaltest(col_data)
                    is_normal_dagostino = p_dagostino > 0.05
                    
                    # Anderson-Darling test
                    result = stats.anderson(col_data, dist='norm')
                    # Critical values at 5% significance
                    critical_value = result.critical_values[2]  # index 2 corresponds to 5% significance
                    is_normal_anderson = result.statistic < critical_value
                    
                    # Combine results
                    if is_normal_shapiro is not None:
                        is_normal = is_normal_shapiro and is_normal_dagostino and is_normal_anderson
                    else:
                        is_normal = is_normal_dagostino and is_normal_anderson
                    
                    distributions[col] = "gaussian" if is_normal else "non_gaussian"
                    
                    # Additional distribution checks if not Gaussian
                    if not is_normal:
                        # Check skewness
                        skewness = stats.skew(col_data)
                        if abs(skewness) > 1:
                            distributions[col] = "skewed_non_gaussian"
                        
                        # Check for heavy tails
                        kurtosis = stats.kurtosis(col_data)
                        if kurtosis > 2:
                            distributions[col] = "heavy_tailed"
                except Exception as e:
                    logger.warning(f"Error testing distribution for column {col}: {str(e)}")
                    distributions[col] = "unknown"
            
            elif var_types[col] in ["discrete", "binary", "categorical"]:
                distributions[col] = "discrete"
            
            elif var_types[col] == "datetime":
                distributions[col] = "datetime"
            
            else:
                distributions[col] = "unknown"
        
        return distributions
    
    def _determine_overall_distribution(self, overall_type: str, 
                                      distributions: Dict[str, str], 
                                      n_features: int) -> str:
        """
        Determine the overall distribution type of the dataset
        
        Args:
            overall_type: Overall data type
            distributions: Dictionary of column distributions
            n_features: Total number of features
            
        Returns:
            Overall distribution characteristic
        """
        if overall_type in ["continuous", "mostly_continuous"]:
            gaussian_count = sum(1 for d in distributions.values() if d == "gaussian")
            gaussian_ratio = gaussian_count / n_features
            
            if gaussian_ratio > 0.8:
                return "gaussian"
            elif gaussian_ratio > 0.5:
                return "mostly_gaussian"
            else:
                skewed_count = sum(1 for d in distributions.values() if d == "skewed_non_gaussian")
                heavy_tailed_count = sum(1 for d in distributions.values() if d == "heavy_tailed")
                
                if skewed_count > heavy_tailed_count:
                    return "skewed"
                elif heavy_tailed_count > 0:
                    return "heavy_tailed"
                else:
                    return "non_gaussian"
        
        elif overall_type in ["discrete", "mostly_discrete"]:
            return "discrete"
        
        else:  # mixed
            return "mixed"
    
    def _detect_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect temporal patterns and time series characteristics
        
        Args:
            data: The input DataFrame
            
        Returns:
            Dictionary with temporal pattern information
        """
        result = {
            "possible_time_series": False,
            "temporal_columns": [],
            "sequential_indices": False
        }
        
        # Check for datetime columns
        datetime_cols = [col for col in data.columns 
                       if pd.api.types.is_datetime64_any_dtype(data[col])]
        
        # Check for columns with time-related names
        time_related_names = ["time", "date", "year", "month", "day", "hour", "minute", 
                             "second", "timestamp", "period", "week", "quarter"]
        
        time_name_cols = [col for col in data.columns 
                         if any(time_term in col.lower() for time_term in time_related_names)]
        
        # Check if index is datetime
        index_is_datetime = pd.api.types.is_datetime64_any_dtype(data.index)
        
        # Check if index is sequential
        if isinstance(data.index, pd.RangeIndex):
            result["sequential_indices"] = True
        else:
            try:
                # Check if index is numeric and sequential
                index_as_array = np.array(data.index)
                if np.issubdtype(index_as_array.dtype, np.number):
                    diffs = np.diff(index_as_array)
                    if np.all(diffs == diffs[0]) and diffs[0] > 0:
                        result["sequential_indices"] = True
            except:
                pass
        
        # Combine findings
        temporal_columns = list(set(datetime_cols + time_name_cols))
        result["temporal_columns"] = temporal_columns
        result["possible_time_series"] = (len(temporal_columns) > 0 or 
                                         index_is_datetime or 
                                         result["sequential_indices"])
        
        # Extra check: if more than 90% of columns have time-based names, it's likely time series
        if len(time_name_cols) / len(data.columns) > 0.9:
            result["possible_time_series"] = True
            result["likely_panel_data"] = True
        
        return result
    
    def _evaluate_sample_sufficiency(self, n_samples: int, n_features: int) -> Dict[str, bool]:
        """
        Evaluate if sample size is sufficient for different algorithm types
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Dictionary with sample sufficiency judgments
        """
        return {
            "parametric": n_samples >= 30,  # Minimum for parametric tests
            "nonparametric": n_samples >= 100,  # More samples for nonparametric
            "structure_learning": n_samples >= max(30, n_features * 10),  # Heuristic: 10x samples per feature
            "nonlinear_methods": n_samples >= 200,  # Nonlinear methods need more data
            "high_dimensional": n_samples >= n_features * 20,  # For high-dimensional learning
            "time_series": n_samples >= 50  # Minimum for time series analysis
        }
    
    def _compute_column_stats(self, data: pd.DataFrame, var_types: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Compute detailed statistics for each column
        
        Args:
            data: The input DataFrame
            var_types: Dictionary mapping column names to variable types
            
        Returns:
            Dictionary with column statistics
        """
        column_stats = {}
        
        for col in data.columns:
            col_data = data[col].dropna()
            
            # Skip empty columns
            if len(col_data) == 0:
                column_stats[col] = {"empty": True}
                continue
            
            if var_types[col] in ["continuous", "discrete"]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        # Basic statistics
                        stats_dict = {
                            "mean": col_data.mean(),
                            "median": col_data.median(),
                            "std": col_data.std(),
                            "min": col_data.min(),
                            "max": col_data.max(),
                            "range": col_data.max() - col_data.min(),
                            "iqr": col_data.quantile(0.75) - col_data.quantile(0.25),
                            "skewness": stats.skew(col_data),
                            "kurtosis": stats.kurtosis(col_data)
                        }
                        
                        # Quantiles
                        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
                        stats_dict.update({
                            f"q{int(q*100)}": col_data.quantile(q) for q in quantiles
                        })
                        
                        # Define additional metrics based on variable type
                        if var_types[col] == "discrete":
                            stats_dict.update({
                                "unique_values": col_data.nunique(),
                                "mode": col_data.mode().iloc[0] if not col_data.mode().empty else None,
                                "entropy": stats.entropy(col_data.value_counts(normalize=True))
                            })
                        
                        column_stats[col] = stats_dict
                except Exception as e:
                    logger.warning(f"Error computing statistics for column {col}: {str(e)}")
                    column_stats[col] = {"error": str(e)}
            
            elif var_types[col] in ["categorical", "binary"]:
                try:
                    # Value counts
                    value_counts = col_data.value_counts()
                    norm_counts = col_data.value_counts(normalize=True)
                    
                    # Most common values
                    most_common = value_counts.index[0] if not value_counts.empty else None
                    
                    stats_dict = {
                        "unique_values": col_data.nunique(),
                        "most_common": most_common,
                        "most_common_percentage": norm_counts.iloc[0] * 100 if not norm_counts.empty else None,
                        "entropy": stats.entropy(norm_counts),
                        "top_categories": value_counts.head(5).to_dict() if not value_counts.empty else {}
                    }
                    
                    column_stats[col] = stats_dict
                except Exception as e:
                    logger.warning(f"Error computing statistics for column {col}: {str(e)}")
                    column_stats[col] = {"error": str(e)}
            
            elif var_types[col] == "datetime":
                try:
                    # Time-specific statistics
                    stats_dict = {
                        "min_date": col_data.min(),
                        "max_date": col_data.max(),
                        "range_days": (col_data.max() - col_data.min()).days 
                                     if hasattr((col_data.max() - col_data.min()), 'days') else None,
                        "unique_dates": col_data.nunique()
                    }
                    
                    # Try to extract year, month components if possible
                    try:
                        year_counts = col_data.dt.year.value_counts()
                        month_counts = col_data.dt.month.value_counts()
                        
                        stats_dict.update({
                            "year_range": [col_data.dt.year.min(), col_data.dt.year.max()],
                            "most_common_year": year_counts.index[0] if not year_counts.empty else None,
                            "most_common_month": month_counts.index[0] if not month_counts.empty else None
                        })
                    except:
                        # Some datetime columns may not support these attributes
                        pass
                    
                    column_stats[col] = stats_dict
                except Exception as e:
                    logger.warning(f"Error computing statistics for datetime column {col}: {str(e)}")
                    column_stats[col] = {"error": str(e)}
            
            else:  # unknown type
                column_stats[col] = {"type": "unknown"}
        
        return column_stats
    
    def _analyze_correlations(self, data: pd.DataFrame, var_types: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze correlations and relationships between variables
        
        Args:
            data: The input DataFrame
            var_types: Dictionary mapping column names to variable types
            
        Returns:
            Dictionary with correlation analysis
        """
        result = {
            "correlation_matrix": None,
            "strong_correlations": [],
            "correlation_clusters": []
        }
        
        try:
            # Get numeric columns
            numeric_cols = [col for col in data.columns 
                          if var_types[col] in ["continuous", "discrete", "binary"]]
            
            if len(numeric_cols) >= 2:
                # Compute correlation matrix
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr_matrix = data[numeric_cols].corr('pearson')
                
                result["correlation_matrix"] = corr_matrix.to_dict()
                
                # Find strongly correlated pairs
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) >= 0.5:  # Moderate to strong correlation
                            corr_pairs.append({
                                "var1": col1,
                                "var2": col2,
                                "correlation": corr,
                                "strength": "very_strong" if abs(corr) > 0.8 else
                                           "strong" if abs(corr) > 0.7 else
                                           "moderate" if abs(corr) > 0.5 else "weak"
                            })
                
                result["strong_correlations"] = sorted(
                    corr_pairs, 
                    key=lambda x: abs(x["correlation"]), 
                    reverse=True
                )
                
                # Find correlation clusters (groups of highly correlated variables)
                # This can be useful for dimensionality reduction and detecting potential latent factors
                clusters = self._find_correlation_clusters(corr_matrix, threshold=0.7)
                result["correlation_clusters"] = clusters
                
                # Rank variables by average absolute correlation with others
                avg_correlations = {}
                for col in numeric_cols:
                    correlations = [abs(corr_matrix.loc[col, other_col]) 
                                  for other_col in numeric_cols if other_col != col]
                    avg_correlations[col] = sum(correlations) / len(correlations) if correlations else 0
                
                result["avg_correlations"] = avg_correlations
            
            # For categorical columns, compute Cramer's V statistic for pairs
            categorical_cols = [col for col in data.columns if var_types[col] in ["categorical", "binary"]]
            if len(categorical_cols) >= 2:
                cat_associations = []
                
                for i in range(len(categorical_cols)):
                    for j in range(i+1, len(categorical_cols)):
                        col1, col2 = categorical_cols[i], categorical_cols[j]
                        try:
                            cramers_v = self._cramers_v(data[col1], data[col2])
                            if cramers_v >= 0.3:  # Moderate to strong association
                                cat_associations.append({
                                    "var1": col1,
                                    "var2": col2,
                                    "cramers_v": cramers_v,
                                    "strength": "very_strong" if cramers_v > 0.8 else
                                              "strong" if cramers_v > 0.6 else
                                              "moderate" if cramers_v > 0.3 else "weak"
                                })
                        except Exception as e:
                            logger.warning(f"Error computing Cramer's V for {col1} and {col2}: {str(e)}")
                
                result["categorical_associations"] = sorted(
                    cat_associations, 
                    key=lambda x: x["cramers_v"], 
                    reverse=True
                )
            
            # Analyze potential nonlinear relationships
            result["nonlinear_candidates"] = self._detect_nonlinear_relationships(data, numeric_cols)
            
        except Exception as e:
            logger.warning(f"Error in correlation analysis: {str(e)}")
        
        return result
    
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate Cramer's V statistic for categorical-categorical association
        
        Args:
            x: First categorical variable
            y: Second categorical variable
            
        Returns:
            Cramer's V statistic
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    
    def _find_correlation_clusters(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find clusters of highly correlated variables
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold for considering variables as clustered
            
        Returns:
            List of correlation clusters
        """
        clusters = []
        remaining_columns = set(corr_matrix.columns)
        
        while remaining_columns:
            # Start a new cluster with the first remaining column
            current_col = next(iter(remaining_columns))
            cluster = {current_col}
            remaining_columns.remove(current_col)
            
            # Find all correlated columns
            for col in list(remaining_columns):
                if abs(corr_matrix.loc[current_col, col]) >= threshold:
                    cluster.add(col)
                    remaining_columns.remove(col)
            
            # Add the cluster if it has at least 2 variables
            if len(cluster) > 1:
                # Calculate average intra-cluster correlation
                intra_corr = 0
                count = 0
                for i, col1 in enumerate(cluster):
                    for col2 in list(cluster)[i+1:]:
                        intra_corr += abs(corr_matrix.loc[col1, col2])
                        count += 1
                
                avg_intra_corr = intra_corr / count if count > 0 else 0
                
                clusters.append({
                    "variables": list(cluster),
                    "size": len(cluster),
                    "avg_correlation": avg_intra_corr
                })
        
        # Sort clusters by size and average correlation
        return sorted(clusters, key=lambda x: (x["size"], x["avg_correlation"]), reverse=True)
    
    def _detect_nonlinear_relationships(self, data: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """
        Detect potential nonlinear relationships between variables
        
        Args:
            data: The input DataFrame
            numeric_cols: List of numeric column names
            
        Returns:
            List of potential nonlinear relationships
        """
        nonlinear_candidates = []
        
        # Check pairs with low linear correlation but potential nonlinear relationship
        if len(numeric_cols) < 2:
            return nonlinear_candidates
        
        try:
            # For computational efficiency, limit to 10 variables
            if len(numeric_cols) > 10:
                # Select variables with highest variance
                variances = {col: data[col].var() for col in numeric_cols if not pd.isna(data[col].var())}
                sorted_vars = sorted(variances.items(), key=lambda x: x[1], reverse=True)
                selected_cols = [col for col, _ in sorted_vars[:10]]
            else:
                selected_cols = numeric_cols
            
            # Check for non-linear relationships using Spearman's rank correlation
            pearson_corr = data[selected_cols].corr('pearson')
            spearman_corr = data[selected_cols].corr('spearman')
            
            for i in range(len(selected_cols)):
                for j in range(i+1, len(selected_cols)):
                    col1, col2 = selected_cols[i], selected_cols[j]
                    
                    pearson = pearson_corr.loc[col1, col2]
                    spearman = spearman_corr.loc[col1, col2]
                    
                    # If Spearman is significantly higher than Pearson, it suggests nonlinear relationship
                    if abs(spearman) - abs(pearson) > 0.3 and abs(spearman) > 0.5:
                        nonlinear_candidates.append({
                            "var1": col1,
                            "var2": col2,
                            "pearson": pearson,
                            "spearman": spearman,
                            "difference": abs(spearman) - abs(pearson)
                        })
            
            return sorted(nonlinear_candidates, key=lambda x: x["difference"], reverse=True)
            
        except Exception as e:
            logger.warning(f"Error detecting nonlinear relationships: {str(e)}")
            return []
    
    def _detect_outliers(self, data: pd.DataFrame, var_types: Dict[str, str]) -> Dict[str, Any]:
        """
        Detect potential outliers in the dataset
        
        Args:
            data: The input DataFrame
            var_types: Dictionary mapping column names to variable types
            
        Returns:
            Dictionary with outlier information
        """
        outlier_info = {}
        
        for col in data.columns:
            if var_types[col] in ["continuous", "discrete"]:
                try:
                    col_data = data[col].dropna()
                    if len(col_data) < 5:  # Need enough data
                        continue
                    
                    # Z-score method
                    z_scores = stats.zscore(col_data)
                    outliers_z = np.abs(z_scores) > 3
                    
                    # IQR method
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers_iqr = (col_data < lower_bound) | (col_data > upper_bound)
                    
                    # Combined result
                    outlier_count_z = outliers_z.sum()
                    outlier_count_iqr = outliers_iqr.sum()
                    
                    # Consider a variable as having outliers if both methods find some
                    if outlier_count_z > 0 and outlier_count_iqr > 0:
                        outlier_percentage = (outlier_count_iqr / len(col_data)) * 100
                        if outlier_percentage > 0.1:  # Only report if at least 0.1%
                            outlier_info[col] = {
                                "outlier_count_z": int(outlier_count_z),
                                "outlier_count_iqr": int(outlier_count_iqr),
                                "outlier_percentage": outlier_percentage,
                                "lower_bound": float(lower_bound),
                                "upper_bound": float(upper_bound)
                            }
                except Exception as e:
                    logger.warning(f"Error detecting outliers for column {col}: {str(e)}")
        
        # Overall assessment
        if outlier_info:
            total_outliers = sum(info["outlier_count_iqr"] for info in outlier_info.values())
            total_values = sum(len(data[col].dropna()) for col in outlier_info.keys())
            overall_percentage = (total_outliers / total_values) * 100 if total_values > 0 else 0
            
            return {
                "variables_with_outliers": outlier_info,
                "total_outlier_count": total_outliers,
                "overall_percentage": overall_percentage,
                "outliers_present": len(outlier_info) > 0
            }
        else:
            return {
                "variables_with_outliers": {},
                "total_outlier_count": 0,
                "overall_percentage": 0,
                "outliers_present": False
            }
    
    def _check_multicollinearity(self, data: pd.DataFrame, var_types: Dict[str, str]) -> Dict[str, Any]:
        """
        Check for multicollinearity in the dataset
        
        Args:
            data: The input DataFrame
            var_types: Dictionary mapping column names to variable types
            
        Returns:
            Dictionary with multicollinearity information
        """
        result = {
            "has_multicollinearity": False,
            "vif_scores": {},
            "highly_collinear_variables": []
        }
        
        # Get numeric columns
        numeric_cols = [col for col in data.columns 
                       if var_types[col] in ["continuous", "discrete", "binary"]]
        
        if len(numeric_cols) < 3:  # Need at least 3 variables for meaningful multicollinearity
            return result
        
        try:
            # Copy data to avoid modifying the original
            num_data = data[numeric_cols].copy()
            
            # Fill missing values with mean for this analysis
            for col in num_data.columns:
                if num_data[col].isna().any():
                    num_data[col] = num_data[col].fillna(num_data[col].mean())
            
            # Try to estimate VIF scores
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                
                # Add constant for statsmodels
                from statsmodels.tools.tools import add_constant
                X = add_constant(num_data)
                
                # Calculate VIF for each feature
                vif_scores = {}
                
                for i, col in enumerate(X.columns):
                    if col != 'const':  # Skip the constant column
                        try:
                            vif = variance_inflation_factor(X.values, i)
                            if not np.isnan(vif) and not np.isinf(vif):
                                vif_scores[col] = vif
                        except:
                            continue
                
                # Identify highly collinear variables (VIF > 5)
                high_vif_vars = {var: vif for var, vif in vif_scores.items() if vif > 5}
                
                if high_vif_vars:
                    result["has_multicollinearity"] = True
                    result["vif_scores"] = vif_scores
                    result["highly_collinear_variables"] = [{
                        "variable": var, 
                        "vif": vif,
                        "severity": "extreme" if vif > 10 else "high"
                    } for var, vif in high_vif_vars.items()]
            except:
                # If statsmodels is not available, use correlation-based approximation
                corr_matrix = num_data.corr()
                
                # Identify pairs with correlation > 0.8
                high_corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > 0.8:
                            high_corr_pairs.append({
                                "var1": col1,
                                "var2": col2,
                                "correlation": corr
                            })
                
                # Count how many times each variable appears in high correlation pairs
                var_counts = Counter()
                for pair in high_corr_pairs:
                    var_counts[pair["var1"]] += 1
                    var_counts[pair["var2"]] += 1
                
                # Variables that appear in multiple high correlation pairs likely have multicollinearity
                if high_corr_pairs:
                    result["has_multicollinearity"] = True
                    result["high_correlation_pairs"] = high_corr_pairs
                    result["variable_high_corr_counts"] = dict(var_counts)
        
        except Exception as e:
            logger.warning(f"Error checking multicollinearity: {str(e)}")
        
        return result
    
    def _derive_dataset_characteristics(self, data: pd.DataFrame, 
                                      overall_type: str, 
                                      overall_distribution: str) -> Dict[str, Any]:
        """
        Derive additional dataset characteristics for algorithm selection
        
        Args:
            data: The input DataFrame
            overall_type: Overall data type
            overall_distribution: Overall distribution type
            
        Returns:
            Dictionary with derived characteristics
        """
        n_samples, n_features = data.shape
        
        characteristics = {
            "high_dimensional": n_features > 20,
            "small_sample": n_samples < 100,
            "large_sample": n_samples > 1000,
            "balanced_dimensions": 0.1 <= (n_features / n_samples) <= 10,
            "sparse_data": data.isna().sum().sum() / (n_samples * n_features) > 0.1,
            "gaussian_compatible": overall_distribution in ["gaussian", "mostly_gaussian"],
            "discrete_compatible": overall_type in ["discrete", "mostly_discrete"],
            "linear_compatible": overall_type in ["continuous", "mostly_continuous"] and 
                                n_samples >= n_features * 10,
            "nonlinear_compatible": n_samples >= 200 and n_features <= 20,
            "complexity_level": "high" if n_features > 20 else
                              "medium" if n_features > 10 else "low"
        }
        
        return characteristics
    
    def _make_algorithm_judgments(self, profile: Dict[str, Any]) -> Dict[str, bool]:
        """
        Make judgments about suitable algorithm classes based on the profile
        
        Args:
            profile: Data profile
            
        Returns:
            Dictionary with algorithm suitability judgments
        """
        judgments = {
            "prefer_nonparametric": profile["overall_distribution"] in ["non_gaussian", "skewed", "heavy_tailed"],
            "may_have_latent_confounders": True,  # Default assumption for safety
            "prefer_constraint_based": profile["n_samples"] < profile["n_features"] * 15 or profile["n_features"] > 15,
            "prefer_score_based": profile["n_samples"] >= profile["n_features"] * 15 and profile["n_features"] <= 15,
            "suitable_for_lingam": profile["overall_distribution"] in ["non_gaussian", "skewed"] and 
                                 profile["overall_type"] in ["continuous", "mostly_continuous"],
            "suitable_for_nonlinear_methods": profile["sufficient_sample_size"]["nonlinear_methods"] and 
                                           profile["n_features"] <= 20,
            "may_be_time_series": profile.get("possible_time_series", False),
            "may_have_nonlinear_relationships": len(profile.get("nonlinear_candidates", [])) > 0 
                                              if "nonlinear_candidates" in profile else False,
            "has_outliers": profile.get("outliers", {}).get("outliers_present", False) 
                          if "outliers" in profile else False,
            "has_multicollinearity": profile.get("multicollinearity", {}).get("has_multicollinearity", False) 
                                   if "multicollinearity" in profile else False,
            "suitable_for_exact_methods": profile["n_features"] <= 8,
            "needs_robust_methods": profile.get("outliers", {}).get("outliers_present", False) 
                                 if "outliers" in profile else False,
            "may_have_distribution_shifts": profile.get("dataset_characteristics", {}).get("sparse_data", False)
        }
        
        return judgments
    
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
        
        # Check if there are potential nonlinear relationships
        has_nonlinear = profile["judgments"].get("may_have_nonlinear_relationships", False)
        
        # Check if data has outliers
        has_outliers = profile["judgments"].get("has_outliers", False)
        
        # Check for multicollinearity
        has_multicollinearity = profile["judgments"].get("has_multicollinearity", False)
        
        # Check dimension
        high_dimensional = profile["n_features"] > 20
        
        # Set for time series data
        if is_time_series:
            suggestions["primary"].extend(["lingam_var", "granger_lasso", "var_lingam"])
            suggestions["secondary"].extend(["cdnod", "timeseries_grangervar", "timeseries_granger_pairwise"])
            
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
                
                if can_use_nonlinear and has_nonlinear:
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
                    if has_outliers:
                        # LiNGAM can be sensitive to outliers
                        suggestions["secondary"].append("lingam_direct")
                    else:
                        suggestions["primary"].append("lingam_direct")
                        suggestions["secondary"].append("lingam_ica")
            
            elif is_discrete:
                suggestions["primary"].append("pc_chisq")
                suggestions["secondary"].append("pc_gsq")
            
            else:  # Mixed
                suggestions["primary"].append("pc_fisherz")  # With limitations
            
            # Score-based methods
            if is_continuous and not has_outliers:
                suggestions["primary"].append("ges_bic")
            elif is_discrete:
                suggestions["primary"].append("ges_bdeu")
            
            # For small graphs, exact methods are feasible
            if profile["n_features"] <= 8:
                suggestions["secondary"].append("exact_astar")
            
            if profile["n_features"] <= 20:
                suggestions["secondary"].append("boss")
            
            # Permutation-based methods
            suggestions["secondary"].append("grasp")
            
            # For nonlinear relationships, if sample size is sufficient
            if can_use_nonlinear and has_nonlinear:
                suggestions["primary" if has_nonlinear else "secondary"].append("pc_kci")
                
                # Pairwise methods (only if few variables)
                if profile["n_features"] <= 5:
                    suggestions["primary" if has_nonlinear else "secondary"].append("anm")
                    suggestions["secondary"].append("pnl")
        
        # Methods that generally don't work well with small sample sizes
        if not has_sufficient_samples:
            not_recommended_small_sample = [
                "pc_kci", "fci_kci", "anm", "pnl", 
                "lingam_camuv", "lingam_rcd", "gin",
                "boss", "exact_astar", "exact_dp"
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
        
        # Methods that are sensitive to multicollinearity
        if has_multicollinearity:
            methods_sensitive_to_multicollinearity = ["ges_bic", "ges_bdeu", "lingam_ica"]
            
            for method in methods_sensitive_to_multicollinearity:
                if method in suggestions["primary"]:
                    suggestions["primary"].remove(method)
                    suggestions["secondary"].append(method)
        
        # Methods that may struggle with high-dimensional data
        if high_dimensional:
            methods_not_for_high_dim = ["exact_astar", "exact_dp", "boss", "anm", "pnl"]
            
            for method in methods_not_for_high_dim:
                if method in suggestions["primary"]:
                    suggestions["primary"].remove(method)
                    suggestions["not_recommended"].append(method)
                if method in suggestions["secondary"]:
                    suggestions["secondary"].remove(method)
                    suggestions["not_recommended"].append(method)
            
            # Add methods good for high dimensions
            if "grasp" not in suggestions["primary"] and "grasp" not in suggestions["secondary"]:
                suggestions["secondary"].append("grasp")
        
        # Check for distribution shifts
        if profile["judgments"].get("may_have_distribution_shifts", False):
            suggestions["secondary"].append("nonstationary")
            suggestions["secondary"].append("cdnod")
        
        # Ensure at least one algorithm is recommended
        if not suggestions["primary"] and not suggestions["secondary"]:
            suggestions["primary"].append("pc_fisherz")  # Default fallback
        
        return suggestions

    def generate_variable_summaries(self, profile: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate natural language summaries for each variable
        
        Args:
            profile: Data profile from profile_data method
            
        Returns:
            Dictionary mapping column names to text summaries
        """
        summaries = {}
        
        var_types = profile.get("variable_types", {})
        col_stats = profile.get("column_stats", {})
        distributions = profile.get("distributions", {})
        
        for col in var_types:
            # Skip if no statistics available
            if col not in col_stats:
                summaries[col] = f"No summary available for {col}"
                continue
            
            # Get statistics
            stats = col_stats[col]
            var_type = var_types[col]
            distribution = distributions.get(col, "unknown")
            
            # Generate appropriate summary based on variable type
            if var_type in ["continuous", "discrete"]:
                # Handle edge case when stats dictionary is missing keys
                if "mean" not in stats or "min" not in stats or "max" not in stats:
                    summaries[col] = f"Limited statistics available for {col} ({var_type})"
                    continue
                    
                # Summary for numeric variables
                summary = f"{col} is a {var_type} variable "
                
                # Range information
                summary += f"ranging from {stats.get('min', 'unknown')} to {stats.get('max', 'unknown')} "
                
                # Central tendency
                summary += f"with a mean of {stats.get('mean', 'unknown'):.2f} and median of {stats.get('median', 'unknown'):.2f}. "
                
                # Distribution
                if distribution == "gaussian":
                    summary += "The distribution appears to be normal. "
                elif distribution == "skewed_non_gaussian":
                    skew = stats.get("skewness", 0)
                    if skew > 0:
                        summary += "The distribution is positively skewed (right-tailed). "
                    else:
                        summary += "The distribution is negatively skewed (left-tailed). "
                elif distribution == "heavy_tailed":
                    summary += "The distribution has heavy tails with potential outliers. "
                
                # Variability
                if "std" in stats:
                    summary += f"Standard deviation is {stats.get('std', 'unknown'):.2f}. "
                
                # Unique values for discrete variables
                if var_type == "discrete" and "unique_values" in stats:
                    summary += f"There are {stats.get('unique_values', 'unknown')} unique values. "
                    
                summaries[col] = summary.strip()
                
            elif var_type in ["categorical", "binary"]:
                # Edge case handling
                if "unique_values" not in stats:
                    summaries[col] = f"Limited statistics available for {col} ({var_type})"
                    continue
                    
                # Summary for categorical variables
                summary = f"{col} is a {var_type} variable "
                
                # Number of categories
                summary += f"with {stats.get('unique_values', 'unknown')} unique values. "
                
                # Most common value
                if "most_common" in stats and "most_common_percentage" in stats:
                    summary += f"The most common value is '{stats.get('most_common', 'unknown')}' "
                    summary += f"occurring in {stats.get('most_common_percentage', 'unknown'):.1f}% of the data. "
                    
                # Entropy
                if "entropy" in stats:
                    entropy = stats.get("entropy", 0)
                    if entropy < 0.5:
                        summary += "The distribution is highly imbalanced. "
                    elif entropy < 1.5:
                        summary += "The distribution is moderately balanced. "
                    else:
                        summary += "The distribution is well balanced across categories. "
                        
                summaries[col] = summary.strip()
                
            elif var_type == "datetime":
                # Edge case handling
                if "min_date" not in stats or "max_date" not in stats:
                    summaries[col] = f"Limited statistics available for {col} (datetime)"
                    continue
                    
                # Summary for datetime variables
                summary = f"{col} is a datetime variable "
                
                # Date range
                summary += f"ranging from {stats.get('min_date', 'unknown')} to {stats.get('max_date', 'unknown')}. "
                
                # Time span
                if "range_days" in stats:
                    days = stats.get("range_days", 0)
                    years = days / 365.25 if days else 0
                    
                    if years >= 1:
                        summary += f"The time span is approximately {years:.1f} years. "
                    else:
                        summary += f"The time span is {days} days. "
                
                # Unique dates
                if "unique_dates" in stats:
                    summary += f"There are {stats.get('unique_dates', 'unknown')} unique timestamps. "
                    
                summaries[col] = summary.strip()
                
            else:
                # Summary for unknown types
                summaries[col] = f"{col} is a variable of unknown or unsupported type."
        
        return summaries