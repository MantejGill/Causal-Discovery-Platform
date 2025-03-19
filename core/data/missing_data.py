# core/data/missing_data.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
import networkx as nx
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissingDataHandler:
    """
    Advanced methods for handling missing data with causal awareness
    """
    
    def __init__(self):
        """Initialize MissingDataHandler"""
        pass
    
    def detect_missingness_type(self, data: pd.DataFrame) -> str:
        """
        Detect the type of missingness in the data (MCAR, MAR, MNAR)
        
        Args:
            data: DataFrame with missing values
            
        Returns:
            String indicating the likely missingness mechanism
        """
        # Create missingness indicators
        miss_indicators = data.isna().astype(int)
        
        # If no missing values, return early
        if not miss_indicators.any().any():
            return "NONE"
        
        # Test for MCAR: missingness indicators should be independent of observed values
        mcar_pvals = []
        for col in data.columns:
            miss_col = miss_indicators[col]
            if miss_col.sum() > 0:  # If there are missing values
                for other_col in data.columns:
                    if other_col != col:
                        # Test independence between missingness and observed values
                        observed = data[other_col][~data[col].isna()]
                        miss_indicator = miss_indicators[other_col][~data[col].isna()]
                        
                        if len(observed) > 0 and len(miss_indicator) > 0 and miss_indicator.sum() > 0:
                            if pd.api.types.is_numeric_dtype(observed):
                                # For numeric data, compare means between missing and non-missing groups
                                missing_group = observed[miss_indicator == 1]
                                nonmissing_group = observed[miss_indicator == 0]
                                
                                if len(missing_group) > 0 and len(nonmissing_group) > 0:
                                    try:
                                        _, p_val = stats.ttest_ind(missing_group, nonmissing_group, 
                                                                 equal_var=False, nan_policy='omit')
                                        mcar_pvals.append(p_val)
                                    except:
                                        # If t-test fails, try non-parametric test
                                        try:
                                            _, p_val = stats.mannwhitneyu(missing_group, nonmissing_group)
                                            mcar_pvals.append(p_val)
                                        except:
                                            pass
                            else:
                                # For categorical data, use chi-square test
                                try:
                                    table = pd.crosstab(miss_indicator, observed)
                                    if table.shape[0] > 1 and table.shape[1] > 1:
                                        _, p_val, _, _ = stats.chi2_contingency(table)
                                        mcar_pvals.append(p_val)
                                except:
                                    pass
        
        # If all p-values are high, likely MCAR
        if mcar_pvals and np.mean([p < 0.05 for p in mcar_pvals]) < 0.1:
            return "MCAR"
        
        # Test for MAR vs MNAR
        # MAR: missingness in one variable depends on observed values of other variables
        # MNAR: missingness depends on the missing values themselves
        
        # This is a heuristic approach - true MNAR is hard to detect without additional data
        mar_evidence = 0
        total_tests = 0
        
        for col in data.columns:
            miss_col = miss_indicators[col]
            if miss_col.sum() > 0:  # If there are missing values
                for other_col in data.columns:
                    if other_col != col:
                        # Check if other column's values predict missingness in this column
                        nonmissing_indices = ~data[other_col].isna()
                        if nonmissing_indices.sum() > 0:
                            other_values = data[other_col][nonmissing_indices]
                            miss_flags = miss_col[nonmissing_indices]
                            
                            if pd.api.types.is_numeric_dtype(other_values) and miss_flags.sum() > 0:
                                try:
                                    # Logistic regression to predict missingness
                                    from sklearn.linear_model import LogisticRegression
                                    X = other_values.values.reshape(-1, 1)
                                    y = miss_flags.values
                                    
                                    if np.unique(y).size > 1:  # Need both classes
                                        model = LogisticRegression(solver='liblinear')
                                        model.fit(X, y)
                                        
                                        # If other column predicts missingness, evidence for MAR
                                        score = model.score(X, y)
                                        if score > 0.6:  # Simple threshold
                                            mar_evidence += 1
                                        total_tests += 1
                                except:
                                    pass
        
        # Based on the proportion of tests showing MAR evidence
        if total_tests > 0:
            mar_ratio = mar_evidence / total_tests
            if mar_ratio > 0.3:  # If there's substantial evidence for MAR
                return "MAR"
            else:
                # If not clearly MAR or MCAR, default to MNAR
                return "MNAR"
        else:
            # Not enough tests to determine
            return "MAR"  # Default assumption (safest)
    
    def multiple_imputation_causal(self, 
                                 data: pd.DataFrame, 
                                 causal_graph: Optional[nx.DiGraph] = None,
                                 num_imputations: int = 5) -> pd.DataFrame:
        """
        Perform multiple imputation with causal awareness
        
        Args:
            data: DataFrame with missing values
            causal_graph: Optional causal graph to guide imputation
            num_imputations: Number of imputations to perform
            
        Returns:
            DataFrame with imputed values
        """
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        except ImportError:
            logger.error("Required libraries not found. Install scikit-learn with: pip install scikit-learn")
            # Fallback to simple imputation
            from sklearn.impute import SimpleImputer
            logger.warning("Falling back to simple mean imputation")
            imputer = SimpleImputer(strategy='mean')
            return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        # Determine column types
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()
        
        # If causal graph is provided, use it to determine imputation order
        if causal_graph is not None:
            # Topological sort to impute causes before effects
            try:
                topo_order = list(nx.topological_sort(causal_graph))
                # Map node indices to column names
                col_mapping = {}
                for node in causal_graph.nodes():
                    # Get column name from node attribute or index
                    if 'name' in causal_graph.nodes[node]:
                        col_mapping[node] = causal_graph.nodes[node]['name']
                    elif isinstance(node, int) and node < len(data.columns):
                        col_mapping[node] = data.columns[node]
                
                # Create ordered list of columns
                columns_ordered = [col_mapping.get(node, str(node)) for node in topo_order 
                                 if node in col_mapping and col_mapping[node] in data.columns]
                
                # Add any columns not in the graph
                missing_cols = [col for col in data.columns if col not in columns_ordered]
                columns_ordered.extend(missing_cols)
            except nx.NetworkXUnfeasible:
                # Graph has cycles, can't do topological sort
                logger.warning("Causal graph has cycles, using default column order")
                columns_ordered = data.columns.tolist()
        else:
            columns_ordered = data.columns.tolist()
        
        # Create a DataFrame with reordered columns
        data_ordered = data[columns_ordered].copy()
        
        # Handle categorical variables
        if cat_cols:
            # One-hot encode categorical variables
            data_with_dummies = pd.get_dummies(data_ordered, columns=cat_cols, drop_first=False)
            
            # Remember original categorical columns and their encoded columns
            cat_mappings = {}
            for col in cat_cols:
                encoded_cols = [c for c in data_with_dummies.columns if c.startswith(f"{col}_")]
                cat_mappings[col] = encoded_cols
        else:
            data_with_dummies = data_ordered
            cat_mappings = {}
        
        # Create a estimator that works well with mixed data types
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Create causal-aware imputer
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=10,
            random_state=42,
            sample_posterior=True,  # Sample from predictive distribution
            skip_complete=True  # Skip columns with no missing values
        )
        
        # Perform multiple imputations
        imputations = []
        for i in range(num_imputations):
            # Different random seed for each imputation
            imputer.random_state = 42 + i
            
            # Impute
            imp_data = imputer.fit_transform(data_with_dummies)
            imp_df = pd.DataFrame(imp_data, columns=data_with_dummies.columns)
            
            # Convert back from one-hot encoding to categorical
            if cat_mappings:
                for cat_col, encoded_cols in cat_mappings.items():
                    # Convert one-hot back to categorical
                    # Get index of maximum value for each row
                    cat_values = []
                    for _, row in imp_df[encoded_cols].iterrows():
                        max_idx = row.values.argmax()
                        cat_values.append(encoded_cols[max_idx].split('_')[-1])
                    
                    # Add back original column
                    imp_df[cat_col] = cat_values
                
                # Keep only original columns
                imp_df = imp_df[data_ordered.columns]
            
            imputations.append(imp_df)
        
        # Combine imputations (e.g., average for numeric, mode for categorical)
        combined = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            if col in numeric_cols:
                # Average for numeric columns
                combined[col] = pd.concat([imp[col] for imp in imputations]).groupby(level=0).mean()
            else:
                # Mode for categorical columns
                combined[col] = pd.concat([imp[col] for imp in imputations]).groupby(level=0).agg(
                    lambda x: x.value_counts().index[0]
                )
        
        return combined
    
    def regression_imputation(self, 
                            data: pd.DataFrame, 
                            target_col: str,
                            predictor_cols: Optional[List[str]] = None) -> pd.Series:
        """
        Impute missing values in a column using regression on other columns
        
        Args:
            data: DataFrame with data
            target_col: Column to impute
            predictor_cols: Columns to use as predictors (None for all other columns)
            
        Returns:
            Series with imputed values
        """
        # Get rows where target is missing
        missing_mask = data[target_col].isna()
        if not missing_mask.any():
            return data[target_col]  # No missing values
            
        # Copy the column to avoid modifying the input
        target = data[target_col].copy()
        
        # If no predictor columns specified, use all other columns
        if predictor_cols is None:
            predictor_cols = [col for col in data.columns if col != target_col]
        
        # Filter to only include columns with no missing values in the relevant rows
        valid_predictors = []
        for col in predictor_cols:
            if not data.loc[~missing_mask, col].isna().any():
                valid_predictors.append(col)
        
        if not valid_predictors:
            logger.warning(f"No valid predictor columns for {target_col}. Using mean imputation.")
            target[missing_mask] = target[~missing_mask].mean()
            return target
        
        # Determine if target is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(target)
        
        if is_numeric:
            # Numeric target - use linear regression
            try:
                from sklearn.linear_model import LinearRegression
                
                # Prepare data
                X_train = data.loc[~missing_mask, valid_predictors]
                y_train = data.loc[~missing_mask, target_col]
                
                # Handle categorical predictors
                X_train = pd.get_dummies(X_train, drop_first=True)
                
                # Fit model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Prepare test data
                X_test = data.loc[missing_mask, valid_predictors]
                X_test = pd.get_dummies(X_test, drop_first=True)
                
                # Ensure train and test have the same columns
                missing_cols = set(X_train.columns) - set(X_test.columns)
                for col in missing_cols:
                    X_test[col] = 0
                
                X_test = X_test[X_train.columns]
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Impute
                target[missing_mask] = y_pred
                
            except Exception as e:
                logger.error(f"Error in regression imputation: {str(e)}")
                # Fallback to mean imputation
                target[missing_mask] = target[~missing_mask].mean()
        else:
            # Categorical target - use logistic regression or classification
            try:
                from sklearn.ensemble import RandomForestClassifier
                
                # Prepare data
                X_train = data.loc[~missing_mask, valid_predictors]
                y_train = data.loc[~missing_mask, target_col]
                
                # Handle categorical predictors
                X_train = pd.get_dummies(X_train, drop_first=True)
                
                # Fit model
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Prepare test data
                X_test = data.loc[missing_mask, valid_predictors]
                X_test = pd.get_dummies(X_test, drop_first=True)
                
                # Ensure train and test have the same columns
                missing_cols = set(X_train.columns) - set(X_test.columns)
                for col in missing_cols:
                    X_test[col] = 0
                
                X_test = X_test[X_train.columns]
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Impute
                target[missing_mask] = y_pred
                
            except Exception as e:
                logger.error(f"Error in classification imputation: {str(e)}")
                # Fallback to mode imputation
                target[missing_mask] = target[~missing_mask].mode()[0]
        
        return target
    
    def causal_imputation(self, 
                        data: pd.DataFrame, 
                        causal_graph: nx.DiGraph) -> pd.DataFrame:
        """
        Impute missing values using causal relations
        
        Args:
            data: DataFrame with missing values
            causal_graph: Causal graph with relationships
            
        Returns:
            DataFrame with imputed values
        """
        # Create a copy of the data
        imputed_data = data.copy()
        
        # Get column names from graph if available
        col_mapping = {}
        for node in causal_graph.nodes():
            # Get column name from node attribute or index
            if 'name' in causal_graph.nodes[node]:
                col_mapping[node] = causal_graph.nodes[node]['name']
            elif isinstance(node, int) and node < len(data.columns):
                col_mapping[node] = data.columns[node]
        
        # Try to get topological order to impute causes before effects
        try:
            topo_order = list(nx.topological_sort(causal_graph))
            # Map node indices to column names
            ordered_columns = [col_mapping.get(node, str(node)) for node in topo_order 
                             if node in col_mapping and col_mapping[node] in data.columns]
            
            # Add any columns not in the graph
            missing_cols = [col for col in data.columns if col not in ordered_columns]
            ordered_columns.extend(missing_cols)
            
        except nx.NetworkXUnfeasible:
            # Graph has cycles, can't do topological sort
            logger.warning("Causal graph has cycles, using default column order")
            ordered_columns = data.columns.tolist()
        
        # Impute each column in order
        for col in ordered_columns:
            if imputed_data[col].isna().any():
                # Find causal parents (predictors) for this column
                predictor_cols = []
                
                # Get node ID for this column
                target_node = None
                for node, name in col_mapping.items():
                    if name == col:
                        target_node = node
                        break
                
                if target_node is not None:
                    # Get parents (causes) of this node
                    for parent in causal_graph.predecessors(target_node):
                        if parent in col_mapping and col_mapping[parent] in data.columns:
                            predictor_cols.append(col_mapping[parent])
                
                # If no causal parents found, use all previously imputed columns
                if not predictor_cols:
                    predictor_cols = [c for c in ordered_columns[:ordered_columns.index(col)]]
                
                # Impute using regression
                imputed_data[col] = self.regression_imputation(
                    imputed_data, col, predictor_cols)
        
        return imputed_data