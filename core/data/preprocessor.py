# core/data/preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles data preprocessing operations for causal discovery.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataPreprocessor
        
        Args:
            data: DataFrame to preprocess
        """
        self.original_data = data.copy()
        self.current_data = data.copy()
        self.preprocessing_steps = []
        self.column_transformations = {}
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the current preprocessed data
        
        Returns:
            Current version of preprocessed DataFrame
        """
        return self.current_data
    
    def reset(self) -> None:
        """Reset preprocessing to original data"""
        self.current_data = self.original_data.copy()
        self.preprocessing_steps = []
        self.column_transformations = {}
    
    def get_preprocessing_steps(self) -> List[Dict[str, Any]]:
        """
        Get list of preprocessing steps applied
        
        Returns:
            List of preprocessing step details
        """
        return self.preprocessing_steps
    
    def handle_missing_values(self, 
                            method: str = 'drop', 
                            columns: Optional[List[str]] = None,
                            fill_value: Optional[Any] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            method: Method to handle missing values ('drop', 'mean', 'median', 'mode', 'constant', 'knn')
            columns: Specific columns to process (None for all)
            fill_value: Value to use for 'constant' method
            
        Returns:
            Preprocessed DataFrame
        """
        target_columns = columns if columns is not None else self.current_data.columns
        data_before = self.current_data.copy()
        
        try:
            if method == 'drop':
                if columns is None:
                    # Drop rows with any missing values
                    self.current_data = self.current_data.dropna()
                else:
                    # Drop rows with missing values in specified columns
                    self.current_data = self.current_data.dropna(subset=columns)
            
            elif method == 'mean':
                for col in target_columns:
                    if pd.api.types.is_numeric_dtype(self.current_data[col]):
                        imputer = SimpleImputer(strategy='mean')
                        self.current_data[col] = imputer.fit_transform(self.current_data[[col]])
                        self.column_transformations[col] = {"type": "imputer", "method": "mean", "imputer": imputer}
            
            elif method == 'median':
                for col in target_columns:
                    if pd.api.types.is_numeric_dtype(self.current_data[col]):
                        imputer = SimpleImputer(strategy='median')
                        self.current_data[col] = imputer.fit_transform(self.current_data[[col]])
                        self.column_transformations[col] = {"type": "imputer", "method": "median", "imputer": imputer}
            
            elif method == 'mode':
                for col in target_columns:
                    imputer = SimpleImputer(strategy='most_frequent')
                    self.current_data[col] = imputer.fit_transform(self.current_data[[col]])
                    self.column_transformations[col] = {"type": "imputer", "method": "mode", "imputer": imputer}
            
            elif method == 'constant':
                for col in target_columns:
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                    self.current_data[col] = imputer.fit_transform(self.current_data[[col]])
                    self.column_transformations[col] = {"type": "imputer", "method": "constant", "imputer": imputer}
            
            elif method == 'knn':
                numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    imputer = KNNImputer(n_neighbors=5)
                    target_numeric_cols = [col for col in target_columns if col in numeric_cols]
                    
                    if target_numeric_cols:
                        self.current_data[target_numeric_cols] = imputer.fit_transform(self.current_data[target_numeric_cols])
                        for col in target_numeric_cols:
                            self.column_transformations[col] = {"type": "imputer", "method": "knn", "imputer": imputer}
            
            else:
                raise ValueError(f"Unknown missing value handling method: {method}")
            
            # Record the preprocessing step
            self.preprocessing_steps.append({
                "type": "missing_values",
                "method": method,
                "columns": target_columns,
                "fill_value": fill_value,
                "rows_before": len(data_before),
                "rows_after": len(self.current_data)
            })
            
            return self.current_data
        
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def normalize_data(self, 
                      method: str = 'standard', 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize numeric data
        
        Args:
            method: Method to normalize ('standard', 'minmax', 'robust')
            columns: Specific columns to normalize (None for all numeric)
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Get numeric columns if none specified
            if columns is None:
                columns = self.current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Skip if no columns to normalize
            if not columns:
                logger.warning("No numeric columns to normalize")
                return self.current_data
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Apply normalization
            self.current_data[columns] = scaler.fit_transform(self.current_data[columns])
            
            # Record transformation for each column
            for col in columns:
                self.column_transformations[col] = {"type": "scaler", "method": method, "scaler": scaler}
            
            # Record the preprocessing step
            self.preprocessing_steps.append({
                "type": "normalization",
                "method": method,
                "columns": columns
            })
            
            return self.current_data
        
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            raise
    
    def encode_categorical(self, 
                          method: str = 'onehot', 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            method: Method to encode ('onehot', 'label', 'ordinal')
            columns: Specific columns to encode (None for all categorical)
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Get categorical columns if none specified
            if columns is None:
                columns = self.current_data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Skip if no columns to encode
            if not columns:
                logger.warning("No categorical columns to encode")
                return self.current_data
            
            if method == 'onehot':
                # Use pandas get_dummies
                data_encoded = pd.get_dummies(self.current_data, columns=columns, drop_first=False)
                
                # Record original columns and created dummy columns
                dummy_columns = [c for c in data_encoded.columns if c not in self.current_data.columns]
                
                self.current_data = data_encoded
                
                for col in columns:
                    self.column_transformations[col] = {
                        "type": "encoder", 
                        "method": "onehot",
                        "original_column": col,
                        "encoded_columns": [c for c in dummy_columns if c.startswith(f"{col}_")]
                    }
            
            elif method == 'label':
                from sklearn.preprocessing import LabelEncoder
                
                for col in columns:
                    encoder = LabelEncoder()
                    self.current_data[col] = encoder.fit_transform(self.current_data[col])
                    self.column_transformations[col] = {
                        "type": "encoder", 
                        "method": "label",
                        "encoder": encoder,
                        "classes": encoder.classes_.tolist()
                    }
            
            elif method == 'ordinal':
                # This requires a mapping for each column
                logger.warning("Ordinal encoding requires a custom mapping for each column. Using label encoding instead.")
                from sklearn.preprocessing import LabelEncoder
                
                for col in columns:
                    encoder = LabelEncoder()
                    self.current_data[col] = encoder.fit_transform(self.current_data[col])
                    self.column_transformations[col] = {
                        "type": "encoder", 
                        "method": "label",
                        "encoder": encoder,
                        "classes": encoder.classes_.tolist()
                    }
            
            else:
                raise ValueError(f"Unknown encoding method: {method}")
            
            # Record the preprocessing step
            self.preprocessing_steps.append({
                "type": "categorical_encoding",
                "method": method,
                "columns": columns
            })
            
            return self.current_data
        
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {str(e)}")
            raise
    
    def filter_data(self, 
                  conditions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Filter data based on conditions
        
        Args:
            conditions: List of condition dictionaries with 'column', 'operator', and 'value'
            
        Returns:
            Filtered DataFrame
        """
        try:
            data_before = self.current_data.copy()
            mask = pd.Series(True, index=self.current_data.index)
            
            for condition in conditions:
                column = condition['column']
                operator = condition['operator']
                value = condition['value']
                
                if column not in self.current_data.columns:
                    raise ValueError(f"Column '{column}' not found in data")
                
                if operator == '==':
                    mask &= (self.current_data[column] == value)
                elif operator == '!=':
                    mask &= (self.current_data[column] != value)
                elif operator == '>':
                    mask &= (self.current_data[column] > value)
                elif operator == '>=':
                    mask &= (self.current_data[column] >= value)
                elif operator == '<':
                    mask &= (self.current_data[column] < value)
                elif operator == '<=':
                    mask &= (self.current_data[column] <= value)
                elif operator == 'in':
                    mask &= (self.current_data[column].isin(value))
                elif operator == 'not in':
                    mask &= (~self.current_data[column].isin(value))
                else:
                    raise ValueError(f"Unknown operator: {operator}")
            
            self.current_data = self.current_data[mask]
            
            # Record the preprocessing step
            self.preprocessing_steps.append({
                "type": "filter",
                "conditions": conditions,
                "rows_before": len(data_before),
                "rows_after": len(self.current_data)
            })
            
            return self.current_data
        
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            raise
    
    def select_columns(self, columns: List[str]) -> pd.DataFrame:
        """
        Select specific columns
        
        Args:
            columns: List of column names to keep
            
        Returns:
            DataFrame with selected columns
        """
        try:
            # Validate columns
            missing_cols = [col for col in columns if col not in self.current_data.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")
            
            self.current_data = self.current_data[columns]
            
            # Record the preprocessing step
            self.preprocessing_steps.append({
                "type": "select_columns",
                "columns": columns
            })
            
            return self.current_data
        
        except Exception as e:
            logger.error(f"Error selecting columns: {str(e)}")
            raise
    
    def remove_outliers(self, 
                       method: str = 'zscore', 
                       threshold: float = 3.0,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove outliers from the data
        
        Args:
            method: Method to detect outliers ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            columns: Specific columns to check (None for all numeric)
            
        Returns:
            DataFrame with outliers removed
        """
        try:
            # Get numeric columns if none specified
            if columns is None:
                columns = self.current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Skip if no columns to process
            if not columns:
                logger.warning("No numeric columns to remove outliers from")
                return self.current_data
            
            data_before = self.current_data.copy()
            
            if method == 'zscore':
                # Z-score method
                from scipy import stats
                z_scores = stats.zscore(self.current_data[columns], nan_policy='omit')
                abs_z_scores = np.abs(z_scores)
                mask = (abs_z_scores < threshold).all(axis=1)
                self.current_data = self.current_data[mask]
            
            elif method == 'iqr':
                # IQR method
                mask = pd.Series(True, index=self.current_data.index)
                
                for col in columns:
                    Q1 = self.current_data[col].quantile(0.25)
                    Q3 = self.current_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    mask &= (self.current_data[col] >= lower_bound) & (self.current_data[col] <= upper_bound)
                
                self.current_data = self.current_data[mask]
            
            else:
                raise ValueError(f"Unknown outlier removal method: {method}")
            
            # Record the preprocessing step
            self.preprocessing_steps.append({
                "type": "remove_outliers",
                "method": method,
                "threshold": threshold,
                "columns": columns,
                "rows_before": len(data_before),
                "rows_after": len(self.current_data),
                "outliers_removed": len(data_before) - len(self.current_data)
            })
            
            return self.current_data
        
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
    
    def create_lagged_variables(self, 
                              columns: List[str], 
                              lags: List[int]) -> pd.DataFrame:
        """
        Create lagged variables for time series analysis
        
        Args:
            columns: Columns to create lags for
            lags: List of lag values
            
        Returns:
            DataFrame with lagged variables
        """
        try:
            # Validate columns
            missing_cols = [col for col in columns if col not in self.current_data.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")
            
            # Create lagged variables
            for col in columns:
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}"
                    self.current_data[lag_col_name] = self.current_data[col].shift(lag)
            
            # Drop rows with NaN values from lag creation
            self.current_data = self.current_data.dropna()
            
            # Record the preprocessing step
            self.preprocessing_steps.append({
                "type": "create_lagged_variables",
                "columns": columns,
                "lags": lags
            })
            
            return self.current_data
        
        except Exception as e:
            logger.error(f"Error creating lagged variables: {str(e)}")
            raise
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps and their effects
        
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            "original_shape": self.original_data.shape,
            "current_shape": self.current_data.shape,
            "steps": self.preprocessing_steps,
            "columns_added": [col for col in self.current_data.columns if col not in self.original_data.columns],
            "columns_removed": [col for col in self.original_data.columns if col not in self.current_data.columns],
            "rows_removed": len(self.original_data) - len(self.current_data)
        }
        
        return summary