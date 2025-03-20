# app/components/timeseries_preprocessor.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    """
    Preprocessor for time series data specifically for causal analysis.
    Provides functions for lag creation, stationarity transformations,
    and feature engineering for time series.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the time series preprocessor
        
        Args:
            data: DataFrame with time series data
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.transformations = []
    
    def set_datetime_index(self, column: str, format: Optional[str] = None) -> pd.DataFrame:
        """
        Set a column as the datetime index
        
        Args:
            column: Column to use as index
            format: Optional datetime format string
            
        Returns:
            DataFrame with datetime index
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        try:
            if format:
                self.data[column] = pd.to_datetime(self.data[column], format=format)
            else:
                self.data[column] = pd.to_datetime(self.data[column])
            
            self.data.set_index(column, inplace=True)
            self.transformations.append({
                "type": "set_datetime_index", 
                "column": column,
                "format": format
            })
            
            return self.data
        except Exception as e:
            logger.error(f"Error setting datetime index: {str(e)}")
            raise
    
    def create_lags(self, columns: List[str], lags: Union[int, List[int]]) -> pd.DataFrame:
        """
        Create lagged versions of specified columns
        
        Args:
            columns: Columns to create lags for
            lags: Number of lags or list of lag values
            
        Returns:
            DataFrame with added lag columns
        """
        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Convert single lag value to list
        if isinstance(lags, int):
            lag_values = list(range(1, lags + 1))
        else:
            lag_values = lags
        
        # Create lag features
        for col in columns:
            for lag in lag_values:
                lag_col_name = f"{col}_lag{lag}"
                self.data[lag_col_name] = self.data[col].shift(lag)
        
        # Record transformation
        self.transformations.append({
            "type": "create_lags",
            "columns": columns,
            "lags": lag_values
        })
        
        return self.data
    
    def create_rolling_features(self, 
                               columns: List[str], 
                               window_sizes: List[int],
                               functions: List[str]) -> pd.DataFrame:
        """
        Create rolling window features (e.g., rolling mean, std)
        
        Args:
            columns: Columns to create features for
            window_sizes: List of window sizes
            functions: List of functions to apply ('mean', 'std', 'min', 'max', 'median')
        
        Returns:
            DataFrame with added rolling features
        """
        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Validate functions
        valid_functions = ['mean', 'std', 'min', 'max', 'median']
        for func in functions:
            if func not in valid_functions:
                raise ValueError(f"Invalid function '{func}'. Must be one of {valid_functions}")
        
        # Create rolling features
        for col in columns:
            for window in window_sizes:
                for func in functions:
                    feature_name = f"{col}_roll{window}_{func}"
                    
                    if func == 'mean':
                        self.data[feature_name] = self.data[col].rolling(window=window).mean()
                    elif func == 'std':
                        self.data[feature_name] = self.data[col].rolling(window=window).std()
                    elif func == 'min':
                        self.data[feature_name] = self.data[col].rolling(window=window).min()
                    elif func == 'max':
                        self.data[feature_name] = self.data[col].rolling(window=window).max()
                    elif func == 'median':
                        self.data[feature_name] = self.data[col].rolling(window=window).median()
        
        # Record transformation
        self.transformations.append({
            "type": "create_rolling_features",
            "columns": columns,
            "window_sizes": window_sizes,
            "functions": functions
        })
        
        return self.data
    
    def create_diff_features(self, columns: List[str], orders: List[int] = [1]) -> pd.DataFrame:
        """
        Create differenced features for stationarity
        
        Args:
            columns: Columns to create differences for
            orders: List of differencing orders
            
        Returns:
            DataFrame with added difference features
        """
        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Create differenced features
        for col in columns:
            for order in orders:
                diff_col_name = f"{col}_diff{order}"
                self.data[diff_col_name] = self.data[col].diff(order)
        
        # Record transformation
        self.transformations.append({
            "type": "create_diff_features",
            "columns": columns,
            "orders": orders
        })
        
        return self.data
    
    def create_seasonal_features(self, 
                               column: str, 
                               period: int,
                               model: str = 'additive') -> pd.DataFrame:
        """
        Create seasonal decomposition features
        
        Args:
            column: Column to decompose
            period: Seasonality period
            model: Decomposition model ('additive' or 'multiplicative')
            
        Returns:
            DataFrame with added seasonal components
        """
        # Validate column
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        # Validate model
        if model not in ['additive', 'multiplicative']:
            raise ValueError(f"Invalid model '{model}'. Must be 'additive' or 'multiplicative'")
        
        # Perform seasonal decomposition
        try:
            # Handle missing values
            decomposition = seasonal_decompose(
                self.data[column].fillna(method='ffill').fillna(method='bfill'),
                model=model,
                period=period
            )
            
            # Add components to dataframe
            self.data[f"{column}_trend"] = decomposition.trend
            self.data[f"{column}_seasonal"] = decomposition.seasonal
            self.data[f"{column}_residual"] = decomposition.resid
            
            # Record transformation
            self.transformations.append({
                "type": "create_seasonal_features",
                "column": column,
                "period": period,
                "model": model
            })
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {str(e)}")
            raise
    
    def detect_stationarity(self, column: str) -> Dict[str, Any]:
        """
        Test for stationarity using ADF and KPSS tests
        
        Args:
            column: Column to test
            
        Returns:
            Dictionary with test results
        """
        # Validate column
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        # Perform ADF test
        try:
            adf_result = adfuller(self.data[column].dropna())
            
            adf_output = {
                'Test Statistic': adf_result[0],
                'p-value': adf_result[1],
                'Critical Values': adf_result[4]
            }
            
            # Perform KPSS test
            kpss_result = kpss(self.data[column].dropna())
            
            kpss_output = {
                'Test Statistic': kpss_result[0],
                'p-value': kpss_result[1],
                'Critical Values': kpss_result[3]
            }
            
            # Determine stationarity
            # ADF: p-value < 0.05 suggests stationarity
            # KPSS: p-value > 0.05 suggests stationarity
            adf_stationary = adf_result[1] < 0.05
            kpss_stationary = kpss_result[1] > 0.05
            
            if adf_stationary and kpss_stationary:
                conclusion = "Stationary"
            elif not adf_stationary and not kpss_stationary:
                conclusion = "Non-stationary"
            elif adf_stationary:
                conclusion = "Trend stationary"
            else:
                conclusion = "Difference stationary"
            
            return {
                "column": column,
                "adf_test": adf_output,
                "kpss_test": kpss_output,
                "conclusion": conclusion,
                "is_stationary": adf_stationary or kpss_stationary
            }
            
        except Exception as e:
            logger.error(f"Error testing stationarity: {str(e)}")
            return {
                "column": column,
                "error": str(e),
                "is_stationary": None
            }
    
    def detect_stationarity_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Test stationarity for all numeric columns
        
        Returns:
            Dictionary mapping column names to stationarity test results
        """
        results = {}
        
        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            results[col] = self.detect_stationarity(col)
        
        return results
    
    def create_time_features(self) -> pd.DataFrame:
        """
        Create time-based features from datetime index
        
        Returns:
            DataFrame with added time features
        """
        # Check if index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index is not a DatetimeIndex. Use set_datetime_index first.")
        
        # Create time features
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['dayofweek'] = self.data.index.dayofweek
        self.data['quarter'] = self.data.index.quarter
        self.data['dayofyear'] = self.data.index.dayofyear
        self.data['week'] = self.data.index.isocalendar().week
        
        # Add hour, minute, second if available
        if self.data.index.hour.nunique() > 1:
            self.data['hour'] = self.data.index.hour
        if self.data.index.minute.nunique() > 1:
            self.data['minute'] = self.data.index.minute
        if self.data.index.second.nunique() > 1:
            self.data['second'] = self.data.index.second
        
        # Record transformation
        self.transformations.append({
            "type": "create_time_features"
        })
        
        return self.data
    
    def resample(self, rule: str, method: str = 'mean') -> pd.DataFrame:
        """
        Resample time series to a different frequency
        
        Args:
            rule: Resampling rule (e.g., 'D', 'W', 'M')
            method: Aggregation method
            
        Returns:
            Resampled DataFrame
        """
        # Check if index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index is not a DatetimeIndex. Use set_datetime_index first.")
        
        # Determine aggregation method
        if method == 'mean':
            self.data = self.data.resample(rule).mean()
        elif method == 'sum':
            self.data = self.data.resample(rule).sum()
        elif method == 'min':
            self.data = self.data.resample(rule).min()
        elif method == 'max':
            self.data = self.data.resample(rule).max()
        elif method == 'median':
            self.data = self.data.resample(rule).median()
        elif method == 'first':
            self.data = self.data.resample(rule).first()
        elif method == 'last':
            self.data = self.data.resample(rule).last()
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'mean', 'sum', 'min', 'max', 'median', 'first', or 'last'")
        
        # Record transformation
        self.transformations.append({
            "type": "resample",
            "rule": rule,
            "method": method
        })
        
        return self.data
    
    def handle_missing_values(self, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in time series data
        
        Args:
            method: Method for handling missing values
            
        Returns:
            DataFrame with handled missing values
        """
        if method == 'interpolate':
            self.data = self.data.interpolate(method='time')
        elif method == 'forward':
            self.data = self.data.fillna(method='ffill')
        elif method == 'backward':
            self.data = self.data.fillna(method='bfill')
        elif method == 'drop':
            self.data = self.data.dropna()
        elif method == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif method == 'median':
            self.data = self.data.fillna(self.data.median())
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'interpolate', 'forward', 'backward', 'drop', 'mean', or 'median'")
        
        # Record transformation
        self.transformations.append({
            "type": "handle_missing_values",
            "method": method
        })
        
        return self.data
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all transformations applied
        
        Returns:
            Dictionary with transformation information
        """
        return {
            "original_shape": self.original_data.shape,
            "current_shape": self.data.shape,
            "transformations": self.transformations,
            "missing_values_before": self.original_data.isna().sum().sum(),
            "missing_values_after": self.data.isna().sum().sum()
        }
    
    def reset(self) -> pd.DataFrame:
        """
        Reset to original data
        
        Returns:
            Original DataFrame
        """
        self.data = self.original_data.copy()
        self.transformations = []
        return self.data


def render_timeseries_preprocessor(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Render the time series preprocessor interface in Streamlit
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple of (processed_dataframe, transformation_summary)
    """
    st.subheader("Time Series Preprocessing")
    
    # Initialize preprocessor
    preprocessor = TimeSeriesPreprocessor(data)
    
    # Step 1: Set datetime index
    st.markdown("### 1. Set Datetime Index")
    
    datetime_cols = st.multiselect(
        "Select potential datetime columns",
        options=data.columns.tolist(),
        default=[]
    )
    
    if datetime_cols:
        datetime_col = st.selectbox("Select datetime column", datetime_cols)
        datetime_format = st.text_input("Datetime format (optional, e.g., %Y-%m-%d)")
        
        if st.button("Set Datetime Index"):
            try:
                format_arg = datetime_format if datetime_format else None
                preprocessor.set_datetime_index(datetime_col, format_arg)
                st.success(f"Set {datetime_col} as datetime index")
            except Exception as e:
                st.error(f"Error setting datetime index: {str(e)}")
    
    # Only show remaining steps if index is datetime
    if isinstance(preprocessor.data.index, pd.DatetimeIndex):
        # Step 2: Handle missing values
        st.markdown("### 2. Handle Missing Values")
        
        missing_counts = preprocessor.data.isna().sum()
        if missing_counts.sum() > 0:
            st.write("Missing values found:")
            st.write(missing_counts[missing_counts > 0])
            
            missing_method = st.selectbox(
                "Missing value handling method",
                options=['interpolate', 'forward', 'backward', 'drop', 'mean', 'median'],
                index=0
            )
            
            if st.button("Handle Missing Values"):
                preprocessor.handle_missing_values(missing_method)
                st.success("Missing values handled")
        else:
            st.write("No missing values found.")
        
        # Step 3: Check stationarity
        st.markdown("### 3. Check Stationarity")
        
        numeric_cols = preprocessor.data.select_dtypes(include=[np.number]).columns
        stationarity_col = st.selectbox("Select column to check stationarity", numeric_cols)
        
        if st.button("Check Stationarity"):
            result = preprocessor.detect_stationarity(stationarity_col)
            
            st.write(f"**ADF Test**: p-value = {result['adf_test']['p-value']:.4f}")
            st.write(f"**KPSS Test**: p-value = {result['kpss_test']['p-value']:.4f}")
            st.write(f"**Conclusion**: {result['conclusion']}")
            
            if not result.get('is_stationary', False):
                st.info("Series is non-stationary. Consider using differencing.")
        
        # Step 4: Create time series features
        st.markdown("### 4. Create Time Series Features")
        
        feature_options = st.multiselect(
            "Select feature types to create",
            options=['Lag Features', 'Differenced Features', 'Rolling Features', 'Seasonal Features', 'Time Features'],
            default=[]
        )
        
        if 'Lag Features' in feature_options:
            lag_cols = st.multiselect("Select columns for lag features", numeric_cols)
            max_lag = st.number_input("Maximum lag", value=3, min_value=1, max_value=50)
            
            if st.button("Create Lag Features"):
                preprocessor.create_lags(lag_cols, max_lag)
                st.success(f"Created lag features up to {max_lag} for {len(lag_cols)} columns")
        
        if 'Differenced Features' in feature_options:
            diff_cols = st.multiselect("Select columns for differencing", numeric_cols)
            diff_orders = st.multiselect("Differencing orders", options=[1, 2, 3, 4], default=[1])
            
            if st.button("Create Differenced Features"):
                preprocessor.create_diff_features(diff_cols, diff_orders)
                st.success(f"Created differenced features for {len(diff_cols)} columns")
        
        if 'Rolling Features' in feature_options:
            roll_cols = st.multiselect("Select columns for rolling features", numeric_cols)
            roll_windows = st.text_input("Window sizes (comma-separated)", "3, 7, 14").split(',')
            roll_windows = [int(w.strip()) for w in roll_windows if w.strip().isdigit()]
            
            roll_funcs = st.multiselect(
                "Select rolling functions",
                options=['mean', 'std', 'min', 'max', 'median'],
                default=['mean']
            )
            
            if st.button("Create Rolling Features"):
                preprocessor.create_rolling_features(roll_cols, roll_windows, roll_funcs)
                st.success(f"Created rolling features for {len(roll_cols)} columns")
        
        if 'Seasonal Features' in feature_options:
            seasonal_col = st.selectbox("Select column for seasonal decomposition", numeric_cols)
            seasonal_period = st.number_input("Seasonal period", value=12, min_value=2, max_value=365)
            seasonal_model = st.selectbox("Decomposition model", options=['additive', 'multiplicative'])
            
            if st.button("Create Seasonal Features"):
                try:
                    preprocessor.create_seasonal_features(seasonal_col, seasonal_period, seasonal_model)
                    st.success(f"Created seasonal features for {seasonal_col}")
                except Exception as e:
                    st.error(f"Error creating seasonal features: {str(e)}")
        
        if 'Time Features' in feature_options:
            if st.button("Create Time Features"):
                preprocessor.create_time_features()
                st.success("Created time features from datetime index")
        
        # Step 5: Resample data
        st.markdown("### 5. Resample Data")
        
        resample_rule = st.text_input("Resampling rule", "D")
        resample_method = st.selectbox(
            "Aggregation method",
            options=['mean', 'sum', 'min', 'max', 'median', 'first', 'last'],
            index=0
        )
        
        if st.button("Resample Data"):
            try:
                preprocessor.resample(resample_rule, resample_method)
                st.success(f"Resampled data to {resample_rule} frequency using {resample_method} aggregation")
            except Exception as e:
                st.error(f"Error resampling data: {str(e)}")
    
    # Preview processed data
    st.markdown("### Preview Processed Data")
    st.write(preprocessor.data.head())
    
    # Get transformation summary
    transformation_summary = preprocessor.get_transformation_summary()
    
    with st.expander("Transformation Summary"):
        st.write(f"Original shape: {transformation_summary['original_shape']}")
        st.write(f"Current shape: {transformation_summary['current_shape']}")
        st.write(f"Missing values before: {transformation_summary['missing_values_before']}")
        st.write(f"Missing values after: {transformation_summary['missing_values_after']}")
        st.write("Transformations:")
        for i, t in enumerate(transformation_summary['transformations']):
            st.write(f"{i+1}. {t['type']}")
    
    # Option to reset
    if st.button("Reset to Original Data"):
        preprocessor.reset()
        st.success("Reset to original data")
    
    return preprocessor.data, transformation_summary


def component_executor(node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the time series preprocessor component
    
    Args:
        node_data: Node configuration data
        inputs: Input data from connected nodes
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Get input data
        input_data = None
        for node_id, node_output in inputs.items():
            if "data" in node_output and isinstance(node_output["data"], pd.DataFrame):
                input_data = node_output["data"]
                break
        
        if input_data is None:
            return {
                "status": "error",
                "message": "No input dataframe provided",
                "data": {}
            }
        
        # Create preprocessor
        preprocessor = TimeSeriesPreprocessor(input_data)
        
        # Get operations from node data
        operations = node_data.get("operations", [])
        
        # Execute each operation in sequence
        for op in operations:
            op_type = op.get("type")
            
            if op_type == "set_datetime_index":
                column = op.get("column")
                format = op.get("format")
                
                if not column:
                    return {
                        "status": "error",
                        "message": "Datetime column not specified",
                        "data": {}
                    }
                
                try:
                    preprocessor.set_datetime_index(column, format)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error setting datetime index: {str(e)}",
                        "data": {}
                    }
            
            elif op_type == "handle_missing_values":
                method = op.get("method", "interpolate")
                
                try:
                    preprocessor.handle_missing_values(method)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error handling missing values: {str(e)}",
                        "data": {}
                    }
            
            elif op_type == "create_lags":
                columns = op.get("columns", [])
                lags = op.get("lags", 3)
                
                if not columns:
                    return {
                        "status": "error",
                        "message": "No columns specified for lag features",
                        "data": {}
                    }
                
                try:
                    preprocessor.create_lags(columns, lags)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error creating lag features: {str(e)}",
                        "data": {}
                    }
            
            elif op_type == "create_diff_features":
                columns = op.get("columns", [])
                orders = op.get("orders", [1])
                
                if not columns:
                    return {
                        "status": "error",
                        "message": "No columns specified for differencing",
                        "data": {}
                    }
                
                try:
                    preprocessor.create_diff_features(columns, orders)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error creating differenced features: {str(e)}",
                        "data": {}
                    }
            
            elif op_type == "create_rolling_features":
                columns = op.get("columns", [])
                window_sizes = op.get("window_sizes", [3, 7])
                functions = op.get("functions", ["mean"])
                
                if not columns:
                    return {
                        "status": "error",
                        "message": "No columns specified for rolling features",
                        "data": {}
                    }
                
                try:
                    preprocessor.create_rolling_features(columns, window_sizes, functions)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error creating rolling features: {str(e)}",
                        "data": {}
                    }
            
            elif op_type == "create_seasonal_features":
                column = op.get("column")
                period = op.get("period", 12)
                model = op.get("model", "additive")
                
                if not column:
                    return {
                        "status": "error",
                        "message": "No column specified for seasonal decomposition",
                        "data": {}
                    }
                
                try:
                    preprocessor.create_seasonal_features(column, period, model)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error creating seasonal features: {str(e)}",
                        "data": {}
                    }
            
            elif op_type == "create_time_features":
                try:
                    preprocessor.create_time_features()
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error creating time features: {str(e)}",
                        "data": {}
                    }
            
            elif op_type == "resample":
                rule = op.get("rule", "D")
                method = op.get("method", "mean")
                
                try:
                    preprocessor.resample(rule, method)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error resampling data: {str(e)}",
                        "data": {}
                    }
        
        # Get transformation summary
        transformation_summary = preprocessor.get_transformation_summary()
        
        # Return the results
        return {
            "status": "completed",
            "message": f"Time series preprocessing completed with {len(transformation_summary['transformations'])} transformations",
            "data": {
                "data": preprocessor.data,
                "transformation_summary": transformation_summary
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in time series preprocessor: {str(e)}",
            "data": {}
        }