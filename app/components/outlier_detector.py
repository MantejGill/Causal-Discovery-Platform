# app/components/outlier_detector.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats


def render_outlier_detection(df: pd.DataFrame, method: str = "z_score", threshold: float = 3.0, columns: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect outliers in the dataset using the specified method.
    
    Args:
        df: Pandas DataFrame with the data
        method: Outlier detection method ('z_score', 'iqr', 'isolation_forest', 'dbscan')
        threshold: Threshold for detection (interpretation depends on method)
        columns: List of columns to check for outliers (if None, checks all numeric columns)
    
    Returns:
        Tuple of (DataFrame with outlier flags, detection results dictionary)
    """
    # Select columns to analyze
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Ensure all specified columns exist and are numeric
        for col in columns:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in the dataset.")
                return df, {}
            if not pd.api.types.is_numeric_dtype(df[col]):
                st.warning(f"Column '{col}' is not numeric and will be skipped.")
                columns.remove(col)
    
    if not columns:
        st.warning("No numeric columns available for outlier detection.")
        return df, {}
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Dictionary to store outlier info
    outlier_info = {
        "method": method,
        "threshold": threshold,
        "columns_checked": columns,
        "total_outliers": 0,
        "outliers_by_column": {},
        "outlier_indices": set()
    }
    
    # Detect outliers based on chosen method
    if method == "z_score":
        for col in columns:
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            
            # Get indices of rows with outliers (need to handle NaN values)
            valid_indices = df[col].dropna().index
            outlier_indices = valid_indices[z_scores > threshold]
            
            # Update outlier info
            outlier_info["outliers_by_column"][col] = {
                "count": len(outlier_indices),
                "indices": outlier_indices.tolist(),
                "percentage": len(outlier_indices) / len(df) * 100
            }
            
            # Add outlier column to result DataFrame
            outlier_col_name = f"{col}_is_outlier"
            result_df[outlier_col_name] = False
            result_df.loc[outlier_indices, outlier_col_name] = True
            
            # Update total outliers and indices
            outlier_info["outlier_indices"].update(outlier_indices)
            outlier_info["total_outliers"] += len(outlier_indices)
    
    elif method == "iqr":
        for col in columns:
            # Calculate IQR
            Q1 = df[col].dropna().quantile(0.25)
            Q3 = df[col].dropna().quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Get indices of rows with outliers
            valid_indices = df[col].dropna().index
            outlier_mask = (df.loc[valid_indices, col] < lower_bound) | (df.loc[valid_indices, col] > upper_bound)
            outlier_indices = valid_indices[outlier_mask]
            
            # Update outlier info
            outlier_info["outliers_by_column"][col] = {
                "count": len(outlier_indices),
                "indices": outlier_indices.tolist(),
                "percentage": len(outlier_indices) / len(df) * 100,
                "bounds": (lower_bound, upper_bound)
            }
            
            # Add outlier column to result DataFrame
            outlier_col_name = f"{col}_is_outlier"
            result_df[outlier_col_name] = False
            result_df.loc[outlier_indices, outlier_col_name] = True
            
            # Update total outliers and indices
            outlier_info["outlier_indices"].update(outlier_indices)
            outlier_info["total_outliers"] += len(outlier_indices)
    
    elif method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
            
            # Select only numeric columns for the model
            X = df[columns].dropna()
            
            # Get indices of valid rows
            valid_indices = X.index
            
            # Fit model
            model = IsolationForest(contamination=threshold/100, random_state=42)
            outlier_labels = model.fit_predict(X)
            
            # Outliers are labeled as -1
            outlier_indices = valid_indices[outlier_labels == -1]
            
            # Add general outlier column
            result_df["is_outlier"] = False
            result_df.loc[outlier_indices, "is_outlier"] = True
            
            # Update outlier info
            outlier_info["outliers_by_column"]["all_columns"] = {
                "count": len(outlier_indices),
                "indices": outlier_indices.tolist(),
                "percentage": len(outlier_indices) / len(df) * 100
            }
            
            # Update total outliers and indices
            outlier_info["outlier_indices"].update(outlier_indices)
            outlier_info["total_outliers"] = len(outlier_indices)
            
        except ImportError:
            st.error("scikit-learn is required for Isolation Forest method.")
            return df, {}
    
    elif method == "dbscan":
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import DBSCAN
            
            # Select only numeric columns for the model
            X = df[columns].dropna()
            
            # Get indices of valid rows
            valid_indices = X.index
            
            # Standardize the data
            X_scaled = StandardScaler().fit_transform(X)
            
            # Fit model
            model = DBSCAN(eps=threshold, min_samples=5)
            cluster_labels = model.fit_predict(X_scaled)
            
            # Outliers are labeled as -1
            outlier_indices = valid_indices[cluster_labels == -1]
            
            # Add general outlier column
            result_df["is_outlier"] = False
            result_df.loc[outlier_indices, "is_outlier"] = True
            
            # Update outlier info
            outlier_info["outliers_by_column"]["all_columns"] = {
                "count": len(outlier_indices),
                "indices": outlier_indices.tolist(),
                "percentage": len(outlier_indices) / len(df) * 100
            }
            
            # Update total outliers and indices
            outlier_info["outlier_indices"].update(outlier_indices)
            outlier_info["total_outliers"] = len(outlier_indices)
            
        except ImportError:
            st.error("scikit-learn is required for DBSCAN method.")
            return df, {}
    
    else:
        st.error(f"Unknown outlier detection method: {method}")
        return df, {}
    
    # Convert outlier indices from set to list
    outlier_info["outlier_indices"] = list(outlier_info["outlier_indices"])
    
    return result_df, outlier_info


def render_outlier_visualization(df: pd.DataFrame, outlier_info: Dict[str, Any], column: str = None):
    """
    Visualize outliers in the dataset.
    
    Args:
        df: Pandas DataFrame with the data
        outlier_info: Outlier detection results from render_outlier_detection
        column: Specific column to visualize (if None, visualizes all columns with outliers)
    """
    if "total_outliers" not in outlier_info or outlier_info["total_outliers"] == 0:
        st.info("No outliers detected in the dataset.")
        return
    
    # Show summary of outliers
    st.subheader("Outlier Detection Summary")
    
    cols = st.columns(3)
    cols[0].metric("Total Outliers", outlier_info["total_outliers"])
    cols[1].metric("Affected Rows", len(outlier_info["outlier_indices"]))
    cols[2].metric("Percentage", f"{len(outlier_info['outlier_indices']) / len(df) * 100:.2f}%")
    
    # Visualize outliers by column
    if column is not None and column in outlier_info["outliers_by_column"]:
        columns_to_visualize = [column]
    else:
        columns_to_visualize = list(outlier_info["outliers_by_column"].keys())
    
    for col in columns_to_visualize:
        if col == "all_columns":
            # For methods that detect outliers across all columns
            st.subheader(f"Multivariate Outliers (using {outlier_info['method']})")
            
            # Try to visualize in 2D space using the first two columns
            if len(outlier_info["columns_checked"]) >= 2:
                x_col = outlier_info["columns_checked"][0]
                y_col = outlier_info["columns_checked"][1]
                
                # Create scatter plot
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    color=df["is_outlier"] if "is_outlier" in df.columns else None,
                    color_discrete_map={True: "red", False: "blue"},
                    title=f"Outliers in {x_col} vs {y_col} space",
                    labels={"color": "Is Outlier"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show 3D plot if we have at least 3 columns
                if len(outlier_info["columns_checked"]) >= 3:
                    z_col = outlier_info["columns_checked"][2]
                    
                    fig_3d = px.scatter_3d(
                        df, 
                        x=x_col, 
                        y=y_col, 
                        z=z_col,
                        color=df["is_outlier"] if "is_outlier" in df.columns else None,
                        color_discrete_map={True: "red", False: "blue"},
                        title=f"Outliers in 3D space ({x_col}, {y_col}, {z_col})",
                        labels={"color": "Is Outlier"}
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
            
        else:
            # For individual column outliers
            st.subheader(f"Outliers in '{col}'")
            
            col_info = outlier_info["outliers_by_column"][col]
            
            st.write(f"Found {col_info['count']} outliers ({col_info['percentage']:.2f}% of data)")
            
            # Create box plot
            fig = go.Figure()
            
            # Add all points
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(
                    color='blue',
                    size=4
                ),
                line=dict(color='blue')
            ))
            
            # Highlight outliers
            if col_info['count'] > 0:
                fig.add_trace(go.Scatter(
                    y=df.loc[col_info['indices'], col],
                    mode='markers',
                    name='Outliers',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='circle',
                        line=dict(color='black', width=1)
                    )
                ))
                
                # Add bounds for IQR method
                if outlier_info["method"] == "iqr" and "bounds" in col_info:
                    lower_bound, upper_bound = col_info["bounds"]
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=0.5,
                        y0=lower_bound,
                        y1=lower_bound,
                        line=dict(color="red", dash="dash"),
                        name="Lower Bound"
                    )
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=0.5,
                        y0=upper_bound,
                        y1=upper_bound,
                        line=dict(color="red", dash="dash"),
                        name="Upper Bound"
                    )
            
            fig.update_layout(
                title=f"Distribution of {col} with Outliers Highlighted",
                yaxis_title=col,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show histogram with outliers
            hist_fig = px.histogram(
                df, 
                x=col,
                color=df[f"{col}_is_outlier"] if f"{col}_is_outlier" in df.columns else None,
                color_discrete_map={True: "red", False: "blue"},
                title=f"Histogram of {col} with Outliers Highlighted",
                labels={"color": "Is Outlier"}
            )
            
            st.plotly_chart(hist_fig, use_container_width=True)


def render_outlier_handling(df: pd.DataFrame, outlier_info: Dict[str, Any], method: str = "remove") -> pd.DataFrame:
    """
    Handle outliers in the dataset using the specified method.
    
    Args:
        df: Pandas DataFrame with the data
        outlier_info: Outlier detection results from render_outlier_detection
        method: Outlier handling method ('remove', 'clip', 'mean', 'median', 'mode')
    
    Returns:
        DataFrame with handled outliers
    """
    if "total_outliers" not in outlier_info or outlier_info["total_outliers"] == 0:
        st.info("No outliers to handle in the dataset.")
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Get outlier indices
    outlier_indices = outlier_info["outlier_indices"]
    
    # Handle outliers based on method
    if method == "remove":
        # Remove rows with outliers
        result_df = result_df.drop(outlier_indices).reset_index(drop=True)
        st.success(f"Removed {len(outlier_indices)} rows with outliers.")
        
    elif method == "clip":
        # Clip outliers to bounds
        for col, col_info in outlier_info["outliers_by_column"].items():
            if col == "all_columns":
                # Skip for multivariate methods
                continue
                
            if outlier_info["method"] == "iqr" and "bounds" in col_info:
                lower_bound, upper_bound = col_info["bounds"]
                
                # Get only the outlier indices for this column
                col_outlier_indices = col_info["indices"]
                
                # Clip values
                result_df.loc[col_outlier_indices, col] = result_df.loc[col_outlier_indices, col].clip(lower_bound, upper_bound)
                
                st.success(f"Clipped {len(col_outlier_indices)} outliers in column '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}].")
            elif outlier_info["method"] == "z_score":
                # Calculate bounds using mean and std
                mean = df[col].mean()
                std = df[col].std()
                
                lower_bound = mean - outlier_info["threshold"] * std
                upper_bound = mean + outlier_info["threshold"] * std
                
                # Get only the outlier indices for this column
                col_outlier_indices = col_info["indices"]
                
                # Clip values
                result_df.loc[col_outlier_indices, col] = result_df.loc[col_outlier_indices, col].clip(lower_bound, upper_bound)
                
                st.success(f"Clipped {len(col_outlier_indices)} outliers in column '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}].")
    
    elif method == "mean":
        # Replace outliers with column mean
        for col, col_info in outlier_info["outliers_by_column"].items():
            if col == "all_columns":
                # For multivariate methods, handle each column separately
                for feature_col in outlier_info["columns_checked"]:
                    mean_val = df[feature_col].mean()
                    result_df.loc[outlier_indices, feature_col] = mean_val
                    st.success(f"Replaced outliers in column '{feature_col}' with mean ({mean_val:.2f}).")
            else:
                # Get only the outlier indices for this column
                col_outlier_indices = col_info["indices"]
                
                # Replace with mean
                mean_val = df[col].mean()
                result_df.loc[col_outlier_indices, col] = mean_val
                
                st.success(f"Replaced {len(col_outlier_indices)} outliers in column '{col}' with mean ({mean_val:.2f}).")
    
    elif method == "median":
        # Replace outliers with column median
        for col, col_info in outlier_info["outliers_by_column"].items():
            if col == "all_columns":
                # For multivariate methods, handle each column separately
                for feature_col in outlier_info["columns_checked"]:
                    median_val = df[feature_col].median()
                    result_df.loc[outlier_indices, feature_col] = median_val
                    st.success(f"Replaced outliers in column '{feature_col}' with median ({median_val:.2f}).")
            else:
                # Get only the outlier indices for this column
                col_outlier_indices = col_info["indices"]
                
                # Replace with median
                median_val = df[col].median()
                result_df.loc[col_outlier_indices, col] = median_val
                
                st.success(f"Replaced {len(col_outlier_indices)} outliers in column '{col}' with median ({median_val:.2f}).")
    
    elif method == "mode":
        # Replace outliers with column mode
        for col, col_info in outlier_info["outliers_by_column"].items():
            if col == "all_columns":
                # For multivariate methods, handle each column separately
                for feature_col in outlier_info["columns_checked"]:
                    mode_val = df[feature_col].mode().iloc[0]
                    result_df.loc[outlier_indices, feature_col] = mode_val
                    st.success(f"Replaced outliers in column '{feature_col}' with mode ({mode_val}).")
            else:
                # Get only the outlier indices for this column
                col_outlier_indices = col_info["indices"]
                
                # Replace with mode
                mode_val = df[col].mode().iloc[0]
                result_df.loc[col_outlier_indices, col] = mode_val
                
                st.success(f"Replaced {len(col_outlier_indices)} outliers in column '{col}' with mode ({mode_val}).")
    
    else:
        st.error(f"Unknown outlier handling method: {method}")
        return df
    
    # Remove outlier flag columns if they exist
    for col in result_df.columns:
        if col.endswith("_is_outlier") or col == "is_outlier":
            result_df = result_df.drop(columns=[col])
    
    return result_df


def render_outlier_interface(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render a complete interface for outlier detection and handling.
    
    Args:
        df: Pandas DataFrame with the data
        
    Returns:
        DataFrame with handled outliers (if user chooses to handle them)
    """
    st.header("Outlier Detection and Handling")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns available for outlier detection.")
        return df
    
    # Outlier detection settings
    st.subheader("Detection Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Detection Method",
            ["z_score", "iqr", "isolation_forest", "dbscan"],
            help="""
            Z-Score: Flag values outside z standard deviations from the mean
            IQR: Flag values outside the bounds of Q1-k*IQR and Q3+k*IQR
            Isolation Forest: Use machine learning to detect outliers in high dimensions
            DBSCAN: Density-based outlier detection
            """
        )
    
    with col2:
        if method == "z_score":
            threshold = st.slider("Z-Score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                                help="Higher values are more conservative (fewer outliers)")
        elif method == "iqr":
            threshold = st.slider("IQR Factor", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                               help="Higher values are more conservative (fewer outliers)")
        elif method == "isolation_forest":
            threshold = st.slider("Contamination Percentage", min_value=1.0, max_value=20.0, value=5.0, step=0.5,
                               help="Estimated percentage of outliers in the dataset")
        elif method == "dbscan":
            threshold = st.slider("Epsilon", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                               help="Maximum distance between samples in the same neighborhood")
    
    # Column selection
    columns = st.multiselect(
        "Select Columns for Analysis",
        options=numeric_cols,
        default=numeric_cols,
        help="Select the columns to analyze for outliers (defaults to all numeric columns)"
    )
    
    # Run detection
    if st.button("Detect Outliers"):
        with st.spinner("Detecting outliers..."):
            result_df, outlier_info = render_outlier_detection(df, method=method, threshold=threshold, columns=columns)
            
            # Store results in session state
            st.session_state.outlier_df = result_df
            st.session_state.outlier_info = outlier_info
            
            # Visualize the outliers
            render_outlier_visualization(df, outlier_info)
            
            # Outlier handling
            st.subheader("Outlier Handling")
            
            if outlier_info["total_outliers"] > 0:
                handling_method = st.selectbox(
                    "Handling Method",
                    ["remove", "clip", "mean", "median", "mode"],
                    help="""
                    Remove: Delete rows with outliers
                    Clip: Cap outliers at the threshold values
                    Mean: Replace outliers with the column mean
                    Median: Replace outliers with the column median
                    Mode: Replace outliers with the column mode
                    """
                )
                
                if st.button("Handle Outliers"):
                    with st.spinner("Handling outliers..."):
                        handled_df = render_outlier_handling(df, outlier_info, method=handling_method)
                        
                        # Compare before and after
                        st.subheader("Before vs After Handling")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Original Rows", len(df))
                            
                        with col2:
                            st.metric("After Handling", len(handled_df), delta=len(handled_df) - len(df))
                        
                        # Display sample of the handled data
                        st.dataframe(handled_df.head(10))
                        
                        return handled_df
    
    # Check if we already have results in session state
    if "outlier_df" in st.session_state and "outlier_info" in st.session_state:
        # Reuse visualization
        render_outlier_visualization(df, st.session_state.outlier_info)
        
        # Outlier handling
        st.subheader("Outlier Handling")
        
        if st.session_state.outlier_info["total_outliers"] > 0:
            handling_method = st.selectbox(
                "Handling Method",
                ["remove", "clip", "mean", "median", "mode"],
                help="""
                Remove: Delete rows with outliers
                Clip: Cap outliers at the threshold values
                Mean: Replace outliers with the column mean
                Median: Replace outliers with the column median
                Mode: Replace outliers with the column mode
                """
            )
            
            if st.button("Handle Outliers"):
                with st.spinner("Handling outliers..."):
                    handled_df = render_outlier_handling(df, st.session_state.outlier_info, method=handling_method)
                    
                    # Compare before and after
                    st.subheader("Before vs After Handling")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Original Rows", len(df))
                        
                    with col2:
                        st.metric("After Handling", len(handled_df), delta=len(handled_df) - len(df))
                    
                    # Display sample of the handled data
                    st.dataframe(handled_df.head(10))
                    
                    return handled_df
    
    return df