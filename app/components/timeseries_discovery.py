# app/components/timeseries_discovery.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
from datetime import datetime

# Import custom modules
from core.algorithms.timeseries import TimeSeriesCausalDiscovery, VARLiNGAM
from core.viz.graph import CausalGraphVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def render_timeseries_preprocessing(df: pd.DataFrame, 
                                  date_col: Optional[str] = None,
                                  freq: Optional[str] = None,
                                  max_lag: int = 5) -> Dict[str, Any]:
    """
    Preprocess time series data for causal discovery
    
    Args:
        df: DataFrame containing the time series data
        date_col: Column containing datetime information (None if index is datetime)
        freq: Frequency to resample data to (None to keep as is)
        max_lag: Maximum number of lags to create
        
    Returns:
        Dictionary with preprocessed data and metadata
    """
    st.subheader("Time Series Preprocessing")
    
    # Check if we have a date column or datetime index
    df_ts = df.copy()
    has_datetime = False
    
    if date_col is not None and date_col in df.columns:
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df_ts[date_col] = pd.to_datetime(df[date_col])
                has_datetime = True
                st.success(f"Converted '{date_col}' to datetime")
            except Exception as e:
                st.error(f"Could not convert '{date_col}' to datetime: {str(e)}")
        else:
            has_datetime = True
            
        # Set date as index if not already
        if date_col != df_ts.index.name:
            df_ts = df_ts.set_index(date_col)
            st.info(f"Set '{date_col}' as index")
    
    elif pd.api.types.is_datetime64_any_dtype(df.index):
        has_datetime = True
        st.info("Using existing datetime index")
    
    if not has_datetime:
        st.warning("No datetime information found. Using row numbers as time index.")
        # Create a simple time index
        df_ts = df_ts.reset_index(drop=True)
        df_ts.index = pd.RangeIndex(start=0, stop=len(df_ts), step=1)
        df_ts.index.name = 'time_idx'
    
    # Resample if frequency is provided
    if freq and has_datetime:
        try:
            # Get numeric columns for resampling
            numeric_cols = df_ts.select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols:
                st.error("No numeric columns found for resampling")
            else:
                # Resample
                df_ts = df_ts[numeric_cols].resample(freq).mean()
                st.success(f"Resampled data to '{freq}' frequency")
                
                # Handle NaN values from resampling
                if df_ts.isna().any().any():
                    st.warning("Resampling introduced NaN values. Filling with forward fill followed by backward fill.")
                    df_ts = df_ts.fillna(method='ffill').fillna(method='bfill')
        except Exception as e:
            st.error(f"Error during resampling: {str(e)}")
    
    # Create lagged variables
    if max_lag > 0:
        # Get numeric columns for lag creation
        numeric_cols = df_ts.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found for creating lags")
        else:
            # Create lag columns
            orig_cols = numeric_cols.copy()
            for col in orig_cols:
                for lag in range(1, max_lag + 1):
                    lag_col = f"{col}_lag{lag}"
                    df_ts[lag_col] = df_ts[col].shift(lag)
            
            # Remove rows with NaN values due to lag
            df_ts = df_ts.dropna()
            st.success(f"Created {max_lag} lags for each variable and dropped {max_lag} rows with NaN values")
    
    # Display basic time series info
    st.subheader("Time Series Information")
    
    # Show the resulting data shape
    st.write(f"Processed data shape: {df_ts.shape}")
    
    # Display time range if we have datetime
    if has_datetime:
        st.write(f"Time range: {df_ts.index.min()} to {df_ts.index.max()}")
        time_span = (df_ts.index.max() - df_ts.index.min()).total_seconds()
        if time_span < 60:
            st.write(f"Time span: {time_span:.1f} seconds")
        elif time_span < 3600:
            st.write(f"Time span: {time_span/60:.1f} minutes")
        elif time_span < 86400:
            st.write(f"Time span: {time_span/3600:.1f} hours")
        elif time_span < 31536000:
            st.write(f"Time span: {time_span/86400:.1f} days")
        else:
            st.write(f"Time span: {time_span/31536000:.1f} years")
    
    # Return preprocessed data and metadata
    return {
        "data": df_ts,
        "has_datetime": has_datetime,
        "orig_shape": df.shape,
        "processed_shape": df_ts.shape,
        "max_lag": max_lag,
        "numeric_columns": df_ts.select_dtypes(include=np.number).columns.tolist(),
        "resampled": freq is not None
    }


def render_timeseries_visualization(df: pd.DataFrame, 
                                  variables: List[str] = None, 
                                  date_col: Optional[str] = None):
    """
    Visualize time series data
    
    Args:
        df: DataFrame containing the time series data
        variables: List of variables to visualize (None for all numeric)
        date_col: Column containing datetime information (None if index is datetime)
    """
    st.subheader("Time Series Visualization")
    
    # Check if we have a date column or datetime index
    df_ts = df.copy()
    has_datetime = False
    
    if date_col is not None and date_col in df.columns:
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df_ts[date_col] = pd.to_datetime(df[date_col])
                has_datetime = True
            except:
                pass
            
        # Set date as index if not already
        if date_col != df_ts.index.name:
            df_ts = df_ts.set_index(date_col)
    
    elif pd.api.types.is_datetime64_any_dtype(df.index):
        has_datetime = True
    
    # Get numeric columns for visualization
    if variables is None:
        variables = df_ts.select_dtypes(include=np.number).columns.tolist()
    else:
        # Filter to ensure all variables exist and are numeric
        variables = [var for var in variables if var in df_ts.columns 
                    and pd.api.types.is_numeric_dtype(df_ts[var])]
    
    if not variables:
        st.error("No numeric variables found for visualization")
        return
    
    # Create the time series plot
    st.write("### Time Series Plot")
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add a line for each variable
    for var in variables:
        fig.add_trace(go.Scatter(
            x=df_ts.index if has_datetime else np.arange(len(df_ts)),
            y=df_ts[var],
            mode='lines',
            name=var
        ))
    
    # Update layout
    fig.update_layout(
        title='Time Series Data',
        xaxis_title='Time' if has_datetime else 'Index',
        yaxis_title='Value',
        legend_title='Variables',
        hovermode='x unified'
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Create correlation plot
    if len(variables) > 1:
        st.write("### Correlation Matrix")
        
        # Calculate correlation matrix
        corr_matrix = df_ts[variables].corr()
        
        # Create heatmap
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        
        fig_corr.update_layout(
            title='Correlation Matrix',
            height=400
        )
        
        # Display correlation heatmap
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Create decomposition plot for a selected variable
    if len(variables) > 0:
        st.write("### Time Series Decomposition")
        
        selected_var = st.selectbox("Select Variable for Decomposition", variables)
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Perform decomposition
            decomposition = seasonal_decompose(df_ts[selected_var], model='additive', extrapolate_trend='freq')
            
            # Create subplots for decomposition components
            fig_decomp = make_subplots(rows=4, cols=1, 
                                       subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
            
            # Add original data
            fig_decomp.add_trace(
                go.Scatter(x=df_ts.index if has_datetime else np.arange(len(df_ts)), 
                           y=df_ts[selected_var], mode='lines', name='Original'),
                row=1, col=1
            )
            
            # Add trend component
            fig_decomp.add_trace(
                go.Scatter(x=df_ts.index if has_datetime else np.arange(len(df_ts)), 
                           y=decomposition.trend, mode='lines', name='Trend'),
                row=2, col=1
            )
            
            # Add seasonal component
            fig_decomp.add_trace(
                go.Scatter(x=df_ts.index if has_datetime else np.arange(len(df_ts)), 
                           y=decomposition.seasonal, mode='lines', name='Seasonal'),
                row=3, col=1
            )
            
            # Add residual component
            fig_decomp.add_trace(
                go.Scatter(x=df_ts.index if has_datetime else np.arange(len(df_ts)), 
                           y=decomposition.resid, mode='lines', name='Residual'),
                row=4, col=1
            )
            
            # Update layout
            fig_decomp.update_layout(
                height=800,
                title_text=f"Time Series Decomposition for {selected_var}",
                showlegend=False
            )
            
            # Display decomposition plot
            st.plotly_chart(fig_decomp, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in time series decomposition: {str(e)}")


def render_granger_causality(df: pd.DataFrame, 
                           max_lag: int = 5,
                           alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform Granger causality analysis on time series data
    
    Args:
        df: DataFrame containing the time series data
        max_lag: Maximum number of lags to test
        alpha: Significance level for tests
        
    Returns:
        Dictionary with Granger causality results
    """
    st.subheader("Granger Causality Analysis")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter out lag columns if they exist
    orig_cols = [col for col in numeric_cols if not col.endswith(tuple([f"_lag{i}" for i in range(1, max_lag+1)]))]
    
    if len(orig_cols) < 2:
        st.error("Need at least 2 numeric columns for Granger causality analysis")
        return {"status": "error", "message": "Not enough numeric columns"}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Import statsmodels for Granger causality tests
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Create a directed graph for causal relationships
        G = nx.DiGraph()
        for col in orig_cols:
            G.add_node(col)
        
        # Perform Granger causality tests for each pair of variables
        n_pairs = len(orig_cols) * (len(orig_cols) - 1)
        pair_count = 0
        test_results = {}
        
        progress_text.text("Running Granger causality tests...")
        
        for i, cause_var in enumerate(orig_cols):
            for j, effect_var in enumerate(orig_cols):
                if i != j:  # Skip self-causality
                    pair_count += 1
                    progress_bar.progress(int(100 * pair_count / n_pairs))
                    progress_text.text(f"Testing {cause_var} → {effect_var} ({pair_count}/{n_pairs})")
                    
                    # Prepare data
                    data = df[[effect_var, cause_var]].dropna()
                    
                    # Run Granger causality test
                    try:
                        gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                        
                        # Extract p-values (using ssr F-test)
                        p_values = [gc_res[lag+1][0]['ssr_ftest'][1] for lag in range(max_lag)]
                        min_p_value = min(p_values)
                        min_p_lag = p_values.index(min_p_value) + 1
                        
                        # Store results
                        test_results[(cause_var, effect_var)] = {
                            "p_values": p_values,
                            "min_p_value": min_p_value,
                            "best_lag": min_p_lag,
                            "significant": min_p_value < alpha
                        }
                        
                        # Add edge to graph if significant
                        if min_p_value < alpha:
                            G.add_edge(
                                cause_var, 
                                effect_var, 
                                weight=1.0 - min_p_value,
                                p_value=min_p_value,
                                lag=min_p_lag,
                                causality="granger"
                            )
                    
                    except Exception as e:
                        logger.error(f"Error in Granger test {cause_var} → {effect_var}: {str(e)}")
                        test_results[(cause_var, effect_var)] = {
                            "error": str(e),
                            "significant": False
                        }
        
        progress_bar.progress(100)
        progress_text.text("Granger causality tests completed!")
        
        # Count significant relationships
        significant_count = sum(1 for result in test_results.values() if result.get("significant", False))
        
        # Display summary
        st.write(f"### Summary of Granger Causality Tests")
        st.write(f"Tested {n_pairs} potential causal relationships")
        st.write(f"Found {significant_count} significant Granger-causal relationships (p < {alpha})")
        
        # Create a graph visualizer
        graph_viz = CausalGraphVisualizer()
        
        # Visualize the causal graph
        fig = graph_viz.visualize_graph(
            graph=G,
            show_confidence=True,
            title="Granger Causality Graph"
        )
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Create detailed results table
        results_data = []
        for (cause, effect), result in test_results.items():
            if result.get("significant", False):
                results_data.append({
                    "Cause": cause,
                    "Effect": effect,
                    "Best Lag": result["best_lag"],
                    "P-Value": f"{result['min_p_value']:.4f}",
                    "Significant": "Yes"
                })
        
        if results_data:
            st.write("### Significant Granger-Causal Relationships")
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df)
        
        # Return results
        return {
            "status": "completed",
            "graph": G,
            "test_results": test_results,
            "significant_count": significant_count,
            "total_tests": n_pairs,
            "max_lag": max_lag,
            "alpha": alpha,
            "method": "Granger causality"
        }
    
    except Exception as e:
        progress_text.text(f"Error in Granger causality analysis: {str(e)}")
        logger.error(f"Error in Granger causality analysis: {str(e)}")
        return {"status": "error", "message": str(e)}


def render_var_lingam_analysis(df: pd.DataFrame, 
                             max_lag: int = 2) -> Dict[str, Any]:
    """
    Perform VAR-LiNGAM analysis on time series data
    
    Args:
        df: DataFrame containing the time series data
        max_lag: Maximum number of lags to include in the model
        
    Returns:
        Dictionary with VAR-LiNGAM results
    """
    st.subheader("VAR-LiNGAM Analysis")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter out lag columns if they exist
    orig_cols = [col for col in numeric_cols if not col.endswith(tuple([f"_lag{i}" for i in range(1, max_lag+1)]))]
    
    if len(orig_cols) < 2:
        st.error("Need at least 2 numeric columns for VAR-LiNGAM analysis")
        return {"status": "error", "message": "Not enough numeric columns"}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        progress_text.text("Initializing VAR-LiNGAM model...")
        progress_bar.progress(20)
        
        # Extract data for analysis
        data = df[orig_cols].dropna().values
        
        # Initialize and fit VAR-LiNGAM model
        progress_text.text("Fitting VAR-LiNGAM model...")
        model = VARLiNGAM(lags=max_lag)
        
        progress_bar.progress(40)
        result = model.fit(data)
        
        if not result.get("success", False):
            progress_text.text(f"Error in VAR-LiNGAM fitting: {result.get('error', 'Unknown error')}")
            return {"status": "error", "message": result.get('error', 'Model fitting failed')}
        
        progress_text.text("Creating causal graph...")
        progress_bar.progress(70)
        
        # Generate the causal graph
        G = model.to_networkx_graph(var_names=orig_cols)
        
        progress_text.text("Analyzing causality matrix...")
        progress_bar.progress(90)
        
        # Create a visualization of the graph
        graph_viz = CausalGraphVisualizer()
        fig = graph_viz.visualize_graph(
            graph=G,
            show_confidence=True,
            title="VAR-LiNGAM Causal Graph"
        )
        
        progress_bar.progress(100)
        progress_text.text("VAR-LiNGAM analysis completed!")
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyze instantaneous effects (contemporaneous relationships)
        st.write("### Contemporaneous Causal Effects")
        
        # Get coefficients for instantaneous effects
        if "lingam_coefs" in result:
            # Convert to DataFrame for better display
            coef_df = pd.DataFrame(
                result["lingam_coefs"],
                index=orig_cols,
                columns=orig_cols
            )
            
            # Display the coefficients
            st.write("Contemporaneous coefficient matrix (rows → columns):")
            st.dataframe(coef_df)
            
            # Create heatmap of coefficients
            fig_heatmap = px.imshow(
                coef_df,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Contemporaneous Causal Effects"
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Analyze lagged effects
        st.write("### Time-Lagged Causal Effects")
        
        if "ar_coefs" in result:
            ar_coefs = result["ar_coefs"]
            
            # Create a visualization for each lag
            for lag in range(max_lag):
                # Convert to DataFrame for better display
                lag_df = pd.DataFrame(
                    ar_coefs[lag],
                    index=orig_cols,
                    columns=orig_cols
                )
                
                # Display the coefficients
                st.write(f"Lag {lag+1} coefficient matrix (rows → columns):")
                st.dataframe(lag_df)
                
                # Create heatmap of coefficients
                fig_lag = px.imshow(
                    lag_df,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title=f"Lag {lag+1} Causal Effects"
                )
                
                st.plotly_chart(fig_lag, use_container_width=True)
        
        # Return results
        return {
            "status": "completed",
            "graph": G,
            "model_result": result,
            "max_lag": max_lag,
            "method": "VAR-LiNGAM",
            "variables": orig_cols
        }
    
    except Exception as e:
        progress_text.text(f"Error in VAR-LiNGAM analysis: {str(e)}")
        logger.error(f"Error in VAR-LiNGAM analysis: {str(e)}")
        return {"status": "error", "message": str(e)}


def render_transfer_entropy_analysis(df: pd.DataFrame, 
                                   max_lag: int = 2,
                                   bins: int = 10,
                                   alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform transfer entropy analysis on time series data
    
    Args:
        df: DataFrame containing the time series data
        max_lag: Maximum number of lags to test
        bins: Number of bins for discretization
        alpha: Significance level for significance testing
        
    Returns:
        Dictionary with transfer entropy results
    """
    st.subheader("Transfer Entropy Analysis")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter out lag columns if they exist
    orig_cols = [col for col in numeric_cols if not col.endswith(tuple([f"_lag{i}" for i in range(1, max_lag+1)]))]
    
    if len(orig_cols) < 2:
        st.error("Need at least 2 numeric columns for transfer entropy analysis")
        return {"status": "error", "message": "Not enough numeric columns"}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Check if jpype and jtransforms are available
        try:
            import jpype
            import jpype.imports
            from jpype.types import *
            
            # Warning about dependency
            st.warning("""
            Transfer entropy requires the Java Partial Information Decomposition (JIDT) toolkit.
            This is a simplified implementation without the full dependency.
            """)
        except ImportError:
            st.error("""
            Transfer entropy requires jpype and the Java Partial Information Decomposition (JIDT) toolkit.
            These dependencies are not available. Using a simplified implementation.
            """)
        
        # Create a directed graph for causal relationships
        G = nx.DiGraph()
        for col in orig_cols:
            G.add_node(col)
        
        # Perform transfer entropy calculation for each pair
        n_pairs = len(orig_cols) * (len(orig_cols) - 1)
        pair_count = 0
        te_results = {}
        
        progress_text.text("Calculating transfer entropy...")
        
        for i, source in enumerate(orig_cols):
            for j, target in enumerate(orig_cols):
                if i != j:  # Skip self-causality
                    pair_count += 1
                    progress_bar.progress(int(100 * pair_count / n_pairs))
                    progress_text.text(f"Calculating TE: {source} → {target} ({pair_count}/{n_pairs})")
                    
                    # Get data for source and target
                    source_data = df[source].dropna().values
                    target_data = df[target].dropna().values
                    
                    # Make sure data lengths match
                    min_len = min(len(source_data), len(target_data))
                    source_data = source_data[:min_len]
                    target_data = target_data[:min_len]
                    
                    # Calculate transfer entropy for each lag
                    te_values = []
                    for lag in range(1, max_lag + 1):
                        # Simplified transfer entropy calculation
                        # In a real implementation, use a proper TE library
                        
                        # Prepare lagged data
                        source_lagged = source_data[:-lag]
                        target_current = target_data[lag:]
                        target_lagged = target_data[:-lag]
                        
                        # Discretize data
                        from sklearn.preprocessing import KBinsDiscretizer
                        
                        disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
                        source_disc = disc.fit_transform(source_lagged.reshape(-1, 1)).flatten()
                        target_disc = disc.fit_transform(target_current.reshape(-1, 1)).flatten()
                        target_hist_disc = disc.fit_transform(target_lagged.reshape(-1, 1)).flatten()
                        
                        # Calculate entropy terms
                        from scipy.stats import entropy
                        
                        # Joint distributions
                        joint_target_hist = np.zeros((bins, bins))
                        joint_source_target_hist = np.zeros((bins, bins, bins))
                        
                        for a, b, c in zip(source_disc, target_disc, target_hist_disc):
                            joint_target_hist[int(c), int(b)] += 1
                            joint_source_target_hist[int(a), int(c), int(b)] += 1
                        
                        # Normalize
                        joint_target_hist /= joint_target_hist.sum()
                        joint_source_target_hist /= joint_source_target_hist.sum()
                        
                        # Calculate conditional entropies
                        h_target_given_hist = 0
                        h_target_given_source_hist = 0
                        
                        for a in range(bins):
                            for c in range(bins):
                                p_ch = joint_target_hist[c, :].sum()
                                if p_ch > 0:
                                    p_t_given_ch = joint_target_hist[c, :] / p_ch
                                    h_target_given_hist -= p_ch * np.sum(p_t_given_ch * np.log2(p_t_given_ch + 1e-10))
                                
                                for b in range(bins):
                                    p_sch = joint_source_target_hist[a, c, :].sum()
                                    if p_sch > 0:
                                        p_t_given_sch = joint_source_target_hist[a, c, :] / p_sch
                                        p_sc = joint_source_target_hist[a, c, :].sum()
                                        h_target_given_source_hist -= p_sc * np.sum(p_t_given_sch * np.log2(p_t_given_sch + 1e-10))
                        
                        # Transfer entropy
                        te = h_target_given_hist - h_target_given_source_hist
                        te_values.append(max(0, te))  # TE should be non-negative
                    
                    # Get maximum TE and corresponding lag
                    if te_values:
                        max_te = max(te_values)
                        best_lag = te_values.index(max_te) + 1
                        
                        # Determine significance through permutation test
                        # This is a simplified approach
                        significant = max_te > 0.01  # Use a fixed threshold as a simple heuristic
                        
                        # Store results
                        te_results[(source, target)] = {
                            "te_values": te_values,
                            "max_te": max_te,
                            "best_lag": best_lag,
                            "significant": significant
                        }
                        
                        # Add edge to graph if significant
                        if significant:
                            G.add_edge(
                                source, 
                                target, 
                                weight=max_te,
                                lag=best_lag,
                                causality="transfer_entropy"
                            )
                    
        progress_bar.progress(100)
        progress_text.text("Transfer entropy analysis completed!")
        
        # Count significant relationships
        significant_count = sum(1 for result in te_results.values() if result.get("significant", False))
        
        # Display summary
        st.write(f"### Summary of Transfer Entropy Analysis")
        st.write(f"Tested {n_pairs} potential causal relationships")
        st.write(f"Found {significant_count} significant information transfers")
        
        # Create a graph visualizer
        graph_viz = CausalGraphVisualizer()
        
        # Visualize the causal graph
        fig = graph_viz.visualize_graph(
            graph=G,
            show_confidence=True,
            title="Transfer Entropy Causal Graph"
        )
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Create detailed results table
        results_data = []
        for (source, target), result in te_results.items():
            if result.get("significant", False):
                results_data.append({
                    "Source": source,
                    "Target": target,
                    "Max TE": f"{result['max_te']:.4f}",
                    "Best Lag": result["best_lag"],
                    "Significant": "Yes"
                })
        
        if results_data:
            st.write("### Significant Information Transfers")
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df)
        
        # Show transfer entropy matrix as heatmap
        st.write("### Transfer Entropy Matrix")
        
        te_matrix = np.zeros((len(orig_cols), len(orig_cols)))
        for i, source in enumerate(orig_cols):
            for j, target in enumerate(orig_cols):
                if (source, target) in te_results:
                    te_matrix[i, j] = te_results[(source, target)].get("max_te", 0)
        
        te_df = pd.DataFrame(te_matrix, index=orig_cols, columns=orig_cols)
        
        fig_te = px.imshow(
            te_df,
            text_auto=True,
            color_continuous_scale='Viridis',
            title="Maximum Transfer Entropy"
        )
        
        st.plotly_chart(fig_te, use_container_width=True)
        
        # Return results
        return {
            "status": "completed",
            "graph": G,
            "te_results": te_results,
            "significant_count": significant_count,
            "total_tests": n_pairs,
            "max_lag": max_lag,
            "method": "Transfer Entropy"
        }
    
    except Exception as e:
        progress_text.text(f"Error in transfer entropy analysis: {str(e)}")
        logger.error(f"Error in transfer entropy analysis: {str(e)}")
        return {"status": "error", "message": str(e)}


def render_nonstationary_discovery(df: pd.DataFrame, 
                                time_index: Optional[List[int]] = None,
                                alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform causal discovery using non-stationarity
    
    Args:
        df: DataFrame containing the time series data
        time_index: List of time period indices (None to auto-detect periods)
        alpha: Significance level for independence tests
        
    Returns:
        Dictionary with nonstationary causal discovery results
    """
    st.subheader("Non-Stationary Causal Discovery")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter out any obvious lag columns
    orig_cols = [col for col in numeric_cols if not col.lower().endswith(("_lag", "_lag1", "_lag2", "_lag3"))]
    
    if len(orig_cols) < 2:
        st.error("Need at least 2 numeric columns for non-stationary causal discovery")
        return {"status": "error", "message": "Not enough numeric columns"}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        progress_text.text("Initializing non-stationary causal discovery...")
        progress_bar.progress(10)
        
        # Create time index if not provided
        if time_index is None:
            # Try to detect time periods
            if pd.api.types.is_datetime64_any_dtype(df.index):
                # Use yearly periods if time span is long enough
                time_span = (df.index.max() - df.index.min()).days
                
                if time_span > 365*2:  # More than 2 years
                    # Create yearly time indices
                    time_index = df.index.year.values - df.index.year.min()
                    st.info(f"Created {len(np.unique(time_index))} time periods based on years")
                elif time_span > 30*3:  # More than 3 months
                    # Create monthly time indices
                    time_index = (df.index.year - df.index.year.min()) * 12 + df.index.month - 1
                    st.info(f"Created {len(np.unique(time_index))} time periods based on months")
                else:
                    # Create weekly time indices
                    time_index = (df.index - df.index.min()).days // 7
                    st.info(f"Created {len(np.unique(time_index))} time periods based on weeks")
            else:
                # Split into 4 equal periods
                time_index = (np.arange(len(df)) * 4 // len(df))
                st.info(f"Created 4 equal time periods")
        
        # Extract data for analysis
        data = df[orig_cols].values
        
        # Initialize non-stationary causal discovery
        progress_text.text("Running non-stationary causal discovery...")
        progress_bar.progress(30)
        
        # Create an instance of NonStationaryCausalDiscovery
        from core.algorithms.nonstationarity import NonStationaryCausalDiscovery
        nscd = NonStationaryCausalDiscovery()
        
        # Run causal discovery
        G, additional_info = nscd.causal_discovery_nonstationary(
            data=data,
            time_index=time_index,
            alpha=alpha
        )
        
        progress_text.text("Analyzing variable distribution changes...")
        progress_bar.progress(60)
        
        # Check which variables have changing distributions
        if "variable_changes" in additional_info:
            var_changes = additional_info["variable_changes"]
            
            # Map numeric variable indices to column names
            var_changes_named = {}
            for var_idx, change_info in var_changes.items():
                if var_idx < len(orig_cols):
                    var_changes_named[orig_cols[var_idx]] = change_info
            
            # Display variables with changing distributions
            changing_vars = [var for var, info in var_changes_named.items() 
                            if info.get("changing", False)]
            
            st.write("### Variables with Changing Distributions")
            if changing_vars:
                st.write(f"Found {len(changing_vars)} variables with changing distributions:")
                for var in changing_vars:
                    info = var_changes_named[var]
                    st.write(f"- {var}: p-value = {info['p_value']:.4f}, strength = {info['strength']:.4f}")
                
                # Visualize distribution changes for one variable
                if len(changing_vars) > 0:
                    selected_var = st.selectbox("Select variable to visualize distribution changes", changing_vars)
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    # Get unique time periods
                    unique_periods = np.unique(time_index)
                    
                    # Add a distribution for each time period
                    for period in unique_periods:
                        period_data = df.loc[time_index == period, selected_var].dropna()
                        
                        # Skip if not enough data
                        if len(period_data) < 5:
                            continue
                        
                        # Add histogram for this period
                        fig.add_trace(go.Histogram(
                            x=period_data,
                            name=f"Period {period}",
                            opacity=0.7,
                            nbinsx=20
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Distribution Changes for {selected_var} Across Time Periods",
                        xaxis_title=selected_var,
                        yaxis_title="Frequency",
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No variables with significantly changing distributions found.")
        
        progress_text.text("Analyzing relationship changes...")
        progress_bar.progress(80)
        
        # Create a graph visualizer
        graph_viz = CausalGraphVisualizer()
        
        # Create a graph with original column names
        G_named = nx.DiGraph()
        
        # Map nodes and edges to column names
        for node in G.nodes():
            if node < len(orig_cols):
                G_named.add_node(orig_cols[node])
        
        for u, v, data in G.edges(data=True):
            if u < len(orig_cols) and v < len(orig_cols):
                G_named.add_edge(orig_cols[u], orig_cols[v], **data)
        
        # Visualize the causal graph
        fig = graph_viz.visualize_graph(
            graph=G_named,
            show_confidence=True,
            title="Non-Stationary Causal Graph"
        )
        
        progress_bar.progress(100)
        progress_text.text("Non-stationary causal discovery completed!")
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyze and display the results
        st.write("### Summary of Non-Stationary Causal Discovery")
        st.write(f"Number of edges discovered: {G_named.number_of_edges()}")
        
        # Check for relationship changes
        if "relationship_changes" in additional_info:
            rel_changes = additional_info["relationship_changes"]
            
            # Count changing relationships
            changing_relationships = sum(1 for info in rel_changes.values() 
                                     if info.get("changing", False))
            
            if changing_relationships > 0:
                st.write(f"Found {changing_relationships} changing relationships between variables")
                
                # Create a table of changing relationships
                change_data = []
                for (i, j), info in rel_changes.items():
                    if info.get("changing", False) and i < len(orig_cols) and j < len(orig_cols):
                        change_data.append({
                            "Variable 1": orig_cols[i],
                            "Variable 2": orig_cols[j],
                            "Strength": f"{info['strength']:.4f}",
                            "P-Value": f"{info['p_value']:.4f}"
                        })
                
                if change_data:
                    st.write("### Changing Relationships")
                    change_df = pd.DataFrame(change_data)
                    st.dataframe(change_df)
        
        # Return results
        return {
            "status": "completed",
            "graph": G_named,
            "raw_graph": G,
            "additional_info": additional_info,
            "method": "Non-stationary causal discovery",
            "variables": orig_cols,
            "time_periods": len(np.unique(time_index)),
            "alpha": alpha
        }
    
    except Exception as e:
        progress_text.text(f"Error in non-stationary causal discovery: {str(e)}")
        logger.error(f"Error in non-stationary causal discovery: {str(e)}")
        return {"status": "error", "message": str(e)}


def render_timeseries_discovery_interface(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Render an interface for time series causal discovery methods
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        Dictionary with selected method results
    """
    st.header("Time Series Causal Discovery")
    
    # Check if data has datetime index
    has_datetime = pd.api.types.is_datetime64_any_dtype(df.index)
    
    # Preprocessing options
    with st.expander("Time Series Preprocessing", expanded=not has_datetime):
        # Display preprocessing options
        date_col = None
        if not has_datetime:
            # List columns that might contain date information
            potential_date_cols = [col for col in df.columns if 
                                'date' in col.lower() or 
                                'time' in col.lower() or 
                                'day' in col.lower() or 
                                'year' in col.lower() or
                                'month' in col.lower()]
            
            all_cols = ["None"] + list(df.columns)
            default_idx = 0
            if potential_date_cols:
                for col in potential_date_cols:
                    if col in all_cols:
                        default_idx = all_cols.index(col)
                        break
            
            date_col_selection = st.selectbox(
                "Date/Time Column",
                all_cols,
                index=default_idx,
                help="Select the column containing date/time information"
            )
            
            if date_col_selection != "None":
                date_col = date_col_selection
        
        # Frequency resampling
        freq_options = ["None", "D", "W", "M", "Q", "Y"]
        freq_labels = ["No resampling", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        
        freq = st.selectbox(
            "Resample Frequency",
            options=freq_options,
            format_func=lambda x: freq_labels[freq_options.index(x)],
            help="Resample time series to a different frequency (requires datetime index)"
        )
        
        if freq == "None":
            freq = None
        
        # Maximum lag for analysis
        max_lag = st.slider(
            "Maximum Lag",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of time lags to consider in the analysis"
        )
        
        # Button to run preprocessing
        if st.button("Preprocess Time Series Data"):
            with st.spinner("Preprocessing time series data..."):
                # Run preprocessing
                preproc_result = render_timeseries_preprocessing(df, date_col, freq, max_lag)
                
                # Store processed data in session state
                if "ts_data" not in st.session_state:
                    st.session_state.ts_data = {}
                
                st.session_state.ts_data["processed_df"] = preproc_result["data"]
                st.session_state.ts_data["metadata"] = preproc_result
                
                # Display time series plots
                render_timeseries_visualization(
                    preproc_result["data"], 
                    variables=None, 
                    date_col=None  # Already set as index
                )
    
    # Get data for analysis
    analysis_df = df
    if "ts_data" in st.session_state and "processed_df" in st.session_state.ts_data:
        analysis_df = st.session_state.ts_data["processed_df"]
        st.success("Using preprocessed time series data")
    
    # Select method
    method = st.selectbox(
        "Time Series Causal Discovery Method",
        ["Granger Causality", 
         "VAR-LiNGAM",
         "Transfer Entropy",
         "Non-Stationary Causal Discovery"],
        help="""
        Granger Causality: Tests if past values of X help predict future values of Y
        VAR-LiNGAM: Combines vector autoregression with non-Gaussian causal discovery
        Transfer Entropy: Information-theoretic approach for directed information flow
        Non-Stationary: Exploits changing data distributions to identify causality
        """
    )
    
    # Method-specific parameters
    if method == "Granger Causality":
        alpha = st.slider(
            "Significance Level",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Significance threshold for Granger causality tests"
        )
        
        if st.button("Run Granger Causality Analysis"):
            with st.spinner("Running Granger causality analysis..."):
                # Get max_lag from preprocessed data or use default
                ml = st.session_state.ts_data.get("metadata", {}).get("max_lag", 5) if "ts_data" in st.session_state else 5
                return render_granger_causality(analysis_df, max_lag=ml, alpha=alpha)
                
    elif method == "VAR-LiNGAM":
        if st.button("Run VAR-LiNGAM Analysis"):
            with st.spinner("Running VAR-LiNGAM analysis..."):
                # Get max_lag from preprocessed data or use default
                ml = st.session_state.ts_data.get("metadata", {}).get("max_lag", 2) if "ts_data" in st.session_state else 2
                return render_var_lingam_analysis(analysis_df, max_lag=ml)
                
    elif method == "Transfer Entropy":
        alpha = st.slider(
            "Significance Level",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Significance threshold for transfer entropy"
        )
        
        bins = st.slider(
            "Number of Bins",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Number of bins for discretizing continuous variables"
        )
        
        if st.button("Run Transfer Entropy Analysis"):
            with st.spinner("Running transfer entropy analysis..."):
                # Get max_lag from preprocessed data or use default
                ml = st.session_state.ts_data.get("metadata", {}).get("max_lag", 2) if "ts_data" in st.session_state else 2
                return render_transfer_entropy_analysis(analysis_df, max_lag=ml, bins=bins, alpha=alpha)
    
    elif method == "Non-Stationary Causal Discovery":
        alpha = st.slider(
            "Significance Level",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Significance threshold for independence tests"
        )
        
        if st.button("Run Non-Stationary Causal Discovery"):
            with st.spinner("Running non-stationary causal discovery..."):
                return render_nonstationary_discovery(analysis_df, time_index=None, alpha=alpha)
    
    # If no analysis has been run yet
    return {"status": "not_run", "message": "Select a method and click the run button"}


def combine_timeseries_analyses(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run multiple time series causal analyses and combine the results
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        Dictionary with combined results
    """
    st.header("Comprehensive Time Series Causal Discovery")
    
    # Preprocessing options (simplified)
    with st.expander("Time Series Preprocessing"):
        # Display preprocessing options
        date_col = None
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            # List columns that might contain date information
            potential_date_cols = [col for col in df.columns if 
                                'date' in col.lower() or 
                                'time' in col.lower() or 
                                'day' in col.lower() or 
                                'year' in col.lower() or
                                'month' in col.lower()]
            
            all_cols = ["None"] + list(df.columns)
            default_idx = 0
            if potential_date_cols:
                for col in potential_date_cols:
                    if col in all_cols:
                        default_idx = all_cols.index(col)
                        break
            
            date_col_selection = st.selectbox(
                "Date/Time Column",
                all_cols,
                index=default_idx,
                help="Select the column containing date/time information"
            )
            
            if date_col_selection != "None":
                date_col = date_col_selection
        
        max_lag = st.slider(
            "Maximum Lag",
            min_value=1,
            max_value=10,
            value=2,
            help="Maximum number of time lags to consider in the analysis"
        )
        
        # Button to run preprocessing
        if st.button("Preprocess Time Series Data"):
            with st.spinner("Preprocessing time series data..."):
                # Run preprocessing (with no resampling)
                preproc_result = render_timeseries_preprocessing(df, date_col, None, max_lag)
                
                # Store processed data in session state
                if "ts_data" not in st.session_state:
                    st.session_state.ts_data = {}
                
                st.session_state.ts_data["processed_df"] = preproc_result["data"]
                st.session_state.ts_data["metadata"] = preproc_result
                
                # Display time series plots
                render_timeseries_visualization(
                    preproc_result["data"], 
                    variables=None, 
                    date_col=None  # Already set as index
                )
    
    # Get data for analysis
    analysis_df = df
    if "ts_data" in st.session_state and "processed_df" in st.session_state.ts_data:
        analysis_df = st.session_state.ts_data["processed_df"]
        st.success("Using preprocessed time series data")
    
    # Select methods to use
    methods = st.multiselect(
        "Select Methods to Use",
        options=["Granger Causality", "VAR-LiNGAM", "Transfer Entropy"],
        default=["Granger Causality", "VAR-LiNGAM"],
        help="Select which time series methods to include in the analysis"
    )
    
    if not methods:
        st.error("Need to select at least one method")
        return {"status": "error", "message": "No methods selected"}
    
    # Set global parameters
    alpha = st.slider(
        "Significance Level",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Significance threshold for causal tests"
    )
    
    # Get max_lag from preprocessed data or use default
    max_lag = st.session_state.ts_data.get("metadata", {}).get("max_lag", 2) if "ts_data" in st.session_state else 2
    
    if st.button("Run Comprehensive Analysis"):
        with st.spinner("Running comprehensive time series causal discovery..."):
            # Create progress indicators
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Track results for each method
            all_results = {}
            
            # Get numeric columns (not lag columns)
            numeric_cols = analysis_df.select_dtypes(include=np.number).columns.tolist()
            orig_cols = [col for col in numeric_cols if not col.endswith(tuple([f"_lag{i}" for i in range(1, max_lag+1)]))]
            
            # Create a directed graph for final results
            G_final = nx.DiGraph()
            for col in orig_cols:
                G_final.add_node(col)
            
            # Run each selected method
            method_idx = 0
            for method in methods:
                method_idx += 1
                progress = int(100 * method_idx / (len(methods) + 1))
                progress_bar.progress(progress)
                progress_text.text(f"Running {method} analysis...")
                
                if method == "Granger Causality":
                    result = render_granger_causality(analysis_df, max_lag=max_lag, alpha=alpha)
                    if result["status"] == "completed":
                        all_results["Granger"] = result
                        
                        # Add edges to final graph
                        for u, v, data in result["graph"].edges(data=True):
                            if G_final.has_edge(u, v):
                                # Edge already exists, update weight
                                current_weight = G_final[u][v].get("weight", 0)
                                G_final[u][v]["weight"] = max(current_weight, data.get("weight", 0))
                                G_final[u][v]["methods"] = G_final[u][v].get("methods", []) + ["Granger"]
                            else:
                                # Add new edge
                                G_final.add_edge(u, v, weight=data.get("weight", 0), methods=["Granger"])
                
                elif method == "VAR-LiNGAM":
                    result = render_var_lingam_analysis(analysis_df, max_lag=max_lag)
                    if result["status"] == "completed":
                        all_results["VAR-LiNGAM"] = result
                        
                        # Add edges to final graph
                        for u, v, data in result["graph"].edges(data=True):
                            if G_final.has_edge(u, v):
                                # Edge already exists, update weight
                                current_weight = G_final[u][v].get("weight", 0)
                                G_final[u][v]["weight"] = max(current_weight, data.get("weight", 0))
                                G_final[u][v]["methods"] = G_final[u][v].get("methods", []) + ["VAR-LiNGAM"]
                            else:
                                # Add new edge
                                G_final.add_edge(u, v, weight=data.get("weight", 0), methods=["VAR-LiNGAM"])
                
                elif method == "Transfer Entropy":
                    result = render_transfer_entropy_analysis(analysis_df, max_lag=max_lag, bins=10, alpha=alpha)
                    if result["status"] == "completed":
                        all_results["TE"] = result
                        
                        # Add edges to final graph
                        for u, v, data in result["graph"].edges(data=True):
                            if G_final.has_edge(u, v):
                                # Edge already exists, update weight
                                current_weight = G_final[u][v].get("weight", 0)
                                G_final[u][v]["weight"] = max(current_weight, data.get("weight", 0))
                                G_final[u][v]["methods"] = G_final[u][v].get("methods", []) + ["Transfer Entropy"]
                            else:
                                # Add new edge
                                G_final.add_edge(u, v, weight=data.get("weight", 0), methods=["Transfer Entropy"])
            
            progress_text.text("Creating consensus graph...")
            progress_bar.progress(100)
            
            # Visualize the combined graph
            graph_viz = CausalGraphVisualizer()
            fig = graph_viz.visualize_graph(
                graph=G_final,
                show_confidence=True,
                title="Combined Time Series Causal Graph"
            )
            
            progress_text.text("Analysis complete!")
            
            # Display the graph
            st.subheader("Combined Time Series Causal Graph")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display method agreement
            st.subheader("Method Agreement")
            
            # Count edges by number of methods
            edge_counts = {}
            for u, v, data in G_final.edges(data=True):
                method_count = len(data.get("methods", []))
                edge_counts[method_count] = edge_counts.get(method_count, 0) + 1
            
            # Display counts
            for count, num_edges in sorted(edge_counts.items(), reverse=True):
                method_text = "methods" if count > 1 else "method"
                st.write(f"- {num_edges} edges supported by {count} {method_text}")
            
            # Create table of causal relationships
            st.subheader("Causal Relationships")
            
            edge_data = []
            for u, v, data in G_final.edges(data=True):
                edge_data.append({
                    "Cause": u,
                    "Effect": v,
                    "Weight": f"{data.get('weight', 0):.4f}",
                    "Methods": ", ".join(data.get("methods", [])),
                    "Agreement": len(data.get("methods", []))
                })
            
            if edge_data:
                edge_df = pd.DataFrame(edge_data)
                edge_df = edge_df.sort_values("Agreement", ascending=False)
                st.dataframe(edge_df)
            
            # Return combined results
            return {
                "status": "completed",
                "graph": G_final,
                "all_results": all_results,
                "methods_used": methods,
                "max_lag": max_lag,
                "alpha": alpha,
                "edge_counts": edge_counts
            }
    
    # If no analysis has been run yet
    return {"status": "not_run", "message": "Click the run button to start analysis"}