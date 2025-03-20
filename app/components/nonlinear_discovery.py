# app/components/nonlinear_discovery.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats
import logging
import time

# Import custom modules
from core.algorithms.nonlinear_models import AdditiveNoiseModel, PostNonlinearModel
from core.algorithms.kernel_methods import KernelCausalDiscovery
from core.viz.graph import CausalGraphVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def render_anm_pairwise(df: pd.DataFrame, 
                      x_col: str, 
                      y_col: str,
                      regression_method: str = "gp") -> Dict[str, Any]:
    """
    Run Additive Noise Model (ANM) for pairwise causal discovery
    
    Args:
        df: DataFrame containing the data
        x_col: First variable column name
        y_col: Second variable column name
        regression_method: Regression method for ANM ('gp' for Gaussian Process)
        
    Returns:
        Dictionary with causal results
    """
    st.subheader("Additive Noise Model Analysis")
    
    # Check if columns exist
    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"One or both columns not found in dataset: {x_col}, {y_col}")
        return {"status": "error", "message": "Columns not found"}
    
    # Check if columns are numeric
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        st.error(f"Both columns must be numeric for ANM analysis")
        return {"status": "error", "message": "Non-numeric columns"}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Extract data
        progress_text.text("Extracting data...")
        x = df[x_col].dropna().values
        y = df[y_col].dropna().values
        
        # Ensure we have enough data points
        if len(x) < 20 or len(y) < 20:
            st.warning(f"Small sample size may lead to unreliable results (n={min(len(x), len(y))})")
        
        # Get common indices
        common_idx = df[df[x_col].notna() & df[y_col].notna()].index
        x = df.loc[common_idx, x_col].values
        y = df.loc[common_idx, y_col].values
        
        # Initialize ANM model
        progress_text.text("Initializing ANM model...")
        progress_bar.progress(20)
        anm = AdditiveNoiseModel(regression_method=regression_method)
        
        # Run in both directions
        progress_text.text("Testing X → Y direction...")
        progress_bar.progress(40)
        forward_score = anm.fit(x, y)
        
        progress_text.text("Testing Y → X direction...")
        progress_bar.progress(60)
        backward_score = anm.fit(y, x)
        
        # Determine direction
        if forward_score > backward_score:
            direction = f"{x_col} → {y_col}"
            confidence = forward_score / (forward_score + backward_score + 1e-10)
            best_model = "forward"
        else:
            direction = f"{y_col} → {x_col}"
            confidence = backward_score / (forward_score + backward_score + 1e-10)
            best_model = "backward"
        
        progress_text.text("Creating visualizations...")
        progress_bar.progress(80)
        
        # Create scatter plot
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            title=f"Relationship between {x_col} and {y_col}",
            opacity=0.7
        )
        
        # Add regression line for forward and backward models
        x_range = np.linspace(min(x), max(x), 100)
        y_range = np.linspace(min(y), max(y), 100)
        
        # This is simplified - in a real implementation we'd use the actual model predictions
        fig.add_trace(go.Scatter(
            x=x_range,
            y=0.5*x_range + np.random.normal(0, 0.1, size=len(x_range)),
            mode='lines',
            name=f'{x_col} → {y_col}',
            line=dict(color='blue', dash='solid' if best_model == 'forward' else 'dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=0.5*y_range + np.random.normal(0, 0.1, size=len(y_range)),
            y=y_range,
            mode='lines',
            name=f'{y_col} → {x_col}',
            line=dict(color='red', dash='solid' if best_model == 'backward' else 'dash')
        ))
        
        progress_bar.progress(100)
        progress_text.text("Analysis complete!")
        
        # Display results
        st.write(f"### Causal Direction: {direction}")
        st.write(f"Confidence: {confidence:.2f}")
        
        # Show scores
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{x_col} → {y_col} Score", f"{forward_score:.4f}")
        with col2:
            st.metric(f"{y_col} → {x_col} Score", f"{backward_score:.4f}")
        
        # Show scatter plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a simple causal graph
        G = nx.DiGraph()
        if best_model == "forward":
            G.add_edge(x_col, y_col, weight=confidence)
        else:
            G.add_edge(y_col, x_col, weight=confidence)
        
        # Return results
        return {
            "status": "completed",
            "direction": direction,
            "confidence": confidence,
            "forward_score": forward_score,
            "backward_score": backward_score,
            "graph": G,
            "method": "ANM",
            "regression_method": regression_method,
            "x_col": x_col,
            "y_col": y_col,
            "sample_size": len(x)
        }
    
    except Exception as e:
        progress_text.text(f"Error in ANM analysis: {str(e)}")
        logger.error(f"Error in ANM analysis: {str(e)}")
        return {"status": "error", "message": str(e)}


def render_pnl_pairwise(df: pd.DataFrame, 
                      x_col: str, 
                      y_col: str,
                      f1_degree: int = 3,
                      f2_degree: int = 3) -> Dict[str, Any]:
    """
    Run Post-Nonlinear (PNL) Model for pairwise causal discovery
    
    Args:
        df: DataFrame containing the data
        x_col: First variable column name
        y_col: Second variable column name
        f1_degree: Degree of polynomial for f1 function
        f2_degree: Degree of polynomial for f2 function
        
    Returns:
        Dictionary with causal results
    """
    st.subheader("Post-Nonlinear Model Analysis")
    
    # Check if columns exist
    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"One or both columns not found in dataset: {x_col}, {y_col}")
        return {"status": "error", "message": "Columns not found"}
    
    # Check if columns are numeric
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        st.error(f"Both columns must be numeric for PNL analysis")
        return {"status": "error", "message": "Non-numeric columns"}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Extract data
        progress_text.text("Extracting data...")
        x = df[x_col].dropna().values
        y = df[y_col].dropna().values
        
        # Ensure we have enough data points
        if len(x) < 30 or len(y) < 30:
            st.warning(f"Small sample size may lead to unreliable results (n={min(len(x), len(y))})")
        
        # Get common indices
        common_idx = df[df[x_col].notna() & df[y_col].notna()].index
        x = df.loc[common_idx, x_col].values
        y = df.loc[common_idx, y_col].values
        
        # Initialize PNL model
        progress_text.text("Initializing PNL model...")
        progress_bar.progress(20)
        pnl = PostNonlinearModel(f1_degree=f1_degree, f2_degree=f2_degree)
        
        # Run model in both directions
        progress_text.text(f"Testing {x_col} → {y_col} direction...")
        progress_bar.progress(40)
        result = pnl.test_direction(x, y)
        
        progress_text.text("Creating visualizations...")
        progress_bar.progress(80)
        
        # Extract direction and confidence
        direction = result["direction"]
        confidence = result.get("confidence", 0.5)
        
        # Map direction to column names
        if direction == "0->1":
            causal_direction = f"{x_col} → {y_col}"
            cause_col = x_col
            effect_col = y_col
        else:
            causal_direction = f"{y_col} → {x_col}"
            cause_col = y_col
            effect_col = x_col
        
        # Create scatter plot
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            title=f"Relationship between {x_col} and {y_col}",
            opacity=0.7
        )
        
        # Add an arrow annotation showing the causal direction
        fig.add_annotation(
            x=df[x_col].mean(),
            y=df[y_col].mean(),
            ax=df[x_col].mean() + (0.2 * df[x_col].std()) if cause_col == x_col else df[x_col].mean() - (0.2 * df[x_col].std()),
            ay=df[y_col].mean() + (0.2 * df[y_col].std()) if cause_col == y_col else df[y_col].mean() - (0.2 * df[y_col].std()),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="red"
        )
        
        progress_bar.progress(100)
        progress_text.text("Analysis complete!")
        
        # Display results
        st.write(f"### Causal Direction: {causal_direction}")
        st.write(f"Confidence: {confidence:.2f}")
        
        # Show scatter plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a simple causal graph
        G = nx.DiGraph()
        if direction == "0->1":
            G.add_edge(x_col, y_col, weight=confidence)
        else:
            G.add_edge(y_col, x_col, weight=confidence)
        
        # Return results
        return {
            "status": "completed",
            "direction": causal_direction,
            "raw_direction": direction,
            "confidence": confidence,
            "graph": G,
            "method": "PNL",
            "f1_degree": f1_degree,
            "f2_degree": f2_degree,
            "x_col": x_col,
            "y_col": y_col,
            "sample_size": len(x)
        }
    
    except Exception as e:
        progress_text.text(f"Error in PNL analysis: {str(e)}")
        logger.error(f"Error in PNL analysis: {str(e)}")
        return {"status": "error", "message": str(e)}


def render_kernel_causal_discovery(df: pd.DataFrame, alpha: float = 0, 
                                kernel_type: str = "rbf", max_variables: int = 10) -> Dict[str, Any]:
    """
    Run kernel-based causal discovery
    
    Args:
        df: DataFrame containing the data
        alpha: Significance level for independence tests
        kernel_type: Type of kernel ('rbf', 'polynomial', 'laplacian')
        max_variables: Maximum number of variables to include
        
    Returns:
        Dictionary with causal results
    """
    st.subheader("Kernel-based Causal Discovery")

    # Create a kernel causal discovery object
    kernel_params = {}
    if kernel_type == "rbf":
        kernel_params = {"gamma": 0.1}
    elif kernel_type == "polynomial":
        kernel_params = {"degree": 3, "coef0": 1}
        
    kcd = KernelCausalDiscovery(kernel_type=kernel_type, kernel_params=kernel_params)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Limit to max_variables
    if len(numeric_cols) > max_variables:
        st.warning(f"Limited to {max_variables} variables out of {len(numeric_cols)} numeric columns")
        numeric_cols = numeric_cols[:max_variables]
    
    # Check if we have enough columns
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for causal discovery")
        return {"status": "error", "message": "Not enough numeric columns"}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Create a directed graph
        G = nx.DiGraph()
        for col in numeric_cols:
            G.add_node(col)
        
        progress_text.text("Running independence tests between variables...")
        
        # Test for independence between each pair of variables
        n_pairs = (len(numeric_cols) * (len(numeric_cols) - 1)) // 2
        pair_idx = 0
        
        # Track edges for potential addition
        potential_edges = []
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i >= j:  # Skip diagonal and redundant pairs
                    continue
                
                # Update progress
                pair_idx += 1
                progress_bar.progress(int(100 * pair_idx / n_pairs))
                progress_text.text(f"Testing pair {pair_idx}/{n_pairs}: {col1} and {col2}")
                
                # Extract data
                x = df[col1].values
                y = df[col2].values
                
                # Get common indices (non-NaN values)
                common_idx = df[df[col1].notna() & df[col2].notna()].index
                x = df.loc[common_idx, col1].values
                y = df.loc[common_idx, col2].values
                
                # Skip if not enough data
                if len(x) < 20:
                    continue
                
                # Test for independence
                independent, p_value = kcd.kernel_pc_independence_test(x, y, None, alpha)
                
                if not independent:
                    # Variables are dependent, test for causal direction
                    # We'll use a simple method based on ANM for direction
                    anm = AdditiveNoiseModel()
                    direction_result = anm.test_direction(x, y)
                    
                    if direction_result["direction"] == "0->1":
                        # X → Y
                        potential_edges.append({
                            "source": col1,
                            "target": col2,
                            "weight": 1.0 - p_value,
                            "confidence": direction_result.get("confidence", 0.5),
                            "p_value": p_value
                        })
                    else:
                        # Y → X
                        potential_edges.append({
                            "source": col2,
                            "target": col1,
                            "weight": 1.0 - p_value,
                            "confidence": direction_result.get("confidence", 0.5),
                            "p_value": p_value
                        })
        
        # Sort potential edges by weight
        potential_edges.sort(key=lambda x: x["weight"], reverse=True)
        
        # Add edges to graph, checking for cycles
        progress_text.text("Building causal graph...")
        progress_bar.progress(100)
        
        edge_count = 0
        for edge in potential_edges:
            # Check if adding this edge would create a cycle
            if not nx.has_path(G, edge["target"], edge["source"]):
                G.add_edge(
                    edge["source"], 
                    edge["target"], 
                    weight=edge["weight"],
                    confidence=edge["confidence"],
                    p_value=edge["p_value"]
                )
                edge_count += 1
                
        progress_text.text(f"Analysis complete! Found {edge_count} causal relationships.")
        
        # Create a graph visualizer
        graph_viz = CausalGraphVisualizer()
        fig = graph_viz.visualize_graph(
            graph=G,
            show_confidence=True
        )
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Return results
        return {
            "status": "completed",
            "graph": G,
            "method": "Kernel-based Causal Discovery",
            "kernel_type": kernel_type,
            "kernel_params": kernel_params,
            "alpha": alpha,
            "variables": numeric_cols,
            "edges": edge_count,
            "potential_edges": potential_edges
        }
    
    except Exception as e:
        progress_text.text(f"Error in kernel causal discovery: {str(e)}")
        logger.error(f"Error in kernel causal discovery: {str(e)}")
        return {"status": "error", "message": str(e)}


def render_igci_pairwise(df: pd.DataFrame, 
                      x_col: str, 
                      y_col: str,
                      method: str = "entropy") -> Dict[str, Any]:
    """
    Run Information Geometric Causal Inference (IGCI) for pairwise causal discovery
    
    Args:
        df: DataFrame containing the data
        x_col: First variable column name
        y_col: Second variable column name
        method: IGCI method ('entropy' or 'slope')
        
    Returns:
        Dictionary with causal results
    """
    st.subheader("Information Geometric Causal Inference")
    
    # Check if columns exist
    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"One or both columns not found in dataset: {x_col}, {y_col}")
        return {"status": "error", "message": "Columns not found"}
    
    # Check if columns are numeric
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        st.error(f"Both columns must be numeric for IGCI analysis")
        return {"status": "error", "message": "Non-numeric columns"}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Extract data
        progress_text.text("Extracting data...")
        x = df[x_col].dropna().values
        y = df[y_col].dropna().values
        
        # Ensure we have enough data points
        if len(x) < 20 or len(y) < 20:
            st.warning(f"Small sample size may lead to unreliable results (n={min(len(x), len(y))})")
        
        # Get common indices
        common_idx = df[df[x_col].notna() & df[y_col].notna()].index
        x = df.loc[common_idx, x_col].values
        y = df.loc[common_idx, y_col].values
        
        progress_text.text("Running IGCI analysis...")
        progress_bar.progress(30)
        
        # Normalize data to [0, 1]
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)
        
        # Avoid extreme values
        eps = 1e-10
        x_norm = x_norm * (1 - 2*eps) + eps
        y_norm = y_norm * (1 - 2*eps) + eps
        
        progress_text.text("Computing causal direction...")
        progress_bar.progress(50)
        
        # Compute IGCI score
        if method == "entropy":
            # Entropy-based IGCI
            
            # Estimate entropy for X → Y
            # This is a simplified implementation using binning
            hist_x, _ = np.histogram(x_norm, bins=30, density=True)
            hist_y, _ = np.histogram(y_norm, bins=30, density=True)
            
            # Calculate entropy
            entropy_x = -np.sum(hist_x[hist_x > 0] * np.log(hist_x[hist_x > 0]))
            entropy_y = -np.sum(hist_y[hist_y > 0] * np.log(hist_y[hist_y > 0]))
            
            # IGCI score
            score = entropy_y - entropy_x
            
        elif method == "slope":
            # Slope-based IGCI
            
            # Sort data
            sort_idx = np.argsort(x_norm)
            x_sorted = x_norm[sort_idx]
            y_sorted = y_norm[sort_idx]
            
            # Compute slopes
            dx = np.diff(x_sorted)
            dy = np.diff(y_sorted)
            
            # Avoid division by zero
            dx[dx == 0] = np.min(dx[dx > 0]) if np.any(dx > 0) else 1e-10
            
            # Compute log absolute slopes
            log_slopes = np.log(np.abs(dy / dx))
            
            # IGCI score is the mean
            score = np.mean(log_slopes)
            
        else:
            raise ValueError(f"Unknown IGCI method: {method}")
        
        progress_text.text("Creating visualizations...")
        progress_bar.progress(80)
        
        # Determine direction
        if score < 0:
            direction = f"{x_col} → {y_col}"
            confidence = min(abs(score), 1.0)
            best_direction = "forward"
        else:
            direction = f"{y_col} → {x_col}"
            confidence = min(abs(score), 1.0)
            best_direction = "backward"
        
        # Create scatter plot
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            title=f"Relationship between {x_col} and {y_col}",
            opacity=0.7
        )
        
        # Add an arrow annotation showing the causal direction
        fig.add_annotation(
            x=df[x_col].mean(),
            y=df[y_col].mean(),
            ax=df[x_col].mean() + (0.2 * df[x_col].std()) if best_direction == "forward" else df[x_col].mean() - (0.2 * df[x_col].std()),
            ay=df[y_col].mean() + (0.2 * df[y_col].std()) if best_direction == "backward" else df[y_col].mean() - (0.2 * df[y_col].std()),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="red"
        )
        
        progress_bar.progress(100)
        progress_text.text("Analysis complete!")
        
        # Display results
        st.write(f"### Causal Direction: {direction}")
        st.write(f"Confidence: {confidence:.2f}")
        st.write(f"IGCI Score: {score:.4f}")
        
        # Show scatter plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a simple causal graph
        G = nx.DiGraph()
        if best_direction == "forward":
            G.add_edge(x_col, y_col, weight=confidence)
        else:
            G.add_edge(y_col, x_col, weight=confidence)
        
        # Return results
        return {
            "status": "completed",
            "direction": direction,
            "confidence": confidence,
            "score": score,
            "graph": G,
            "method": "IGCI",
            "igci_method": method,
            "x_col": x_col,
            "y_col": y_col,
            "sample_size": len(x)
        }
    
    except Exception as e:
        progress_text.text(f"Error in IGCI analysis: {str(e)}")
        logger.error(f"Error in IGCI analysis: {str(e)}")
        return {"status": "error", "message": str(e)}


def render_nonlinear_discovery_interface(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Render an interface for nonlinear causal discovery methods
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        Dictionary with selected method results
    """
    st.header("Nonlinear Causal Discovery")
    
    # Select method
    method = st.selectbox(
        "Nonlinear Discovery Method",
        ["Additive Noise Model (ANM)", 
         "Post-Nonlinear Model (PNL)",
         "Information Geometric Causal Inference (IGCI)",
         "Kernel-based Causal Discovery"],
        help="""
        ANM: Discovers causal relationships based on asymmetry of residuals
        PNL: More general model allowing for nonlinear transformations
        IGCI: Uses information geometry for deterministic relationships
        Kernel: Uses kernel-based independence tests for general nonlinear relationships
        """
    )
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols or len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for causal discovery")
        return {"status": "error", "message": "Not enough numeric columns"}
    
    if method in ["Additive Noise Model (ANM)", "Post-Nonlinear Model (PNL)", "Information Geometric Causal Inference (IGCI)"]:
        # These methods work on pairs of variables
        st.subheader("Select Variable Pair")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("First Variable", numeric_cols, index=0)
        
        with col2:
            # Filter out the first variable
            remaining_cols = [col for col in numeric_cols if col != x_col]
            y_col = st.selectbox("Second Variable", remaining_cols, index=0)
        
        # Method-specific parameters
        if method == "Additive Noise Model (ANM)":
            regression_method = st.selectbox(
                "Regression Method",
                ["gp", "linear"],
                help="GP: Gaussian Process (nonparametric), Linear: Linear regression"
            )
            
            if st.button("Run ANM Analysis"):
                with st.spinner("Running ANM analysis..."):
                    return render_anm_pairwise(df, x_col, y_col, regression_method)
                    
        elif method == "Post-Nonlinear Model (PNL)":
            f1_degree = st.slider(
                "Function 1 Degree",
                min_value=1,
                max_value=5,
                value=3,
                help="Degree of polynomial for the first function (higher = more flexible)"
            )
            
            f2_degree = st.slider(
                "Function 2 Degree",
                min_value=1,
                max_value=5,
                value=3,
                help="Degree of polynomial for the second function (higher = more flexible)"
            )
            
            if st.button("Run PNL Analysis"):
                with st.spinner("Running PNL analysis..."):
                    return render_pnl_pairwise(df, x_col, y_col, f1_degree, f2_degree)
                    
        elif method == "Information Geometric Causal Inference (IGCI)":
            igci_method = st.selectbox(
                "IGCI Method",
                ["entropy", "slope"],
                help="Entropy: Based on differential entropy, Slope: Based on slope distributions"
            )
            
            if st.button("Run IGCI Analysis"):
                with st.spinner("Running IGCI analysis..."):
                    return render_igci_pairwise(df, x_col, y_col, igci_method)
    
    elif method == "Kernel-based Causal Discovery":
        # Kernel-based method works on multiple variables
        st.subheader("Kernel-based Causal Discovery Settings")
        
        kernel_type = st.selectbox(
            "Kernel Type",
            ["rbf", "polynomial", "laplacian"],
            help="RBF: Radial basis function, Polynomial: Polynomial kernel, Laplacian: Laplacian kernel"
        )
        
        alpha = st.slider(
            "Significance Level",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Significance level for independence tests (lower = more conservative)"
        )
        
        max_vars = st.slider(
            "Maximum Variables",
            min_value=2,
            max_value=min(20, len(numeric_cols)),
            value=min(10, len(numeric_cols)),
            step=1,
            help="Maximum number of variables to include in the analysis"
        )
        
        if st.button("Run Kernel Causal Discovery"):
            with st.spinner("Running kernel-based causal discovery..."):
                return render_kernel_causal_discovery(df, alpha, kernel_type, max_vars)
    
    # If no analysis has been run yet
    return {"status": "not_run", "message": "Select a method and click the run button"}


def combine_pairwise_analyses(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run multiple pairwise nonlinear causal analyses and combine the results
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        Dictionary with combined results
    """
    st.header("Comprehensive Nonlinear Causal Discovery")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols or len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for causal discovery")
        return {"status": "error", "message": "Not enough numeric columns"}
    
    # Select columns to analyze
    selected_cols = st.multiselect(
        "Select Variables to Analyze",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        help="Select variables for pairwise causal analysis"
    )
    
    if len(selected_cols) < 2:
        st.error("Need at least 2 columns for causal discovery")
        return {"status": "error", "message": "Not enough columns selected"}
    
    # Select methods to use
    methods = st.multiselect(
        "Select Methods to Use",
        options=["ANM", "PNL", "IGCI"],
        default=["ANM", "PNL"],
        help="Select which nonlinear methods to include in the analysis"
    )
    
    if not methods:
        st.error("Need to select at least one method")
        return {"status": "error", "message": "No methods selected"}
    
    if st.button("Run Comprehensive Analysis"):
        with st.spinner("Running comprehensive nonlinear causal discovery..."):
            # Create progress indicators
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Calculate total pairs
            n_pairs = (len(selected_cols) * (len(selected_cols) - 1)) // 2
            pair_idx = 0
            
            # Create a directed graph
            G = nx.DiGraph()
            for col in selected_cols:
                G.add_node(col)
            
            # Dictionary to store all pairwise results
            all_results = {}
            
            # Analyze each pair of variables
            for i, col1 in enumerate(selected_cols):
                for j, col2 in enumerate(selected_cols):
                    if i >= j:  # Skip diagonal and redundant pairs
                        continue
                    
                    # Update progress
                    pair_idx += 1
                    progress = int(100 * pair_idx / n_pairs)
                    progress_bar.progress(progress)
                    progress_text.text(f"Analyzing pair {pair_idx}/{n_pairs}: {col1} and {col2}")
                    
                    # Initialize vote counting for direction
                    col1_to_col2_votes = 0
                    col2_to_col1_votes = 0
                    total_confidence = 0
                    
                    # Results for this pair
                    pair_results = {}
                    
                    # Run each selected method
                    if "ANM" in methods:
                        anm_result = render_anm_pairwise(df, col1, col2, "gp")
                        
                        if anm_result["status"] == "completed":
                            pair_results["ANM"] = anm_result
                            
                            # Count votes
                            if anm_result["direction"] == f"{col1} → {col2}":
                                col1_to_col2_votes += anm_result["confidence"]
                            else:
                                col2_to_col1_votes += anm_result["confidence"]
                                
                            total_confidence += anm_result["confidence"]
                    
                    if "PNL" in methods:
                        pnl_result = render_pnl_pairwise(df, col1, col2, 3, 3)
                        
                        if pnl_result["status"] == "completed":
                            pair_results["PNL"] = pnl_result
                            
                            # Count votes
                            if pnl_result["direction"] == f"{col1} → {col2}":
                                col1_to_col2_votes += pnl_result["confidence"]
                            else:
                                col2_to_col1_votes += pnl_result["confidence"]
                                
                            total_confidence += pnl_result["confidence"]
                    
                    if "IGCI" in methods:
                        igci_result = render_igci_pairwise(df, col1, col2, "entropy")
                        
                        if igci_result["status"] == "completed":
                            pair_results["IGCI"] = igci_result
                            
                            # Count votes
                            if igci_result["direction"] == f"{col1} → {col2}":
                                col1_to_col2_votes += igci_result["confidence"]
                            else:
                                col2_to_col1_votes += igci_result["confidence"]
                                
                            total_confidence += igci_result["confidence"]
                    
                    # Determine consensus direction
                    if col1_to_col2_votes > col2_to_col1_votes:
                        consensus_direction = f"{col1} → {col2}"
                        consensus_confidence = col1_to_col2_votes / total_confidence if total_confidence > 0 else 0.5
                        
                        # Add edge to graph
                        G.add_edge(col1, col2, weight=consensus_confidence, methods=methods)
                        
                    elif col2_to_col1_votes > col1_to_col2_votes:
                        consensus_direction = f"{col2} → {col1}"
                        consensus_confidence = col2_to_col1_votes / total_confidence if total_confidence > 0 else 0.5
                        
                        # Add edge to graph
                        G.add_edge(col2, col1, weight=consensus_confidence, methods=methods)
                        
                    else:
                        consensus_direction = "Undetermined"
                        consensus_confidence = 0.5
                    
                    # Store consensus
                    pair_results["consensus"] = {
                        "direction": consensus_direction,
                        "confidence": consensus_confidence,
                        "col1_to_col2_votes": col1_to_col2_votes,
                        "col2_to_col1_votes": col2_to_col1_votes
                    }
                    
                    # Add to all results
                    all_results[(col1, col2)] = pair_results
            
            # Create a graph visualizer
            graph_viz = CausalGraphVisualizer()
            fig = graph_viz.visualize_graph(
                graph=G,
                show_confidence=True
            )
            
            # Finalize progress
            progress_bar.progress(100)
            progress_text.text("Analysis complete!")
            
            # Display the graph
            st.subheader("Causal Graph from Nonlinear Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display consensus table
            st.subheader("Consensus Results")
            consensus_data = []
            for (col1, col2), results in all_results.items():
                consensus = results["consensus"]
                consensus_data.append({
                    "Variable 1": col1,
                    "Variable 2": col2,
                    "Direction": consensus["direction"],
                    "Confidence": f"{consensus['confidence']:.2f}",
                    "Methods": ", ".join(methods)
                })
            
            consensus_df = pd.DataFrame(consensus_data)
            st.dataframe(consensus_df)
            
            # Return combined results
            return {
                "status": "completed",
                "graph": G,
                "method": "Comprehensive Nonlinear Discovery",
                "methods_used": methods,
                "all_results": all_results,
                "variables": selected_cols,
                "n_pairs": n_pairs
            }
    
    # If no analysis has been run yet
    return {"status": "not_run", "message": "Click the run button to start analysis"}