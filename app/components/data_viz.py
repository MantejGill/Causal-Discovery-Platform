"""
Streamlit components for data visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Tuple


def render_feature_distribution(df: pd.DataFrame, column: str, theme: str = "light"):
    """
    Render a distribution plot for a single feature.
    
    Args:
        df: Pandas DataFrame with the data
        column: Column name to visualize
        theme: Color theme (light or dark)
    """
    from core.viz.distribution import DataVisualizer
    
    # Create visualizer with the selected theme
    visualizer = DataVisualizer(theme=theme)
    
    # Generate the distribution figure
    fig = visualizer.plot_feature_distribution(df, column)
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df: pd.DataFrame, method: str = "pearson", theme: str = "light"):
    """
    Render a correlation heatmap.
    
    Args:
        df: Pandas DataFrame with the data
        method: Correlation method (pearson, spearman, kendall)
        theme: Color theme (light or dark)
    """
    from core.viz.distribution import DataVisualizer
    
    # Create visualizer with the selected theme
    visualizer = DataVisualizer(theme=theme)
    
    # Generate the heatmap figure
    fig = visualizer.plot_correlation_heatmap(df, method=method)
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_by: str = None, theme: str = "light"):
    """
    Render a scatter plot between two variables.
    
    Args:
        df: Pandas DataFrame with the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_by: Column name for coloring points (optional)
        theme: Color theme (light or dark)
    """
    # Create figure
    if color_by:
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_by,
            opacity=0.7,
            title=f"{x_col} vs {y_col}"
        )
    else:
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            opacity=0.7,
            title=f"{x_col} vs {y_col}"
        )
    
    # Set theme colors
    if theme == "dark":
        bg_color = "#1e1e1e"
        plot_bg_color = "#2d2d2d"
        text_color = "#ffffff"
    else:
        bg_color = "#ffffff"
        plot_bg_color = "#f8f9fa"
        text_color = "#000000"
    
    fig.update_layout(
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)


def render_pairwise_scatter(df: pd.DataFrame, columns: List[str] = None, 
                          color_by: str = None, theme: str = "light"):
    """
    Render a matrix of pairwise scatter plots.
    
    Args:
        df: Pandas DataFrame with the data
        columns: List of columns to include (defaults to all numeric columns)
        color_by: Column name for coloring points (optional)
        theme: Color theme (light or dark)
    """
    from core.viz.distribution import DataVisualizer
    
    # Create visualizer with the selected theme
    visualizer = DataVisualizer(theme=theme)
    
    # Generate the scatter matrix figure
    fig = visualizer.plot_pairwise_scatter(df, columns=columns, color_by=color_by)
    
    if fig:
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for pairwise scatter plots.")


def render_missing_values(df: pd.DataFrame, theme: str = "light"):
    """
    Render a visualization of missing values.
    
    Args:
        df: Pandas DataFrame with the data
        theme: Color theme (light or dark)
    """
    from core.viz.distribution import DataVisualizer
    
    # Create visualizer with the selected theme
    visualizer = DataVisualizer(theme=theme)
    
    # Check if there are missing values
    if df.isnull().sum().sum() > 0:
        # Generate the missing values figure
        fig = visualizer.plot_missing_values(df)
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing values found in the dataset.")


def render_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                    color_by: str = None, theme: str = "light"):
    """
    Render a 3D scatter plot.
    
    Args:
        df: Pandas DataFrame with the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        z_col: Column name for z-axis
        color_by: Column name for coloring points (optional)
        theme: Color theme (light or dark)
    """
    from core.viz.distribution import DataVisualizer
    
    # Create visualizer with the selected theme
    visualizer = DataVisualizer(theme=theme)
    
    # Generate the 3D scatter figure
    fig = visualizer.plot_3d_scatter(df, x_col, y_col, z_col, color_by)
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)


def render_data_summary(df: pd.DataFrame):
    """
    Render a summary of the dataset.
    
    Args:
        df: Pandas DataFrame with the data
    """
    st.subheader("Dataset Overview")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rows", df.shape[0])
    
    with col2:
        st.metric("Columns", df.shape[1])
    
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    
    # Data types
    st.subheader("Column Types")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=np.datetime64).columns.tolist()
    other_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols + datetime_cols]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Numeric", len(numeric_cols))
        if numeric_cols:
            with st.expander("Numeric Columns"):
                for col in numeric_cols:
                    st.markdown(f"- {col}")
    
    with col2:
        st.metric("Categorical", len(categorical_cols))
        if categorical_cols:
            with st.expander("Categorical Columns"):
                for col in categorical_cols:
                    st.markdown(f"- {col}")
    
    with col3:
        st.metric("Datetime", len(datetime_cols))
        if datetime_cols:
            with st.expander("Datetime Columns"):
                for col in datetime_cols:
                    st.markdown(f"- {col}")
    
    with col4:
        st.metric("Other", len(other_cols))
        if other_cols:
            with st.expander("Other Columns"):
                for col in other_cols:
                    st.markdown(f"- {col}")