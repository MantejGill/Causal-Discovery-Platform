"""
Distribution explorer component for visualizing and analyzing variable distributions.
Provides interactive exploration of univariate and bivariate distributions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributionExplorer:
    """
    Component for exploring the distributions of variables in a dataset.
    Provides visualizations and statistical summaries for understanding
    variable properties important for causal discovery.
    """
    
    def __init__(self, data: pd.DataFrame, theme: str = "light"):
        """
        Initialize the distribution explorer with a dataset
        
        Args:
            data: DataFrame containing the data to explore
            theme: Visual theme ('light' or 'dark')
        """
        self.data = data
        self.theme = theme
        
        # Set theme colors
        if theme == "dark":
            self.bg_color = "#1e1e1e"
            self.plot_bg_color = "#2d2d2d"
            self.text_color = "#ffffff"
            self.color_sequence = ["#2c6fbb", "#4b8bbf", "#66a3ff", "#99c2ff", "#cce0ff"]
        else:
            self.bg_color = "#ffffff"
            self.plot_bg_color = "#f8f9fa"
            self.text_color = "#000000"
            self.color_sequence = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    def get_variable_types(self) -> Dict[str, List[str]]:
        """
        Categorize variables by their types
        
        Returns:
            Dictionary with variable types
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.data.select_dtypes(include=[np.datetime64]).columns.tolist()
        boolean_cols = []
        
        # Check for boolean columns (might be encoded as numeric)
        for col in numeric_cols:
            if set(self.data[col].unique()).issubset({0, 1, np.nan}):
                boolean_cols.append(col)
                numeric_cols.remove(col)
        
        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "boolean": boolean_cols
        }
    
    def create_univariate_plot(self, variable: str, plot_type: str = "auto", bins: int = 30) -> go.Figure:
        """
        Create a univariate distribution plot for a variable
        
        Args:
            variable: Name of the variable to plot
            plot_type: Type of plot (auto, histogram, kde, box, violin)
            bins: Number of bins for histogram
            
        Returns:
            Plotly figure
        """
        if variable not in self.data.columns:
            return self._create_error_figure(f"Variable '{variable}' not found in data")
        
        # Detect variable type if plot_type is auto
        var_type = "numeric"
        if variable in self.get_variable_types()["categorical"]:
            var_type = "categorical"
        elif variable in self.get_variable_types()["boolean"]:
            var_type = "boolean"
        elif variable in self.get_variable_types()["datetime"]:
            var_type = "datetime"
        
        # Choose plot type based on variable type if auto
        if plot_type == "auto":
            if var_type == "categorical" or var_type == "boolean":
                plot_type = "bar"
            elif var_type == "datetime":
                plot_type = "line"
            else:
                plot_type = "histogram"
        
        # Create figure based on plot type
        try:
            if plot_type == "histogram":
                # Create histogram with KDE
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add histogram
                hist = go.Histogram(
                    x=self.data[variable],
                    nbinsx=bins,
                    name="Histogram",
                    marker_color=self.color_sequence[0]
                )
                fig.add_trace(hist)
                
                # Try to add KDE curve
                try:
                    if var_type == "numeric":
                        # Remove NaN values
                        data_no_nan = self.data[variable].dropna()
                        
                        # Create KDE
                        kde = stats.gaussian_kde(data_no_nan)
                        x_range = np.linspace(data_no_nan.min(), data_no_nan.max(), 1000)
                        y_kde = kde(x_range)
                        
                        # Scale KDE to match histogram
                        hist_values, bin_edges = np.histogram(data_no_nan, bins=bins)
                        hist_max = hist_values.max()
                        y_kde_scaled = y_kde * (hist_max / y_kde.max())
                        
                        # Add KDE curve
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_kde_scaled,
                                mode='lines',
                                name='KDE',
                                line=dict(color=self.color_sequence[1], width=2)
                            )
                        )
                except Exception as e:
                    logger.warning(f"Could not add KDE for {variable}: {str(e)}")
                
                # Update layout
                fig.update_layout(
                    title=f"Distribution of {variable}",
                    xaxis_title=variable,
                    yaxis_title="Count",
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
                
            elif plot_type == "kde" and var_type == "numeric":
                # Create KDE plot
                data_no_nan = self.data[variable].dropna()
                
                # Create KDE
                kde = stats.gaussian_kde(data_no_nan)
                x_range = np.linspace(data_no_nan.min(), data_no_nan.max(), 1000)
                y_kde = kde(x_range)
                
                # Create figure
                fig = go.Figure()
                
                # Add KDE curve
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_kde,
                        mode='lines',
                        fill='tozeroy',
                        name='KDE',
                        line=dict(color=self.color_sequence[0], width=2)
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Density of {variable}",
                    xaxis_title=variable,
                    yaxis_title="Density",
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
                
            elif plot_type == "box":
                # Create box plot
                fig = go.Figure()
                
                # Add box plot
                fig.add_trace(
                    go.Box(
                        y=self.data[variable],
                        name=variable,
                        marker_color=self.color_sequence[0],
                        boxmean=True
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Box Plot of {variable}",
                    yaxis_title=variable,
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
                
            elif plot_type == "violin":
                # Create violin plot
                fig = go.Figure()
                
                # Add violin plot
                fig.add_trace(
                    go.Violin(
                        y=self.data[variable],
                        name=variable,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=self.color_sequence[0],
                        line_color=self.color_sequence[1]
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Violin Plot of {variable}",
                    yaxis_title=variable,
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
                
            elif plot_type == "bar" and (var_type == "categorical" or var_type == "boolean"):
                # Create bar chart of value counts
                value_counts = self.data[variable].value_counts().reset_index()
                value_counts.columns = [variable, 'Count']
                
                # Sort by count if there are many categories
                if len(value_counts) > 10:
                    value_counts = value_counts.sort_values('Count', ascending=False)
                
                # Create figure
                fig = go.Figure()
                
                # Add bar chart
                fig.add_trace(
                    go.Bar(
                        x=value_counts[variable],
                        y=value_counts['Count'],
                        marker_color=self.color_sequence[0]
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Value Counts for {variable}",
                    xaxis_title=variable,
                    yaxis_title="Count",
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
                
                # Rotate x-axis labels if there are many categories
                if len(value_counts) > 5:
                    fig.update_layout(
                        xaxis=dict(tickangle=45)
                    )
                
            elif plot_type == "line" and var_type == "datetime":
                # Create time series plot
                fig = go.Figure()
                
                # Sort by date
                sorted_data = self.data.sort_values(variable)
                
                # Count occurrences per date or use value column if specified
                if len(self.data) > 1000:
                    # For large datasets, bin by date
                    date_counts = self.data[variable].dt.date.value_counts().sort_index()
                    fig.add_trace(
                        go.Scatter(
                            x=date_counts.index,
                            y=date_counts.values,
                            mode='lines',
                            line=dict(color=self.color_sequence[0], width=2)
                        )
                    )
                else:
                    # For smaller datasets, show individual points
                    fig.add_trace(
                        go.Scatter(
                            x=sorted_data[variable],
                            y=np.arange(len(sorted_data)),
                            mode='markers',
                            marker=dict(color=self.color_sequence[0])
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"Timeline of {variable}",
                    xaxis_title=variable,
                    yaxis_title="Count",
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
                
            else:
                # Default to histogram for unknown plot types
                fig = px.histogram(
                    self.data,
                    x=variable,
                    nbins=bins,
                    title=f"Distribution of {variable}"
                )
                
                # Update layout for theme
                fig.update_layout(
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distribution plot for {variable}: {str(e)}")
            return self._create_error_figure(f"Error creating plot: {str(e)}")
    
    def create_bivariate_plot(self, 
                            x_var: str, 
                            y_var: str, 
                            color_var: Optional[str] = None,
                            plot_type: str = "auto") -> go.Figure:
        """
        Create a bivariate relationship plot for two variables
        
        Args:
            x_var: Name of the variable for x-axis
            y_var: Name of the variable for y-axis
            color_var: Optional variable for color encoding
            plot_type: Type of plot (auto, scatter, density, violin, box)
            
        Returns:
            Plotly figure
        """
        # Check if variables exist
        for var in [x_var, y_var]:
            if var not in self.data.columns:
                return self._create_error_figure(f"Variable '{var}' not found in data")
        
        if color_var and color_var not in self.data.columns:
            return self._create_error_figure(f"Color variable '{color_var}' not found in data")
        
        # Detect variable types
        var_types = self.get_variable_types()
        
        x_type = "numeric"
        if x_var in var_types["categorical"]:
            x_type = "categorical"
        elif x_var in var_types["boolean"]:
            x_type = "boolean"
        
        y_type = "numeric"
        if y_var in var_types["categorical"]:
            y_type = "categorical"
        elif y_var in var_types["boolean"]:
            y_type = "boolean"
        
        # Choose plot type based on variable types if auto
        if plot_type == "auto":
            if x_type == "numeric" and y_type == "numeric":
                plot_type = "scatter"
            elif x_type == "categorical" and y_type == "numeric":
                plot_type = "box"
            elif x_type == "numeric" and y_type == "categorical":
                plot_type = "violin"
            elif x_type == "categorical" and y_type == "categorical":
                plot_type = "heatmap"
            elif x_type == "boolean" or y_type == "boolean":
                plot_type = "bar"
        
        # Create figure based on plot type
        try:
            if plot_type == "scatter" and x_type == "numeric" and y_type == "numeric":
                # Create scatter plot
                if color_var:
                    fig = px.scatter(
                        self.data,
                        x=x_var,
                        y=y_var,
                        color=color_var,
                        opacity=0.7,
                        title=f"{y_var} vs {x_var}"
                    )
                else:
                    fig = px.scatter(
                        self.data,
                        x=x_var,
                        y=y_var,
                        opacity=0.7,
                        color_discrete_sequence=[self.color_sequence[0]],
                        title=f"{y_var} vs {x_var}"
                    )
                
                # Add trendline
                try:
                    # Calculate correlation
                    corr = self.data[[x_var, y_var]].corr().iloc[0, 1]
                    
                    # Add trendline
                    fig.add_trace(
                        go.Scatter(
                            x=self.data[x_var].sort_values(),
                            y=np.poly1d(np.polyfit(self.data[x_var], self.data[y_var], 1))(self.data[x_var].sort_values()),
                            mode='lines',
                            name=f'Trendline (r={corr:.3f})',
                            line=dict(color='red', width=2, dash='dash')
                        )
                    )
                    
                    # Add correlation annotation
                    fig.add_annotation(
                        x=0.05,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"Correlation: {corr:.3f}",
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                except Exception as e:
                    logger.warning(f"Could not add trendline: {str(e)}")
            
            elif plot_type == "density" and x_type == "numeric" and y_type == "numeric":
                # Create density heatmap
                fig = go.Figure()
                
                # Remove missing values
                df_clean = self.data[[x_var, y_var]].dropna()
                
                # Calculate 2D histogram
                x_bins = np.linspace(df_clean[x_var].min(), df_clean[x_var].max(), 50)
                y_bins = np.linspace(df_clean[y_var].min(), df_clean[y_var].max(), 50)
                
                hist, x_edges, y_edges = np.histogram2d(
                    df_clean[x_var], df_clean[y_var], 
                    bins=[x_bins, y_bins]
                )
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        x=x_edges,
                        y=y_edges,
                        z=hist.T,
                        colorscale='Blues',
                        colorbar=dict(title='Count')
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Density Heatmap: {y_var} vs {x_var}",
                    xaxis_title=x_var,
                    yaxis_title=y_var,
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
                
            elif plot_type == "box" and y_type == "numeric":
                # Create box plot
                if color_var:
                    fig = px.box(
                        self.data, 
                        x=x_var, 
                        y=y_var,
                        color=color_var,
                        title=f"{y_var} by {x_var}"
                    )
                else:
                    fig = px.box(
                        self.data,
                        x=x_var,
                        y=y_var,
                        color_discrete_sequence=self.color_sequence,
                        title=f"{y_var} by {x_var}"
                    )
                
            elif plot_type == "violin" and y_type == "numeric":
                # Create violin plot
                if color_var:
                    fig = px.violin(
                        self.data,
                        x=x_var,
                        y=y_var,
                        color=color_var,
                        box=True,
                        points="all",
                        title=f"{y_var} by {x_var}"
                    )
                else:
                    fig = px.violin(
                        self.data,
                        x=x_var,
                        y=y_var,
                        box=True,
                        points="all",
                        color_discrete_sequence=self.color_sequence,
                        title=f"{y_var} by {x_var}"
                    )
                
            elif plot_type == "heatmap" and x_type in ["categorical", "boolean"] and y_type in ["categorical", "boolean"]:
                # Create contingency table
                contingency = pd.crosstab(self.data[y_var], self.data[x_var])
                
                # Create heatmap
                fig = px.imshow(
                    contingency,
                    labels=dict(x=x_var, y=y_var, color="Count"),
                    x=contingency.columns,
                    y=contingency.index,
                    color_continuous_scale="Blues",
                    title=f"Contingency Table: {y_var} vs {x_var}"
                )
                
                # Add text annotations
                annotations = []
                for i, row in enumerate(contingency.values):
                    for j, val in enumerate(row):
                        annotations.append(
                            dict(
                                x=j,
                                y=i,
                                text=str(val),
                                showarrow=False,
                                font=dict(color="black" if val < np.max(contingency.values) / 2 else "white")
                            )
                        )
                
                fig.update_layout(annotations=annotations)
                
            elif plot_type == "bar":
                # Create grouped bar chart
                if x_type == "categorical" and y_type == "categorical":
                    # For two categorical variables
                    counts = pd.crosstab(self.data[y_var], self.data[x_var])
                    
                    fig = go.Figure()
                    
                    for category in counts.index:
                        fig.add_trace(
                            go.Bar(
                                name=str(category),
                                x=counts.columns,
                                y=counts.loc[category],
                            )
                        )
                    
                    fig.update_layout(
                        title=f"{y_var} vs {x_var}",
                        xaxis_title=x_var,
                        yaxis_title="Count",
                        barmode='group',
                        plot_bgcolor=self.plot_bg_color,
                        paper_bgcolor=self.bg_color,
                        font=dict(color=self.text_color)
                    )
                    
                elif x_type == "categorical" and y_type == "numeric":
                    # For categorical x and numeric y
                    fig = px.bar(
                        self.data.groupby(x_var)[y_var].mean().reset_index(),
                        x=x_var,
                        y=y_var,
                        color=x_var if len(self.data[x_var].unique()) <= 10 else None,
                        title=f"Mean {y_var} by {x_var}"
                    )
                    
                elif x_type == "numeric" and y_type == "categorical":
                    # For numeric x and categorical y
                    fig = px.bar(
                        self.data.groupby(y_var)[x_var].mean().reset_index(),
                        y=y_var,
                        x=x_var,
                        color=y_var if len(self.data[y_var].unique()) <= 10 else None,
                        title=f"Mean {x_var} by {y_var}",
                        orientation='h'
                    )
                
                else:
                    # Default to scatter plot for other combinations
                    fig = px.scatter(
                        self.data,
                        x=x_var,
                        y=y_var,
                        title=f"{y_var} vs {x_var}"
                    )
            
            else:
                # Default to scatter plot for unknown combinations
                fig = px.scatter(
                    self.data,
                    x=x_var,
                    y=y_var,
                    title=f"{y_var} vs {x_var}"
                )
            
            # Update layout for theme
            fig.update_layout(
                plot_bgcolor=self.plot_bg_color,
                paper_bgcolor=self.bg_color,
                font=dict(color=self.text_color)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating bivariate plot: {str(e)}")
            return self._create_error_figure(f"Error creating plot: {str(e)}")
    
    def create_correlation_matrix(self, methods: List[str] = ['pearson']) -> Dict[str, go.Figure]:
        """
        Create correlation matrix visualizations
        
        Args:
            methods: List of correlation methods to use ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary of Plotly figures for each method
        """
        # Get numeric columns
        numeric_cols = self.get_variable_types()["numeric"]
        
        if len(numeric_cols) < 2:
            return {
                "error": self._create_error_figure("Need at least 2 numeric columns for correlation matrix")
            }
        
        figures = {}
        
        for method in methods:
            try:
                # Calculate correlation matrix
                corr_matrix = self.data[numeric_cols].corr(method=method)
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color=f"{method.capitalize()} Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title=f"{method.capitalize()} Correlation Matrix"
                )
                
                # Add correlation values as text
                annotations = []
                for i, row in enumerate(corr_matrix.values):
                    for j, val in enumerate(row):
                        annotations.append(
                            dict(
                                x=j,
                                y=i,
                                text=f"{val:.2f}",
                                showarrow=False,
                                font=dict(
                                    color="white" if abs(val) > 0.5 else "black"
                                )
                            )
                        )
                
                fig.update_layout(annotations=annotations)
                
                # Update layout for theme
                fig.update_layout(
                    plot_bgcolor=self.plot_bg_color,
                    paper_bgcolor=self.bg_color,
                    font=dict(color=self.text_color)
                )
                
                figures[method] = fig
                
            except Exception as e:
                logger.error(f"Error creating {method} correlation matrix: {str(e)}")
                figures[method] = self._create_error_figure(f"Error creating {method} correlation matrix: {str(e)}")
        
        return figures
    
    def create_pairwise_scatter_matrix(self, 
                                     variables: Optional[List[str]] = None, 
                                     max_vars: int = 5,
                                     color_var: Optional[str] = None) -> go.Figure:
        """
        Create a pairwise scatter plot matrix
        
        Args:
            variables: List of variables to include (defaults to numeric columns)
            max_vars: Maximum number of variables to include
            color_var: Variable to use for color encoding
            
        Returns:
            Plotly figure
        """
        # Get numeric columns if variables not specified
        if variables is None:
            variables = self.get_variable_types()["numeric"]
        
        # Check if we have valid variables
        for var in variables:
            if var not in self.data.columns:
                return self._create_error_figure(f"Variable '{var}' not found in data")
        
        # Limit number of variables
        if len(variables) > max_vars:
            variables = variables[:max_vars]
            logger.warning(f"Limited to {max_vars} variables for scatter matrix")
        
        try:
            # Create scatter matrix
            if color_var:
                fig = px.scatter_matrix(
                    self.data,
                    dimensions=variables,
                    color=color_var,
                    title="Scatter Plot Matrix",
                    opacity=0.7
                )
            else:
                fig = px.scatter_matrix(
                    self.data,
                    dimensions=variables,
                    title="Scatter Plot Matrix",
                    opacity=0.7,
                    color_discrete_sequence=[self.color_sequence[0]]
                )
            
            # Update layout for theme
            fig.update_layout(
                plot_bgcolor=self.plot_bg_color,
                paper_bgcolor=self.bg_color,
                font=dict(color=self.text_color)
            )
            
            # Update trace settings
            fig.update_traces(
                diagonal_visible=True,
                showupperhalf=False,
                marker=dict(size=5)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter matrix: {str(e)}")
            return self._create_error_figure(f"Error creating scatter matrix: {str(e)}")
    
    def create_missing_values_heatmap(self) -> go.Figure:
        """
        Create a heatmap of missing values
        
        Returns:
            Plotly figure
        """
        try:
            # Check if there are any missing values
            if not self.data.isnull().values.any():
                return self._create_error_figure("No missing values found in data")
            
            # Create missing values mask
            missing_mask = self.data.isnull().astype(int)
            
            # Create heatmap
            fig = px.imshow(
                missing_mask.T,
                labels=dict(x="Data Point Index", y="Variable", color="Missing"),
                color_continuous_scale=["#2c6fbb", "#f44336"],
                title="Missing Values Heatmap"
            )
            
            # Update layout
            fig.update_layout(
                xaxis=dict(
                    title="Data Point Index",
                    showgrid=False,
                    showticklabels=False if len(self.data) > 100 else True,
                    tickmode='array',
                    tickvals=list(range(0, len(self.data), max(1, len(self.data) // 10)))
                ),
                yaxis=dict(
                    title="Variable",
                    showgrid=False
                ),
                plot_bgcolor=self.plot_bg_color,
                paper_bgcolor=self.bg_color,
                font=dict(color=self.text_color)
            )
            
            # Add summary of missing values
            missing_count = self.data.isnull().sum().sort_values(ascending=False)
            missing_percent = (missing_count / len(self.data) * 100).round(2)
            
            annotations = []
            for i, var in enumerate(missing_mask.columns):
                if missing_count[var] > 0:
                    annotations.append(
                        dict(
                            x=len(self.data) + 10,  # Position at the right edge
                            y=i,
                            text=f"{missing_count[var]} ({missing_percent[var]}%)",
                            showarrow=False,
                            xref="x",
                            yref="y",
                            align="left"
                        )
                    )
            
            fig.update_layout(annotations=annotations)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating missing values heatmap: {str(e)}")
            return self._create_error_figure(f"Error creating missing values heatmap: {str(e)}")
    
    def calculate_variable_statistics(self, variable: str) -> Dict[str, Any]:
        """
        Calculate detailed statistics for a variable
        
        Args:
            variable: Name of the variable
            
        Returns:
            Dictionary of statistics
        """
        if variable not in self.data.columns:
            return {"error": f"Variable '{variable}' not found in data"}
        
        # Get variable type
        var_types = self.get_variable_types()
        
        var_type = "numeric"
        if variable in var_types["categorical"]:
            var_type = "categorical"
        elif variable in var_types["boolean"]:
            var_type = "boolean"
        elif variable in var_types["datetime"]:
            var_type = "datetime"
        
        # Calculate basic statistics for all types
        stats = {
            "name": variable,
            "type": var_type,
            "count": len(self.data[variable]),
            "missing": self.data[variable].isnull().sum(),
            "missing_percent": round(self.data[variable].isnull().sum() / len(self.data) * 100, 2),
            "unique_values": self.data[variable].nunique()
        }
        
        # Add type-specific statistics
        if var_type == "numeric":
            # Add numeric statistics
            stats.update({
                "mean": float(self.data[variable].mean()),
                "std": float(self.data[variable].std()),
                "min": float(self.data[variable].min()),
                "25%": float(self.data[variable].quantile(0.25)),
                "median": float(self.data[variable].median()),
                "75%": float(self.data[variable].quantile(0.75)),
                "max": float(self.data[variable].max()),
                "skewness": float(stats.skew(self.data[variable].dropna())),
                "kurtosis": float(stats.kurtosis(self.data[variable].dropna()))
            })
            
            # Check for normality
            if len(self.data[variable].dropna()) >= 8:
                try:
                    shapiro_test = stats.shapiro(self.data[variable].dropna())
                    stats["shapiro_test_statistic"] = shapiro_test[0]
                    stats["shapiro_p_value"] = shapiro_test[1]
                    stats["is_normal"] = shapiro_test[1] > 0.05
                except:
                    stats["is_normal"] = "Unknown"
            
            # Check if binary
            unique_vals = set(self.data[variable].dropna().unique())
            if len(unique_vals) == 2:
                stats["is_binary"] = True
                stats["binary_values"] = list(unique_vals)
            else:
                stats["is_binary"] = False
            
            # Check for zeros and negative values
            stats["zero_count"] = int((self.data[variable] == 0).sum())
            stats["negative_count"] = int((self.data[variable] < 0).sum())
            
        elif var_type == "categorical" or var_type == "boolean":
            # Add categorical statistics
            value_counts = self.data[variable].value_counts()
            value_percents = value_counts / value_counts.sum() * 100
            
            # Add top categories
            top_n = min(10, len(value_counts))
            stats["top_categories"] = {
                str(val): {
                    "count": int(count),
                    "percent": float(value_percents[val])
                }
                for val, count in value_counts.iloc[:top_n].items()
            }
            
            # Add entropy as a measure of dispersion
            stats["entropy"] = float(stats.entropy(value_counts))
            
            # Add mode
            stats["mode"] = str(self.data[variable].mode().iloc[0]) if not self.data[variable].mode().empty else None
            
        elif var_type == "datetime":
            # Add datetime statistics
            try:
                stats["min_date"] = str(self.data[variable].min())
                stats["max_date"] = str(self.data[variable].max())
                stats["range_days"] = (self.data[variable].max() - self.data[variable].min()).days
                
                # Add counts by year, month, day of week
                try:
                    stats["year_counts"] = {
                        str(year): int(count)
                        for year, count in self.data[variable].dt.year.value_counts().items()
                    }
                    
                    stats["month_counts"] = {
                        str(month): int(count)
                        for month, count in self.data[variable].dt.month.value_counts().items()
                    }
                    
                    stats["weekday_counts"] = {
                        str(day): int(count)
                        for day, count in self.data[variable].dt.dayofweek.value_counts().items()
                    }
                except:
                    # Some datetime operations might not be supported
                    pass
            except:
                # Handle potential errors with datetime operations
                pass
        
        return stats
    
    def calculate_bivariate_statistics(self, x_var: str, y_var: str) -> Dict[str, Any]:
        """
        Calculate statistics for a pair of variables
        
        Args:
            x_var: First variable
            y_var: Second variable
            
        Returns:
            Dictionary of statistics
        """
        # Check if variables exist
        for var in [x_var, y_var]:
            if var not in self.data.columns:
                return {"error": f"Variable '{var}' not found in data"}
        
        # Get variable types
        var_types = self.get_variable_types()
        
        x_type = "numeric"
        if x_var in var_types["categorical"]:
            x_type = "categorical"
        elif x_var in var_types["boolean"]:
            x_type = "boolean"
        
        y_type = "numeric"
        if y_var in var_types["categorical"]:
            y_type = "categorical"
        elif y_var in var_types["boolean"]:
            y_type = "boolean"
        
        # Basic statistics
        stats = {
            "x_variable": x_var,
            "y_variable": y_var,
            "x_type": x_type,
            "y_type": y_type,
            "pair_count": len(self.data[[x_var, y_var]].dropna())
        }
        
        # Calculate appropriate statistics based on variable types
        if x_type == "numeric" and y_type == "numeric":
            # Calculate correlation
            corr_pearson = self.data[[x_var, y_var]].corr().iloc[0, 1]
            corr_spearman = self.data[[x_var, y_var]].corr(method='spearman').iloc[0, 1]
            
            stats.update({
                "pearson_correlation": float(corr_pearson),
                "spearman_correlation": float(corr_spearman),
                "covariance": float(self.data[[x_var, y_var]].cov().iloc[0, 1])
            })
            
            # Linear regression
            try:
                from scipy import stats as sc_stats
                
                # Remove NaN values
                clean_data = self.data[[x_var, y_var]].dropna()
                
                # Simple linear regression
                slope, intercept, r_value, p_value, std_err = sc_stats.linregress(
                    clean_data[x_var], clean_data[y_var]
                )
                
                stats.update({
                    "regression": {
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value),
                        "std_error": float(std_err)
                    }
                })
            except Exception as e:
                logger.warning(f"Could not compute regression: {str(e)}")
            
        elif (x_type == "categorical" or x_type == "boolean") and (y_type == "categorical" or y_type == "boolean"):
            # Create contingency table
            contingency = pd.crosstab(self.data[y_var], self.data[x_var])
            
            # Chi-square test of independence
            try:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                
                stats.update({
                    "chi2_test": {
                        "chi2": float(chi2),
                        "p_value": float(p),
                        "dof": int(dof),
                        "significant": p < 0.05
                    }
                })
                
                # Cramer's V for effect size
                n = contingency.sum().sum()
                phi2 = chi2 / n
                r, k = contingency.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
                
                stats["cramers_v"] = float(cramers_v)
                
            except Exception as e:
                logger.warning(f"Could not compute chi-square test: {str(e)}")
            
        elif (x_type == "categorical" or x_type == "boolean") and y_type == "numeric":
            # ANOVA
            try:
                groups = []
                labels = []
                
                for category in self.data[x_var].dropna().unique():
                    group_data = self.data[self.data[x_var] == category][y_var].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                        labels.append(str(category))
                
                if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    stats.update({
                        "anova": {
                            "f_statistic": float(f_stat),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05
                        }
                    })
                    
                    # Calculate effect size (eta-squared)
                    # Between-group sum of squares
                    group_means = [group.mean() for group in groups]
                    grand_mean = np.mean([val for group in groups for val in group])
                    n_total = sum(len(group) for group in groups)
                    
                    ss_between = sum(len(group) * (mean - grand_mean)**2 for group, mean in zip(groups, group_means))
                    
                    # Total sum of squares
                    ss_total = sum((val - grand_mean)**2 for group in groups for val in group)
                    
                    # Eta-squared
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    stats["eta_squared"] = float(eta_squared)
                    
                    # Add group statistics
                    stats["group_statistics"] = {
                        label: {
                            "count": len(group),
                            "mean": float(group.mean()),
                            "std": float(group.std())
                        }
                        for label, group in zip(labels, groups)
                    }
            except Exception as e:
                logger.warning(f"Could not compute ANOVA: {str(e)}")
            
        elif x_type == "numeric" and (y_type == "categorical" or y_type == "boolean"):
            # Logistic regression for binary outcome
            if len(self.data[y_var].dropna().unique()) == 2:
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import roc_auc_score
                    
                    # Get clean data
                    clean_data = self.data[[x_var, y_var]].dropna()
                    
                    # Encode categorical y if needed
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(clean_data[y_var])
                    
                    # Fit logistic regression
                    model = LogisticRegression(random_state=42)
                    X = clean_data[x_var].values.reshape(-1, 1)
                    model.fit(X, y_encoded)
                    
                    # Calculate AUC
                    y_pred_proba = model.predict_proba(X)[:, 1]
                    auc = roc_auc_score(y_encoded, y_pred_proba)
                    
                    stats.update({
                        "logistic_regression": {
                            "coefficient": float(model.coef_[0][0]),
                            "intercept": float(model.intercept_[0]),
                            "auc": float(auc),
                            "classes": list(le.classes_)
                        }
                    })
                except Exception as e:
                    logger.warning(f"Could not compute logistic regression: {str(e)}")
            
            # Point-biserial correlation (treat categorical as binary numeric)
            try:
                if len(self.data[y_var].dropna().unique()) == 2:
                    # Encode y as 0/1
                    y_encoded = pd.Categorical(self.data[y_var]).codes
                    
                    # Calculate point-biserial correlation
                    correlation, p_value = stats.pointbiserialr(self.data[x_var], y_encoded)
                    
                    stats.update({
                        "point_biserial": {
                            "correlation": float(correlation),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05
                        }
                    })
            except Exception as e:
                logger.warning(f"Could not compute point-biserial correlation: {str(e)}")
        
        return stats
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create an error figure with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=error_message,
            showarrow=False,
            font=dict(color="red", size=14)
        )
        
        fig.update_layout(
            plot_bgcolor=self.plot_bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        
        return fig

# Streamlit component wrapper functions
def render_univariate_distribution(st, data, variable, plot_type="auto", bins=30, theme="light"):
    """
    Render a univariate distribution plot in Streamlit
    
    Args:
        st: Streamlit instance
        data: DataFrame
        variable: Variable to plot
        plot_type: Type of plot
        bins: Number of bins for histogram
        theme: Color theme
    """
    explorer = DistributionExplorer(data, theme=theme)
    
    # Get information about variable
    var_stats = explorer.calculate_variable_statistics(variable)
    
    # Create plot
    fig = explorer.create_univariate_plot(variable, plot_type, bins)
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics in expandable section
    if "error" not in var_stats:
        with st.expander("Variable Statistics"):
            # Basic information
            st.markdown(f"**Variable Type**: {var_stats['type'].title()}")
            st.markdown(f"**Count**: {var_stats['count']} (missing: {var_stats['missing']} / {var_stats['missing_percent']}%)")
            st.markdown(f"**Unique Values**: {var_stats['unique_values']}")
            
            # Type-specific statistics
            if var_stats["type"] == "numeric":
                # Create a summary table
                summary_df = pd.DataFrame({
                    "Statistic": ["Mean", "Standard Deviation", "Min", "25%", "Median", "75%", "Max", "Skewness", "Kurtosis"],
                    "Value": [
                        var_stats["mean"],
                        var_stats["std"],
                        var_stats["min"],
                        var_stats["25%"],
                        var_stats["median"],
                        var_stats["75%"],
                        var_stats["max"],
                        var_stats["skewness"],
                        var_stats["kurtosis"]
                    ]
                })
                
                st.table(summary_df)
                
                # Normality test
                if "is_normal" in var_stats and var_stats["is_normal"] is not True and var_stats["is_normal"] is not False:
                    st.markdown("**Normality**: Not tested (insufficient data)")
                elif "is_normal" in var_stats:
                    normality_result = "Normal" if var_stats["is_normal"] else "Non-normal"
                    normality_text = f"**Normality**: {normality_result} (Shapiro-Wilk p-value: {var_stats['shapiro_p_value']:.4f})"
                    st.markdown(normality_text)
                
                # Binary check
                if var_stats["is_binary"]:
                    st.markdown(f"**Binary Variable**: Yes (values: {var_stats['binary_values']})")
                
                # Zero and negative counts
                if var_stats["zero_count"] > 0 or var_stats["negative_count"] > 0:
                    st.markdown(f"**Zeros**: {var_stats['zero_count']} ({var_stats['zero_count'] / var_stats['count'] * 100:.1f}%)")
                    st.markdown(f"**Negative Values**: {var_stats['negative_count']} ({var_stats['negative_count'] / var_stats['count'] * 100:.1f}%)")
                
            elif var_stats["type"] == "categorical" or var_stats["type"] == "boolean":
                # Show top categories
                st.markdown("**Top Categories:**")
                
                # Create categories dataframe
                categories_df = pd.DataFrame([
                    {"Category": cat, "Count": info["count"], "Percentage": f"{info['percent']:.1f}%"}
                    for cat, info in var_stats["top_categories"].items()
                ])
                
                st.table(categories_df)
                
                # Show mode
                if "mode" in var_stats and var_stats["mode"]:
                    st.markdown(f"**Mode**: {var_stats['mode']}")
                
                # Show entropy
                if "entropy" in var_stats:
                    st.markdown(f"**Entropy**: {var_stats['entropy']:.4f}")
                
            elif var_stats["type"] == "datetime":
                # Show datetime info
                if "min_date" in var_stats and "max_date" in var_stats:
                    st.markdown(f"**Date Range**: {var_stats['min_date']} to {var_stats['max_date']}")
                    
                    if "range_days" in var_stats:
                        st.markdown(f"**Range**: {var_stats['range_days']} days")

def render_bivariate_relationship(st, data, x_var, y_var, color_var=None, plot_type="auto", theme="light"):
    """
    Render a bivariate relationship plot in Streamlit
    
    Args:
        st: Streamlit instance
        data: DataFrame
        x_var: X-axis variable
        y_var: Y-axis variable
        color_var: Variable for color encoding
        plot_type: Type of plot
        theme: Color theme
    """
    explorer = DistributionExplorer(data, theme=theme)
    
    # Create plot
    fig = explorer.create_bivariate_plot(x_var, y_var, color_var, plot_type)
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display statistics
    stats = explorer.calculate_bivariate_statistics(x_var, y_var)
    
    if "error" not in stats:
        with st.expander("Relationship Statistics"):
            # Basic information
            st.markdown(f"**Variables**: {x_var} ({stats['x_type']}) vs {y_var} ({stats['y_type']})")
            st.markdown(f"**Observations**: {stats['pair_count']}")
            
            # Type-specific statistics
            if stats['x_type'] == "numeric" and stats['y_type'] == "numeric":
                # Show correlations
                st.markdown(f"**Pearson Correlation**: {stats['pearson_correlation']:.4f}")
                st.markdown(f"**Spearman Correlation**: {stats['spearman_correlation']:.4f}")
                st.markdown(f"**Covariance**: {stats['covariance']:.4f}")
                
                # Show regression results if available
                if "regression" in stats:
                    reg = stats["regression"]
                    st.markdown("**Linear Regression:**")
                    st.markdown(f"- Equation: y = {reg['slope']:.4f}x + {reg['intercept']:.4f}")
                    st.markdown(f"- R-squared: {reg['r_squared']:.4f}")
                    st.markdown(f"- p-value: {reg['p_value']:.4g}")
                    st.markdown(f"- Standard Error: {reg['std_error']:.4f}")
            
            elif (stats['x_type'] in ["categorical", "boolean"] and 
                  stats['y_type'] in ["categorical", "boolean"]):
                # Show chi-square results
                if "chi2_test" in stats:
                    chi2 = stats["chi2_test"]
                    st.markdown("**Chi-square Test of Independence:**")
                    st.markdown(f"- Chi-square: {chi2['chi2']:.4f}")
                    st.markdown(f"- p-value: {chi2['p_value']:.4g}")
                    st.markdown(f"- Degrees of Freedom: {chi2['dof']}")
                    st.markdown(f"- Significant: {'Yes' if chi2['significant'] else 'No'}")
                
                # Show Cramer's V
                if "cramers_v" in stats:
                    st.markdown(f"**Cramer's V**: {stats['cramers_v']:.4f}")
            
            elif (stats['x_type'] in ["categorical", "boolean"] and 
                  stats['y_type'] == "numeric"):
                # Show ANOVA results
                if "anova" in stats:
                    anova = stats["anova"]
                    st.markdown("**ANOVA:**")
                    st.markdown(f"- F-statistic: {anova['f_statistic']:.4f}")
                    st.markdown(f"- p-value: {anova['p_value']:.4g}")
                    st.markdown(f"- Significant: {'Yes' if anova['significant'] else 'No'}")
                
                # Show effect size
                if "eta_squared" in stats:
                    st.markdown(f"**Eta-squared**: {stats['eta_squared']:.4f}")
                
                # Show group statistics
                if "group_statistics" in stats:
                    st.markdown("**Group Statistics:**")
                    
                    # Create group statistics dataframe
                    group_df = pd.DataFrame([
                        {"Group": group, "Count": info["count"], "Mean": info["mean"], "Std Dev": info["std"]}
                        for group, info in stats["group_statistics"].items()
                    ])
                    
                    st.table(group_df)
            
            elif (stats['x_type'] == "numeric" and 
                  stats['y_type'] in ["categorical", "boolean"]):
                # Show logistic regression for binary outcome
                if "logistic_regression" in stats:
                    logreg = stats["logistic_regression"]
                    st.markdown("**Logistic Regression:**")
                    st.markdown(f"- Coefficient: {logreg['coefficient']:.4f}")
                    st.markdown(f"- Intercept: {logreg['intercept']:.4f}")
                    st.markdown(f"- AUC: {logreg['auc']:.4f}")
                    st.markdown(f"- Classes: {logreg['classes']}")
                
                # Show point-biserial correlation
                if "point_biserial" in stats:
                    pb = stats["point_biserial"]
                    st.markdown("**Point-Biserial Correlation:**")
                    st.markdown(f"- Correlation: {pb['correlation']:.4f}")
                    st.markdown(f"- p-value: {pb['p_value']:.4g}")
                    st.markdown(f"- Significant: {'Yes' if pb['significant'] else 'No'}")

def render_correlation_matrix(st, data, methods=['pearson'], theme="light"):
    """
    Render correlation matrix in Streamlit
    
    Args:
        st: Streamlit instance
        data: DataFrame
        methods: Correlation methods to use
        theme: Color theme
    """
    explorer = DistributionExplorer(data, theme=theme)
    
    # Create correlation matrices
    corr_figs = explorer.create_correlation_matrix(methods)
    
    # Display each correlation matrix
    for method, fig in corr_figs.items():
        if method != "error":
            st.subheader(f"{method.capitalize()} Correlation")
            st.plotly_chart(fig, use_container_width=True)

def render_missing_values_heatmap(st, data, theme="light"):
    """
    Render missing values heatmap in Streamlit
    
    Args:
        st: Streamlit instance
        data: DataFrame
        theme: Color theme
    """
    explorer = DistributionExplorer(data, theme=theme)
    
    # Check if there are missing values
    if not data.isnull().values.any():
        st.info("No missing values found in the dataset.")
        return
    
    # Create missing values heatmap
    fig = explorer.create_missing_values_heatmap()
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Show summary of missing values
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data) * 100).round(2)
    
    # Create summary dataframe
    missing_df = pd.DataFrame({
        'Variable': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing Percent': missing_percent.values
    })
    
    # Filter to show only variables with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        st.subheader("Missing Values Summary")
        st.dataframe(missing_df, use_container_width=True)

def render_pairwise_scatter_matrix(st, data, variables=None, max_vars=5, color_var=None, theme="light"):
    """
    Render pairwise scatter plot matrix in Streamlit
    
    Args:
        st: Streamlit instance
        data: DataFrame
        variables: List of variables to include
        max_vars: Maximum number of variables
        color_var: Variable for color encoding
        theme: Color theme
    """
    explorer = DistributionExplorer(data, theme=theme)
    
    # Create scatter matrix
    fig = explorer.create_pairwise_scatter_matrix(variables, max_vars, color_var)
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)