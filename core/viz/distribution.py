# core/viz/distribution.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    Creates exploratory data visualizations for causal discovery.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataVisualizer
        
        Args:
            data: DataFrame to visualize
        """
        self.data = data
    
    def create_distribution_plot(self, 
                                column: str, 
                                plot_type: str = 'histogram',
                                bins: int = 30,
                                kde: bool = True) -> go.Figure:
        """
        Create a distribution plot for a numeric column
        
        Args:
            column: Column name to visualize
            plot_type: Type of plot ('histogram', 'box', 'violin')
            bins: Number of bins for histogram
            kde: Whether to overlay KDE curve
            
        Returns:
            Plotly figure
        """
        try:
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' not found in data")
            
            if not pd.api.types.is_numeric_dtype(self.data[column]):
                raise ValueError(f"Column '{column}' is not numeric")
            
            if plot_type == 'histogram':
                fig = px.histogram(
                    self.data, 
                    x=column, 
                    nbins=bins,
                    marginal="box" if kde else None,
                    title=f"Distribution of {column}"
                )
                
                if kde:
                    # Add density curve
                    from scipy.stats import gaussian_kde
                    data_no_nan = self.data[column].dropna()
                    kde_func = gaussian_kde(data_no_nan)
                    x_range = np.linspace(data_no_nan.min(), data_no_nan.max(), 200)
                    y_density = kde_func(x_range)
                    
                    # Scale KDE to match histogram
                    hist_values, bin_edges = np.histogram(data_no_nan, bins=bins)
                    hist_max = hist_values.max()
                    y_density_scaled = y_density * (hist_max / y_density.max())
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range, 
                            y=y_density_scaled, 
                            mode='lines', 
                            name='KDE',
                            line=dict(color='red')
                        )
                    )
            
            elif plot_type == 'box':
                fig = px.box(
                    self.data, 
                    y=column,
                    title=f"Box Plot of {column}"
                )
            
            elif plot_type == 'violin':
                fig = px.violin(
                    self.data, 
                    y=column,
                    box=True,
                    points="all",
                    title=f"Violin Plot of {column}"
                )
            
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Count" if plot_type == 'histogram' else column,
                legend_title="Legend"
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating plot: {str(e)}",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                ]
            )
            return fig
    
    def create_scatterplot(self, 
                          x: str, 
                          y: str, 
                          color: Optional[str] = None,
                          size: Optional[str] = None,
                          add_regression: bool = True) -> go.Figure:
        """
        Create a scatter plot between two variables
        
        Args:
            x: Column name for x-axis
            y: Column name for y-axis
            color: Optional column name for color encoding
            size: Optional column name for size encoding
            add_regression: Whether to add regression line
            
        Returns:
            Plotly figure
        """
        try:
            # Validate columns
            for col in [x, y]:
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' not found in data")
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    raise ValueError(f"Column '{col}' is not numeric")
            
            # Create scatterplot
            fig = px.scatter(
                self.data,
                x=x,
                y=y,
                color=color,
                size=size,
                title=f"Relationship between {x} and {y}",
                trendline="ols" if add_regression else None
            )
            
            # Add correlation annotation
            corr = self.data[[x, y]].corr().iloc[0, 1]
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
            
            fig.update_layout(
                xaxis_title=x,
                yaxis_title=y
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating plot: {str(e)}",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                ]
            )
            return fig
    
    def create_correlation_matrix(self, 
                               method: str = 'pearson', 
                               columns: Optional[List[str]] = None) -> go.Figure:
        """
        Create a correlation matrix heatmap
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            columns: Specific columns to include (None for all numeric)
            
        Returns:
            Plotly figure
        """
        try:
            # Get numeric columns if none specified
            if columns is None:
                columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                # Validate columns
                for col in columns:
                    if col not in self.data.columns:
                        raise ValueError(f"Column '{col}' not found in data")
                    if not pd.api.types.is_numeric_dtype(self.data[col]):
                        raise ValueError(f"Column '{col}' is not numeric")
            
            # Skip if no columns to process
            if not columns:
                raise ValueError("No numeric columns to create correlation matrix")
            
            # Calculate correlation matrix
            corr_matrix = self.data[columns].corr(method=method)
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"{method.capitalize()} Correlation Matrix"
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating correlation matrix: {str(e)}",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                ]
            )
            return fig
    
    def create_pairplot(self, 
                      columns: Optional[List[str]] = None, 
                      color: Optional[str] = None,
                      max_cols: int = 5) -> go.Figure:
        """
        Create a pairplot of multiple variables
        
        Args:
            columns: Specific columns to include (None for all numeric)
            color: Optional column name for color encoding
            max_cols: Maximum number of columns to include
            
        Returns:
            Plotly figure
        """
        try:
            # Get numeric columns if none specified
            if columns is None:
                columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Limit number of columns
            if len(columns) > max_cols:
                logger.warning(f"Limiting pairplot to {max_cols} columns")
                columns = columns[:max_cols]
            
            # Validate columns
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' not found in data")
            
            n_cols = len(columns)
            
            # Create subplots grid
            fig = make_subplots(
                rows=n_cols, cols=n_cols,
                shared_xaxes=True, shared_yaxes=True,
                subplot_titles=[f"{col_i} vs {col_j}" for col_i in columns for col_j in columns]
            )
            
            # Fill in the pairplot
            for i, col_i in enumerate(columns):
                for j, col_j in enumerate(columns):
                    row, col = i + 1, j + 1
                    
                    if i == j:
                        # Diagonal: show histogram
                        hist_data = self.data[col_i].dropna()
                        fig.add_trace(
                            go.Histogram(x=hist_data, name=col_i),
                            row=row, col=col
                        )
                    else:
                        # Off-diagonal: show scatter
                        if color:
                            # With color encoding
                            for color_val in self.data[color].unique():
                                subset = self.data[self.data[color] == color_val]
                                fig.add_trace(
                                    go.Scatter(
                                        x=subset[col_j],
                                        y=subset[col_i],
                                        mode="markers",
                                        name=f"{color}={color_val}",
                                        showlegend=False,
                                        marker=dict(
                                            opacity=0.6
                                        )
                                    ),
                                    row=row, col=col
                                )
                        else:
                            # Without color encoding
                            fig.add_trace(
                                go.Scatter(
                                    x=self.data[col_j],
                                    y=self.data[col_i],
                                    mode="markers",
                                    marker=dict(
                                        opacity=0.6
                                    ),
                                    showlegend=False
                                ),
                                row=row, col=col
                            )
            
            # Update layout
            fig.update_layout(
                title="Pair Plot Matrix",
                height=250 * n_cols,
                width=250 * n_cols,
                showlegend=False
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating pairplot: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating pairplot: {str(e)}",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                ]
            )
            return fig
    
    def create_summary_dashboard(self, 
                               max_cols: int = 5) -> Dict[str, go.Figure]:
        """
        Create a dashboard of summary visualizations
        
        Args:
            max_cols: Maximum number of columns to include in visualizations
            
        Returns:
            Dictionary of Plotly figures
        """
        try:
            # Get numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Limit number of columns
            if len(numeric_cols) > max_cols:
                logger.info(f"Limiting dashboard to {max_cols} columns")
                numeric_cols = numeric_cols[:max_cols]
            
            # Create different visualizations
            dashboard = {}
            
            # 1. Correlation matrix
            dashboard["correlation_matrix"] = self.create_correlation_matrix(
                method='pearson', 
                columns=numeric_cols
            )
            
            # 2. Distribution plots for each column
            dashboard["distributions"] = {}
            for col in numeric_cols:
                dashboard["distributions"][col] = self.create_distribution_plot(
                    column=col,
                    plot_type='histogram',
                    kde=True
                )
            
            # 3. Scatter plot matrix (for top 3 columns)
            if len(numeric_cols) >= 2:
                top_cols = numeric_cols[:min(3, len(numeric_cols))]
                pairs = [(top_cols[i], top_cols[j]) 
                         for i in range(len(top_cols)) 
                         for j in range(i+1, len(top_cols))]
                
                dashboard["scatterplots"] = {}
                for x, y in pairs:
                    dashboard["scatterplots"][f"{x}_vs_{y}"] = self.create_scatterplot(
                        x=x,
                        y=y,
                        add_regression=True
                    )
            
            return dashboard
        
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {str(e)}")
            
            # Return error figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating dashboard: {str(e)}",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False
                    )
                ]
            )
            
            return {"error": fig}