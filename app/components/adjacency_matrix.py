## app/components/adjacency_matrix.py

```python
# app/components/adjacency_matrix.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdjacencyMatrixVisualizer:
    """
    Component for visualizing causal graphs as adjacency matrices.
    Provides multiple ways to display and interact with causal relationships
    in a matrix format, with various customization options.
    """
    
    def __init__(self, graph: nx.DiGraph, data: Optional[pd.DataFrame] = None):
        """
        Initialize the adjacency matrix visualizer
        
        Args:
            graph: NetworkX DiGraph representing the causal structure
            data: Optional DataFrame containing the data (for correlation info)
        """
        self.graph = graph
        self.data = data
    
    def _get_node_name(self, node_id: Any) -> str:
        """
        Get the readable name for a node
        
        Args:
            node_id: The node ID in the graph
            
        Returns:
            The readable name for the node
        """
        # Check for name in node attributes
        if "name" in self.graph.nodes[node_id]:
            return self.graph.nodes[node_id]["name"]
        
        # If node_id is an integer index into the dataframe columns
        if isinstance(node_id, int) and self.data is not None and node_id < len(self.data.columns):
            return self.data.columns[node_id]
        
        # Default to string representation
        return str(node_id)
    
    def create_adjacency_matrix(self, 
                              use_edge_weights: bool = False,
                              edge_attribute: Optional[str] = None) -> np.ndarray:
        """
        Create an adjacency matrix from the graph
        
        Args:
            use_edge_weights: If True, use edge weights instead of binary values
            edge_attribute: Optional edge attribute to use for weights
            
        Returns:
            NumPy array containing the adjacency matrix
        """
        # Get sorted list of nodes
        nodes = sorted(self.graph.nodes())
        n_nodes = len(nodes)
        
        # Create empty matrix
        matrix = np.zeros((n_nodes, n_nodes))
        
        # Fill matrix with edge information
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if self.graph.has_edge(source, target):
                    if use_edge_weights:
                        if edge_attribute and edge_attribute in self.graph.edges[source, target]:
                            # Use specified edge attribute
                            weight = self.graph.edges[source, target][edge_attribute]
                        elif 'weight' in self.graph.edges[source, target]:
                            # Default to 'weight' attribute
                            weight = self.graph.edges[source, target]['weight']
                        else:
                            # Default to 1.0 if no weight is found
                            weight = 1.0
                        
                        matrix[i, j] = weight
                    else:
                        # Binary adjacency matrix
                        matrix[i, j] = 1.0
        
        return matrix, nodes
    
    def create_correlation_matrix(self) -> Optional[np.ndarray]:
        """
        Create a correlation matrix from the data
        
        Returns:
            NumPy array containing the correlation matrix or None if data is not available
        """
        if self.data is None:
            return None, None
        
        # Get numeric columns only
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        return corr_matrix.values, corr_matrix.columns.tolist()
    
    def visualize_adjacency_matrix(self,
                                use_edge_weights: bool = False,
                                edge_attribute: Optional[str] = None,
                                colorscale: str = "Blues",
                                title: str = "Causal Graph Adjacency Matrix") -> go.Figure:
        """
        Visualize the adjacency matrix
        
        Args:
            use_edge_weights: If True, use edge weights instead of binary values
            edge_attribute: Optional edge attribute to use for weights
            colorscale: Color scale for the heatmap
            title: Title for the visualization
            
        Returns:
            Plotly figure with the adjacency matrix visualization
        """
        # Create adjacency matrix
        adj_matrix, nodes = self.create_adjacency_matrix(use_edge_weights, edge_attribute)
        
        # Get node names
        node_names = [self._get_node_name(node) for node in nodes]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=adj_matrix,
            x=node_names,
            y=node_names,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Weight" if use_edge_weights else "Connection"),
            text=adj_matrix.round(3) if use_edge_weights else None,
            hoverinfo="text",
            hovertext=[[f"{node_names[i]} → {node_names[j]}: {adj_matrix[i, j]:.3f}" if adj_matrix[i, j] > 0 else "No edge"
                        for j in range(len(node_names))] for i in range(len(node_names))]
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Target", tickangle=45),
            yaxis=dict(title="Source", autorange="reversed"),
            width=700,
            height=700
        )
        
        return fig
    
    def visualize_correlation_matrix(self,
                                  colorscale: str = "RdBu_r",
                                  title: str = "Variable Correlation Matrix") -> Optional[go.Figure]:
        """
        Visualize the correlation matrix from the data
        
        Args:
            colorscale: Color scale for the heatmap
            title: Title for the visualization
            
        Returns:
            Plotly figure with the correlation matrix visualization or None if data is not available
        """
        # Check if data is available
        if self.data is None:
            return None
        
        # Create correlation matrix
        corr_matrix, columns = self.create_correlation_matrix()
        
        if corr_matrix is None or len(corr_matrix) == 0:
            return None
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=columns,
            y=columns,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(title="Correlation"),
            text=np.round(corr_matrix, 3),
            hoverinfo="text",
            hovertext=[[f"{columns[i]} ↔ {columns[j]}: {corr_matrix[i, j]:.3f}"
                        for j in range(len(columns))] for i in range(len(columns))]
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Variable", tickangle=45),
            yaxis=dict(title="Variable", autorange="reversed"),
            width=700,
            height=700
        )
        
        return fig
    
    def visualize_comparison(self,
                         compare_to: str = 'correlation',
                         use_edge_weights: bool = False,
                         edge_attribute: Optional[str] = None,
                         title: str = "Adjacency vs. Correlation Matrix") -> Optional[go.Figure]:
        """
        Visualize a comparison between the adjacency matrix and another matrix
        
        Args:
            compare_to: What to compare with ('correlation')
            use_edge_weights: If True, use edge weights instead of binary values
            edge_attribute: Optional edge attribute to use for weights
            title: Title for the visualization
            
        Returns:
            Plotly figure with the comparison visualization or None if comparison is not available
        """
        # Check if comparison is supported
        if compare_to == 'correlation' and self.data is None:
            return None
        
        # Create adjacency matrix
        adj_matrix, nodes = self.create_adjacency_matrix(use_edge_weights, edge_attribute)
        node_names = [self._get_node_name(node) for node in nodes]
        
        # Create comparison matrix
        if compare_to == 'correlation':
            corr_matrix, columns = self.create_correlation_matrix()
            
            if corr_matrix is None or len(corr_matrix) == 0:
                return None
            
            # Check if all graph nodes exist in correlation matrix
            if not all(name in columns for name in node_names):
                return None
            
            # Reorder correlation matrix to match adjacency matrix order
            indices = [columns.index(name) for name in node_names]
            comp_matrix = corr_matrix[indices, :][:, indices]
            comp_title = "Correlation Matrix"
        else:
            return None
        
        # Create figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Adjacency Matrix", comp_title],
            horizontal_spacing=0.1
        )
        
        # Add adjacency matrix heatmap
        fig.add_trace(
            go.Heatmap(
                z=adj_matrix,
                x=node_names,
                y=node_names,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(
                    title="Weight" if use_edge_weights else "Connection",
                    x=0.46
                ),
                hoverinfo="text",
                hovertext=[[f"{node_names[i]} → {node_names[j]}: {adj_matrix[i, j]:.3f}" if adj_matrix[i, j] > 0 else "No edge"
                            for j in range(len(node_names))] for i in range(len(node_names))]
            ),
            row=1, col=1
        )
        
        # Add comparison matrix heatmap
        fig.add_trace(
            go.Heatmap(
                z=comp_matrix,
                x=node_names,
                y=node_names,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                showscale=True,
                colorbar=dict(
                    title="Correlation",
                    x=1.02
                ),
                hoverinfo="text",
                hovertext=[[f"{node_names[i]} ↔ {node_names[j]}: {comp_matrix[i, j]:.3f}"
                            for j in range(len(node_names))] for i in range(len(node_names))]
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=700,
            width=1200
        )
        
        # Update axes
        fig.update_xaxes(title="Target", tickangle=45, row=1, col=1)
        fig.update_yaxes(title="Source", autorange="reversed", row=1, col=1)
        fig.update_xaxes(title="Variable", tickangle=45, row=1, col=2)
        fig.update_yaxes(title="Variable", autorange="reversed", row=1, col=2)
        
        return fig
    
    def visualize_edge_metrics(self, 
                            metrics: List[str] = ['weight', 'confidence'],
                            title: str = "Edge Metrics Comparison") -> Optional[go.Figure]:
        """
        Visualize multiple edge metric matrices side by side
        
        Args:
            metrics: List of edge attributes to visualize
            title: Title for the visualization
            
        Returns:
            Plotly figure with the edge metrics visualization or None if no metrics are found
        """
        # Get sorted list of nodes
        nodes = sorted(self.graph.nodes())
        n_nodes = len(nodes)
        node_names = [self._get_node_name(node) for node in nodes]
        
        # Check if metrics exist in the graph
        available_metrics = set()
        for _, _, data in self.graph.edges(data=True):
            available_metrics.update(data.keys())
        
        metrics = [m for m in metrics if m in available_metrics]
        
        if not metrics:
            return None
        
        # Create figure with subplots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=1, cols=n_metrics,
            subplot_titles=[f"{metric.capitalize()} Matrix" for metric in metrics],
            horizontal_spacing=0.1 / n_metrics
        )
        
        # Create and add each metric matrix
        for i, metric in enumerate(metrics):
            # Create matrix for this metric
            metric_matrix = np.zeros((n_nodes, n_nodes))
            
            # Fill matrix with edge information
            for source_idx, source in enumerate(nodes):
                for target_idx, target in enumerate(nodes):
                    if self.graph.has_edge(source, target) and metric in self.graph.edges[source, target]:
                        metric_matrix[source_idx, target_idx] = self.graph.edges[source, target][metric]
            
            # Determine color scale based on metric type
            if metric in ['weight', 'confidence']:
                colorscale = "Blues"
                zmin = 0
                zmax = 1
            else:
                colorscale = "Viridis"
                zmin = None
                zmax = None
            
            # Add heatmap for this metric
            fig.add_trace(
                go.Heatmap(
                    z=metric_matrix,
                    x=node_names,
                    y=node_names,
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    showscale=True,
                    colorbar=dict(
                        title=metric.capitalize(),
                        x=(i + 1) / n_metrics
                    ),
                    hoverinfo="text",
                    hovertext=[[f"{node_names[i]} → {node_names[j]}: {metric_matrix[i, j]:.3f}" if metric_matrix[i, j] > 0 else "No edge"
                                for j in range(len(node_names))] for i in range(len(node_names))]
                ),
                row=1, col=i+1
            )
            
            # Update axes
            fig.update_xaxes(title="Target", tickangle=45, row=1, col=i+1)
            fig.update_yaxes(title="Source" if i == 0 else None, autorange="reversed", row=1, col=i+1)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=700,
            width=300 * n_metrics + 100
        )
        
        return fig
    
    def visualize_difference(self,
                          other_graph: nx.DiGraph,
                          use_edge_weights: bool = False,
                          title: str = "Graph Difference Matrix") -> go.Figure:
        """
        Visualize the difference between this graph and another graph
        
        Args:
            other_graph: Another NetworkX DiGraph to compare with
            use_edge_weights: If True, use edge weights for comparison
            title: Title for the visualization
            
        Returns:
            Plotly figure with the difference visualization
        """
        # Create adjacency matrices for both graphs
        adj_matrix1, nodes1 = self.create_adjacency_matrix(use_edge_weights)
        
        # Create temporary visualizer for other graph
        other_viz = AdjacencyMatrixVisualizer(other_graph, self.data)
        adj_matrix2, nodes2 = other_viz.create_adjacency_matrix(use_edge_weights)
        
        # Check if the nodes are the same
        if nodes1 != nodes2:
            # Create a unified list of nodes
            unified_nodes = sorted(set(nodes1).union(set(nodes2)))
            n_nodes = len(unified_nodes)
            
            # Create new matrices with unified node set
            new_adj1 = np.zeros((n_nodes, n_nodes))
            new_adj2 = np.zeros((n_nodes, n_nodes))
            
            # Map from unified nodes to original indices
            for i, node_i in enumerate(unified_nodes):
                for j, node_j in enumerate(unified_nodes):
                    if node_i in nodes1 and node_j in nodes1:
                        idx1_i = nodes1.index(node_i)
                        idx1_j = nodes1.index(node_j)
                        new_adj1[i, j] = adj_matrix1[idx1_i, idx1_j]
                    
                    if node_i in nodes2 and node_j in nodes2:
                        idx2_i = nodes2.index(node_i)
                        idx2_j = nodes2.index(node_j)
                        new_adj2[i, j] = adj_matrix2[idx2_i, idx2_j]
            
            adj_matrix1 = new_adj1
            adj_matrix2 = new_adj2
            nodes = unified_nodes
        else:
            nodes = nodes1
        
        # Calculate difference matrix
        diff_matrix = adj_matrix1 - adj_matrix2
        
        # Get node names
        node_names = [self._get_node_name(node) for node in nodes]
        
        # Create difference matrix visualization
        fig = go.Figure(data=go.Heatmap(
            z=diff_matrix,
            x=node_names,
            y=node_names,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title="Difference",
                tickvals=[-1, 0, 1],
                ticktext=["Only in Graph 2", "No Difference", "Only in Graph 1"]
            ),
            hoverinfo="text",
            hovertext=[[self._get_difference_text(node_names[i], node_names[j], diff_matrix[i, j])
                        for j in range(len(node_names))] for i in range(len(node_names))]
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Target", tickangle=45),
            yaxis=dict(title="Source", autorange="reversed"),
            width=700,
            height=700
        )
        
        return fig
    
    def _get_difference_text(self, source: str, target: str, diff_value: float) -> str:
        """Generate hover text for difference matrix"""
        if abs(diff_value) < 1e-6:  # Close to zero
            return f"{source} → {target}: No difference"
        elif diff_value > 0:
            return f"{source} → {target}: Only in Graph 1 ({diff_value:.3f})"
        else:
            return f"{source} → {target}: Only in Graph 2 ({-diff_value:.3f})"


# Helper functions for use in the Streamlit UI
def render_adjacency_matrix_ui(graph: nx.DiGraph, 
                            data: Optional[pd.DataFrame] = None,
                            comparison_graph: Optional[nx.DiGraph] = None,
                            key_prefix: str = ""):
    """
    Render the UI for the adjacency matrix component
    
    Args:
        graph: The causal graph
        data: Optional dataset for correlation analysis
        comparison_graph: Optional second graph for comparison
        key_prefix: Prefix for Streamlit widget keys
    """
    st.subheader("Adjacency Matrix Visualization")
    
    # Create the visualizer
    viz = AdjacencyMatrixVisualizer(graph, data)
    
    # Visualization type
    viz_type = st.selectbox(
        "Visualization Type",
        ["Adjacency Matrix", "Edge Metrics", "Correlation Comparison", "Graph Difference"],
        key=f"{key_prefix}viz_type"
    )
    
    if viz_type == "Adjacency Matrix":
        # Basic options
        col1, col2 = st.columns(2)
        
        with col1:
            use_weights = st.checkbox(
                "Use Edge Weights",
                value=True,
                key=f"{key_prefix}use_weights"
            )
        
        with col2:
            if use_weights:
                # Get available edge attributes
                edge_attrs = ["weight"]
                if graph.edges:
                    sample_edge = next(iter(graph.edges(data=True)))
                    edge_attrs.extend([attr for attr in sample_edge[2].keys() 
                                     if attr != "weight" and not attr.startswith("_")])
                
                edge_attr = st.selectbox(
                    "Edge Attribute",
                    edge_attrs,
                    key=f"{key_prefix}edge_attr"
                )
            else:
                edge_attr = None
        
        # Create and display the adjacency matrix
        fig = viz.visualize_adjacency_matrix(
            use_edge_weights=use_weights,
            edge_attribute=edge_attr
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Edge Metrics":
        # Get available edge attributes
        edge_attrs = []
        if graph.edges:
            sample_edge = next(iter(graph.edges(data=True)))
            edge_attrs.extend([attr for attr in sample_edge[2].keys() 
                             if not attr.startswith("_")])
        
        if not edge_attrs:
            st.warning("No edge attributes found in the graph")
            return
        
        # Select metrics to visualize
        metrics = st.multiselect(
            "Edge Metrics to Visualize",
            edge_attrs,
            default=["weight"] if "weight" in edge_attrs else edge_attrs[:1],
            key=f"{key_prefix}metrics"
        )
        
        if not metrics:
            st.warning("Please select at least one metric to visualize")
            return
        
        # Create and display edge metrics visualization
        fig = viz.visualize_edge_metrics(metrics=metrics)
        
        if fig is None:
            st.warning("Selected metrics not available in the graph")
            return
            
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Comparison":
        if data is None:
            st.warning("Dataset is required for correlation comparison")
            return
        
        # Options
        use_weights = st.checkbox(
            "Use Edge Weights",
            value=True,
            key=f"{key_prefix}use_weights_corr"
        )
        
        # Create and display comparison visualization
        fig = viz.visualize_comparison(
            compare_to='correlation',
            use_edge_weights=use_weights
        )
        
        if fig is None:
            st.warning("Could not create comparison visualization")
            return
            
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("Understanding the Comparison"):
            st.markdown("""
            This comparison shows:
            
            **Left:** The adjacency matrix of the causal graph, where entries indicate direct causal relationships.
            
            **Right:** The correlation matrix of the variables, where entries indicate statistical associations.
            
            **Key insights:**
            - Variables that are causally related typically show correlation
            - However, correlation can exist without causation (e.g., due to confounding)
            - The difference between these matrices can reveal potential hidden confounders
            """)
    
    elif viz_type == "Graph Difference":
        if comparison_graph is None:
            st.warning("Please provide a second graph for comparison")
            return
        
        # Options
        use_weights = st.checkbox(
            "Use Edge Weights",
            value=False,
            key=f"{key_prefix}use_weights_diff"
        )
        
        # Create and display difference visualization
        fig = viz.visualize_difference(
            other_graph=comparison_graph,
            use_edge_weights=use_weights
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("Understanding the Difference Matrix"):
            st.markdown("""
            The difference matrix highlights:
            
            - **Blue cells:** Edges present only in the first graph
            - **Red cells:** Edges present only in the second graph
            - **White cells:** No difference between the graphs
            
            This is useful for comparing:
            - Before/after refinement
            - Different algorithm results
            - Predicted vs. true causal structure
            """)


def create_adjacency_matrix_figure(graph: nx.DiGraph,
                               use_edge_weights: bool = False,
                               edge_attribute: Optional[str] = None) -> go.Figure:
    """
    Create an adjacency matrix visualization for a graph
    
    Args:
        graph: The causal graph
        use_edge_weights: Whether to use edge weights
        edge_attribute: Optional edge attribute to use
        
    Returns:
        Plotly figure with the adjacency matrix
    """
    viz = AdjacencyMatrixVisualizer(graph)
    return viz.visualize_adjacency_matrix(
        use_edge_weights=use_edge_weights,
        edge_attribute=edge_attribute
    )


def compare_adjacency_matrices(graph1: nx.DiGraph, 
                            graph2: nx.DiGraph,
                            use_edge_weights: bool = False) -> go.Figure:
    """
    Compare two graphs using difference matrix
    
    Args:
        graph1: First causal graph
        graph2: Second causal graph
        use_edge_weights: Whether to use edge weights
        
    Returns:
        Plotly figure with the difference matrix
    """
    viz = AdjacencyMatrixVisualizer(graph1)
    return viz.visualize_difference(
        other_graph=graph2,
        use_edge_weights=use_edge_weights
    )
```