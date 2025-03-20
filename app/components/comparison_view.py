# app/components/comparison_view.py
"""
Comparison view for causal graphs to visualize differences between multiple graphs.
Allows side-by-side comparison, overlay view, and difference highlighting.
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

class ComparisonView:
    """
    Component for comparing multiple causal graphs visually.
    Provides multiple visualization modes and difference analysis.
    """
    
    def __init__(self, theme: str = "light"):
        """
        Initialize the comparison view component
        
        Args:
            theme: Visual theme ('light' or 'dark')
        """
        self.theme = theme
        
        # Set theme colors
        if theme == "dark":
            self.bg_color = "#1e1e1e"
            self.text_color = "#ffffff"
            self.added_color = "#4caf50"  # Green
            self.removed_color = "#f44336"  # Red
            self.unchanged_color = "#2196f3"  # Blue
            self.changed_color = "#ff9800"  # Orange
        else:
            self.bg_color = "#ffffff"
            self.text_color = "#000000"
            self.added_color = "#4caf50"  # Green
            self.removed_color = "#f44336"  # Red
            self.unchanged_color = "#2196f3"  # Blue
            self.changed_color = "#ff9800"  # Orange
    
    def render_side_by_side(self, 
                          graphs: List[nx.DiGraph], 
                          graph_names: List[str],
                          node_labels: Optional[Dict[Any, str]] = None) -> go.Figure:
        """
        Render multiple graphs side by side for comparison
        
        Args:
            graphs: List of NetworkX DiGraphs to compare
            graph_names: Names for each graph
            node_labels: Optional dictionary mapping node IDs to display labels
            
        Returns:
            Plotly figure with side-by-side comparison
        """
        n_graphs = len(graphs)
        
        # Validate inputs
        if n_graphs < 2:
            raise ValueError("At least two graphs are required for comparison")
        if len(graph_names) != n_graphs:
            raise ValueError("Number of graph names must match number of graphs")
        
        # Create a union of all graphs to get consistent layout
        union_graph = nx.DiGraph()
        for graph in graphs:
            union_graph.add_nodes_from(graph.nodes(data=True))
            union_graph.add_edges_from(graph.edges(data=True))
        
        # Create a layout for the union graph
        pos = nx.spring_layout(union_graph, seed=42)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=n_graphs,
            subplot_titles=graph_names,
            horizontal_spacing=0.05
        )
        
        # Add each graph to its subplot
        for i, graph in enumerate(graphs):
            col = i + 1
            
            # Add edges
            for u, v, data in graph.edges(data=True):
                if u in pos and v in pos:  # Check that nodes exist in layout
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    
                    # Get edge weight if available
                    weight = data.get('weight', 1.0)
                    width = 1.0 + (2.0 * min(weight, 2.0))
                    
                    # Add edge
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(width=width, color=self.unchanged_color),
                            showlegend=False,
                            hoverinfo='text',
                            text=f"{u} → {v}" if not node_labels else f"{node_labels.get(u, u)} → {node_labels.get(v, v)}"
                        ),
                        row=1, col=col
                    )
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            
            for node in graph.nodes():
                if node in pos:  # Check that node exists in layout
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Use label if provided
                    if node_labels and node in node_labels:
                        node_text.append(node_labels[node])
                    else:
                        node_text.append(str(node))
            
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=self.unchanged_color,
                        line=dict(width=1, color=self.text_color)
                    ),
                    text=node_text,
                    textposition="top center",
                    hoverinfo='text',
                    showlegend=False
                ),
                row=1, col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Graph Comparison",
            height=600,
            width=max(250 * n_graphs, 500),
            showlegend=False,
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        # Remove axis labels
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        return fig
    
    def render_difference_view(self, 
                             base_graph: nx.DiGraph, 
                             comparison_graph: nx.DiGraph,
                             base_name: str = "Base Graph",
                             comparison_name: str = "Comparison Graph",
                             node_labels: Optional[Dict[Any, str]] = None) -> go.Figure:
        """
        Render a view highlighting differences between two graphs
        
        Args:
            base_graph: Base graph for comparison
            comparison_graph: Graph to compare against the base
            base_name: Name for the base graph
            comparison_name: Name for the comparison graph
            node_labels: Optional dictionary mapping node IDs to display labels
            
        Returns:
            Plotly figure with difference visualization
        """
        # Create a union graph for layout
        union_graph = nx.DiGraph()
        union_graph.add_nodes_from(base_graph.nodes(data=True))
        union_graph.add_nodes_from(comparison_graph.nodes(data=True))
        union_graph.add_edges_from(base_graph.edges(data=True))
        union_graph.add_edges_from(comparison_graph.edges(data=True))
        
        # Create a layout for the union graph
        pos = nx.spring_layout(union_graph, seed=42)
        
        # Identify differences
        base_edges = set(base_graph.edges())
        comparison_edges = set(comparison_graph.edges())
        
        # Categorize edges
        added_edges = comparison_edges - base_edges
        removed_edges = base_edges - comparison_edges
        common_edges = base_edges.intersection(comparison_edges)
        
        # Create figure
        fig = go.Figure()
        
        # Add common edges
        for u, v in common_edges:
            if u in pos and v in pos:  # Check that nodes exist in layout
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Get edge data
                base_data = dict(base_graph.get_edge_data(u, v))
                comp_data = dict(comparison_graph.get_edge_data(u, v))
                
                # Check if edge attributes differ
                is_changed = False
                for key in set(base_data.keys()).union(comp_data.keys()):
                    if key not in base_data or key not in comp_data:
                        is_changed = True
                        break
                    if base_data[key] != comp_data[key]:
                        is_changed = True
                        break
                
                # Add edge with appropriate color
                color = self.changed_color if is_changed else self.unchanged_color
                
                # Get edge text
                edge_text = f"{u} → {v}" if not node_labels else f"{node_labels.get(u, u)} → {node_labels.get(v, v)}"
                if is_changed:
                    edge_text += " (attributes changed)"
                
                # Add edge
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color=color),
                        name='Changed' if is_changed else 'Unchanged',
                        hoverinfo='text',
                        text=edge_text
                    )
                )
        
        # Add added edges
        for u, v in added_edges:
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Get edge text
                edge_text = f"{u} → {v}" if not node_labels else f"{node_labels.get(u, u)} → {node_labels.get(v, v)}"
                edge_text += " (added)"
                
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color=self.added_color),
                        name='Added',
                        hoverinfo='text',
                        text=edge_text
                    )
                )
        
        # Add removed edges
        for u, v in removed_edges:
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Get edge text
                edge_text = f"{u} → {v}" if not node_labels else f"{node_labels.get(u, u)} → {node_labels.get(v, v)}"
                edge_text += " (removed)"
                
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color=self.removed_color, dash='dash'),
                        name='Removed',
                        hoverinfo='text',
                        text=edge_text
                    )
                )
        
        # Add nodes
        for node in union_graph.nodes():
            if node in pos:
                x, y = pos[node]
                
                # Determine node status
                in_base = node in base_graph
                in_comparison = node in comparison_graph
                
                if in_base and in_comparison:
                    color = self.unchanged_color
                    node_name = 'Unchanged'
                elif in_base:
                    color = self.removed_color
                    node_name = 'Removed'
                else:
                    color = self.added_color
                    node_name = 'Added'
                
                # Get node text
                if node_labels and node in node_labels:
                    node_text = node_labels[node]
                else:
                    node_text = str(node)
                
                if not (in_base and in_comparison):
                    node_text += f" ({node_name.lower()})"
                
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=color,
                            line=dict(width=1, color=self.text_color)
                        ),
                        text=node_text,
                        textposition="top center",
                        name=node_name,
                        hoverinfo='text',
                        showlegend=False
                    )
                )
        
        # Create a categorical legend trace
        for color, name in [
            (self.unchanged_color, "Unchanged"),
            (self.changed_color, "Changed"),
            (self.added_color, "Added"),
            (self.removed_color, "Removed")
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=name,
                    showlegend=True
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Differences: {base_name} vs {comparison_name}",
            height=600,
            width=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        # Remove axis labels
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        return fig
    
    def render_overlay_view(self,
                          base_graph: nx.DiGraph,
                          comparison_graph: nx.DiGraph,
                          base_name: str = "Base Graph",
                          comparison_name: str = "Comparison Graph",
                          node_labels: Optional[Dict[Any, str]] = None) -> go.Figure:
        """
        Render an overlay view of two graphs
        
        Args:
            base_graph: Base graph for comparison
            comparison_graph: Graph to compare against the base
            base_name: Name for the base graph
            comparison_name: Name for the comparison graph
            node_labels: Optional dictionary mapping node IDs to display labels
            
        Returns:
            Plotly figure with overlay visualization
        """
        # Create a union graph for layout
        union_graph = nx.DiGraph()
        union_graph.add_nodes_from(base_graph.nodes(data=True))
        union_graph.add_nodes_from(comparison_graph.nodes(data=True))
        union_graph.add_edges_from(base_graph.edges(data=True))
        union_graph.add_edges_from(comparison_graph.edges(data=True))
        
        # Create a layout for the union graph
        pos = nx.spring_layout(union_graph, seed=42)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges from base graph
        for u, v, data in base_graph.edges(data=True):
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Get edge weight if available
                weight = data.get('weight', 1.0)
                width = 1.0 + (2.0 * min(weight, 2.0))
                
                # Add edge
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=width, color=self.unchanged_color),
                        name=base_name,
                        hoverinfo='text',
                        text=f"{u} → {v}" if not node_labels else f"{node_labels.get(u, u)} → {node_labels.get(v, v)}"
                    )
                )
        
        # Add edges from comparison graph with offset
        for u, v, data in comparison_graph.edges(data=True):
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Add a small offset to make edges distinguishable
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                offset_x = -dy * 0.05  # Create perpendicular offset
                offset_y = dx * 0.05
                
                # Create curved path
                edge_x = [x0, 
                         (x0 + mid_x) / 2 + offset_x, 
                         mid_x + offset_x, 
                         (mid_x + x1) / 2 + offset_x, 
                         x1, None]
                edge_y = [y0, 
                         (y0 + mid_y) / 2 + offset_y, 
                         mid_y + offset_y, 
                         (mid_y + y1) / 2 + offset_y, 
                         y1, None]
                
                # Get edge weight if available
                weight = data.get('weight', 1.0)
                width = 1.0 + (2.0 * min(weight, 2.0))
                
                # Add edge
                fig.add_trace(
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode='lines',
                        line=dict(width=width, color=self.changed_color, dash='dot'),
                        name=comparison_name,
                        hoverinfo='text',
                        text=f"{u} → {v}" if not node_labels else f"{node_labels.get(u, u)} → {node_labels.get(v, v)}"
                    )
                )
        
        # Add nodes (show all nodes from union graph)
        for node in union_graph.nodes():
            if node in pos:
                x, y = pos[node]
                
                # Determine node status
                in_base = node in base_graph
                in_comparison = node in comparison_graph
                
                if in_base and in_comparison:
                    color = self.unchanged_color
                    marker_line_color = self.unchanged_color
                    marker_line_width = 2
                elif in_base:
                    color = self.unchanged_color
                    marker_line_color = self.text_color
                    marker_line_width = 1
                else:
                    color = self.changed_color
                    marker_line_color = self.text_color
                    marker_line_width = 1
                
                # Get node text
                if node_labels and node in node_labels:
                    node_text = node_labels[node]
                else:
                    node_text = str(node)
                
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=color,
                            line=dict(width=marker_line_width, color=marker_line_color)
                        ),
                        text=node_text,
                        textposition="top center",
                        hoverinfo='text',
                        showlegend=False
                    )
                )
        
        # Create a legend just for the edge types
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(width=2, color=self.unchanged_color),
                name=base_name,
                showlegend=True
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(width=2, color=self.changed_color, dash='dot'),
                name=comparison_name,
                showlegend=True
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Overlay: {base_name} & {comparison_name}",
            height=600,
            width=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor=self.bg_color,
            paper_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        # Remove axis labels
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        return fig
    
    def create_difference_table(self,
                              base_graph: nx.DiGraph,
                              comparison_graph: nx.DiGraph,
                              base_name: str = "Base Graph",
                              comparison_name: str = "Comparison Graph",
                              node_labels: Optional[Dict[Any, str]] = None) -> pd.DataFrame:
        """
        Create a DataFrame summarizing the differences between two graphs
        
        Args:
            base_graph: Base graph for comparison
            comparison_graph: Graph to compare against the base
            base_name: Name for the base graph
            comparison_name: Name for the comparison graph
            node_labels: Optional dictionary mapping node IDs to display labels
            
        Returns:
            DataFrame with difference summary
        """
        # Identify differences
        base_edges = set(base_graph.edges())
        comparison_edges = set(comparison_graph.edges())
        
        base_nodes = set(base_graph.nodes())
        comparison_nodes = set(comparison_graph.nodes())
        
        # Categorize differences
        added_edges = comparison_edges - base_edges
        removed_edges = base_edges - comparison_edges
        common_edges = base_edges.intersection(comparison_edges)
        
        added_nodes = comparison_nodes - base_nodes
        removed_nodes = base_nodes - comparison_nodes
        
        # Create list for DataFrame
        rows = []
        
        # Format node labels if provided
        def format_node(node):
            if node_labels and node in node_labels:
                return f"{node_labels[node]} ({node})"
            return str(node)
        
        # Add common edges with changed attributes
        for u, v in common_edges:
            base_data = dict(base_graph.get_edge_data(u, v))
            comp_data = dict(comparison_graph.get_edge_data(u, v))
            
            # Check if edge attributes differ
            changed_attrs = {}
            for key in set(base_data.keys()).union(comp_data.keys()):
                if key not in base_data:
                    changed_attrs[key] = (None, comp_data[key])
                elif key not in comp_data:
                    changed_attrs[key] = (base_data[key], None)
                elif base_data[key] != comp_data[key]:
                    changed_attrs[key] = (base_data[key], comp_data[key])
            
            if changed_attrs:
                rows.append({
                    "Type": "Edge Attribute",
                    "Element": f"{format_node(u)} → {format_node(v)}",
                    "Change": "Changed",
                    "Details": ", ".join([f"{k}: {v[0]} → {v[1]}" for k, v in changed_attrs.items()])
                })
        
        # Add added edges
        for u, v in added_edges:
            rows.append({
                "Type": "Edge",
                "Element": f"{format_node(u)} → {format_node(v)}",
                "Change": "Added",
                "Details": f"Edge added in {comparison_name}"
            })
        
        # Add removed edges
        for u, v in removed_edges:
            rows.append({
                "Type": "Edge",
                "Element": f"{format_node(u)} → {format_node(v)}",
                "Change": "Removed",
                "Details": f"Edge removed in {comparison_name}"
            })
        
        # Add node changes
        for node in added_nodes:
            rows.append({
                "Type": "Node",
                "Element": format_node(node),
                "Change": "Added",
                "Details": f"Node added in {comparison_name}"
            })
        
        for node in removed_nodes:
            rows.append({
                "Type": "Node",
                "Element": format_node(node),
                "Change": "Removed",
                "Details": f"Node removed in {comparison_name}"
            })
        
        # For nodes in both graphs, check for attribute changes
        common_nodes = base_nodes.intersection(comparison_nodes)
        for node in common_nodes:
            base_attrs = dict(base_graph.nodes[node])
            comp_attrs = dict(comparison_graph.nodes[node])
            
            changed_attrs = {}
            for key in set(base_attrs.keys()).union(comp_attrs.keys()):
                if key not in base_attrs:
                    changed_attrs[key] = (None, comp_attrs[key])
                elif key not in comp_attrs:
                    changed_attrs[key] = (base_attrs[key], None)
                elif base_attrs[key] != comp_attrs[key]:
                    changed_attrs[key] = (base_attrs[key], comp_attrs[key])
            
            if changed_attrs:
                rows.append({
                    "Type": "Node Attribute",
                    "Element": format_node(node),
                    "Change": "Changed",
                    "Details": ", ".join([f"{k}: {v[0]} → {v[1]}" for k, v in changed_attrs.items()])
                })
        
        # Create DataFrame
        return pd.DataFrame(rows)
    
    def compute_graph_stats(self, graphs: List[nx.DiGraph], graph_names: List[str]) -> pd.DataFrame:
        """
        Compute various graph statistics for comparison
        
        Args:
            graphs: List of NetworkX DiGraphs to compare
            graph_names: Names for each graph
            
        Returns:
            DataFrame with graph statistics
        """
        stats_rows = []
        
        for i, (graph, name) in enumerate(zip(graphs, graph_names)):
            # Basic statistics
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            density = nx.density(graph)
            
            # Degree statistics
            in_degrees = [d for _, d in graph.in_degree()]
            out_degrees = [d for _, d in graph.out_degree()]
            total_degrees = [d for _, d in graph.degree()]
            
            avg_in_degree = np.mean(in_degrees) if in_degrees else 0
            avg_out_degree = np.mean(out_degrees) if out_degrees else 0
            avg_degree = np.mean(total_degrees) if total_degrees else 0
            
            max_in_degree = max(in_degrees) if in_degrees else 0
            max_out_degree = max(out_degrees) if out_degrees else 0
            
            # Structural properties
            try:
                is_dag = nx.is_directed_acyclic_graph(graph)
            except:
                is_dag = False
            
            try:
                n_connected_components = nx.number_weakly_connected_components(graph)
            except:
                n_connected_components = 0
            
            # Check for self-loops and parallel edges
            n_self_loops = nx.number_of_selfloops(graph)
            
            # Compile statistics
            stats_rows.append({
                "Graph": name,
                "Nodes": n_nodes,
                "Edges": n_edges,
                "Density": density,
                "Avg In-Degree": avg_in_degree,
                "Avg Out-Degree": avg_out_degree,
                "Avg Degree": avg_degree,
                "Max In-Degree": max_in_degree,
                "Max Out-Degree": max_out_degree,
                "Is DAG": is_dag,
                "Connected Components": n_connected_components,
                "Self-Loops": n_self_loops
            })
        
        return pd.DataFrame(stats_rows)

# Streamlit component wrapper
def render_comparison_view(st, graphs, graph_names, view_type="side_by_side", node_labels=None, theme="light"):
    """
    Render comparison view in Streamlit
    
    Args:
        st: Streamlit instance
        graphs: List of graphs to compare
        graph_names: Names for each graph
        view_type: Type of view ('side_by_side', 'difference', 'overlay')
        node_labels: Optional dictionary mapping node IDs to display labels
        theme: Color theme ('light' or 'dark')
    """
    comparison_view = ComparisonView(theme=theme)
    
    if len(graphs) < 2:
        st.error("At least two graphs are required for comparison")
        return
    
    if view_type == "side_by_side":
        fig = comparison_view.render_side_by_side(graphs, graph_names, node_labels)
        st.plotly_chart(fig, use_container_width=True)
        
    elif view_type == "difference":
        if len(graphs) != 2:
            st.warning("Difference view requires exactly two graphs. Using the first two.")
        
        base_graph = graphs[0]
        comparison_graph = graphs[1]
        base_name = graph_names[0]
        comparison_name = graph_names[1]
        
        fig = comparison_view.render_difference_view(
            base_graph, comparison_graph, 
            base_name, comparison_name, 
            node_labels
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show difference table
        st.subheader("Detailed Differences")
        diff_df = comparison_view.create_difference_table(
            base_graph, comparison_graph,
            base_name, comparison_name,
            node_labels
        )
        
        if len(diff_df) > 0:
            st.dataframe(diff_df, use_container_width=True)
        else:
            st.info("No differences found between the graphs.")
        
    elif view_type == "overlay":
        if len(graphs) != 2:
            st.warning("Overlay view requires exactly two graphs. Using the first two.")
        
        base_graph = graphs[0]
        comparison_graph = graphs[1]
        base_name = graph_names[0]
        comparison_name = graph_names[1]
        
        fig = comparison_view.render_overlay_view(
            base_graph, comparison_graph, 
            base_name, comparison_name, 
            node_labels
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Show graph statistics
    st.subheader("Graph Statistics")
    stats_df = comparison_view.compute_graph_stats(graphs, graph_names)
    st.dataframe(stats_df, use_container_width=True)
