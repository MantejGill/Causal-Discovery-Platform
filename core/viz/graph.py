# core/viz/graph.py
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CausalGraphVisualizer:
    """
    Visualizes causal graphs from causal discovery algorithms.
    """
    
    def __init__(self):
        """Initialize CausalGraphVisualizer"""
        pass
    
    # Fix for core/viz/graph.py, in the visualize_graph method

    def visualize_graph(self, 
                    graph: nx.DiGraph, 
                    node_labels: Optional[Dict[int, str]] = None,
                    edge_weights: bool = True,
                    layout_type: str = 'spring',
                    show_bidirected: bool = True,
                    show_confidence: bool = True) -> go.Figure:
        """
        Visualize a causal graph using Plotly
        """
        try:
            # Create a copy of the graph for visualization
            G = graph.copy()
            
            # Get node positions
            if layout_type == 'spring':
                pos = nx.spring_layout(G, seed=42)
            elif layout_type == 'circular':
                pos = nx.circular_layout(G)
            elif layout_type == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            elif layout_type == 'planar':
                # Try planar layout, fall back to spring layout if not planar
                try:
                    pos = nx.planar_layout(G)
                except nx.NetworkXException:
                    pos = nx.spring_layout(G, seed=42)
            else:
                raise ValueError(f"Unknown layout type: {layout_type}")
            
            # Create figure
            fig = go.Figure()
            
            # Process normal edges
            normal_edges = []
            for u, v, data in G.edges(data=True):
                # Skip bidirected edges (we'll add them separately)
                if show_bidirected and 'bidirected' in data and data['bidirected']:
                    continue
                
                normal_edges.append((u, v, data))
            
            if normal_edges:
                # Create individual traces for each edge
                for u, v, data in normal_edges:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    
                    # Edge text
                    text = f"{u} → {v}"
                    
                    # Add weight if available
                    if edge_weights and 'weight' in data:
                        text += f"<br>Weight: {data['weight']:.3f}"
                    
                    # Determine color and width based on confidence
                    if show_confidence and 'confidence' in data:
                        text += f"<br>Confidence: {data['confidence']:.3f}"
                        width = 1 + 3 * data['confidence']
                        
                        # Color based on confidence
                        confidence = data['confidence']
                        color = f"rgba({int(255 * (1 - confidence))}, {int(255 * confidence)}, 0, 0.8)"
                    else:
                        width = 1.5
                        color = "rgba(0, 0, 200, 0.5)"
                    
                    # Create edge trace for this edge
                    edge_trace = go.Scatter(
                        x=[x0, x1, None], 
                        y=[y0, y1, None],
                        line=dict(width=width, color=color),
                        hoverinfo='text',
                        text=text,
                        mode='lines',
                        name='Edge',
                        showlegend=False
                    )
                    
                    fig.add_trace(edge_trace)
                
                # Add a dummy trace just for legend
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    line=dict(width=1.5, color="rgba(0, 0, 200, 0.5)"),
                    mode='lines',
                    name='Directed Edges'
                ))
            
            # Process bidirected edges if enabled
            if show_bidirected:
                bidirected_edges = []
                for u, v, data in G.edges(data=True):
                    if 'bidirected' in data and data['bidirected']:
                        # Only add once (since bidirected edges appear twice in DiGraph)
                        if (v, u) not in [(e[0], e[1]) for e in bidirected_edges]:
                            bidirected_edges.append((u, v, data))
                
                if bidirected_edges:
                    # Create individual traces for each bidirected edge
                    for u, v, data in bidirected_edges:
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        
                        # Create curved line for bidirected edges
                        # Calculate midpoint and offset for curve
                        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                        dx, dy = x1 - x0, y1 - y0
                        offset_x = -dy * 0.1  # Perpendicular offset
                        offset_y = dx * 0.1   # Perpendicular offset
                        
                        # Create curved path with some interpolated points
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
                        
                        # Edge text
                        text = f"{u} ↔ {v}"
                        
                        # Add confidence if available
                        width = 1.0
                        if show_confidence and 'confidence' in data:
                            text += f"<br>Confidence: {data['confidence']:.3f}"
                            width = 1 + 2 * data['confidence']
                        
                        # Create bidirected edge trace
                        bidirect_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=width, color="rgba(200, 0, 0, 0.5)", dash="dot"),
                            hoverinfo='text',
                            text=text,
                            mode='lines',
                            name='Bidirected Edge',
                            showlegend=False
                        )
                        
                        fig.add_trace(bidirect_trace)
                    
                    # Add a dummy trace just for legend
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        line=dict(width=1.0, color="rgba(200, 0, 0, 0.5)", dash="dot"),
                        mode='lines',
                        name='Bidirected Edges'
                    ))
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node label
                if node_labels and node in node_labels:
                    label = node_labels[node]
                    node_text.append(f"Node ID: {node}<br>Label: {label}")
                else:
                    node_text.append(f"Node ID: {node}")
                
                # Node color and size
                node_attrs = G.nodes[node]
                
                # Check if it's a hidden variable
                if 'is_hidden' in node_attrs and node_attrs['is_hidden']:
                    node_color.append('rgba(255, 0, 0, 0.8)')  # Red for hidden variables
                    node_size.append(15)
                else:
                    node_color.append('rgba(0, 100, 255, 0.8)')  # Blue for observed variables
                    node_size.append(10)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[str(i) for i in G.nodes()],
                textposition="top center",
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    color=node_color,
                    size=node_size,
                    line=dict(width=1, color='rgba(0, 0, 0, 0.8)')
                ),
                name='Nodes'
            )
            
            fig.add_trace(node_trace)
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text="Causal Graph Visualization",
                    font=dict(size=16)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                ),
                plot_bgcolor='rgba(255, 255, 255, 1)',
                paper_bgcolor='rgba(255, 255, 255, 1)'
            )

            # Add a legend title
            fig.update_layout(
                legend_title_text='Graph Elements'
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating graph visualization: {str(e)}",
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
    
    def visualize_graph_comparison(self, 
                                graph1: nx.DiGraph, 
                                graph2: nx.DiGraph,
                                title1: str = "Graph 1",
                                title2: str = "Graph 2",
                                node_labels: Optional[Dict[int, str]] = None) -> go.Figure:
        """
        Visualize a comparison between two causal graphs
        
        Args:
            graph1: First NetworkX DiGraph
            graph2: Second NetworkX DiGraph
            title1: Title for first graph
            title2: Title for second graph
            node_labels: Dictionary mapping node IDs to labels
            
        Returns:
            Plotly figure with side-by-side comparison
        """
        try:
            # Create a copy of the graphs
            G1 = graph1.copy()
            G2 = graph2.copy()
            
            # Check if graphs have the same nodes
            if set(G1.nodes()) != set(G2.nodes()):
                logger.warning("Graphs have different nodes, visualization may be misleading")
            
            # Get common layout for both graphs
            # Use union of both graphs to calculate layout
            union_graph = nx.DiGraph()
            union_graph.add_nodes_from(list(G1.nodes()) + list(G2.nodes()))
            union_graph.add_edges_from(list(G1.edges()) + list(G2.edges()))
            
            pos = nx.spring_layout(union_graph, seed=42)
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(title1, title2),
                horizontal_spacing=0.05
            )
            
            # Add first graph
            for graph_idx, G in enumerate([G1, G2]):
                col = graph_idx + 1
                
                # Add edges
                edge_x = []
                edge_y = []
                edge_text = []
                
                for u, v, data in G.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    
                    # Add line
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                    # Edge text
                    text = f"{u} → {v}"
                    if 'weight' in data:
                        text += f"<br>Weight: {data['weight']:.3f}"
                    if 'confidence' in data:
                        text += f"<br>Confidence: {data['confidence']:.3f}"
                    
                    edge_text.append(text)
                
                # Add edges to subplot
                fig.add_trace(
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color='rgba(0, 0, 200, 0.5)'),
                        hoverinfo='text',
                        text=edge_text,
                        mode='lines',
                        name=f'Edges {title1 if graph_idx == 0 else title2}'
                    ),
                    row=1, col=col
                )
                
                # Add nodes
                node_x = []
                node_y = []
                node_text = []
                
                for node in G.nodes():
                    if node in pos:  # Check if node is in layout
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        # Node label
                        if node_labels and node in node_labels:
                            label = node_labels[node]
                            node_text.append(f"Node ID: {node}<br>Label: {label}")
                        else:
                            node_text.append(f"Node ID: {node}")
                
                # Add nodes to subplot
                fig.add_trace(
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=[str(i) for i in G.nodes() if i in pos],
                        textposition="top center",
                        hoverinfo='text',
                        hovertext=node_text,
                        marker=dict(
                            color='rgba(0, 100, 255, 0.8)',
                            size=10,
                            line=dict(width=1, color='rgba(0, 0, 0, 0.8)')
                        ),
                        name=f'Nodes {title1 if graph_idx == 0 else title2}'
                    ),
                    row=1, col=col
                )
            
            # Update layout
            fig.update_layout(
                title="Causal Graph Comparison",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                height=500,
                width=1000
            )
            
            # Remove axis for better visualization
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            
            return fig
        
        except Exception as e:
            logger.error(f"Error visualizing graph comparison: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating graph comparison: {str(e)}",
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
    
    def create_adjacency_matrix_plot(self, graph: nx.DiGraph, node_labels: Optional[Dict[int, str]] = None) -> go.Figure:
        """
        Create a visualization of the adjacency matrix
        
        Args:
            graph: NetworkX DiGraph representing the causal graph
            node_labels: Dictionary mapping node IDs to labels
            
        Returns:
            Plotly figure of adjacency matrix
        """
        try:
            # Get adjacency matrix
            nodes = sorted(list(graph.nodes()))
            n_nodes = len(nodes)
            
            # Create empty matrix
            adj_matrix = np.zeros((n_nodes, n_nodes))
            
            # Fill adjacency matrix
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if graph.has_edge(u, v):
                        # Check for bidirected edge
                        if graph.has_edge(v, u) and 'bidirected' in graph.edges[u, v]:
                            adj_matrix[i, j] = 0.5  # Use 0.5 for bidirected
                        else:
                            adj_matrix[i, j] = 1  # Use 1 for directed
            
            # Create labels
            if node_labels:
                labels = [node_labels.get(node, str(node)) for node in nodes]
            else:
                labels = [str(node) for node in nodes]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=adj_matrix,
                x=labels,
                y=labels,
                colorscale=[[0, 'white'], [0.5, 'orange'], [1, 'blue']],
                showscale=False,
                text=[[f"{labels[i]} → {labels[j]}" if adj_matrix[i, j] == 1 else
                       f"{labels[i]} ↔ {labels[j]}" if adj_matrix[i, j] == 0.5 else
                       "No edge" for j in range(n_nodes)] for i in range(n_nodes)],
                hoverinfo='text'
            ))
            
            # Update layout
            fig.update_layout(
                title="Adjacency Matrix",
                xaxis=dict(title="To"),
                yaxis=dict(title="From", autorange='reversed')
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating adjacency matrix plot: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating adjacency matrix: {str(e)}",
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
    
    def visualize_causal_paths(self, 
                             graph: nx.DiGraph, 
                             source: int, 
                             target: int,
                             node_labels: Optional[Dict[int, str]] = None) -> go.Figure:
        """
        Visualize all causal paths between source and target nodes
        
        Args:
            graph: NetworkX DiGraph representing the causal graph
            source: Source node ID
            target: Target node ID
            node_labels: Dictionary mapping node IDs to labels
            
        Returns:
            Plotly figure highlighting causal paths
        """
        try:
            if source not in graph.nodes() or target not in graph.nodes():
                raise ValueError(f"Source or target node not in graph")
            
            # Create a copy of the graph for visualization
            G = graph.copy()
            
            # Find all simple paths
            try:
                paths = list(nx.all_simple_paths(G, source, target))
            except nx.NetworkXNoPath:
                paths = []
            
            if not paths:
                # Create simple figure with message
                fig = go.Figure()
                fig.update_layout(
                    title=f"No causal paths found from {source} to {target}",
                    annotations=[
                        dict(
                            text="No causal paths exist between these nodes",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False
                        )
                    ]
                )
                return fig
            
            # Get node positions
            pos = nx.spring_layout(G, seed=42)
            
            # Create figure
            fig = go.Figure()
            
            # Add all edges with low opacity
            edge_x = []
            edge_y = []
            
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Add background edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(200, 200, 200, 0.2)'),
                hoverinfo='none',
                mode='lines',
                name='Background Edges'
            ))
            
            # Add paths with different colors
            colors = ['rgba(255,0,0,0.8)', 'rgba(0,255,0,0.8)', 'rgba(0,0,255,0.8)', 
                    'rgba(255,255,0,0.8)', 'rgba(255,0,255,0.8)', 'rgba(0,255,255,0.8)']
            
            for i, path in enumerate(paths):
                path_edge_x = []
                path_edge_y = []
                path_edge_text = []
                
                # Get path edges
                path_edges = list(zip(path[:-1], path[1:]))
                
                for u, v in path_edges:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    path_edge_x.extend([x0, x1, None])
                    path_edge_y.extend([y0, y1, None])
                    
                    # Edge text
                    if node_labels:
                        u_label = node_labels.get(u, str(u))
                        v_label = node_labels.get(v, str(v))
                        text = f"{u_label} → {v_label}"
                    else:
                        text = f"{u} → {v}"
                    
                    path_edge_text.append(text)
                
                # Add path edges
                color_idx = i % len(colors)
                fig.add_trace(go.Scatter(
                    x=path_edge_x, y=path_edge_y,
                    line=dict(width=2, color=colors[color_idx]),
                    hoverinfo='text',
                    text=path_edge_text,
                    mode='lines',
                    name=f'Path {i+1}'
                ))
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node label
                if node_labels and node in node_labels:
                    label = node_labels[node]
                    node_text.append(f"Node ID: {node}<br>Label: {label}")
                else:
                    node_text.append(f"Node ID: {node}")
                
                # Node color and size
                if node == source:
                    node_color.append('rgba(0, 255, 0, 0.8)')  # Green for source
                    node_size.append(15)
                elif node == target:
                    node_color.append('rgba(255, 0, 0, 0.8)')  # Red for target
                    node_size.append(15)
                elif any(node in path for path in paths):
                    node_color.append('rgba(0, 100, 255, 0.8)')  # Blue for path nodes
                    node_size.append(10)
                else:
                    node_color.append('rgba(200, 200, 200, 0.5)')  # Gray for other nodes
                    node_size.append(8)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[str(i) for i in G.nodes()],
                textposition="top center",
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    color=node_color,
                    size=node_size,
                    line=dict(width=1, color='rgba(0, 0, 0, 0.8)')
                ),
                name='Nodes'
            )
            
            fig.add_trace(node_trace)
            
            # Update layout
            source_label = node_labels.get(source, str(source)) if node_labels else str(source)
            target_label = node_labels.get(target, str(target)) if node_labels else str(target)
            
            fig.update_layout(
                # title=f"Causal Paths from {source_label} to {target_label}",
                title=dict(
                    text=f"Causal Paths from {source_label} to {target_label}",
                    font=dict(size=16)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=[
                    dict(
                        text=f"Found {len(paths)} path(s)",
                        xref="paper", yref="paper",
                        x=0.01, y=0.01,
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                ]
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error visualizing causal paths: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error visualizing causal paths: {str(e)}",
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
        

    def visualize_temporal_graph(self, 
                           graph: nx.DiGraph, 
                           max_lags: int = 3,
                           node_labels: Optional[Dict[int, str]] = None) -> go.Figure:
        """
        Visualize a temporal causal graph with time lags
        
        Args:
            graph: NetworkX DiGraph with time lag information
            max_lags: Maximum number of time lags to display
            node_labels: Optional node labels
        
        Returns:
            Plotly figure
        """
        try:
            # Create a figure with subplots for each time lag
            fig = make_subplots(rows=1, cols=max_lags+1, 
                            subplot_titles=[f"Contemporaneous" if i==0 else f"Lag {i}" for i in range(max_lags+1)])
            
            # Extract all nodes and edges
            all_nodes = list(graph.nodes())
            all_edges = list(graph.edges(data=True))
            
            # Organize edges by lag
            lag_edges = {}
            for lag in range(max_lags+1):
                lag_edges[lag] = []
            
            for u, v, data in all_edges:
                lag = data.get('time_lag', 0)
                if isinstance(lag, int) and lag <= max_lags:
                    lag_edges[lag].append((u, v, data))
            
            # For each time lag, create a graph visualization
            for lag in range(max_lags+1):
                # Create a subgraph with edges for this lag
                subgraph = nx.DiGraph()
                subgraph.add_nodes_from(all_nodes)
                subgraph.add_edges_from([(u, v, d) for u, v, d in lag_edges[lag]])
                
                # Use spring layout for node positions (consistent across subgraphs)
                if lag == 0:
                    pos = nx.spring_layout(subgraph, seed=42)
                
                # Add nodes
                node_x = []
                node_y = []
                node_text = []
                
                for node in subgraph.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Node label
                    if node_labels and node in node_labels:
                        label = node_labels[node]
                        node_text.append(f"Node ID: {node}<br>Label: {label}")
                    else:
                        node_text.append(f"Node ID: {node}")
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=[str(i) for i in subgraph.nodes()],
                    textposition="top center",
                    hoverinfo='text',
                    hovertext=node_text,
                    marker=dict(
                        color='rgba(0, 100, 255, 0.8)',
                        size=10,
                        line=dict(width=1, color='rgba(0, 0, 0, 0.8)')
                    ),
                    name='Nodes'
                )
                
                # Add edges
                for u, v, data in subgraph.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    
                    # Edge text
                    text = f"{u} → {v}"
                    if 'weight' in data:
                        text += f"<br>Weight: {data['weight']:.3f}"
                    if 'confidence' in data:
                        text += f"<br>Confidence: {data['confidence']:.3f}"
                    
                    edge_trace = go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=1.5, color='rgba(0, 0, 200, 0.5)'),
                        hoverinfo='text',
                        text=text,
                        mode='lines',
                        showlegend=False
                    )
                    
                    fig.add_trace(edge_trace, row=1, col=lag+1)
                
                # Add node trace
                fig.add_trace(node_trace, row=1, col=lag+1)
            
            # Update layout
            fig.update_layout(
                title="Temporal Causal Graph Visualization",
                showlegend=False,
                height=500,
                width=300 * (max_lags + 1)
            )
            
            # Remove axis labels
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            
            return fig
        
        except Exception as e:
            logger.error(f"Error visualizing temporal graph: {str(e)}")
            
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title=f"Error creating temporal graph visualization: {str(e)}",
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