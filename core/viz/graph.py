# core/viz/graph.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go


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
    
    # Modified version of CausalGraphVisualizer to add edge weights before arrow markers

    def visualize_graph(self, 
                    graph: nx.DiGraph, 
                    node_labels: Optional[Dict[int, str]] = None,
                    edge_weights: bool = True,
                    layout_type: str = 'spring',
                    show_bidirected: bool = True,
                    show_confidence: bool = True) -> go.Figure:
        """
        Visualize a causal graph using Plotly
        
        Args:
            graph: NetworkX DiGraph representing the causal graph
            node_labels: Dictionary mapping node IDs to labels
            edge_weights: Whether to show edge weights
            layout_type: Layout algorithm to use ('spring', 'circular', 'kamada_kawai', 'planar')
            show_bidirected: Whether to show bidirected edges
            show_confidence: Whether to show confidence values
            
        Returns:
            Plotly figure
        """
        try:
            # Ensure required imports are available
            import networkx as nx
            import numpy as np
            import plotly.graph_objects as go
            
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
                pos = nx.spring_layout(G, seed=42)  # Default to spring layout
            
            # Create figure
            fig = go.Figure()
            
            # Process edges - separate into normal and bidirectional edges
            normal_edges = []
            bidirected_edges = []
            
            for u, v, data in G.edges(data=True):
                if show_bidirected and 'bidirected' in data and data['bidirected'] and G.has_edge(v, u):
                    # Only add bidirected edge once (not twice)
                    if (v, u) not in [(e[0], e[1]) for e in bidirected_edges]:
                        bidirected_edges.append((u, v, data))
                else:
                    normal_edges.append((u, v, data))
            
            # Add normal edges with arrows
            for u, v, data in normal_edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Edge text/tooltip
                edge_text = f"{u} → {v}"
                
                # Handle edge weights and confidence
                edge_width = 1.5  # Default width
                edge_color = 'rgba(50, 50, 200, 0.8)'  # Default color
                edge_weight_text = ""
                
                if 'weight' in data and edge_weights:
                    weight = data['weight']
                    edge_text += f"<br>Weight: {weight:.3f}"
                    edge_weight_text = f"{weight:.2f}"
                    
                    # Adjust width based on weight
                    # edge_width = 1.5 + abs(weight)
                    
                    # Optionally change color based on positive/negative weight
                    if weight < 0:
                        edge_color = 'rgba(200, 50, 50, 0.8)'  # Red for negative
                
                if 'confidence' in data and show_confidence:
                    conf = data['confidence']
                    edge_text += f"<br>Confidence: {conf:.3f}"
                    
                    # Adjust width based on confidence
                    # edge_width = 1.0 + 2.0 * conf
                
                # Create edge trace with arrow
                # First create the line
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='text',
                    text=edge_text,
                    showlegend=False
                ))
                
                # Calculate the position for weight label (at 70% along the edge)
                if edge_weight_text:
                    weight_pos = 0.7  # Position along the edge for weight label
                    weight_x = x0 + weight_pos * (x1 - x0)
                    weight_y = y0 + weight_pos * (y1 - y0)
                    
                    # Add weight label
                    fig.add_trace(go.Scatter(
                        x=[weight_x],
                        y=[weight_y],
                        mode='text',
                        text=[edge_weight_text],
                        textposition="middle center",
                        textfont=dict(
                            size=10,
                            color=edge_color
                        ),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                
                # Calculate the position for the arrowhead (slightly before end point)
                # This ensures the arrow is visible and not hidden by the target node
                t = 0.9  # Position along the edge (0 is start, 1 is end)
                arrow_x = x0 + t * (x1 - x0)
                arrow_y = y0 + t * (y1 - y0)
                
                # Calculate angle for arrow
                angle = np.arctan2(y1 - y0, x1 - x0)
                
                # Create arrowhead using a triangle marker
                fig.add_trace(go.Scatter(
                    x=[arrow_x],
                    y=[arrow_y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color=edge_color,
                        angle=np.degrees(angle) + 90  # Rotate to point along edge
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))
            
            # Add bidirected edges (with curved lines)
            for u, v, data in bidirected_edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Create curved path
                # Calculate the midpoint and perpendicular offset
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                # Get vector perpendicular to edge
                dx = x1 - x0
                dy = y1 - y0
                length = np.sqrt(dx*dx + dy*dy)
                nx = -dy / length  # Perpendicular normalized vector
                ny = dx / length
                
                # Control point offset
                offset = length * 0.2
                
                # Control point coordinates
                cx = mid_x + offset * nx
                cy = mid_y + offset * ny
                
                # Generate curved path with Bezier curve
                t = np.linspace(0, 1, 50)
                curve_x = (1-t)**2 * x0 + 2 * (1-t) * t * cx + t**2 * x1
                curve_y = (1-t)**2 * y0 + 2 * (1-t) * t * cy + t**2 * y1
                
                # Edge text/tooltip
                edge_text = f"{u} ↔ {v}"
                edge_weight_text = ""
                
                if 'weight' in data and edge_weights:
                    weight = data['weight']
                    edge_text += f"<br>Weight: {weight:.3f}"
                    edge_weight_text = f"{weight:.2f}"
                
                if 'confidence' in data and show_confidence:
                    edge_text += f"<br>Confidence: {data['confidence']:.3f}"
                
                # Add curved edge
                fig.add_trace(go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode='lines',
                    line=dict(width=1.5, color='rgba(150, 50, 150, 0.8)', dash='dash'),
                    hoverinfo='text',
                    text=edge_text,
                    showlegend=False
                ))
                
                # Add weight label near the middle of the curve if weight exists
                if edge_weight_text:
                    # Position for the weight is at the peak of the curve
                    t_weight = 0.5  # Middle of the curve
                    weight_x = (1-t_weight)**2 * x0 + 2 * (1-t_weight) * t_weight * cx + t_weight**2 * x1
                    weight_y = (1-t_weight)**2 * y0 + 2 * (1-t_weight) * t_weight * cy + t_weight**2 * y1
                    
                    # Add a slight offset to move the weight label away from the curve
                    weight_x += nx * (length * 0.05)
                    weight_y += ny * (length * 0.05)
                    
                    # Add weight label
                    fig.add_trace(go.Scatter(
                        x=[weight_x],
                        y=[weight_y],
                        mode='text',
                        text=[edge_weight_text],
                        textposition="middle center",
                        textfont=dict(
                            size=10,
                            color='rgba(150, 50, 150, 0.8)'
                        ),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                
                # Add arrowhead in both directions
                # First direction (u to v)
                t1 = 0.85  # Position for first arrowhead
                arrow1_x = (1-t1)**2 * x0 + 2 * (1-t1) * t1 * cx + t1**2 * x1
                arrow1_y = (1-t1)**2 * y0 + 2 * (1-t1) * t1 * cy + t1**2 * y1
                
                # Calculate tangent angle at t1
                tangent_x = -2 * (1-t1) * x0 + 2 * (1-2*t1) * cx + 2 * t1 * x1
                tangent_y = -2 * (1-t1) * y0 + 2 * (1-2*t1) * cy + 2 * t1 * y1
                angle1 = np.arctan2(tangent_y, tangent_x)
                
                # Second direction (v to u)
                t2 = 0.15  # Position for second arrowhead
                arrow2_x = (1-t2)**2 * x0 + 2 * (1-t2) * t2 * cx + t2**2 * x1
                arrow2_y = (1-t2)**2 * y0 + 2 * (1-t2) * t2 * cy + t2**2 * y1
                
                # Calculate tangent angle at t2
                tangent_x = -2 * (1-t2) * x0 + 2 * (1-2*t2) * cx + 2 * t2 * x1
                tangent_y = -2 * (1-t2) * y0 + 2 * (1-2*t2) * cy + 2 * t2 * y1
                angle2 = np.arctan2(tangent_y, tangent_x) + np.pi  # Reverse direction
                
                # Add arrowheads
                fig.add_trace(go.Scatter(
                    x=[arrow1_x],
                    y=[arrow1_y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color='rgba(150, 50, 150, 0.8)',
                        angle=np.degrees(angle1) + 90
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=[arrow2_x],
                    y=[arrow2_y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color='rgba(150, 50, 150, 0.8)',
                        angle=np.degrees(angle2) + 90
                    ),
                    hoverinfo='skip',
                    showlegend=False
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
                    node_text.append(f"{label}")
                else:
                    node_text.append(f"{node}")
                
                # Node color and size based on attributes
                if 'is_hidden' in G.nodes[node] and G.nodes[node]['is_hidden']:
                    node_color.append('rgba(255, 150, 150, 0.8)')  # Red for hidden variables
                    node_size.append(15)
                else:
                    node_color.append('rgba(100, 200, 255, 0.8)')  # Blue for observed variables
                    node_size.append(12)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                marker=dict(
                    color=node_color,
                    size=node_size,
                    line=dict(width=2, color='white')
                ),
                name='Nodes'
            )
            
            fig.add_trace(node_trace)
            
            # Add legend traces for edge types (won't show on plot but will show in legend)
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(width=2, color='rgba(50, 50, 200, 0.8)'),
                name='Directed Edge'
            ))
            
            if bidirected_edges:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(width=1.5, color='rgba(150, 50, 150, 0.8)', dash='dash'),
                    name='Bidirected Edge'
                ))
            
            # Update layout
            fig.update_layout(
                title="Causal Graph",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                plot_bgcolor='rgba(255, 255, 255, 1)',
                paper_bgcolor='rgba(255, 255, 255, 1)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            # Make it more square for better visualization
            fig.update_layout(
                autosize=False,
                width=700,
                height=700
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