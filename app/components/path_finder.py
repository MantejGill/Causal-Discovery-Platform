# app/components/path_finder.py

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathFinder:
    """
    Find and analyze causal paths between variables in a causal graph.
    Identifies direct and indirect effects, mediators, and path-specific
    effects between variables.
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize path finder with a causal graph
        
        Args:
            graph: NetworkX DiGraph representing the causal structure
        """
        self.graph = graph
        self.paths = []
        self.path_analysis = {}
    
    def _get_node_name(self, node: Any) -> str:
        """Get the readable name for a node"""
        if "name" in self.graph.nodes[node]:
            return self.graph.nodes[node]["name"]
        else:
            return str(node)
    
    def find_all_paths(self, 
                     source: Any, 
                     target: Any, 
                     max_length: Optional[int] = None) -> List[List[Any]]:
        """
        Find all causal paths from source to target
        
        Args:
            source: Source node
            target: Target node
            max_length: Maximum path length (None for no limit)
            
        Returns:
            List of paths, where each path is a list of nodes
        """
        try:
            # Find all simple paths
            if max_length is not None:
                paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            else:
                paths = list(nx.all_simple_paths(self.graph, source, target))
            
            # Sort paths by length
            paths = sorted(paths, key=len)
            
            # Store the paths
            self.paths = paths
            
            return paths
            
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.error(f"Error finding paths: {str(e)}")
            self.paths = []
            return []
    
    def analyze_paths(self) -> Dict[str, Any]:
        """
        Analyze the found paths
        
        Returns:
            Dictionary with analysis results
        """
        if not self.paths:
            self.path_analysis = {
                "status": "no_paths",
                "message": "No paths found"
            }
            return self.path_analysis
        
        # Analyze paths
        direct_path = None
        indirect_paths = []
        all_mediators = set()
        
        for path in self.paths:
            path_length = len(path) - 1  # Number of edges
            
            if path_length == 1:
                # Direct path
                direct_path = path
            else:
                # Indirect path
                indirect_paths.append(path)
                
                # Add mediators (all nodes except source and target)
                all_mediators.update(path[1:-1])
        
        # Group paths by length
        paths_by_length = {}
        for path in self.paths:
            length = len(path) - 1
            if length not in paths_by_length:
                paths_by_length[length] = []
            paths_by_length[length].append(path)
        
        # Calculate path probabilities if edge weights present
        path_probabilities = self._calculate_path_probabilities()
        
        # Store analysis results
        self.path_analysis = {
            "status": "completed",
            "total_paths": len(self.paths),
            "direct_path": direct_path,
            "has_direct_effect": direct_path is not None,
            "indirect_paths": indirect_paths,
            "has_indirect_effect": len(indirect_paths) > 0,
            "all_mediators": list(all_mediators),
            "paths_by_length": paths_by_length,
            "path_probabilities": path_probabilities
        }
        
        return self.path_analysis
    
    def _calculate_path_probabilities(self) -> Dict[int, float]:
        """
        Calculate probability for each path based on edge weights/confidences
        
        Returns:
            Dictionary mapping path index to probability
        """
        path_probs = {}
        
        # Check if we have weight or confidence attributes
        has_weights = False
        weight_attr = None
        
        if self.paths:
            # Check first path for edge attributes
            first_path = self.paths[0]
            if len(first_path) > 1:
                u, v = first_path[0], first_path[1]
                if 'weight' in self.graph.edges[u, v]:
                    has_weights = True
                    weight_attr = 'weight'
                elif 'confidence' in self.graph.edges[u, v]:
                    has_weights = True
                    weight_attr = 'confidence'
        
        if not has_weights:
            # No weights available, assign equal probabilities
            for i, path in enumerate(self.paths):
                path_probs[i] = 1.0 / len(self.paths)
            return path_probs
        
        # Calculate path probabilities based on edge weights
        path_weights = []
        
        for i, path in enumerate(self.paths):
            path_weight = 1.0
            
            # Multiply weights of all edges in the path
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                edge_weight = self.graph.edges[u, v].get(weight_attr, 1.0)
                path_weight *= edge_weight
            
            path_weights.append(path_weight)
            path_probs[i] = path_weight
        
        # Normalize to sum to 1.0
        if sum(path_weights) > 0:
            for i in range(len(path_weights)):
                path_probs[i] = path_weights[i] / sum(path_weights)
        
        return path_probs
    
    def find_mediators(self) -> List[Dict[str, Any]]:
        """
        Find mediators between source and target
        
        Returns:
            List of mediator details
        """
        if not self.paths or "all_mediators" not in self.path_analysis:
            return []
        
        mediators = []
        all_mediator_nodes = self.path_analysis["all_mediators"]
        
        # Count paths going through each mediator
        mediator_counts = {mediator: 0 for mediator in all_mediator_nodes}
        mediator_paths = {mediator: [] for mediator in all_mediator_nodes}
        
        for i, path in enumerate(self.paths):
            if len(path) <= 2:  # Skip direct paths
                continue
                
            # Get mediators in this path
            path_mediators = path[1:-1]
            
            for mediator in path_mediators:
                mediator_counts[mediator] += 1
                mediator_paths[mediator].append(i)
        
        # Get details for each mediator
        for mediator in all_mediator_nodes:
            # Get mediator name
            mediator_name = self._get_node_name(mediator)
            
            # Get paths through this mediator
            paths_through = mediator_paths[mediator]
            
            # Calculate importance based on path count and probabilities
            path_probs = self.path_analysis.get("path_probabilities", {})
            mediator_importance = sum(path_probs.get(i, 1.0/len(self.paths)) for i in paths_through)
            
            mediators.append({
                "node": mediator,
                "name": mediator_name,
                "paths_count": mediator_counts[mediator],
                "paths_indices": paths_through,
                "importance": mediator_importance
            })
        
        # Sort mediators by importance
        mediators.sort(key=lambda x: x["importance"], reverse=True)
        
        return mediators
    
    def find_confounders(self) -> List[Dict[str, Any]]:
        """
        Find potential confounders between source and target
        
        Returns:
            List of potential confounder details
        """
        if not self.paths:
            return []
        
        # Get source and target from first path
        source = self.paths[0][0]
        target = self.paths[0][-1]
        
        confounders = []
        
        # Check each node as potential confounder
        for node in self.graph.nodes():
            # Skip source and target
            if node == source or node == target:
                continue
            
            # Check if node has paths to both source and target
            path_to_source = False
            path_to_target = False
            
            try:
                # Check path to source
                for _ in nx.all_simple_paths(self.graph, node, source):
                    path_to_source = True
                    break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
                
            try:
                # Check path to target
                for _ in nx.all_simple_paths(self.graph, node, target):
                    path_to_target = True
                    break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            
            # If node has paths to both, it's a potential confounder
            if path_to_source and path_to_target:
                # Get confounder name
                node_name = self._get_node_name(node)
                
                confounders.append({
                    "node": node,
                    "name": node_name,
                    "is_hidden": self.graph.nodes[node].get("is_hidden", False)
                })
        
        return confounders
    
    def find_colliders(self) -> List[Dict[str, Any]]:
        """
        Find colliders (common effects) between source and target
        
        Returns:
            List of collider details
        """
        if not self.paths:
            return []
        
        # Get source and target from first path
        source = self.paths[0][0]
        target = self.paths[0][-1]
        
        colliders = []
        
        # Check all nodes as potential colliders
        for node in self.graph.nodes():
            # Skip source and target
            if node == source or node == target:
                continue
            
            # Check if both source and target have paths to node
            path_from_source = False
            path_from_target = False
            
            try:
                # Check path from source
                for _ in nx.all_simple_paths(self.graph, source, node):
                    path_from_source = True
                    break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
                
            try:
                # Check path from target
                for _ in nx.all_simple_paths(self.graph, target, node):
                    path_from_target = True
                    break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            
            # If node has paths from both, it's a collider
            if path_from_source and path_from_target:
                # Get node name
                node_name = self._get_node_name(node)
                
                colliders.append({
                    "node": node,
                    "name": node_name
                })
        
        return colliders
    
    def calculate_path_effects(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate causal effects along different paths
        
        Args:
            data: Optional DataFrame for calculating effects from data
            
        Returns:
            Dictionary with path effects
        """
        if not self.paths:
            return {"status": "no_paths", "message": "No paths found"}
        
        # If no data provided, use path probabilities as proxy for effects
        if data is None:
            path_probs = self.path_analysis.get("path_probabilities", {})
            
            # Calculate total effect as sum of path probabilities
            total_effect = sum(path_probs.values())
            
            # Calculate direct effect (if exists)
            direct_effect = 0.0
            direct_path_idx = -1
            
            for i, path in enumerate(self.paths):
                if len(path) == 2:  # Direct path has length 2 (source and target)
                    direct_effect = path_probs.get(i, 0.0)
                    direct_path_idx = i
                    break
            
            # Calculate indirect effect
            indirect_effect = total_effect - direct_effect
            
            return {
                "status": "completed",
                "total_effect": total_effect,
                "direct_effect": direct_effect,
                "indirect_effect": indirect_effect,
                "path_effects": path_probs,
                "source_type": "path_probabilities",
                "message": "Effects estimated from path probabilities (no data provided)"
            }
        
        # If data is provided, try to estimate effects
        try:
            # Get source and target from first path
            source = self.paths[0][0]
            target = self.paths[0][-1]
            
            # Get node names for data lookup
            source_name = self._get_node_name(source)
            target_name = self._get_node_name(target)
            
            # Check if names are in data columns
            if source_name not in data.columns or target_name not in data.columns:
                logger.warning(f"Source or target not found in data columns: {source_name}, {target_name}")
                return {
                    "status": "error",
                    "message": f"Source or target not found in data columns: {source_name}, {target_name}"
                }
            
            # Simple linear regression to estimate total effect
            import statsmodels.api as sm
            
            # Prepare data for total effect
            X_total = data[source_name].values.reshape(-1, 1)
            X_total = sm.add_constant(X_total)
            y_total = data[target_name].values
            
            # Fit model for total effect
            model_total = sm.OLS(y_total, X_total).fit()
            total_effect = model_total.params[1]  # Coefficient for source
            
            # Calculate direct effect (controlling for all mediators)
            direct_effect = total_effect
            
            if "all_mediators" in self.path_analysis and self.path_analysis["all_mediators"]:
                # Get mediator names
                mediator_names = []
                for mediator in self.path_analysis["all_mediators"]:
                    mediator_name = self._get_node_name(mediator)
                    if mediator_name in data.columns:
                        mediator_names.append(mediator_name)
                
                if mediator_names:
                    # Create design matrix with source and all mediators
                    X_columns = [source_name] + mediator_names
                    X_direct = data[X_columns].values
                    X_direct = sm.add_constant(X_direct)
                    
                    # Fit model for direct effect
                    model_direct = sm.OLS(y_total, X_direct).fit()
                    direct_effect = model_direct.params[1]  # Coefficient for source
            
            # Calculate indirect effect
            indirect_effect = total_effect - direct_effect
            
            # Estimate individual path effects (simplified approach)
            path_effects = {}
            
            # 1. Direct path (if exists)
            for i, path in enumerate(self.paths):
                if len(path) == 2:  # Direct path
                    path_effects[i] = direct_effect
                    break
            
            # 2. Indirect paths (if data for all mediators is available)
            if "indirect_paths" in self.path_analysis and mediator_names:
                remaining_effect = indirect_effect
                remaining_paths = len(self.path_analysis.get("indirect_paths", []))
                
                if remaining_paths > 0:
                    # Simplified: distribute indirect effect equally among paths
                    for i, path in enumerate(self.paths):
                        if len(path) > 2 and i not in path_effects:
                            path_effects[i] = remaining_effect / remaining_paths
            
            return {
                "status": "completed",
                "total_effect": total_effect,
                "direct_effect": direct_effect,
                "indirect_effect": indirect_effect,
                "path_effects": path_effects,
                "source_type": "regression",
                "message": "Effects estimated from data using regression models"
            }
            
        except Exception as e:
            logger.error(f"Error calculating path effects: {str(e)}")
            return {
                "status": "error",
                "message": f"Error calculating path effects: {str(e)}"
            }
    
    def visualize_paths(self, 
                      highlight_mediators: bool = True, 
                      show_weights: bool = True) -> go.Figure:
        """
        Create visualization of the causal paths
        
        Args:
            highlight_mediators: Whether to highlight mediator nodes
            show_weights: Whether to show edge weights/confidences
            
        Returns:
            Plotly figure with path visualization
        """
        if not self.paths:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No paths found",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Get source and target
        source = self.paths[0][0]
        target = self.paths[0][-1]
        
        # Create a subgraph with only the nodes and edges on the paths
        path_graph = nx.DiGraph()
        
        # Add all nodes and edges from all paths
        for path in self.paths:
            path_graph.add_nodes_from(path)
            
            # Add edges with their attributes from the original graph
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                
                if self.graph.has_edge(u, v):
                    path_graph.add_edge(u, v, **self.graph.edges[u, v])
                else:
                    path_graph.add_edge(u, v)
        
        # Create node positions using a hierarchical layout
        # Place source at left, target at right, and arrange other nodes by their distance from source
        pos = {}
        
        # Calculate hierarchical levels
        levels = {}
        levels[source] = 0
        levels[target] = 1  # Will be updated later
        
        # Use shortest path length from source to place nodes
        for node in path_graph.nodes():
            if node != source and node != target:
                try:
                    levels[node] = nx.shortest_path_length(path_graph, source, node)
                except nx.NetworkXNoPath:
                    # If no path from source, use reverse path from target
                    try:
                        # Place relative to target, but at a higher level
                        revs_dist = nx.shortest_path_length(path_graph, node, target)
                        levels[node] = -revs_dist  # Negative to place before source
                    except nx.NetworkXNoPath:
                        # If disconnected, place at level 0.5
                        levels[node] = 0.5
        
        # Update target level to be maximum + 1
        max_level = max(levels.values())
        levels[target] = max_level + 1
        
        # Convert levels to horizontal positions
        # Count nodes at each level
        level_counts = {}
        for node, level in levels.items():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Assign positions
        level_positions = {}
        for level in sorted(level_counts.keys()):
            level_positions[level] = []
        
        # Place nodes
        for node, level in levels.items():
            count = level_counts[level]
            position = len(level_positions[level])
            
            # Calculate y position based on position within level
            if count > 1:
                y = position / (count - 1)
            else:
                y = 0.5
                
            # Special positions for source and target
            if node == source:
                y = 0.5  # Center source vertically
            elif node == target:
                y = 0.5  # Center target vertically
            
            # Scale x position to [0, 1]
            x = level / max(max_level + 1, 1)
            
            # Store position
            pos[node] = (x, y)
            level_positions[level].append(node)
        
        # Create figure
        fig = go.Figure()
        
        # Get mediators if highlighting
        mediators = []
        if highlight_mediators and "all_mediators" in self.path_analysis:
            mediators = self.path_analysis["all_mediators"]
        
        # Add edges
        for u, v in path_graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Determine edge attributes
            edge_attrs = path_graph.edges[u, v]
            
            # Determine edge width based on weight/confidence
            width = 1.5
            if 'weight' in edge_attrs:
                width = 1 + 3 * edge_attrs['weight']
            elif 'confidence' in edge_attrs:
                width = 1 + 3 * edge_attrs['confidence']
            
            # Determine edge color
            color = 'rgba(0, 0, 200, 0.5)'
            
            # Determine hover text
            hover_text = f"{self._get_node_name(u)} → {self._get_node_name(v)}"
            
            if show_weights:
                if 'weight' in edge_attrs:
                    hover_text += f"<br>Weight: {edge_attrs['weight']:.3f}"
                if 'confidence' in edge_attrs:
                    hover_text += f"<br>Confidence: {edge_attrs['confidence']:.3f}"
            
            # Add edge
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=hover_text,
                showlegend=False
            ))
            
            # Add arrowhead
            # Calculate position for arrowhead (slightly before end point)
            t = 0.9  # Position along the edge (0 = start, 1 = end)
            arrow_x = x0 + t * (x1 - x0)
            arrow_y = y0 + t * (y1 - y0)
            
            # Calculate angle for arrow
            angle = np.arctan2(y1 - y0, x1 - x0)
            
            # Add arrowhead marker
            fig.add_trace(go.Scatter(
                x=[arrow_x],
                y=[arrow_y],
                mode='markers',
                marker=dict(
                    symbol='triangle-right',
                    size=10,
                    angle=angle * 180 / np.pi,  # Convert to degrees
                    color=color
                ),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        for node in path_graph.nodes():
            x, y = pos[node]
            
            # Determine node size and color
            size = 15
            color = 'rgba(0, 100, 255, 0.8)'
            
            if node == source:
                size = 20
                color = 'rgba(0, 200, 0, 0.8)'  # Green for source
            elif node == target:
                size = 20
                color = 'rgba(200, 0, 0, 0.8)'  # Red for target
            elif node in mediators:
                size = 18
                color = 'rgba(255, 165, 0, 0.8)'  # Orange for mediators
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=1, color='black')
                ),
                text=[self._get_node_name(node)],
                textposition="top center",
                hoverinfo='text',
                hovertext=f"Node: {self._get_node_name(node)}",
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title="Causal Paths Visualization",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.05, 1.05]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.05, 1.05]
            ),
            height=600,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def visualize_path_effects(self, effects: Dict[str, Any]) -> go.Figure:
        """
        Visualize the effects along different paths
        
        Args:
            effects: Effects dictionary from calculate_path_effects
            
        Returns:
            Plotly figure with effects visualization
        """
        if effects.get("status") != "completed":
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=effects.get("message", "No effects information available"),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Get effects information
        total_effect = effects.get("total_effect", 0)
        direct_effect = effects.get("direct_effect", 0)
        indirect_effect = effects.get("indirect_effect", 0)
        path_effects = effects.get("path_effects", {})
        
        # Create figure
        fig = go.Figure()
        
        # Add overall effects bar chart
        fig.add_trace(go.Bar(
            x=["Total Effect", "Direct Effect", "Indirect Effect"],
            y=[total_effect, direct_effect, indirect_effect],
            marker_color=["blue", "green", "orange"],
            name="Overall Effects"
        ))
        
        # Create text for path effects
        if self.paths and path_effects:
            # Only show individual paths if there are multiple
            if len(self.paths) > 1:
                # Create subplot with second bar chart for individual path effects
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=["Overall Effects", "Individual Path Effects"],
                    vertical_spacing=0.2,
                    row_heights=[0.4, 0.6]
                )
                
                # Add overall effects to first subplot
                fig.add_trace(go.Bar(
                    x=["Total Effect", "Direct Effect", "Indirect Effect"],
                    y=[total_effect, direct_effect, indirect_effect],
                    marker_color=["blue", "green", "orange"],
                    name="Overall Effects"
                ), row=1, col=1)
                
                # Add individual path effects to second subplot
                path_labels = []
                path_effect_values = []
                path_colors = []
                
                for i, path in enumerate(self.paths):
                    if i in path_effects:
                        # Create path label
                        path_str = " → ".join([self._get_node_name(node) for node in path])
                        if len(path_str) > 30:
                            path_str = f"Path {i+1}: " + path_str[:27] + "..."
                        else:
                            path_str = f"Path {i+1}: " + path_str
                        
                        path_labels.append(path_str)
                        path_effect_values.append(path_effects[i])
                        
                        # Determine color based on path length
                        if len(path) == 2:  # Direct path
                            path_colors.append("green")
                        else:  # Indirect path
                            path_colors.append("orange")
                
                fig.add_trace(go.Bar(
                    x=path_labels,
                    y=path_effect_values,
                    marker_color=path_colors,
                    name="Path Effects"
                ), row=2, col=1)
                
                # Update yaxis for path effects
                fig.update_yaxes(title_text="Effect Strength", row=2, col=1)
                
                # Update layout
                fig.update_layout(
                    title="Causal Effects Analysis",
                    showlegend=False,
                    height=800
                )
            else:
                # Just add title
                fig.update_layout(
                    title="Causal Effects Analysis",
                    yaxis_title="Effect Strength"
                )
        
        return fig


def render_path_finder(graph: nx.DiGraph, data: Optional[pd.DataFrame] = None):
    """
    Render the path finder UI in Streamlit
    
    Args:
        graph: The causal graph
        data: Optional dataset for effect estimation
    """
    st.subheader("Causal Path Finder")
    
    # Get node names for selection
    node_names = {}
    for node in graph.nodes():
        if "name" in graph.nodes[node]:
            node_names[node] = graph.nodes[node]["name"]
        else:
            node_names[node] = str(node)
    
    # Select source and target
    col1, col2 = st.columns(2)
    
    with col1:
        source_node = st.selectbox(
            "Source Node",
            options=list(graph.nodes()),
            format_func=lambda x: node_names.get(x, str(x))
        )
    
    with col2:
        target_node = st.selectbox(
            "Target Node",
            options=list(graph.nodes()),
            format_func=lambda x: node_names.get(x, str(x))
        )
    
    # Path options
    max_length = st.slider(
        "Maximum Path Length",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of edges in a path (-1 for no limit)"
    )
    
    # Analyze button
    if st.button("Find and Analyze Paths"):
        # Create path finder
        path_finder = PathFinder(graph)
        
        # Find paths
        with st.spinner("Finding paths..."):
            paths = path_finder.find_all_paths(source_node, target_node, max_length)
        
        if not paths:
            st.warning(f"No paths found from {node_names[source_node]} to {node_names[target_node]}")
            return
        
        # Analyze paths
        analysis = path_finder.analyze_paths()
        
        # Show visualizations
        st.subheader("Path Visualization")
        fig = path_finder.visualize_paths(highlight_mediators=True, show_weights=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate path effects if data is provided
        if data is not None:
            effects = path_finder.calculate_path_effects(data)
            
            st.subheader("Path Effects")
            effects_fig = path_finder.visualize_path_effects(effects)
            st.plotly_chart(effects_fig, use_container_width=True)
        
        # Show path details
        st.subheader("Path Details")
        
        # Summary statistics
        total_paths = len(paths)
        path_lengths = [len(path) - 1 for path in paths]
        max_path_length = max(path_lengths)
        min_path_length = min(path_lengths)
        avg_path_length = sum(path_lengths) / len(path_lengths)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Paths", total_paths)
        col2.metric("Min Length", min_path_length)
        col3.metric("Max Length", max_path_length)
        col4.metric("Avg. Length", f"{avg_path_length:.1f}")
        
        # Direct vs indirect effects
        st.markdown("### Direct and Indirect Effects")
        
        if analysis["has_direct_effect"]:
            direct_path = analysis["direct_path"]
            direct_path_str = " → ".join([node_names.get(node, str(node)) for node in direct_path])
            st.success(f"Direct effect present: {direct_path_str}")
        else:
            st.info("No direct effect found")
        
        if analysis["has_indirect_effect"]:
            st.success(f"Indirect effects present: {len(analysis['indirect_paths'])} indirect paths")
        else:
            st.info("No indirect effects found")
        
        # Mediators
        mediators = path_finder.find_mediators()
        if mediators:
            st.markdown("### Mediators")
            
            for i, mediator in enumerate(mediators):
                with st.expander(f"{mediator['name']} (in {mediator['paths_count']} paths)"):
                    st.markdown(f"**Importance:** {mediator['importance']:.3f}")
                    
                    st.markdown("**Paths through this mediator:**")
                    for path_idx in mediator['paths_indices']:
                        path = paths[path_idx]
                        path_str = " → ".join([node_names.get(node, str(node)) for node in path])
                        st.markdown(f"- Path {path_idx+1}: {path_str}")
        
        # Potential confounders
        confounders = path_finder.find_confounders()
        if confounders:
            st.markdown("### Potential Confounders")
            
            for confounder in confounders:
                hidden_tag = " (Hidden)" if confounder["is_hidden"] else ""
                st.markdown(f"- {confounder['name']}{hidden_tag}")
        
        # All paths
        st.markdown("### All Paths")
        
        for i, path in enumerate(paths):
            path_str = " → ".join([node_names.get(node, str(node)) for node in path])
            with st.expander(f"Path {i+1} (Length: {len(path)-1})"):
                st.markdown(path_str)
                
                # Show path details
                if "path_probabilities" in analysis and i in analysis["path_probabilities"]:
                    st.markdown(f"**Relative Strength:** {analysis['path_probabilities'][i]:.3f}")


def find_causal_paths(graph: nx.DiGraph, 
                    source: Any, 
                    target: Any, 
                    max_length: Optional[int] = None) -> Dict[str, Any]:
    """
    Find and analyze causal paths between two variables
    
    Args:
        graph: The causal graph
        source: Source node
        target: Target node
        max_length: Maximum path length (None for no limit)
        
    Returns:
        Dictionary with path analysis results
    """
    path_finder = PathFinder(graph)
    
    # Find paths
    paths = path_finder.find_all_paths(source, target, max_length)
    
    if not paths:
        return {
            "status": "no_paths",
            "message": f"No paths found from {source} to {target}"
        }
    
    # Analyze paths
    analysis = path_finder.analyze_paths()
    
    # Find mediators
    mediators = path_finder.find_mediators()
    analysis["mediators"] = mediators
    
    # Find confounders
    confounders = path_finder.find_confounders()
    analysis["confounders"] = confounders
    
    return analysis