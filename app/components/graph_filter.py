# app/components/graph_filter.py

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphFilter:
    """
    Filter causal graphs based on various criteria such as
    edge confidence, node connectivity, and causal structures.
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize graph filter with a causal graph
        
        Args:
            graph: NetworkX DiGraph representing the causal structure
        """
        self.original_graph = graph
        self.filtered_graph = graph.copy()
        
    def reset_filter(self):
        """Reset all filters and return to original graph"""
        self.filtered_graph = self.original_graph.copy()
        return self.filtered_graph
    
    def filter_by_confidence(self, 
                           threshold: float, 
                           comparison: str = ">=") -> nx.DiGraph:
        """
        Filter edges by their confidence level
        
        Args:
            threshold: Confidence threshold value
            comparison: Type of comparison (">=", ">", "<=", "<", "==")
            
        Returns:
            Filtered graph
        """
        G = self.original_graph.copy()
        edges_to_remove = []
        
        for u, v, data in G.edges(data=True):
            # Check if confidence is a property of the edge
            if 'confidence' not in data:
                # Keep edges without confidence values if using >=, >, or ==
                if comparison in [">=", ">", "=="]:
                    continue
                # Remove edges without confidence values if using <= or 
                else:
                    edges_to_remove.append((u, v))
                continue
            
            # Get confidence value
            confidence = data['confidence']
            
            # Apply comparison
            if comparison == ">=" and confidence < threshold:
                edges_to_remove.append((u, v))
            elif comparison == ">" and confidence <= threshold:
                edges_to_remove.append((u, v))
            elif comparison == "<=" and confidence > threshold:
                edges_to_remove.append((u, v))
            elif comparison == "<" and confidence >= threshold:
                edges_to_remove.append((u, v))
            elif comparison == "==" and confidence != threshold:
                edges_to_remove.append((u, v))
        
        # Remove edges
        G.remove_edges_from(edges_to_remove)
        
        # Update filtered graph
        self.filtered_graph = G
        return G
    
    def filter_by_edge_type(self, 
                          include_directed: bool = True,
                          include_bidirected: bool = True,
                          include_hidden_causes: bool = True) -> nx.DiGraph:
        """
        Filter edges by type (directed, bidirected, hidden causes)
        
        Args:
            include_directed: Whether to include regular directed edges
            include_bidirected: Whether to include bidirected edges
            include_hidden_causes: Whether to include edges from hidden variables
            
        Returns:
            Filtered graph
        """
        G = self.original_graph.copy()
        edges_to_remove = []
        
        for u, v, data in G.edges(data=True):
            # Check edge type
            is_bidirected = data.get('bidirected', False)
            is_hidden_cause = data.get('is_hidden_cause', False)
            is_regular = not is_bidirected and not is_hidden_cause
            
            # Apply filter
            if (is_regular and not include_directed) or \
               (is_bidirected and not include_bidirected) or \
               (is_hidden_cause and not include_hidden_causes):
                edges_to_remove.append((u, v))
        
        # Remove edges
        G.remove_edges_from(edges_to_remove)
        
        # Update filtered graph
        self.filtered_graph = G
        return G
    
    def filter_by_path_existence(self, 
                               source: Any, 
                               target: Any, 
                               max_length: Optional[int] = None) -> nx.DiGraph:
        """
        Keep only nodes and edges on paths between source and target
        
        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length to consider
            
        Returns:
            Filtered graph with only path nodes and edges
        """
        G = self.original_graph.copy()
        
        try:
            # Find all paths from source to target
            if max_length is not None:
                paths = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
            else:
                paths = list(nx.all_simple_paths(G, source, target))
            
            if not paths:
                logger.warning(f"No paths found from {source} to {target}")
                # Create an empty graph with just the source and target
                path_graph = nx.DiGraph()
                path_graph.add_nodes_from([source, target])
                
                # Copy node attributes
                for node in [source, target]:
                    for key, value in G.nodes[node].items():
                        path_graph.nodes[node][key] = value
                
                self.filtered_graph = path_graph
                return path_graph
            
            # Create a new graph with only nodes and edges on paths
            path_graph = nx.DiGraph()
            
            # Add all nodes and edges on all paths
            for path in paths:
                path_graph.add_nodes_from(path)
                path_graph.add_edges_from(zip(path[:-1], path[1:]))
            
            # Copy node and edge attributes from original graph
            for node in path_graph.nodes():
                for key, value in G.nodes[node].items():
                    path_graph.nodes[node][key] = value
            
            for u, v in path_graph.edges():
                for key, value in G.edges[u, v].items():
                    path_graph.edges[u, v][key] = value
            
            # Update filtered graph
            self.filtered_graph = path_graph
            return path_graph
            
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.error(f"Error finding paths: {str(e)}")
            # Keep original graph if error
            self.filtered_graph = G
            return G
    
    def filter_roots_and_leaves(self, 
                              include_roots: bool = True, 
                              include_leaves: bool = True,
                              include_intermediates: bool = True) -> nx.DiGraph:
        """
        Filter nodes based on their role in the causal graph
        
        Args:
            include_roots: Whether to include root nodes (no parents)
            include_leaves: Whether to include leaf nodes (no children)
            include_intermediates: Whether to include intermediate nodes
            
        Returns:
            Filtered graph
        """
        G = self.original_graph.copy()
        nodes_to_remove = []
        
        for node in G.nodes():
            is_root = G.in_degree(node) == 0
            is_leaf = G.out_degree(node) == 0
            is_intermediate = not is_root and not is_leaf
            
            # Apply filter
            if (is_root and not include_roots) or \
               (is_leaf and not include_leaves) or \
               (is_intermediate and not include_intermediates):
                nodes_to_remove.append(node)
        
        # Remove nodes
        G.remove_nodes_from(nodes_to_remove)
        
        # Update filtered graph
        self.filtered_graph = G
        return G
    
    def filter_by_node_attributes(self, 
                                attribute: str,
                                values: List[Any],
                                include: bool = True) -> nx.DiGraph:
        """
        Filter nodes based on their attributes
        
        Args:
            attribute: Node attribute to filter on
            values: List of attribute values to include/exclude
            include: Whether to include (True) or exclude (False) the values
            
        Returns:
            Filtered graph
        """
        G = self.original_graph.copy()
        nodes_to_remove = []
        
        for node in G.nodes():
            # Get attribute value (None if not present)
            attr_value = G.nodes[node].get(attribute)
            
            # Check if node should be included/excluded
            if include and attr_value not in values:
                nodes_to_remove.append(node)
            elif not include and attr_value in values:
                nodes_to_remove.append(node)
        
        # Remove nodes
        G.remove_nodes_from(nodes_to_remove)
        
        # Update filtered graph
        self.filtered_graph = G
        return G
    
    def filter_by_connected_component(self, node: Any) -> nx.DiGraph:
        """
        Keep only the connected component containing the specified node
        
        Args:
            node: Node ID in the graph
            
        Returns:
            Filtered graph
        """
        G = self.original_graph.copy()
        
        try:
            # Get the weakly connected component containing the node
            component = nx.node_connected_component(G.to_undirected(), node)
            
            # Create a subgraph with just this component
            subgraph = G.subgraph(component).copy()
            
            # Update filtered graph
            self.filtered_graph = subgraph
            return subgraph
            
        except nx.NodeNotFound:
            logger.error(f"Node {node} not found in graph")
            # Keep original graph if error
            self.filtered_graph = G
            return G
    
    def apply_all_filters(self, 
                        confidence_threshold: Optional[float] = None,
                        confidence_comparison: str = ">=",
                        include_directed: bool = True,
                        include_bidirected: bool = True,
                        include_hidden_causes: bool = True,
                        filter_path: bool = False,
                        path_source: Optional[Any] = None,
                        path_target: Optional[Any] = None,
                        path_max_length: Optional[int] = None,
                        include_roots: bool = True,
                        include_leaves: bool = True,
                        include_intermediates: bool = True) -> nx.DiGraph:
        """
        Apply multiple filters in sequence
        
        Args:
            Various filter parameters
            
        Returns:
            Filtered graph
        """
        # Start with original graph
        self.filtered_graph = self.original_graph.copy()
        
        # Apply confidence filter if threshold provided
        if confidence_threshold is not None:
            self.filter_by_confidence(confidence_threshold, confidence_comparison)
        
        # Apply edge type filter
        self.filter_by_edge_type(include_directed, include_bidirected, include_hidden_causes)
        
        # Apply path filter if enabled
        if filter_path and path_source is not None and path_target is not None:
            self.filter_by_path_existence(path_source, path_target, path_max_length)
        
        # Apply node role filter
        self.filter_roots_and_leaves(include_roots, include_leaves, include_intermediates)
        
        return self.filtered_graph


def render_graph_filter_controls(graph: nx.DiGraph):
    """
    Render controls for filtering a causal graph in a Streamlit app
    
    Args:
        graph: NetworkX DiGraph representing the causal structure
        
    Returns:
        Filtered graph based on user selections
    """
    # Create graph filter instance
    graph_filter = GraphFilter(graph)
    
    st.subheader("Graph Filtering Options")
    
    # Confidence threshold
    use_confidence = st.checkbox("Filter by confidence", value=False)
    if use_confidence:
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider(
                "Confidence threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05
            )
        with col2:
            confidence_comparison = st.selectbox(
                "Comparison",
                options=[">=", ">", "<=", "<", "=="],
                index=0
            )
    
    # Edge type filtering
    st.subheader("Edge Types")
    include_directed = st.checkbox("Include directed edges", value=True)
    include_bidirected = st.checkbox("Include bidirected edges", value=True)
    include_hidden_causes = st.checkbox("Include hidden cause edges", value=True)
    
    # Path filtering
    use_path_filter = st.checkbox("Show only specific paths", value=False)
    if use_path_filter:
        # Get node names for selection
        node_names = {}
        for node in graph.nodes():
            if "name" in graph.nodes[node]:
                node_names[node] = graph.nodes[node]["name"]
            else:
                node_names[node] = str(node)
        
        # Create selections for source and target
        col1, col2 = st.columns(2)
        with col1:
            source_node = st.selectbox(
                "Source node",
                options=list(node_names.keys()),
                format_func=lambda x: node_names.get(x, str(x))
            )
        with col2:
            target_node = st.selectbox(
                "Target node",
                options=list(node_names.keys()),
                format_func=lambda x: node_names.get(x, str(x))
            )
        
        path_max_length = st.slider(
            "Maximum path length",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Maximum length of paths to consider (-1 for unlimited)"
        )
    
    # Node role filtering
    st.subheader("Node Types")
    include_roots = st.checkbox("Include root nodes", value=True)
    include_leaves = st.checkbox("Include leaf nodes", value=True)
    include_intermediates = st.checkbox("Include intermediate nodes", value=True)
    
    # Apply button
    if st.button("Apply Filters"):
        # Apply all selected filters
        filtered_graph = graph_filter.apply_all_filters(
            confidence_threshold=confidence_threshold if use_confidence else None,
            confidence_comparison=confidence_comparison if use_confidence else ">=",
            include_directed=include_directed,
            include_bidirected=include_bidirected,
            include_hidden_causes=include_hidden_causes,
            filter_path=use_path_filter,
            path_source=source_node if use_path_filter else None,
            path_target=target_node if use_path_filter else None,
            path_max_length=path_max_length if use_path_filter else None,
            include_roots=include_roots,
            include_leaves=include_leaves,
            include_intermediates=include_intermediates
        )
        
        # Display filtering summary
        original_nodes = graph.number_of_nodes()
        original_edges = graph.number_of_edges()
        filtered_nodes = filtered_graph.number_of_nodes()
        filtered_edges = filtered_graph.number_of_edges()
        
        st.success(f"Filtered graph: {filtered_nodes}/{original_nodes} nodes, {filtered_edges}/{original_edges} edges")
        
        return filtered_graph
    
    # Reset button
    if st.button("Reset Filters"):
        return graph
    
    # By default, return the original graph
    return graph


def filter_causal_graph(graph: nx.DiGraph, 
                      min_confidence: float = 0.0,
                      max_confidence: float = 1.0,
                      include_directed: bool = True,
                      include_bidirected: bool = True,
                      include_hidden_causes: bool = True,
                      simplify: bool = False) -> nx.DiGraph:
    """
    Simple function to filter a causal graph with common options
    
    Args:
        graph: The causal graph to filter
        min_confidence: Minimum confidence threshold
        max_confidence: Maximum confidence threshold
        include_directed: Whether to include regular directed edges
        include_bidirected: Whether to include bidirected edges 
        include_hidden_causes: Whether to include edges from hidden variables
        simplify: Whether to remove nodes with no connections
        
    Returns:
        Filtered causal graph
    """
    # Create filter
    graph_filter = GraphFilter(graph)
    
    # Apply confidence filter
    if min_confidence > 0:
        graph_filter.filter_by_confidence(min_confidence, ">=")
    
    if max_confidence < 1.0:
        graph_filter.filter_by_confidence(max_confidence, "<=")
    
    # Apply edge type filter
    graph_filter.filter_by_edge_type(
        include_directed=include_directed,
        include_bidirected=include_bidirected,
        include_hidden_causes=include_hidden_causes
    )
    
    filtered_graph = graph_filter.filtered_graph
    
    # Remove isolated nodes if simplify is True
    if simplify:
        nodes_to_remove = [node for node in filtered_graph.nodes() 
                          if filtered_graph.degree(node) == 0]
        filtered_graph.remove_nodes_from(nodes_to_remove)
    
    return filtered_graph