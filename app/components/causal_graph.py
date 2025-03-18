"""
Streamlit components for causal graph visualization.
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Tuple


def render_causal_graph(graph: nx.DiGraph, 
                       show_confidence: bool = True,
                       highlight_nodes: List[str] = None,
                       highlight_edges: List[Tuple[str, str]] = None,
                       title: str = "Causal Graph", 
                       theme: str = "light"):
    """
    Render a causal graph visualization using Plotly.
    
    Args:
        graph: NetworkX DiGraph with the causal structure
        show_confidence: Whether to color edges by confidence
        highlight_nodes: List of node names to highlight
        highlight_edges: List of edges to highlight
        title: Title for the graph
        theme: Color theme (light or dark)
    """
    from core.viz.graph import CausalGraphVisualizer
    
    # Create the visualizer with the selected theme
    visualizer = CausalGraphVisualizer(theme=theme)
    
    # Generate the graph figure
    fig = visualizer.plot_causal_graph(
        graph=graph,
        show_confidence=show_confidence,
        highlight_nodes=highlight_nodes,
        highlight_edges=highlight_edges,
        title=title
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)


def render_graph_statistics(graph: nx.DiGraph):
    """
    Render basic statistics about the causal graph.
    
    Args:
        graph: NetworkX DiGraph with the causal structure
    """
    # Create columns for statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nodes", graph.number_of_nodes())
    
    with col2:
        st.metric("Edges", graph.number_of_edges())
    
    with col3:
        density = nx.density(graph)
        st.metric("Density", f"{density:.3f}")
    
    # Show root and leaf nodes
    st.subheader("Causal Structure")
    
    root_nodes = [n for n, d in graph.in_degree() if d == 0]
    leaf_nodes = [n for n, d in graph.out_degree() if d == 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Root Causes:**")
        if root_nodes:
            for node in root_nodes:
                st.markdown(f"- {node}")
        else:
            st.markdown("None (cyclic graph)")
    
    with col2:
        st.markdown("**Leaf Effects:**")
        if leaf_nodes:
            for node in leaf_nodes:
                st.markdown(f"- {node}")
        else:
            st.markdown("None (cyclic graph)")


def render_edge_list(graph: nx.DiGraph):
    """
    Render a table of edges in the causal graph.
    
    Args:
        graph: NetworkX DiGraph with the causal structure
    """
    import pandas as pd
    
    # Create dataframe of edges
    edges_data = []
    for u, v, data in graph.edges(data=True):
        confidence = data.get('confidence', None)
        validation = data.get('validation', None)
        
        edges_data.append({
            "From": u,
            "To": v,
            "Confidence": f"{confidence:.2f}" if confidence is not None else "N/A",
            "Validated": "Yes" if validation else "No" if validation is False else "N/A"
        })
    
    # Convert to dataframe
    df = pd.DataFrame(edges_data)
    
    # Display the table
    st.dataframe(df, hide_index=True)


def render_graph_comparison(original_graph: nx.DiGraph, refined_graph: nx.DiGraph, theme: str = "light"):
    """
    Render a comparison between two causal graphs.
    
    Args:
        original_graph: Original causal graph
        refined_graph: Refined causal graph
        theme: Color theme (light or dark)
    """
    from core.viz.graph import CausalGraphVisualizer
    
    # Create the visualizer with the selected theme
    visualizer = CausalGraphVisualizer(theme=theme)
    
    # Generate the comparison figure
    fig = visualizer.plot_comparison(
        original_graph=original_graph,
        refined_graph=refined_graph,
        title="Comparison: Original vs. Refined Causal Graph"
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare the two graphs
    st.subheader("Graph Comparison")
    
    original_edges = set(original_graph.edges())
    refined_edges = set(refined_graph.edges())
    
    added_edges = refined_edges - original_edges
    removed_edges = original_edges - refined_edges
    common_edges = original_edges.intersection(refined_edges)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Common Edges", len(common_edges))
    
    with col2:
        st.metric("Added Edges", len(added_edges))
    
    with col3:
        st.metric("Removed Edges", len(removed_edges))
    
    # Display edge details
    if added_edges:
        with st.expander("Added Edges"):
            for u, v in added_edges:
                st.markdown(f"- {u} → {v}")
    
    if removed_edges:
        with st.expander("Removed Edges"):
            for u, v in removed_edges:
                st.markdown(f"- {u} → {v}")