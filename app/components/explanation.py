"""
Streamlit components for explanations and interpretations.
"""

import streamlit as st
import networkx as nx
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple


def render_causal_explanation(graph: nx.DiGraph, level: str = "intermediate"):
    """
    Render a causal explanation at specified level.
    
    Args:
        graph: NetworkX DiGraph with the causal structure
        level: Explanation level (basic, intermediate, technical)
    """
    # Basic explanation is suitable for non-technical users
    basic_explanation = """
    This causal graph shows the relationships between variables in your dataset. 
    Arrows indicate that one variable causes another. The thickness of the arrows 
    indicates the confidence in that causal relationship.
    
    Root nodes (variables with no incoming arrows) represent primary causal factors, 
    while leaf nodes (variables with no outgoing arrows) represent final effects.
    """
    
    # Intermediate explanation adds more details about the structure
    intermediate_explanation = f"""
    The causal graph contains {graph.number_of_nodes()} variables and {graph.number_of_edges()} causal relationships.
    
    The structure reveals:
    - Root causes: {', '.join([n for n, d in graph.in_degree() if d == 0]) or 'None'}
    - Final effects: {', '.join([n for n, d in graph.out_degree() if d == 0]) or 'None'}
    - Key mediators: {', '.join([n for n in graph.nodes() if graph.in_degree(n) > 0 and graph.out_degree(n) > 0]) or 'None'}
    
    The graph captures both direct effects and indirect effects through mediating variables.
    Confidence in the edges varies, with darker/thicker arrows indicating stronger relationships.
    """
    
    # Technical explanation includes graph theory metrics and causal assumptions
    technical_explanation = f"""
    ## Technical Details
    
    **Graph Properties:**
    - Nodes: {graph.number_of_nodes()}
    - Edges: {graph.number_of_edges()}
    - Density: {nx.density(graph):.4f}
    - Average in-degree: {sum(d for _, d in graph.in_degree()) / graph.number_of_nodes():.2f}
    - Average out-degree: {sum(d for _, d in graph.out_degree()) / graph.number_of_nodes():.2f}
    
    **Centrality Measures:**
    Top 3 by degree centrality:
    {', '.join([f"{n} ({v:.3f})" for n, v in sorted(nx.degree_centrality(graph).items(), key=lambda x: x[1], reverse=True)[:3]])}
    
    **Paths and Connectivity:**
    The graph has {1 if nx.is_weakly_connected(graph) else nx.number_weakly_connected_components(graph)} weakly connected component(s).
    
    **Causal Interpretation:**
    The graph represents a Directed Acyclic Graph (DAG) where edges represent causal relationships
    under the causal Markov condition and faithfulness assumptions. 
    The direction of edges indicates the direction of causality from cause to effect.
    """
    
    # Display the appropriate explanation
    if level.lower() == "basic":
        st.markdown(basic_explanation)
    elif level.lower() == "intermediate":
        st.markdown(intermediate_explanation)
    else:  # technical
        st.markdown(technical_explanation)


def render_path_explanation(graph: nx.DiGraph, source: str, target: str):
    """
    Explain causal paths from source to target.
    
    Args:
        graph: NetworkX DiGraph with the causal structure
        source: Source node
        target: Target node
    """
    if source not in graph.nodes() or target not in graph.nodes():
        st.warning(f"Either {source} or {target} is not in the graph.")
        return
    
    # Find all paths from source to target
    try:
        all_paths = list(nx.all_simple_paths(graph, source, target))
        
        if not all_paths:
            st.info(f"No causal paths found from {source} to {target}.")
            return
        
        # Group paths by length
        paths_by_length = {}
        for path in all_paths:
            length = len(path) - 1  # Number of edges
            if length not in paths_by_length:
                paths_by_length[length] = []
            paths_by_length[length].append(path)
        
        # Display information about the paths
        st.markdown(f"### Causal Paths from {source} to {target}")
        
        st.markdown(f"Found {len(all_paths)} causal path(s) from {source} to {target}.")
        
        # Direct effect
        if 1 in paths_by_length:
            st.markdown("**Direct Effect:** Present")
        else:
            st.markdown("**Direct Effect:** None")
        
        # Indirect effects
        indirect_paths = sum(len(paths) for length, paths in paths_by_length.items() if length > 1)
        if indirect_paths > 0:
            st.markdown(f"**Indirect Effects:** {indirect_paths} path(s)")
            
            # Collect all mediators
            mediators = set()
            for length, paths in paths_by_length.items():
                if length > 1:
                    for path in paths:
                        mediators.update(path[1:-1])
            
            st.markdown(f"**Mediators:** {', '.join(mediators)}")
        else:
            st.markdown("**Indirect Effects:** None")
        
        # Show the paths
        st.markdown("### Path Details")
        
        for length in sorted(paths_by_length.keys()):
            with st.expander(f"{length}-step paths ({len(paths_by_length[length])})"):
                for i, path in enumerate(paths_by_length[length]):
                    st.markdown(f"Path {i+1}: {' → '.join(path)}")
        
    except nx.NetworkXNoPath:
        st.info(f"No causal paths found from {source} to {target}.")


def render_counterfactual_explanation(intervention_var: str, target_var: str, 
                                    intervention_value: float, effect_size: float,
                                    baseline: float):
    """
    Explain a counterfactual intervention.
    
    Args:
        intervention_var: Intervention variable
        target_var: Target variable
        intervention_value: Value set for intervention
        effect_size: Size of the causal effect
        baseline: Baseline value of target
    """
    # Calculate percentage change
    if baseline != 0:
        effect_percentage = (effect_size / abs(baseline)) * 100
    else:
        effect_percentage = float('inf')
    
    # Determine direction
    if effect_size > 0:
        direction = "increase"
    elif effect_size < 0:
        direction = "decrease"
    else:
        direction = "not change"
    
    # Generate explanation
    st.markdown("### Counterfactual Explanation")
    
    st.markdown(f"""
    Setting {intervention_var} to {intervention_value:.2f} is predicted to {direction} {target_var} 
    by {abs(effect_size):.4f} ({abs(effect_percentage):.2f}%).
    
    - **Original value (baseline):** {baseline:.4f}
    - **New predicted value:** {baseline + effect_size:.4f}
    - **Change:** {effect_size:+.4f} ({effect_percentage:+.2f}%)
    
    This prediction is based on the causal relationships identified in the graph and
    assumes the absence of unmeasured confounders that might affect both variables.
    """)
    
    # Add interpretation of effect size
    if abs(effect_percentage) < 5:
        st.markdown("This represents a **minimal effect** that might not be practically significant.")
    elif abs(effect_percentage) < 20:
        st.markdown("This represents a **moderate effect** that could be practically relevant.")
    else:
        st.markdown("This represents a **substantial effect** that is likely to be practically significant.")


def render_hidden_variables_explanation(hidden_variables: List[Dict[str, Any]]):
    """
    Explain hypothesized hidden variables.
    
    Args:
        hidden_variables: List of dictionaries describing hidden variables
    """
    if not hidden_variables:
        st.info("No hidden variables have been identified.")
        return
    
    st.markdown("### Hidden Variables Analysis")
    
    st.markdown(f"""
    The causal analysis has identified {len(hidden_variables)} potential hidden variable(s) 
    that might influence the observed relationships. Hidden variables represent unmeasured 
    factors that could be acting as confounders or mediators.
    """)
    
    # Show each hidden variable
    for i, var in enumerate(hidden_variables):
        var_name = var.get("name", f"Hidden Variable {i+1}")
        relationship = var.get("relationship_type", "unknown").capitalize()
        confidence = var.get("confidence", 0)
        reasoning = var.get("reasoning", "No reasoning provided")
        connected = var.get("connected_variables", [])
        
        st.markdown(f"#### {var_name}")
        st.markdown(f"**Type:** {relationship}")
        st.markdown(f"**Confidence:** {confidence}/10")
        st.markdown(f"**Connected to:** {', '.join(connected)}")
        st.markdown(f"**Reasoning:** {reasoning}")
        
        # Add a divider if not the last variable
        if i < len(hidden_variables) - 1:
            st.markdown("---")


def render_graph_export_options(graph: nx.DiGraph, explanation_text: str):
    """
    Render export options for the graph and explanations.
    
    Args:
        graph: NetworkX DiGraph with the causal structure
        explanation_text: Explanation text to export
    """
    st.subheader("Export Options")
    
    export_format = st.selectbox(
        "Export Format:",
        options=["Markdown", "Graph (GML)", "Graph (GraphML)"],
        index=0
    )
    
    if export_format == "Markdown":
        # Add graph structure as text
        export_content = explanation_text
        
        export_content += "\n\n## Graph Structure\n\n"
        export_content += "### Nodes\n"
        for node in sorted(graph.nodes()):
            export_content += f"- {node}\n"
        
        export_content += "\n### Edges\n"
        for u, v, data in graph.edges(data=True):
            confidence = data.get('confidence', 'N/A')
            export_content += f"- {u} → {v} (Confidence: {confidence})\n"
        
        st.download_button(
            label="Download Explanation",
            data=export_content,
            file_name="causal_graph_explanation.md",
            mime="text/markdown"
        )
        
    elif export_format == "Graph (GML)":
        # Export as GML
        import io
        from networkx.readwrite import gml
        
        buffer = io.StringIO()
        gml.write_gml(graph, buffer)
        
        st.download_button(
            label="Download Graph (GML)",
            data=buffer.getvalue(),
            file_name="causal_graph.gml",
            mime="text/plain"
        )
        
    elif export_format == "Graph (GraphML)":
        # Export as GraphML
        import io
        from networkx.readwrite import graphml
        
        buffer = io.StringIO()
        graphml.write_graphml(graph, buffer)
        
        st.download_button(
            label="Download Graph (GraphML)",
            data=buffer.getvalue(),
            file_name="causal_graph.graphml",
            mime="application/xml"
        )