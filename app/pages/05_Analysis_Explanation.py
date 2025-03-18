import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

# Import core modules
from core.viz.graph import CausalGraphVisualizer

# Ensure session state is initialized
for var in ['data_loaded', 'df', 'metadata', 'data_profile', 'preprocessor', 
            'causal_graphs', 'current_graph', 'refined_graph', 'llm_adapter', 'theme']:
    if var not in st.session_state:
        if var in ['data_loaded']:
            st.session_state[var] = False
        elif var in ['causal_graphs']:
            st.session_state[var] = {}
        elif var in ['theme']:
            st.session_state[var] = "light"
        else:
            st.session_state[var] = None

st.title("Causal Analysis & Explanation")

if not st.session_state.causal_graphs or not st.session_state.current_graph:
    st.warning("Please run causal discovery algorithms first in the 'Causal Discovery' page.")
else:
    # Get the current graph
    graph_options = list(st.session_state.causal_graphs.keys())
    selected_graph = st.selectbox(
        "Select graph to analyze",
        graph_options,
        index=graph_options.index(st.session_state.current_graph)
    )
    
    st.session_state.current_graph = selected_graph
    graph_result = st.session_state.causal_graphs[selected_graph]
    G = graph_result["graph"]
    
    # Get column names for node labels
    node_labels = {i: name for i, name in enumerate(st.session_state.df.columns)}
    
    # If there are hidden variables, update node labels
    if "refinement_info" in graph_result and "hidden_var_mapping" in graph_result["refinement_info"]:
        for hidden_name, hidden_id in graph_result["refinement_info"]["hidden_var_mapping"].items():
            node_labels[hidden_id] = hidden_name
    
    # Create tabs for different analysis tools
    tab1, tab2, tab3 = st.tabs(["Causal Paths", "Causal Effects", "Explanations"])
    
    with tab1:
        st.header("Causal Paths Analysis")
        
        # Let user select source and target variables
        col1, col2 = st.columns(2)
        
        with col1:
            source_options = [(node, node_labels.get(node, f"Node {node}")) for node in G.nodes()]
            source_node = st.selectbox(
                "Source variable",
                options=[node for node, _ in source_options],
                format_func=lambda x: next((label for node, label in source_options if node == x), str(x))
            )
        
        with col2:
            target_options = [(node, node_labels.get(node, f"Node {node}")) for node in G.nodes() if node != source_node]
            target_node = st.selectbox(
                "Target variable",
                options=[node for node, _ in target_options],
                format_func=lambda x: next((label for node, label in target_options if node == x), str(x))
            )
        
        if st.button("Find Causal Paths"):
            with st.spinner("Finding causal paths..."):
                try:
                    # Find all paths
                    try:
                        paths = list(nx.all_simple_paths(G, source_node, target_node))
                        
                        # Visualize causal paths
                        graph_viz = CausalGraphVisualizer()
                        
                        # Create a subgraph with the path nodes highlighted
                        subgraph = G.copy()
                        path_edges = []
                        for path in paths:
                            for i in range(len(path) - 1):
                                path_edges.append((path[i], path[i+1]))
                        
                        # Visualize full graph with paths highlighted
                        fig = graph_viz.visualize_graph(
                            graph=G,
                            node_labels=node_labels,
                            layout_type='spring'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if paths:
                            st.subheader("Causal Paths Found")
                            
                            for i, path in enumerate(paths):
                                path_str = " → ".join([node_labels.get(node, f"Node {node}") for node in path])
                                st.write(f"**Path {i+1}:** {path_str}")
                            
                            # Find the shortest path
                            shortest_path = min(paths, key=len)
                            shortest_path_str = " → ".join([node_labels.get(node, f"Node {node}") for node in shortest_path])
                            st.write(f"**Shortest path:** {shortest_path_str}")
                            
                            # If using LLM, generate textual explanation
                            if st.session_state.llm_adapter:
                                with st.spinner("Generating explanation..."):
                                    try:
                                        source_name = node_labels.get(source_node, f"Node {source_node}")
                                        target_name = node_labels.get(target_node, f"Node {target_node}")
                                        
                                        # Fixed: Use the API compatible with both base class and implementations
                                        explanation = st.session_state.llm_adapter.explain_causal_relationship(
                                            from_var=source_name,
                                            to_var=target_name,
                                            graph_data={},  # We'll build this properly later
                                            variable_descriptions={k: f"Variable {v}" for k, v in node_labels.items()},
                                            detail_level="intermediate"
                                        )
                                        
                                        st.subheader("Causal Explanation")
                                        st.write(explanation["explanation"])
                                    
                                    except Exception as e:
                                        st.error(f"Error generating explanation: {str(e)}")
                                        st.write("Try using the LLM in Settings to enable explanation generation.")
                        else:
                            st.write("No causal paths found.")
                            
                    except nx.NetworkXNoPath:
                        st.write("No causal paths found.")
                    
                except Exception as e:
                    st.error(f"Error analyzing causal paths: {str(e)}")
    
    with tab2:
        st.header("Causal Effects Analysis")
        
        st.write("""
        This tool allows you to estimate the causal effect of one variable on another based on the causal graph.
        Note that this is a simple analysis and assumes no unmeasured confounders unless they are explicitly included in the graph.
        """)
        
        # Let user select treatment and outcome variables
        col1, col2 = st.columns(2)
        
        with col1:
            treatment_options = [(node, node_labels.get(node, f"Node {node}")) for node in G.nodes()]
            treatment_node = st.selectbox(
                "Treatment variable",
                options=[node for node, _ in treatment_options],
                format_func=lambda x: next((label for node, label in treatment_options if node == x), str(x)),
                key="treatment_node"
            )
        
        with col2:
            outcome_options = [(node, node_labels.get(node, f"Node {node}")) for node in G.nodes() if node != treatment_node]
            outcome_node = st.selectbox(
                "Outcome variable",
                options=[node for node, _ in outcome_options],
                format_func=lambda x: next((label for node, label in outcome_options if node == x), str(x)),
                key="outcome_node"
            )
        
        # Allow controlling for variables
        control_options = [(node, node_labels.get(node, f"Node {node}")) for node in G.nodes() 
                          if node != treatment_node and node != outcome_node]
        
        control_nodes = st.multiselect(
            "Control for variables (optional)",
            options=[node for node, _ in control_options],
            format_func=lambda x: next((label for node, label in control_options if node == x), str(x))
        )
        
        if st.button("Estimate Causal Effect"):
            with st.spinner("Estimating causal effect..."):
                try:
                    # First check if there's a causal path
                    try:
                        paths = list(nx.all_simple_paths(G, treatment_node, outcome_node))
                        
                        if not paths:
                            st.warning("No causal paths found from treatment to outcome.")
                        
                        # Visualize full graph
                        graph_viz = CausalGraphVisualizer()
                        fig = graph_viz.visualize_graph(
                            graph=G,
                            node_labels=node_labels,
                            layout_type='spring'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # If we have LLM, generate textual explanation
                        if st.session_state.llm_adapter:
                            with st.spinner("Generating causal effect explanation..."):
                                try:
                                    treatment_name = node_labels.get(treatment_node, f"Node {treatment_node}")
                                    outcome_name = node_labels.get(outcome_node, f"Node {outcome_node}")
                                    control_names = [node_labels.get(node, f"Node {node}") for node in control_nodes]
                                    
                                    # Create variable descriptions dictionary
                                    variable_descriptions = {node_labels.get(node, f"Node {node}"): f"Variable {node_labels.get(node, node)}" 
                                                          for node in G.nodes()}
                                    
                                    # Create minimal graph data dictionary
                                    graph_data = {
                                        "nodes": [{"id": n, "name": node_labels.get(n, f"Node {n}")} for n in G.nodes()],
                                        "edges": [{"source": u, "target": v} for u, v in G.edges()]
                                    }
                                    
                                    # Fixed: Use the explain_causal_relationship method directly
                                    effect_explanation = st.session_state.llm_adapter.explain_causal_relationship(
                                        from_var=treatment_name,
                                        to_var=outcome_name,
                                        graph_data=graph_data,
                                        variable_descriptions=variable_descriptions,
                                        domain_context=f"Control variables: {', '.join(control_names)}" if control_names else None,
                                        detail_level="intermediate"
                                    )
                                    
                                    st.subheader("Causal Effect Explanation")
                                    st.write(effect_explanation["explanation"])
                                    
                                    # Also offer technical explanation
                                    with st.expander("Technical Explanation"):
                                        technical_explanation = st.session_state.llm_adapter.explain_causal_relationship(
                                            from_var=treatment_name,
                                            to_var=outcome_name,
                                            graph_data=graph_data,
                                            variable_descriptions=variable_descriptions,
                                            domain_context=f"Control variables: {', '.join(control_names)}" if control_names else None,
                                            detail_level="technical"
                                        )
                                        st.write(technical_explanation["explanation"])
                                
                                except Exception as e:
                                    st.error(f"Error generating explanation: {str(e)}")
                                    st.write("Try using the LLM in Settings to enable explanation generation.")
                        
                    except nx.NetworkXNoPath:
                        st.warning("No causal path exists from treatment to outcome.")
                    
                except Exception as e:
                    st.error(f"Error analyzing causal effect: {str(e)}")
    
    with tab3:
        st.header("Multi-Level Explanations")
        
        st.write("""
        Get natural language explanations of causal relationships at different levels of detail.
        """)
        
        # Let user select edge to explain
        edges = list(G.edges())
        
        if edges:
            edge_options = []
            
            for u, v in edges:
                u_name = node_labels.get(u, f"Node {u}")
                v_name = node_labels.get(v, f"Node {v}")
                edge_options.append(((u, v), f"{u_name} → {v_name}"))
            
            selected_edge = st.selectbox(
                "Select relationship to explain",
                options=[edge for edge, _ in edge_options],
                format_func=lambda x: next((label for edge, label in edge_options if edge == x), str(x))
            )
            
            u, v = selected_edge
            
            # Choose explanation level
            explanation_level = st.radio(
                "Explanation level",
                ["basic", "intermediate", "technical"],
                horizontal=True
            )
            
            if st.button("Generate Explanation"):
                if not st.session_state.llm_adapter:
                    st.warning("LLM adapter not available. Please configure OpenAI API key in Settings.")
                else:
                    with st.spinner("Generating explanation..."):
                        try:
                            u_name = node_labels.get(u, f"Node {u}")
                            v_name = node_labels.get(v, f"Node {v}")
                            
                            # Create minimal graph data dictionary
                            graph_data = {
                                "nodes": [{"id": n, "name": node_labels.get(n, f"Node {n}")} for n in G.nodes()],
                                "edges": [{"source": e[0], "target": e[1]} for e in G.edges()]
                            }
                            
                            # Create variable descriptions dictionary
                            variable_descriptions = {node_labels.get(node, f"Node {node}"): f"Variable {node_labels.get(node, node)}" 
                                                  for node in G.nodes()}
                            
                            # Fixed: Use the explain_causal_relationship method directly
                            explanation_result = st.session_state.llm_adapter.explain_causal_relationship(
                                from_var=u_name,
                                to_var=v_name,
                                graph_data=graph_data,
                                variable_descriptions=variable_descriptions,
                                detail_level=explanation_level
                            )
                            
                            st.subheader(f"{explanation_level.title()} Explanation")
                            st.write(explanation_result["explanation"])
                        
                        except Exception as e:
                            st.error(f"Error generating explanation: {str(e)}")
                            st.write("Try using the LLM in Settings to enable explanation generation.")
        else:
            st.info("No edges in the graph to explain.")