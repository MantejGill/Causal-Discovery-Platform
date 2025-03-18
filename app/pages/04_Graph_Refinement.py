import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

# Import core modules
from core.llm.refinement import CausalGraphRefiner, RefinementResult
from core.viz.graph import CausalGraphVisualizer

def initialize_session_state():
    """Ensure session state variables are initialized"""
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

def render_graph_refinement(llm_adapter):
    if not st.session_state.causal_graphs or not st.session_state.current_graph:
        st.warning("Please run causal discovery algorithms first in the 'Causal Discovery' page.")
        return
    
    if not llm_adapter:
        st.warning("LLM adapter not available. Please configure OpenAI API key in Settings.")
        return
    
    st.header("Causal Graph Refinement")
    
    # Let user select which graph to refine
    graph_options = list(st.session_state.causal_graphs.keys())
    selected_graph = st.selectbox(
        "Select graph to refine",
        graph_options,
        index=graph_options.index(st.session_state.current_graph) if st.session_state.current_graph in graph_options else 0
    )
    
    graph_result = st.session_state.causal_graphs[selected_graph]
    
    # Domain knowledge input
    st.subheader("Domain Knowledge")
    
    st.write("""
    You can provide domain knowledge to help the LLM refine the causal graph. 
    This is optional but can significantly improve results.
    """)
    
    domain_knowledge_text = st.text_area(
        "Domain knowledge (optional)",
        height=150,
        help="Describe any domain knowledge about variables and their relationships"
    )
    
    # Parse domain knowledge into structured format
    domain_knowledge = {}
    if domain_knowledge_text:
        domain_knowledge = {
            "description": domain_knowledge_text,
            "variables": {},
            "relationships": []
        }
    
    # Advanced options
    with st.expander("Advanced Options"):
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            help="Minimum confidence level for keeping edges"
        )
        
        discover_hidden = st.checkbox(
            "Discover hidden variables", 
            value=True,
            help="Use LLM to discover potential hidden confounders"
        )
    
    if st.button("Refine Causal Graph"):
        with st.spinner("Refining causal graph with LLM..."):
            try:
                # Initialize graph refiner
                graph_refiner = CausalGraphRefiner(llm_adapter=llm_adapter)
                
                # Get column names for node labels
                node_labels = {i: name for i, name in enumerate(st.session_state.df.columns)}
                
                # Refine graph - access the graph via the dictionary key
                refinement_result = graph_refiner.refine_graph(
                    graph=graph_result["graph"],
                    data=st.session_state.df,
                    domain_knowledge=domain_knowledge,
                    confidence_threshold=confidence_threshold
                )
                
                # Store refined graph - convert to dict format for session state
                refined_id = f"refined_{selected_graph}"
                st.session_state.causal_graphs[refined_id] = {
                    "status": "success",
                    "algorithm_id": refined_id,
                    "graph": refinement_result.graph,
                    "refinement_info": refinement_result.to_dict(),
                    "original_graph_id": selected_graph
                }
                
                st.session_state.current_graph = refined_id
                st.session_state.refined_graph = refinement_result.to_dict()
                
                st.success(f"Successfully refined causal graph")
                
            except Exception as e:
                st.error(f"Error refining graph: {str(e)}")
    
    # Visualize refined graph if available
    if 'refined_graph' in st.session_state and st.session_state.refined_graph:
        st.header("Refined Graph Visualization")
        
        # Visualize graph
        graph_viz = CausalGraphVisualizer()
        
        # Get column names for node labels
        node_labels = {i: name for i, name in enumerate(st.session_state.df.columns)}
        
        # If there are hidden variables, update node labels
        refinement_info = st.session_state.refined_graph
        if "hidden_var_mapping" in refinement_info:
            for hidden_name, hidden_id in refinement_info["hidden_var_mapping"].items():
                node_labels[hidden_id] = hidden_name
        
        with st.spinner("Creating refined graph visualization..."):
            try:
                fig = graph_viz.visualize_graph(
                    graph=st.session_state.causal_graphs[st.session_state.current_graph]["graph"],
                    node_labels=node_labels,
                    edge_weights=True,
                    layout_type="spring",
                    show_confidence=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error visualizing refined graph: {str(e)}")
        
        # Compare with original graph
        st.subheader("Compare with Original Graph")
        
        original_graph_id = st.session_state.causal_graphs[st.session_state.current_graph].get("original_graph_id")
        if original_graph_id and original_graph_id in st.session_state.causal_graphs:
            with st.spinner("Creating comparison visualization..."):
                try:
                    # Create side-by-side visualization instead of using comparison method
                    original_graph = st.session_state.causal_graphs[original_graph_id]["graph"]
                    refined_graph = st.session_state.causal_graphs[st.session_state.current_graph]["graph"]
                    
                    # Visualize original graph
                    orig_fig = graph_viz.visualize_graph(
                        graph=original_graph,
                        node_labels=node_labels,
                        layout_type="spring",
                        show_confidence=False
                    )
                    orig_fig.update_layout(title="Original Graph")
                    
                    # Visualize refined graph
                    ref_fig = graph_viz.visualize_graph(
                        graph=refined_graph,
                        node_labels=node_labels,
                        layout_type="spring",
                        show_confidence=True
                    )
                    ref_fig.update_layout(title="Refined Graph")
                    
                    # Display side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(orig_fig, use_container_width=True)
                    with col2:
                        st.plotly_chart(ref_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating comparison visualization: {str(e)}")
        
        # Edge validation details
        if "edge_validations" in refinement_info:
            st.subheader("Edge Validation Details")
            
            validations = refinement_info["edge_validations"]
            
            for edge, validation in validations.items():
                # Convert the tuple key string back to source and target
                try:
                    if isinstance(edge, str):
                        # Handle the case where the edge might be stored as a string representation of a tuple
                        edge_eval = eval(edge)
                        source_id, target_id = edge_eval if isinstance(edge_eval, tuple) else (None, None)
                    else:
                        source_id, target_id = edge
                        
                    source = validation["source"] if "source" in validation else (node_labels[source_id] if source_id in node_labels else f"Node {source_id}")
                    target = validation["target"] if "target" in validation else (node_labels[target_id] if target_id in node_labels else f"Node {target_id}")
                except Exception as e:
                    # Fallback if there's an issue parsing the edge
                    source = "Unknown"
                    target = "Unknown"
                
                is_valid = validation["is_valid"]
                confidence = validation["confidence"]
                reasoning = validation["reasoning"]
                reverse_suggested = validation.get("reverse_suggested", False)
                
                with st.expander(f"{source} → {target} (Confidence: {confidence:.2f})"):
                    st.write(f"**Valid:** {is_valid}")
                    st.write(f"**Confidence:** {confidence:.2f}")
                    st.write(f"**Reverse Direction Suggested:** {reverse_suggested}")
                    st.write("**Reasoning:**")
                    st.write(reasoning)
        
        # Hidden variables
        if "hidden_variables" in refinement_info and refinement_info["hidden_variables"]:
            st.subheader("Discovered Hidden Variables")
            
            for hidden_var in refinement_info["hidden_variables"]:
                with st.expander(f"{hidden_var['name']} (Confidence: {hidden_var['confidence']:.2f})"):
                    st.write(f"**Description:** {hidden_var['description']}")
                    st.write("**Affects:**")
                    for child in hidden_var["children"]:
                        st.write(f"- {child}")
                    st.write(f"**Confidence:** {hidden_var['confidence']:.2f}")

def main():
    initialize_session_state()
    st.title("LLM-Based Graph Refinement")
    render_graph_refinement(st.session_state.llm_adapter)

if __name__ == "__main__":
    main()