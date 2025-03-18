import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

# Import core modules
from core.algorithms.selector import AlgorithmSelector
from core.algorithms.executor import AlgorithmExecutor
from core.algorithms.ensemble import AlgorithmEnsemble
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

st.title("Causal Discovery")

if not st.session_state.data_loaded:
    st.warning("Please load a dataset first in the 'Data Loading' page.")
else:
    st.header("Algorithm Selection")
    
    # Initialize algorithm selector and executor
    algorithm_selector = AlgorithmSelector()
    algorithm_executor = AlgorithmExecutor()
    
    # Get all available algorithms
    all_algorithms = algorithm_selector.get_all_algorithms()
    
    # Create tabs for different algorithm selection methods
    tab1, tab2, tab3 = st.tabs(["Recommended", "Manual Selection", "Ensemble"])
    
    with tab1:
        st.subheader("Recommended Algorithms")
        
        if st.session_state.data_profile:
            # Get recommended algorithms based on data profile
            algorithm_suggestions = algorithm_selector.suggest_algorithms(st.session_state.data_profile)
            
            # Display primary recommendations
            st.write("#### Primary Recommendations")
            primary_options = algorithm_suggestions["primary"]
            
            if primary_options:
                for alg_id in primary_options:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{alg_id}**: {all_algorithms.get(alg_id, '')}")
                    with col2:
                        if st.button("Run", key=f"run_primary_{alg_id}"):
                            with st.spinner(f"Running {alg_id}..."):
                                try:
                                    result = algorithm_executor.execute_algorithm(alg_id, st.session_state.df)
                                    
                                    if result["status"] == "success":
                                        st.session_state.causal_graphs[alg_id] = result
                                        st.session_state.current_graph = alg_id
                                        st.success(f"Successfully ran {alg_id}")
                                    else:
                                        st.error(f"Error running {alg_id}: {result.get('error', 'Unknown error')}")
                                
                                except Exception as e:
                                    st.error(f"Error running {alg_id}: {str(e)}")
            
            # Display secondary recommendations
            st.write("#### Secondary Recommendations")
            secondary_options = algorithm_suggestions["secondary"]
            
            if secondary_options:
                for alg_id in secondary_options:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{alg_id}**: {all_algorithms.get(alg_id, '')}")
                    with col2:
                        if st.button("Run", key=f"run_secondary_{alg_id}"):
                            with st.spinner(f"Running {alg_id}..."):
                                try:
                                    result = algorithm_executor.execute_algorithm(alg_id, st.session_state.df)
                                    
                                    if result["status"] == "success":
                                        st.session_state.causal_graphs[alg_id] = result
                                        st.session_state.current_graph = alg_id
                                        st.success(f"Successfully ran {alg_id}")
                                    else:
                                        st.error(f"Error running {alg_id}: {result.get('error', 'Unknown error')}")
                                
                                except Exception as e:
                                    st.error(f"Error running {alg_id}: {str(e)}")
        else:
            st.info("Data profile not available. Please go to Data Exploration page first.")
    
    with tab2:
        st.subheader("Manual Algorithm Selection")
        
        # Group algorithms by category
        constraint_based = algorithm_selector.get_algorithms_by_group("constraint_based")
        score_based = algorithm_selector.get_algorithms_by_group("score_based")
        fcm_based = algorithm_selector.get_algorithms_by_group("fcm_based")
        hidden_causal = algorithm_selector.get_algorithms_by_group("hidden_causal")
        granger = algorithm_selector.get_algorithms_by_group("granger")
        
        # Create tabs for each algorithm group
        alg_tab1, alg_tab2, alg_tab3, alg_tab4, alg_tab5 = st.tabs([
            "Constraint-Based", "Score-Based", "FCM-Based", "Hidden Causal", "Granger Causality"
        ])
        
        with alg_tab1:
            st.write("These algorithms use conditional independence tests to infer causal structure.")
            
            for alg_id, description in constraint_based.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{alg_id}**: {description}")
                with col2:
                    if st.button("Run", key=f"run_cb_{alg_id}"):
                        with st.spinner(f"Running {alg_id}..."):
                            try:
                                result = algorithm_executor.execute_algorithm(alg_id, st.session_state.df)
                                
                                if result["status"] == "success":
                                    st.session_state.causal_graphs[alg_id] = result
                                    st.session_state.current_graph = alg_id
                                    st.success(f"Successfully ran {alg_id}")
                                else:
                                    st.error(f"Error running {alg_id}: {result.get('error', 'Unknown error')}")
                            
                            except Exception as e:
                                st.error(f"Error running {alg_id}: {str(e)}")
        
        with alg_tab2:
            st.write("These algorithms use score functions to find the optimal causal graph.")
            
            for alg_id, description in score_based.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{alg_id}**: {description}")
                with col2:
                    if st.button("Run", key=f"run_sb_{alg_id}"):
                        with st.spinner(f"Running {alg_id}..."):
                            try:
                                result = algorithm_executor.execute_algorithm(alg_id, st.session_state.df)
                                
                                if result["status"] == "success":
                                    st.session_state.causal_graphs[alg_id] = result
                                    st.session_state.current_graph = alg_id
                                    st.success(f"Successfully ran {alg_id}")
                                else:
                                    st.error(f"Error running {alg_id}: {result.get('error', 'Unknown error')}")
                            
                            except Exception as e:
                                st.error(f"Error running {alg_id}: {str(e)}")
        
        with alg_tab3:
            st.write("These algorithms use functional causal models for causal discovery.")
            
            for alg_id, description in fcm_based.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{alg_id}**: {description}")
                with col2:
                    if st.button("Run", key=f"run_fcm_{alg_id}"):
                        with st.spinner(f"Running {alg_id}..."):
                            try:
                                result = algorithm_executor.execute_algorithm(alg_id, st.session_state.df)
                                
                                if result["status"] == "success":
                                    st.session_state.causal_graphs[alg_id] = result
                                    st.session_state.current_graph = alg_id
                                    st.success(f"Successfully ran {alg_id}")
                                else:
                                    st.error(f"Error running {alg_id}: {result.get('error', 'Unknown error')}")
                            
                            except Exception as e:
                                st.error(f"Error running {alg_id}: {str(e)}")
        
        with alg_tab4:
            st.write("These algorithms discover hidden causal variables.")
            
            for alg_id, description in hidden_causal.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{alg_id}**: {description}")
                with col2:
                    if st.button("Run", key=f"run_hc_{alg_id}"):
                        with st.spinner(f"Running {alg_id}..."):
                            try:
                                result = algorithm_executor.execute_algorithm(alg_id, st.session_state.df)
                                
                                if result["status"] == "success":
                                    st.session_state.causal_graphs[alg_id] = result
                                    st.session_state.current_graph = alg_id
                                    st.success(f"Successfully ran {alg_id}")
                                else:
                                    st.error(f"Error running {alg_id}: {result.get('error', 'Unknown error')}")
                            
                            except Exception as e:
                                st.error(f"Error running {alg_id}: {str(e)}")
        
        with alg_tab5:
            st.write("These algorithms use Granger causality for time series data.")
            
            for alg_id, description in granger.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{alg_id}**: {description}")
                with col2:
                    if st.button("Run", key=f"run_gr_{alg_id}"):
                        with st.spinner(f"Running {alg_id}..."):
                            try:
                                result = algorithm_executor.execute_algorithm(alg_id, st.session_state.df)
                                
                                if result["status"] == "success":
                                    st.session_state.causal_graphs[alg_id] = result
                                    st.session_state.current_graph = alg_id
                                    st.success(f"Successfully ran {alg_id}")
                                else:
                                    st.error(f"Error running {alg_id}: {result.get('error', 'Unknown error')}")
                            
                            except Exception as e:
                                st.error(f"Error running {alg_id}: {str(e)}")
    
    with tab3:
        st.subheader("Ensemble Method")
        
        st.write("""
        Ensemble methods combine multiple causal discovery algorithms to produce a more robust result.
        Select the algorithms you want to include in the ensemble.
        """)
        
        # Let user select algorithms for ensemble
        ensemble_algs = st.multiselect(
            "Select algorithms for ensemble",
            list(st.session_state.causal_graphs.keys()),
            help="Select at least 2 algorithms to create an ensemble"
        )
        
        edge_threshold = st.slider(
            "Edge Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            help="Minimum frequency of edge appearance across algorithms"
        )
        
        use_llm = st.checkbox(
            "Use LLM to resolve conflicts", 
            value=True,
            help="Use Large Language Model to resolve conflicting edge directions"
        )
        
        if use_llm and not st.session_state.llm_adapter:
            st.warning("LLM adapter not available. Please configure OpenAI API key in Settings.")
        
        if len(ensemble_algs) >= 2 and st.button("Create Ensemble Graph"):
            with st.spinner("Creating ensemble causal graph..."):
                try:
                    # Get algorithm results
                    algorithm_results = [st.session_state.causal_graphs[alg] for alg in ensemble_algs]
                    
                    # Create ensemble
                    ensemble = AlgorithmEnsemble(llm_adapter=st.session_state.llm_adapter if use_llm else None)
                    
                    # Get column names for node labels
                    node_names = {i: name for i, name in enumerate(st.session_state.df.columns)}
                    
                    # Create ensemble graph
                    ensemble_result = ensemble.create_ensemble_graph(
                        algorithm_results=algorithm_results,
                        edge_threshold=edge_threshold,
                        node_names=node_names,
                        resolve_conflicts=use_llm
                    )
                    
                    # Store ensemble result
                    ensemble_id = f"ensemble_{'+'.join(ensemble_algs)}"
                    st.session_state.causal_graphs[ensemble_id] = {
                        "status": "success",
                        "algorithm_id": ensemble_id,
                        "graph": ensemble_result["graph"],
                        "ensemble_info": ensemble_result
                    }
                    
                    st.session_state.current_graph = ensemble_id
                    st.success(f"Successfully created ensemble graph from {len(ensemble_algs)} algorithms")
                    
                except Exception as e:
                    st.error(f"Error creating ensemble graph: {str(e)}")
    
    # Visualize current graph
    st.header("Causal Graph Visualization")
    
    if st.session_state.causal_graphs and st.session_state.current_graph:
        # Let user select which graph to visualize
        graph_options = list(st.session_state.causal_graphs.keys())
        selected_graph = st.selectbox(
            "Select graph to visualize",
            graph_options,
            index=graph_options.index(st.session_state.current_graph)
        )
        
        st.session_state.current_graph = selected_graph
        graph_result = st.session_state.causal_graphs[selected_graph]
        
        # Visualization options
        col1, col2, col3 = st.columns(3)
        with col1:
            layout_type = st.selectbox(
                "Layout", 
                ["spring", "circular", "kamada_kawai", "planar"],
                help="Graph layout algorithm"
            )
        with col2:
            show_weights = st.checkbox("Show edge weights", value=True)
        with col3:
            show_confidence = st.checkbox("Show confidence", value=True)
        
        # Visualize graph
        graph_viz = CausalGraphVisualizer()
        
        # Get column names for node labels
        node_labels = {i: name for i, name in enumerate(st.session_state.df.columns)}
        
        with st.spinner("Creating graph visualization..."):
            try:
                fig = graph_viz.visualize_graph(
                    graph=graph_result["graph"],
                    node_labels=node_labels,
                    edge_weights=show_weights,
                    layout_type=layout_type,
                    show_confidence=show_confidence
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Also show adjacency matrix
                st.subheader("Adjacency Matrix")
                adj_fig = graph_viz.create_adjacency_matrix_plot(
                    graph=graph_result["graph"],
                    node_labels=node_labels
                )
                st.plotly_chart(adj_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error visualizing graph: {str(e)}")
        
        # Algorithm details
        st.subheader("Algorithm Details")
        st.write(f"Algorithm: **{graph_result['algorithm_id']}**")
        
        if "params" in graph_result:
            st.write("Parameters:")
            for param, value in graph_result["params"].items():
                st.write(f"- {param}: {value}")
        
        # Graph statistics
        st.subheader("Graph Statistics")
        G = graph_result["graph"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes", len(G.nodes()))
        with col2:
            st.metric("Edges", len(G.edges()))
        with col3:
            # Calculate average degree
            avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
            st.metric("Avg. Degree", f"{avg_degree:.2f}")
    else:
        st.info("No causal graphs available. Run an algorithm first.")