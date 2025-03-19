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
        # Create tabs for different algorithm groups
        alg_tab1, alg_tab2, alg_tab3, alg_tab4, alg_tab5, alg_tab6 = st.tabs([
            "Constraint-Based", "Score-Based", "FCM-Based", "Hidden Causal", 
            "Granger Causality", "Advanced Methods"  # Added new tab
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
        
        with alg_tab6:
            st.write("These advanced methods support nonlinear relationships, time series, and nonstationarity.")
            
            advanced_method_type = st.selectbox(
                "Advanced method type",
                ["Pairwise Nonlinear", "Time Series", "Nonstationary Data"]
            )
            
            if advanced_method_type == "Pairwise Nonlinear":
                st.write("These methods determine causal direction between two variables with nonlinear relationships.")
                
                # Select two variables
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("First variable", st.session_state.df.columns, key="nonlinear_x")
                with col2:
                    other_cols = [col for col in st.session_state.df.columns if col != x_var]
                    y_var = st.selectbox("Second variable", other_cols, key="nonlinear_y")
                
                # Select nonlinear method
                nonlinear_method = st.selectbox(
                    "Nonlinear method",
                    ["anm", "pnl"],
                    format_func=lambda x: {
                        "anm": "Additive Noise Model (ANM)",
                        "pnl": "Post-Nonlinear Model (PNL)"
                    }.get(x, x)
                )
                
                # Method-specific parameters
                if nonlinear_method == "anm":
                    regression_method = st.selectbox(
                        "Regression method",
                        ["gp"],
                        format_func=lambda x: {
                            "gp": "Gaussian Process Regression"
                        }.get(x, x)
                    )
                    params = {"regression_method": regression_method}
                elif nonlinear_method == "pnl":
                    col1, col2 = st.columns(2)
                    with col1:
                        f1_degree = st.slider("f1 polynomial degree", 1, 5, 3)
                    with col2:
                        f2_degree = st.slider("f2 polynomial degree", 1, 5, 3)
                    
                    independence_test = st.selectbox(
                        "Independence test",
                        ["hsic", "pearson"],
                        format_func=lambda x: {
                            "hsic": "Hilbert-Schmidt Independence Criterion",
                            "pearson": "Pearson Correlation"
                        }.get(x, x)
                    )
                    
                    params = {
                        "f1_degree": f1_degree,
                        "f2_degree": f2_degree,
                        "independence_test": independence_test
                    }
                
                # Button to run nonlinear algorithm
                if st.button("Run Nonlinear Analysis"):
                    with st.spinner(f"Running {nonlinear_method}..."):
                        try:
                            # Get indices of selected variables
                            x_idx = list(st.session_state.df.columns).index(x_var)
                            y_idx = list(st.session_state.df.columns).index(y_var)
                            
                            # Extract data for selected variables
                            selected_data = st.session_state.df[[x_var, y_var]].values
                            
                            # Run algorithm
                            result = algorithm_executor.execute_algorithm(nonlinear_method, 
                                                                    st.session_state.df[[x_var, y_var]], 
                                                                    params)
                            
                            if result["status"] == "success":
                                # Store result
                                st.session_state.causal_graphs[f"{nonlinear_method}_{x_var}_{y_var}"] = result
                                st.session_state.current_graph = f"{nonlinear_method}_{x_var}_{y_var}"
                                
                                # Display result
                                st.success(f"Successfully determined causal direction")
                                
                                # Show direction and confidence
                                causal_result = result["causal_learn_result"]
                                direction = causal_result["direction"]
                                confidence = causal_result.get("confidence", 0.0)
                                
                                direction_str = f"{x_var} → {y_var}" if direction == "0->1" else f"{y_var} → {x_var}"
                                
                                st.write(f"**Causal Direction:** {direction_str}")
                                st.write(f"**Confidence:** {confidence:.3f}")
                                
                                # Visualize result
                                node_labels = {0: x_var, 1: y_var}
                                graph_viz = CausalGraphVisualizer()
                                fig = graph_viz.visualize_graph(
                                    graph=result["graph"],
                                    node_labels=node_labels,
                                    show_confidence=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"Error running {nonlinear_method}: {result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error running {nonlinear_method}: {str(e)}")
            
            elif advanced_method_type == "Time Series":
                st.write("These methods are designed for time-ordered data to discover causal relationships over time.")
                
                # Time series method selection
                ts_method = st.selectbox(
                    "Time series method",
                    ["timeseries_grangervar", "timeseries_granger_pairwise", "var_lingam"],
                    format_func=lambda x: {
                        "timeseries_grangervar": "Vector Autoregression with Granger Causality",
                        "timeseries_granger_pairwise": "Pairwise Granger Causality",
                        "var_lingam": "VAR-LiNGAM (Vector Autoregressive LiNGAM)"
                    }.get(x, x)
                )
                
                # Common parameters
                col1, col2 = st.columns(2)
                with col1:
                    max_lags = st.slider("Maximum Lags", 1, 10, 3)
                with col2:
                    alpha = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
                
                detect_instantaneous = st.checkbox("Detect instantaneous effects", value=True)
                combine_graphs = st.checkbox("Combine temporal and instantaneous graphs", value=True)
                
                # Method-specific parameters
                params = {
                    "lags": max_lags,
                    "alpha": alpha,
                    "detect_instantaneous": detect_instantaneous,
                    "combine_graphs": combine_graphs,
                    "var_names": list(st.session_state.df.columns)
                }
                
                if ts_method == "timeseries_grangervar" or ts_method == "timeseries_granger_pairwise":
                    params["method"] = ts_method.replace("timeseries_", "")
                
                # Button to run time series algorithm
                if st.button("Run Time Series Analysis"):
                    with st.spinner(f"Running {ts_method}..."):
                        try:
                            # Check if data appears to be time-ordered
                            st.info("Note: Ensure your data is properly time-ordered for meaningful time series analysis.")
                            
                            # Run algorithm
                            result = algorithm_executor.execute_algorithm(ts_method, 
                                                                    st.session_state.df, 
                                                                    params)
                            
                            if result["status"] == "success":
                                # Store result
                                ts_id = f"{ts_method}_lag{max_lags}"
                                st.session_state.causal_graphs[ts_id] = result
                                st.session_state.current_graph = ts_id
                                
                                # Display result
                                st.success(f"Successfully ran {ts_method}")
                                
                                # Visualize result
                                node_labels = {i: name for i, name in enumerate(st.session_state.df.columns)}
                                graph_viz = CausalGraphVisualizer()
                                
                                # If available, show temporal visualization
                                try:
                                    fig = graph_viz.visualize_temporal_graph(
                                        graph=result["graph"],
                                        max_lags=max_lags,
                                        node_labels=node_labels
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except:
                                    # Fall back to standard visualization
                                    fig = graph_viz.visualize_graph(
                                        graph=result["graph"],
                                        node_labels=node_labels,
                                        show_confidence=True
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Show instantaneous effects if available
                                if detect_instantaneous and not combine_graphs:
                                    if "instantaneous_graph" in result["causal_learn_result"]:
                                        st.subheader("Instantaneous Effects")
                                        fig_inst = graph_viz.visualize_graph(
                                            graph=result["causal_learn_result"]["instantaneous_graph"],
                                            node_labels=node_labels,
                                            show_confidence=True
                                        )
                                        st.plotly_chart(fig_inst, use_container_width=True)
                            else:
                                st.error(f"Error running {ts_method}: {result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error running {ts_method}: {str(e)}")
            
            elif advanced_method_type == "Nonstationary Data":
                st.write("These methods leverage changes in data distributions across time or domains to discover causal relationships.")
                
                st.info("To use nonstationary methods, you need to provide time/domain indices for your data.")
                
                # Time/domain index selection
                time_index_type = st.radio(
                    "Time/domain index type",
                    ["Generate from column", "Provide manually"]
                )
                
                if time_index_type == "Generate from column":
                    time_col = st.selectbox(
                        "Column containing time/domain information",
                        st.session_state.df.columns
                    )
                    
                    # Preview the unique values
                    unique_values = st.session_state.df[time_col].unique()
                    st.write(f"Unique values detected: {len(unique_values)}")
                    st.write(unique_values[:10])
                    
                    # Generate time indices
                    time_indices = st.session_state.df[time_col].astype('category').cat.codes.values
                else:
                    # Manual input - split points
                    splits = st.number_input(
                        "Number of equal-sized data segments", 
                        min_value=2, 
                        max_value=10,
                        value=3
                    )
                    
                    n_samples = len(st.session_state.df)
                    time_indices = np.zeros(n_samples)
                    
                    # Divide data into equal segments
                    for i in range(1, splits):
                        segment_size = n_samples // splits
                        time_indices[i*segment_size:] = i
                
                # Parameters for nonstationary algorithm
                alpha = st.slider("Significance Level", 0.01, 0.10, 0.05, 0.01, key="ns_alpha")
                
                params = {
                    "time_index": time_indices,
                    "alpha": alpha
                }
                
                # Button to run nonstationary algorithm
                if st.button("Run Nonstationary Analysis"):
                    with st.spinner("Running nonstationary causal discovery..."):
                        try:
                            # Run algorithm
                            result = algorithm_executor.execute_algorithm("nonstationary", 
                                                                    st.session_state.df, 
                                                                    params)
                            
                            if result["status"] == "success":
                                # Store result
                                ns_id = "nonstationary"
                                st.session_state.causal_graphs[ns_id] = result
                                st.session_state.current_graph = ns_id
                                
                                # Display result
                                st.success("Successfully ran nonstationary causal discovery")
                                
                                # Show variable changes
                                st.subheader("Detected Distribution Changes")
                                
                                # Get variable change info
                                var_changes = result["causal_learn_result"].get("variable_changes", {})
                                
                                if var_changes:
                                    # Create table of variable changes
                                    var_change_data = []
                                    for var_idx, info in var_changes.items():
                                        if isinstance(var_idx, int) and var_idx < len(st.session_state.df.columns):
                                            var_name = st.session_state.df.columns[var_idx]
                                            var_change_data.append({
                                                "Variable": var_name,
                                                "Changing": info.get("changing", False),
                                                "p-value": info.get("p_value", 1.0),
                                                "Strength": info.get("strength", 0.0)
                                            })
                                    
                                    if var_change_data:
                                        import pandas as pd
                                        var_change_df = pd.DataFrame(var_change_data)
                                        st.dataframe(var_change_df)
                                
                                # Visualize causal graph
                                node_labels = {i: name for i, name in enumerate(st.session_state.df.columns)}
                                graph_viz = CausalGraphVisualizer()
                                fig = graph_viz.visualize_graph(
                                    graph=result["graph"],
                                    node_labels=node_labels,
                                    show_confidence=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"Error running nonstationary analysis: {result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error running nonstationary analysis: {str(e)}")
    
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