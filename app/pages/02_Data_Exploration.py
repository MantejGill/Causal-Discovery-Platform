import streamlit as st
import pandas as pd
import numpy as np

# Import core modules
from core.data.profiler import DataProfiler
from core.data.preprocessor import DataPreprocessor
from core.viz.distribution import DataVisualizer
from core.algorithms.selector import AlgorithmSelector

# Ensure session state is initialized
for var in ['data_loaded', 'df', 'metadata', 'data_profile', 'preprocessor', 
            'causal_graphs', 'current_graph', 'refined_graph', 'llm_adapter', 'theme',
            'current_judgments', 'algorithm_suggestions']:
    if var not in st.session_state:
        if var in ['data_loaded']:
            st.session_state[var] = False
        elif var in ['causal_graphs']:
            st.session_state[var] = {}
        elif var in ['theme']:
            st.session_state[var] = "light"
        else:
            st.session_state[var] = None

st.title("Data Exploration")

if not st.session_state.data_loaded:
    st.warning("Please load a dataset first in the 'Data Loading' page.")
else:
    st.header("Data Preprocessing")
    
    preprocess_tab1, preprocess_tab2, preprocess_tab3 = st.tabs([
        "Missing Values", "Normalization", "Feature Selection"
    ])
    
    with preprocess_tab1:
        st.subheader("Handle Missing Values")
        
        missing_method = st.selectbox(
            "Method", 
            ["drop", "mean", "median", "mode", "constant", "knn"],
            help="Method to handle missing values"
        )
        
        columns = st.multiselect(
            "Apply to columns", 
            st.session_state.df.columns.tolist(),
            help="Leave empty to apply to all columns"
        )
        
        if missing_method == "constant":
            fill_value = st.text_input("Fill Value", "0")
            try:
                fill_value = float(fill_value)
            except ValueError:
                fill_value = fill_value  # Keep as string
        else:
            fill_value = None
        
        if st.button("Apply Missing Value Handling"):
            with st.spinner("Processing..."):
                try:
                    columns = columns if columns else None
                    st.session_state.preprocessor.handle_missing_values(
                        method=missing_method,
                        columns=columns,
                        fill_value=fill_value
                    )
                    
                    # Update the dataframe
                    st.session_state.df = st.session_state.preprocessor.get_data()
                    
                    # Show summary
                    summary = st.session_state.preprocessor.get_preprocessing_summary()
                    st.success(f"Successfully processed missing values. Rows removed: {summary['rows_removed']}")
                    
                except Exception as e:
                    st.error(f"Error handling missing values: {str(e)}")
    
    with preprocess_tab2:
        st.subheader("Normalize Data")
        
        norm_method = st.selectbox(
            "Normalization Method", 
            ["standard", "minmax", "robust"],
            help="Method to normalize numeric data"
        )
        
        norm_columns = st.multiselect(
            "Apply to columns", 
            st.session_state.df.select_dtypes(include=[np.number]).columns.tolist(),
            help="Leave empty to apply to all numeric columns"
        )
        
        if st.button("Apply Normalization"):
            with st.spinner("Processing..."):
                try:
                    norm_columns = norm_columns if norm_columns else None
                    st.session_state.preprocessor.normalize_data(
                        method=norm_method,
                        columns=norm_columns
                    )
                    
                    # Update the dataframe
                    st.session_state.df = st.session_state.preprocessor.get_data()
                    
                    st.success(f"Successfully normalized data")
                    
                except Exception as e:
                    st.error(f"Error normalizing data: {str(e)}")
    
    with preprocess_tab3:
        st.subheader("Select Features")
        
        selected_columns = st.multiselect(
            "Select columns to keep", 
            st.session_state.df.columns.tolist(),
            default=st.session_state.df.columns.tolist()
        )
        
        if st.button("Apply Feature Selection"):
            with st.spinner("Processing..."):
                try:
                    st.session_state.preprocessor.select_columns(selected_columns)
                    
                    # Update the dataframe
                    st.session_state.df = st.session_state.preprocessor.get_data()
                    
                    st.success(f"Successfully selected {len(selected_columns)} columns")
                    
                except Exception as e:
                    st.error(f"Error selecting features: {str(e)}")
    
    # Reset preprocessing
    if st.button("Reset All Preprocessing"):
        st.session_state.preprocessor.reset()
        st.session_state.df = st.session_state.preprocessor.get_data()
        st.success("Reset preprocessing to original data")
    
    st.divider()
    
    # Data Visualization
    st.header("Data Visualization")
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "Distribution Plots", "Correlation Matrix", "Scatter Plots", "Pair Plots"
    ])
    
    # Initialize data visualizer
    data_viz = DataVisualizer(st.session_state.df)
    
    with viz_tab1:
        st.subheader("Distribution of Variables")
        
        dist_col = st.selectbox(
            "Select column", 
            st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        dist_type = st.selectbox(
            "Plot type", 
            ["histogram", "box", "violin"]
        )
        
        if dist_col:
            with st.spinner("Creating plot..."):
                try:
                    fig = data_viz.create_distribution_plot(
                        column=dist_col,
                        plot_type=dist_type,
                        kde=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
    
    with viz_tab2:
        st.subheader("Correlation Matrix")
        
        corr_method = st.selectbox(
            "Correlation method", 
            ["pearson", "spearman", "kendall"]
        )
        
        with st.spinner("Creating correlation matrix..."):
            try:
                fig = data_viz.create_correlation_matrix(method=corr_method)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating correlation matrix: {str(e)}")
    
    with viz_tab3:
        st.subheader("Scatter Plots")
        
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y", index=min(1, len(numeric_cols)-1))
        
        color_col = st.selectbox("Color by", ["None"] + st.session_state.df.columns.tolist(), key="scatter_color")
        color_col = None if color_col == "None" else color_col
        
        add_regression = st.checkbox("Add regression line", value=True)
        
        if x_col and y_col:
            with st.spinner("Creating scatter plot..."):
                try:
                    fig = data_viz.create_scatterplot(
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        add_regression=add_regression
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
    
    with viz_tab4:
        st.subheader("Pair Plots")
        
        pair_cols = st.multiselect(
            "Select columns", 
            st.session_state.df.select_dtypes(include=[np.number]).columns.tolist(),
            default=st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()[:min(4, len(st.session_state.df.select_dtypes(include=[np.number]).columns))]
        )
        
        pair_color = st.selectbox("Color by", ["None"] + st.session_state.df.columns.tolist(), key="pair_color")
        pair_color = None if pair_color == "None" else pair_color
        
        if pair_cols and st.button("Create Pair Plot"):
            with st.spinner("Creating pair plot (this may take a while for large datasets)..."):
                try:
                    fig = data_viz.create_pairplot(
                        columns=pair_cols,
                        color=pair_color,
                        max_cols=5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating pair plot: {str(e)}")
    
    # Data Profile Summary
    st.header("Data Profile Summary")
    
    if st.session_state.data_profile:
        profile = st.session_state.data_profile
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", profile["n_samples"])
        with col2:
            st.metric("Features", profile["n_features"])
        with col3:
            st.metric("Data Type", profile["overall_type"])
        
        st.subheader("Variable Types")
        st.write(profile["type_counts"])
        
        # Initialize current judgments from profile if not already set
        if st.session_state.current_judgments is None:
            st.session_state.current_judgments = profile["judgments"].copy()
        
        # Function to update algorithm recommendations when judgments change
        def update_recommendations():
            # Create a temporary profile with updated judgments
            updated_profile = profile.copy()
            updated_profile["judgments"] = st.session_state.current_judgments
            
            # Get new algorithm recommendations
            algorithm_selector = AlgorithmSelector()
            st.session_state.algorithm_suggestions = algorithm_selector.suggest_algorithms(updated_profile)
        
        st.subheader("Key Judgments for Algorithm Selection")
        
        # Use multiple columns for the dropdowns
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)
        
        # First row of dropdowns
        with col1:
            st.session_state.current_judgments["prefer_nonparametric"] = st.selectbox(
                "Prefer non-parametric methods",
                [True, False],
                index=0 if st.session_state.current_judgments["prefer_nonparametric"] else 1,
                on_change=update_recommendations
            )
        
        with col2:
            st.session_state.current_judgments["may_have_latent_confounders"] = st.selectbox(
                "May have latent confounders",
                [True, False],
                index=0 if st.session_state.current_judgments["may_have_latent_confounders"] else 1,
                on_change=update_recommendations
            )
        
        # Second row of dropdowns
        with col3:
            st.session_state.current_judgments["prefer_constraint_based"] = st.selectbox(
                "Prefer constraint-based methods",
                [True, False],
                index=0 if st.session_state.current_judgments["prefer_constraint_based"] else 1,
                on_change=update_recommendations
            )
        
        with col4:
            st.session_state.current_judgments["suitable_for_lingam"] = st.selectbox(
                "Suitable for LiNGAM methods",
                [True, False],
                index=0 if st.session_state.current_judgments["suitable_for_lingam"] else 1,
                on_change=update_recommendations
            )
        
        # Third row of dropdowns
        with col5:
            st.session_state.current_judgments["suitable_for_nonlinear_methods"] = st.selectbox(
                "Suitable for nonlinear methods",
                [True, False],
                index=0 if st.session_state.current_judgments["suitable_for_nonlinear_methods"] else 1,
                on_change=update_recommendations
            )
        
        with col6:
            st.session_state.current_judgments["may_be_time_series"] = st.selectbox(
                "May be time series data",
                [True, False],
                index=0 if st.session_state.current_judgments["may_be_time_series"] else 1,
                on_change=update_recommendations
            )
        
        # Add explanation of key judgments
        st.markdown("### Key Judgments Explanation")
        
        with st.expander("Understanding Key Judgments", expanded=True):
            st.markdown("""
            Each judgment affects which causal discovery algorithms are recommended:
            
            - **Prefer non-parametric methods**: Select if your data doesn't follow standard distributions or contains outliers. Non-parametric methods make fewer assumptions about the data distribution.
            
            - **May have latent confounders**: Select if you suspect there are unmeasured variables that influence multiple observed variables. This will prioritize algorithms like FCI that can handle hidden confounders.
            
            - **Prefer constraint-based methods**: Select if you want algorithms that rely on conditional independence tests (like PC, FCI) rather than score-based optimization. These are often more interpretable but can be sensitive to errors in independence tests.
            
            - **Suitable for LiNGAM methods**: Select if your data has continuous, non-Gaussian distributions. LiNGAM methods exploit non-Gaussianity to determine causal direction.
            
            - **Suitable for nonlinear methods**: Select if relationships between variables are likely nonlinear. This will prioritize kernel-based methods and specific nonlinear algorithms.
            
            - **May be time series data**: Select if your data has a temporal ordering or represents measurements over time. This will prioritize time-series specific methods like Granger causality or VAR-LiNGAM.
            """)
        
        # Update recommendations if they haven't been initialized yet
        if st.session_state.algorithm_suggestions is None:
            update_recommendations()
        
        # Display algorithm recommendations
        st.subheader("Recommended Algorithms")
        
        algorithm_suggestions = st.session_state.algorithm_suggestions
        
        if algorithm_suggestions["primary"]:
            st.markdown("**Primary recommendations:**")
            for algo in algorithm_suggestions["primary"]:
                st.markdown(f"* {algo}")
        else:
            st.markdown("**No primary recommendations based on current judgments.**")
        
        if algorithm_suggestions["secondary"]:
            st.markdown("**Secondary recommendations:**")
            for algo in algorithm_suggestions["secondary"]:
                st.markdown(f"* {algo}")
        else:
            st.markdown("**No secondary recommendations based on current judgments.**")
        
        if algorithm_suggestions["not_recommended"]:
            st.markdown("**Not recommended for this dataset:**")
            for algo in algorithm_suggestions["not_recommended"]:
                st.markdown(f"* {algo}")
        
        # Add a button to reset judgments to original values
        if st.button("Reset to Original Judgments"):
            st.session_state.current_judgments = profile["judgments"].copy()
            update_recommendations()
            st.experimental_rerun()
    else:
        st.info("Data profile not available. Please load or preprocess data first.")