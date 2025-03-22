import streamlit as st
import pandas as pd
import numpy as np

# Import core modules
from core.data.profiler import DataProfiler
from core.data.preprocessor import DataPreprocessor
from core.viz.distribution import DataVisualizer
from core.algorithms.selector import AlgorithmSelector
from core.llm.algorithm_recommender import LLMAlgorithmRecommender, display_llm_recommendations

def add_explanation_button(st, viz_type, data_description, viz_description):
    """
    Add an explanation button for a visualization
    
    Args:
        st: Streamlit instance
        viz_type: Type of visualization
        data_description: Description of the data
        viz_description: Description of the visualization
    """
    # Create a unique key for each button
    button_key = f"explain_{viz_type}_{hash(str(viz_description))}"
    
    if st.button(f"🧠 Explain this {viz_type}", key=button_key):
        with st.spinner("Generating AI explanation..."):
            if st.session_state.llm_adapter:
                try:
                    # Get user preferences for explanation
                    detail_level = st.session_state.get("explanation_detail_level", "intermediate")
                    focus = st.session_state.get("explanation_focus", "statistical")
                    
                    # Use OpenRouter adapter if available
                    if hasattr(st.session_state.llm_adapter, "explain_visualization"):
                        explanation = st.session_state.llm_adapter.explain_visualization(
                            viz_type=viz_type,
                            data_description=data_description,
                            viz_description=viz_description,
                            detail_level=detail_level,
                            focus=focus
                        )
                        
                        # Create an attractive container for the explanation
                        with st.expander("📊 **Visualization Explained**", expanded=True):
                            st.markdown(explanation["explanation"])
                            st.caption(f"Explanation generated using {explanation.get('model_used', 'LLM')} • {detail_level.capitalize()} level • {focus.capitalize()} focus")
                    
                    # Fallback to generic LLM adapter
                    else:
                        # Create a prompt for explanation
                        prompt = f"""
                        Please explain this {viz_type} visualization:
                        
                        Data Description:
                        {data_description}
                        
                        Visualization Parameters:
                        {viz_description}
                        
                        Detail Level: {detail_level}
                        Focus: {focus}
                        
                        Provide a clear explanation of what this visualization shows, key patterns or insights,
                        and how to interpret it.
                        """
                        
                        system_prompt = """You are an expert in data visualization and statistics. 
                        Explain visualizations clearly and insightfully at the appropriate level of detail."""
                        
                        # Call the LLM
                        response = st.session_state.llm_adapter.complete(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=0.4
                        )
                        
                        # Display the explanation
                        with st.expander("📊 **Visualization Explained**", expanded=True):
                            st.markdown(response["completion"])
                            st.caption(f"Explanation generated using {st.session_state.llm_adapter.get_name()} • {detail_level.capitalize()} level • {focus.capitalize()} focus")
                
                except Exception as e:
                    st.error(f"Error generating explanation: {str(e)}")
            else:
                st.warning("LLM adapter not available. Please configure OpenAI or OpenRouter API key in Settings.")

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

# Add explanation settings
with st.expander("🧠 AI Explanation Settings", expanded=False):
    st.markdown("Configure how AI-powered visualization explanations are generated:")
    col1, col2 = st.columns(2)
    
    with col1:
        explanation_detail_level = st.selectbox(
            "Detail Level",
            options=["beginner", "intermediate", "advanced"],
            index=1,  # Default to intermediate
            help="Select the level of detail for visualization explanations"
        )
        st.session_state["explanation_detail_level"] = explanation_detail_level
    
    with col2:
        explanation_focus = st.selectbox(
            "Focus Area",
            options=["statistical", "causal", "domain"],
            index=0,  # Default to statistical
            help="Select the focus area for visualization explanations"
        )
        st.session_state["explanation_focus"] = explanation_focus
    
    st.markdown("""
    - **Beginner**: Simple, non-technical explanations
    - **Intermediate**: Moderate technical depth with explanations of concepts
    - **Advanced**: In-depth technical analysis
    
    - **Statistical**: Focus on statistical properties and distributions
    - **Causal**: Focus on potential causal relationships
    - **Domain**: Focus on domain-specific insights and implications
    """)

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
                    # Add explanation button
                    data_description = {
                        "column": dist_col,
                        "data_type": str(st.session_state.df[dist_col].dtype),
                        "summary_stats": {
                            "mean": float(st.session_state.df[dist_col].mean()),
                            "median": float(st.session_state.df[dist_col].median()),
                            "std": float(st.session_state.df[dist_col].std()),
                            "min": float(st.session_state.df[dist_col].min()),
                            "max": float(st.session_state.df[dist_col].max())
                        }
                    }
                    viz_description = {
                        "plot_type": dist_type,
                        "kde": True
                    }
                    add_explanation_button(st, f"{dist_type} plot", data_description, viz_description)

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
                # Add explanation button
                numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
                corr_matrix = st.session_state.df[numeric_cols].corr(method=corr_method).to_dict()
                data_description = {
                    "columns": numeric_cols,
                    "data_types": {col: str(st.session_state.df[col].dtype) for col in numeric_cols}
                }
                viz_description = {
                    "method": corr_method,
                    "correlation_values": corr_matrix
                }
                add_explanation_button(st, "correlation matrix", data_description, viz_description)

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

        # Add option for square aspect ratio
        use_square = st.checkbox("Square plot shape", value=True)

        if x_col and y_col:
            with st.spinner("Creating scatter plot..."):
                try:
                    fig = data_viz.create_scatterplot(
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        add_regression=add_regression
                    )
                    
                    # Set aspect ratio to be equal (square) if selected
                    if use_square:
                        fig.update_layout(
                            height=600,  # You can adjust the size as needed
                            width=600,   # Equal height and width for square shape
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation button
                    data_description = {
                        "x_column": x_col,
                        "y_column": y_col,
                        "color_column": color_col,
                        "x_summary": {
                            "mean": float(st.session_state.df[x_col].mean()),
                            "std": float(st.session_state.df[x_col].std())
                        },
                        "y_summary": {
                            "mean": float(st.session_state.df[y_col].mean()),
                            "std": float(st.session_state.df[y_col].std())
                        },
                        "correlation": float(st.session_state.df[[x_col, y_col]].corr().iloc[0, 1])
                    }
                    viz_description = {
                        "regression_line": add_regression,
                        "color_encoding": color_col is not None,
                        "square_shape": use_square
                    }
                    add_explanation_button(st, "scatter plot", data_description, viz_description)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")

    # Modify the pair plot section in viz_tab4 to persist the plot state across reruns

    with viz_tab4:
        st.subheader("Pair Plots")
        
        # Initialize pair plot state in session state if not already present
        if 'pair_plot_config' not in st.session_state:
            st.session_state.pair_plot_config = None
        
        pair_cols = st.multiselect(
            "Select columns", 
            st.session_state.df.select_dtypes(include=[np.number]).columns.tolist(),
            default=st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()[:min(4, len(st.session_state.df.select_dtypes(include=[np.number]).columns))]
        )
        
        pair_color = st.selectbox("Color by", ["None"] + st.session_state.df.columns.tolist(), key="pair_color")
        pair_color = None if pair_color == "None" else pair_color
        
        # Create pair plot button
        if pair_cols and st.button("Create Pair Plot"):
            # Store configuration in session state when button is clicked
            st.session_state.pair_plot_config = {
                "columns": pair_cols,
                "color": pair_color
            }
        
        # If we have a configuration, create and display the plot
        if st.session_state.pair_plot_config:
            with st.spinner("Creating pair plot (this may take a while for large datasets)..."):
                try:
                    # Get configuration from session state
                    config = st.session_state.pair_plot_config
                    
                    # Create the plot using the stored configuration
                    fig = data_viz.create_pairplot(
                        columns=config["columns"],
                        color=config["color"],
                        max_cols=5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation button (this will still work after reruns)
                    data_description = {
                        "columns": config["columns"],
                        "data_types": {col: str(st.session_state.df[col].dtype) for col in config["columns"]}
                    }
                    viz_description = {
                        "color_by": config["color"],
                        "plot_type": "pairplot"
                    }
                    add_explanation_button(st, "pair plot", data_description, viz_description)
                    
                    # Add a button to clear the plot if desired
                    if st.button("Clear Plot"):
                        st.session_state.pair_plot_config = None
                        st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error creating pair plot: {str(e)}")
                    # Clear the configuration on error to avoid repeating errors
                    st.session_state.pair_plot_config = None
    
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
        
        # We have moved algorithm selection judgments and recommendations to the Causal Discovery page
    else:
        st.info("Data profile not available. Please load or preprocess data first.")