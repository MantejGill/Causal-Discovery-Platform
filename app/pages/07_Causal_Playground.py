# app/pages/07_Causal_Playground.py

import streamlit as st
import streamlit_elements as elements
from streamlit_elements import dashboard, mui, html, editor, media, lazy, sync, nivo
import pandas as pd
import networkx as nx
import json
import uuid
from datetime import datetime
import os
from typing import Dict, List, Any, Optional

# Import core modules
from core.viz.graph import CausalGraphVisualizer
from core.data.loader import DataLoader
from core.algorithms.executor import AlgorithmExecutor
from core.algorithms.ensemble import AlgorithmEnsemble
from core.llm.factory import LLMFactory

# Initialize session state variables needed for the playground
if "playground_experiments" not in st.session_state:
    st.session_state.playground_experiments = {}
if "current_experiment_id" not in st.session_state:
    st.session_state.current_experiment_id = None
if "component_registry" not in st.session_state:
    st.session_state.component_registry = {}
if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = {}
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

# Main app
st.set_page_config(
    page_title="Causal Playground",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions for experiment management
def create_new_experiment():
    """Create a new experiment with a unique ID"""
    experiment_id = str(uuid.uuid4())
    st.session_state.playground_experiments[experiment_id] = {
        "name": f"Experiment {len(st.session_state.playground_experiments) + 1}",
        "created_at": datetime.now().isoformat(),
        "nodes": [],
        "edges": [],
        "results": {},
        "status": "draft"  # draft, running, completed, failed
    }
    st.session_state.current_experiment_id = experiment_id
    return experiment_id

def get_current_experiment():
    """Get the current experiment data or create a new one if none exists"""
    if not st.session_state.current_experiment_id or st.session_state.current_experiment_id not in st.session_state.playground_experiments:
        create_new_experiment()
    return st.session_state.playground_experiments[st.session_state.current_experiment_id]

def save_experiment(experiment_data):
    """Save experiment data to session state"""
    if st.session_state.current_experiment_id:
        st.session_state.playground_experiments[st.session_state.current_experiment_id] = experiment_data

def delete_experiment(experiment_id):
    """Delete an experiment by ID"""
    if experiment_id in st.session_state.playground_experiments:
        del st.session_state.playground_experiments[experiment_id]
        if st.session_state.current_experiment_id == experiment_id:
            if st.session_state.playground_experiments:
                st.session_state.current_experiment_id = list(st.session_state.playground_experiments.keys())[0]
            else:
                st.session_state.current_experiment_id = None

def duplicate_experiment(experiment_id):
    """Duplicate an existing experiment"""
    if experiment_id in st.session_state.playground_experiments:
        new_id = str(uuid.uuid4())
        experiment = st.session_state.playground_experiments[experiment_id].copy()
        experiment["name"] = f"{experiment['name']} (Copy)"
        experiment["created_at"] = datetime.now().isoformat()
        st.session_state.playground_experiments[new_id] = experiment
        st.session_state.current_experiment_id = new_id

def get_component_types():
    """Return the available component types for the palette"""
    return {
        "data_sources": [
            {"id": "dataset_loader", "name": "Dataset Loader", "icon": "database"},
            {"id": "synthetic_data", "name": "Synthetic Data", "icon": "schema"}
        ],
        "preprocessing": [
            {"id": "missing_values", "name": "Missing Values", "icon": "healing"},
            {"id": "normalization", "name": "Normalization", "icon": "equalizer"},
            {"id": "feature_selection", "name": "Feature Selection", "icon": "filter_list"}
        ],
        "algorithms": [
            {"id": "pc_algorithm", "name": "PC Algorithm", "icon": "account_tree"},
            {"id": "fci_algorithm", "name": "FCI Algorithm", "icon": "share"},
            {"id": "lingam", "name": "LiNGAM", "icon": "linear_scale"},
            {"id": "ensemble", "name": "Ensemble", "icon": "merge_type"}
        ],
        "refinement": [
            {"id": "llm_refinement", "name": "LLM Refinement", "icon": "psychology"},
            {"id": "hidden_variables", "name": "Hidden Variables", "icon": "visibility_off"}
        ],
        "visualization": [
            {"id": "graph_viz", "name": "Graph Visualization", "icon": "bubble_chart"},
            {"id": "path_analysis", "name": "Path Analysis", "icon": "timeline"}
        ],
        "evaluation": [
            {"id": "causal_effects", "name": "Causal Effects", "icon": "trending_up"},
            {"id": "counterfactual", "name": "Counterfactual", "icon": "compare_arrows"}
        ]
    }

def execute_node(node_id, experiment_data):
    """Execute a single node in the experiment"""
    # Find the node in the experiment
    node = next((n for n in experiment_data["nodes"] if n["id"] == node_id), None)
    if not node:
        return {"status": "error", "message": f"Node {node_id} not found"}
    
    # Find input nodes (nodes that have edges pointing to this node)
    input_edges = [e for e in experiment_data["edges"] if e["target"] == node_id]
    input_nodes = [e["source"] for e in input_edges]
    
    # Check if all input nodes have been executed
    for input_node_id in input_nodes:
        if input_node_id not in experiment_data["results"] or experiment_data["results"][input_node_id]["status"] != "completed":
            return {"status": "waiting", "message": f"Waiting for input node {input_node_id}"}
    
    # Collect input data from parent nodes
    input_data = {e["source"]: experiment_data["results"][e["source"]]["data"] for e in input_edges}
    
    # Execute the node based on its type
    try:
        # This is a simplified execution - in a real implementation, you would call
        # different functions based on the node type and handle the specific parameters
        result = {"status": "completed", "message": f"Executed {node['type']}", "data": {}}
        
        # Store the result
        experiment_data["results"][node_id] = result
        return result
    
    except Exception as e:
        result = {"status": "error", "message": str(e)}
        experiment_data["results"][node_id] = result
        return result

def execute_experiment(experiment_data):
    """Execute all nodes in the experiment in topological order"""
    # Create a directed graph from the experiment
    G = nx.DiGraph()
    for node in experiment_data["nodes"]:
        G.add_node(node["id"])
    for edge in experiment_data["edges"]:
        G.add_edge(edge["source"], edge["target"])
    
    # Check for cycles
    if not nx.is_directed_acyclic_graph(G):
        return {"status": "error", "message": "Experiment contains cycles and cannot be executed"}
    
    # Get topological sorting of nodes
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return {"status": "error", "message": "Could not determine execution order"}
    
    # Execute nodes in topological order
    experiment_data["status"] = "running"
    experiment_data["results"] = {}
    
    for node_id in sorted_nodes:
        result = execute_node(node_id, experiment_data)
        if result["status"] == "error":
            experiment_data["status"] = "failed"
            return {"status": "failed", "message": result["message"]}
    
    experiment_data["status"] = "completed"
    return {"status": "completed", "message": "Experiment executed successfully"}

def explain_result(node_id, experiment_data):
    """Generate an explanation for a node's result using LLM"""
    if not st.session_state.llm_adapter:
        return "LLM adapter not configured. Please set up an LLM provider in Settings."
    
    node = next((n for n in experiment_data["nodes"] if n["id"] == node_id), None)
    if not node:
        return "Node not found"
    
    if node_id not in experiment_data["results"]:
        return "No results available for this node"
    
    result = experiment_data["results"][node_id]
    
    # Create a prompt for the LLM based on the node type and result
    prompt = f"""Explain the results of a {node['type']} node in a causal discovery experiment.
    
    Node parameters: {json.dumps(node.get('data', {}))}
    Result status: {result['status']}
    Result data: {json.dumps(result.get('data', {}))}
    
    Please provide a clear explanation of what this result means, its significance for causal discovery,
    and any recommendations for next steps or improvements.
    """
    
    # Call the LLM
    try:
        response = st.session_state.llm_adapter.complete(
            prompt=prompt,
            system_prompt="You are an expert in causal discovery and data analysis. Explain experiment results clearly and provide helpful insights.",
            temperature=0.3
        )
        return response.get("completion", "Failed to generate explanation")
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# Main UI for the Causal Playground
st.title("🧪 Causal Playground")

# Sidebar for experiment management
with st.sidebar:
    st.header("Experiment Management")
    
    # List existing experiments
    if st.session_state.playground_experiments:
        st.subheader("Your Experiments")
        for exp_id, exp in st.session_state.playground_experiments.items():
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                if st.button(exp["name"], key=f"select_{exp_id}", use_container_width=True):
                    st.session_state.current_experiment_id = exp_id
            with col2:
                if st.button("📋", key=f"duplicate_{exp_id}", help="Duplicate experiment"):
                    duplicate_experiment(exp_id)
            with col3:
                if st.button("🗑️", key=f"delete_{exp_id}", help="Delete experiment"):
                    delete_experiment(exp_id)
    
    # Create new experiment button
    if st.button("Create New Experiment", use_container_width=True):
        create_new_experiment()
    
    # Component palette
    st.header("Component Palette")
    
    component_types = get_component_types()
    
    # Display components by category
    for category, components in component_types.items():
        with st.expander(category.replace("_", " ").title(), expanded=True):
            for component in components:
                st.button(
                    f"{component['name']}",
                    key=f"component_{component['id']}",
                    help=f"Drag to add {component['name']} to canvas",
                    use_container_width=True
                )
    
    # Execution controls
    st.header("Execution Controls")
    
    if st.button("▶️ Run Experiment", use_container_width=True):
        experiment = get_current_experiment()
        result = execute_experiment(experiment)
        st.write(result["message"])
    
    if st.button("🔄 Reset Results", use_container_width=True):
        experiment = get_current_experiment()
        experiment["results"] = {}
        experiment["status"] = "draft"
        save_experiment(experiment)

# Main content area with tabs for different views
tab1, tab2, tab3 = st.tabs(["Canvas", "Results", "Export"])

with tab1:
    # Experiment canvas - this is a placeholder that will be replaced by the actual React Flow canvas
    st.subheader("Experiment Canvas")
    
    # Show current experiment info
    experiment = get_current_experiment()
    st.write(f"Current Experiment: {experiment['name']}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Canvas placeholder - in the real implementation, this would be replaced with Streamlit Elements
        st.info("This is where the drag-and-drop canvas will be implemented using Streamlit Elements.")
        # Display the experiment nodes and edges as a placeholder
        st.json({
            "nodes": experiment["nodes"],
            "edges": experiment["edges"]
        })
    
    with col2:
        # Node properties panel
        st.subheader("Node Properties")
        st.info("Select a node to edit its properties")

with tab2:
    # Results view
    st.subheader("Experiment Results")
    
    experiment = get_current_experiment()
    
    if experiment["status"] == "draft":
        st.info("Experiment has not been run yet. Click 'Run Experiment' to execute.")
    elif experiment["status"] == "running":
        st.warning("Experiment is currently running...")
    elif experiment["status"] == "failed":
        st.error("Experiment execution failed. Check the logs for details.")
    elif experiment["status"] == "completed":
        st.success("Experiment completed successfully!")
        
        # Display results for each node
        for node in experiment["nodes"]:
            node_id = node["id"]
            if node_id in experiment["results"]:
                with st.expander(f"{node['data']['label']} ({node_id})"):
                    result = experiment["results"][node_id]
                    st.write(f"Status: {result['status']}")
                    st.write(f"Message: {result.get('message', 'No message')}")
                    
                    # Display data based on node type
                    data = result.get("data", {})
                    if node["type"] == "graph_viz" and "graph" in data:
                        # Display graph visualization
                        st.write("Graph Visualization")
                    elif "data" in data and isinstance(data["data"], pd.DataFrame):
                        # Display dataframe
                        st.dataframe(data["data"])
                    else:
                        # Generic data display
                        st.json(data)
                    
                    # LLM explanation button
                    if st.button("Explain Results", key=f"explain_{node_id}"):
                        explanation = explain_result(node_id, experiment)
                        st.markdown(explanation)

with tab3:
    # Export options
    st.subheader("Export Experiment")
    
    export_format = st.selectbox("Export Format", ["JSON", "Python Script", "Image"])
    
    if st.button("Export"):
        experiment = get_current_experiment()
        if export_format == "JSON":
            st.download_button(
                "Download JSON",
                data=json.dumps(experiment, indent=2),
                file_name=f"{experiment['name'].replace(' ', '_')}.json",
                mime="application/json"
            )
        elif export_format == "Python Script":
            st.info("Python script export is not implemented yet")
        elif export_format == "Image":
            st.info("Image export is not implemented yet")

# Footer
st.divider()
st.markdown("""
This Causal Playground allows you to create, run, and analyze causal experiments through a visual interface.
Drag components from the palette onto the canvas, connect them to build a workflow, and run the experiment to see the results.
""")

# Apply selected theme
if "theme" in st.session_state and st.session_state.theme == "dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)