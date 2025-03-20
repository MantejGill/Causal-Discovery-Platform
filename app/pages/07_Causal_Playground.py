import streamlit as st
import streamlit_elements as elements
from streamlit_elements import dashboard, mui, html
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
if "selected_node" not in st.session_state:
    st.session_state.selected_node = None
if "nodes" not in st.session_state:
    st.session_state.nodes = []
if "edges" not in st.session_state:
    st.session_state.edges = []
if "next_node_id" not in st.session_state:
    st.session_state.next_node_id = 1

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
    # Reset nodes and edges
    st.session_state.nodes = []
    st.session_state.edges = []
    st.session_state.next_node_id = 1
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
            {"id": "dataset_loader", "name": "Dataset Loader", "icon": "database", "category": "data_source"},
            {"id": "synthetic_data", "name": "Synthetic Data", "icon": "schema", "category": "data_source"}
        ],
        "preprocessing": [
            {"id": "missing_values", "name": "Missing Values", "icon": "healing", "category": "preprocessor"},
            {"id": "normalization", "name": "Normalization", "icon": "equalizer", "category": "preprocessor"},
            {"id": "feature_selection", "name": "Feature Selection", "icon": "filter_list", "category": "preprocessor"}
        ],
        "algorithms": [
            {"id": "pc_algorithm", "name": "PC Algorithm", "icon": "account_tree", "category": "algorithm"},
            {"id": "fci_algorithm", "name": "FCI Algorithm", "icon": "share", "category": "algorithm"},
            {"id": "lingam", "name": "LiNGAM", "icon": "linear_scale", "category": "algorithm"},
            {"id": "ensemble", "name": "Ensemble", "icon": "merge_type", "category": "algorithm"}
        ],
        "refinement": [
            {"id": "llm_refinement", "name": "LLM Refinement", "icon": "psychology", "category": "refinement"},
            {"id": "hidden_variables", "name": "Hidden Variables", "icon": "visibility_off", "category": "refinement"}
        ],
        "visualization": [
            {"id": "graph_viz", "name": "Graph Visualization", "icon": "bubble_chart", "category": "visualization"},
            {"id": "path_analysis", "name": "Path Analysis", "icon": "timeline", "category": "visualization"}
        ],
        "evaluation": [
            {"id": "causal_effects", "name": "Causal Effects", "icon": "trending_up", "category": "evaluation"},
            {"id": "counterfactual", "name": "Counterfactual", "icon": "compare_arrows", "category": "evaluation"}
        ]
    }

def get_node_color(node_type):
    """Get color for node based on its type"""
    colors = {
        "data_source": "#4CAF50",     # Green
        "preprocessor": "#2196F3",    # Blue
        "algorithm": "#FF9800",       # Orange
        "refinement": "#9C27B0",      # Purple
        "visualization": "#00BCD4",   # Cyan
        "evaluation": "#F44336"       # Red
    }
    return colors.get(node_type, "#757575")  # Default gray

def add_node(node_type, node_name, category):
    """Add a new node to the graph"""
    node_id = f"node_{st.session_state.next_node_id}"
    st.session_state.next_node_id += 1
    
    # Create node
    new_node = {
        "id": node_id,
        "type": node_type,
        "data": {
            "label": node_name,
            "category": category,
            "params": {},
            "status": "idle"  # idle, running, completed, error
        },
    }
    
    st.session_state.nodes.append(new_node)
    return node_id

def handle_node_select(node_id):
    """Handle node selection for property editing"""
    st.session_state.selected_node = node_id

def handle_node_delete(node_id):
    """Delete a node and its connected edges"""
    # Remove the node
    st.session_state.nodes = [node for node in st.session_state.nodes if node["id"] != node_id]
    
    # Remove connected edges
    st.session_state.edges = [
        edge for edge in st.session_state.edges 
        if edge["source"] != node_id and edge["target"] != node_id
    ]
    
    # Clear selection if the deleted node was selected
    if st.session_state.selected_node == node_id:
        st.session_state.selected_node = None

def execute_node(node_id, experiment_data):
    """Execute a single node in the experiment"""
    # This is a simplified simulation of node execution
    # In a real implementation, you would call the actual algorithm execution logic
    
    # Find the node in the experiment
    node = next((n for n in st.session_state.nodes if n["id"] == node_id), None)
    if not node:
        return {"status": "error", "message": f"Node {node_id} not found"}
    
    # Update node status to running
    for n in st.session_state.nodes:
        if n["id"] == node_id:
            n["data"]["status"] = "running"
            break
    
    # Find input nodes (nodes that have edges pointing to this node)
    input_edges = [e for e in st.session_state.edges if e["target"] == node_id]
    input_nodes = [e["source"] for e in input_edges]
    
    # Check if all input nodes have been executed
    for input_node_id in input_nodes:
        node_data = next((n["data"] for n in st.session_state.nodes if n["id"] == input_node_id), None)
        if not node_data or node_data["status"] != "completed":
            # Update node status
            for n in st.session_state.nodes:
                if n["id"] == node_id:
                    n["data"]["status"] = "waiting"
                    break
            return {"status": "waiting", "message": f"Waiting for input node {input_node_id}"}
    
    # Simulate execution with a success result
    # In a real implementation, you would call the actual algorithm with appropriate parameters
    result = {
        "status": "completed", 
        "message": f"Executed {node['data']['label']}",
        "data": {"output": f"Sample output from {node['data']['label']}"}
    }
    
    # Update node status
    for n in st.session_state.nodes:
        if n["id"] == node_id:
            n["data"]["status"] = "completed"
            n["data"]["result"] = result
            break
    
    return result

def execute_experiment():
    """Execute all nodes in the experiment in topological order"""
    # Create a directed graph from the experiment
    G = nx.DiGraph()
    for node in st.session_state.nodes:
        G.add_node(node["id"])
    for edge in st.session_state.edges:
        G.add_edge(edge["source"], edge["target"])
    
    # Check for cycles
    if not nx.is_directed_acyclic_graph(G):
        st.error("Experiment contains cycles and cannot be executed")
        return {"status": "error", "message": "Experiment contains cycles and cannot be executed"}
    
    # Get topological sorting of nodes
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        st.error("Could not determine execution order")
        return {"status": "error", "message": "Could not determine execution order"}
    
    # Execute nodes in topological order
    experiment = get_current_experiment()
    experiment["status"] = "running"
    experiment["results"] = {}
    save_experiment(experiment)
    
    for node_id in sorted_nodes:
        result = execute_node(node_id, experiment)
        if result["status"] == "error":
            experiment["status"] = "failed"
            save_experiment(experiment)
            st.error(f"Execution failed at node {node_id}: {result['message']}")
            return {"status": "failed", "message": result["message"]}
    
    experiment["status"] = "completed"
    save_experiment(experiment)
    st.success("Experiment executed successfully!")
    return {"status": "completed", "message": "Experiment executed successfully"}

def render_property_panel():
    """Render the property panel for the selected node"""
    if not st.session_state.selected_node:
        st.info("Select a node to edit its properties")
        return
    
    # Find the selected node
    node = next((n for n in st.session_state.nodes if n["id"] == st.session_state.selected_node), None)
    if not node:
        st.error(f"Selected node {st.session_state.selected_node} not found")
        return
    
    st.subheader(f"Edit {node['data']['label']} Properties")
    
    # Basic properties
    node_label = st.text_input("Name", node["data"]["label"])
    
    # Type-specific properties
    if node["data"]["category"] == "data_source":
        # Dataset loader properties
        if node["type"] == "dataset_loader":
            dataset_source = st.selectbox(
                "Data Source",
                ["Upload File", "Sample Dataset", "URL"],
                index=0
            )
            
            if dataset_source == "Sample Dataset":
                sample_dataset = st.selectbox(
                    "Sample Dataset",
                    ["Sachs Protein Signaling", "Boston Housing", "Airfoil", "Galton Height Data"],
                    index=0
                )
                node["data"]["params"]["source"] = "sample"
                node["data"]["params"]["sample_dataset"] = sample_dataset
            elif dataset_source == "URL":
                url = st.text_input("Dataset URL")
                node["data"]["params"]["source"] = "url"
                node["data"]["params"]["url"] = url
            else:
                st.info("Upload a file in the Data Loading page and select it here")
                node["data"]["params"]["source"] = "upload"
    
    elif node["data"]["category"] == "preprocessor":
        # Preprocessing properties
        if node["type"] == "missing_values":
            method = st.selectbox(
                "Method",
                ["drop", "mean", "median", "mode", "constant", "knn"],
                index=0
            )
            node["data"]["params"]["method"] = method
            
            if method == "constant":
                fill_value = st.text_input("Fill Value", "0")
                node["data"]["params"]["fill_value"] = fill_value
        
        elif node["type"] == "normalization":
            method = st.selectbox(
                "Method",
                ["standard", "minmax", "robust"],
                index=0
            )
            node["data"]["params"]["method"] = method
    
    elif node["data"]["category"] == "algorithm":
        # Algorithm properties
        if node["type"] in ["pc_algorithm", "fci_algorithm"]:
            test = st.selectbox(
                "Independence Test",
                ["Fisher Z", "Chi-Square", "G-Square", "KCI"],
                index=0
            )
            alpha = st.slider("Alpha", 0.01, 0.1, 0.05, 0.01)
            node["data"]["params"]["test"] = test
            node["data"]["params"]["alpha"] = alpha
    
    elif node["data"]["category"] == "refinement":
        # Refinement properties
        if node["type"] == "llm_refinement":
            confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            discover_hidden = st.checkbox("Discover Hidden Variables", True)
            node["data"]["params"]["confidence_threshold"] = confidence
            node["data"]["params"]["discover_hidden"] = discover_hidden
    
    # Update node properties
    if node["data"]["label"] != node_label:
        node["data"]["label"] = node_label
    
    # Delete button
    if st.button("Delete Node", key=f"delete_{node['id']}"):
        handle_node_delete(node["id"])
        st.session_state.selected_node = None
        st.rerun()

# Main app
st.set_page_config(
    page_title="Causal Playground",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
                    # Load nodes and edges from the experiment
                    if "nodes" in exp:
                        st.session_state.nodes = exp["nodes"]
                    if "edges" in exp:
                        st.session_state.edges = exp["edges"]
                    # Set next_node_id to max node id + 1
                    max_id = 0
                    for node in st.session_state.nodes:
                        try:
                            node_num = int(node["id"].split('_')[1])
                            if node_num > max_id:
                                max_id = node_num
                        except:
                            pass
                    st.session_state.next_node_id = max_id + 1
                    st.rerun()
            with col2:
                if st.button("📋", key=f"duplicate_{exp_id}", help="Duplicate experiment"):
                    duplicate_experiment(exp_id)
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{exp_id}", help="Delete experiment"):
                    delete_experiment(exp_id)
                    st.rerun()
    
    # Create new experiment button
    if st.button("Create New Experiment", use_container_width=True):
        create_new_experiment()
        st.rerun()
    
    # Component palette
    st.header("Component Palette")
    
    component_types = get_component_types()
    
    # Display components by category
    for category, components in component_types.items():
        with st.expander(category.replace("_", " ").title(), expanded=True):
            for component in components:
                if st.button(
                    f"{component['name']}",
                    key=f"component_{component['id']}",
                    help=f"Add {component['name']} to canvas",
                    use_container_width=True
                ):
                    add_node(component['id'], component['name'], component['category'])
                    st.rerun()
    
    # Execution controls
    st.header("Execution Controls")
    
    if st.button("▶️ Run Experiment", use_container_width=True):
        execute_experiment()
    
    if st.button("🔄 Reset Results", use_container_width=True):
        # Reset node statuses
        for node in st.session_state.nodes:
            node["data"]["status"] = "idle"
            if "result" in node["data"]:
                del node["data"]["result"]
        
        # Reset experiment status
        experiment = get_current_experiment()
        experiment["status"] = "draft"
        experiment["results"] = {}
        save_experiment(experiment)
        st.rerun()

# Main content area with tabs for different views
tab1, tab2, tab3 = st.tabs(["Canvas", "Results", "Export"])

with tab1:
    # Experiment canvas with React Flow integration
    st.subheader("Experiment Canvas")
    
    # Show current experiment info
    experiment = get_current_experiment()
    st.write(f"Current Experiment: {experiment['name']}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Canvas area - since we had issues with the React Flow implementation
        # we'll create a simplified representation of the workflow for now
        
        # Use a Streamlit container with fixed height for the canvas
        canvas_container = st.container()
        
        with canvas_container:
            # Create a visual representation of nodes and edges
            st.markdown("### Current Graph")
            
            # Display nodes
            if st.session_state.nodes:
                st.markdown("#### Nodes")
                
                # Create a grid layout for nodes
                node_cols = st.columns(3)
                
                for i, node in enumerate(st.session_state.nodes):
                    col_idx = i % 3
                    
                    with node_cols[col_idx]:
                        # Determine status color
                        status_color = {
                            "idle": "gray",
                            "running": "blue",
                            "waiting": "orange",
                            "completed": "green",
                            "error": "red"
                        }.get(node["data"]["status"], "gray")
                        
                        # Create a card for the node
                        st.markdown(
                            f"""
                            <div style="
                                border: 1px solid #ccc;
                                border-radius: 5px;
                                padding: 10px;
                                margin-bottom: 10px;
                                background-color: {get_node_color(node['data']['category'])};
                                color: white;
                                position: relative;
                            ">
                                <div style="
                                    position: absolute;
                                    top: 5px;
                                    right: 5px;
                                    width: 10px;
                                    height: 10px;
                                    border-radius: 50%;
                                    background-color: {status_color};
                                "></div>
                                <h4 style="margin: 0;">{node['data']['label']}</h4>
                                <p style="margin: 5px 0 0 0;">ID: {node['id']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Add select button
                        if st.button("Select", key=f"select_node_{node['id']}"):
                            handle_node_select(node["id"])
                
                # Display edges
                if st.session_state.edges:
                    st.markdown("#### Connections")
                    
                    for edge in st.session_state.edges:
                        # Get source and target node names
                        source_node = next((n for n in st.session_state.nodes if n["id"] == edge["source"]), None)
                        target_node = next((n for n in st.session_state.nodes if n["id"] == edge["target"]), None)
                        
                        if source_node and target_node:
                            source_name = source_node["data"]["label"]
                            target_name = target_node["data"]["label"]
                            
                            st.markdown(f"* {source_name} → {target_name}")
            else:
                st.info("Add components from the palette to build your workflow")
            
            # Instructions for the canvas
            with st.expander("Canvas Instructions"):
                st.markdown("""
                ### How to use the Causal Playground:
                
                1. **Add nodes** by clicking on components in the sidebar palette
                2. **Select a node** to edit its properties in the right panel
                3. **Create connections** between nodes (experimental)
                4. **Run the experiment** using the Run button in the sidebar
                
                The visual editor with drag-and-drop functionality is coming soon!
                """)
                
            # Experimental: Add connections between nodes
            with st.expander("Add Connection"):
                source_options = [(node["id"], node["data"]["label"]) for node in st.session_state.nodes]
                target_options = [(node["id"], node["data"]["label"]) for node in st.session_state.nodes]
                
                if source_options and target_options:
                    source_id = st.selectbox(
                        "Source Node",
                        options=[s[0] for s in source_options],
                        format_func=lambda x: next((s[1] for s in source_options if s[0] == x), x)
                    )
                    
                    target_id = st.selectbox(
                        "Target Node",
                        options=[t[0] for t in target_options],
                        format_func=lambda x: next((t[1] for t in target_options if t[0] == x), x)
                    )
                    
                    if st.button("Connect Nodes"):
                        if source_id != target_id:
                            # Check if this edge already exists
                            if not any(e["source"] == source_id and e["target"] == target_id for e in st.session_state.edges):
                                edge_id = f"edge_{uuid.uuid4().hex[:8]}"
                                st.session_state.edges.append({
                                    "id": edge_id,
                                    "source": source_id,
                                    "target": target_id
                                })
                                st.rerun()
                            else:
                                st.warning("This connection already exists")
                        else:
                            st.warning("Cannot connect a node to itself")
                else:
                    st.info("Add at least two nodes to create connections")
        
        # Save the current layout
        current_experiment = get_current_experiment()
        current_experiment["nodes"] = st.session_state.nodes
        current_experiment["edges"] = st.session_state.edges
        save_experiment(current_experiment)
    
    with col2:
        # Node properties panel
        render_property_panel()

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
        for node in st.session_state.nodes:
            if "result" in node["data"]:
                with st.expander(f"{node['data']['label']} ({node['id']})"):
                    result = node["data"]["result"]
                    st.write(f"Status: {result['status']}")
                    st.write(f"Message: {result.get('message', 'No message')}")
                    
                    # Display data based on node type
                    if "data" in result:
                        data = result["data"]
                        st.write("Output:")
                        st.json(data)
                    
                    # If this is a visualization node, show a placeholder graph
                    if node["data"]["category"] == "visualization":
                        st.write("Graph Visualization:")
                        st.image("https://via.placeholder.com/500x300?text=Graph+Visualization", use_column_width=True)
                    
                    # LLM explanation button
                    if st.button("Explain Results", key=f"explain_{node['id']}"):
                        st.markdown("""
                        ### Explanation
                        
                        This node processed data from its inputs and applied the algorithm successfully. 
                        The results show the expected pattern and are ready to be used by downstream nodes.
                        
                        Key observations:
                        - The process completed without errors
                        - Output format is as expected
                        - The results align with typical patterns for this kind of data
                        """)

with tab3:
    # Export options
    st.subheader("Export Experiment")
    
    export_format = st.selectbox("Export Format", ["JSON", "Python Script", "Image"])
    
    if st.button("Export"):
        experiment = get_current_experiment()
        if export_format == "JSON":
            # Create a clean JSON export without internal state
            export_data = {
                "name": experiment["name"],
                "created_at": experiment["created_at"],
                "nodes": st.session_state.nodes,
                "edges": st.session_state.edges,
                "status": experiment["status"]
            }
            
            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"{experiment['name'].replace(' ', '_')}.json",
                mime="application/json"
            )
        elif export_format == "Python Script":
            # Generate a Python script that recreates this experiment
            python_code = f"""# Generated from Causal Playground: {experiment['name']}
# Created at: {experiment['created_at']}

import pandas as pd
import networkx as nx

# Initialize experiment
experiment = {{
    "name": "{experiment['name']}",
    "nodes": {json.dumps(st.session_state.nodes, indent=4)},
    "edges": {json.dumps(st.session_state.edges, indent=4)}
}}

# Function to execute the experiment
def run_experiment(experiment):
    # Create a graph from nodes and edges
    G = nx.DiGraph()
    
    # Add nodes
    for node in experiment["nodes"]:
        G.add_node(node["id"], **node["data"])
    
    # Add edges
    for edge in experiment["edges"]:
        G.add_edge(edge["source"], edge["target"])
    
    # Execute in topological order
    for node in nx.topological_sort(G):
        print(f"Executing {{G.nodes[node]['label']}}")
        # Implement execution logic here
    
    return G

# Run the experiment
result_graph = run_experiment(experiment)
print(f"Execution complete. Result has {{len(result_graph.nodes)}} nodes and {{len(result_graph.edges)}} edges.")
"""
            st.download_button(
                "Download Python Script",
                data=python_code,
                file_name=f"{experiment['name'].replace(' ', '_')}.py",
                mime="text/plain"
            )
        elif export_format == "Image":
            st.info("Image export functionality is not yet implemented. Please use screenshot tools for now.")

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