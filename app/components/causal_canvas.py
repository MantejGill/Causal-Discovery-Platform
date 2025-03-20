# app/components/causal_canvas.py

import streamlit as st
import streamlit_elements as elements
from streamlit_elements import dashboard, mui, html, editor, media, lazy, sync, nivo
from streamlit_elements import react_flow as rf
import json
import uuid
from typing import Dict, List, Any, Optional
import pandas as pd
import networkx as nx

class CausalCanvas:
    """
    Implementation of the drag-and-drop canvas for the Causal Playground.
    Uses Streamlit Elements to create a React Flow-based interface.
    """
    
    def __init__(self, key: str = "causal_canvas"):
        """Initialize the causal canvas with a unique key"""
        self.key = key
        self.nodes_key = f"{key}_nodes"
        self.edges_key = f"{key}_edges"
        self.selected_node_key = f"{key}_selected_node"
        
        # Initialize session state for the canvas
        if self.nodes_key not in st.session_state:
            st.session_state[self.nodes_key] = []
        if self.edges_key not in st.session_state:
            st.session_state[self.edges_key] = []
        if self.selected_node_key not in st.session_state:
            st.session_state[self.selected_node_key] = None
    
    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """Get the current nodes on the canvas"""
        return st.session_state[self.nodes_key]
    
    @nodes.setter
    def nodes(self, value: List[Dict[str, Any]]):
        """Set the nodes on the canvas"""
        st.session_state[self.nodes_key] = value
    
    @property
    def edges(self) -> List[Dict[str, Any]]:
        """Get the current edges on the canvas"""
        return st.session_state[self.edges_key]
    
    @edges.setter
    def edges(self, value: List[Dict[str, Any]]):
        """Set the edges on the canvas"""
        st.session_state[self.edges_key] = value
    
    @property
    def selected_node(self) -> Optional[str]:
        """Get the currently selected node ID"""
        return st.session_state[self.selected_node_key]
    
    @selected_node.setter
    def selected_node(self, value: Optional[str]):
        """Set the currently selected node ID"""
        st.session_state[self.selected_node_key] = value
    
    def load_experiment(self, experiment_data: Dict[str, Any]):
        """Load an experiment into the canvas"""
        self.nodes = experiment_data.get("nodes", [])
        self.edges = experiment_data.get("edges", [])
        self.selected_node = None
    
    def save_to_experiment(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save the current canvas state to the experiment data"""
        experiment_data["nodes"] = self.nodes
        experiment_data["edges"] = self.edges
        return experiment_data
    
    def handle_nodes_change(self, changes):
        """Handle changes to nodes on the canvas"""
        # This is called when nodes are added, removed, or modified
        for change in changes:
            change_type = change.get("type")
            node_id = change.get("id")
            
            if change_type == "add":
                # A new node was added
                new_node = change.get("item", {})
                self.nodes.append(new_node)
            
            elif change_type == "remove":
                # A node was removed
                self.nodes = [n for n in self.nodes if n["id"] != node_id]
                # Also remove any edges connected to this node
                self.edges = [
                    e for e in self.edges 
                    if e["source"] != node_id and e["target"] != node_id
                ]
            
            elif change_type == "position":
                # A node was moved
                for node in self.nodes:
                    if node["id"] == node_id:
                        node["position"] = change.get("position", {"x": 0, "y": 0})
                        break
    
    def handle_edges_change(self, changes):
        """Handle changes to edges on the canvas"""
        # This is called when edges are added, removed, or modified
        for change in changes:
            change_type = change.get("type")
            edge_id = change.get("id")
            
            if change_type == "add":
                # A new edge was added
                new_edge = change.get("item", {})
                self.edges.append(new_edge)
            
            elif change_type == "remove":
                # An edge was removed
                self.edges = [e for e in self.edges if e["id"] != edge_id]
    
    def handle_node_selection(self, node_id):
        """Handle node selection on the canvas"""
        self.selected_node = node_id
    
    def add_node(self, node_type: str, position: Dict[str, float], data: Dict[str, Any] = None):
        """Add a new node to the canvas"""
        node_id = str(uuid.uuid4())
        
        # Default node data
        node_data = {
            "label": self._get_node_label(node_type),
            "type": node_type,
            "status": "draft"
        }
        
        # Add custom data if provided
        if data:
            node_data.update(data)
        
        # Create the node
        new_node = {
            "id": node_id,
            "type": self._get_node_component_type(node_type),
            "position": position,
            "data": node_data
        }
        
        # Add node to the canvas
        self.nodes.append(new_node)
        
        return node_id
    
    def _get_node_label(self, node_type: str) -> str:
        """Get a human-readable label for a node type"""
        labels = {
            "dataset_loader": "Dataset Loader",
            "synthetic_data": "Synthetic Data",
            "missing_values": "Missing Values",
            "normalization": "Normalization",
            "feature_selection": "Feature Selection",
            "pc_algorithm": "PC Algorithm",
            "fci_algorithm": "FCI Algorithm",
            "lingam": "LiNGAM",
            "ensemble": "Ensemble",
            "llm_refinement": "LLM Refinement",
            "hidden_variables": "Hidden Variables",
            "graph_viz": "Graph Visualization",
            "path_analysis": "Path Analysis",
            "causal_effects": "Causal Effects",
            "counterfactual": "Counterfactual"
        }
        return labels.get(node_type, node_type.replace("_", " ").title())
    
    def _get_node_component_type(self, node_type: str) -> str:
        """
        Get the React Flow component type for a node type.
        This maps our logical node types to the visual components in React Flow.
        """
        # Define component types
        component_map = {
            "dataset_loader": "dataNode",
            "synthetic_data": "dataNode",
            "missing_values": "processNode",
            "normalization": "processNode",
            "feature_selection": "processNode",
            "pc_algorithm": "algorithmNode",
            "fci_algorithm": "algorithmNode",
            "lingam": "algorithmNode",
            "ensemble": "algorithmNode",
            "llm_refinement": "refinementNode",
            "hidden_variables": "refinementNode",
            "graph_viz": "visualizationNode",
            "path_analysis": "visualizationNode",
            "causal_effects": "analysisNode",
            "counterfactual": "analysisNode"
        }
        
        return component_map.get(node_type, "default")
    
    def connect_nodes(self, source_id: str, target_id: str):
        """Connect two nodes with an edge"""
        edge_id = f"e{source_id}-{target_id}"
        
        # Check if edge already exists
        for edge in self.edges:
            if edge["source"] == source_id and edge["target"] == target_id:
                return  # Edge already exists
        
        # Create the edge
        new_edge = {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "type": "smoothstep",  # Use a smooth step edge type
        }
        
        # Add edge to the canvas
        self.edges.append(new_edge)
    
    def render(self, height: int = 600):
        """Render the canvas using Streamlit Elements"""
        # Create a container for the canvas
        with elements(f"{self.key}_container"):
            # Define custom node types
            node_types = {
                "dataNode": lazy(rf.CustomNode)(
                    className="data-node",
                    style={"background": "#2196f3", "color": "white"}
                ),
                "processNode": lazy(rf.CustomNode)(
                    className="process-node",
                    style={"background": "#4caf50", "color": "white"}
                ),
                "algorithmNode": lazy(rf.CustomNode)(
                    className="algorithm-node",
                    style={"background": "#ff9800", "color": "white"}
                ),
                "refinementNode": lazy(rf.CustomNode)(
                    className="refinement-node",
                    style={"background": "#9c27b0", "color": "white"}
                ),
                "visualizationNode": lazy(rf.CustomNode)(
                    className="visualization-node",
                    style={"background": "#e91e63", "color": "white"}
                ),
                "analysisNode": lazy(rf.CustomNode)(
                    className="analysis-node",
                    style={"background": "#795548", "color": "white"}
                ),
                "default": lazy(rf.CustomNode)(
                    className="default-node",
                    style={"background": "#607d8b", "color": "white"}
                )
            }
            
            # Define the React Flow canvas
            with dashboard.Item("react-flow", 0, 0, 12, height // 10):
                with mui.Paper(sx={"height": height}):
                    # Include CSS for the React Flow canvas
                    html.link(
                        href="https://unpkg.com/reactflow@11.5.2/dist/base.css",
                        rel="stylesheet"
                    )
                    
                    # CSS for custom nodes and styling
                    html.style("""
                    .react-flow__node {
                        padding: 10px;
                        border-radius: 5px;
                        width: 150px;
                        font-size: 12px;
                        text-align: center;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    
                    .react-flow__node.selected {
                        box-shadow: 0 0 0 2px #000;
                    }
                    
                    .react-flow__edge path {
                        stroke-width: 2;
                    }
                    
                    .react-flow__edge-path {
                        stroke: #555;
                    }
                    
                    .react-flow__node.executing {
                        animation: pulse 1.5s infinite;
                    }
                    
                    .react-flow__node.completed {
                        border: 2px solid #4caf50;
                    }
                    
                    .react-flow__node.error {
                        border: 2px solid #f44336;
                    }
                    
                    @keyframes pulse {
                        0% {
                            box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.7);
                        }
                        70% {
                            box-shadow: 0 0 0 10px rgba(33, 150, 243, 0);
                        }
                        100% {
                            box-shadow: 0 0 0 0 rgba(33, 150, 243, 0);
                        }
                    }
                    """)
                    
                    # Create the React Flow component
                    rf.ReactFlow(
                        nodes=self.nodes,
                        edges=self.edges,
                        onNodesChange=sync(self.handle_nodes_change),
                        onEdgesChange=sync(self.handle_edges_change),
                        onNodeClick=sync(lambda _, node: self.handle_node_selection(node["id"])),
                        onConnect=sync(lambda params: self.connect_nodes(params["source"], params["target"])),
                        nodeTypes=node_types,
                        fitView=True,
                        deleteKeyCode=["Backspace", "Delete"],
                        minZoom=0.2,
                        maxZoom=4,
                        proOptions={"hideAttribution": True}
                    )
    
    def render_node_properties(self):
        """Render the properties panel for the selected node"""
        if not self.selected_node:
            st.info("Select a node to edit its properties")
            return
        
        # Find the selected node
        node = next((n for n in self.nodes if n["id"] == self.selected_node), None)
        if not node:
            st.warning("Selected node not found")
            return
        
        st.subheader(f"Edit {node['data']['label']} Properties")
        
        # Node type specific properties
        node_type = node["data"]["type"]
        
        # Common properties
        new_label = st.text_input("Label", value=node["data"]["label"])
        
        # Update the node data
        if new_label != node["data"]["label"]:
            for n in self.nodes:
                if n["id"] == node["id"]:
                    n["data"]["label"] = new_label
                    break
        
        # Render type-specific properties
        if node_type == "dataset_loader":
            st.subheader("Dataset Options")
            dataset_type = st.selectbox(
                "Dataset Source",
                ["Upload", "Sample Dataset"],
                index=0 if node["data"].get("source") != "sample" else 1
            )
            
            if dataset_type == "Upload":
                st.file_uploader("Upload dataset", key=f"file_{node['id']}")
            else:
                st.selectbox(
                    "Sample Dataset",
                    ["Sachs", "Boston Housing", "Airfoil", "Galton"],
                    key=f"sample_{node['id']}"
                )
        
        elif node_type in ["pc_algorithm", "fci_algorithm"]:
            st.subheader("Algorithm Parameters")
            st.selectbox(
                "Independence Test",
                ["Fisher Z", "Chi-Square", "G-Square", "KCI"],
                key=f"indep_test_{node['id']}"
            )
            st.slider(
                "Alpha",
                min_value=0.01,
                max_value=0.1,
                value=0.05,
                step=0.01,
                key=f"alpha_{node['id']}"
            )
        
        elif node_type == "graph_viz":
            st.subheader("Visualization Options")
            st.selectbox(
                "Layout",
                ["Spring", "Circular", "Kamada Kawai", "Planar"],
                key=f"layout_{node['id']}"
            )
            st.checkbox(
                "Show Edge Weights",
                value=True,
                key=f"weights_{node['id']}"
            )
            st.checkbox(
                "Show Confidence",
                value=True,
                key=f"confidence_{node['id']}"
            )
        
        # Node execution status
        if "status" in node["data"]:
            st.subheader("Execution Status")
            st.info(f"Status: {node['data']['status']}")
        
        # Delete node button
        if st.button("Delete Node", key=f"delete_{node['id']}"):
            self.nodes = [n for n in self.nodes if n["id"] != node["id"]]
            self.edges = [
                e for e in self.edges 
                if e["source"] != node["id"] and e["target"] != node["id"]
            ]
            self.selected_node = None

# Function to use the canvas in the main app
def render_causal_canvas(experiment_data, height=600):
    """Render the causal canvas with the given experiment data"""
    # Create the canvas
    canvas = CausalCanvas()
    
    # Load the experiment data
    canvas.load_experiment(experiment_data)
    
    # Create a layout with the canvas and properties panel
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Render the canvas
        canvas.render(height=height)
    
    with col2:
        # Render the properties panel
        canvas.render_node_properties()
    
    # Save the updated canvas state to the experiment
    return canvas.save_to_experiment(experiment_data)

# Component palette handler
def handle_component_drag(component_id, experiment_data):
    """Handle dragging a component from the palette onto the canvas"""
    # Create the canvas
    canvas = CausalCanvas()
    
    # Load the experiment data
    canvas.load_experiment(experiment_data)
    
    # Add the new node at a default position
    position = {"x": 100, "y": 100}
    node_id = canvas.add_node(component_id, position)
    
    # Save the updated canvas state to the experiment
    return canvas.save_to_experiment(experiment_data)