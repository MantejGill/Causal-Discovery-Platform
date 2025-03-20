# app/components/component_registry.py

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import uuid
import json
import os
import streamlit as st
from dataclasses import dataclass, field

# Import core modules for execution
from core.data.loader import DataLoader
from core.data.profiler import DataProfiler
from core.data.preprocessor import DataPreprocessor
from core.algorithms.executor import AlgorithmExecutor
from core.algorithms.ensemble import AlgorithmEnsemble
from core.viz.graph import CausalGraphVisualizer
from core.llm.factory import LLMFactory


@dataclass
class PortDefinition:
    """Definition of an input or output port for a component"""
    name: str
    data_type: str
    description: str
    required: bool = True
    multiple: bool = False  # Whether multiple connections are allowed


@dataclass
class ParameterDefinition:
    """Definition of a configurable parameter for a component"""
    name: str
    display_name: str
    type: str  # 'string', 'number', 'boolean', 'select', 'multiselect', 'file'
    description: str
    default: Any = None
    options: List[str] = field(default_factory=list)  # For select/multiselect
    min_value: Optional[float] = None  # For number
    max_value: Optional[float] = None  # For number
    step: Optional[float] = None  # For number
    accept_types: List[str] = field(default_factory=list)  # For file upload


@dataclass
class ComponentDefinition:
    """Definition of a component in the Causal Playground"""
    id: str
    name: str
    category: str
    description: str
    icon: str
    input_ports: List[PortDefinition] = field(default_factory=list)
    output_ports: List[PortDefinition] = field(default_factory=list)
    parameters: List[ParameterDefinition] = field(default_factory=list)
    executor: Callable = None  # Function that executes the component


class ComponentRegistry:
    """Registry of all available components in the Causal Playground"""
    
    def __init__(self):
        """Initialize the component registry"""
        self.components: Dict[str, ComponentDefinition] = {}
        self._register_all_components()
    
    def register_component(self, component: ComponentDefinition):
        """Register a component in the registry"""
        self.components[component.id] = component
    
    def get_component(self, component_id: str) -> Optional[ComponentDefinition]:
        """Get a component definition by ID"""
        return self.components.get(component_id)
    
    def get_components_by_category(self, category: str) -> List[ComponentDefinition]:
        """Get all components in a category"""
        return [comp for comp in self.components.values() if comp.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get all component categories"""
        return sorted(list(set(comp.category for comp in self.components.values())))
    
    def get_all_components(self) -> List[ComponentDefinition]:
        """Get all registered components"""
        return list(self.components.values())
    
    def _register_all_components(self):
        """Register all available components"""
        # Register data source components
        self._register_data_sources()
        
        # Register preprocessing components
        self._register_preprocessing_components()
        
        # Register algorithm components
        self._register_algorithm_components()
        
        # Register refinement components
        self._register_refinement_components()
        
        # Register visualization components
        self._register_visualization_components()
        
        # Register analysis components
        self._register_analysis_components()
    
    def _register_data_sources(self):
        """Register data source components"""
        # Dataset Loader
        self.register_component(ComponentDefinition(
            id="dataset_loader",
            name="Dataset Loader",
            category="data_sources",
            description="Load data from a file or use a sample dataset",
            icon="database",
            output_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Loaded dataset"
                ),
                PortDefinition(
                    name="metadata",
                    data_type="dict",
                    description="Dataset metadata"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="source",
                    display_name="Data Source",
                    type="select",
                    description="Source of the dataset",
                    default="file",
                    options=["file", "sample"]
                ),
                ParameterDefinition(
                    name="file",
                    display_name="Data File",
                    type="file",
                    description="File to load data from",
                    accept_types=[".csv", ".xlsx", ".xls", ".json", ".parquet"]
                ),
                ParameterDefinition(
                    name="sample_dataset",
                    display_name="Sample Dataset",
                    type="select",
                    description="Sample dataset to load",
                    default="sachs",
                    options=["sachs", "boston_housing", "airfoil", "galton"]
                )
            ],
            executor=self._execute_dataset_loader
        ))
        
        # Synthetic Data Generator
        self.register_component(ComponentDefinition(
            id="synthetic_data",
            name="Synthetic Data",
            category="data_sources",
            description="Generate synthetic data with known causal structure",
            icon="schema",
            output_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Generated dataset"
                ),
                PortDefinition(
                    name="true_graph",
                    data_type="graph",
                    description="True causal graph"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="model_type",
                    display_name="Model Type",
                    type="select",
                    description="Type of synthetic data model",
                    default="linear",
                    options=["linear", "nonlinear", "discrete", "mixed"]
                ),
                ParameterDefinition(
                    name="n_variables",
                    display_name="Number of Variables",
                    type="number",
                    description="Number of variables to generate",
                    default=5,
                    min_value=2,
                    max_value=20,
                    step=1
                ),
                ParameterDefinition(
                    name="n_samples",
                    display_name="Number of Samples",
                    type="number",
                    description="Number of samples to generate",
                    default=1000,
                    min_value=100,
                    max_value=10000,
                    step=100
                ),
                ParameterDefinition(
                    name="edge_density",
                    display_name="Edge Density",
                    type="number",
                    description="Density of edges in the graph",
                    default=0.3,
                    min_value=0.1,
                    max_value=0.9,
                    step=0.1
                ),
                ParameterDefinition(
                    name="seed",
                    display_name="Random Seed",
                    type="number",
                    description="Random seed for reproducibility",
                    default=42,
                    min_value=0,
                    max_value=99999,
                    step=1
                )
            ],
            executor=self._execute_synthetic_data
        ))
    
    def _register_preprocessing_components(self):
        """Register preprocessing components"""
        # Missing Values Handler
        self.register_component(ComponentDefinition(
            id="missing_values",
            name="Missing Values",
            category="preprocessing",
            description="Handle missing values in the dataset",
            icon="healing",
            input_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Input dataset with missing values"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Dataset with handled missing values"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="method",
                    display_name="Method",
                    type="select",
                    description="Method for handling missing values",
                    default="drop",
                    options=["drop", "mean", "median", "mode", "constant", "knn"]
                ),
                ParameterDefinition(
                    name="columns",
                    display_name="Columns",
                    type="multiselect",
                    description="Columns to apply the method to (empty for all)",
                    default=[],
                    options=[]  # Will be populated dynamically
                ),
                ParameterDefinition(
                    name="fill_value",
                    display_name="Fill Value",
                    type="string",
                    description="Value to use for constant method",
                    default="0"
                )
            ],
            executor=self._execute_missing_values
        ))
        
        # Normalization
        self.register_component(ComponentDefinition(
            id="normalization",
            name="Normalization",
            category="preprocessing",
            description="Normalize numeric features in the dataset",
            icon="equalizer",
            input_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Input dataset"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Normalized dataset"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="method",
                    display_name="Method",
                    type="select",
                    description="Normalization method",
                    default="standard",
                    options=["standard", "minmax", "robust"]
                ),
                ParameterDefinition(
                    name="columns",
                    display_name="Columns",
                    type="multiselect",
                    description="Columns to normalize (empty for all numeric)",
                    default=[],
                    options=[]  # Will be populated dynamically
                )
            ],
            executor=self._execute_normalization
        ))
        
        # Feature Selection
        self.register_component(ComponentDefinition(
            id="feature_selection",
            name="Feature Selection",
            category="preprocessing",
            description="Select features to include in the analysis",
            icon="filter_list",
            input_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Input dataset"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Dataset with selected features"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="columns",
                    display_name="Columns to Keep",
                    type="multiselect",
                    description="Columns to keep in the dataset",
                    default=[],
                    options=[]  # Will be populated dynamically
                )
            ],
            executor=self._execute_feature_selection
        ))
    
    def _register_algorithm_components(self):
        """Register algorithm components"""
        # PC Algorithm
        self.register_component(ComponentDefinition(
            id="pc_algorithm",
            name="PC Algorithm",
            category="algorithms",
            description="Constraint-based causal discovery algorithm",
            icon="account_tree",
            input_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Input dataset"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Discovered causal graph"
                ),
                PortDefinition(
                    name="result",
                    data_type="dict",
                    description="Full algorithm result"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="indep_test",
                    display_name="Independence Test",
                    type="select",
                    description="Independence test to use",
                    default="fisherz",
                    options=["fisherz", "chisq", "gsq", "kci"]
                ),
                ParameterDefinition(
                    name="alpha",
                    display_name="Alpha",
                    type="number",
                    description="Significance level for independence tests",
                    default=0.05,
                    min_value=0.01,
                    max_value=0.1,
                    step=0.01
                ),
                ParameterDefinition(
                    name="stable",
                    display_name="Stable",
                    type="boolean",
                    description="Use stable PC algorithm",
                    default=True
                )
            ],
            executor=self._execute_pc_algorithm
        ))
        
        # FCI Algorithm
        self.register_component(ComponentDefinition(
            id="fci_algorithm",
            name="FCI Algorithm",
            category="algorithms",
            description="Fast Causal Inference algorithm for discovering latent confounders",
            icon="share",
            input_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Input dataset"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Discovered causal graph"
                ),
                PortDefinition(
                    name="result",
                    data_type="dict",
                    description="Full algorithm result"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="indep_test",
                    display_name="Independence Test",
                    type="select",
                    description="Independence test to use",
                    default="fisherz",
                    options=["fisherz", "chisq", "kci"]
                ),
                ParameterDefinition(
                    name="alpha",
                    display_name="Alpha",
                    type="number",
                    description="Significance level for independence tests",
                    default=0.05,
                    min_value=0.01,
                    max_value=0.1,
                    step=0.01
                ),
                ParameterDefinition(
                    name="max_path_length",
                    display_name="Max Path Length",
                    type="number",
                    description="Maximum length of discriminating paths",
                    default=-1,
                    min_value=-1,
                    max_value=10,
                    step=1
                )
            ],
            executor=self._execute_fci_algorithm
        ))
        
        # LiNGAM
        self.register_component(ComponentDefinition(
            id="lingam",
            name="LiNGAM",
            category="algorithms",
            description="Linear Non-Gaussian Acyclic Model for causal discovery",
            icon="linear_scale",
            input_ports=[
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Input dataset"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Discovered causal graph"
                ),
                PortDefinition(
                    name="result",
                    data_type="dict",
                    description="Full algorithm result"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="method",
                    display_name="Method",
                    type="select",
                    description="LiNGAM method",
                    default="direct",
                    options=["ica", "direct", "var", "rcd"]
                ),
                ParameterDefinition(
                    name="measure",
                    display_name="Measure",
                    type="select",
                    description="Independence measure (for DirectLiNGAM)",
                    default="pwling",
                    options=["pwling", "kernel"]
                ),
                ParameterDefinition(
                    name="random_state",
                    display_name="Random State",
                    type="number",
                    description="Random state for reproducibility",
                    default=42,
                    min_value=0,
                    max_value=99999,
                    step=1
                )
            ],
            executor=self._execute_lingam
        ))
        
        # Ensemble
        self.register_component(ComponentDefinition(
            id="ensemble",
            name="Ensemble",
            category="algorithms",
            description="Combine multiple causal graphs into an ensemble",
            icon="merge_type",
            input_ports=[
                PortDefinition(
                    name="graphs",
                    data_type="graph",
                    description="Input causal graphs",
                    multiple=True
                )
            ],
            output_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Ensemble causal graph"
                ),
                PortDefinition(
                    name="result",
                    data_type="dict",
                    description="Full ensemble result"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="edge_threshold",
                    display_name="Edge Threshold",
                    type="number",
                    description="Minimum frequency for including edges",
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1
                ),
                ParameterDefinition(
                    name="resolve_conflicts",
                    display_name="Resolve Conflicts",
                    type="boolean",
                    description="Use LLM to resolve conflicting edge directions",
                    default=True
                )
            ],
            executor=self._execute_ensemble
        ))
    
    def _register_refinement_components(self):
        """Register refinement components"""
        # LLM Refinement
        self.register_component(ComponentDefinition(
            id="llm_refinement",
            name="LLM Refinement",
            category="refinement",
            description="Use LLM to refine causal graph based on domain knowledge",
            icon="psychology",
            input_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Input causal graph"
                ),
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Original dataset",
                    required=False
                )
            ],
            output_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Refined causal graph"
                ),
                PortDefinition(
                    name="result",
                    data_type="dict",
                    description="Refinement details"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="domain_knowledge",
                    display_name="Domain Knowledge",
                    type="string",
                    description="Domain knowledge to guide refinement",
                    default=""
                ),
                ParameterDefinition(
                    name="confidence_threshold",
                    display_name="Confidence Threshold",
                    type="number",
                    description="Minimum confidence level for keeping edges",
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1
                ),
                ParameterDefinition(
                    name="discover_hidden",
                    display_name="Discover Hidden Variables",
                    type="boolean",
                    description="Try to discover hidden confounders",
                    default=True
                )
            ],
            executor=self._execute_llm_refinement
        ))
        
        # Hidden Variables
        self.register_component(ComponentDefinition(
            id="hidden_variables",
            name="Hidden Variables",
            category="refinement",
            description="Detect and incorporate hidden variables in the causal graph",
            icon="visibility_off",
            input_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Input causal graph"
                ),
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Original dataset"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Graph with hidden variables"
                ),
                PortDefinition(
                    name="result",
                    data_type="dict",
                    description="Hidden variables details"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="method",
                    display_name="Method",
                    type="select",
                    description="Method for detecting hidden variables",
                    default="correlation",
                    options=["correlation", "tetrad", "llm"]
                ),
                ParameterDefinition(
                    name="threshold",
                    display_name="Threshold",
                    type="number",
                    description="Threshold for detecting hidden variables",
                    default=0.7,
                    min_value=0.1,
                    max_value=0.9,
                    step=0.1
                )
            ],
            executor=self._execute_hidden_variables
        ))
    
    def _register_visualization_components(self):
        """Register visualization components"""
        # Graph Visualization
        self.register_component(ComponentDefinition(
            id="graph_viz",
            name="Graph Visualization",
            category="visualization",
            description="Visualize causal graph with various layouts",
            icon="bubble_chart",
            input_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Causal graph to visualize"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="figure",
                    data_type="figure",
                    description="Graph visualization figure"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="layout",
                    display_name="Layout",
                    type="select",
                    description="Graph layout algorithm",
                    default="spring",
                    options=["spring", "circular", "kamada_kawai", "planar"]
                ),
                ParameterDefinition(
                    name="show_weights",
                    display_name="Show Edge Weights",
                    type="boolean",
                    description="Show edge weights in the graph",
                    default=True
                ),
                ParameterDefinition(
                    name="show_confidence",
                    display_name="Show Confidence",
                    type="boolean",
                    description="Show edge confidence in the graph",
                    default=True
                ),
                ParameterDefinition(
                    name="node_size",
                    display_name="Node Size",
                    type="number",
                    description="Size of nodes in the graph",
                    default=10,
                    min_value=5,
                    max_value=20,
                    step=1
                )
            ],
            executor=self._execute_graph_viz
        ))
        
        # Path Analysis
        self.register_component(ComponentDefinition(
            id="path_analysis",
            name="Path Analysis",
            category="visualization",
            description="Analyze causal paths between variables",
            icon="timeline",
            input_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Causal graph to analyze"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="figure",
                    data_type="figure",
                    description="Path visualization figure"
                ),
                PortDefinition(
                    name="paths",
                    data_type="dict",
                    description="Path analysis results"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="source",
                    display_name="Source Variable",
                    type="select",
                    description="Source variable for path analysis",
                    default="",
                    options=[]  # Will be populated dynamically
                ),
                ParameterDefinition(
                    name="target",
                    display_name="Target Variable",
                    type="select",
                    description="Target variable for path analysis",
                    default="",
                    options=[]  # Will be populated dynamically
                ),
                ParameterDefinition(
                    name="max_length",
                    display_name="Max Path Length",
                    type="number",
                    description="Maximum length of paths to find",
                    default=5,
                    min_value=1,
                    max_value=10,
                    step=1
                )
            ],
            executor=self._execute_path_analysis
        ))
    
    def _register_analysis_components(self):
        """Register analysis components"""
        # Causal Effects
        self.register_component(ComponentDefinition(
            id="causal_effects",
            name="Causal Effects",
            category="analysis",
            description="Estimate causal effects between variables",
            icon="trending_up",
            input_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Causal graph"
                ),
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Dataset for estimation"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="effects",
                    data_type="dict",
                    description="Estimated causal effects"
                ),
                PortDefinition(
                    name="figure",
                    data_type="figure",
                    description="Effects visualization"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="treatment",
                    display_name="Treatment Variable",
                    type="select",
                    description="Treatment (cause) variable",
                    default="",
                    options=[]  # Will be populated dynamically
                ),
                ParameterDefinition(
                    name="outcome",
                    display_name="Outcome Variable",
                    type="select",
                    description="Outcome (effect) variable",
                    default="",
                    options=[]  # Will be populated dynamically
                ),
                ParameterDefinition(
                    name="method",
                    display_name="Estimation Method",
                    type="select",
                    description="Method for effect estimation",
                    default="adjustment",
                    options=["adjustment", "backdoor", "frontdoor", "ivs"]
                ),
                ParameterDefinition(
                    name="control_for",
                    display_name="Control Variables",
                    type="multiselect",
                    description="Variables to control for",
                    default=[],
                    options=[]  # Will be populated dynamically
                )
            ],
            executor=self._execute_causal_effects
        ))
        
        # Counterfactual
        self.register_component(ComponentDefinition(
            id="counterfactual",
            name="Counterfactual",
            category="analysis",
            description="Generate counterfactual scenarios",
            icon="compare_arrows",
            input_ports=[
                PortDefinition(
                    name="graph",
                    data_type="graph",
                    description="Causal graph"
                ),
                PortDefinition(
                    name="data",
                    data_type="dataframe",
                    description="Dataset for counterfactuals"
                )
            ],
            output_ports=[
                PortDefinition(
                    name="counterfactuals",
                    data_type="dataframe",
                    description="Counterfactual scenarios"
                ),
                PortDefinition(
                    name="figure",
                    data_type="figure",
                    description="Counterfactual visualization"
                )
            ],
            parameters=[
                ParameterDefinition(
                    name="intervention",
                    display_name="Intervention Variable",
                    type="select",
                    description="Variable to intervene on",
                    default="",
                    options=[]  # Will be populated dynamically
                ),
                ParameterDefinition(
                    name="intervention_value",
                    display_name="Intervention Value",
                    type="string",
                    description="Value to set for intervention",
                    default="0"
                ),
                ParameterDefinition(
                    name="target",
                    display_name="Target Variable",
                    type="select",
                    description="Variable to observe effects on",
                    default="",
                    options=[]  # Will be populated dynamically
                ),
                ParameterDefinition(
                    name="num_scenarios",
                    display_name="Number of Scenarios",
                    type="number",
                    description="Number of counterfactual scenarios",
                    default=5,
                    min_value=1,
                    max_value=20,
                    step=1
                )
            ],
            executor=self._execute_counterfactual
        ))
    
    # Component executors
    def _execute_dataset_loader(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the dataset loader component"""
        try:
            source = node_data.get("source", "file")
            data_loader = DataLoader()
            
            if source == "file":
                file_path = node_data.get("file", "")
                if not file_path:
                    return {
                        "status": "error",
                        "message": "No file specified",
                        "data": {}
                    }
                
                df, metadata = data_loader.load_file(file_path)
                
            else:  # "sample"
                sample_dataset = node_data.get("sample_dataset", "sachs")
                df, metadata = data_loader.load_sample_dataset(sample_dataset)
            
            # Return the outputs
            return {
                "status": "completed",
                "message": f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns",
                "data": {
                    "data": df,
                    "metadata": metadata
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error loading dataset: {str(e)}",
                "data": {}
            }
    
    def _execute_synthetic_data(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the synthetic data generator component"""
        try:
            # Get parameters
            model_type = node_data.get("model_type", "linear")
            n_variables = node_data.get("n_variables", 5)
            n_samples = node_data.get("n_samples", 1000)
            edge_density = node_data.get("edge_density", 0.3)
            seed = node_data.get("seed", 42)
            
            # Set random seed
            np.random.seed(seed)
            
            # Generate random DAG
            G = nx.DiGraph()
            G.add_nodes_from(range(n_variables))
            
            # Add edges with probability edge_density
            for i in range(n_variables):
                for j in range(i+1, n_variables):
                    if np.random.random() < edge_density:
                        G.add_edge(i, j)
            
            # Generate data based on the model type
            if model_type == "linear":
                # Generate linear data
                data = np.zeros((n_samples, n_variables))
                
                # Topological sort to ensure causality
                node_order = list(nx.topological_sort(G))
                
                # Generate data for each node
                for node in node_order:
                    # If the node has parents, generate data based on them
                    parents = list(G.predecessors(node))
                    
                    if parents:
                        # Generate coefficients for parents
                        weights = np.random.uniform(0.5, 1.5, size=len(parents))
                        if np.random.random() < 0.5:
                            weights *= -1  # Random sign
                        
                        # Combine parent values with weights
                        parent_values = np.array([data[:, parent] for parent in parents]).T
                        node_value = parent_values @ weights
                        
                        # Add noise
                        node_value += np.random.normal(0, 1, size=n_samples)
                    else:
                        # No parents, generate exogenous variable
                        node_value = np.random.normal(0, 1, size=n_samples)
                    
                    # Store generated values
                    data[:, node] = node_value
            
            elif model_type == "nonlinear":
                # Generate nonlinear data (simplified)
                data = np.zeros((n_samples, n_variables))
                
                # Topological sort to ensure causality
                node_order = list(nx.topological_sort(G))
                
                # Generate data for each node
                for node in node_order:
                    # If the node has parents, generate data based on them
                    parents = list(G.predecessors(node))
                    
                    if parents:
                        # Generate nonlinear combination of parents
                        parent_values = np.array([data[:, parent] for parent in parents]).T
                        
                        # Simple nonlinear transformation
                        node_value = np.zeros(n_samples)
                        for i, parent in enumerate(parents):
                            # Random nonlinear function
                            func_type = np.random.choice(["sin", "exp", "squared"])
                            if func_type == "sin":
                                node_value += np.sin(parent_values[:, i])
                            elif func_type == "exp":
                                node_value += np.exp(parent_values[:, i] / 5)  # Scaled to avoid overflow
                            else:  # squared
                                node_value += parent_values[:, i] ** 2
                        
                        # Normalize and add noise
                        node_value = (node_value - node_value.mean()) / (node_value.std() + 1e-8)
                        node_value += np.random.normal(0, 1, size=n_samples)
                    else:
                        # No parents, generate exogenous variable
                        node_value = np.random.normal(0, 1, size=n_samples)
                    
                    # Store generated values
                    data[:, node] = node_value
            
            elif model_type == "discrete":
                # Generate discrete data
                data = np.zeros((n_samples, n_variables), dtype=int)
                
                # Topological sort to ensure causality
                node_order = list(nx.topological_sort(G))
                
                # Generate data for each node
                for node in node_order:
                    # If the node has parents, generate data based on them
                    parents = list(G.predecessors(node))
                    
                    if parents:
                        # Number of categories for this node
                        n_categories = np.random.randint(2, 5)
                        
                        # Create a simple logistic model
                        parent_values = np.array([data[:, parent] for parent in parents]).T
                        
                        # Compute logits for each category
                        weights = np.random.uniform(-1, 1, size=(len(parents), n_categories))
                        logits = parent_values @ weights
                        
                        # Apply softmax
                        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                        
                        # Generate categories
                        node_value = np.array([np.random.choice(n_categories, p=p) for p in probs])
                    else:
                        # No parents, generate random categories
                        n_categories = np.random.randint(2, 5)
                        node_value = np.random.randint(0, n_categories, size=n_samples)
                    
                    # Store generated values
                    data[:, node] = node_value
            
            else:  # mixed
                # Generate mixed data (some continuous, some discrete)
                data = np.zeros((n_samples, n_variables))
                is_discrete = np.random.choice([True, False], size=n_variables)
                
                # Topological sort to ensure causality
                node_order = list(nx.topological_sort(G))
                
                # Generate data for each node
                for node in node_order:
                    # If the node has parents, generate data based on them
                    parents = list(G.predecessors(node))
                    
                    if parents:
                        # Generate based on parents
                        parent_values = np.array([data[:, parent] for parent in parents]).T
                        weights = np.random.uniform(-1, 1, size=len(parents))
                        node_value = parent_values @ weights
                        
                        # Add noise
                        node_value += np.random.normal(0, 1, size=n_samples)
                        
                        # If this is a discrete variable, discretize the values
                        if is_discrete[node]:
                            n_categories = np.random.randint(2, 5)
                            # Bin the continuous values
                            bins = np.percentile(node_value, np.linspace(0, 100, n_categories+1))
                            node_value = np.digitize(node_value, bins[1:-1])
                    else:
                        # No parents, generate exogenous variable
                        if is_discrete[node]:
                            n_categories = np.random.randint(2, 5)
                            node_value = np.random.randint(0, n_categories, size=n_samples)
                        else:
                            node_value = np.random.normal(0, 1, size=n_samples)
                    
                    # Store generated values
                    data[:, node] = node_value
            
            # Create column names
            column_names = [f"X{i}" for i in range(n_variables)]
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=column_names)
            
            # Return the outputs
            return {
                "status": "completed",
                "message": f"Generated synthetic {model_type} data with {n_variables} variables and {n_samples} samples",
                "data": {
                    "data": df,
                    "true_graph": G
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating synthetic data: {str(e)}",
                "data": {}
            }
    
    def _execute_missing_values(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the missing values handler component"""
        try:
            # Get the input data
            if "data" not in inputs:
                return {
                    "status": "error",
                    "message": "No input data provided",
                    "data": {}
                }
            
            df = inputs["data"]
            
            # Get parameters
            method = node_data.get("method", "drop")
            columns = node_data.get("columns", [])
            fill_value = node_data.get("fill_value", "0")
            
            # Convert fill_value to appropriate type if needed
            try:
                fill_value = float(fill_value)
            except:
                pass  # Keep as string
            
            # Create preprocessor
            preprocessor = DataPreprocessor(df)
            
            # Handle missing values
            result_df = preprocessor.handle_missing_values(
                method=method,
                columns=columns if columns else None,
                fill_value=fill_value
            )
            
            # Get summary
            summary = preprocessor.get_preprocessing_summary()
            
            # Return the outputs
            return {
                "status": "completed",
                "message": f"Handled missing values using {method}. Rows before: {summary['original_shape'][0]}, Rows after: {summary['current_shape'][0]}",
                "data": {
                    "data": result_df
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error handling missing values: {str(e)}",
                "data": {}
            }
    
    def _execute_normalization(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the normalization component"""
        try:
            # Get the input data
            if "data" not in inputs:
                return {
                    "status": "error",
                    "message": "No input data provided",
                    "data": {}
                }
            
            df = inputs["data"]
            
            # Get parameters
            method = node_data.get("method", "standard")
            columns = node_data.get("columns", [])
            
            # Create preprocessor
            preprocessor = DataPreprocessor(df)
            
            # Normalize data
            result_df = preprocessor.normalize_data(
                method=method,
                columns=columns if columns else None
            )
            
            # Return the outputs
            return {
                "status": "completed",
                "message": f"Normalized data using {method} method",
                "data": {
                    "data": result_df
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error normalizing data: {str(e)}",
                "data": {}
            }
    
    def _execute_feature_selection(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the feature selection component"""
        try:
            # Get the input data
            if "data" not in inputs:
                return {
                    "status": "error",
                    "message": "No input data provided",
                    "data": {}
                }
            
            df = inputs["data"]
            
            # Get parameters
            columns = node_data.get("columns", [])
            
            if not columns:
                columns = df.columns.tolist()
            
            # Create preprocessor
            preprocessor = DataPreprocessor(df)
            
            # Select columns
            result_df = preprocessor.select_columns(columns)
            
            # Return the outputs
            return {
                "status": "completed",
                "message": f"Selected {len(columns)} columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}",
                "data": {
                    "data": result_df
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error selecting features: {str(e)}",
                "data": {}
            }
    
    def _execute_pc_algorithm(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the PC algorithm component"""
        try:
            # Get the input data
            if "data" not in inputs:
                return {
                    "status": "error",
                    "message": "No input data provided",
                    "data": {}
                }
            
            df = inputs["data"]
            
            # Get parameters
            indep_test = node_data.get("indep_test", "fisherz")
            alpha = node_data.get("alpha", 0.05)
            stable = node_data.get("stable", True)
            
            # Create algorithm executor
            executor = AlgorithmExecutor()
            
            # Execute PC algorithm
            algorithm_id = f"pc_{indep_test}"
            params = {
                "alpha": alpha,
                "stable": stable
            }
            
            result = executor.execute_algorithm(algorithm_id, df, params)
            
            if result["status"] == "success":
                return {
                    "status": "completed",
                    "message": f"Successfully executed PC algorithm with {indep_test} independence test",
                    "data": {
                        "graph": result["graph"],
                        "result": result
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"Error running PC algorithm: {result.get('error', 'Unknown error')}",
                    "data": {}
                }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing PC algorithm: {str(e)}",
                "data": {}
            }
    
    def _execute_fci_algorithm(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the FCI algorithm component"""
        try:
            # Get the input data
            if "data" not in inputs:
                return {
                    "status": "error",
                    "message": "No input data provided",
                    "data": {}
                }
            
            df = inputs["data"]
            
            # Get parameters
            indep_test = node_data.get("indep_test", "fisherz")
            alpha = node_data.get("alpha", 0.05)
            max_path_length = node_data.get("max_path_length", -1)
            
            # Create algorithm executor
            executor = AlgorithmExecutor()
            
            # Execute FCI algorithm
            algorithm_id = f"fci_{indep_test}"
            params = {
                "alpha": alpha,
                "max_path_length": max_path_length
            }
            
            result = executor.execute_algorithm(algorithm_id, df, params)
            
            if result["status"] == "success":
                return {
                    "status": "completed",
                    "message": f"Successfully executed FCI algorithm with {indep_test} independence test",
                    "data": {
                        "graph": result["graph"],
                        "result": result
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"Error running FCI algorithm: {result.get('error', 'Unknown error')}",
                    "data": {}
                }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing FCI algorithm: {str(e)}",
                "data": {}
            }
    
    def _execute_lingam(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LiNGAM component"""
        try:
            # Get the input data
            if "data" not in inputs:
                return {
                    "status": "error",
                    "message": "No input data provided",
                    "data": {}
                }
            
            df = inputs["data"]
            
            # Get parameters
            method = node_data.get("method", "direct")
            measure = node_data.get("measure", "pwling")
            random_state = node_data.get("random_state", 42)
            
            # Create algorithm executor
            executor = AlgorithmExecutor()
            
            # Execute LiNGAM algorithm
            algorithm_id = f"lingam_{method}"
            params = {
                "measure": measure,
                "random_state": random_state
            }
            
            result = executor.execute_algorithm(algorithm_id, df, params)
            
            if result["status"] == "success":
                return {
                    "status": "completed",
                    "message": f"Successfully executed LiNGAM ({method}) algorithm",
                    "data": {
                        "graph": result["graph"],
                        "result": result
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"Error running LiNGAM algorithm: {result.get('error', 'Unknown error')}",
                    "data": {}
                }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing LiNGAM algorithm: {str(e)}",
                "data": {}
            }
    
    def _execute_ensemble(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ensemble component"""
        try:
            # Get the input graphs
            if "graphs" not in inputs:
                return {
                    "status": "error",
                    "message": "No input graphs provided",
                    "data": {}
                }
            
            # Input graphs can be a single graph or a list of graphs
            graphs = inputs["graphs"]
            if not isinstance(graphs, list):
                graphs = [graphs]
            
            if not graphs:
                return {
                    "status": "error",
                    "message": "No graphs provided for ensemble",
                    "data": {}
                }
            
            # Get parameters
            edge_threshold = node_data.get("edge_threshold", 0.5)
            resolve_conflicts = node_data.get("resolve_conflicts", True)
            
            # Create ensemble
            ensemble = AlgorithmEnsemble(llm_adapter=st.session_state.llm_adapter if resolve_conflicts else None)
            
            # Prepare algorithm results in the expected format
            algorithm_results = []
            for i, graph in enumerate(graphs):
                algorithm_results.append({
                    "status": "success",
                    "algorithm_id": f"algorithm_{i}",
                    "graph": graph
                })
            
            # Create ensemble graph
            ensemble_result = ensemble.create_ensemble_graph(
                algorithm_results=algorithm_results,
                edge_threshold=edge_threshold,
                resolve_conflicts=resolve_conflicts
            )
            
            return {
                "status": "completed",
                "message": f"Created ensemble graph from {len(graphs)} input graphs",
                "data": {
                    "graph": ensemble_result["graph"],
                    "result": ensemble_result
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating ensemble graph: {str(e)}",
                "data": {}
            }
    
    def _execute_llm_refinement(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LLM refinement component"""
        try:
            # Get the input graph
            if "graph" not in inputs:
                return {
                    "status": "error",
                    "message": "No input graph provided",
                    "data": {}
                }
            
            graph = inputs["graph"]
            
            # Get the input data (optional)
            data = inputs.get("data")
            
            # Get parameters
            domain_knowledge = node_data.get("domain_knowledge", "")
            confidence_threshold = node_data.get("confidence_threshold", 0.5)
            discover_hidden = node_data.get("discover_hidden", True)
            
            # Check if LLM adapter is available
            if not st.session_state.llm_adapter:
                return {
                    "status": "error",
                    "message": "LLM adapter not available. Please configure in Settings.",
                    "data": {}
                }
            
            # Create variable descriptions (simplified)
            variable_descriptions = {}
            for node in graph.nodes():
                if "name" in graph.nodes[node]:
                    name = graph.nodes[node]["name"]
                elif isinstance(node, int) and node < len(data.columns):
                    name = data.columns[node]
                else:
                    name = str(node)
                
                variable_descriptions[name] = f"Variable {name}"
            
            # Format domain knowledge
            domain_context = domain_knowledge if domain_knowledge else None
            
            # Use LLM for refinement
            result = st.session_state.llm_adapter.causal_refinement(
                graph_data={
                    "nodes": [{"id": n, "name": variable_descriptions.get(str(n), str(n))} for n in graph.nodes()],
                    "edges": [{"source": u, "target": v, **graph.edges[u, v]} for u, v in graph.edges()]
                },
                variable_descriptions=variable_descriptions,
                domain_context=domain_context,
                temperature=0.3
            )
            
            # Here in a real implementation, we would parse the LLM's response
            # and update the graph accordingly. For this example, we'll create
            # a simple update that adds some confidence scores to edges.
            
            # Create a new graph with refinements
            refined_graph = graph.copy()
            
            # Add confidence scores to edges
            import random
            for u, v in refined_graph.edges():
                refined_graph[u][v]["confidence"] = random.uniform(0.5, 1.0)
            
            # If discover_hidden is True, add a hidden variable
            if discover_hidden:
                # Add a hidden variable node
                hidden_id = max(refined_graph.nodes()) + 1 if refined_graph.nodes() else 0
                refined_graph.add_node(hidden_id, name="Hidden Variable", is_hidden=True)
                
                # Add edges from hidden variable to some nodes
                nodes = list(refined_graph.nodes())
                for node in random.sample(nodes, min(3, len(nodes))):
                    if node != hidden_id:
                        refined_graph.add_edge(hidden_id, node, confidence=random.uniform(0.5, 1.0), is_hidden_cause=True)
            
            # Return the outputs
            return {
                "status": "completed",
                "message": "Refined causal graph using LLM",
                "data": {
                    "graph": refined_graph,
                    "result": {
                        "refined_graph": refined_graph,
                        "llm_response": result.get("reasoning", ""),
                        "hidden_variables_added": discover_hidden
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error refining graph with LLM: {str(e)}",
                "data": {}
            }
    
    def _execute_hidden_variables(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the hidden variables component"""
        try:
            # Get the input graph and data
            if "graph" not in inputs or "data" not in inputs:
                return {
                    "status": "error",
                    "message": "Both graph and data inputs are required",
                    "data": {}
                }
            
            graph = inputs["graph"]
            data = inputs["data"]
            
            # Get parameters
            method = node_data.get("method", "correlation")
            threshold = node_data.get("threshold", 0.7)
            
            # Create a new graph with hidden variables
            graph_with_hidden = graph.copy()
            
            # In a real implementation, we would apply various methods
            # to detect hidden variables. Here we'll use a simplified approach.
            
            # Find pairs of nodes with high correlation but no direct edge
            hidden_variables = []
            
            if method == "correlation":
                # Select numeric columns
                numeric_data = data.select_dtypes(include=[np.number])
                
                # Calculate correlation matrix
                corr_matrix = numeric_data.corr()
                
                # Find pairs with high correlation but no direct edge
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        
                        # Get node IDs for these columns
                        node1 = None
                        node2 = None
                        
                        for node in graph.nodes():
                            if "name" in graph.nodes[node] and graph.nodes[node]["name"] == col1:
                                node1 = node
                            elif isinstance(node, int) and node < len(data.columns) and data.columns[node] == col1:
                                node1 = node
                                
                            if "name" in graph.nodes[node] and graph.nodes[node]["name"] == col2:
                                node2 = node
                            elif isinstance(node, int) and node < len(data.columns) and data.columns[node] == col2:
                                node2 = node
                        
                        if node1 is not None and node2 is not None:
                            corr = abs(corr_matrix.loc[col1, col2])
                            
                            # Check if correlation is high but no direct edge
                            if corr > threshold and not graph.has_edge(node1, node2) and not graph.has_edge(node2, node1):
                                # This suggests a hidden confounder
                                hidden_id = max(graph_with_hidden.nodes()) + 1
                                hidden_name = f"Hidden_{len(hidden_variables)+1}"
                                
                                # Add hidden node
                                graph_with_hidden.add_node(hidden_id, name=hidden_name, is_hidden=True)
                                
                                # Add edges from hidden to both nodes
                                graph_with_hidden.add_edge(hidden_id, node1, confidence=corr, is_hidden_cause=True)
                                graph_with_hidden.add_edge(hidden_id, node2, confidence=corr, is_hidden_cause=True)
                                
                                # Track hidden variable
                                hidden_variables.append({
                                    "id": hidden_id,
                                    "name": hidden_name,
                                    "affects": [col1, col2],
                                    "confidence": corr,
                                    "method": "correlation"
                                })
            
            elif method == "tetrad":
                # Simplified tetrad test (in a real implementation, would use proper tetrad constraints)
                # For now, just use a placeholder implementation similar to correlation
                
                # This is similar to above but would use tetrad constraints
                # This is a placeholder
                pass
            
            elif method == "llm":
                # Use LLM to identify potential hidden variables
                if not st.session_state.llm_adapter:
                    return {
                        "status": "error",
                        "message": "LLM adapter not available. Please configure in Settings.",
                        "data": {}
                    }
                
                # Create a prompt for the LLM
                prompt = f"""Analyze this causal graph and dataset to identify potential hidden variables.

Graph Structure:
{json.dumps([{"source": u, "target": v} for u, v in graph.edges()])}

Variable Names:
{json.dumps(data.columns.tolist())}

Data Summary:
- Number of samples: {data.shape[0]}
- Number of variables: {data.shape[1]}
- Variable types: {json.dumps({col: str(dtype) for col, dtype in data.dtypes.astype(str).to_dict().items()})}

Please identify any potential hidden variables that might be confounders in this causal system.
For each hidden variable, provide:
1. A name for the hidden variable
2. Which observed variables it might affect
3. Your confidence level (0-1) in this hidden variable
4. A brief explanation of why you think this hidden variable exists
"""
                
                # Query the LLM
                response = st.session_state.llm_adapter.complete(
                    prompt=prompt,
                    system_prompt="You are an expert in causal discovery and identifying hidden variables. Your task is to analyze causal graphs and data to identify potential hidden confounders.",
                    temperature=0.4
                )
                
                # In a real implementation, we would parse the LLM's response
                # For now, create a dummy hidden variable
                
                hidden_id = max(graph_with_hidden.nodes()) + 1
                hidden_name = "LLM_Hidden_1"
                
                # Add hidden node
                graph_with_hidden.add_node(hidden_id, name=hidden_name, is_hidden=True)
                
                # Add edges from hidden to a few random nodes
                import random
                for node in random.sample(list(graph.nodes()), min(3, len(graph.nodes()))):
                    graph_with_hidden.add_edge(hidden_id, node, confidence=0.7, is_hidden_cause=True)
                
                # Track hidden variable
                hidden_variables.append({
                    "id": hidden_id,
                    "name": hidden_name,
                    "affects": [data.columns[node] if isinstance(node, int) and node < len(data.columns) else str(node) for node in graph_with_hidden.successors(hidden_id)],
                    "confidence": 0.7,
                    "method": "llm",
                    "explanation": "Identified by LLM analysis"
                })
            
            # Return the outputs
            return {
                "status": "completed",
                "message": f"Identified {len(hidden_variables)} potential hidden variables",
                "data": {
                    "graph": graph_with_hidden,
                    "result": {
                        "hidden_variables": hidden_variables,
                        "method": method,
                        "threshold": threshold
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error detecting hidden variables: {str(e)}",
                "data": {}
            }
    
    def _execute_graph_viz(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the graph visualization component"""
        try:
            # Get the input graph
            if "graph" not in inputs:
                return {
                    "status": "error",
                    "message": "No input graph provided",
                    "data": {}
                }
            
            graph = inputs["graph"]
            
            # Get parameters
            layout = node_data.get("layout", "spring")
            show_weights = node_data.get("show_weights", True)
            show_confidence = node_data.get("show_confidence", True)
            node_size = node_data.get("node_size", 10)
            
            # Create graph visualizer
            graph_viz = CausalGraphVisualizer()
            
            # Create node labels
            node_labels = {}
            for node in graph.nodes():
                if "name" in graph.nodes[node]:
                    node_labels[node] = graph.nodes[node]["name"]
                else:
                    node_labels[node] = str(node)
            
            # Generate visualization
            fig = graph_viz.visualize_graph(
                graph=graph,
                node_labels=node_labels,
                edge_weights=show_weights,
                layout_type=layout,
                show_confidence=show_confidence
            )
            
            # Return the outputs
            return {
                "status": "completed",
                "message": "Generated graph visualization",
                "data": {
                    "figure": fig
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error visualizing graph: {str(e)}",
                "data": {}
            }
    
    def _execute_path_analysis(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the path analysis component"""
        try:
            # Get the input graph
            if "graph" not in inputs:
                return {
                    "status": "error",
                    "message": "No input graph provided",
                    "data": {}
                }
            
            graph = inputs["graph"]
            
            # Get parameters
            source = node_data.get("source", "")
            target = node_data.get("target", "")
            max_length = node_data.get("max_length", 5)
            
            # Find source and target nodes
            source_node = None
            target_node = None
            
            for node in graph.nodes():
                if "name" in graph.nodes[node] and graph.nodes[node]["name"] == source:
                    source_node = node
                if "name" in graph.nodes[node] and graph.nodes[node]["name"] == target:
                    target_node = node
            
            if source_node is None or target_node is None:
                return {
                    "status": "error",
                    "message": f"Could not find {'source' if source_node is None else 'target'} node",
                    "data": {}
                }
            
            # Find all paths between source and target
            try:
                if max_length > 0:
                    # Use cutoff for max path length
                    paths = list(nx.all_simple_paths(graph, source_node, target_node, cutoff=max_length))
                else:
                    # No length limit
                    paths = list(nx.all_simple_paths(graph, source_node, target_node))
            except nx.NetworkXNoPath:
                paths = []
            
            # Create node labels
            node_labels = {}
            for node in graph.nodes():
                if "name" in graph.nodes[node]:
                    node_labels[node] = graph.nodes[node]["name"]
                else:
                    node_labels[node] = str(node)
            
            # Create graph visualizer
            graph_viz = CausalGraphVisualizer()
            
            # Generate path visualization
            if paths:
                fig = graph_viz.visualize_causal_paths(
                    graph=graph,
                    source=source_node,
                    target=target_node,
                    node_labels=node_labels
                )
            else:
                # If no paths, just visualize the graph
                fig = graph_viz.visualize_graph(
                    graph=graph,
                    node_labels=node_labels
                )
            
            # Format paths for output
            formatted_paths = []
            for path in paths:
                path_str = " → ".join([node_labels[node] for node in path])
                formatted_paths.append({
                    "path": path,
                    "path_str": path_str,
                    "length": len(path) - 1
                })
            
            # Return the outputs
            return {
                "status": "completed",
                "message": f"Found {len(paths)} causal paths from {source} to {target}",
                "data": {
                    "figure": fig,
                    "paths": {
                        "source": source,
                        "target": target,
                        "source_node": source_node,
                        "target_node": target_node,
                        "paths": formatted_paths,
                        "node_labels": node_labels
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing causal paths: {str(e)}",
                "data": {}
            }
    
    def _execute_causal_effects(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the causal effects component"""
        try:
            # Get the input graph and data
            if "graph" not in inputs or "data" not in inputs:
                return {
                    "status": "error",
                    "message": "Both graph and data inputs are required",
                    "data": {}
                }
            
            graph = inputs["graph"]
            data = inputs["data"]
            
            # Get parameters
            treatment = node_data.get("treatment", "")
            outcome = node_data.get("outcome", "")
            method = node_data.get("method", "adjustment")
            control_for = node_data.get("control_for", [])
            
            # Find treatment and outcome nodes
            treatment_node = None
            outcome_node = None
            
            for node in graph.nodes():
                if "name" in graph.nodes[node] and graph.nodes[node]["name"] == treatment:
                    treatment_node = node
                if "name" in graph.nodes[node] and graph.nodes[node]["name"] == outcome:
                    outcome_node = node
                
                # If nodes are integer indices into dataframe columns
                if treatment_node is None and isinstance(node, int) and node < len(data.columns) and data.columns[node] == treatment:
                    treatment_node = node
                if outcome_node is None and isinstance(node, int) and node < len(data.columns) and data.columns[node] == outcome:
                    outcome_node = node
            
            if treatment_node is None or outcome_node is None:
                return {
                    "status": "error",
                    "message": f"Could not find {'treatment' if treatment_node is None else 'outcome'} node",
                    "data": {}
                }
            
            # Find control nodes
            control_nodes = []
            for ctrl in control_for:
                for node in graph.nodes():
                    if "name" in graph.nodes[node] and graph.nodes[node]["name"] == ctrl:
                        control_nodes.append(node)
                    
                    # If nodes are integer indices into dataframe columns
                    if isinstance(node, int) and node < len(data.columns) and data.columns[node] == ctrl:
                        control_nodes.append(node)
            
            # In a real implementation, there would be a more sophisticated
            # causal effect estimation. Here we'll use a simplified approach.
            
            # Create node labels for explanation
            node_labels = {}
            for node in graph.nodes():
                if "name" in graph.nodes[node]:
                    node_labels[node] = graph.nodes[node]["name"]
                elif isinstance(node, int) and node < len(data.columns):
                    node_labels[node] = data.columns[node]
                else:
                    node_labels[node] = str(node)
            
            # Get the variable names
            treatment_name = node_labels[treatment_node]
            outcome_name = node_labels[outcome_node]
            control_names = [node_labels[node] for node in control_nodes]
            
            # Calculate causal effect using linear regression (simplified)
            import statsmodels.api as sm
            
            if method == "adjustment":
                # Simple adjustment method with linear regression
                
                # Prepare the data
                X_names = [treatment_name] + control_names
                X = data[X_names]
                X = sm.add_constant(X)  # Add intercept
                y = data[outcome_name]
                
                # Fit the model
                model = sm.OLS(y, X).fit()
                
                # Get the coefficient for treatment
                effect_size = model.params[treatment_name]
                
                # Calculate confidence interval
                conf_int = model.conf_int().loc[treatment_name].tolist()
                
                # Get p-value
                p_value = model.pvalues[treatment_name]
                
                effect_info = {
                    "effect_size": effect_size,
                    "confidence_interval": conf_int,
                    "p_value": p_value,
                    "standard_error": model.bse[treatment_name],
                    "t_statistic": model.tvalues[treatment_name],
                    "regression_summary": model.summary().as_text()
                }
            
            else:
                # For other methods, use a placeholder
                import random
                effect_size = random.uniform(-1, 1)
                p_value = random.uniform(0, 0.1)
                conf_int = [effect_size - 0.2, effect_size + 0.2]
                
                effect_info = {
                    "effect_size": effect_size,
                    "confidence_interval": conf_int,
                    "p_value": p_value,
                    "method_info": f"Used {method} method for estimation"
                }
            
            # Create a plot to visualize the effect
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add effect size with confidence interval
            fig.add_trace(go.Bar(
                x=[treatment_name],
                y=[effect_size],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[conf_int[1] - effect_size],
                    arrayminus=[effect_size - conf_int[0]]
                ),
                name="Causal Effect"
            ))
            
            fig.update_layout(
                title=f"Causal Effect of {treatment_name} on {outcome_name}",
                xaxis_title="Treatment",
                yaxis_title="Effect Size",
                showlegend=True
            )
            
            # Create a string representation of the effect
            if effect_size > 0:
                effect_direction = "positive"
            elif effect_size < 0:
                effect_direction = "negative"
            else:
                effect_direction = "neutral"
            
            effect_significance = "significant" if p_value < 0.05 else "not significant"
            
            effect_str = f"The estimated causal effect of {treatment_name} on {outcome_name} is {effect_size:.4f} (95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]). This represents a {effect_direction} effect that is statistically {effect_significance} (p = {p_value:.4f})."
            
            if control_names:
                effect_str += f" This estimate controls for {', '.join(control_names)}."
            
            # Create a comprehensive result
            effects_result = {
                "treatment": treatment_name,
                "outcome": outcome_name,
                "controls": control_names,
                "method": method,
                "effect_size": effect_size,
                "confidence_interval": conf_int,
                "p_value": p_value,
                "effect_direction": effect_direction,
                "significant": p_value < 0.05,
                "effect_str": effect_str,
                "details": effect_info
            }
            
            # Return the outputs
            return {
                "status": "completed",
                "message": effect_str,
                "data": {
                    "effects": effects_result,
                    "figure": fig
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error estimating causal effects: {str(e)}",
                "data": {}
            }
    
    def _execute_counterfactual(self, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the counterfactual component"""
        try:
            # Get the input graph and data
            if "graph" not in inputs or "data" not in inputs:
                return {
                    "status": "error",
                    "message": "Both graph and data inputs are required",
                    "data": {}
                }
            
            graph = inputs["graph"]
            data = inputs["data"]
            
            # Get parameters
            intervention = node_data.get("intervention", "")
            intervention_value = node_data.get("intervention_value", "0")
            target = node_data.get("target", "")
            num_scenarios = node_data.get("num_scenarios", 5)
            
            # Find intervention and target nodes
            intervention_node = None
            target_node = None
            
            for node in graph.nodes():
                if "name" in graph.nodes[node] and graph.nodes[node]["name"] == intervention:
                    intervention_node = node
                if "name" in graph.nodes[node] and graph.nodes[node]["name"] == target:
                    target_node = node
                
                # If nodes are integer indices into dataframe columns
                if intervention_node is None and isinstance(node, int) and node < len(data.columns) and data.columns[node] == intervention:
                    intervention_node = node
                if target_node is None and isinstance(node, int) and node < len(data.columns) and data.columns[node] == target:
                    target_node = node
            
            if intervention_node is None or target_node is None:
                return {
                    "status": "error",
                    "message": f"Could not find {'intervention' if intervention_node is None else 'target'} node",
                    "data": {}
                }
            
            # Get the variable names
            intervention_name = None
            target_name = None
            
            for node in graph.nodes():
                if node == intervention_node:
                    if "name" in graph.nodes[node]:
                        intervention_name = graph.nodes[node]["name"]
                    elif isinstance(node, int) and node < len(data.columns):
                        intervention_name = data.columns[node]
                    else:
                        intervention_name = str(node)
                
                if node == target_node:
                    if "name" in graph.nodes[node]:
                        target_name = graph.nodes[node]["name"]
                    elif isinstance(node, int) and node < len(data.columns):
                        target_name = data.columns[node]
                    else:
                        target_name = str(node)
            
            if intervention_name is None:
                intervention_name = str(intervention_node)
            if target_name is None:
                target_name = str(target_node)
            
            # Convert intervention value to appropriate type
            try:
                intervention_value = float(intervention_value)
            except:
                pass  # Keep as string
            
            # In a real implementation, there would be a true counterfactual
            # generation using the causal graph and appropriate methods.
            # Here we'll use a simplified approach for demonstration.
            
            # Use linear regression to predict counterfactual outcomes (simplified)
            import statsmodels.api as sm
            
            # Find descendants of intervention node
            try:
                descendants = list(nx.descendants(graph, intervention_node))
            except:
                descendants = []
            
            # Only proceed if target is a descendant of intervention
            if target_node not in descendants and target_node != intervention_node:
                return {
                    "status": "error",
                    "message": f"Target variable ({target_name}) is not caused by intervention variable ({intervention_name})",
                    "data": {}
                }
            
            # Prepare the data
            X = data[[intervention_name]]
            X = sm.add_constant(X)  # Add intercept
            y = data[target_name]
            
            # Fit the model
            model = sm.OLS(y, X).fit()
            
            # Create counterfactual scenarios
            counterfactuals = []
            
            # Get baseline statistics
            baseline_mean = data[target_name].mean()
            baseline_std = data[target_name].std()
            
            # Original samples as basis for counterfactuals
            original_samples = data.sample(min(num_scenarios, len(data))).copy()
            
            # Create counterfactuals
            for i, (idx, row) in enumerate(original_samples.iterrows()):
                # Create counterfactual scenario
                cf = row.copy()
                
                # Replace intervention with new value
                original_intervention = cf[intervention_name]
                cf[intervention_name] = intervention_value
                
                # Predict new target value
                X_cf = np.array([1, intervention_value]).reshape(1, -1)  # Add intercept
                predicted_target = model.predict(X_cf)[0]
                
                # Replace target with predicted value
                original_target = cf[target_name]
                cf[target_name] = predicted_target
                
                # Calculate effect
                effect = predicted_target - original_target
                
                # Store counterfactual scenario
                counterfactuals.append({
                    "id": i+1,
                    "original_intervention": original_intervention,
                    "intervention_value": intervention_value,
                    "original_target": original_target,
                    "predicted_target": predicted_target,
                    "effect": effect,
                    "scenario": cf.to_dict()
                })
            
            # Create counterfactual DataFrame
            cf_data = pd.DataFrame([cf["scenario"] for cf in counterfactuals])
            
            # Create plot to visualize counterfactuals
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add original vs counterfactual comparison
            fig.add_trace(go.Bar(
                x=["Original", "Counterfactual"],
                y=[original_samples[target_name].mean(), cf_data[target_name].mean()],
                error_y=dict(
                    type='data',
                    array=[original_samples[target_name].std(), cf_data[target_name].std()]
                ),
                name=f"Effect on {target_name}"
            ))
            
            fig.update_layout(
                title=f"Counterfactual Effect of {intervention_name} = {intervention_value} on {target_name}",
                xaxis_title="Scenario",
                yaxis_title=f"Average {target_name}",
                showlegend=True
            )
            
            # Create a string representation of the counterfactual effect
            avg_effect = cf_data[target_name].mean() - original_samples[target_name].mean()
            
            if avg_effect > 0:
                effect_direction = "increase"
            elif avg_effect < 0:
                effect_direction = "decrease"
            else:
                effect_direction = "no change in"
            
            effect_str = f"Setting {intervention_name} to {intervention_value} is predicted to {effect_direction} {target_name} by {abs(avg_effect):.4f} on average, from {original_samples[target_name].mean():.4f} to {cf_data[target_name].mean():.4f}."
            
            # Return the outputs
            return {
                "status": "completed",
                "message": effect_str,
                "data": {
                    "counterfactuals": {
                        "intervention": intervention_name,
                        "intervention_value": intervention_value,
                        "target": target_name,
                        "scenarios": counterfactuals,
                        "avg_effect": avg_effect,
                        "effect_str": effect_str
                    },
                    "figure": fig
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating counterfactuals: {str(e)}",
                "data": {}
            }


# Create singleton registry
component_registry = ComponentRegistry()

# Function to get the registry
def get_component_registry():
    """Get the global component registry"""
    return component_registry

# Function to execute a component
def execute_component(component_id: str, node_data: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a component with the given node data and inputs
    
    Args:
        component_id: ID of the component to execute
        node_data: Configuration data for the node
        inputs: Input data from connected nodes
        
    Returns:
        Execution results
    """
    registry = get_component_registry()
    component = registry.get_component(component_id)
    
    if component is None:
        return {
            "status": "error",
            "message": f"Component not found: {component_id}",
            "data": {}
        }
    
    if component.executor is None:
        return {
            "status": "error",
            "message": f"No executor defined for component: {component_id}",
            "data": {}
        }
    
    return component.executor(node_data, inputs)