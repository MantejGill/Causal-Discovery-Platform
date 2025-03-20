# app/components/execution_engine.py

import streamlit as st
import pandas as pd
import networkx as nx
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the component registry for accessing component definitions
from app.components.component_registry import get_component_registry, execute_component

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Execution engine for the Causal Playground.
    Handles the execution of components in a workflow, manages execution state,
    and provides caching and parallelization features.
    """
    
    def __init__(self):
        """Initialize the execution engine"""
        self.registry = get_component_registry()
        
        # Initialize execution state if not already in session state
        if "execution_state" not in st.session_state:
            st.session_state.execution_state = {}
        if "execution_cache" not in st.session_state:
            st.session_state.execution_cache = {}
    
    def execute_workflow(self, 
                        experiment_data: Dict[str, Any], 
                        reset_cache: bool = False, 
                        parallel: bool = False,
                        max_workers: int = 4) -> Dict[str, Any]:
        """
        Execute a workflow defined in an experiment
        
        Args:
            experiment_data: The experiment data containing nodes and edges
            reset_cache: Whether to reset the execution cache
            parallel: Whether to use parallel execution
            max_workers: Maximum number of parallel workers
            
        Returns:
            Updated experiment data with results
        """
        experiment_id = experiment_data.get("id", "unknown")
        
        # Reset execution state for this experiment
        if experiment_id not in st.session_state.execution_state:
            st.session_state.execution_state[experiment_id] = {}
        
        # Reset cache if requested
        if reset_cache and experiment_id in st.session_state.execution_cache:
            st.session_state.execution_cache[experiment_id] = {}
        
        # Create a directed graph from the experiment
        G = nx.DiGraph()
        for node in experiment_data.get("nodes", []):
            G.add_node(node["id"])
        for edge in experiment_data.get("edges", []):
            G.add_edge(edge["source"], edge["target"])
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            experiment_data["status"] = "error"
            experiment_data["error_message"] = "Workflow contains cycles and cannot be executed"
            return experiment_data
        
        # Get topological sorting of nodes
        try:
            sorted_nodes = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            experiment_data["status"] = "error"
            experiment_data["error_message"] = "Could not determine execution order"
            return experiment_data
        
        # Initialize results container if not present
        if "results" not in experiment_data:
            experiment_data["results"] = {}
        
        # Initialize cache for this experiment if not present
        if experiment_id not in st.session_state.execution_cache:
            st.session_state.execution_cache[experiment_id] = {}
        
        # Update experiment status
        experiment_data["status"] = "running"
        experiment_data["start_time"] = time.time()
        
        if parallel:
            # Execute nodes in parallel where possible
            self._execute_parallel(experiment_data, G, sorted_nodes, max_workers)
        else:
            # Execute nodes sequentially
            self._execute_sequential(experiment_data, G, sorted_nodes)
        
        # Update execution end time
        experiment_data["end_time"] = time.time()
        
        # Check if all nodes executed successfully
        all_completed = all(
            result.get("status") == "completed"
            for result in experiment_data["results"].values()
        )
        
        if all_completed:
            experiment_data["status"] = "completed"
        else:
            # Check if any node failed
            any_failed = any(
                result.get("status") == "error"
                for result in experiment_data["results"].values()
            )
            if any_failed:
                experiment_data["status"] = "failed"
            else:
                experiment_data["status"] = "partial"
        
        # Calculate execution time
        if "start_time" in experiment_data and "end_time" in experiment_data:
            experiment_data["execution_time"] = experiment_data["end_time"] - experiment_data["start_time"]
        
        return experiment_data
    
    def _execute_sequential(self, experiment_data: Dict[str, Any], G: nx.DiGraph, sorted_nodes: List[str]):
        """Execute nodes sequentially in topological order"""
        for node_id in sorted_nodes:
            self._execute_single_node(experiment_data, G, node_id)
    
    def _execute_parallel(self, experiment_data: Dict[str, Any], G: nx.DiGraph, sorted_nodes: List[str], max_workers: int):
        """Execute nodes in parallel where dependencies allow"""
        # Group nodes by their topological level (distance from root)
        levels = {}
        for node in sorted_nodes:
            # Maximum distance from any root node
            level = max([0] + [levels[pred] + 1 for pred in G.predecessors(node)])
            levels[node] = level
        
        # Group nodes by level
        nodes_by_level = {}
        for node, level in levels.items():
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)
        
        # Execute each level in parallel
        for level in sorted(nodes_by_level.keys()):
            level_nodes = nodes_by_level[level]
            
            # Skip empty levels
            if not level_nodes:
                continue
            
            # If only one node at this level, execute it directly
            if len(level_nodes) == 1:
                self._execute_single_node(experiment_data, G, level_nodes[0])
                continue
            
            # Execute nodes at this level in parallel
            with ThreadPoolExecutor(max_workers=min(max_workers, len(level_nodes))) as executor:
                # Submit all nodes at this level
                futures = {
                    executor.submit(self._execute_single_node, experiment_data, G, node_id): node_id
                    for node_id in level_nodes
                }
                
                # Process results as they complete
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        # Handle unexpected errors
                        logger.error(f"Error executing node {node_id}: {str(e)}")
                        experiment_data["results"][node_id] = {
                            "status": "error",
                            "message": f"Execution error: {str(e)}",
                            "error": str(e)
                        }
    
    def _execute_single_node(self, experiment_data: Dict[str, Any], G: nx.DiGraph, node_id: str) -> Dict[str, Any]:
        """Execute a single node in the workflow"""
        experiment_id = experiment_data.get("id", "unknown")
        
        # Find the node configuration
        node = next((n for n in experiment_data.get("nodes", []) if n["id"] == node_id), None)
        if not node:
            logger.error(f"Node {node_id} not found in experiment")
            return {"status": "error", "message": f"Node {node_id} not found"}
        
        # Check if the node has already been executed and cached
        cache = st.session_state.execution_cache.get(experiment_id, {})
        if node_id in cache and not node.get("data", {}).get("force_execution", False):
            # Use cached result
            experiment_data["results"][node_id] = cache[node_id]
            return cache[node_id]
        
        # Update node status to indicate execution
        for n in experiment_data["nodes"]:
            if n["id"] == node_id:
                if "data" not in n:
                    n["data"] = {}
                n["data"]["status"] = "running"
                break
        
        # Check if all input dependencies have been executed
        input_nodes = list(G.predecessors(node_id))
        for input_node in input_nodes:
            # Skip check if the input node is not found (shouldn't happen)
            if input_node not in experiment_data.get("results", {}):
                continue
                
            input_result = experiment_data["results"][input_node]
            if input_result.get("status") != "completed":
                logger.warning(f"Input node {input_node} for {node_id} has not completed successfully")
                # Update node status
                for n in experiment_data["nodes"]:
                    if n["id"] == node_id:
                        if "data" not in n:
                            n["data"] = {}
                        n["data"]["status"] = "waiting"
                        break
                
                return {"status": "waiting", "message": f"Waiting for input node {input_node}"}
        
        # Collect inputs from predecessor nodes
        inputs = {}
        for input_node in input_nodes:
            if input_node in experiment_data.get("results", {}):
                # Get the data from the input node result
                input_result = experiment_data["results"][input_node]
                if "data" in input_result:
                    inputs[input_node] = input_result["data"]
        
        # Get component configuration from node data
        component_id = node.get("data", {}).get("type", "unknown")
        node_config = node.get("data", {})
        
        # Log execution
        logger.info(f"Executing node {node_id} (type: {component_id})")
        
        try:
            # Start execution time
            start_time = time.time()
            
            # Execute the component
            result = execute_component(component_id, node_config, inputs)
            
            # End execution time
            end_time = time.time()
            result["execution_time"] = end_time - start_time
            
            # Update node status based on result
            for n in experiment_data["nodes"]:
                if n["id"] == node_id:
                    if "data" not in n:
                        n["data"] = {}
                    n["data"]["status"] = result.get("status", "unknown")
                    break
            
            # Store result in experiment data and cache
            experiment_data["results"][node_id] = result
            st.session_state.execution_cache.setdefault(experiment_id, {})[node_id] = result
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing node {node_id}: {str(e)}")
            
            # Update node status
            for n in experiment_data["nodes"]:
                if n["id"] == node_id:
                    if "data" not in n:
                        n["data"] = {}
                    n["data"]["status"] = "error"
                    break
            
            # Create error result
            error_result = {
                "status": "error",
                "message": f"Execution error: {str(e)}",
                "error": str(e)
            }
            
            # Store error result in experiment data
            experiment_data["results"][node_id] = error_result
            
            return error_result
    
    def get_node_status(self, experiment_data: Dict[str, Any], node_id: str) -> str:
        """Get the execution status of a node"""
        # Check in the node data first
        for node in experiment_data.get("nodes", []):
            if node["id"] == node_id and "data" in node and "status" in node["data"]:
                return node["data"]["status"]
        
        # Check in the results
        if node_id in experiment_data.get("results", {}):
            return experiment_data["results"][node_id].get("status", "unknown")
        
        # Default status
        return "draft"
    
    def reset_execution(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reset the execution state and results for an experiment"""
        experiment_id = experiment_data.get("id", "unknown")
        
        # Clear the results
        experiment_data["results"] = {}
        
        # Clear the execution cache
        if experiment_id in st.session_state.execution_cache:
            st.session_state.execution_cache[experiment_id] = {}
        
        # Reset the status on all nodes
        for node in experiment_data.get("nodes", []):
            if "data" in node:
                node["data"]["status"] = "draft"
        
        # Reset experiment status
        experiment_data["status"] = "draft"
        
        return experiment_data
    
    def get_execution_statistics(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the execution of an experiment"""
        statistics = {
            "total_nodes": len(experiment_data.get("nodes", [])),
            "executed_nodes": 0,
            "completed_nodes": 0,
            "error_nodes": 0,
            "waiting_nodes": 0,
            "draft_nodes": 0,
            "total_execution_time": 0
        }
        
        # Count nodes by status
        for node_id in [n["id"] for n in experiment_data.get("nodes", [])]:
            status = self.get_node_status(experiment_data, node_id)
            
            if status in ["completed", "error", "waiting", "running"]:
                statistics["executed_nodes"] += 1
            
            if status == "completed":
                statistics["completed_nodes"] += 1
            elif status == "error":
                statistics["error_nodes"] += 1
            elif status == "waiting":
                statistics["waiting_nodes"] += 1
            elif status == "draft":
                statistics["draft_nodes"] += 1
            
            # Add execution time if available
            if node_id in experiment_data.get("results", {}):
                result = experiment_data["results"][node_id]
                if "execution_time" in result:
                    statistics["total_execution_time"] += result["execution_time"]
        
        # Overall experiment execution time
        if "execution_time" in experiment_data:
            statistics["experiment_execution_time"] = experiment_data["execution_time"]
        
        return statistics


# Helper functions for use in the UI
def execute_experiment(experiment_data: Dict[str, Any], 
                     reset_cache: bool = False,
                     parallel: bool = False) -> Dict[str, Any]:
    """
    Execute an experiment workflow
    
    Args:
        experiment_data: The experiment data
        reset_cache: Whether to reset the execution cache
        parallel: Whether to use parallel execution
    
    Returns:
        Updated experiment data with results
    """
    engine = ExecutionEngine()
    return engine.execute_workflow(experiment_data, reset_cache, parallel)

def reset_experiment_execution(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Reset the execution state of an experiment"""
    engine = ExecutionEngine()
    return engine.reset_execution(experiment_data)

def get_execution_stats(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get execution statistics for an experiment"""
    engine = ExecutionEngine()
    return engine.get_execution_statistics(experiment_data)

def explain_execution_result(experiment_data: Dict[str, Any], node_id: str) -> str:
    """Generate an LLM explanation for a node's execution result"""
    # Check if LLM adapter is available
    if not hasattr(st.session_state, 'llm_adapter') or st.session_state.llm_adapter is None:
        return "LLM adapter not available. Please configure in Settings."
    
    # Find the node
    node = next((n for n in experiment_data.get("nodes", []) if n["id"] == node_id), None)
    if not node:
        return "Node not found in experiment."
    
    # Get the node result
    if node_id not in experiment_data.get("results", {}):
        return "No execution results available for this node."
    
    result = experiment_data["results"][node_id]
    
    # Create a system prompt for the LLM
    system_prompt = """You are an expert in causal discovery, statistical analysis, and data science.
    Your task is to explain execution results from a causal discovery workflow in a clear, accurate,
    and helpful manner. Focus on the practical implications, potential next steps, and any 
    important findings or issues that the user should be aware of."""
    
    # Create the user prompt
    user_prompt = f"""Please explain the following execution result from a causal discovery workflow.

Node Type: {node.get('data', {}).get('type', 'Unknown')}
Node Label: {node.get('data', {}).get('label', 'Unnamed Node')}
Execution Status: {result.get('status', 'Unknown')}

Result Message: {result.get('message', 'No message')}

Result Data: 
{json.dumps(result.get('data', {}), indent=2, default=str)}

Node Configuration:
{json.dumps(node.get('data', {}), indent=2, default=str)}

Please provide:
1. A summary of what this component was trying to do
2. An explanation of the result and what it means
3. Any issues or warnings that should be addressed
4. Recommendations for next steps or improvements
"""
    
    # Query the LLM
    try:
        response = st.session_state.llm_adapter.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        return response.get("completion", "Failed to generate explanation.")
    except Exception as e:
        return f"Error generating explanation: {str(e)}"