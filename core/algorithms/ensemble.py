# core/algorithms/ensemble.py
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlgorithmEnsemble:
    """
    Combines results from multiple causal discovery algorithms 
    to create a consensus causal graph.
    """
    
    def __init__(self, llm_adapter=None):
        """
        Initialize AlgorithmEnsemble
        
        Args:
            llm_adapter: Optional LLM adapter for resolving conflicts
        """
        self.llm_adapter = llm_adapter
    
    def create_ensemble_graph(self, 
                             algorithm_results: List[Dict[str, Any]], 
                             edge_threshold: float = 0.5,
                             node_names: Optional[List[str]] = None,
                             resolve_conflicts: bool = True,
                             domain_knowledge: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an ensemble causal graph from multiple algorithm results
        
        Args:
            algorithm_results: List of results from AlgorithmExecutor
            edge_threshold: Threshold for including edges in ensemble
            node_names: Optional list of node names for the graph
            resolve_conflicts: Whether to use LLM to resolve edge conflicts
            domain_knowledge: Optional domain knowledge for conflict resolution
            
        Returns:
            Dictionary with ensemble graph and metadata
        """
        # Check if results are valid
        if not algorithm_results:
            raise ValueError("No algorithm results provided")
        
        valid_results = [r for r in algorithm_results if r.get("status") == "success"]
        if not valid_results:
            raise ValueError("No valid algorithm results provided")
        
        # Get number of nodes from first valid result
        first_graph = valid_results[0]["graph"]
        n_nodes = len(first_graph.nodes())
        
        # Check that all graphs have the same number of nodes
        for result in valid_results:
            if len(result["graph"].nodes()) != n_nodes:
                raise ValueError(f"Inconsistent number of nodes in graphs: expected {n_nodes}, got {len(result['graph'].nodes())}")
        
        # Initialize edge counts and directions
        edge_counts = np.zeros((n_nodes, n_nodes))
        edge_directions = np.zeros((n_nodes, n_nodes))
        total_algorithms = len(valid_results)
        
        # Count edges and directions
        for result in valid_results:
            G = result["graph"]
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        continue
                    
                    # Check if edge i->j exists
                    if G.has_edge(i, j):
                        edge_counts[i, j] += 1
                        
                        # Check if bidirected (i<->j)
                        if G.has_edge(j, i) and "bidirected" in G.edges[i, j]:
                            edge_directions[i, j] = 0  # Bidirected or undirected
                        else:
                            edge_directions[i, j] += 1  # Directed i->j
        
        # Normalize counts
        edge_probs = edge_counts / total_algorithms
        
        # Create the ensemble graph
        G_ensemble = nx.DiGraph()
        G_ensemble.add_nodes_from(range(n_nodes))
        
        # Add node names if provided
        if node_names:
            if len(node_names) != n_nodes:
                raise ValueError(f"Length of node_names ({len(node_names)}) does not match number of nodes ({n_nodes})")
            for i in range(n_nodes):
                G_ensemble.nodes[i]["name"] = node_names[i]
        
        # Identify common edges, conflicting edges, and unique edges
        common_edges = []
        conflicting_edges = []
        unique_edges = []
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                
                if edge_probs[i, j] >= edge_threshold and edge_probs[j, i] >= edge_threshold:
                    # Conflict: both directions have high probability
                    conflicting_edges.append((i, j))
                elif edge_probs[i, j] >= edge_threshold:
                    # Common edge: i->j
                    common_edges.append((i, j))
                elif edge_probs[j, i] >= edge_threshold:
                    # Common edge: j->i
                    common_edges.append((j, i))
                elif edge_probs[i, j] > 0 and edge_probs[i, j] < edge_threshold:
                    # Unique edge: some evidence for i->j but below threshold
                    unique_edges.append((i, j))
                elif edge_probs[j, i] > 0 and edge_probs[j, i] < edge_threshold:
                    # Unique edge: some evidence for j->i but below threshold
                    unique_edges.append((j, i))
        
        # Resolve conflicting edges using LLM if available
        resolved_conflicts = {}
        if resolve_conflicts and self.llm_adapter and conflicting_edges:
            resolved_conflicts = self._resolve_conflicts_with_llm(
                conflicting_edges, node_names, domain_knowledge
            )
        
        # Add edges to ensemble graph
        for i, j in common_edges:
            G_ensemble.add_edge(i, j, weight=edge_probs[i, j], confidence=edge_probs[i, j])
        
        # Add resolved conflicts
        for (i, j), direction in resolved_conflicts.items():
            if direction == "forward":
                G_ensemble.add_edge(i, j, weight=edge_probs[i, j], confidence=edge_probs[i, j], resolved=True)
            elif direction == "backward":
                G_ensemble.add_edge(j, i, weight=edge_probs[j, i], confidence=edge_probs[j, i], resolved=True)
            elif direction == "bidirected":
                G_ensemble.add_edge(i, j, weight=edge_probs[i, j], confidence=edge_probs[i, j], bidirected=True, resolved=True)
                G_ensemble.add_edge(j, i, weight=edge_probs[j, i], confidence=edge_probs[j, i], bidirected=True, resolved=True)
        
        # Add unresolved conflicts as bidirected edges
        for i, j in conflicting_edges:
            if (i, j) not in resolved_conflicts and (j, i) not in resolved_conflicts:
                G_ensemble.add_edge(i, j, weight=edge_probs[i, j], confidence=edge_probs[i, j], bidirected=True, unresolved=True)
                G_ensemble.add_edge(j, i, weight=edge_probs[j, i], confidence=edge_probs[j, i], bidirected=True, unresolved=True)
        
        return {
            "graph": G_ensemble,
            "algorithm_count": total_algorithms,
            "edge_probs": edge_probs,
            "common_edges": common_edges,
            "conflicting_edges": conflicting_edges,
            "unique_edges": unique_edges,
            "resolved_conflicts": resolved_conflicts
        }
    
    def _resolve_conflicts_with_llm(self, 
                                   conflicting_edges: List[Tuple[int, int]], 
                                   node_names: Optional[List[str]] = None,
                                   domain_knowledge: Optional[Dict[str, Any]] = None) -> Dict[Tuple[int, int], str]:
        """
        Use LLM to resolve conflicting edges in the causal graph
        
        Args:
            conflicting_edges: List of conflicting edge tuples (i, j)
            node_names: Optional list of node names
            domain_knowledge: Optional domain knowledge
            
        Returns:
            Dictionary mapping edge tuples to resolved directions ('forward', 'backward', or 'bidirected')
        """
        if not self.llm_adapter:
            return {}
        
        resolved_conflicts = {}
        
        for i, j in conflicting_edges:
            i_name = node_names[i] if node_names else f"Variable {i}"
            j_name = node_names[j] if node_names else f"Variable {j}"
            
            # Format domain knowledge if available
            domain_context = ""
            if domain_knowledge:
                if "variables" in domain_knowledge:
                    var_info = domain_knowledge["variables"]
                    i_info = var_info.get(i_name, {}).get("description", "")
                    j_info = var_info.get(j_name, {}).get("description", "")
                    
                    if i_info:
                        domain_context += f"Information about {i_name}: {i_info}\n"
                    if j_info:
                        domain_context += f"Information about {j_name}: {j_info}\n"
                
                if "relationships" in domain_knowledge:
                    rel_info = [r for r in domain_knowledge["relationships"] 
                               if (r.get("source") == i_name and r.get("target") == j_name) or
                                  (r.get("source") == j_name and r.get("target") == i_name)]
                    
                    if rel_info:
                        domain_context += "Known relationships:\n"
                        for r in rel_info:
                            domain_context += f"- {r.get('description', '')}\n"
            
            # Construct prompt
            prompt = f"""
            I need to determine the causal relationship between these two variables:
            - {i_name}
            - {j_name}

            {domain_context if domain_context else ""}

            Given the information, which of the following is most likely:
            1. {i_name} causes {j_name}
            2. {j_name} causes {i_name}
            3. Both have a bidirectional relationship or share a common cause
            
            Think step by step about the most plausible causal relationship. Provide a short reasoning for your choice.

            Output format:
            Direction: [1/2/3]
            Confidence: [1-10]
            Reasoning: Your explanation here
            """
            
            # Query LLM
            try:
                response = self.llm_adapter.query(prompt)
                
                # Parse response
                direction_line = [line for line in response.split('\n') if line.strip().startswith('Direction:')]
                confidence_line = [line for line in response.split('\n') if line.strip().startswith('Confidence:')]
                
                if direction_line and confidence_line:
                    direction_value = direction_line[0].split(':')[1].strip()
                    confidence_value = int(confidence_line[0].split(':')[1].strip())
                    
                    if direction_value == '1':
                        resolved_conflicts[(i, j)] = "forward"
                    elif direction_value == '2':
                        resolved_conflicts[(i, j)] = "backward"
                    elif direction_value == '3':
                        resolved_conflicts[(i, j)] = "bidirected"
                    
                    logger.info(f"Resolved conflict between {i_name} and {j_name}: {resolved_conflicts.get((i, j))}")
                else:
                    logger.warning(f"Failed to parse LLM response for conflict between {i_name} and {j_name}")
            
            except Exception as e:
                logger.error(f"Error resolving conflict with LLM: {str(e)}")
        
        return resolved_conflicts