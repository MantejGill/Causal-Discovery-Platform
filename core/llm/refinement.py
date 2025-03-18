import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RefinementResult:
    """
    Class to hold the results of causal graph refinement.
    """
    
    def __init__(self, 
                graph: nx.DiGraph,
                edge_validations: Dict[Tuple[Any, Any], Dict[str, Any]],
                hidden_variables: List[Dict[str, Any]],
                hidden_var_mapping: Dict[str, Any],
                original_graph: Optional[nx.DiGraph] = None):
        """
        Initialize RefinementResult
        
        Args:
            graph: Refined causal graph
            edge_validations: Dictionary of edge validations
            hidden_variables: List of discovered hidden variables
            hidden_var_mapping: Mapping of hidden variable names to node IDs
            original_graph: Original graph before refinement (optional)
        """
        self.graph = graph
        self.edge_validations = edge_validations
        self.hidden_variables = hidden_variables
        self.hidden_var_mapping = hidden_var_mapping
        self.original_graph = original_graph
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any], original_graph: Optional[nx.DiGraph] = None):
        """
        Create a RefinementResult from a dictionary
        
        Args:
            result_dict: Dictionary with refinement results
            original_graph: Original graph before refinement (optional)
            
        Returns:
            RefinementResult instance
        """
        return cls(
            graph=result_dict.get("graph", nx.DiGraph()),
            edge_validations=result_dict.get("edge_validations", {}),
            hidden_variables=result_dict.get("hidden_variables", []),
            hidden_var_mapping=result_dict.get("hidden_var_mapping", {}),
            original_graph=original_graph
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Dictionary representation of the refinement result
        """
        return {
            "graph": self.graph,
            "edge_validations": self.edge_validations,
            "hidden_variables": self.hidden_variables,
            "hidden_var_mapping": self.hidden_var_mapping
        }
    
    def get_added_edges(self) -> List[Tuple[Any, Any]]:
        """
        Get edges that were added during refinement
        
        Returns:
            List of (source, target) tuples for added edges
        """
        if self.original_graph is None:
            return []
        
        original_edges = set(self.original_graph.edges())
        current_edges = set(self.graph.edges())
        
        return list(current_edges - original_edges)
    
    def get_removed_edges(self) -> List[Tuple[Any, Any]]:
        """
        Get edges that were removed during refinement
        
        Returns:
            List of (source, target) tuples for removed edges
        """
        if self.original_graph is None:
            return []
        
        original_edges = set(self.original_graph.edges())
        current_edges = set(self.graph.edges())
        
        return list(original_edges - current_edges)
    
    def get_hidden_nodes(self) -> List[Any]:
        """
        Get nodes that represent hidden variables
        
        Returns:
            List of node IDs for hidden variables
        """
        return list(self.hidden_var_mapping.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of refinement changes
        
        Returns:
            Dictionary with summary information
        """
        added_edges = self.get_added_edges()
        removed_edges = self.get_removed_edges()
        hidden_nodes = self.get_hidden_nodes()
        
        return {
            "num_added_edges": len(added_edges),
            "num_removed_edges": len(removed_edges),
            "num_hidden_variables": len(hidden_nodes),
            "added_edges": added_edges,
            "removed_edges": removed_edges,
            "hidden_nodes": hidden_nodes
        }

class CausalGraphRefiner:
    """
    Uses LLM to refine causal graphs, validate causal relationships,
    and discover hidden variables.
    """
    
    def __init__(self, llm_adapter):
        """
        Initialize CausalGraphRefiner
        
        Args:
            llm_adapter: LLM adapter for causal refinement
        """
        self.llm_adapter = llm_adapter
    
    def refine_graph(self, 
                    graph: nx.DiGraph, 
                    data: pd.DataFrame,
                    domain_knowledge: Optional[Dict[str, Any]] = None,
                    confidence_threshold: float = 0.5) -> RefinementResult:
        """
        Refine a causal graph using LLM validation
        
        Args:
            graph: NetworkX DiGraph representing the causal graph
            data: DataFrame containing the original data
            domain_knowledge: Optional domain knowledge
            confidence_threshold: Threshold for including edges based on LLM confidence
            
        Returns:
            RefinementResult object with refined graph and metadata
        """
        if not self.llm_adapter:
            raise ValueError("LLM adapter is required for graph refinement")
        
        # Create a copy of the original graph
        refined_graph = graph.copy()
        
        # Get variable names or use indices
        variable_names = {}
        for node in graph.nodes():
            if "name" in graph.nodes[node]:
                variable_names[node] = graph.nodes[node]["name"]
            else:
                variable_names[node] = data.columns[node] if node < len(data.columns) else f"Variable_{node}"
        
        # Prepare data summary for LLM context
        data_summary = self._create_data_summary(data)
        
        # Format domain knowledge if available
        domain_context = ""
        if domain_knowledge:
            domain_context = self._format_domain_knowledge(domain_knowledge)
        
        # Process edges for validation
        edge_validations = {}
        for u, v in graph.edges():
            # Skip bidirected edges (only process in one direction)
            if graph.has_edge(v, u) and "bidirected" in graph.edges[u, v]:
                if (v, u) in edge_validations:
                    continue
            
            # Validate edge u->v
            u_name = variable_names[u]
            v_name = variable_names[v]
            
            validation_result = self._validate_edge(u_name, v_name, data_summary, domain_context)
            edge_validations[(u, v)] = validation_result
            
            # Update edge in refined graph based on validation
            if validation_result["is_valid"]:
                if validation_result["confidence"] >= confidence_threshold:
                    # Keep the edge with updated attributes
                    refined_graph[u][v]["validation"] = validation_result
                    refined_graph[u][v]["confidence"] = validation_result["confidence"]
                else:
                    # Remove low-confidence edge
                    refined_graph.remove_edge(u, v)
            else:
                # Remove invalid edge
                refined_graph.remove_edge(u, v)
                
                # Check if the reverse edge was suggested
                if validation_result.get("reverse_suggested", False):
                    # Add the reverse edge
                    refined_graph.add_edge(v, u, 
                                          validation=validation_result,
                                          confidence=validation_result["confidence"],
                                          reversed=True)
        
        # Discover hidden variables
        hidden_variables = self._discover_hidden_variables(graph, variable_names, data_summary, domain_context)
        
        # Add hidden variables to the refined graph
        next_node_id = max(graph.nodes()) + 1 if graph.nodes() else 0
        hidden_var_mapping = {}
        
        for i, hidden_var in enumerate(hidden_variables):
            hidden_id = next_node_id + i
            hidden_var_mapping[hidden_var["name"]] = hidden_id
            
            # Add hidden variable node
            refined_graph.add_node(hidden_id, 
                                 name=hidden_var["name"], 
                                 description=hidden_var["description"],
                                 is_hidden=True,
                                 confidence=hidden_var["confidence"])
            
            # Add edges from hidden variable to its children
            for child_name in hidden_var["children"]:
                # Find child node id
                child_id = None
                for node, name in variable_names.items():
                    if name == child_name:
                        child_id = node
                        break
                
                if child_id is not None:
                    refined_graph.add_edge(hidden_id, child_id,
                                         confidence=hidden_var["edge_confidence"],
                                         is_hidden_cause=True)
        
        # Create and return RefinementResult object
        result = RefinementResult(
            graph=refined_graph,
            edge_validations=edge_validations,
            hidden_variables=hidden_variables,
            hidden_var_mapping=hidden_var_mapping,
            original_graph=graph
        )
        
        return result
    
    def _create_data_summary(self, data: pd.DataFrame) -> str:
        """
        Create a summary of the dataset for LLM context
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            String summary of dataset
        """
        n_samples, n_features = data.shape
        
        summary = f"Dataset with {n_samples} samples and {n_features} variables: {', '.join(data.columns)}\n\n"
        
        # Add basic statistics for each variable
        summary += "Variable statistics:\n"
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                summary += f"- {col}: mean={data[col].mean():.2f}, std={data[col].std():.2f}, min={data[col].min():.2f}, max={data[col].max():.2f}\n"
            else:
                summary += f"- {col}: {data[col].nunique()} unique values, most common: {data[col].value_counts().index[0]}\n"
        
        # Add correlation information
        try:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                corr_matrix = data[numeric_cols].corr()
                
                # Find strong correlations
                strong_corrs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) >= 0.5:
                            strong_corrs.append((col1, col2, corr))
                
                if strong_corrs:
                    summary += "\nStrong correlations:\n"
                    for col1, col2, corr in strong_corrs:
                        summary += f"- {col1} and {col2}: correlation={corr:.2f}\n"
        except Exception as e:
            logger.warning(f"Could not compute correlations: {str(e)}")
        
        return summary
    
    def _format_domain_knowledge(self, domain_knowledge: Dict[str, Any]) -> str:
        """
        Format domain knowledge for LLM context
        
        Args:
            domain_knowledge: Dictionary containing domain knowledge
            
        Returns:
            Formatted domain knowledge string
        """
        context = "Domain knowledge:\n"
        
        if "variables" in domain_knowledge:
            context += "Variable information:\n"
            for var, info in domain_knowledge["variables"].items():
                context += f"- {var}: {info.get('description', '')}\n"
        
        if "relationships" in domain_knowledge:
            context += "\nKnown relationships:\n"
            for rel in domain_knowledge["relationships"]:
                context += f"- {rel.get('description', '')}\n"
        
        if "constraints" in domain_knowledge:
            context += "\nConstrained relationships:\n"
            for constraint in domain_knowledge["constraints"]:
                context += f"- {constraint.get('description', '')}\n"
        
        return context
    
    def _validate_edge(self, 
                      source: str, 
                      target: str, 
                      data_summary: str, 
                      domain_context: str) -> Dict[str, Any]:
        """
        Validate a causal edge using LLM
        
        Args:
            source: Source variable name
            target: Target variable name
            data_summary: Summary of the dataset
            domain_context: Domain knowledge context
            
        Returns:
            Dictionary with validation results
        """
        system_prompt = """You are an expert in causal inference and data analysis. 
        Your task is to validate proposed causal relationships based on data and domain knowledge."""
        
        user_prompt = f"""
        I need to validate a potential causal relationship: {source} -> {target}
        
        {data_summary}
        
        {domain_context}
        
        Please analyze this proposed causal relationship and determine if it's valid based on the data statistics and domain knowledge.
        
        Think step by step:
        1. Is there a plausible mechanism by which {source} could directly cause {target}?
        2. Are there any temporal considerations (does {source} precede {target})?
        3. Are there any statistical indications supporting this relationship?
        4. Could this relationship be explained by a common cause instead?
        5. Could the causal direction be reversed ({target} -> {source})?
        
        After your analysis, provide a structured response:
        
        Valid: [Yes/No]
        Confidence: [1-10 scale]
        Reverse direction suggested: [Yes/No]
        Reasoning: [Your step-by-step reasoning]
        """
        
        try:
            # Use the complete method instead of query
            response = self.llm_adapter.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            # Get the completion text
            response_text = response.get("completion", "")
            
            # Parse response
            valid_line = [line for line in response_text.split('\n') if line.strip().startswith('Valid:')]
            confidence_line = [line for line in response_text.split('\n') if line.strip().startswith('Confidence:')]
            reverse_line = [line for line in response_text.split('\n') if line.strip().startswith('Reverse direction suggested:')]
            
            is_valid = False
            confidence = 0.5
            reverse_suggested = False
            reasoning = ""
            
            if valid_line:
                is_valid = "yes" in valid_line[0].split(':')[1].strip().lower()
            
            if confidence_line:
                try:
                    confidence_str = confidence_line[0].split(':')[1].strip()
                    confidence = float(confidence_str) / 10.0  # Convert 1-10 scale to 0-1
                except ValueError:
                    confidence = 0.5
            
            if reverse_line:
                reverse_suggested = "yes" in reverse_line[0].split(':')[1].strip().lower()
            
            # Extract reasoning
            reasoning_start = response_text.find("Reasoning:")
            if reasoning_start != -1:
                reasoning = response_text[reasoning_start + len("Reasoning:"):].strip()
            
            return {
                "is_valid": is_valid,
                "confidence": confidence,
                "reverse_suggested": reverse_suggested,
                "reasoning": reasoning,
                "source": source,
                "target": target
            }
        
        except Exception as e:
            logger.error(f"Error in edge validation: {str(e)}")
            return {
                "is_valid": True,  # Default to keeping edge on error
                "confidence": 0.5,
                "reverse_suggested": False,
                "reasoning": f"Error during validation: {str(e)}",
                "source": source,
                "target": target,
                "error": str(e)
            }
    
    def _discover_hidden_variables(self, 
                                 graph: nx.DiGraph, 
                                 variable_names: Dict[int, str],
                                 data_summary: str, 
                                 domain_context: str) -> List[Dict[str, Any]]:
        """
        Discover potential hidden variables
        
        Args:
            graph: Original causal graph
            variable_names: Mapping of node IDs to variable names
            data_summary: Summary of the dataset
            domain_context: Domain knowledge context
            
        Returns:
            List of discovered hidden variables
        """
        # Create graph description
        graph_desc = "Causal graph structure:\n"
        for u, v in graph.edges():
            u_name = variable_names[u]
            v_name = variable_names[v]
            graph_desc += f"- {u_name} -> {v_name}\n"
        
        # Look for unexplained correlations
        corr_desc = self._find_unexplained_correlations(graph, variable_names)
        
        system_prompt = """You are an expert in causal inference and latent variable discovery.
        Your task is to identify potential hidden variables in causal systems based on observed
        relationships and domain knowledge."""
        
        user_prompt = f"""
        I need to analyze a causal graph and discover potential hidden/latent variables that might be causing observed relationships.
        
        {data_summary}
        
        {graph_desc}
        
        {corr_desc}
        
        {domain_context}
        
        Please identify potential hidden variables that might explain patterns in the data or unaccounted correlations.
        For each hidden variable:
        1. Provide a name
        2. Explain what it represents
        3. List which observed variables it likely affects
        4. Estimate your confidence in its existence (1-10 scale)
        
        Format your response as a list of hidden variables:
        
        Hidden Variable 1:
        Name: [name]
        Description: [description]
        Affects: [list of affected variables]
        Confidence: [1-10]
        
        Hidden Variable 2:
        ...
        """
        
        try:
            # Use the complete method instead of query
            response = self.llm_adapter.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.4
            )
            
            # Get the completion text
            response_text = response.get("completion", "")
            
            # Parse response to extract hidden variables
            hidden_vars = []
            sections = response_text.split("Hidden Variable")[1:]  # Skip intro text
            
            for section in sections:
                lines = section.strip().split('\n')
                
                # Initialize variable data
                var_data = {
                    "name": "",
                    "description": "",
                    "children": [],
                    "confidence": 0.5,
                    "edge_confidence": 0.5  # Default edge confidence
                }
                
                for line in lines:
                    if line.startswith("Name:"):
                        var_data["name"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Description:"):
                        var_data["description"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Affects:"):
                        affects = line.split(":", 1)[1].strip()
                        var_data["children"] = [v.strip() for v in affects.split(",")]
                    elif line.startswith("Confidence:"):
                        try:
                            conf = float(line.split(":", 1)[1].strip())
                            var_data["confidence"] = conf / 10.0  # Convert to 0-1 scale
                            var_data["edge_confidence"] = conf / 10.0  # Same for edges
                        except ValueError:
                            pass
                
                if var_data["name"] and var_data["children"]:
                    hidden_vars.append(var_data)
            
            return hidden_vars
        
        except Exception as e:
            logger.error(f"Error in hidden variable discovery: {str(e)}")
            return []
    
    def _find_unexplained_correlations(self, 
                                     graph: nx.DiGraph, 
                                     variable_names: Dict[int, str]) -> str:
        """
        Find patterns that might suggest hidden variables
        
        Args:
            graph: Original causal graph
            variable_names: Mapping of node IDs to variable names
            
        Returns:
            Description of unexplained patterns
        """
        desc = "Patterns suggesting hidden variables:\n"
        
        # Find nodes with common children
        nodes_with_common_children = {}
        for node in graph.nodes():
            children = list(graph.successors(node))
            if len(children) >= 2:
                node_name = variable_names[node]
                child_names = [variable_names[c] for c in children]
                desc += f"- {node_name} affects multiple variables: {', '.join(child_names)}\n"
        
        # Find variables with no parents (potential confounders)
        for node in graph.nodes():
            if graph.in_degree(node) == 0 and graph.out_degree(node) >= 2:
                node_name = variable_names[node]
                desc += f"- {node_name} has no parents but affects other variables\n"
        
        # Find unexplained connections (nodes that are correlated but not connected in graph)
        # This would require correlation data which we don't have here
        # For now, we'll add a general note
        desc += "- There may be variables that are correlated but lack direct connections in the graph\n"
        
        # Find clusters of nodes (suggesting common cause)
        # Simple heuristic: nodes that share multiple edges
        for i in graph.nodes():
            for j in graph.nodes():
                if i < j:  # To avoid duplicates
                    # Find common neighbors
                    i_neighbors = set(list(graph.successors(i)) + list(graph.predecessors(i)))
                    j_neighbors = set(list(graph.successors(j)) + list(graph.predecessors(j)))
                    common = i_neighbors.intersection(j_neighbors)
                    
                    if len(common) >= 2:
                        i_name = variable_names[i]
                        j_name = variable_names[j]
                        common_names = [variable_names[c] for c in common]
                        desc += f"- {i_name} and {j_name} share multiple connections: {', '.join(common_names)}\n"
        
        return desc