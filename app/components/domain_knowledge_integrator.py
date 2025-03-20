# app/components/domain_knowledge_integrator.py

import streamlit as st
import pandas as pd
import networkx as nx
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainKnowledgeIntegrator:
    """
    Integrates domain knowledge into causal discovery process.
    Uses LLM to structure domain expertise and apply it to causal graphs.
    """
    
    def __init__(self, llm_adapter=None):
        """
        Initialize the domain knowledge integrator
        
        Args:
            llm_adapter: LLM adapter for processing domain knowledge
        """
        self.llm_adapter = llm_adapter
        self.domain_knowledge = {}
    
    def extract_knowledge_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured domain knowledge from text using LLM
        
        Args:
            text: Free-form text containing domain knowledge
            
        Returns:
            Dictionary of structured domain knowledge
        """
        if not self.llm_adapter:
            raise ValueError("LLM adapter is required to extract domain knowledge")
        
        # Create a system prompt for the LLM
        system_prompt = """You are an expert in causal relationships and domain knowledge formalization.
        Your task is to extract structured domain knowledge from free-form text and organize it in a way
        that can be used for causal discovery. Focus on identifying variables, relationships between variables,
        and constraints that should be enforced in a causal graph."""
        
        # Create the user prompt
        user_prompt = f"""Please extract structured domain knowledge from the following text for use in causal discovery:

{text}

Please structure the knowledge in the following format:
1. Variables: List of variables mentioned in the text with their descriptions
2. Relationships: Known causal relationships between variables
3. Constraints: Constraints on the causal structure (e.g., A cannot cause B)
4. Background: General background knowledge about the domain

Return your response as a structured JSON object with the following format:
{{
    "variables": {{
        "variable_name": {{
            "description": "description of the variable",
            "type": "continuous/categorical/binary",
            "possible_values": ["value1", "value2"] // for categorical variables
        }},
        // more variables...
    }},
    "relationships": [
        {{
            "source": "source_variable",
            "target": "target_variable",
            "description": "description of the relationship",
            "confidence": 0.9, // 0-1 scale
            "type": "direct/indirect/correlation"
        }},
        // more relationships...
    ],
    "constraints": [
        {{
            "type": "forbidden_edge/required_edge/timing",
            "source": "source_variable", // if applicable
            "target": "target_variable", // if applicable
            "description": "description of the constraint"
        }},
        // more constraints...
    ],
    "background": "General background knowledge about the domain"
}}

Focus on extracting specific, actionable knowledge that can guide causal discovery.
"""
        
        # Call the LLM to extract domain knowledge
        try:
            response = self.llm_adapter.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for more structured output
                max_tokens=2000
            )
            
            # Parse the JSON response
            response_text = response.get("completion", "")
            
            # Extract the JSON part from the response (it might be wrapped in markdown code blocks)
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            
            knowledge = json.loads(json_str)
            
            # Store the extracted knowledge
            self.domain_knowledge = knowledge
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Error extracting domain knowledge: {str(e)}")
            # Return empty structure on error
            return {
                "variables": {},
                "relationships": [],
                "constraints": [],
                "background": ""
            }
    
    def apply_to_graph(self, 
                       graph: nx.DiGraph, 
                       knowledge: Optional[Dict[str, Any]] = None,
                       confidence_threshold: float = 0.7) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Apply domain knowledge to refine a causal graph
        
        Args:
            graph: NetworkX DiGraph to refine
            knowledge: Domain knowledge to apply (uses stored knowledge if None)
            confidence_threshold: Minimum confidence to apply a relationship
            
        Returns:
            Tuple of (refined graph, changes made)
        """
        if knowledge is None:
            knowledge = self.domain_knowledge
        
        if not knowledge:
            return graph, {"no_changes": True, "reason": "No domain knowledge provided"}
        
        # Create a copy of the graph to modify
        refined_graph = graph.copy()
        
        # Track changes made to the graph
        changes = {
            "added_edges": [],
            "removed_edges": [],
            "direction_changed": [],
            "attributes_updated": []
        }
        
        # Get node names from graph
        node_names = {}
        for node in graph.nodes():
            if "name" in graph.nodes[node]:
                node_names[graph.nodes[node]["name"]] = node
        
        # 1. Apply relationship knowledge
        for relationship in knowledge.get("relationships", []):
            source = relationship.get("source")
            target = relationship.get("target")
            confidence = relationship.get("confidence", 0.8)
            rel_type = relationship.get("type", "direct")
            description = relationship.get("description", "")
            
            # Skip if confidence is below threshold
            if confidence < confidence_threshold:
                continue
            
            # Find node IDs for source and target
            source_node = node_names.get(source)
            target_node = node_names.get(target)
            
            # If node names not found in graph, try direct matching
            if source_node is None and source in graph.nodes():
                source_node = source
            if target_node is None and target in graph.nodes():
                target_node = target
            
            # Skip if either node is not in the graph
            if source_node is None or target_node is None:
                continue
            
            # Apply the relationship based on its type
            if rel_type == "direct":
                # Check if the edge already exists in the opposite direction
                if refined_graph.has_edge(target_node, source_node):
                    # Remove the existing edge if we're confident about the direction
                    if confidence > 0.8:
                        refined_graph.remove_edge(target_node, source_node)
                        changes["direction_changed"].append((target_node, source_node, source_node, target_node))
                
                # Add or update the edge
                if not refined_graph.has_edge(source_node, target_node):
                    refined_graph.add_edge(source_node, target_node, 
                                         weight=confidence, 
                                         confidence=confidence,
                                         description=description,
                                         from_domain_knowledge=True)
                    changes["added_edges"].append((source_node, target_node))
                else:
                    # Update edge attributes
                    refined_graph[source_node][target_node]["confidence"] = confidence
                    refined_graph[source_node][target_node]["description"] = description
                    refined_graph[source_node][target_node]["from_domain_knowledge"] = True
                    changes["attributes_updated"].append((source_node, target_node))
            
            elif rel_type == "correlation" or rel_type == "indirect":
                # For correlation or indirect relationship, we might add bidirected edges
                # or update existing edges with lower confidence
                
                # Only add/update if neither direction exists
                if not refined_graph.has_edge(source_node, target_node) and not refined_graph.has_edge(target_node, source_node):
                    # For indirect/correlation, add with lower confidence
                    mod_confidence = min(confidence, 0.7)  # Cap confidence for indirect relationships
                    refined_graph.add_edge(source_node, target_node, 
                                         weight=mod_confidence, 
                                         confidence=mod_confidence,
                                         description=description,
                                         bidirected=True,
                                         from_domain_knowledge=True)
                    refined_graph.add_edge(target_node, source_node, 
                                         weight=mod_confidence, 
                                         confidence=mod_confidence,
                                         description=description,
                                         bidirected=True,
                                         from_domain_knowledge=True)
                    changes["added_edges"].append((source_node, target_node))
                    changes["added_edges"].append((target_node, source_node))
        
        # 2. Apply constraints
        for constraint in knowledge.get("constraints", []):
            constraint_type = constraint.get("type", "")
            source = constraint.get("source")
            target = constraint.get("target")
            description = constraint.get("description", "")
            
            # Find node IDs for source and target
            source_node = node_names.get(source)
            target_node = node_names.get(target)
            
            # If node names not found in graph, try direct matching
            if source_node is None and source in graph.nodes():
                source_node = source
            if target_node is None and target in graph.nodes():
                target_node = target
            
            # Skip if either node is not in the graph
            if constraint_type in ["forbidden_edge", "required_edge"] and (source_node is None or target_node is None):
                continue
            
            # Apply the constraint based on its type
            if constraint_type == "forbidden_edge":
                # Remove the edge if it exists
                if refined_graph.has_edge(source_node, target_node):
                    refined_graph.remove_edge(source_node, target_node)
                    changes["removed_edges"].append((source_node, target_node))
                    
                # Also remove the edge in the opposite direction
                if refined_graph.has_edge(target_node, source_node):
                    refined_graph.remove_edge(target_node, source_node)
                    changes["removed_edges"].append((target_node, source_node))
            
            elif constraint_type == "required_edge":
                # Add the edge if it doesn't exist
                if not refined_graph.has_edge(source_node, target_node):
                    refined_graph.add_edge(source_node, target_node, 
                                         weight=1.0, 
                                         confidence=1.0,
                                         description=description,
                                         from_domain_knowledge=True,
                                         required=True)
                    changes["added_edges"].append((source_node, target_node))
                    
                    # Remove the edge in the opposite direction if it exists
                    if refined_graph.has_edge(target_node, source_node):
                        refined_graph.remove_edge(target_node, source_node)
                        changes["removed_edges"].append((target_node, source_node))
            
            # For timing constraints, we could implement temporal ordering checks
            # but this would require additional information about the graph
        
        # Update node attributes with variable descriptions from domain knowledge
        for var_name, var_info in knowledge.get("variables", {}).items():
            # Find the node in the graph
            node_id = node_names.get(var_name)
            
            # If node name not found in graph, try direct matching
            if node_id is None and var_name in graph.nodes():
                node_id = var_name
            
            # Skip if node is not in the graph
            if node_id is None:
                continue
            
            # Update node attributes
            refined_graph.nodes[node_id]["description"] = var_info.get("description", "")
            refined_graph.nodes[node_id]["variable_type"] = var_info.get("type", "")
            
            if "possible_values" in var_info:
                refined_graph.nodes[node_id]["possible_values"] = var_info["possible_values"]
        
        return refined_graph, changes
    
    def identify_missing_variables(self, 
                                  graph: nx.DiGraph, 
                                  data: pd.DataFrame,
                                  text: str) -> List[Dict[str, Any]]:
        """
        Use domain knowledge to identify potentially missing variables (confounders)
        
        Args:
            graph: NetworkX DiGraph to analyze
            data: DataFrame with the data
            text: Domain knowledge text
            
        Returns:
            List of dictionaries describing potential missing variables
        """
        if not self.llm_adapter:
            raise ValueError("LLM adapter is required to identify missing variables")
        
        # Extract structured domain knowledge first
        knowledge = self.extract_knowledge_from_text(text)
        
        # Create a system prompt for the LLM
        system_prompt = """You are an expert in causal discovery and identifying hidden variables.
        Your task is to analyze a causal graph and domain knowledge to identify potentially missing
        variables that could be important confounders in the causal structure."""
        
        # Create a representation of the graph
        graph_repr = []
        for u, v in graph.edges():
            source = graph.nodes[u].get("name", str(u))
            target = graph.nodes[v].get("name", str(v))
            graph_repr.append(f"{source} -> {target}")
        
        # Get data summary
        data_summary = f"Dataset with {data.shape[0]} rows and {data.shape[1]} columns.\n"
        data_summary += f"Columns: {', '.join(data.columns)}"
        
        # Create the user prompt
        user_prompt = f"""Please analyze this causal graph and domain knowledge to identify potentially missing variables (confounders):

## Current Causal Graph
{'\n'.join(graph_repr)}

## Data Summary
{data_summary}

## Domain Knowledge
{text}

Based on the graph, data, and domain knowledge, identify potentially missing variables that:
1. Are mentioned in the domain knowledge but not present in the graph
2. Could explain suspicious associations between variables
3. Would make the graph more consistent with domain knowledge
4. Are common confounders in this domain

For each potential missing variable, provide:
1. Name: A descriptive name for the variable
2. Description: What this variable represents
3. Influenced Variables: Which observed variables it likely affects
4. Evidence: Why you believe this variable might be missing
5. Importance: How important this variable is (1-10 scale)

Format your response as a JSON array of missing variable objects:
[
  {{
    "name": "variable_name",
    "description": "description of what this variable represents",
    "influenced_variables": ["var1", "var2", ...],
    "evidence": "explanation of why this variable might be missing",
    "importance": 8 // 1-10 scale
  }},
  // more variables...
]

Focus on identifying meaningful confounders that would significantly impact causal inference if included.
"""
        
        # Call the LLM
        try:
            response = self.llm_adapter.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse the JSON response
            response_text = response.get("completion", "")
            
            # Extract the JSON part from the response
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            
            missing_vars = json.loads(json_str)
            
            return missing_vars
            
        except Exception as e:
            logger.error(f"Error identifying missing variables: {str(e)}")
            return []
    
    def suggest_relationship_directions(self, 
                                      graph: nx.DiGraph, 
                                      undirected_edges: List[Tuple[Any, Any]],
                                      domain_knowledge: str) -> Dict[Tuple[Any, Any], str]:
        """
        Suggest directions for undirected or conflicting edges using domain knowledge
        
        Args:
            graph: NetworkX DiGraph with undirected edges
            undirected_edges: List of edge tuples that need direction assignment
            domain_knowledge: Domain knowledge text
            
        Returns:
            Dictionary mapping edges to suggested directions ("forward", "backward", "bidirected")
        """
        if not self.llm_adapter:
            raise ValueError("LLM adapter is required to suggest edge directions")
        
        # No edges to resolve
        if not undirected_edges:
            return {}
        
        # Create a system prompt for the LLM
        system_prompt = """You are an expert in causal inference and domain knowledge application.
        Your task is to suggest directions for undirected or conflicting edges in a causal graph
        based on domain knowledge and causal reasoning principles."""
        
        # Create a representation of the graph
        graph_repr = []
        for u, v in graph.edges():
            source = graph.nodes[u].get("name", str(u))
            target = graph.nodes[v].get("name", str(v))
            
            # Mark undirected edges
            if (u, v) in undirected_edges or (v, u) in undirected_edges:
                graph_repr.append(f"{source} -- {target}")
            else:
                graph_repr.append(f"{source} -> {target}")
        
        # Create the user prompt
        user_prompt = f"""Please suggest directions for the undirected edges in this causal graph based on domain knowledge:

## Current Causal Graph
{'\n'.join(graph_repr)}

## Undirected Edges to Resolve
{'\n'.join([f"{graph.nodes[u].get('name', str(u))} -- {graph.nodes[v].get('name', str(v))}" for u, v in undirected_edges])}

## Domain Knowledge
{domain_knowledge}

For each undirected edge, determine the most likely causal direction based on the domain knowledge.
Consider timing information, cause-effect relationships mentioned in the domain knowledge,
and general causal principles.

For each edge, suggest one of:
- "forward": The first node causes the second (A -> B)
- "backward": The second node causes the first (A <- B)
- "bidirected": Both directions are plausible or there's a hidden common cause (A <-> B)

Format your response as a JSON object where keys are edge identifiers and values are the suggested directions:
{{
  "edge1": "forward",
  "edge2": "backward",
  "edge3": "bidirected"
}}

Provide a brief explanation for each decision.
"""
        
        # Call the LLM
        try:
            response = self.llm_adapter.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse the JSON response
            response_text = response.get("completion", "")
            
            # Extract the JSON part from the response
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            
            # The response format might need processing to match our expected output
            directions_raw = json.loads(json_str)
            
            # Convert to the expected format (map edge tuples to directions)
            directions = {}
            
            # Create a mapping from edge names to edge tuples
            edge_map = {}
            for i, edge in enumerate(undirected_edges):
                edge_name = f"edge{i+1}"
                edge_map[edge_name] = edge
            
            # Also try with the format "node1--node2"
            for edge in undirected_edges:
                u, v = edge
                u_name = graph.nodes[u].get("name", str(u))
                v_name = graph.nodes[v].get("name", str(v))
                edge_name = f"{u_name}--{v_name}"
                edge_map[edge_name] = edge
                
                # Try the reverse format too
                edge_name_rev = f"{v_name}--{u_name}"
                edge_map[edge_name_rev] = edge
            
            # Process each direction from the response
            for key, direction in directions_raw.items():
                if key in edge_map:
                    directions[edge_map[key]] = direction
            
            # If formats don't match, try to process the result another way
            if not directions and len(directions_raw) == len(undirected_edges):
                # Assume the response is in order of the undirected_edges
                for i, edge in enumerate(undirected_edges):
                    directions[edge] = list(directions_raw.values())[i]
            
            return directions
            
        except Exception as e:
            logger.error(f"Error suggesting edge directions: {str(e)}")
            return {}

def render_domain_knowledge_editor():
    """
    Render the domain knowledge editor component in the Streamlit UI
    
    Returns:
        The entered domain knowledge text
    """
    # Check if previous domain knowledge exists in session state
    if "domain_knowledge_text" not in st.session_state:
        st.session_state.domain_knowledge_text = ""
    
    # Title and description
    st.subheader("Domain Knowledge Editor")
    st.markdown("""
    Enter domain knowledge in natural language. Describe causal relationships, constraints, 
    and background knowledge about the variables in your dataset. This information will be 
    used to refine the causal graph.
    """)
    
    # Examples dropdown
    with st.expander("See examples of domain knowledge descriptions"):
        st.markdown("""
        ### Example 1: Medical
        
        In hypertension, age directly affects blood pressure. Smoking causes both increased blood pressure and decreased lung function. 
        Exercise improves cardiovascular health and reduces blood pressure. Salt intake increases blood pressure. 
        Genetic factors influence both blood pressure and response to medications. Blood pressure medication cannot cause age or genetic factors.
        
        ### Example 2: Economics
        
        Education level directly influences income. Job experience affects income and job satisfaction.
        Economic conditions impact unemployment rates and market demand. Government policies affect tax rates and business regulations.
        Income cannot cause education level in our model. Age influences both education level and job experience.
        
        ### Example 3: Environmental
        
        Temperature affects plant growth rate. Rainfall directly influences soil moisture and plant growth.
        Pollution levels impact both air quality and water quality. Solar radiation affects temperature and plant photosynthesis.
        Geographic location influences rainfall patterns and temperature. Human activity affects pollution levels but not geographic location.
        """)
    
    # Domain knowledge text area
    domain_knowledge_text = st.text_area(
        "Enter domain knowledge",
        value=st.session_state.domain_knowledge_text,
        height=200,
        placeholder="Describe causal relationships and constraints in natural language..."
    )
    
    # Update session state
    st.session_state.domain_knowledge_text = domain_knowledge_text
    
    # Add formatting options (could be expanded in a more sophisticated implementation)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear"):
            st.session_state.domain_knowledge_text = ""
            st.rerun()
    
    with col2:
        if st.button("Reset to Example"):
            st.session_state.domain_knowledge_text = """
            Age directly affects blood pressure. Smoking causes both increased blood pressure and decreased lung function. 
            Exercise improves cardiovascular health and reduces blood pressure. Salt intake increases blood pressure. 
            Genetic factors influence both blood pressure and response to medications. Blood pressure medication cannot cause age or genetic factors.
            """
            st.rerun()
    
    return domain_knowledge_text

def render_domain_knowledge_integration(graph: nx.DiGraph, domain_knowledge_text: str):
    """
    Render the domain knowledge integration component in the Streamlit UI
    
    Args:
        graph: The causal graph to refine
        domain_knowledge_text: The domain knowledge text
        
    Returns:
        The refined graph if integration was performed
    """
    st.subheader("Domain Knowledge Integration")
    
    # Create the integrator
    integrator = DomainKnowledgeIntegrator(st.session_state.llm_adapter)
    
    # Show integration options
    st.markdown("Apply domain knowledge to refine the causal graph")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence level to apply a relationship from domain knowledge"
        )
    
    with col2:
        structuring_method = st.selectbox(
            "Knowledge Structuring Method",
            options=["Automatic (LLM)", "Manual Specification"],
            index=0,
            help="How to extract structured knowledge from the text"
        )
    
    # Button to apply domain knowledge
    if st.button("Apply Domain Knowledge", type="primary"):
        if not domain_knowledge_text:
            st.warning("Please enter domain knowledge text first.")
            return graph
        
        if not st.session_state.llm_adapter:
            st.error("LLM adapter not available. Please configure in Settings page.")
            return graph
        
        # Process based on selected method
        with st.spinner("Processing domain knowledge..."):
            if structuring_method == "Automatic (LLM)":
                # Extract knowledge using LLM
                knowledge = integrator.extract_knowledge_from_text(domain_knowledge_text)
                
                # Show the extracted knowledge
                with st.expander("View Extracted Knowledge Structure"):
                    st.json(knowledge)
                
                # Apply the knowledge to the graph
                refined_graph, changes = integrator.apply_to_graph(
                    graph=graph,
                    knowledge=knowledge,
                    confidence_threshold=confidence_threshold
                )
                
                # Show changes
                st.subheader("Applied Changes")
                
                if changes.get("no_changes", False):
                    st.info("No changes were made to the graph.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Added Edges")
                        if changes["added_edges"]:
                            for edge in changes["added_edges"]:
                                source, target = edge
                                source_name = graph.nodes[source].get("name", str(source))
                                target_name = graph.nodes[target].get("name", str(target))
                                st.write(f"- {source_name} → {target_name}")
                        else:
                            st.write("No edges added")
                    
                    with col2:
                        st.markdown("### Removed Edges")
                        if changes["removed_edges"]:
                            for edge in changes["removed_edges"]:
                                source, target = edge
                                source_name = graph.nodes[source].get("name", str(source))
                                target_name = graph.nodes[target].get("name", str(target))
                                st.write(f"- {source_name} → {target_name}")
                        else:
                            st.write("No edges removed")
                
                return refined_graph
                
            else:  # Manual Specification
                st.warning("Manual specification interface not implemented yet.")
                
                # In a complete implementation, this would include a UI for
                # manually specifying relationships and constraints
                
                return graph
    
    # Return original graph if no integration performed
    return graph

def identify_missing_variables_ui(graph: nx.DiGraph, data: pd.DataFrame, domain_knowledge_text: str):
    """
    UI component for identifying missing variables
    
    Args:
        graph: The causal graph to analyze
        data: The dataset
        domain_knowledge_text: Domain knowledge text
        
    Returns:
        List of identified missing variables
    """
    st.subheader("Identify Missing Variables")
    
    st.markdown("""
    Use domain knowledge to identify potentially missing variables (confounders) 
    that might improve the causal graph.
    """)
    
    if st.button("Identify Missing Variables"):
        if not domain_knowledge_text:
            st.warning("Please enter domain knowledge text first.")
            return []
        
        if not st.session_state.llm_adapter:
            st.error("LLM adapter not available. Please configure in Settings page.")
            return []
        
        # Create the integrator
        integrator = DomainKnowledgeIntegrator(st.session_state.llm_adapter)
        
        # Identify missing variables
        with st.spinner("Analyzing for missing variables..."):
            missing_vars = integrator.identify_missing_variables(
                graph=graph,
                data=data,
                text=domain_knowledge_text
            )
        
        # Display results
        if missing_vars:
            st.success(f"Identified {len(missing_vars)} potential missing variables")
            
            for var in missing_vars:
                with st.expander(f"{var['name']} (Importance: {var['importance']}/10)"):
                    st.markdown(f"**Description:** {var['description']}")
                    st.markdown(f"**Influenced Variables:** {', '.join(var['influenced_variables'])}")
                    st.markdown(f"**Evidence:** {var['evidence']}")
        else:
            st.info("No missing variables identified.")
        
        return missing_vars
    
    return []