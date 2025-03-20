# app/components/hidden_variable_detector.py

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HiddenVariableDetector:
    """
    Detect potential hidden variables (latent confounders) in causal graphs.
    Uses statistical analysis, tetrad constraints, and LLM-based methods
    to identify unobserved variables that may be influencing the data.
    """
    
    def __init__(self, graph: nx.DiGraph, data: pd.DataFrame):
        """
        Initialize hidden variable detector
        
        Args:
            graph: NetworkX DiGraph representing the causal structure
            data: DataFrame containing the data
        """
        self.graph = graph
        self.data = data
        self.hidden_variables = []
        self.analysis_results = {}
    
    def _get_node_name(self, node: Any) -> str:
        """Get the readable name for a node"""
        if isinstance(node, int) and node < len(self.data.columns):
            return self.data.columns[node]
        elif "name" in self.graph.nodes[node]:
            return self.graph.nodes[node]["name"]
        else:
            return str(node)
    
    def detect_by_correlation(self, 
                            threshold: float = 0.7, 
                            min_uncorrelated_path_length: int = 2) -> List[Dict[str, Any]]:
        """
        Detect hidden variables using correlation patterns
        
        Args:
            threshold: Correlation threshold for detection
            min_uncorrelated_path_length: Minimum path length to check for uncorrelated variables
            
        Returns:
            List of detected hidden variables
        """
        hidden_vars = []
        
        # Get all node pairs with minimum path length
        all_pairs = []
        for node1 in self.graph.nodes():
            for node2 in self.graph.nodes():
                if node1 != node2:
                    try:
                        # Check if there's a path between nodes
                        path_length = nx.shortest_path_length(self.graph, node1, node2)
                        
                        # Check if the path length meets the minimum
                        if path_length >= min_uncorrelated_path_length:
                            all_pairs.append((node1, node2))
                    except nx.NetworkXNoPath:
                        # No path exists, include these pairs as they may have a hidden common cause
                        all_pairs.append((node1, node2))
        
        # Find pairs with high correlation but no direct edge
        for node1, node2 in all_pairs:
            # Skip if there's a direct edge
            if self.graph.has_edge(node1, node2) or self.graph.has_edge(node2, node1):
                continue
            
            # Get node names for data lookup
            name1 = self._get_node_name(node1)
            name2 = self._get_node_name(node2)
            
            # Skip if either name is not in data columns
            if name1 not in self.data.columns or name2 not in self.data.columns:
                continue
            
            # Calculate correlation if data is numeric
            if pd.api.types.is_numeric_dtype(self.data[name1]) and pd.api.types.is_numeric_dtype(self.data[name2]):
                corr = abs(self.data[name1].corr(self.data[name2]))
                
                # If correlation is high, suggest hidden variable
                if corr > threshold:
                    # Check if there's a hidden common parent already
                    has_hidden_parent = False
                    for parent in self.graph.nodes():
                        if (self.graph.has_edge(parent, node1) and self.graph.has_edge(parent, node2) and 
                            self.graph.nodes[parent].get("is_hidden", False)):
                            has_hidden_parent = True
                            break
                    
                    if not has_hidden_parent:
                        hidden_vars.append({
                            "name": f"Hidden_Correlation_{len(hidden_vars) + 1}",
                            "method": "correlation",
                            "affects": [name1, name2],
                            "node1": node1,
                            "node2": node2,
                            "correlation": corr,
                            "confidence": (corr - threshold) / (1 - threshold),  # Scale to 0-1
                            "explanation": (f"Variables {name1} and {name2} have high correlation ({corr:.3f}) "
                                          f"but no direct causal link, suggesting a hidden common cause.")
                        })
        
        # Store results
        self.analysis_results["correlation"] = {
            "threshold": threshold,
            "pairs_analyzed": len(all_pairs),
            "hidden_variables_found": len(hidden_vars)
        }
        
        # Extend the list of hidden variables
        self.hidden_variables.extend(hidden_vars)
        
        return hidden_vars
    
    def detect_by_tetrad_constraints(self) -> List[Dict[str, Any]]:
        """
        Detect hidden variables using tetrad constraints
        
        Returns:
            List of detected hidden variables
        """
        hidden_vars = []
        
        # Get all quartets of variables for tetrad testing
        quartets = []
        nodes = list(self.graph.nodes())
        
        if len(nodes) < 4:
            logger.warning("Not enough nodes for tetrad testing (need at least 4)")
            # Store results
            self.analysis_results["tetrad"] = {
                "status": "skipped",
                "reason": "Not enough nodes (need at least 4)"
            }
            return []
        
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                for k in range(j+1, len(nodes)):
                    for l in range(k+1, len(nodes)):
                        quartets.append((nodes[i], nodes[j], nodes[k], nodes[l]))
        
        # Test tetrad constraints for each quartet
        for node_a, node_b, node_c, node_d in quartets:
            # Get node names
            name_a = self._get_node_name(node_a)
            name_b = self._get_node_name(node_b)
            name_c = self._get_node_name(node_c)
            name_d = self._get_node_name(node_d)
            
            # Skip if any name is not in data columns or not numeric
            if (name_a not in self.data.columns or name_b not in self.data.columns or 
                name_c not in self.data.columns or name_d not in self.data.columns):
                continue
                
            if (not pd.api.types.is_numeric_dtype(self.data[name_a]) or 
                not pd.api.types.is_numeric_dtype(self.data[name_b]) or
                not pd.api.types.is_numeric_dtype(self.data[name_c]) or
                not pd.api.types.is_numeric_dtype(self.data[name_d])):
                continue
            
            # Calculate covariance matrix
            columns = [name_a, name_b, name_c, name_d]
            cov_matrix = self.data[columns].cov()
            
            # Calculate tetrad differences
            tetrad_abc_d = cov_matrix.loc[name_a, name_c] * cov_matrix.loc[name_b, name_d] - \
                          cov_matrix.loc[name_a, name_d] * cov_matrix.loc[name_b, name_c]
            
            tetrad_abd_c = cov_matrix.loc[name_a, name_b] * cov_matrix.loc[name_d, name_c] - \
                          cov_matrix.loc[name_a, name_c] * cov_matrix.loc[name_b, name_d]
            
            tetrad_acd_b = cov_matrix.loc[name_a, name_c] * cov_matrix.loc[name_d, name_b] - \
                          cov_matrix.loc[name_a, name_b] * cov_matrix.loc[name_c, name_d]
            
            # Check if tetrads vanish (approximately zero)
            # Scale by average covariance magnitude to ensure relative comparison
            avg_cov = np.mean(np.abs(cov_matrix.values))
            threshold = 0.1 * avg_cov  # 10% of average covariance
            
            tetrads_vanish = [
                abs(tetrad_abc_d) < threshold,
                abs(tetrad_abd_c) < threshold,
                abs(tetrad_acd_b) < threshold
            ]
            
            # If exactly one tetrad constraint is satisfied, suggest latent variable
            if sum(tetrads_vanish) == 1:
                # Determine which variables are connected by the latent
                if tetrads_vanish[0]:  # tetrad_abc_d vanishes
                    connected = [name_a, name_b, name_c, name_d]
                    pattern = "latent connects all four variables"
                elif tetrads_vanish[1]:  # tetrad_abd_c vanishes
                    connected = [name_a, name_b, name_c, name_d]
                    pattern = "latent connects all four variables"
                elif tetrads_vanish[2]:  # tetrad_acd_b vanishes
                    connected = [name_a, name_b, name_c, name_d]
                    pattern = "latent connects all four variables"
                
                # Check if this hidden variable is already suggested
                duplicate = False
                for hv in hidden_vars:
                    if set(hv["affects"]) == set(connected) and hv["method"] == "tetrad":
                        duplicate = True
                        break
                
                if not duplicate:
                    hidden_vars.append({
                        "name": f"Hidden_Tetrad_{len(hidden_vars) + 1}",
                        "method": "tetrad",
                        "affects": connected,
                        "pattern": pattern,
                        "confidence": 0.7,  # Fixed confidence for tetrad detections
                        "explanation": (f"Tetrad constraint satisfied by variables {', '.join(connected)}, "
                                      f"suggesting a hidden common cause with pattern: {pattern}.")
                    })
        
        # Store results
        self.analysis_results["tetrad"] = {
            "quartets_tested": len(quartets),
            "hidden_variables_found": len(hidden_vars)
        }
        
        # Extend the list of hidden variables
        self.hidden_variables.extend(hidden_vars)
        
        return hidden_vars
    
    def detect_by_collider_patterns(self, min_dependence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect hidden variables using collider patterns
        
        Args:
            min_dependence: Minimum dependence (correlation) for detection
            
        Returns:
            List of detected hidden variables
        """
        hidden_vars = []
        
        # Find all collider patterns (v-structures) in the graph
        colliders = []
        for node in self.graph.nodes():
            # Find nodes with at least two parents
            parents = list(self.graph.predecessors(node))
            if len(parents) >= 2:
                # Check each pair of parents
                for i in range(len(parents)):
                    for j in range(i+1, len(parents)):
                        parent1 = parents[i]
                        parent2 = parents[j]
                        
                        # Check if parents are not directly connected
                        if not self.graph.has_edge(parent1, parent2) and not self.graph.has_edge(parent2, parent1):
                            colliders.append((parent1, node, parent2))
        
        # Analyze each collider
        for parent1, collider, parent2 in colliders:
            # Get node names
            name1 = self._get_node_name(parent1)
            name_collider = self._get_node_name(collider)
            name2 = self._get_node_name(parent2)
            
            # Skip if any name is not in data columns
            if (name1 not in self.data.columns or 
                name_collider not in self.data.columns or 
                name2 not in self.data.columns):
                continue
            
            # Check if parents are correlated in the data
            if (pd.api.types.is_numeric_dtype(self.data[name1]) and 
                pd.api.types.is_numeric_dtype(self.data[name2])):
                corr = abs(self.data[name1].corr(self.data[name2]))
                
                # If correlation is high, suggest hidden variable
                if corr > min_dependence:
                    # Check if there's a common parent already
                    has_common_parent = False
                    for node in self.graph.nodes():
                        if (self.graph.has_edge(node, parent1) and self.graph.has_edge(node, parent2)):
                            has_common_parent = True
                            break
                    
                    if not has_common_parent:
                        hidden_vars.append({
                            "name": f"Hidden_Collider_{len(hidden_vars) + 1}",
                            "method": "collider",
                            "affects": [name1, name2],
                            "node1": parent1,
                            "node2": parent2,
                            "collider": collider,
                            "correlation": corr,
                            "confidence": (corr - min_dependence) / (1 - min_dependence),  # Scale to 0-1
                            "explanation": (f"Variables {name1} and {name2} are correlated ({corr:.3f}) "
                                          f"but form a collider at {name_collider}, suggesting a hidden common cause.")
                        })
        
        # Store results
        self.analysis_results["collider"] = {
            "colliders_analyzed": len(colliders),
            "hidden_variables_found": len(hidden_vars)
        }
        
        # Extend the list of hidden variables
        self.hidden_variables.extend(hidden_vars)
        
        return hidden_vars
    
    def detect_by_llm(self, llm_adapter) -> List[Dict[str, Any]]:
        """
        Detect hidden variables using LLM analysis
        
        Args:
            llm_adapter: LLM adapter for querying the language model
            
        Returns:
            List of detected hidden variables
        """
        if llm_adapter is None:
            logger.warning("LLM adapter not provided, skipping LLM-based detection")
            return []
        
        hidden_vars = []
        
        # Create a description of the graph
        graph_description = "Graph structure:\n"
        for u, v in self.graph.edges():
            u_name = self._get_node_name(u)
            v_name = self._get_node_name(v)
            graph_description += f"- {u_name} → {v_name}\n"
        
        # Create a description of the data
        data_description = "Data characteristics:\n"
        data_description += f"- {self.data.shape[0]} samples, {self.data.shape[1]} variables\n"
        
        # Add correlation information
        try:
            # Only use numeric columns for correlation
            numeric_data = self.data.select_dtypes(include=[np.number])
            if not numeric_data.empty and numeric_data.shape[1] >= 2:
                corr_matrix = numeric_data.corr()
                
                # Find strong correlations
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr = abs(corr_matrix.loc[col1, col2])
                        if corr > 0.5:
                            strong_corrs.append((col1, col2, corr))
                
                if strong_corrs:
                    data_description += "\nStrong correlations:\n"
                    for col1, col2, corr in strong_corrs[:5]:  # Limit to top 5 correlations
                        data_description += f"- {col1} and {col2}: {corr:.3f}\n"
                    if len(strong_corrs) > 5:
                        data_description += f"- ... and {len(strong_corrs) - 5} more\n"
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
        
        # Create prompt for the LLM
        prompt = f"""
        I need your help identifying potential hidden variables (latent confounders) in a causal system.
        Here is the information:
        
        {graph_description}
        
        {data_description}
        
        Given the graph structure and data characteristics, please suggest potential hidden variables 
        that might be influencing the observed variables but are not captured in the graph.
        
        For each hidden variable:
        1. Provide a name for the hidden variable
        2. List which observed variables it might affect
        3. Provide a brief explanation of why you think this hidden variable exists
        4. Rate your confidence on a scale of 0.0 to 1.0
        
        Format your response as follows:
        
        HIDDEN_VARIABLE
        Name: [name]
        Affects: [list of affected variables]
        Explanation: [your reasoning]
        Confidence: [0.0-1.0]
        
        HIDDEN_VARIABLE
        ... (for each hidden variable)
        """
        
        try:
            # Query the LLM
            response = llm_adapter.complete(
                prompt=prompt,
                system_prompt="You are an expert in causal inference and hidden variable detection. Your task is to analyze causal graphs and identify potential hidden confounders.",
                temperature=0.4
            )
            
            # Extract completion
            completion = response.get("completion", "")
            
            # Parse response to extract hidden variables
            sections = completion.split("HIDDEN_VARIABLE")
            for section in sections[1:]:  # Skip the first empty section
                lines = section.strip().split("\n")
                
                name = ""
                affects = []
                explanation = ""
                confidence = 0.5  # Default confidence
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Name:"):
                        name = line[5:].strip()
                    elif line.startswith("Affects:"):
                        affects_str = line[8:].strip()
                        affects = [var.strip() for var in affects_str.split(",")]
                    elif line.startswith("Explanation:"):
                        explanation = line[12:].strip()
                    elif line.startswith("Confidence:"):
                        try:
                            conf_str = line[11:].strip()
                            confidence = float(conf_str)
                            # Ensure confidence is in range 0-1
                            confidence = max(0.0, min(1.0, confidence))
                        except ValueError:
                            confidence = 0.5
                
                if name and affects:
                    hidden_vars.append({
                        "name": name,
                        "method": "llm",
                        "affects": affects,
                        "confidence": confidence,
                        "explanation": explanation
                    })
        
        except Exception as e:
            logger.error(f"Error in LLM detection: {e}")
        
        # Store results
        self.analysis_results["llm"] = {
            "hidden_variables_found": len(hidden_vars)
        }
        
        # Extend the list of hidden variables
        self.hidden_variables.extend(hidden_vars)
        
        return hidden_vars
    
    def run_all_detection_methods(self, 
                                llm_adapter=None, 
                                correlation_threshold: float = 0.7,
                                min_dependence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Run all detection methods and consolidate results
        
        Args:
            llm_adapter: LLM adapter for LLM-based detection (optional)
            correlation_threshold: Threshold for correlation-based detection
            min_dependence: Minimum dependence for collider pattern detection
            
        Returns:
            List of all detected hidden variables
        """
        # Reset hidden variables
        self.hidden_variables = []
        
        # Run all detection methods
        self.detect_by_correlation(threshold=correlation_threshold)
        self.detect_by_tetrad_constraints()
        self.detect_by_collider_patterns(min_dependence=min_dependence)
        
        # Run LLM detection if adapter is provided
        if llm_adapter is not None:
            self.detect_by_llm(llm_adapter)
        
        # Deduplicate hidden variables by affected variables
        deduplicated = []
        for hv in self.hidden_variables:
            # Check if there's a similar hidden variable already
            duplicate = False
            for existing in deduplicated:
                if set(hv["affects"]) == set(existing["affects"]):
                    # Take the one with higher confidence
                    if hv["confidence"] > existing["confidence"]:
                        # Replace existing with current
                        deduplicated.remove(existing)
                        deduplicated.append(hv)
                    duplicate = True
                    break
            
            if not duplicate:
                deduplicated.append(hv)
        
        # Replace with deduplicated list
        self.hidden_variables = deduplicated
        
        return self.hidden_variables
    
    def add_hidden_variables_to_graph(self) -> nx.DiGraph:
        """
        Add detected hidden variables to the graph
        
        Returns:
            Updated graph with hidden variables
        """
        # Create a copy of the original graph
        updated_graph = self.graph.copy()
        
        # Add each hidden variable to the graph
        for i, hv in enumerate(self.hidden_variables):
            # Create a new node ID
            hidden_id = max(updated_graph.nodes()) + 1 if updated_graph.nodes() else 0
            
            # Add the hidden node
            updated_graph.add_node(
                hidden_id,
                name=hv["name"],
                is_hidden=True,
                method=hv["method"],
                confidence=hv["confidence"],
                explanation=hv.get("explanation", "")
            )
            
            # Add edges to affected variables
            for affected in hv["affects"]:
                # Find the node ID for the affected variable
                affected_id = None
                for node in updated_graph.nodes():
                    if self._get_node_name(node) == affected:
                        affected_id = node
                        break
                
                if affected_id is not None:
                    # Add edge from hidden to affected
                    updated_graph.add_edge(
                        hidden_id,
                        affected_id,
                        is_hidden_cause=True,
                        confidence=hv["confidence"]
                    )
        
        return updated_graph
    
    def visualize_hidden_variables(self) -> go.Figure:
        """
        Create a visualization of detected hidden variables
        
        Returns:
            Plotly figure with hidden variables visualization
        """
        if not self.hidden_variables:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No hidden variables detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Create node positions
        pos = {}
        
        # Place hidden variables at the top
        hidden_count = len(self.hidden_variables)
        for i, hv in enumerate(self.hidden_variables):
            x_pos = (i + 1) / (hidden_count + 1)
            pos[hv["name"]] = (x_pos, 0.1)
        
        # Get all affected variables
        all_affected = set()
        for hv in self.hidden_variables:
            all_affected.update(hv["affects"])
        
        # Place affected variables at the bottom
        affected_count = len(all_affected)
        affected_pos = {}
        for i, var in enumerate(sorted(all_affected)):
            x_pos = (i + 1) / (affected_count + 1)
            pos[var] = (x_pos, 0.9)
            affected_pos[var] = (x_pos, 0.9)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for hv in self.hidden_variables:
            hv_x, hv_y = pos[hv["name"]]
            
            for affected in hv["affects"]:
                if affected in affected_pos:
                    aff_x, aff_y = affected_pos[affected]
                    
                    # Add edge
                    fig.add_trace(go.Scatter(
                        x=[hv_x, aff_x],
                        y=[hv_y, aff_y],
                        mode='lines',
                        line=dict(
                            width=2, 
                            color=f"rgba(100, 100, 255, {hv['confidence']})"
                        ),
                        hoverinfo='text',
                        text=f"Confidence: {hv['confidence']:.2f}",
                        showlegend=False
                    ))
                    
                    # Add arrow
                    fig.add_trace(go.Scatter(
                        x=[aff_x],
                        y=[aff_y - 0.03],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color=f"rgba(100, 100, 255, {hv['confidence']})",
                        ),
                        hoverinfo='none',
                        showlegend=False
                    ))
        
        # Add hidden variable nodes
        for hv in self.hidden_variables:
            x, y = pos[hv["name"]]
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='rgba(255, 100, 100, 0.8)',
                    line=dict(width=2, color='darkred')
                ),
                text=[hv["name"]],
                textposition="top center",
                hoverinfo='text',
                hovertext=f"{hv['name']}<br>Method: {hv['method']}<br>Confidence: {hv['confidence']:.2f}<br>{hv.get('explanation', '')}",
                name=hv["method"].capitalize()
            ))
        
        # Add affected variable nodes
        for var, (x, y) in affected_pos.items():
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='rgba(100, 100, 255, 0.8)',
                    line=dict(width=1, color='darkblue')
                ),
                text=[var],
                textposition="bottom center",
                hoverinfo='text',
                hovertext=var,
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title="Detected Hidden Variables",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.1, 1.1]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.1, 1.1]
            ),
            legend=dict(
                title="Detection Method",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=600,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the hidden variable detection analysis
        
        Returns:
            Dictionary with summary information
        """
        methods_count = {}
        for hv in self.hidden_variables:
            method = hv["method"]
            methods_count[method] = methods_count.get(method, 0) + 1
        
        all_affected = set()
        for hv in self.hidden_variables:
            all_affected.update(hv["affects"])
        
        # Calculate average confidence
        avg_confidence = 0
        if self.hidden_variables:
            avg_confidence = sum(hv["confidence"] for hv in self.hidden_variables) / len(self.hidden_variables)
        
        summary = {
            "total_hidden_variables": len(self.hidden_variables),
            "by_method": methods_count,
            "affected_variables": len(all_affected),
            "average_confidence": avg_confidence,
            "analysis_results": self.analysis_results
        }
        
        return summary


def render_hidden_variable_detector(graph: nx.DiGraph, data: pd.DataFrame):
    """
    Render the hidden variable detector UI in Streamlit
    
    Args:
        graph: The causal graph
        data: The dataset
    """
    st.subheader("Hidden Variable Detector")
    
    # Create detector
    detector = HiddenVariableDetector(graph, data)
    
    # Set up detection options
    st.markdown("#### Detection Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_correlation = st.checkbox("Use correlation analysis", value=True)
        correlation_threshold = st.slider(
            "Correlation threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.7, 
            step=0.05,
            disabled=not use_correlation
        )
    
    with col2:
        use_tetrad = st.checkbox("Use tetrad constraints", value=True)
        use_collider = st.checkbox("Use collider patterns", value=True)
        min_dependence = st.slider(
            "Minimum dependence", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.3, 
            step=0.05,
            disabled=not use_collider
        )
    
    # LLM-based detection
    use_llm = st.checkbox("Use LLM-based detection", value=True)
    
    # Detect button
    if st.button("Detect Hidden Variables"):
        with st.spinner("Detecting hidden variables..."):
            # Get LLM adapter if enabled
            llm_adapter = None
            if use_llm and hasattr(st.session_state, "llm_adapter") and st.session_state.llm_adapter is not None:
                llm_adapter = st.session_state.llm_adapter
            elif use_llm:
                st.warning("LLM adapter not available. LLM-based detection will be skipped.")
            
            # Run selected detection methods
            if use_correlation:
                detector.detect_by_correlation(correlation_threshold)
            
            if use_tetrad:
                detector.detect_by_tetrad_constraints()
            
            if use_collider:
                detector.detect_by_collider_patterns(min_dependence)
            
            if use_llm and llm_adapter:
                detector.detect_by_llm(llm_adapter)
            
            # Show results
            hidden_vars = detector.hidden_variables
            
            if hidden_vars:
                st.success(f"Detected {len(hidden_vars)} potential hidden variables!")
                
                # Show visualization
                fig = detector.visualize_hidden_variables()
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed results
                st.subheader("Detailed Results")
                
                for i, hv in enumerate(hidden_vars):
                    with st.expander(f"{hv['name']} (Confidence: {hv['confidence']:.2f})"):
                        st.markdown(f"**Method:** {hv['method'].capitalize()}")
                        st.markdown(f"**Affects:** {', '.join(hv['affects'])}")
                        st.markdown(f"**Explanation:** {hv.get('explanation', 'No explanation provided')}")
                
                # Option to add to graph
                if st.button("Add Hidden Variables to Graph"):
                    updated_graph = detector.add_hidden_variables_to_graph()
                    st.session_state.current_graph = updated_graph
                    st.success("Hidden variables added to graph!")
            else:
                st.info("No hidden variables detected with the current settings.")
                
                # Show analysis summary
                summary = detector.get_summary()
                st.markdown("#### Analysis Summary")
                
                for method, results in summary["analysis_results"].items():
                    st.markdown(f"**{method.capitalize()} analysis:**")
                    for key, value in results.items():
                        st.markdown(f"- {key}: {value}")


def detect_hidden_variables(graph: nx.DiGraph, 
                          data: pd.DataFrame,
                          methods: List[str] = ["correlation", "tetrad", "collider"],
                          llm_adapter=None,
                          correlation_threshold: float = 0.7) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
    """
    Detect hidden variables and return results plus updated graph
    
    Args:
        graph: The causal graph
        data: The dataset
        methods: List of detection methods to use
        llm_adapter: LLM adapter for LLM-based detection
        correlation_threshold: Threshold for correlation-based detection
        
    Returns:
        Tuple of (hidden variables list, updated graph with hidden variables)
    """
    detector = HiddenVariableDetector(graph, data)
    
    # Run selected methods
    if "correlation" in methods:
        detector.detect_by_correlation(threshold=correlation_threshold)
    
    if "tetrad" in methods:
        detector.detect_by_tetrad_constraints()
    
    if "collider" in methods:
        detector.detect_by_collider_patterns()
    
    if "llm" in methods and llm_adapter:
        detector.detect_by_llm(llm_adapter)
    
    # Get updated graph
    updated_graph = detector.add_hidden_variables_to_graph()
    
    return detector.hidden_variables, updated_graph