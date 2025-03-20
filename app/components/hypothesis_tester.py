# app/components/hypothesis_tester.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HypothesisTester:
    """
    Component for testing causal hypotheses against data and causal graphs.
    Provides statistical tests to validate causal relationships and
    evaluate the strength of evidence for causal assertions.
    """
    
    def __init__(self, graph: nx.DiGraph, data: pd.DataFrame):
        """
        Initialize the hypothesis tester
        
        Args:
            graph: NetworkX DiGraph representing the causal structure
            data: DataFrame containing the data for testing
        """
        self.graph = graph
        self.data = data
    
    def _get_node_name(self, node_id: Any) -> str:
        """
        Get the readable name for a node
        
        Args:
            node_id: The node ID in the graph
            
        Returns:
            The readable name for the node
        """
        # Check for name in node attributes
        if "name" in self.graph.nodes[node_id]:
            return self.graph.nodes[node_id]["name"]
        
        # If node_id is an integer index into the dataframe columns
        if isinstance(node_id, int) and node_id < len(self.data.columns):
            return self.data.columns[node_id]
        
        # Default to string representation
        return str(node_id)
    
    def _get_node_id(self, node_name: str) -> Optional[Any]:
        """
        Find the node ID for a given node name
        
        Args:
            node_name: The name of the node to find
            
        Returns:
            The node ID or None if not found
        """
        # Search by name attribute
        for node_id in self.graph.nodes():
            if "name" in self.graph.nodes[node_id] and self.graph.nodes[node_id]["name"] == node_name:
                return node_id
        
        # Search by dataframe column name
        for node_id in self.graph.nodes():
            if isinstance(node_id, int) and node_id < len(self.data.columns) and self.data.columns[node_id] == node_name:
                return node_id
        
        # Direct match to node_id
        if node_name in self.graph.nodes():
            return node_name
        
        return None
    
    def test_causal_relationship(self, 
                               cause: Union[str, int], 
                               effect: Union[str, int],
                               test_type: str = 'regression',
                               control_for: List[Union[str, int]] = None) -> Dict[str, Any]:
        """
        Test if a causal relationship exists between cause and effect variables
        
        Args:
            cause: The cause variable (name or ID)
            effect: The effect variable (name or ID)
            test_type: Type of test to perform ('regression', 'correlation', 'partial_correlation')
            control_for: Optional list of variables to control for
            
        Returns:
            Dictionary with test results
        """
        # Convert strings to node IDs if needed
        cause_id = self._get_node_id(cause) if isinstance(cause, str) else cause
        effect_id = self._get_node_id(effect) if isinstance(effect, str) else effect
        
        if cause_id is None or effect_id is None:
            return {
                "status": "error",
                "message": f"Could not find {'cause' if cause_id is None else 'effect'} variable in graph",
                "hypothesis_confirmed": False
            }
        
        # Get variable names for data access
        cause_name = self._get_node_name(cause_id)
        effect_name = self._get_node_name(effect_id)
        
        # Check if the data contains the variables
        if cause_name not in self.data.columns or effect_name not in self.data.columns:
            return {
                "status": "error",
                "message": f"Variable {'cause_name' if cause_name not in self.data.columns else 'effect_name'} not found in data",
                "hypothesis_confirmed": False
            }
        
        # Convert control variables to node IDs if needed
        control_ids = []
        control_names = []
        if control_for:
            for ctrl in control_for:
                ctrl_id = self._get_node_id(ctrl) if isinstance(ctrl, str) else ctrl
                if ctrl_id is not None:
                    control_ids.append(ctrl_id)
                    control_names.append(self._get_node_name(ctrl_id))
        
        # Check if there's a direct edge in the graph
        edge_exists = self.graph.has_edge(cause_id, effect_id)
        
        # Perform the requested test
        if test_type == 'regression':
            return self._regression_test(cause_name, effect_name, control_names, edge_exists)
        elif test_type == 'correlation':
            return self._correlation_test(cause_name, effect_name, edge_exists)
        elif test_type == 'partial_correlation':
            return self._partial_correlation_test(cause_name, effect_name, control_names, edge_exists)
        else:
            return {
                "status": "error",
                "message": f"Unknown test type: {test_type}",
                "hypothesis_confirmed": False
            }
    
    def _regression_test(self, 
                       cause_name: str, 
                       effect_name: str, 
                       control_names: List[str],
                       edge_exists: bool) -> Dict[str, Any]:
        """
        Perform a regression-based test of causality
        
        Args:
            cause_name: Name of cause variable
            effect_name: Name of effect variable
            control_names: Names of control variables
            edge_exists: Whether a direct edge exists in the graph
            
        Returns:
            Dictionary with test results
        """
        try:
            import statsmodels.api as sm
            
            # Prepare data for regression
            X_names = [cause_name] + control_names
            X = self.data[X_names]
            X = sm.add_constant(X)  # Add intercept
            y = self.data[effect_name]
            
            # Fit the regression model
            model = sm.OLS(y, X).fit()
            
            # Get statistics for the cause variable
            cause_index = list(model.params.index).index(cause_name)
            coefficient = model.params[cause_name]
            p_value = model.pvalues[cause_name]
            t_stat = model.tvalues[cause_name]
            conf_int = model.conf_int().loc[cause_name].tolist()
            
            # Determine if hypothesis is confirmed (significant effect in expected direction)
            hypothesis_confirmed = p_value < 0.05 and (
                (edge_exists and coefficient != 0) or 
                (not edge_exists and coefficient == 0)
            )
            
            # Create result dictionary
            result = {
                "status": "completed",
                "test_type": "regression",
                "cause": cause_name,
                "effect": effect_name,
                "controls": control_names,
                "edge_exists_in_graph": edge_exists,
                "coefficient": coefficient,
                "std_error": model.bse[cause_name],
                "t_statistic": t_stat,
                "p_value": p_value,
                "confidence_interval": conf_int,
                "hypothesis_confirmed": hypothesis_confirmed,
                "model_summary": model.summary().as_text(),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in regression test: {str(e)}")
            return {
                "status": "error",
                "message": f"Error in regression test: {str(e)}",
                "hypothesis_confirmed": False
            }
    
    def _correlation_test(self, 
                        cause_name: str, 
                        effect_name: str, 
                        edge_exists: bool) -> Dict[str, Any]:
        """
        Perform a correlation-based test of causality
        
        Args:
            cause_name: Name of cause variable
            effect_name: Name of effect variable
            edge_exists: Whether a direct edge exists in the graph
            
        Returns:
            Dictionary with test results
        """
        try:
            # Calculate Pearson correlation
            corr, p_value = stats.pearsonr(self.data[cause_name], self.data[effect_name])
            
            # Determine if hypothesis is confirmed (significant correlation in expected direction)
            hypothesis_confirmed = p_value < 0.05 and (
                (edge_exists and corr != 0) or 
                (not edge_exists and corr == 0)
            )
            
            # Create result dictionary
            result = {
                "status": "completed",
                "test_type": "correlation",
                "cause": cause_name,
                "effect": effect_name,
                "edge_exists_in_graph": edge_exists,
                "correlation": corr,
                "p_value": p_value,
                "hypothesis_confirmed": hypothesis_confirmed
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in correlation test: {str(e)}")
            return {
                "status": "error",
                "message": f"Error in correlation test: {str(e)}",
                "hypothesis_confirmed": False
            }
    
    def _partial_correlation_test(self, 
                               cause_name: str, 
                               effect_name: str,
                               control_names: List[str],
                               edge_exists: bool) -> Dict[str, Any]:
        """
        Perform a partial correlation test of causality
        
        Args:
            cause_name: Name of cause variable
            effect_name: Name of effect variable
            control_names: Names of control variables
            edge_exists: Whether a direct edge exists in the graph
            
        Returns:
            Dictionary with test results
        """
        try:
            # If no control variables, fall back to regular correlation
            if not control_names:
                return self._correlation_test(cause_name, effect_name, edge_exists)
            
            # Calculate partial correlation
            from scipy.stats import pearsonr
            from pingouin import partial_corr
            
            # Create DataFrame with relevant variables
            data_subset = self.data[[cause_name, effect_name] + control_names]
            
            # Calculate partial correlation
            pcorr_result = partial_corr(data=data_subset, x=cause_name, y=effect_name, covar=control_names)
            
            # Get results
            pcorr = pcorr_result['r'].values[0]
            p_value = pcorr_result['p-val'].values[0]
            
            # Determine if hypothesis is confirmed (significant partial correlation in expected direction)
            hypothesis_confirmed = p_value < 0.05 and (
                (edge_exists and pcorr != 0) or 
                (not edge_exists and pcorr == 0)
            )
            
            # Create result dictionary
            result = {
                "status": "completed",
                "test_type": "partial_correlation",
                "cause": cause_name,
                "effect": effect_name,
                "controls": control_names,
                "edge_exists_in_graph": edge_exists,
                "partial_correlation": pcorr,
                "p_value": p_value,
                "hypothesis_confirmed": hypothesis_confirmed
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in partial correlation test: {str(e)}")
            return {
                "status": "error",
                "message": f"Error in partial correlation test: {str(e)}",
                "hypothesis_confirmed": False
            }
    
    def test_backdoor_adjustment(self,
                              treatment: Union[str, int],
                              outcome: Union[str, int],
                              adjustment_sets: List[List[Union[str, int]]] = None) -> Dict[str, Any]:
        """
        Test effectiveness of backdoor adjustment sets
        
        Args:
            treatment: The treatment variable (name or ID)
            outcome: The outcome variable (name or ID)
            adjustment_sets: Optional list of adjustment sets to test
            
        Returns:
            Dictionary with test results
        """
        # Convert strings to node IDs if needed
        treatment_id = self._get_node_id(treatment) if isinstance(treatment, str) else treatment
        outcome_id = self._get_node_id(outcome) if isinstance(outcome, str) else outcome
        
        if treatment_id is None or outcome_id is None:
            return {
                "status": "error",
                "message": f"Could not find {'treatment' if treatment_id is None else 'outcome'} variable in graph",
                "hypothesis_confirmed": False
            }
        
        # Get variable names for data access
        treatment_name = self._get_node_name(treatment_id)
        outcome_name = self._get_node_name(outcome_id)
        
        # Try to identify valid adjustment sets if none provided
        if adjustment_sets is None:
            adjustment_sets = self._find_backdoor_adjustment_sets(treatment_id, outcome_id)
        
        # Convert adjustment sets to variable names
        named_adjustment_sets = []
        for adj_set in adjustment_sets:
            named_set = []
            for var in adj_set:
                var_id = self._get_node_id(var) if isinstance(var, str) else var
                if var_id is not None:
                    named_set.append(self._get_node_name(var_id))
            named_adjustment_sets.append(named_set)
        
        # Test each adjustment set using regression
        adjustment_results = []
        for adj_set in named_adjustment_sets:
            result = self._regression_test(treatment_name, outcome_name, adj_set, 
                                         self.graph.has_edge(treatment_id, outcome_id))
            
            if result["status"] == "completed":
                adjustment_results.append({
                    "adjustment_set": adj_set,
                    "coefficient": result["coefficient"],
                    "p_value": result["p_value"],
                    "t_statistic": result["t_statistic"],
                    "confidence_interval": result["confidence_interval"],
                    "r_squared": result["r_squared"]
                })
        
        # Compare against unadjusted estimate
        unadjusted = self._regression_test(treatment_name, outcome_name, [], 
                                         self.graph.has_edge(treatment_id, outcome_id))
        
        # Create final result
        return {
            "status": "completed",
            "treatment": treatment_name,
            "outcome": outcome_name,
            "adjustment_sets": named_adjustment_sets,
            "adjustment_results": adjustment_results,
            "unadjusted": {
                "coefficient": unadjusted["coefficient"],
                "p_value": unadjusted["p_value"],
                "confidence_interval": unadjusted["confidence_interval"]
            },
            "hypothesis_confirmed": any(r["p_value"] < 0.05 for r in adjustment_results)
        }
    
    def _find_backdoor_adjustment_sets(self, treatment_id: Any, outcome_id: Any) -> List[List[Any]]:
        """
        Find valid backdoor adjustment sets for deconfounding
        
        Args:
            treatment_id: Treatment variable ID
            outcome_id: Outcome variable ID
            
        Returns:
            List of adjustment sets (each one is a list of variable IDs)
        """
        # Simple implementation: find parents of treatment
        parents = list(self.graph.predecessors(treatment_id))
        
        # Return basic adjustment sets
        if not parents:
            return [[]]  # No adjustment needed
        else:
            return [parents]  # Adjust for parents of treatment
    
    def test_multiple_hypotheses(self,
                              hypotheses: List[Dict[str, Any]],
                              correction_method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Test multiple causal hypotheses with correction for multiple testing
        
        Args:
            hypotheses: List of hypothesis dictionaries, each with 'cause' and 'effect'
            correction_method: Method for multiple testing correction ('bonferroni', 'fdr')
            
        Returns:
            Dictionary with results for all hypotheses
        """
        # Test each hypothesis individually
        results = []
        p_values = []
        
        for h in hypotheses:
            cause = h.get("cause")
            effect = h.get("effect")
            test_type = h.get("test_type", "regression")
            control_for = h.get("control_for", [])
            
            if not cause or not effect:
                continue
                
            result = self.test_causal_relationship(
                cause=cause,
                effect=effect,
                test_type=test_type,
                control_for=control_for
            )
            
            if result["status"] == "completed":
                results.append(result)
                p_values.append(result["p_value"])
        
        # Apply multiple testing correction
        adjusted_p_values = []
        
        if correction_method == 'bonferroni':
            # Bonferroni correction
            n_tests = len(p_values)
            adjusted_p_values = [min(p * n_tests, 1.0) for p in p_values]
            
        elif correction_method == 'fdr':
            # Benjamini-Hochberg procedure (FDR)
            from statsmodels.stats.multitest import multipletests
            
            if p_values:
                _, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')
            else:
                adjusted_p_values = []
                
        else:
            # No correction
            adjusted_p_values = p_values
        
        # Update results with adjusted p-values and significance
        for i, result in enumerate(results):
            if i < len(adjusted_p_values):
                result["adjusted_p_value"] = adjusted_p_values[i]
                result["significant_after_correction"] = adjusted_p_values[i] < 0.05
        
        # Create final result
        return {
            "status": "completed",
            "n_hypotheses": len(hypotheses),
            "n_tested": len(results),
            "correction_method": correction_method,
            "results": results,
            "overall_confirmed": any(r.get("significant_after_correction", False) for r in results)
        }
    
    def visualize_test_results(self, test_result: Dict[str, Any]) -> go.Figure:
        """
        Visualize the results of a hypothesis test
        
        Args:
            test_result: Result from a hypothesis test
            
        Returns:
            Plotly figure with visualization
        """
        if test_result.get("status") != "completed":
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=test_result.get("message", "Error in hypothesis test"),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig
        
        test_type = test_result.get("test_type", "")
        
        if test_type == "regression":
            return self._visualize_regression_results(test_result)
        elif test_type in ["correlation", "partial_correlation"]:
            return self._visualize_correlation_results(test_result)
        elif "adjustment_results" in test_result:
            return self._visualize_adjustment_comparison(test_result)
        elif "results" in test_result and "correction_method" in test_result:
            return self._visualize_multiple_tests(test_result)
        else:
            # Generic fallback visualization
            fig = go.Figure()
            fig.add_annotation(
                text="No specific visualization available for this test type",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
    
    def _visualize_regression_results(self, test_result: Dict[str, Any]) -> go.Figure:
        """Visualize regression test results"""
        # Extract data
        cause = test_result["cause"]
        effect = test_result["effect"]
        coefficient = test_result["coefficient"]
        p_value = test_result["p_value"]
        conf_int = test_result["confidence_interval"]
        
        # Create figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Coefficient with 95% CI", "Scatterplot with Regression Line"]
        )
        
        # Add bar plot for coefficient
        fig.add_trace(
            go.Bar(
                x=[cause],
                y=[coefficient],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[conf_int[1] - coefficient],
                    arrayminus=[coefficient - conf_int[0]]
                ),
                name="Coefficient",
                marker_color="blue" if p_value < 0.05 else "gray"
            ),
            row=1, col=1
        )
        
        # Add horizontal line at zero
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=0.5,
            y0=0,
            y1=0,
            line=dict(color="red", dash="dash"),
            row=1, col=1
        )
        
        # Add scatter plot with regression line
        x = self.data[cause]
        y = self.data[effect]
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=8, opacity=0.6),
                name="Data Points"
            ),
            row=1, col=2
        )
        
        # Add regression line
        x_range = np.linspace(x.min(), x.max(), 100)
        if "controls" in test_result and test_result["controls"]:
            # For partial regression, use the coefficient to draw line
            # This is simplified and not entirely accurate for partial regression
            y_mean = y.mean()
            y_range = y_mean + coefficient * (x_range - x.mean())
        else:
            # For simple regression, calculate the line directly
            slope, intercept = np.polyfit(x, y, 1)
            y_range = slope * x_range + intercept
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                line=dict(color="red"),
                name="Regression Line"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Regression Test: {cause} → {effect}",
            showlegend=True,
            height=400,
            width=800
        )
        
        # Add p-value annotation
        fig.add_annotation(
            text=f"p-value: {p_value:.4f}" + (" (significant)" if p_value < 0.05 else ""),
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor=("rgba(0,255,0,0.1)" if p_value < 0.05 else "rgba(255,0,0,0.1)")
        )
        
        return fig
    
    def _visualize_correlation_results(self, test_result: Dict[str, Any]) -> go.Figure:
        """Visualize correlation test results"""
        # Extract data
        cause = test_result["cause"]
        effect = test_result["effect"]
        
        if "correlation" in test_result:
            corr = test_result["correlation"]
            corr_type = "Correlation"
        else:
            corr = test_result["partial_correlation"]
            corr_type = "Partial Correlation"
            
        p_value = test_result["p_value"]
        
        # Create figure with two subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"{corr_type} Value", "Scatterplot"]
        )
        
        # Add bar for correlation
        fig.add_trace(
            go.Bar(
                x=[f"{cause} ↔ {effect}"],
                y=[corr],
                name=corr_type,
                marker_color="blue" if p_value < 0.05 else "gray"
            ),
            row=1, col=1
        )
        
        # Add reference lines for correlation (-1, 0, 1)
        for val, label in [(-1, "Perfect Negative"), (0, "No Correlation"), (1, "Perfect Positive")]:
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=0.5,
                y0=val,
                y1=val,
                line=dict(color="red" if val == 0 else "gray", dash="dash"),
                row=1, col=1
            )
        
        # Add scatter plot
        x = self.data[cause]
        y = self.data[effect]
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=8, opacity=0.6),
                name="Data Points"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"{corr_type} Test: {cause} ↔ {effect}",
            showlegend=True,
            height=400,
            width=800,
            yaxis=dict(range=[-1.1, 1.1])  # For correlation axis
        )
        
        # Add p-value annotation
        fig.add_annotation(
            text=f"p-value: {p_value:.4f}" + (" (significant)" if p_value < 0.05 else ""),
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor=("rgba(0,255,0,0.1)" if p_value < 0.05 else "rgba(255,0,0,0.1)")
        )
        
        # Add correlation value annotation
        fig.add_annotation(
            text=f"{corr_type}: {corr:.4f}",
            xref="paper", yref="paper",
            x=0.75, y=0.98,
            showarrow=False
        )
        
        return fig
    
    def _visualize_adjustment_comparison(self, test_result: Dict[str, Any]) -> go.Figure:
        """Visualize comparison of adjustment sets"""
        # Extract data
        treatment = test_result["treatment"]
        outcome = test_result["outcome"]
        adj_results = test_result["adjustment_results"]
        unadjusted = test_result["unadjusted"]
        
        # Create lists for plotting
        labels = ["Unadjusted"]
        coefficients = [unadjusted["coefficient"]]
        p_values = [unadjusted["p_value"]]
        lower_cis = [unadjusted["confidence_interval"][0]]
        upper_cis = [unadjusted["confidence_interval"][1]]
        colors = ["gray" if unadjusted["p_value"] >= 0.05 else "blue"]
        
        # Add adjustment sets
        for i, adj in enumerate(adj_results):
            adj_set = adj["adjustment_set"]
            label = f"Adjusted for {', '.join(adj_set)}" if adj_set else "No adjustment"
            
            labels.append(label)
            coefficients.append(adj["coefficient"])
            p_values.append(adj["p_value"])
            lower_cis.append(adj["confidence_interval"][0])
            upper_cis.append(adj["confidence_interval"][1])
            colors.append("gray" if adj["p_value"] >= 0.05 else "blue")
        
        # Create figure
        fig = go.Figure()
        
        # Add bars with error bars
        fig.add_trace(
            go.Bar(
                x=labels,
                y=coefficients,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[u - c for u, c in zip(upper_cis, coefficients)],
                    arrayminus=[c - l for l, c in zip(lower_cis, coefficients)]
                ),
                marker_color=colors,
                name="Coefficients"
            )
        )
        
        # Add horizontal line at zero
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(labels) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="red", dash="dash")
        )
        
        # Update layout
        fig.update_layout(
            title=f"Causal Effect Estimation: {treatment} → {outcome}",
            xaxis_title="Adjustment Set",
            yaxis_title="Coefficient (Causal Effect)",
            showlegend=False
        )
        
        # Add p-value annotations
        for i, (label, p_val) in enumerate(zip(labels, p_values)):
            fig.add_annotation(
                text=f"p={p_val:.3f}",
                x=i,
                y=coefficients[i] + (upper_cis[i] - coefficients[i]) + 0.02,
                showarrow=False,
                font=dict(size=10, color="black")
            )
        
        return fig
    
    def _visualize_multiple_tests(self, test_result: Dict[str, Any]) -> go.Figure:
        """Visualize multiple hypothesis tests with corrections"""
        # Extract data
        results = test_result["results"]
        correction = test_result["correction_method"]
        
        # Create lists for plotting
        hypotheses = []
        p_values = []
        adj_p_values = []
        colors = []
        
        for result in results:
            cause = result.get("cause", "")
            effect = result.get("effect", "")
            
            hypotheses.append(f"{cause} → {effect}")
            p_values.append(result.get("p_value", 1.0))
            adj_p_values.append(result.get("adjusted_p_value", 1.0))
            
            # Determine color based on significance
            if result.get("significant_after_correction", False):
                colors.append("green")
            elif result.get("p_value", 1.0) < 0.05:
                colors.append("orange")  # Significant before but not after correction
            else:
                colors.append("red")
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for raw p-values
        fig.add_trace(
            go.Bar(
                x=hypotheses,
                y=p_values,
                name="Raw p-values",
                marker_color="blue",
                opacity=0.6
            )
        )
        
        # Add bars for adjusted p-values
        fig.add_trace(
            go.Bar(
                x=hypotheses,
                y=adj_p_values,
                name=f"Adjusted p-values ({correction})",
                marker_color="red",
                opacity=0.6
            )
        )
        
        # Add horizontal line at 0.05
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(hypotheses) - 0.5,
            y0=0.05,
            y1=0.05,
            line=dict(color="black", dash="dash")
        )
        
        # Update layout
        fig.update_layout(
            title=f"Multiple Hypothesis Testing ({correction} correction)",
            xaxis_title="Hypothesis",
            yaxis_title="p-value",
            yaxis_type="log",  # Log scale for p-values
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
            yaxis=dict(range=[-3, 0])  # Log scale from 0.001 to 1
        )
        
        # Add annotation for significance threshold
        fig.add_annotation(
            text="Significance threshold (α=0.05)",
            x=len(hypotheses) - 1,
            y=0.05,
            xshift=10,
            yshift=5,
            showarrow=True,
            arrowhead=2
        )
        
        return fig


# Helper functions for use in the Streamlit UI
def render_hypothesis_tester_ui(graph: nx.DiGraph, data: pd.DataFrame, key_prefix: str = ""):
    """
    Render the UI for the hypothesis tester component
    
    Args:
        graph: The causal graph
        data: The dataset
        key_prefix: Prefix for Streamlit widget keys
    """
    st.subheader("Causal Hypothesis Testing")
    
    # Create the tester
    tester = HypothesisTester(graph, data)
    
    # Get node names for dropdowns
    node_names = []
    for node in graph.nodes():
        name = None
        if "name" in graph.nodes[node]:
            name = graph.nodes[node]["name"]
        elif isinstance(node, int) and node < len(data.columns):
            name = data.columns[node]
        else:
            name = str(node)
            
        node_names.append(name)
    
    # Test type selection
    test_type = st.selectbox(
        "Test Type",
        ["Causal Relationship", "Backdoor Adjustment", "Multiple Hypotheses"],
        key=f"{key_prefix}test_type"
    )
    
    # UI based on test type
    if test_type == "Causal Relationship":
        # Select cause and effect variables
        col1, col2 = st.columns(2)
        
        with col1:
            cause = st.selectbox(
                "Cause Variable",
                node_names,
                key=f"{key_prefix}cause"
            )
        
        with col2:
            effect = st.selectbox(
                "Effect Variable",
                node_names,
                key=f"{key_prefix}effect"
            )
        
        # Test method
        test_method = st.selectbox(
            "Test Method",
            ["Regression", "Correlation", "Partial Correlation"],
            key=f"{key_prefix}test_method"
        )
        
        # Control variables (for regression and partial correlation)
        if test_method in ["Regression", "Partial Correlation"]:
            controls = st.multiselect(
                "Control Variables",
                [n for n in node_names if n != cause and n != effect],
                key=f"{key_prefix}controls"
            )
        else:
            controls = []
        
        # Run test button
        if st.button("Run Test", key=f"{key_prefix}run_test"):
            with st.spinner("Running test..."):
                # Perform the test
                result = tester.test_causal_relationship(
                    cause=cause,
                    effect=effect,
                    test_type=test_method.lower(),
                    control_for=controls
                )
                
                # Display results
                if result["status"] == "error":
                    st.error(result["message"])
                else:
                    # Display visualization
                    st.subheader("Test Results")
                    fig = tester.visualize_test_results(result)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed results
                    with st.expander("Detailed Results"):
                        # Test conclusion
                        if result["hypothesis_confirmed"]:
                            st.success("✅ Hypothesis Confirmed")
                        else:
                            st.error("❌ Hypothesis Not Confirmed")
                        
                        # Test statistics
                        st.write("### Test Statistics")
                        
                        if test_method == "Regression":
                            st.write(f"**Coefficient:** {result['coefficient']:.4f}")
                            st.write(f"**Standard Error:** {result['std_error']:.4f}")
                            st.write(f"**t-statistic:** {result['t_statistic']:.4f}")
                            st.write(f"**p-value:** {result['p_value']:.4f}")
                            st.write(f"**95% Confidence Interval:** [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
                            st.write(f"**R-squared:** {result['r_squared']:.4f}")
                            
                            # Show model summary
                            st.text(result["model_summary"])
                            
                        elif test_method == "Correlation":
                            st.write(f"**Correlation:** {result['correlation']:.4f}")
                            st.write(f"**p-value:** {result['p_value']:.4f}")
                            
                        elif test_method == "Partial Correlation":
                            st.write(f"**Partial Correlation:** {result['partial_correlation']:.4f}")
                            st.write(f"**p-value:** {result['p_value']:.4f}")
                            st.write(f"**Controlling for:** {', '.join(result['controls'])}")
    
    elif test_type == "Backdoor Adjustment":
        # Select treatment and outcome
        col1, col2 = st.columns(2)
        
        with col1:
            treatment = st.selectbox(
                "Treatment Variable",
                node_names,
                key=f"{key_prefix}treatment"
            )
        
        with col2:
            outcome = st.selectbox(
                "Outcome Variable",
                node_names,
                key=f"{key_prefix}outcome"
            )
        
        # Optional: manual adjustment sets
        use_manual = st.checkbox("Specify Adjustment Sets Manually", key=f"{key_prefix}use_manual")
        
        if use_manual:
            adjustment_vars = st.multiselect(
                "Adjustment Variables",
                [n for n in node_names if n != treatment and n != outcome],
                key=f"{key_prefix}adjustment_vars"
            )
            
            adjustment_sets = [adjustment_vars] if adjustment_vars else [[]]
        else:
            adjustment_sets = None
        
        # Run test button
        if st.button("Run Test", key=f"{key_prefix}run_backdoor"):
            with st.spinner("Running backdoor adjustment test..."):
                # Perform the test
                result = tester.test_backdoor_adjustment(
                    treatment=treatment,
                    outcome=outcome,
                    adjustment_sets=adjustment_sets
                )
                
                # Display results
                if result["status"] == "error":
                    st.error(result["message"])
                else:
                    # Display visualization
                    st.subheader("Adjustment Test Results")
                    fig = tester.visualize_test_results(result)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed results
                    with st.expander("Detailed Results"):
                        # Test conclusion
                        if result["hypothesis_confirmed"]:
                            st.success("✅ Causal Effect Confirmed with Adjustment")
                        else:
                            st.error("❌ Causal Effect Not Confirmed")
                        
                        # Unadjusted estimate
                        st.write("### Unadjusted Estimate")
                        st.write(f"**Coefficient:** {result['unadjusted']['coefficient']:.4f}")
                        st.write(f"**p-value:** {result['unadjusted']['p_value']:.4f}")
                        st.write(f"**95% Confidence Interval:** [{result['unadjusted']['confidence_interval'][0]:.4f}, {result['unadjusted']['confidence_interval'][1]:.4f}]")
                        
                        # Adjusted estimates
                        st.write("### Adjusted Estimates")
                        for i, adj in enumerate(result["adjustment_results"]):
                            st.write(f"**Adjustment Set {i+1}:** {', '.join(adj['adjustment_set']) if adj['adjustment_set'] else 'Empty set'}")
                            st.write(f"**Coefficient:** {adj['coefficient']:.4f}")
                            st.write(f"**p-value:** {adj['p_value']:.4f}")
                            st.write(f"**95% Confidence Interval:** [{adj['confidence_interval'][0]:.4f}, {adj['confidence_interval'][1]:.4f}]")
                            st.write(f"**R-squared:** {adj['r_squared']:.4f}")
                            st.write("---")
    
    elif test_type == "Multiple Hypotheses":
        # Create UI for multiple hypotheses
        st.write("### Define Multiple Hypotheses")
        
        # Number of hypotheses
        num_hypotheses = st.number_input(
            "Number of Hypotheses",
            min_value=1,
            max_value=10,
            value=2,
            key=f"{key_prefix}num_hypotheses"
        )
        
        # Create fields for each hypothesis
        hypotheses = []
        
        for i in range(int(num_hypotheses)):
            st.write(f"#### Hypothesis {i+1}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                h_cause = st.selectbox(
                    "Cause",
                    node_names,
                    key=f"{key_prefix}h{i}_cause"
                )
            
            with col2:
                h_effect = st.selectbox(
                    "Effect",
                    node_names,
                    key=f"{key_prefix}h{i}_effect"
                )
            
            with col3:
                h_test = st.selectbox(
                    "Test",
                    ["Regression", "Correlation"],
                    key=f"{key_prefix}h{i}_test"
                )
            
            # Optional control variables
            if h_test == "Regression":
                h_controls = st.multiselect(
                    "Control Variables",
                    [n for n in node_names if n != h_cause and n != h_effect],
                    key=f"{key_prefix}h{i}_controls"
                )
            else:
                h_controls = []
            
            # Add to hypotheses list
            hypotheses.append({
                "cause": h_cause,
                "effect": h_effect,
                "test_type": h_test.lower(),
                "control_for": h_controls
            })
        
        # Multiple testing correction method
        correction = st.selectbox(
            "Multiple Testing Correction",
            ["Bonferroni", "FDR (Benjamini-Hochberg)", "None"],
            key=f"{key_prefix}correction"
        )
        
        correction_map = {
            "Bonferroni": "bonferroni",
            "FDR (Benjamini-Hochberg)": "fdr",
            "None": "none"
        }
        
        # Run tests button
        if st.button("Run Multiple Tests", key=f"{key_prefix}run_multiple"):
            with st.spinner("Running multiple hypothesis tests..."):
                # Perform the tests
                result = tester.test_multiple_hypotheses(
                    hypotheses=hypotheses,
                    correction_method=correction_map[correction]
                )
                
                # Display results
                if result["status"] == "error":
                    st.error(result["message"])
                else:
                    # Display visualization
                    st.subheader("Multiple Tests Results")
                    fig = tester.visualize_test_results(result)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed results
                    with st.expander("Detailed Results"):
                        # Overall results
                        st.write(f"**Number of Hypotheses Tested:** {result['n_tested']}")
                        st.write(f"**Correction Method:** {result['correction_method']}")
                        
                        if result['overall_confirmed']:
                            st.success("✅ At least one hypothesis confirmed after correction")
                        else:
                            st.error("❌ No hypotheses confirmed after correction")
                        
                        # Individual results
                        st.write("### Individual Test Results")
                        
                        for i, r in enumerate(result["results"]):
                            cause = r.get("cause", "")
                            effect = r.get("effect", "")
                            
                            st.write(f"**Hypothesis {i+1}:** {cause} → {effect}")
                            
                            # Display different statistics based on test type
                            if r.get("test_type") == "regression":
                                st.write(f"**Coefficient:** {r.get('coefficient', 0):.4f}")
                                st.write(f"**t-statistic:** {r.get('t_statistic', 0):.4f}")
                            else:
                                if "correlation" in r:
                                    st.write(f"**Correlation:** {r.get('correlation', 0):.4f}")
                                else:
                                    st.write(f"**Partial Correlation:** {r.get('partial_correlation', 0):.4f}")
                            
                            # Common statistics
                            st.write(f"**p-value:** {r.get('p_value', 1.0):.4f}")
                            st.write(f"**Adjusted p-value:** {r.get('adjusted_p_value', 1.0):.4f}")
                            
                            # Significance
                            if r.get("significant_after_correction", False):
                                st.success("✅ Significant after correction")
                            elif r.get("p_value", 1.0) < 0.05:
                                st.warning("⚠️ Significant before correction but not after")
                            else:
                                st.error("❌ Not significant")
                            
                            st.write("---")


def test_causal_hypothesis(graph: nx.DiGraph, 
                         data: pd.DataFrame,
                         cause: str,
                         effect: str,
                         test_type: str = 'regression',
                         control_for: List[str] = None) -> Dict[str, Any]:
    """
    Test a specific causal hypothesis
    
    Args:
        graph: Causal graph
        data: Dataset
        cause: Cause variable
        effect: Effect variable
        test_type: Type of test
        control_for: Variables to control for
        
    Returns:
        Test result dictionary
    """
    tester = HypothesisTester(graph, data)
    return tester.test_causal_relationship(
        cause=cause,
        effect=effect,
        test_type=test_type,
        control_for=control_for if control_for else []
    )


def visualize_hypothesis_test(graph: nx.DiGraph,
                            data: pd.DataFrame,
                            test_result: Dict[str, Any]) -> go.Figure:
    """
    Visualize a hypothesis test result
    
    Args:
        graph: Causal graph
        data: Dataset
        test_result: Result from test_causal_hypothesis
        
    Returns:
        Plotly figure
    """
    tester = HypothesisTester(graph, data)
    return tester.visualize_test_results(test_result)