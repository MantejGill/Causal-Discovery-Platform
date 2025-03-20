# core/viz/counterfactual.py

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CounterfactualAnalyzer:
    """
    Generates and visualizes counterfactual scenarios based on causal graphs.
    Simulates the effect of interventions on causal variables and predicts
    the impact on downstream variables.
    """
    
    def __init__(self, graph: nx.DiGraph, data: pd.DataFrame):
        """
        Initialize the counterfactual analyzer
        
        Args:
            graph: NetworkX DiGraph representing the causal structure
            data: DataFrame containing the data for variable distributions
        """
        self.graph = graph
        self.data = data
        self.causal_models = {}  # Cache for fitted causal models
    
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
    
    def get_valid_intervention_targets(self) -> List[Dict[str, Any]]:
        """
        Get a list of valid targets for intervention
        
        Returns:
            List of node details for intervention targets
        """
        targets = []
        
        for node in self.graph.nodes():
            # Get node name
            node_name = self._get_node_name(node)
            
            # Create details
            targets.append({
                "id": node,
                "name": node_name,
                "type": "categorical" if pd.api.types.is_categorical_dtype(self.data[node_name]) or 
                                        pd.api.types.is_object_dtype(self.data[node_name]) else "continuous"
            })
        
        return targets
    
    def get_valid_outcome_targets(self, intervention_node: Any) -> List[Dict[str, Any]]:
        """
        Get a list of valid outcome targets for a given intervention
        
        Args:
            intervention_node: The node being intervened upon
            
        Returns:
            List of node details for valid outcome targets
        """
        outcomes = []
        
        # Get descendants of intervention node
        try:
            descendants = nx.descendants(self.graph, intervention_node)
        except:
            descendants = []
        
        # Include descendants and intervention node itself
        valid_targets = list(descendants) + [intervention_node]
        
        for node in valid_targets:
            # Get node name
            node_name = self._get_node_name(node)
            
            # Create details
            outcomes.append({
                "id": node,
                "name": node_name,
                "type": "categorical" if pd.api.types.is_categorical_dtype(self.data[node_name]) or 
                                        pd.api.types.is_object_dtype(self.data[node_name]) else "continuous" 
            })
        
        return outcomes
    
    def _get_model(self, target_node: Any, feature_nodes: List[Any] = None) -> Tuple[Any, List[str]]:
        """
        Get or create a causal model for a target node
        
        Args:
            target_node: The node to predict
            feature_nodes: List of feature nodes (default: use parents from the graph)
            
        Returns:
            Tuple of (model, feature_names)
        """
        target_name = self._get_node_name(target_node)
        
        # If feature nodes not provided, use parents from the graph
        if feature_nodes is None:
            feature_nodes = list(self.graph.predecessors(target_node))
        
        # Get feature names
        feature_names = [self._get_node_name(node) for node in feature_nodes]
        
        # Check if model is cached
        model_key = (target_node, tuple(feature_nodes))
        if model_key in self.causal_models:
            return self.causal_models[model_key], feature_names
        
        # Create a new model
        target_data = self.data[target_name]
        
        # Handle different target types
        if pd.api.types.is_categorical_dtype(target_data) or pd.api.types.is_object_dtype(target_data):
            # Categorical target - use logistic regression or classification
            try:
                from sklearn.linear_model import LogisticRegression
                
                # Prepare data
                X = self.data[feature_names]
                y = target_data
                
                # Handle categorical features
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.pipeline import Pipeline
                
                # Identify categorical features
                categorical_features = [i for i, name in enumerate(feature_names) 
                                        if pd.api.types.is_categorical_dtype(self.data[name]) or 
                                        pd.api.types.is_object_dtype(self.data[name])]
                numeric_features = [i for i, name in enumerate(feature_names) 
                                   if i not in categorical_features]
                
                # Create preprocessing
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ])
                
                # Create pipeline
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(max_iter=1000))
                ])
                
                # Fit model
                model.fit(X, y)
                
            except Exception as e:
                logger.error(f"Error creating categorical model for {target_name}: {str(e)}")
                # Fallback to simple model
                model = None
        else:
            # Continuous target - use linear regression
            try:
                from sklearn.linear_model import LinearRegression
                
                # Prepare data
                X = self.data[feature_names]
                y = target_data
                
                # Handle categorical features
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.pipeline import Pipeline
                
                # Identify categorical features
                categorical_features = [i for i, name in enumerate(feature_names) 
                                        if pd.api.types.is_categorical_dtype(self.data[name]) or 
                                        pd.api.types.is_object_dtype(self.data[name])]
                numeric_features = [i for i, name in enumerate(feature_names) 
                                   if i not in categorical_features]
                
                # Create preprocessing
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ], remainder='passthrough')
                
                # Create pipeline
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', LinearRegression())
                ])
                
                # Fit model
                model.fit(X, y)
                
            except Exception as e:
                logger.error(f"Error creating continuous model for {target_name}: {str(e)}")
                # Fallback to simple model
                model = None
        
        # Cache the model
        self.causal_models[model_key] = model
        
        return model, feature_names
    
    def generate_counterfactual(self, 
                              intervention_node: Any, 
                              intervention_value: Any,
                              outcome_node: Any,
                              num_samples: int = 100) -> Dict[str, Any]:
        """
        Generate a counterfactual scenario
        
        Args:
            intervention_node: The node to intervene on
            intervention_value: The value to set the intervention node to
            outcome_node: The node to observe the outcome on
            num_samples: Number of counterfactual samples to generate
            
        Returns:
            Dictionary with counterfactual details
        """
        # Get node names
        intervention_name = self._get_node_name(intervention_node)
        outcome_name = self._get_node_name(outcome_node)
        
        # Convert intervention value to appropriate type if needed
        try:
            original_type = self.data[intervention_name].dtype
            intervention_value = original_type.type(intervention_value)
        except:
            # Keep as is if conversion fails
            pass
        
        # Check if outcome is causally downstream from intervention
        if intervention_node != outcome_node:
            try:
                if outcome_node not in nx.descendants(self.graph, intervention_node):
                    return {
                        "status": "error",
                        "message": f"Outcome variable {outcome_name} is not causally affected by intervention on {intervention_name}"
                    }
            except:
                # If descendants cannot be computed, continue anyway
                pass
        
        # Sample from original data for counterfactuals
        cf_samples = self.data.sample(n=min(num_samples, len(self.data)), replace=num_samples > len(self.data))
        
        # Save original values
        original_intervention = cf_samples[intervention_name].copy()
        original_outcome = cf_samples[outcome_name].copy()
        
        # Create a copy for counterfactual prediction
        cf_data = cf_samples.copy()
        
        # Set intervention value
        cf_data[intervention_name] = intervention_value
        
        # If intervention node equals outcome node, the outcome is just the intervention value
        if intervention_node == outcome_node:
            predicted_outcome = pd.Series([intervention_value] * len(cf_data), index=cf_data.index)
        else:
            # Otherwise, predict using causal model
            try:
                # Get nodes on paths from intervention to outcome
                # This identifies the necessary variables to update
                all_paths = list(nx.all_simple_paths(self.graph, intervention_node, outcome_node))
                
                if not all_paths:
                    return {
                        "status": "error",
                        "message": f"No causal paths found from {intervention_name} to {outcome_name}"
                    }
                
                # Collect all nodes on all paths
                path_nodes = set()
                for path in all_paths:
                    path_nodes.update(path)
                
                # Remove intervention node from path_nodes
                path_nodes.discard(intervention_node)
                
                # Topologically sort the nodes on the paths
                affected_nodes = [n for n in nx.topological_sort(self.graph) if n in path_nodes]
                
                # Update each affected node in topological order
                for node in affected_nodes:
                    node_name = self._get_node_name(node)
                    
                    # Get parent nodes
                    parents = list(self.graph.predecessors(node))
                    
                    # Get or create model for this node
                    model, feature_names = self._get_model(node, parents)
                    
                    if model is not None:
                        # Predict using the model
                        X_pred = cf_data[feature_names]
                        cf_data[node_name] = model.predict(X_pred)
                    else:
                        # If model creation failed, use original values
                        logger.warning(f"Using original values for {node_name} due to model failure")
                
                # Get the predicted outcome
                predicted_outcome = cf_data[outcome_name]
                
            except Exception as e:
                logger.error(f"Error generating counterfactual: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error generating counterfactual: {str(e)}"
                }
        
        # Calculate summary statistics
        original_mean = original_outcome.mean()
        original_std = original_outcome.std()
        cf_mean = predicted_outcome.mean()
        cf_std = predicted_outcome.std()
        
        # Calculate effect size
        effect = cf_mean - original_mean
        
        # Calculate percentage change
        if original_mean != 0:
            percentage_change = (effect / abs(original_mean)) * 100
        else:
            percentage_change = float('inf') if effect > 0 else float('-inf') if effect < 0 else 0
        
        # Determine confidence interval
        if len(predicted_outcome) >= 30:
            # Use t-distribution for confidence interval
            t_value = stats.t.ppf(0.975, df=len(predicted_outcome)-1)
            margin = t_value * (cf_std / np.sqrt(len(predicted_outcome)))
            ci_lower = cf_mean - margin
            ci_upper = cf_mean + margin
        else:
            # Simple percentile-based interval for small samples
            ci_lower = np.percentile(predicted_outcome, 2.5)
            ci_upper = np.percentile(predicted_outcome, 97.5)
        
        # Create detailed results
        details = []
        for i in range(len(cf_samples)):
            details.append({
                "original_intervention": original_intervention.iloc[i],
                "intervention_value": intervention_value,
                "original_outcome": original_outcome.iloc[i],
                "counterfactual_outcome": predicted_outcome.iloc[i],
                "effect": predicted_outcome.iloc[i] - original_outcome.iloc[i]
            })
        
        # Create narrative explanation based on results
        if effect > 0:
            direction = "increase"
        elif effect < 0:
            direction = "decrease"
        else:
            direction = "no change in"
        
        # Create explanation text
        explanation = f"Setting {intervention_name} to {intervention_value} is predicted to {direction} {outcome_name} "
        explanation += f"by {abs(effect):.4f} ({abs(percentage_change):.2f}%).\n\n"
        explanation += f"Original {outcome_name} (mean): {original_mean:.4f}\n"
        explanation += f"Counterfactual {outcome_name} (mean): {cf_mean:.4f}\n"
        explanation += f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
        
        # Add interpretation of effect size
        if abs(percentage_change) < 5:
            explanation += "This represents a **minimal effect** that might not be practically significant."
        elif abs(percentage_change) < 20:
            explanation += "This represents a **moderate effect** that could be practically relevant."
        else:
            explanation += "This represents a **substantial effect** that is likely to be practically significant."
        
        # Create results dictionary
        result = {
            "status": "completed",
            "intervention": {
                "node": intervention_node,
                "name": intervention_name,
                "value": intervention_value
            },
            "outcome": {
                "node": outcome_node,
                "name": outcome_name
            },
            "effect": {
                "original_mean": original_mean,
                "original_std": original_std,
                "counterfactual_mean": cf_mean,
                "counterfactual_std": cf_std,
                "absolute_effect": effect,
                "percentage_change": percentage_change,
                "confidence_interval": [ci_lower, ci_upper]
            },
            "details": details,
            "explanation": explanation
        }
        
        return result
    
    def compare_interventions(self, 
                            intervention_node: Any,
                            intervention_values: List[Any],
                            outcome_node: Any,
                            num_samples: int = 100) -> Dict[str, Any]:
        """
        Compare multiple intervention values
        
        Args:
            intervention_node: The node to intervene on
            intervention_values: List of values to set the intervention node to
            outcome_node: The node to observe the outcome on
            num_samples: Number of counterfactual samples per intervention
            
        Returns:
            Dictionary with comparison details
        """
        # Check if there are intervention values to compare
        if not intervention_values:
            return {
                "status": "error",
                "message": "No intervention values provided for comparison"
            }
        
        # Generate counterfactuals for each intervention value
        counterfactuals = []
        
        for value in intervention_values:
            cf = self.generate_counterfactual(
                intervention_node=intervention_node,
                intervention_value=value,
                outcome_node=outcome_node,
                num_samples=num_samples
            )
            
            if cf.get("status") == "error":
                return cf  # Return error if any counterfactual fails
                
            counterfactuals.append(cf)
        
        # Collect results for comparison
        intervention_name = self._get_node_name(intervention_node)
        outcome_name = self._get_node_name(outcome_node)
        
        comparison = {
            "status": "completed",
            "intervention_node": {
                "id": intervention_node,
                "name": intervention_name
            },
            "outcome_node": {
                "id": outcome_node,
                "name": outcome_name
            },
            "counterfactuals": counterfactuals,
            "comparison": []
        }
        
        # Create comparison data
        for cf in counterfactuals:
            comparison["comparison"].append({
                "intervention_value": cf["intervention"]["value"],
                "outcome_mean": cf["effect"]["counterfactual_mean"],
                "outcome_ci_lower": cf["effect"]["confidence_interval"][0],
                "outcome_ci_upper": cf["effect"]["confidence_interval"][1],
                "absolute_effect": cf["effect"]["absolute_effect"],
                "percentage_change": cf["effect"]["percentage_change"]
            })
        
        # Find the "best" intervention based on absolute effect
        # In a more sophisticated implementation, this could consider various optimization criteria
        comparison["comparison"].sort(key=lambda x: abs(x["absolute_effect"]), reverse=True)
        comparison["best_intervention"] = comparison["comparison"][0]["intervention_value"]
        
        return comparison
    
    def visualize_counterfactual(self, counterfactual_result: Dict[str, Any]) -> go.Figure:
        """
        Create visualization for a counterfactual scenario
        
        Args:
            counterfactual_result: The result from generate_counterfactual
            
        Returns:
            Plotly figure with counterfactual visualization
        """
        if counterfactual_result.get("status") != "completed":
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=counterfactual_result.get("message", "Error generating counterfactual"),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig
        
        # Extract data
        intervention_name = counterfactual_result["intervention"]["name"]
        intervention_value = counterfactual_result["intervention"]["value"]
        outcome_name = counterfactual_result["outcome"]["name"]
        
        original_mean = counterfactual_result["effect"]["original_mean"]
        cf_mean = counterfactual_result["effect"]["counterfactual_mean"]
        
        ci_lower = counterfactual_result["effect"]["confidence_interval"][0]
        ci_upper = counterfactual_result["effect"]["confidence_interval"][1]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for original and counterfactual
        fig.add_trace(go.Bar(
            x=["Original", "Counterfactual"],
            y=[original_mean, cf_mean],
            error_y=dict(
                type="data",
                array=[0, (ci_upper - cf_mean)],
                arrayminus=[0, (cf_mean - ci_lower)]
            ),
            text=[f"{original_mean:.3f}", f"{cf_mean:.3f}"],
            textposition="auto",
            marker_color=["#1f77b4", "#ff7f0e"]
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Effect of {intervention_name} = {intervention_value} on {outcome_name}",
            xaxis_title="Scenario",
            yaxis_title=f"Average {outcome_name}",
            showlegend=False
        )
        
        # Add effect size annotation
        effect = cf_mean - original_mean
        percentage = counterfactual_result["effect"]["percentage_change"]
        
        if effect > 0:
            direction = "increase"
        elif effect < 0:
            direction = "decrease"
        else:
            direction = "no change"
            
        fig.add_annotation(
            text=f"{direction.capitalize()}: {abs(effect):.3f} ({abs(percentage):.1f}%)",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig
    
    def visualize_intervention_comparison(self, comparison_result: Dict[str, Any]) -> go.Figure:
        """
        Create visualization for an intervention comparison
        
        Args:
            comparison_result: The result from compare_interventions
            
        Returns:
            Plotly figure with comparison visualization
        """
        if comparison_result.get("status") != "completed":
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=comparison_result.get("message", "Error comparing interventions"),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig
        
        # Extract data
        intervention_name = comparison_result["intervention_node"]["name"]
        outcome_name = comparison_result["outcome_node"]["name"]
        
        # Create data for plotting
        intervention_values = []
        outcome_means = []
        ci_lowers = []
        ci_uppers = []
        
        for item in comparison_result["comparison"]:
            intervention_values.append(str(item["intervention_value"]))
            outcome_means.append(item["outcome_mean"])
            ci_lowers.append(item["outcome_ci_lower"])
            ci_uppers.append(item["outcome_ci_upper"])
        
        # Calculate error bar arrays
        error_y = np.array(ci_uppers) - np.array(outcome_means)
        error_y_minus = np.array(outcome_means) - np.array(ci_lowers)
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each intervention value
        fig.add_trace(go.Bar(
            x=intervention_values,
            y=outcome_means,
            error_y=dict(
                type="data",
                array=error_y,
                arrayminus=error_y_minus
            ),
            text=[f"{mean:.3f}" for mean in outcome_means],
            textposition="auto"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Effect of {intervention_name} on {outcome_name}",
            xaxis_title=intervention_name,
            yaxis_title=f"Average {outcome_name}",
            showlegend=False
        )
        
        # Highlight the best intervention
        best_intervention = comparison_result.get("best_intervention")
        if best_intervention is not None:
            best_idx = intervention_values.index(str(best_intervention))
            
            # Update marker color for best intervention
            fig.data[0].marker.color = ["#1f77b4"] * len(intervention_values)
            fig.data[0].marker.color[best_idx] = "#ff7f0e"
            
            # Add annotation
            fig.add_annotation(
                text="Best Intervention",
                x=intervention_values[best_idx],
                y=outcome_means[best_idx] + error_y[best_idx] + 0.1 * max(outcome_means),
                showarrow=True,
                arrowhead=2,
                arrowcolor="#ff7f0e",
                font=dict(color="#ff7f0e")
            )
        
        return fig
    
    def create_distribution_visualization(self, 
                                       counterfactual_result: Dict[str, Any],
                                       use_kde: bool = True) -> go.Figure:
        """
        Create a visualization of original vs counterfactual distributions
        
        Args:
            counterfactual_result: The result from generate_counterfactual
            use_kde: Whether to use KDE for continuous variables
            
        Returns:
            Plotly figure with distribution visualization
        """
        if counterfactual_result.get("status") != "completed":
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=counterfactual_result.get("message", "Error generating counterfactual"),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig
        
        # Extract data
        details = counterfactual_result.get("details", [])
        if not details:
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text="No detailed data available for distribution visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig
        
        # Extract original and counterfactual outcomes
        original_outcomes = [d["original_outcome"] for d in details]
        cf_outcomes = [d["counterfactual_outcome"] for d in details]
        
        intervention_name = counterfactual_result["intervention"]["name"]
        intervention_value = counterfactual_result["intervention"]["value"]
        outcome_name = counterfactual_result["outcome"]["name"]
        
        # Check if outcome is categorical
        is_categorical = False
        if not all(isinstance(x, (int, float)) for x in original_outcomes) or not all(isinstance(x, (int, float)) for x in cf_outcomes):
            is_categorical = True
        else:
            # Check number of unique values - if few, treat as categorical
            unique_vals = set(original_outcomes + cf_outcomes)
            is_categorical = len(unique_vals) < 10
        
        # Create figure
        fig = go.Figure()
        
        if is_categorical:
            # Create frequency tables
            from collections import Counter
            original_counts = Counter(original_outcomes)
            cf_counts = Counter(cf_outcomes)
            
            # Get all unique categories
            all_categories = sorted(set(original_counts.keys()) | set(cf_counts.keys()))
            
            # Create bar chart of frequencies
            for category in all_categories:
                fig.add_trace(go.Bar(
                    name="Original",
                    x=[f"{category}"],
                    y=[original_counts.get(category, 0)],
                    marker_color="#1f77b4"
                ))
                
                fig.add_trace(go.Bar(
                    name="Counterfactual",
                    x=[f"{category}"],
                    y=[cf_counts.get(category, 0)],
                    marker_color="#ff7f0e"
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Distribution of {outcome_name} under {intervention_name} = {intervention_value}",
                xaxis_title=outcome_name,
                yaxis_title="Frequency",
                barmode="group"
            )
        else:
            # Continuous data - use histogram with optional KDE
            fig.add_trace(go.Histogram(
                x=original_outcomes,
                name="Original",
                opacity=0.7,
                marker_color="#1f77b4",
                histnorm="probability density"
            ))
            
            fig.add_trace(go.Histogram(
                x=cf_outcomes,
                name="Counterfactual",
                opacity=0.7,
                marker_color="#ff7f0e",
                histnorm="probability density"
            ))
            
            # Add KDE if requested
            if use_kde:
                try:
                    from scipy.stats import gaussian_kde
                    
                    # KDE for original outcomes
                    if len(set(original_outcomes)) > 1:  # Need at least 2 different values
                        kde_orig = gaussian_kde(original_outcomes)
                        x_range = np.linspace(min(original_outcomes), max(original_outcomes), 200)
                        y_kde_orig = kde_orig(x_range)
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_kde_orig,
                            mode="lines",
                            name="Original KDE",
                            line=dict(color="#1f77b4", width=2)
                        ))
                    
                    # KDE for counterfactual outcomes
                    if len(set(cf_outcomes)) > 1:  # Need at least 2 different values
                        kde_cf = gaussian_kde(cf_outcomes)
                        x_range = np.linspace(min(cf_outcomes), max(cf_outcomes), 200)
                        y_kde_cf = kde_cf(x_range)
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_kde_cf,
                            mode="lines",
                            name="Counterfactual KDE",
                            line=dict(color="#ff7f0e", width=2)
                        ))
                except Exception as e:
                    logger.warning(f"Could not generate KDE: {str(e)}")
            
            # Update layout
            fig.update_layout(
                title=f"Distribution of {outcome_name} under {intervention_name} = {intervention_value}",
                xaxis_title=outcome_name,
                yaxis_title="Probability Density",
                barmode="overlay"
            )
        
        return fig
    
    def create_detailed_report(self, counterfactual_result: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create a detailed report with multiple visualizations
        
        Args:
            counterfactual_result: The result from generate_counterfactual
            
        Returns:
            Dictionary of visualization figures
        """
        figures = {}
        
        # Main effect visualization
        figures["main_effect"] = self.visualize_counterfactual(counterfactual_result)
        
        # Distribution visualization
        figures["distribution"] = self.create_distribution_visualization(counterfactual_result)
        
        # Individual effects scatter plot
        details = counterfactual_result.get("details", [])
        if details:
            original_outcomes = [d["original_outcome"] for d in details]
            cf_outcomes = [d["counterfactual_outcome"] for d in details]
            effects = [d["effect"] for d in details]
            
            intervention_name = counterfactual_result["intervention"]["name"]
            intervention_value = counterfactual_result["intervention"]["value"]
            outcome_name = counterfactual_result["outcome"]["name"]
            
            # Create scatter plot of original vs counterfactual
            scatter_fig = go.Figure()
            
            scatter_fig.add_trace(go.Scatter(
                x=original_outcomes,
                y=cf_outcomes,
                mode="markers",
                marker=dict(
                    color=effects,
                    colorscale="RdBu",
                    colorbar=dict(title="Effect"),
                    size=10,
                    line=dict(width=1)
                ),
                text=[f"Effect: {e:.3f}" for e in effects],
                hoverinfo="text"
            ))
            
            # Add diagonal line (no effect)
            min_val = min(min(original_outcomes), min(cf_outcomes))
            max_val = max(max(original_outcomes), max(cf_outcomes))
            scatter_fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="No effect"
            ))
            
            scatter_fig.update_layout(
                title=f"Individual Effects of {intervention_name} = {intervention_value} on {outcome_name}",
                xaxis_title=f"Original {outcome_name}",
                yaxis_title=f"Counterfactual {outcome_name}",
                showlegend=False
            )
            
            figures["individual_effects"] = scatter_fig
        
        return figures


# Helper functions for use in UI components
def generate_counterfactual_scenario(graph: nx.DiGraph, 
                                  data: pd.DataFrame,
                                  intervention_node: str,
                                  intervention_value: Any,
                                  outcome_node: str,
                                  num_samples: int = 100) -> Dict[str, Any]:
    """
    Generate a counterfactual scenario
    
    Args:
        graph: Causal graph
        data: Dataset
        intervention_node: Node to intervene on
        intervention_value: Value to set intervention node to
        outcome_node: Node to observe outcome on
        num_samples: Number of samples
        
    Returns:
        Counterfactual result dictionary
    """
    analyzer = CounterfactualAnalyzer(graph, data)
    return analyzer.generate_counterfactual(
        intervention_node=intervention_node,
        intervention_value=intervention_value,
        outcome_node=outcome_node,
        num_samples=num_samples
    )

def compare_intervention_values(graph: nx.DiGraph, 
                             data: pd.DataFrame,
                             intervention_node: str,
                             intervention_values: List[Any],
                             outcome_node: str,
                             num_samples: int = 100) -> Dict[str, Any]:
    """
    Compare multiple intervention values
    
    Args:
        graph: Causal graph
        data: Dataset
        intervention_node: Node to intervene on
        intervention_values: List of values to try
        outcome_node: Node to observe outcome on
        num_samples: Number of samples per intervention
        
    Returns:
        Comparison result dictionary
    """
    analyzer = CounterfactualAnalyzer(graph, data)
    return analyzer.compare_interventions(
        intervention_node=intervention_node,
        intervention_values=intervention_values,
        outcome_node=outcome_node,
        num_samples=num_samples
    )

def visualize_counterfactual_result(graph: nx.DiGraph,
                                 data: pd.DataFrame,
                                 counterfactual_result: Dict[str, Any]) -> go.Figure:
    """
    Visualize a counterfactual result
    
    Args:
        graph: Causal graph
        data: Dataset
        counterfactual_result: Result from generate_counterfactual
        
    Returns:
        Plotly figure
    """
    analyzer = CounterfactualAnalyzer(graph, data)
    return analyzer.visualize_counterfactual(counterfactual_result)

def get_possible_intervention_targets(graph: nx.DiGraph, data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Get list of possible intervention targets
    
    Args:
        graph: Causal graph
        data: Dataset
        
    Returns:
        List of intervention target details
    """
    analyzer = CounterfactualAnalyzer(graph, data)
    return analyzer.get_valid_intervention_targets()

def get_possible_outcome_targets(graph: nx.DiGraph, 
                             data: pd.DataFrame, 
                             intervention_node: str) -> List[Dict[str, Any]]:
    """
    Get list of possible outcome targets
    
    Args:
        graph: Causal graph
        data: Dataset
        intervention_node: Selected intervention node
        
    Returns:
        List of outcome target details
    """
    analyzer = CounterfactualAnalyzer(graph, data)
    return analyzer.get_valid_outcome_targets(intervention_node)