# app/components/effect_estimator.py

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EffectEstimator:
    """
    Estimate causal effects between variables using various causal inference methods.
    Supports backdoor adjustment, frontdoor adjustment, instrumental variables,
    and counterfactual estimation.
    """
    
    def __init__(self, graph: nx.DiGraph, data: pd.DataFrame):
        """
        Initialize effect estimator
        
        Args:
            graph: NetworkX DiGraph representing the causal structure
            data: DataFrame containing the data
        """
        self.graph = graph
        self.data = data
        self.estimates = {}
        self.models = {}
    
    def _get_node_name(self, node: Any) -> str:
        """Get the readable name for a node"""
        if "name" in self.graph.nodes[node]:
            return self.graph.nodes[node]["name"]
        elif isinstance(node, int) and node < len(self.data.columns):
            return self.data.columns[node]
        else:
            return str(node)
    
    def _get_node_id(self, node_name: str) -> Optional[Any]:
        """Find the node ID for a given node name"""
        for node in self.graph.nodes():
            if self._get_node_name(node) == node_name:
                return node
        return None
    
    def _find_backdoor_adjustment_set(self, treatment: Any, outcome: Any) -> List[Any]:
        """
        Find a valid backdoor adjustment set
        
        Args:
            treatment: Treatment variable node
            outcome: Outcome variable node
            
        Returns:
            List of nodes forming a valid adjustment set
        """
        # Method 1: Use parents of treatment
        parents = list(self.graph.predecessors(treatment))
        
        # Method 2: If available, use a more sophisticated algorithm (requires other packages)
        # For now, we'll use the simple parent-based approach
        return parents
    
    def _find_frontdoor_adjustment_set(self, treatment: Any, outcome: Any) -> List[Any]:
        """
        Find a valid frontdoor adjustment set
        
        Args:
            treatment: Treatment variable node
            outcome: Outcome variable node
            
        Returns:
            List of nodes forming a valid frontdoor adjustment set
        """
        # Find all descendants of treatment that are also ancestors of outcome
        descendants_of_treatment = nx.descendants(self.graph, treatment)
        
        try:
            # This will fail if outcome is not reachable from treatment
            ancestors_of_outcome = nx.ancestors(self.graph, outcome)
        except:
            ancestors_of_outcome = set()
        
        mediators = descendants_of_treatment.intersection(ancestors_of_outcome)
        
        # Exclude outcome if it's in the set
        if outcome in mediators:
            mediators.remove(outcome)
        
        # Find mediators that block all paths from treatment to outcome
        valid_frontdoor_set = []
        for mediator in mediators:
            # Check if all paths from treatment to outcome go through this mediator
            all_paths_blocked = True
            
            # Create a copy of the graph without the mediator
            G_temp = self.graph.copy()
            G_temp.remove_node(mediator)
            
            # Check if outcome is still reachable from treatment
            try:
                path = nx.shortest_path(G_temp, treatment, outcome)
                all_paths_blocked = False
            except nx.NetworkXNoPath:
                pass
            
            if all_paths_blocked:
                valid_frontdoor_set.append(mediator)
        
        return valid_frontdoor_set
    
    def _find_instrumental_variables(self, treatment: Any, outcome: Any) -> List[Any]:
        """
        Find valid instrumental variables
        
        Args:
            treatment: Treatment variable node
            outcome: Outcome variable node
            
        Returns:
            List of nodes that can serve as instruments
        """
        # An instrument Z must:
        # 1. Affect the treatment (Z->X)
        # 2. Affect the outcome only through the treatment (no direct Z->Y)
        # 3. Have no common causes with the outcome
        
        instruments = []
        
        # Find parents of treatment
        treatment_parents = list(self.graph.predecessors(treatment))
        
        # Check each parent as a potential instrument
        for parent in treatment_parents:
            # Check if parent has a direct path to outcome
            has_direct_path = False
            
            # Create a copy of the graph without the treatment
            G_temp = self.graph.copy()
            G_temp.remove_node(treatment)
            
            # Check if outcome is reachable from parent in this modified graph
            try:
                path = nx.shortest_path(G_temp, parent, outcome)
                has_direct_path = True
            except nx.NetworkXNoPath:
                pass
            
            if not has_direct_path:
                # Check for common causes with outcome
                has_common_cause = False
                parent_ancestors = nx.ancestors(self.graph, parent)
                outcome_ancestors = nx.ancestors(self.graph, outcome)
                
                common_ancestors = parent_ancestors.intersection(outcome_ancestors)
                if common_ancestors:
                    has_common_cause = True
                
                if not has_common_cause:
                    instruments.append(parent)
        
        return instruments
    
    def estimate_average_treatment_effect(self, 
                                        treatment: Any, 
                                        outcome: Any,
                                        method: str = "adjustment",
                                        covariates: Optional[List[Any]] = None,
                                        bootstrap_samples: int = 0) -> Dict[str, Any]:
        """
        Estimate the average treatment effect
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            method: Estimation method ("adjustment", "backdoor", "frontdoor", "iv", "simple")
            covariates: Optional list of covariates to adjust for (overrides automatic selection)
            bootstrap_samples: Number of bootstrap samples for confidence intervals (0 for no bootstrapping)
            
        Returns:
            Dictionary with effect estimates and details
        """
        # Get variable names for data access
        treatment_name = self._get_node_name(treatment)
        outcome_name = self._get_node_name(treatment)
        
        # Check that variables exist in the data
        if treatment_name not in self.data.columns:
            return {
                "status": "error",
                "message": f"Treatment variable '{treatment_name}' not found in data"
            }
        
        if outcome_name not in self.data.columns:
            return {
                "status": "error",
                "message": f"Outcome variable '{outcome_name}' not found in data"
            }
        
        # Convert covariates to names
        covariate_names = []
        if covariates:
            covariate_names = [self._get_node_name(cov) for cov in covariates]
            # Check that all covariates exist in the data
            for cov in covariate_names:
                if cov not in self.data.columns:
                    return {
                        "status": "error",
                        "message": f"Covariate '{cov}' not found in data"
                    }
        
        # Check if the treatment is binary or continuous
        is_binary_treatment = self.data[treatment_name].nunique() <= 2
        
        # Select estimation method
        if method == "simple":
            estimate = self._estimate_simple_effect(treatment_name, outcome_name, is_binary_treatment)
        elif method == "adjustment" or method == "backdoor":
            # If covariates not provided, find backdoor adjustment set
            if not covariates:
                backdoor_nodes = self._find_backdoor_adjustment_set(treatment, outcome)
                covariate_names = [self._get_node_name(node) for node in backdoor_nodes]
            
            estimate = self._estimate_backdoor_effect(
                treatment_name, outcome_name, covariate_names, is_binary_treatment
            )
        elif method == "frontdoor":
            # If covariates not provided, find frontdoor adjustment set
            if not covariates:
                frontdoor_nodes = self._find_frontdoor_adjustment_set(treatment, outcome)
                covariate_names = [self._get_node_name(node) for node in frontdoor_nodes]
            
            if not covariate_names:
                return {
                    "status": "error",
                    "message": "No valid frontdoor adjustment set found"
                }
                
            estimate = self._estimate_frontdoor_effect(
                treatment_name, outcome_name, covariate_names, is_binary_treatment
            )
        elif method == "iv":
            # If covariates not provided, find instrumental variables
            if not covariates:
                iv_nodes = self._find_instrumental_variables(treatment, outcome)
                covariate_names = [self._get_node_name(node) for node in iv_nodes]
            
            if not covariate_names:
                return {
                    "status": "error",
                    "message": "No valid instrumental variables found"
                }
                
            estimate = self._estimate_iv_effect(
                treatment_name, outcome_name, covariate_names[0], is_binary_treatment
            )
        else:
            return {
                "status": "error",
                "message": f"Unknown estimation method: {method}"
            }
        
        # Add bootstrap confidence intervals if requested
        if bootstrap_samples > 0:
            bootstrap_results = self._bootstrap_effect_estimate(
                method, treatment_name, outcome_name, covariate_names,
                is_binary_treatment, bootstrap_samples
            )
            
            estimate.update(bootstrap_results)
        
        # Store the estimate
        estimate_key = f"{treatment_name}_{outcome_name}_{method}"
        self.estimates[estimate_key] = estimate
        
        return estimate
    
    def _estimate_simple_effect(self, 
                              treatment_name: str, 
                              outcome_name: str,
                              is_binary_treatment: bool) -> Dict[str, Any]:
        """
        Estimate effect without any adjustment
        
        Args:
            treatment_name: Name of treatment variable in data
            outcome_name: Name of outcome variable in data
            is_binary_treatment: Whether treatment is binary
            
        Returns:
            Dictionary with effect estimate and details
        """
        if is_binary_treatment:
            # For binary treatment, compare means
            treatment_values = self.data[treatment_name].unique()
            
            if len(treatment_values) == 1:
                return {
                    "status": "error",
                    "message": f"Treatment variable '{treatment_name}' has only one value"
                }
            
            # Determine the treatment and control values
            t1 = treatment_values[0]
            t0 = treatment_values[1]
            
            # Calculate means
            y1 = self.data[self.data[treatment_name] == t1][outcome_name].mean()
            y0 = self.data[self.data[treatment_name] == t0][outcome_name].mean()
            
            # Calculate effect
            effect = y1 - y0
            
            # Calculate standard error
            n1 = len(self.data[self.data[treatment_name] == t1])
            n0 = len(self.data[self.data[treatment_name] == t0])
            
            var1 = self.data[self.data[treatment_name] == t1][outcome_name].var()
            var0 = self.data[self.data[treatment_name] == t0][outcome_name].var()
            
            se = np.sqrt(var1/n1 + var0/n0)
            
            # T-test for significance
            t_stat = effect / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n0 - 2))
            
            # Confidence interval
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se
            
            return {
                "status": "completed",
                "method": "simple",
                "effect": effect,
                "standard_error": se,
                "t_statistic": t_stat,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "treatment_values": {
                    "treatment": t1,
                    "control": t0
                },
                "outcome_means": {
                    "treatment": y1,
                    "control": y0
                },
                "sample_sizes": {
                    "treatment": n1,
                    "control": n0
                }
            }
        else:
            # For continuous treatment, use linear regression
            X = sm.add_constant(self.data[treatment_name])
            y = self.data[outcome_name]
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Save model
            model_key = f"{treatment_name}_{outcome_name}_simple"
            self.models[model_key] = model
            
            # Extract effect (coefficient)
            effect = model.params[treatment_name]
            se = model.bse[treatment_name]
            t_stat = model.tvalues[treatment_name]
            p_value = model.pvalues[treatment_name]
            
            # Confidence interval
            ci_lower, ci_upper = model.conf_int().loc[treatment_name]
            
            return {
                "status": "completed",
                "method": "simple",
                "effect": effect,
                "standard_error": se,
                "t_statistic": t_stat,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "model_summary": model.summary().as_text(),
                "r_squared": model.rsquared
            }
    
    def _estimate_backdoor_effect(self, 
                                treatment_name: str, 
                                outcome_name: str,
                                covariate_names: List[str],
                                is_binary_treatment: bool) -> Dict[str, Any]:
        """
        Estimate effect using backdoor adjustment
        
        Args:
            treatment_name: Name of treatment variable in data
            outcome_name: Name of outcome variable in data
            covariate_names: Names of covariates to adjust for
            is_binary_treatment: Whether treatment is binary
            
        Returns:
            Dictionary with effect estimate and details
        """
        # Filter out any covariates not in the data
        covariate_names = [cov for cov in covariate_names if cov in self.data.columns]
        
        if is_binary_treatment:
            # For binary treatment, use stratification (simplified approach)
            treatment_values = self.data[treatment_name].unique()
            
            if len(treatment_values) != 2:
                return {
                    "status": "error",
                    "message": f"Binary treatment required, but found {len(treatment_values)} values"
                }
            
            # Determine the treatment and control values
            t1 = treatment_values[0]
            t0 = treatment_values[1]
            
            # Check if we have any covariates
            if not covariate_names:
                # No covariates, fall back to simple estimate
                return self._estimate_simple_effect(treatment_name, outcome_name, is_binary_treatment)
            
            # Use regression adjustment
            X = pd.get_dummies(self.data[covariate_names], drop_first=True)
            X = sm.add_constant(X)
            X[treatment_name] = self.data[treatment_name]
            y = self.data[outcome_name]
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Save model
            model_key = f"{treatment_name}_{outcome_name}_backdoor"
            self.models[model_key] = model
            
            # Extract effect (coefficient)
            effect = model.params[treatment_name]
            se = model.bse[treatment_name]
            t_stat = model.tvalues[treatment_name]
            p_value = model.pvalues[treatment_name]
            
            # Confidence interval
            ci_lower, ci_upper = model.conf_int().loc[treatment_name]
            
            return {
                "status": "completed",
                "method": "backdoor",
                "effect": effect,
                "standard_error": se,
                "t_statistic": t_stat,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "covariates": covariate_names,
                "model_summary": model.summary().as_text(),
                "r_squared": model.rsquared
            }
        else:
            # For continuous treatment, use regression adjustment
            # Create design matrix with treatment and covariates
            X = self.data[covariate_names + [treatment_name]].copy()
            
            # Convert categorical covariates to dummies
            for col in covariate_names:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X, dummies], axis=1)
                    X.drop(col, axis=1, inplace=True)
            
            X = sm.add_constant(X)
            y = self.data[outcome_name]
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Save model
            model_key = f"{treatment_name}_{outcome_name}_backdoor"
            self.models[model_key] = model
            
            # Extract effect (coefficient)
            effect = model.params[treatment_name]
            se = model.bse[treatment_name]
            t_stat = model.tvalues[treatment_name]
            p_value = model.pvalues[treatment_name]
            
            # Confidence interval
            ci_lower, ci_upper = model.conf_int().loc[treatment_name]
            
            return {
                "status": "completed",
                "method": "backdoor",
                "effect": effect,
                "standard_error": se,
                "t_statistic": t_stat,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "covariates": covariate_names,
                "model_summary": model.summary().as_text(),
                "r_squared": model.rsquared
            }
    
    def _estimate_frontdoor_effect(self, 
                                 treatment_name: str, 
                                 outcome_name: str,
                                 mediator_names: List[str],
                                 is_binary_treatment: bool) -> Dict[str, Any]:
        """
        Estimate effect using frontdoor adjustment
        
        Args:
            treatment_name: Name of treatment variable in data
            outcome_name: Name of outcome variable in data
            mediator_names: Names of mediator variables
            is_binary_treatment: Whether treatment is binary
            
        Returns:
            Dictionary with effect estimate and details
        """
        if not mediator_names:
            return {
                "status": "error",
                "message": "No mediators provided for frontdoor adjustment"
            }
        
        # Frontdoor adjustment is more complex
        # Step 1: Estimate effect of treatment on mediator(s)
        # Step 2: Estimate effect of mediator(s) on outcome
        # Step 3: Combine the estimates
        
        # For simplicity, we'll implement a version for a single mediator
        if len(mediator_names) == 1:
            mediator_name = mediator_names[0]
            
            # Step 1: Treatment -> Mediator
            X1 = sm.add_constant(self.data[treatment_name])
            y1 = self.data[mediator_name]
            
            model1 = sm.OLS(y1, X1).fit()
            effect1 = model1.params[treatment_name]
            
            # Step 2: Mediator -> Outcome (controlling for treatment)
            X2 = self.data[[treatment_name, mediator_name]].copy()
            X2 = sm.add_constant(X2)
            y2 = self.data[outcome_name]
            
            model2 = sm.OLS(y2, X2).fit()
            effect2 = model2.params[mediator_name]
            
            # Step 3: Combine effects
            effect = effect1 * effect2
            
            # Calculate standard error using delta method (simplified)
            var_effect1 = model1.bse[treatment_name]**2
            var_effect2 = model2.bse[mediator_name]**2
            
            # Approximate standard error
            se = np.sqrt(var_effect1 * effect2**2 + var_effect2 * effect1**2)
            
            # Calculate t-statistic and p-value
            t_stat = effect / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(self.data) - 3))
            
            # Confidence interval
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se
            
            return {
                "status": "completed",
                "method": "frontdoor",
                "effect": effect,
                "standard_error": se,
                "t_statistic": t_stat,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "mediators": mediator_names,
                "effect_on_mediator": effect1,
                "effect_of_mediator": effect2,
                "model1_summary": model1.summary().as_text(),
                "model2_summary": model2.summary().as_text()
            }
        else:
            # Multiple mediators - we'd implement a more complex strategy
            # For now, return a partial implementation note
            return {
                "status": "error",
                "message": "Multiple mediator frontdoor adjustment not fully implemented"
            }
    
    def _estimate_iv_effect(self, 
                          treatment_name: str, 
                          outcome_name: str,
                          instrument_name: str,
                          is_binary_treatment: bool) -> Dict[str, Any]:
        """
        Estimate effect using instrumental variable
        
        Args:
            treatment_name: Name of treatment variable in data
            outcome_name: Name of outcome variable in data
            instrument_name: Name of instrumental variable
            is_binary_treatment: Whether treatment is binary
            
        Returns:
            Dictionary with effect estimate and details
        """
        # Two-stage least squares (2SLS)
        # Stage 1: Instrument -> Treatment
        X1 = sm.add_constant(self.data[instrument_name])
        y1 = self.data[treatment_name]
        
        model1 = sm.OLS(y1, X1).fit()
        
        # Predicted treatment
        treatment_pred = model1.predict(X1)
        
        # Stage 2: Predicted Treatment -> Outcome
        X2 = sm.add_constant(treatment_pred)
        y2 = self.data[outcome_name]
        
        model2 = sm.OLS(y2, X2).fit()
        
        # Extract effect (coefficient of predicted treatment)
        effect = model2.params[1]  # Index 1 for the treatment coefficient
        se = model2.bse[1]
        t_stat = model2.tvalues[1]
        p_value = model2.pvalues[1]
        
        # Confidence interval
        ci_lower = effect - 1.96 * se
        ci_upper = effect + 1.96 * se
        
        # Check first stage strength
        f_stat_first_stage = model1.fvalue
        r2_first_stage = model1.rsquared
        
        # Rule of thumb: F > 10 for strong instrument
        is_strong_instrument = f_stat_first_stage > 10
        
        return {
            "status": "completed",
            "method": "iv",
            "effect": effect,
            "standard_error": se,
            "t_statistic": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "instrument": instrument_name,
            "first_stage_f": f_stat_first_stage,
            "first_stage_r2": r2_first_stage,
            "is_strong_instrument": is_strong_instrument,
            "model1_summary": model1.summary().as_text(),
            "model2_summary": model2.summary().as_text()
        }
    
    def _bootstrap_effect_estimate(self,
                                  method: str,
                                  treatment_name: str,
                                  outcome_name: str,
                                  covariate_names: List[str],
                                  is_binary_treatment: bool,
                                  bootstrap_samples: int) -> Dict[str, Any]:
        """
        Perform bootstrap to estimate confidence intervals
        
        Args:
            method: Estimation method
            treatment_name: Name of treatment variable
            outcome_name: Name of outcome variable
            covariate_names: Names of covariates
            is_binary_treatment: Whether treatment is binary
            bootstrap_samples: Number of bootstrap samples
            
        Returns:
            Dictionary with bootstrap results
        """
        bootstrap_effects = []
        
        n_samples = len(self.data)
        
        # Store original data
        original_data = self.data.copy()
        
        for i in range(bootstrap_samples):
            # Sample with replacement
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = original_data.iloc[bootstrap_indices].copy()
            
            # Save bootstrap data
            self.data = bootstrap_data
            
            # Estimate effect using the specified method
            if method == "simple":
                estimate = self._estimate_simple_effect(treatment_name, outcome_name, is_binary_treatment)
            elif method == "adjustment" or method == "backdoor":
                estimate = self._estimate_backdoor_effect(
                    treatment_name, outcome_name, covariate_names, is_binary_treatment
                )
            elif method == "frontdoor":
                estimate = self._estimate_frontdoor_effect(
                    treatment_name, outcome_name, covariate_names, is_binary_treatment
                )
            elif method == "iv":
                if covariate_names:
                    estimate = self._estimate_iv_effect(
                        treatment_name, outcome_name, covariate_names[0], is_binary_treatment
                    )
                else:
                    continue  # Skip this iteration if no instrument
            
            # Add effect to bootstrap results
            if estimate["status"] == "completed":
                bootstrap_effects.append(estimate["effect"])
        
        # Restore original data
        self.data = original_data
        
        # Calculate bootstrap confidence intervals
        bootstrap_effects.sort()
        
        # 95% confidence interval
        lower_idx = int(0.025 * bootstrap_samples)
        upper_idx = int(0.975 * bootstrap_samples)
        
        # If indices are out of bounds, adjust them
        lower_idx = max(0, min(lower_idx, bootstrap_samples - 1))
        upper_idx = max(0, min(upper_idx, bootstrap_samples - 1))
        
        bootstrap_ci_lower = bootstrap_effects[lower_idx]
        bootstrap_ci_upper = bootstrap_effects[upper_idx]
        
        # Calculate bootstrap standard error
        bootstrap_se = np.std(bootstrap_effects)
        
        return {
            "bootstrap_ci_lower": bootstrap_ci_lower,
            "bootstrap_ci_upper": bootstrap_ci_upper,
            "bootstrap_se": bootstrap_se,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_effects": bootstrap_effects
        }
    
    def estimate_conditional_effects(self,
                                   treatment: Any,
                                   outcome: Any,
                                   method: str = "adjustment",
                                   covariates: Optional[List[Any]] = None,
                                   interaction_vars: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Estimate conditional treatment effects (heterogeneous effects)
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            method: Estimation method
            covariates: Optional covariates to adjust for
            interaction_vars: Variables to interact with treatment
            
        Returns:
            Dictionary with conditional effect estimates
        """
        # Get variable names
        treatment_name = self._get_node_name(treatment)
        outcome_name = self._get_node_name(outcome)
        
        # Convert covariates to names
        covariate_names = []
        if covariates:
            covariate_names = [self._get_node_name(cov) for cov in covariates]
        
        # Convert interaction variables to names
        interaction_names = []
        if interaction_vars:
            interaction_names = [self._get_node_name(var) for var in interaction_vars]
        
        # If no interaction variables specified, return average effect
        if not interaction_names:
            return self.estimate_average_treatment_effect(
                treatment, outcome, method, covariates
            )
        
        # Check if the treatment and interaction variables exist in the data
        if treatment_name not in self.data.columns:
            return {
                "status": "error",
                "message": f"Treatment variable '{treatment_name}' not found in data"
            }
        
        if outcome_name not in self.data.columns:
            return {
                "status": "error",
                "message": f"Outcome variable '{outcome_name}' not found in data"
            }
        
        for var in interaction_names:
            if var not in self.data.columns:
                return {
                    "status": "error",
                    "message": f"Interaction variable '{var}' not found in data"
                }
        
        # For now, implement a simple interaction model
        # This could be extended with more sophisticated methods
        
        # Create design matrix
        X = self.data[covariate_names + [treatment_name] + interaction_names].copy()
        
        # Create interaction terms
        for var in interaction_names:
            interaction_col = f"{treatment_name}_{var}"
            X[interaction_col] = X[treatment_name] * X[var]
        
        X = sm.add_constant(X)
        y = self.data[outcome_name]
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Extract main effect
        main_effect = model.params[treatment_name]
        main_effect_se = model.bse[treatment_name]
        
        # Extract interaction effects
        interaction_effects = {}
        
        for var in interaction_names:
            interaction_col = f"{treatment_name}_{var}"
            if interaction_col in model.params:
                effect = model.params[interaction_col]
                se = model.bse[interaction_col]
                t_stat = model.tvalues[interaction_col]
                p_value = model.pvalues[interaction_col]
                
                interaction_effects[var] = {
                    "effect": effect,
                    "se": se,
                    "t_stat": t_stat,
                    "p_value": p_value
                }
        
        # Save model
        model_key = f"{treatment_name}_{outcome_name}_conditional"
        self.models[model_key] = model
        
        return {
            "status": "completed",
            "method": "conditional",
            "main_effect": main_effect,
            "main_effect_se": main_effect_se,
            "interaction_effects": interaction_effects,
            "interaction_vars": interaction_names,
            "covariates": covariate_names,
            "model_summary": model.summary().as_text(),
            "r_squared": model.rsquared
        }
    
    def conduct_sensitivity_analysis(self,
                                   treatment: Any,
                                   outcome: Any,
                                   method: str = "adjustment",
                                   covariates: Optional[List[Any]] = None,
                                   bounds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Conduct sensitivity analysis for unobserved confounding
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            method: Estimation method
            covariates: Optional covariates to adjust for
            bounds: Optional parameter bounds for sensitivity
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        # Simple implementation of sensitivity analysis
        # For a more comprehensive implementation, consider using specialized packages
        
        # Get variable names
        treatment_name = self._get_node_name(treatment)
        outcome_name = self._get_node_name(outcome)
        
        # Default bounds if not provided
        if bounds is None:
            bounds = {
                "r2_confounder_treatment": [0.0, 0.3, 0.05],  # [min, max, step]
                "r2_confounder_outcome": [0.0, 0.3, 0.05]
            }
        
        # Get base estimate
        base_estimate = self.estimate_average_treatment_effect(
            treatment, outcome, method, covariates
        )
        
        if base_estimate["status"] != "completed":
            return base_estimate
        
        # Get parameter ranges
        r2_tx_range = np.arange(
            bounds["r2_confounder_treatment"][0],
            bounds["r2_confounder_treatment"][1] + 0.001,
            bounds["r2_confounder_treatment"][2]
        )
        
        r2_y_range = np.arange(
            bounds["r2_confounder_outcome"][0],
            bounds["r2_confounder_outcome"][1] + 0.001,
            bounds["r2_confounder_outcome"][2]
        )
        
        # Create grid for sensitivity analysis
        grid = []
        
        for r2_tx in r2_tx_range:
            for r2_y in r2_y_range:
                # Skip zero-zero point (no confounding)
                if r2_tx == 0 and r2_y == 0:
                    continue
                
                # Calculate adjusted effect (using Cinelli & Hazlett's omitted variable bias formula)
                # This is a simplified version
                base_effect = base_estimate["effect"]
                
                # Simplistic adjustment (proper implementation would be more complex)
                # The key idea is that stronger correlation with treatment and outcome
                # leads to more bias
                adjusted_effect = base_effect * (1 - np.sqrt(r2_tx * r2_y))
                
                grid.append({
                    "r2_tx": r2_tx,
                    "r2_y": r2_y,
                    "adjusted_effect": adjusted_effect,
                    "bias": adjusted_effect - base_effect
                })
        
        # Robustness value: minimum r2 product needed to change sign
        min_r2_product_sign_change = float('inf')
        
        for entry in grid:
            r2_product = entry["r2_tx"] * entry["r2_y"]
            adjusted_effect = entry["adjusted_effect"]
            
            # Check if sign changes
            if base_effect * adjusted_effect <= 0:
                if r2_product < min_r2_product_sign_change:
                    min_r2_product_sign_change = r2_product
        
        # If no sign change found, set to None
        if min_r2_product_sign_change == float('inf'):
            min_r2_product_sign_change = None
        
        return {
            "status": "completed",
            "base_estimate": base_estimate,
            "sensitivity_grid": grid,
            "min_r2_product_sign_change": min_r2_product_sign_change,
            "bounds": bounds,
            "method": "ols_sensitivity"
        }
    
    def visualize_effect(self, effect_result: Dict[str, Any]) -> go.Figure:
        """
        Visualize causal effect estimate
        
        Args:
            effect_result: The result from an effect estimation method
            
        Returns:
            Plotly figure with effect visualization
        """
        if effect_result["status"] != "completed":
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=effect_result.get("message", "Error estimating effect"),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Create figure based on estimation method
        method = effect_result.get("method", "unknown")
        
        if method in ["simple", "backdoor", "frontdoor", "iv"]:
            # Create basic effect visualization
            fig = go.Figure()
            
            # Add effect with confidence interval
            effect = effect_result["effect"]
            ci_lower = effect_result.get("ci_lower", effect - 1.96 * effect_result.get("standard_error", 0))
            ci_upper = effect_result.get("ci_upper", effect + 1.96 * effect_result.get("standard_error", 0))
            
            # Add effect and CI
            fig.add_trace(go.Bar(
                x=["Causal Effect"],
                y=[effect],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[ci_upper - effect],
                    arrayminus=[effect - ci_lower]
                ),
                marker_color='blue',
                width=0.4
            ))
            
            # Add reference line at zero
            fig.add_shape(
                type="line",
                x0=-0.5, y0=0,
                x1=0.5, y1=0,
                line=dict(
                    color="red",
                    width=2,
                    dash="dot"
                )
            )
            
            # Add annotations
            fig.add_annotation(
                x="Causal Effect",
                y=effect,
                text=f"{effect:.3f}",
                showarrow=False,
                yshift=10,
                font=dict(size=14)
            )
            
            # Add p-value annotation
            if "p_value" in effect_result:
                p_value = effect_result["p_value"]
                sig_text = f"p = {p_value:.3f}" + ("*" if p_value < 0.05 else "")
                
                fig.add_annotation(
                    x="Causal Effect",
                    y=effect,
                    text=sig_text,
                    showarrow=False,
                    yshift=-20,
                    font=dict(size=12)
                )
            
            # Update layout
            title = f"Estimated Causal Effect ({method.capitalize()} Method)"
            
            fig.update_layout(
                title=title,
                xaxis_title="",
                yaxis_title="Effect Size",
                xaxis=dict(
                    tickangle=0,
                    tickfont=dict(size=14)
                ),
                width=500,
                height=400
            )
            
            return fig
            
        elif method == "conditional":
            # Create visualization for conditional effects
            fig = go.Figure()
            
            # Get main effect and interaction effects
            main_effect = effect_result["main_effect"]
            interaction_effects = effect_result.get("interaction_effects", {})
            
            # Bar for main effect
            fig.add_trace(go.Bar(
                x=["Main Effect"],
                y=[main_effect],
                marker_color='blue',
                name="Main Effect"
            ))
            
            # Bars for interaction effects
            interaction_x = []
            interaction_y = []
            
            for var, effect_data in interaction_effects.items():
                interaction_x.append(f"Interaction: {var}")
                interaction_y.append(effect_data["effect"])
            
            if interaction_x:
                fig.add_trace(go.Bar(
                    x=interaction_x,
                    y=interaction_y,
                    marker_color='orange',
                    name="Interaction Effects"
                ))
            
            # Add reference line at zero
            fig.add_shape(
                type="line",
                x0=-0.5, y0=0,
                x1=len(interaction_effects) + 0.5, y1=0,
                line=dict(
                    color="red",
                    width=2,
                    dash="dot"
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Conditional Treatment Effects",
                xaxis_title="",
                yaxis_title="Effect Size",
                xaxis=dict(
                    tickangle=45,
                    tickfont=dict(size=12)
                ),
                width=max(500, 100 * (len(interaction_effects) + 1)),
                height=500
            )
            
            return fig
            
        else:
            # Generic visualization
            fig = go.Figure()
            
            # Add effect
            fig.add_trace(go.Bar(
                x=["Causal Effect"],
                y=[effect_result["effect"]],
                marker_color='blue'
            ))
            
            # Add reference line at zero
            fig.add_shape(
                type="line",
                x0=-0.5, y0=0,
                x1=0.5, y1=0,
                line=dict(
                    color="red",
                    width=2,
                    dash="dot"
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Estimated Causal Effect ({method.capitalize()})",
                xaxis_title="",
                yaxis_title="Effect Size"
            )
            
            return fig
    
    def visualize_sensitivity(self, sensitivity_result: Dict[str, Any]) -> go.Figure:
        """
        Visualize sensitivity analysis results
        
        Args:
            sensitivity_result: Result from sensitivity analysis
            
        Returns:
            Plotly figure with sensitivity visualization
        """
        if sensitivity_result["status"] != "completed":
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=sensitivity_result.get("message", "Error in sensitivity analysis"),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Extract sensitivity grid
        grid = sensitivity_result["sensitivity_grid"]
        
        if not grid:
            # No grid points
            fig = go.Figure()
            fig.add_annotation(
                text="No sensitivity analysis results available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Create heatmap data
        r2_tx_values = sorted(list(set([entry["r2_tx"] for entry in grid])))
        r2_y_values = sorted(list(set([entry["r2_y"] for entry in grid])))
        
        # Create empty matrix for heatmap
        z = np.zeros((len(r2_y_values), len(r2_tx_values)))
        
        # Fill the matrix with adjusted effects
        for i, r2_y in enumerate(r2_y_values):
            for j, r2_tx in enumerate(r2_tx_values):
                # Find the grid entry
                for entry in grid:
                    if entry["r2_tx"] == r2_tx and entry["r2_y"] == r2_y:
                        z[i, j] = entry["adjusted_effect"]
                        break
        
        # Get base effect for reference
        base_effect = sensitivity_result["base_estimate"]["effect"]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=r2_tx_values,
            y=r2_y_values,
            colorscale='RdBu',
            zmid=0,  # Center colorscale at zero
            colorbar=dict(
                title="Adjusted Effect",
                titleside="right"
            ),
            hovertemplate=(
                "R² Treatment-Confounder: %{x}<br>" +
                "R² Outcome-Confounder: %{y}<br>" +
                "Adjusted Effect: %{z:.3f}<br>" +
                "Bias: %{z:.3f} - " + str(base_effect) + " = %{text:.3f}"
            ),
            text=[[z_val - base_effect for z_val in z_row] for z_row in z]
        ))
        
        # Add contour line at zero effect
        if np.min(z) < 0 and np.max(z) > 0:
            fig.add_contour(
                z=z,
                x=r2_tx_values,
                y=r2_y_values,
                contours=dict(
                    coloring='none',
                    showlabels=True,
                    start=0,
                    end=0,
                    size=0
                ),
                line=dict(color='black', width=2),
                showscale=False
            )
        
        # Update layout
        fig.update_layout(
            title="Sensitivity Analysis for Unobserved Confounding",
            xaxis_title="R² Treatment-Confounder",
            yaxis_title="R² Outcome-Confounder",
            xaxis=dict(
                tickfont=dict(size=10),
                tickformat=".2f"
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                tickformat=".2f"
            ),
            width=700,
            height=600
        )
        
        return fig


def render_effect_estimator(graph: nx.DiGraph, data: pd.DataFrame):
    """
    Render the effect estimator UI in Streamlit
    
    Args:
        graph: Causal graph
        data: Dataset
    """
    st.subheader("Causal Effect Estimator")
    
    # Create estimator
    estimator = EffectEstimator(graph, data)
    
    # Get node names for selection
    node_names = {}
    for node in graph.nodes():
        if "name" in graph.nodes[node]:
            node_names[node] = graph.nodes[node]["name"]
        else:
            node_names[node] = str(node)
    
    # Select treatment and outcome
    col1, col2 = st.columns(2)
    
    with col1:
        treatment_node = st.selectbox(
            "Treatment Variable",
            options=list(graph.nodes()),
            format_func=lambda x: node_names.get(x, str(x))
        )
    
    with col2:
        outcome_node = st.selectbox(
            "Outcome Variable",
            options=list(graph.nodes()),
            format_func=lambda x: node_names.get(x, str(x))
        )
    
    # Select estimation method
    method = st.selectbox(
        "Estimation Method",
        options=["adjustment", "backdoor", "frontdoor", "iv", "simple"],
        index=0,
        help=("adjustment/backdoor: Control for confounders | " 
              "frontdoor: Use mediator variables | " 
              "iv: Instrumental variables | " 
              "simple: No adjustment")
    )
    
    # Automatically find adjustment variables
    st.markdown("### Adjustment Variables")
    
    auto_select = st.checkbox("Auto-select adjustment variables", value=True)
    
    if auto_select:
        if method == "backdoor" or method == "adjustment":
            adjustment_nodes = estimator._find_backdoor_adjustment_set(treatment_node, outcome_node)
            st.write("Backdoor adjustment set:")
        elif method == "frontdoor":
            adjustment_nodes = estimator._find_frontdoor_adjustment_set(treatment_node, outcome_node)
            st.write("Frontdoor adjustment set:")
        elif method == "iv":
            adjustment_nodes = estimator._find_instrumental_variables(treatment_node, outcome_node)
            st.write("Potential instrumental variables:")
        else:
            adjustment_nodes = []
        
        if adjustment_nodes:
            adjustment_names = [node_names.get(node, str(node)) for node in adjustment_nodes]
            st.write(", ".join(adjustment_names))
        else:
            st.write("No suitable adjustment variables found automatically.")
    else:
        # Manual selection
        remaining_nodes = [node for node in graph.nodes() 
                          if node != treatment_node and node != outcome_node]
        
        adjustment_nodes = st.multiselect(
            "Select adjustment variables",
            options=remaining_nodes,
            format_func=lambda x: node_names.get(x, str(x))
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        bootstrap = st.checkbox("Bootstrap confidence intervals", value=False)
        bootstrap_samples = st.number_input(
            "Bootstrap samples",
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
            disabled=not bootstrap
        )
        
        sensitivity = st.checkbox("Perform sensitivity analysis", value=False)
    
    # Estimate button
    if st.button("Estimate Causal Effect"):
        with st.spinner("Estimating causal effect..."):
            # Estimate effect
            if bootstrap:
                effect = estimator.estimate_average_treatment_effect(
                    treatment_node,
                    outcome_node,
                    method=method,
                    covariates=adjustment_nodes,
                    bootstrap_samples=bootstrap_samples
                )
            else:
                effect = estimator.estimate_average_treatment_effect(
                    treatment_node,
                    outcome_node,
                    method=method,
                    covariates=adjustment_nodes
                )
            
            if effect["status"] == "completed":
                # Show effect visualization
                st.subheader("Causal Effect Results")
                
                # Display effect size
                effect_size = effect["effect"]
                p_value = effect.get("p_value", float('nan'))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Effect Size",
                        f"{effect_size:.4f}",
                        delta=None,
                        help="Estimated causal effect"
                    )
                
                with col2:
                    # Show significance
                    is_significant = p_value < 0.05 if not np.isnan(p_value) else False
                    sig_text = "✓ Significant" if is_significant else "✗ Not significant"
                    sig_delta = "p < 0.05" if is_significant else f"p = {p_value:.4f}"
                    
                    st.metric(
                        "Statistical Significance",
                        sig_text,
                        delta=sig_delta,
                        delta_color="normal",
                        help="Statistical significance at p < 0.05"
                    )
                
                with col3:
                    # Show confidence interval
                    ci_lower = effect.get("ci_lower", effect_size - 1.96 * effect.get("standard_error", 0))
                    ci_upper = effect.get("ci_upper", effect_size + 1.96 * effect.get("standard_error", 0))
                    
                    ci_contains_zero = ci_lower <= 0 <= ci_upper
                    ci_text = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                    ci_delta = "Contains zero" if ci_contains_zero else "Does not contain zero"
                    
                    st.metric(
                        "95% Confidence Interval",
                        ci_text,
                        delta=ci_delta,
                        delta_color="off" if ci_contains_zero else "normal",
                        help="95% confidence interval for the effect"
                    )
                
                # Show effect visualization
                fig = estimator.visualize_effect(effect)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show bootstrap results if available
                if bootstrap and "bootstrap_ci_lower" in effect:
                    st.subheader("Bootstrap Results")
                    
                    bootstrap_ci_lower = effect["bootstrap_ci_lower"]
                    bootstrap_ci_upper = effect["bootstrap_ci_upper"]
                    bootstrap_se = effect["bootstrap_se"]
                    
                    st.markdown(f"**Bootstrap 95% CI:** [{bootstrap_ci_lower:.4f}, {bootstrap_ci_upper:.4f}]")
                    st.markdown(f"**Bootstrap SE:** {bootstrap_se:.4f}")
                    
                    # Show bootstrap distribution
                    if "bootstrap_effects" in effect:
                        bootstrap_effects = effect["bootstrap_effects"]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=bootstrap_effects,
                            nbinsx=30,
                            marker_color='blue',
                            opacity=0.7
                        ))
                        
                        # Add vertical line for the point estimate
                        fig.add_shape(
                            type="line",
                            x0=effect_size, y0=0,
                            x1=effect_size, y1=1,
                            yref="paper",
                            line=dict(
                                color="red",
                                width=2,
                                dash="solid"
                            )
                        )
                        
                        # Add vertical lines for bootstrap CI
                        fig.add_shape(
                            type="line",
                            x0=bootstrap_ci_lower, y0=0,
                            x1=bootstrap_ci_lower, y1=0.9,
                            yref="paper",
                            line=dict(
                                color="green",
                                width=2,
                                dash="dash"
                            )
                        )
                        
                        fig.add_shape(
                            type="line",
                            x0=bootstrap_ci_upper, y0=0,
                            x1=bootstrap_ci_upper, y1=0.9,
                            yref="paper",
                            line=dict(
                                color="green",
                                width=2,
                                dash="dash"
                            )
                        )
                        
                        # Add annotation for the point estimate
                        fig.add_annotation(
                            x=effect_size,
                            y=1,
                            text="Point Estimate",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40,
                            yref="paper"
                        )
                        
                        fig.update_layout(
                            title="Bootstrap Distribution of Effect Estimates",
                            xaxis_title="Effect Size",
                            yaxis_title="Frequency",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Show sensitivity analysis if requested
                if sensitivity:
                    st.subheader("Sensitivity Analysis")
                    
                    with st.spinner("Performing sensitivity analysis..."):
                        # Perform sensitivity analysis
                        sensitivity_result = estimator.conduct_sensitivity_analysis(
                            treatment_node,
                            outcome_node,
                            method=method,
                            covariates=adjustment_nodes
                        )
                        
                        if sensitivity_result["status"] == "completed":
                            # Show robustness value
                            min_r2 = sensitivity_result.get("min_r2_product_sign_change", None)
                            
                            if min_r2 is not None:
                                st.markdown(f"**Robustness Value (RV):** {min_r2:.4f}")
                                st.markdown("The minimum strength of confounding needed to change the sign of the effect.")
                            else:
                                st.markdown("**Robustness Value (RV):** The effect estimate is robust to the range of confounding strengths tested.")
                            
                            # Show sensitivity plot
                            fig = estimator.visualize_sensitivity(sensitivity_result)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interpretation
                            st.markdown("### Interpretation")
                            
                            if min_r2 is not None:
                                if min_r2 < 0.1:
                                    st.warning("The effect estimate is sensitive to relatively weak unobserved confounding.")
                                elif min_r2 < 0.2:
                                    st.info("The effect estimate is moderately robust to unobserved confounding.")
                                else:
                                    st.success("The effect estimate is robust to moderate levels of unobserved confounding.")
                            else:
                                st.success("The effect estimate is highly robust to unobserved confounding within the tested range.")
                        else:
                            st.error(f"Error in sensitivity analysis: {sensitivity_result.get('message', 'Unknown error')}")
                
                # Show method details
                with st.expander("Method Details"):
                    st.markdown(f"**Method:** {method}")
                    
                    if "covariates" in effect and effect["covariates"]:
                        st.markdown(f"**Adjustment Variables:** {', '.join(effect['covariates'])}")
                    
                    if "standard_error" in effect:
                        st.markdown(f"**Standard Error:** {effect['standard_error']:.4f}")
                    
                    if "t_statistic" in effect:
                        st.markdown(f"**t-statistic:** {effect['t_statistic']:.4f}")
                    
                    if "model_summary" in effect:
                        st.text(effect["model_summary"])
            else:
                st.error(f"Error estimating effect: {effect.get('message', 'Unknown error')}")


def estimate_causal_effect(graph: nx.DiGraph, 
                         data: pd.DataFrame,
                         treatment: Any, 
                         outcome: Any,
                         method: str = "adjustment",
                         covariates: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Estimate causal effect between treatment and outcome
    
    Args:
        graph: The causal graph
        data: The dataset
        treatment: Treatment variable
        outcome: Outcome variable
        method: Estimation method
        covariates: Optional adjustment variables
        
    Returns:
        Dictionary with effect estimate and details
    """
    estimator = EffectEstimator(graph, data)
    return estimator.estimate_average_treatment_effect(
        treatment, outcome, method, covariates
    )