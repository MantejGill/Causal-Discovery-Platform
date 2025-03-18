import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Any, Optional

class LLMAlgorithmRecommender:
    """
    Uses LLMs to provide algorithm recommendations with detailed explanations
    based on data characteristics.
    """
    
    def __init__(self, llm_adapter):
        """
        Initialize the LLM Algorithm Recommender.
        
        Args:
            llm_adapter: The configured LLM adapter (OpenAI or OpenRouter)
        """
        self.llm_adapter = llm_adapter
        
    def get_algorithm_recommendations(self, 
                                    data_profile: Dict[str, Any],
                                    judgments: Dict[str, bool]) -> Dict[str, Any]:
        """
        Use LLM to recommend algorithms with detailed explanations.
        
        Args:
            data_profile: The profile of the dataset with characteristics
            judgments: User judgments about the data properties
            
        Returns:
            Dictionary with recommendations and explanations
        """
        if not self.llm_adapter:
            return {"error": "LLM adapter not configured or unavailable"}
            
        # Create a prompt for the LLM with data characteristics
        prompt = self._create_recommendation_prompt(data_profile, judgments=None)
        
        try:
            # Call LLM with structured output format
            system_prompt = """You are an expert in causal discovery algorithms. Your task is to recommend 
            appropriate causal discovery algorithms based on dataset characteristics and provide 
            detailed explanations for each recommendation. Provide your response in this format:

            PRIMARY RECOMMENDATIONS:
            - algorithm_name: explanation why this is recommended
            - algorithm_name: explanation why this is recommended

            SECONDARY RECOMMENDATIONS:
            - algorithm_name: explanation why this could be a secondary choice
            - algorithm_name: explanation why this could be a secondary choice

            NOT RECOMMENDED:
            - algorithm_name: explanation why this is not recommended
            - algorithm_name: explanation why this is not recommended
            """
            
            response = self.llm_adapter.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )

            print(response)
            print(prompt)
            
            # Extract and parse the response
            return self._parse_llm_response(response.get("completion", ""))
            
        except Exception as e:
            return {
                "error": f"Error generating recommendations: {str(e)}",
                "primary": [],
                "secondary": [],
                "not_recommended": []
            }
    
    def _create_recommendation_prompt(self, 
                                     data_profile: Dict[str, Any],
                                     judgments: Dict[str, bool]) -> str:
        """
        Create a detailed prompt for the LLM based on data profile and judgments.
        
        Args:
            data_profile: The profile of the dataset
            judgments: User judgments about the data
            
        Returns:
            Formatted prompt string
        """
        # Extract key information from the data profile
        n_samples = data_profile.get("n_samples", 0)
        n_features = data_profile.get("n_features", 0)
        overall_type = data_profile.get("overall_type", "unknown")
        overall_distribution = data_profile.get("overall_distribution", "unknown")
        has_missing_values = data_profile.get("has_missing_values", False)
        
        # Format all judgments
        if judgments:
            judgments_text = "\n".join([f"- {key}: {value}" for key, value in judgments.items()])
        else:
            judgments_text = "No user judgments provided."
        
        # Create the prompt
        prompt = f"""
        # Dataset Characteristics for Causal Discovery Algorithm Recommendation
        
        ## Basic Dataset Information
        - Number of samples (rows): {n_samples}
        - Number of features (columns): {n_features}
        - Overall data type: {overall_type}
        - Overall distribution: {overall_distribution}
        - Has missing values: {has_missing_values}
        
        ## User Judgments About the Data
        {judgments_text}
        
        ## Type Counts
        {self._format_dict(data_profile.get("type_counts", {}))}
        
        ## Available Algorithms
        
        ### Constraint-based Methods
        - pc_fisherz: PC algorithm with Fisher's Z test (for continuous, Gaussian data)
        - pc_chisq: PC algorithm with Chi-square test (for discrete data)
        - pc_gsq: PC algorithm with G-square test (for discrete data)
        - pc_kci: PC algorithm with Kernel CI test (for nonlinear dependencies)
        - fci_fisherz: FCI algorithm with Fisher's Z test (for latent confounders, continuous data)
        - fci_chisq: FCI algorithm with Chi-square test (for latent confounders, discrete data)
        - fci_kci: FCI algorithm with Kernel CI test (for latent confounders, nonlinear)
        - cdnod: CD-NOD algorithm for heterogeneous/nonstationary data
        
        ### Score-based Methods
        - ges_bic: GES algorithm with BIC score (for continuous, Gaussian data)
        - ges_bdeu: GES algorithm with BDeu score (for discrete data)
        - ges_cv: GES algorithm with CV score (for nonlinear relationships)
        - grasp: GRaSP algorithm (permutation-based)
        - boss: BOSS algorithm (permutation-based)
        - exact_dp: Exact search with dynamic programming
        - exact_astar: Exact search with A* algorithm
        
        ### FCM-based Methods
        - lingam_ica: ICA-based LiNGAM (for linear non-Gaussian acyclic models)
        - lingam_direct: DirectLiNGAM (for linear non-Gaussian acyclic models)
        - lingam_var: VAR-LiNGAM (for time series data)
        - lingam_rcd: RCD (for linear non-Gaussian with latent confounders)
        - lingam_camuv: CAM-UV (for causal additive models with unobserved variables)
        - anm: Additive Noise Model (for nonlinear relationships)
        - pnl: Post-Nonlinear causal model (for nonlinear relationships)
        
        ### Hidden Causal Methods
        - gin: GIN (for linear non-Gaussian latent variable models)
        
        ### Granger Causality
        - granger_test: Linear Granger causality test (for time series, 2 variables)
        - granger_lasso: Linear Granger causality with Lasso (for multivariate time series)
        
        ## Task
        Based on the dataset characteristics and user judgments provided above, recommend appropriate causal discovery algorithms.
        
        1. Provide PRIMARY recommendations - algorithms most suitable for this dataset
        2. Provide SECONDARY recommendations - algorithms that could work but may not be optimal
        3. Provide NOT RECOMMENDED algorithms - algorithms that are likely inappropriate for this dataset
        
        For each algorithm, provide a detailed explanation of why it is recommended or not recommended.
        
        Return your response in the following JSON format:
        ```json
        {{
          "primary": [
            {{"algorithm": "algorithm_name", "reason": "detailed explanation of why this algorithm is recommended"}}
          ],
          "secondary": [
            {{"algorithm": "algorithm_name", "reason": "detailed explanation of why this could be a secondary choice"}}
          ],
          "not_recommended": [
            {{"algorithm": "algorithm_name", "reason": "detailed explanation of why this algorithm is not recommended"}}
          ]
        }}
        ```
        
        Focus on being thorough in your explanations, considering all relevant characteristics and judgments.
        """
        
        return prompt
    
    def _format_dict(self, d: Dict) -> str:
        """Format a dictionary as a string with one item per line"""
        return "\n".join([f"- {key}: {value}" for key, value in d.items()])
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM's response to extract structured recommendations.
        
        Args:
            response_text: The raw text response from the LLM
            
        Returns:
            Dictionary with parsed recommendations
        """
        # Default structure if parsing fails
        default_result = {
            "primary": [],
            "secondary": [],
            "not_recommended": [],
            "raw_response": response_text
        }
        
        try:
            # Try to extract JSON from the response
            json_str = response_text
            
            # If response is wrapped in ```json ... ``` or ``` ... ```, extract just the JSON part
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse the JSON
            parsed = json.loads(json_str)
            
            # Ensure the required keys exist
            result = {
                "primary": parsed.get("primary", []),
                "secondary": parsed.get("secondary", []),
                "not_recommended": parsed.get("not_recommended", []),
                "raw_response": response_text
            }
            
            return result
            
        except Exception as e:
            # If JSON parsing fails, try to extract recommendations in a more forgiving way
            try:
                # Attempt to find sections in the text response
                result = default_result.copy()
                
                if "PRIMARY" in response_text or "Primary" in response_text:
                    primary_section = self._extract_section(response_text, ["PRIMARY", "Primary"], ["SECONDARY", "Secondary", "NOT RECOMMENDED", "Not recommended"])
                    result["primary"] = self._extract_algorithms_from_text(primary_section)
                
                if "SECONDARY" in response_text or "Secondary" in response_text:
                    secondary_section = self._extract_section(response_text, ["SECONDARY", "Secondary"], ["NOT RECOMMENDED", "Not recommended"])
                    result["secondary"] = self._extract_algorithms_from_text(secondary_section)
                
                if "NOT RECOMMENDED" in response_text or "Not recommended" in response_text:
                    not_recommended_section = self._extract_section(response_text, ["NOT RECOMMENDED", "Not recommended"], [])
                    result["not_recommended"] = self._extract_algorithms_from_text(not_recommended_section)
                
                return result
                
            except Exception as nested_e:
                # Return the default structure if all parsing attempts fail
                default_result["error"] = f"Error parsing LLM response: {str(nested_e)}"
                return default_result
    
    def _extract_section(self, text: str, section_starters: List[str], section_enders: List[str]) -> str:
        """Extract a section of text between any of the starters and any of the enders"""
        # Find the start of the section
        start_pos = len(text)
        for starter in section_starters:
            pos = text.find(starter)
            if pos != -1 and pos < start_pos:
                start_pos = pos
        
        if start_pos == len(text):
            return ""
        
        # Find the end of the section
        end_pos = len(text)
        for ender in section_enders:
            pos = text.find(ender, start_pos + 1)
            if pos != -1 and pos < end_pos:
                end_pos = pos
        
        # Extract the section
        return text[start_pos:end_pos].strip()
    
    def _extract_algorithms_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Extract algorithm names and reasons from text when JSON parsing fails.
        This is a best-effort extraction for robustness.
        """
        algorithms = []
        
        # Split by newlines and look for algorithm names
        lines = text.split('\n')
        current_algorithm = ""
        current_reason = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains an algorithm name
            algorithm_match = False
            for alg in ["pc_", "fci_", "ges_", "lingam_", "anm", "pnl", "gin", "granger_", "boss", "grasp", "exact_", "cdnod"]:
                if alg in line.lower():
                    # If we already have an algorithm, save it before starting a new one
                    if current_algorithm:
                        algorithms.append({"algorithm": current_algorithm, "reason": current_reason.strip()})
                    
                    # Extract the algorithm name - everything up to a colon or space after the matched text
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        current_algorithm = parts[0].strip()
                        current_reason = parts[1].strip()
                    else:
                        current_algorithm = alg
                        current_reason = line
                    
                    algorithm_match = True
                    break
            
            # If this line doesn't contain a new algorithm, append to the current reason
            if not algorithm_match and current_algorithm:
                current_reason += " " + line
        
        # Add the last algorithm if there is one
        if current_algorithm:
            algorithms.append({"algorithm": current_algorithm, "reason": current_reason.strip()})
        
        return algorithms

def display_llm_recommendations(llm_adapter, data_profile, judgments):
    """
    Display LLM-based algorithm recommendations in the Streamlit UI.
    
    Args:
        llm_adapter: The configured LLM adapter
        data_profile: The dataset profile
        judgments: User judgments about the data
    """
    # Only attempt if LLM adapter is available
    if not llm_adapter:
        st.warning("LLM adapter not available. Please configure LLM in Settings page.")
        return
    
    # Create the recommender
    recommender = LLMAlgorithmRecommender(llm_adapter)
    
    # Get recommendations with explanations
    with st.spinner("Generating LLM-based algorithm recommendations..."):
        recommendations = recommender.get_algorithm_recommendations(data_profile, judgments)
    
    # Check for errors
    if "error" in recommendations:
        st.error(f"Error generating recommendations: {recommendations['error']}")
        return
    
    # Display primary recommendations
    if recommendations["primary"]:
        st.markdown("### Primary Recommendations")
        for rec in recommendations["primary"]:
            with st.expander(f"**{rec['algorithm']}**"):
                st.markdown(f"**Why recommended:** {rec['reason']}")
    else:
        st.info("No primary recommendations from LLM.")
    
    # Display secondary recommendations
    if recommendations["secondary"]:
        st.markdown("### Secondary Recommendations")
        for rec in recommendations["secondary"]:
            with st.expander(f"**{rec['algorithm']}**"):
                st.markdown(f"**Why considered:** {rec['reason']}")
    else:
        st.info("No secondary recommendations from LLM.")
    
    # Display not recommended
    if recommendations["not_recommended"]:
        st.markdown("### Not Recommended")
        for rec in recommendations["not_recommended"]:
            with st.expander(f"**{rec['algorithm']}**"):
                st.markdown(f"**Why not recommended:** {rec['reason']}")
    
    # Add a note about the source of recommendations
    st.info("These recommendations are generated by an LLM and should be used as guidance. The LLM analyzes your data characteristics and judgments to provide tailored explanations.")