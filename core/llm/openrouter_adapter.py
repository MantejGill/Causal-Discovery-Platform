import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
import requests
from openai import OpenAI

from core.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)

class OpenRouterAdapter(LLMAdapter):
    """
    Adapter for OpenRouter API, which provides access to multiple models including DeepSeek R1.
    OpenRouter exposes an OpenAI-compatible API, making the integration straightforward.
    """
    
    def __init__(
        self, 
        api_key: str,
        model: str = "deepseek/deepseek-r1:free",
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "https://causal-discovery-framework.app",
        site_name: str = "LLM-Augmented Causal Discovery Framework",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        """
        Initialize the OpenRouter adapter.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier for OpenRouter (default is DeepSeek R1)
            base_url: OpenRouter API base URL
            site_url: Your application URL for attribution in OpenRouter
            site_name: Your application name for attribution in OpenRouter
            temperature: Sampling temperature (higher is more random)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for token frequency
            presence_penalty: Penalty for token presence
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.site_url = site_url
        self.site_name = site_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
        # Initialize OpenAI client for OpenRouter
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def get_name(self) -> str:
        """Get the name of the LLM adapter."""
        return f"OpenRouter ({self.model})"
    
    def get_supported_models(self) -> List[str]:
        """
        Get a list of supported models from OpenRouter.
        This method fetches the available models from OpenRouter's API.
        
        Returns:
            List of model identifiers
        """
        try:
            # Get the list of models from OpenRouter
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            if response.status_code == 200:
                models_data = response.json()
                return [model['id'] for model in models_data]
            else:
                logger.error(f"Failed to get models from OpenRouter: {response.status_code}, {response.text}")
                return [self.model]  # Return default model as fallback
        except Exception as e:
            logger.error(f"Error fetching models from OpenRouter: {str(e)}")
            return [self.model]  # Return default model as fallback
    
    def complete(
    self, 
    prompt: str, 
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Optional[Union[str, List[str]]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
        """
        Complete a prompt using the OpenRouter API.
        
        Args:
            prompt: The user prompt to complete
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (overrides instance setting if provided)
            max_tokens: Maximum tokens to generate (overrides instance setting if provided)
            top_p: Nucleus sampling parameter (overrides instance setting if provided)
            stop: Optional stop sequences
            model: Optional model override (use a different model for this request)
            
        Returns:
            Dictionary containing the completion result
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Set up parameters
            params = {
                "model": model if model is not None else self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "top_p": top_p if top_p is not None else self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }
            
            # Add max_tokens if specified
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            elif self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens
                
            # Add stop sequences if specified
            if stop:
                params["stop"] = stop
            
            # Add OpenRouter-specific headers
            extra_headers = {
                "HTTP-Referer": self.site_url,  # Optional. Site URL for rankings on openrouter.ai
                "X-Title": self.site_name,      # Optional. Site title for rankings on openrouter.ai
            }
            
            # Make the API call
            response = self.client.chat.completions.create(
                extra_headers=extra_headers,
                **params
            )
            
            # Extract and return the completion
            completion = response.choices[0].message.content
            
            return {
                "completion": completion,
                "model": params["model"],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        
        except Exception as e:
            logger.error(f"Error in OpenRouter completion: {str(e)}")
            return {
                "completion": f"Error: {str(e)}",
                "model": model if model is not None else self.model,
                "error": str(e)
            }

    def explain_visualization(
    self,
    viz_type: str,
    data_description: Dict[str, Any],
    viz_description: Dict[str, Any],
    detail_level: str = "intermediate",
    focus: str = "statistical",
    model: Optional[str] = None
) -> Dict[str, Any]:
        """
        Generate an explanation for a data visualization.
        
        Args:
            viz_type: Type of visualization (e.g., "histogram", "scatter plot", "correlation matrix")
            data_description: Description of the data being visualized
            viz_description: Description of visualization parameters
            detail_level: Level of detail for explanation ("beginner", "intermediate", "advanced")
            focus: Focus area for explanation ("statistical", "causal", "domain")
            model: Optional override of the default model
            
        Returns:
            Dictionary containing the explanation
        """
        # Create a system prompt focused on data visualization explanation
        system_prompt = """You are an expert in data visualization, statistics, and data science.
        Your task is to explain visualizations clearly and insightfully at the appropriate level of detail.
        Focus on helping the user understand what the visualization shows, any patterns or insights revealed,
        and how this might inform their analysis or decision-making."""
        
        # Create the prompt based on visualization type
        prompt = f"""# Visualization Explanation Request

    ## Visualization Type
    {viz_type}

    ## Data Description
    ```json
    {json.dumps(data_description, indent=2)}
    ```

    ## Visualization Parameters
    ```json
    {json.dumps(viz_description, indent=2)}
    ```

    ## Explanation Parameters
    - Detail Level: {detail_level} (beginner, intermediate, or advanced)
    - Focus: {focus} (statistical, causal, or domain)

    ## Instructions

    Please explain this {viz_type} visualization with the following considerations:

    1. Use a {detail_level} level of detail
    - Beginner: Simple, non-technical explanations for those new to data analysis
    - Intermediate: Moderate technical depth with explanations of concepts
    - Advanced: In-depth technical analysis suitable for data scientists

    2. Focus on {focus} aspects:
    - Statistical: Focus on distributions, correlations, statistical properties
    - Causal: Discuss potential causal relationships and implications
    - Domain: Emphasize practical implications and domain-specific insights

    3. Include the following in your explanation:
    - What this visualization shows and how to interpret it
    - Key patterns or insights visible in the visualization
    - Potential limitations or things to be cautious about
    - Suggestions for further analysis based on what's shown

    4. Format your explanation in clear markdown with appropriate headers and bullet points.
    """

        # If model is specified, use it instead of the default model
        use_model = model if model else self.model
        
        # Get the completion
        result = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1000,
            model=use_model
        )
        
        # Return the explanation
        return {
            "explanation": result["completion"],
            "viz_type": viz_type,
            "detail_level": detail_level,
            "focus": focus,
            "model_used": use_model
        }
    
    def causal_refinement(
        self,
        graph_data: Dict[str, Any],
        variable_descriptions: Dict[str, str],
        domain_context: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Use the LLM to refine a causal graph.
        
        Args:
            graph_data: The causal graph data to refine
            variable_descriptions: Descriptions of the variables in the graph
            domain_context: Optional domain-specific context to guide refinement
            temperature: Optional temperature override
            
        Returns:
            Dictionary containing the refined graph and explanations
        """
        # Convert graph data to a readable format for the LLM
        graph_description = self._format_graph_for_prompt(graph_data)
        
        # Format variable descriptions
        var_descriptions = "\n".join([f"- {var}: {desc}" for var, desc in variable_descriptions.items()])
        
        # Create the prompt for causal refinement
        system_prompt = """You are an expert in causal inference and domain knowledge integration. 
        Your task is to refine a causal graph based on statistical evidence and domain knowledge. 
        Analyze the graph carefully, identify potential issues, suggest refinements, and explain your reasoning."""
        
        # Construct the prompt using string concatenation instead of f-strings
        prompt = """# Causal Graph Refinement Task

I have a causal graph derived from data using causal discovery algorithms. 
Please help refine this graph by evaluating the plausibility of relationships, 
suggesting missing links or variables, and identifying potential confounders.

## Graph Structure
""" + graph_description + """

## Variable Descriptions
""" + var_descriptions + """

""" + ("## Domain Context\n" + domain_context if domain_context else "") + """

## Refinement Tasks
1. Evaluate the plausibility of each identified causal relationship
2. Suggest any missing relationships that should be present based on domain knowledge
3. Identify any potential hidden confounders
4. Propose a refined graph structure

For each suggestion, please provide your reasoning based on both causal principles and domain knowledge.

Format your response as:
1. Evaluation of existing relationships
2. Suggested new relationships 
3. Potential hidden confounders
4. Refined graph structure (as an adjacency list)
5. Confidence levels for each relationship (High/Medium/Low)"""

        # Get the completion
        result = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else 0.3,
            max_tokens=2000
        )
        
        # Return the refinement result
        return {
            "refined_graph": result["completion"],
            "original_graph": graph_data,
            "reasoning": result["completion"],
        }
    
    def explain_causal_relationship(
        self,
        from_var: str,
        to_var: str,
        graph_data: Dict[str, Any],
        variable_descriptions: Dict[str, str],
        data_summary: Optional[Dict[str, Any]] = None,
        domain_context: Optional[str] = None,
        detail_level: str = "medium",
    ) -> Dict[str, Any]:
        """
        Generate an explanation for a specific causal relationship.
        
        Args:
            from_var: Source variable in the relationship
            to_var: Target variable in the relationship
            graph_data: The causal graph data
            variable_descriptions: Descriptions of the variables
            data_summary: Optional summary statistics of the data
            domain_context: Optional domain context
            detail_level: Detail level for the explanation (simple, medium, detailed)
            
        Returns:
            Dictionary containing the explanation
        """
        # Format variable descriptions
        var_descriptions = "\n".join([f"- {var}: {desc}" for var, desc in variable_descriptions.items()])
        
        # Create a readable description of the graph
        graph_description = self._format_graph_for_prompt(graph_data)
        
        # Create data summary text if provided
        data_summary_text = ""
        if data_summary:
            data_summary_text = "## Data Summary\n"
            for var, stats in data_summary.items():
                data_summary_text += f"- {var}: {stats}\n"
        
        # Set detail level instruction
        detail_instructions = {
            "simple": "Provide a simple, non-technical explanation suitable for someone without statistical background.",
            "medium": "Provide a moderately detailed explanation with some technical concepts but explained clearly.",
            "detailed": "Provide a detailed technical explanation suitable for someone with statistical knowledge."
        }.get(detail_level, "Provide a moderately detailed explanation with some technical concepts but explained clearly.")
        
        # Create the prompt
        system_prompt = """You are an expert in causal inference, statistics, and scientific communication.
        Your task is to explain causal relationships in a clear, accurate, and helpful manner,
        adapting your explanation to the requested level of detail."""
        
        # Construct the prompt using string concatenation instead of f-strings
        prompt = """# Causal Relationship Explanation

Please explain the causal relationship from '""" + from_var + """' to '""" + to_var + """' based on the provided information.

## Variable Descriptions
""" + var_descriptions + """

## Causal Graph Structure
""" + graph_description + """

""" + data_summary_text + """

""" + ("## Domain Context\n" + domain_context if domain_context else "") + """

## Instructions
""" + detail_instructions + """

In your explanation, please address:
1. The nature of the causal relationship
2. Possible mechanisms explaining how """ + from_var + """ affects """ + to_var + """
3. Any mediating factors or pathways
4. Potential confounding factors
5. Strength and confidence in this causal relationship
6. Practical implications of this relationship"""

        # Get the completion
        result = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # Return the explanation
        return {
            "explanation": result["completion"],
            "from_var": from_var,
            "to_var": to_var,
            "detail_level": detail_level
        }
    
    def generate_causal_explanation(
        self,
        from_var: Optional[str] = None,
        to_var: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        variable_descriptions: Optional[Dict[str, str]] = None,
        data_summary: Optional[Dict[str, Any]] = None,
        domain_context: Optional[str] = None,
        detail_level: str = "medium",
        relationship: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an explanation for a specific causal relationship.
        This is an alias for explain_causal_relationship for compatibility.
        
        Args:
            from_var: Source variable in the relationship
            to_var: Target variable in the relationship
            graph_data: The causal graph data
            variable_descriptions: Descriptions of the variables
            data_summary: Optional summary statistics of the data
            domain_context: Optional domain context
            detail_level: Detail level for the explanation (simple, medium, detailed)
            relationship: Optional relationship data (for backward compatibility)
            
        Returns:
            Dictionary containing the explanation
        """
        # Handle relationship parameter for backward compatibility
        if relationship is not None:
            from_var = relationship.get('from', from_var)
            to_var = relationship.get('to', to_var)
        
        return self.explain_causal_relationship(
            from_var=from_var,
            to_var=to_var,
            graph_data=graph_data,
            variable_descriptions=variable_descriptions,
            data_summary=data_summary,
            domain_context=domain_context,
            detail_level=detail_level
        )
    
    def _format_graph_for_prompt(self, graph_data: Dict[str, Any]) -> str:
        """
        Format graph data into a readable text representation for the prompt.
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            Formatted string representation of the graph
        """
        # Extract nodes and edges
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Format as adjacency list and edge list
        formatted_text = "### Graph Structure\n"
        
        # Node list
        formatted_text += "**Nodes:**\n"
        for node in nodes:
            node_id = node.get("id", "")
            node_name = node.get("name", node_id)
            formatted_text += f"- {node_name} (ID: {node_id})\n"
        
        # Edge list
        formatted_text += "\n**Causal Relationships:**\n"
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            weight = edge.get("weight", "")
            confidence = edge.get("confidence", "")
            
            # Find source and target node names
            source_name = source
            target_name = target
            for node in nodes:
                if node.get("id") == source:
                    source_name = node.get("name", source)
                if node.get("id") == target:
                    target_name = node.get("name", target)
            
            # Format edge information
            edge_info = f"- {source_name} → {target_name}"
            if weight:
                edge_info += f" (weight: {weight})"
            if confidence:
                edge_info += f" (confidence: {confidence})"
            
            formatted_text += edge_info + "\n"
        
        return formatted_text
    

    def explain_visualization(
    self,
    viz_type: str,
    data_description: Dict[str, Any],
    viz_description: Dict[str, Any],
    detail_level: str = "intermediate",
    focus: str = "statistical",
    model: Optional[str] = None
) -> Dict[str, Any]:
        """
        Generate an explanation for a data visualization.
        
        Args:
            viz_type: Type of visualization (e.g., "histogram", "scatter plot", "correlation matrix")
            data_description: Description of the data being visualized
            viz_description: Description of visualization parameters
            detail_level: Level of detail for explanation ("beginner", "intermediate", "advanced")
            focus: Focus area for explanation ("statistical", "causal", "domain")
            model: Optional override of the default model
            
        Returns:
            Dictionary containing the explanation
        """
        # Create a system prompt focused on data visualization explanation
        system_prompt = """You are an expert in data visualization, statistics, and data science.
        Your task is to explain visualizations clearly and insightfully at the appropriate level of detail.
        Focus on helping the user understand what the visualization shows, any patterns or insights revealed,
        and how this might inform their analysis or decision-making."""
        
        # Create the prompt based on visualization type
        prompt = f"""# Visualization Explanation Request

    ## Visualization Type
    {viz_type}

    ## Data Description
    ```json
    {json.dumps(data_description, indent=2)}
    ```

    ## Visualization Parameters
    ```json
    {json.dumps(viz_description, indent=2)}
    ```

    ## Explanation Parameters
    - Detail Level: {detail_level} (beginner, intermediate, or advanced)
    - Focus: {focus} (statistical, causal, or domain)

    ## Instructions

    Please explain this {viz_type} visualization with the following considerations:

    1. Use a {detail_level} level of detail
    - Beginner: Simple, non-technical explanations for those new to data analysis
    - Intermediate: Moderate technical depth with explanations of concepts
    - Advanced: In-depth technical analysis suitable for data scientists

    2. Focus on {focus} aspects:
    - Statistical: Focus on distributions, correlations, statistical properties
    - Causal: Discuss potential causal relationships and implications
    - Domain: Emphasize practical implications and domain-specific insights

    3. Include the following in your explanation:
    - What this visualization shows and how to interpret it
    - Key patterns or insights visible in the visualization
    - Potential limitations or things to be cautious about
    - Suggestions for further analysis based on what's shown

    4. Format your explanation in clear markdown with appropriate headers and bullet points.
    """

        # If model is specified, use it instead of the default model
        use_model = model if model else self.model
        
        # Get the completion
        result = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=1000,
            model=use_model
        )
        
        # Return the explanation
        return {
            "explanation": result["completion"],
            "viz_type": viz_type,
            "detail_level": detail_level,
            "focus": focus,
            "model_used": use_model
        }