from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

# Define LLMResponse for backward compatibility
class LLMResponse:
    """
    Simple class to hold LLM response data for backward compatibility.
    """
    def __init__(self, content: str, model: str, usage: Optional[Dict[str, int]] = None, error: Optional[str] = None):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.error = error
        self.success = error is None

# Abstract base class for LLM adapters
class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    Defines the interface that all LLM adapters must implement.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the LLM adapter.
        
        Returns:
            String name of the adapter
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get a list of supported models.
        
        Returns:
            List of model identifiers
        """
        pass
    
    @abstractmethod
    def complete(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Complete a prompt using the LLM.
        
        Args:
            prompt: The user prompt to complete
            system_prompt: Optional system prompt to set context
            temperature: Optional sampling temperature
            max_tokens: Optional maximum tokens to generate
            top_p: Optional nucleus sampling parameter
            stop: Optional stop sequences
            
        Returns:
            Dictionary containing the completion result
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
        
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
        
# For backward compatibility
BaseLLMAdapter = LLMAdapter