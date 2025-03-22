from typing import Dict, List, Optional, Union, Any
import logging

from core.llm.adapter import LLMAdapter
from core.rag.rag_manager import RAGManager

logger = logging.getLogger(__name__)

class RAGLLMAdapter(LLMAdapter):
    """
    Wraps an existing LLM adapter with RAG capabilities.
    
    This adapter maintains the same interface as the base adapter
    but enhances prompts with relevant context retrieved from the
    RAG system before passing them to the underlying LLM.
    """
    
    def __init__(self, 
                base_adapter: LLMAdapter,
                rag_manager: RAGManager):
        """
        Initialize the RAG-enabled LLM adapter
        
        Args:
            base_adapter: The base LLM adapter to wrap
            rag_manager: The RAG manager to use for context retrieval
        """
        self.base_adapter = base_adapter
        self.rag_manager = rag_manager
    
    def get_name(self) -> str:
        """
        Get the name of the LLM adapter.
        
        Returns:
            String name of the adapter with RAG indicator
        """
        return f"{self.base_adapter.get_name()} with RAG"
    
    def get_supported_models(self) -> List[str]:
        """
        Get a list of supported models.
        
        Returns:
            List of model identifiers from the base adapter
        """
        return self.base_adapter.get_supported_models()
    
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
        Complete a prompt using the LLM with RAG enhancement.
        
        Args:
            prompt: The user prompt to complete
            system_prompt: Optional system prompt to set context
            temperature: Optional sampling temperature
            max_tokens: Optional maximum tokens to generate
            top_p: Optional nucleus sampling parameter
            stop: Optional stop sequences
            
        Returns:
            Dictionary containing the completion result with RAG metadata
        """
        try:
            # Check if RAG is ready
            if not self.rag_manager.is_ready:
                logger.info("RAG system not ready, using base adapter without augmentation")
                result = self.base_adapter.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop
                )
                result["rag_applied"] = False
                return result
            
            # Augment the prompt with relevant context
            augmented = self.rag_manager.augment_prompt(prompt, system_prompt)
            
            # Forward to base adapter with augmented prompts
            result = self.base_adapter.complete(
                prompt=augmented["augmented_prompt"],
                system_prompt=augmented["augmented_system_prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop
            )
            
            # Add RAG metadata to the result
            result["rag_applied"] = augmented["context_used"]
            result["rag_documents"] = augmented["documents"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced completion: {str(e)}")
            # Fall back to base adapter on error
            try:
                logger.info("Falling back to base adapter without RAG")
                result = self.base_adapter.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop
                )
                result["rag_applied"] = False
                result["rag_error"] = str(e)
                return result
            except Exception as inner_e:
                logger.error(f"Base adapter fallback also failed: {str(inner_e)}")
                return {
                    "content": f"Error: {str(e)}. Fallback also failed: {str(inner_e)}",
                    "status": "error",
                    "rag_applied": False
                }
    
    def _enhance_domain_context(self, 
                             query: str, 
                             domain_context: Optional[str] = None) -> str:
        """
        Enhance domain context with relevant literature from the RAG system
        
        Args:
            query: The query to retrieve context for
            domain_context: Original domain context provided
            
        Returns:
            Enhanced domain context with RAG literature
        """
        if not self.rag_manager.is_ready:
            return domain_context
        
        try:
            # Retrieve context documents
            docs = self.rag_manager.retrieve_context(query)
            
            if not docs:
                return domain_context
            
            # Format context from retrieved documents
            rag_context = "\n\n".join([
                f"From '{doc['metadata'].get('title', 'literature')}' "
                f"by {doc['metadata'].get('author', 'unknown author')}:\n"
                f"{doc['content']}"
                for doc in docs[:3]  # Limit to top 3 documents
            ])
            
            # Combine with original domain context
            if domain_context:
                return f"{domain_context}\n\nAdditional domain knowledge from literature:\n{rag_context}"
            else:
                return f"Domain knowledge from literature:\n{rag_context}"
                
        except Exception as e:
            logger.error(f"Error enhancing domain context: {str(e)}")
            return domain_context
    
    def causal_refinement(
        self,
        graph_data: Dict[str, Any],
        variable_descriptions: Dict[str, str],
        domain_context: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Use the LLM to refine a causal graph with RAG enhancement.
        
        Args:
            graph_data: The causal graph data to refine
            variable_descriptions: Descriptions of the variables in the graph
            domain_context: Optional domain-specific context to guide refinement
            temperature: Optional temperature override
            
        Returns:
            Dictionary containing the refined graph and explanations
        """
        try:
            # Build a query from the graph data and variables
            query = f"Causal graph refinement for variables: {', '.join(variable_descriptions.keys())}"
            
            # Enhance domain context with relevant literature
            enhanced_context = self._enhance_domain_context(query, domain_context)
            
            # Forward to base adapter with enhanced context
            return self.base_adapter.causal_refinement(
                graph_data=graph_data,
                variable_descriptions=variable_descriptions,
                domain_context=enhanced_context,
                temperature=temperature
            )
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced causal refinement: {str(e)}")
            # Fall back to base adapter
            return self.base_adapter.causal_refinement(
                graph_data=graph_data,
                variable_descriptions=variable_descriptions,
                domain_context=domain_context,
                temperature=temperature
            )
    
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
        Generate an explanation for a specific causal relationship with RAG enhancement.
        
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
        try:
            # Build a query specifically for this causal relationship
            query = f"Causal relationship from {from_var} to {to_var}"
            if from_var in variable_descriptions and to_var in variable_descriptions:
                query += f": {variable_descriptions[from_var]} affects {variable_descriptions[to_var]}"
            
            # Enhance domain context with relevant literature
            enhanced_context = self._enhance_domain_context(query, domain_context)
            
            # Forward to base adapter with enhanced context
            return self.base_adapter.explain_causal_relationship(
                from_var=from_var,
                to_var=to_var,
                graph_data=graph_data,
                variable_descriptions=variable_descriptions,
                data_summary=data_summary,
                domain_context=enhanced_context,
                detail_level=detail_level
            )
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced causal explanation: {str(e)}")
            # Fall back to base adapter
            return self.base_adapter.explain_causal_relationship(
                from_var=from_var,
                to_var=to_var,
                graph_data=graph_data,
                variable_descriptions=variable_descriptions,
                data_summary=data_summary,
                domain_context=domain_context,
                detail_level=detail_level
            )
    
    # Ensure compatibility with base adapter
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
