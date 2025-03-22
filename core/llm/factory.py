import logging
from typing import Dict, Any, Optional

from core.llm.adapter import LLMAdapter
from core.llm.openai_adapter import OpenAIAdapter
from core.llm.openrouter_adapter import OpenRouterAdapter
from core.llm.rag_adapter import RAGLLMAdapter

# Import RAG manager type for type checking
try:
    from core.rag.rag_manager import RAGManager
except ImportError:
    # Create a type alias for type checking if RAGManager is not available
    from typing import Protocol
    class RAGManager(Protocol):
        def augment_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]: ...

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Factory class for creating and managing LLM adapters.
    Provides a centralized way to create different LLM adapters based on configuration.
    """
    
    @staticmethod
    def create_adapter(config: Dict[str, Any]) -> Optional[LLMAdapter]:
        """
        Create an LLM adapter based on the provided configuration.
        
        Args:
            config: Dictionary containing configuration parameters
                Required keys depend on the provider:
                - provider: The LLM provider ("openai" or "openrouter")
                - api_keys: Dictionary with API keys for different providers
                - For OpenAI: openai_model, temperature, max_tokens
                - For OpenRouter: openrouter_model, temperature, max_tokens
                - Optional RAG configuration:
                  - use_rag: Boolean flag to enable RAG
                  - rag_manager: Optional instance of RAGManager
                  - rag_db_dir: Directory for RAG database (if creating new manager)
                  - embeddings_model: Model for embeddings ("openai" or "huggingface")
        
        Returns:
            An initialized LLM adapter or None if initialization fails
        """
        try:
            # Create base adapter
            provider = config.get("provider", "openai")
            base_adapter = None
            
            if provider == "openai":
                api_key = config.get("api_keys", {}).get("openai")
                if not api_key:
                    logger.error("OpenAI API key not provided")
                    return None
                
                model = config.get("openai_model", "gpt-4o")
                temperature = config.get("temperature", 0.7)
                max_tokens = config.get("max_tokens", 1000)
                
                base_adapter = OpenAIAdapter(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            elif provider == "openrouter":
                api_key = config.get("api_keys", {}).get("openrouter")
                if not api_key:
                    logger.error("OpenRouter API key not provided")
                    return None
                
                model = config.get("openrouter_model", "deepseek/deepseek-r1:free")
                temperature = config.get("temperature", 0.7)
                max_tokens = config.get("max_tokens", 1000)
                
                base_adapter = OpenRouterAdapter(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            else:
                logger.error(f"Unknown LLM provider: {provider}")
                return None
            
            # Check if RAG is enabled
            use_rag = config.get("use_rag", False)
            
            if use_rag and base_adapter is not None:
                try:
                    # Import RAGManager here to avoid circular imports
                    from core.rag.rag_manager import RAGManager
                    
                    # Get or create RAG manager
                    rag_manager = config.get("rag_manager")
                    
                    if rag_manager is None:
                        # Create new RAG manager with provided configuration
                        logger.info("Creating new RAG manager")
                        rag_manager = RAGManager(
                            llm_adapter=base_adapter,
                            embeddings_model=config.get("embeddings_model", "openai"),
                            openai_api_key=config.get("api_keys", {}).get("openai"),
                            db_dir=config.get("rag_db_dir", "./data/rag_db")
                        )
                    
                    # Create RAG-enabled adapter
                    logger.info(f"Creating RAG-enabled adapter wrapping {base_adapter.get_name()}")
                    return RAGLLMAdapter(
                        base_adapter=base_adapter,
                        rag_manager=rag_manager
                    )
                except ImportError as e:
                    logger.error(f"Could not import RAG components: {str(e)}")
                    logger.warning("Falling back to base adapter without RAG")
                    return base_adapter
                except Exception as e:
                    logger.error(f"Error creating RAG-enabled adapter: {str(e)}")
                    logger.warning("Falling back to base adapter without RAG")
                    return base_adapter
            
            return base_adapter
                
        except Exception as e:
            logger.error(f"Error creating LLM adapter: {str(e)}")
            return None
    
    @staticmethod
    def get_provider_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported LLM providers.
        
        Returns:
            Dictionary with information about each supported provider
        """
        return {
            "openai": {
                "name": "OpenAI",
                "description": "Commercial provider with powerful models like GPT-4",
                "models": [
                    {"id": "gpt-4o", "name": "GPT-4o", "description": "Latest and most capable OpenAI model"},
                    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "Smaller, faster version of GPT-4o"},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "Previous generation top model"},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Faster, lower-cost option"}
                ],
                "requires_api_key": True,
                "free_tier": False
            },
            "openrouter": {
                "name": "OpenRouter (DeepSeek R1)",
                "description": "Access to DeepSeek's open-source models via OpenRouter",
                "models": [
                    {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1", "description": "Open-source model comparable to OpenAI's o1, MIT licensed"},
                    {"id": "deepseek/deepseek-r1-distill-llama-70b", "name": "DeepSeek R1 Distill Llama 70B", "description": "Distilled version based on Llama 3.3 70B"},
                    {"id": "deepseek/deepseek-r1-distill-qwen-32b", "name": "DeepSeek R1 Distill Qwen 32B", "description": "Distilled version based on Qwen 2.5 32B"},
                    {"id": "deepseek/deepseek-r1-distill-qwen-14b", "name": "DeepSeek R1 Distill Qwen 14B", "description": "Distilled version based on Qwen 2.5 14B"}
                ],
                "requires_api_key": True,
                "free_tier": True
            }
        }
    
    @staticmethod
    def create_rag_enabled_adapter(base_adapter: LLMAdapter, 
                                rag_config: Dict[str, Any]) -> Optional[LLMAdapter]:
        """
        Create a RAG-enabled adapter wrapping an existing base adapter.
        
        Args:
            base_adapter: Existing LLM adapter to wrap
            rag_config: Configuration for RAG system
                - openai_api_key: Optional API key for OpenAI embeddings
                - embeddings_model: Model for embeddings ("openai" or "huggingface")
                - db_dir: Directory for RAG database
                - existing_rag_manager: Optional existing RAG manager to use
        
        Returns:
            RAG-enabled LLM adapter or None if initialization fails
        """
        try:
            # Import RAGManager here to avoid circular imports
            from core.rag.rag_manager import RAGManager
            
            # Get or create RAG manager
            rag_manager = rag_config.get("existing_rag_manager")
            
            if rag_manager is None:
                # Create new RAG manager
                rag_manager = RAGManager(
                    llm_adapter=base_adapter,
                    embeddings_model=rag_config.get("embeddings_model", "openai"),
                    openai_api_key=rag_config.get("openai_api_key"),
                    db_dir=rag_config.get("db_dir", "./data/rag_db")
                )
            
            # Create RAG-enabled adapter
            return RAGLLMAdapter(
                base_adapter=base_adapter,
                rag_manager=rag_manager
            )
            
        except ImportError as e:
            logger.error(f"Could not import RAG components: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error creating RAG-enabled adapter: {str(e)}")
            return None