import logging
from typing import Dict, Any, Optional

from core.llm.adapter import LLMAdapter
from core.llm.openai_adapter import OpenAIAdapter
from core.llm.openrouter_adapter import OpenRouterAdapter

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
        
        Returns:
            An initialized LLM adapter or None if initialization fails
        """
        try:
            provider = config.get("provider", "openai")
            
            if provider == "openai":
                api_key = config.get("api_keys", {}).get("openai")
                if not api_key:
                    logger.error("OpenAI API key not provided")
                    return None
                
                model = config.get("openai_model", "gpt-4o")
                temperature = config.get("temperature", 0.7)
                max_tokens = config.get("max_tokens", 1000)
                
                return OpenAIAdapter(
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
                
                return OpenRouterAdapter(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            else:
                logger.error(f"Unknown LLM provider: {provider}")
                return None
                
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