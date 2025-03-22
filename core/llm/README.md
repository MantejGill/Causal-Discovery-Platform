# Core LLM Module

This directory contains the Large Language Model integration components for the Causal Discovery Platform.

## Overview

The LLM module provides integration with various Large Language Models to enhance the causal discovery process. It enables:

- Intelligent algorithm selection
- Causal graph refinement
- Natural language explanations of causal relationships
- Domain knowledge integration
- Hidden variable detection

## Key Components

### LLM Adapter (`adapter.py`)

The base adapter class that defines the interface for LLM integrations. It provides methods for:

- Completing prompts
- Generating explanations
- Structured outputs for causal judgments
- Error handling and retries

### LLM Factory (`factory.py`)

A factory pattern implementation that creates the appropriate LLM adapter based on configuration. Supports multiple providers:

- OpenAI
- OpenRouter
- Local models (e.g., Llama)

### OpenAI Adapter (`openai_adapter.py`)

Implementation of the LLM Adapter interface for OpenAI models:

- GPT-4o
- GPT-4o-mini
- GPT-4-turbo
- GPT-3.5-turbo

### OpenRouter Adapter (`openrouter_adapter.py`)

Implementation of the LLM Adapter interface for models available through OpenRouter:

- Claude
- DeepSeek R1
- Mistral
- Various open models

### Llama Adapter (`llama_adapter.py`)

Implementation of the LLM Adapter interface for local Llama models.

### Algorithm Recommender (`algorithm_recommender.py`)

Uses LLMs to:

- Analyze data characteristics
- Recommend appropriate causal discovery algorithms
- Explain recommendations and tradeoffs
- Consider domain-specific factors

### Graph Refinement (`refinement.py`)

Uses LLMs to:

- Validate causal relationships
- Detect spurious correlations
- Suggest missing edges
- Identify potential hidden variables
- Apply domain knowledge to causal graphs

### Prompt Templates (`prompts.py`)

Contains structured prompts for different LLM tasks:

- Algorithm recommendation
- Graph refinement
- Explanation generation
- Effect estimation
- Domain knowledge integration

### RAG Adapter (`rag_adapter.py`)

Extends the LLM Adapter to integrate with the Retrieval-Augmented Generation module for enhanced domain knowledge.

## Usage

The LLM components are typically accessed through the UI, but can also be used programmatically:

```python
from core.llm.factory import LLMFactory

# Create an LLM adapter
llm_factory = LLMFactory()
llm_adapter = llm_factory.create_adapter(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Generate a causal explanation
explanation = llm_adapter.explain_causal_relationship(
    cause="Exercise",
    effect="Heart Health",
    detail_level="intermediate"
)
```

## Related Documentation

- [Core Module Overview](../README.md)
- [RAG Module](../rag/README.md)
- [Algorithm Module](../algorithms/README.md)
- [Graph Refinement Page](../../app/pages/README.md)
