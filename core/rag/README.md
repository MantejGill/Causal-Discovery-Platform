# Core RAG Module

This directory contains the Retrieval-Augmented Generation (RAG) components for the Causal Discovery Platform.

## Overview

The RAG module enhances LLM capabilities by retrieving and integrating domain-specific knowledge from uploaded documents. It enables:

- More accurate causal analysis based on domain literature
- Evidence-based causal judgments
- Domain-specific explanations
- Integration of research findings into causal discovery

## Key Components

### RAG Manager (`rag_manager.py`)

The central component that:

- Coordinates document processing
- Manages the document database
- Handles document retrieval
- Integrates with LLM adapters

### Document Processor (`document_processor.py`)

Processes documents for retrieval:

- Extracts text from various file formats
- Segments documents into chunks
- Processes metadata
- Handles document updates and versions

### Simple Document Processor (`simple_document_processor.py`)

A lightweight implementation of the document processor for basic use cases.

### Query Engine (`query_engine.py`)

Specialized query handling for causal discovery:

- Formulates effective retrieval queries
- Ranks and filters relevant passages
- Structures retrieved content for LLM consumption
- Integrates domain knowledge into causal analysis

### Async Utilities (`async_utils.py`)

Utilities for asynchronous document processing and retrieval.

## Document Storage

The RAG module uses a vector database to store document embeddings for efficient retrieval. The platform supports:

- ChromaDB (default, included)
- Potential integration with other vector stores

Documents are automatically processed when uploaded through the Knowledge Base page.

## Usage

The RAG components integrate with the LLM module and are typically accessed through the UI, but can also be used programmatically:

```python
from core.rag.rag_manager import RAGManager
from core.llm.factory import LLMFactory

# Initialize RAG manager
rag_manager = RAGManager()

# Process documents
rag_manager.add_document("path/to/document.pdf")

# Get LLM adapter with RAG integration
llm_factory = LLMFactory()
llm_adapter = llm_factory.create_adapter(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    rag_manager=rag_manager
)

# Use RAG-enhanced LLM for causal analysis
result = llm_adapter.analyze_causal_relationship(
    cause="Treatment A",
    effect="Outcome B",
    context="medical research"
)
```

## Related Documentation

- [Core Module Overview](../README.md)
- [LLM Module](../llm/README.md)
- [Knowledge Base Page](../../app/pages/README.md)
