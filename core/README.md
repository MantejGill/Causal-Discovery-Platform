# Core Module

The Core module contains the business logic and algorithms for the Causal Discovery Platform.

## Overview

This module implements the core functionality of the platform, separate from the user interface. It's organized into several submodules:

- **algorithms**: Causal discovery algorithms and their execution
- **data**: Data loading, processing, and profiling
- **llm**: Large Language Model integration
- **rag**: Retrieval-Augmented Generation
- **viz**: Visualization tools for causal graphs and data

## Directory Structure

```
core/
├── __init__.py
├── algorithms/     # Causal discovery algorithms
├── data/          # Data handling
├── llm/           # LLM integration
├── rag/           # Retrieval-Augmented Generation
└── viz/           # Visualization tools
```

## Submodules

### algorithms

The algorithms submodule integrates with causal discovery libraries and provides:

- Algorithm selection based on data characteristics
- Algorithm execution with error handling
- Ensemble methods for combining multiple algorithms
- Specialized algorithms for nonlinear and time series data

Learn more in the [Algorithms README](./algorithms/README.md).

### data

The data submodule handles:

- Data loading from various formats
- Data preprocessing and cleaning
- Missing data handling
- Data profiling for algorithm selection

Learn more in the [Data README](./data/README.md).

### llm

The llm submodule provides:

- Integration with various LLM providers (OpenAI, OpenRouter)
- Algorithm recommendation based on data characteristics
- Graph refinement using LLM analysis
- Natural language explanation generation

Learn more in the [LLM README](./llm/README.md).

### rag

The rag submodule implements:

- Document processing for domain-specific knowledge
- Query enhancement for LLM interactions
- Knowledge retrieval from documents
- Integration with LLM components

Learn more in the [RAG README](./rag/README.md).

### viz

The viz submodule provides:

- Causal graph visualization tools
- Data visualization utilities
- Counterfactual visualization
- Interactive visualization components

Learn more in the [Visualization README](./viz/README.md).

## Related Modules

- [App Module](../app/README.md): User interface components and pages
- [Database Module](../db/README.md): Data persistence
- [Utils Module](../utils/README.md): Utility functions
