# Core Algorithms

This directory contains the causal discovery algorithms and related functionality for the Causal Discovery Platform.

## Overview

The algorithms module integrates with causal learning libraries (primarily CausalLearn) and provides a unified interface for:

- Selecting appropriate algorithms based on data characteristics
- Executing algorithms with appropriate parameters
- Processing and interpreting algorithm results
- Combining results from multiple algorithms (ensemble methods)

## Key Components

### Algorithm Selector (`selector.py`)

The Algorithm Selector analyzes data characteristics and recommends appropriate causal discovery algorithms. It considers:

- Data types (continuous, discrete, mixed)
- Sample size
- Missing data
- Assumptions about latent confounders
- Assumptions about nonlinearity
- Time series characteristics

### Algorithm Executor (`executor.py`)

The Algorithm Executor:

- Runs selected algorithms with appropriate parameters
- Handles errors and exceptions
- Standardizes algorithm outputs
- Provides progress tracking
- Caches results when appropriate

### Ensemble Methods (`ensemble.py`)

The Ensemble module:

- Combines results from multiple causal discovery algorithms
- Resolves conflicts between different algorithm outputs
- Uses voting mechanisms for edge presence and direction
- Integrates LLM assistance for conflict resolution

### Specialized Algorithms

- **Kernel Methods** (`kernel_methods.py`): Implements kernel-based independence tests
- **Nonlinear Models** (`nonlinear_models.py`): Specialized algorithms for nonlinear causal discovery
- **Nonstationarity** (`nonstationarity.py`): Algorithms that leverage distributional shifts
- **Time Series** (`timeseries.py`): Specialized algorithms for time series data

## Supported Algorithms

The platform supports a variety of causal discovery algorithms:

### Constraint-Based Algorithms
- PC (Peter-Clark Algorithm)
- FCI (Fast Causal Inference)
- RFCI (Really Fast Causal Inference)

### Score-Based Algorithms
- GES (Greedy Equivalence Search)
- MMHC (Max-Min Hill Climbing)

### FCM-Based Algorithms
- LiNGAM (Linear Non-Gaussian Acyclic Models)
- DirectLiNGAM
- ICA-LiNGAM

### Hidden Causal Algorithms
- FCI
- RFCI
- BIDIRECTIONAL

### Granger Causality Algorithms
- Granger VAR
- Granger Pairwise
- VAR-LiNGAM

## Usage

These algorithms are typically not called directly from application code but are invoked through the App's user interface. However, they can be used programmatically:

```python
from core.algorithms.selector import AlgorithmSelector
from core.algorithms.executor import AlgorithmExecutor

# Select appropriate algorithms
selector = AlgorithmSelector()
recommended_algorithms = selector.suggest_algorithms(data_profile)

# Execute algorithm
executor = AlgorithmExecutor()
result = executor.execute_algorithm("pc_fisherz", data, params={"alpha": 0.05})

# Process results
graph = result["graph"]
```

## Related Documentation

- [Core Module Overview](../README.md)
- [LLM Integration](../llm/README.md)
- [Causal Discovery Page](../../app/pages/README.md)
