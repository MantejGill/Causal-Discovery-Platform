# App Components

This directory contains reusable UI components for the Causal Discovery Platform.

## Overview

These components encapsulate specific functionality and UI elements that are used across different pages of the application. They provide a modular approach to building the user interface.

## Key Components

### Visualization Components

- **AdjacencyMatrix** (`adjacency_matrix.py`): Displays the causal graph as an adjacency matrix
- **CausalGraph** (`causal_graph.py`): Interactive causal graph visualization
- **DataViz** (`data_viz.py`): Data visualization tools for exploratory analysis
- **DistributionExplorer** (`distribution_explorer.py`): Visualize variable distributions

### Data Processing Components

- **CategoricalTransformer** (`categorical_transformer.py`): Handles categorical data conversion
- **DataConnector** (`data_connector.py`): Manages data connections and imports
- **OutlierDetector** (`outlier_detector.py`): Identifies outliers in the data
- **TimeseriesPreprocessor** (`timeseries_preprocessor.py`): Prepares time series data

### Causal Analysis Components

- **CausalCanvas** (`causal_canvas.py`): Interface for building and editing causal graphs
- **EffectEstimator** (`effect_estimator.py`): Estimates causal effects from graphs
- **GraphFilter** (`graph_filter.py`): Filters and transforms causal graphs
- **HiddenVariableDetector** (`hidden_variable_detector.py`): Detects potential latent confounders
- **NonlinearDiscovery** (`nonlinear_discovery.py`): Specialized discovery for nonlinear relationships
- **TimeseriesDiscovery** (`timeseries_discovery.py`): Specialized discovery for time series data

### Integration Components

- **ComparisonView** (`comparison_view.py`): Compares multiple causal graphs
- **DomainKnowledgeIntegrator** (`domain_knowledge_integrator.py`): Integrates expert knowledge
- **ExecutionEngine** (`execution_engine.py`): Executes algorithms with progress tracking
- **Explanation** (`explanation.py`): Generates explanations of causal relationships
- **HypothesisTester** (`hypothesis_tester.py`): Tests causal hypotheses
- **PathFinder** (`path_finder.py`): Finds paths between variables in causal graphs

### System Components

- **ComponentRegistry** (`component_registry.py`): Registry for dynamic component loading

## Usage

Components are designed to be imported and used within the application pages. They typically follow this pattern:

```python
from app.components.causal_graph import CausalGraph

# In a Streamlit page:
causal_graph = CausalGraph(graph_data)
causal_graph.render()
```

## Related Documentation

- [App Pages](../pages/README.md)
- [App Module Overview](../README.md)
- [Core Visualization](../../core/viz/README.md)
