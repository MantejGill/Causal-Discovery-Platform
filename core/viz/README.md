# Core Visualization Module

This directory contains the visualization tools for the Causal Discovery Platform.

## Overview

The visualization module provides tools for creating interactive and informative visualizations of:

- Causal graphs
- Data distributions
- Correlation patterns
- Counterfactual analysis
- Time series data

These visualizations help users understand both their data and the causal relationships discovered by the platform.

## Key Components

### Graph Visualization (`graph.py`)

The Graph Visualization component:

- Renders causal graphs using different layouts
- Provides interactive graph manipulation
- Displays edge weights and confidence
- Supports various node and edge styling
- Creates adjacency matrix visualizations
- Enables graph comparison views

### Distribution Visualization (`distribution.py`)

The Distribution Visualization component creates:

- Histograms and density plots
- Box plots and violin plots
- Scatter plots and pair plots
- Correlation heatmaps
- Time series plots

### Counterfactual Visualization (`counterfactual.py`)

The Counterfactual Visualization component:

- Visualizes the effects of interventions
- Creates comparison plots for different scenarios
- Shows uncertainty in causal effect estimates
- Displays treatment effect distributions

## Technologies

The visualization module uses:

- **Plotly**: For interactive graph and data visualizations
- **NetworkX**: For graph data structures and algorithms
- **Matplotlib**: For additional plotting capabilities
- **Streamlit Components**: For integration with the Streamlit UI

## Usage

The visualization components are typically accessed through the App's UI components, but can also be used programmatically:

```python
from core.viz.graph import CausalGraphVisualizer
import networkx as nx

# Create a sample graph
G = nx.DiGraph()
G.add_edge('A', 'B', weight=0.7)
G.add_edge('B', 'C', weight=0.5)
G.add_edge('A', 'C', weight=0.3)

# Visualize the graph
graph_viz = CausalGraphVisualizer()
fig = graph_viz.visualize_graph(
    graph=G,
    layout_type="circular",
    show_weights=True
)

# The figure can be displayed in Streamlit or saved
```

## Related Documentation

- [Core Module Overview](../README.md)
- [App Components](../../app/components/README.md)
- [Causal Discovery Page](../../app/pages/README.md)
- [Analysis & Explanation Page](../../app/pages/README.md)
