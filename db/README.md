# Database Module

This directory contains the database models and session handling for the Causal Discovery Platform.

## Overview

The database module provides:

- Data models for persistent storage
- Session management
- Database connection handling
- Data serialization and deserialization

## Key Components

### Database Models (`models.py`)

Defines the data models for:

- User sessions
- Saved datasets
- Causal graphs
- Analysis results
- User settings

### Session Management (`session.py`)

Provides:

- Database session creation and management
- Connection pooling
- Transaction handling
- Error recovery

## Database Schema

The platform uses a simple database schema to persist:

1. **User Sessions**: Tracks user sessions and their associated data
2. **Datasets**: Stores metadata about uploaded datasets
3. **CausalGraphs**: Stores discovered and refined causal graphs
4. **AnalysisResults**: Stores the results of causal analyses
5. **UserSettings**: Stores user preferences and settings

## Configuration

Database configuration is handled through environment variables:

- `DB_CONNECTION_STRING`: Database connection string
- `DB_POOL_SIZE`: Connection pool size
- `DB_TIMEOUT`: Query timeout in seconds

The default configuration uses SQLite for simplicity, but the module can be configured to use other database backends.

## Usage

The database components are typically used internally by the platform, but can also be used directly:

```python
from db.session import get_session
from db.models import CausalGraph
import networkx as nx

# Save a causal graph
def save_graph(graph_name, graph_data):
    graph = nx.DiGraph()
    # Add graph data...
    
    with get_session() as session:
        graph_model = CausalGraph(
            name=graph_name,
            data=nx.node_link_data(graph),  # Serialize the graph
            created_at=datetime.now()
        )
        session.add(graph_model)
        session.commit()
        return graph_model.id
```

## Related Documentation

- [Utils Module](../utils/README.md)
- [Core Module](../core/README.md)
- [Settings Page](../app/pages/README.md)
