# Utils Module

This directory contains utility functions and helpers for the Causal Discovery Platform.

## Overview

The utils module provides common utilities that are used across different parts of the platform:

- Logging utilities
- Serialization helpers
- Common helper functions
- System utilities

## Key Components

### Logger (`logger.py`)

Provides a standardized logging system for the entire platform:

- Configurable log levels
- Log file management
- Structured logging
- Console and file output

### Serialization (`serialization.py`)

Utilities for serializing and deserializing complex objects:

- NetworkX graph serialization
- Pandas DataFrame serialization
- Custom object serialization
- JSON conversion helpers

## Usage

These utilities are used throughout the codebase:

```python
from utils.logger import get_logger
from utils.serialization import serialize_graph

# Create a logger
logger = get_logger(__name__)

# Log messages
logger.info("Processing dataset")
logger.error("Error in algorithm execution", exc_info=True)

# Serialize a graph
graph_data = serialize_graph(graph)
```

## Related Documentation

- [Core Module](../core/README.md)
- [Database Module](../db/README.md)
- [App Module](../app/README.md)
