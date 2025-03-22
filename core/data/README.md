# Core Data Module

This directory contains the data handling functionality of the Causal Discovery Platform.

## Overview

The data module handles all aspects of data processing, including:

- Loading data from various sources and formats
- Preprocessing and cleaning data
- Profiling data for algorithm selection
- Handling missing values
- Feature transformation

## Key Components

### Data Loader (`loader.py`)

The Data Loader:

- Imports data from various file formats (CSV, Excel, Parquet)
- Handles different delimiters and file encodings
- Processes sample datasets
- Validates data integrity
- Performs initial data type inference

### Data Preprocessor (`preprocessor.py`)

The Data Preprocessor:

- Cleans and standardizes data
- Handles categorical variables
- Performs feature scaling and normalization
- Transforms data for specific algorithm requirements
- Filters variables based on user selection

### Data Profiler (`profiler.py`)

The Data Profiler analyzes datasets to:

- Determine statistical properties
- Identify data distributions
- Detect outliers
- Assess data suitability for different algorithms
- Generate judgments for algorithm selection

### Missing Data Handler (`missing_data.py`)

The Missing Data Handler:

- Detects missing values
- Provides different imputation strategies
- Recommends appropriate handling based on data characteristics
- Estimates the impact of missing data on causal discovery

## Usage

These components are typically used through the App's user interface, but can also be used programmatically:

```python
from core.data.loader import DataLoader
from core.data.preprocessor import DataPreprocessor
from core.data.profiler import DataProfiler

# Load data
loader = DataLoader()
df = loader.load_csv("data.csv")

# Preprocess data
preprocessor = DataPreprocessor(df)
preprocessor.handle_missing_values(method="mean")
preprocessor.normalize_data(method="standard")
processed_df = preprocessor.get_data()

# Profile data
profiler = DataProfiler()
profile = profiler.profile_data(processed_df)
```

## Related Documentation

- [Core Module Overview](../README.md)
- [Core Algorithms](../algorithms/README.md)
- [Data Loading Page](../../app/pages/README.md)
- [Data Exploration Page](../../app/pages/README.md)
