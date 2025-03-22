# Application Pages

This directory contains the different pages that make up the Causal Discovery Platform workflow.

## Overview

Each page represents a step in the causal discovery process, guiding users from data loading through discovery, refinement, and analysis.

## Pages

### 1. Guide (`00_Guide.py`)

The introduction to the platform that explains:
- Key concepts of causal discovery
- How to use the platform effectively
- A guide to the workflow

### 2. Data Loading (`01_Data_Loading.py`)

Allows users to:
- Upload custom datasets (CSV, Excel, Parquet)
- Select from sample datasets
- Configure data import settings
- View initial data summary

### 3. Data Exploration (`02_Data_Exploration.py`)

Enables users to:
- Explore data distributions
- Handle missing values
- Visualize correlations
- Prepare data for causal discovery
- Generate data profiles

### 4. Causal Discovery (`03_Causal_Discovery.py`)

The core page where users:
- Configure algorithm judgments
- Select appropriate algorithms
- Execute causal discovery algorithms
- View the discovered causal graph
- Compare results from different algorithms
- See AI-powered recommendations

### 5. Graph Refinement (`04_Graph_Refinement.py`)

Allows users to:
- Edit causal graphs
- Add domain knowledge
- Validate with LLM assistance
- Detect hidden variables
- Compare original and refined graphs

### 6. Analysis & Explanation (`05_Analysis_Explanation.py`)

Provides tools to:
- Interpret discovered causal relationships
- Perform counterfactual analysis
- Generate explanations at different technical levels
- Export results and findings

### 7. Settings (`06_Settings.py`)

Configuration page for:
- API keys for LLM services
- UI preferences
- Performance settings
- Model selection

### 8. Causal Playground (`07_Causal_Playground.py`)

A sandbox environment to:
- Experiment with causal concepts
- Create and simulate causal models
- Test interventions
- Learn through interactive examples

### 9. Knowledge Base (`08_Knowledge_Base.py`)

Resource center providing:
- Documentation on algorithms
- Explanation of causal concepts
- Guides and tutorials
- Reference information

## Navigation

The pages are numbered to indicate the typical workflow sequence, but users can navigate freely between pages once data is loaded.

## Related Documentation

- [App Components](../components/README.md)
- [App Module Overview](../README.md)
- [Core Algorithms](../../core/algorithms/README.md)
