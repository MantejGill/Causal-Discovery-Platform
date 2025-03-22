# App Module

The App module contains the user interface components and pages of the Causal Discovery Platform.

## Overview

This module is built using Streamlit and provides an interactive web interface for the platform. It consists of:

- **Pages**: Different screens of the application that guide users through the causal discovery workflow
- **Components**: Reusable UI elements that provide specific functionality across pages

## Directory Structure

```
app/
├── Welcome.py           # The main entry point/welcome page
├── __init__.py
├── components/          # Reusable UI components 
└── pages/               # Application pages
```

## Pages

The application follows a sequential workflow represented by the following pages:

1. **Guide** (`00_Guide.py`): Introduction and guidance on using the platform
2. **Data Loading** (`01_Data_Loading.py`): Upload datasets or use sample data
3. **Data Exploration** (`02_Data_Exploration.py`): Visualize and understand your data
4. **Causal Discovery** (`03_Causal_Discovery.py`): Run causal discovery algorithms
5. **Graph Refinement** (`04_Graph_Refinement.py`): Refine discovered causal graphs
6. **Analysis & Explanation** (`05_Analysis_Explanation.py`): Interpret results
7. **Settings** (`06_Settings.py`): Configure application settings
8. **Causal Playground** (`07_Causal_Playground.py`): Experiment with causal concepts
9. **Knowledge Base** (`08_Knowledge_Base.py`): Access documentation and resources

Learn more about each page in the [Pages README](./pages/README.md).

## Components

The app includes various reusable UI components for:

- Data visualization
- Causal graph interaction
- Effect estimation
- Domain knowledge integration
- and more

Each component is designed to be modular and reusable across different pages. Learn more in the [Components README](./components/README.md).

## Related Modules

- [Core Module](../core/README.md): Contains the business logic and algorithms
- [Database Module](../db/README.md): Handles data persistence
- [Utils Module](../utils/README.md): Provides utility functions
