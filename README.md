# LLM-Augmented Causal Discovery Framework

A Python-based platform for causal discovery that integrates Large Language Models with established causal discovery algorithms.

## Overview

This platform enables domain experts without specialized statistical training to:

1. Load and explore datasets
2. Run appropriate causal discovery algorithms based on data characteristics
3. Refine causal graphs using Large Language Models (LLMs)
4. Analyze and interpret causal relationships
5. Perform counterfactual analysis

## Features

- **Data Exploration**: Visualize distributions, correlations, and scatter plots
- **Algorithm Selection**: Automatically choose appropriate causal discovery algorithms based on data characteristics
- **Causal Discovery**: Execute algorithms from the CausalLearn library with sensible defaults
- **Ensemble Integration**: Combine results from multiple algorithms for more robust causal graphs
- **LLM Refinement**: Use LLMs to validate, refine, and explain causal relationships
- **Hidden Variable Discovery**: Hypothesize potential latent confounders
- **Counterfactual Analysis**: Estimate effects of interventions
- **Multi-level Explanations**: Generate explanations at different technical levels

## System Architecture

![System Architecture](docs/architecture-diagram.png)

## Getting Started

### Prerequisites

- Docker and Docker Compose (recommended)
- Python 3.9+ (for local installation)
- OpenAI API key (for LLM integration)

### Installation with Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/causal-discovery-platform.git
   cd causal-discovery-platform
   ```

2. Create a `.env` file with your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

3. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/causal-discovery-platform.git
   cd causal-discovery-platform
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-api-key-here  # On Windows, use: set OPENAI_API_KEY=your-api-key-here
   ```

5. Run the application:
   ```bash
   streamlit run app/Welcome.py
   ```

## Usage

1. **Data Loading**:
   - Upload your dataset (CSV, Excel, Parquet)
   - Or select from sample datasets (Asia, Sachs, Insurance)

2. **Data Exploration**:
   - Visualize feature distributions
   - Examine correlations
   - Create scatter plots
   - Analyze missing values

3. **Causal Discovery**:
   - Select algorithms based on data characteristics
   - Execute algorithms individually or in ensemble
   - Visualize the resulting causal graph

4. **LLM Refinement**:
   - Configure LLM connection (OpenAI)
   - Refine the causal graph with LLM validation
   - Discover potential hidden variables
   - Compare original and refined graphs

5. **Analysis & Interpretation**:
   - Analyze causal graph structure
   - Perform counterfactual analysis
   - Generate explanations at different technical levels

## Supported Algorithms

The platform supports a variety of causal discovery algorithms from the CausalLearn library:

- PC Algorithm with variations:
  - PC + Fisher Z (continuous Gaussian data)
  - PC + Chi-square (discrete data)
  - PC + KCI (nonlinear data)

- Fast Causal Inference (FCI) for latent variables:
  - FCI + Fisher Z
  - FCI + Chi-square
  - FCI + KCI

- Greedy Equivalence Search (GES):
  - GES + BIC (continuous Gaussian data)
  - GES + BDeu (discrete data)

- LiNGAM (Linear Non-Gaussian Acyclic Models)
- GIN (Greedy Interventional Network)
- RCD (Recursive Causal Discovery)
- CAM-UV (Causal Additive Model with Unobserved Variables)

## LLM Integration

The platform currently supports OpenAI's models (GPT-3.5, GPT-4) for LLM-based refinement. The LLM is used to:

1. Validate causal edges based on statistical evidence and domain knowledge
2. Hypothesize potential hidden variables
3. Generate natural language explanations of causal relationships
4. Provide context-specific insights about the causal structure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The platform uses the [CausalLearn](https://github.com/cmu-phil/causallearn) library for causal discovery algorithms
- LLM integration is powered by OpenAI's API
- Visualization components use Plotly and NetworkX