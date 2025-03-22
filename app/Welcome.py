import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

# Import core modules
from core.llm.factory import LLMFactory

# Load environment variables
load_dotenv()

# Initialize session state variables if they don't exist
for var in ['data_loaded', 'df', 'metadata', 'data_profile', 'preprocessor', 
            'causal_graphs', 'current_graph', 'refined_graph', 'llm_adapter', 'theme',
            'current_judgments', 'algorithm_suggestions', 'api_keys', 'llm_settings']:
    if var not in st.session_state:
        if var == 'data_loaded':
            st.session_state[var] = False
        elif var == 'causal_graphs':
            st.session_state[var] = {}
        elif var == 'theme':
            st.session_state[var] = "light"
        elif var == 'api_keys':
            st.session_state[var] = {
                "openai": os.getenv("OPENAI_API_KEY", ""),
                "openrouter": os.getenv("OPENROUTER_API_KEY", "")
            }
        elif var == 'llm_settings':
            st.session_state[var] = {
                "provider": "openrouter",  # Default to OpenRouter
                "openai_model": "gpt-4o",
                "openrouter_model": "deepseek/deepseek-r1:free",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        else:
            st.session_state[var] = None

# Try to initialize LLM adapter if settings are available
def initialize_llm_adapter_if_needed():
    if st.session_state.llm_adapter is None and all([
        st.session_state.llm_settings is not None,
        st.session_state.api_keys is not None
    ]):
        # Create configuration for LLM factory
        config = {
            "provider": st.session_state.llm_settings["provider"],
            "openai_model": st.session_state.llm_settings.get("openai_model", "gpt-4o"),
            "openrouter_model": st.session_state.llm_settings.get("openrouter_model", "deepseek/deepseek-r1:free"),
            "temperature": st.session_state.llm_settings.get("temperature", 0.7),
            "max_tokens": st.session_state.llm_settings.get("max_tokens", 1000),
            "api_keys": st.session_state.api_keys
        }
        
        # Try to create adapter
        try:
            st.session_state.llm_adapter = LLMFactory.create_adapter(config)
        except Exception as e:
            st.error(f"Error initializing LLM adapter: {str(e)}")


# Initialize LLM adapter if needed
initialize_llm_adapter_if_needed()

# Main app
st.set_page_config(
    page_title="LLM-Augmented Causal Discovery",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 LLM-Augmented Causal Discovery Framework")

# Introduction
st.markdown("""
This application integrates traditional causal discovery algorithms with Large Language Models (LLMs) to:

1. Discover causal relationships from data
2. Refine causal graphs using domain knowledge
3. Explain causal relationships in natural language
4. Identify potential hidden variables and confounders

Navigate using the sidebar to explore the different functionalities.
""")

# Feature highlights
st.subheader("Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🧠 Automatic Algorithm Selection
    
    The framework analyzes your data characteristics and recommends appropriate causal discovery algorithms.
    """)

with col2:
    st.markdown("""
    ### 🌐 LLM-Enhanced Refinement
    
    Leverage large language models to refine causal graphs based on domain knowledge.
    
    *Now with DeepSeek R1 support!*
    """)

with col3:
    st.markdown("""
    ### 📊 Interactive Visualization
    
    Explore and interact with causal graphs through intuitive visualizations.
    """)

# Getting started
st.subheader("Getting Started")

st.markdown("""
1. Start by loading your data in the **Data Loading** page
2. Explore and preprocess your data in the **Data Exploration** page
3. Discover causal relationships in the **Causal Discovery** page
4. Refine your causal graph in the **Graph Refinement** page
5. Analyze and explain relationships in the **Analysis & Explanation** page
""")

# LLM Status
st.subheader("LLM Integration Status")

# Show LLM information based on the configuration
if st.session_state.llm_adapter is not None:
    provider_info = LLMFactory.get_provider_info()
    current_provider = st.session_state.llm_settings["provider"]
    
    if current_provider == "openai":
        model = st.session_state.llm_settings["openai_model"]
        st.success(f"✅ Connected to OpenAI API using model: {model}")
    elif current_provider == "openrouter":
        model = st.session_state.llm_settings["openrouter_model"]
        st.success(f"✅ Connected to OpenRouter using model: {model}")
        
        # Add special note for DeepSeek R1
        if "deepseek-r1" in model:
            st.info("""
            **DeepSeek R1** is actively integrated! This model offers:
            - Performance comparable to OpenAI's o1
            - 163,840 token context window
            - MIT license (fully open source)
            - 671B parameters with 37B active during inference
            """)
    
else:
    st.warning("""
    ⚠️ LLM integration not configured. Please go to the **Settings** page to set up your LLM provider.
    
    This application supports:
    - OpenAI (GPT-4o, GPT-4o Mini, etc.)
    - OpenRouter's DeepSeek R1 (free, open-source model)
    """)

# Bottom section with additional info if needed
st.divider()

# Add a note about DeepSeek R1 if not already configured
if st.session_state.llm_adapter is None or st.session_state.llm_settings["provider"] != "openrouter":
    st.markdown("""
    ### 🆕 Now Supporting DeepSeek R1!
    
    We've added support for DeepSeek R1, a powerful open-source model available for free through OpenRouter.
    
    DeepSeek R1 features:
    - Performance comparable to OpenAI's o1
    - 163,840 token context window
    - MIT license for commercial use
    - Free access via OpenRouter
    
    Configure it in the **Settings** page.
    """)

# Apply theme if dark mode is selected
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)