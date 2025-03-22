import streamlit as st
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Import core modules
from core.llm.adapter import LLMAdapter
from core.llm.openai_adapter import OpenAIAdapter
from core.llm.openrouter_adapter import OpenRouterAdapter
from core.llm.factory import LLMFactory

# Try to import RAG components (optional)
try:
    from core.rag.rag_manager import RAGManager
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize session state variables if they don't exist
if 'llm_adapter' not in st.session_state:
    st.session_state.llm_adapter = None
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "openrouter": os.getenv("OPENROUTER_API_KEY", "")
    }
if 'llm_settings' not in st.session_state:
    st.session_state.llm_settings = {
        "provider": "openrouter",  # Default to OpenRouter
        "openai_model": "gpt-4o",
        "openrouter_model": "deepseek/deepseek-r1:free",
        "temperature": 0.7,
        "max_tokens": 1000
    }
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = False
if 'rag_manager' not in st.session_state:
    st.session_state.rag_manager = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = "huggingface"  # Default to HuggingFace

# Define the function to save settings to a file
def save_settings_to_file():
    """Save settings to a JSON file (excluding API keys for security)"""
    settings_to_save = {
        "theme": st.session_state.theme,
        "llm_settings": st.session_state.llm_settings,
        "rag_enabled": st.session_state.rag_enabled,
        "embeddings_model": st.session_state.embeddings_model
    }
    
    # Create settings directory if it doesn't exist
    settings_dir = Path("./settings")
    settings_dir.mkdir(exist_ok=True)
    
    # Save settings
    with open(settings_dir / "app_settings.json", "w") as f:
        json.dump(settings_to_save, f, indent=2)
    
    st.success("Settings saved successfully!")

# Define the function to load settings from a file
def load_settings_from_file():
    """Load settings from a JSON file"""
    settings_file = Path("./settings/app_settings.json")
    
    if settings_file.exists():
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)
            
            # Update session state with loaded settings
            if "theme" in settings:
                st.session_state.theme = settings["theme"]
            if "llm_settings" in settings:
                st.session_state.llm_settings = settings["llm_settings"]
            if "rag_enabled" in settings:
                st.session_state.rag_enabled = settings["rag_enabled"]
            if "embeddings_model" in settings:
                st.session_state.embeddings_model = settings["embeddings_model"]
            
            st.success("Settings loaded successfully!")
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
    else:
        st.info("No saved settings found. Using default settings.")

# Function to initialize LLM adapter based on settings
def initialize_llm_adapter():
    """Initialize LLM adapter based on current settings"""
    # Create configuration for LLM factory
    config = {
        "provider": st.session_state.llm_settings["provider"],
        "openai_model": st.session_state.llm_settings.get("openai_model", "gpt-4o"),
        "openrouter_model": st.session_state.llm_settings.get("openrouter_model", "deepseek/deepseek-r1:free"),
        "temperature": st.session_state.llm_settings.get("temperature", 0.7),
        "max_tokens": st.session_state.llm_settings.get("max_tokens", 1000),
        "api_keys": st.session_state.api_keys,
        "use_rag": st.session_state.get("rag_enabled", False),
        "rag_manager": st.session_state.get("rag_manager", None),
        "rag_db_dir": "./data/rag_db",
        "embeddings_model": st.session_state.get("embeddings_model", "huggingface")  # Default to HuggingFace
    }
    
    # Try to create adapter using factory
    try:
        adapter = LLMFactory.create_adapter(config)
        
        if adapter:
            st.success(f"Successfully initialized {adapter.get_name()}")
            return adapter
        else:
            st.error("Failed to initialize LLM adapter")
            return None
    except Exception as e:
        st.error(f"Error initializing LLM adapter: {str(e)}")
        return None

# Main app
st.title("Settings")

tab1, tab2, tab3 = st.tabs(["LLM Settings", "UI Settings", "Advanced"])

with tab1:
    st.header("LLM Provider Settings")
    
    # LLM Provider selection
    provider = st.selectbox(
        "LLM Provider",
        options=["openai", "openrouter"],
        index=0 if st.session_state.llm_settings["provider"] == "openai" else 1,
        format_func=lambda x: {"openai": "OpenAI", "openrouter": "OpenRouter"}[x],
        help="Select the LLM provider to use for causal graph refinement and explanations"
    )
    
    # Update provider in session state
    st.session_state.llm_settings["provider"] = provider
    
    # Provider-specific settings
    if provider == "openai":
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_keys["openai"],
            type="password",
            help="Your OpenAI API key. This is stored in your session and not saved to disk."
        )
        st.session_state.api_keys["openai"] = openai_api_key
        
        # OpenAI Model selection
        openai_model = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"].index(
                st.session_state.llm_settings["openai_model"]
            ) if st.session_state.llm_settings["openai_model"] in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"] else 0,
            help="Select the OpenAI model to use"
        )
        st.session_state.llm_settings["openai_model"] = openai_model
    
    elif provider == "openrouter":
        # OpenRouter API Key
        openrouter_api_key = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.api_keys["openrouter"],
            type="password",
            help="Your OpenRouter API key. This is stored in your session and not saved to disk."
        )
        st.session_state.api_keys["openrouter"] = openrouter_api_key
        
        openrouter_model = st.selectbox(
            "OpenRouter Model",
            options=[
                # Gemini models
                "google/gemini-2.0-flash-thinking-exp:free",  # Gemini 2.0 Flash Thinking Experimental 01-21 (free)
                # DeepSeek models
                "deepseek/deepseek-r1:free", 
                "deepseek/deepseek-r1-distill-llama-70b",
                "deepseek/deepseek-r1-distill-qwen-32b", 
                "deepseek/deepseek-r1-distill-qwen-14b",
                # Claude models
                "anthropic/claude-3-opus:20240229",
                "anthropic/claude-3-sonnet:20240229",
                "anthropic/claude-3-haiku:20240307"
            ],
            index=0,  # Default to Gemini Flash
            help="Select the OpenRouter model to use"
        )
        st.session_state.llm_settings["openrouter_model"] = openrouter_model
        
        # Add information about available models
        st.info("""
        **Available Models:**
        - **Gemini 1.5 Flash**: Google's fast Gemini model, excellent for visualization explanations (FREE)
        - **DeepSeek R1**: Powerful open-source LLM with MIT license
        - **DeepSeek R1 Distill**: Smaller, faster versions based on Llama 3 and Qwen
        - **Claude 3 Opus**: Most capable Claude model for complex tasks
        - **Claude 3 Sonnet**: Balanced Claude model for most use cases
        - **Claude 3 Haiku**: Fastest Claude model for simpler tasks

        Gemini 1.5 Flash is recommended for visualization explanations due to its strong understanding of data and visuals.
        """)
    
    # Common LLM settings
    st.subheader("Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.llm_settings["temperature"],
            step=0.1,
            help="Higher values produce more random outputs"
        )
        st.session_state.llm_settings["temperature"] = temperature
    
    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=st.session_state.llm_settings["max_tokens"],
            step=100,
            help="Maximum number of tokens to generate"
        )
        st.session_state.llm_settings["max_tokens"] = max_tokens
    
    # RAG Integration
    st.header("RAG Integration")
    
    if RAG_AVAILABLE:
        rag_enabled = st.toggle(
            "Enable Retrieval-Augmented Generation",
            value=st.session_state.get("rag_enabled", False),
            help="Enhance LLMs with document retrieval from causal literature"
        )
        
        st.session_state.rag_enabled = rag_enabled
        
        if rag_enabled:
            st.markdown("""
            With RAG enabled, the LLM will use your knowledge base of causal literature to provide
            more accurate and domain-specific responses.
            
            To manage your knowledge base, go to the **Knowledge Base** page.
            """)
            
            # Embedding model selection
            embeddings_model = st.radio(
                "Embeddings Model",
                options=["huggingface", "openai"],
                index=0 if st.session_state.embeddings_model == "huggingface" else 1,
                help="Model to use for document embeddings. HuggingFace doesn't require an API key, but OpenAI may provide better quality for some use cases."
            )
            st.session_state.embeddings_model = embeddings_model
            
            # OpenAI model warning
            if embeddings_model == "openai" and not st.session_state.api_keys.get("openai"):
                st.warning("You selected OpenAI embeddings, but no OpenAI API key is provided. Please enter an OpenAI API key or switch to HuggingFace embeddings.")
            
            # Show RAG status if available
            if "rag_manager" in st.session_state and st.session_state.rag_manager:
                try:
                    is_ready = st.session_state.rag_manager.is_ready
                    status = st.session_state.rag_manager.get_status()
                    
                    if is_ready:
                        st.success(f"RAG system is ready with {status['document_count']} documents ({status['chunk_count']} chunks)")
                    else:
                        st.warning("Knowledge base is empty. Please add documents in the Knowledge Base page.")
                    
                    # Add link to Knowledge Base page
                    if st.button("Go to Knowledge Base"):
                        # Use Streamlit navigation to go to the Knowledge Base page
                        st.experimental_set_query_params(page="08_Knowledge_Base.py")
                except Exception as e:
                    st.error(f"Error checking RAG status: {str(e)}")
    else:
        st.warning("""
        RAG components are not available. Make sure all dependencies are installed:
        ```
        pip install langchain langchain-community chromadb sentence-transformers pypdf2
        ```
        """)
    
    # Initialize LLM button
    if st.button("Initialize LLM Adapter", key="initialize_llm"):
        st.session_state.llm_adapter = initialize_llm_adapter()
    
    st.subheader("Visualization Explanation Settings")

    col1, col2 = st.columns(2)

    with col1:
        explanation_detail_level = st.selectbox(
            "Default Detail Level",
            options=["beginner", "intermediate", "advanced"],
            index=1,  # Default to intermediate
            help="Select the default level of detail for visualization explanations"
        )
        st.session_state["explanation_detail_level"] = explanation_detail_level

    with col2:
        explanation_focus = st.selectbox(
            "Default Focus Area",
            options=["statistical", "causal", "domain"],
            index=0,  # Default to statistical
            help="Select the default focus area for visualization explanations"
        )
        st.session_state["explanation_focus"] = explanation_focus

    st.markdown("""
    ### About Visualization Explanations

    The AI-powered visualization explanation feature uses LLMs to provide insights into your data visualizations, helping you:

    - Understand what patterns are shown in charts and graphs
    - Interpret statistical relationships and significance
    - Identify potential causal connections
    - Discover insights you might have missed

    You can adjust the level of detail and focus area to suit your needs and expertise level.
    """)

with tab2:
    st.header("UI Settings")
    
    # Theme selection
    theme = st.selectbox(
        "Theme",
        options=["light", "dark"],
        index=0 if st.session_state.theme == "light" else 1,
        help="Select the application theme"
    )
    st.session_state.theme = theme
    
    # Example color preview based on theme
    if theme == "light":
        st.write("Light theme preview:")
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
            <h4 style="color: #0e1117;">This is how the light theme looks</h4>
            <p style="color: #262730;">Text appears in dark color on light background</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Dark theme preview:")
        st.markdown("""
        <div style="background-color: #0e1117; padding: 10px; border-radius: 5px;">
            <h4 style="color: #fafafa;">This is how the dark theme looks</h4>
            <p style="color: #e6e6e6;">Text appears in light color on dark background</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.header("Advanced Settings")
    
    # Save/Load settings
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Settings", help="Save current settings to a file (API keys are not saved)"):
            save_settings_to_file()
    
    with col2:
        if st.button("Load Settings", help="Load settings from a file"):
            load_settings_from_file()
    
    # Debug info
    if st.checkbox("Show Debug Information"):
        st.subheader("Session State")
        
        # Show LLM settings (excluding API keys)
        debug_info = {
            "llm_settings": st.session_state.llm_settings,
            "theme": st.session_state.theme,
            "llm_adapter_initialized": st.session_state.llm_adapter is not None,
            "rag_enabled": st.session_state.rag_enabled,
            "rag_available": RAG_AVAILABLE,
            "rag_manager_initialized": st.session_state.rag_manager is not None,
            "embeddings_model": st.session_state.embeddings_model
        }
        
        st.json(debug_info)

# Footer with additional info
st.divider()
st.markdown("""
### About LLM and RAG Integration

This application uses Large Language Models (LLMs) to enhance causal discovery by:
- Refining causal graphs based on domain knowledge
- Explaining causal relationships in natural language
- Identifying potential hidden variables

With Retrieval-Augmented Generation (RAG), you can:
- Enhance the LLM with knowledge from causal literature
- Get more accurate and domain-specific responses
- Ground responses in specific sources with proper citations

The LLM providers we support are:
- **OpenAI**: Commercial provider with powerful models like GPT-4
- **OpenRouter (DeepSeek R1)**: Free, open-source model with MIT license and performance comparable to OpenAI's o1
""")

# Check if we need to apply the theme
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)