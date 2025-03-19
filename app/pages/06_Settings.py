import streamlit as st
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Import core modules
from core.llm.adapter import LLMAdapter
from core.llm.openai_adapter import OpenAIAdapter
from core.llm.openrouter_adapter import OpenRouterAdapter

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
        "provider": "openai",
        "openai_model": "gpt-4o",
        "openrouter_model": "deepseek/deepseek-r1:free",
        "temperature": 0.7,
        "max_tokens": 1000
    }

# Define the function to save settings to a file
def save_settings_to_file():
    """Save settings to a JSON file (excluding API keys for security)"""
    settings_to_save = {
        "theme": st.session_state.theme,
        "llm_settings": st.session_state.llm_settings
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
            
            st.success("Settings loaded successfully!")
        except Exception as e:
            st.error(f"Error loading settings: {str(e)}")
    else:
        st.info("No saved settings found. Using default settings.")

# Function to initialize LLM adapter based on settings
def initialize_llm_adapter():
    """Initialize LLM adapter based on current settings"""
    provider = st.session_state.llm_settings["provider"]
    
    if provider == "openai":
        api_key = st.session_state.api_keys["openai"]
        if not api_key:
            st.error("OpenAI API key not set. Please enter your API key.")
            return None
        
        model = st.session_state.llm_settings["openai_model"]
        temperature = st.session_state.llm_settings["temperature"]
        max_tokens = st.session_state.llm_settings["max_tokens"]
        
        try:
            adapter = OpenAIAdapter(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            st.success(f"Successfully initialized OpenAI adapter with model: {model}")
            return adapter
        except Exception as e:
            st.error(f"Error initializing OpenAI adapter: {str(e)}")
            return None
    
    elif provider == "openrouter":
        api_key = st.session_state.api_keys["openrouter"]
        if not api_key:
            st.error("OpenRouter API key not set. Please enter your API key.")
            return None
        
        model = st.session_state.llm_settings["openrouter_model"]
        temperature = st.session_state.llm_settings["temperature"]
        max_tokens = st.session_state.llm_settings["max_tokens"]
        
        try:
            adapter = OpenRouterAdapter(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            st.success(f"Successfully initialized OpenRouter adapter with model: {model}")
            return adapter
        except Exception as e:
            st.error(f"Error initializing OpenRouter adapter: {str(e)}")
            return None
    
    else:
        st.error(f"Unknown provider: {provider}")
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
        format_func=lambda x: {"openai": "OpenAI", "openrouter": "OpenRouter (DeepSeek R1)"}[x],
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
        st.json({
            "llm_settings": st.session_state.llm_settings,
            "theme": st.session_state.theme,
            "llm_adapter_initialized": st.session_state.llm_adapter is not None
        })

# Footer with additional info
st.divider()
st.markdown("""
### About LLM Integration

This application uses Large Language Models (LLMs) to enhance causal discovery by:
- Refining causal graphs based on domain knowledge
- Explaining causal relationships in natural language
- Identifying potential hidden variables

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