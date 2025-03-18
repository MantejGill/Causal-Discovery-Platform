"""
Configuration settings for the causal discovery platform.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class LLMSettings:
    """Settings for LLM integration."""
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000
    

@dataclass
class AlgorithmSettings:
    """Settings for causal discovery algorithms."""
    # Default timeout in seconds for algorithm execution
    default_timeout: int = 300
    
    # Default parameters for specific algorithms
    pc_fisher_z_params: Dict[str, Any] = None
    pc_chi_square_params: Dict[str, Any] = None
    pc_kci_params: Dict[str, Any] = None
    fci_fisher_z_params: Dict[str, Any] = None
    ges_bic_params: Dict[str, Any] = None
    lingam_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default parameters if not provided."""
        if self.pc_fisher_z_params is None:
            self.pc_fisher_z_params = {'indep_test': 'fisher_z', 'alpha': 0.05}
        
        if self.pc_chi_square_params is None:
            self.pc_chi_square_params = {'indep_test': 'chi_square', 'alpha': 0.05}
        
        if self.pc_kci_params is None:
            self.pc_kci_params = {'indep_test': 'kci', 'alpha': 0.05}
        
        if self.fci_fisher_z_params is None:
            self.fci_fisher_z_params = {'indep_test': 'fisher_z', 'alpha': 0.05}
        
        if self.ges_bic_params is None:
            self.ges_bic_params = {'score_func': 'local_score_BIC'}
        
        if self.lingam_params is None:
            self.lingam_params = {}


@dataclass
class DatabaseSettings:
    """Settings for database connections."""
    use_db: bool = False
    db_url: str = "sqlite:///data/causal_discovery.db"
    pool_size: int = 5
    max_overflow: int = 10
    

@dataclass
class VisualizationSettings:
    """Settings for visualizations."""
    default_theme: str = "light"
    color_scales: Dict[str, List[str]] = None
    default_node_size: int = 15
    default_edge_width: float = 1.5
    
    def __post_init__(self):
        """Initialize default color scales if not provided."""
        if self.color_scales is None:
            self.color_scales = {
                "light": ["#2c6fbb", "#4b8bbf", "#66a3ff", "#99c2ff", "#cce0ff"],
                "dark": ["#0052cc", "#0066cc", "#0080ff", "#3399ff", "#66b3ff"]
            }


@dataclass
class Settings:
    """Global application settings."""
    # Application metadata
    app_name: str = "LLM-Augmented Causal Discovery"
    version: str = "0.1.0"
    debug: bool = os.environ.get("DEBUG", "0") == "1"
    
    # Component settings
    llm: LLMSettings = None
    algorithms: AlgorithmSettings = None
    database: DatabaseSettings = None
    visualization: VisualizationSettings = None
    
    # File paths
    data_dir: str = "data"
    sample_data_dir: str = "data/samples"
    
    def __post_init__(self):
        """Initialize component settings if not provided."""
        if self.llm is None:
            self.llm = LLMSettings(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        
        if self.algorithms is None:
            self.algorithms = AlgorithmSettings()
        
        if self.database is None:
            self.database = DatabaseSettings()
        
        if self.visualization is None:
            self.visualization = VisualizationSettings()


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings object
    """
    return settings


def load_settings_from_env():
    """Load settings from environment variables."""
    # Load LLM settings
    if os.environ.get("LLM_PROVIDER"):
        settings.llm.provider = os.environ.get("LLM_PROVIDER")
    
    if os.environ.get("OPENAI_API_KEY"):
        settings.llm.api_key = os.environ.get("OPENAI_API_KEY")
    
    if os.environ.get("LLM_MODEL"):
        settings.llm.model = os.environ.get("LLM_MODEL")
    
    # Load database settings
    if os.environ.get("DATABASE_URL"):
        settings.database.db_url = os.environ.get("DATABASE_URL")
        settings.database.use_db = True
    
    # Load visualization settings
    if os.environ.get("DEFAULT_THEME"):
        settings.visualization.default_theme = os.environ.get("DEFAULT_THEME")
    
    # Set debug mode
    settings.debug = os.environ.get("DEBUG", "0") == "1"


# Load settings from environment when module is imported
load_settings_from_env()