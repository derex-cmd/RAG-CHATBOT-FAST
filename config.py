# ====================================================================
# CONFIGURATION MODULE
# File name: config.py
# ====================================================================

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Model configuration settings"""
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "google/flan-t5-base"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.15

@dataclass
class RetrievalConfig:
    """Retrieval configuration settings"""
    top_k_retrieve: int = 6
    similarity_threshold: float = 0.3
    max_context_length: int = 3000
    query_expansion_limit: int = 2

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    chroma_path: str = "./chroma_db"
    collection_name: str = "pdf_text_data"

@dataclass
class UIConfig:
    """UI configuration settings"""
    page_title: str = "🎓 FAST NUCES AI Assistant"
    page_icon: str = "🤖"
    theme_color: str = "#667eea"
    max_chat_history: int = 50

@dataclass
class PerformanceConfig:
    """Performance configuration settings"""
    use_gpu: bool = True
    cache_size: int = 100
    max_response_time: float = 30.0

@dataclass
class AnalyticsConfig:
    """Analytics configuration settings"""
    enable_analytics: bool = True
    max_log_entries: int = 1000

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.retrieval = RetrievalConfig()
        self.database = DatabaseConfig()
        self.ui = UIConfig()
        self.performance = PerformanceConfig()
        self.analytics = AnalyticsConfig()
        
        # Load environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Model configuration
        self.model.embedding_model = os.getenv("EMBEDDING_MODEL", self.model.embedding_model)
        self.model.llm_model = os.getenv("LLM_MODEL", self.model.llm_model)
        self.model.chunk_size = int(os.getenv("CHUNK_SIZE", self.model.chunk_size))
        self.model.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", self.model.chunk_overlap))
        self.model.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", self.model.max_new_tokens))
        self.model.temperature = float(os.getenv("TEMPERATURE", self.model.temperature))
        self.model.top_p = float(os.getenv("TOP_P", self.model.top_p))
        self.model.repetition_penalty = float(os.getenv("REPETITION_PENALTY", self.model.repetition_penalty))
        
        # Retrieval configuration
        self.retrieval.top_k_retrieve = int(os.getenv("TOP_K_RETRIEVE", self.retrieval.top_k_retrieve))
        self.retrieval.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", self.retrieval.similarity_threshold))
        self.retrieval.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", self.retrieval.max_context_length))
        
        # Database configuration
        self.database.chroma_path = os.getenv("CHROMA_PATH", self.database.chroma_path)
        self.database.collection_name = os.getenv("COLLECTION_NAME", self.database.collection_name)
        
        # UI configuration
        self.ui.page_title = os.getenv("PAGE_TITLE", self.ui.page_title)
        self.ui.page_icon = os.getenv("PAGE_ICON", self.ui.page_icon)
        self.ui.theme_color = os.getenv("THEME_COLOR", self.ui.theme_color)
        
        # Performance configuration
        self.performance.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        self.performance.cache_size = int(os.getenv("CACHE_SIZE", self.performance.cache_size))
        
        # Analytics configuration
        self.analytics.enable_analytics = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
        self.analytics.max_log_entries = int(os.getenv("MAX_LOG_ENTRIES", self.analytics.max_log_entries))
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        
        errors = []
        
        # Validate model configuration
        if self.model.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.model.temperature < 0 or self.model.temperature > 2:
            errors.append("temperature must be between 0 and 2")
        
        if self.model.top_p <= 0 or self.model.top_p > 1:
            errors.append("top_p must be between 0 and 1")
        
        # Validate retrieval configuration
        if self.retrieval.top_k_retrieve <= 0:
            errors.append("top_k_retrieve must be positive")
        
        if self.retrieval.similarity_threshold < 0 or self.retrieval.similarity_threshold > 1:
            errors.append("similarity_threshold must be between 0 and 1")
        
        # Validate performance configuration
        if self.performance.cache_size < 0:
            errors.append("cache_size must be non-negative")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global configuration instance
config = ConfigManager()

# Validate configuration on import
if not config.validate():
    print("⚠️ Configuration validation failed. Please check your settings.")

# Application deployment settings
DEPLOYMENT_SETTINGS = {
    "default_port": 8501,
    "default_host": "localhost",
    "auto_open_browser": True,
    "enable_cors": False,
    "max_upload_size": 200  # MB
}

# Example usage:
if __name__ == "__main__":
    print("Configuration Manager")
    print("=" * 40)
    
    print(f"Model: {config.model.llm_model}")
    print(f"Embeddings: {config.model.embedding_model}")
    print(f"Database: {config.database.chroma_path}")
    print(f"UI Title: {config.ui.page_title}")
    
    print(f"\nValidation Result: {'✅ PASSED' if config.validate() else '❌ FAILED'}")