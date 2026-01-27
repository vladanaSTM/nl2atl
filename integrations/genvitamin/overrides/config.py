import os
from typing import List, Optional

from pydantic_settings import BaseSettings


def _available_mem_gb() -> Optional[float]:
    """Best-effort available memory in GiB using /proc/meminfo; returns None on failure."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            fields = {
                line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
                for line in f
            }
        if "MemAvailable" in fields:
            kb = float(fields["MemAvailable"].split()[0])
            return kb / (1024 * 1024)
    except Exception:
        return None
    return None


def _default_ollama_model() -> str:
    """
    Select a model based on available memory:
    - If < ~6 GiB available, pick a smaller 3B model for speed.
    - Otherwise, keep the 8B default for better quality.
    Environment variables still override this default via BaseSettings.
    """
    mem_gb = _available_mem_gb()
    if mem_gb is not None and mem_gb < 6:
        return "llama3.2:3b"
    return "llama3:8b-instruct-q4_0"


class Settings(BaseSettings):
    # API Configuration
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "VITAMIN API"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # CORS Configuration
    # Allowed origins for cross-origin requests:
    # - 3000: React/Next.js frontend dev server
    # - 5173: Vite frontend dev server
    # - 8501: Streamlit application (kept for current Streamlit UI)
    CORS_ORIGINS: str = (
        "http://localhost:3000,http://localhost:5173,http://localhost:8501"
    )

    # Ollama AI Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_MAX_TOKENS: int = (
        2048  # Maximum tokens the model can GENERATE per response (output limit)
    )

    # Model Selection - Optimized for your available models
    # Available models categorized by size:
    #   Small (1-3B): llama3.2:3b, qwen2.5:3b, gemma2:2b
    #     - Fast, low memory (~2GB), good for simple tasks
    #   Medium (6-8B): llama3:8b, llama3.1:8b, deepseek-coder:6.7b, mistral:7b, llama2:7b, codellama:7b
    #     - Balanced quality/memory (~2.5-4GB quantized), good for most tasks
    #   Large (13B+): llama2:13b, codellama:13b, qwen2.5:14b (if available)
    #     - Best quality but high memory (~7GB+), use only for complex tasks
    # Strategy: Use smaller models by default, larger models only when needed
    OLLAMA_MODEL: str = _default_ollama_model()
    OLLAMA_CODE_MODEL: str = "deepseek-coder"  # Best for code (~2.4GB)
    OLLAMA_REASONING_MODEL: str = "qwen2.5:3b"  # Best for reasoning (~2GB)
    OLLAMA_LARGE_MODEL: str = "llama3.1:8b"  # Complex tasks (~4.7GB, lazy-loaded)

    # Ollama Memory & Performance
    OLLAMA_NUM_CTX: int = 4096  # Total context window size (input + output combined)
    OLLAMA_MAX_LOADED_MODELS: int = 1  # Max models in memory (0 = unlimited)
    OLLAMA_NUM_PARALLEL: int = 1  # Parallel requests
    OLLAMA_KEEP_ALIVE: str = (
        "10m"  # Keep models loaded (format: "5m", "10m", "1h", "0")
    )

    # Model Selection Thresholds (query length in characters)
    OLLAMA_QUERY_LENGTH_MEDIUM: int = 200
    OLLAMA_QUERY_LENGTH_LONG: int = 500
    OLLAMA_QUERY_LENGTH_VERY_LONG: int = 1000

    # Ollama API Timeouts (seconds)
    OLLAMA_API_TIMEOUT: float = 2.0
    OLLAMA_PULL_TIMEOUT: float = 300.0
    OLLAMA_STREAM_TIMEOUT: float = 300.0  # Timeout for streaming responses (5 minutes)

    # Fallback Context Window (tokens)
    OLLAMA_FALLBACK_NUM_CTX: int = 2048
    OLLAMA_FALLBACK_NUM_PREDICT: int = 256  # Max tokens for fallback models

    # Fallback Configuration
    FALLBACK_ENABLED: bool = True
    FALLBACK_MAX_ATTEMPTS: int = 5  # Max fallback attempts before giving up
    FALLBACK_CIRCUIT_BREAKER_ENABLED: bool = True
    FALLBACK_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    FALLBACK_CIRCUIT_BREAKER_TIMEOUT: float = 60.0
    FALLBACK_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = (
        2  # Successes needed to close circuit
    )
    FALLBACK_HEALTH_CHECK_ENABLED: bool = True
    FALLBACK_HEALTH_CHECK_TIMEOUT: float = 5.0
    FALLBACK_METRICS_ENABLED: bool = True

    # Response Streaming Configuration
    STREAM_WORD_DELAY: float = (
        0.05  # Delay between words in fallback streaming (seconds)
    )
    STREAM_ERROR_DELAY: float = 0.02  # Delay for error messages (seconds)

    # NL2ATL Integration (optional)
    NL2ATL_URL: Optional[str] = None
    NL2ATL_MODEL: str = "qwen-3b"
    NL2ATL_FEW_SHOT: bool = True
    NL2ATL_NUM_FEW_SHOT: Optional[int] = None
    NL2ATL_ADAPTER: Optional[str] = "qwen-3b_finetuned_few_shot/final"
    NL2ATL_MAX_NEW_TOKENS: int = 128
    NL2ATL_TIMEOUT: int = 30

    # ChromaDB Configuration
    CHROMADB_URL: str = "http://localhost:8000"
    CHROMADB_PERSIST_DIR: str = "./chroma_db"
    CHROMADB_COLLECTION_NAME: str = "vitamin_knowledge"

    # Embeddings Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG Configuration
    RAG_DEFAULT_K: int = 3  # Default number of documents to retrieve
    RAG_SIMILARITY_THRESHOLD: float = (
        1.0  # Maximum distance score to accept (lower = more similar)
    )

    # File Upload Settings
    UPLOAD_DIR: str = "./data/tmp"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024

    # Model Checker Configuration
    MODEL_CHECKER_TIMEOUT: int = 30
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600

    # Model Manager Configuration
    MODEL_REGISTRATION_DELAY: float = 1.0  # seconds to wait after pulling model
    ERROR_TEXT_LIMIT: int = 200  # characters to show in error messages
    MODEL_LIST_DISPLAY_LIMIT: int = 5  # number of models to show in error messages

    # Security (for production) - Currently not used
    # SECRET_KEY: Optional[str] = None
    # JWT_SECRET: Optional[str] = None
    # ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert CORS_ORIGINS from string to list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


settings = Settings()
