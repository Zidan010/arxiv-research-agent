"""
Centralized application configuration.

Uses pydantic-settings so config is:
  - loaded from environment variables (with a local `.env` file for dev),
  - validated and type-checked at startup (fail fast on bad config),
  - a single source of truth injected wherever it's needed, rather than
    scattered `os.environ.get(...)` calls throughout the codebase.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- API ---------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Logging -------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    
    # --- Ingestion (arXiv) -----------------------------------------------------
    ARXIV_CATEGORY: str = "cs.AI"
    ARXIV_MAX_RESULTS: int = 10
    # arXiv's documented limit is 3 requests/second; default throttles to
    # comfortably below that (one request every 3 seconds).
    ARXIV_RATE_LIMIT_SECONDS: float = 3.0
    RAW_DATA_DIR: str = "data/raw"

    # --- Parsing -----------------------------------------------------------
    PROCESSED_DATA_DIR: str = "data/processed"
    # Target max characters per chunk (soft limit — a chunk only exceeds this
    # if a single paragraph, kept intact, is already longer on its own).
    PARSING_MAX_CHUNK_CHARS: int = 1200

    # --- Embeddings --------------------------------------------------------
    # Local sentence-transformers model name (downloaded once, cached by the
    # library — no API key required).
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32

    # --- Vector store --------------------------------------------------------
    VECTOR_STORE_DIR: str = "data/vector_store"
    # Default number of chunks retrieved per query, and the cap on how many
    # of those may come from a single paper (see FaissVectorStore.search).
    RETRIEVAL_TOP_K: int = 6
    RETRIEVAL_MAX_CHUNKS_PER_PAPER: int = 2

    # --- LLM (synthesis) -------------------------------------------------------
    LLM_PROVIDER: str = "groq"  # "groq" | "openai"
 
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
 
    LLM_MAX_TOKENS: int = 1024
    # Kept low (rather than a more "creative" default) since this system
    # synthesizes factual claims from retrieved research, not open-ended text.
    LLM_TEMPERATURE: float = 0.3 

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    Using a cached factory (rather than a module-level singleton constructed
    at import time) makes settings easy to override in tests via
    `get_settings.cache_clear()` + monkeypatched environment variables.
    """
    return Settings()