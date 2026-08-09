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

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    Using a cached factory (rather than a module-level singleton constructed
    at import time) makes settings easy to override in tests via
    `get_settings.cache_clear()` + monkeypatched environment variables.
    """
    return Settings()