"""
Local embedding pipeline.

Design notes:
  - Uses a local sentence-transformers model rather than a hosted embeddings
    API. This removes an external dependency.
  - Kept as a class behind a small interface (not bare module-level
    functions) so it can be swapped for a different embedding backend later
    without changing any caller -- the same "pluggable behind an interface"
    pattern used for the LLM provider layer (app/llm).
  - Embeddings are L2-normalized at generation time so that cosine
    similarity search downstream can use a
    plain inner-product index rather than a more expensive similarity
    computation.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _load_model() -> SentenceTransformer:
    """
    Loads (and caches) the configured sentence-transformers model.

    Cached at module level via lru_cache because loading the model is
    expensive (weights are read from disk / downloaded and moved onto the
    inference device); every Embedder instance should share one loaded
    model rather than re-loading it.
    """
    settings = get_settings()
    logger.info(
        "Loading embedding model '%s' (first run downloads and caches the "
        "weights locally; subsequent runs load from cache)...",
        settings.EMBEDDING_MODEL,
    )
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
    logger.info(
        "Embedding model loaded. Output dimension: %d",
        model.get_sentence_embedding_dimension(),
    )
    return model


class Embedder:
    """Thin wrapper around a local sentence-transformers model."""

    def __init__(self) -> None:
        self._model = _load_model()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embeds a batch of texts (e.g. chunk texts at ingestion time).
        Returns an (N, dimension) float32 array, L2-normalized per row.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string. Returns a (dimension,) float32 vector."""
        return self.embed_texts([query])[0]


@lru_cache
def get_embedder() -> Embedder:
    """Cached factory, mirroring the get_settings() pattern in app/config.py."""
    return Embedder()