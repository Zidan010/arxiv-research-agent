"""
Vector store interface.

Kept a base so the concrete implementation (FAISS today) can be
swapped for Qdrant, Chroma, etc. later without any caller needing to change — the same pattern already used for
the LLM provider layer (app/llm) and the embedding backend (app/embeddings).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        """Adds a batch of vectors with their associated metadata records."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        max_per_paper: int | None = None,
    ) -> list[SearchResult]:
        """
        Returns the top_k most similar entries to query_vector.

        max_per_paper, when set, caps how many results may come from the
        same arxiv_id — this is what lets retrieval favor evidence from
        multiple independent papers rather than returning five chunks of
        whichever single paper happens to embed closest to the query (see
        ARCHITECTURE.md, Section 3).
        """

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persists the index and its metadata sidecar to directory."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Loads a previously saved index and metadata sidecar from directory."""