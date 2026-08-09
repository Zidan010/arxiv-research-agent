"""
FAISS-backed vector store, with a JSON metadata sidecar.

Design notes:
  - FAISS stores vectors and returns integer row indices on
    search -- it has no concept of metadata. This module maintains a
    parallel Python list (`_metadatas`), indexed by the same row position
    FAISS assigns on `add()`, so every vector has a corresponding
    {chunk_id, arxiv_id, title, authors, url, section, chunk_text} record.
    This sidecar is what makes the strict source-traceability requirement
    possible -- the vector store is the single place chunk provenance is
    joined to embeddings, not something reconstructed later at query time.
  - Uses IndexFlatIP (inner product). Embeddings are L2-normalized at
    generation time (see app/embeddings/embedder.py), so inner product is
    mathematically equivalent to cosine similarity here, without the extra
    computation a dedicated cosine index would add.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from app.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._metadatas: list[dict] = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        if vectors.shape[0] != len(metadatas):
            raise ValueError(
                f"vectors/metadatas length mismatch: {vectors.shape[0]} vs {len(metadatas)}"
            )
        if vectors.shape[0] == 0:
            return
        self.index.add(vectors.astype("float32"))
        self._metadatas.extend(metadatas)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        max_per_paper: int | None = None,
    ) -> list[SearchResult]:
        if self.index.ntotal == 0:
            return []

        # Over-fetch when diversity is requested, since some candidates will
        # be skipped once their paper's cap is hit -- 5x is generous enough
        # in practice without materially hurting query latency at this scale.
        fetch_k = min(self.index.ntotal, top_k * 5) if max_per_paper else min(self.index.ntotal, top_k)

        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, fetch_k)

        results: list[SearchResult] = []
        per_paper_counts: dict[str, int] = {}

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if fetch_k > ntotal
                continue
            metadata = self._metadatas[idx]

            if max_per_paper is not None:
                arxiv_id = metadata.get("arxiv_id", "")
                count = per_paper_counts.get(arxiv_id, 0)
                if count >= max_per_paper:
                    continue
                per_paper_counts[arxiv_id] = count + 1

            results.append(
                SearchResult(chunk_id=metadata["chunk_id"], score=float(score), metadata=metadata)
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        (directory / METADATA_FILENAME).write_text(json.dumps(self._metadatas))
        logger.info(
            "Saved FAISS index (%d vectors, dim=%d) to %s",
            self.index.ntotal, self.dimension, directory,
        )

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No vector store found at {directory} "
                f"(expected {INDEX_FILENAME} and {METADATA_FILENAME})"
            )

        index = faiss.read_index(str(index_path))
        metadatas = json.loads(metadata_path.read_text())

        store = cls(dimension=index.d)
        store.index = index
        store._metadatas = metadatas
        logger.info(
            "Loaded FAISS index (%d vectors, dim=%d) from %s",
            index.ntotal, index.d, directory,
        )
        return store