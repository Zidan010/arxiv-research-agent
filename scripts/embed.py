"""
Embedding entrypoint: reads the chunks produced by scripts/parse.py
(data/processed/chunks.json), embeds them with the local sentence-transformers
model, and writes:
  - data/processed/embeddings.npy         (float32 array, shape [N, dim])
  - data/processed/embeddings_index.json  (chunk_id for each row, same order)

Kept as a separate array + index file (rather than embedding vectors inline
in chunks.json) so the vector store build step can load a
compact numeric array directly into FAISS without re-parsing JSON floats,
and so chunks.json stays human-readable on its own.

Usage:
    python -m scripts.embed
"""

import json
import logging
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    processed_dir = Path(settings.PROCESSED_DATA_DIR)
    chunks_path = processed_dir / "chunks.json"

    if not chunks_path.exists():
        logger.error(
            "No parsed chunks found at %s -- run `python -m scripts.parse` first.",
            chunks_path,
        )
        return

    chunks = json.loads(chunks_path.read_text())
    if not chunks:
        logger.error("chunks.json is empty -- nothing to embed.")
        return

    logger.info(
        "Embedding %d chunks with model '%s'", len(chunks), settings.EMBEDDING_MODEL
    )

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    embeddings = embedder.embed_texts(
        texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        show_progress=True,
    )

    embeddings_path = processed_dir / "embeddings.npy"
    index_path = processed_dir / "embeddings_index.json"

    np.save(embeddings_path, embeddings)
    index_path.write_text(json.dumps(chunk_ids, indent=2))

    logger.info(
        "Embedding complete: %d vectors of dimension %d written to %s "
        "(chunk-id index: %s)",
        embeddings.shape[0], embeddings.shape[1], embeddings_path, index_path,
    )


if __name__ == "__main__":
    configure_logging()
    run()