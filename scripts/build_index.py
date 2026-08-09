"""
Index-build entrypoint: joins embeddings into a single searchable vector store.

Inputs:
  - data/raw/papers.json                    
  - data/processed/chunks.json               
  - data/processed/embeddings.npy            
  - data/processed/embeddings_index.json    

Output:
  - <VECTOR_STORE_DIR>/index.faiss
  - <VECTOR_STORE_DIR>/metadata.json

This is the step where per-chunk metadata (arxiv_id, section) gets joined
with per-paper metadata (title, authors, url) into the single record that
makes strict source traceability possible at query time.
Usage:
    python -m scripts.build_index
"""

import json
import logging
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.logging_config import configure_logging
from app.vectorstore.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    raw_dir = Path(settings.RAW_DATA_DIR)
    processed_dir = Path(settings.PROCESSED_DATA_DIR)
    vector_store_dir = Path(settings.VECTOR_STORE_DIR)

    papers_path = raw_dir / "papers.json"
    chunks_path = processed_dir / "chunks.json"
    embeddings_path = processed_dir / "embeddings.npy"
    embeddings_index_path = processed_dir / "embeddings_index.json"

    missing = [p for p in (papers_path, chunks_path, embeddings_path, embeddings_index_path) if not p.exists()]
    if missing:
        logger.error(
            "Missing required input file(s): %s. Run scripts.ingest, "
            "scripts.parse, and scripts.embed first, in that order.",
            [str(p) for p in missing],
        )
        return

    papers = {p["arxiv_id"]: p for p in json.loads(papers_path.read_text())}
    chunks = {c["chunk_id"]: c for c in json.loads(chunks_path.read_text())}
    embeddings = np.load(embeddings_path)
    chunk_id_order = json.loads(embeddings_index_path.read_text())

    if embeddings.shape[0] != len(chunk_id_order):
        logger.error(
            "embeddings.npy row count (%d) does not match embeddings_index.json "
            "length (%d) -- these files are out of sync. Re-run scripts.embed.",
            embeddings.shape[0], len(chunk_id_order),
        )
        return

    metadatas: list[dict] = []
    skipped = 0
    for chunk_id in chunk_id_order:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            # chunks.json and embeddings_index.json came from different runs
            skipped += 1
            continue
        paper = papers.get(chunk["arxiv_id"], {})
        metadatas.append(
            {
                "chunk_id": chunk_id,
                "arxiv_id": chunk["arxiv_id"],
                "section": chunk["section"],
                "text": chunk["text"],
                "source_type": chunk["source_type"],
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "url": paper.get("abs_url", ""),
            }
        )

    if skipped:
        logger.warning(
            "%d chunk_id(s) from embeddings_index.json were not found in "
            "chunks.json and were skipped -- inputs may be out of sync.",
            skipped,
        )

    store = FaissVectorStore(dimension=embeddings.shape[1])
    # metadatas is already in the same row order as `embeddings` (both built
    # by iterating chunk_id_order), so a direct add() keeps FAISS row index
    # aligned with the metadata sidecar.
    store.add(embeddings[: len(metadatas)], metadatas)
    store.save(vector_store_dir)

    logger.info(
        "Index build complete: %d chunks indexed from %d papers, written to %s",
        len(metadatas), len(papers), vector_store_dir,
    )


if __name__ == "__main__":
    configure_logging()
    run()