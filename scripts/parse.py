"""
Parsing entrypoint: reads the papers ingested by scripts/ingest.py
(data/raw/papers.json), parses each one — LaTeX source when available,
PDF fallback otherwise — and writes the resulting chunks to
data/processed/chunks.json for the embedding stage to consume.

Usage:
    python -m scripts.parse
"""

import json
import logging
from pathlib import Path

from app.config import get_settings
from app.ingestion.arxiv_client import ArxivClient
from app.ingestion.models import Paper
from app.logging_config import configure_logging
from app.parsing.latex_chunker import parse_latex_source
from app.parsing.pdf_fallback import parse_pdf

logger = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    raw_dir = Path(settings.RAW_DATA_DIR)
    processed_dir = Path(settings.PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = raw_dir / "papers.json"
    if not metadata_path.exists():
        logger.error(
            "No ingested metadata found at %s — run `python -m scripts.ingest` first.",
            metadata_path,
        )
        return

    papers_raw = json.loads(metadata_path.read_text())
    logger.info("Parsing %d ingested papers", len(papers_raw))

    client = ArxivClient(rate_limit_seconds=settings.ARXIV_RATE_LIMIT_SECONDS)
    pdf_dir = raw_dir / "pdfs"

    all_chunks = []
    latex_count = 0
    fallback_count = 0
    failed_count = 0

    for entry in papers_raw:
        arxiv_id = entry["arxiv_id"]

        if entry.get("source_available") and entry.get("source_path"):
            chunks = parse_latex_source(
                Path(entry["source_path"]),
                arxiv_id,
                max_chunk_chars=settings.PARSING_MAX_CHUNK_CHARS,
            )
            if chunks:
                latex_count += 1
            else:
                # LaTeX unpack/parse failed even though a source file existed
                # (e.g. no \documentclass found) — fall back to PDF rather
                # than losing the paper entirely.
                logger.info("LaTeX parse yielded no chunks for %s; falling back to PDF.", arxiv_id)
                entry["source_available"] = False

        if not entry.get("source_available"):
            paper = Paper(
                arxiv_id=arxiv_id,
                title=entry["title"],
                authors=entry["authors"],
                abstract=entry["abstract"],
                published=entry["published"],
                updated=entry["updated"],
                primary_category=entry["primary_category"],
                categories=entry["categories"],
                pdf_url=entry["pdf_url"],
            )
            pdf_path = client.fetch_pdf(paper, destination_dir=pdf_dir)
            chunks = parse_pdf(pdf_path, arxiv_id, max_chunk_chars=settings.PARSING_MAX_CHUNK_CHARS) if pdf_path else []
            if chunks:
                fallback_count += 1
            else:
                failed_count += 1
                logger.error("Could not parse %s via LaTeX or PDF fallback; skipping.", arxiv_id)

        all_chunks.extend(chunks)

    output_path = processed_dir / "chunks.json"
    output_path.write_text(json.dumps([c.to_dict() for c in all_chunks], indent=2))

    logger.info(
        "Parsing complete: %d total chunks written to %s "
        "(%d papers via LaTeX, %d via PDF fallback, %d failed)",
        len(all_chunks), output_path, latex_count, fallback_count, failed_count,
    )


if __name__ == "__main__":
    configure_logging()
    run()