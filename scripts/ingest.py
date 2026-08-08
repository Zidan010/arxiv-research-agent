"""
Ingestion entrypoint: fetches metadata + LaTeX source for the N most recent
papers in a given arXiv category, and writes them to disk as raw input for
the parsing stage.

Usage:
    python -m scripts.ingest

Configuration is read from Settings :
    ARXIV_CATEGORY, ARXIV_MAX_RESULTS, ARXIV_RATE_LIMIT_SECONDS, RAW_DATA_DIR
"""

import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.ingestion.arxiv_client import ArxivClient
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    client = ArxivClient(rate_limit_seconds=settings.ARXIV_RATE_LIMIT_SECONDS)

    raw_dir = Path(settings.RAW_DATA_DIR)
    sources_dir = raw_dir / "sources"
    metadata_path = raw_dir / "papers.json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting ingestion: category=%s max_results=%d",
        settings.ARXIV_CATEGORY,
        settings.ARXIV_MAX_RESULTS,
    )

    papers = client.fetch_recent_papers(
        category=settings.ARXIV_CATEGORY,
        max_results=settings.ARXIV_MAX_RESULTS,
    )

    if not papers:
        logger.error(
            "No papers returned for category=%s — aborting before writing output.",
            settings.ARXIV_CATEGORY,
        )
        return

    completed = []
    for i, paper in enumerate(papers, start=1):
        logger.info("[%d/%d] Fetching source for %s", i, len(papers), paper.arxiv_id)
        paper = client.fetch_source(paper, destination_dir=sources_dir)
        completed.append(paper.to_dict())

    metadata_path.write_text(json.dumps(completed, indent=2))

    with_source = sum(1 for p in completed if p["source_available"])
    logger.info(
        "Ingestion complete: %d papers written to %s (%d with LaTeX source, "
        "%d will need the PDF fallback in parsing)",
        len(completed),
        metadata_path,
        with_source,
        len(completed) - with_source,
    )


if __name__ == "__main__":
    configure_logging()
    run()