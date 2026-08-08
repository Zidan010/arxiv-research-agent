"""
Rate-limit client for arXiv's official Atom API and e-print source endpoint.

Design notes:
  - Uses the documented Atom query API (export.arxiv.org/api/query), never
    HTML scraping. 
  - Enforces the documented 3 requests/second limit via a simple minimum-
    interval throttle applied before every outbound request, regardless of
    endpoint (metadata query or source download).
  - Retries transient failures (network errors, 5xx, 429) with exponential
    backoff via tenacity, rather than failing the whole ingestion run on one
    flaky request.
  - Fetches LaTeX source (the "e-print") separately from metadata, since the
    parsing strategy depends on having the raw source, not just the abstract.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import feedparser
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.ingestion.models import Paper

logger = logging.getLogger(__name__)

ATOM_API_URL = "http://export.arxiv.org/api/query"
EPRINT_URL_TEMPLATE = "https://arxiv.org/e-print/{arxiv_id}"

# "http://arxiv.org/abs/2508.01234v2" — this strips it down to "2508.01234".
_ID_VERSION_RE = re.compile(r"arxiv\.org/abs/([^v]+)(v\d+)?$")


class RateLimiter:
    """Enforces a minimum interval between calls, shared across endpoints."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


class ArxivClient:
    def __init__(
        self,
        rate_limit_seconds: float = 3.0,
        user_agent: str = "arxiv-research-agent/0.1",
        timeout_seconds: int = 30,
    ) -> None:
        # arXiv documents a 3 requests/second ceiling; we throttle to one
        # request per `rate_limit_seconds` to stay comfortably under it
        # rather than trying to burst right up to the limit.
        self._rate_limiter = RateLimiter(rate_limit_seconds)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._timeout = timeout_seconds

    @staticmethod
    def _extract_arxiv_id(entry_id_url: str) -> str:
        match = _ID_VERSION_RE.search(entry_id_url)
        if match:
            return match.group(1)
        # Fallback: last path segment, version-stripped
        last_segment = entry_id_url.rstrip("/").split("/")[-1]
        return re.sub(r"v\d+$", "", last_segment)

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        self._rate_limiter.wait()
        logger.debug("GET %s params=%s", url, params)
        response = self._session.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()
        return response

    def fetch_recent_papers(self, category: str, max_results: int) -> list[Paper]:
        """
        Query the Atom API for the most recent papers in `category`, sorted
        by submission date descending. Returns parsed Paper objects with
        metadata.
        """
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
        response = self._get(ATOM_API_URL, params=params)
        feed = feedparser.parse(response.text)

        if feed.bozo:
            # feedparser sets `bozo` when the XML wasn't well-formed. We still
            # attempt to use whatever entries were parsed, but log loudly 
            # a malformed feed silently yielding zero papers is a worse
            # failure mode than a noisy warning.
            logger.warning(
                "arXiv Atom feed for category=%s was not well-formed (bozo=%s): %s",
                category,
                feed.bozo,
                getattr(feed, "bozo_exception", "unknown"),
            )

        papers: list[Paper] = []
        for entry in feed.entries:
            arxiv_id = self._extract_arxiv_id(entry.id)

            pdf_url = ""
            for link in entry.get("links", []):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
                    break

            categories = [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")]
            primary_category = getattr(entry, "arxiv_primary_category", {}).get(
                "term", categories[0] if categories else category
            )

            papers.append(
                Paper(
                    arxiv_id=arxiv_id,
                    title=" ".join(entry.title.split()),  # collapse embedded newlines/whitespace
                    authors=[author.name for author in entry.get("authors", [])],
                    abstract=" ".join(entry.summary.split()),
                    published=entry.get("published", ""),
                    updated=entry.get("updated", ""),
                    primary_category=primary_category,
                    categories=categories,
                    abs_url=entry.id,
                    pdf_url=pdf_url,
                )
            )

        logger.info(
            "Fetched %d papers from category=%s (requested max_results=%d)",
            len(papers),
            category,
            max_results,
        )
        return papers

    def fetch_source(self, paper: Paper, destination_dir: Path) -> Paper:
        """
        Downloads the LaTeX source (e-print) tarball for a paper.
        Mutates and returns the paper with `source_path` / `source_available`
        set. Never raises on a missing source — some submissions (e.g.
        PDF-only or withdrawn papers) simply don't have one, which is the
        expected trigger for the PDF-parsing fallback, not an ingestion failure.
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        target_path = destination_dir / f"{paper.arxiv_id.replace('/', '_')}.tar.gz"

        url = EPRINT_URL_TEMPLATE.format(arxiv_id=paper.arxiv_id)
        try:
            response = self._get(url)
        except requests.RequestException as exc:
            logger.warning(
                "No LaTeX source available for %s (%s); PDF fallback will be used.",
                paper.arxiv_id,
                exc,
            )
            paper.source_available = False
            return paper

        target_path.write_bytes(response.content)
        paper.source_path = str(target_path)
        paper.source_available = True
        logger.info("Downloaded source for %s -> %s", paper.arxiv_id, target_path)
        return paper

    def fetch_pdf(self, paper: Paper, destination_dir: Path) -> Path | None:
        """
        Downloads the rendered PDF for a paper. Used only by the parsing
        fallback path, for papers where fetch_source() found no LaTeX
        source available. Shares the same rate limiter as every other
        request this client makes.
        """
        if not paper.pdf_url:
            logger.warning("No pdf_url recorded for %s; cannot fetch PDF fallback.", paper.arxiv_id)
            return None
 
        destination_dir.mkdir(parents=True, exist_ok=True)
        target_path = destination_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
 
        try:
            response = self._get(paper.pdf_url)
        except requests.RequestException as exc:
            logger.error("Failed to download PDF for %s: %s", paper.arxiv_id, exc)
            return None
 
        target_path.write_bytes(response.content)
        logger.info("Downloaded PDF fallback for %s -> %s", paper.arxiv_id, target_path)
        return target_path