"""
PDF-based parsing fallback, used only when a paper has no LaTeX source
available.

This is deliberately the secondary path: it cannot recover equation
structure the way the LaTeX chunker can, since by the time content is a
PDF, math has typically been rendered to glyphs, not preserved as markup.
What it *does* do properly is respect the two-column layout that naive
text extraction destroys — reading a two-column PDF top-to-bottom without
column awareness interleaves unrelated sentences from both columns.

Approach: cluster words by x-position into left/right columns (using the
page midpoint as the split), reconstruct line order within each column by
vertical position, and read the left column fully before the right column
— matching how a human actually reads the page.

Known limitation: this relies on pdfplumber's own word-boundary detection,
which merges adjacent text into one "word" if the horizontal gap between
them is small. On a page where the column gutter is very narrow, the last
word of the left column and the first word of the right column can be
merged into a single token, corrupting both.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.parsing.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHUNK_CHARS = 1200
LINE_Y_TOLERANCE = 3.0  # points; words within this vertical distance are the same line


def _extract_column_aware_text(page) -> str:
    """
    Extracts a single page's text respecting a two-column layout.
    Falls back gracefully to whatever text exists if the page has no
    extractable words (e.g. a pure-image page).
    """
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return ""

    midpoint = page.width / 2
    left_words = [w for w in words if w["x0"] < midpoint]
    right_words = [w for w in words if w["x0"] >= midpoint]

    # If the split is wildly uneven, this probably isn't a two-column page
    # (e.g. a full-width title page or figure) — just read it as one column
    # in natural top-to-bottom, left-to-right order instead of forcing a
    # column split that doesn't apply.
    if not left_words or not right_words or min(len(left_words), len(right_words)) < 0.15 * len(words):
        ordered = sorted(words, key=lambda w: (round(w["top"] / LINE_Y_TOLERANCE), w["x0"]))
        return _words_to_text(ordered)

    return _words_to_text(sorted(left_words, key=lambda w: (round(w["top"] / LINE_Y_TOLERANCE), w["x0"]))) \
        + "\n\n" \
        + _words_to_text(sorted(right_words, key=lambda w: (round(w["top"] / LINE_Y_TOLERANCE), w["x0"])))


def _words_to_text(sorted_words: list[dict]) -> str:
    lines: list[list[str]] = []
    current_line: list[str] = []
    current_top: float | None = None

    for word in sorted_words:
        if current_top is None or abs(word["top"] - current_top) > LINE_Y_TOLERANCE:
            if current_line:
                lines.append(current_line)
            current_line = [word["text"]]
            current_top = word["top"]
        else:
            current_line.append(word["text"])
    if current_line:
        lines.append(current_line)

    return "\n".join(" ".join(line) for line in lines)


def parse_pdf(
    pdf_path: Path,
    arxiv_id: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """
    Extracts column-aware text from every page and chunks it into
    paragraph-respecting windows. Section names are not recoverable from
    PDF layout alone without a much heavier document-structure model, so
    chunks are tagged with a page-range section label instead — still
    traceable, just coarser-grained than the LaTeX path.
    """
    import pdfplumber  # imported lazily so the LaTeX-only path has no PDF dependency at import time

    chunks: list[Chunk] = []
    order_index = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = _extract_column_aware_text(page)
                if not page_text.strip():
                    continue

                paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                current = ""
                for paragraph in paragraphs:
                    candidate = f"{current}\n\n{paragraph}" if current else paragraph
                    if len(candidate) > max_chunk_chars and current:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{arxiv_id}::page_{page_number}::{order_index}",
                                arxiv_id=arxiv_id,
                                section=f"Page {page_number}",
                                text=current,
                                order_index=order_index,
                                source_type="pdf_fallback",
                            )
                        )
                        order_index += 1
                        current = paragraph
                    else:
                        current = candidate
                if current:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{arxiv_id}::page_{page_number}::{order_index}",
                            arxiv_id=arxiv_id,
                            section=f"Page {page_number}",
                            text=current,
                            order_index=order_index,
                            source_type="pdf_fallback",
                        )
                    )
                    order_index += 1
    except Exception as exc:  # pdfplumber can raise a range of parser-internal errors on malformed PDFs
        logger.error("Failed to parse PDF for %s: %s", arxiv_id, exc)
        return []

    logger.info(
        "Parsed %s into %d chunks via PDF fallback (column-aware extraction)",
        arxiv_id, len(chunks),
    )
    return chunks