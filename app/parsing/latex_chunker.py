"""
LaTeX-aware parsing and chunking.

Design notes:
  - Operates on the raw LaTeX source (unpacked e-print tarball), not the PDF.
    Equations are already well-formed markup here (`$...$`,
    `\\begin{equation}...\\end{equation}`) — no OCR or layout-
    reconstruction problem to solve.
  - Chunks on logical structure (\\section / \\subsection boundaries), not
    character count, so each chunk corresponds to a real unit of meaning.
  - Equations are treated as atomic, non-splittable units: before chunking,
    every equation environment and every $...$ / $$...$$ / \\[...\\] span is
    replaced with a short placeholder token, so plain paragraph/character-
    based chunking downstream can never land its cut point in the middle of
    an expression. Placeholders are restored after chunk boundaries are
    decided.
"""

from __future__ import annotations

import logging
import re
import tarfile
from pathlib import Path

from app.parsing.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHUNK_CHARS = 1200

# Equation environments we treat as atomic. Covers the environments that
# actually appear in the wild; deliberately not attempting a full LaTeX
# parser, just protecting the structures that matter for "don't mangle math."
_EQUATION_ENV_NAMES = (
    "equation", "equation*",
    "align", "align*",
    "gather", "gather*",
    "multline", "multline*",
    "eqnarray", "eqnarray*",
)
_ENV_PATTERN = re.compile(
    r"\\begin\{(" + "|".join(re.escape(n) for n in _EQUATION_ENV_NAMES) + r")\}"
    r".*?"
    r"\\end\{\1\}",
    re.DOTALL,
)
# $$...$$ and \[...\] (display math), then $...$ (inline math). Order matters:
# $$ must be matched before a naive $...$ pattern would misfire on it.
_DISPLAY_DOLLAR_PATTERN = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_BRACKET_DISPLAY_PATTERN = re.compile(r"\\\[.*?\\\]", re.DOTALL)
_INLINE_DOLLAR_PATTERN = re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL)

_SECTION_PATTERN = re.compile(
    r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}"
)
_INPUT_INCLUDE_PATTERN = re.compile(r"\\(?:input|include)\{([^}]*)\}")
_COMMENT_PATTERN = re.compile(r"(?<!\\)%.*")


def _strip_comments(text: str) -> str:
    return "\n".join(_COMMENT_PATTERN.sub("", line) for line in text.split("\n"))


def find_main_tex_file(extracted_dir: Path) -> Path | None:
    """
    Heuristic for locating the entry-point .tex file in a multi-file source
    tarball: prefer a file containing \\documentclass and \\begin{document};
    if several qualify, prefer the one with the most \\section commands
    (proxy for "the actual paper body" vs. a supplementary/appendix file).
    """
    candidates = list(extracted_dir.rglob("*.tex"))
    if not candidates:
        return None

    scored: list[tuple[int, Path]] = []
    for path in candidates:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "\\documentclass" in text and "\\begin{document}" in text:
            score = len(_SECTION_PATTERN.findall(text))
            scored.append((score, path))

    if not scored:
        # No file has \documentclass (unusual, but be defensive) — fall back
        # to the largest .tex file as a best guess.
        return max(candidates, key=lambda p: p.stat().st_size)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def _inline_inputs(text: str, base_dir: Path, _depth: int = 0, _seen: set | None = None) -> str:
    """Recursively inlines \\input{...} / \\include{...} referenced files."""
    if _depth > 10:  # guard against pathological or cyclic includes
        return text
    _seen = _seen or set()

    def replace(match: re.Match) -> str:
        ref = match.group(1)
        candidate = ref if ref.endswith(".tex") else f"{ref}.tex"
        path = (base_dir / candidate).resolve()
        if path in _seen or not path.exists():
            return ""
        _seen.add(path)
        try:
            included = path.read_text(errors="ignore")
        except OSError:
            return ""
        return _inline_inputs(included, base_dir, _depth + 1, _seen)

    return _INPUT_INCLUDE_PATTERN.sub(replace, text)


def _protect_equations(text: str) -> tuple[str, dict[str, str]]:
    """Replaces equation spans with placeholder tokens; returns the mapping to restore them."""
    mapping: dict[str, str] = {}
    counter = 0

    def protect(pattern: re.Pattern) -> None:
        nonlocal text, counter
        def _sub(match: re.Match) -> str:
            nonlocal counter
            token = f"@@EQN_{counter}@@"
            mapping[token] = match.group(0)
            counter += 1
            return token
        text = pattern.sub(_sub, text)

    # Environments first (they may internally contain $ signs that would
    # otherwise confuse the inline-dollar pattern), then display math, then
    # inline math.
    protect(_ENV_PATTERN)
    protect(_DISPLAY_DOLLAR_PATTERN)
    protect(_BRACKET_DISPLAY_PATTERN)
    protect(_INLINE_DOLLAR_PATTERN)
    return text, mapping


def _restore_equations(text: str, mapping: dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """
    Splits document body into (section_name, section_text) pairs using
    \\section/\\subsection/\\subsubsection boundaries. Content before the
    first heading is kept as a "Preamble" section (usually the abstract /
    intro-before-first-heading) rather than silently dropped.
    """
    matches = list(_SECTION_PATTERN.finditer(body))
    if not matches:
        return [("Full Text", body)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = body[: matches[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))

    for i, match in enumerate(matches):
        name = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((name, body[start:end].strip()))

    return sections


def _chunk_section_text(
    section_name: str,
    protected_text: str,
    eqn_map: dict[str, str],
    max_chunk_chars: int,
) -> list[str]:
    """
    Splits a section's (equation-protected) text into paragraph-respecting
    chunks under max_chunk_chars, then restores equations in each resulting
    chunk. Because equations are single short tokens at this point, a chunk
    boundary can never fall inside one.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", protected_text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > max_chunk_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)

    return [_restore_equations(c, eqn_map) for c in chunks]


def parse_latex_source(
    tar_path: Path,
    arxiv_id: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """
    Full pipeline: unpack tarball -> locate main .tex -> inline \\input/\\include
    -> strip comments -> split into sections -> equation-protected chunking
    -> restore equations -> return Chunk objects.
    """
    extract_dir = tar_path.parent / f"{tar_path.stem.replace('.tar', '')}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(extract_dir, filter="data")
    except tarfile.ReadError:
        # Some e-prints are a single gzipped .tex file, not a tarball.
        import gzip
        try:
            with gzip.open(tar_path, "rb") as gz:
                content = gz.read()
            single_file = extract_dir / f"{arxiv_id.replace('/', '_')}.tex"
            single_file.write_bytes(content)
        except OSError as exc:
            logger.warning(
                "Could not unpack source for %s as tar or gzip (%s); "
                "PDF fallback should be used instead.",
                arxiv_id, exc,
            )
            return []

    main_file = find_main_tex_file(extract_dir)
    if main_file is None:
        logger.warning(
            "No .tex file with \\documentclass found for %s; "
            "PDF fallback should be used instead.",
            arxiv_id,
        )
        return []

    raw_text = main_file.read_text(errors="ignore")
    raw_text = _inline_inputs(raw_text, main_file.parent)
    raw_text = _strip_comments(raw_text)

    # Work only on the document body (between \begin{document}/\end{document})
    # to avoid chunking preamble/macro-definition noise.
    body_match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", raw_text, re.DOTALL)
    body = body_match.group(1) if body_match else raw_text

    protected_body, eqn_map = _protect_equations(body)
    sections = _split_into_sections(protected_body)

    chunks: list[Chunk] = []
    order_index = 0
    for section_name, section_text in sections:
        for chunk_text in _chunk_section_text(section_name, section_text, eqn_map, max_chunk_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{arxiv_id}::{section_name}::{order_index}",
                    arxiv_id=arxiv_id,
                    section=section_name,
                    text=chunk_text,
                    order_index=order_index,
                    source_type="latex",
                )
            )
            order_index += 1

    logger.info(
        "Parsed %s into %d chunks across %d sections (LaTeX source)",
        arxiv_id, len(chunks), len(sections),
    )
    return chunks