"""
RAG synthesis: retrieval -> prompt construction -> generation -> citation
validation.

Design notes:
  - Retrieval favors cross-paper evidence (via the vector store's
    max_per_paper cap), not just raw top-k similarity.
  - The synthesis prompt (app/rag/prompts.py) forces paraphrasing,
    per-claim citation, and explicit surfacing of disagreement between
    sources.
  - Citations are validated after generation, not trusted: every [n]
    marker in the answer must correspond to a source that was actually
    provided as context, and the answer must contain at least one
    citation. A failure here returns an explicit validation_failed result
    rather than silently shipping an uncited or hallucinated-citation
    answer -- making traceability a system-enforced guarantee, not a
    prompting convention the model can quietly ignore.

Dependencies (embedder, vector store, LLM provider) are injected via the
constructor rather than imported/constructed internally. This is what
makes the class fully unit-testable with fakes, with no network access
required.
"""

from __future__ import annotations

import logging
import re

from app.embeddings.embedder import Embedder
from app.llm.base import LLMProvider, LLMProviderError
from app.rag.models import ChunkUsed, SourceCitation, SynthesisResult
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from app.vectorstore.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


class RAGSynthesizer:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        top_k: int = 6,
        max_chunks_per_paper: int = 2,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm_provider
        self._top_k = top_k
        self._max_chunks_per_paper = max_chunks_per_paper
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _group_by_paper(self, results: list[SearchResult]) -> list[dict]:
        """
        Assigns a citation_id (1-based) to each distinct arxiv_id, in order
        of first appearance in the (already-ranked) results, grouping every
        matching chunk under that single citation number -- this is what
        makes "[1]" mean "this paper," not "this specific chunk."
        """
        grouped: dict[str, dict] = {}
        order: list[str] = []

        for result in results:
            meta = result.metadata
            arxiv_id = meta["arxiv_id"]
            if arxiv_id not in grouped:
                order.append(arxiv_id)
                grouped[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", []),
                    "url": meta.get("url", ""),
                    "chunks": [],
                }
            grouped[arxiv_id]["chunks"].append(
                {"section": meta.get("section", ""), "text": meta.get("text", "")}
            )

        numbered = []
        for i, arxiv_id in enumerate(order, start=1):
            entry = grouped[arxiv_id]
            entry["citation_id"] = i
            numbered.append(entry)
        return numbered

    def _validate_citations(self, answer: str, valid_ids: set[int]) -> str | None:
        """
        Returns an error message if validation fails, or None if the
        answer passes. Checks:
          1. At least one citation marker is present.
          2. Every citation marker in the answer refers to a source that
             was actually provided as context (no hallucinated numbers).
        """
        cited_ids = {int(n) for n in CITATION_MARKER_PATTERN.findall(answer)}

        if not cited_ids:
            return "Generated answer contained no citation markers."

        invalid_ids = cited_ids - valid_ids
        if invalid_ids:
            return (
                f"Generated answer cited source number(s) {sorted(invalid_ids)}, "
                f"which were not among the {len(valid_ids)} source(s) provided as context."
            )

        return None

    def answer_query(self, query: str) -> SynthesisResult:
        query_vector = self._embedder.embed_query(query)
        results = self._vector_store.search(
            query_vector,
            top_k=self._top_k,
            max_per_paper=self._max_chunks_per_paper,
        )

        if not results:
            logger.info("No results retrieved for query: %r", query)
            return SynthesisResult(
                query=query, answer=None, sources=[], status="no_results",
                error="No relevant papers were found in the index for this query.",
            )

        numbered_sources = self._group_by_paper(results)
        user_prompt = build_user_prompt(query, numbered_sources)

        try:
            answer = self._llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except LLMProviderError as exc:
            logger.error("LLM generation failed for query %r: %s", query, exc)
            return SynthesisResult(
                query=query, answer=None, sources=[], status="llm_error",
                error=f"Answer generation failed: {exc}",
            )

        valid_ids = {s["citation_id"] for s in numbered_sources}
        validation_error = self._validate_citations(answer, valid_ids)
        if validation_error:
            logger.warning("Citation validation failed for query %r: %s", query, validation_error)
            return SynthesisResult(
                query=query, answer=None, sources=[], status="validation_failed",
                error=validation_error,
            )

        # Only include sources actually cited in the final answer -- a
        # source retrieved but not referenced shouldn't clutter the
        # traceability output.
        cited_ids = {int(n) for n in CITATION_MARKER_PATTERN.findall(answer)}
        sources = [
            SourceCitation(
                citation_id=s["citation_id"],
                arxiv_id=s["arxiv_id"],
                title=s["title"],
                authors=s["authors"],
                url=s["url"],
                chunks_used=[ChunkUsed(**c) for c in s["chunks"]],
            )
            for s in numbered_sources
            if s["citation_id"] in cited_ids
        ]

        return SynthesisResult(query=query, answer=answer, sources=sources, status="ok")