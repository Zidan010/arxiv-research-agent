"""
Data model for the output of RAG synthesis
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkUsed:
    section: str
    text: str


@dataclass
class SourceCitation:
    citation_id: int
    arxiv_id: str
    title: str
    authors: list[str]
    url: str
    chunks_used: list[ChunkUsed]

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "chunks_used": [{"section": c.section, "text": c.text} for c in self.chunks_used],
        }


@dataclass
class SynthesisResult:
    query: str
    status: str  # "ok" | "validation_failed" | "no_results" | "llm_error"
    answer: str | None = None
    sources: list[SourceCitation] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "status": self.status,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "error": self.error,
        }