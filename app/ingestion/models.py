"""
Data model for a paper fetched from arXiv.

Kept intentionally separate from any specific fetch mechanism (Atom API,
OAI-PMH, or S3 bulk manifest) so parsing/embedding code downstream depends on
this stable shape, not on how the data was retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    """Metadata + raw content for a single arXiv paper."""

    arxiv_id: str  
    title: str
    authors: list[str]
    abstract: str
    published: str  
    updated: str
    primary_category: str
    categories: list[str] = field(default_factory=list)

    abs_url: str = ""  # https://arxiv.org/abs/{id}
    pdf_url: str = ""  # https://arxiv.org/pdf/{id}

    # Populated after a successful source download 
    source_path: str | None = None  # local path to the downloaded e-print 
    source_available: bool = False  # False triggers the PDF-parsing fallback

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "updated": self.updated,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "source_path": self.source_path,
            "source_available": self.source_available,
        }