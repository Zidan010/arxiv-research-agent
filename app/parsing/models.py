"""
Data model for a parsed chunk of a paper.

A Chunk is the atomic unit that gets embedded and stored in the vector
index. Its shape is what makes source traceability possible downstream:
every chunk carries enough metadata to point straight back to the exact
paper and section it came from, without any later reconstruction step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str  # f"{arxiv_id}::{section}::{order_index}"
    arxiv_id: str
    section: str
    text: str
    order_index: int
    source_type: str  # "latex" | "pdf_fallback"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "arxiv_id": self.arxiv_id,
            "section": self.section,
            "text": self.text,
            "order_index": self.order_index,
            "source_type": self.source_type,
        }