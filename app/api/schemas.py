"""
Pydantic request/response schemas for /api/research/query.

This maintains the synthesized answer AND a sources array
detailing exactly where the data came from (arXiv ID, title, author, URL,
and the specific text chunks used). 
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The research question to answer using the indexed papers.",
    )


class ChunkUsedSchema(BaseModel):
    section: str
    text: str


class SourceSchema(BaseModel):
    citation_id: int
    arxiv_id: str
    title: str
    authors: list[str]
    url: str
    chunks_used: list[ChunkUsedSchema]


class QueryResponse(BaseModel):
    query: str
    status: str  # "ok" | "no_results" | "llm_error" | "validation_failed"
    answer: str | None
    sources: list[SourceSchema]
    error: str | None