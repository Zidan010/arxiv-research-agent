"""
The research query endpoint.

The synthesizer is built once at application
startup and stored on app.state, rather
than being reconstructed per request -- loading the embedding model and the
FAISS index are both too expensive to repeat on every call.

Defined as a plain `def` (not `async def`) so FastAPI runs it in its worker
threadpool automatically: embedding, FAISS search, and the LLM API call are
all blocking/synchronous, and running them directly in an async def would
block the event loop for every other concurrent request.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("/query", response_model=QueryResponse)
def query_research(request: QueryRequest, http_request: Request) -> QueryResponse:
    synthesizer = getattr(http_request.app.state, "synthesizer", None)

    if synthesizer is None:
        startup_error = getattr(http_request.app.state, "synthesizer_error", None)
        detail = (
            "The research agent is not ready to serve queries. "
            + (startup_error or "See server logs for details.")
        )
        logger.error("Query rejected -- synthesizer not initialized: %s", detail)
        raise HTTPException(status_code=503, detail=detail)

    logger.info("Received query: %r", request.query)
    result = synthesizer.answer_query(request.query)

    return QueryResponse(**result.to_dict())