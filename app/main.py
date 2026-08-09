"""
FastAPI application entrypoint.
 
Registers the /api/research/query router (app/api/routes.py). The
RAGSynthesizer -- which depends on the embedding model, the FAISS index,
and the LLM provider client -- is constructed once at startup and stored
on app.state, rather than per-request, since loading the embedding model
and the vector index are both too expensive to repeat on every call.
 
if the vector store hasn't been built yet
(scripts.ingest / scripts.parse / scripts.embed / scripts.build_index
haven't been run), or the configured LLM provider has no API key, the app
still boots and /health still reports ok.
"""
from pathlib import Path
import sys
import logging
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI

from app.api.routes import router as research_router
from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.llm.base import LLMProviderError
from app.llm.factory import get_llm_provider
from app.logging_config import configure_logging
from app.rag.synthesizer import RAGSynthesizer
from app.vectorstore.faiss_store import FaissVectorStore

configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="arXiv Research Agent",
    description=(
        "Ingests, parses, and answers questions about recent AI/ML research "
        "from arXiv, with per-claim source traceability."
    ),
    version="0.1.0",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting arXiv Research Agent API on %s:%s (log level: %s)",
        settings.API_HOST,
        settings.API_PORT,
        settings.LOG_LEVEL,
    )

    app.state.synthesizer = None
    app.state.synthesizer_error = None

    try:
        vector_store = FaissVectorStore.load(Path(settings.VECTOR_STORE_DIR))
    except FileNotFoundError:
        app.state.synthesizer_error = (
            f"No vector store found at '{settings.VECTOR_STORE_DIR}'. "
            "Run scripts.ingest, scripts.parse, scripts.embed, and "
            "scripts.build_index (in that order) before querying."
        )
        logger.warning(app.state.synthesizer_error)
        yield
        return

    try:
        embedder = get_embedder()
        llm_provider = get_llm_provider()
    except LLMProviderError as exc:
        app.state.synthesizer_error = str(exc)
        logger.warning("LLM provider not ready: %s", exc)
        yield
        return

    app.state.synthesizer = RAGSynthesizer(
        embedder=embedder,
        vector_store=vector_store,
        llm_provider=llm_provider,
        top_k=settings.RETRIEVAL_TOP_K,
        max_chunks_per_paper=settings.RETRIEVAL_MAX_CHUNKS_PER_PAPER,
        max_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
    )
    logger.info(
        "Synthesizer ready: %d vectors indexed, LLM provider '%s'",
        vector_store.index.ntotal,
        settings.LLM_PROVIDER,
    )

    yield

    logger.info("Shutting down arXiv Research Agent API")


app = FastAPI(
    title="arXiv Research Agent",
    description=(
        "Ingests, parses, and answers questions about recent AI/ML research "
        "from arXiv, with per-claim source traceability."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(research_router)

@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Basic liveness check — used by Docker/orchestrators and for smoke testing."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )