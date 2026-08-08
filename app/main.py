from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI

from app.config import get_settings
from app.logging_config import configure_logging

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


@app.on_event("startup")
async def on_startup() -> None:
    logger.info(
        "Starting arXiv Research Agent API on %s:%s",
        settings.API_HOST,
        settings.API_PORT
    )


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