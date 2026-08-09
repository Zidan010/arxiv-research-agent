# syntax=docker/dockerfile:1

FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE avoids .pyc clutter in the image;
# PYTHONUNBUFFERED makes log output show up immediately (important for
# `docker logs` / docker-compose log streaming, since Python buffers
# stdout by default when it isn't attached to a real terminal).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first, separately from application code, so Docker's
# layer cache is only invalidated by a requirements.txt change -- not by
# every code edit. This is the single biggest build-time win available here,
# since this project's dependencies (torch, via sentence-transformers) are
# large and slow to install.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Mount point for persisted data (raw ingestion output, parsed chunks,
# embeddings, and the FAISS index + metadata sidecar). Created here so the
# directory exists with correct permissions even before a volume is mounted
# over it; docker-compose.yml mounts ./data here for the actual persisted,
# host-visible content.
RUN mkdir -p data/raw data/processed data/vector_store

EXPOSE 8000

# Uses Python's stdlib rather than installing curl/wget just for the
# healthcheck -- keeps the image smaller. Fails (non-zero exit) if the
# request errors or times out, which Docker interprets as unhealthy.
# Reads $API_PORT from the container's own environment via os.environ,
# consistent with the actual port uvicorn binds to below, rather than
# hardcoding 8000 and risking it drifting out of sync if API_PORT is
# overridden.
# start-period is deliberately generous: importing sentence-transformers
# (and therefore torch) at app startup measurably takes ~15-20s on a
# single-core host during local testing, and Docker's HEALTHCHECK would
# otherwise mark the container unhealthy mid-import on modest hardware.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('API_PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health', timeout=5)" || exit 1

# Shell form (not exec-form JSON array) so ${API_HOST}/${API_PORT} are
# expanded from the container's environment at startup -- this keeps
# app/config.py's Settings as the single source of truth for host/port,
# whether the app is run locally (`python -m app.main`) or in Docker,
# rather than having two different hardcoded values that can drift apart.
CMD uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}