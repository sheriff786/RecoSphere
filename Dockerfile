# RecoSphere serving image.
# Models (models/) and gold data (data/gold/) are mounted as volumes, not baked in:
# retraining (Phase 7) swaps files without rebuilding the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    RANKFORGE_DATA=/app/data \
    RANKFORGE_REPO=/app \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prom \
    EXPERIMENT_CONFIG=/app/experiments/active.yaml \
    EXPOSURE_DIR=/app/ab_logs

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-serve.txt

COPY eval/ eval/
COPY retrieval/ retrieval/
COPY ranking/ ranking/
COPY serve/ serve/
COPY ab/ ab/

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fs http://localhost:8000/health || exit 1
# API_WORKERS > 1 needs REDIS_URL (workers share user state through Redis)
# PROMETHEUS_MULTIPROC_DIR must be EMPTY at start (stale files from a previous container = wrong counters)
CMD ["sh", "-c", "rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && uvicorn serve.app:app --host 0.0.0.0 --port 8000 --workers ${API_WORKERS:-1}"]
