# Serving image for the Rossmann demand forecast API.
#
# Ships the trained artifacts and the two modules needed to serve them.
# Training code, MLflow, and the 38 MB source CSVs are deliberately absent:
# the container serves a model, it does not build one.

FROM python:3.13-slim

# libgomp is LightGBM's OpenMP runtime. The wheel links against it but the
# slim base image does not include it, and the failure appears at import time
# rather than at build time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the
# model or the code, so rebuilds after a retrain reuse this layer.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Trained artifacts: booster, frozen categories, precomputed feature table.
COPY serving/ /app/serving/

# The only two modules the service needs. encoding.py is shared with the
# training pipeline so the serve-time encoding cannot drift from it.
COPY src/api.py src/encoding.py src/landing.html /app/src/

# Run unprivileged. Nothing here needs to write to disk.
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

# Cloud Run injects PORT and expects the process to bind it on 0.0.0.0.
# 8080 is the default it uses; the fallback keeps `docker run` simple locally.
ENV PORT=8080
EXPOSE 8080

# One worker on purpose. The model and the 8.2 MB feature table load into
# memory at import, so each extra worker duplicates them. Cloud Run scales by
# adding container instances, not by adding workers inside one.
CMD ["sh", "-c", "uvicorn api:app --app-dir /app/src --host 0.0.0.0 --port ${PORT}"]
