# Slim production image for the ChGK Model website.
#
# The DuckDB file is NOT baked into the image — it is bind-mounted at
# runtime from the host (see website/deploy/docker-compose.yml).  This
# keeps the image small (~150 MB instead of ~550 MB) and lets the DB be
# refreshed without rebuilding the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime deps only (build-only deps like pyarrow/psycopg2 are skipped).
RUN pip install --no-cache-dir \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27" \
      "jinja2>=3.1" \
      "duckdb>=1.0"

COPY website/app /app/app

# Mount point for the DuckDB file; expected to be bind-mounted read-only.
RUN mkdir -p /app/data
ENV CHGK_DB_PATH=/app/data/chgk.duckdb

EXPOSE 8000

# Single worker keeps /admin/reload-db deterministic (only 1 process to
# refresh).  Throughput is plenty for the expected load; bump --workers
# if needed and accept that reload-db only reloads the worker that
# received the POST (or `docker compose restart app` for a clean reload).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--no-access-log"]
