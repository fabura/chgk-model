# Slim production image for the ChGK Model website.
# Expects chgk.duckdb to be baked into the image at /app/data/chgk.duckdb.
# Build the DB locally (see README) and copy it to website/data/ before docker build.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Only the runtime deps (skip build-only ones).
COPY website/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27" \
      "jinja2>=3.1" \
      "duckdb>=1.0"

COPY website/app /app/app
COPY website/data/chgk.duckdb /app/data/chgk.duckdb

ENV CHGK_DB_PATH=/app/data/chgk.duckdb
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
