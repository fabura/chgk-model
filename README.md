# ChGK Model

Probabilistic rating for Что? Где? Когда?: estimates player strength **θ**,
question difficulty **b**, and question discrimination **a** from team
take/not-take outcomes and rosters.

The project has three main parts:

- `rating/` trains the sequential online model.
- `website/` serves a FastAPI + Jinja2 site over a baked DuckDB.
- `scripts/refresh_*.sh` keep Postgres, model artifacts, and the website DB fresh.

For model details, table schemas, and navigation, start with
[`docs/INDEX.md`](docs/INDEX.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # or Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Common Commands

```bash
# Synthetic smoke test
python -m rating --mode synthetic

# Build training cache from a restored rating-db Postgres
export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/postgres
python -m rating --mode db --cache_file data.npz

# Train from cache and export model artifacts
python -m rating --mode cached --cache_file data.npz --results_npz results/seq.npz

# Run focused tests
python -m pytest tests/ -q
```

## Data Flow

```
rating-db Postgres
  -> data.py
  -> data.npz
  -> python -m rating
  -> results/seq.npz
  -> website/build/build_db.py
  -> website/data/chgk.duckdb
  -> website/app
```

Details:

- Input schema: [`docs/schema/postgres.md`](docs/schema/postgres.md)
- Cache/results formats: [`docs/schema/cache.md`](docs/schema/cache.md)
- Website DuckDB schema: [`docs/schema/duckdb.md`](docs/schema/duckdb.md)
- Full flow and relationships: [`docs/schema/README.md`](docs/schema/README.md)

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/INDEX.md`](docs/INDEX.md) | Documentation hub and data flow |
| [`docs/model.md`](docs/model.md) | Formula, defaults, training history |
| [`docs/repo-map.md`](docs/repo-map.md) | Repository structure and module roles |
| [`docs/schema/`](docs/schema/) | Postgres, DuckDB, npz, overlays |
| [`docs/interpretation.md`](docs/interpretation.md) | How to interpret θ |
| [`docs/experiments/experiments_summary_ru.md`](docs/experiments/experiments_summary_ru.md) | Experiment index |
| [`docs/experiments/README.md`](docs/experiments/README.md) | Experiment docs layout |
| [`website/README.md`](website/README.md) | Local website build/run/deploy |
| [`AGENTS.md`](AGENTS.md) | Operational context for AI agents |

## Repository Layout

| Path | Purpose |
|------|---------|
| `data.py` | Load rating DB data, build `IndexMaps`, read/write cache |
| `rating/` | Sequential model, training engine, simulation, backtest |
| `rating_api/` | Incremental mirror from `api.rating.chgk.info` |
| `website/` | Public site and DuckDB build pipeline |
| `venue_overlay/` | Venue overlay DuckDB for venue analysis |
| `scripts/` | Refresh pipeline, experiments, diagnostics, reports |
| `tests/` | Unit and regression tests |
| `docs/` | Project documentation and schema contracts |
| `docs/experiments/` | Experiment history, ablations, analysis drafts |

More detail: [`docs/repo-map.md`](docs/repo-map.md).

## Website and Refresh

```bash
# Full local refresh: Postgres -> data.npz -> seq.npz -> DuckDB -> hot reload
./scripts/refresh_data.sh

# Development website
CHGK_DB_PATH=$(pwd)/website/data/chgk.duckdb PYTHONPATH=website \
  uvicorn app.main:app --reload --port 8000
```

Website commands, admin token, Docker, and deploy details live in
[`website/README.md`](website/README.md) and `website/deploy/README.md`.
