# AGENTS.md — ChGK Model

Guidance for AI agents working with this codebase.

## Start here (do not rediscover layout)

| Need | Read first |
|------|------------|
| Model, hyperparams, training defaults | [`docs/model.md`](docs/model.md) |
| Navigation, data flow, doc index | [`docs/INDEX.md`](docs/INDEX.md) |
| Which file/module does what | [`docs/repo-map.md`](docs/repo-map.md) |
| DB tables and relationships | [`docs/schema/README.md`](docs/schema/README.md) → `docs/schema/*.md` |
| Website, refresh, API mirror (below) | This file |

Schema docs must stay in sync with code — see `.cursor/rules/docs-maintenance.mdc`.

## Project overview

Probabilistic ChGK model: player strength **θ**, question difficulty **b**,
discrimination **a** from team take/not-take outcomes and rosters.

**Full formula, defaults, hyperparameter list, training history:**
[`docs/model.md`](docs/model.md).

Data: rating DB `tournaments`, `tournament_results.points_mask`, `tournament_rosters`.
Load/cache: `data.py`; train: `python -m rating --mode cached --cache_file data.npz`.

## Sequential online rating (`rating/`)

See [`docs/model.md`](docs/model.md) for formula, defaults, and `Config` fields.

Quick reference:

| File | Role |
|------|------|
| `rating/engine.py` | `Config` + `run_sequential()` |
| `rating/model.py` | Noisy-OR forward + gradients |
| `rating/players.py`, `rating/questions.py` | θ and b/a state |
| `rating/backtest.py` | Cell-holdout evaluation |
| `rating/io.py` | `load_results_npz()` |

```bash
python -m rating --mode cached --cache_file data.npz --results_npz results/seq.npz
python -m rating --mode cached --cache_file data.npz --backtest
```

## Website (`website/`)

Read-only FastAPI + Jinja2 frontend over a baked DuckDB (~390 MB):

| Path | Role |
|------|------|
| `website/build/build_db.py` | Joins `data.npz`, `results/seq.npz` and the questions sqlite into `website/data/chgk.duckdb`; precomputes per-team expected takes from pre-tournament θ snapshots, a `theta_display` (inactivity-shrunk θ) column, and a `team_theta_implied` per (team, tournament) — the per-player θ that a hypothetical team of identical players of the team's actual size would need to take exactly the observed score on that pack (strips out δ_size/δ_pos; same scale as player θ; powers the team-page chart) |
| `website/app/main.py` | Routes: `/`, `/teams`, `/tournaments`, `/player/{id}`, `/team/{id}`, `/tournament/{id}`, `/search`, `/methodology`, `/admin/reload-db` (guarded by `X-Admin-Token`) |
| `website/app/db.py` | Single read-only DuckDB connection, re-opened on hot reload |
| `website/app/templates/` | `top_players.html` (sorts by `theta_display`, shows raw θ in a sub-column), `tournaments_list.html` (paginated, type filter), `teams_list.html` (paginated; default `core=offline`: ranks active teams by **venue-loyalty-weighted** mean `theta_display` of top-≤6 players by `n_app_venue` (очник+синхрон) in window — weight `share_venue * min(1, n/3)`, guests with `<2` venue games excluded, async ignored; `core=all` keeps legacy all-mode share; filter `min_eff_base` default 3.0), `player.html` (cold-start warning for <15-game rookies; inactivity warning when `theta_display` ≠ `theta`), `team.html` (expandable per-tournament rosters; roster split into venue **ядро** / остальные, sorted by `n_app_venue`; trend chart uses `team_theta_implied`), `tournament.html` (expected-vs-actual takes), `methodology.html` (cold-start + inactivity-decay sections), `search.html`, `base.html`, `_macros.html` (shared paginator) |
| `website/Dockerfile` | Production image (Python deps only — DuckDB file is bind-mounted, NOT baked) |
| `website/deploy/` | Production deployment to a single VPS behind nginx (`docker-compose.yml` + `nginx.conf` + `deploy.sh` + `refresh-db.sh`); see `website/deploy/README.md` |

Local secrets (`.admin_token`) and the on-disk DuckDB are gitignored
(`website/.gitignore` covers `data/*.duckdb*`).

### Deployment

Single VPS (`65.21.62.193`), two containers on a docker bridge:
nginx (`:80`) → app (uvicorn `:8000`).  The DuckDB file lives on the
host at `/srv/chgk-model/data/chgk.duckdb` and is bind-mounted
read-only — model rebuilds only re-rsync that file.

```bash
./website/deploy/deploy.sh --db          # full deploy (build image + ship + restart + DB)
./website/deploy/deploy.sh --image-only  # only Python/template changes
./website/deploy/refresh-db.sh           # only refresh DB after a model retrain
```

`ADMIN_TOKEN` lives in `/srv/chgk-model/.env` (chmod 600) and protects
`/admin/reload-db`.  HTTPS is intentionally not configured yet — add
Cloudflare or a certbot sidecar once a domain is registered.

### Display-only inactivity decay

The model itself does not decay θ over calendar time
(`rho_calendar = 1.0` — see `docs/experiments/mechanisms/calendar_decay_experiments.md`),
which leaves long-retired players at the very top of the raw θ board.
The website hides this artefact by precomputing a `theta_display`
column at build time (`compute_theta_display` in `website/build/build_db.py`):

```
factor       = 0.5 ** (max(0, days_inactive - grace) / halflife)
theta_display = prior + (theta - prior) * factor
```

Defaults: `grace = 365 days` (no penalty for a season off),
`halflife = 4 * 365 days` (very slow), `prior = 0.0`. So θ = 1.0
becomes 0.84 / 0.71 / 0.50 / 0.21 after 2 / 3 / 5 / 10 years of
inactivity. `theta_display` is what `/`, the `#rank`, and most
profile UI use; raw `theta` is still displayed in parentheses and
used in every historical computation (e.g. `expected_takes`).

## Daily refresh pipeline (`scripts/refresh_*.sh`)

End-to-end nightly refresh, single-instance via PID lock at
`logs/refresh.lock`:

1. `scripts/refresh_postgres.sh` — downloads the most recent
   `YYYY-MM-DD_rating.backup` from R2 (walks back day by day if today's
   dump isn't out yet — backups appear ~23:00 UTC), validates with
   `pg_restore --list` **before** touching the running DB, then
   re-restores into the local docker-compose postgres.
2. `python -m rating_api` — mirrors fresh tournaments from
   `api.rating.chgk.info` into the same Postgres (see "API mirror"
   below). Lets us pick up data that landed after the dump was taken,
   or skip step 1 entirely (`--skip-postgres`) for a daily light refresh
   when the dump is stale or broken.
3. `python -m rating --mode db --cache_file data.npz --results_npz results/seq.npz`
   — pulls `data.npz` from PG and trains in a single CLI call.
4. `python -m website.build.build_db` → `chgk.duckdb.new`.
5. Atomic `mv .new → .duckdb`, then `POST /admin/reload-db` to swap
   the inode under the running uvicorn.

```bash
./scripts/refresh_data.sh                       # full refresh
./scripts/refresh_data.sh --skip-postgres       # reuse current PG state
./scripts/refresh_data.sh --skip-api            # don't pull API deltas
./scripts/refresh_data.sh --api-only            # alias for --skip-postgres
./scripts/refresh_data.sh --skip-train          # reuse data.npz + seq.npz
./scripts/refresh_data.sh --skip-build          # don't rebuild DuckDB
SKIP_RELOAD=1 ./scripts/refresh_data.sh         # don't ping the website
```

## API mirror (`rating_api/`)

Incremental loader for `api.rating.chgk.info`. Complements (does not
replace) the dump-based path. Both write into the same local PG, so
`load_from_db` is unchanged.

| File | Role |
|------|------|
| `rating_api/client.py` | stdlib HTTP client with retry/throttle; `iter_tournaments_changed_since` walks `/tournaments?lastEditDate[strictly_after]=…&order[lastEditDate]=asc` paginated |
| `rating_api/parse.py` | JSON → dataclasses shaped like rows of `public.tournaments` / `tournament_results` / `tournament_rosters` / `tournament_editors`. `type.name` is written verbatim (matches `_normalize_type`); `questionQty` (dict per-tour) is summed into `questions_count` |
| `rating_api/upsert.py` | Per-tournament `DELETE … WHERE tournament_id=%s` + bulk INSERT in a single transaction. `tournaments.id` has no UNIQUE constraint, so ON CONFLICT isn't available; the delete-then-insert pattern is intentional |
| `rating_api/pg_state.py` | `api_overlay.fetch_state` (in a separate schema because `restore.sh` drops `public`). Discovery cursor = `MAX(public.tournaments.last_edited_at)` — the dump fills this column for every historical tournament |
| `rating_api/sync.py` | Orchestrator: discover → fetch → parse → upsert → record outcome |

```bash
python -m rating_api                       # default: write deltas since
                                            # MAX(public.tournaments.last_edited_at)
python -m rating_api --limit 5             # smoke test (cap at 5 tournaments)
python -m rating_api --since 2026-05-01    # force a specific cursor
python -m rating_api --dry-run             # inspect-only, no writes to public.*
```

**Empty-payload rule**: tournaments whose `/results` returns `[]`
(e.g. results not posted yet) keep their previous dump-loaded rows in
`tournament_results` / `tournament_rosters` — we only refresh
`tournaments` metadata and `tournament_editors` for them.

**Not mirrored**: `true_dls` (per-team granularity, FK on `models`,
API only surfaces the aggregate `trueDL`). The dump remains the source
of truth here; `load_from_db` already handles missing rows gracefully.

End-to-end takes ~20 min on macOS (most of it is the train pass).

## Key files

| File | Role |
|------|------|
| `data.py` | Index maps, `Sample`, synthetic data, `load_from_db`, cache (`.npz` compressed / `.pkl`), paired tournament detection |

## Data flow

1. **Load**: `load_from_db()` or `load_cached()` → arrays + `IndexMaps`
2. **Sequential**: `run_sequential(arrays, maps)` — processes tournaments by date
3. **Bake**: `website/build/build_db.py` produces the website DuckDB

## Index mapping

- `player_id` (DB) ↔ `player_idx` (0..num_players-1) via `IndexMaps`
- `(tournament_id, question_index)` ↔ `question_idx` via `IndexMaps`
- **Paired tournaments**: `canonical_q_idx` maps raw question slots to shared canonical params

## Conventions

- **Python**: `__future__` annotations, type hints, dataclasses
- **Russian terms**: tournament types 0=очник, 1=синхрон, 2=асинхрон

## Common tasks

- **Change sequential hyperparams** → `Config` in `rating/engine.py`
- **Tune hyperparams** → `python -m rating --mode cached --cache_file data.npz --tune` (grid search) or `--tune --tune-trials 24` (random search)
- **Per-mode and per-tournament offsets** → removed in 2026-04; only
  `δ_size` and `δ_pos` remain (use `--no-use-team-size-effect` /
  `--no-use-pos-effect` to disable individually)
- **Add DB filter** → `load_from_db()` in `data.py`
- **Export sequential results** → `--results_npz` (compact) or `--players_out`, `--questions_out`, `--history_out` (CSV)
- **Refresh production data** → `./scripts/refresh_data.sh` (see above)
- **Hot-reload website only** → `curl -X POST -H "X-Admin-Token: $(cat website/.admin_token)" http://127.0.0.1:8765/admin/reload-db`
- **Interpret θ** → `docs/interpretation.md`

## Cache

- **`.npz`** — compressed, ~50× smaller than `.pkl`, faster load. Prefer for new caches.
- **Convert** existing `.pkl` → `.npz`:
  `python data.py --convert_cache data/cache_all.pkl data/cache_all.npz`

## Setup

```bash
pip install -r requirements.txt
# DB: set DATABASE_URL or use --cache_file for cached runs
```

## Scripts

- `scripts/refresh_data.sh`, `scripts/refresh_postgres.sh` — daily refresh pipeline
- `scripts/fetch_venue_overlay.py` — sync venue assignments from
  `api.rating.chgk.info` into `data/venue_overlay.duckdb` (see `venue_overlay/`)
- `scripts/analyse_venue_effects.py` — residual slices by venue size (mono vs multi)
- `scripts/exp_cold_start_grid.py`, `scripts/exp_cold_start_grid_extra.py` — `(θ_init, games_offset)` sweeps
- `scripts/run_simple_experiments.py` — single-knob configuration sweeps (calendar decay etc.)
- `scripts/compare_to_baselines.py` — side-by-side variant comparison on the backtest split
- `scripts/question_uncertainties.py` — posterior std on b / a per question
- `scripts/show_top_players.py` — current top-N by θ with name lookup
- `scripts/theta_to_prob.py` — convert θ to probability
- `scripts/lookup_players.py` — player lookup
- `scripts/build_strongest_100plus.py`, `scripts/count_*.py` — analysis

## Refactor backlog

- **Unify the noisy-OR + lapse + recal kernel.** The same formula is
  currently implemented twice: `website/build/build_db.py::compute_expected_takes`
  (bake-time, batched per tournament, also computes `_solve_implied_theta`)
  and `rating/simulate.py::simulate_roster_on_pack` (request-time,
  per-roster, used by `/forecast/*`).  They are logically identical
  (verified via the `model_params` JSON written by `build_db`), but
  structurally duplicated.  Fixing requires a PR whose only goal is
  bit-exact agreement: route `compute_expected_takes` through
  `simulate_roster_on_pack`, then diff the resulting `expected_takes`
  in DuckDB against the previous build and assert ≤ 1 ULP per row.

## Docs

- `docs/INDEX.md` — documentation hub (start here)
- `docs/model.md` — formula, Config defaults, training history
- `docs/repo-map.md` — modules, scripts, routes, typical commands
- `docs/schema/` — Postgres, DuckDB, npz, questions.db, overlays (tables + ER)
- `docs/experiments/README.md` — layout of experiment docs
- `docs/experiments/experiments_summary_ru.md` — Russian index (✅/❌/⚠️); **update in the same PR** when promoting/rejecting experiments or changing `Config` defaults
- `docs/experiments/mechanisms/` — long-lived production mechanisms
- `docs/experiments/cycles/` — monthly ablation cycles (2026-04 …)
- `docs/experiments/analysis/` — observational reports and post drafts (no Config changes)
- `docs/interpretation.md` — θ interpretation and tables
