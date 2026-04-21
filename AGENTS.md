# AGENTS.md — ChGK Model

Guidance for AI agents working with this codebase.

## Project overview

**ChGK** (Что? Где? Когда?) is a probabilistic model that estimates:

- **θ** (theta) — player strength
- **b** — question difficulty
- **a** — question discrimination (selectivity)

from binary team answers (taken / not taken) and team rosters. Data comes from the rating DB: `tournaments`, `tournament_results.points_mask`, `tournament_rosters`.

Core formula (noisy-OR):
`z_k = -(b_i + δ) + a_i * θ_k` → `λ_k = exp(z_k)` → `S = Σ_k λ_k` → `p_take = 1 - exp(-S)`

`δ = δ_size[clip(team_size, 1, K)] + δ_pos[q_index_in_tournament % tour_len]` where:

- `δ_size` = per-team-size shift (anchored at 6 = 0; corrects noisy-OR's
  naive composition of player contributions; see
  `docs/team_size_experiments.md`)
- `δ_pos` = per-position-in-tour shift (anchored at 0 = 0, the easiest
  position; captures the empirical "first questions easier, mid-tour
  hardest, end-tour slight rebound" pattern; see
  `docs/position_in_tour_experiments.md`)

The per-mode (`μ_type`) and per-tournament (`ε_t`) offsets were removed
in 2026-04 after an ablation showed those 8 746 parameters were
net-negative for backtest quality (logloss −0.0043, AUC +0.0044 when
removed). Mode differences are now captured only via the per-mode
update weights (`w_online`, `w_sync`, …) in `rating.engine`.

## Sequential online rating (`rating/`)

Sequential model: computes player strength changes week by week, tournament by tournament.

- **Location**: `rating/` package
- **Run**: `python -m rating --mode cached --cache_file data.npz`
- **Important defaults**:
  - tuned `t6` mode handling — see `docs/async_mode_experiments.md`
  - per-player calendar decay (`use_calendar_decay=True`,
    `rho_calendar=1.0`) — see `docs/calendar_decay_experiments.md`
  - learned per-team-size effect (`use_team_size_effect=True`,
    anchor at 6) — see `docs/team_size_experiments.md`
  - learned per-position-in-tour effect (`use_pos_effect=True`,
    anchor at 0, `tour_len=12`) — see `docs/position_in_tour_experiments.md`
  - separate solo update channel (`use_solo_channel=True`,
    `w_solo=0.3`) — observations with `team_size==1` are routed
    through their own gradient pass with a much smaller θ-update
    weight, so prolific soloists in online quizzes (M-Лига etc.)
    don't get artefactually inflated θ via the noisy-OR
    identifiability shortcut. See `docs/solo_channel_experiments.md`.
  - fixed cold-start prior (`cold_init_theta=-1.0`,
    `cold_init_use_team_mean=False`) plus chess-Elo "rookie boost"
    (`games_offset=0.25`, so first-game η = 2·η0) — breaks the
    team-mean inheritance feedback loop that produced multi-year
    population θ drift; rationale and the 12-cell sweep that picked
    these defaults are in `scripts/exp_cold_start_grid.py` (and the
    extra boundary sweep in `..._extra.py`).
  - Backtest logloss on the full DB: **0.522** (was 0.602 with the
    old per-tournament decay; 0.532 before adding team-size; 0.527
    before the position effect; ~1.3 % further improvement from the
    fixed cold-start prior).
- **Hyperparameters**: `eta0`, `rho`, `rho_calendar`, `decay_period_days`,
  `cold_init_theta`, `cold_init_use_team_mean`, `cold_init_factor`,
  `games_offset`, `w_online`, `w_online_questions`, `w_online_log_a`,
  `w_offline`, `w_sync`, `eta_size`, `eta_pos`, `eta_teammate`,
  `reg_size`, `reg_pos`, `reg_theta`, `reg_b`, `reg_log_a`,
  `team_size_max`, `team_size_anchor`, `w_size_offline/sync/async`,
  `tour_len`, `pos_anchor`, `use_solo_channel`, `w_solo`,
  `w_solo_questions`, `w_solo_log_a`, `w_size_solo`, `w_pos_solo`,
  `recenter_period_days`, `recenter_target`, `recenter_min_games`,
  `recenter_active_days`. Full list in `Config` (`rating/engine.py`).
- **Drift fix (yearly gauge re-centering)**: every
  `recenter_period_days` (365 by default) the median θ of "active
  veterans" (`games >= recenter_min_games=200`, seen within
  `recenter_active_days=365`) is pinned to `recenter_target` (default
  **−0.70**, tuned via backtest sweep). Implemented as a strict gauge
  transform — `θ ↑ Δ`, `b ↑ a·Δ` — so predictions are exactly
  invariant; the only effect is to keep absolute θ comparable across
  years and stop the multi-year cold-start drift (median θ used to
  drift from −0.15 in 2020 to −0.66 in 2025; now pinned at −0.70 with
  per-year Δ < 0.04). Also yields a small predictive-quality win
  (logloss 0.5350 at target=−0.80, 0.5357 at −0.70 vs 0.5486 with
  re-centering disabled). Filter for "season aggregate" tournaments
  (`exclude_seasonal_aggregates=True` in `data.py`) is applied at load
  time before training to remove "12 граней"-style broken
  `points_mask` rows.
- **Re-tuned defaults (2026-04, post-cleanup)**: after the seasonal-
  aggregate filter and yearly re-centering, the per-update step sizes
  needed re-tuning. A focused sweep (`/tmp/chgk_retune*.py`,
  `results/retune_2026-04*.csv`, 38 trials) on the 20 % time-split
  hold-out picked **`eta0=0.05`** (was 0.10), **`w_sync=0.7`** (was
  0.9) and **`w_online_questions=0.30`** (was 0.45) as the then-current
  `Config` defaults.
- **2026-04 lean refactor**: a follow-up ablation removed the per-mode
  shift `μ_type` and per-tournament residual `ε_t` (8 746 params,
  net-negative for backtest), added a small teammate-θ shrinkage
  (`eta_teammate=0.005`) to soften the noisy-OR identifiability
  problem on stable rosters, and re-tuned `eta0` (`0.05 → 0.07`) for
  the leaner model. Cumulative gain on the 20 % hold-out:
  `logloss 0.5365 → 0.5309` (−0.0056), `AUC 0.8065 → 0.8115` (+0.0050).
  The full sweep tables live in `/tmp/exp_*.py`.
- **Paired tournaments**: Uses `canonical_q_idx` — sync+async pairs share question params (b, a)
- **Tournament ordering**: By `start_datetime` (date of start, not end)

| File | Role |
|------|------|
| `rating/model.py` | Noisy-OR `forward`, gradients (stable `expm1` formulation) |
| `rating/players.py` | `PlayerState` — θ, adaptive η = η0/√(games_offset + games); fixed-prior or team-mean cold-start |
| `rating/questions.py` | `QuestionState` — b, log_a, init from take rate |
| `rating/decay.py` | θ ← ρ·θ between tournaments (or per-week calendar decay) |
| `rating/tournaments.py` | Tournament-type encoding helpers (`TYPE_OFFLINE/SYNC/ASYNC`, `game_type_to_idx`); the per-mode/per-tournament shift was removed in 2026-04 |
| `rating/engine.py` (`delta_size`) | per-team-size shift, anchored at 6, learned online |
| `rating/engine.py` (`delta_pos`) | per-position-in-tour shift (length `tour_len`, anchored at 0), learned online |
| `rating/engine.py` | `Config` + `run_sequential()` — chronological online SGD |
| `rating/backtest.py` | Time-split evaluation (logloss, Brier, AUC) |
| `rating/io.py` | `load_results_npz()` / `save_results_npz()` — compact results |

```bash
# From DB (prefer .npz — compressed, faster load)
python -m rating --mode db --cache_file data.npz

# From cache
python -m rating --mode cached --cache_file data.npz

# Backtest
python -m rating --mode cached --cache_file data.npz --backtest

# Export (compact .npz or CSV)
python -m rating --mode cached --cache_file data.npz --results_npz results/seq.npz
python -m rating --mode cached --cache_file data.npz \
    --players_out results/seq_players.csv \
    --questions_out results/seq_questions.csv
```

## Website (`website/`)

Read-only FastAPI + Jinja2 frontend over a baked DuckDB (~390 MB):

| Path | Role |
|------|------|
| `website/build/build_db.py` | Joins `data.npz`, `results/seq.npz` and the questions sqlite into `website/data/chgk.duckdb`; precomputes per-team expected takes from pre-tournament θ snapshots, a `theta_display` (inactivity-shrunk θ) column, and a `team_theta_implied` per (team, tournament) — the per-player θ that a hypothetical team of identical players of the team's actual size would need to take exactly the observed score on that pack (strips out δ_size/δ_pos; same scale as player θ; powers the team-page chart) |
| `website/app/main.py` | Routes: `/`, `/teams`, `/tournaments`, `/player/{id}`, `/team/{id}`, `/tournament/{id}`, `/search`, `/methodology`, `/admin/reload-db` (guarded by `X-Admin-Token`) |
| `website/app/db.py` | Single read-only DuckDB connection, re-opened on hot reload |
| `website/app/templates/` | `top_players.html` (sorts by `theta_display`, shows raw θ in a sub-column), `tournaments_list.html` (paginated, type filter), `teams_list.html` (paginated, ranks active teams by mean `theta_display` of their top-≤6 most-frequent players in a configurable window; filter by `min_base` excludes lone-wolf teams), `player.html` (cold-start warning for <15-game rookies; inactivity warning when `theta_display` ≠ `theta`), `team.html` (expandable per-tournament rosters; trend chart uses `team_theta_implied` instead of raw take counts), `tournament.html` (expected-vs-actual takes), `methodology.html` (cold-start + inactivity-decay sections), `search.html`, `base.html`, `_macros.html` (shared paginator) |
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
(`rho_calendar = 1.0` — see `docs/calendar_decay_experiments.md`),
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
2. `python -m rating --mode db --cache_file data.npz --results_npz results/seq.npz`
   — pulls `data.npz` from PG and trains in a single CLI call.
3. `python -m website.build.build_db` → `chgk.duckdb.new`.
4. Atomic `mv .new → .duckdb`, then `POST /admin/reload-db` to swap
   the inode under the running uvicorn.

```bash
./scripts/refresh_data.sh                  # full refresh
./scripts/refresh_data.sh --skip-postgres  # reuse current PG state
./scripts/refresh_data.sh --skip-train     # reuse data.npz + seq.npz
./scripts/refresh_data.sh --skip-build     # don't rebuild DuckDB
SKIP_RELOAD=1 ./scripts/refresh_data.sh    # don't ping the website
```

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
- `scripts/exp_cold_start_grid.py`, `scripts/exp_cold_start_grid_extra.py` — `(θ_init, games_offset)` sweeps
- `scripts/run_simple_experiments.py` — single-knob configuration sweeps (calendar decay etc.)
- `scripts/compare_to_baselines.py` — side-by-side variant comparison on the backtest split
- `scripts/question_uncertainties.py` — posterior std on b / a per question
- `scripts/show_top_players.py` — current top-N by θ with name lookup
- `scripts/theta_to_prob.py` — convert θ to probability
- `scripts/lookup_players.py` — player lookup
- `scripts/build_strongest_100plus.py`, `scripts/count_*.py` — analysis

## Docs

- `docs/current_model_mechanics.md` — detailed model and filters
- `docs/interpretation.md` — θ interpretation and tables
- `docs/async_mode_experiments.md` — async/sync/offline mode effects, verified hypotheses, chosen `t6` defaults
- `docs/calendar_decay_experiments.md` — calendar-based decay sweep, why per-tournament decay was wrong, current defaults
- `docs/team_size_experiments.md` — per-team-size difficulty shift (δ_size) and backtest gains
- `docs/position_in_tour_experiments.md` — per-position-in-tour shift (δ_pos), empirical curve, anchor choice, backtest gains
