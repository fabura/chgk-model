# ChGK probabilistic model

Sequential online rating: estimates **player strength** θ_k, **question difficulty** b_i, and **question discrimination** a_i from binary team answers (taken / not taken) and team compositions. Processes tournaments week by week.

Current default handling of tournament modes:

- per-mode (`mu_type`) and per-tournament (`eps_t`) shifts were removed
  in 2026-04 — see `AGENTS.md` and the comment in `rating/tournaments.py`
- `offline` is the baseline mode
- `async` updates are intentionally weaker for players, question
  difficulty, and especially question discrimination (per-mode update
  weights `w_online` / `w_sync` / `w_offline` in `rating.engine`)
- tuned defaults currently use the `t6` configuration from
  `docs/async_mode_experiments.md`

Default decay model:

- `use_calendar_decay=True`, `rho_calendar=1.0` — no decay applied;
  the per-tournament global decay was found to throw away most of the
  accumulated history (cumulative multiplier ≈ 0.014 over 8 years).
  Switching to per-player calendar decay cut backtest logloss from
  0.602 → 0.532. Details: `docs/calendar_decay_experiments.md`.

## Data mapping from rating DB

- **Games**: `public.tournaments` (one tournament = one game with `questions_count` questions).
- **Question-level outcomes**: `public.tournament_results.points_mask` — string of `'0'`/`'1'` per question (1 = taken).
- **Rosters**: `public.tournament_rosters` (tournament_id, team_id, player_id).

So we have: `(game_id, question_id, team_id, taken)` with `game_id = tournament_id`, `question_id = (tournament_id, question_index)` mapped to a global question index.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # or Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run synthetic demo

```bash
python -m rating --mode synthetic
```

## Run on real DB (after `docker-compose up` in rating-db)

```bash
export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/postgres
python -m rating --mode db --cache_file data.npz
```

### Cache: avoid re-querying the DB

First run: load from DB and save to a cache file. Later runs: load from the file (no DB needed).

```bash
# First run: fetches from DB and writes data.npz
python -m rating --mode db --cache_file data.npz

# Next runs: use cache only (DB can be stopped)
python -m rating --mode cached --cache_file data.npz
```

Use different cache paths for different options (e.g. `data/cache_100.npz` for `--max_tournaments 100`). Delete the file to force a fresh DB export.

### Export results

```bash
# Compact .npz (players, questions, history)
python -m rating --mode cached --cache_file data.npz --results_npz results/seq.npz

# CSV exports
python -m rating --mode cached --cache_file data.npz \
    --players_out results/seq_players.csv \
    --questions_out results/seq_questions.csv
```

### Интерпретация силы θ

Сила игрока θ задаёт его вклад в вероятность взятия вопроса; сама вероятность зависит ещё от сложности вопроса \(b_i\), дискриминации \(a_i\) и состава команды. В **эталонном** случае (один игрок, «средний» вопрос с \(b=0\), \(a=1\)): вероятность взять вопрос = \(1 - \exp(-\exp(\theta))\) — например θ=0 → ~63%, θ=1 → ~93%. Подробно и таблицы: [docs/interpretation.md](docs/interpretation.md). Перевести θ в вероятность: `python scripts/theta_to_prob.py 0.5 1.0` или `python scripts/theta_to_prob.py --table`.

## Periodic data refresh

The model + website live downstream of the rating-DB postgres dump. A
fresh dump appears in R2 every ~23:00 UTC; the helper scripts pull it
through the whole pipeline and hot-reload the running site.

```bash
# One-off full refresh (≈10–15 min: postgres restore + train + DuckDB build)
./scripts/refresh_data.sh

# Skip stages for faster iteration:
./scripts/refresh_data.sh --skip-postgres   # reuse current rating-db
./scripts/refresh_data.sh --skip-train      # reuse data.npz + seq.npz
./scripts/refresh_data.sh --skip-build      # don't rebuild DuckDB
SKIP_RELOAD=1 ./scripts/refresh_data.sh     # don't ping the website
```

What it does:

1. `scripts/refresh_postgres.sh` — `download.sh` from `rating-db/`, then
   re-runs `restore.sh` inside the running postgres container.
2. `python -m rating --mode db --cache_file data.npz --results_npz results/seq.npz`
   — pull observations, train the sequential model, export the compact
   results.
3. `python -m website.build.build_db` — rebuild `chgk.duckdb.new`,
   atomic `mv` over `chgk.duckdb` (the running uvicorn keeps reading
   the old inode through its open fd until reload).
4. `POST /admin/reload-db` with `X-Admin-Token` — close + re-open the
   site's DuckDB connection so new data becomes visible.

A PID-file lock (`logs/refresh.lock`) prevents two refreshes from
running concurrently. Logs land in `logs/refresh-<UTC-timestamp>.log`,
with `logs/refresh-latest.log` symlinked to the most recent run.

### Admin token

The website reads its admin token from `$ADMIN_TOKEN` (env var, takes
precedence) or `website/.admin_token` (gitignored). Generate one with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))" \
  > website/.admin_token
chmod 600 website/.admin_token
```

Restart uvicorn so it picks up the token. Without a token configured,
`/admin/reload-db` and `/admin/db-info` return HTTP 503.

### Cron / launchd

Daily refresh ~01:00 UTC (after the new R2 backup is published):

```cron
# crontab -e
0 1 * * * cd /Users/fbr/Projects/personal/сhgk-model && ./scripts/refresh_data.sh >> logs/cron.log 2>&1
```

For launchd on macOS, drop a plist into `~/Library/LaunchAgents/`
calling the same script with `StartCalendarInterval` set to `Hour=1`.

## Layout

- `data.py` — data loader, index maps, synthetic data, DB loader (points_mask → samples).
- `rating/` — sequential online rating: model, engine, players, questions, decay, tournaments.
- `scripts/refresh_*.sh` — periodic data-refresh pipeline (postgres → train → DuckDB → hot-reload).
- `website/` — FastAPI site reading from `website/data/chgk.duckdb`. Admin endpoints: `/admin/reload-db` (POST), `/admin/db-info` (GET).

## Notes

- `docs/async_mode_experiments.md` — checked hypotheses, measured
  backtest results, chosen defaults, and future ideas for mode handling
