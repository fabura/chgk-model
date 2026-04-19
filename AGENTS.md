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

`δ = μ_type[type_t] + ε_t + δ_size[clip(team_size, 1, K)] + δ_pos[q_index_in_tournament % tour_len]` where:

- `μ_type` = systematic mode effect (`offline`, `sync`, `async`)
- `ε_t` = residual tournament offset
- `δ_size` = per-team-size shift (anchored at 6 = 0; corrects noisy-OR's
  naive composition of player contributions; see
  `docs/team_size_experiments.md`)
- `δ_pos` = per-position-in-tour shift (anchored at 0 = 0, the easiest
  position; captures the empirical "first questions easier, mid-tour
  hardest, end-tour slight rebound" pattern; see
  `docs/position_in_tour_experiments.md`)

Residual offsets are centered within type each week.

## Sequential online rating (`rating/`)

Sequential model: computes player strength changes week by week, tournament by tournament.

- **Location**: `rating/` package
- **Run**: `python -m rating --mode cached --cache_file data.npz`
- **Important defaults**: tuned `t6` mode handling from `docs/async_mode_experiments.md`, per-player calendar decay (`use_calendar_decay=True`, `rho_calendar=1.0`) — see `docs/calendar_decay_experiments.md`, learned per-team-size effect (`use_team_size_effect=True`, anchor at 6) — see `docs/team_size_experiments.md`, and learned per-position-in-tour effect (`use_pos_effect=True`, anchor at 0, `tour_len=12`) — see `docs/position_in_tour_experiments.md`. Backtest logloss on the full DB: **0.522** (was 0.602 with the old per-tournament decay; 0.532 before adding team-size; 0.527 before adding the position effect).
- **Hyperparameters**: `eta0`, `rho`, `rho_calendar`, `decay_period_days`, `cold_init_factor`, `w_online`, `w_online_questions`, `w_online_log_a`, `w_async_mode`, `w_async_residual`, `eta_mu`, `eta_eps`, `eta_size`, `eta_pos`, `reg_mu_type`, `reg_eps`, `reg_size`, `reg_pos`, `reg_theta`, `reg_b`, `reg_log_a`
- **Paired tournaments**: Uses `canonical_q_idx` — sync+async pairs share question params (b, a)
- **Tournament ordering**: By `start_datetime` (date of start, not end)

| File | Role |
|------|------|
| `rating/model.py` | Noisy-OR `forward`, gradients (stable `expm1` formulation) |
| `rating/players.py` | `PlayerState` — θ, adaptive η = η0/√(1+games) |
| `rating/questions.py` | `QuestionState` — b, log_a, init from take rate |
| `rating/decay.py` | θ ← ρ·θ between tournaments |
| `rating/tournaments.py` | `TournamentState` — `μ_type + ε_t`, centering, type prior |
| `rating/engine.py` (`delta_size`) | per-team-size shift, anchored at 6, learned online |
| `rating/engine.py` (`delta_pos`) | per-position-in-tour shift (length `tour_len`, anchored at 0), learned online |
| `rating/engine.py` | `run_sequential()` — chronological online SGD |
| `rating/backtest.py` | Time-split evaluation (logloss, Brier, AUC) |
| `rating/io.py` | `load_results_npz()` — load compact results |

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

## Key files

| File | Role |
|------|------|
| `data.py` | Index maps, `Sample`, synthetic data, `load_from_db`, cache (`.npz` compressed / `.pkl`), paired tournament detection |

## Data flow

1. **Load**: `load_from_db()` or `load_cached()` → arrays + `IndexMaps`
2. **Sequential**: `run_sequential(arrays, maps)` — processes tournaments by date

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
- **Disable tournament offsets** → `--no-tournament-delta` or `Config(use_tournament_delta=False)`
- **Add DB filter** → `load_from_db()` in `data.py`
- **Export sequential results** → `--results_npz` (compact) or `--players_out`, `--questions_out`, `--history_out` (CSV)
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

- `scripts/theta_to_prob.py` — convert θ to probability
- `scripts/lookup_players.py` — player lookup
- `scripts/build_strongest_100plus.py`, `scripts/count_*.py` — analysis

## Docs

- `docs/current_model_mechanics.md` — detailed model and filters
- `docs/interpretation.md` — θ interpretation and tables
- `docs/async_mode_experiments.md` — verified hypotheses, backtest results, chosen defaults, future work
- `docs/calendar_decay_experiments.md` — calendar-based decay vs the legacy per-tournament one
- `docs/team_size_experiments.md` — per-team-size difficulty shift (δ_size) and backtest gains
- `docs/position_in_tour_experiments.md` — per-position-in-tour shift (δ_pos), empirical curve, anchor choice, backtest gains
- `docs/calendar_decay_experiments.md` — calendar-based decay sweep, why per-tournament decay was wrong, current defaults
