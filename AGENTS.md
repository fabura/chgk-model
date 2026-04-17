# AGENTS.md — ChGK Model

Guidance for AI agents working with this codebase.

## Project overview

**ChGK** (Что? Где? Когда?) is a probabilistic model that estimates:

- **θ** (theta) — player strength
- **b** — question difficulty
- **a** — question discrimination (selectivity)

from binary team answers (taken / not taken) and team rosters. Data comes from the rating DB: `tournaments`, `tournament_results.points_mask`, `tournament_rosters`.

Core formula (noisy-OR): 
`z_k = -(b_i + δ_t) + a_i * θ_k` → `λ_k = exp(z_k)` → `S = Σ_k λ_k` → `p_take = 1 - exp(-S)`

`δ_t = μ_type[type_t] + ε_t` where:

- `μ_type` = systematic mode effect (`offline`, `sync`, `async`)
- `ε_t` = residual tournament offset

Residual offsets are centered within type each week.

## Sequential online rating (`rating/`)

Sequential model: computes player strength changes week by week, tournament by tournament.

- **Location**: `rating/` package
- **Run**: `python -m rating --mode cached --cache_file data.npz`
- **Important defaults**: current default mode-handling is the tuned `t6` configuration from `docs/async_mode_experiments.md`
- **Hyperparameters**: `eta0`, `rho`, `w_online`, `w_online_questions`, `w_online_log_a`, `w_async_mode`, `w_async_residual`, `eta_mu`, `eta_eps`, `reg_mu_type`, `reg_eps`
- **Paired tournaments**: Uses `canonical_q_idx` — sync+async pairs share question params (b, a)
- **Tournament ordering**: By `start_datetime` (date of start, not end)

| File | Role |
|------|------|
| `rating/model.py` | Noisy-OR `forward`, gradients (stable `expm1` formulation) |
| `rating/players.py` | `PlayerState` — θ, adaptive η = η0/√(1+games) |
| `rating/questions.py` | `QuestionState` — b, log_a, init from take rate |
| `rating/decay.py` | θ ← ρ·θ between tournaments |
| `rating/tournaments.py` | `TournamentState` — `μ_type + ε_t`, centering, type prior |
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
