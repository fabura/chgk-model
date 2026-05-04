# Config cleanup, 2026-05

After the leakage-free re-validation of the model
(`docs/leakage_2026-05.md`), several `Config` knobs that had been
either dead-by-default or shown to provide no value were removed.

## Removed parameters and code paths

### `rho`, `use_calendar_decay`, `apply_decay`

* **What**: legacy global per-tournament rating decay (`θ ← ρ · θ`
  applied once per tournament for every player).
* **Why removed**: `use_calendar_decay=True` was the production default
  since the calendar-decay experiments (`docs/calendar_decay_experiments.md`)
  showed that the per-tournament path cut backtest logloss from 0.532 to
  0.602.  Calendar-decay (per-player, per-day) replaced it. The `rho`
  knob and `apply_decay` function were only reachable via
  `use_calendar_decay=False`, which had not been a useful configuration
  in over a year.
* **Migration**: configure decay solely via `rho_calendar` (default
  `1.0` = disabled) and `decay_period_days` (default `7.0` = per-week
  natural unit).

### `cold_init_factor`, `cold_init_use_team_mean`

* **What**: legacy "inherit team-mean θ" path for first-time players,
  with optional shrinkage toward `cold_init_theta`.
* **Why removed**: `cold_init_use_team_mean=False` had been the
  production default since the cold-start grid sweep
  (`scripts/exp_cold_start_grid.py`,
  `docs/cold_start_experiments.md` — implicit) showed that fixed-prior
  cold-start with the chess-Elo "rookie boost"
  (`games_offset=0.25`) eliminated the multi-year θ drift that
  team-mean inheritance produced. With team-mean off,
  `cold_init_factor` is dead.
* **Migration**: configure cold-start solely via `cold_init_theta`
  (default `-1.0`) and `games_offset` (default `0.25`).

## Defaults flipped

### `freeze_log_a: bool = False → True`

* **What**: stop learning per-question discrimination `a_i`; the
  `forward` pass uses `a = exp(0) = 1` for every question.
* **Why**: the cell-holdout ablation
  (`results/exp_holdout_ablations.csv`) showed `freeze_log_a=True`
  gives an **identical** (0.5078 vs 0.5083) honest logloss to the
  learnable variant, with a microscopic improvement on async
  (-0.0016).  Removing ~25 k learnable parameters also makes SGD
  marginally faster.  The knob is preserved for ablation: pass
  `--no-freeze-log-a` to re-enable learning.

### `team_size_max: int = 8 → 12`

* **What**: separate `δ_size` per size up to 12 (sizes 13+ collapse
  into `[12]`).
* **Why**: under cell-holdout, the `[10+]` collapse bucket was
  under-predicted by ~3 p.p.; `team_size_max=12` cuts that to ~1 p.p.
  on those observations.  Overall logloss is unchanged (those sizes
  are 0.36 % of obs), but the local calibration improves and the
  cost is negligible (a few extra learnable scalars).  See
  `results/exp_size_max.csv`.

## CLI default flipped

### `--backtest` now implies `--holdout 0.10`

* **What**: `python -m rating --backtest` now runs in honest cell-
  holdout mode by default (10 % of observations are randomly held
  out, deterministic via `--holdout-seed=42`, and the metric is
  computed only on those cells).
* **Why**: the legacy time-split was leaky by ~+5 % logloss overall
  (~+16 % on offline tournaments) — see `docs/leakage_2026-05.md`.
  Reaching for the honest mode every single time is cumbersome and
  error-prone.  The legacy mode is still available via
  `--holdout 0.0` for direct comparison to historical numbers.

## Affected scripts

Four historical experiment scripts that used removed knobs were
either fixed (where the removed knob's value was the new
not-an-option default) or guarded with a `sys.exit(...)` notice:

| Script | Action |
|---|---|
| `scripts/exp_cold_start.py` | Guarded — needed both `True` and `False` of `cold_init_use_team_mean` for the A/B comparison. Saved output: `results/exp_cold_start.json`. |
| `scripts/exp_cold_start_grid.py` | Fixed — dropped `cold_init_use_team_mean=False` (now default behaviour). |
| `scripts/exp_cold_start_grid_extra.py` | Fixed — same. |
| `scripts/run_simple_experiments.py` | Edited — dropped `cold_init_0.5` and `combo_v1` variants, simplified `cal_decay_*` to use `rho_calendar` alone. |

Restore the relevant scripts from git history before this commit if
you need to reproduce the legacy-knob baselines exactly.
