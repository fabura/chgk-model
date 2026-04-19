# Calendar Decay Experiments

This note captures the cheap-but-impactful changes evaluated against
the previously-tuned `t6` baseline (see
`docs/async_mode_experiments.md`).  The biggest win came from replacing
the global per-tournament decay with a per-player calendar-based decay.

## Why the per-tournament decay was wrong

The old `apply_decay(theta, rho)` was called *once per tournament* in
chronological order, with `rho=0.9995`.  The dataset has 8595
tournaments over ~8 years (~21 tournaments/week).  The cumulative
multiplier on the strength of every player by the end of training is

    0.9995 ** 8595 ≈ 0.014

In other words: every player's accumulated θ was effectively wiped out;
the "current" rating was dominated by the last few hundred SGD steps.
Two undesirable consequences:

1. The model used only a short trailing history despite owning a
   chronological signal of 8 years.  Most of its evidence was thrown
   away on every step.
2. Active and inactive players were decayed identically, so a player
   present in 200 tournaments per year was punished as much as a
   long-retired one.  The decay had no relation to elapsed time.

## Replacement: calendar decay applied per-player

The new mechanism (`rating/decay.py::apply_calendar_decay`) decays each
player's θ *only when that player next appears in a tournament*, by

    theta_k *= rho_calendar ** (Δdays / decay_period_days)

with `decay_period_days = 7` (per-week unit).  ``last_seen_ordinal[k]``
is updated by the engine to the ordinal of the tournament's
`start_datetime`.  When `rho_calendar = 1.0` no decay is applied at
all.

## Backtest sweep on full data (32 296 888 obs, 1 719 test tournaments)

All numbers come from the prequential backtest in `rating/backtest.py`
(predictions made before each test tournament is used for updates).

| config                | logloss | Brier  | AUC    | Δ logloss vs base |
|-----------------------|--------:|-------:|-------:|------------------:|
| baseline (t6)         | 0.6019  | 0.2018 | 0.8145 |              0    |
| reg_theta = 0.001     | 0.6028  | 0.2021 | 0.8144 |          +0.001   |
| reg_theta = 0.01      | 0.6096  | 0.2044 | 0.8132 |          +0.008   |
| cold_init = 0.5       | 0.6032  | 0.2022 | 0.8143 |          +0.001   |
| cal_decay rho=0.99    | 0.5959  | 0.1997 | 0.8155 |          −0.006   |
| cal_decay rho=0.995   | 0.5763  | 0.1931 | 0.8180 |          −0.026   |
| cal_decay rho=0.997   | 0.5627  | 0.1885 | 0.8195 |          −0.039   |
| cal_decay rho=0.998   | 0.5537  | 0.1854 | 0.8204 |          −0.048   |
| cal_decay rho=0.999   | 0.5433  | 0.1818 | 0.8213 |          −0.059   |
| **cal_decay rho=1.0** | **0.5318** | **0.1777** | **0.8220** |   **−0.070** |

### Per-tournament-type breakdown of the chosen default

`cal_decay rho=1.0` (the new default):

| type    | n_obs       | logloss | Brier  | AUC    |
|---------|------------:|--------:|-------:|-------:|
| offline |   1 008 576 | 0.4826  | 0.1581 | 0.8725 |
| sync    |   2 201 059 | 0.5389  | 0.1814 | 0.8165 |
| async   |   1 238 185 | 0.5592  | 0.1870 | 0.7906 |

The biggest improvement is on offline (logloss −0.166 vs t6
baseline), which matches expectation: offline tournaments are exactly
the events for which a long, stable history is most informative.

## What did NOT work

The other "cheap" ideas were either neutral or slightly negative:

* **L2 shrinkage on θ** (`reg_theta`).  Even small values
  (0.001–0.01) hurt logloss.  The adaptive learning rate
  `eta = eta0/√(1+games)` already provides effective shrinkage; adding
  per-observation L2 over-shrinks active players.
* **Cold-start factor < 1.0** (`cold_init_factor=0.5`).  Slight
  regression.  Initialising a rookie at the team mean is already a
  reasonable prior; halving it does not help on the metric.
* **Combining cold_init=0.5 with cal_decay** is slightly worse than
  cal_decay alone — same conclusion as above.

## Defaults adopted

```python
Config(
    use_calendar_decay = True,
    rho_calendar       = 1.0,
    decay_period_days  = 7.0,
    # rest unchanged: t6 mode handling
)
```

`reg_theta`, `reg_b`, `reg_log_a`, `cold_init_factor` all default to
the no-op value (0.0 / 1.0).  CLI flags expose them for experiments.

## Caveats

* `rho_calendar = 1.0` literally never decays.  A long-retired player
  keeps the last θ they had.  For "current strength" boards this can
  look weird; consider exporting `theta_shrunk` (already added to
  `--players_out`) or move to `rho_calendar ≈ 0.999` (≈ 0.95 over a
  year, 0.77 over 5 years inactive) at the cost of ~0.011 logloss.
* The win is measured against the prequential test (last 20 % of
  tournaments by date).  If the use-case puts more weight on rapid
  detection of recent skill changes, smaller `rho_calendar` should be
  re-evaluated against that specific objective.
