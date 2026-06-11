# Offline Tilt-Resilience Diagnostics (2026-06)

Purpose: measure whether teams keep underperforming after a short run of
"avoidable" misses in the same tour, and use that historical signal to rank
registered teams for tournament `12826` ("II ЧР по интеллектуальным играм").

This is a diagnostic only. It does not change the rating model.

## Definition

The script is `scripts/diagnostic_tilt_resilience.py`.

Default run:

```bash
.venv/bin/python scripts/diagnostic_tilt_resilience.py \
  --event-id 12826 --null-sims 50 --quiet
```

The diagnostic:

1. runs the current sequential model with per-observation predictions;
2. keeps `mode=offline` observations, excludes obvious online-like titles
   (`онлайн`, `online`, `м-лига`, `m-лига`, `m-liga`), and drops solo rosters;
3. groups observations into within-tournament sequences by `(tournament,
   sorted roster)`;
4. marks an avoidable miss as `taken == 0` with `pred_p >= 0.60`;
5. after exactly two consecutive avoidable misses, measures residual
   `actual - expected` on the next three questions.

Main score:

```text
resilience_score = mean(actual - expected) after the trigger
```

Higher is better. `tilt_pp = -100 * resilience_score`, so positive `tilt_pp`
means a penalty in percentage points; negative `tilt_pp` means the team/player
historically outperformed expectation after the trigger.

## First Run

Run completed on `data.npz` with current `Config` defaults.

Filtered sample:

- excluded by online-like title terms: `1080` tournaments;
- observations kept for offline team sequences: `3,833,100`;
- team/tournament sequences: `54,880`;
- trigger events: `11,915`;
- after-trigger observations: `35,319`.

Global aftershock:

| metric | value |
| --- | ---: |
| observed residual/question after trigger | `-0.0402` |
| null residual/question (50 Bernoulli sims from same `pred_p`) | `-0.00009` |
| null sd | `0.00205` |
| z vs null | `-19.6` |

Recovery by current avoidable-miss streak:

| streak before next question | actual take rate | expected take rate | residual |
| --- | ---: | ---: | ---: |
| `0` | `0.421` | `0.420` | `+0.001` |
| `1` | `0.502` | `0.513` | `-0.012` |
| `2` | `0.502` | `0.543` | `-0.040` |
| `3+` | `0.395` | `0.568` | `-0.172` |

Interpretation: conditional on model difficulty and team strength, there is a
large aggregate aftershock after repeated avoidable misses. The effect is not
explained by random streakiness under independent Bernoulli outcomes from the
same probabilities.

## Outputs

All outputs are in `results/tilt_resilience/`:

- `summary.csv` / `summary.json` — global aftershock, recovery buckets, null
  baseline metadata.
- `teams_history.csv` — historical roster-level scores. A historical "team"
  is an exact sorted roster because `data.npz` does not carry `team_id`.
- `players.csv` — player-level attribution. Each team event is assigned to all
  roster members equally; this is diagnostic, not causal.
- `events.csv` — worst concrete aftershock episodes for manual inspection.
- `event_12826_teams.csv` — registered teams for `12826`, ranked by combined
  historical resilience evidence.

## Tournament `12826`

The script pulled `90` registered teams from the rating API.

Top rows in `event_12826_teams.csv` after filtering:

| rank | team | score | evidence note |
| ---: | --- | ---: | --- |
| 1 | Клуб анонимных зануд | `+0.131` | direct exact-roster signal, low sample (`3` weighted events) |
| 2 | Пятница | `+0.084` | mostly player-level, weak evidence (`1` scored player) |
| 3 | Хайвмайнд | `+0.082` | player-level, `4/6` scored players |
| 4 | Спонсора.net | `+0.069` | player-level, `3/6` scored players |
| 5 | Кудрявчик и Вахривка | `+0.065` | player-level, `6/6` scored players |

Use the ranking with the evidence columns, not by score alone. In particular,
many registered teams include players absent from the local historical model
or with too few trigger events. Columns to check:

- `scored_players`;
- `unknown_players`;
- `player_event_sum`;
- `direct_overlap_events`;
- `best_overlap`.

## Caveats

- Continuous streaks require continuous within-tour sequences, so the default
  uses `--eval-scope all`. This is better for tilt measurement but not a pure
  leakage-free backtest. Use `--eval-scope holdout` only as a sparse sanity
  check.
- Historical team identity is roster-based, not `team_id` based. Exact team
  continuity across seasons is approximated through overlapping registered
  rosters for `12826`.
- Player attribution is equal-across-roster. It says "this player was on teams
  that recovered/collapsed", not "this player caused recovery/collapse".
- The metric is observational. Strong opponents, room conditions, tour rhythm,
  and captain decisions can still confound it.
