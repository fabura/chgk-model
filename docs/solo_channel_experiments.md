# Solo channel

## Motivation

The noisy-OR likelihood `p_take = 1 − exp(−Σ_k λ_k)` plus the per-
team-size shift `δ_size` can fit *the average* solo result reasonably
well — but it leaves a structural identifiability shortcut for
prolific soloists.

Concretely: a player who plays almost exclusively solo online quizzes
(M-Лига, "Гостиный двор", various Discord packs) competes against a
self-selected, hyper-engaged subset of the population. Take rates of
30–80 % on their personal packs are routine. With the standard online
update, every one of those 36-question packs tugs their θ toward the
top of the distribution because the gradient is computed exactly the
same way as a team result, and there is no other player to share
credit with.

The triggering case was player **Андрей Белов (id=2954)**: 531 games,
of which ~12 % solo, but those solo wins drove his θ to **+0.58
(rank 3 / 49 433)** — well above multi-decade team veterans like Иван
Семушин who routinely beat him in the same tournaments when actually
on the same roster.

The fix could in principle be either of:

1. Drop solo samples entirely (`scripts/exp_no_solo.py`). Cheap, but
   throws away genuine signal — soloists *are* informative, just
   over-weighted.
2. **Down-weight solo updates via a separate channel** (the chosen
   solution). Solo results still contribute to the loss and to the
   `δ_size[1]` learning, but their gradient on θ / b / log_a is
   scaled down so a single solo result tugs ~3× less than a team
   result.

## Implementation

When `Config.use_solo_channel = True`, the per-tournament update loop
(`rating/engine.py`) splits each tournament's observations into two
buckets at gradient time:

* `obs_order_team` — every observation with `team_size ≥ 2`. Updated
  with the legacy per-tournament-type weights (`w_offline`, `w_sync`,
  `w_online`).
* `obs_order_solo` — every observation with `team_size == 1`,
  regardless of the tournament's nominal type (offline / sync /
  async). Updated with the dedicated `w_solo*` weights.

Both buckets go through the same Numba `process_batch_nb` and into
the same θ / b / log_a / `δ_size` / `δ_pos` arrays — only the
multiplicative weights change. The `forward()` predictions, `δ_size[1]`
correction, teammate shrinkage, calendar decay, and game-counter
increment are all unchanged.

Default weights:

| weight              | value | rationale                                        |
|---------------------|-------|--------------------------------------------------|
| `w_solo`            | 0.3   | Best logloss in the 5-cell sweep (see below).    |
| `w_solo_questions`  | 0.0   | Don't let the narrow soloist pool bias `b`.      |
| `w_solo_log_a`      | 0.0   | Same — don't bias question discrimination.       |
| `w_size_solo`       | 1.0   | Keep learning `δ_size[1]`; otherwise the         |
|                     |       | noisy-OR forward pass for solo predictions       |
|                     |       | breaks (anchor at 6 leaves [1] at 0).            |
| `w_pos_solo`        | 0.0   | Solo packs are mostly 36-question online quizzes |
|                     |       | with atypical positional structure.              |

CLI: not exposed yet; flip in code via `Config(use_solo_channel=...)`.

## Experiments

### Backtest sweep over `w_solo`

`scripts/exp_solo_channel.py data.npz`, 20 % time-split hold-out, full
DB. Log: `results/exp_solo_channel.log`.

```
config              logloss     AUC   ll_off  ll_syn  ll_asy   Belov θ   rank
baseline (legacy)    0.5276  0.8151   0.5102  0.5226  0.5450    +0.58      3
solo w=0.0           0.5272  0.8156   0.5088  0.5221  0.5452    -0.13    182
solo w=0.1           0.5270  0.8158   0.5087  0.5220  0.5446    +0.07     31
solo w=0.3           0.5268  0.8159   0.5088  0.5220  0.5442    +0.34      8   ← chosen
solo w=0.5           0.5268  0.8160   0.5089  0.5220  0.5440    +0.48      3
```

Every solo-channel variant beats the legacy baseline on both metrics.
Improvement plateaus at `w=0.3 / w=0.5`:

* `w=0.3` and `w=0.5` are tied on logloss (0.5268, −0.0008 vs baseline);
  AUC differs by 0.0001 in favour of `w=0.5`.
* But at `w=0.5` the channel becomes too weak: Belov bounces back to
  rank 3 and his solo wins almost recover their full pull. The
  artefact the channel was introduced to fix is barely corrected.
* `w=0.3` is the *lowest-logloss point that still meaningfully
  deflates soloists* (Belov rank 3 → 8, θ +0.58 → +0.34) and was
  picked as the default.

Per-type breakdown shows the gain is concentrated in async (the
mode where most solo packs live): −0.0008 on async logloss, ~flat
on offline / sync.

### Top-N sanity check

`scripts/exp_solo_topn.py data.npz` — full-history training with the
same defaults, no backtest. Log: `results/exp_solo_topn.log`.

* **Pure team players** (Семушин 30 % solo, Хайбуллин/Шешуков/
  Коробейников/Брутер 0 % solo): |Δθ| ≤ 0.075 — almost no movement.
* **Mixed players** (Белов 11 % solo): Δθ −0.244, rank 3 → 8.
* **100 %-solo entries in baseline top-30**: drop hard.
  Examples: Имя 248699 (20 g) +0.52 → +0.35, Комаров Фёдор (12 g)
  +0.48 → +0.20, Витюгов Евгений (1 g) +0.27 → −0.30.
* **Top-20 by solo % within baseline top-1000**: every 1-game phantom
  gets pushed past rank 1000; multi-game soloists drop ~150–500 ranks
  proportional to solo %.

The new top-30 by θ now reads as a list of established team players
rather than a mix of veterans and online-quiz outliers.

### Counterfactual: dropping solo samples entirely

`scripts/exp_no_solo.py` — for context. Globally filters every
`team_size==1` sample (~4 % of the corpus) and retrains. Belov's θ
falls to **−0.12 (rank 187)** — too aggressive; the down-weighted
channel at `w_solo=0.3` lands at +0.34 / rank 8, which is much closer
to where his actual team performance places him.

## What this does NOT change

* `forward()` is unchanged. Solo predictions still combine the
  player's θ, the question's `b/a`, and `δ_size[1]` exactly as before.
* `δ_size[1]` keeps being learned (`w_size_solo=1.0`).
* Solo samples still:
  * contribute to the training log-likelihood;
  * count toward `players.games[k]` (so subsequent η is still scaled
    down); and
  * trigger calendar decay / teammate shrinkage as before.
* Tournament type assignment is unchanged — a solo observation in an
  offline tournament still routes through the solo channel
  (`team_size==1` is the only criterion). This is intentional;
  conceptually the channel is "downweight 1-player observations",
  not "downweight async tournaments".

## Open questions

* Sweep `w_solo_log_a` and `w_solo_questions` separately (currently
  pinned at 0). Plausible that a small non-zero value on `b` would
  help, since solo packs do contain real signal about question
  difficulty.
* Consider exposing `--use-solo-channel` / `--w-solo` via the CLI in
  `rating/__main__.py` if we ever want to A/B in production.
* Re-run `scripts/exp_cold_start_grid.py` with the solo channel
  enabled — the "rookie boost" was tuned without it, and the channel
  changes how fast 1-game soloists move.
