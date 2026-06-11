# Roster-sticking investigation, 2026-05

## Motivation

Three veteran players were flagged by a user as sitting "much lower in
the rating than they should": Ирина Чернуха (`id=34909`), Анастасия
Рекшинская (`id=26818`), Вера Монина (`id=158668`). The common thread:
all three are long-tenured players who spend most of their games as the
weakest-θ member of a strong, stable roster. This is the textbook
failure mode of noisy-OR on team-level data — credit for a taken
question is attributed ∝ λ_k, so the strongest teammate absorbs almost
all of it and a perennial "floor" player can stay frozen low for years.

This cycle built diagnostics to measure how widespread the effect is,
then tested three fixes. **One was adopted (`eta_teammate` bump); two
were rejected with evidence** (residual-aware adaptive η; mixed credit
attribution). The headline lesson: the three players' low θ is mostly
*correct given recent evidence*, and the residual "stickiness" signal is
~80 % team-context, not individual θ-staleness.

## Diagnostics (read-only, over the baked DuckDB)

| Script | What it measures |
|---|---|
| `scripts/diagnostic_roster_sticking.py` | per-veteran fraction of games as the lowest-θ in the roster; 1-year `implied − θ` gap; career & recent `actual − expected`; team-departure θ recovery |
| `scripts/diagnostic_residual_persistence.py` | does a player's past residual predict their next residual? (precondition for an adaptive-η fix) |
| `scripts/diagnostic_residual_individual.py` | is the residual an **individual** trait (cross-team consistency, ICC) or **team-context**? |

Results CSVs: `results/diagnostic_roster_sticking.csv`,
`results/diagnostic_team_departures.csv`,
`results/diagnostic_residual_persistence.csv`.

### Population picture (3 784 active veterans, ≥200 games)

* **Roster sticking is rare but real**: only 85 / 3 784 (2.2 %) are the
  lowest-θ member in ≥70 % of their games; p90 = 43 %, p95 = 58 %.
  Чернуха is at 55 % (#222).
* **`implied − θ` gap ≥ 0.20**: 630 / 3 784 (16.6 %). Чернуха +0.27,
  Монина +0.20; Рекшинская −0.11 (i.e. the model does **not** under-rate
  her on recent evidence).

| Player | θ | % lowest in roster | implied gap (1y) | career `actual − exp` |
|---|---:|---:|---:|---:|
| Чернуха | −0.53 | 55 % | +0.27 | −70 |
| Рекшинская | −0.16 | 39 % | −0.11 | +26 |
| Монина | −0.18 | 24 % | +0.20 | **+437** |

---

## Hypothesis 1 — bump `eta_teammate` (ACCEPTED)

`eta_teammate` is a per-tournament shrinkage that, after each gradient
step, pulls every roster member's θ toward the team mean
(`θ_k −= eta_teammate · (θ_k − mean_team θ)`). It is the existing,
principled-heuristic lever against exactly this identifiability problem.
It was `0.005`; this cycle re-swept it under honest cell-holdout
(`holdout 0.10`, `seed 42`) with `scripts/exp_eta_teammate_sweep_honest.py`.

| `eta_teammate` | logloss | AUC | Чернуха θ | Рекшинская | Монина |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.50310 | 0.83590 | **−1.13** | +0.18 | −0.05 |
| 0.005 (old) | 0.50226 | 0.83676 | −0.53 | −0.16 | −0.18 |
| 0.010 | 0.50202 | 0.83704 | −0.44 | −0.19 | −0.21 |
| 0.015 | 0.50184 | 0.83720 | −0.41 | −0.17 | −0.20 |
| **0.020 (new)** | **0.50178** | 0.83726 | −0.39 | −0.15 | −0.19 |
| 0.030 | 0.50177 | 0.83728 | −0.37 | −0.13 | −0.17 |

Findings:

* Turning the shrinkage **off** is clearly worst on both logloss
  *and* the floor player (Чернуха collapses to −1.13).
* logloss plateaus at 0.02–0.03 (−0.0005 vs the old 0.005). Picked
  **0.02** (within noise of 0.03, more conservative).
* Net for Чернуха: +0.14 θ vs the old default, with no harm to overall
  quality or to the other two players.

Adopted as the new `Config.eta_teammate` default (and the
`--eta_teammate` CLI default). Reproduced exactly by
`scripts/sanity_eta_teammate.py` (logloss 0.5018, θ within ±0.001 of the
sweep value). The change is **not yet on the website** — it needs a
retrain + DuckDB rebuild.

### Side check — `w_online` (no change)

After the `eta_teammate` bump, swept the async/online θ-update weight
(`scripts/exp_w_online_sweep_post_teammate.py`,
`results/exp_w_online_sweep_post_teammate.csv`). Lowering `w_online`
raises Рекшинская (to +0.02 at 0.30) but **worsens logloss**, especially
the async slice (+0.004). The current `w_online = 1.0` is the logloss
optimum; left unchanged. Рекшинская's async drag is not fixable through
the global online step size.

---

## Hypothesis 2 — residual-aware / Adam-style adaptive η (REJECTED, offline)

The current schedule `η_k = η0 / √(games_offset + games_k)` assumes a
**stationary** target. The idea (user-proposed): track each player's
persistent over/under-performance and boost η when it is one-signed, so
veterans whose form/context changed are not frozen.

**Attribution caveat (flagged up front):** the residual
`actual − expected` is a *team* quantity. Driving an *individual*'s η
from it mis-attributes a teammate's bias. The clean version gates on the
persistence of the player's **own gradient sign** (≈ per-player Adam),
because the gradient already attributes credit ∝ λ_k.

### Offline test (precondition: does past residual predict future?)

`diagnostic_residual_persistence.py`, per-question-normalised residual,
veterans only:

| Slice | corr | OLS slope |
|---|---:|---:|
| all veteran games | +0.249 | **+0.543** |
| rookie (game ≤50) | +0.344 | +0.572 |
| mid (50–200) | +0.244 | +0.541 |
| **veteran (>200)** | +0.199 | **+0.472** |
| carry games (θ matters most) | +0.256 | +0.551 |
| passenger games | +0.240 | +0.527 |
| prior-30 → recent-30 block split | +0.648 (corr) | — |

The persistence is large (slope 0.54) **but**:

1. It **falls** with experience (0.57 → 0.47), the opposite of the
   "1/√games too slow for veterans" hypothesis — the engine has already
   absorbed *more* of a veteran's stable signal.
2. **carry ≈ passenger** (0.551 vs 0.527). If the persistence were
   individual θ-staleness, carry games (where the player's own θ
   dominates `expected`) would persist more. Equality ⇒ the signal is
   team-context, not individual.

### Refinement: how much is individual? (`diagnostic_residual_individual.py`)

* **Cross-team consistency**: a player's two largest-team mean residuals
  correlate only **+0.24** across players; one-way ICC = **+0.20**. So
  ~20 % of the residual is a stable per-player trait, ~80 % is
  team-specific or noise.
* **Context regress-out is uninformative by construction**: `mate_avg_θ`
  + roster size explain **0.0 %** of the residual variance — because
  `expected` already incorporates teammate θ and δ_size. The residual is
  orthogonal to these inputs; the persistence lives in structure the
  model does not capture.
* **Per-team residuals of the three players** confirm it: Чернуха is
  ≈0 (−0.016 … +0.012) on *every* team — there is no team where she
  over-performs, so her low θ is not contradicted by any roster context.

**Verdict: NO-GO.** A per-player adaptive/boosted η would chase
team-level autocorrelation and mis-attribute it. Only ~20 % of the
signal is individual, and that part is already being absorbed.

---

## Hypothesis 3 — mixed credit attribution (REJECTED, honest ablation)

The exact noisy-OR θ-gradient is `∂L/∂θ_k = ∂L/∂S · a · λ_k`, i.e.
credit ∝ λ_k, concentrating on the strongest member. C blends it toward
uniform, conserving total credit:

```
credit_k = (1 − mix) · λ_k + mix · (S / n)
```

This is a deliberate **departure from the likelihood gradient** (a
heuristic, like `eta_teammate`). Implemented behind
`Config.credit_uniform_mix` (0.0 = exact gradient) and ablated under
honest cell-holdout. Expectation set in advance from Hypothesis 2:
marginal-to-negative, since the persistence is mostly team-context.

| `credit_uniform_mix` | logloss | AUC | Чернуха | Рекшинская | Монина |
|---:|---:|---:|---:|---:|---:|
| **0.00** | 0.50178 | 0.83726 | −0.392 | −0.154 | −0.192 |
| 0.25 | 0.50178 | 0.83726 | −0.393 | −0.129 | −0.185 |
| 0.50 | 0.50178 | 0.83724 | −0.392 | −0.100 | −0.167 |

**ΔLogloss = +0.00000; AUC marginally worse.** Two conclusions:

1. **Zero predictive benefit** — it is a gauge shift in who gets credit,
   not a quality gain.
2. **It does not even move the motivating case** (Чернуха −0.392 →
   −0.392). As Hypothesis 2 predicted: with no persistent residual to
   redistribute, extra credit on takes is exactly cancelled by extra
   blame on misses. Only Рекшинская/Монина drift up, and only because
   they carry a little residual — again with no logloss gain.

`mix = 0.0` reproduced the baseline to 5 significant figures, confirming
the change was correct but useless.

**Verdict: REJECT.** Reverted from `model.py` / `engine.py` per the
project's failed-ablation convention (cf. `b_pack_shrinkage`,
`pack_prior_w` in 2026-04). The result CSV
`results/exp_credit_mix_holdout.csv` is kept as the record.

**2026-06 follow-up** (`docs/floor_player_experiments_2026-06.md`): difficulty-
weighted loss, Model C promotion, and per-tournament overperformance θ
floor were also tried and rejected / not promoted.  Production still
`eta_teammate=0.02` only.

---

## Per-player conclusions

* **Чернуха** — under-rated relative to her *Инк-era* teammates (96 %
  the lowest-θ on Инк), but her per-team residual is ≈0 everywhere, so
  recent data does not say her θ is wrong. The historical artefact is
  partially softened by `eta_teammate = 0.02` (−0.53 → −0.39); no
  mechanism that keys on residuals will lift her further, because there
  is no residual signal to act on.
* **Рекшинская** — not under-rated on recent evidence (implied gap
  −0.11; carry-game residual −0.03). Her decline from +0.24 (2019) to
  −0.16 is the model correctly tracking a real performance drop plus a
  weak-async-roster context ("Самозанятый сомелье", −1.56/game). Not a
  model bug.
* **Монина** — the only genuine (but small) under-rating: career
  `actual − expected` = +437, positive carry-game residual (+0.022). The
  engine is already raising her slowly (−0.77 in 2019 → −0.18 now).

## Open / residual item

The only remaining cheap, zero-backtest-risk lever for Монина-type
players (model lags observed performance) is a **display-time blend** of
`theta_display` toward `team_theta_implied`, analogous to the existing
inactivity decay in `website/build/build_db.py`. It is cosmetic (shifts
displayed numbers only) and should be done only if moving the public
board is desired. Everything that affects the backtest has reached
diminishing returns for this class of player; `eta_teammate = 0.02`
captured the principled gain.
