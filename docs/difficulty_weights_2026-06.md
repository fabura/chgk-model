# Difficulty-weighted loss experiments, 2026-06

Tests the intuition (raised by a user, motivated by the Семушин/Руссо
"Борский корабел" case) that **player strength is better defined by
which *hard* questions they take than by consistency on *easy* ones** —
so the model should blame a strong soloist less for missing an easy
question and reward them more for taking a hard one.

**Verdict: rejected as a loss-reweighting mechanism.** Every non-zero
weight degrades honest holdout logloss; applying it to team
observations is catastrophic. The intuition is real and has exact
mathematical grounding (§3), but the right way to encode it is a
*structural* model change (Model C γ_k, already validated; or a
per-player lapse rate), not a distortion of the MLE gradient.

Source artefacts:

- `results/exp_difficulty_weights_sweep.csv` — full sweep table
- `scripts/exp_difficulty_weights_sweep.py` — experiment script
- `rating/engine.py` — `Config.diff_w_miss_power`,
  `diff_w_take_boost`, `diff_w_solo_only`; `_difficulty_weight_args`
- `rating/model.py` — per-observation weight in `process_batch_nb`

---

## 1. Mechanism

Forward pass is **unchanged** (predictions identical). Only the
per-observation gradient (and its log-likelihood contribution) is
scaled by a weight derived from the model's own predicted probability
`p = p_final` (after lapse + recalibration):

```
miss (y=0): w = (1 − p) ** diff_w_miss_power      # ↓ blame for easy misses
take (y=1): w = 1 + diff_w_take_boost · (1 − p)   # ↑ credit for hard takes
```

`diff_w_solo_only=True` (default) applies it only to `team_size == 1`
observations, leaving the noisy-OR team channel untouched.

## 2. Results (honest cell-holdout, 10 %, seed 42)

All rows are `n_extra_epochs=1` (production default) unless noted.

| config | miss α | take β | scope | holdout ll | solo ll | AUC | θ Семушин | θ Руссо |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **baseline** | 0 | 0 | — | **0.5018** | **0.5040** | **0.8373** | 1.29 | 1.36 |
| miss_a03 | 0.3 | 0 | solo | 0.5019 | 0.5060 | 0.8372 | 1.41 | 1.30 |
| miss_a05 | 0.5 | 0 | solo | 0.5020 | 0.5098 | 0.8372 | 1.56 | 1.28 |
| miss_a10 | 1.0 | 0 | solo | 0.5025 | 0.5208 | 0.8368 | 1.93 | 1.10 |
| take_b05 | 0 | 0.5 | solo | 0.5023 | 0.5169 | 0.8370 | 1.83 | 1.16 |
| take_b10 | 0 | 1.0 | solo | 0.5031 | 0.5358 | 0.8364 | 1.97 | 1.07 |
| both_a05_b10 | 0.5 | 1.0 | solo | 0.5037 | 0.5486 | 0.8360 | 2.09 | 1.08 |
| both_a10_b10_all | 1.0 | 1.0 | **all** | **0.6472** | 0.5812 | 0.738 | 0.13 | −0.15 |

### Findings

1. **Monotone degradation.** Holdout logloss only ever gets worse
   (+0.0001 → +0.0019 in the solo-only rows). The *solo* slice — the
   exact population the change targets — degrades fastest
   (0.5040 → 0.5486, +0.045 at `both_a05_b10`). The mechanism makes the
   model worse precisely where it was meant to help.

2. **θ inflation, not skill recovery.** The soloist's θ balloons
   (Семушин 1.29 → 2.09) and the order with Руссо flips. But this is
   the calibration breaking, not evidence Семушин is stronger: on
   held-out easy async questions, predicting him at p≈0.92 (baseline)
   beats predicting p≈0.99 (down-weighted misses). The "+2 takes but
   −0.06 θ" feeling is partly a prior the held-out data does not
   support.

3. **Team channel collapse.** Applying the weights to all observations
   (`both_a10_b10_all`) destroys the model: ll 0.5018 → 0.6472, AUC
   0.837 → 0.738. The noisy-OR team gradient is the exact MLE gradient;
   distorting it is fatal (mirrors the 2026-06 temperature result).

## 2b. Convergence check

| config | passes | holdout ll | θ Семушин |
|---|---|---:|---:|
| baseline | 2 (1 extra) | 0.5018 | 1.29 |
| baseline_1pass | 1 | 0.5030 | 0.99 |
| both_a05_b10 | 2 | 0.5037 | 2.09 |
| both_a05_b10_1pass | 1 | 0.5049 | 1.13 |

The model converges cleanly in both regimes — `train_avg_ll` is stable
(≈ −0.49) and never diverges. The extra chronological pass is worth
−0.0012 logloss (consistent with `results/exp_multi_epoch_honest.csv`),
and θ has clearly *not* converged after a single pass (Семушин 0.99 vs
1.29). So the degradation under difficulty weights is a genuine
quality loss, not an optimisation artefact.

---

## 3. Why the intuition is correct — and why reweighting is the wrong lever

The intuition has exact mathematical grounding. For a single solo
observation the noisy-OR θ-gradient is

```
∂L/∂θ = ∂L/∂S · a · λ ,   λ = exp(−b + a·θ)
```

For a **miss** (y=0), `∂L/∂S = −1`, so the push on θ is `−a·λ`. Because
`λ = exp(−b + a·θ)`, an **easy** question (low `b`) has large `λ` →
**large** negative gradient, while a **hard** question (high `b`) has
small `λ` → **small** gradient. I.e. *easy-question misses already
dominate the θ update*, which is exactly the asymmetry the user noticed.

But this domination is **statistically correct** under the model: a
player who misses a question the model thought they'd take with p=0.92
*is* weaker at the margin than the model assumed, and nudging θ down is
the calibrated response. Reweighting the loss to suppress this
(`(1−p)**α`) makes the estimator inconsistent — it stops being an MLE,
θ drifts upward, and held-out predictions get worse. This is the same
failure class as temperature-scaled credit
(`docs/temperature_credit_experiments_2026-06.md`).

**The lesson is the same as every prior credit-attribution experiment:
encode structural priors in the *model*, not heuristics in the
*gradient*.**

## 4. The principled ways to encode "strength = hard questions"

### (a) Model C — per-player difficulty slope γ_k  *(already validated)*

```
z_k = θ_k + γ_k·b − b − δ
```

`γ_k > 0` ⇒ relatively better on hard questions. This *changes the
prediction* (it is not a loss hack), so it stays a clean MLE — and it
already shows **−0.0040 logloss** at `eta_gamma=0.01`
(`docs/2d_player_experiments_2026-06.md`). It directly expresses
"some players are hard-question specialists" with θ = baseline ability
and γ = difficulty profile.

**This directly confirms the intuition — in a calibrated way.** Running
`use_2d_players=True, eta_gamma=0.01, freeze_log_a=True` on the full DB
gives holdout logloss **0.4978** (vs 0.5018 baseline, −0.0040; AUC
0.8392 vs 0.8373) — i.e. the change *improves* prediction — and the γ
read-off for the Борский корабел soloists is exactly what the user's
intuition predicted (γ population mean +0.037, std 0.113):

| Player | θ | γ | reading |
|---|---:|---:|---|
| Брутер | −0.333 | **+0.573** | strong hard-question specialist (~+4.7σ) |
| Руссо | −0.259 | **+0.449** | strong hard-question specialist (~+3.6σ) |
| Семушин | +0.083 | **+0.317** | hard-question specialist (~+2.5σ) |
| Рекшинская | −0.878 | +0.282 | hard-question specialist |
| Чернуха | −1.110 | −0.064 | neutral / slight easy lean |

The 1D model was forced to compress "elite on hard packs, fumbles some
easy ones" into a single θ; Model C lets it split that into θ (baseline)
+ γ (difficulty slope), and the held-out data *rewards* the split. Note
θ drops sharply (it now means "ability at b=0"); to rank players you
compare effective ability `θ + γ·b` at the relevant difficulty. For the
two motivating soloists: `Семушин = 0.083 + 0.317·b`,
`Руссо = −0.259 + 0.449·b`. Семушин has the higher baseline and stays
ahead across essentially the entire observed difficulty range (they
only cross at `b ≈ 2.6`, i.e. extreme packs, where Руссо's steeper slope
edges in). So in the 2D view Семушин ≥ Руссо on normal-to-hard questions
— closer to the "Ваня is a different animal" intuition than the 1D model,
and achieved without any gradient hack.

**Recommended next step**: promote Model C to a production default
(re-train + DuckDB rebuild), and surface γ on the player page as a
"hard-question specialism" indicator.

### (b) Per-player lapse rate  *(untested, conceptually exact)*

The current lapse `p = (1 − π_{mode,solo})·p_noisy_or` already says
"even strong players whiff easy ones ~5 % of the time", but π is global
per (mode × solo). A **per-player** lapse component would let a player
who reliably takes hard questions but occasionally fumbles easy ones
carry a higher π *without* deflating θ — i.e. "strength on hard
questions, noise on easy ones" as two separate parameters. This is the
most literal encoding of the user's intuition. Risks: +N parameters and
θ↔π identifiability; would need an honest sweep and likely strong
shrinkage. Not yet implemented.

### (c) What *not* to do

Loss reweighting (this doc), temperature credit, asymmetric win/loss
temperature, uniform credit blend — all distort the MLE gradient and
all either degrade logloss or do nothing. Four independent experiments
now agree.

---

## 5. Status

The `diff_w_*` knobs are left in `Config` (default 0.0 = disabled) as a
documented dead-end, mirroring how the temperature knobs were handled.
No production default changes.  Model C was **not** promoted after rank
validation — see `docs/floor_player_experiments_2026-06.md`.
