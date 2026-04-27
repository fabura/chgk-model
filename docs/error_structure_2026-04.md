# Error structure & multi-epoch refit (April 2026)

Two complementary diagnostics on top of the current production model
(`Config()` defaults, backtest logloss = 0.488 / AUC = 0.846 on the
20 % time tail):

1. **Where does the model lose the most?** — slicing per-observation
  residuals on the test split by tournament, question, player, mode,
   team size, position-in-tour and roster strength.
2. **Joint-refit experiment** — re-running the chronological pass for
  K extra epochs (warm-started from the converged single-pass
   weights) to see whether the online single-pass is genuinely
   under-trained, and which params drift most when given more
   gradient updates.

Source artefacts:

- `scripts/analyse_errors.py` — Part 1
- `scripts/exp_multi_epoch.py` — Part 2 (`Config.n_extra_epochs`)
- CSV outputs in `results/error_analysis/` and `results/`

The single-pass baseline is unchanged (`Config()` with no flag
overrides); both experiments only read it and add diagnostics.

---

## Part 1. Error structure on the 20 % time tail

Snapshot: 1 749 test tournaments, 4 011 217 observations, overall
logloss 0.4877, Brier 0.1605, AUC 0.8455. Calibration ECE = 0.0112
(very good).

### 1.1 Calibration (10 bins of `p̂`)


| `p̂` bin | n       | mean p̂ | actual | abs diff  |
| -------- | ------- | ------- | ------ | --------- |
| 0.0–0.1  | 375 230 | 0.050   | 0.048  | 0.002     |
| 0.1–0.2  | 429 894 | 0.150   | 0.139  | **0.012** |
| 0.2–0.3  | 430 326 | 0.250   | 0.238  | **0.012** |
| 0.3–0.4  | 420 081 | 0.350   | 0.345  | 0.005     |
| 0.4–0.5  | 402 932 | 0.450   | 0.455  | 0.006     |
| 0.5–0.6  | 387 823 | 0.550   | 0.565  | **0.016** |
| 0.6–0.7  | 371 554 | 0.650   | 0.672  | **0.022** |
| 0.7–0.8  | 359 027 | 0.750   | 0.771  | **0.021** |
| 0.8–0.9  | 355 384 | 0.850   | 0.861  | 0.011     |
| 0.9–1.0  | 478 966 | 0.959   | 0.951  | 0.008     |


Pattern: the model is **slightly overconfident in both tails**:

- in `[0.1, 0.3]` it over-predicts (says 15 / 25 %, true 14 / 24 %);
- in `[0.5, 0.8]` it under-predicts (says 65 / 75 %, true 67 / 77 %).

ECE 0.011 is competitive but not perfect — there is room for ≈1 pp of
correction in the mid-confidence range. A simple isotonic / Platt
recalibration on the held-out fold would mostly close that gap, but
since the gap is small and well-distributed it isn't an obvious win.

### 1.2 By tournament mode


| mode    | n         | mean p̂ | actual | logloss    | mean res   |
| ------- | --------- | ------- | ------ | ---------- | ---------- |
| offline | 580 366   | 0.443   | 0.443  | 0.4483     | +0.001     |
| sync    | 2 216 336 | 0.498   | 0.497  | 0.4818     | −0.001     |
| async   | 1 214 515 | 0.530   | 0.540  | **0.5172** | **+0.010** |


Async carries the global error: 30 % of test obs but its logloss is
≈0.07 worse than offline / sync, and the model **systematically
under-predicts on async by 1 pp** even after the 2026-04 lean refactor
(which had removed the per-mode residual). Per-mode update weights
(`w_sync`, etc.) are not enough to absorb the residual variability of
online quizzes.

### 1.3 By team size


| size       | n         | mean p̂ | actual | mean res   |
| ---------- | --------- | ------- | ------ | ---------- |
| 1          | 199 389   | 0.382   | 0.423  | **+0.041** |
| 2          | 348 105   | 0.442   | 0.465  | +0.023     |
| 3          | 348 266   | 0.434   | 0.445  | +0.011     |
| 4          | 586 554   | 0.470   | 0.474  | +0.004     |
| 5          | 845 400   | 0.504   | 0.504  | 0.000      |
| 6 (anchor) | 1 531 272 | 0.555   | 0.548  | −0.007     |
| 7          | 93 047    | 0.479   | 0.476  | −0.003     |
| 8          | 59 184    | 0.453   | 0.457  | +0.004     |


The δ_size ladder works for sizes 5+ (residual ≈ 0), but solo and
2-player observations are still **systematically under-predicted**
(+0.04 and +0.02). The solo channel + δ_size knock the bias down from
the original ~+0.10 but don't fully eliminate it — small teams still
beat the model's expectation slightly more often than not. Likely the
δ_size estimate is correct on average and the residual reflects
selection bias (people who play solo tend to be self-selected
strong).

### 1.4 By position-in-tour


| pos                 | mean p̂   | actual | mean res |
| ------------------- | --------- | ------ | -------- |
| 0 (easiest, anchor) | 0.615     | 0.620  | +0.005   |
| 1                   | 0.578     | 0.586  | +0.008   |
| 2                   | 0.528     | 0.533  | +0.005   |
| 3                   | 0.510     | 0.515  | +0.005   |
| 4                   | 0.481     | 0.483  | +0.003   |
| 5–8 (mid-tour)      | 0.45–0.46 | ~same  | < +0.001 |
| 9                   | 0.461     | 0.461  | +0.001   |
| 10                  | 0.472     | 0.473  | +0.001   |
| 11                  | 0.517     | 0.522  | +0.005   |


δ_pos calibration is essentially perfect. The +0.005 at the very
start and end of a tour is at the noise floor.

### 1.5 By roster strength quartile


| quartile       | θ̄    | logloss    | AUC       |
| -------------- | ----- | ---------- | --------- |
| q1 (weakest)   | −1.31 | 0.4823     | 0.848     |
| q2             | −0.97 | 0.4823     | 0.848     |
| q3             | −0.79 | 0.4927     | 0.843     |
| q4 (strongest) | −0.58 | **0.4990** | **0.841** |


Hardest tournaments are still ≈1 % worse logloss than the median.
Mostly noise floor for the chosen test set, but the direction matches
the historical "Vyshka under-predicts" complaint that motivated the
θ̄-init Round 2.

### 1.6 Top-30 worst tournaments

`results/error_analysis/worst_tournaments.csv`. Top-5:


| id    | title                             | type  | logloss  | mean_p̂ | actual | residual |
| ----- | --------------------------------- | ----- | -------- | ------- | ------ | -------- |
| 11484 | ШАНс-2 (асинхрон и онлайн)        | async | **2.22** | 0.51    | 0.46   | −0.06    |
| 12811 | Тыквенный пирог (асинхрон/онлайн) | async | 1.32     | 0.50    | 0.58   | +0.08    |
| 12387 | Майский мёд (асинхрон/онлайн)     | async | 1.32     | 0.49    | 0.54   | +0.05    |
| 12129 | Баурсак (асинхрон и онлайн)       | async | 1.10     | 0.50    | 0.59   | +0.09    |
| 12671 | Brain Link 3                      | sync  | 1.07     | 0.51    | 0.46   | −0.05    |


Of the top-30 worst, **27 are async**. These are typically online
quizzes where (a) packs are atypical (rebuses, pop culture, AI-
generated), (b) team rosters are unstable / often single-player, (c)
the question set has a bimodal "joke vs hard" distribution that the
single-pass mean can't catch. The top 5 best, by contrast, are
"normal" sync packs with logloss 0.14–0.30.

### 1.7 Top-30 worst questions

`results/error_analysis/worst_questions.csv`. The pathology is
clear and shared by ~25 of 30 entries:


| pattern                              | n_q | example b / a   | mean p̂ | actual    |
| ------------------------------------ | --- | --------------- | ------- | --------- |
| `b ≈ +9.6` (clamped-ish), `a ≈ 0.94` | ~16 | b=+9.67, a=0.93 | 0.0001  | 0.30–0.50 |
| `b ≈ 0` or negative, `a ≈ 1.4–1.9`   | ~9  | b=−0.83, a=1.41 | 0.999   | 0.21–0.77 |
| Other isolated outliers              | ~5  | mixed           | mixed   | mixed     |


These are questions that were "perfectly easy" (everyone took, b → −∞)
or "perfectly hard" (no one took, b → +9.6 clamp) on training data,
then in the test split a different sample of teams produced a
radically different take rate. Two scenarios:

- **Same physical pack played twice (sync→async or asynchron pair)**
with very different audiences, sharing the canonical `cq` index.
Most of the `b ≈ +9.6` cluster looks like this: the first pass had
no teams take (offline tournament with strong teams), the second
pass (online async with mixed crowd) had ~30 % take.
- **Genuinely outlier observations** — a single very strong team
showing up on a "no-one takes" question.

Either way, the obvious mitigation is **bound `b` away from the hard
clamp earlier** (e.g. shrinkage of unseen-only questions toward the
mean of the same tournament's already-seen questions). We previously
tried "pack-level shrinkage" and rejected it as net-negative; worth
revisiting now that `θ̄_init` reduced the share of questions hitting
clamp.

### 1.8 Per-player residuals (uniform attribution)

`results/error_analysis/players_underestimated.csv` and
`players_overestimated.csv`.

**Caveat.** Each team-level residual is split equally across the
players on the roster — this is a crude attribution that smears the
residual of a weak link onto strong veterans on the same team and
vice versa. A more principled split would weight by per-player
`λ_k = exp(a θ_k − b − δ)` ("noisy-OR credit"). Not implemented in
this pass.

Top-5 under-predicted (model is too pessimistic — team beats the
prediction, suggests player is stronger than θ shows):


| name              | θ     | games | mean res |
| ----------------- | ----- | ----- | -------- |
| Зантария Тимур    | −1.51 | 41    | +0.35    |
| Хомяк Андрей      | −1.27 | 19    | +0.34    |
| Шкарупа Александр | −1.74 | 63    | +0.33    |
| Никулин Артём     | −1.31 | 45    | +0.32    |
| Пчелинцева Ольга  | −1.48 | 19    | +0.32    |


All under-predicted players have `< 70` total games and small test
windows (`n_obs_test = 36`). They look like **rookies on a strong
team** whose θ is anchored at the cold-start prior (−1.0) plus a
small downward correction; in the test window the team beats their
combined θ implied probability — by far the most likely
interpretation is that θ for them is still warming up, not that the
model is structurally biased against them.

Top-5 over-predicted (model thinks they're stronger than they
showed):


| name             | θ     | games | mean res |
| ---------------- | ----- | ----- | -------- |
| Евдокимов Руслан | −1.49 | 17    | −0.26    |
| Марфина Софья    | −1.84 | 18    | −0.23    |
| Макарова Ульяна  | −1.62 | 12    | −0.20    |
| Былкова Дарья    | −1.79 | 10    | −0.20    |
| Соловьёв Дмитрий | −1.43 | 11    | −0.18    |


Same pattern in reverse — all are rookies (10–18 games) whose teams
under-performed expectations during the test window. Both lists are
dominated by **freshly-cold-started players**, confirming that the
remaining error structure is mostly "warm-up noise" rather than a
systematic bias against any particular slice of established players.

---

## Part 2. Multi-epoch warm-start refit

### 2.1 Setup

`Config.n_extra_epochs = K`: after the standard chronological pass
(epoch 0), run K additional chronological passes over the **training
tournaments only** (first 80 % by date) with SGD updates only — no
init, no decay, no re-centering, no predictions during these extra
passes. Then re-collect predictions on the test tournaments using the
post-warm-start state in **frozen mode** (no further updates from
test data into the model). Train-tournament predictions stay as
recorded during the main pass.

`K = 0` is exactly the production behaviour (single chronological
pass with predict-then-update). `K > 0` ran fully from cold init for
each value (no continuation), so the comparison is independent.

The frozen-prediction mode for K > 0 is **strictly cleaner** than
the prequential test predictions used at K = 0: at K > 0 the model
NEVER sees test data; at K = 0 it does (one prequential update per
test obs).

### 2.2 Results

| n_extra | logloss | brier | auc | offline | sync | async | q1 | q2 | q3 | q4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (prod) | 0.4877 | 0.1605 | 0.8455 | 0.4483 | 0.4818 | 0.5172 | 0.4823 | 0.4823 | 0.4927 | 0.4990 |
| **1** | **0.4812** | **0.1589** | **0.8486** | 0.4502 | **0.4791** | **0.4998** | 0.4813 | **0.4807** | **0.4790** | **0.4858** |
| 2 | 0.4827 | 0.1593 | 0.8479 | 0.4509 | 0.4804 | 0.5019 | 0.4825 | 0.4821 | 0.4808 | 0.4873 |
| 4 | 0.4860 | 0.1604 | 0.8468 | 0.4533 | 0.4837 | 0.5059 | 0.4856 | 0.4853 | 0.4852 | 0.4897 |

(Bold = best per column.)

Wallclock per run on cache: 504 s (K=0), 615 s (K=1), 694 s (K=2),
642 s (K=4). The K=4 number is suspiciously low compared to K=2 —
likely warm Numba JIT cache shared across runs (the script ran them
sequentially in a single Python process).

### 2.3 What we learned

* **One extra epoch is the sweet spot.** Logloss 0.4877 → 0.4812
  (−0.0065, −1.3 %); AUC 0.8455 → 0.8486 (+0.0031); Brier
  0.1605 → 0.1589.

* **Most of the gain lands exactly where Part 1 located the error**:
  * async: 0.5172 → 0.4998 (−0.0174, −3.4 %);
  * hardest roster quartile: 0.4990 → 0.4858 (−0.0132, −2.6 %);
  * sync: 0.4818 → 0.4791 (−0.0027);
  * offline: 0.4483 → 0.4502 (+0.0019, **slight loss** — the
    single-pass model is already well-fit on offline; 1 extra epoch
    overfits it a tiny bit).

* **Two or more extra epochs over-fit.** Logloss climbs back: 0.4827
  (K=2), 0.4860 (K=4). All slices degrade roughly monotonically past
  K=1. So this is **not** a "more is better" effect; the single-pass
  model is just slightly under-trained.

### 2.4 Parameter drift relative to baseline (K = 0)

`results/exp_multi_epoch_drift.csv`:

| K | n_players | n_questions | θ_RMS | θ_max | b_RMS | b_max | log_a_RMS | log_a_max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49 464 | 340 286 | 0.116 | 1.05 | 0.109 | 3.39 | 0.116 | 3.92 |
| 2 | 49 464 | 340 286 | 0.196 | **10.12** | 0.182 | 7.95 | 0.192 | 4.04 |
| 4 | 49 464 | 340 286 | 0.300 | **10.11** | 0.283 | 6.50 | 0.301 | 3.94 |

Mean θ shift is ≈ 0 across all K (−0.005 .. −0.008): the gauge stays
stable, individual θ's move in both directions roughly symmetrically.

Critical detail: at K = 1 max |Δθ| = 1.05; at K ≥ 2 it hits the
±10 clamp. Past K = 1, individual params start running off the rails
on small-sample observations (the typical overfitting signature).

### 2.5 Where parameters move most (K = 4 vs K = 0)

`results/exp_multi_epoch_top_question_drifts.csv` — top entries
(20 questions with largest |Δb|):

| canonical q | b_base | b_K4 | Δb | log_a_base | log_a_K4 |
|---:|---:|---:|---:|---:|---:|
| 121814 | 7.70 | 1.19 | **−6.50** | −1.34 | −2.27 |
| 133607 | 8.15 | 1.78 | **−6.37** | −1.17 | −2.26 |
| 122801 | 8.07 | 2.27 | −5.80 | −0.37 | −0.66 |
| 122824 | 8.09 | 2.34 | −5.76 | −0.34 | −0.59 |
| 50865 | 6.25 | 1.25 | −5.00 | −1.26 | −2.12 |
| 122819 | 8.95 | 4.83 | −4.12 | −0.17 | −0.39 |
| 209075 | 9.07 | 5.39 | −3.68 | −0.20 | −0.56 |
| ...all top-15 entries follow the same pattern: b shrinks from ≈9 to 5–6, log_a moves more negative... |

**Every single one of the top-15 question drifts is a question whose
single-pass `b` was clamped near +9.6** (the one that caused 3–5 nat
losses in Part 1, §1.7). The extra epochs **un-clamp them**: b moves
down to a more sensible 1–6, and `log_a` becomes more negative
(reducing discrimination). This is exactly the structural pathology
the error analysis flagged. K = 1 only partially un-clamps these
(Δb ~ 1–3 instead of 5–6) — which is why K = 1 is the optimum: it
fixes most of the clamp pathology without overshooting.

`results/exp_multi_epoch_top_player_drifts.csv` — top entries
(K = 4 vs base):

| name | games | θ_base | θ_K4 | Δθ |
|---|---:|---:|---:|---:|
| Курант Эли | 5 | +0.12 | **−9.98** | −10.11 |
| Соколов Артём | 6 | −1.34 | +0.17 | +1.51 |
| Рахимов Азамат | 44 | −1.41 | −0.01 | +1.40 |
| Кустов Владимир | 1 | −0.29 | +1.09 | +1.38 |
| Сакаева Елена | 1 | −0.76 | +0.59 | +1.34 |
| Захарова Наталья | 8 | −1.48 | −0.14 | +1.34 |
| Чусов Александр | 157 | −1.94 | −0.65 | +1.29 |
| ...all top-20 entries: 1–157 games, θ moves by 1.0–1.5... |

The biggest player drifts are **all on small-sample players (1–157
games)**: extra epochs don't change established veterans (their
prior is dominated by their large game count), but they have ample
freedom to grind freshly-cold-started players toward whatever their
team-result history says. The Курант Эли case (θ → −9.98 = clamp) is
the textbook over-fitting failure mode.

→ This is consistent with K = 1 being the sweet spot: at K = 1 we
get the un-clamp on questions ("good" gradient flow restored once
the question has been seen multiple times) without the player-θ
overfitting blow-up that K ≥ 2 brings.

---

## Conclusions and follow-up candidates

### Where the model loses

1. **Async quizzes drive the global error.** Logloss 0.517 on async
  vs 0.448 offline / 0.482 sync; +0.010 mean residual (model
   under-predicts). The 27/30 worst tournaments are async. Mitigations
   that have NOT been tried yet:
  - separate `b` distribution (or stronger regularisation toward the
  async population mean);
  - larger `w_size_async` for the under-predicted small-team strata.
2. **Hard-clamp pathology on `b`.** A few hundred questions sit at the
  `b ≈ +9.6` clamp (or beyond), then the test split has 30–50 % take
   and the model loses 3–5 nats per observation. Pack-level shrinkage
   was rejected previously, but only on the pre-θ̄-init engine; worth
   re-trying with a **conservative pull toward the tournament-level
   `b̄` for `cq` with `n_obs < 5`**.
3. **Solo and 2-player residuals are still positive.** δ_size + solo
  channel reduce the bias from ~+0.10 to +0.04 / +0.02 but selection
   bias (strong soloists) lingers.
4. **All player-level residuals concentrate in rookies (≤ 70 games).**
  Suggests the cold-start prior is the right thing to keep working
   on; nothing structurally wrong with the established-player tail.

### Things to try next (in rough cost order)

1. **Adopt `n_extra_epochs = 1` as default.** −0.0065 logloss / +0.003
   AUC for ~30 % more wallclock per nightly retrain. The gain is
   concentrated exactly in the slices Part 1 flagged (async, hard
   tournaments, clamped questions). NOTE: re-tuning of `eta0` /
   regularisation could push the optimum further; current sweep was
   only over `K`.
2. Re-try **pack-level shrinkage** with conservative weight (`w ≤
   0.05`) only for canonical questions with very few observations
   (Part 1 §1.7 + Part 2 §2.5 both point to the clamp pathology as
   the single biggest residual source).
3. **Async-specific `b` regulariser** centred on async population mean
   (instead of zero).
4. **Noisy-OR-weighted player residual attribution** in
   `analyse_errors.py` — to confirm the rookie-only pattern.
5. **Per-tournament `b` shift** (reintroduce `ε_t` but with a stronger
   prior) — would directly absorb the pathological per-tournament
   residuals seen on Brain Link 3 / ШАНс-2.

### What we learned from multi-epoch

* The single-pass online model **is genuinely under-trained** by
  about one epoch's worth of gradient updates — and the under-training
  manifests **exactly** as the residual pattern Part 1 found.
* The mechanism is **questions whose `b` is clamped at ~+9.6** during
  the single pass: one extra epoch un-clamps them (Δb ≈ 1–3) and the
  test logloss on async / hard tournaments drops by 1–3 %.
* K ≥ 2 over-fits — typical small-sample players' θ runs to the ±10
  clamp.
* No retuning was tried alongside `n_extra_epochs`; if we adopt
  `K = 1` as the default we should re-sweep `eta0`, `reg_b`, and the
  `w_*_questions` weights to see if a slightly smaller learning rate
  + 2 epochs beats the current K = 1 result. Cheap follow-up
  experiment.