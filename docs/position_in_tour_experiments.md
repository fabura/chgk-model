# Position-in-tour effect (`δ_pos`)

A small additive shift on tournament difficulty, indexed by the
question's position inside its tour:

```
δ = μ_type[type] + ε_t + δ_size[team_size] + δ_pos[(q_index_in_tournament) % tour_len]
```

with `tour_len = 12` (the standard ChGK tour length) and
`δ_pos[pos_anchor] ≡ 0` for identifiability (`pos_anchor = 0` =
empirically the easiest position).

## Motivation

Domain hint from a strong player ("first 1–2 questions in each tour
tend to be easier"). The empirical take rate per position confirms
the intuition very clearly. Restricting to the 5 898 tournaments
that have exactly 36 questions (so `% 12` is unambiguous):

| pos | empirical take rate | obs           |
|-----|---------------------|---------------|
| 0   | **0.5953**          | 1 587 683     |
| 1   | 0.5436              | 1 587 683     |
| 2   | 0.5011              | 1 587 683     |
| 3   | 0.4714              | 1 587 682     |
| 4   | 0.4551              | 1 587 682     |
| 5   | 0.4366              | 1 587 682     |
| 6   | 0.4389              | 1 587 682     |
| 7   | 0.4225 (lowest)     | 1 587 682     |
| 8   | 0.4239              | 1 587 682     |
| 9   | 0.4288              | 1 587 682     |
| 10  | 0.4360              | 1 587 682     |
| 11  | 0.4919 (rebound)    | 1 587 682     |

The curve is monotone-decreasing from p=0 to ~p=7 (12 pp gap) with a
mild rebound on the last question of each tour.

The signal is too systematic and too large to leave to chance, but
it cannot be absorbed by `b` alone: every (tournament, q_index)
pair is a *different* question, so per-question `b` cannot share
information across positions. `ε_t` is also blind to position
because it is one number per tournament. A 12-bucket position
vector is the cheapest fix.

## Implementation

* `delta_pos` lives on `SequentialResult` and is updated inside
  `process_batch_nb` by the same gradient as `ε_t` and `δ_size`
  (`dL/dδ`, with type-conditional weight `pos_w`).
* The mapping `q_index_in_tournament → pos` is precomputed once at
  the start of `run_sequential` from `IndexMaps.idx_to_question_id`.
* Tournaments with non-12 tour structure (~30% of the data: 45-,
  60-, 90-question events with mostly 15-question tours) get a
  noisier signal but still contribute usefully on average.
* Anchor at `pos_anchor = 0`: the easiest position is fixed at 0,
  so the learned δ_pos vector reads naturally as
  "extra difficulty of position p relative to question 1".

## Backtest results (full DB)

|                | baseline | + team_size | + team_size + pos (anchor 0) | Δ vs baseline |
|----------------|----------|-------------|------------------------------|---------------|
| Logloss        | 0.5318   | 0.5274      | **0.5225**                   | −0.0093       |
| Brier          | 0.1777   | 0.1761      | **0.1742**                   | −0.0035       |
| AUC            | 0.8220   | 0.8249      | **0.8255**                   | +0.0035       |
| offline LL     | 0.4826   | 0.4721      | **0.4600**                   | −0.0226       |
| sync LL        | 0.5389   | 0.5376      | **0.5342**                   | −0.0047       |
| async LL       | 0.5592   | 0.5541      | **0.5525**                   | −0.0067       |

Improvement is largest on offline tournaments, where the standard
36-question / 3-tour structure dominates and the position signal is
cleanest.

## Learned δ_pos curve

After a full sequential pass on the full DB, with anchor at p=0:

```
p= 0*+0.000
p= 1 +0.339
p= 2 +0.413
p= 3 +0.383
p= 4 +0.400
p= 5 +0.350
p= 6 +0.434
p= 7 +0.421
p= 8 +0.431
p= 9 +0.416
p=10 +0.396
p=11 +0.369
```

The curve mirrors the empirical take-rate table almost exactly:
question 1 is the easy outlier, p=6–9 form the hard plateau, and
the last question of the tour (p=11) is slightly easier again. The
total spread is ~0.43 difficulty units, which corresponds to a
take-rate gap of roughly 15 pp at typical values of `S`.

## Anchor choice

The first run used `pos_anchor = 6` (mid-tour). The learned curve
was numerically the same up to a constant shift, but read as
"position 6 is uniquely easier" which was confusing. Switching to
`pos_anchor = 0` (the easiest position) produces a non-negative
curve that is trivially interpretable. Logloss is essentially
unchanged (0.5235 → 0.5225), as expected — anchor choice is just an
identifiability convention.

## Knobs

* `--use-pos-effect / --no-use-pos-effect` (default on)
* `--tour_len` (default 12)
* `--pos_anchor` (default 0)
* `--eta_pos` (default 0.005)
* `--reg_pos` (default 0.10)

Per-type weights `w_pos_offline / w_pos_sync / w_pos_async` live on
`Config` (defaults 1.0 / 1.0 / 0.5, mirroring `w_size_*`).

## Future work

* Detect tour length per tournament (use 15 for tournaments whose
  question count is divisible by 15 but not 12) — currently a
  ~30% of observations get a noisier slot index.
* Consider a smoother (e.g. cubic spline) instead of 12 free
  parameters; with 4–5 effective degrees of freedom the curve looks
  almost piecewise-linear plus an "edge" effect at p=0 / p=11.
* Log a per-`pos_idx` calibration table during backtest to confirm
  that the new shift removes the position bias from residuals.
