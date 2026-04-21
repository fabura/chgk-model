"""Per-question uncertainty estimates for the rating model.

For every canonical question we report:

  * ``n_obs``                – number of (team, question) observations
  * ``take_rate``            – empirical fraction of teams that took it
  * ``take_rate_lo``,
    ``take_rate_hi``         – 95% Wilson confidence interval
  * ``b``, ``b_se``,
    ``b_lo``, ``b_hi``       – difficulty estimate (positive = harder)
                                with asymptotic 95% CI from Fisher info
  * ``a``, ``a_se``,
    ``a_lo``, ``a_hi``       – discrimination estimate (selectivity)
                                with asymptotic 95% CI; ``a_se`` is the
                                delta-method SE (a = exp(log_a))
  * ``r_pb``                 – point-biserial correlation between average
                                team θ and y (classical IRT
                                discrimination index, range −1..+1)
  * ``info``                 – Fisher information at the median observed
                                team strength (peak of the IRF), useful as
                                a one-number "how informative is this
                                question" score.

The Fisher information matrix for one observation is

    I_t(b, a) = (1 / expm1(S_t)) · [[ S_t² , −S_t·T_t ],
                                     [ −S_t·T_t , T_t² ]]

with ``S_t = Σ_k λ_{kt}`` and ``T_t = Σ_k θ_k · λ_{kt}`` (the noisy-OR
intensity and its θ-weighted analogue).  Summing across the teams that
played the question gives the per-question Fisher matrix; its inverse
yields SEs for ``b`` and ``log_a`` (then ``a_se = a · log_a_se`` by the
delta method).

Run from the repo root:

    python scripts/question_uncertainties.py \
        --cache_file data.npz \
        --out results/question_uncertainties.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_cached
from rating.engine import Config, run_sequential

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is in requirements
    def njit(*a, **kw):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn


@njit(cache=True)
def _per_obs_S_T(
    offsets: np.ndarray,
    pflat: np.ndarray,
    q_idx: np.ndarray,
    cq: np.ndarray,
    theta: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    delta_obs: np.ndarray,
    out_S: np.ndarray,
    out_T: np.ndarray,
    out_avg_theta: np.ndarray,
) -> None:
    """For each observation, compute S_t, T_t and the team-average θ.

    ``a`` is the linear discrimination ``exp(log_a)`` per *canonical*
    question.  Outputs are written in-place.
    """
    n_obs = len(out_S)
    for i in range(n_obs):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        qi = int(cq[int(q_idx[i])])
        b_i = b[qi]
        a_i = a[qi]
        d_i = delta_obs[i]
        S = 0.0
        T = 0.0
        th_sum = 0.0
        n_team = e - s
        for k in range(s, e):
            th_k = theta[int(pflat[k])]
            z = a_i * th_k - b_i - d_i
            if z > 20.0:
                z = 20.0
            elif z < -20.0:
                z = -20.0
            lam = math.exp(z)
            S += lam
            T += th_k * lam
            th_sum += th_k
        out_S[i] = S
        out_T[i] = T
        if n_team > 0:
            out_avg_theta[i] = th_sum / n_team
        else:
            out_avg_theta[i] = 0.0


def _wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Wilson 95% CI for binomial proportion."""
    p_hat = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    denom = 1.0 + z * z / np.maximum(n, 1)
    centre = (p_hat + z * z / (2.0 * np.maximum(n, 1))) / denom
    spread = (
        z
        * np.sqrt(
            (p_hat * (1.0 - p_hat) + z * z / (4.0 * np.maximum(n, 1)))
            / np.maximum(n, 1)
        )
        / denom
    )
    return centre - spread, centre + spread


def main(
    cache_file: str = "data.npz",
    out_path: str = "results/question_uncertainties.csv",
    min_obs: int = 1,
) -> None:
    print(f"Loading {cache_file} ...")
    arrays, maps = load_cached(cache_file)

    print("Training model with current defaults ...")
    t0 = time.time()
    cfg = Config()
    result = run_sequential(arrays, maps, cfg, verbose=False)
    print(f"  done in {time.time() - t0:.1f}s")

    theta = result.players.theta
    b = result.questions.b
    log_a = np.clip(result.questions.log_a, -3.0, 3.0)
    a = np.exp(log_a)
    n_canonical = b.shape[0]

    delta_size = (
        result.delta_size
        if result.delta_size is not None
        else np.zeros(1, dtype=np.float64)
    )
    delta_pos = (
        result.delta_pos
        if result.delta_pos is not None
        else np.zeros(1, dtype=np.float64)
    )
    size_anchor = int(result.team_size_anchor)
    pos_anchor = int(result.pos_anchor)

    q_idx = arrays["q_idx"].astype(np.int64)
    taken = arrays["taken"].astype(np.float64)
    team_sizes = arrays["team_sizes"].astype(np.int64)
    pflat = arrays["player_indices_flat"].astype(np.int64)
    offsets = np.cumsum(np.concatenate([[0], team_sizes])).astype(np.int64)
    cq = (
        maps.canonical_q_idx.astype(np.int64)
        if hasattr(maps, "canonical_q_idx")
        else q_idx.copy()
    )

    n_obs = len(q_idx)
    print(f"  {n_obs:,} obs over {n_canonical:,} canonical questions")

    # ---- per-obs δ_obs (vectorised) -------------------------------------
    # δ now only collects δ_size + δ_pos (μ_type and ε_t were removed in
    # 2026-04 — see rating/tournaments.py).
    print("Computing per-observation δ_obs ...")
    delta_obs = np.zeros(n_obs, dtype=np.float64)

    if delta_size is not None and len(delta_size) > 1:
        max_size_idx = len(delta_size) - 1
        ts_clip = np.clip(team_sizes, 1, max_size_idx).astype(np.int64)
        size_contrib = delta_size[ts_clip]
        size_contrib[ts_clip == size_anchor] = 0.0
        delta_obs += size_contrib

    if delta_pos is not None and len(delta_pos) > 1:
        tour_len = len(delta_pos)
        qids = maps.idx_to_question_id
        q_pos_in_tour = np.zeros(n_canonical_or_raw := len(qids), dtype=np.int64)
        for raw_qi in range(len(qids)):
            qq = qids[raw_qi]
            q_pos_in_tour[raw_qi] = (
                int(qq[1]) % tour_len if isinstance(qq, tuple) else raw_qi % tour_len
            )
        pos_per_obs = q_pos_in_tour[q_idx]
        pos_contrib = delta_pos[pos_per_obs]
        pos_contrib[pos_per_obs == pos_anchor] = 0.0
        delta_obs += pos_contrib

    delta_obs = delta_obs.astype(np.float64)

    # ---- per-obs S_t, T_t (jitted) --------------------------------------
    print("Computing per-observation S_t, T_t (numba) ...")
    out_S = np.zeros(n_obs, dtype=np.float64)
    out_T = np.zeros(n_obs, dtype=np.float64)
    out_avg_theta = np.zeros(n_obs, dtype=np.float64)
    t0 = time.time()
    _per_obs_S_T(
        offsets,
        pflat,
        q_idx,
        cq,
        theta,
        b,
        a,
        delta_obs,
        out_S,
        out_T,
        out_avg_theta,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    # ---- aggregate per canonical question via bincount ------------------
    print("Aggregating Fisher information per question ...")
    qi_arr = cq[q_idx]
    valid = (out_S > 1e-12) & (out_S < 500.0)
    qi_v = qi_arr[valid]
    S_v = out_S[valid]
    T_v = out_T[valid]
    w_v = 1.0 / np.expm1(S_v)
    w_v[~np.isfinite(w_v)] = 0.0

    I_bb_q = np.bincount(qi_v, weights=w_v * S_v * S_v, minlength=n_canonical)
    I_ba_q = -np.bincount(qi_v, weights=w_v * S_v * T_v, minlength=n_canonical)
    I_aa_q = np.bincount(qi_v, weights=w_v * T_v * T_v, minlength=n_canonical)
    n_q = np.bincount(qi_arr, minlength=n_canonical)
    k_q = np.bincount(qi_arr, weights=taken, minlength=n_canonical)

    # 2×2 inverse — analytical
    det_q = I_bb_q * I_aa_q - I_ba_q * I_ba_q
    safe = det_q > 1e-12
    var_b = np.full(n_canonical, np.nan)
    var_loga = np.full(n_canonical, np.nan)
    var_b[safe] = I_aa_q[safe] / det_q[safe]
    var_loga[safe] = I_bb_q[safe] / det_q[safe]
    se_b = np.sqrt(np.maximum(var_b, 0.0))
    se_loga = np.sqrt(np.maximum(var_loga, 0.0))
    # Note: the Fisher matrix above is wrt (b, a), not (b, log_a).  We
    # actually want SE(log_a) for symmetric CIs on a; but I_aa is wrt a
    # so SE(a)≈√var.  Use delta-method for a and exponentiate to get
    # log-symmetric CIs.
    se_a = se_loga  # because we computed I in (b, a) space
    # 95% CIs
    b_lo, b_hi = b - 1.96 * se_b, b + 1.96 * se_b
    a_lo = np.maximum(a - 1.96 * se_a, 0.0)
    a_hi = a + 1.96 * se_a

    take_rate = np.where(n_q > 0, k_q / np.maximum(n_q, 1), np.nan)
    tr_lo, tr_hi = _wilson_ci(k_q, n_q)
    tr_lo[n_q == 0] = np.nan
    tr_hi[n_q == 0] = np.nan

    # ---- point-biserial r per question via bincount sums ---------------
    print("Computing point-biserial r per question ...")
    th = out_avg_theta
    y = taken
    sum_th = np.bincount(qi_arr, weights=th, minlength=n_canonical)
    sum_th2 = np.bincount(qi_arr, weights=th * th, minlength=n_canonical)
    sum_y = k_q
    sum_y2 = sum_y  # y∈{0,1} ⇒ y²=y
    sum_thy = np.bincount(qi_arr, weights=th * y, minlength=n_canonical)

    n_safe = np.maximum(n_q, 1)
    mean_th = sum_th / n_safe
    mean_y = sum_y / n_safe
    var_th = np.maximum(sum_th2 / n_safe - mean_th * mean_th, 0.0)
    var_y = np.maximum(sum_y2 / n_safe - mean_y * mean_y, 0.0)
    cov_thy = sum_thy / n_safe - mean_th * mean_y
    denom = np.sqrt(var_th * var_y)
    r_pb = np.where(denom > 1e-12, cov_thy / np.maximum(denom, 1e-12), np.nan)
    r_pb[n_q < 2] = np.nan

    # ---- Fisher information at the median team strength of the obs ----
    # info(θ*) = a² · S(θ*) / (e^{S(θ*)} - 1)  (per-obs Fisher info on θ_k
    # marginal); a one-number "how informative" score.  Use median of
    # observed average-team-θ as θ*.
    median_th = np.full(n_canonical, np.nan)
    # Cheap: per-question median via sorted partitioning.  For most
    # questions n_obs is small, so the simple approach is fine.
    print("Computing per-question median team θ ...")
    order = np.argsort(qi_arr, kind="stable")
    qi_sorted = qi_arr[order]
    th_sorted = th[order]
    starts = np.searchsorted(qi_sorted, np.arange(n_canonical), side="left")
    ends = np.searchsorted(qi_sorted, np.arange(n_canonical), side="right")
    for qi in range(n_canonical):
        s_, e_ = int(starts[qi]), int(ends[qi])
        if e_ > s_:
            median_th[qi] = float(np.median(th_sorted[s_:e_]))

    # info at median θ — single-obs information for a *single* player at
    # that θ (treating a "team" of size 1 with strength = median team-avg)
    z_star = a * median_th - b
    z_star_clip = np.clip(z_star, -20, 20)
    S_star = np.exp(z_star_clip)
    info_med = np.where(
        np.isfinite(median_th) & (S_star > 1e-12),
        (a * a) * S_star / np.expm1(np.maximum(S_star, 1e-12)),
        np.nan,
    )

    # ---- write CSV ------------------------------------------------------
    print(f"Writing {out_path} ...")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    qids = maps.idx_to_question_id
    # canonical → first raw question that maps to it (good enough for
    # producing a tournament_id / q_in_tournament tag for display)
    canon_to_first_raw = np.full(n_canonical, -1, dtype=np.int64)
    for raw_qi in range(len(qids)):
        c = int(cq[raw_qi])
        if canon_to_first_raw[c] < 0:
            canon_to_first_raw[c] = raw_qi

    fields = [
        "canonical_idx",
        "tournament_id",
        "q_in_tournament",
        "n_obs",
        "take_rate",
        "take_rate_lo",
        "take_rate_hi",
        "b",
        "b_se",
        "b_lo",
        "b_hi",
        "a",
        "a_se",
        "a_lo",
        "a_hi",
        "r_pb",
        "info_at_median_theta",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        kept = 0
        for qi in range(n_canonical):
            if int(n_q[qi]) < min_obs:
                continue
            raw = int(canon_to_first_raw[qi])
            if raw >= 0 and isinstance(qids[raw], tuple):
                tid, qpos = int(qids[raw][0]), int(qids[raw][1])
            else:
                tid, qpos = -1, -1

            def _fmt(x: float) -> str:
                if x is None or not np.isfinite(x):
                    return ""
                return f"{x:.4f}"

            w.writerow(
                [
                    qi,
                    tid,
                    qpos,
                    int(n_q[qi]),
                    _fmt(take_rate[qi]),
                    _fmt(tr_lo[qi]),
                    _fmt(tr_hi[qi]),
                    _fmt(b[qi]),
                    _fmt(se_b[qi]),
                    _fmt(b_lo[qi]),
                    _fmt(b_hi[qi]),
                    _fmt(a[qi]),
                    _fmt(se_a[qi]),
                    _fmt(a_lo[qi]),
                    _fmt(a_hi[qi]),
                    _fmt(r_pb[qi]),
                    _fmt(info_med[qi]),
                ]
            )
            kept += 1
    print(f"Wrote {kept:,} rows to {out_path}")

    # ---- quick summary printout -----------------------------------------
    finite_se_b = se_b[np.isfinite(se_b)]
    finite_se_a = se_a[np.isfinite(se_a)]
    print()
    print(f"SE(b)  median={np.median(finite_se_b):.3f}  90%-quantile={np.quantile(finite_se_b, 0.9):.3f}")
    print(f"SE(a)  median={np.median(finite_se_a):.3f}  90%-quantile={np.quantile(finite_se_a, 0.9):.3f}")
    print(f"r_pb  median={np.nanmedian(r_pb):.3f}  std={np.nanstd(r_pb):.3f}")
    print(f"n_obs median={int(np.median(n_q[n_q>0]))}  90%-quantile={int(np.quantile(n_q[n_q>0], 0.9))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_file", default="data.npz")
    p.add_argument("--out", default="results/question_uncertainties.csv")
    p.add_argument("--min_obs", type=int, default=1)
    args = p.parse_args()
    main(args.cache_file, args.out, args.min_obs)
