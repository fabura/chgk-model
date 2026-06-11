#!/usr/bin/env python3
"""Find players who over-contribute on hard questions but underperform on easy ones.

Heuristic for «знает сложное, но слаб в ЧГК-культуре на простом»:
  - on med-hard+hard questions (by learned b): team residual (y−p) not worse than baseline;
  - on easy questions (low b): team residual clearly below expectation;
  - when the team *does* take, player's noisy-OR marginal share is higher on hard than easy.

Uses data.npz observations + results/seq.npz parameters. Read-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import load_cached
from rating.io import load_results_npz

try:
    from numba import njit
except ImportError:
    njit = lambda fn, **kw: fn  # type: ignore


@njit(cache=True)
def _aggregate(
    offsets: np.ndarray,
    team_sizes: np.ndarray,
    player_flat: np.ndarray,
    taken: np.ndarray,
    b_obs: np.ndarray,
    delta_obs: np.ndarray,
    theta: np.ndarray,
    b_easy_max: float,
    b_hard_min: float,
    # per-player accumulators (num_players,)
    n_easy: np.ndarray,
    sum_res_easy: np.ndarray,
    n_hard: np.ndarray,
    sum_res_hard: np.ndarray,
    n_easy_take: np.ndarray,
    sum_cred_easy: np.ndarray,
    n_hard_take: np.ndarray,
    sum_cred_hard: np.ndarray,
) -> None:
    clamp = 20.0
    n_obs = len(team_sizes)
    for i in range(n_obs):
        start = offsets[i]
        n = team_sizes[i]
        if n <= 0:
            continue
        b = b_obs[i]
        delta = delta_obs[i]
        y = taken[i]
        eff_b = b + delta
        S = 0.0
        # first pass: S and store local thetas / player idx
        for j in range(n):
            pidx = player_flat[start + j]
            zk = -eff_b + theta[pidx]
            if zk < -clamp:
                zk = -clamp
            elif zk > clamp:
                zk = clamp
            S += np.exp(zk)
        if S < 1e-12:
            S = 1e-12
        if S > 500.0:
            expm_s = 0.0
        else:
            expm_s = np.exp(-S)
        p = 1.0 - expm_s if S > 1e-10 else S
        res = float(y) - p
        is_easy = b < b_easy_max
        is_hard = b >= b_hard_min
        if not is_easy and not is_hard:
            continue
        for j in range(n):
            pidx = player_flat[start + j]
            zk = -eff_b + theta[pidx]
            if zk < -clamp:
                zk = -clamp
            elif zk > clamp:
                zk = clamp
            lam = np.exp(zk)
            cred = expm_s * lam if y == 1 else 0.0
            if is_easy:
                n_easy[pidx] += 1
                sum_res_easy[pidx] += res
                if y == 1:
                    n_easy_take[pidx] += 1
                    sum_cred_easy[pidx] += cred
            if is_hard:
                n_hard[pidx] += 1
                sum_res_hard[pidx] += res
                if y == 1:
                    n_hard_take[pidx] += 1
                    sum_cred_hard[pidx] += cred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data.npz")
    ap.add_argument("--results", default="results/seq.npz")
    ap.add_argument("--min-games", type=int, default=120)
    ap.add_argument("--min-easy-obs", type=int, default=600)
    ap.add_argument("--min-hard-obs", type=int, default=600)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    arrays, maps = load_cached(args.cache)
    res = load_results_npz(args.results)

    n_players = maps.num_players
    pid_to_idx = maps.player_id_to_idx
    theta = np.zeros(n_players, dtype=np.float64)
    games = np.zeros(n_players, dtype=np.int64)
    for i, pid in enumerate(res.player_id):
        j = pid_to_idx.get(int(pid))
        if j is not None:
            theta[j] = float(res.theta[i])
            games[j] = int(res.games[i])

    cq = res.canonical_q_idx
    if cq is None:
        cq = np.arange(len(maps.idx_to_question_id), dtype=np.int32)
    b_canon = res.b.astype(np.float64)
    q_idx = arrays["q_idx"].astype(np.int64)
    b_obs = b_canon[cq[q_idx]]

  # difficulty quartiles on canonical b (weighted by obs count is expensive; use canon dist)
    b_q25, b_q50, b_q75 = np.quantile(b_canon, [0.25, 0.5, 0.75])
    b_easy_max = float(b_q25)
    b_hard_min = float(b_q75)
    print(f"Question b quartiles: Q25={b_q25:.3f} Q50={b_q50:.3f} Q75={b_q75:.3f}")
    print(f"  easy: b < {b_easy_max:.3f}   hard: b >= {b_hard_min:.3f}\n")

    tour_len = len(res.delta_pos) if res.delta_pos is not None else 12
    pos_anchor = int(res.pos_anchor or 0)
    size_anchor = int(res.team_size_anchor or 6)
    delta_pos = res.delta_pos.astype(np.float64) if res.delta_pos is not None else np.zeros(tour_len)
    delta_size = res.delta_size.astype(np.float64) if res.delta_size is not None else np.zeros(13)
    pos_shift = delta_pos - delta_pos[pos_anchor % tour_len]
    size_shift_table = delta_size - delta_size[min(size_anchor, len(delta_size) - 1)]

    qi = res.question_qi.astype(np.int64)
    pos_obs = (qi[q_idx] % tour_len).astype(np.int64)
    delta_obs = size_shift_table[np.clip(arrays["team_sizes"], 0, len(size_shift_table) - 1)] + pos_shift[pos_obs]

    team_sizes = arrays["team_sizes"].astype(np.int64)
    offsets = np.zeros(len(team_sizes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(team_sizes)
    player_flat = arrays["player_indices_flat"].astype(np.int64)
    taken = arrays["taken"].astype(np.int8)

    n_easy = np.zeros(n_players, dtype=np.int64)
    sum_res_easy = np.zeros(n_players, dtype=np.float64)
    n_hard = np.zeros(n_players, dtype=np.int64)
    sum_res_hard = np.zeros(n_players, dtype=np.float64)
    n_easy_take = np.zeros(n_players, dtype=np.int64)
    sum_cred_easy = np.zeros(n_players, dtype=np.float64)
    n_hard_take = np.zeros(n_players, dtype=np.int64)
    sum_cred_hard = np.zeros(n_players, dtype=np.float64)

    print(f"Aggregating {len(team_sizes):,} observations…")
    _aggregate(
        offsets,
        team_sizes,
        player_flat,
        taken,
        b_obs.astype(np.float64),
        delta_obs.astype(np.float64),
        theta,
        b_easy_max,
        b_hard_min,
        n_easy,
        sum_res_easy,
        n_hard,
        sum_res_hard,
        n_easy_take,
        sum_cred_easy,
        n_hard_take,
        sum_cred_hard,
    )

    # player-level metrics
    candidates = []
    for pidx in range(n_players):
        ne, nh = int(n_easy[pidx]), int(n_hard[pidx])
        if ne < args.min_easy_obs or nh < args.min_hard_obs:
            continue
        g = int(games[pidx])
        if g < args.min_games:
            continue
        res_e = sum_res_easy[pidx] / ne
        res_h = sum_res_hard[pidx] / nh
        gap = res_h - res_e
        net_e = int(n_easy_take[pidx])
        net_h = int(n_hard_take[pidx])
        cred_e = (sum_cred_easy[pidx] / net_e) if net_e >= 30 else np.nan
        cred_h = (sum_cred_hard[pidx] / net_h) if net_h >= 30 else np.nan
        cred_ratio = (cred_h / cred_e) if cred_e and cred_e > 1e-9 and not np.isnan(cred_h) else np.nan

        # profile filters
        if res_e > -0.012:  # must underperform on easy (team-level when in roster)
            continue
        if gap < 0.018:  # must be relatively better on hard
            continue
        if res_h > 0.025:  # not overall weak on hard — we're not looking for pure weaklings
            continue
        if theta[pidx] > 0.55:  # skip established top θ
            continue
        if cred_ratio == cred_ratio and cred_ratio < 1.08:  # marginal credit on takes
            continue

        pid = maps.idx_to_player_id[pidx]
        candidates.append(
            dict(
                pid=pid,
                theta=float(theta[pidx]),
                games=g,
                n_easy=ne,
                n_hard=nh,
                res_easy=res_e,
                res_hard=res_h,
                gap=gap,
                cred_easy=cred_e,
                cred_hard=cred_h,
                cred_ratio=cred_ratio,
            )
        )

    candidates.sort(key=lambda x: (-x["gap"], x["res_easy"]))
    print(f"\nFound {len(candidates)} players matching profile\n")
    print(
        f"{'name':32} {'θ':>6} {'games':>5}  "
        f"{'res_easy':>9} {'res_hard':>9} {'gap':>6}  "
        f"{'cred_e':>7} {'cred_h':>7} {'ratio':>5}"
    )
    print("-" * 100)

    # optional names from duckdb
    names: dict[int, str] = {}
    duck = Path("website/data/chgk.duckdb")
    if duck.exists():
        import duckdb

        con = duckdb.connect(str(duck), read_only=True)
        for c in candidates[: args.top]:
            row = con.execute(
                "SELECT first_name, last_name FROM players WHERE player_id=?",
                [c["pid"]],
            ).fetchone()
            if row:
                names[c["pid"]] = f"{row[1] or ''} {row[0] or ''}".strip()

    for c in candidates[: args.top]:
        nm = names.get(c["pid"], str(c["pid"]))
        cr = c["cred_ratio"]
        cr_s = f"{cr:.2f}" if cr == cr else "  —"
        ce = c["cred_easy"]
        ch = c["cred_hard"]
        ce_s = f"{ce:.4f}" if ce == ce else "   —"
        ch_s = f"{ch:.4f}" if ch == ch else "   —"
        print(
            f"{nm[:32]:32} {c['theta']:+.2f} {c['games']:5d}  "
            f"{c['res_easy']:+.4f}  {c['res_hard']:+.4f}  {c['gap']:+.4f}  "
            f"{ce_s} {ch_s} {cr_s:>5}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
