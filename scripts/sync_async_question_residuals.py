"""Find canonical questions whose empirical take rate differs most
between sync and async observations on the SAME pack.

The model currently shares (b, a) across paired sync↔async tournaments
(via ``canonical_q_idx``); mode differences are absorbed only by the
per-mode SGD weights.  This script does NOT re-fit anything and does
NOT depend on the model's per-observation predictions (the engine
collects them in chronological-game order, which is not the same as
the order of ``arrays['q_idx']``, so any "pair pred_p with q_idx via
the is_known mask" trick is silently wrong on the mode bucket — see
the alignment check below).

Approach: pure empirical, with a coarse adjustment for team strength.

For every observation:

    canonical_idx[i]    — which question (shared across pair)
    bucket[i]           — sync / async / offline (offline ignored)
    team_theta_mean[i]  — average pre-tournament θ of the team's
                          mature players (history-derived)

For each canonical question seen ≥ ``--min-obs`` times in BOTH sync and
async, we report:

    rate_sync, rate_async               — raw empirical take rates
    Δrate = rate_async − rate_sync
    θ̄_sync, θ̄_async                    — mean team θ per bucket
    Δrate_adj                           — Δrate after subtracting the
                                          rate change you'd expect
                                          purely from the team-θ gap,
                                          using the within-canonical
                                          rate-vs-θ slope estimated
                                          across BOTH buckets pooled.

Output: a CSV ranked by ``|Δrate_adj|``.  A question whose adjusted
delta is large means the same canonical pack — fed to teams of
comparable strength — yields a markedly different take rate by mode.
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_cached  # noqa: E402
from rating.io import load_results_npz  # noqa: E402


def _bucket(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def _fetch_question_text(qdb: Optional[Path], tid_qi_pairs: list[tuple[int, int]]) -> dict:
    """Best-effort lookup of question text+answer from the questions sqlite."""
    if qdb is None or not qdb.exists() or not tid_qi_pairs:
        return {}
    out: dict[tuple[int, int], tuple[str, str]] = {}
    with sqlite3.connect(str(qdb)) as conn:
        cur = conn.cursor()
        try:
            cur.execute("PRAGMA table_info(questions)")
            cols = {r[1] for r in cur.fetchall()}
        except sqlite3.Error:
            return {}
        if not {"tournament_id", "q_in_tournament", "text"}.issubset(cols):
            return {}
        ans_col = "answer" if "answer" in cols else None
        placeholders = ",".join(["(?,?)"] * len(tid_qi_pairs))
        params: list = []
        for tid, qi in tid_qi_pairs:
            params.extend([int(tid), int(qi)])
        sel = "tournament_id, q_in_tournament, text" + (
            f", {ans_col}" if ans_col else ""
        )
        try:
            cur.execute(
                f"SELECT {sel} FROM questions "
                f"WHERE (tournament_id, q_in_tournament) IN ({placeholders})",
                params,
            )
            for row in cur.fetchall():
                tid, qi, txt = int(row[0]), int(row[1]), (row[2] or "").strip()
                ans = (row[3] or "").strip() if ans_col else ""
                out[(tid, qi)] = (txt, ans)
        except sqlite3.Error:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data.npz")
    ap.add_argument("--results", default="results/seq.npz")
    ap.add_argument(
        "--questions-db",
        default=os.path.expanduser(
            "~/Projects/personal/chgk-embedings/data/questions.db"
        ),
    )
    ap.add_argument(
        "--out", default="results/sync_async_question_residuals.csv"
    )
    ap.add_argument(
        "--min-obs", type=int, default=30,
        help="Minimum observations per (canonical, mode) bucket.",
    )
    ap.add_argument(
        "--top", type=int, default=200,
        help="How many top |Δrate_adj| rows to write to the CSV (0 = all).",
    )
    args = ap.parse_args()

    print(f"[sa-resid] loading {args.cache}…", flush=True)
    arrays, maps = load_cached(args.cache)

    print(f"[sa-resid] loading {args.results}…", flush=True)
    res = load_results_npz(args.results)

    # --- per-observation arrays we need ---
    q_idx = arrays["q_idx"]
    game_idx = arrays["game_idx"]
    taken = arrays["taken"].astype(np.float64)
    team_sizes = arrays["team_sizes"]
    pflat = arrays["player_indices_flat"].astype(np.int64)
    offsets = np.cumsum(np.concatenate([[0], team_sizes])).astype(np.int64)

    # canonical question id (shared across pair); fall back to raw if absent.
    canonical = (
        maps.canonical_q_idx
        if getattr(maps, "canonical_q_idx", None) is not None
        else q_idx
    ).astype(np.int64)
    cq = canonical[q_idx]

    # mode bucket per observation.
    gtype = maps.game_type
    if gtype is None:
        raise RuntimeError("game_type missing from cache; can't bucket")
    g_bucket = np.array([_bucket(str(gtype[g])) for g in range(len(gtype))], dtype=object)
    obs_bucket = g_bucket[game_idx]

    # We only want sync and async observations (offline ignored — there's
    # no canonical-pack overlap with sync/async by construction).
    keep = (obs_bucket == "sync") | (obs_bucket == "async")
    print(
        f"[sa-resid] obs total: {len(taken):,}, "
        f"sync+async kept: {int(keep.sum()):,}",
        flush=True,
    )

    # --- pre-tournament team θ̄ per observation ---
    # We use the model's per-(player, tournament) pre-tournament θ from the
    # history arrays in seq.npz.  Build a lookup
    # (db_player_id, db_tournament_id) → θ̄ once, then average over the
    # team's roster per observation.
    if (
        res.history_player_id is None
        or res.history_game_id is None
        or res.history_theta is None
    ):
        raise RuntimeError(
            "results file lacks per-(player, tournament) θ history; "
            "rerun training to produce it."
        )

    # Build (player_id, tournament_id) → θ̄ dict.  We expect ~3M entries.
    print("[sa-resid] building θ̄ lookup…", flush=True)
    h_pid = res.history_player_id.astype(np.int64)
    h_gid = res.history_game_id.astype(np.int64)
    h_th = res.history_theta.astype(np.float64)
    theta_lookup: dict[tuple[int, int], float] = {
        (int(p), int(g)): float(t) for p, g, t in zip(h_pid, h_gid, h_th)
    }

    pid_arr = np.array(maps.idx_to_player_id, dtype=np.int64)
    gid_arr = np.array(maps.idx_to_game_id, dtype=np.int64)

    # Per-observation team-θ mean.  Vectorising the dict lookup is awkward,
    # so we do it group-by-game to amortise the (player → θ̄_at_this_game)
    # build into one fast numpy op per game.
    print("[sa-resid] computing per-obs team θ̄…", flush=True)
    team_theta = np.full(len(taken), np.nan, dtype=np.float64)
    # Index observations by game.
    obs_order = np.argsort(game_idx, kind="stable")
    g_sorted = game_idx[obs_order]
    # find boundaries
    boundaries = np.concatenate(
        ([0], np.where(np.diff(g_sorted) != 0)[0] + 1, [len(g_sorted)])
    )
    for bi in range(len(boundaries) - 1):
        lo, hi = boundaries[bi], boundaries[bi + 1]
        g = int(g_sorted[lo])
        gid_db = int(gid_arr[g])
        # theta_at_this_game per local player_idx (only those with history)
        # local map: player_idx -> θ̄
        # We'll build an array of size num_players with NaN, then fill.
        # That's wasteful for 50k players × 8.7k games; instead, gather
        # only the players we need.
        idxs_in_game = obs_order[lo:hi]
        # collect unique players touched in this game
        players_per_obs: list[np.ndarray] = []
        for i in idxs_in_game:
            s, e = offsets[i], offsets[i] + team_sizes[i]
            players_per_obs.append(pflat[s:e])
        if not players_per_obs:
            continue
        all_players = np.concatenate(players_per_obs)
        unique_p = np.unique(all_players)
        # lookup θ̄ for each
        th_for_p = np.full(unique_p.size, np.nan, dtype=np.float64)
        for k, p_local in enumerate(unique_p):
            pid_db = int(pid_arr[p_local])
            v = theta_lookup.get((pid_db, gid_db))
            if v is not None:
                th_for_p[k] = v
        # build a small map from player_local → θ̄
        sort_unique = unique_p
        # for each obs in this game compute mean θ̄ (skipping NaN players)
        for j, i in enumerate(idxs_in_game):
            ps = players_per_obs[j]
            # find positions of ps in sort_unique (already sorted by unique)
            pos = np.searchsorted(sort_unique, ps)
            ths = th_for_p[pos]
            valid = ~np.isnan(ths)
            if valid.any():
                team_theta[int(i)] = float(ths[valid].mean())

    have_theta = ~np.isnan(team_theta)
    keep &= have_theta
    print(
        f"[sa-resid] obs with θ̄ available: {int(keep.sum()):,}",
        flush=True,
    )

    # --- per (canonical, mode) aggregation ---
    cq_k = cq[keep]
    bk_k = obs_bucket[keep]
    y_k = taken[keep]
    th_k = team_theta[keep]

    # Aggregate sums.
    agg: dict[tuple[int, str], dict] = {}
    for c, b, y, t in zip(cq_k, bk_k, y_k, th_k):
        s = agg.setdefault(
            (int(c), str(b)),
            {"n": 0, "sum_y": 0.0, "sum_th": 0.0, "sum_y_th": 0.0, "sum_th2": 0.0},
        )
        s["n"] += 1
        s["sum_y"] += float(y)
        s["sum_th"] += float(t)
        s["sum_y_th"] += float(y) * float(t)
        s["sum_th2"] += float(t) * float(t)

    # Pivot to per-canonical, keep only those present in both buckets.
    by_cq: dict[int, dict[str, dict]] = defaultdict(dict)
    for (c, b), s in agg.items():
        by_cq[c][b] = s

    rows: list[dict] = []
    for c, modes in by_cq.items():
        s_sync = modes.get("sync")
        s_async = modes.get("async")
        if s_sync is None or s_async is None:
            continue
        if s_sync["n"] < args.min_obs or s_async["n"] < args.min_obs:
            continue

        n_total = s_sync["n"] + s_async["n"]
        sum_y = s_sync["sum_y"] + s_async["sum_y"]
        sum_th = s_sync["sum_th"] + s_async["sum_th"]
        sum_y_th = s_sync["sum_y_th"] + s_async["sum_y_th"]
        sum_th2 = s_sync["sum_th2"] + s_async["sum_th2"]
        # Pooled within-canonical OLS slope: rate ≈ α + β·θ̄.
        denom = n_total * sum_th2 - sum_th * sum_th
        if denom > 1e-9:
            slope = (n_total * sum_y_th - sum_y * sum_th) / denom
        else:
            slope = 0.0

        rate_sync = s_sync["sum_y"] / s_sync["n"]
        rate_async = s_async["sum_y"] / s_async["n"]
        th_sync = s_sync["sum_th"] / s_sync["n"]
        th_async = s_async["sum_th"] / s_async["n"]
        delta_rate = rate_async - rate_sync
        delta_rate_adj = delta_rate - slope * (th_async - th_sync)

        rows.append(
            {
                "canonical_idx": c,
                "n_sync": s_sync["n"],
                "n_async": s_async["n"],
                "rate_sync": rate_sync,
                "rate_async": rate_async,
                "th_sync": th_sync,
                "th_async": th_async,
                "delta_rate": delta_rate,
                "slope": slope,
                "delta_rate_adj": delta_rate_adj,
                "abs_delta_adj": abs(delta_rate_adj),
            }
        )

    rows.sort(key=lambda r: r["abs_delta_adj"], reverse=True)
    if args.top > 0:
        rows = rows[: args.top]

    # Resolve canonical → primary (tid, qi) + alias list for context.
    cq_to_aliases: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for raw_idx, key in enumerate(maps.idx_to_question_id):
        c = int(canonical[raw_idx])
        cq_to_aliases[c].append((int(key[0]), int(key[1])))
    primary: dict[int, tuple[int, int]] = {
        c: sorted(lst)[0] for c, lst in cq_to_aliases.items()
    }

    qdb_path = Path(args.questions_db)
    text_lookup = _fetch_question_text(
        qdb_path, [primary[r["canonical_idx"]] for r in rows]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sa-resid] writing {len(rows)} rows to {out_path}", flush=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "canonical_idx",
                "primary_tournament_id",
                "primary_q_in_tournament",
                "n_sync", "n_async",
                "rate_sync", "rate_async",
                "th_sync", "th_async",
                "delta_rate", "slope", "delta_rate_adj",
                "all_tournament_ids",
                "text", "answer",
            ]
        )
        for r in rows:
            cq_v = r["canonical_idx"]
            ptid, pqi = primary[cq_v]
            text, answer = text_lookup.get((ptid, pqi), ("", ""))
            all_tids = ",".join(
                str(tid) for tid, _ in sorted(set(cq_to_aliases[cq_v]))
            )
            w.writerow(
                [
                    cq_v, ptid, pqi,
                    r["n_sync"], r["n_async"],
                    f"{r['rate_sync']:.4f}", f"{r['rate_async']:.4f}",
                    f"{r['th_sync']:+.3f}", f"{r['th_async']:+.3f}",
                    f"{r['delta_rate']:+.4f}", f"{r['slope']:+.3f}",
                    f"{r['delta_rate_adj']:+.4f}",
                    all_tids,
                    text, answer,
                ]
            )

    if rows:
        deltas = np.array([r["delta_rate_adj"] for r in rows])
        print(
            f"[sa-resid] kept {len(rows):,} canonical questions "
            f"(both modes ≥ {args.min_obs} obs).  "
            f"Δrate_adj on top-{len(rows)}: mean {deltas.mean():+.4f}, "
            f"median {np.median(deltas):+.4f}, "
            f"|Δ| max {np.abs(deltas).max():.4f}"
        )


if __name__ == "__main__":
    main()
