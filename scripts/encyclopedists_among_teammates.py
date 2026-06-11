#!/usr/bin/env python3
"""Encyclopedist profile for players who shared a roster with a given player."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np

from data import load_cached
from rating.io import load_results_npz
from scripts.find_hard_specialist_players import _aggregate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("player_id", type=int, help="Your player_id (DB)")
    ap.add_argument("--min-together", type=int, default=2)
    ap.add_argument("--min-obs", type=int, default=200, help="Min easy and hard question-obs")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    args = ap.parse_args()

    duck = Path(args.duckdb)
    if not duck.exists():
        raise SystemExit(f"No DuckDB at {duck}")

    con = duckdb.connect(str(duck), read_only=True)
    me = args.player_id
    you = con.execute(
        "SELECT first_name, last_name, theta, games FROM players WHERE player_id=?",
        [me],
    ).fetchone()
    if not you:
        raise SystemExit(f"player_id {me} not in DuckDB")
    print(f"You: {you[1]} {you[0]} (id={me})  θ={you[2]:+.2f}  games={you[3]}\n")

    together_rows = con.execute(
        """
        WITH my AS (
          SELECT tournament_id, team_id FROM player_games WHERE player_id = ?
        )
        SELECT pg.player_id, COUNT(DISTINCT pg.tournament_id) AS n
        FROM player_games pg
        INNER JOIN my ON my.tournament_id = pg.tournament_id AND my.team_id = pg.team_id
        WHERE pg.player_id != ?
        GROUP BY 1
        HAVING n >= ?
        """,
        [me, me, args.min_together],
    ).fetchall()
    together = {int(pid): int(n) for pid, n in together_rows}
    print(f"Co-players (>={args.min_together} tournaments): {len(together)}\n")

    arrays, maps = load_cached("data.npz")
    res = load_results_npz("results/seq.npz")
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
    b_canon = res.b.astype(np.float64)
    b_obs = b_canon[cq[arrays["q_idx"]]]
    b_q25, b_q75 = np.quantile(b_canon, [0.25, 0.75])

    tour_len = len(res.delta_pos)
    pos_shift = res.delta_pos.astype(np.float64) - res.delta_pos[
        int(res.pos_anchor or 0) % tour_len
    ]
    size_shift_table = res.delta_size.astype(np.float64) - res.delta_size[
        int(res.team_size_anchor or 6)
    ]
    qi = res.question_qi.astype(np.int64)
    delta_obs = size_shift_table[
        np.clip(arrays["team_sizes"], 0, len(size_shift_table) - 1)
    ] + pos_shift[qi[arrays["q_idx"]] % tour_len]
    offsets = np.zeros(len(arrays["team_sizes"]) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(arrays["team_sizes"])

    n_easy = np.zeros(n_players, dtype=np.int64)
    sum_re = np.zeros(n_players, dtype=np.float64)
    n_hard = np.zeros(n_players, dtype=np.int64)
    sum_rh = np.zeros(n_players, dtype=np.float64)
    ne_t = np.zeros(n_players, dtype=np.int64)
    sce = np.zeros(n_players, dtype=np.float64)
    nh_t = np.zeros(n_players, dtype=np.int64)
    sch = np.zeros(n_players, dtype=np.float64)
    print("Aggregating question-level stats…")
    _aggregate(
        offsets,
        arrays["team_sizes"].astype(np.int64),
        arrays["player_indices_flat"].astype(np.int64),
        arrays["taken"].astype(np.int8),
        b_obs.astype(np.float64),
        delta_obs.astype(np.float64),
        theta,
        float(b_q25),
        float(b_q75),
        n_easy,
        sum_re,
        n_hard,
        sum_rh,
        ne_t,
        sce,
        nh_t,
        sch,
    )

    rows = []
    for pid in together:
        j = pid_to_idx.get(pid)
        if j is None:
            continue
        ne, nh = int(n_easy[j]), int(n_hard[j])
        if ne < args.min_obs or nh < args.min_obs:
            continue
        re = float(sum_re[j] / ne)
        rh = float(sum_rh[j] / nh)
        gap = rh - re
        net_e, net_h = int(ne_t[j]), int(nh_t[j])
        ce = float(sce[j] / net_e) if net_e >= 20 else float("nan")
        ch = float(sch[j] / net_h) if net_h >= 20 else float("nan")
        cr = (ch / ce) if ce > 0 and ce == ce and ch == ch else float("nan")
        rows.append(
            dict(
                pid=pid,
                together=together[pid],
                theta=float(theta[j]),
                games=int(games[j]),
                res_easy=re,
                res_hard=rh,
                gap=gap,
                cred_ratio=cr,
            )
        )

    if not rows:
        print("No co-players with enough easy/hard observations.")
        return 0

    def enc_score(r: dict) -> float:
        cr = r["cred_ratio"] if r["cred_ratio"] == r["cred_ratio"] else 1.0
        return r["gap"] + min(0.0, r["res_easy"]) * 0.5 + (cr - 1.0) * 0.02

    rows.sort(key=lambda r: (-enc_score(r), -r["gap"], r["res_easy"]))

    print(
        f"{'name':30} {'with':>4} {'θ':>6} {'res_easy':>9} {'res_hard':>9} "
        f"{'gap':>7} {'cred×':>6}"
    )
    print("-" * 82)
    for r in rows[: args.top]:
        nm = con.execute(
            "SELECT last_name, first_name FROM players WHERE player_id=?", [r["pid"]]
        ).fetchone()
        name = f"{nm[0]} {nm[1]}" if nm else str(r["pid"])
        cr = r["cred_ratio"]
        cr_s = f"{cr:.2f}" if cr == cr else "  —"
        print(
            f"{name[:30]:30} {r['together']:4d} {r['theta']:+.2f} "
            f"{r['res_easy']:+.4f}  {r['res_hard']:+.4f}  {r['gap']:+.4f}  {cr_s:>6}"
        )

    strict = [
        r
        for r in rows
        if r["res_easy"] <= -0.02
        and r["gap"] >= 0.03
        and (r["cred_ratio"] != r["cred_ratio"] or r["cred_ratio"] >= 1.15)
    ]
    print(f"\nStrict «encyclopedists» among your mates: {len(strict)}")
    for r in strict[:15]:
        nm = con.execute(
            "SELECT last_name, first_name FROM players WHERE player_id=?", [r["pid"]]
        ).fetchone()
        name = f"{nm[0]} {nm[1]}" if nm else str(r["pid"])
        cr = r["cred_ratio"]
        cr_s = f"{cr:.2f}" if cr == cr else "—"
        print(
            f"  {name[:28]:28}  together={r['together']:3d}  "
            f"gap={r['gap']:+.3f}  easy={r['res_easy']:+.3f}  cred×{cr_s}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
