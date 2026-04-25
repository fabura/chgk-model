"""Compare per-team actual vs expected takes for a given tournament,
using two DuckDB files (before/after a model change).

Usage:
    .venv/bin/python scripts/diagnostic_compare.py \
        --old website/data/chgk.duckdb.old \
        --new website/data/chgk.duckdb \
        13104 12469 13041

Designed to validate that changes to question-init / hyperparameters
fix systematic over-prediction on hard packs (the original Vyshka
issue).  Prints, for each tournament:

    place team                       n  actual  exp_old  exp_new  Δ_old   Δ_new
    1     ...                        6      28    32.4    29.1    -4.4    -1.1
    ...

Where Δ = actual - expected.  Negative Δ means the model over-predicted
the team (and would have penalised θ for these players).
"""
from __future__ import annotations

import argparse
import sys

import duckdb


def fetch(con, tid: int):
    rows = con.execute(
        """
        SELECT tg.team_id, tg.team_name, tg.n_players_active,
               tg.score_actual, tg.expected_takes, tg.place
        FROM team_games tg
        WHERE tg.tournament_id = ?
        ORDER BY tg.score_actual DESC NULLS LAST, tg.team_id
        """,
        [tid],
    ).fetchall()
    return rows


def title_of(con, tid: int) -> str:
    r = con.execute(
        "SELECT title, start_date, type, n_questions, n_teams "
        "FROM tournaments WHERE tournament_id = ?", [tid]
    ).fetchone()
    if not r:
        return f"<unknown {tid}>"
    title, dt, typ, nq, nt = r
    return f"{title} | {dt} | {typ} | {nq}q × {nt}t"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Old DuckDB path (before-state)")
    ap.add_argument("--new", required=True, help="New DuckDB path (after-state)")
    ap.add_argument(
        "tournament_ids", nargs="+", type=int, help="Tournament IDs to inspect",
    )
    args = ap.parse_args()

    old_con = duckdb.connect(args.old, read_only=True)
    new_con = duckdb.connect(args.new, read_only=True)

    for tid in args.tournament_ids:
        title = title_of(new_con, tid) if title_of(new_con, tid) else title_of(old_con, tid)
        print(f"\n{'=' * 90}")
        print(f"  T{tid}  {title}")
        print(f"{'=' * 90}")

        old_rows = {r[0]: r for r in fetch(old_con, tid)}
        new_rows = {r[0]: r for r in fetch(new_con, tid)}
        common = sorted(
            set(old_rows) & set(new_rows),
            key=lambda t: (
                -(old_rows[t][3] or 0),  # by actual desc
                t,
            ),
        )

        # Header
        hdr = (
            f"{'place':>5} {'team':30s} {'n':>2} "
            f"{'actual':>6} {'exp_old':>7} {'exp_new':>7} "
            f"{'Δ_old':>6} {'Δ_new':>6} {'Δ_swing':>7}"
        )
        print(hdr)
        print("-" * len(hdr))

        sums = {"actual": 0.0, "old": 0.0, "new": 0.0, "n": 0}
        for tid_ in common:
            o = old_rows[tid_]
            n = new_rows[tid_]
            place = o[5] if o[5] is not None else n[5]
            place_s = f"{int(place):>5}" if place is not None else "    -"
            name = (o[1] or n[1] or "")[:30]
            np_ = o[2] if o[2] is not None else n[2]
            actual = o[3] if o[3] is not None else 0
            exp_old = o[4] or 0.0
            exp_new = n[4] or 0.0
            d_old = actual - exp_old
            d_new = actual - exp_new
            swing = d_new - d_old  # positive = new is more generous
            print(
                f"{place_s} {name:30s} {np_ or 0:>2} "
                f"{actual:>6d} {exp_old:>7.1f} {exp_new:>7.1f} "
                f"{d_old:>+6.1f} {d_new:>+6.1f} {swing:>+7.2f}"
            )
            sums["actual"] += actual
            sums["old"] += exp_old
            sums["new"] += exp_new
            sums["n"] += 1
        if sums["n"]:
            n_t = sums["n"]
            ma = sums["actual"] / n_t
            mo = sums["old"] / n_t
            mn = sums["new"] / n_t
            print("-" * len(hdr))
            print(
                f"{'mean':>5} {'(' + str(n_t) + ' teams)':30s} "
                f"   {ma:>6.1f} {mo:>7.1f} {mn:>7.1f} "
                f"{ma - mo:>+6.1f} {ma - mn:>+6.1f} "
                f"{(ma - mn) - (ma - mo):>+7.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
