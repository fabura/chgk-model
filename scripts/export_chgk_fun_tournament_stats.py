#!/usr/bin/env python3
"""Export per-tournament fun vs ours stats to CSV (adult offline).

Default: offline tournaments since 2024-01-01 with the same title filters
as ``compare_chgk_fun_offline_adult.py``.

Run:
  .venv/bin/python scripts/export_chgk_fun_tournament_stats.py
  .venv/bin/python scripts/export_chgk_fun_tournament_stats.py --since 2024-01-01 \\
      --out results/chgk_fun_offline_adult_by_tournament_2024.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.compare_chgk_fun import DEFAULT_DUCKDB, DEFAULT_PARQUET  # noqa: E402
from scripts.compare_chgk_fun_offline_adult import _passes_filter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="2024-01-01")
    ap.add_argument("--until", default=None)
    ap.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "chgk_fun_offline_adult_by_tournament_2024.csv",
    )
    args = ap.parse_args()

    if not args.parquet.is_file():
        raise SystemExit(f"Parquet not found: {args.parquet}")
    if not args.duckdb.is_file():
        raise SystemExit(f"DuckDB not found: {args.duckdb}")

    fun: dict[int, dict[int, float]] = {}
    for row in pq.read_table(
        args.parquet, columns=["tournament_id", "team_id", "predictedquestions"]
    ).to_pylist():
        fun.setdefault(int(row["tournament_id"]), {})[int(row["team_id"])] = float(
            row["predictedquestions"]
        )

    con = duckdb.connect(str(args.duckdb), read_only=True)
    sql = """
        SELECT t.tournament_id, t.title, t.start_date, t.n_questions,
               tg.team_id, tg.score_actual, tg.expected_takes
        FROM tournaments t
        JOIN team_games tg USING (tournament_id)
        WHERE t.type = 'offline'
          AND t.start_date >= ?
          AND tg.expected_takes IS NOT NULL
          AND tg.score_actual IS NOT NULL
    """
    params: list = [args.since]
    if args.until:
        sql += " AND t.start_date <= ?"
        params.append(args.until)
    sql += " ORDER BY t.start_date, t.tournament_id"
    rows = con.execute(sql, params).fetchall()
    con.close()

    by_tid: dict[int, dict] = {}
    for tid, title, start_date, n_q, team_id, actual, expected in rows:
        if not _passes_filter(title or "")[0]:
            continue
        pf = fun.get(int(tid), {}).get(int(team_id))
        if pf is None:
            continue
        slot = by_tid.setdefault(
            int(tid),
            {
                "tournament_id": int(tid),
                "title": title or "",
                "start_date": str(start_date),
                "n_questions": int(n_q) if n_q else 0,
                "actual": [],
                "pred_fun": [],
                "pred_ours": [],
            },
        )
        slot["actual"].append(float(actual))
        slot["pred_fun"].append(float(pf))
        slot["pred_ours"].append(float(expected))

    out_rows: list[dict] = []
    for tid in sorted(by_tid, key=lambda t: (by_tid[t]["start_date"], t)):
        s = by_tid[tid]
        act = np.array(s["actual"])
        pf = np.array(s["pred_fun"])
        po = np.array(s["pred_ours"])
        n_q = int(s["n_questions"])
        ef, eo = np.abs(pf - act), np.abs(po - act)

        out_rows.append(
            {
                "tournament_id": tid,
                "title": s["title"],
                "start_date": s["start_date"],
                "n_questions": n_q,
                "n_teams": len(act),
                "mae_fun": round(float(ef.mean()), 4),
                "mae_ours": round(float(eo.mean()), 4),
                "rmse_fun": round(float(np.sqrt((ef**2).mean())), 4),
                "rmse_ours": round(float(np.sqrt((eo**2).mean())), 4),
                "bias_fun": round(float((pf - act).mean()), 4),
                "bias_ours": round(float((po - act).mean()), 4),
            }
        )

    if not out_rows:
        raise SystemExit("No tournaments matched filters / parquet overlap")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} tournaments → {args.out}")


if __name__ == "__main__":
    main()
