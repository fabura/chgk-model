#!/usr/bin/env python3
"""Per-team dump + factor breakdown for chgk.fun vs ours.

Pipeline:
  1. Pick tournaments from DuckDB (date window + min field size).
  2. Pull pre-tournament forecasts from rating.chgk.fun (cached on disk).
  3. Join with DuckDB ``team_games`` (our ``expected_takes`` uses the
     pre-tournament θ snapshot, same epoch as fun's ``predictedquestions``).
  4. Dump a per-team CSV with every factor we slice by.
  5. Print breakdowns: by tournament type, team size (especially solo),
     field size, pack difficulty, team strength quintile.

Run:
  .venv/bin/python scripts/compare_chgk_fun_detail.py
  .venv/bin/python scripts/compare_chgk_fun_detail.py --limit 600 --since 2022-01-01
  .venv/bin/python scripts/compare_chgk_fun_detail.py --dump-only   # re-print stats from cached per-team CSV

The full per-tournament JSON is cached under ``data/chgk_fun_cache/`` so
re-runs only hit network for genuinely new tournaments.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.compare_chgk_fun import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DUCKDB,
    FUN_MODEL_START,
    _fetch_fun_tournament,
)

# ---------------------------------------------------------------------------
# Per-team dump
# ---------------------------------------------------------------------------

PER_TEAM_FIELDS = [
    "tournament_id",
    "start_date",
    "type",
    "n_questions",
    "n_teams_field",
    "team_id",
    "n_players_active",
    "actual",
    "pred_fun",
    "pred_ours",
    "place",
]


def _gather_per_team(
    *,
    duckdb_path: Path,
    cache_dir: Path,
    since: str,
    until: str | None,
    limit: int,
    min_teams: int,
    sleep: float,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    import duckdb

    con = duckdb.connect(str(duckdb_path), read_only=True)
    sql = """
        SELECT t.tournament_id, t.start_date, t.type, t.n_questions,
               COUNT(*) AS n_teams_field
        FROM tournaments t
        JOIN team_games tg USING (tournament_id)
        WHERE t.start_date >= ?
          AND tg.expected_takes IS NOT NULL
          AND tg.score_actual IS NOT NULL
    """
    params: list = [since]
    if until:
        sql += " AND t.start_date <= ?"
        params.append(until)
    sql += """
        GROUP BY 1, 2, 3, 4
        HAVING n_teams_field >= ?
        ORDER BY t.start_date DESC
        LIMIT ?
    """
    params.extend([min_teams, limit])
    tourns = con.execute(sql, params).fetchall()

    if not tourns:
        con.close()
        raise SystemExit("No tournaments matched filters")

    rows: list[dict] = []
    stats = {"api_miss": 0, "no_overlap": 0, "kept": 0}

    for i, (tid, start_date, ttype, n_q, n_teams_field) in enumerate(tourns, 1):
        tid = int(tid)
        fun = _fetch_fun_tournament(tid, cache_dir=cache_dir)
        if fun is None:
            stats["api_miss"] += 1
        if i % 25 == 0 and verbose:
            print(
                f"  [{i}/{len(tourns)}]  kept={stats['kept']}  "
                f"miss={stats['api_miss']}  no_overlap={stats['no_overlap']}"
            )
        if not fun or not fun.get("tourresults"):
            time.sleep(sleep)
            continue
        time.sleep(sleep)  # be polite, regardless of cache hit

        fun_by_team = {int(t["teamid"]): t for t in fun["tourresults"]}
        ours = con.execute(
            """
            SELECT team_id, n_players_active, score_actual, expected_takes, place
            FROM team_games
            WHERE tournament_id = ?
              AND expected_takes IS NOT NULL
              AND score_actual IS NOT NULL
            """,
            [tid],
        ).fetchall()

        kept_local = 0
        for team_id, n_pl, actual, expected, place in ours:
            ft = fun_by_team.get(int(team_id))
            if ft is None:
                continue
            rows.append(
                {
                    "tournament_id": tid,
                    "start_date": str(start_date),
                    "type": ttype,
                    "n_questions": int(n_q) if n_q is not None else None,
                    "n_teams_field": int(n_teams_field),
                    "team_id": int(team_id),
                    "n_players_active": int(n_pl) if n_pl is not None else None,
                    "actual": float(actual),
                    "pred_fun": float(ft["predictedquestions"]),
                    "pred_ours": float(expected),
                    "place": float(place) if place is not None else None,
                }
            )
            kept_local += 1
        if kept_local == 0:
            stats["no_overlap"] += 1
        else:
            stats["kept"] += 1

    con.close()
    return rows, stats


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _bucket_summary(name: str, rows: list[dict]) -> dict:
    if not rows:
        return {
            "bucket": name,
            "n_teams": 0,
            "n_tourn": 0,
            "mae_fun": float("nan"),
            "mae_ours": float("nan"),
            "rmse_fun": float("nan"),
            "rmse_ours": float("nan"),
            "bias_fun": float("nan"),
            "bias_ours": float("nan"),
            "mae_norm_fun": float("nan"),
            "mae_norm_ours": float("nan"),
            "ours_team_wins": 0,
            "fun_team_wins": 0,
        }
    act = np.array([r["actual"] for r in rows])
    pf = np.array([r["pred_fun"] for r in rows])
    po = np.array([r["pred_ours"] for r in rows])
    nq = np.array([r["n_questions"] or 0 for r in rows], dtype=float)
    nq_safe = np.where(nq > 0, nq, 1.0)
    err_f = np.abs(pf - act)
    err_o = np.abs(po - act)
    return {
        "bucket": name,
        "n_teams": len(rows),
        "n_tourn": len({r["tournament_id"] for r in rows}),
        "mae_fun": float(err_f.mean()),
        "mae_ours": float(err_o.mean()),
        "rmse_fun": float(np.sqrt((err_f**2).mean())),
        "rmse_ours": float(np.sqrt((err_o**2).mean())),
        "bias_fun": float((pf - act).mean()),
        "bias_ours": float((po - act).mean()),
        "mae_norm_fun": float((err_f / nq_safe).mean()),
        "mae_norm_ours": float((err_o / nq_safe).mean()),
        "ours_team_wins": int((err_o < err_f).sum()),
        "fun_team_wins": int((err_f < err_o).sum()),
    }


def _print_table(rows: list[dict]) -> None:
    """Compact, fixed-width breakdown table."""
    if not rows:
        print("  (empty)")
        return
    cols = [
        ("bucket", "bucket", 24),
        ("n_teams", "n_teams", 8),
        ("n_tourn", "n_tourn", 7),
        ("mae_fun", "MAE_fun", 8),
        ("mae_ours", "MAE_ours", 9),
        ("rmse_fun", "RMSE_fun", 9),
        ("rmse_ours", "RMSE_our", 9),
        ("bias_fun", "bias_fun", 9),
        ("bias_ours", "bias_our", 9),
        ("mae_norm_fun", "MAE/Q_fun", 10),
        ("mae_norm_ours", "MAE/Q_our", 10),
        ("ours_team_wins", "ours_win", 9),
        ("fun_team_wins", "fun_win", 9),
    ]
    header = "  ".join(f"{lab:>{w}}" for _, lab, w in cols)
    print("  " + header)
    print("  " + "-" * len(header))
    for r in rows:
        cells = []
        for key, _, w in cols:
            v = r.get(key)
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append(f"{'—':>{w}}")
                else:
                    cells.append(f"{v:>{w}.3f}")
            else:
                cells.append(f"{v!s:>{w}}")
        print("  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------


def _size_bucket(n: int) -> str:
    if n <= 1:
        return "1 (solo)"
    if n <= 3:
        return "2–3"
    if n == 6:
        return "6 (full)"
    if n <= 5:
        return "4–5"
    return "7+"


def _field_bucket(n: int) -> str:
    if n < 20:
        return "field <20"
    if n < 50:
        return "field 20–49"
    if n < 100:
        return "field 50–99"
    return "field 100+"


def _take_rate_bucket(rows: list[dict]) -> dict[int, str]:
    """Per-tournament: bucket by mean actual take rate (pack difficulty)."""
    by_tid: dict[int, list[dict]] = {}
    for r in rows:
        by_tid.setdefault(r["tournament_id"], []).append(r)
    mean_rate = {}
    for tid, rs in by_tid.items():
        nq = rs[0]["n_questions"] or 1
        mean_rate[tid] = float(np.mean([r["actual"] for r in rs])) / max(nq, 1)
    if not mean_rate:
        return {}
    vals = np.array(list(mean_rate.values()))
    q1, q2, q3 = np.quantile(vals, [0.25, 0.5, 0.75])

    def label(v: float) -> str:
        if v < q1:
            return f"pack hard (<{q1:.2f})"
        if v < q2:
            return f"pack med-hard"
        if v < q3:
            return f"pack med-easy"
        return f"pack easy (>{q3:.2f})"

    return {tid: label(r) for tid, r in mean_rate.items()}


def _per_tournament_team_quintile(rows: list[dict]) -> dict[tuple[int, int], int]:
    """Within each tournament, assign teams to actual-take quintiles (1..5)."""
    by_tid: dict[int, list[dict]] = {}
    for r in rows:
        by_tid.setdefault(r["tournament_id"], []).append(r)
    out: dict[tuple[int, int], int] = {}
    for tid, rs in by_tid.items():
        if len(rs) < 5:
            # Too few teams for a meaningful quintile split.
            for r in rs:
                out[(tid, r["team_id"])] = 0
            continue
        actuals = np.array([r["actual"] for r in rs])
        # quintile labels 1..5 by actual takes (1 = weakest)
        ranks = actuals.argsort().argsort()
        q = (ranks * 5) // len(rs) + 1
        for r, qi in zip(rs, q):
            out[(tid, r["team_id"])] = int(qi)
    return out


def _print_breakdowns(rows: list[dict]) -> None:
    print()
    print(f"=== Overall ({len(rows):,} teams, "
          f"{len({r['tournament_id'] for r in rows})} tournaments) ===")
    _print_table([_bucket_summary("all", rows)])

    # By type
    print("\n=== By tournament type ===")
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"] or "?", []).append(r)
    rows_table = [
        _bucket_summary(t, rs)
        for t, rs in sorted(by_type.items(), key=lambda x: -len(x[1]))
    ]
    _print_table(rows_table)

    # By team size
    print("\n=== By team size (n_players_active) ===")
    size_order = ["1 (solo)", "2–3", "4–5", "6 (full)", "7+"]
    by_size: dict[str, list[dict]] = {k: [] for k in size_order}
    for r in rows:
        if r["n_players_active"] is None:
            continue
        by_size[_size_bucket(int(r["n_players_active"]))].append(r)
    _print_table([_bucket_summary(k, by_size[k]) for k in size_order])

    # Solo × type cross-tab (the headline question)
    print("\n=== Solo (n=1) split by tournament type ===")
    solo_rows = by_size["1 (solo)"]
    sub: dict[str, list[dict]] = {}
    for r in solo_rows:
        sub.setdefault(r["type"] or "?", []).append(r)
    _print_table(
        [
            _bucket_summary(f"solo · {t}", rs)
            for t, rs in sorted(sub.items(), key=lambda x: -len(x[1]))
        ]
    )

    # By field size
    print("\n=== By field size ===")
    by_field: dict[str, list[dict]] = {}
    for r in rows:
        by_field.setdefault(_field_bucket(int(r["n_teams_field"])), []).append(r)
    field_order = ["field <20", "field 20–49", "field 50–99", "field 100+"]
    _print_table([_bucket_summary(k, by_field.get(k, [])) for k in field_order])

    # By pack difficulty (per-tournament quartile of mean take rate)
    print("\n=== By pack difficulty (per-tournament mean take rate quartile) ===")
    pack_label = _take_rate_bucket(rows)
    by_pack: dict[str, list[dict]] = {}
    for r in rows:
        lab = pack_label.get(r["tournament_id"])
        if lab:
            by_pack.setdefault(lab, []).append(r)
    for k in sorted(by_pack):
        pass  # ensure deterministic
    _print_table([_bucket_summary(k, by_pack[k]) for k in sorted(by_pack)])

    # By team strength quintile (within each tournament)
    print("\n=== By team strength quintile (per-tournament actual-takes rank) ===")
    qmap = _per_tournament_team_quintile(rows)
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        q = qmap.get((r["tournament_id"], r["team_id"]), 0)
        if q == 0:
            continue
        label = {1: "Q1 (weakest)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (strongest)"}[q]
        by_q.setdefault(label, []).append(r)
    _print_table(
        [
            _bucket_summary(k, by_q.get(k, []))
            for k in ["Q1 (weakest)", "Q2", "Q3", "Q4", "Q5 (strongest)"]
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_dump(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            row["n_questions"] = int(row["n_questions"]) if row["n_questions"] else None
            row["n_teams_field"] = int(row["n_teams_field"])
            row["team_id"] = int(row["team_id"])
            row["n_players_active"] = (
                int(row["n_players_active"]) if row["n_players_active"] else None
            )
            row["actual"] = float(row["actual"])
            row["pred_fun"] = float(row["pred_fun"])
            row["pred_ours"] = float(row["pred_ours"])
            row["place"] = float(row["place"]) if row["place"] else None
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--since", default=FUN_MODEL_START)
    ap.add_argument("--until", default=None)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--min-teams", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--dump", type=Path, default=_REPO / "results" / "compare_chgk_fun_per_team.csv")
    ap.add_argument(
        "--dump-only", action="store_true",
        help="skip fetching; re-print stats from the existing --dump CSV",
    )
    args = ap.parse_args()

    if args.dump_only:
        if not args.dump.is_file():
            raise SystemExit(f"--dump not found: {args.dump}")
        rows = _load_dump(args.dump)
        print(f"Loaded {len(rows):,} per-team rows from {args.dump}")
    else:
        if not args.duckdb.is_file():
            raise SystemExit(f"DuckDB not found: {args.duckdb}")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        rows, stats = _gather_per_team(
            duckdb_path=args.duckdb,
            cache_dir=args.cache_dir,
            since=args.since,
            until=args.until,
            limit=args.limit,
            min_teams=args.min_teams,
            sleep=args.sleep,
        )
        print(f"\nKept {stats['kept']} tournaments, "
              f"{stats['api_miss']} api_miss, {stats['no_overlap']} no_overlap")
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PER_TEAM_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote per-team dump: {args.dump}  ({len(rows):,} rows)")

    _print_breakdowns(rows)


if __name__ == "__main__":
    main()
