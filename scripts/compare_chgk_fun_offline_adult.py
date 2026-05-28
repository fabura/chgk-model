#!/usr/bin/env python3
"""Compare ours vs rating.chgk.fun on top adult-offline tournaments.

Filters:
  * ``type='offline'`` per DuckDB (source-of-truth from rating.chgk.info)
  * exclude title prefix ``Онлайн:`` — М-Лига etc. are flagged as offline
    by the rating policy but played online; doesn't match user intent
  * exclude school: ``школьн``, ``ШЧР``, ``юношеск``, ``юниор``, ``детск``,
    ``малыш``, ``первенство сибири`` (юношеский трофей), ``воронёнок``
  * exclude student tournaments EXCEPT
    ``Студенческий чемпионат России / Беларуси``
  * exclude obviously mixed ones (``школьники и студенты``)
  * take the top ``--limit`` by ``n_teams`` (largest fields = most signal)

Output: per-team and aggregated tables, plus a per-tournament CSV.
"""
from __future__ import annotations

import argparse
import csv
import re
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
    _fetch_fun_tournament,
)
from scripts.compare_chgk_fun_detail import _bucket_summary, _print_table  # noqa: E402

# Match anywhere in the title (case-insensitive). Patterns are conservative —
# false-positive exclusions are preferred over including school/junior packs.
EXCLUDE_SCHOOL_RE = re.compile(
    r"(онлайн:|школьн|шчр|юношеск|юниор|детск(?:ий|ая|ое|ие)|малыш"
    r"|школьники и студенты|воронёнок|воронен[ок]"
    r"|первенство сибири|первенство дв|первенство урал"
    r"|кубок дв v\b|олимпиад\w* среди школ)",
    re.IGNORECASE,
)

# Pattern for student tournaments we *do* keep (the explicit student
# championships of Russia and Belarus).
KEEP_STUDENT_RE = re.compile(
    r"студенческ\w*\s+(чемпионат|кубок)\s+(росси|беларус|белорус)",
    re.IGNORECASE,
)
# Pattern for any other student tournament we drop.
STUDENT_GENERIC_RE = re.compile(r"студент|студенч", re.IGNORECASE)


def _passes_filter(title: str) -> tuple[bool, str]:
    """Return (keep, reason)."""
    t = title or ""
    if EXCLUDE_SCHOOL_RE.search(t):
        return False, f"school: matched {EXCLUDE_SCHOOL_RE.search(t).group(0)!r}"
    if STUDENT_GENERIC_RE.search(t):
        if KEEP_STUDENT_RE.search(t):
            return True, "student-championship-RU/BY"
        return False, "student (not Russia/Belarus championship)"
    return True, "adult-offline"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--since", default="2022-01-01")
    ap.add_argument("--until", default=None)
    ap.add_argument("--limit", type=int, default=150,
                    help="top-N largest qualifying tournaments")
    ap.add_argument("--min-teams", type=int, default=24,
                    help="ignore very small fields")
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--out-tourn", type=Path,
                    default=_REPO / "results" / "compare_chgk_fun_offline_adult.csv")
    ap.add_argument("--out-team", type=Path,
                    default=_REPO / "results" / "compare_chgk_fun_offline_adult_team.csv")
    ap.add_argument("--show-excluded", action="store_true",
                    help="print first 30 excluded tournaments with reasons")
    args = ap.parse_args()

    import duckdb

    con = duckdb.connect(str(args.duckdb), read_only=True)
    sql = """
        SELECT t.tournament_id, t.title, t.start_date, t.n_questions,
               COUNT(*) AS n_teams_field
        FROM tournaments t
        JOIN team_games tg USING (tournament_id)
        WHERE t.type='offline'
          AND t.start_date >= ?
          AND tg.expected_takes IS NOT NULL
          AND tg.score_actual IS NOT NULL
    """
    params: list = [args.since]
    if args.until:
        sql += " AND t.start_date <= ?"
        params.append(args.until)
    sql += """
        GROUP BY 1, 2, 3, 4
        HAVING n_teams_field >= ?
        ORDER BY n_teams_field DESC
    """
    params.append(args.min_teams)
    all_offline = con.execute(sql, params).fetchall()

    kept, excluded = [], []
    for row in all_offline:
        ok, reason = _passes_filter(row[1] or "")
        (kept if ok else excluded).append((row, reason))

    kept = kept[: args.limit]
    print(
        f"Candidates: {len(all_offline)} offline tournaments since {args.since}; "
        f"after filter: {len(kept) + max(0, len([r for r in all_offline if _passes_filter(r[1] or '')[0]]) - len(kept))} "
        f"({len([r for r in all_offline if _passes_filter(r[1] or '')[0]])} pass filter); "
        f"taking top {len(kept)} by n_teams (min_teams={args.min_teams})"
    )

    if args.show_excluded:
        print("\n--- first 30 excluded ---")
        for (row, reason) in excluded[:30]:
            print(f"  {row[4]:>3}  {row[2]}  #{row[0]:<6}  {row[1][:80]:<82}  [{reason}]")
        print("--- end excluded ---")

    print(f"\nKept top {len(kept)} adult-offline tournaments. Examples:")
    for (row, _) in kept[:10]:
        print(f"  {row[4]:>3}  {row[2]}  #{row[0]:<6}  {row[1][:80]}")

    # ---- fetch & compare ----
    team_rows: list[dict] = []
    tourn_rows: list[dict] = []

    for i, ((tid, title, start_date, n_q, n_teams_field), _) in enumerate(kept, 1):
        tid = int(tid)
        fun = _fetch_fun_tournament(tid, cache_dir=args.cache_dir)
        if not fun or not fun.get("tourresults"):
            time.sleep(args.sleep)
            continue
        time.sleep(args.sleep)

        fun_by_team = {int(t["teamid"]): t for t in fun["tourresults"]}
        ours = con.execute(
            """
            SELECT team_id, n_players_active, score_actual, expected_takes, place
            FROM team_games WHERE tournament_id = ?
              AND expected_takes IS NOT NULL AND score_actual IS NOT NULL
            """,
            [tid],
        ).fetchall()

        rows_local: list[dict] = []
        for team_id, n_pl, actual, expected, place in ours:
            ft = fun_by_team.get(int(team_id))
            if ft is None:
                continue
            rows_local.append({
                "tournament_id": tid,
                "start_date": str(start_date),
                "title": title,
                "type": "offline",
                "n_questions": int(n_q) if n_q else None,
                "n_teams_field": int(n_teams_field),
                "team_id": int(team_id),
                "n_players_active": int(n_pl) if n_pl else None,
                "actual": float(actual),
                "pred_fun": float(ft["predictedquestions"]),
                "pred_ours": float(expected),
                "place": float(place) if place is not None else None,
            })
        if not rows_local:
            continue
        team_rows.extend(rows_local)

        act = np.array([r["actual"] for r in rows_local])
        pf = np.array([r["pred_fun"] for r in rows_local])
        po = np.array([r["pred_ours"] for r in rows_local])
        err_f, err_o = np.abs(pf - act), np.abs(po - act)

        # Place-prediction Spearman: rank by predicted takes vs actual place.
        places = np.array([r["place"] for r in rows_local if r["place"] is not None])
        sp_f = sp_o = float("nan")
        if len(places) == len(rows_local) and len(places) >= 5:
            rank_o = (-po).argsort().argsort() + 1
            rank_f = (-pf).argsort().argsort() + 1
            try:
                from scipy.stats import spearmanr
                sp_f = float(spearmanr(rank_f, places).correlation)
                sp_o = float(spearmanr(rank_o, places).correlation)
            except ImportError:
                pass

        tourn_rows.append({
            "tournament_id": tid,
            "start_date": str(start_date),
            "title": title,
            "n_teams": len(rows_local),
            "mae_fun": float(err_f.mean()),
            "mae_ours": float(err_o.mean()),
            "rmse_fun": float(np.sqrt((err_f**2).mean())),
            "rmse_ours": float(np.sqrt((err_o**2).mean())),
            "bias_fun": float((pf - act).mean()),
            "bias_ours": float((po - act).mean()),
            "spearman_fun": sp_f,
            "spearman_ours": sp_o,
        })
        if i % 25 == 0:
            print(f"  [{i}/{len(kept)}]  ours_mae_running="
                  f"{np.mean([t['mae_ours'] for t in tourn_rows]):.3f}")

    con.close()

    # ---- write CSVs ----
    args.out_tourn.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tourn.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tourn_rows[0].keys()))
        w.writeheader()
        w.writerows(tourn_rows)
    with args.out_team.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(team_rows[0].keys()))
        w.writeheader()
        w.writerows(team_rows)

    # ---- aggregate stats ----
    print(f"\n=== Aggregate over {len(tourn_rows)} adult-offline tournaments, "
          f"{len(team_rows):,} teams ===")
    _print_table([_bucket_summary("all (adult offline)", team_rows)])

    # By team size on this slice
    print("\n=== Adult offline · by team size ===")
    by_size: dict[str, list[dict]] = {}
    for r in team_rows:
        n = r["n_players_active"]
        if n is None:
            continue
        if n == 1:
            k = "1 (solo)"
        elif n <= 3:
            k = "2–3"
        elif n <= 5:
            k = "4–5"
        elif n == 6:
            k = "6 (full)"
        else:
            k = "7+"
        by_size.setdefault(k, []).append(r)
    _print_table([
        _bucket_summary(k, by_size.get(k, []))
        for k in ["1 (solo)", "2–3", "4–5", "6 (full)", "7+"]
    ])

    # By field size on this slice
    print("\n=== Adult offline · by field size ===")
    by_field: dict[str, list[dict]] = {}
    for r in team_rows:
        n = r["n_teams_field"]
        k = "<30" if n < 30 else "30–49" if n < 50 else "50–74" if n < 75 else "75+"
        by_field.setdefault(k, []).append(r)
    _print_table([
        _bucket_summary(k, by_field.get(k, []))
        for k in ["<30", "30–49", "50–74", "75+"]
    ])

    # By team strength quintile
    print("\n=== Adult offline · by team strength quintile (actual takes rank) ===")
    by_tid: dict[int, list[dict]] = {}
    for r in team_rows:
        by_tid.setdefault(r["tournament_id"], []).append(r)
    qmap = {}
    for tid, rs in by_tid.items():
        if len(rs) < 5:
            continue
        actuals = np.array([r["actual"] for r in rs])
        ranks = actuals.argsort().argsort()
        q = (ranks * 5) // len(rs) + 1
        for r, qi in zip(rs, q):
            qmap[(tid, r["team_id"])] = int(qi)
    by_q: dict[str, list[dict]] = {}
    for r in team_rows:
        q = qmap.get((r["tournament_id"], r["team_id"]))
        if q is None:
            continue
        label = {1: "Q1 (weakest)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (strongest)"}[q]
        by_q.setdefault(label, []).append(r)
    _print_table([
        _bucket_summary(k, by_q.get(k, []))
        for k in ["Q1 (weakest)", "Q2", "Q3", "Q4", "Q5 (strongest)"]
    ])

    # Place ranking
    sp_f = np.array([t["spearman_fun"] for t in tourn_rows])
    sp_o = np.array([t["spearman_ours"] for t in tourn_rows])
    print("\n=== Adult offline · place ranking (mean tournament Spearman) ===")
    print(f"  fun  mean ρ = {np.nanmean(sp_f):.4f}")
    print(f"  ours mean ρ = {np.nanmean(sp_o):.4f}")
    print(f"  tournaments where ours ρ > fun ρ: "
          f"{(sp_o > sp_f).sum()} / {len(tourn_rows)}")

    # Tournament-MAE summary + worst outliers
    mae_o = np.array([t["mae_ours"] for t in tourn_rows])
    mae_f = np.array([t["mae_fun"] for t in tourn_rows])
    print("\n=== Adult offline · per-tournament MAE summary ===")
    print(f"  mean MAE   fun  {mae_f.mean():.3f}   ours {mae_o.mean():.3f}")
    print(f"  median MAE fun  {np.median(mae_f):.3f}   ours {np.median(mae_o):.3f}")
    print(f"  tournaments where ours beats fun: "
          f"{(mae_o < mae_f).sum()} / {len(tourn_rows)}")

    # Smaller (mae_ours − mae_fun) ⇒ ours wins by more.
    tourn_rows.sort(key=lambda t: t["mae_ours"] - t["mae_fun"])
    print("\n  TOP 5 where OURS beats fun by MAE (negative Δ = ours wins):")
    for t in tourn_rows[:5]:
        print(f"    Δ={(t['mae_ours']-t['mae_fun']):+.2f}  {t['start_date']}  "
              f"#{t['tournament_id']:<6}  {t['title'][:70]}")
    print("\n  TOP 5 where FUN beats ours by MAE (positive Δ = fun wins):")
    for t in tourn_rows[-5:]:
        print(f"    Δ={(t['mae_ours']-t['mae_fun']):+.2f}  {t['start_date']}  "
              f"#{t['tournament_id']:<6}  {t['title'][:70]}")

    print(f"\nWrote per-tournament: {args.out_tourn}")
    print(f"Wrote per-team:       {args.out_team}")


if __name__ == "__main__":
    main()
