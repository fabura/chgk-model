#!/usr/bin/env python3
"""Find players/teams with repeated extreme positive forecast outliers.

Flags entities whose best games (top-K by actual − expected) cluster far
above typical performance — possible calibration artefacts or one-off luck.

Uses bake-time ``expected_takes`` from website DuckDB (read-only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "website" / "data" / "chgk.duckdb"


def _spike_stats(residuals: np.ndarray, top_k: int) -> dict:
    if residuals.size == 0:
        return {}
    s = np.sort(residuals)[::-1]
    k = min(top_k, s.size)
    top = s[:k]
    top5 = s[: min(5, s.size)]
    return {
        "n_games": int(s.size),
        "max_res": float(top[0]),
        "mean_top5": float(top5.mean()),
        f"mean_top{top_k}": float(top.mean()),
        f"sum_top{top_k}": float(top.sum()),
        "n_ge_8": int((s >= 8).sum()),
        "n_ge_10": int((s >= 10).sum()),
        "n_ge_12": int((s >= 12).sum()),
        "n_ge_15": int((s >= 15).sum()),
        "median_res": float(np.median(s)),
    }


def _observation_filters(
    *,
    team_only: bool,
    exclude_async: bool,
    exclude_marathons: bool,
    marathon_score_min: int,
) -> str:
    """SQL fragment (AND …) for team_games / player_games queries."""
    parts: list[str] = []
    if exclude_async:
        parts.append("t.type IN ('sync', 'offline')")
    if team_only:
        parts.append("tg.n_players_active > 1")
    if exclude_marathons:
        parts.append(
            "("
            "lower(t.title) NOT LIKE '%марафон%' "
            "AND lower(t.title) NOT LIKE '%marathon%' "
            f"AND coalesce(tg.score_actual, 0) < {int(marathon_score_min)}"
            ")"
        )
    return "".join(f"\n          AND {p}" for p in parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--min-player-games", type=int, default=40)
    ap.add_argument("--min-team-games", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--include-async",
        action="store_true",
        help="Include async tournaments (default: sync+offline only)",
    )
    ap.add_argument(
        "--include-solo",
        action="store_true",
        help="Include solo (1-player) team observations",
    )
    ap.add_argument(
        "--include-marathons",
        action="store_true",
        help="Include marathons (title or score_actual threshold)",
    )
    ap.add_argument(
        "--marathon-score-min",
        type=int,
        default=150,
        help="Also drop games with score_actual >= this when excluding marathons",
    )
    ap.add_argument(
        "--min-spikes-10",
        type=int,
        default=3,
        help="Flag if at least this many games with residual >= +10",
    )
    ap.add_argument("--players-out", type=Path, default=REPO_ROOT / "results" / "outlier_spike_players.csv")
    ap.add_argument("--teams-out", type=Path, default=REPO_ROOT / "results" / "outlier_spike_teams.csv")
    ap.add_argument("--top", type=int, default=40, help="Rows to print per table")
    args = ap.parse_args()

    if not args.db.is_file():
        raise SystemExit(f"DuckDB not found: {args.db}")

    con = duckdb.connect(str(args.db), read_only=True)
    top_k = int(args.top_k)
    filt = _observation_filters(
        team_only=not args.include_solo,
        exclude_async=not args.include_async,
        exclude_marathons=not args.include_marathons,
        marathon_score_min=args.marathon_score_min,
    )
    filter_note = []
    if not args.include_async:
        filter_note.append("sync+offline")
    if not args.include_solo:
        filter_note.append("без solo")
    if not args.include_marathons:
        filter_note.append(f"без марафонов (score<{args.marathon_score_min})")
    print("Фильтры:", ", ".join(filter_note) or "нет", "\n")

    # --- players ---
    pg = con.execute(
        f"""
        SELECT
            pg.player_id,
            p.last_name,
            p.first_name,
            p.theta,
            p.games AS career_games,
            pg.tournament_id,
            t.title,
            t.start_date,
            t.type,
            tg.team_name,
            tg.n_players_active,
            pg.n_takes_team AS actual,
            pg.expected_takes_team AS expected,
            pg.n_takes_team - pg.expected_takes_team AS res
        FROM player_games pg
        JOIN players p ON p.player_id = pg.player_id
        JOIN tournaments t ON t.tournament_id = pg.tournament_id
        JOIN team_games tg
          ON tg.team_id = pg.team_id AND tg.tournament_id = pg.tournament_id
        WHERE pg.expected_takes_team IS NOT NULL
        {filt}
        """
    ).fetchdf()

    player_rows = []
    for pid, grp in pg.groupby("player_id"):
        if len(grp) < args.min_player_games:
            continue
        res = grp["res"].to_numpy(dtype=np.float64)
        st = _spike_stats(res, top_k)
        if not st:
            continue
        ln = grp["last_name"].iloc[0] or ""
        fn = grp["first_name"].iloc[0] or ""
        player_rows.append(
            {
                "player_id": int(pid),
                "name": f"{ln} {fn}".strip(),
                "theta": float(grp["theta"].iloc[0]),
                "career_games": int(grp["career_games"].iloc[0]),
                **st,
            }
        )

    import pandas as pd

    pdf = pd.DataFrame(player_rows)
    if pdf.empty:
        print("No players matched filters.")
    else:
        # Strict: repeated +10 games; soft flag kept for CSV exploration only
        pdf["flag_strict"] = (pdf["n_ge_10"] >= 8) | (
            (pdf["n_ge_10"] >= 5) & (pdf["mean_top5"] >= 14.0)
        )
        pdf["flag"] = pdf["flag_strict"] | (
            (pdf["n_ge_10"] >= args.min_spikes_10) & (pdf["mean_top5"] >= 10.0)
        )
        pdf = pdf.sort_values(
            ["flag_strict", "n_ge_10", "mean_top5", "max_res"],
            ascending=[False, False, False, False],
        )
        args.players_out.parent.mkdir(parents=True, exist_ok=True)
        pdf.to_csv(args.players_out, index=False)

        print(f"=== Игроки: повторяющиеся выбросы (≥{args.min_player_games} игр) ===")
        print(
            f"    flag_strict: ≥8×res≥+10  ИЛИ  (≥5×res≥+10 и mean_top5≥14); "
            f"flag: ещё ≥{args.min_spikes_10}×res≥+10 и mean_top5≥10"
        )
        print(f"    CSV: {args.players_out}\n")
        cols = ["name", "theta", "n_games", "n_ge_10", "n_ge_12", "mean_top5", f"mean_top{top_k}", "max_res", "median_res"]
        flagged = pdf[pdf["flag_strict"]].head(args.top)
        print(
            f"--- Строгие ({len(pdf[pdf['flag_strict']])} всего), "
            f"топ-{min(args.top, len(flagged))} ---\n"
        )
        print(flagged[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}" if isinstance(x, float) else str(x)))

        print(f"\n--- Топ по mean_top{top_k} (все игроки) ---\n")
        print(pdf.head(args.top)[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}" if isinstance(x, float) else str(x)))

    # --- teams ---
    tg = con.execute(
        f"""
        SELECT
            tg.team_id,
            tg.team_name,
            tg.tournament_id,
            t.title,
            t.start_date,
            t.type,
            tg.n_players_active,
            tg.score_actual AS actual,
            tg.expected_takes AS expected,
            tg.score_actual - tg.expected_takes AS res,
            tg.place,
            t.n_teams
        FROM team_games tg
        JOIN tournaments t ON t.tournament_id = tg.tournament_id
        WHERE tg.expected_takes IS NOT NULL
        {filt}
        """
    ).fetchdf()

    team_rows = []
    for tid, grp in tg.groupby("team_id"):
        if len(grp) < args.min_team_games:
            continue
        res = grp["res"].to_numpy(dtype=np.float64)
        st = _spike_stats(res, top_k)
        if not st:
            continue
        team_rows.append(
            {
                "team_id": int(tid),
                "team_name": grp["team_name"].iloc[0] or "",
                **st,
            }
        )

    tdf = pd.DataFrame(team_rows)
    if not tdf.empty:
        tdf["flag"] = (tdf["n_ge_10"] >= max(2, args.min_spikes_10 - 1)) | (
            (tdf[f"mean_top{top_k}"] >= 9.0) & (tdf["n_ge_8"] >= 3)
        )
        tdf = tdf.sort_values(
            ["flag", "n_ge_10", f"mean_top{top_k}", "max_res"],
            ascending=[False, False, False, False],
        )
        tdf.to_csv(args.teams_out, index=False)

        print(f"\n=== Команды (≥{args.min_team_games} турниров) ===")
        print(f"    CSV: {args.teams_out}\n")
        tcols = ["team_name", "n_games", "n_ge_10", "n_ge_12", f"mean_top{top_k}", "max_res", "median_res"]
        print(tdf[tdf["flag"]].head(args.top)[tcols].to_string(index=False))

    # --- detail top spikes for a few flagged players ---
    if not pdf.empty:
        print("\n=== Примеры: топ-5 выбросов у самых «шипастых» игроков ===\n")
        for _, row in pdf[pdf["flag_strict"]].head(8).iterrows():
            pid = int(row["player_id"])
            sub = pg[pg["player_id"] == pid].nlargest(5, "res")
            print(f"{row['name']} (θ={row['theta']:+.2f}, n_ge_10={int(row['n_ge_10'])})")
            for _, g in sub.iterrows():
                print(
                    f"  {g['start_date']}  res={g['res']:+.0f}  {g['actual']:.0f}/{g['expected']:.0f}  "
                    f"[{g['type']}] {str(g['title'])[:50]}  ({g['team_name']})"
                )
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
