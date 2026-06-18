#!/usr/bin/env python3
"""Forecast bias diagnostic for a tournament (DuckDB bake vs hypotheses).

Usage:
    python scripts/forecast_diagnostic.py --tournament-id 12826
    python scripts/forecast_diagnostic.py --tournament-id 12826 --compare 11749
    python scripts/forecast_diagnostic.py --tournament-id 12826 --baseline-offline
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rating.io import load_results_npz
from rating.pack_calib import (
    PACK_B_GAP_THRESHOLD,
    init_b_from_take_rate,
    pack_adjust_b,
    pack_b_gap,
    should_use_pack_adj_retrospective,
)
from rating.simulate import simulate_roster_on_pack
from scripts.analyse_chr import (
    DEFAULT_DUCKDB,
    DEFAULT_SEQ,
    _fetch_teams,
    _spearman,
)

DEFAULT_KVRM_XLSX = REPO_ROOT / "data/КВРМ.xlsx"


def _quartile_stats(rows: list[dict]) -> list[dict]:
    by_place = sorted(rows, key=lambda r: r["place"])
    n = len(by_place)
    q_size = max(1, n // 4)
    out = []
    for i, label in enumerate(["Q1 (top)", "Q2", "Q3", "Q4 (bottom)"]):
        chunk = by_place[i * q_size : (i + 1) * q_size if i < 3 else n]
        if not chunk:
            continue
        deltas = [r["delta"] for r in chunk]
        out.append(
            {
                "quartile": label,
                "n": len(chunk),
                "mean_delta": round(statistics.mean(deltas), 2),
                "median_delta": round(statistics.median(deltas), 2),
                "mean_actual": round(statistics.mean([r["actual"] for r in chunk]), 1),
                "mean_expected": round(statistics.mean([r["expected"] for r in chunk]), 1),
                "place_range": f"{chunk[0]['place']:.0f}–{chunk[-1]['place']:.0f}",
            }
        )
    return out


def _load_team_games(con: duckdb.DuckDBPyConnection, tid: int) -> list[dict]:
    rows = con.execute(
        """
        SELECT team_id, team_name, place, score_actual,
               expected_takes, team_theta_implied
        FROM team_games
        WHERE tournament_id = ? AND place IS NOT NULL
        ORDER BY place
        """,
        [tid],
    ).fetchall()
    return [
        {
            "team_id": int(r[0]),
            "team_name": str(r[1]),
            "place": float(r[2]),
            "actual": int(r[3]),
            "expected": float(r[4]),
            "delta": int(r[3]) - float(r[4]),
            "theta_implied": float(r[5]) if r[5] is not None else float("nan"),
        }
        for r in rows
    ]


def _load_questions(con: duckdb.DuckDBPyConnection, tid: int) -> list[dict]:
    rows = con.execute(
        """
        SELECT qa.q_in_tournament, q.b, q.a, qa.n_taken, qa.n_obs
        FROM question_aliases qa
        JOIN questions q ON q.canonical_idx = qa.canonical_idx
        WHERE qa.tournament_id = ?
        ORDER BY qa.q_in_tournament
        """,
        [tid],
    ).fetchall()
    return [
        {
            "q_index": int(r[0]),
            "b": float(r[1]),
            "a": float(r[2]),
            "n_taken": int(r[3] or 0),
            "n_teams": int(r[4] or 0),
            "take_rate": (int(r[3] or 0) / int(r[4])) if r[4] else 0.0,
        }
        for r in rows
    ]


def _theta_before_map(
    con: duckdb.DuckDBPyConnection,
    tid: int,
    pids: list[int],
    *,
    cold_init: float,
) -> dict[int, float]:
    gidx = con.execute(
        "SELECT game_idx FROM tournaments WHERE tournament_id = ?", [tid]
    ).fetchone()[0]
    hist = con.execute(
        """
        SELECT ph.player_id, ph.theta, t.game_idx
        FROM player_history ph
        JOIN tournaments t ON t.tournament_id = ph.tournament_id
        WHERE ph.player_id IN (SELECT UNNEST(?))
        ORDER BY ph.player_id, t.game_idx
        """,
        [pids],
    ).fetchall()
    by_player: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for pid, th, gi in hist:
        by_player[int(pid)].append((int(gi), float(th)))
    out: dict[int, float] = {}
    for pid in pids:
        prev = cold_init
        for gi, th in by_player.get(pid, []):
            if gi >= gidx:
                break
            prev = th
        out[pid] = prev
    return out


def _team_strength(pids: list[int], pmap: dict[int, float]) -> float:
    vals = sorted((pmap.get(p, -1.0) for p in pids), reverse=True)
    return float(sum(vals[:6])) if vals else 0.0


def _mean_expected(
    teams,
    questions: list[dict],
    pmap: dict[int, float],
    res,
    *,
    b_override: np.ndarray | None = None,
    lapse_arr=None,
    recal_arr=None,
) -> tuple[float, list[float], np.ndarray]:
    b = b_override if b_override is not None else np.array(
        [q["b"] for q in questions], dtype=np.float64
    )
    a = np.array([q["a"] for q in questions], dtype=np.float64)
    qi = np.array([q["q_index"] for q in questions], dtype=np.int64)
    exps: list[float] = []
    sum_q = np.zeros(len(questions), dtype=np.float64)
    for team in teams:
        thetas = np.array([pmap.get(p, -1.0) for p in team.player_ids], dtype=np.float64)
        p_q = simulate_roster_on_pack(
            thetas,
            b,
            a,
            q_in_tour=qi,
            delta_size=res.delta_size,
            team_size_anchor=res.team_size_anchor,
            delta_pos=res.delta_pos,
            pos_anchor=res.pos_anchor,
            mode="offline",
            lapse_arr=lapse_arr if lapse_arr is not None else res.lapse,
            recal_arr=recal_arr if recal_arr is not None else res.recal,
        )
        exps.append(float(p_q.sum()))
        sum_q += p_q
    return float(statistics.mean(exps)), exps, sum_q


def _init_b_pack(
    questions: list[dict],
    teams,
    theta_before: dict[int, float],
) -> np.ndarray:
    out = []
    for q in questions:
        qi = q["q_index"]
        take_rate = q["n_taken"] / q["n_teams"] if q["n_teams"] else 0.0
        taken_means = []
        for t in teams:
            if len(t.mask) <= qi or t.mask[qi] != "1" or not t.player_ids:
                continue
            taken_means.append(
                statistics.mean([theta_before.get(p, -1.0) for p in t.player_ids])
            )
        theta_bar = statistics.mean(taken_means) if taken_means else 0.0
        out.append(
            init_b_from_take_rate(take_rate, team_size_avg=6.0, theta_bar=theta_bar)
        )
    return np.array(out, dtype=np.float64)


def _team_expected_rows(
    teams,
    questions: list[dict],
    tg: list[dict],
    pmap: dict[int, float],
    res,
    *,
    b_override: np.ndarray | None = None,
) -> list[dict]:
    b = (
        b_override
        if b_override is not None
        else np.array([q["b"] for q in questions], dtype=np.float64)
    )
    a = np.array([q["a"] for q in questions], dtype=np.float64)
    qi = np.array([q["q_index"] for q in questions], dtype=np.int64)
    team_by_id = {t.team_id: t for t in teams}
    rows: list[dict] = []
    for row in tg:
        team = team_by_id.get(row["team_id"])
        if team is None or not team.player_ids:
            continue
        thetas = np.array(
            [pmap.get(p, -1.0) for p in team.player_ids], dtype=np.float64
        )
        p_q = simulate_roster_on_pack(
            thetas,
            b,
            a,
            q_in_tour=qi,
            delta_size=res.delta_size,
            team_size_anchor=res.team_size_anchor,
            delta_pos=res.delta_pos,
            pos_anchor=res.pos_anchor,
            mode="offline",
            lapse_arr=res.lapse,
            recal_arr=res.recal,
        )
        exp = float(p_q.sum())
        rows.append(
            {
                "place": row["place"],
                "actual": row["actual"],
                "expected": exp,
                "delta": row["actual"] - exp,
            }
        )
    return rows


def _fix_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"mean_delta": None, "by_quartile": []}
    deltas = [r["delta"] for r in rows]
    return {
        "mean_delta": round(statistics.mean(deltas), 2),
        "median_delta": round(statistics.median(deltas), 2),
        "by_quartile": _quartile_stats(rows),
    }


def _offline_baseline(con: duckdb.DuckDBPyConnection, exclude: set[int]) -> dict:
    rows = con.execute(
        """
        SELECT tg.tournament_id, tg.score_actual, tg.expected_takes
        FROM team_games tg
        JOIN tournaments t ON t.tournament_id = tg.tournament_id
        WHERE t.type = 'offline'
          AND tg.place IS NOT NULL
          AND tg.expected_takes IS NOT NULL
          AND tg.n_players_active >= 4
        """
    ).fetchall()
    by_tid: dict[int, list[float]] = defaultdict(list)
    for tid, actual, expected in rows:
        if int(tid) in exclude:
            continue
        by_tid[int(tid)].append(float(actual) - float(expected))
    all_d = [d for v in by_tid.values() for d in v]
    tour_means = [statistics.mean(v) for v in by_tid.values()]
    return {
        "n_tournaments": len(by_tid),
        "n_teams": len(all_d),
        "mean_delta_per_team": round(statistics.mean(all_d), 2),
        "median_delta_per_team": round(statistics.median(all_d), 2),
        "median_delta_per_tournament": round(statistics.median(tour_means), 2),
        "p10_tournament_mean": round(float(np.quantile(tour_means, 0.10)), 2),
        "p90_tournament_mean": round(float(np.quantile(tour_means, 0.90)), 2),
    }


def diagnose(
    tournament_id: int,
    *,
    duckdb_path: Path,
    seq_path: Path,
    xlsx_path: Path | None,
    baseline_offline: bool,
) -> dict:
    con = duckdb.connect(str(duckdb_path), read_only=True)
    res = load_results_npz(seq_path)
    cold_init = float(getattr(res, "cold_init_theta", -1.0))

    teams, _ = _fetch_teams(tournament_id, xlsx_path=xlsx_path)
    team_by_id = {t.team_id: t for t in teams}
    tg = _load_team_games(con, tournament_id)
    questions = _load_questions(con, tournament_id)
    if not tg or not questions:
        raise SystemExit(f"Missing team_games or questions for #{tournament_id}")

    all_pids = sorted({p for t in teams for p in t.player_ids})
    theta_before = _theta_before_map(con, tournament_id, all_pids, cold_init=cold_init)
    pmap_display = {
        int(r[0]): float(r[1])
        for r in con.execute(
            f"SELECT player_id, theta_display FROM players WHERE player_id IN ({','.join('?'*len(all_pids))})",
            all_pids,
        ).fetchall()
        if r[1] is not None
    }

    mean_actual = statistics.mean([r["actual"] for r in tg])
    mean_expected = statistics.mean([r["expected"] for r in tg])
    n_q = len(questions)
    mean_take_rate = mean_actual / n_q
    mean_pred_rate = mean_expected / n_q

    strengths = []
    deltas = []
    for row in tg:
        team = team_by_id.get(row["team_id"])
        if not team:
            continue
        strengths.append(_team_strength(team.player_ids, theta_before))
        deltas.append(row["delta"])

    exp_before, _, sum_q = _mean_expected(teams, questions, theta_before, res)
    exp_display, _, _ = _mean_expected(teams, questions, pmap_display, res)
    b_init = _init_b_pack(questions, teams, theta_before)
    exp_binit, _, _ = _mean_expected(teams, questions, theta_before, res, b_override=b_init)
    exp_nocal, _, _ = _mean_expected(
        teams, questions, theta_before, res, lapse_arr=None, recal_arr=None
    )
    b_tr = np.array([q["b"] for q in questions], dtype=np.float64)
    b_adj, b_gap, pack_adj_on = pack_adjust_b(b_tr, b_init)
    exp_pack_adj, _, _ = _mean_expected(
        teams, questions, theta_before, res, b_override=b_adj
    )
    b_shift = b_tr + b_gap
    exp_shift, _, _ = _mean_expected(
        teams, questions, theta_before, res, b_override=b_shift
    )
    trained_rows = _team_expected_rows(teams, questions, tg, theta_before, res)
    oracle_rows = _team_expected_rows(
        teams, questions, tg, theta_before, res, b_override=b_init
    )
    pack_adj_rows = _team_expected_rows(
        teams, questions, tg, theta_before, res, b_override=b_adj
    )
    shift_rows = _team_expected_rows(
        teams, questions, tg, theta_before, res, b_override=b_shift
    )
    retro_on = should_use_pack_adj_retrospective(
        mean_expected_trained=mean_expected,
        mean_delta_trained=mean_actual - mean_expected,
        b_gap=b_gap,
    )

    actual_q = np.array([q["n_taken"] for q in questions], dtype=float)
    qdelta = actual_q - sum_q
    worst_q = sorted(
        [
            {
                "q": q["q_index"] + 1,
                "delta": round(float(actual_q[i] - sum_q[i]), 1),
                "b": round(q["b"], 2),
                "taken": f"{q['n_taken']}/{q['n_teams']}",
            }
            for i, q in enumerate(questions)
        ],
        key=lambda x: x["delta"],
    )[:8]

    tour_stats = []
    if n_q == 90:
        for tour in range(6):
            chunk = [q for q in questions if tour * 15 <= q["q_index"] < (tour + 1) * 15]
            idxs = [q["q_index"] for q in chunk]
            act = sum(q["n_taken"] for q in chunk)
            exp = float(sum(sum_q[i] for i in idxs))
            tour_stats.append(
                {
                    "tour": tour + 1,
                    "mean_b": round(statistics.mean([q["b"] for q in chunk]), 3),
                    "mean_take_rate": round(statistics.mean([q["take_rate"] for q in chunk]), 3),
                    "actual_takes": act,
                    "expected_takes": round(exp, 1),
                    "delta": round(act - exp, 1),
                }
            )

    report = {
        "tournament_id": tournament_id,
        "n_teams": len(tg),
        "n_questions": n_q,
        "summary": {
            "mean_actual": round(mean_actual, 2),
            "mean_expected": round(mean_expected, 2),
            "mean_delta": round(mean_actual - mean_expected, 2),
            "median_delta": round(statistics.median([r["delta"] for r in tg]), 2),
            "mean_take_rate_pct": round(mean_take_rate * 100, 2),
            "mean_pred_rate_pct": round(mean_pred_rate * 100, 2),
            "rate_gap_pp": round((mean_pred_rate - mean_take_rate) * 100, 2),
            "pct_teams_under": round(100 * sum(1 for r in tg if r["delta"] < -1) / len(tg), 1),
            "pct_teams_over": round(100 * sum(1 for r in tg if r["delta"] > 1) / len(tg), 1),
        },
        "by_quartile": _quartile_stats(tg),
        "correlations": {
            "strength_before_vs_delta": round(_spearman(strengths, deltas), 3),
        },
        "counterfactuals": {
            "site_duckdb_delta": round(mean_actual - mean_expected, 2),
            "theta_before_recompute_delta": round(mean_actual - exp_before, 2),
            "theta_display_recompute_delta": round(mean_actual - exp_display, 2),
            "b_init_oracle_delta": round(mean_actual - exp_binit, 2),
            "pack_adj_delta": round(mean_actual - exp_pack_adj, 2),
            "uniform_b_shift_delta": round(mean_actual - exp_shift, 2),
            "no_lapse_recal_delta": round(mean_actual - exp_nocal, 2),
            "mean_b_trained": round(statistics.mean([q["b"] for q in questions]), 3),
            "mean_b_init": round(float(b_init.mean()), 3),
            "b_gap_init_minus_trained": round(b_gap, 3),
            "b_gap_trained_minus_init": round(-b_gap, 3),
            "pack_adj_applied": pack_adj_on,
            "pack_adj_retrospective_gate": retro_on,
        },
        "fixes": {
            "trained": _fix_summary(trained_rows),
            "oracle_b": _fix_summary(oracle_rows),
            "pack_adj": _fix_summary(pack_adj_rows),
            "uniform_b_shift": _fix_summary(shift_rows),
        },
        "per_question_worst": worst_q,
        "by_tour": tour_stats,
    }
    if baseline_offline:
        report["offline_baseline"] = _offline_baseline(con, exclude={tournament_id})
    con.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Forecast bias diagnostic")
    ap.add_argument("--tournament-id", type=int, default=12826)
    ap.add_argument("--compare", type=int, default=None)
    ap.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    ap.add_argument("--seq", type=Path, default=DEFAULT_SEQ)
    ap.add_argument("--xlsx", type=Path, default=None)
    ap.add_argument("--baseline-offline", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    xlsx = args.xlsx
    if xlsx is None and DEFAULT_KVRM_XLSX.exists():
        xlsx = DEFAULT_KVRM_XLSX

    report = diagnose(
        args.tournament_id,
        duckdb_path=args.duckdb,
        seq_path=args.seq,
        xlsx_path=xlsx,
        baseline_offline=args.baseline_offline or args.compare is not None,
    )
    if args.compare:
        report["compare"] = diagnose(
            args.compare,
            duckdb_path=args.duckdb,
            seq_path=args.seq,
            xlsx_path=None,
            baseline_offline=False,
        )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    s = report["summary"]
    print(f"\n=== Forecast diagnostic #{report['tournament_id']} ===")
    print(
        f"Mean actual={s['mean_actual']} expected={s['mean_expected']} "
        f"Δ={s['mean_delta']:+.2f}; take rate {s['mean_take_rate_pct']:.1f}% "
        f"vs predicted {s['mean_pred_rate_pct']:.1f}% ({s['rate_gap_pp']:+.1f} p.p.)"
    )
    print(f"Teams Δ<-1: {s['pct_teams_under']}%  Δ>+1: {s['pct_teams_over']}%")
    for q in report["by_quartile"]:
        print(
            f"  {q['quartile']} ({q['place_range']}): Δ={q['mean_delta']:+.2f} "
            f"actual={q['mean_actual']} expected={q['mean_expected']}"
        )
    cf = report["counterfactuals"]
    print(
        f"\nCounterfactuals (mean Δ): site={cf['site_duckdb_delta']:+.2f} "
        f"oracle={cf['b_init_oracle_delta']:+.2f} "
        f"pack_adj={cf['pack_adj_delta']:+.2f} "
        f"b_shift={cf['uniform_b_shift_delta']:+.2f} "
        f"no lapse/recal={cf['no_lapse_recal_delta']:+.2f}"
    )
    print(
        f"b trained={cf['mean_b_trained']} init={cf['mean_b_init']} "
        f"gap(init−trained)={cf['b_gap_init_minus_trained']:+.3f} "
        f"pack_adj={'yes' if cf['pack_adj_applied'] else 'no'} "
        f"retro_gate={'yes' if cf['pack_adj_retrospective_gate'] else 'no'}"
    )
    fx = report.get("fixes") or {}
    for key, label in (
        ("trained", "trained b"),
        ("oracle_b", "oracle b"),
        ("pack_adj", "pack-adj b"),
    ):
        block = fx.get(key) or {}
        if block.get("mean_delta") is None:
            continue
        print(f"\n  [{label}] mean Δ={block['mean_delta']:+.2f}")
        for q in block.get("by_quartile") or []:
            print(
                f"    {q['quartile']} ({q['place_range']}): "
                f"Δ={q['mean_delta']:+.2f}"
            )
    print(f"ρ(strength, Δ)={report['correlations']['strength_before_vs_delta']:+.3f}")
    if args.compare:
        cs = report["compare"]["summary"]
        print(f"\nCompare #{args.compare}: Δ={cs['mean_delta']:+.2f}")
    if report.get("offline_baseline"):
        b = report["offline_baseline"]
        print(
            f"\nOffline baseline: median Δ={b['median_delta_per_team']:+.2f}/team, "
            f"median tournament mean={b['median_delta_per_tournament']:+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
