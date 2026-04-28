"""Where does the current model lose the most?

Runs ``backtest()`` once with ``collect_predictions=True`` and slices the
per-observation residuals on the held-out 20 % time tail along several
axes:

* calibration in 10 probability bins (+ ECE);
* per-tournament mean logloss (top-K best / worst);
* per-canonical-question mean logloss (top-K worst);
* per-player accumulated residual (uniform attribution of each team
  residual across the players on that roster — naive but simple, see
  README caveat below);
* structural slices: tournament mode, team size, position-in-tour,
  roster-strength quartile.

Outputs go to ``results/error_analysis/``:

    calibration.csv
    worst_tournaments.csv  best_tournaments.csv
    worst_questions.csv
    players_underestimated.csv  players_overestimated.csv
    slices.csv
    summary.json

Tournament titles and player names are joined in from the website
DuckDB (``website/data/chgk.duckdb``) when available; otherwise only
ids are shown.

CAVEAT on player residuals: the team-level residual is split equally
across the players on the roster.  This is simple but biased — strong
veterans on a roster of one weak link get smeared with the residual
that mostly belongs to the weak link.  For a more principled split
you'd weight by per-player ``λ_k = exp(a·θ_k − b − δ)`` ("noisy-OR
credit"); not done here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config


def _build_offsets(team_sizes: np.ndarray) -> np.ndarray:
    off = np.empty(len(team_sizes) + 1, dtype=np.int64)
    off[0] = 0
    np.cumsum(team_sizes.astype(np.int64), out=off[1:])
    return off


def _try_open_duckdb(path: str):
    try:
        import duckdb

        return duckdb.connect(path, read_only=True)
    except Exception as e:
        print(f"  (no DuckDB titles: {e})")
        return None


def _tournament_titles(con, tournament_ids: list[int]) -> dict[int, dict]:
    if con is None or not tournament_ids:
        return {}
    rows = con.execute(
        "SELECT tournament_id, title, type, start_date, n_questions, n_teams "
        "FROM tournaments WHERE tournament_id = ANY(?)",
        [tournament_ids],
    ).fetchall()
    return {
        r[0]: {
            "title": r[1],
            "type": r[2],
            "start_date": str(r[3]),
            "n_q": r[4],
            "n_teams": r[5],
        }
        for r in rows
    }


def _player_names(con, player_ids: list[int]) -> dict[int, str]:
    if con is None or not player_ids:
        return {}
    rows = con.execute(
        "SELECT player_id, COALESCE(last_name,'') || ' ' || "
        "COALESCE(first_name,'') AS name "
        "FROM players WHERE player_id = ANY(?)",
        [player_ids],
    ).fetchall()
    return {r[0]: r[1].strip() for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out_dir", default="results/error_analysis")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--min_obs_q", type=int, default=20,
                    help="min n_obs per question to include in worst-Q list")
    ap.add_argument("--min_obs_g", type=int, default=200,
                    help="min n_obs per tournament to include in worst/best")
    ap.add_argument("--min_obs_p", type=int, default=20,
                    help="min n test obs per player to include in player lists")
    ap.add_argument("--duckdb",
                    default="website/data/chgk.duckdb",
                    help="optional: DuckDB for title/name lookups")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)
    cfg = Config()  # current production defaults

    print("[backtest] running once with predictions…")
    metrics = backtest(arrays, maps, cfg, verbose=True)
    result = metrics["result"]
    pred = result.predictions
    if pred is None or "obs_idx" not in pred:
        raise RuntimeError(
            "predictions missing obs_idx — engine.py needs the "
            "obs_idx patch from this commit."
        )

    pred_p = pred["pred_p"]
    actual_y = pred["actual_y"]
    pred_g = pred["game_idx"]
    pred_obs = pred["obs_idx"]

    # Reconstruct test mask the same way backtest() does.
    gdo = maps.game_date_ordinal
    all_games = np.unique(pred_g)
    if gdo is not None:
        known = all_games[
            np.array([int(gdo[g]) >= 0 for g in all_games], dtype=bool)
        ]
        ordered = known[
            np.argsort(np.array([int(gdo[g]) for g in known]))
        ]
    else:
        ordered = np.sort(all_games)
    n_test = max(1, int(len(ordered) * 0.2))
    test_games = set(int(g) for g in ordered[-n_test:])
    test_mask = np.fromiter(
        (int(g) in test_games for g in pred_g),
        count=len(pred_g),
        dtype=bool,
    )

    p = pred_p[test_mask]
    y = actual_y[test_mask].astype(np.float64)
    g_t = pred_g[test_mask]
    obs_t = pred_obs[test_mask]

    eps = 1e-15
    p_clip = np.clip(p, eps, 1.0 - eps)
    ll = -(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
    res = y - p  # >0: model under-predicted (team beat expectation)

    # === 1. CALIBRATION ===============================================
    bins = np.linspace(0.0, 1.0, 11)
    bin_id = np.clip(np.digitize(p, bins) - 1, 0, 9)
    cal_rows = []
    ece_num = 0.0
    for b in range(10):
        m = bin_id == b
        n_b = int(m.sum())
        if n_b == 0:
            continue
        mp = float(p[m].mean())
        ay = float(y[m].mean())
        cal_rows.append({
            "bin": b,
            "p_lo": float(bins[b]),
            "p_hi": float(bins[b + 1]),
            "n": n_b,
            "mean_p": mp,
            "actual_freq": ay,
            "abs_diff": abs(mp - ay),
        })
        ece_num += n_b * abs(mp - ay)
    ece = ece_num / max(1, len(p))
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(out / "calibration.csv", index=False)
    print(f"[calibration] ECE = {ece:.4f}")
    print(cal_df.to_string(index=False))

    # === 2. PER-TOURNAMENT ===========================================
    df = pd.DataFrame({
        "g": g_t, "obs": obs_t, "p": p, "y": y, "ll": ll, "res": res,
    })
    by_g = (
        df.groupby("g")
          .agg(n_obs=("y", "size"), mean_ll=("ll", "mean"),
               mean_p=("p", "mean"), mean_y=("y", "mean"),
               mean_res=("res", "mean"))
          .reset_index()
    )
    by_g["tournament_id"] = by_g["g"].map(
        lambda gi: maps.idx_to_game_id[gi]
        if gi < len(maps.idx_to_game_id) else int(gi)
    )

    con = _try_open_duckdb(args.duckdb)
    titles = _tournament_titles(con, by_g["tournament_id"].tolist())
    by_g["title"] = by_g["tournament_id"].map(
        lambda tid: titles.get(tid, {}).get("title", "")
    )
    by_g["type"] = by_g["tournament_id"].map(
        lambda tid: titles.get(tid, {}).get("type", "")
    )
    by_g["start_date"] = by_g["tournament_id"].map(
        lambda tid: titles.get(tid, {}).get("start_date", "")
    )
    by_g_keep = by_g[by_g["n_obs"] >= args.min_obs_g].copy()

    cols = ["tournament_id", "title", "type", "start_date",
            "n_obs", "mean_ll", "mean_p", "mean_y", "mean_res"]
    by_g_keep.sort_values("mean_ll", ascending=False).head(args.top_k)[
        cols
    ].to_csv(out / "worst_tournaments.csv", index=False)
    by_g_keep.sort_values("mean_ll", ascending=True).head(args.top_k)[
        cols
    ].to_csv(out / "best_tournaments.csv", index=False)

    # === 3. PER-QUESTION =============================================
    q_idx_arr = arrays["q_idx"]
    cq_arr = (
        maps.canonical_q_idx if maps.canonical_q_idx is not None
        else np.arange(maps.num_questions, dtype=np.int32)
    )
    q_obs = q_idx_arr[obs_t]
    cq_obs = cq_arr[q_obs]
    df["cq"] = cq_obs

    by_q = (
        df.groupby("cq")
          .agg(n=("y", "size"), mean_ll=("ll", "mean"),
               mean_p=("p", "mean"), mean_y=("y", "mean"),
               mean_res=("res", "mean"))
          .reset_index()
    )
    by_q_keep = by_q[by_q["n"] >= args.min_obs_q].copy()
    by_q_keep["b"] = result.questions.b[by_q_keep["cq"].values]
    by_q_keep["a"] = np.exp(np.clip(
        result.questions.log_a[by_q_keep["cq"].values], -3, 3))
    cols_q = ["cq", "n", "mean_ll", "mean_p", "mean_y",
              "mean_res", "b", "a"]
    by_q_keep.sort_values("mean_ll", ascending=False).head(args.top_k)[
        cols_q
    ].to_csv(out / "worst_questions.csv", index=False)

    # === 4. PER-PLAYER ===============================================
    team_sizes = arrays["team_sizes"]
    offsets = _build_offsets(team_sizes)
    player_flat = arrays["player_indices_flat"]
    n_players = maps.num_players

    sum_res = np.zeros(n_players, dtype=np.float64)
    sum_abs_res = np.zeros(n_players, dtype=np.float64)
    sum_ll = np.zeros(n_players, dtype=np.float64)
    n_obs_per = np.zeros(n_players, dtype=np.int64)

    for k in range(len(obs_t)):
        oi = int(obs_t[k])
        s, e = int(offsets[oi]), int(offsets[oi + 1])
        sz = e - s
        if sz <= 0:
            continue
        share = 1.0 / sz
        ll_k = ll[k]
        res_k = res[k]
        for pid in player_flat[s:e]:
            pid_i = int(pid)
            sum_res[pid_i] += res_k * share
            sum_abs_res[pid_i] += abs(res_k) * share
            sum_ll[pid_i] += ll_k * share
            n_obs_per[pid_i] += 1

    p_df = pd.DataFrame({
        "player_idx": np.arange(n_players),
        "n_obs_test": n_obs_per,
        "sum_res": sum_res,
        "mean_res": sum_res / np.maximum(n_obs_per, 1),
        "mean_abs_res": sum_abs_res / np.maximum(n_obs_per, 1),
        "mean_ll": sum_ll / np.maximum(n_obs_per, 1),
        "theta": result.players.theta[:n_players],
        "games_total": result.players.games[:n_players],
    })
    p_df["player_id"] = p_df["player_idx"].map(
        lambda i: maps.idx_to_player_id[i] if i < len(maps.idx_to_player_id) else int(i)
    )
    p_df_active = p_df[p_df["n_obs_test"] >= args.min_obs_p].copy()
    names = _player_names(con, p_df_active["player_id"].tolist())
    p_df_active["name"] = p_df_active["player_id"].map(
        lambda pid: names.get(pid, "")
    )

    cols_p = ["player_id", "name", "theta", "games_total",
              "n_obs_test", "mean_res", "sum_res", "mean_ll",
              "mean_abs_res"]
    # Underestimated: mean_res > 0 (team beat expectation); player
    # appears to be stronger than θ predicts.
    p_df_active.sort_values("mean_res", ascending=False).head(args.top_k)[
        cols_p
    ].to_csv(out / "players_underestimated.csv", index=False)
    p_df_active.sort_values("mean_res", ascending=True).head(args.top_k)[
        cols_p
    ].to_csv(out / "players_overestimated.csv", index=False)

    # === 5. STRUCTURAL SLICES ========================================
    df["team_size"] = np.clip(team_sizes[obs_t], 1, 8)

    qids = maps.idx_to_question_id
    q_pos = np.zeros(maps.num_questions, dtype=np.int32)
    if qids is not None and len(qids) > 0 and isinstance(qids[0], tuple):
        for qi in range(min(maps.num_questions, len(qids))):
            q_pos[qi] = int(qids[qi][1]) % 12
    df["pos_in_tour"] = q_pos[q_obs]

    gtype = maps.game_type
    if gtype is not None:
        def _bucket(g):
            s = str(gtype[g]) if g < len(gtype) else "offline"
            if "async" in s:
                return "async"
            if "sync" in s:
                return "sync"
            return "offline"
        df["mode"] = [_bucket(int(g)) for g in g_t]
    else:
        df["mode"] = "offline"

    # Per-game roster strength quartile (mean pre-tournament θ over
    # predicting teams).  Use team_theta_mean from predictions.
    thbar = pred.get("team_theta_mean")
    if thbar is not None:
        thbar_t = thbar[test_mask]
        per_g = pd.DataFrame({"g": g_t, "thbar": thbar_t}).groupby("g")["thbar"].mean()
        cuts = np.quantile(per_g.values, [0.25, 0.5, 0.75])
        cuts = np.unique(cuts)
        g_to_q = pd.Series(
            np.searchsorted(cuts, per_g.values, side="right") + 1,
            index=per_g.index,
        )
        df["hardness_q"] = df["g"].map(g_to_q).fillna(0).astype(int)
    else:
        df["hardness_q"] = 0

    slice_rows = []
    for col in ("mode", "team_size", "pos_in_tour", "hardness_q"):
        for v, sub in df.groupby(col):
            slice_rows.append({
                "slice": col,
                "value": v,
                "n": len(sub),
                "mean_p": float(sub["p"].mean()),
                "mean_y": float(sub["y"].mean()),
                "mean_ll": float(sub["ll"].mean()),
                "mean_res": float(sub["res"].mean()),
                "mean_abs_res": float(np.abs(sub["res"]).mean()),
            })
    pd.DataFrame(slice_rows).to_csv(out / "slices.csv", index=False)

    # === 6. SUMMARY ==================================================
    summary = {
        "n_test_obs": int(test_mask.sum()),
        "n_test_games": len(test_games),
        "logloss": float(metrics["logloss"]),
        "brier": float(metrics["brier"]),
        "auc": float(metrics["auc"]),
        "ECE": float(ece),
        "calibration": cal_rows,
        "worst_tournaments_top5": (
            by_g_keep.sort_values("mean_ll", ascending=False)
                     .head(5)[cols].to_dict("records")
        ),
        "best_tournaments_top5": (
            by_g_keep.sort_values("mean_ll", ascending=True)
                     .head(5)[cols].to_dict("records")
        ),
        "worst_questions_top5": (
            by_q_keep.sort_values("mean_ll", ascending=False)
                     .head(5)[cols_q].to_dict("records")
        ),
        "underestimated_players_top5": (
            p_df_active.sort_values("mean_res", ascending=False)
                       .head(5)[cols_p].to_dict("records")
        ),
        "overestimated_players_top5": (
            p_df_active.sort_values("mean_res", ascending=True)
                       .head(5)[cols_p].to_dict("records")
        ),
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=float)
    print(f"\n[ok] wrote analysis to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
