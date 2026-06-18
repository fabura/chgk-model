#!/usr/bin/env python3
"""Comprehensive residual / calibration analysis with honest cell-holdout.

Slices hold-out observations (Config.holdout_obs_fraction) by team size,
tournament type, pack length, question difficulty (b) and discrimination (a),
player experience, geography, editors, online/offline player mix, roster
stability, play frequency, position-in-tour, field strength, and more.

Outputs under ``results/error_analysis_comprehensive/``.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

REPO = Path(__file__).resolve().parents[1]
MOSCOW_TOWN = 201
SPB_TOWN = 285


def _build_offsets(team_sizes: np.ndarray) -> np.ndarray:
    off = np.empty(len(team_sizes) + 1, dtype=np.int64)
    off[0] = 0
    np.cumsum(team_sizes.astype(np.int64), out=off[1:])
    return off


def _bucket_mode(gt: str) -> str:
    s = str(gt)
    if "async" in s:
        return "async"
    if "sync" in s:
        return "sync"
    return "offline"


def _agg_slice(
    name: str,
    labels: np.ndarray,
    p: np.ndarray,
    y: np.ndarray,
    ll: np.ndarray,
    *,
    min_n: int = 500,
) -> list[dict]:
    rows: list[dict] = []
    for v in np.unique(labels):
        m = labels == v
        n = int(m.sum())
        if n < min_n:
            continue
        pm, ym = p[m], y[m]
        llm = ll[m]
        rows.append({
            "dimension": name,
            "slice": str(v),
            "n": n,
            "mean_p": float(pm.mean()),
            "mean_y": float(ym.mean()),
            "bias": float((ym - pm).mean()),  # actual - predicted
            "mean_ll": float(llm.mean()),
            "mean_abs_bias": float(np.abs(ym - pm).mean()),
            "brier": float(np.mean((pm - ym) ** 2)),
        })
    return rows


def _calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bid = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = bid == b
        nb = int(m.sum())
        if nb == 0:
            continue
        mp = float(p[m].mean())
        ay = float(y[m].mean())
        rows.append({
            "bin": b,
            "p_lo": float(bins[b]),
            "p_hi": float(bins[b + 1]),
            "n": nb,
            "mean_p": mp,
            "actual_freq": ay,
            "bias": ay - mp,
        })
        ece += nb * abs(mp - ay)
    ece /= max(1, len(p))
    return ece, rows


def _load_duckdb_aux(maps, duckdb_path: Path) -> dict:
    aux: dict = {}
    try:
        import duckdb

        con = duckdb.connect(str(duckdb_path), read_only=True)
    except Exception as e:
        print(f"  [warn] DuckDB unavailable: {e}")
        return aux

    # tournament_id -> editor names (top editors only used later)
    try:
        ed = con.execute(
            "SELECT tournament_id, editor_name FROM pack_editors"
        ).fetchall()
        t_ed: dict[int, list[str]] = defaultdict(list)
        for tid, name in ed:
            t_ed[int(tid)].append(str(name))
        aux["editors"] = dict(t_ed)
    except Exception:
        aux["editors"] = {}

    # player online fraction: share of games in async/sync vs offline
    try:
        pf = con.execute(
            """
            SELECT pg.player_id,
                   avg(CASE WHEN t.type IN ('async','sync') THEN 1.0 ELSE 0.0 END) AS online_frac,
                   count(*) AS n_games
            FROM player_games pg
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
            GROUP BY pg.player_id
            """
        ).fetchall()
        aux["player_online_frac"] = {int(r[0]): (float(r[1]), int(r[2])) for r in pf}
    except Exception:
        aux["player_online_frac"] = {}

    # roster key -> team_id per tournament
    try:
        tr = con.execute(
            """
            SELECT pg.tournament_id, pg.team_id,
                   list(pg.player_id ORDER BY pg.player_id) AS pids
            FROM player_games pg
            GROUP BY pg.tournament_id, pg.team_id
            """
        ).fetchall()
        roster_team: dict[tuple[int, tuple[int, ...]], int] = {}
        for tid, team_id, pids in tr:
            roster_team[(int(tid), tuple(int(x) for x in pids))] = int(team_id)
        aux["roster_team"] = roster_team
    except Exception:
        aux["roster_team"] = {}

    # team roster stability (Jaccard vs previous appearance)
    try:
        hist = con.execute(
            """
            SELECT pg.team_id, pg.tournament_id, t.start_date,
                   list(pg.player_id ORDER BY pg.player_id) AS pids
            FROM player_games pg
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
            WHERE pg.team_id IS NOT NULL
            GROUP BY pg.team_id, pg.tournament_id, t.start_date
            ORDER BY pg.team_id, t.start_date
            """
        ).fetchall()
        team_jacc: dict[int, float] = {}
        prev: dict[int, set[int]] = {}
        sums: dict[int, list[float]] = defaultdict(list)
        for team_id, _tid, _dt, pids in hist:
            cur = set(int(x) for x in pids)
            if team_id in prev:
                inter = len(prev[team_id] & cur)
                union = len(prev[team_id] | cur)
                if union > 0:
                    sums[int(team_id)].append(inter / union)
            prev[int(team_id)] = cur
        for tid, vals in sums.items():
            team_jacc[tid] = float(np.mean(vals)) if vals else 1.0
        aux["team_jaccard"] = team_jacc
    except Exception:
        aux["team_jaccard"] = {}

    # tournament town from baked DB + json fallback handled outside
    try:
        towns = con.execute(
            "SELECT tournament_id, town_id FROM map_venues mv "
            "JOIN tournaments t ON t.tournament_id = mv.tournament_id"
        ).fetchall()
        aux["tournament_town"] = {int(r[0]): int(r[1]) for r in towns if r[1] is not None}
    except Exception:
        aux["tournament_town"] = {}

    con.close()
    return aux


def _load_tournament_towns_json() -> dict[int, int | None]:
    path = REPO / "website" / "data" / "tournament_towns.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): (int(v) if v is not None else None) for k, v in raw.items()}


def _load_venue_town(venue_db: Path) -> dict[tuple[int, int], int]:
    """(tournament_id, team_id) -> town_id via venue town."""
    out: dict[tuple[int, int], int] = {}
    if not venue_db.is_file():
        return out
    try:
        import duckdb

        con = duckdb.connect(str(venue_db), read_only=True)
        rows = con.execute(
            """
            SELECT ttv.tournament_id, ttv.team_id, v.town_id
            FROM team_tournament_venue ttv
            JOIN venues v ON v.venue_id = ttv.venue_id
            WHERE v.town_id IS NOT NULL AND NOT coalesce(v.is_online, false)
            """
        ).fetchall()
        for tid, team_id, town_id in rows:
            out[(int(tid), int(team_id))] = int(town_id)
        con.close()
    except Exception as e:
        print(f"  [warn] venue overlay: {e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out_dir", default="results/error_analysis_comprehensive")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--venue_db", default="data/venue_overlay.duckdb")
    ap.add_argument("--min_slice_n", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)
    n_obs = len(arrays["q_idx"])
    n_players = maps.num_players
    offsets = _build_offsets(arrays["team_sizes"])
    player_flat = arrays["player_indices_flat"]
    idx_to_pid = maps.idx_to_player_id
    idx_to_tid = maps.idx_to_game_id

    cfg = Config(holdout_obs_fraction=args.holdout, holdout_seed=args.seed)
    print(
        f"[train] cell-holdout run (fraction={args.holdout}, seed={args.seed})…"
    )
    t0 = time.time()
    result = run_sequential(
        arrays, maps, cfg, verbose=True, collect_predictions=True,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    pred = result.predictions
    if pred is None or "is_holdout" not in pred:
        raise RuntimeError("predictions missing is_holdout")

    holdout_mask = pred["is_holdout"].astype(bool)
    if holdout_mask.sum() == 0:
        raise RuntimeError("no holdout observations")

    p_all = pred["pred_p"]
    y_all = pred["actual_y"].astype(np.float64)
    obs_idx = pred["obs_idx"]
    g_idx = pred["game_idx"]
    thbar = pred.get("team_theta_mean")

    p = p_all[holdout_mask]
    y = y_all[holdout_mask]
    obs_h = obs_idx[holdout_mask]
    g_h = g_idx[holdout_mask]
    thbar_h = thbar[holdout_mask] if thbar is not None else None

    eps = 1e-15
    p_clip = np.clip(p, eps, 1.0 - eps)
    ll = -(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
    res = y - p

    metrics = compute_metrics(p, y.astype(int))
    ece, cal_rows = _calibration_bins(p, y)
    print(
        f"[holdout] n={len(p):,}  logloss={metrics['logloss']:.4f}  "
        f"brier={metrics['brier']:.4f}  auc={metrics['auc']:.4f}  ECE={ece:.4f}"
    )

    # --- precompute per-obs features (vectorized where possible) ---
    print("[features] building slice labels…")
    team_sizes_h = arrays["team_sizes"][obs_h]
    q_obs = arrays["q_idx"][obs_h]

    cq = (
        maps.canonical_q_idx
        if maps.canonical_q_idx is not None
        else np.arange(maps.num_questions, dtype=np.int32)
    )
    cq_h = cq[q_obs]
    b_h = result.questions.b[cq_h]
    a_h = np.exp(np.clip(result.questions.log_a[cq_h], -3.0, 3.0))

    # position in tour
    q_pos = np.zeros(maps.num_questions, dtype=np.int32)
    qids = maps.idx_to_question_id
    if qids is not None and len(qids) > 0 and isinstance(qids[0], tuple):
        for qi in range(min(maps.num_questions, len(qids))):
            q_pos[qi] = int(qids[qi][1]) % 12
    pos_h = q_pos[q_obs]

    # mode
    gtype = maps.game_type
    mode_h = np.array(
        [_bucket_mode(gtype[g] if g < len(gtype) else "offline") for g in g_h],
        dtype=object,
    )

    # questions per tournament (unique q_idx per game, not obs count)
    uq_by_game: dict[int, set[int]] = defaultdict(set)
    for qi, g in zip(arrays["q_idx"], arrays["game_idx"]):
        uq_by_game[int(g)].add(int(qi))
    q_per_game = np.array(
        [len(uq_by_game.get(g, ())) for g in range(len(maps.idx_to_game_id))],
        dtype=np.int32,
    )
    pack_len_h = q_per_game[g_h]

    # games at prediction (chronological pass)
    print("[features] games-at-prediction pass…")
    games_count = np.zeros(n_players, dtype=np.int32)
    obs_by_game: dict[int, list[int]] = defaultdict(list)
    for i in range(n_obs):
        obs_by_game[int(arrays["game_idx"][i])].append(i)
    gdo = maps.game_date_ordinal
    all_games = np.array(sorted(obs_by_game.keys()), dtype=np.int32)
    if gdo is not None:
        known = all_games[np.array([gdo[g] >= 0 for g in all_games], dtype=bool)]
        game_order = known[np.argsort([gdo[g] for g in known])]
    else:
        game_order = all_games

    mean_games_pred = np.zeros(n_obs, dtype=np.float32)
    for g in game_order:
        for i in obs_by_game[int(g)]:
            s, e = int(offsets[i]), int(offsets[i + 1])
            if e > s:
                mean_games_pred[i] = float(games_count[player_flat[s:e]].mean())
        # after tournament: increment (matches engine — all obs players)
        seen_p: set[int] = set()
        for i in obs_by_game[int(g)]:
            s, e = int(offsets[i]), int(offsets[i + 1])
            for pi in player_flat[s:e]:
                seen_p.add(int(pi))
        for pi in seen_p:
            games_count[pi] += 1

    mg_h = mean_games_pred[obs_h]
    exp_bucket = np.where(
        mg_h < 15, "newbie_lt15",
        np.where(mg_h < 200, "mid_15_199", "veteran_200plus"),
    )

    # difficulty / discrimination buckets (quintiles on holdout)
    b_q = np.quantile(b_h, [0.2, 0.4, 0.6, 0.8])
    b_bucket = np.searchsorted(b_q, b_h, side="right")
    b_labels = np.array([f"b_q{i+1}" for i in b_bucket])

    # a buckets (mostly 1.0 if frozen)
    a_unique = np.unique(np.round(a_h, 4))
    if len(a_unique) <= 3:
        a_labels = np.full(len(a_h), f"a≈{float(a_unique[0]):.3f}", dtype=object)
    else:
        a_q = np.quantile(a_h, [0.2, 0.4, 0.6, 0.8])
        a_labels = np.array([f"a_q{int(np.searchsorted(a_q, v, side='right'))+1}" for v in a_h])

    # pack length buckets
    pl_labels = np.where(
        pack_len_h <= 24, "short_le24",
        np.where(pack_len_h <= 36, "medium_25_36", "long_37plus"),
    )

    # team size labels
    ts_labels = np.where(
        team_sizes_h == 1, "solo_1",
        np.where(team_sizes_h <= 3, "small_2_3",
        np.where(team_sizes_h <= 5, "mid_4_5",
        np.where(team_sizes_h == 6, "std_6", "large_7plus"))),
    )

    # field strength quartile
    if thbar_h is not None:
        cuts = np.unique(np.quantile(thbar_h, [0.25, 0.5, 0.75]))
        field_q = np.searchsorted(cuts, thbar_h, side="right") + 1
        field_labels = np.array([f"field_q{int(v)}" for v in field_q])
    else:
        field_labels = np.full(len(p), "unknown", dtype=object)

    # DuckDB / venue aux
    print("[features] loading DuckDB / venue metadata…")
    aux = _load_duckdb_aux(maps, Path(args.duckdb))
    town_json = _load_tournament_towns_json()
    venue_town = _load_venue_town(Path(args.venue_db))
    roster_team = aux.get("roster_team", {})
    team_jacc = aux.get("team_jaccard", {})
    player_online = aux.get("player_online_frac", {})
    editors_map = aux.get("editors", {})

    # per-obs geo, team, player mix (group by tournament + roster to avoid redundant work)
    print("[features] per-obs metadata (team/geo/editor)…")
    geo_labels = np.full(len(p), "unknown", dtype=object)
    roster_stab = np.full(len(p), "unknown", dtype=object)
    player_mix = np.full(len(p), "unknown", dtype=object)
    play_freq = np.full(len(p), "unknown", dtype=object)
    editor_labels = np.full(len(p), "no_editor", dtype=object)

    top_editors = {
        "Максим Мерзляков", "Антон Саксонов", "Сергей Терентьев",
        "Александр Кудрявцев", "Александр Коробейников",
    }

    pid_to_idx = maps.player_id_to_idx
    roster_team_idx: dict[tuple[int, tuple[int, ...]], int] = {}
    for (tid, pids), team_id in roster_team.items():
        idxs = tuple(sorted(pid_to_idx.get(pid, -1) for pid in pids))
        if idxs and idxs[0] >= 0:
            roster_team_idx[(tid, idxs)] = team_id

    tids_h = np.array(
        [int(idx_to_tid[int(g)]) if int(g) < len(idx_to_tid) else int(g) for g in g_h],
        dtype=np.int64,
    )

    group_meta: dict[tuple[int, tuple[int, ...]], tuple] = {}
    group_obs: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    for k in range(len(obs_h)):
        oi = int(obs_h[k])
        tid = int(tids_h[k])
        s, e = int(offsets[oi]), int(offsets[oi + 1])
        key = (tid, tuple(sorted(int(pi) for pi in player_flat[s:e])))
        group_obs[key].append(k)

    for key, obs_ks in group_obs.items():
        tid, idxs = key
        pids = tuple(int(idx_to_pid[pi]) for pi in idxs if pi < len(idx_to_pid))

        town_id = town_json.get(tid) or aux.get("tournament_town", {}).get(tid)
        team_id = roster_team_idx.get(key)
        if team_id is not None and (tid, team_id) in venue_town:
            town_id = venue_town[(tid, team_id)]

        if town_id == MOSCOW_TOWN:
            geo = "moscow"
        elif town_id == SPB_TOWN:
            geo = "spb"
        elif town_id is not None:
            geo = "regions"
        elif mode_h[obs_ks[0]] != "offline":
            geo = "sync_async_no_geo"
        else:
            geo = "unknown"

        if team_id is not None and team_id in team_jacc:
            j = team_jacc[team_id]
            if j >= 0.85:
                rstab = "stable_ge85"
            elif j >= 0.60:
                rstab = "medium_60_84"
            else:
                rstab = "pickup_lt60"
        else:
            rstab = "unknown"

        fracs = [player_online[pid][0] for pid in pids if pid in player_online]
        if fracs:
            mf = float(np.mean(fracs))
            if mf >= 0.70:
                pmix = "online_heavy_ge70"
            elif mf >= 0.30:
                pmix = "mixed_30_69"
            else:
                pmix = "offline_heavy_lt30"
        else:
            pmix = "unknown"

        games_list = [player_online[pid][1] for pid in pids if pid in player_online]
        if games_list:
            mg = float(np.mean(games_list))
            if mg < 50:
                pfreq = "rare_lt50"
            elif mg < 300:
                pfreq = "regular_50_299"
            else:
                pfreq = "frequent_300plus"
        else:
            pfreq = "unknown"

        eds = editors_map.get(tid, [])
        if eds:
            primary = eds[0]
            if primary in top_editors:
                edlab = f"top5:{primary.split()[0]}"
            else:
                edlab = "other_editor"
        else:
            edlab = "no_editor"

        for k in obs_ks:
            geo_labels[k] = geo
            roster_stab[k] = rstab
            player_mix[k] = pmix
            play_freq[k] = pfreq
            editor_labels[k] = edlab

    # --- aggregate all slices ---
    all_rows: list[dict] = []
    min_n = args.min_slice_n
    for name, labels in [
        ("mode", mode_h),
        ("team_size_bucket", ts_labels),
        ("team_size_exact", team_sizes_h.astype(str)),
        ("pack_length", pl_labels),
        ("difficulty_b_quintile", b_labels),
        ("discrimination_a", a_labels),
        ("player_experience", exp_bucket),
        ("pos_in_tour", np.array([f"pos_{v}" for v in pos_h])),
        ("field_strength", field_labels),
        ("geo", geo_labels),
        ("roster_stability", roster_stab),
        ("player_online_mix", player_mix),
        ("play_frequency", play_freq),
        ("editor", editor_labels),
        ("mode_x_team_size", np.array([f"{m}_sz{ts}" for m, ts in zip(mode_h, team_sizes_h)])),
    ]:
        all_rows.extend(_agg_slice(name, labels, p, y, ll, min_n=min_n))

    slices_df = pd.DataFrame(all_rows)
    slices_df = slices_df.sort_values("mean_ll", ascending=False)
    slices_df.to_csv(out / "slices_all.csv", index=False)

    # calibration overall + by mode
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(out / "calibration_overall.csv", index=False)
    cal_mode_rows = []
    for m in ("offline", "sync", "async"):
        mask = mode_h == m
        if mask.sum() < min_n:
            continue
        _, cr = _calibration_bins(p[mask], y[mask])
        for row in cr:
            row["mode"] = m
            cal_mode_rows.append(row)
    pd.DataFrame(cal_mode_rows).to_csv(out / "calibration_by_mode.csv", index=False)

    # top problem slices
    top_bias = slices_df.reindex(slices_df["bias"].abs().sort_values(ascending=False).index).head(25)
    top_ll = slices_df.head(25)
    top_bias.to_csv(out / "top_abs_bias.csv", index=False)
    top_ll.to_csv(out / "top_logloss.csv", index=False)

    # cross-tab mode x experience
    cross_rows = []
    for m in ("offline", "sync", "async"):
        for e in ("newbie_lt15", "mid_15_199", "veteran_200plus"):
            mask = (mode_h == m) & (exp_bucket == e)
            n = int(mask.sum())
            if n < min_n:
                continue
            cross_rows.append({
                "dimension": "mode_x_experience",
                "slice": f"{m}_{e}",
                "n": n,
                "mean_p": float(p[mask].mean()),
                "mean_y": float(y[mask].mean()),
                "bias": float(res[mask].mean()),
                "mean_ll": float(ll[mask].mean()),
            })
    pd.DataFrame(cross_rows).to_csv(out / "mode_x_experience.csv", index=False)

    summary = {
        "holdout_fraction": args.holdout,
        "holdout_seed": args.seed,
        "n_holdout_obs": int(len(p)),
        "logloss": metrics["logloss"],
        "brier": metrics["brier"],
        "auc": metrics["auc"],
        "ECE": ece,
        "top10_worst_logloss": top_ll.head(10).to_dict("records"),
        "top10_worst_abs_bias": top_bias.head(10).to_dict("records"),
        "calibration_overall": cal_rows,
        "limitations": {
            "geo_coverage": (
                "offline town from tournament_towns.json (~4k tournaments); "
                "sync via venue_overlay team venue; async mostly unmapped"
            ),
            "editors": "pack_editors in DuckDB — primary editor only for slicing",
            "roster_stability": "Jaccard of consecutive rosters per team_id from player_games",
            "player_experience": "mean pre-tournament games on roster (chronological replay)",
            "discrimination": "a frozen at 1.0 in production — slice is mostly flat",
        },
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=float)

    print(f"\n[ok] wrote {out}/")
    print("\nTop-10 worst slices by logloss:")
    print(top_ll.head(10)[["dimension", "slice", "n", "bias", "mean_ll"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
