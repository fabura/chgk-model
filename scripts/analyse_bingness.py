"""Bing-ness diagnostic: does team activity predict systematic residual bias?

"Bing-ness" = team activity measured as COUNT(DISTINCT tournament_id) in
player_games.  Hypothesis: frequently-playing teams ("binge" teams) may be
systematically over- or under-predicted by the model.

Metrics computed from website/data/chgk.duckdb (read-only):
    residual  = score_actual - expected_takes   (from team_games, has_breakdown=true)
    activity  = COUNT(DISTINCT tournament_id) per team  (from player_games)

Outputs:
    1. Global Spearman(residual, log1p(activity)) + 95 % bootstrap CI
    2. Same split by format (offline / sync / async)
    3. Per-tournament pack-level: mean residual gap (Q4 - Q1 activity quartile)
       printed as histogram and written to results/bingness_pack_gap.csv
    4. --per-question: per-canonical-question bing affinity, written to
       results/bingness_per_question.csv  (uses data.npz + results/seq.npz)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman r and two-tailed p-value."""
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _bootstrap_ci(
    rx: np.ndarray,
    ry: np.ndarray,
    n_boot: int = 2000,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI on pre-ranked arrays (fast: avoids re-sorting per resample)."""
    rng = np.random.default_rng(rng_seed)
    n = len(rx)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.corrcoef(rx[idx], ry[idx])[0, 1])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _fmt_r(r: float, p: float, lo: float, hi: float) -> str:
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"r={r:+.4f}  p={p:.2e}{stars}  95% CI [{lo:+.4f}, {hi:+.4f}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-question bing affinity
# ---------------------------------------------------------------------------

def _per_question_affinity(
    cache_file: str = "data.npz",
    seq_file: str = "results/seq.npz",
    out_path: str = "results/bingness_per_question.csv",
    min_obs: int = 20,
    top_n: int = 30,
) -> int:
    """Compute per-canonical-question bing affinity.

    Uses FINAL model parameters (from seq.npz) to compute predicted
    probabilities for all 27M observations.  Residual = actual - predicted.
    Activity proxy = mean(player.games) over the team roster (final snapshot).

    For each canonical question with n_obs >= min_obs:
        affinity = mean(residual | activity in Q4) - mean(residual | activity in Q1)
        spearman_r = Spearman(residual, log1p(activity)) over the question

    The "predicted" value uses only the core noisy-OR + delta_size correction
    (omits delta_pos and lapse/recal which are uniform across teams on a given
    question, so they do not affect the within-question activity ranking).
    """
    cache_path = Path(cache_file)
    seq_path = Path(seq_file)
    for p, name in [(cache_path, "--cache_file"), (seq_path, "--seq_file")]:
        if not p.exists():
            print(f"[error] {name} not found at {p}")
            return 1

    print(f"[load] {cache_path}  +  {seq_path}")
    d = np.load(str(cache_path), allow_pickle=True)
    s = np.load(str(seq_path), allow_pickle=True)

    q_idx        = d["q_idx"]           # (N,) observation → question slot idx
    taken        = d["taken"]           # (N,) 0/1 outcome
    team_sizes   = d["team_sizes"]      # (N,) roster size
    player_flat  = d["player_indices_flat"]  # (M,) flattened player indices
    game_idx_arr = d["game_idx"]        # (N,) game index
    game_type    = d["game_type"]       # (G,) 'offline'/'sync'/'async'

    # question slot → canonical question index
    cq_map = d["canonical_q_idx"]       # (Q,) question slot → canonical idx
    cq_per_obs = cq_map[q_idx]          # (N,) canonical idx per obs

    # tournament type per obs (via question slot → game_idx mapping)
    q_game_idx = d["question_game_idx"]   # (Q,) question slot → game idx
    game_per_obs = q_game_idx[q_idx]      # (N,)

    theta        = s["theta"].astype(np.float32)   # (P,) final player strengths
    games_final  = s["games"].astype(np.int32)      # (P,) total games per player
    b            = s["b"].astype(np.float32)        # (CQ,) canonical difficulties
    delta_size   = s["delta_size"].astype(np.float32)
    team_size_max = int(len(delta_size) - 1)

    N = len(q_idx)
    print(f"[data] {N:,} observations, {int(cq_per_obs.max())+1:,} canonical questions")

    # ------------------------------------------------------------------
    # 1. Build per-observation predicted probability (vectorized)
    # ------------------------------------------------------------------
    print("[pred] building player→obs mapping…")
    obs_per_player = np.repeat(
        np.arange(N, dtype=np.int32), team_sizes.astype(np.int32)
    )

    ts_clipped   = np.clip(team_sizes, 1, team_size_max).astype(np.int32)
    b_eff_per_obs = b[cq_per_obs] + delta_size[ts_clipped]   # (N,)

    print("[pred] computing noisy-OR probabilities…")
    z         = (-b_eff_per_obs[obs_per_player]
                 + theta[player_flat]).astype(np.float64)
    lam       = np.exp(z)
    S         = np.bincount(obs_per_player.astype(np.int64),
                            weights=lam, minlength=N).astype(np.float32)
    p_hat     = (1.0 - np.exp(-S.astype(np.float64))).astype(np.float32)

    residual  = taken.astype(np.float32) - p_hat   # (N,)

    # ------------------------------------------------------------------
    # 2. Team activity proxy: mean(player.games) over roster
    # ------------------------------------------------------------------
    print("[act]  computing per-obs team mean games…")
    games_per_player  = games_final[player_flat].astype(np.float64)
    sum_games         = np.bincount(obs_per_player.astype(np.int64),
                                    weights=games_per_player, minlength=N)
    mean_games        = (sum_games / np.maximum(team_sizes, 1)).astype(np.float32)
    log_act           = np.log1p(mean_games)

    # ------------------------------------------------------------------
    # 3. Tournament type per obs
    # ------------------------------------------------------------------
    def _gtype(g: int) -> str:
        s_val = str(game_type[g])
        if "async" in s_val:
            return "async"
        if "sync" in s_val:
            return "sync"
        return "offline"

    # Per canonical question: majority tournament type among its obs
    # (we'll compute this by plurality vote below)

    # ------------------------------------------------------------------
    # 4. Per-canonical-question affinity
    # ------------------------------------------------------------------
    print("[group] aggregating per canonical question…")

    n_cq = int(cq_per_obs.max()) + 1

    # Sort observations by canonical question for fast group iteration
    sort_order = np.argsort(cq_per_obs, kind="stable")
    cq_sorted  = cq_per_obs[sort_order]
    res_sorted = residual[sort_order]
    act_sorted = log_act[sort_order]
    gidx_sorted = game_per_obs[sort_order]

    # Split points
    _, first_idx, counts = np.unique(
        cq_sorted, return_index=True, return_counts=True
    )

    records: list[dict] = []

    # Pre-compute game type by cq_idx via majority vote
    # (most canonical questions appear in one format; pairing is offline↔sync)
    for k, (cq, fi, cnt) in enumerate(zip(
        np.unique(cq_sorted), first_idx, counts
    )):
        if cnt < min_obs:
            continue

        sl = slice(fi, fi + cnt)
        r  = res_sorted[sl].astype(np.float64)
        a  = act_sorted[sl].astype(np.float64)
        gs = gidx_sorted[sl]

        # Affinity: Q4 - Q1 mean residual
        q25, q75 = np.percentile(a, [25, 75])
        bot      = r[a <= q25]
        top      = r[a >= q75]
        if len(bot) < 2 or len(top) < 2:
            continue
        affinity  = float(top.mean() - bot.mean())

        # Spearman
        if np.unique(a).size < 3:
            spear_r, spear_p = float("nan"), float("nan")
        else:
            _sr = stats.spearmanr(a, r)
            spear_r = float(_sr.statistic if hasattr(_sr, "statistic") else _sr.correlation)
            spear_p = float(_sr.pvalue)

        # Dominant tournament type
        types = [_gtype(int(g)) for g in gs[:min(100, len(gs))]]
        dominant_fmt = max(set(types), key=types.count)

        records.append({
            "canonical_idx": int(cq),
            "b":             float(b[cq]),
            "n_obs":         int(cnt),
            "fmt":           dominant_fmt,
            "affinity":      affinity,
            "spearman_r":    round(spear_r, 4) if not np.isnan(spear_r) else "",
            "spearman_p":    f"{spear_p:.3e}" if not np.isnan(spear_p) else "",
            "mean_res_q1":   round(float(bot.mean()), 4),
            "mean_res_q4":   round(float(top.mean()), 4),
            "mean_p_hat":    round(float(p_hat[sort_order[sl]].mean()), 4),
        })

    if not records:
        print("[warn] no canonical questions passed the n_obs filter")
        return 0

    print(f"[ok]   {len(records):,} canonical questions with n_obs ≥ {min_obs}")

    aff = np.array([r["affinity"] for r in records], dtype=np.float64)
    sp_r = np.array([r["spearman_r"] if r["spearman_r"] != "" else float("nan")
                     for r in records], dtype=np.float64)
    b_arr = np.array([r["b"] for r in records], dtype=np.float64)

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print("\n=== Per-question bing affinity (Q4 − Q1 mean residual) ===")
    print(f"  Questions analysed : {len(records):,}")
    print(f"  Mean affinity      : {float(np.nanmean(aff)):+.4f}")
    print(f"  Median affinity    : {float(np.nanmedian(aff)):+.4f}")
    print(f"  Std affinity       : {float(np.nanstd(aff)):.4f}")
    thr = 0.10
    frac_pos  = float((aff > 0).mean())
    frac_high = float((aff > thr).mean())
    print(f"  Frac affinity > 0  : {frac_pos:.2%}")
    print(f"  Frac affinity > {thr:.2f}: {frac_high:.2%}  "
          f"(n={int((aff > thr).sum())})")
    print(f"  Mean |Spearman r|  : {float(np.nanmean(np.abs(sp_r))):.4f}")

    # Affinity vs b correlation
    valid = (~np.isnan(aff)) & (~np.isnan(b_arr))
    r_aff_b, p_aff_b = stats.spearmanr(aff[valid], b_arr[valid])
    print(f"\n  Spearman(affinity, b)  r={r_aff_b:+.4f}  p={p_aff_b:.2e}")
    print("  (positive = harder questions have more bing affinity)")

    # By format
    print("\n  Mean affinity by format:")
    for fmt in ("offline", "sync", "async"):
        fa = [r["affinity"] for r in records if r["fmt"] == fmt]
        if fa:
            print(f"    {fmt:<8s} n={len(fa):5d}  "
                  f"mean={float(np.mean(fa)):+.4f}  "
                  f"median={float(np.median(fa)):+.4f}")

    # Affinity histogram
    hist, edges = np.histogram(aff, bins=12)
    max_bar = max(hist)
    scale = 40 / max(max_bar, 1)
    print("\n  Distribution of per-question affinity scores:")
    for i, cnt in enumerate(hist):
        bar = "█" * int(cnt * scale)
        print(f"  [{edges[i]:+5.2f}, {edges[i+1]:+5.2f})  {bar}  ({cnt})")

    # Top questions by affinity (bing candidates)
    records_sorted = sorted(records, key=lambda x: x["affinity"], reverse=True)
    print(f"\n  Top {top_n} canonical questions by affinity (strongest bing candidates):")
    header = (f"  {'cq_idx':>8}  {'b':>6}  {'n_obs':>6}  {'fmt':>7}  "
              f"{'affinity':>9}  {'spear_r':>8}  {'spear_p':>10}  "
              f"{'q1_res':>8}  {'q4_res':>8}")
    print(header)
    for rec in records_sorted[:top_n]:
        print(f"  {rec['canonical_idx']:>8d}  {rec['b']:>+6.2f}  "
              f"{rec['n_obs']:>6d}  {rec['fmt']:>7s}  "
              f"{rec['affinity']:>+9.4f}  "
              f"{str(rec['spearman_r']):>8s}  "
              f"{str(rec['spearman_p']):>10s}  "
              f"{rec['mean_res_q1']:>+8.4f}  "
              f"{rec['mean_res_q4']:>+8.4f}")

    # Check if high-affinity questions cluster by difficulty
    high_b = b_arr[aff > thr]
    low_b  = b_arr[aff <= thr]
    print(f"\n  b (difficulty) stats:")
    print(f"    All questions : mean={float(np.nanmean(b_arr)):+.3f}  "
          f"median={float(np.nanmedian(b_arr)):+.3f}")
    if len(high_b):
        print(f"    Affinity>{thr:.2f} : mean={float(np.nanmean(high_b)):+.3f}  "
              f"median={float(np.nanmedian(high_b)):+.3f}  "
              f"n={len(high_b)}")
    if len(low_b):
        print(f"    Affinity≤{thr:.2f} : mean={float(np.nanmean(low_b)):+.3f}  "
              f"median={float(np.nanmedian(low_b)):+.3f}  "
              f"n={len(low_b)}")

    # ------------------------------------------------------------------
    # 6. Write CSV
    # ------------------------------------------------------------------
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    records_sorted_full = sorted(records, key=lambda x: x["affinity"], reverse=True)
    fieldnames = list(records_sorted_full[0].keys())
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records_sorted_full)
    print(f"\n[ok] per-question CSV → {out}  ({len(records_sorted_full)} rows)")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="website/data/chgk.duckdb",
                    help="path to chgk.duckdb (default: website/data/chgk.duckdb)")
    ap.add_argument("--out", default="results/bingness_pack_gap.csv",
                    help="CSV output for pack-level quartile gaps")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for CI (default 2000)")
    ap.add_argument("--min-activity", type=int, default=5,
                    help="drop teams with fewer than this many tournaments")
    # --- per-question section ---
    ap.add_argument("--per-question", action="store_true",
                    help="run per-canonical-question affinity analysis "
                         "(uses data.npz + results/seq.npz; skips DuckDB section)")
    ap.add_argument("--cache_file", default="data.npz",
                    help="path to data.npz  (default: data.npz)")
    ap.add_argument("--seq_file", default="results/seq.npz",
                    help="path to seq.npz  (default: results/seq.npz)")
    ap.add_argument("--per-question-out", default="results/bingness_per_question.csv",
                    help="CSV output for per-question affinity")
    ap.add_argument("--min-obs", type=int, default=20,
                    help="min observations per canonical question (default 20)")
    ap.add_argument("--top-n", type=int, default=30,
                    help="number of top bing-candidate questions to print")
    args = ap.parse_args()

    if args.per_question:
        return _per_question_affinity(
            cache_file=args.cache_file,
            seq_file=args.seq_file,
            out_path=args.per_question_out,
            min_obs=args.min_obs,
            top_n=args.top_n,
        )

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[error] DuckDB not found at {db_path}")
        print("Run `python -m website.build.build_db` first.")
        return 1

    import duckdb  # local import — optional dep

    con = duckdb.connect(str(db_path), read_only=True)

    # ------------------------------------------------------------------
    # 1. Pull data: one row per (team × tournament) with residual + activity
    # ------------------------------------------------------------------
    print(f"[load] {db_path}")
    df = con.execute("""
        WITH team_activity AS (
            SELECT team_id, COUNT(DISTINCT tournament_id) AS activity
            FROM player_games
            GROUP BY team_id
        )
        SELECT
            tg.tournament_id,
            tg.team_id,
            tg.score_actual - tg.expected_takes   AS residual,
            t.type                                  AS fmt,
            ta.activity
        FROM team_games tg
        JOIN tournaments t  ON t.tournament_id = tg.tournament_id
        JOIN team_activity ta ON ta.team_id    = tg.team_id
        WHERE tg.has_breakdown = true
        ORDER BY tg.tournament_id, tg.team_id
    """).df()

    n_raw = len(df)
    df = df[df["activity"] >= args.min_activity]
    print(f"[data] {n_raw:,} rows → {len(df):,} after min_activity≥{args.min_activity}"
          f"  ({df['team_id'].nunique():,} teams, {df['tournament_id'].nunique():,} tournaments)")

    residual = df["residual"].to_numpy(dtype=float)
    log_act  = np.log1p(df["activity"].to_numpy(dtype=float))

    # ------------------------------------------------------------------
    # 2. Global Spearman
    # ------------------------------------------------------------------
    print("\n=== Global Spearman(residual, log1p(activity)) ===")
    r_global, p_global = _spearman(log_act, residual)
    # Pre-rank once; bootstrap on Pearson of ranks = Spearman (fast)
    rx_global = stats.rankdata(log_act).astype(float)
    ry_global = stats.rankdata(residual).astype(float)
    lo, hi = _bootstrap_ci(rx_global, ry_global, n_boot=args.n_boot)
    print(f"  Global  {_fmt_r(r_global, p_global, lo, hi)}")

    # Mean residual by quartile
    q25, q75 = np.percentile(log_act, [25, 75])
    bot_mask = log_act <= q25
    top_mask = log_act >= q75
    mean_bot = float(residual[bot_mask].mean())
    mean_top = float(residual[top_mask].mean())
    print(f"  Q1 (activity ≤ p25) mean residual = {mean_bot:+.3f}  n={bot_mask.sum():,}")
    print(f"  Q4 (activity ≥ p75) mean residual = {mean_top:+.3f}  n={top_mask.sum():,}")
    print(f"  Q4 − Q1 gap = {mean_top - mean_bot:+.3f}")

    # ------------------------------------------------------------------
    # 3. By format
    # ------------------------------------------------------------------
    print("\n=== By format ===")
    for fmt in ("offline", "sync", "async"):
        mask = df["fmt"].to_numpy() == fmt
        if mask.sum() < 50:
            print(f"  {fmt:<8s} — skipped (n={mask.sum()})")
            continue
        x, y = log_act[mask], residual[mask]
        r, p = _spearman(x, y)
        rx = stats.rankdata(x).astype(float)
        ry = stats.rankdata(y).astype(float)
        lo, hi = _bootstrap_ci(rx, ry, n_boot=args.n_boot)
        print(f"  {fmt:<8s} n={mask.sum():>7,}  {_fmt_r(r, p, lo, hi)}")

    # ------------------------------------------------------------------
    # 4. Per-tournament: Q4−Q1 gap
    # ------------------------------------------------------------------
    print("\n=== Per-tournament pack-level Q4−Q1 activity quartile gap ===")
    records: list[dict] = []
    for tid, grp in df.groupby("tournament_id"):
        if len(grp) < 12:   # skip tiny tournaments
            continue
        la = np.log1p(grp["activity"].to_numpy(dtype=float))
        re = grp["residual"].to_numpy(dtype=float)
        q1_th = np.percentile(la, 25)
        q4_th = np.percentile(la, 75)
        bot = re[la <= q1_th]
        top = re[la >= q4_th]
        if len(bot) < 2 or len(top) < 2:
            continue
        gap = float(top.mean() - bot.mean())
        records.append({
            "tournament_id": int(tid),
            "fmt": grp["fmt"].iloc[0],
            "n_teams": len(grp),
            "gap_q4_minus_q1": gap,
        })

    if not records:
        print("  [warn] no tournaments passed the size filter")
        return 0

    gaps = np.array([r["gap_q4_minus_q1"] for r in records])
    print(f"  Tournaments analysed: {len(records):,}")
    print(f"  Median gap (Q4−Q1):   {float(np.median(gaps)):+.3f}")
    print(f"  Mean gap (Q4−Q1):     {float(np.mean(gaps)):+.3f}")
    print(f"  Std:                  {float(np.std(gaps)):.3f}")
    print(f"  Frac gap > 0:         {float((gaps > 0).mean()):.2%}")

    # quick ASCII histogram
    hist, edges = np.histogram(gaps, bins=12)
    max_bar = max(hist)
    scale = 40 / max(max_bar, 1)
    print("\n  Distribution of per-tournament Q4−Q1 gaps:")
    for i, cnt in enumerate(hist):
        bar = "█" * int(cnt * scale)
        print(f"  [{edges[i]:+5.2f}, {edges[i+1]:+5.2f})  {bar}  ({cnt})")

    # by format
    print("\n  Mean gap by format:")
    for fmt in ("offline", "sync", "async"):
        fg = [r["gap_q4_minus_q1"] for r in records if r["fmt"] == fmt]
        if fg:
            print(f"    {fmt:<8s} n={len(fg):4d}  mean={float(np.mean(fg)):+.3f}"
                  f"  median={float(np.median(fg)):+.3f}")

    # write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"\n[ok] pack-gap CSV → {out_path}  ({len(records)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
