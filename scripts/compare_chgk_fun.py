#!/usr/bin/env python3
"""Compare team-level take forecasts: rating.chgk.fun vs our baked DuckDB.

rating.chgk.fun (AVSirotkin/ChGK) publishes pre-tournament expectations as
``predictedquestions`` per team (Elo-style: teams vs question difficulties).
Our ``team_games.expected_takes`` uses the same *pre-tournament* θ snapshot
(``theta_before`` at bake time in ``build_db.py``).

Data source for the external model:
  GET https://rating.chgk.fun/api/tournament_full/{tournament_id}
  (JSON despite text/html Content-Type)

Forecasts are read from ``data/chgk_fun_predictions.parquet`` (the
~3 MB committed bundle); only tournaments missing from the parquet
fall back to the JSON cache and the API.  After fetching new ones, run
``scripts/build_chgk_fun_predictions.py`` to refold the bundle.

Run from repo root:
  .venv/bin/python scripts/compare_chgk_fun.py
  .venv/bin/python scripts/compare_chgk_fun.py --limit 80 --since 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FUN_API = "https://rating.chgk.fun/api/tournament_full/{tid}"
DEFAULT_DUCKDB = _REPO / "website" / "data" / "chgk.duckdb"
DEFAULT_CACHE = _REPO / "data" / "chgk_fun_cache"
DEFAULT_PARQUET = _REPO / "data" / "chgk_fun_predictions.parquet"
# Model on chgk.fun is documented from 2021-09-02 onward.
FUN_MODEL_START = "2021-09-02"

# Lazy-loaded in-memory index built from ``data/chgk_fun_predictions.parquet``.
# Shape: {tournament_id: [ {teamid, predictedquestions, totalquestions, ...}, … ]}
# The downstream comparison code consumes ``data["tourresults"]`` so we mimic
# that shape exactly, allowing the parquet path to be a drop-in for the raw
# JSON cache without further code changes.
_PARQUET_INDEX: dict[int, list[dict]] | None = None


def _load_parquet_index(path: Path) -> dict[int, list[dict]]:
    """Load the compact parquet into a per-tournament dict."""
    global _PARQUET_INDEX
    if _PARQUET_INDEX is not None:
        return _PARQUET_INDEX
    if not path.is_file():
        _PARQUET_INDEX = {}
        return _PARQUET_INDEX
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        _PARQUET_INDEX = {}
        return _PARQUET_INDEX

    cols = ["tournament_id", "team_id", "place", "totalquestions",
            "predictedquestions", "mask", "teamrating"]
    table = pq.read_table(path, columns=cols)
    pylist = table.to_pylist()
    idx: dict[int, list[dict]] = {}
    for row in pylist:
        tid = int(row["tournament_id"])
        idx.setdefault(tid, []).append(
            {
                "teamid": int(row["team_id"]),
                "place": row["place"],
                "totalquestions": int(row["totalquestions"] or 0),
                "predictedquestions": float(row["predictedquestions"]),
                "mask": row["mask"],
                "teamrating": row["teamrating"],
            }
        )
    _PARQUET_INDEX = idx
    return idx


def _fetch_fun_tournament(
    tid: int,
    *,
    cache_dir: Path | None = None,
    parquet_path: Path | None = DEFAULT_PARQUET,
    timeout: float = 45.0,
) -> dict | None:
    """Resolve a tournament's per-team forecasts from rating.chgk.fun.

    Lookup order:
      1. compact parquet bundle (``data/chgk_fun_predictions.parquet``) —
         the only artefact committed to the repo;
      2. raw per-tournament JSON cache (``{cache_dir}/{tid}.json``) —
         developer-local, gitignored;
      3. live network fetch, persisted into the JSON cache so a follow-up
         ``scripts/build_chgk_fun_predictions.py`` can fold it into the
         parquet bundle.

    A network 404 is recorded as the sentinel string ``"NOT_FOUND"`` in
    the cache so repeated runs don't hammer the API for tournaments that
    rating.chgk.fun simply didn't process.
    """
    if parquet_path is not None:
        idx = _load_parquet_index(parquet_path)
        teams = idx.get(int(tid))
        if teams:
            return {"tourresults": teams}

    cache_path = (cache_dir / f"{tid}.json") if cache_dir else None
    if cache_path is not None and cache_path.is_file():
        raw = cache_path.read_text()
        if raw == "NOT_FOUND":
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cache_path.unlink(missing_ok=True)

    url = FUN_API.format(tid=tid)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            if cache_path is not None:
                cache_path.write_text("NOT_FOUND")
            return None
        raise
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(raw)
    return data


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y).correlation)
    except ImportError:
        # Rank with average ties — good enough for a script.
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rx = rx.astype(np.float64)
        ry = ry.astype(np.float64)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = np.sqrt((rx**2).sum() * (ry**2).sum())
        if denom <= 0:
            return float("nan")
        return float((rx * ry).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    parser.add_argument("--since", default=FUN_MODEL_START, help="start_date lower bound")
    parser.add_argument("--until", default=None, help="start_date upper bound (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=100, help="max tournaments to evaluate")
    parser.add_argument("--min-teams", type=int, default=6, help="skip tiny fields")
    parser.add_argument("--sleep", type=float, default=0.15, help="seconds between API calls")
    parser.add_argument("--out", type=Path, default=_REPO / "results" / "compare_chgk_fun.csv")
    args = parser.parse_args()

    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("duckdb required — use the project venv") from exc

    if not args.duckdb.is_file():
        raise SystemExit(f"DuckDB not found: {args.duckdb}")

    con = duckdb.connect(str(args.duckdb), read_only=True)
    sql = """
        SELECT t.tournament_id, t.title, t.start_date, t.type,
               COUNT(*) AS n_teams,
               COUNT(tg.expected_takes) AS n_with_expected
        FROM tournaments t
        JOIN team_games tg USING (tournament_id)
        WHERE t.start_date >= ?
          AND tg.expected_takes IS NOT NULL
          AND tg.score_actual IS NOT NULL
    """
    params: list = [args.since]
    if args.until:
        sql += " AND t.start_date <= ?"
        params.append(args.until)
    sql += """
        GROUP BY 1, 2, 3, 4
        HAVING n_teams >= ? AND n_with_expected = n_teams
        ORDER BY t.start_date DESC
        LIMIT ?
    """
    params.extend([args.min_teams, args.limit])
    candidates = con.execute(sql, params).fetchall()
    con.close()

    if not candidates:
        raise SystemExit("No tournaments matched filters in DuckDB")

    rows_out: list[dict] = []
    all_err_fun: list[float] = []
    all_err_ours: list[float] = []
    n_api_miss = 0
    n_no_overlap = 0

    print(
        f"Comparing up to {len(candidates)} tournaments "
        f"(since {args.since}, min_teams={args.min_teams})…"
    )

    for tid, title, start_date, ttype, n_teams, _ in candidates:
        tid = int(tid)
        fun = _fetch_fun_tournament(tid)
        time.sleep(args.sleep)
        if not fun or not fun.get("tourresults"):
            n_api_miss += 1
            continue

        fun_by_team = {int(t["teamid"]): t for t in fun["tourresults"]}

        con = duckdb.connect(str(args.duckdb), read_only=True)
        ours = con.execute(
            """
            SELECT team_id, score_actual, expected_takes, place
            FROM team_games
            WHERE tournament_id = ?
              AND expected_takes IS NOT NULL
              AND score_actual IS NOT NULL
            """,
            [tid],
        ).fetchall()
        con.close()

        actuals, pred_fun, pred_ours, places = [], [], [], []
        for team_id, actual, expected, place in ours:
            ft = fun_by_team.get(int(team_id))
            if ft is None:
                continue
            a = float(actual)
            pf = float(ft["predictedquestions"])
            po = float(expected)
            actuals.append(a)
            pred_fun.append(pf)
            pred_ours.append(po)
            if place is not None:
                places.append((po, pf, float(place)))

        if len(actuals) < args.min_teams:
            n_no_overlap += 1
            continue

        act = np.array(actuals)
        pf = np.array(pred_fun)
        po = np.array(pred_ours)
        err_f = np.abs(pf - act)
        err_o = np.abs(po - act)

        all_err_fun.extend(err_f.tolist())
        all_err_ours.extend(err_o.tolist())

        # Rank by predicted takes (higher → better place).
        rank_o = (-po).argsort().argsort() + 1
        rank_f = (-pf).argsort().argsort() + 1
        place_arr = np.array([p[2] for p in places]) if places else None
        sp_o = sp_f = float("nan")
        if place_arr is not None and len(place_arr) == len(rank_o):
            sp_o = _spearman(rank_o.astype(float), place_arr)
            sp_f = _spearman(rank_f.astype(float), place_arr)

        row = {
            "tournament_id": tid,
            "start_date": str(start_date),
            "type": ttype,
            "n_teams": len(actuals),
            "mae_fun": float(err_f.mean()),
            "mae_ours": float(err_o.mean()),
            "rmse_fun": float(np.sqrt((err_f**2).mean())),
            "rmse_ours": float(np.sqrt((err_o**2).mean())),
            "bias_fun": float((pf - act).mean()),
            "bias_ours": float((po - act).mean()),
            "spearman_place_fun": sp_f,
            "spearman_place_ours": sp_o,
            "fun_wins_teams": int((err_f < err_o).sum()),
            "ours_wins_teams": int((err_o < err_f).sum()),
        }
        rows_out.append(row)

    if not rows_out:
        raise SystemExit(
            f"No overlapping tournaments (api_miss={n_api_miss}, "
            f"no_overlap={n_no_overlap})"
        )

    ef = np.array(all_err_fun)
    eo = np.array(all_err_ours)

    print()
    print(f"Evaluated tournaments: {len(rows_out)}  (API miss: {n_api_miss})")
    print(f"Team observations:     {len(ef):,}")
    print()
    print("=== Pooled team-level error (|pred − actual takes|) ===")
    print(f"  MAE   fun {ef.mean():.4f}   ours {eo.mean():.4f}   Δ(ours−fun) {eo.mean()-ef.mean():+.4f}")
    print(f"  RMSE  fun {np.sqrt((ef**2).mean()):.4f}   ours {np.sqrt((eo**2).mean()):.4f}")
    print(f"  Per-team wins: fun {(ef < eo).sum():,}   ours {(eo < ef).sum():,}")
    print()
    print("=== Tournament-level MAE (mean over fields) ===")
    mae_f = np.array([r["mae_fun"] for r in rows_out])
    mae_o = np.array([r["mae_ours"] for r in rows_out])
    print(f"  mean MAE   fun {mae_f.mean():.4f}   ours {mae_o.mean():.4f}")
    print(f"  tournaments where ours beats fun: {(mae_o < mae_f).sum()} / {len(rows_out)}")
    print()
    sp_f = np.array([r["spearman_place_fun"] for r in rows_out])
    sp_o = np.array([r["spearman_place_ours"] for r in rows_out])
    print("=== Place ranking (Spearman predicted rank vs actual place) ===")
    print(f"  mean ρ   fun {np.nanmean(sp_f):.4f}   ours {np.nanmean(sp_o):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote per-tournament breakdown: {args.out}")


if __name__ == "__main__":
    main()
