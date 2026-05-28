#!/usr/bin/env python3
"""Pack the rating.chgk.fun JSON cache into a compact parquet.

The raw per-tournament JSON (``data/chgk_fun_cache/{tid}.json``) carries
~40 KB each (tournament_info + leg_dict + fulljson Russian-escape blob),
but our comparison scripts only need a few per-team fields.  This script
collapses everything into one zstd-compressed parquet file:

  data/chgk_fun_predictions.parquet
    tournament_id  int32   (filename)
    team_id        int32   (tourresults[].teamid)
    place          float32 (tourresults[].place, may be NULL/half)
    totalquestions int32   (actual takes — sanity check vs our DuckDB)
    predictedquestions float64  (THE forecast we compare against)
    mask           string  (per-question 0/1, kept for future per-Q work)
    teamrating     float32 (their Elo-style team strength)

Drops: tournament_info, leg_dict, fulljson, teamname (in our DuckDB),
atleastprob/atmostprob/teamperformance (we don't use).

The compact parquet is small enough to commit (~1–2 MB vs 65 MB of JSON);
the JSON cache directory should be gitignored.

Usage:
  .venv/bin/python scripts/build_chgk_fun_predictions.py
  .venv/bin/python scripts/build_chgk_fun_predictions.py --cache-dir data/chgk_fun_cache --out data/chgk_fun_predictions.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = _REPO / "data" / "chgk_fun_cache"
DEFAULT_OUT = _REPO / "data" / "chgk_fun_predictions.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--compression", default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
    )
    args = ap.parse_args()

    if not args.cache_dir.is_dir():
        raise SystemExit(f"Cache dir not found: {args.cache_dir}")

    files = sorted(args.cache_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {args.cache_dir}")

    tids: list[int] = []
    team_ids: list[int] = []
    places: list[float | None] = []
    actuals: list[int] = []
    preds: list[float] = []
    masks: list[str | None] = []
    ratings: list[float | None] = []

    skipped_404 = 0
    skipped_bad = 0
    total_bytes = 0

    for path in files:
        total_bytes += path.stat().st_size
        raw = path.read_text()
        if raw == "NOT_FOUND":
            skipped_404 += 1
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            skipped_bad += 1
            continue
        rows = data.get("tourresults") or []
        for r in rows:
            try:
                tids.append(int(r["teamid"] and int(path.stem)))
                team_ids.append(int(r["teamid"]))
                places.append(
                    float(r["place"]) if r.get("place") is not None else None
                )
                actuals.append(int(r.get("totalquestions") or 0))
                preds.append(float(r["predictedquestions"]))
                masks.append(r.get("mask"))
                tr = r.get("teamrating")
                ratings.append(float(tr) if tr is not None else None)
            except (KeyError, TypeError, ValueError):
                continue

    schema = pa.schema([
        ("tournament_id", pa.int32()),
        ("team_id", pa.int32()),
        ("place", pa.float32()),
        ("totalquestions", pa.int32()),
        ("predictedquestions", pa.float64()),
        ("mask", pa.string()),
        ("teamrating", pa.float32()),
    ])
    table = pa.table(
        {
            "tournament_id": pa.array(tids, type=pa.int32()),
            "team_id": pa.array(team_ids, type=pa.int32()),
            "place": pa.array(places, type=pa.float32()),
            "totalquestions": pa.array(actuals, type=pa.int32()),
            "predictedquestions": pa.array(preds, type=pa.float64()),
            "mask": pa.array(masks, type=pa.string()),
            "teamrating": pa.array(ratings, type=pa.float32()),
        },
        schema=schema,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    comp = None if args.compression == "none" else args.compression
    pq.write_table(table, args.out, compression=comp)

    out_bytes = args.out.stat().st_size
    n_tourns = len({int(t) for t in tids})
    print(f"Read {len(files)} JSON files ({total_bytes/1024/1024:.1f} MB)")
    print(f"  skipped 404: {skipped_404}, malformed: {skipped_bad}")
    print(f"Packed {len(tids):,} team rows from {n_tourns} tournaments")
    print(f"Wrote {args.out} ({out_bytes/1024:.1f} KB, "
          f"compression={args.compression}; "
          f"{out_bytes / max(total_bytes, 1) * 100:.1f}% of JSON)")


if __name__ == "__main__":
    main()
