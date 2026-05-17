"""Pull observations from PG → .npz cache, parameterised by min_games.

Used by exp_min_games_cold_grid.py to prepare per-min_games caches once,
so the sweep doesn't re-pull PG on every cell.
"""
from __future__ import annotations

import argparse
import sys

from data import load_from_db, save_cached, _samples_to_arrays, _save_arrays_maps_npz
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-games", type=int, required=True)
    ap.add_argument("--cache-file", type=str, required=True)
    ap.add_argument(
        "--min-tournament-date", type=str, default="2015-01-01"
    )
    args = ap.parse_args()

    print(
        f"Pulling from PG with min_games={args.min_games} → "
        f"{args.cache_file}",
        flush=True,
    )
    samples, maps = load_from_db(
        min_games=args.min_games,
        min_tournament_date=args.min_tournament_date,
    )
    print(
        f"Got {len(samples):,} samples, {maps.num_players:,} players, "
        f"{maps.num_questions:,} questions",
        flush=True,
    )
    # Force NPZ format regardless of file extension (save_cached
    # picks pkl vs npz from the suffix and our sweep paths look like
    # data.mg0.npz where save_cached's heuristic still works only if
    # the LAST suffix is .npz — be explicit to be safe).
    arrays = _samples_to_arrays(samples)
    _save_arrays_maps_npz(arrays, maps, Path(args.cache_file))
    print(f"Wrote {args.cache_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
