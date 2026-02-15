#!/usr/bin/env python3
"""
Скрипт для оценки влияния порога min_games при текущей логике:
- игроки с < N игр удаляются из состава ("пустой стул"),
- команда отбрасывается только если после этого состав пустой.
По умолчанию N=10.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_cached


def get_player_games_from_db(player_ids: list[int]) -> dict[int, int]:
    try:
        import psycopg2
    except ImportError:
        return {}
    url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT player_id, COUNT(DISTINCT tournament_id)
            FROM public.tournament_rosters
            WHERE player_id = ANY(%s)
            GROUP BY player_id
            """,
            (player_ids,),
        )
        out = {r[0]: r[1] for r in cur.fetchall()}
        conn.close()
        return out
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate impact of removing low-game players from rosters")
    parser.add_argument("cache_file", nargs="?", default=None, help="Path to cache (required if no default)")
    parser.add_argument("--min_games", type=int, default=10, help="Threshold: players with fewer games are 'low-game' (default 10)")
    args = parser.parse_args()

    cache_path = args.cache_file
    if not cache_path:
        print("Usage: python count_low_game_impact.py <path_to_cache.pkl> [--min_games 10]", file=sys.stderr)
        print("Pass the same cache file you use for training (e.g. --cache_file in train.py).", file=sys.stderr)
        return 1
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        print(f"Cache not found: {cache_path}", file=sys.stderr)
        return 1

    print(f"Loading cache: {cache_path}")
    arrays, maps = load_cached(cache_path)
    q_idx = arrays["q_idx"]
    team_sizes = arrays["team_sizes"]
    player_indices_flat = arrays["player_indices_flat"]
    idx_to_player_id = maps.idx_to_player_id
    n_players = len(idx_to_player_id)
    n_samples = len(q_idx)

    offsets = np.zeros(n_samples + 1, dtype=np.int64)
    np.cumsum(team_sizes, out=offsets[1:])

    print(f"Players in cache: {n_players}, samples: {n_samples}")

    print("Querying DB for games per player...")
    db_games = get_player_games_from_db(list(idx_to_player_id))
    if not db_games:
        print("DB unavailable. Cannot compute games; run with DB to get real counts.", file=sys.stderr)
        return 1

    player_games = np.array([db_games.get(pid, 0) for pid in idx_to_player_id], dtype=np.int64)
    active = player_games >= args.min_games  # True = enough games
    n_low = (~active).sum()
    n_active = active.sum()
    print(f"\nPlayers with < {args.min_games} games: {n_low} ({100 * n_low / n_players:.1f}%)")
    print(f"Players with >= {args.min_games} games: {n_active} ({100 * n_active / n_players:.1f}%)")

    # Count samples dropped only when roster becomes empty after removing low-game players.
    # Also estimate compute impact by reduced total roster slots.
    print("\nCounting impact with 'remove low-game players, drop only empty teams' logic...")
    active_int = active.astype(np.int64)          # 1 if active, 0 if low-game
    active_per_slot = active_int[player_indices_flat]
    active_counts = np.add.reduceat(active_per_slot, offsets[:-1])  # active players per sample after filtering
    original_counts = team_sizes.astype(np.int64)

    dropped = int((active_counts == 0).sum())
    kept = n_samples - dropped
    pct_dropped = 100 * dropped / n_samples
    pct_kept = 100 * kept / n_samples

    total_slots_before = int(original_counts.sum())
    total_slots_after = int(active_counts.sum())
    slots_removed = total_slots_before - total_slots_after
    pct_slots_removed = 100 * slots_removed / max(1, total_slots_before)

    print(f"\nSamples DROPPED (became empty after removing <{args.min_games}): {dropped} ({pct_dropped:.1f}%)")
    print(f"Samples KEPT: {kept} ({pct_kept:.1f}%)")
    print(f"Total roster slots before: {total_slots_before}")
    print(f"Total roster slots after:  {total_slots_after}")
    print(f"Roster slots removed:      {slots_removed} ({pct_slots_removed:.1f}%)")

    if total_slots_after > 0:
        speedup_slots = total_slots_before / total_slots_after
        print(f"\nОценка ускорения по количеству игроков в командах: ~{speedup_slots:.2f}x")
        print(f"(и меньше игроков в модели: {n_active} вместо {n_players})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
