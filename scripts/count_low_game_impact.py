#!/usr/bin/env python3
"""
Скрипт для оценки: сколько игроков с < N играми и какая доля сэмплов будет отброшена,
если исключать из датасета все сэмплы, где в команде есть хотя бы один такой игрок.
По умолчанию N=10. Оценка ускорения: 1 / (1 - доля_отброшенных).
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
    parser = argparse.ArgumentParser(description="Count players with <N games and sample drop impact")
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

    # Count samples that contain at least one low-game player (would be dropped)
    # Vectorized: inv_active[player_indices_flat] then segment sum per sample via reduceat
    print("\nCounting samples that would be dropped (team has at least one low-game player)...")
    inv_active = (1 - active).astype(np.int64)  # 1 if low-game, 0 else
    values = inv_active[player_indices_flat]    # per appearance in flat
    segment_sums = np.add.reduceat(values, offsets[:-1])
    dropped = int((segment_sums > 0).sum())
    kept = n_samples - dropped
    pct_dropped = 100 * dropped / n_samples
    pct_kept = 100 * kept / n_samples

    print(f"\nSamples that would be DROPPED (team has any player with <{args.min_games} games): {dropped} ({pct_dropped:.1f}%)")
    print(f"Samples that would be KEPT: {kept} ({pct_kept:.1f}%)")

    if kept > 0:
        speedup = n_samples / kept
        print(f"\nОценка ускорения (при исключении этих сэмплов): в ~{speedup:.2f}x раз быстрее эпоха")
        print(f"(плюс меньше игроков в модели: {n_active} вместо {n_players})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
