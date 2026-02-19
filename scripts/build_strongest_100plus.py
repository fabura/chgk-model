#!/usr/bin/env python3
"""Build strongest_30plus_games.csv from players.csv. Keeps only players with >30 games in DB."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    results_dir = Path(__file__).resolve().parent.parent / "results"
    players_csv = results_dir / "players.csv"
    out_csv = results_dir / "strongest_30plus_games.csv"

    if not players_csv.exists():
        print(f"File not found: {players_csv}", file=sys.stderr)
        return 1

    # Load players.csv: player_id, theta, SE, rating
    players: list[tuple[int, float, float, float]] = []
    with open(players_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                pid = int(row["player_id"])
                theta = float(row["theta"])
                se = float(row.get("SE", "0"))
                rating = float(row.get("rating", str(theta)))
                players.append((pid, theta, se, rating))
            except (KeyError, ValueError):
                continue

    print(f"Loaded {len(players)} players from {players_csv}")

    try:
        import psycopg2
    except ImportError:
        print("pip install psycopg2-binary", file=sys.stderr)
        return 1

    url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
    player_ids = [p[0] for p in players]
    player_map = {p[0]: p for p in players}

    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()

        # Get games count and last_game per player from tournament_rosters + tournaments
        cur.execute(
            """
            SELECT tr.player_id,
                   COUNT(DISTINCT tr.tournament_id) AS games,
                   MAX(t.start_datetime::date) AS last_game
            FROM public.tournament_rosters tr
            JOIN public.tournaments t ON t.id = tr.tournament_id AND t.start_datetime IS NOT NULL
            WHERE tr.player_id = ANY(%s)
            GROUP BY tr.player_id
            """,
            (player_ids,),
        )
        db_stats = {r[0]: {"games": r[1], "last_game": r[2]} for r in cur.fetchall()}

        # Get first_name, last_name from players
        cur.execute(
            "SELECT id, first_name, last_name FROM public.players WHERE id = ANY(%s)",
            (player_ids,),
        )
        names = {r[0]: (r[1] or "", r[2] or "") for r in cur.fetchall()}

        conn.close()
    except Exception as e:
        print(f"DB error: {e}", file=sys.stderr)
        return 1

    # Build rows: only players with >30 games
    rows = []
    for pid, theta, se, rating in players:
        stats = db_stats.get(pid)
        if not stats or stats["games"] <= 30:
            continue
        first, last = names.get(pid, ("", ""))
        last_game = stats["last_game"]
        last_game_str = last_game.strftime("%Y-%m-%d") if last_game else ""
        rows.append((pid, first, last, theta, se, rating, stats["games"], last_game_str))

    # Sort by conservative rating (theta - c*SE) descending
    rows.sort(key=lambda r: -r[5])

    # Write output
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "player_id", "first_name", "last_name", "rating", "theta", "SE", "games", "last_game"])
        for i, (pid, first, last, theta, se, rating, games, last_game) in enumerate(rows, 1):
            w.writerow([i, pid, first, last, round(rating, 6), round(theta, 6), round(se, 6), games, last_game])

    print(f"Wrote {len(rows)} players (games > 30) to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
