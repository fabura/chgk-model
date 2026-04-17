#!/usr/bin/env python3
"""Build strongest_30plus_games.csv from seq.npz (or players.csv). Keeps only players with ≥30 games."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strongest_30plus_games.csv")
    parser.add_argument(
        "--input",
        choices=["seq", "players"],
        default=None,
        help="Source: seq=seq.npz (sequential model), players=players.csv. Auto-detect if not set.",
    )
    parser.add_argument("--min_games", type=int, default=30, help="Min games to include. Default 30.")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path. Default results/strongest_30plus_games.csv")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    seq_npz = results_dir / "seq.npz"
    players_csv = results_dir / "players.csv"
    out_csv = Path(args.out) if args.out else results_dir / "strongest_30plus_games.csv"

    # Choose input
    use_seq = args.input == "seq" or (args.input is None and seq_npz.exists())
    if use_seq and seq_npz.exists():
        from rating import load_results_npz

        r = load_results_npz(seq_npz)
        mask = r.games >= args.min_games
        order = np.argsort(r.theta)[::-1]
        players = [
            (int(r.player_id[i]), float(r.theta[i]), 0.0, float(r.theta[i]), int(r.games[i]))
            for i in order
            if mask[i]
        ]
        print(f"Loaded {len(players)} players (≥{args.min_games} games) from {seq_npz}")
    elif players_csv.exists():
        players = []
        with open(players_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    pid = int(row["player_id"])
                    theta = float(row["theta"])
                    se = float(row.get("SE", "0"))
                    rating = float(row.get("rating", str(theta)))
                    games = int(row.get("num_games", row.get("games", 0)))
                    players.append((pid, theta, se, rating, games))
                except (KeyError, ValueError):
                    continue
        players = [p for p in players if p[4] >= args.min_games]
        players.sort(key=lambda p: -p[3])  # by rating
        print(f"Loaded {len(players)} players (≥{args.min_games} games) from {players_csv}")
    else:
        print(f"No input found. Need {seq_npz} or {players_csv}", file=sys.stderr)
        return 1

    if not players:
        print("No players with enough games.", file=sys.stderr)
        return 1

    try:
        import psycopg2
    except ImportError:
        print("pip install psycopg2-binary", file=sys.stderr)
        return 1

    player_ids = [p[0] for p in players]
    url = "postgresql://postgres:password@127.0.0.1:5432/postgres"
    import os

    url = os.environ.get("DATABASE_URL", url)

    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()

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

        cur.execute(
            "SELECT id, first_name, last_name FROM public.players WHERE id = ANY(%s)",
            (player_ids,),
        )
        names = {r[0]: (r[1] or "", r[2] or "") for r in cur.fetchall()}

        conn.close()
    except Exception as e:
        print(f"DB error: {e}", file=sys.stderr)
        return 1

    rows = []
    for pid, theta, se, rating, games_in in players:
        stats = db_stats.get(pid)
        first, last = names.get(pid, ("", ""))
        last_game = stats["last_game"] if stats else None
        last_game_str = last_game.strftime("%Y-%m-%d") if last_game else ""
        games = stats["games"] if stats and stats["games"] else games_in
        rows.append((pid, first, last, rating, theta, se, games, last_game_str))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "player_id", "first_name", "last_name", "rating", "theta", "SE", "games", "last_game"])
        for i, (pid, first, last, rating, theta, se, games, last_game) in enumerate(rows, 1):
            w.writerow([i, pid, first, last, round(rating, 6), round(theta, 6), round(se, 6), games, last_game])

    print(f"Wrote {len(rows)} players to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
