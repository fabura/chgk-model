#!/usr/bin/env python3
"""Look up player names by ID from the rating DB. Usage: python scripts/lookup_players.py 288681 33194 ..."""
from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/lookup_players.py <player_id> [player_id ...]", file=sys.stderr)
        return 1
    ids = [int(x) for x in sys.argv[1:]]

    try:
        import psycopg2
    except ImportError:
        print("pip install psycopg2-binary", file=sys.stderr)
        return 1

    url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, last_name, first_name FROM public.players WHERE id = ANY(%s)",
        (ids,),
    )
    rows = cur.fetchall()
    conn.close()

    by_id = {r[0]: (r[1] or "", r[2] or "") for r in rows}
    for i, pid in enumerate(ids, 1):
        last, first = by_id.get(pid, ("?", "?"))
        print(f"  {i}. {last} {first} (id={pid})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
