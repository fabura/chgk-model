#!/usr/bin/env python3
"""Посмотреть в БД: сколько турниров с true_dl и когда самый ранний."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    try:
        import psycopg2
    except ImportError:
        print("pip install psycopg2-binary", file=sys.stderr)
        return 1

    url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(DISTINCT tournament_id)
            FROM public.true_dls
            WHERE tournament_id IS NOT NULL
        """)
        (n,) = cur.fetchone()
        print(f"Турниров с true_dl: {n}")

        cur.execute("""
            SELECT t.id, t.start_datetime, t.title
            FROM public.tournaments t
            WHERE t.id IN (SELECT DISTINCT tournament_id FROM public.true_dls WHERE tournament_id IS NOT NULL)
              AND t.start_datetime IS NOT NULL
            ORDER BY t.start_datetime ASC
            LIMIT 15
        """)
        rows = cur.fetchall()
        if not rows:
            print("Самых ранних по дате: нет (или у всех start_datetime NULL)")
        else:
            print("\nСамые ранние турниры с true_dl (по start_datetime):")
            for tid, start, title in rows:
                date_str = start.strftime("%Y-%m-%d") if start else "?"
                title_short = (title or "")[:50]
                print(f"  id={tid}  {date_str}  {title_short}")

        conn.close()
        return 0
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
