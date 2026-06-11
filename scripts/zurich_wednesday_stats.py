#!/usr/bin/env python3
"""Zurich Wednesday stats from synch request dateStart — Telegram-ready output."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached  # noqa: E402

ZURICH = "(v.name = 'Цюрих' OR v.name LIKE 'Цюрих /%')"
PERIOD_START = date(2025, 1, 1)
EXCL_START = date(2025, 10, 16)
EXCL_END = date(2025, 10, 19)


def all_wednesdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() == 2:
            out.append(d)
        d += timedelta(days=1)
    return out


def _as_date(d) -> date:
    return d.date() if hasattr(d, "date") else d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chgk-db", type=Path, default=REPO_ROOT / "website/data/chgk.duckdb")
    ap.add_argument("--venue-db", type=Path, default=REPO_ROOT / "data/venue_overlay.duckdb")
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "data.npz")
    ap.add_argument("-o", "--out", type=Path, default=REPO_ROOT / "results/zurich_wednesday_telegram.md")
    args = ap.parse_args()

    import duckdb

    con = duckdb.connect()
    con.execute(f"ATTACH '{args.chgk_db}' AS site (READ_ONLY)")
    con.execute(f"ATTACH '{args.venue_db}' AS vo (READ_ONLY)")

    con.execute(f"""
    CREATE TEMP TABLE zurich_sessions AS
    SELECT
        tv.tournament_id, tv.venue_id, v.name AS venue_name,
        cast(sr.date_start AS DATE) AS play_date,
        t.title, tv.teams_played
    FROM vo.tournament_venues tv
    JOIN vo.venues v ON v.venue_id = tv.venue_id
    JOIN vo.synch_requests sr ON sr.synch_request_id = tv.synch_request_id
    JOIN site.tournaments t ON t.tournament_id = tv.tournament_id
    WHERE {ZURICH}
      AND sr.date_start IS NOT NULL
      AND cast(sr.date_start AS DATE) >= '{PERIOD_START}'
      AND NOT (cast(sr.date_start AS DATE) BETWEEN '{EXCL_START}' AND '{EXCL_END}')
      AND dayname(cast(sr.date_start AS DATE)) = 'Wednesday'
    """)

    n_sess = con.execute("SELECT count(*) FROM zurich_sessions").fetchone()[0]
    if n_sess == 0:
        print("No sessions — run: python scripts/backfill_synch_request_dates.py --zurich-only --since 2025-01-01", file=sys.stderr)
        sys.exit(1)

    period_end = _as_date(con.execute("SELECT max(play_date) FROM zurich_sessions").fetchone()[0])
    play_dates = {_as_date(r[0]) for r in con.execute("SELECT DISTINCT play_date FROM zurich_sessions").fetchall()}
    expected = all_wednesdays(PERIOD_START, period_end)
    skipped = [w for w in expected if w not in play_dates]

    overview = con.execute("""
    SELECT count(DISTINCT play_date), count(*), count(DISTINCT tournament_id), sum(teams_played)
    FROM zurich_sessions
    """).fetchone()

    con.execute("""
    CREATE TEMP TABLE games AS
    SELECT DISTINCT tg.tournament_id, tg.team_id, tg.team_name, ttv.venue_id,
           tg.score_actual, tg.expected_takes,
           tg.score_actual - tg.expected_takes AS residual,
           zs.play_date, zs.title
    FROM site.team_games tg
    JOIN vo.team_tournament_venue ttv
      ON ttv.tournament_id = tg.tournament_id AND ttv.team_id = tg.team_id
    JOIN zurich_sessions zs ON zs.tournament_id = tg.tournament_id AND zs.venue_id = ttv.venue_id
    """)
    con.execute("""
    CREATE TEMP TABLE ranked AS
    SELECT g.*, rank() OVER (PARTITION BY g.tournament_id, g.venue_id ORDER BY g.score_actual DESC) AS rnk
    FROM games g
    """)

    n_players = con.execute("""
    SELECT count(DISTINCT pg.player_id)
    FROM site.player_games pg
    JOIN games g ON g.tournament_id = pg.tournament_id AND g.team_id = pg.team_id
    """).fetchone()[0]
    n_games = con.execute("SELECT count(*) FROM games").fetchone()[0]
    avg_sc = con.execute("SELECT round(avg(score_actual),1), round(avg(residual),2) FROM games").fetchone()

    top_players = con.execute("""
    SELECT p.last_name, p.first_name,
           count(DISTINCT g.play_date || '-' || cast(g.venue_id AS varchar)) AS n
    FROM site.player_games pg
    JOIN games g ON g.tournament_id = pg.tournament_id AND g.team_id = pg.team_id
    JOIN site.players p ON p.player_id = pg.player_id
    GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 8
    """).fetchall()

    top_teams = con.execute("""
    SELECT max(team_name), count(*), sum(CASE WHEN rnk=1 THEN 1 ELSE 0 END),
           round(100.0*sum(CASE WHEN rnk=1 THEN 1 ELSE 0 END)/count(*),0)
    FROM ranked GROUP BY team_id HAVING count(*)>=5
    ORDER BY 2 DESC LIMIT 6
    """).fetchall()

    eligible = {int(r[0]) for r in con.execute("SELECT DISTINCT tournament_id FROM zurich_sessions").fetchall()}
    arrays, maps = load_cached(args.cache)
    q_idx, taken = arrays["q_idx"], arrays["taken"]
    game_idx = arrays["game_idx"] if "game_idx" in arrays else maps.question_game_idx[q_idx]
    player_flat, team_sizes = arrays["player_indices_flat"], arrays["team_sizes"]
    pid_to_pidx = maps.player_id_to_idx

    zteam_set: dict[int, set[int]] = {}
    for tid, team_id in con.execute("SELECT DISTINCT tournament_id, team_id FROM games").fetchall():
        zteam_set.setdefault(int(tid), set()).add(int(team_id))

    roster_to_team = {}
    for tid, team_id, pids in con.execute("""
        SELECT pg.tournament_id, pg.team_id, list_sort(list(pg.player_id))
        FROM site.player_games pg
        JOIN games g ON g.tournament_id = pg.tournament_id AND g.team_id = pg.team_id
        GROUP BY 1, 2
    """).fetchall():
        pidxs = tuple(sorted(pid_to_pidx.get(int(p), -1) for p in pids))
        roster_to_team[(int(tid), pidxs)] = int(team_id)

    qid_map = {
        i: (int(qid[0]), int(qid[1]))
        for i, qid in enumerate(maps.idx_to_question_id)
        if isinstance(qid, tuple) and qid[0] in eligible
    }
    take_counts: Counter = Counter()
    p_off = 0
    for si in range(len(q_idx)):
        g = int(game_idx[si])
        tid = maps.idx_to_game_id[g]
        ts = int(team_sizes[si])
        pidxs = tuple(sorted(int(player_flat[p_off + j]) for j in range(ts)))
        p_off += ts
        if tid not in eligible or taken[si] != 1:
            continue
        qi_global = int(q_idx[si])
        if qi_global not in qid_map:
            continue
        team_id = roster_to_team.get((tid, pidxs))
        if team_id is not None and team_id in zteam_set.get(tid, set()):
            take_counts[qid_map[qi_global]] += 1

    hardest = []
    for (tid, qi), cnt in take_counts.items():
        row = con.execute("""
            SELECT q.b, q.text, q.answer, qa.n_obs, qa.n_taken, t.title
            FROM site.question_aliases qa
            JOIN site.questions q ON q.canonical_idx = qa.canonical_idx
            JOIN site.tournaments t ON t.tournament_id = qa.tournament_id
            WHERE qa.tournament_id=? AND qa.q_in_tournament=?
        """, [tid, qi]).fetchone()
        if row:
            hardest.append((row[0], cnt, row[5], qi, row[1] or "", row[2] or ""))
    hardest.sort(key=lambda x: (-x[0], -x[1]))

    double_nights = con.execute("""
    SELECT play_date, count(*) AS n, string_agg(title, ' / ') AS titles
    FROM zurich_sessions GROUP BY play_date HAVING count(*)>1 ORDER BY play_date
    """).fetchall()

    lines = [
        "📊 **Цюрих — статистика сред с 2025**",
        "_даты из заявок на синхрон (dateStart), без 16–19 окт 2025_",
        "",
        f"**{PERIOD_START.strftime('%d.%m.%Y')} — {period_end.strftime('%d.%m.%Y')}**",
        "",
        f"🗓 Игровых сред: **{overview[0]}** из {len(expected)} ({100*overview[0]/len(expected):.0f}%)",
        f"⏭ Пропущено: **{len(skipped)}**",
        f"🎯 Сессий на площадках: **{overview[1]}** · турниров: **{overview[2]}**",
        f"👥 Уникальных игроков: **{n_players}** · игр команд: **{n_games}**",
        f"📈 Ср. результат: **{avg_sc[0]}** (Δ vs модель: **{avg_sc[1]:+.2f}**)",
        "",
        "**Топ игроков**",
    ]
    for i, (ln, fn, n) in enumerate(top_players, 1):
        lines.append(f"{i}. {ln} {fn} — **{n}**")

    lines += ["", "**Топ команд** _(≥5 игр)_"]
    for team, n, wins, pct in top_teams:
        lines.append(f"• **{team}** — {n} игр, {int(wins)} побед ({int(pct)}%)")

    lines += ["", "**Самые сложные взятые вопросы**"]
    for b, cnt, title, qi, text, ans in hardest[:5]:
        lines.append(f"• b={b:.1f} · {cnt} ком. · _{title[:38]}_ Q{qi+1}")
        if text.strip():
            lines.append(f"  {text.strip().replace(chr(10), ' ')[:85]}…")
        if ans.strip():
            lines.append(f"  → {ans.strip()[:45]}")

    if skipped:
        lines += ["", f"**Пропуски** ({len(skipped)}):"]
        lines.append(", ".join(d.strftime("%d.%m.%y") for d in skipped))

    if double_nights:
        lines += ["", "**Двойные вечера:**"]
        for pd, _n, titles in double_nights:
            d = _as_date(pd)
            lines.append(f"• {d.strftime('%d.%m.%Y')}: {titles[:75]}")

    lines += ["", "—", "_Источник: api.rating.chgk.info · chgk-model_"]

    text = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
