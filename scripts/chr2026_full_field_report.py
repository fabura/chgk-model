#!/usr/bin/env python3
"""Full field report for upcoming ЧР (default API event 12826)."""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import duckdb

from website.app import forecast_api
from website.app.forecast import forecast_for_event

EVENT_ID = 12826
LAST_CHR_ID = 11749
ZIGA_CHR_ID = 12459
MCHR_ID = 13717
SINCE_DATE = "2025-06-14"


@dataclass
class TeamRoster:
    team_id: int
    team_name: str
    town: str
    player_ids: list[int] = field(default_factory=list)


def fetch_rosters(event_id: int) -> list[TeamRoster]:
    out: list[TeamRoster] = []
    for row in forecast_api.get_rosters(event_id):
        team = row.get("team") or {}
        tid = team.get("id")
        if not isinstance(tid, int):
            continue
        pids = [
            int(tm["player"]["id"])
            for tm in (row.get("teamMembers") or [])
            if isinstance((tm.get("player") or {}).get("id"), int)
        ]
        out.append(
            TeamRoster(
                team_id=int(tid),
                team_name=str(team.get("name") or f"#{tid}"),
                town=str((team.get("town") or {}).get("name") or ""),
                player_ids=pids,
            )
        )
    return out


def top_sets(con: duckdb.DuckDBPyConnection, col: str = "theta_display") -> dict[int, set[int]]:
    rows = con.execute(
        f"SELECT player_id FROM players WHERE {col} IS NOT NULL "
        f"ORDER BY {col} DESC, games DESC, player_id"
    ).fetchall()
    ordered = [int(r[0]) for r in rows]
    return {n: set(ordered[:n]) for n in (10, 100, 1000, 10000)}


def player_rows(con: duckdb.DuckDBPyConnection, pids: set[int]) -> dict[int, dict]:
    if not pids:
        return {}
    ph = ",".join("?" * len(pids))
    out: dict[int, dict] = {}
    for r in con.execute(
        f"SELECT player_id, last_name, first_name, theta, theta_display, games, last_game_date "
        f"FROM players WHERE player_id IN ({ph})",
        sorted(pids),
    ).fetchall():
        out[int(r[0])] = {
            "last_name": r[1],
            "first_name": r[2],
            "theta": float(r[3] or 0),
            "theta_display": float(r[4] or 0),
            "games": int(r[5] or 0),
            "last_game_date": r[6],
        }
    return out


def name_of(pid: int, pmap: dict[int, dict]) -> str:
    p = pmap.get(pid, {})
    return f"{p.get('last_name', '?')} {p.get('first_name', '?')}"


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "—"


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", type=int, default=EVENT_ID)
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--no-forecast", action="store_true")
    args = ap.parse_args()

    teams = fetch_rosters(args.event_id)
    all_pids = {p for t in teams for p in t.player_ids}
    con = duckdb.connect(str(REPO_ROOT / args.duckdb), read_only=True)
    pmap = player_rows(con, all_pids)
    tops = top_sets(con)

    last_chr = con.execute(
        "SELECT title, start_date, end_date, n_teams FROM tournaments WHERE tournament_id=?",
        [LAST_CHR_ID],
    ).fetchone()

    section("0. Поле")
    print(f"Событие #{args.event_id}: {len(teams)} команд, {len(all_pids)} игроков")
    print(f"Прошлый ЧР #{LAST_CHR_ID}: «{last_chr[0]}» ({last_chr[1]}, {last_chr[2]}), {last_chr[3]} команд")
    print(f"Окно активности: турниры с start_date > {SINCE_DATE}")

    # --- 1. Top-N ---
    section("1. Игроки из топ-N (theta_display)")
    for n in (10, 100, 1000, 10000):
        hit = all_pids & tops[n]
        print(f"  топ-{n:>5}: {len(hit):>3} ({pct(len(hit), len(all_pids))} поля, {len(hit)/len(teams):.2f}/команду)")
    hit10 = sorted(all_pids & tops[10], key=lambda p: -pmap[p]["theta_display"])
    print("  топ-10 в поле:", ", ".join(name_of(p, pmap) for p in hit10))

    teams_top100 = sum(1 for t in teams if any(p in tops[100] for p in t.player_ids))
    teams_2top100 = sum(1 for t in teams if sum(1 for p in t.player_ids if p in tops[100]) >= 2)
    print(f"  команд с ≥1 из топ-100: {teams_top100}/{len(teams)}; с ≥2: {teams_2top100}/{len(teams)}")

    # --- 2. Team strength ---
    section("2. Сила составов (theta_display)")
    team_stats = []
    for t in teams:
        thetas = sorted((pmap[p]["theta_display"] for p in t.player_ids if p in pmap), reverse=True)
        top6 = thetas[:6]
        if not top6:
            continue
        s6 = sum(top6)
        gap = top6[0] - top6[-1] if len(top6) >= 2 else 0.0
        top2_share = sum(top6[:2]) / s6 if s6 > 0 else 0
        team_stats.append((s6, gap, top2_share, t.team_name, top6[0]))
    team_stats.sort(reverse=True)
    sums = [x[0] for x in team_stats]
    gaps = [x[1] for x in team_stats]
    conc = [x[2] for x in team_stats]
    print(f"  Σθ top-6: среднее {statistics.mean(sums):.2f}, медиана {statistics.median(sums):.2f}, "
          f"мин/макс {min(sums):.2f}/{max(sums):.2f}")
    print(f"  разрыв 1-й/6-й: среднее {statistics.mean(gaps):.2f}, медиана {statistics.median(gaps):.2f}")
    print(f"  доля top-2 в Σθ: среднее {statistics.mean(conc):.1%}")
    print("  топ-5 по Σθ:", ", ".join(f"{n} ({s:.1f})" for s, _, _, n, _ in team_stats[:5]))
    print("  самые ровные (малый разрыв):", ", ".join(
        f"{n} ({g:.2f})" for _, g, _, n, _ in sorted(team_stats, key=lambda x: x[1])[:5]
    ))

    # --- 3. Activity ---
    section("3. Активность с прошлого ЧР")
    ph = ",".join("?" * len(all_pids))
    gs_rows = con.execute(
        f"""
        SELECT pg.player_id, COUNT(DISTINCT pg.tournament_id) AS n,
               SUM(CASE WHEN t.type='offline' THEN 1 ELSE 0 END) AS n_off,
               SUM(CASE WHEN t.type='sync' THEN 1 ELSE 0 END) AS n_sync,
               SUM(CASE WHEN t.type='async' THEN 1 ELSE 0 END) AS n_async
        FROM player_games pg
        JOIN tournaments t ON t.tournament_id = pg.tournament_id
        WHERE pg.player_id IN ({ph}) AND t.start_date > ?
        GROUP BY 1
        """,
        sorted(all_pids) + [SINCE_DATE],
    ).fetchall()
    gs = {int(r[0]): int(r[1]) for r in gs_rows}
    mode_split = {int(r[0]): (int(r[2] or 0), int(r[3] or 0), int(r[4] or 0)) for r in gs_rows}
    vals = [gs.get(p, 0) for p in all_pids]
    print(f"  турниров/игрок: среднее {statistics.mean(vals):.1f}, медиана {statistics.median(vals):.0f}, "
          f"мин/макс {min(vals)}/{max(vals)}, без игр {sum(v==0 for v in vals)}")
    q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else []
    if q:
        print(f"  квартили: Q1={q[0]:.0f} Q2={q[1]:.0f} Q3={q[2]:.0f}")

    today = date.today()
    inactive = []
    for p in all_pids:
        ld = pmap[p].get("last_game_date")
        if ld:
            inactive.append((today - ld, p))
    inactive.sort(reverse=True)
    print(f"  дней с последней игры: медиана {statistics.median([x[0].days for x in inactive]):.0f}, "
          f"макс {inactive[0][0].days} ({name_of(inactive[0][1], pmap)})")

    off = sum(m[0] for m in mode_split.values())
    sync = sum(m[1] for m in mode_split.values())
    async_ = sum(m[2] for m in mode_split.values())
    tot_mode = off + sync + async_ or 1
    print(f"  режимы (турнир-участий): очник {pct(off, tot_mode)}, синхрон {pct(sync, tot_mode)}, "
          f"асинхрон {pct(async_, tot_mode)}")

    hist = {
        int(r[0]): float(r[1])
        for r in con.execute(
            f"SELECT player_id, theta FROM player_history WHERE tournament_id=? "
            f"AND player_id IN ({ph})",
            [LAST_CHR_ID] + sorted(all_pids),
        ).fetchall()
        if r[1] is not None
    }
    by_delta = sorted(
        ((pmap[p]["theta_display"] - hist[p], p) for p in all_pids if p in hist),
        reverse=True,
    )
    deltas = [d for d, _ in by_delta]
    print(f"  Δθ с прошлого ЧР (n={len(deltas)}): среднее {statistics.mean(deltas):+.3f}, "
          f"медиана {statistics.median(deltas):+.3f}")
    if by_delta:
        print("  рост θ:", ", ".join(f"{name_of(p,pmap)} ({d:+.2f})" for d, p in by_delta[:3]))
        print("  падение θ:", ", ".join(f"{name_of(p,pmap)} ({d:+.2f})" for d, p in by_delta[-3:]))

    # --- 4. Chemistry ---
    section("4. Химия и стабильность составов")
    pair_counts: list[int] = []
    for t in teams:
        pids = t.player_ids
        for i, a in enumerate(pids):
            for b in pids[i + 1 :]:
                n = con.execute(
                    """
                    SELECT COUNT(DISTINCT a.tournament_id)
                    FROM player_games a
                    JOIN player_games b ON a.tournament_id=b.tournament_id AND a.team_id=b.team_id
                    JOIN tournaments t ON t.tournament_id=a.tournament_id
                    WHERE a.player_id=? AND b.player_id=? AND t.start_date>?
                    """,
                    [a, b, SINCE_DATE],
                ).fetchone()[0]
                pair_counts.append(int(n or 0))
    print(f"  пар в заявках: {len(pair_counts)}")
    print(f"  турниров вместе (пара): среднее {statistics.mean(pair_counts):.1f}, медиана {statistics.median(pair_counts):.0f}")
    for thr in (0, 1, 5, 10, 20):
        c = sum(1 for x in pair_counts if x >= thr)
        print(f"  пар с ≥{thr} совместных турниров: {c} ({pct(c, len(pair_counts))})")

    # legionnaires
    shared: dict[int, int] = defaultdict(int)
    for t in teams:
        rs = set(t.player_ids)
        if len(rs) < 2:
            continue
        ph2 = ",".join("?" * len(rs))
        for _tid, _team, pids in con.execute(
            f"""
            SELECT pg.tournament_id, pg.team_id, list(DISTINCT pg.player_id)
            FROM player_games pg JOIN tournaments tt ON tt.tournament_id=pg.tournament_id
            WHERE pg.player_id IN ({ph2}) AND tt.start_date>?
            GROUP BY 1,2 HAVING count(DISTINCT pg.player_id)>=2
            """,
            sorted(rs) + [SINCE_DATE],
        ).fetchall():
            present = set(int(x) for x in pids) & rs
            for pid in present:
                shared[pid] += 1
    legion = [p for p in all_pids if shared.get(p, 0) <= 1]
    print(f"  легионеры (≤1 турнир с текущими): {len(legion)} ({pct(len(legion), len(all_pids))})")

    # --- 5. CHR 2025 overlap ---
    section("5. Связь с прошлым ЧР (#11749)")
    last_pids = {
        int(r[0])
        for r in con.execute(
            "SELECT DISTINCT player_id FROM player_games WHERE tournament_id=?",
            [LAST_CHR_ID],
        ).fetchall()
    }
    overlap = all_pids & last_pids
    print(f"  из прошлогоднего поля заявлены: {len(overlap)}/{len(last_pids)} ({pct(len(overlap), len(last_pids))})")
    print(f"  дебютанты серии (не играли #11749): {len(all_pids - last_pids)}")
    debut_teams = [t.team_name for t in teams if not any(p in last_pids for p in t.player_ids)]
    print(f"  команд без участника прошлого ЧР: {len(debut_teams)}")

    # roster churn per team (match by player overlap with 2025 team at CHR)
    last_team_rosters: dict[int, set[int]] = {}
    for tid, in con.execute(
        "SELECT DISTINCT team_id FROM team_games WHERE tournament_id=?", [LAST_CHR_ID]
    ).fetchall():
        pset = {
            int(r[0])
            for r in con.execute(
                "SELECT player_id FROM player_games WHERE tournament_id=? AND team_id=?",
                [LAST_CHR_ID, tid],
            ).fetchall()
        }
        last_team_rosters[int(tid)] = pset

    churn_stats = []
    for t in teams:
        cur = set(t.player_ids)
        best = max(
            (len(cur & s), len(s), s) for s in last_team_rosters.values() if cur & s
        ) if any(cur & s for s in last_team_rosters.values()) else (0, 0, set())
        inter, _, prev = best
        if inter == 0:
            churn_stats.append((len(cur), 0, len(cur), t.team_name, "новая команда"))
        else:
            churn_stats.append((inter, inter, len(cur - prev), t.team_name, f"{inter} общих"))
    retained = [x[0] for x in churn_stats if x[0] > 0]
    joined = [x[2] for x in churn_stats if x[0] > 0]
    print(f"  при совпадении с прошлогодним составом: удержано игроков среднее {statistics.mean(retained):.1f}, "
          f"новых {statistics.mean(joined):.1f} (n={len(retained)} команд)")

    # --- 6. Other tournaments ---
    section("6. Пересечения с другими турнирами")
    refs = [
        (ZIGA_CHR_ID, "Зига-ЧР 2025"),
        (MCHR_ID, "МЧР-2026"),
    ]
    for tid, label in refs:
        trow = con.execute(
            "SELECT title, start_date FROM tournaments WHERE tournament_id=?", [tid]
        ).fetchone()
        if not trow:
            continue
        played = {
            int(r[0])
            for r in con.execute(
                "SELECT DISTINCT player_id FROM player_games WHERE tournament_id=?", [tid]
            ).fetchall()
        }
        hit = all_pids & played
        print(f"  {label} (#{tid}): {len(hit)} игроков ({pct(len(hit), len(all_pids))}), "
              f"{sum(1 for t in teams if any(p in played for p in t.player_ids))} команд")

    # --- 7. Geography ---
    section("7. География (город команды в API)")
    towns = Counter(t.town or "?" for t in teams)
    print("  топ городов:", ", ".join(f"{k} ({v})" for k, v in towns.most_common(8)))
    msk = sum(1 for t in teams if "Москва" in t.town)
    spb = sum(1 for t in teams if "Петербург" in t.town or t.town == "Санкт-Петербург")
    print(f"  Москва: {msk}, СПб: {spb}, прочие: {len(teams)-msk-spb}")

    # --- 8. Bench / roster size ---
    section("8. Заявки")
    sizes = [len(t.player_ids) for t in teams]
    print(f"  размер: среднее {statistics.mean(sizes):.2f}, медиана {statistics.median(sizes):.0f}, "
          f"мин/макс {min(sizes)}/{max(sizes)}")
    big = [t for t in teams if len(t.player_ids) > 6]
    if big:
        print(f"  >6 игроков: {len(big)} команд")
        for t in sorted(big, key=lambda x: -len(x.player_ids))[:5]:
            thetas = sorted((pmap[p]["theta_display"] for p in t.player_ids), reverse=True)
            print(f"    {t.team_name}: {len(t.player_ids)} чел., 7-й θ={thetas[6]:+.2f}" if len(thetas)>6 else "")

    # --- 9. Network: players on multiple strong teams ---
    section("9. Гостевые связи / концентрация звёзд")
    # players in top-1000 who appear - each only on one team in registration
    # but count how many top-500 played with many distinct teammates in strong events
    star_spread = []
    for p in all_pids & tops[1000]:
        n_teams = con.execute(
            f"""
            SELECT COUNT(DISTINCT pg.team_id) FROM player_games pg
            JOIN tournaments t ON t.tournament_id=pg.tournament_id
            WHERE pg.player_id=? AND t.start_date>? AND t.type='offline'
            """,
            [p, SINCE_DATE],
        ).fetchone()[0]
        star_spread.append(int(n_teams or 0))
    if star_spread:
        print(f"  топ-1000: офлайн-команд с прошлого ЧР на игрока — среднее {statistics.mean(star_spread):.1f}, "
              f"макс {max(star_spread)}")

    # --- 10. Forecast ---
    fc = None
    if not args.no_forecast:
        section("10. Прогноз (пакет #11749, MC=500)")
        fc = forecast_for_event(args.event_id, pack_kind="past", pack_id=LAST_CHR_ID, n_mc_samples=500)
        if fc and fc.get("teams"):
            ft = fc["teams"]
            print(f"  топ-5 E[T]:", ", ".join(
                f"{r['team_name']} ({r['expected_takes']:.1f})" for r in ft[:5]
            ))
            widths = [r.get("place_q95", 0) - r.get("place_q05", 0) for r in ft if r.get("place_q95")]
            if widths:
                print(f"  ширина MC-места (p95-p05): среднее {statistics.mean(widths):.0f}, "
                      f"макс {max(widths):.0f}")
            volatile = sorted(ft, key=lambda r: -(r.get("place_q95", 0) - r.get("place_q05", 0)))[:3]
            print("  самые «волатильные» места:", ", ".join(
                f"{r['team_name']} ({r.get('place_q05',0):.0f}–{r.get('place_q95',0):.0f})" for r in volatile
            ))

    # --- 11. Absent stars ---
    section("11. Сильнейшие из прошлогоднего ЧР без заявки")
    absent = []
    for r in con.execute(
        """
        SELECT p.player_id, p.last_name, p.first_name, p.theta_display, tg.team_name
        FROM player_games pg
        JOIN players p ON p.player_id=pg.player_id
        LEFT JOIN team_games tg ON tg.tournament_id=pg.tournament_id AND tg.team_id=pg.team_id
        WHERE pg.tournament_id=?
        ORDER BY p.theta_display DESC
        """,
        [LAST_CHR_ID],
    ).fetchall():
        pid = int(r[0])
        if pid not in all_pids and pid not in {a[0] for a in absent}:
            absent.append((pid, r[1], r[2], float(r[3]), r[4]))
    print("  топ-5:", ", ".join(f"{a[1]} {a[2]} ({a[3]:+.2f}, был в {a[4]})" for a in absent[:5]))

    ziga_hit = {
        int(r[0])
        for r in con.execute(
            "SELECT DISTINCT player_id FROM player_games WHERE tournament_id=?",
            [ZIGA_CHR_ID],
        ).fetchall()
    }

    # --- CONCLUSIONS ---
    section("ВЫВОДЫ (компактно)")
    conclusions = [
        f"Поле выросло: {last_chr[3]}→{len(teams)} команд, {len(last_pids)}→{len(all_pids)} игроков в заявках.",
        f"Преемственность высокая: {pct(len(overlap), len(last_pids))} прошлогодних участников вернулись ({len(overlap)}/{len(last_pids)}).",
        f"Дебютантов серии {len(all_pids - last_pids)} ({pct(len(all_pids - last_pids), len(all_pids))}); {len(debut_teams)} команд без игрока с ЧР-2025.",
        f"Элита: {len(all_pids & tops[10])} из глобального топ-10, {len(all_pids & tops[100])} из топ-100; {teams_2top100} команд с двумя топ-100.",
        f"Составы обкатаны: пары играли вместе в среднем {statistics.mean(pair_counts):.0f} турниров; легионеров мало ({pct(len(legion), len(all_pids))}).",
        f"Активность высокая: медиана {statistics.median(vals):.0f} турниров/игрока с прошлого ЧР; без игр — {sum(v==0 for v in vals)} чел.",
        f"Звёздный перекос: Σθ top-6 лидирует «{team_stats[0][3]}»; самый ровный верх — «{sorted(team_stats, key=lambda x: x[1])[0][3]}».",
        f"Зига-ЧР пересекается слабо ({pct(len(all_pids & ziga_hit), len(all_pids))} поля) — другой контингент.",
        f"Главный отсутствующий ветеран: {absent[0][1]} {absent[0][2]} (θ={absent[0][3]:+.2f}).",
    ]
    if not args.no_forecast and fc and fc.get("teams"):
        conclusions.append(
            f"Прогнозный фаворит на пакете ЧР-2025: {fc['teams'][0]['team_name']} (E[T]={fc['teams'][0]['expected_takes']:.1f})."
        )
    for i, c in enumerate(conclusions, 1):
        print(f"  {i}. {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
