#!/usr/bin/env python3
"""Field statistics for an upcoming ЧР event (default api tid 12826)."""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import duckdb

from website.app import forecast_api

DEFAULT_EVENT_ID = 12826
DEFAULT_LAST_CHR_ID = 11749  # ЧР по интеллектуальным играм 2025
DEFAULT_ZIGA_CHR_ID = 12459  # Чемпионат России 2025 ("зига-ЧР")


@dataclass
class TeamRoster:
    team_id: int
    team_name: str
    player_ids: list[int] = field(default_factory=list)


def _fetch_rosters(event_id: int) -> list[TeamRoster]:
    payload = forecast_api.get_rosters(event_id)
    out: list[TeamRoster] = []
    for row in payload:
        team = row.get("team") or {}
        team_id = team.get("id")
        if not isinstance(team_id, int):
            continue
        pids: list[int] = []
        for tm in row.get("teamMembers") or []:
            pl = tm.get("player") or {}
            pid = pl.get("id")
            if isinstance(pid, int):
                pids.append(int(pid))
        out.append(
            TeamRoster(
                team_id=int(team_id),
                team_name=str(team.get("name") or f"#{team_id}"),
                player_ids=pids,
            )
        )
    return out


def _top_n_sets(con: duckdb.DuckDBPyConnection, metric: str) -> dict[int, set[int]]:
    """Return top-N player_id sets for metric in {theta, theta_display}."""
    col = "theta_display" if metric == "theta_display" else "theta"
    rows = con.execute(
        f"""
        SELECT player_id
        FROM players
        WHERE {col} IS NOT NULL
        ORDER BY {col} DESC, games DESC, player_id
        """
    ).fetchall()
    ordered = [int(r[0]) for r in rows]
    limits = (10, 100, 1000, 10000)
    return {n: set(ordered[:n]) for n in limits}


def _games_since(
    con: duckdb.DuckDBPyConnection, player_ids: set[int], since_date: str
) -> dict[int, int]:
    if not player_ids:
        return {}
    placeholders = ",".join("?" * len(player_ids))
    rows = con.execute(
        f"""
        SELECT pg.player_id, COUNT(DISTINCT pg.tournament_id) AS n
        FROM player_games pg
        JOIN tournaments t ON t.tournament_id = pg.tournament_id
        WHERE pg.player_id IN ({placeholders})
          AND t.start_date > ?
        GROUP BY 1
        """,
        sorted(player_ids) + [since_date],
    ).fetchall()
    return {int(pid): int(n) for pid, n in rows}


def _played_tournament(
    con: duckdb.DuckDBPyConnection, player_ids: set[int], tournament_id: int
) -> set[int]:
    if not player_ids:
        return set()
    placeholders = ",".join("?" * len(player_ids))
    rows = con.execute(
        f"""
        SELECT DISTINCT player_id
        FROM player_games
        WHERE player_id IN ({placeholders})
          AND tournament_id = ?
        """,
        sorted(player_ids) + [tournament_id],
    ).fetchall()
    return {int(r[0]) for r in rows}


def _shared_tournaments_since(
    con: duckdb.DuckDBPyConnection,
    *,
    since_date: str,
    team_rosters: list[TeamRoster],
) -> dict[int, int]:
    """For each player, count tournaments since ``since_date`` with any current teammate."""
    shared_count: dict[int, int] = defaultdict(int)
    for tr in team_rosters:
        roster_set = set(tr.player_ids)
        if len(roster_set) < 2:
            continue
        placeholders = ",".join("?" * len(roster_set))
        rows = con.execute(
            f"""
            SELECT pg.tournament_id, pg.team_id, ARRAY_AGG(DISTINCT pg.player_id) AS pids
            FROM player_games pg
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
            WHERE pg.player_id IN ({placeholders})
              AND t.start_date > ?
            GROUP BY 1, 2
            HAVING COUNT(DISTINCT pg.player_id) >= 2
            """,
            sorted(roster_set) + [since_date],
        ).fetchall()
        for _tid, _team_id, pids in rows:
            present = set(int(x) for x in pids) & roster_set
            if len(present) < 2:
                continue
            for pid in present:
                shared_count[pid] += 1
    return dict(shared_count)


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "—"
    return f"{100.0 * n / d:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", type=int, default=DEFAULT_EVENT_ID)
    ap.add_argument("--last-chr-id", type=int, default=DEFAULT_LAST_CHR_ID)
    ap.add_argument("--ziga-chr-id", type=int, default=DEFAULT_ZIGA_CHR_ID)
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    args = ap.parse_args()

    duck_path = REPO_ROOT / args.duckdb
    if not duck_path.exists():
        raise SystemExit(f"No DuckDB at {duck_path}")

    teams = _fetch_rosters(args.event_id)
    all_pids: set[int] = set()
    for t in teams:
        all_pids.update(t.player_ids)

    con = duckdb.connect(str(duck_path), read_only=True)

    last_chr = con.execute(
        "SELECT tournament_id, title, start_date, end_date FROM tournaments WHERE tournament_id = ?",
        [args.last_chr_id],
    ).fetchone()
    ziga = con.execute(
        "SELECT tournament_id, title, start_date, end_date FROM tournaments WHERE tournament_id = ?",
        [args.ziga_chr_id],
    ).fetchone()
    if not last_chr or not ziga:
        raise SystemExit("Reference tournaments not found in DuckDB")

    since_date = str(last_chr[3])  # after last ЧР ended

    # Player DB rows
    placeholders = ",".join("?" * len(all_pids)) if all_pids else "NULL"
    db_players: dict[int, dict] = {}
    if all_pids:
        for row in con.execute(
            f"""
            SELECT player_id, theta, theta_display, games, last_game_date
            FROM players WHERE player_id IN ({placeholders})
            """,
            sorted(all_pids),
        ).fetchall():
            db_players[int(row[0])] = {
                "theta": row[1],
                "theta_display": row[2],
                "games": int(row[3] or 0),
                "last_game_date": row[4],
            }

    unknown_pids = all_pids - set(db_players)

    print(f"Событие API #{args.event_id}: {len(teams)} команд, {len(all_pids)} заявленных игроков")
    print(f"Прошлый ЧР: #{last_chr[0]} «{last_chr[1]}» ({last_chr[2]} — {last_chr[3]})")
    print(f"Зига-ЧР:    #{ziga[0]} «{ziga[1]}» ({ziga[2]} — {ziga[3]})")
    print(f"Окно «с прошлого ЧР»: турниры с start_date > {since_date}\n")

    # 1) Top-N coverage
    for metric_label, metric in [("theta_display (доска сайта)", "theta_display"), ("raw θ", "theta")]:
        tops = _top_n_sets(con, metric)
        print(f"=== 1. Игроки из топ-N ({metric_label}) ===")
        for n in (10, 100, 1000, 10000):
            hit = all_pids & tops[n]
            print(
                f"  топ-{n:>5}: {len(hit):>3} игроков "
                f"({len(hit)/len(teams):.2f} на команду, {_pct(len(hit), len(all_pids))} поля)"
            )
        print()

    # 2) Games since last ЧР
    games_since = _games_since(con, all_pids, since_date)
    gs_vals = [games_since.get(pid, 0) for pid in all_pids]
    print("=== 2. Число турниров с прошлого ЧР (на игрока) ===")
    print(f"  среднее: {statistics.mean(gs_vals):.2f}")
    print(f"  медиана: {statistics.median(gs_vals):.1f}")
    print(f"  мин / макс: {min(gs_vals)} / {max(gs_vals)}")
    zero = sum(1 for v in gs_vals if v == 0)
    print(f"  без игр с прошлого ЧР: {zero} ({_pct(zero, len(gs_vals))})")
    print()

    # 3) Legionnaires
    shared = _shared_tournaments_since(con, since_date=since_date, team_rosters=teams)
    legion = [pid for pid in all_pids if shared.get(pid, 0) <= 1]
    print("=== 3. Легионеры (≤1 турнир с текущими сокомандниками с прошлого ЧР) ===")
    print(f"  {len(legion)} / {len(all_pids)} ({_pct(len(legion), len(all_pids))})")
    full_legion = sum(
        1 for tr in teams
        if tr.player_ids and all(shared.get(pid, 0) <= 1 for pid in tr.player_ids)
    )
    print(f"  команд, где все заявленные — легионеры: {full_legion} / {len(teams)}")
    print()

    # 4) Ziga ЧР overlap
    ziga_players = _played_tournament(con, all_pids, args.ziga_chr_id)
    last_chr_players = _played_tournament(con, all_pids, args.last_chr_id)
    print("=== 4. Пересечение с прошлогодними чемпионатами ===")
    print(
        f"  играли на зига-ЧР #{ziga[0]}: "
        f"{len(ziga_players)} / {len(all_pids)} ({_pct(len(ziga_players), len(all_pids))})"
    )
    print(
        f"  играли на прошлом ЧР #{last_chr[0]}: "
        f"{len(last_chr_players)} / {len(all_pids)} ({_pct(len(last_chr_players), len(all_pids))})"
    )
    both = ziga_players & last_chr_players
    print(f"  и там, и там: {len(both)}")
    print()

    # 5) Extra metrics
    roster_sizes = [len(t.player_ids) for t in teams]
    print("=== 5. Дополнительные метрики ===")
    print(
        f"  размер заявки: среднее {statistics.mean(roster_sizes):.2f}, "
        f"медиана {statistics.median(roster_sizes):.0f}, "
        f"мин/макс {min(roster_sizes)}/{max(roster_sizes)}"
    )
    print(f"  не в нашей базе (нет θ): {len(unknown_pids)} ({_pct(len(unknown_pids), len(all_pids))})")
    rookies = sum(1 for pid in all_pids if db_players.get(pid, {}).get("games", 0) < 15)
    print(f"  «новички» (<15 игр в модели): {rookies} ({_pct(rookies, len(all_pids))})")

    thetas = [db_players[pid]["theta_display"] for pid in all_pids if pid in db_players]
    if thetas:
        print(
            f"  θ_display поля: среднее {statistics.mean(thetas):+.2f}, "
            f"медиана {statistics.median(thetas):+.2f}"
        )

    # Team stability: mean pairwise shared tournaments since last ЧР
    pair_shared: list[int] = []
    for tr in teams:
        pids = tr.player_ids
        for i, a in enumerate(pids):
            for b in pids[i + 1 :]:
                rows = con.execute(
                    """
                    SELECT COUNT(DISTINCT a.tournament_id)
                    FROM player_games a
                    JOIN player_games b
                      ON a.tournament_id = b.tournament_id
                     AND a.team_id = b.team_id
                    JOIN tournaments t ON t.tournament_id = a.tournament_id
                    WHERE a.player_id = ? AND b.player_id = ?
                      AND t.start_date > ?
                    """,
                    [a, b, since_date],
                ).fetchone()
                pair_shared.append(int(rows[0] or 0))
    if pair_shared:
        print(
            f"  сыгранность пар (турниров вместе с прошлого ЧР): "
            f"среднее {statistics.mean(pair_shared):.2f}, медиана {statistics.median(pair_shared):.1f}"
        )

    teams_with_last_chr = sum(
        1 for tr in teams if any(pid in last_chr_players for pid in tr.player_ids)
    )
    teams_with_2plus_last_chr = sum(
        1
        for tr in teams
        if sum(1 for pid in tr.player_ids if pid in last_chr_players) >= 2
    )
    print(
        f"  команд с ≥1 участником прошлого ЧР: {teams_with_last_chr}/{len(teams)}; "
        f"с ≥2: {teams_with_2plus_last_chr}/{len(teams)}"
    )

    # Top-band concentration
    tops = _top_n_sets(con, "theta_display")
    for n in (100, 1000):
        hit = all_pids & tops[n]
        teams_with_top = sum(1 for tr in teams if any(pid in tops[n] for pid in tr.player_ids))
        print(
            f"  команд с игроком из топ-{n}: {teams_with_top}/{len(teams)} "
            f"({len(hit)} игроков в поле)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
