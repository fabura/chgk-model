"""FastAPI application for the ChGK model website."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="ChGK Model")
# Note: cache_size=0 to work around a Jinja2 LRUCache bug under Python 3.14.
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.cache = None
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Jinja helpers
# ---------------------------------------------------------------------------


def _player_full_name(p: dict) -> str:
    last = (p.get("last_name") or "").strip()
    first = (p.get("first_name") or "").strip()
    name = f"{last} {first}".strip()
    return name or f"#{p.get('player_id')}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.1f}%"


def _fmt_signed(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "–"
    return f"{x:+.{digits}f}"


templates.env.filters["fullname"] = _player_full_name
templates.env.filters["pct"] = _fmt_pct
templates.env.filters["signed"] = _fmt_signed


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


_TYPE_LABELS = {
    "offline": "очный",
    "sync": "синхрон",
    "async": "асинхрон",
}


def _type_label(t: Optional[str]) -> str:
    return _TYPE_LABELS.get(t or "", t or "—")


templates.env.filters["type_label"] = _type_label


def _build_page_links(page: int, total_pages: int, window: int = 2) -> list[Optional[int]]:
    """
    Build a paginator: list of page numbers (1-indexed) and ``None`` for
    "…" gaps.  Always includes the first and last pages plus a ``window``
    around the current one, so the bar stays compact even on 200+ pages.
    """
    if total_pages <= 1:
        return [1] if total_pages == 1 else []
    pages: set[int] = {1, total_pages}
    for i in range(page - window, page + window + 1):
        if 1 <= i <= total_pages:
            pages.add(i)
    out: list[Optional[int]] = []
    prev = 0
    for p in sorted(pages):
        if p > prev + 1:
            out.append(None)
        out.append(p)
        prev = p
    return out


@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    min_games: int = Query(50, ge=1, le=10_000),
    per_page: int = Query(100, ge=10, le=500),
    page: int = Query(1, ge=1),
):
    """Top players by theta_display (raw theta, shrunk for inactivity)."""
    total = (
        db.query_one(
            "SELECT COUNT(*) AS n FROM players WHERE games >= ?",
            [min_games],
        )
        or {"n": 0}
    )["n"]
    total_pages = max(1, math.ceil(total / per_page))
    page = min(page, total_pages)  # clamp to last page if user overshoots
    offset = (page - 1) * per_page

    rows = db.query(
        """
        SELECT player_id, last_name, first_name, theta, theta_display,
               games, last_game_date
        FROM players
        WHERE games >= ?
        ORDER BY theta_display DESC, player_id ASC
        LIMIT ? OFFSET ?
        """,
        [min_games, per_page, offset],
    )
    for i, r in enumerate(rows, start=1):
        r["rank"] = offset + i
    return templates.TemplateResponse(
        request,
        "top_players.html",
        {
            "players": rows,
            "min_games": min_games,
            "per_page": per_page,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "page_links": _build_page_links(page, total_pages),
        },
    )


@app.get("/tournaments", response_class=HTMLResponse)
def tournaments_list(
    request: Request,
    type: str = Query("", pattern=r"^(|offline|sync|async)$"),
    per_page: int = Query(100, ge=10, le=500),
    page: int = Query(1, ge=1),
):
    """List of all tournaments, sorted by date desc, with optional type filter."""
    where = ""
    params: list = []
    if type:
        where = "WHERE type = ?"
        params.append(type)

    total = (
        db.query_one(f"SELECT COUNT(*) AS n FROM tournaments {where}", params)
        or {"n": 0}
    )["n"]
    total_pages = max(1, math.ceil(total / per_page))
    page = min(page, total_pages)
    offset = (page - 1) * per_page

    rows = db.query(
        f"""
        SELECT tournament_id, title, type, start_date,
               n_questions, n_teams
        FROM tournaments
        {where}
        ORDER BY start_date DESC NULLS LAST, tournament_id DESC
        LIMIT ? OFFSET ?
        """,
        params + [per_page, offset],
    )

    return templates.TemplateResponse(
        request,
        "tournaments_list.html",
        {
            "tournaments": rows,
            "type": type,
            "per_page": per_page,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "page_links": _build_page_links(page, total_pages),
            "type_options": [
                ("", "все"),
                ("offline", "очный"),
                ("sync", "синхрон"),
                ("async", "асинхрон"),
            ],
        },
    )


# Cached SQL for the teams ranking page.  The "current strength" of a
# team is the average ``theta_display`` of the up-to-6 players who
# appeared most often in that team over the last ``window`` days
# (relative to the latest tournament in the database).  Only teams
# active in that window are listed.
_TEAMS_RANK_SQL = """
WITH cutoff AS (
    SELECT MAX(start_date) - INTERVAL ({window}) DAY AS dt FROM tournaments
),
recent_pg AS (
    SELECT pg.team_id, pg.player_id, COUNT(*) AS n_app
    FROM player_games pg
    JOIN tournaments t USING (tournament_id)
    WHERE t.start_date >= (SELECT dt FROM cutoff)
    GROUP BY pg.team_id, pg.player_id
),
ranked AS (
    SELECT r.team_id, r.player_id, r.n_app, p.theta_display,
           ROW_NUMBER() OVER (
               PARTITION BY r.team_id
               ORDER BY r.n_app DESC, p.theta_display DESC, r.player_id
           ) AS rk
    FROM recent_pg r JOIN players p USING (player_id)
),
base AS (
    SELECT team_id,
           AVG(theta_display) AS base_theta,
           COUNT(*) AS n_base
    FROM ranked WHERE rk <= 6
    GROUP BY team_id
),
team_meta AS (
    SELECT tg.team_id,
           arg_max(tg.team_name, t.start_date) AS team_name,
           COUNT(*) AS n_recent_games,
           MAX(t.start_date) AS last_played
    FROM team_games tg
    JOIN tournaments t USING (tournament_id)
    WHERE t.start_date >= (SELECT dt FROM cutoff)
    GROUP BY tg.team_id
)
"""


@app.get("/teams", response_class=HTMLResponse)
def teams_list(
    request: Request,
    min_games: int = Query(3, ge=1, le=200),
    min_base: int = Query(4, ge=1, le=6),
    window: int = Query(365, ge=30, le=3650),
    per_page: int = Query(100, ge=10, le=500),
    page: int = Query(1, ge=1),
):
    """Teams sorted by current base-roster strength (top-6 regulars in the window).

    Filters out teams whose "core" has fewer than ``min_base`` distinct regulars,
    so that lone-wolf single-player teams do not dominate the leaderboard.
    """
    sql_head = _TEAMS_RANK_SQL.format(window=int(window))

    total = (
        db.query_one(
            sql_head
            + """
            SELECT COUNT(*) AS n
            FROM base b JOIN team_meta m USING (team_id)
            WHERE m.n_recent_games >= ? AND b.n_base >= ?
            """,
            [min_games, min_base],
        )
        or {"n": 0}
    )["n"]
    total_pages = max(1, math.ceil(total / per_page))
    page = min(page, total_pages)
    offset = (page - 1) * per_page

    rows = db.query(
        sql_head
        + """
        SELECT m.team_id, m.team_name, b.base_theta, b.n_base,
               m.n_recent_games, m.last_played
        FROM base b JOIN team_meta m USING (team_id)
        WHERE m.n_recent_games >= ? AND b.n_base >= ?
        ORDER BY b.base_theta DESC, m.team_id
        LIMIT ? OFFSET ?
        """,
        [min_games, min_base, per_page, offset],
    )
    for i, r in enumerate(rows, start=1):
        r["rank"] = offset + i

    cutoff_row = db.query_one(
        "SELECT MAX(start_date) AS today, "
        "MAX(start_date) - INTERVAL (?) DAY AS cutoff FROM tournaments",
        [int(window)],
    ) or {"today": None, "cutoff": None}

    return templates.TemplateResponse(
        request,
        "teams_list.html",
        {
            "teams": rows,
            "min_games": min_games,
            "min_base": min_base,
            "window": window,
            "per_page": per_page,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "page_links": _build_page_links(page, total_pages),
            "ref_today": cutoff_row.get("today"),
            "ref_cutoff": cutoff_row.get("cutoff"),
        },
    )


@app.get("/player/{player_id}", response_class=HTMLResponse)
def player_profile(
    request: Request,
    player_id: int,
    min_games: int = Query(30, ge=1, le=10_000),
):
    player = db.query_one(
        "SELECT * FROM players WHERE player_id = ?", [player_id]
    )
    if player is None:
        raise HTTPException(status_code=404, detail="player not found")

    # Rank by theta_display (raw θ shrunk for inactivity) among players
    # with ≥ ``min_games`` games (overall).  Using theta_display keeps the
    # rank consistent with the public top-N board.
    rank_row = db.query_one(
        """
        WITH eligible AS (
            SELECT player_id, theta_display,
                   RANK() OVER (ORDER BY theta_display DESC) AS rk,
                   COUNT(*) OVER () AS pop
            FROM players
            WHERE games >= ?
        )
        SELECT rk, pop FROM eligible WHERE player_id = ?
        """,
        [min_games, player_id],
    )
    rank_info = None
    if rank_row is not None:
        rank_info = {
            "rank": int(rank_row["rk"]),
            "total": int(rank_row["pop"]),
            "min_games": min_games,
            "pct_top": 100.0 * rank_row["rk"] / rank_row["pop"],
        }
    elif (player.get("games") or 0) < min_games:
        # Player exists but doesn't meet the threshold – mention it.
        rank_info = {
            "rank": None,
            "total": None,
            "min_games": min_games,
            "pct_top": None,
        }

    # Theta + rank history (date, theta, rank_global, n_active) — joined
    # directly on tournament_id.
    history = db.query(
        """
        SELECT
            ph.theta,
            ph.rank_global,
            ph.n_active,
            t.tournament_id,
            t.title,
            t.start_date
        FROM player_history ph
        JOIN tournaments t USING (tournament_id)
        WHERE ph.player_id = ?
        ORDER BY t.start_date NULLS LAST, t.tournament_id
        """,
        [player_id],
    )

    games = db.query(
        """
        SELECT
            pg.tournament_id,
            t.title,
            t.type,
            t.start_date,
            pg.team_id,
            COALESCE(tg.team_name, '') AS team_name,
            tg.score_actual,
            tg.expected_takes,
            tg.place,
            t.n_questions,
            tg.n_players_active,
            pg.theta_after
        FROM player_games pg
        JOIN tournaments t USING (tournament_id)
        LEFT JOIN team_games tg USING (tournament_id, team_id)
        WHERE pg.player_id = ?
        ORDER BY t.start_date DESC NULLS LAST, t.tournament_id DESC
        """,
        [player_id],
    )

    # ----- Yearly relative position --------------------------------------
    # Compares the player to all "active" players that year (≥10 games),
    # using each player's last θ snapshot in that year.  Helps separate
    # an absolute θ drop (population drift) from a real percentile drop.
    yearly = db.query(
        """
        WITH games_by_year AS (
            SELECT
                ph.player_id,
                EXTRACT(year FROM t.start_date)::INT AS yr,
                t.start_date,
                ph.theta
            FROM player_history ph
            JOIN tournaments t USING (tournament_id)
            WHERE t.start_date IS NOT NULL
        ),
        per_player_year AS (
            SELECT player_id, yr,
                   COUNT(*) AS n_games,
                   ARG_MAX(theta, start_date) AS end_theta
            FROM games_by_year
            GROUP BY player_id, yr
            HAVING COUNT(*) >= 10
        ),
        my_yrs AS (
            SELECT yr, n_games, end_theta FROM per_player_year WHERE player_id = ?
        ),
        agg AS (
            SELECT yr,
                   COUNT(*) AS pop,
                   approx_quantile(end_theta, 0.5) AS p50,
                   approx_quantile(end_theta, 0.95) AS p95,
                   approx_quantile(end_theta, 0.99) AS p99
            FROM per_player_year
            GROUP BY yr
        ),
        my_with_rank AS (
            SELECT my.yr, my.n_games, my.end_theta,
                   (SELECT COUNT(*) FROM per_player_year p
                    WHERE p.yr = my.yr AND p.end_theta < my.end_theta) AS n_below
            FROM my_yrs my
        )
        SELECT m.yr, m.n_games, m.end_theta,
               a.pop, a.p50, a.p95, a.p99,
               m.n_below,
               (m.n_below * 1.0 / NULLIF(a.pop, 0)) AS pct_below
        FROM my_with_rank m
        JOIN agg a USING (yr)
        ORDER BY m.yr
        """,
        [player_id],
    )

    # Top teammates (top-K most-frequent collaborators)
    teammates = db.query(
        """
        WITH my_games AS (
            SELECT tournament_id, team_id FROM player_games WHERE player_id = ?
        )
        SELECT
            p.player_id,
            p.last_name,
            p.first_name,
            p.theta,
            COUNT(*) AS n_games_together
        FROM player_games pg
        JOIN my_games USING (tournament_id, team_id)
        JOIN players p ON p.player_id = pg.player_id
        WHERE pg.player_id <> ?
        GROUP BY p.player_id, p.last_name, p.first_name, p.theta
        ORDER BY n_games_together DESC
        LIMIT 15
        """,
        [player_id, player_id],
    )

    return templates.TemplateResponse(
        request,
        "player.html",
        {
            "player": player,
            "rank": rank_info,
            "yearly": yearly,
            "history_json": json.dumps(
                [
                    {
                        "date": (r["start_date"].isoformat() if r["start_date"] else None),
                        "theta": r["theta"],
                        "rank": r["rank_global"],
                        "n_active": r["n_active"],
                        "tournament_id": r["tournament_id"],
                        "title": r["title"],
                    }
                    for r in history
                ],
                ensure_ascii=False,
            ),
            "yearly_json": json.dumps(
                [
                    {
                        "yr": int(r["yr"]),
                        "end_theta": r["end_theta"],
                        "p50": r["p50"],
                        "p95": r["p95"],
                        "p99": r["p99"],
                        "pct_below": r["pct_below"],
                    }
                    for r in yearly
                ],
                ensure_ascii=False,
            ),
            "games": games,
            "teammates": teammates,
        },
    )


@app.get("/tournament/{tournament_id}", response_class=HTMLResponse)
def tournament_page(request: Request, tournament_id: int):
    tournament = db.query_one(
        "SELECT * FROM tournaments WHERE tournament_id = ?", [tournament_id]
    )
    if tournament is None:
        raise HTTPException(status_code=404, detail="tournament not found")

    editors = db.query(
        "SELECT editor_name FROM pack_editors WHERE tournament_id = ? ORDER BY editor_name",
        [tournament_id],
    )

    # Teams table
    teams = db.query(
        """
        SELECT
            tg.team_id,
            tg.team_name,
            tg.score_actual,
            tg.expected_takes,
            tg.place,
            tg.n_players_active
        FROM team_games tg
        WHERE tg.tournament_id = ?
        ORDER BY tg.score_actual DESC, tg.team_name
        """,
        [tournament_id],
    )

    # For each team, top players by theta (cap to keep query small)
    if teams:
        team_ids = [t["team_id"] for t in teams]
        team_rosters = db.query(
            f"""
            SELECT
                pg.team_id,
                p.player_id,
                p.last_name,
                p.first_name,
                p.theta
            FROM player_games pg
            JOIN players p ON p.player_id = pg.player_id
            WHERE pg.tournament_id = ?
              AND pg.team_id IN ({','.join('?' * len(team_ids))})
            ORDER BY p.theta DESC
            """,
            [tournament_id] + team_ids,
        )
        roster_by_team: dict[int, list[dict]] = {}
        for r in team_rosters:
            roster_by_team.setdefault(r["team_id"], []).append(r)
        for t in teams:
            t["roster"] = roster_by_team.get(t["team_id"], [])

    # Questions table (joined to canonical params + take stats).
    # Take rate is per-tournament (qa.n_obs / qa.n_taken) — the
    # ``questions.n_obs/n_taken`` columns are aggregated by canonical_idx
    # and so double-count for sync+async pairs.
    questions = db.query(
        """
        SELECT
            qa.q_in_tournament,
            qa.canonical_idx,
            q.b,
            q.a,
            qa.n_obs,
            qa.n_taken,
            CASE WHEN qa.n_obs > 0 THEN qa.n_taken::DOUBLE / qa.n_obs ELSE NULL END AS take_rate,
            q.text,
            q.answer
        FROM question_aliases qa
        JOIN questions q ON q.canonical_idx = qa.canonical_idx
        WHERE qa.tournament_id = ?
        ORDER BY qa.q_in_tournament
        """,
        [tournament_id],
    )

    # Scatter data: b vs a, sized by take_rate.  Questions where
    # nobody took it ("гробы", take_rate == 0) are pulled out — they
    # all sit on the right edge with b clipped to 10 and visually
    # squash every other question to the left half of the plot.
    # We list their numbers separately under the chart instead.
    # The header summary (mean b / a) is also computed *without* the
    # dead questions, for the same reason — their clipped b would
    # dominate the average.
    scatter: list[dict] = []
    dead_q_numbers: list[int] = []
    live_b: list[float] = []
    live_a: list[float] = []
    for q in questions:
        rate = q["take_rate"]
        qnum = q["q_in_tournament"] + 1
        if rate is not None and rate <= 0.0:
            dead_q_numbers.append(qnum)
            continue
        live_b.append(q["b"])
        live_a.append(q["a"])
        scatter.append(
            {
                "b": q["b"],
                "a": q["a"],
                "take_rate": rate,
                "q": qnum,
                "has_text": bool(q["text"]),
            }
        )
    summary_stats = (
        {
            "mean_b": sum(live_b) / len(live_b),
            "mean_a": sum(live_a) / len(live_a),
            "n_live": len(live_b),
            "n_dead": len(dead_q_numbers),
        }
        if live_b
        else None
    )

    return templates.TemplateResponse(
        request,
        "tournament.html",
        {
            "tournament": tournament,
            "editors": [e["editor_name"] for e in editors],
            "teams": teams,
            "questions": questions,
            "scatter_json": json.dumps(scatter, ensure_ascii=False),
            "dead_q_numbers": dead_q_numbers,
            "summary_stats": summary_stats,
        },
    )


@app.get("/team/{team_id}", response_class=HTMLResponse)
def team_page(request: Request, team_id: int):
    """Team profile: tournaments played, regular roster, summary stats."""
    # Most recent name for display.
    name_row = db.query_one(
        """
        SELECT tg.team_name
        FROM team_games tg
        JOIN tournaments t USING (tournament_id)
        WHERE tg.team_id = ?
        ORDER BY t.start_date DESC NULLS LAST, t.tournament_id DESC
        LIMIT 1
        """,
        [team_id],
    )
    if name_row is None:
        raise HTTPException(status_code=404, detail="team not found")
    team_name = name_row["team_name"] or f"#{team_id}"

    # Tournaments played, with team result.
    games = db.query(
        """
        SELECT
            tg.tournament_id,
            t.title,
            t.type,
            t.start_date,
            t.n_questions,
            tg.team_name,
            tg.n_players_active,
            tg.score_actual,
            tg.expected_takes,
            tg.team_theta_implied,
            tg.place
        FROM team_games tg
        JOIN tournaments t USING (tournament_id)
        WHERE tg.team_id = ?
        ORDER BY t.start_date DESC NULLS LAST, t.tournament_id DESC
        """,
        [team_id],
    )

    # Regular roster: players who appeared most often, with their θ and last
    # date played for this team.
    roster = db.query(
        """
        SELECT
            p.player_id, p.last_name, p.first_name, p.theta,
            COUNT(*) AS games_with_team,
            MAX(t.start_date) AS last_played
        FROM player_games pg
        JOIN players p USING (player_id)
        JOIN tournaments t USING (tournament_id)
        WHERE pg.team_id = ?
        GROUP BY p.player_id, p.last_name, p.first_name, p.theta
        ORDER BY games_with_team DESC, p.theta DESC
        LIMIT 50
        """,
        [team_id],
    )

    # Per-tournament roster: who actually played in each game, with their
    # θ snapshot right after that tournament.  One batched query, grouped
    # by tournament_id in Python so the template can render expandable
    # rows under each game.
    roster_rows = db.query(
        """
        SELECT
            pg.tournament_id,
            pg.player_id,
            p.last_name,
            p.first_name,
            pg.theta_after
        FROM player_games pg
        JOIN players p USING (player_id)
        WHERE pg.team_id = ?
        ORDER BY pg.tournament_id, pg.theta_after DESC NULLS LAST
        """,
        [team_id],
    )
    rosters_by_tournament: dict[int, list[dict]] = {}
    for r in roster_rows:
        rosters_by_tournament.setdefault(r["tournament_id"], []).append(r)

    # Aggregate summary
    summary_row = db.query_one(
        """
        SELECT
            COUNT(*) AS n_tournaments,
            SUM(score_actual) AS total_actual,
            SUM(expected_takes) AS total_expected,
            AVG(score_actual) AS mean_actual,
            AVG(expected_takes) AS mean_expected,
            AVG(score_actual - expected_takes) AS mean_delta
        FROM team_games WHERE team_id = ? AND expected_takes IS NOT NULL
        """,
        [team_id],
    )

    # Team-strength trend: implied θ per tournament (per-player skill that a
    # hypothetical team of identical players of the same actual size would
    # need to take exactly score_actual on this pack — strips out δ_t,
    # δ_size, δ_pos, so the line is comparable across tournaments).  Raw
    # take counts are useless on their own because pack difficulty varies a
    # lot between tournaments.
    trend_json = [
        {
            "date": (g["start_date"].isoformat() if g["start_date"] else None),
            "tournament_id": g["tournament_id"],
            "title": g["title"],
            "theta": g["team_theta_implied"],
            "actual": g["score_actual"],
            "n_questions": g["n_questions"],
            "expected": g["expected_takes"],
        }
        for g in reversed(games)  # chronological order for the chart
    ]

    return templates.TemplateResponse(
        request,
        "team.html",
        {
            "team_id": team_id,
            "team_name": team_name,
            "games": games,
            "roster": roster,
            "rosters_by_tournament": rosters_by_tournament,
            "summary": summary_row,
            "trend_json": json.dumps(trend_json, ensure_ascii=False),
        },
    )


@app.get("/methodology", response_class=HTMLResponse)
def methodology(request: Request):
    return templates.TemplateResponse(request, "methodology.html", {})


# ---------------------------------------------------------------------------
# Compare players (2-5 players overlaid on charts + common tournaments)
# ---------------------------------------------------------------------------


MAX_COMPARE_PLAYERS = 5
MAX_COMPARE_TOURNAMENTS = 5


def _parse_compare_ids(ids_param: str, *, cap: int = MAX_COMPARE_PLAYERS) -> list[int]:
    """Parse a comma-separated id list, dedupe, ignore non-ints, cap to ``cap``."""
    if not ids_param:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for tok in ids_param.split(","):
        tok = tok.strip()
        if not tok or not tok.lstrip("-").isdigit():
            continue
        pid = int(tok)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        if len(out) >= cap:
            break
    return out


def _sister_tournament_ids(tids: list[int]) -> list[int]:
    """Return tournament_ids that share at least one canonical_idx with any
    of the given ``tids`` (i.e. paired sync+async siblings of the same pack).

    Empty list when no sisters exist.  Result is deduplicated, excludes the
    inputs, and is ordered by descending overlap so the closest sister
    comes first."""
    if not tids:
        return []
    rows = db.query(
        f"""
        SELECT qa2.tournament_id AS sib_id, COUNT(*) AS overlap
        FROM question_aliases qa1
        JOIN question_aliases qa2
          USING (canonical_idx)
        WHERE qa1.tournament_id IN ({','.join('?' * len(tids))})
          AND qa2.tournament_id NOT IN ({','.join('?' * len(tids))})
        GROUP BY qa2.tournament_id
        ORDER BY overlap DESC
        """,
        tids + tids,
    )
    return [int(r["sib_id"]) for r in rows]


@app.get("/compare", response_class=HTMLResponse)
def compare_players(
    request: Request,
    ids: str = Query("", max_length=200),
    add: str = Query("", max_length=100),
    tids: str = Query("", max_length=200),
    add_t: str = Query("", max_length=100),
):
    """Side-by-side comparison of players (``ids``) and/or tournaments (``tids``).

    Sister tournaments (sync+async pairs of the same pack) are auto-added
    so users don't have to discover them manually.  Capped at
    ``MAX_COMPARE_TOURNAMENTS`` after auto-expansion."""
    player_ids = _parse_compare_ids(ids)

    players_by_id: dict[int, dict] = {}
    if player_ids:
        rows = db.query(
            f"""
            SELECT player_id, last_name, first_name, theta, theta_display,
                   games, last_game_date
            FROM players
            WHERE player_id IN ({','.join('?' * len(player_ids))})
            """,
            player_ids,
        )
        players_by_id = {r["player_id"]: r for r in rows}
    # Drop ids that didn't resolve, keep URL order.
    players = [players_by_id[pid] for pid in player_ids if pid in players_by_id]
    player_ids = [p["player_id"] for p in players]
    selected_set = set(player_ids)

    # "+ add player" suggestions (driven by the inline form).
    add = add.strip()
    suggestions: list[dict] = []
    if add and len(player_ids) < MAX_COMPARE_PLAYERS:
        if add.isdigit():
            row = db.query_one(
                "SELECT player_id, last_name, first_name, theta, games "
                "FROM players WHERE player_id = ?",
                [int(add)],
            )
            if row:
                suggestions = [row]
        if not suggestions:
            tokens = [t for t in add.split() if t]
            if tokens:
                cond = (
                    "(last_name ILIKE '%' || ? || '%' "
                    "OR first_name ILIKE '%' || ? || '%')"
                )
                where = " AND ".join(cond for _ in tokens)
                params: list = []
                for t in tokens:
                    params.extend([t, t])
                suggestions = db.query(
                    f"""
                    SELECT player_id, last_name, first_name, theta, games
                    FROM players
                    WHERE {where}
                    ORDER BY theta DESC
                    LIMIT 15
                    """,
                    params,
                )
        suggestions = [s for s in suggestions if s["player_id"] not in selected_set]

    # θ + rank history (single batched query).
    history_by_player: dict[int, list[dict]] = {pid: [] for pid in player_ids}
    if player_ids:
        history_rows = db.query(
            f"""
            SELECT
                ph.player_id, ph.theta, ph.rank_global, ph.n_active,
                t.tournament_id, t.title, t.start_date
            FROM player_history ph
            JOIN tournaments t USING (tournament_id)
            WHERE ph.player_id IN ({','.join('?' * len(player_ids))})
            ORDER BY t.start_date NULLS LAST, t.tournament_id
            """,
            player_ids,
        )
        for r in history_rows:
            history_by_player.setdefault(r["player_id"], []).append(r)

    # Common tournaments (≥ 2 of the selected players played).
    common_tournaments: list[dict] = []
    if len(player_ids) >= 2:
        rows = db.query(
            f"""
            WITH sel AS (
                SELECT pg.tournament_id, pg.player_id, pg.team_id, pg.theta_after
                FROM player_games pg
                WHERE pg.player_id IN ({','.join('?' * len(player_ids))})
            ),
            shared AS (
                SELECT tournament_id
                FROM sel
                GROUP BY tournament_id
                HAVING COUNT(DISTINCT player_id) >= 2
            )
            SELECT
                s.tournament_id,
                t.title,
                t.type,
                t.start_date,
                t.n_questions,
                s.player_id,
                s.team_id,
                tg.team_name,
                tg.score_actual,
                tg.expected_takes,
                tg.place,
                s.theta_after
            FROM sel s
            JOIN shared USING (tournament_id)
            JOIN tournaments t USING (tournament_id)
            LEFT JOIN team_games tg USING (tournament_id, team_id)
            ORDER BY t.start_date DESC NULLS LAST, t.tournament_id DESC,
                     s.player_id
            """,
            player_ids,
        )
        bucket: dict[int, dict] = {}
        for r in rows:
            tid = r["tournament_id"]
            if tid not in bucket:
                bucket[tid] = {
                    "tournament_id": tid,
                    "title": r["title"],
                    "type": r["type"],
                    "start_date": r["start_date"],
                    "n_questions": r["n_questions"],
                    "by_player": {},
                }
            bucket[tid]["by_player"][r["player_id"]] = {
                "team_id": r["team_id"],
                "team_name": r["team_name"],
                "score_actual": r["score_actual"],
                "expected_takes": r["expected_takes"],
                "place": r["place"],
                "theta_after": r["theta_after"],
            }
        for v in bucket.values():
            team_ids = {p["team_id"] for p in v["by_player"].values()}
            v["same_team"] = len(team_ids) == 1 and len(v["by_player"]) >= 2
            v["n_present"] = len(v["by_player"])
        # bucket already preserves insertion order which mirrors the SQL ordering
        common_tournaments = list(bucket.values())

    # Yearly side-by-side (only years where ≥1 selected player has ≥10 games).
    yearly_by_year: list[dict] = []
    if player_ids:
        rows = db.query(
            f"""
            WITH games_by_year AS (
                SELECT
                    ph.player_id,
                    EXTRACT(year FROM t.start_date)::INT AS yr,
                    t.start_date,
                    ph.theta
                FROM player_history ph
                JOIN tournaments t USING (tournament_id)
                WHERE t.start_date IS NOT NULL
            ),
            per_player_year AS (
                SELECT player_id, yr,
                       COUNT(*) AS n_games,
                       ARG_MAX(theta, start_date) AS end_theta
                FROM games_by_year
                GROUP BY player_id, yr
                HAVING COUNT(*) >= 10
            ),
            sel_yrs AS (
                SELECT DISTINCT yr FROM per_player_year
                WHERE player_id IN ({','.join('?' * len(player_ids))})
            ),
            agg AS (
                SELECT yr,
                       COUNT(*) AS pop,
                       approx_quantile(end_theta, 0.5) AS p50,
                       approx_quantile(end_theta, 0.95) AS p95,
                       approx_quantile(end_theta, 0.99) AS p99
                FROM per_player_year
                WHERE yr IN (SELECT yr FROM sel_yrs)
                GROUP BY yr
            ),
            sel AS (
                SELECT pp.yr, pp.player_id, pp.n_games, pp.end_theta,
                       (SELECT COUNT(*) FROM per_player_year p2
                        WHERE p2.yr = pp.yr AND p2.end_theta < pp.end_theta) AS n_below
                FROM per_player_year pp
                WHERE pp.player_id IN ({','.join('?' * len(player_ids))})
            )
            SELECT s.yr, s.player_id, s.n_games, s.end_theta,
                   a.pop, a.p50, a.p95, a.p99,
                   s.n_below,
                   (s.n_below * 1.0 / NULLIF(a.pop, 0)) AS pct_below
            FROM sel s JOIN agg a USING (yr)
            ORDER BY s.yr, s.player_id
            """,
            player_ids + player_ids,
        )
        bucket_y: dict[int, dict] = {}
        for r in rows:
            yr = int(r["yr"])
            if yr not in bucket_y:
                bucket_y[yr] = {
                    "yr": yr,
                    "pop": int(r["pop"]),
                    "p50": r["p50"],
                    "p95": r["p95"],
                    "p99": r["p99"],
                    "by_player": {},
                }
            bucket_y[yr]["by_player"][r["player_id"]] = {
                "n_games": int(r["n_games"]),
                "end_theta": r["end_theta"],
                "pct_below": r["pct_below"],
            }
        yearly_by_year = sorted(bucket_y.values(), key=lambda v: v["yr"])

    chart_series = [
        {
            "player_id": pid,
            "name": _player_full_name(p),
            "theta_points": [
                {
                    "date": (h["start_date"].isoformat()
                             if h["start_date"] else None),
                    "theta": h["theta"],
                    "title": h["title"],
                }
                for h in history_by_player.get(pid, [])
            ],
            "rank_points": [
                {
                    "date": (h["start_date"].isoformat()
                             if h["start_date"] else None),
                    "rank": h["rank_global"],
                    "n_active": h["n_active"],
                    "title": h["title"],
                }
                for h in history_by_player.get(pid, [])
                if h["rank_global"] and h["rank_global"] > 0
            ],
        }
        for pid, p in zip(player_ids, players)
    ]

    def _ids_without(pid: int) -> str:
        return ",".join(str(i) for i in player_ids if i != pid)

    def _ids_with(pid: int) -> str:
        return ",".join(str(i) for i in player_ids + [pid])

    remove_url = {pid: _ids_without(pid) for pid in player_ids}
    add_url = {s["player_id"]: _ids_with(s["player_id"]) for s in suggestions}

    # ----------------------------------------------------------------- tournaments
    requested_tids = _parse_compare_ids(tids, cap=MAX_COMPARE_TOURNAMENTS)
    auto_added_tids: set[int] = set()
    if requested_tids:
        # Auto-add sister (paired sync+async) tournaments.  We expand
        # one sister at a time so the cap stays predictable.
        sisters = _sister_tournament_ids(requested_tids)
        for sib in sisters:
            if len(requested_tids) >= MAX_COMPARE_TOURNAMENTS:
                break
            if sib in requested_tids:
                continue
            requested_tids.append(sib)
            auto_added_tids.add(sib)
    tournament_ids: list[int] = []
    tournaments_data: list[dict] = []
    if requested_tids:
        rows = db.query(
            f"""
            SELECT tournament_id, title, type, start_date, n_questions, n_teams
            FROM tournaments
            WHERE tournament_id IN ({','.join('?' * len(requested_tids))})
            """,
            requested_tids,
        )
        by_tid = {r["tournament_id"]: r for r in rows}
        for tid in requested_tids:
            if tid in by_tid:
                t = by_tid[tid]
                t["auto_added"] = tid in auto_added_tids
                tournaments_data.append(t)
                tournament_ids.append(tid)

    add_t = add_t.strip()
    tournament_suggestions: list[dict] = []
    if add_t and len(tournament_ids) < MAX_COMPARE_TOURNAMENTS:
        if add_t.isdigit():
            row = db.query_one(
                "SELECT tournament_id, title, type, start_date, n_questions, n_teams "
                "FROM tournaments WHERE tournament_id = ?",
                [int(add_t)],
            )
            if row:
                tournament_suggestions = [row]
        if not tournament_suggestions:
            tokens = [t for t in add_t.split() if t]
            if tokens:
                cond = "title ILIKE '%' || ? || '%'"
                where = " AND ".join(cond for _ in tokens)
                params: list = list(tokens)
                tournament_suggestions = db.query(
                    f"""
                    SELECT tournament_id, title, type, start_date, n_questions, n_teams
                    FROM tournaments
                    WHERE {where}
                    ORDER BY start_date DESC NULLS LAST
                    LIMIT 15
                    """,
                    params,
                )
        selected_tset = set(tournament_ids)
        tournament_suggestions = [
            s for s in tournament_suggestions if s["tournament_id"] not in selected_tset
        ]

    # Per-tournament question scatter (canonical b/a + per-tournament take rate).
    # Skip dead questions (take_rate==0) for the same reason as the per-tournament
    # page: their b is clipped to +10 and would dominate the layout.
    tournament_chart_series: list[dict] = []
    if tournament_ids:
        rows = db.query(
            f"""
            SELECT
                qa.tournament_id,
                qa.q_in_tournament,
                qa.canonical_idx,
                q.b,
                q.a,
                CASE WHEN qa.n_obs > 0 THEN qa.n_taken::DOUBLE / qa.n_obs ELSE NULL END AS take_rate
            FROM question_aliases qa
            JOIN questions q ON q.canonical_idx = qa.canonical_idx
            WHERE qa.tournament_id IN ({','.join('?' * len(tournament_ids))})
            ORDER BY qa.tournament_id, qa.q_in_tournament
            """,
            tournament_ids,
        )
        by_tid_qs: dict[int, list[dict]] = {tid: [] for tid in tournament_ids}
        for r in rows:
            by_tid_qs.setdefault(r["tournament_id"], []).append(r)

        title_by_tid = {t["tournament_id"]: t["title"] for t in tournaments_data}
        type_by_tid = {t["tournament_id"]: t["type"] for t in tournaments_data}
        for tid in tournament_ids:
            qs = by_tid_qs.get(tid, [])
            live = [q for q in qs if q["take_rate"] is not None and q["take_rate"] > 0.0]
            n_dead = sum(
                1 for q in qs if q["take_rate"] is not None and q["take_rate"] <= 0.0
            )
            tournament_chart_series.append(
                {
                    "tournament_id": tid,
                    "title": title_by_tid.get(tid, str(tid)),
                    "type": type_by_tid.get(tid, ""),
                    "points": [
                        {
                            "b": q["b"],
                            "a": q["a"],
                            "take_rate": q["take_rate"],
                            "q": q["q_in_tournament"] + 1,
                        }
                        for q in live
                    ],
                    "mean_b": (
                        sum(q["b"] for q in live) / len(live) if live else None
                    ),
                    "mean_a": (
                        sum(q["a"] for q in live) / len(live) if live else None
                    ),
                    "n_live": len(live),
                    "n_dead": n_dead,
                }
            )

    def _tids_without(tid: int) -> str:
        return ",".join(str(i) for i in tournament_ids if i != tid)

    def _tids_with(tid: int) -> str:
        return ",".join(str(i) for i in tournament_ids + [tid])

    tournament_remove_url = {tid: _tids_without(tid) for tid in tournament_ids}
    tournament_add_url = {
        s["tournament_id"]: _tids_with(s["tournament_id"])
        for s in tournament_suggestions
    }

    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "player_ids": player_ids,
            "ids_csv": ",".join(str(i) for i in player_ids),
            "players": players,
            "max_players": MAX_COMPARE_PLAYERS,
            "add_query": add,
            "suggestions": suggestions,
            "remove_url": remove_url,
            "add_url": add_url,
            "chart_series_json": json.dumps(chart_series, ensure_ascii=False),
            "common_tournaments": common_tournaments,
            "yearly_by_year": yearly_by_year,
            # Tournament comparison
            "tournament_ids": tournament_ids,
            "tids_csv": ",".join(str(i) for i in tournament_ids),
            "tournaments": tournaments_data,
            "max_tournaments": MAX_COMPARE_TOURNAMENTS,
            "add_tournament_query": add_t,
            "tournament_suggestions": tournament_suggestions,
            "tournament_remove_url": tournament_remove_url,
            "tournament_add_url": tournament_add_url,
            "tournament_chart_json": json.dumps(
                tournament_chart_series, ensure_ascii=False, default=str
            ),
        },
    )


# ---------------------------------------------------------------------------
# Search (basic)
# ---------------------------------------------------------------------------


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = Query("", min_length=0, max_length=100)):
    q = q.strip()
    players: list[dict] = []
    tournaments: list[dict] = []
    if q:
        if q.isdigit():
            pid = int(q)
            hit = db.query(
                "SELECT player_id FROM players WHERE player_id = ? LIMIT 1",
                [pid],
            )
            if hit:
                return RedirectResponse(url=f"/player/{pid}", status_code=303)
            tid_hit = db.query(
                "SELECT tournament_id FROM tournaments WHERE tournament_id = ? LIMIT 1",
                [pid],
            )
            if tid_hit:
                return RedirectResponse(url=f"/tournament/{pid}", status_code=303)
        # Each whitespace-separated token must match first_name or last_name
        # (so e.g. «Дима Попов» finds rows with both substrings, in any order).
        tokens = [t for t in q.split() if t]
        if tokens:
            token_cond = (
                "(last_name ILIKE '%' || ? || '%' OR first_name ILIKE '%' || ? || '%')"
            )
            where_players = " AND ".join(token_cond for _ in tokens)
            player_params: list[str] = []
            for t in tokens:
                player_params.extend([t, t])
            players = db.query(
                f"""
                SELECT player_id, last_name, first_name, theta, games
                FROM players
                WHERE {where_players}
                ORDER BY theta DESC
                LIMIT 25
                """,
                player_params,
            )
        else:
            players = []
        tournaments = db.query(
            """
            SELECT tournament_id, title, type, start_date, n_questions, n_teams
            FROM tournaments
            WHERE title ILIKE '%' || ? || '%'
            ORDER BY start_date DESC NULLS LAST
            LIMIT 25
            """,
            [q],
        )
    return templates.TemplateResponse(
        request,
        "search.html",
        {"q": q, "players": players, "tournaments": tournaments},
    )


# ---------------------------------------------------------------------------
# Admin: hot-reload the DuckDB connection after `scripts/refresh_data.sh`
# ---------------------------------------------------------------------------


def _admin_token() -> Optional[str]:
    """Resolve the admin token: env var > website/.admin_token file."""
    tok = os.environ.get("ADMIN_TOKEN")
    if tok:
        return tok.strip() or None
    path = BASE_DIR.parent / ".admin_token"
    if path.exists():
        try:
            return path.read_text().strip() or None
        except Exception:
            return None
    return None


@app.post("/admin/reload-db")
def admin_reload_db(x_admin_token: Optional[str] = Header(default=None)):
    """Close and re-open the DuckDB connection so an updated file is picked up.

    Auth: header ``X-Admin-Token`` must match ``$ADMIN_TOKEN`` env var or
    the contents of ``website/.admin_token``. If no token is configured
    on the server we refuse the request to avoid accidental hot-reloads.
    """
    expected = _admin_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="reload disabled: server has no ADMIN_TOKEN configured.",
        )
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="invalid admin token.")
    info = db.reload_conn()
    info["reloaded_at"] = datetime.now(timezone.utc).isoformat()
    return JSONResponse(info)


@app.get("/admin/db-info")
def admin_db_info(x_admin_token: Optional[str] = Header(default=None)):
    """Inspect the currently-open DuckDB file (size, mtime, row counts)."""
    expected = _admin_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="admin disabled: server has no ADMIN_TOKEN configured.",
        )
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="invalid admin token.")
    path = db.db_path()
    st = path.stat()
    counts = {}
    for table in ("players", "tournaments", "team_games", "questions"):
        try:
            row = db.query_one(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = int(row["n"]) if row else None
        except Exception as exc:
            counts[table] = f"err: {exc}"
    return JSONResponse(
        {
            "path": str(path),
            "size_bytes": int(st.st_size),
            "mtime": st.st_mtime,
            "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "row_counts": counts,
        }
    )
