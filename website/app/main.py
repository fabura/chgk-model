"""FastAPI application for the ChGK model website."""
from __future__ import annotations

import json
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


@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    min_games: int = Query(50, ge=1, le=10_000),
    limit: int = Query(100, ge=10, le=2000),
):
    """Top players by theta."""
    rows = db.query(
        """
        SELECT player_id, last_name, first_name, theta, games, last_game_date
        FROM players
        WHERE games >= ?
        ORDER BY theta DESC
        LIMIT ?
        """,
        [min_games, limit],
    )
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return templates.TemplateResponse(
        request,
        "top_players.html",
        {"players": rows, "min_games": min_games, "limit": limit},
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

    # Rank by θ among players with ≥ ``min_games`` games (overall).
    rank_row = db.query_one(
        """
        WITH eligible AS (
            SELECT player_id, theta,
                   RANK() OVER (ORDER BY theta DESC) AS rk,
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

    # Theta history (date, theta) — joined directly on tournament_id.
    history = db.query(
        """
        SELECT
            ph.theta,
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

    # Questions table (joined to canonical params + take stats)
    questions = db.query(
        """
        SELECT
            qa.q_in_tournament,
            qa.canonical_idx,
            q.b,
            q.a,
            q.n_obs,
            q.n_taken,
            CASE WHEN q.n_obs > 0 THEN q.n_taken::DOUBLE / q.n_obs ELSE NULL END AS take_rate,
            q.text,
            q.answer
        FROM question_aliases qa
        JOIN questions q ON q.canonical_idx = qa.canonical_idx
        WHERE qa.tournament_id = ?
        ORDER BY qa.q_in_tournament
        """,
        [tournament_id],
    )

    # Scatter data: b vs a, sized by take_rate
    scatter = [
        {
            "b": q["b"],
            "a": q["a"],
            "take_rate": q["take_rate"],
            "q": q["q_in_tournament"] + 1,
            "has_text": bool(q["text"]),
        }
        for q in questions
    ]

    return templates.TemplateResponse(
        request,
        "tournament.html",
        {
            "tournament": tournament,
            "editors": [e["editor_name"] for e in editors],
            "teams": teams,
            "questions": questions,
            "scatter_json": json.dumps(scatter, ensure_ascii=False),
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

    # θ-trend chart data: average roster θ per tournament (using current θ;
    # quick proxy for a "team strength over time" line).
    trend_json = [
        {
            "date": (g["start_date"].isoformat() if g["start_date"] else None),
            "tournament_id": g["tournament_id"],
            "title": g["title"],
            "actual": g["score_actual"],
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
            "summary": summary_row,
            "trend_json": json.dumps(trend_json, ensure_ascii=False),
        },
    )


@app.get("/methodology", response_class=HTMLResponse)
def methodology(request: Request):
    return templates.TemplateResponse(request, "methodology.html", {})


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
        players = db.query(
            """
            SELECT player_id, last_name, first_name, theta, games
            FROM players
            WHERE last_name ILIKE '%' || ? || '%' OR first_name ILIKE '%' || ? || '%'
            ORDER BY theta DESC
            LIMIT 25
            """,
            [q, q],
        )
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
