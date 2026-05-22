"""
Forecast helpers for the website.

Stage 1 — re-simulate a past tournament with **current** θ values: take
each team's roster from that tournament, look up b/a for the pack from
``question_aliases`` + ``questions``, run ``simulate_roster_on_pack``,
rank by the resulting expected score and join with the actual result.

The function intentionally does as much work as possible in DuckDB and
keeps Python down to roster bookkeeping + the simulation loop.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
from typing import Optional

# ``rating`` is shipped alongside ``app`` in production (Dockerfile copies
# both into ``/app``), but ``rating/`` is *outside* the ``app`` package
# in the dev tree.  Some launchers (e.g. plain ``uvicorn …`` without
# ``--app-dir``) end up with a sys.path that does not include the repo
# root, so we proactively make sure ``rating.simulate`` is importable
# from either location.
_HERE = _Path(__file__).resolve().parent
for _candidate in (_HERE.parent, _HERE.parent.parent):
    if (_candidate / "rating" / "simulate.py").exists():
        if str(_candidate) not in _sys.path:
            _sys.path.insert(0, str(_candidate))
        break

import numpy as np

from rating.simulate import simulate_roster_on_pack

from . import db


def _type_to_mode(ttype: Optional[str]) -> str:
    """Map the DuckDB ``type`` column to the simulation kernel's mode."""
    if ttype == "sync":
        return "sync"
    if ttype == "async":
        return "async"
    return "offline"


def forecast_past_tournament(tournament_id: int) -> Optional[dict]:
    """Build the context for the /forecast/tournament/{tid} page.

    Returns ``None`` when the tournament is not in the DB.  Otherwise
    returns a dict with the tournament metadata, the ranked forecast
    table, calibration warnings, and a model-params status flag.
    """
    tournament = db.query_one(
        "SELECT tournament_id, title, type, start_date, n_questions "
        "FROM tournaments WHERE tournament_id = ?",
        [tournament_id],
    )
    if tournament is None:
        return None

    questions = db.query(
        """
        SELECT
            qa.q_in_tournament,
            q.b,
            q.a
        FROM question_aliases qa
        JOIN questions q ON q.canonical_idx = qa.canonical_idx
        WHERE qa.tournament_id = ?
        ORDER BY qa.q_in_tournament
        """,
        [tournament_id],
    )
    if not questions:
        return {
            "tournament": tournament,
            "teams": [],
            "warnings": ["В базе нет вопросов для этого турнира."],
            "model_params_missing": False,
        }

    b_arr = np.array([q["b"] for q in questions], dtype=np.float64)
    a_arr = np.array([q["a"] for q in questions], dtype=np.float64)
    q_in_tour = np.array(
        [q["q_in_tournament"] for q in questions], dtype=np.int64
    )

    # Rosters for this tournament + each player's current θ.  One batched
    # query, grouped in Python by team_id.  We grab actual score/place
    # from team_games so we can show the comparison column.
    rows = db.query(
        """
        SELECT
            pg.team_id,
            pg.player_id,
            p.theta,
            tg.team_name,
            tg.score_actual,
            tg.expected_takes,
            tg.place,
            tg.n_players_active
        FROM player_games pg
        JOIN players p USING (player_id)
        LEFT JOIN team_games tg USING (tournament_id, team_id)
        WHERE pg.tournament_id = ?
        ORDER BY pg.team_id, p.theta DESC NULLS LAST
        """,
        [tournament_id],
    )

    teams: dict[int, dict] = {}
    for r in rows:
        tid = int(r["team_id"])
        slot = teams.setdefault(
            tid,
            {
                "team_id": tid,
                "team_name": r["team_name"] or f"#{tid}",
                "score_actual": r["score_actual"],
                "expected_takes_baked": r["expected_takes"],
                "place_actual": r["place"],
                "n_players_active": r["n_players_active"],
                "thetas": [],
                "player_ids": [],
            },
        )
        if r["theta"] is not None:
            slot["thetas"].append(float(r["theta"]))
            slot["player_ids"].append(int(r["player_id"]))

    mp = db.get_model_params()
    mode = _type_to_mode(tournament["type"])
    warnings: list[str] = []
    if mp["delta_size"] is None or mp["lapse"] is None or mp["recal"] is None:
        warnings.append(
            "Параметры калибровки модели не найдены в DuckDB; прогноз "
            "будет показан без поправок размера команды, позиции в туре "
            "и калибровки. Пересоберите DuckDB через build_db."
        )

    forecast_rows: list[dict] = []
    for slot in teams.values():
        thetas = np.asarray(slot["thetas"], dtype=np.float64)
        if thetas.size == 0:
            # No θ data for any roster member — skip the team rather than
            # invent zeros (zeros would imply a population-average team and
            # silently rank the team mid-table).
            continue
        p_q = simulate_roster_on_pack(
            thetas=thetas,
            b=b_arr,
            a=a_arr,
            q_in_tour=q_in_tour,
            delta_size=mp["delta_size"],
            team_size_anchor=mp["team_size_anchor"],
            delta_pos=mp["delta_pos"],
            pos_anchor=mp["pos_anchor"],
            mode=mode,
            lapse_arr=mp["lapse"],
            recal_arr=mp["recal"],
        )
        forecast_rows.append(
            {
                "team_id": slot["team_id"],
                "team_name": slot["team_name"],
                "expected_takes": float(p_q.sum()),
                "score_actual": slot["score_actual"],
                "place_actual": slot["place_actual"],
                "n_players": int(thetas.size),
                "n_players_active": slot["n_players_active"],
            }
        )

    forecast_rows.sort(
        key=lambda r: (-r["expected_takes"], r["team_name"] or "")
    )
    for i, r in enumerate(forecast_rows, start=1):
        r["rank_predicted"] = i
        if r["place_actual"] is not None:
            try:
                r["place_delta"] = int(round(float(r["place_actual"]))) - i
            except (TypeError, ValueError):
                r["place_delta"] = None
        else:
            r["place_delta"] = None

    return {
        "tournament": tournament,
        "teams": forecast_rows,
        "warnings": warnings,
        "model_params_missing": mp["delta_size"] is None,
    }
