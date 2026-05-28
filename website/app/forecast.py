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

from . import db, forecast_api


def _type_to_mode(ttype: Optional[str]) -> str:
    """Map the DuckDB ``type`` column to the simulation kernel's mode."""
    if ttype == "sync":
        return "sync"
    if ttype == "async":
        return "async"
    return "offline"


# ---------------------------------------------------------------------------
# Monte-Carlo with θ-bootstrap
# ---------------------------------------------------------------------------
#
# For a forecast of distribution-of-places we sample, per Monte-Carlo
# iteration:
#   1. θ_p ~ Normal(θ̂_p, σ²_p)  per player
#      (a θ-bootstrap reflecting the rating's posterior uncertainty)
#   2. p_q from the noisy-OR + lapse + recal kernel using those θ
#   3. score_t ~ Σ_q Bernoulli(p_q)
# Then rank teams within each iteration → empirical place distribution.
#
# The σ_p proxy ``1 / sqrt(games + 1)`` mirrors the engine's adaptive
# learning rate ``η = η₀ / sqrt(games_offset + games)`` — same inverse-
# information weighting, which is the right first-order approximation
# of posterior std for an online SGD estimate.  Capped at 0.7 so a
# debutant doesn't spread across the full θ range and confuse the chart.

_DEFAULT_N_SAMPLES = 2000
_THETA_SIGMA_CAP = 0.7
_THETA_SIGMA_FLOOR = 0.04  # established players: still some noise


def _theta_sigmas(games: np.ndarray) -> np.ndarray:
    g = np.asarray(games, dtype=np.float64)
    s = 1.0 / np.sqrt(np.clip(g, 0.0, None) + 1.0)
    return np.clip(s, _THETA_SIGMA_FLOOR, _THETA_SIGMA_CAP)


def _player_display_name(
    *,
    player_id: Optional[int],
    last_name: Optional[str] = None,
    first_name: Optional[str] = None,
    api_surname: Optional[str] = None,
    api_name: Optional[str] = None,
) -> str:
    """Best-effort display name for a roster line on forecast pages."""
    parts = [last_name or "", first_name or ""]
    name = " ".join(p for p in parts if p).strip()
    if not name and (api_surname or api_name):
        name = f"{api_surname or ''} {api_name or ''}".strip()
    if name:
        return name
    if player_id is not None:
        return f"#{player_id}"
    return "—"


def _members_sorted_by_theta(members: list[dict]) -> list[dict]:
    return sorted(members, key=lambda m: (-float(m.get("theta", 0.0)), m.get("name") or ""))


def simulate_field(
    *,
    rosters: list[dict],
    b_arr: np.ndarray,
    a_arr: np.ndarray,
    q_in_tour: np.ndarray,
    mode: str,
    n_mc_samples: int = _DEFAULT_N_SAMPLES,
    rng_seed: int = 20260524,
) -> dict:
    """Run the full forecast pipeline (E[T] + MC place distribution).

    Generic over the data source — the page-level helpers (past
    tournament, upcoming tournament, user team) prepare ``rosters`` and
    the pack arrays, then defer ranking and Monte-Carlo to here so the
    formula stays in exactly one place.

    Each roster dict is expected to contain at least:
      * ``team_id``, ``team_name``
      * ``thetas`` (sequence of floats)
      * ``games`` (sequence of ints, same length as ``thetas``)

    Optional fields are forwarded into the result rows verbatim:
    ``score_actual``, ``place_actual``, ``n_players_active``,
    ``n_unknown_players`` (for warnings on /forecast/event/{api_tid}).

    Returns a dict with ``teams`` (sorted by E[T] desc, with MC quantiles
    populated when ``n_mc_samples > 0``), ``warnings``,
    ``model_params_missing``, and ``n_mc_samples``.
    """
    mp = db.get_model_params()
    warnings: list[str] = []
    if mp["delta_size"] is None or mp["lapse"] is None or mp["recal"] is None:
        warnings.append(
            "Параметры калибровки модели не найдены в DuckDB; прогноз "
            "будет показан без поправок размера команды, позиции в туре "
            "и калибровки. Пересоберите DuckDB через build_db."
        )

    # ---- Pass 1: deterministic p_q + E[T] per team ----------------------
    paired: list[tuple[dict, np.ndarray, dict]] = []
    for src in rosters:
        thetas = np.asarray(src.get("thetas", []), dtype=np.float64)
        if thetas.size == 0:
            # No θ data for any roster member — skip rather than invent
            # zeros (zeros would imply a population-average team and
            # silently rank the team mid-table).
            continue
        games = np.asarray(
            src.get("games") or np.zeros_like(thetas), dtype=np.int64
        )
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
        row = {
            "team_id": src.get("team_id"),
            "team_name": src.get("team_name") or f"#{src.get('team_id')}",
            "expected_takes": float(p_q.sum()),
            "score_actual": src.get("score_actual"),
            "place_actual": src.get("place_actual"),
            "n_players": int(thetas.size),
            "n_players_active": src.get("n_players_active"),
            "n_unknown_players": src.get("n_unknown_players", 0),
        }
        if src.get("members"):
            row["members"] = src["members"]
        paired.append((row, p_q, {"thetas": thetas, "games": games}))

    paired.sort(key=lambda t: (-t[0]["expected_takes"], t[0]["team_name"] or ""))
    forecast_rows = [t[0] for t in paired]
    ordered_boot = [t[2] for t in paired]

    for i, r in enumerate(forecast_rows, start=1):
        r["rank_predicted"] = i
        if r["place_actual"] is not None:
            try:
                r["place_delta"] = int(round(float(r["place_actual"]))) - i
            except (TypeError, ValueError):
                r["place_delta"] = None
        else:
            r["place_delta"] = None

    # ---- Pass 2: Monte-Carlo over (θ-bootstrap, Bernoulli takes) -------
    if forecast_rows and n_mc_samples > 0:
        rng = np.random.default_rng(rng_seed)
        scores = _monte_carlo_scores(
            boot_inputs=ordered_boot,
            b_arr=b_arr,
            a_arr=a_arr,
            q_in_tour=q_in_tour,
            mp=mp,
            mode=mode,
            n_samples=int(n_mc_samples),
            rng=rng,
        )
        ranks = _rank_descending_avg(scores)
        place_q = np.quantile(ranks, [0.05, 0.5, 0.95], axis=0)
        score_q = np.quantile(scores, [0.05, 0.5, 0.95], axis=0)
        for i, r in enumerate(forecast_rows):
            r["score_q05"] = float(score_q[0, i])
            r["score_q50"] = float(score_q[1, i])
            r["score_q95"] = float(score_q[2, i])
            r["place_q05"] = float(place_q[0, i])
            r["place_q50"] = float(place_q[1, i])
            r["place_q95"] = float(place_q[2, i])

    return {
        "teams": forecast_rows,
        "warnings": warnings,
        "model_params_missing": mp["delta_size"] is None,
        "n_mc_samples": int(n_mc_samples),
    }


def forecast_past_tournament(
    tournament_id: int,
    *,
    n_mc_samples: int = _DEFAULT_N_SAMPLES,
    rng_seed: int = 20260524,
) -> Optional[dict]:
    """Build the context for the /forecast/tournament/{tid} page."""
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
            "n_mc_samples": 0,
        }

    b_arr = np.array([q["b"] for q in questions], dtype=np.float64)
    a_arr = np.array([q["a"] for q in questions], dtype=np.float64)
    q_in_tour = np.array(
        [q["q_in_tournament"] for q in questions], dtype=np.int64
    )

    rows = db.query(
        """
        SELECT
            pg.team_id,
            pg.player_id,
            p.theta,
            p.games,
            p.last_name,
            p.first_name,
            tg.team_name,
            tg.score_actual,
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
                "place_actual": r["place"],
                "n_players_active": r["n_players_active"],
                "thetas": [],
                "games": [],
                "members": [],
            },
        )
        if r["theta"] is not None:
            slot["thetas"].append(float(r["theta"]))
            slot["games"].append(int(r["games"] or 0))
            slot["members"].append({
                "player_id": int(r["player_id"]),
                "name": _player_display_name(
                    player_id=int(r["player_id"]),
                    last_name=r.get("last_name"),
                    first_name=r.get("first_name"),
                ),
                "theta": float(r["theta"]),
                "unknown": False,
            })
    for slot in teams.values():
        slot["members"] = _members_sorted_by_theta(slot["members"])

    sim = simulate_field(
        rosters=list(teams.values()),
        b_arr=b_arr,
        a_arr=a_arr,
        q_in_tour=q_in_tour,
        mode=_type_to_mode(tournament["type"]),
        n_mc_samples=n_mc_samples,
        rng_seed=rng_seed,
    )
    sim["tournament"] = tournament
    return sim


# ---------------------------------------------------------------------------
# Pack sources for forecasts of *future* tournaments
# ---------------------------------------------------------------------------


_SYNTH_PROFILES: dict[str, dict] = {
    # Loosely matches what the data shows for these formats.  ``a = 1``
    # is the population centre (regularised toward 0 in log space, so
    # the mean stays near 1.0 even with ``freeze_log_a=False``); for
    # past packs we use the actually-learned per-question ``a`` from
    # the questions table instead.
    "easy": {
        "label": "Школьный / лёгкий пакет",
        "b_mean": -0.7, "b_std": 0.5, "a": 1.0,
    },
    "medium": {
        "label": "Средний синхрон",
        "b_mean": 0.0, "b_std": 0.7, "a": 1.0,
    },
    "hard": {
        "label": "Топовый очный / сильный пак",
        "b_mean": 0.7, "b_std": 0.7, "a": 1.0,
    },
}


def _pack_from_past(tournament_id: int) -> Optional[dict]:
    """Pull (b, a, q_in_tour) for a past pack from DuckDB."""
    rows = db.query(
        """
        SELECT qa.q_in_tournament, q.b, q.a
        FROM question_aliases qa
        JOIN questions q ON q.canonical_idx = qa.canonical_idx
        WHERE qa.tournament_id = ?
        ORDER BY qa.q_in_tournament
        """,
        [int(tournament_id)],
    )
    if not rows:
        return None
    meta = db.query_one(
        "SELECT title, type, n_questions FROM tournaments WHERE tournament_id = ?",
        [int(tournament_id)],
    )
    return {
        "kind": "past",
        "tournament_id": int(tournament_id),
        "title": (meta or {}).get("title") or f"турнир {tournament_id}",
        "type": (meta or {}).get("type"),
        "b": np.array([r["b"] for r in rows], dtype=np.float64),
        "a": np.array([r["a"] for r in rows], dtype=np.float64),
        "q_in_tour": np.array(
            [r["q_in_tournament"] for r in rows], dtype=np.int64
        ),
        "n_questions": len(rows),
    }


def _pack_synthetic(profile: str, n_questions: int) -> dict:
    """Generate a deterministic synthetic pack for the given profile."""
    spec = _SYNTH_PROFILES.get(profile, _SYNTH_PROFILES["medium"])
    n = max(1, int(n_questions))
    rng = np.random.default_rng(hash((profile, n)) & 0xFFFFFFFF)
    b = rng.normal(spec["b_mean"], spec["b_std"], size=n)
    a = np.full(n, spec["a"], dtype=np.float64)
    return {
        "kind": "synth",
        "profile": profile,
        "title": spec["label"],
        "b": b.astype(np.float64),
        "a": a,
        "q_in_tour": np.arange(n, dtype=np.int64),
        "n_questions": n,
    }


def resolve_pack(
    *,
    pack_kind: str,
    pack_id: Optional[int],
    profile: Optional[str],
    fallback_n_questions: int,
) -> Optional[dict]:
    """Translate the URL ``pack_*`` query params into a pack dict.

    ``pack_kind == "past"`` looks the tournament up in DuckDB; missing
    data returns ``None``.  ``pack_kind == "synth"`` always succeeds
    (unknown profile silently falls back to ``"medium"``).  Anything
    else is treated as a request for the default synthetic pack.
    """
    kind = (pack_kind or "synth").strip().lower()
    if kind == "past" and pack_id is not None:
        return _pack_from_past(int(pack_id))
    profile = (profile or "medium").strip().lower()
    if profile not in _SYNTH_PROFILES:
        profile = "medium"
    return _pack_synthetic(profile, fallback_n_questions)


# ---------------------------------------------------------------------------
# Forecast for an *upcoming* (or just-now) tournament from rating.chgk.info
# ---------------------------------------------------------------------------


def forecast_for_event(
    api_tid: int,
    *,
    pack_kind: str = "synth",
    pack_id: Optional[int] = None,
    profile: Optional[str] = "medium",
    n_mc_samples: int = _DEFAULT_N_SAMPLES,
    rng_seed: int = 20260524,
) -> Optional[dict]:
    """Build the context for /forecast/event/{api_tid}.

    Pulls metadata + rosters from ``api.rating.chgk.info``, looks up
    each player's current θ in our DuckDB (unknown players are flagged
    so the page can warn), resolves the pack via ``resolve_pack``, and
    runs the same simulation pipeline as past tournaments.
    """
    try:
        meta = forecast_api.get_tournament(int(api_tid))
    except Exception as exc:
        return {
            "tournament": None,
            "api_tid": int(api_tid),
            "fetch_error": f"Не удалось получить турнир из API: {exc}",
            "teams": [],
            "warnings": [],
            "pack": None,
            "n_mc_samples": 0,
            "model_params_missing": False,
        }
    if not isinstance(meta, dict) or "id" not in meta:
        return None

    n_q_total = sum(int(v) for v in (meta.get("questionQty") or {}).values()) or 36
    pack = resolve_pack(
        pack_kind=pack_kind,
        pack_id=pack_id,
        profile=profile,
        fallback_n_questions=n_q_total,
    )
    if pack is None:
        return {
            "tournament": meta,
            "api_tid": int(api_tid),
            "teams": [],
            "warnings": ["Выбранный пакет не найден в нашей БД."],
            "pack": None,
            "n_mc_samples": 0,
            "model_params_missing": False,
        }

    try:
        roster_payload = forecast_api.get_rosters(int(api_tid))
    except Exception as exc:
        return {
            "tournament": meta,
            "api_tid": int(api_tid),
            "fetch_error": f"Не удалось получить составы из API: {exc}",
            "teams": [],
            "warnings": [],
            "pack": pack,
            "n_mc_samples": 0,
            "model_params_missing": False,
        }

    # Collect every (api) player_id we need θ / games for.
    pid_set: set[int] = set()
    for team_row in roster_payload:
        for tm in team_row.get("teamMembers") or []:
            pl = tm.get("player") or {}
            pid = pl.get("id")
            if isinstance(pid, int):
                pid_set.add(int(pid))
    theta_by_pid: dict[int, dict] = {}
    if pid_set:
        rows = db.query(
            "SELECT player_id, theta, games, last_name, first_name "
            "FROM players WHERE player_id IN ("
            + ",".join("?" * len(pid_set)) + ")",
            sorted(pid_set),
        )
        theta_by_pid = {int(r["player_id"]): r for r in rows}

    # Cold-start prior used during training, read from model_params
    # (written by build_db).  Falls back to -1.5 for legacy DuckDB
    # files that pre-date the field.
    cold_init = db.get_model_params().get("cold_init_theta")
    if cold_init is None:
        cold_init = -1.5

    rosters: list[dict] = []
    n_unknown_total = 0
    for team_row in roster_payload:
        team = team_row.get("team") or {}
        team_id = team.get("id")
        if team_id is None:
            continue
        members_raw = team_row.get("teamMembers") or []
        thetas: list[float] = []
        games: list[int] = []
        roster_members: list[dict] = []
        n_unknown = 0
        for tm in members_raw:
            pl = tm.get("player") or {}
            pid = pl.get("id")
            if not isinstance(pid, int):
                continue
            db_row = theta_by_pid.get(int(pid))
            if db_row is None:
                # Player is in the tournament but not in our DB — likely
                # a brand-new debutant.  Use the engine's cold-start
                # prior (read from model_params, so it stays in sync if
                # ``Config.cold_init_theta`` ever changes again).  Treat
                # them as having zero games for the bootstrap (max σ).
                theta_val = float(cold_init)
                thetas.append(theta_val)
                games.append(0)
                n_unknown += 1
                roster_members.append({
                    "player_id": int(pid),
                    "name": _player_display_name(
                        player_id=int(pid),
                        api_surname=pl.get("surname"),
                        api_name=pl.get("name"),
                    ),
                    "theta": theta_val,
                    "unknown": True,
                })
            else:
                theta_val = float(db_row["theta"] or 0.0)
                thetas.append(theta_val)
                games.append(int(db_row["games"] or 0))
                roster_members.append({
                    "player_id": int(pid),
                    "name": _player_display_name(
                        player_id=int(pid),
                        last_name=db_row.get("last_name"),
                        first_name=db_row.get("first_name"),
                        api_surname=pl.get("surname"),
                        api_name=pl.get("name"),
                    ),
                    "theta": theta_val,
                    "unknown": False,
                })
        if not thetas:
            continue
        n_unknown_total += n_unknown
        rosters.append(
            {
                "team_id": int(team_id),
                "team_name": team.get("name") or f"#{team_id}",
                "thetas": thetas,
                "games": games,
                "members": _members_sorted_by_theta(roster_members),
                "n_unknown_players": n_unknown,
                "n_players_active": len(thetas),
            }
        )

    sim = simulate_field(
        rosters=rosters,
        b_arr=pack["b"],
        a_arr=pack["a"],
        q_in_tour=pack["q_in_tour"],
        mode=forecast_api.api_type_to_mode(meta.get("type")),
        n_mc_samples=n_mc_samples,
        rng_seed=rng_seed,
    )
    if n_unknown_total:
        sim["warnings"].append(
            f"Игроков не из нашей базы: {n_unknown_total} — для них θ "
            f"взят как cold-start ({cold_init:+.1f}) с максимальной "
            "неопределённостью."
        )
    sim["tournament"] = meta
    sim["api_tid"] = int(api_tid)
    sim["pack"] = pack
    return sim


def forecast_custom_team(
    *,
    player_ids: list[int],
    pack_kind: str = "synth",
    pack_id: Optional[int] = None,
    profile: Optional[str] = "medium",
    n_questions: int = 36,
    mode: str = "offline",
    n_mc_samples: int = _DEFAULT_N_SAMPLES,
    rng_seed: int = 20260524,
) -> dict:
    """One-team forecast for the /forecast/team-builder page.

    The resulting context has the same ``teams`` shape as the multi-team
    pages so the template can reuse the same MC columns.  Single-team
    place distribution is degenerate (always 1) so we omit it and
    surface the score distribution instead.  Per-question p_q is also
    included so the user can see which questions the team is likely to
    take or fail.
    """
    pack = resolve_pack(
        pack_kind=pack_kind, pack_id=pack_id, profile=profile,
        fallback_n_questions=n_questions,
    )
    if pack is None:
        return {
            "teams": [],
            "warnings": ["Выбранный пакет не найден в нашей БД."],
            "pack": None,
            "players": [],
            "n_mc_samples": 0,
            "model_params_missing": False,
            "p_q": [],
        }

    cleaned: list[int] = []
    for pid in player_ids:
        try:
            v = int(pid)
        except (TypeError, ValueError):
            continue
        if v > 0 and v not in cleaned:
            cleaned.append(v)
    if not cleaned:
        return {
            "teams": [],
            "warnings": [],
            "pack": pack,
            "players": [],
            "n_mc_samples": 0,
            "model_params_missing": False,
            "p_q": [],
        }

    rows = db.query(
        "SELECT player_id, theta, games, last_name, first_name "
        "FROM players WHERE player_id IN ("
        + ",".join("?" * len(cleaned)) + ")",
        cleaned,
    )
    by_pid = {int(r["player_id"]): r for r in rows}
    players_resolved: list[dict] = []
    thetas: list[float] = []
    games: list[int] = []
    missing: list[int] = []
    for pid in cleaned:
        row = by_pid.get(pid)
        if row is None:
            missing.append(pid)
            players_resolved.append({
                "player_id": pid, "name": f"#{pid} (не найден)",
                "theta": None, "games": 0, "missing": True,
            })
        else:
            players_resolved.append({
                "player_id": pid,
                "name": f"{row['last_name'] or ''} {row['first_name'] or ''}".strip()
                        or f"#{pid}",
                "theta": float(row["theta"] or 0.0),
                "games": int(row["games"] or 0),
                "missing": False,
            })
            thetas.append(float(row["theta"] or 0.0))
            games.append(int(row["games"] or 0))

    warnings: list[str] = []
    if missing:
        warnings.append(
            f"Игроков не найдено в нашей базе: {len(missing)}"
            f" (id: {', '.join(str(m) for m in missing)}). Они "
            "пропущены — добавьте корректные ID."
        )

    if not thetas:
        return {
            "teams": [],
            "warnings": warnings or ["Не задано ни одного известного игрока."],
            "pack": pack,
            "players": players_resolved,
            "n_mc_samples": 0,
            "model_params_missing": False,
            "p_q": [],
        }

    sim = simulate_field(
        rosters=[{
            "team_id": 0,
            "team_name": "Ваша команда",
            "thetas": thetas,
            "games": games,
        }],
        b_arr=pack["b"],
        a_arr=pack["a"],
        q_in_tour=pack["q_in_tour"],
        mode=mode,
        n_mc_samples=n_mc_samples,
        rng_seed=rng_seed,
    )
    sim["warnings"] = sim.get("warnings", []) + warnings
    sim["pack"] = pack
    sim["players"] = players_resolved
    # Per-question p_q for the team — re-run the deterministic kernel
    # (the simulate_field discards intermediate p_q).  Cheap.
    mp = db.get_model_params()
    p_q = simulate_roster_on_pack(
        thetas=np.asarray(thetas, dtype=np.float64),
        b=pack["b"],
        a=pack["a"],
        q_in_tour=pack["q_in_tour"],
        delta_size=mp["delta_size"],
        team_size_anchor=mp["team_size_anchor"],
        delta_pos=mp["delta_pos"],
        pos_anchor=mp["pos_anchor"],
        mode=mode,
        lapse_arr=mp["lapse"],
        recal_arr=mp["recal"],
    )
    sim["p_q"] = [float(x) for x in p_q]
    return sim


def list_upcoming_tournaments(
    *, after_iso: str, before_iso: Optional[str] = None
) -> list[dict]:
    """Return upcoming tournaments enriched with ``has_rosters`` flag.

    The ``has_rosters`` flag is *cheap* (the API embeds the team count
    only when ``includeTeamMembers=1``); we don't fetch full rosters per
    item — that would be a separate request per row.  Instead we use the
    presence of ``synchData.dateRequestsAllowedTo`` as a heuristic ('the
    org has set a registration deadline → registrations exist').
    """
    return forecast_api.list_upcoming(
        after_iso=after_iso, before_iso=before_iso, items_per_page=50
    )


def _monte_carlo_scores(
    *,
    boot_inputs: list[dict],
    b_arr: np.ndarray,
    a_arr: np.ndarray,
    q_in_tour: np.ndarray,
    mp: dict,
    mode: str,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised Monte-Carlo scoring.

    For each team:
      * draw ``θ_p ~ Normal(θ̂_p, σ_p²)`` for each player, ``n_samples`` times;
      * recompute the noisy-OR + lapse + recal ``p_q`` per sample;
      * sample ``Bernoulli(p_q)`` and sum to a score per sample.

    Returns an array of shape ``(n_samples, n_teams)``.
    """
    Q = b_arr.shape[0]
    n_teams = len(boot_inputs)
    scores = np.zeros((n_samples, n_teams), dtype=np.int32)

    # Pre-build pos / size shifts once.  They are independent of θ.
    has_size = mp["delta_size"] is not None and mp["team_size_anchor"] is not None
    has_pos = mp["delta_pos"] is not None and mp["pos_anchor"] is not None
    if has_pos:
        dp = np.asarray(mp["delta_pos"], dtype=np.float64)
        tour_len = dp.shape[0]
        qpos = np.asarray(q_in_tour, dtype=np.int64) % tour_len
        pos_shift = dp[qpos] - dp[int(mp["pos_anchor"])]
    else:
        pos_shift = np.zeros(Q, dtype=np.float64)

    if has_size:
        ds = np.asarray(mp["delta_size"], dtype=np.float64)
        size_max = ds.shape[0] - 1
        anchor = int(mp["team_size_anchor"])
    else:
        ds = None
        size_max = 0
        anchor = 0

    # Resolve calibration once per (mode, is_solo).  Fall back to identity.
    from rating.simulate import _calibration_params, apply_probability_calibration
    cal_team = _calibration_params(
        mode, is_solo=False, lapse_arr=mp["lapse"], recal_arr=mp["recal"]
    )
    cal_solo = _calibration_params(
        mode, is_solo=True, lapse_arr=mp["lapse"], recal_arr=mp["recal"]
    )

    for ti, boot in enumerate(boot_inputs):
        thetas = boot["thetas"]
        sigmas = _theta_sigmas(boot["games"])
        n = thetas.shape[0]
        if n == 0:
            continue
        # θ samples: (n_samples, n_players)
        theta_s = thetas[None, :] + sigmas[None, :] * rng.standard_normal(
            size=(n_samples, n)
        )
        # δ_size: depends on actual roster size, fixed per team.
        if has_size and size_max >= 1:
            ts_idx = max(1, min(n, size_max))
            size_shift = float(ds[ts_idx] - ds[anchor])
        else:
            size_shift = 0.0
        b_eff = b_arr + size_shift + pos_shift  # (Q,)
        # Per-question noisy-OR per sample.  Broadcast:
        #   z[s, q] = Σ_p (a[q] · θ_s[s, p]) − b_eff[q]
        # Σ over players of exp(z) gives S_s,q.  Doing the per-player
        # exponential and summing keeps the arithmetic stable for huge
        # negative z (no NaNs), at the cost of an (n_samples, n, Q) tensor.
        # For typical (2000, 6, 36) that's 432 K floats — small.
        z = -b_eff[None, None, :] + theta_s[:, :, None] * a_arr[None, None, :]
        np.clip(z, -20.0, 20.0, out=z)
        lam = np.exp(z)
        S = lam.sum(axis=1)              # (n_samples, Q)
        p_raw = -np.expm1(-S)
        cal = cal_solo if n == 1 else cal_team
        p = apply_probability_calibration(
            p_raw, lapse=cal[0], recal_alpha=cal[1], recal_beta=cal[2]
        )
        # Bernoulli sampling, sum to score.
        u = rng.random(size=p.shape)
        scores[:, ti] = (u < p).sum(axis=1)
    return scores


def _rank_descending_avg(scores: np.ndarray) -> np.ndarray:
    """Average-rank within each row, with rank 1 going to the highest score.

    Matches the standard ЧГК tie-breaking rule (teams sharing a score
    share the average of the places they would otherwise occupy).
    """
    # Convert to "ascending rank of −score" with average ties.
    neg = -scores.astype(np.float64)
    n_samples, n_teams = neg.shape
    out = np.empty_like(neg)
    for i in range(n_samples):
        order = np.argsort(neg[i], kind="mergesort")
        sorted_vals = neg[i, order]
        # Ranks: average over equal-valued runs.
        ranks = np.empty(n_teams, dtype=np.float64)
        j = 0
        while j < n_teams:
            k = j
            while k + 1 < n_teams and sorted_vals[k + 1] == sorted_vals[j]:
                k += 1
            avg_rank = 0.5 * (j + 1 + k + 1)
            ranks[j:k + 1] = avg_rank
            j = k + 1
        out[i, order] = ranks
    return out
