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


def forecast_past_tournament(
    tournament_id: int,
    *,
    n_mc_samples: int = _DEFAULT_N_SAMPLES,
    rng_seed: int = 20260524,
) -> Optional[dict]:
    """Build the context for the /forecast/tournament/{tid} page.

    Returns ``None`` when the tournament is not in the DB.  Otherwise
    returns a dict with the tournament metadata, the ranked forecast
    table (deterministic E[T] + Monte-Carlo place distribution),
    calibration warnings, and a model-params status flag.

    The Monte-Carlo loop draws ``n_mc_samples`` joint samples of
    ``(θ-bootstrap, Bernoulli takes)`` and ranks teams within each
    sample, so the reported place quantiles already capture both
    sources of uncertainty.  ``rng_seed`` makes the page deterministic.
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
            p.games,
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
                "games": [],
                "player_ids": [],
            },
        )
        if r["theta"] is not None:
            slot["thetas"].append(float(r["theta"]))
            slot["games"].append(int(r["games"] or 0))
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

    # ---- Pass 1: deterministic p_q + E[T] per team ----------------------
    # Build a paired list ``(row, p_q, boot)`` so we can sort by E[T]
    # without losing the alignment to per-team simulation state.
    paired: list[tuple[dict, np.ndarray, dict]] = []
    for slot in teams.values():
        thetas = np.asarray(slot["thetas"], dtype=np.float64)
        if thetas.size == 0:
            # No θ data for any roster member — skip the team rather than
            # invent zeros (zeros would imply a population-average team and
            # silently rank the team mid-table).
            continue
        games = np.asarray(slot["games"], dtype=np.int64)
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
            "team_id": slot["team_id"],
            "team_name": slot["team_name"],
            "expected_takes": float(p_q.sum()),
            "score_actual": slot["score_actual"],
            "place_actual": slot["place_actual"],
            "n_players": int(thetas.size),
            "n_players_active": slot["n_players_active"],
        }
        paired.append((row, p_q, {"thetas": thetas, "games": games}))

    paired.sort(key=lambda t: (-t[0]["expected_takes"], t[0]["team_name"] or ""))
    forecast_rows = [t[0] for t in paired]
    ordered_p = [t[1] for t in paired]
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
        )  # (n_samples, n_teams)
        # Rank within each sample (1 = best score).  Ties get the average
        # rank, which is what the actual ChGK rule applies in tournaments.
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
        "tournament": tournament,
        "teams": forecast_rows,
        "warnings": warnings,
        "model_params_missing": mp["delta_size"] is None,
        "n_mc_samples": int(n_mc_samples),
    }


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
