"""
Vectorised simulation of a roster on a pack.

The single entry point ``simulate_roster_on_pack`` computes the model's
predicted take probability ``p_q`` per question for one team, applying
the same noisy-OR + lapse + logit-affine recalibration formula used by
``rating.model.forward`` and ``website.build.build_db.compute_expected_takes``.

This is the shared kernel used both at DuckDB build time (legacy
expected-takes computation) and at website request time (the /forecast
pages).  Keeping the formula in exactly one place is the whole point —
otherwise calibration drift between training and forecasting silently
breaks the predictions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


_MODE_TO_IDX = {"offline": 0, "sync": 1, "async": 2}


def _mode_to_idx(mode: str) -> int:
    return _MODE_TO_IDX.get(mode, 0)


def _calibration_params(
    mode: str,
    is_solo: bool,
    *,
    lapse_arr: Optional[np.ndarray],
    recal_arr: Optional[np.ndarray],
) -> tuple[float, float, float]:
    """Resolve ``(lapse, recal_alpha, recal_beta)`` for a (mode, is_solo) bucket.

    Returns identity calibration ``(0, 0, 1)`` when the corresponding
    array is missing or has an unexpected shape.
    """
    lapse = 0.0
    if lapse_arr is not None:
        arr = np.asarray(lapse_arr, dtype=np.float64)
        if arr.shape == (3, 2):
            lapse = float(arr[_mode_to_idx(mode), 1 if is_solo else 0])
    alpha, beta = 0.0, 1.0
    if recal_arr is not None:
        arr = np.asarray(recal_arr, dtype=np.float64)
        if arr.shape == (3, 2, 2):
            row = arr[_mode_to_idx(mode), 1 if is_solo else 0]
            alpha = float(row[0])
            beta = float(row[1])
    return lapse, alpha, beta


def apply_probability_calibration(
    p_raw: np.ndarray,
    *,
    lapse: float,
    recal_alpha: float,
    recal_beta: float,
    take_floor_min: float = 0.0,
    is_grave: bool = False,
) -> np.ndarray:
    """Apply the lapse cap + logit-affine recal used by ``rating.model``.

    ``p_lapse = (1 - π) · p_raw``;
    ``p_final = sigmoid(α + β · logit(p_lapse))``;
    optional ``p_final = max(p_min, p_final)`` for non-grave questions.

    Identity recalibration (``α=0, β=1``) short-circuits the second step.
    """
    p = (1.0 - float(lapse)) * np.asarray(p_raw, dtype=np.float64)
    if recal_alpha == 0.0 and recal_beta == 1.0:
        out = p
    else:
        p_c = np.clip(p, 1e-15, 1.0 - 1e-15)
        z = recal_alpha + recal_beta * np.log(p_c / (1.0 - p_c))
        out = 1.0 / (1.0 + np.exp(-z))
    if not is_grave and take_floor_min > 0.0:
        out = np.maximum(out, take_floor_min)
    return out


def simulate_roster_on_pack(
    thetas: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    *,
    q_in_tour: Optional[np.ndarray] = None,
    delta_size: Optional[np.ndarray] = None,
    team_size_anchor: Optional[int] = None,
    delta_pos: Optional[np.ndarray] = None,
    pos_anchor: Optional[int] = None,
    team_size: Optional[int] = None,
    mode: str = "offline",
    lapse_arr: Optional[np.ndarray] = None,
    recal_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Predicted take probability ``p_q`` per question for one team.

    Parameters
    ----------
    thetas
        Player strengths, shape ``(n_players,)``.  May be empty.
    b, a
        Question difficulty / discrimination, shape ``(Q,)``.
    q_in_tour
        Per-question 0-based position used to index ``delta_pos``.  When
        ``None``, ``np.arange(Q)`` is used (the typical layout for a
        sequential pack).
    delta_size, team_size_anchor
        Per-team-size shift table; ignored when either is ``None``.  The
        shift used is ``delta_size[ts] - delta_size[anchor]``, where ``ts``
        is clipped to ``[1, len(delta_size) - 1]``.
    delta_pos, pos_anchor
        Per-position shift table (length ``tour_len``); ignored when either
        is ``None``.  Position is taken modulo ``tour_len``.
    team_size
        Override for the size used in ``delta_size`` lookup.  Defaults to
        ``len(thetas)`` — passing this is useful when the caller wants
        to predict for a roster of ``n`` players sized as if it were a
        team of ``team_size`` (e.g. a partial roster that historically
        played as 6).
    mode
        Tournament mode; one of ``"offline" | "sync" | "async"``.
    lapse_arr, recal_arr
        Calibration tables shaped ``(3, 2)`` and ``(3, 2, 2)`` respectively
        (``[mode, is_solo]`` and ``[mode, is_solo, [alpha, beta]]``).
        Both default to identity calibration when ``None`` or wrong shape.

    Returns
    -------
    p : np.ndarray
        Per-question probability of taking, shape ``(Q,)``.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    Q = b_arr.shape[0]
    n = thetas.shape[0]

    if Q == 0:
        return np.zeros(0, dtype=np.float64)

    ts_for_size = team_size if team_size is not None else n

    size_shift = 0.0
    if delta_size is not None and team_size_anchor is not None:
        ds = np.asarray(delta_size, dtype=np.float64)
        size_max = ds.shape[0] - 1
        if size_max >= 1:
            ts_idx = max(1, min(int(ts_for_size), size_max))
            size_shift = float(ds[ts_idx] - ds[int(team_size_anchor)])

    if delta_pos is not None and pos_anchor is not None:
        dp = np.asarray(delta_pos, dtype=np.float64)
        tour_len = dp.shape[0]
        if q_in_tour is None:
            qpos = np.arange(Q, dtype=np.int64) % tour_len
        else:
            qpos = np.asarray(q_in_tour, dtype=np.int64) % tour_len
        pos_shift = dp[qpos] - dp[int(pos_anchor)]
    else:
        pos_shift = np.zeros(Q, dtype=np.float64)

    b_eff = b_arr + size_shift + pos_shift  # (Q,)

    if n == 0:
        # No players → S = 0 → p_raw = 0; calibration only adds lapse-zero
        # which leaves zeros in place.
        p_raw = np.zeros(Q, dtype=np.float64)
    else:
        z = -b_eff[None, :] + np.outer(thetas, a_arr)  # (n, Q)
        lam = np.exp(z)
        S = lam.sum(axis=0)
        p_raw = -np.expm1(-S)

    lapse, alpha, beta = _calibration_params(
        mode, is_solo=(n == 1), lapse_arr=lapse_arr, recal_arr=recal_arr
    )
    return apply_probability_calibration(
        p_raw, lapse=lapse, recal_alpha=alpha, recal_beta=beta
    )
