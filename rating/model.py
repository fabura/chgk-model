"""
Noisy-OR probability model.

    z_k = -(b_i + δ) + a_i · θ_k
    λ_k = exp(z_k)
    S   = Σ_k λ_k
    p   = 1 − exp(−S)   computed as  −expm1(−S)

δ collects the auxiliary shifts (per-team-size + per-position-in-tour;
positive = harder) — see ``rating.engine``.

Numba JIT versions (forward_nb, gradients_nb, process_batch_nb) for speed.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = lambda fn, **kw: fn


def forward(
    theta: np.ndarray,
    b: float,
    a: float,
    delta: float = 0.0,
    clamp: float = 20.0,
    lapse: float = 0.0,
) -> tuple[float, float, np.ndarray]:
    """
    Compute team-answer probability for one (team, question) pair.

    Args:
        theta: player strengths for the team  [team_size]
        b:     question difficulty
        a:     question discrimination  (> 0)
        delta: tournament difficulty offset (positive = harder)
        clamp: bounds on z_k to avoid exp overflow
        lapse: lapse-rate floor π ∈ [0, 1).  ``p = (1 − π) · p_noisy_or``,
            i.e. the predicted probability is capped at ``(1 − π)``.

    Returns:
        (p, S, lam)  where lam[k] = λ_k
    """
    eff_b = b + delta
    z = np.clip(-eff_b + a * theta, -clamp, clamp)
    lam = np.exp(z)
    S = float(lam.sum())
    p_raw = float(-math.expm1(-S))
    p = (1.0 - lapse) * p_raw
    p = max(min(p, 1.0 - 1e-15), 1e-15)
    return p, S, lam


def gradients(
    S: float,
    lam: np.ndarray,
    a: float,
    theta: np.ndarray,
    y: int,
) -> tuple[np.ndarray, float, float, float]:
    """
    Gradients of log-likelihood for one (team, question) observation.

    Args:
        S:     team intensity  Σ λ_k
        lam:   per-player λ_k  [team_size]
        a:     question discrimination
        theta: player strengths  [team_size]
        y:     observed outcome  {0, 1}

    Returns:
        (dL_dtheta, dL_db, dL_dlog_a, dL_ddelta)

    dL_ddelta = dL_db (same formula, since δ appears with b in eff_b = b + δ).
    """
    if y == 0:
        dL_dS = -1.0
    else:
        if S > 500.0:
            dL_dS = 0.0
        else:
            dL_dS = 1.0 / max(math.expm1(S), 1e-15)

    dL_db = dL_dS * (-S)
    dL_ddelta = dL_db
    dL_dtheta = dL_dS * a * lam
    dL_dlog_a = dL_dS * float(np.sum(lam * a * theta))

    return dL_dtheta, dL_db, dL_dlog_a, dL_ddelta


# =============================================================================
# Numba-accelerated (vectorized) batch processing
# =============================================================================

@njit(cache=True)
def _forward_nb(theta: np.ndarray, b: float, a: float, delta: float, clamp: float
) -> tuple[float, float, np.ndarray]:
    """Single obs forward. Returns (p, S, lam)."""
    eff_b = b + delta
    z = np.empty_like(theta)
    for k in range(len(theta)):
        zk = -eff_b + a * theta[k]
        if zk < -clamp:
            z[k] = -clamp
        elif zk > clamp:
            z[k] = clamp
        else:
            z[k] = zk
    S = 0.0
    for k in range(len(theta)):
        S += math.exp(z[k])
    # p = 1 - exp(-S)
    if S > 1e-10:
        p = 1.0 - math.exp(-S)
    else:
        p = S
    if p < 1e-15:
        p = 1e-15
    elif p > 1.0 - 1e-15:
        p = 1.0 - 1e-15
    lam = np.exp(z)
    return p, S, lam


@njit(cache=True)
def _gradients_nb(S: float, lam: np.ndarray, a: float, theta: np.ndarray, y: int
) -> tuple[np.ndarray, float, float, float]:
    """Single obs gradients. Returns (dL_dtheta, dL_db, dL_dlog_a, dL_ddelta)."""
    if y == 0:
        dL_dS = -1.0
    else:
        if S > 500.0:
            dL_dS = 0.0
        else:
            expm1_S = math.expm1(S)
            if expm1_S < 1e-15:
                expm1_S = 1e-15
            dL_dS = 1.0 / expm1_S
    dL_db = dL_dS * (-S)
    dL_dtheta = np.empty_like(lam)
    for k in range(len(lam)):
        dL_dtheta[k] = dL_dS * a * lam[k]
    lam_theta_sum = 0.0
    for k in range(len(lam)):
        lam_theta_sum += lam[k] * a * theta[k]
    dL_dlog_a = dL_dS * lam_theta_sum
    return dL_dtheta, dL_db, dL_dlog_a, dL_db


@njit(cache=True)
def process_batch_nb(
    obs_indices: np.ndarray,
    offsets: np.ndarray,
    player_flat: np.ndarray,
    q_idx: np.ndarray,
    taken: np.ndarray,
    cq: np.ndarray,
    q_pos_in_tour: np.ndarray,
    theta: np.ndarray,
    b: np.ndarray,
    log_a: np.ndarray,
    delta_size: np.ndarray,
    delta_pos: np.ndarray,
    games: np.ndarray,
    eta0: float,
    theta_w: float,
    b_w: float,
    log_a_w: float,
    size_w: float,
    pos_w: float,
    eta_size: float,
    eta_pos: float,
    reg_size: float,
    reg_pos: float,
    size_anchor: int,
    pos_anchor: int,
    reg_theta: float = 0.0,
    reg_b: float = 0.0,
    reg_log_a: float = 0.0,
    clamp: float = 20.0,
    games_offset: float = 1.0,
    lapse_arr: np.ndarray = np.zeros(1, dtype=np.float64),
    eta_lapse: float = 0.0,
    lapse_max: float = 0.30,
) -> float:
    """
    Process a batch of observations.

    Updates ``theta``, ``b``, ``log_a``, ``delta_size``, ``delta_pos``
    and ``lapse_arr[0]`` in-place.

    The effective tournament shift is

        δ = delta_size[team_size] + delta_pos[q_pos_in_tour[qi_raw]]

    where ``delta_size`` is anchored at ``size_anchor`` and
    ``delta_pos`` at ``pos_anchor``: updates are skipped at the anchor
    index, and the anchor entry is treated as structurally zero.

    The forward applies a lapse-rate floor:

        p = (1 − π) · (1 − exp(−S))   with  π = lapse_arr[0]

    so the model can learn to leave the high-p region uncalibrated
    when format-specific noise (typos, distraction, glitches) keeps
    even strong teams from achieving p ≈ 1.  When ``eta_lapse > 0``
    the per-batch lapse is itself updated; otherwise it stays fixed
    at the value the caller put in.

    L2-style shrinkage is applied as a multiplicative pull toward zero
    after each gradient step (``param *= max(0, 1 - lr * reg)``).  For
    ``theta`` the shrinkage uses the player's adaptive lr; for question
    parameters it uses ``eta0`` (matching their gradient update).

    Returns total log-likelihood for the batch.
    """
    total_ll = 0.0
    max_size_idx = len(delta_size) - 1
    n_pos = len(delta_pos)
    lapse = lapse_arr[0]
    if lapse < 0.0:
        lapse = 0.0
    elif lapse > lapse_max:
        lapse = lapse_max
    one_minus_lapse = 1.0 - lapse
    log_one_minus_lapse = math.log(one_minus_lapse) if one_minus_lapse > 1e-15 else math.log(1e-15)
    dlapse_acc = 0.0  # accumulated dL/dπ over the batch
    for j in range(len(obs_indices)):
        i = obs_indices[j]
        s, e = int(offsets[i]), int(offsets[i + 1])
        qi_raw = int(q_idx[i])
        qi = int(cq[qi_raw])
        y = int(taken[i])
        team_size_raw = e - s
        if team_size_raw < 1:
            ts_idx = 1
        elif team_size_raw > max_size_idx:
            ts_idx = max_size_idx
        else:
            ts_idx = team_size_raw
        pos_idx = int(q_pos_in_tour[qi_raw])
        if pos_idx < 0:
            pos_idx = 0
        elif pos_idx >= n_pos:
            pos_idx = n_pos - 1
        delta_g = 0.0
        if ts_idx != size_anchor:
            delta_g += delta_size[ts_idx]
        if pos_idx != pos_anchor:
            delta_g += delta_pos[pos_idx]
        b_val = b[qi]
        log_a_val = log_a[qi]
        if log_a_val > 3.0:
            log_a_val = 3.0
        elif log_a_val < -3.0:
            log_a_val = -3.0
        a_val = math.exp(log_a_val)
        team_size = e - s
        th = np.empty(team_size)
        pids = np.empty(team_size, dtype=np.int32)
        for k in range(team_size):
            pids[k] = player_flat[s + k]
            th[k] = theta[pids[k]]
        p_raw, S, lam = _forward_nb(th, b_val, a_val, delta_g, clamp)
        if y == 1:
            # log p = log(1-π) + log p_raw
            ll_term = log_one_minus_lapse + math.log(p_raw)
            total_ll += ll_term
            # dL/dS unchanged from no-lapse case (constant shift in π)
            grad_scale = 1.0
            # dL/dπ = -1/(1-π)
            dlapse_acc += -1.0 / max(one_minus_lapse, 1e-15)
        else:
            # 1 - p = 1 - (1-π) p_raw = π + (1-π) (1 - p_raw)
            # exp(-S) = 1 - p_raw, so:
            # 1 - p = π + (1-π) exp(-S)
            exp_neg_S = math.exp(-S) if S < 500.0 else 0.0
            q = lapse + one_minus_lapse * exp_neg_S
            if q < 1e-15:
                q = 1e-15
            total_ll += math.log(q)
            # dL/dS = -(1-π) exp(-S) / q  vs  old -1
            # gradient scaling factor relative to the no-lapse case:
            # old dL/dS = -1 ⇒ scale = (1-π) exp(-S) / q
            grad_scale = one_minus_lapse * exp_neg_S / q
            # dL/dπ = (1 - exp(-S)) / q
            dlapse_acc += (1.0 - exp_neg_S) / q
        dL_dth, dL_db, dL_dloga, dL_ddelta = _gradients_nb(S, lam, a_val, th, y)
        if grad_scale != 1.0:
            for k in range(team_size):
                dL_dth[k] *= grad_scale
            dL_db *= grad_scale
            dL_dloga *= grad_scale
            dL_ddelta *= grad_scale
        for k in range(team_size):
            pidx = pids[k]
            lr = eta0 / math.sqrt(games_offset + games[pidx])
            theta[pidx] += theta_w * lr * dL_dth[k]
            if reg_theta > 0.0:
                shrink = 1.0 - lr * reg_theta
                if shrink < 0.0:
                    shrink = 0.0
                theta[pidx] *= shrink
        b_val_new = b[qi] + b_w * eta0 * dL_db
        if reg_b > 0.0:
            shrink_b = 1.0 - eta0 * reg_b
            if shrink_b < 0.0:
                shrink_b = 0.0
            b_val_new *= shrink_b
        b[qi] = b_val_new
        log_a_val_new = log_a[qi] + log_a_w * eta0 * dL_dloga
        if reg_log_a > 0.0:
            shrink_la = 1.0 - eta0 * reg_log_a
            if shrink_la < 0.0:
                shrink_la = 0.0
            log_a_val_new *= shrink_la
        log_a[qi] = log_a_val_new
        if b[qi] > 10.0:
            b[qi] = 10.0
        elif b[qi] < -10.0:
            b[qi] = -10.0
        if log_a[qi] > 3.0:
            log_a[qi] = 3.0
        elif log_a[qi] < -3.0:
            log_a[qi] = -3.0
        if ts_idx != size_anchor and size_w > 0.0 and eta_size > 0.0:
            ds_val = delta_size[ts_idx] + size_w * eta_size * dL_ddelta
            if reg_size > 0.0:
                ds_val *= max(0.0, 1.0 - eta_size * reg_size)
            if ds_val > 10.0:
                ds_val = 10.0
            elif ds_val < -10.0:
                ds_val = -10.0
            delta_size[ts_idx] = ds_val
        if pos_idx != pos_anchor and pos_w > 0.0 and eta_pos > 0.0:
            dp_val = delta_pos[pos_idx] + pos_w * eta_pos * dL_ddelta
            if reg_pos > 0.0:
                dp_val *= max(0.0, 1.0 - eta_pos * reg_pos)
            if dp_val > 10.0:
                dp_val = 10.0
            elif dp_val < -10.0:
                dp_val = -10.0
            delta_pos[pos_idx] = dp_val
    if eta_lapse > 0.0 and len(obs_indices) > 0:
        # Batch update: step in mean-gradient direction.  Gradient
        # ascent (we accumulated dL/dπ, want to maximise log-lik).
        new_lapse = lapse + eta_lapse * (dlapse_acc / float(len(obs_indices)))
        if new_lapse < 0.0:
            new_lapse = 0.0
        elif new_lapse > lapse_max:
            new_lapse = lapse_max
        lapse_arr[0] = new_lapse
    return total_ll
