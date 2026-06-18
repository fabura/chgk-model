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
    recal_alpha: float = 0.0,
    recal_beta: float = 1.0,
    take_floor_min: float = 0.0,
    is_grave: bool = False,
) -> tuple[float, float, np.ndarray]:
    """
    Compute team-answer probability for one (team, question) pair.

    Forward chain:
        S       = Σ exp(−(b + δ) + a · θ_k)
        p_raw   = 1 − exp(−S)
        p_lapse = (1 − π) · p_raw                      # lapse cap
        p_final = sigmoid(α + β · logit(p_lapse))      # logit-affine recal
        p_final = max(p_min, p_final)  if non-grave     # optional take floor

    Identity recalibration: ``α = 0, β = 1``.  The take floor (when
    ``take_floor_min > 0``) is applied **after** recalibration and is
    skipped for grave questions (empirical take rate = 0).

    Args:
        theta:        player strengths for the team  [team_size]
        b:            question difficulty
        a:            question discrimination  (> 0)
        delta:        tournament difficulty offset (positive = harder)
        clamp:        bounds on z_k to avoid exp overflow
        lapse:        π ∈ [0, 1).  Floor on the predicted probability.
        recal_alpha:  α additive shift in logit space (default 0 = identity).
        recal_beta:   β multiplicative scale in logit space (default 1).
        take_floor_min: minimum p_take for non-grave questions (0 = off).
        is_grave:     skip the floor for zero-take-rate questions.

    Returns:
        (p, S, lam)  where lam[k] = λ_k
    """
    eff_b = b + delta
    z_lin = np.clip(-eff_b + a * theta, -clamp, clamp)
    lam = np.exp(z_lin)
    S = float(lam.sum())
    p_raw = float(-math.expm1(-S))
    p_lapse = (1.0 - lapse) * p_raw
    if recal_alpha == 0.0 and recal_beta == 1.0:
        p = p_lapse
    else:
        p_lapse_c = max(min(p_lapse, 1.0 - 1e-15), 1e-15)
        z = recal_alpha + recal_beta * math.log(
            p_lapse_c / (1.0 - p_lapse_c)
        )
        if z >= 0:
            p = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            p = ez / (1.0 + ez)
    if not is_grave and take_floor_min > 0.0 and p < take_floor_min:
        p = take_floor_min
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
def _forward_nb(theta: np.ndarray, b: float, a: float, delta: float, clamp: float,
                gamma: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Single obs forward. Returns (p, S, lam).

    When ``gamma`` is not None: uses the 2D player model
        z_k = -eff_b + a·θ_k + γ_k·b
    instead of the standard
        z_k = -eff_b + a·θ_k
    """
    eff_b = b + delta
    use_gamma = gamma is not None
    z = np.empty_like(theta)
    for k in range(len(theta)):
        zk = -eff_b + a * theta[k]
        if use_gamma:
            zk += gamma[k] * b
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
def _gradients_nb(S: float, lam: np.ndarray, a: float, theta: np.ndarray, y: int,
                  gamma: np.ndarray | None = None,
                  b_val: float = 0.0,
) -> tuple[np.ndarray, float, float, float, np.ndarray | None]:
    """Single obs gradients.

    Returns (dL_dtheta, dL_db, dL_dlog_a, dL_ddelta, dL_dgamma).

    When ``gamma`` is not None, corrects dL_db for the 2D player
    model (dS/db_extra = Σ λ_k·γ_k) and returns per-player
    dL_dγ_k = dL_dS · λ_k · b_val.
    """
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
    use_gamma = gamma is not None
    if use_gamma:
        dL_dgamma = np.empty_like(lam)
        lam_gamma_sum = 0.0
        for k in range(len(lam)):
            dL_dgamma[k] = dL_dS * lam[k] * b_val
            lam_gamma_sum += lam[k] * gamma[k]
        dL_db += dL_dS * lam_gamma_sum  # correction: dS/db includes Σ λ_k·γ_k
    else:
        dL_dgamma_out = np.zeros(1, dtype=np.float64)  # dummy, won't be used
    lam_theta_sum = 0.0
    for k in range(len(lam)):
        lam_theta_sum += lam[k] * a * theta[k]
    dL_dlog_a = dL_dS * lam_theta_sum
    return dL_dtheta, dL_db, dL_dlog_a, dL_db, dL_dgamma if use_gamma else dL_dgamma_out


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
    min_eta: float = 0.0,
    lapse_arr: np.ndarray = np.zeros(1, dtype=np.float64),
    eta_lapse: float = 0.0,
    lapse_max: float = 0.30,
    recal_arr: np.ndarray = np.array([0.0, 1.0], dtype=np.float64),
    eta_recal: float = 0.0,
    recal_alpha_max: float = 3.0,
    recal_beta_min: float = 0.30,
    recal_beta_max: float = 2.00,
    # 2D player model (Model C: per-player difficulty slope).
    # When ``gamma`` is not None (shape == theta.shape), the forward
    # becomes z_k = -eff_b + a·θ_k + γ_k·b and the gradient on θ/b
    # is corrected accordingly.
    gamma: np.ndarray | None = None,
    gamma_w: float = 1.0,
    eta_gamma: float = 0.0,
    reg_gamma: float = 0.0,
    gamma_max: float = 2.0,
    # Difficulty-weighted loss (see ``Config.diff_w_*``).
    diff_w_miss_power: float = 0.0,
    diff_w_take_boost: float = 0.0,
    take_floor_min: float = 0.0,
    grave_q: np.ndarray | None = None,
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
    recal_alpha = recal_arr[0]
    recal_beta = recal_arr[1]
    if recal_beta < recal_beta_min:
        recal_beta = recal_beta_min
    elif recal_beta > recal_beta_max:
        recal_beta = recal_beta_max
    if recal_alpha < -recal_alpha_max:
        recal_alpha = -recal_alpha_max
    elif recal_alpha > recal_alpha_max:
        recal_alpha = recal_alpha_max
    is_identity_recal = (recal_alpha == 0.0) and (recal_beta == 1.0)
    use_2d = gamma is not None
    use_take_floor = take_floor_min > 0.0 and grave_q is not None
    dlapse_acc = 0.0  # accumulated dL/dπ over the batch
    drecal_alpha_acc = 0.0
    drecal_beta_acc = 0.0
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
        gm = np.empty(team_size) if use_2d else np.zeros(1)
        for k in range(team_size):
            pids[k] = player_flat[s + k]
            th[k] = theta[pids[k]]
            if use_2d:
                gm[k] = gamma[pids[k]]
        p_raw, S, lam = _forward_nb(
            th, b_val, a_val, delta_g, clamp,
            gamma=gm if use_2d else None,
        )
        # Forward chain: p_raw → p_lapse = (1-π)·p_raw → p_final = sigmoid(α + β·logit(p_lapse))
        p_lapse = one_minus_lapse * p_raw
        if p_lapse < 1e-15:
            p_lapse = 1e-15
        elif p_lapse > 1.0 - 1e-15:
            p_lapse = 1.0 - 1e-15
        one_minus_p_lapse = 1.0 - p_lapse
        if is_identity_recal:
            p_final = p_lapse
            logit_p_lapse = 0.0  # unused
        else:
            logit_p_lapse = math.log(p_lapse / one_minus_p_lapse)
            z_recal = recal_alpha + recal_beta * logit_p_lapse
            if z_recal >= 0:
                p_final = 1.0 / (1.0 + math.exp(-z_recal))
            else:
                ez = math.exp(z_recal)
                p_final = ez / (1.0 + ez)
            if p_final < 1e-15:
                p_final = 1e-15
            elif p_final > 1.0 - 1e-15:
                p_final = 1.0 - 1e-15
        floor_active = False
        if use_take_floor and grave_q[qi] == 0:
            if p_final < take_floor_min:
                p_final = take_floor_min
                floor_active = True
        # Optional per-observation difficulty weight (forward unchanged).
        obs_w = 1.0
        if diff_w_miss_power > 0.0 or diff_w_take_boost > 0.0:
            one_minus_p_final = 1.0 - p_final
            if one_minus_p_final < 1e-15:
                one_minus_p_final = 1e-15
            if y == 0 and diff_w_miss_power > 0.0:
                obs_w = one_minus_p_final ** diff_w_miss_power
            elif y == 1 and diff_w_take_boost > 0.0:
                obs_w = 1.0 + diff_w_take_boost * one_minus_p_final
        # Log-likelihood + dL/dz_recal (= y - p_final), then unified
        # dL/dS via chain through (take floor → logit-affine → lapse cap
        # → noisy-OR).  When the take floor binds, p_final is constant
        # and no gradient flows to θ / b / lapse / recal.
        if floor_active:
            if y == 1:
                total_ll += obs_w * math.log(p_final)
            else:
                total_ll += obs_w * math.log(1.0 - p_final)
            continue
        if y == 1:
            total_ll += obs_w * math.log(p_final)
            dL_dz = obs_w * (1.0 - p_final)
        else:
            total_ll += obs_w * math.log(1.0 - p_final)
            dL_dz = obs_w * (-p_final)
        # dL/dα = dL/dz · 1
        # dL/dβ = dL/dz · logit(p_lapse)
        if not is_identity_recal:
            drecal_alpha_acc += dL_dz
            drecal_beta_acc += dL_dz * logit_p_lapse
        else:
            # Even at identity (β=1, α=0) we still accumulate so that
            # the params can move off identity if data prefers it.
            drecal_alpha_acc += dL_dz
            drecal_beta_acc += dL_dz * math.log(p_lapse / one_minus_p_lapse)
        # dL/dπ via chain: dL/dπ = -dL/dz · β · p_raw / (p_lapse · (1-p_lapse))
        dlapse_acc += -dL_dz * recal_beta * p_raw / (p_lapse * one_minus_p_lapse)
        # Gradients on θ, b, log_a, δ go through dL/dS.  We compute them via
        # _gradients_nb (which returns base gradients assuming dL/dS_base =
        # 1/expm1(S) for y=1 or -1 for y=0), then multiply by grad_scale =
        # new_dL_dS / dL_dS_base.
        dL_dth, dL_db, dL_dloga, dL_ddelta, dL_dgamma = _gradients_nb(
            S, lam, a_val, th, y,
            gamma=gm if use_2d else None,
            b_val=b_val,
        )
        # Unified grad_scale derivation:
        # new_dL_dS = dL_dz · β · (1-π) · exp(-S) / [p_lapse · (1-p_lapse)]
        # For y=1: base_dL_dS = exp(-S)/p_raw, so
        #          scale = (1-p_final) · β / (1-p_lapse)
        # For y=0: base_dL_dS = -1, so
        #          scale = p_final · β · (1-π) · exp(-S) / [p_lapse·(1-p_lapse)]
        if y == 1:
            grad_scale = (1.0 - p_final) * recal_beta / one_minus_p_lapse
        else:
            exp_neg_S = math.exp(-S) if S < 500.0 else 0.0
            grad_scale = (
                p_final * recal_beta * one_minus_lapse * exp_neg_S
                / (p_lapse * one_minus_p_lapse)
            )
        if grad_scale != 1.0:
            for k in range(team_size):
                dL_dth[k] *= grad_scale
            dL_db *= grad_scale
            dL_dloga *= grad_scale
            dL_ddelta *= grad_scale
            if use_2d:
                for k in range(team_size):
                    dL_dgamma[k] *= grad_scale
        for k in range(team_size):
            pidx = pids[k]
            lr = eta0 / math.sqrt(games_offset + games[pidx])
            if min_eta > 0.0 and lr < min_eta:
                lr = min_eta
            theta[pidx] += theta_w * lr * dL_dth[k]
            if reg_theta > 0.0:
                shrink = 1.0 - lr * reg_theta
                if shrink < 0.0:
                    shrink = 0.0
                theta[pidx] *= shrink
            if use_2d and eta_gamma > 0.0:
                gamma[pidx] += gamma_w * eta_gamma * dL_dgamma[k]
                if reg_gamma > 0.0:
                    shrink_g = 1.0 - eta_gamma * reg_gamma
                    if shrink_g < 0.0:
                        shrink_g = 0.0
                    gamma[pidx] *= shrink_g
                if gamma[pidx] > gamma_max:
                    gamma[pidx] = gamma_max
                elif gamma[pidx] < -gamma_max:
                    gamma[pidx] = -gamma_max
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
        new_lapse = lapse + eta_lapse * (dlapse_acc / float(len(obs_indices)))
        if new_lapse < 0.0:
            new_lapse = 0.0
        elif new_lapse > lapse_max:
            new_lapse = lapse_max
        lapse_arr[0] = new_lapse
    if eta_recal > 0.0 and len(obs_indices) > 0:
        n = float(len(obs_indices))
        new_alpha = recal_alpha + eta_recal * (drecal_alpha_acc / n)
        new_beta = recal_beta + eta_recal * (drecal_beta_acc / n)
        if new_alpha < -recal_alpha_max:
            new_alpha = -recal_alpha_max
        elif new_alpha > recal_alpha_max:
            new_alpha = recal_alpha_max
        if new_beta < recal_beta_min:
            new_beta = recal_beta_min
        elif new_beta > recal_beta_max:
            new_beta = recal_beta_max
        recal_arr[0] = new_alpha
        recal_arr[1] = new_beta
    return total_ll
