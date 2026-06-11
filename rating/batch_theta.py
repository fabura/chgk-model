"""Stationary batch fit of player θ with fixed question / calibration params.

Fits θ jointly on all (or a date-filtered subset of) observations while holding
``b``, ``a``, ``δ_size``, ``δ_pos``, lapse, and recalibration fixed at the values
from a completed sequential run.  No temporal ordering, no per-game learning-rate
decay, no teammate shrinkage — a reference estimate for comparing against
online θ.

See ``scripts/diagnostic_batch_theta_outliers.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rating.engine import _mode_idx
from rating.io import RatingResults

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco if not args else deco(args[0])

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None  # type: ignore


@dataclass
class BatchFitContext:
    """Precomputed observation metadata for repeated loss/grad evaluations."""

    offsets: np.ndarray
    player_flat: np.ndarray
    q_idx: np.ndarray
    taken: np.ndarray
    team_sizes: np.ndarray
    game_idx: np.ndarray
    # Per raw-question index.
    qi_canon: np.ndarray
    q_pos_in_tour: np.ndarray
    # Per canonical question.
    b: np.ndarray
    log_a: np.ndarray
    # Per observation (aligned with obs index).
    delta_g: np.ndarray
    lapse: np.ndarray
    recal_alpha: np.ndarray
    recal_beta: np.ndarray
    # Player index sets.
    veteran_pidx: np.ndarray
    online_theta: np.ndarray
    obs_mask: np.ndarray  # bool, which observations to include


def build_batch_context(
    arrays: dict[str, np.ndarray],
    maps,
    res: RatingResults,
    *,
    min_games: int = 150,
    recent_days: Optional[int] = None,
    obs_subsample: Optional[float] = None,
    subsample_seed: int = 42,
) -> BatchFitContext:
    """Build context; optionally restrict observations to recent calendar days."""
    q_idx = arrays["q_idx"].astype(np.int64)
    taken = arrays["taken"].astype(np.int64)
    team_sizes = arrays["team_sizes"].astype(np.int32)
    player_flat = arrays["player_indices_flat"].astype(np.int64)
    offsets = np.zeros(len(q_idx) + 1, dtype=np.int64)
    np.cumsum(team_sizes, out=offsets[1:])
    n_obs = len(q_idx)
    n_players = maps.num_players
    n_q = maps.num_questions

    gdo = maps.game_date_ordinal
    game_idx = arrays.get("game_idx")
    if game_idx is None and maps.question_game_idx is not None:
        game_idx = maps.question_game_idx[q_idx]
    game_idx = np.asarray(game_idx, dtype=np.int64)

    # Online θ aligned to player index order in cache.
    pid_to_idx = {int(p): i for i, p in enumerate(maps.idx_to_player_id)}
    online_theta = np.zeros(n_players, dtype=np.float64)
    online_games = np.zeros(n_players, dtype=np.int64)
    for pid, th, gm in zip(res.player_id, res.theta, res.games):
        idx = pid_to_idx.get(int(pid))
        if idx is not None:
            online_theta[idx] = float(th)
            online_games[idx] = int(gm)

    veteran_pidx = np.where(online_games >= min_games)[0].astype(np.int64)

    obs_mask = np.ones(n_obs, dtype=bool)
    if recent_days is not None and gdo is not None:
        max_ord = int(np.max(gdo[gdo >= 0])) if np.any(gdo >= 0) else -1
        cutoff = max_ord - int(recent_days)
        obs_mask = np.zeros(n_obs, dtype=bool)
        for i in range(n_obs):
            g = int(game_idx[i])
            if 0 <= g < len(gdo) and int(gdo[g]) >= cutoff:
                obs_mask[i] = True

    if obs_subsample is not None and 0.0 < obs_subsample < 1.0:
        rng = np.random.default_rng(subsample_seed)
        idx = np.where(obs_mask)[0]
        keep = rng.choice(
            idx, size=max(1, int(len(idx) * obs_subsample)), replace=False,
        )
        obs_mask[:] = False
        obs_mask[keep] = True

    cq = res.canonical_q_idx
    if cq is None:
        cq = np.arange(n_q, dtype=np.int64)
    else:
        cq = np.asarray(cq, dtype=np.int64)

    qi_canon = cq[q_idx]
    b = np.asarray(res.b, dtype=np.float64)
    log_a = np.log(np.maximum(np.asarray(res.a, dtype=np.float64), 1e-15))

    tour_len = len(res.delta_pos) if res.delta_pos is not None else 12
    pos_anchor = int(res.pos_anchor or 0)
    size_anchor = int(res.team_size_anchor or 6)
    delta_size = (
        np.asarray(res.delta_size, dtype=np.float64)
        if res.delta_size is not None
        else np.zeros(13, dtype=np.float64)
    )
    delta_pos = (
        np.asarray(res.delta_pos, dtype=np.float64)
        if res.delta_pos is not None
        else np.zeros(tour_len, dtype=np.float64)
    )
    team_size_max = len(delta_size) - 1

    q_pos_in_tour = np.zeros(n_q, dtype=np.int32)
    qids = getattr(maps, "idx_to_question_id", None)
    if qids is not None and len(qids) > 0 and isinstance(qids[0], tuple):
        for raw_qi in range(min(n_q, len(qids))):
            q_pos_in_tour[raw_qi] = int(qids[raw_qi][1]) % tour_len
    else:
        for raw_qi in range(n_q):
            q_pos_in_tour[raw_qi] = raw_qi % tour_len

    delta_g = np.zeros(n_obs, dtype=np.float64)
    lapse = np.zeros(n_obs, dtype=np.float64)
    recal_alpha = np.zeros(n_obs, dtype=np.float64)
    recal_beta = np.ones(n_obs, dtype=np.float64)

    lapse_arr = res.lapse if res.lapse is not None else np.zeros((3, 2))
    recal_arr = res.recal if res.recal is not None else np.zeros((3, 2, 2))
    game_types = maps.game_type

    for i in range(n_obs):
        raw_q = int(q_idx[i])
        ts_raw = int(team_sizes[i])
        ts_idx = max(1, min(ts_raw, team_size_max))
        pos_idx = int(q_pos_in_tour[raw_q])
        dg = 0.0
        if ts_idx != size_anchor:
            dg += float(delta_size[ts_idx])
        if pos_idx != pos_anchor:
            dg += float(delta_pos[pos_idx])
        delta_g[i] = dg

        g = int(game_idx[i])
        gt = str(game_types[g]) if game_types is not None and g < len(game_types) else "offline"
        mi = _mode_idx(gt)
        solo = ts_raw == 1
        lapse[i] = float(lapse_arr[mi, 1 if solo else 0])
        recal_alpha[i] = float(recal_arr[mi, 1 if solo else 0, 0])
        recal_beta[i] = float(recal_arr[mi, 1 if solo else 0, 1])

    return BatchFitContext(
        offsets=offsets,
        player_flat=player_flat,
        q_idx=q_idx,
        taken=taken,
        team_sizes=team_sizes,
        game_idx=game_idx,
        qi_canon=qi_canon,
        q_pos_in_tour=q_pos_in_tour,
        b=b,
        log_a=log_a,
        delta_g=delta_g,
        lapse=lapse,
        recal_alpha=recal_alpha,
        recal_beta=recal_beta,
        veteran_pidx=veteran_pidx,
        online_theta=online_theta,
        obs_mask=obs_mask,
    )


@njit(cache=True)
def _eval_loss_grad_nb(
    theta: np.ndarray,
    is_veteran: np.ndarray,
    obs_indices: np.ndarray,
    offsets: np.ndarray,
    player_flat: np.ndarray,
    taken: np.ndarray,
    qi_canon: np.ndarray,
    b: np.ndarray,
    log_a: np.ndarray,
    delta_g: np.ndarray,
    lapse: np.ndarray,
    recal_alpha: np.ndarray,
    recal_beta: np.ndarray,
    reg_theta: float,
    veteran_pidx: np.ndarray,
    x_vet: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Numba: log-likelihood + θ gradients (mirrors engine lapse/recal chain)."""
    n_players = len(theta)
    grad = np.zeros(n_players, dtype=np.float64)
    ll = 0.0
    eps = 1e-15

    for j in range(len(obs_indices)):
        i = int(obs_indices[j])
        s, e = int(offsets[i]), int(offsets[i + 1])
        team_size = e - s
        qi = int(qi_canon[i])
        log_a_val = log_a[qi]
        if log_a_val > 3.0:
            log_a_val = 3.0
        elif log_a_val < -3.0:
            log_a_val = -3.0
        a_val = math.exp(log_a_val)
        eff_b = b[qi] + delta_g[i]
        lapse_i = lapse[i]
        alpha = recal_alpha[i]
        beta = recal_beta[i]
        y = int(taken[i])

        S = 0.0
        for k in range(team_size):
            pid = int(player_flat[s + k])
            zk = -eff_b + a_val * theta[pid]
            if zk < -20.0:
                zk = -20.0
            elif zk > 20.0:
                zk = 20.0
            S += math.exp(zk)

        if S > 1e-10:
            p_raw = 1.0 - math.exp(-S)
        else:
            p_raw = S
        one_minus_lapse = 1.0 - lapse_i
        p_lapse = one_minus_lapse * p_raw
        if p_lapse < eps:
            p_lapse = eps
        one_minus_p_lapse = 1.0 - p_lapse

        if alpha == 0.0 and beta == 1.0:
            p_final = p_lapse
            logit_p_lapse = 0.0
        else:
            logit_p_lapse = math.log(p_lapse / one_minus_p_lapse)
            z_recal = alpha + beta * logit_p_lapse
            if z_recal >= 0.0:
                p_final = 1.0 / (1.0 + math.exp(-z_recal))
            else:
                ez = math.exp(z_recal)
                p_final = ez / (1.0 + ez)
        if p_final < eps:
            p_final = eps
        elif p_final > 1.0 - eps:
            p_final = 1.0 - eps

        if y == 1:
            ll += math.log(p_final)
        else:
            ll += math.log(1.0 - p_final)

        # Base noisy-OR dL/dS then engine grad_scale (rating/model.py).
        if y == 0:
            dL_dS = -1.0
        else:
            if S > 500.0:
                dL_dS = 0.0
            else:
                expm1_S = math.expm1(S)
                if expm1_S < eps:
                    expm1_S = eps
                dL_dS = 1.0 / expm1_S

        if y == 1:
            grad_scale = (1.0 - p_final) * beta / one_minus_lapse
        else:
            exp_neg_S = math.exp(-S) if S < 500.0 else 0.0
            grad_scale = (
                p_final * beta * one_minus_lapse * exp_neg_S
                / (p_lapse * one_minus_p_lapse)
            )

        for k in range(team_size):
            pid = int(player_flat[s + k])
            zk = -eff_b + a_val * theta[pid]
            if zk < -20.0:
                zk = -20.0
            elif zk > 20.0:
                zk = 20.0
            lam_k = math.exp(zk)
            # ``dL_dS`` is ∂(log-likelihood)/∂S (ascent); scipy minimizes NLL.
            dL_dth_k = -dL_dS * a_val * lam_k * grad_scale
            if is_veteran[pid]:
                grad[pid] += dL_dth_k

    nll = -ll
    for vi in range(len(veteran_pidx)):
        p = int(veteran_pidx[vi])
        nll += reg_theta * x_vet[vi] * x_vet[vi]
        grad[p] += 2.0 * reg_theta * x_vet[vi]
    return nll, grad


def _eval_loss_grad(
    x: np.ndarray,
    ctx: BatchFitContext,
    reg_theta: float,
    obs_indices: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood + L2 and gradient w.r.t. veteran θ only."""
    theta = ctx.online_theta.copy()
    theta[ctx.veteran_pidx] = x
    is_vet = np.zeros(len(theta), dtype=np.bool_)
    is_vet[ctx.veteran_pidx] = True
    nll, grad = _eval_loss_grad_nb(
        theta,
        is_vet,
        obs_indices,
        ctx.offsets,
        ctx.player_flat,
        ctx.taken,
        ctx.qi_canon,
        ctx.b,
        ctx.log_a,
        ctx.delta_g,
        ctx.lapse,
        ctx.recal_alpha,
        ctx.recal_beta,
        reg_theta,
        ctx.veteran_pidx,
        x,
    )
    return nll, grad[ctx.veteran_pidx]


def fit_batch_theta(
    ctx: BatchFitContext,
    *,
    reg_theta: float = 0.01,
    maxiter: int = 40,
    verbose: bool = True,
) -> np.ndarray:
    """Return full-length θ vector (veterans optimized, others = online)."""
    if minimize is None:
        raise RuntimeError("scipy is required for batch θ fit (pip install scipy)")

    x0 = ctx.online_theta[ctx.veteran_pidx].copy()
    obs_indices = np.where(ctx.obs_mask)[0].astype(np.int64)
    n_obs = len(obs_indices)
    if verbose:
        print(
            f"[batch-fit] veterans={len(ctx.veteran_pidx)} obs={n_obs} "
            f"reg_theta={reg_theta} maxiter={maxiter}",
            flush=True,
        )

    iters = {"n": 0}

    def fun(x: np.ndarray) -> float:
        nll, _ = _eval_loss_grad(x, ctx, reg_theta, obs_indices)
        return nll

    def jac(x: np.ndarray) -> np.ndarray:
        iters["n"] += 1
        if verbose and iters["n"] % 5 == 0:
            print(f"  grad eval #{iters['n']}", flush=True)
        _, g = _eval_loss_grad(x, ctx, reg_theta, obs_indices)
        return g

    res = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": maxiter, "ftol": 1e-8},
    )
    if verbose:
        print(
            f"[batch-fit] done success={res.success} nll={res.fun:.2f} "
            f"nit={res.nit} message={res.message}",
            flush=True,
        )

    theta = ctx.online_theta.copy()
    theta[ctx.veteran_pidx] = res.x
    return theta
