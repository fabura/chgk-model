from __future__ import annotations

import numpy as np
import torch

from model import ChGKModel


@torch.no_grad()
def fisher_diag_theta(
    model: ChGKModel,
    question_indices: torch.Tensor,
    player_indices_flat: torch.Tensor,
    team_sizes: torch.Tensor,
    taken: torch.Tensor,
    obs_weights: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """
    Empirical Fisher diagonal for theta:
      fisher[k] ~= sum_obs w_obs * (d/dtheta_k loglik_obs)^2
    """
    device = question_indices.device
    eps = 1e-7

    canon_idx = model.canonical_q_idx[question_indices]
    b_i = model.b[canon_idx]
    type_idx = model.tournament_type[question_indices]
    scale = model.tournament_dl_scale[type_idx]
    bias = model.tournament_type_bias[type_idx]
    dl_raw = model.tournament_dl[question_indices]
    dl_norm = model._normalize_tournament_dl(dl_raw, type_idx)  # pylint: disable=protected-access
    dl_i = model._transform_tournament_dl(dl_norm)  # pylint: disable=protected-access
    b_i = b_i + bias + scale * dl_i

    log_a_clipped = model.log_a[canon_idx].clamp(max=2.0)
    a_i = torch.exp(log_a_clipped).clamp(min=eps)

    theta_flat = model.theta[player_indices_flat]
    b_exp = torch.repeat_interleave(b_i, team_sizes)
    a_exp = torch.repeat_interleave(a_i, team_sizes)
    logits = (-b_exp + a_exp * theta_flat).clamp(min=-20.0, max=20.0)
    lam = torch.exp(logits)

    bsz = question_indices.size(0)
    segment_ids = torch.repeat_interleave(
        torch.arange(bsz, device=device, dtype=torch.long), team_sizes
    )
    lam_sum = torch.zeros(bsz, device=device, dtype=lam.dtype)
    lam_sum.scatter_add_(0, segment_ids, lam)
    ts_indices = team_sizes.clamp(max=model.team_size_bias.size(0) - 1)
    ts_bias = model.team_size_bias[ts_indices]
    lam_sum = lam_sum * torch.exp(ts_bias)

    p = (1.0 - torch.exp(-lam_sum)).clamp(min=eps, max=1.0 - eps)
    dll_dS = taken * (torch.exp(-lam_sum) / p) - (1.0 - taken)

    dS_dtheta_flat = torch.repeat_interleave(dll_dS * torch.exp(ts_bias), team_sizes) * a_exp * lam
    score_sq_weighted = torch.repeat_interleave(obs_weights, team_sizes) * (dS_dtheta_flat ** 2)

    fisher = torch.zeros(num_players, device=device, dtype=lam.dtype)
    fisher.scatter_add_(0, player_indices_flat, score_sq_weighted)
    return fisher


def conservative_rating(
    theta: np.ndarray,
    fisher_diag: np.ndarray,
    prior_precision: float,
    c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    denom = np.maximum(fisher_diag + float(prior_precision), 1e-12)
    se = np.sqrt(1.0 / denom)
    rating = theta - float(c) * se
    return se.astype(np.float32), rating.astype(np.float32)
