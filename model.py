"""
ChGK probabilistic model: player strength θ, question difficulty b, discrimination a.
Team take probability: p = 1 - exp(-Σ_k λ_ik), λ_ik = exp(-b_i + a_i * θ_k).

Aggregation is additive: more players → higher λ_sum → higher p. So players who
often play in small rosters (e.g. alone) and beat full teams can get understated θ;
see docs/interpretation.md.

Stretch (not implemented): allow dynamic strength θ_k(t) = θ_k(t-1) + ε;
or concave aggregation so one strong player can outweigh several weaker ones.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ChGKModel(nn.Module):
    """
    Parameters:
        theta: [num_players]  player strength
        b:     [num_questions]  question difficulty
        log_a: [num_questions]  log discrimination (a_i > 0)
        team_size_bias: [max_team_size]  bias for team size (e.g. 1-6)
        tournament_dl: [num_questions]  optional tournament difficulty (true_dl); higher = harder
        tournament_type: [num_questions]  type index 0=Очник, 1=Синхрон, 2=Асинхрон
        tournament_dl_scale: [3]  learned scale per type so effective b += scale[type] * tournament_dl
    """

    NUM_TOURNAMENT_TYPES = 3  # Очник, Синхрон, Асинхрон

    def __init__(
        self,
        num_players: int,
        num_questions: int,
        *,
        init_theta_scale: float = 0.1,
        init_b_scale: float = 0.1,
        init_log_a_mean: float = 0.0,
        init_log_a_scale: float = 0.1,
        max_team_size: int = 10,
        mode: str = "exp",
        tournament_dl: Optional[torch.Tensor] = None,
        tournament_type: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_players = num_players
        self.num_questions = num_questions
        self.theta = nn.Parameter(torch.zeros(num_players).uniform_(-init_theta_scale, init_theta_scale))
        self.b = nn.Parameter(torch.zeros(num_questions).uniform_(-init_b_scale, init_b_scale))
        self.log_a = nn.Parameter(
            torch.full((num_questions,), init_log_a_mean).uniform_(
                init_log_a_mean - init_log_a_scale,
                init_log_a_mean + init_log_a_scale,
            )
        )
        # Bias per team size (1 to max_team_size). Index 0 is unused.
        self.team_size_bias = nn.Parameter(torch.zeros(max_team_size + 1))
        # Tournament difficulty (true_dl) and type (0/1/2) for type-dependent scale
        if tournament_dl is not None:
            self.register_buffer("tournament_dl", tournament_dl.float())
        else:
            self.register_buffer("tournament_dl", torch.zeros(num_questions))
        if tournament_type is not None:
            self.register_buffer("tournament_type", tournament_type.long().clamp(0, self.NUM_TOURNAMENT_TYPES - 1))
        else:
            self.register_buffer("tournament_type", torch.zeros(num_questions, dtype=torch.long))
        self.tournament_dl_scale = nn.Parameter(torch.zeros(self.NUM_TOURNAMENT_TYPES))
        self.mode = mode
        self._eps = 1e-7

    def forward(
        self,
        question_indices: torch.Tensor,
        player_index_lists: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Batched forward.
        question_indices: [B] long
        player_index_lists: list of [team_size_k] long (ragged)
        Returns: p [B] in (0, 1), clipped for numerical stability.
        """
        # λ_ik = exp(-b_i + a_i * θ_k). For each sample we need λ_sum = Σ_k λ_ik.
        type_idx = self.tournament_type[question_indices]
        scale = self.tournament_dl_scale[type_idx]
        b_i = self.b[question_indices] + scale * self.tournament_dl[question_indices]  # [B]
        a_i = torch.exp(self.log_a[question_indices]).clamp(min=self._eps)  # [B]
        # For each sample s: λ_sum_s = Σ_{k in team_s} exp(-b_s + a_s * θ_k)
        p_list: List[torch.Tensor] = []
        for s in range(question_indices.size(0)):
            players = player_index_lists[s]  # [team_size]
            team_size = players.size(0)
            theta_k = self.theta[players]  # [team_size]
            # λ_ik for this sample: [team_size]
            log_lambda = -b_i[s] + a_i[s] * theta_k
            # Clamp to avoid exp overflow
            log_lambda = log_lambda.clamp(min=-20.0, max=20.0)
            
            # Apply team size bias
            ts_bias = self.team_size_bias[min(team_size, self.team_size_bias.size(0) - 1)]
            lam_sum = torch.exp(log_lambda).sum() * torch.exp(ts_bias)
            
            p = 1.0 - torch.exp(-lam_sum)
            p = p.clamp(min=self._eps, max=1.0 - self._eps)
            p_list.append(p)
        return torch.stack(p_list, dim=0)

    def forward_packed(
        self,
        question_indices: torch.Tensor,
        player_indices_flat: torch.Tensor,
        team_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized forward using packed representation.
        question_indices: [B], player_indices_flat: [total_players], team_sizes: [B] (sum = total_players).
        """
        b_i = self.b[question_indices]  # [B]
        type_idx = self.tournament_type[question_indices]
        scale = self.tournament_dl_scale[type_idx]
        b_i = b_i + scale * self.tournament_dl[question_indices]
        # Clamp log_a to prevent exponential explosion (max a ~ 7.4)
        log_a_clipped = self.log_a[question_indices].clamp(max=2.0)
        a_i = torch.exp(log_a_clipped).clamp(min=self._eps)  # [B]
        theta_flat = self.theta[player_indices_flat]  # [total_players]
        # Expand b_i, a_i to per-player: repeat each b_i[s], a_i[s] by team_sizes[s]
        b_exp = torch.repeat_interleave(b_i, team_sizes)  # [total_players]
        a_exp = torch.repeat_interleave(a_i, team_sizes)  # [total_players]
        
        logits = -b_exp + a_exp * theta_flat
        logits = logits.clamp(min=-20.0, max=20.0)

        B = question_indices.size(0)
        segment_ids = torch.repeat_interleave(
            torch.arange(B, device=team_sizes.device, dtype=torch.long), team_sizes
        )

        if self.mode == "exp":
            lam = torch.exp(logits)
            # Sum per team: vectorized segment sum
            lam_sum = torch.zeros(B, device=lam.device, dtype=lam.dtype)
            lam_sum.scatter_add_(0, segment_ids, lam)
            
            # Apply team size bias
            ts_indices = team_sizes.clamp(max=self.team_size_bias.size(0) - 1)
            ts_bias = self.team_size_bias[ts_indices]
            lam_sum = lam_sum * torch.exp(ts_bias)
            
            p = 1.0 - torch.exp(-lam_sum)
        else:
            # Sigmoid mode: p_team = 1 - prod(1 - sigmoid(logits))
            # log(1 - p_team) = sum(log(1 - sigmoid(logits))) = sum(log(sigmoid(-logits)))
            # log_not_p = -sum(softplus(logits))
            log_not_p_flat = -torch.nn.functional.softplus(logits)
            
            log_not_p_sum = torch.zeros(B, device=logits.device, dtype=logits.dtype)
            log_not_p_sum.scatter_add_(0, segment_ids, log_not_p_flat)
            
            # Apply team size bias (in log space for sigmoid mode)
            ts_indices = team_sizes.clamp(max=self.team_size_bias.size(0) - 1)
            ts_bias = self.team_size_bias[ts_indices]
            
            p = 1.0 - torch.exp(log_not_p_sum + ts_bias)
            
        return p.clamp(min=self._eps, max=1.0 - self._eps)
