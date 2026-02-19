from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass
class WeightConfig:
    type_offline: float = 1.0
    type_sync: float = 0.8
    type_async: float = 0.5
    size_mode: str = "log1p"  # log1p|sqrt
    mix_alpha: float = 0.5
    exp_bins: tuple[int, ...] = (3, 10, 30)  # boundaries for [0..3], [4..10], [11..30], [31+]


@dataclass
class GameWeights:
    w_type: np.ndarray
    w_size: np.ndarray
    w_mix: np.ndarray
    w_game: np.ndarray
    w_norm: np.ndarray
    n_teams: np.ndarray
    entropy: np.ndarray


def _bin_indices(values: np.ndarray, boundaries: Sequence[int]) -> np.ndarray:
    # boundaries interpreted as inclusive upper bounds.
    # np.digitize with right=True gives: <=bound -> corresponding bin.
    return np.digitize(values, boundaries, right=True).astype(np.int32)


def _type_weight(game_type: str, cfg: WeightConfig) -> float:
    if game_type == "async":
        return cfg.type_async
    if game_type == "sync":
        return cfg.type_sync
    return cfg.type_offline


def compute_player_games_by_date(
    game_participants: Sequence[np.ndarray],
    game_date_ordinal: np.ndarray,
    num_players: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - exp_before_game[g, p]: games of player p strictly before game g date,
        or total games for p when date unknown for game g.
      - total_games[p]: total distinct games for player p in dataset.
    """
    n_games = len(game_participants)
    total_games = np.zeros(num_players, dtype=np.int32)
    for players in game_participants:
        total_games[players] += 1

    # Default fallback: no-date games use total games.
    exp_before_game = np.tile(total_games[None, :], (n_games, 1))

    # Chronological pass for games with known date.
    known = np.where(game_date_ordinal >= 0)[0]
    if known.size == 0:
        return exp_before_game, total_games

    order = known[np.argsort(game_date_ordinal[known], kind="stable")]
    running = np.zeros(num_players, dtype=np.int32)
    i = 0
    while i < len(order):
        d = game_date_ordinal[order[i]]
        j = i
        while j < len(order) and game_date_ordinal[order[j]] == d:
            g = order[j]
            exp_before_game[g, :] = running
            j += 1
        for k in range(i, j):
            g = order[k]
            running[game_participants[g]] += 1
        i = j
    return exp_before_game, total_games


def build_game_participants(
    game_idx: np.ndarray,
    offsets: np.ndarray,
    player_indices_flat: np.ndarray,
    num_games: int,
) -> list[np.ndarray]:
    """Distinct players per game from packed team-question observations."""
    per_game: list[set[int]] = [set() for _ in range(num_games)]
    for i, g in enumerate(game_idx):
        s, e = offsets[i], offsets[i + 1]
        per_game[int(g)].update(int(p) for p in player_indices_flat[s:e])
    return [np.array(sorted(list(s)), dtype=np.int32) for s in per_game]


def estimate_n_teams_per_game(game_idx: np.ndarray, q_idx: np.ndarray, num_games: int) -> np.ndarray:
    # observations per game divided by unique questions in game.
    obs = np.zeros(num_games, dtype=np.float64)
    obs_q: list[set[int]] = [set() for _ in range(num_games)]
    for i, g in enumerate(game_idx):
        gi = int(g)
        obs[gi] += 1.0
        obs_q[gi].add(int(q_idx[i]))
    n_teams = np.ones(num_games, dtype=np.float64)
    for g in range(num_games):
        qn = max(1, len(obs_q[g]))
        n_teams[g] = max(1.0, obs[g] / float(qn))
    return n_teams


def compute_game_weights(
    *,
    game_idx: np.ndarray,
    q_idx: np.ndarray,
    offsets: np.ndarray,
    player_indices_flat: np.ndarray,
    num_players: int,
    game_types: Sequence[str],
    game_date_ordinal: np.ndarray,
    train_game_mask: np.ndarray,
    cfg: WeightConfig,
) -> GameWeights:
    num_games = len(game_types)
    participants = build_game_participants(game_idx, offsets, player_indices_flat, num_games)
    exp_before_game, _ = compute_player_games_by_date(participants, game_date_ordinal, num_players)

    # Type and size terms.
    w_type = np.array([_type_weight(str(t), cfg) for t in game_types], dtype=np.float64)
    n_teams = estimate_n_teams_per_game(game_idx, q_idx, num_games)
    if cfg.size_mode == "sqrt":
        w_size = np.sqrt(n_teams)
    else:
        w_size = np.log1p(n_teams)

    # Mixing term by entropy across experience bins.
    entropy = np.zeros(num_games, dtype=np.float64)
    num_bins = len(cfg.exp_bins) + 1
    max_entropy = np.log(float(num_bins)) if num_bins > 1 else 1.0
    for g in range(num_games):
        players = participants[g]
        if players.size == 0:
            continue
        exp = exp_before_game[g, players]
        bins = _bin_indices(exp, cfg.exp_bins)
        counts = np.bincount(bins, minlength=num_bins).astype(np.float64)
        p = counts / max(1.0, counts.sum())
        nz = p > 0
        h = -np.sum(p[nz] * np.log(p[nz]))
        entropy[g] = h / max(max_entropy, 1e-12)
    w_mix = 1.0 + float(cfg.mix_alpha) * entropy

    w_game = w_type * w_size * w_mix
    mean_train = float(np.mean(w_game[train_game_mask])) if np.any(train_game_mask) else float(np.mean(w_game))
    if mean_train <= 0:
        mean_train = 1.0
    w_norm = w_game / mean_train

    return GameWeights(
        w_type=w_type.astype(np.float32),
        w_size=w_size.astype(np.float32),
        w_mix=w_mix.astype(np.float32),
        w_game=w_game.astype(np.float32),
        w_norm=w_norm.astype(np.float32),
        n_teams=n_teams.astype(np.float32),
        entropy=entropy.astype(np.float32),
    )


def _type_to_index(game_type: str) -> int:
    """Map game type string to index: offline=0, sync=1, async=2."""
    if game_type == "async":
        return 2
    if game_type == "sync":
        return 1
    return 0


class LearnableGameWeights(nn.Module):
    """
    Learnable observation weights from game features.
    w[g] = type_w[type[g]] * (1 + size_coef * size_feat[g]) * (1 + mix_coef * entropy[g])
    Normalized to mean 1 over training games.
    """

    def __init__(
        self,
        num_games: int,
        type_index: np.ndarray,
        size_feat: np.ndarray,
        entropy: np.ndarray,
        train_game_mask: np.ndarray,
        init_type: tuple[float, float, float] = (1.0, 0.8, 0.5),
        init_size_coef: float = 0.3,
        init_mix_coef: float = 0.5,
    ):
        super().__init__()
        self.num_games = num_games
        self.register_buffer("type_index", torch.from_numpy(type_index).long().clamp(0, 2))
        self.register_buffer("size_feat", torch.from_numpy(size_feat).float())
        self.register_buffer("entropy", torch.from_numpy(entropy).float())
        self.register_buffer("train_mask", torch.from_numpy(train_game_mask).float())

        # init so that type weights match init_type
        log_t = np.log(np.array(init_type, dtype=np.float32) + 1e-8)
        log_t = log_t - log_t.mean()
        self.log_type_w = nn.Parameter(torch.from_numpy(log_t).float())
        self.log_size_coef = nn.Parameter(torch.tensor(float(np.log(init_size_coef + 1e-8))))
        self.log_mix_coef = nn.Parameter(torch.tensor(float(np.log(init_mix_coef + 1e-8))))

    def forward(self) -> torch.Tensor:
        type_w = torch.exp(self.log_type_w).clamp(min=0.65, max=1.3)
        type_w = type_w / type_w.mean()
        w_type = type_w[self.type_index]

        # Size coefficient: moderate range
        size_coef = torch.exp(self.log_size_coef).clamp(min=0.01, max=3.0)
        w_size = 1.0 + size_coef * (self.size_feat - 1.0).clamp(min=-0.5)

        # Mix coefficient: floor at 0.05 so mixing signal is never fully killed
        mix_coef = torch.exp(self.log_mix_coef).clamp(min=0.05, max=3.0)
        w_mix = 1.0 + mix_coef * self.entropy

        w = w_type * w_size * w_mix
        mean_train = (w * self.train_mask).sum() / self.train_mask.sum().clamp(min=1e-8)
        w = w / mean_train.clamp(min=1e-8)
        return w.clamp(min=0.05)

    def effective_weights(self) -> dict[str, float]:
        """Return the effective learned parameters (after clamping) for display."""
        with torch.no_grad():
            type_w = torch.exp(self.log_type_w).clamp(min=0.65, max=1.3)
            type_w = type_w / type_w.mean()
            return {
                "type_offline": round(type_w[0].item(), 3),
                "type_sync": round(type_w[1].item(), 3),
                "type_async": round(type_w[2].item(), 3),
                "size_coef": round(torch.exp(self.log_size_coef).clamp(min=0.01, max=3.0).item(), 3),
                "mix_coef": round(torch.exp(self.log_mix_coef).clamp(min=0.05, max=3.0).item(), 3),
            }
