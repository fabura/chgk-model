"""
Training loop: penalized MLE (binary CE + L2), identifiability (center θ, optional center b).
Optimized for large datasets (NumPy-based cache and packed data).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import (
    IndexMaps,
    Sample,
    generate_synthetic,
    generate_synthetic_two_populations,
    load_cached,
    load_from_db,
    save_cached,
    samples_to_arrays,
    samples_to_tensors,
    train_val_split,
)
from metrics import auc_roc, brier_score, logloss as metric_logloss, plot_calibration, weighted_logloss
from model import ChGKModel
from uncertainty import conservative_rating, fisher_diag_theta
from weights import WeightConfig, LearnableGameWeights, _type_to_index, compute_game_weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def packed_from_samples(
    samples: list[Sample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert samples to packed: question_indices, player_indices_flat, team_sizes, taken."""
    q_list = np.array([s.question_idx for s in samples], dtype=np.int32)
    taken_list = np.array([s.taken for s in samples], dtype=np.float32)
    team_sizes = np.array([len(s.player_indices) for s in samples], dtype=np.int32)
    flat_players = np.concatenate([s.player_indices for s in samples]).astype(np.int32)
    
    return (
        torch.from_numpy(q_list).long(),
        torch.from_numpy(flat_players).long(),
        torch.from_numpy(team_sizes).long(),
        torch.from_numpy(taken_list).float(),
    )


@dataclass
class PackedData:
    """Packed representation of the dataset for fast training."""
    q_idx: np.ndarray
    taken: np.ndarray
    team_sizes: np.ndarray
    player_indices_flat: np.ndarray
    offsets: np.ndarray  # Precomputed offsets for player_indices_flat
    team_strength: Optional[np.ndarray] = None  # Optional: team strength proxy (e.g. tournament place) per sample
    game_idx: Optional[np.ndarray] = None  # Optional: game index per observation
    obs_weight: Optional[np.ndarray] = None  # Optional: normalized game weight per observation

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> PackedData:
        offsets = np.zeros(len(arrays["team_sizes"]) + 1, dtype=np.int64)
        np.cumsum(arrays["team_sizes"], out=offsets[1:])
        return cls(
            q_idx=arrays["q_idx"],
            taken=arrays["taken"],
            team_sizes=arrays["team_sizes"],
            player_indices_flat=arrays["player_indices_flat"],
            offsets=offsets,
            team_strength=arrays.get("team_strength"),
            game_idx=arrays.get("game_idx"),
            obs_weight=arrays.get("obs_weight"),
        )

    def __len__(self) -> int:
        return len(self.q_idx)

    def get_batch(self, indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a batch of samples by indices and return as tensors."""
        batch_q = torch.from_numpy(self.q_idx[indices]).long()
        batch_taken = torch.from_numpy(self.taken[indices]).float()
        batch_sizes = torch.from_numpy(self.team_sizes[indices]).long()
        
        # Extract flat players for these samples
        flat_list = []
        for idx in indices:
            start = self.offsets[idx]
            end = self.offsets[idx+1]
            flat_list.append(self.player_indices_flat[start:end])
        batch_flat = torch.from_numpy(np.concatenate(flat_list)).long()
        
        return batch_q, batch_flat, batch_sizes, batch_taken

    def split(self, train_indices: np.ndarray, val_indices: np.ndarray) -> Tuple[PackedData, PackedData]:
        """Split into train and validation PackedData."""
        def slice_data(indices):
            out = {
                "q_idx": self.q_idx[indices],
                "taken": self.taken[indices],
                "team_sizes": self.team_sizes[indices],
                "player_indices_flat": np.concatenate([
                    self.player_indices_flat[self.offsets[i]:self.offsets[i+1]] for i in indices
                ])
            }
            if self.team_strength is not None:
                out["team_strength"] = self.team_strength[indices]
            if self.game_idx is not None:
                out["game_idx"] = self.game_idx[indices]
            if self.obs_weight is not None:
                out["obs_weight"] = self.obs_weight[indices]
            return out
        return PackedData.from_arrays(slice_data(train_indices)), PackedData.from_arrays(slice_data(val_indices))


def filter_dataset_by_player_games(
    data: PackedData,
    maps: IndexMaps,
    player_games: np.ndarray,
    min_games: int,
) -> Tuple[PackedData, IndexMaps]:
    """Remove low-game players from rosters; drop only samples that become empty; then reindex players."""
    n_players = len(maps.idx_to_player_id)
    active = (player_games >= min_games).astype(np.int64)
    active_idx = np.where(active)[0]
    old_to_new = np.full(n_players, -1, dtype=np.int32)
    old_to_new[active_idx] = np.arange(len(active_idx), dtype=np.int32)

    # Vectorized filtering:
    # 1) mark active players in the flattened roster array;
    # 2) count active players per observation;
    # 3) keep non-empty observations and only their active players.
    flat_players = data.player_indices_flat
    flat_active_mask = active[flat_players] == 1  # [sum(team_sizes)]
    active_per_obs = np.add.reduceat(flat_active_mask.astype(np.int32), data.offsets[:-1])  # [N_obs]
    kept_indices_arr = np.where(active_per_obs > 0)[0]

    new_q_idx = data.q_idx[kept_indices_arr]
    new_taken = data.taken[kept_indices_arr]
    new_team_sizes = active_per_obs[kept_indices_arr].astype(data.team_sizes.dtype, copy=False)

    if flat_active_mask.any() and kept_indices_arr.size > 0:
        obs_ids_flat = np.repeat(np.arange(len(data), dtype=np.int32), data.team_sizes)
        keep_obs_mask = np.zeros(len(data), dtype=bool)
        keep_obs_mask[kept_indices_arr] = True
        keep_flat_mask = flat_active_mask & keep_obs_mask[obs_ids_flat]
        new_player_indices_flat = old_to_new[flat_players[keep_flat_mask]]
    else:
        new_player_indices_flat = np.empty((0,), dtype=np.int32)

    out_arrays = {
        "q_idx": new_q_idx,
        "taken": new_taken,
        "team_sizes": new_team_sizes,
        "player_indices_flat": new_player_indices_flat,
    }
    if data.team_strength is not None:
        out_arrays["team_strength"] = data.team_strength[kept_indices_arr]
    if data.game_idx is not None:
        out_arrays["game_idx"] = data.game_idx[kept_indices_arr]
    if data.obs_weight is not None:
        out_arrays["obs_weight"] = data.obs_weight[kept_indices_arr]

    new_idx_to_player_id = [maps.idx_to_player_id[j] for j in active_idx]
    new_maps = IndexMaps(
        player_id_to_idx={pid: i for i, pid in enumerate(new_idx_to_player_id)},
        question_id_to_idx=maps.question_id_to_idx,
        idx_to_player_id=new_idx_to_player_id,
        idx_to_question_id=maps.idx_to_question_id,
        tournament_dl=getattr(maps, "tournament_dl", None),
        tournament_type=getattr(maps, "tournament_type", None),
        question_game_idx=getattr(maps, "question_game_idx", None),
        idx_to_game_id=getattr(maps, "idx_to_game_id", []),
        game_type=getattr(maps, "game_type", None),
        game_date_ordinal=getattr(maps, "game_date_ordinal", None),
    )
    return PackedData.from_arrays(out_arrays), new_maps


def loss_fn(
    model: ChGKModel,
    question_indices: torch.Tensor,
    player_indices_flat: torch.Tensor,
    team_sizes: torch.Tensor,
    taken: torch.Tensor,
    *,
    reg_theta: float = 1e-4,
    reg_b: float = 1e-3,
    reg_log_a: float = 1e-3,
    reg_type: float = 0.0,
    reg_team_size: float = 0.0,
    reg_gamma: float = 0.0,
    reg_game_weights: float = 0.0,
    log_gamma_init: Optional[float] = None,
    mean_b: Optional[float] = 0.0,
    obs_weights: Optional[torch.Tensor] = None,
    theta_reg_weights: Optional[torch.Tensor] = None,
    theta_anchor: Optional[torch.Tensor] = None,
    question_reg_weights: Optional[torch.Tensor] = None,
    weight_module: Optional[object] = None,
) -> torch.Tensor:
    """Negative log-likelihood + L2 regularization.

    When *_reg_weights are provided, the L2 penalty for each parameter is
    scaled by its weight (typically 1/sqrt(obs_count)), so that players or
    questions with many observations are penalized less relative to their
    data signal.

    theta_anchor: per-player anchor for theta regularization (rookie prior).
    """
    p = model.forward_packed(question_indices, player_indices_flat, team_sizes)
    nll = (
        weighted_logloss(taken, p, obs_weights)
        if obs_weights is not None
        else metric_logloss(taken, p)
    )

    # --- theta regularization (observation-normalized, with rookie anchor) ---
    anchor = theta_anchor if theta_anchor is not None else 0.0
    theta_diff = model.theta - anchor
    if theta_reg_weights is not None:
        reg_t = reg_theta * (theta_reg_weights * theta_diff ** 2).mean()
    else:
        reg_t = reg_theta * (theta_diff ** 2).mean()

    # --- question regularization (observation-normalized) ---
    if question_reg_weights is not None:
        reg_q = (
            reg_b * (question_reg_weights * model.b ** 2).mean()
            + reg_log_a * (question_reg_weights * model.log_a ** 2).mean()
        )
    else:
        reg_q = (
            reg_b * (model.b ** 2).mean()
            + reg_log_a * (model.log_a ** 2).mean()
        )

    reg = reg_t + reg_q

    # --- tournament type coefficients ---
    if reg_type > 0:
        reg = reg + reg_type * (
            (model.tournament_dl_scale ** 2).mean()
            + (model.tournament_type_bias ** 2).mean()
        )

    # --- team size bias ---
    if reg_team_size > 0:
        reg = reg + reg_team_size * (model.team_size_bias ** 2).mean()

    # --- dl power gamma (pull toward init value, not zero) ---
    if reg_gamma > 0 and hasattr(model, "log_dl_power_gamma"):
        anchor = log_gamma_init if log_gamma_init is not None else 0.0
        reg = reg + reg_gamma * (model.log_dl_power_gamma - anchor) ** 2

    # --- learnable game weights ---
    if reg_game_weights > 0 and weight_module is not None:
        for param in weight_module.parameters():
            reg = reg + reg_game_weights * (param ** 2).mean()

    if mean_b is not None:
        reg = reg + 1e2 * (model.b.mean() - mean_b) ** 2
    return nll + reg


def compute_reg_weights(
    data: PackedData,
    n_players: int,
    n_questions: int,
    device: torch.device,
    canonical_q_idx: Optional[np.ndarray] = None,
    num_canonical: Optional[int] = None,
    question_min_obs: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-player and per-question regularization weights = 1/sqrt(count).

    Players/questions with more observations get lower weight, so the L2 penalty
    doesn't dominate their data signal.  Unseen items get weight 1.0.
    When canonical_q_idx is provided, question weights are aggregated to canonical dimension.
    If question_min_obs > 0, questions with count < question_min_obs get extra boost
    (stronger regularization for rarely-played questions).
    """
    player_counts = np.zeros(n_players, dtype=np.float64)
    np.add.at(player_counts, data.player_indices_flat, 1.0)

    n_q_out = num_canonical if (canonical_q_idx is not None and num_canonical is not None) else n_questions
    if canonical_q_idx is not None and num_canonical is not None:
        canon_obs_idx = canonical_q_idx[data.q_idx]
        question_counts = np.bincount(canon_obs_idx, minlength=n_q_out).astype(np.float64)
    else:
        question_counts = np.bincount(data.q_idx, minlength=n_q_out).astype(np.float64)

    player_w = 1.0 / np.sqrt(np.maximum(player_counts, 1.0))
    question_w = 1.0 / np.sqrt(np.maximum(question_counts, 1.0))

    if question_min_obs > 0:
        low = question_counts < question_min_obs
        question_w[low] *= np.sqrt(question_min_obs / np.maximum(question_counts[low], 1.0))

    player_w /= player_w.mean()
    question_w /= question_w.mean()

    return (
        torch.from_numpy(player_w.astype(np.float32)).to(device),
        torch.from_numpy(question_w.astype(np.float32)).to(device),
    )


def get_player_games_from_db(
    player_ids: List[int],
    database_url: Optional[str] = None,
) -> dict:
    """Per-player count of distinct tournaments (games) in DB, not just in dataset.
    Returns dict player_id -> games. Empty dict on connection/import error.
    """
    try:
        import psycopg2
    except ImportError:
        return {}
    url = database_url or os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT player_id, COUNT(DISTINCT tournament_id)
            FROM public.tournament_rosters
            WHERE player_id = ANY(%s)
            GROUP BY player_id
            """,
            (player_ids,),
        )
        out = {r[0]: r[1] for r in cur.fetchall()}
        conn.close()
        return out
    except Exception:
        return {}


def compute_player_games(data: PackedData, idx_to_question_id: list, n_players: int) -> np.ndarray:
    """Per-player count of distinct tournaments (games) in data (fallback when DB unavailable)."""
    player_tournaments: dict[int, set] = {}
    off = data.offsets
    for i in range(len(data.q_idx)):
        qidx = data.q_idx[i]
        tid = idx_to_question_id[qidx][0] if isinstance(idx_to_question_id[qidx], tuple) else idx_to_question_id[qidx]
        for pidx in data.player_indices_flat[off[i] : off[i + 1]]:
            pidx = int(pidx)
            player_tournaments.setdefault(pidx, set()).add(tid)
    return np.array([len(player_tournaments.get(j, set())) for j in range(n_players)], dtype=np.int64)


def split_indices_by_game_date(
    data: PackedData,
    game_date_ordinal: Optional[np.ndarray],
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split observations into train/val; use chronological holdout when dates are available."""
    n = len(data)
    if game_date_ordinal is None or data.game_idx is None or len(game_date_ordinal) == 0:
        idx = np.random.RandomState(seed).permutation(n)
        n_val = max(1, int(n * val_frac))
        return idx[n_val:], idx[:n_val]

    gidx = data.game_idx
    unique_games = np.unique(gidx)
    known_games = unique_games[game_date_ordinal[unique_games] >= 0]
    if known_games.size < 2:
        idx = np.random.RandomState(seed).permutation(n)
        n_val = max(1, int(n * val_frac))
        return idx[n_val:], idx[:n_val]

    order = known_games[np.argsort(game_date_ordinal[known_games])]
    n_val_games = max(1, int(len(order) * val_frac))
    val_games = set(int(g) for g in order[-n_val_games:])
    val_mask = np.array([int(g) in val_games for g in gidx], dtype=bool)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    if len(val_idx) == 0 or len(train_idx) == 0:
        idx = np.random.RandomState(seed).permutation(n)
        n_val = max(1, int(n * val_frac))
        return idx[n_val:], idx[:n_val]
    return train_idx, val_idx


def compute_empirical_init(
    data: PackedData,
    n_players: int,
    n_questions: int,
    eps: float = 1e-4,
    scale: float = 0.5,
    scale_log_a: float = 0.8,
    log_a_clamp: float = 1.5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute data-driven initial values for theta, b and optionally log_a.

    - b: high take rate on a question -> easy -> lower b. b_init = scale * log((1-r)/r).
    - theta: high take rate when player is in team -> stronger -> higher theta. theta_init = scale * logit(r).
    - log_a: if data.team_strength is set, per-question correlation(taken, team_strength) as selectivity
      (strong teams take more -> higher correlation -> higher a). log_a_init = scale_log_a * clip(corr, -clamp, clamp).
    Returns (theta_init, b_init, log_a_init). log_a_init is None if no team_strength.
    """
    if len(data) == 0:
        return None, None, None
    # Per-question take rate
    sum_taken = np.bincount(data.q_idx, weights=data.taken.astype(np.float64), minlength=n_questions)
    count_q = np.bincount(data.q_idx, minlength=n_questions)
    rate_q = sum_taken / np.maximum(count_q, 1)
    r_q = np.clip(rate_q, eps, 1.0 - eps)
    # b: easy (high rate) -> low b. log((1-r)/r) so high r -> negative log -> low b
    b_init = scale * np.log((1.0 - r_q) / r_q).astype(np.float32)

    # Per-player take rate: each (sample, player) appearance gets the sample's taken value
    flat_taken = np.repeat(data.taken, data.team_sizes)
    player_sum = np.bincount(data.player_indices_flat, weights=flat_taken.astype(np.float64), minlength=n_players)
    player_count = np.bincount(data.player_indices_flat, minlength=n_players)
    rate_p = player_sum / np.maximum(player_count, 1)
    r_p = np.clip(rate_p, eps, 1.0 - eps)
    theta_init = scale * np.log(r_p / (1.0 - r_p)).astype(np.float32)

    # Per-question correlation(taken, team_strength) as selectivity proxy -> init log_a
    log_a_init: Optional[np.ndarray] = None
    if data.team_strength is not None:
        # Vectorized per-question Pearson correlation:
        # corr = (n*sum(ts)-sum(t)sum(s)) / sqrt((n*sum(t^2)-sum(t)^2)*(n*sum(s^2)-sum(s)^2))
        q_idx = data.q_idx
        t = data.taken.astype(np.float64)
        s = data.team_strength.astype(np.float64)
        n = np.bincount(q_idx, minlength=n_questions).astype(np.float64)
        sum_t = np.bincount(q_idx, weights=t, minlength=n_questions)
        sum_s = np.bincount(q_idx, weights=s, minlength=n_questions)
        sum_tt = np.bincount(q_idx, weights=t * t, minlength=n_questions)
        sum_ss = np.bincount(q_idx, weights=s * s, minlength=n_questions)
        sum_ts = np.bincount(q_idx, weights=t * s, minlength=n_questions)

        num = n * sum_ts - sum_t * sum_s
        den_t = n * sum_tt - sum_t * sum_t
        den_s = n * sum_ss - sum_s * sum_s
        den = np.sqrt(np.maximum(den_t * den_s, 0.0))

        corr = np.zeros(n_questions, dtype=np.float64)
        valid = (n >= 2.0) & (den > 1e-12)
        corr[valid] = num[valid] / den[valid]
        corr = np.clip(corr, -1.0, 1.0)
        log_a_init = np.clip(scale_log_a * corr, -log_a_clamp, log_a_clamp).astype(np.float32)

    return theta_init, b_init, log_a_init


def apply_identifiability(
    model: ChGKModel,
    center_theta: bool = True,
    center_b: bool = False,
    theta_center_mask: Optional[torch.Tensor] = None,
) -> None:
    """Center theta (and optionally b) in-place after optimizer step.

    When theta_center_mask is given, theta is centered by the mean of theta
    over masked players only (e.g. experienced players with 500+ games).
    """
    with torch.no_grad():
        if center_theta:
            if theta_center_mask is not None and theta_center_mask.any():
                center = model.theta[theta_center_mask].mean()
            else:
                center = model.theta.mean()
            model.theta.sub_(center)
        if center_b:
            model.b.sub_(model.b.mean())


def train_epoch(
    model: ChGKModel,
    data: list[Sample] | PackedData,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    reg_theta: float,
    reg_b: float,
    reg_log_a: float,
    reg_type: float = 0.0,
    reg_team_size: float = 0.0,
    reg_gamma: float = 0.0,
    reg_game_weights: float = 0.0,
    log_gamma_init: Optional[float] = None,
    center_theta: bool = True,
    center_b: bool = False,
    show_progress: bool = True,
    theta_reg_weights: Optional[torch.Tensor] = None,
    theta_anchor: Optional[torch.Tensor] = None,
    question_reg_weights: Optional[torch.Tensor] = None,
    theta_center_mask: Optional[torch.Tensor] = None,
    weight_module: Optional[object] = None,
    grad_clip: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    total_loss = 0.0
    total_count = 0
    total_p = 0.0
    
    batch_starts = range(0, n_samples, batch_size)
    if show_progress:
        batch_starts = tqdm(batch_starts, desc="Train", unit=" batch", leave=False)
        
    for start in batch_starts:
        batch_idx = indices[start : start + batch_size]
        
        if isinstance(data, PackedData):
            q, flat_p, sizes, taken = data.get_batch(batch_idx)
            obs_w = None
            if weight_module is not None and data.game_idx is not None:
                w_game = weight_module()
                obs_w = w_game[data.game_idx[batch_idx]]
            elif data.obs_weight is not None:
                obs_w = torch.from_numpy(data.obs_weight[batch_idx]).float()
        else:
            batch_samples = [data[i] for i in batch_idx]
            q, flat_p, sizes, taken = packed_from_samples(batch_samples)
            obs_w = None
            
        q, flat_p, sizes, taken = q.to(device), flat_p.to(device), sizes.to(device), taken.to(device)
        if obs_w is not None:
            obs_w = obs_w.to(device)
        optimizer.zero_grad()
        loss = loss_fn(
            model, q, flat_p, sizes, taken,
            reg_theta=reg_theta, reg_b=reg_b, reg_log_a=reg_log_a,
            reg_type=reg_type, reg_team_size=reg_team_size,
            reg_gamma=reg_gamma, reg_game_weights=reg_game_weights,
            log_gamma_init=log_gamma_init,
            mean_b=0.0 if center_b else None,
            obs_weights=obs_w,
            theta_reg_weights=theta_reg_weights,
            theta_anchor=theta_anchor,
            question_reg_weights=question_reg_weights,
            weight_module=weight_module,
        )
        loss.backward()
        if grad_clip > 0:
            all_params = list(model.parameters())
            if weight_module is not None:
                all_params += list(weight_module.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()
        apply_identifiability(
            model, center_theta=center_theta, center_b=center_b,
            theta_center_mask=theta_center_mask,
        )

        total_loss += loss.item() * len(batch_idx)
        total_count += len(batch_idx)
        with torch.no_grad():
            p = model.forward_packed(q, flat_p, sizes)
            total_p += p.sum().item()
            
    return total_loss / total_count, total_p / total_count


@torch.no_grad()
def evaluate(
    model: ChGKModel,
    data: list[Sample] | PackedData,
    device: torch.device,
    batch_size: int = 4096,
    weight_module: Optional[object] = None,
) -> Tuple[float, float, float, float]:
    """Returns mean_loss, mean_p, brier, auc."""
    model.eval()
    n_samples = len(data)
    if n_samples == 0:
        return 0.0, 0.0, 0.0, float("nan")
        
    all_p = []
    all_taken = []
    all_w = []
    total_loss = 0.0
    
    # Evaluate in batches to avoid OOM
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = np.arange(start, end)
        
        if isinstance(data, PackedData):
            q, flat_p, sizes, taken = data.get_batch(batch_idx)
            obs_w = None
            if weight_module is not None and data.game_idx is not None:
                w_game = weight_module()
                obs_w = w_game[data.game_idx[batch_idx]]
            elif data.obs_weight is not None:
                obs_w = torch.from_numpy(data.obs_weight[batch_idx]).float()
        else:
            batch_samples = [data[i] for i in batch_idx]
            q, flat_p, sizes, taken = packed_from_samples(batch_samples)
            obs_w = None
            
        q, flat_p, sizes, taken = q.to(device), flat_p.to(device), sizes.to(device), taken.to(device)
        if obs_w is not None:
            obs_w = obs_w.to(device)
        p = model.forward_packed(q, flat_p, sizes)
        
        if obs_w is not None:
            total_loss += weighted_logloss(taken, p, obs_w).item() * (end - start)
            all_w.append(obs_w.cpu())
        else:
            total_loss += metric_logloss(taken, p).item() * (end - start)
        all_p.append(p.cpu())
        all_taken.append(taken.cpu())
        
    p = torch.cat(all_p)
    taken = torch.cat(all_taken)
    
    mean_loss = total_loss / n_samples
    brier = brier_score(taken, p).item()
    auc = auc_roc(taken, p)
    return mean_loss, p.mean().item(), brier, auc


# ---------------------------------------------------------------------------
# Hyperparameter tuning (random search)
# ---------------------------------------------------------------------------

TUNE_SEARCH_SPACE = {
    "lr":           ("log_uniform", 5e-4, 2e-2),
    "batch_size":   ("choice", [256, 512, 1024, 2048]),
    "reg_theta":    ("log_uniform", 1e-5, 1e-2),
    "reg_b":        ("log_uniform", 1e-4, 1e-1),
    "reg_log_a":    ("log_uniform", 1e-4, 1e-1),
    "reg_type":     ("log_uniform", 1e-3, 1e-1),
    "reg_team_size":("log_uniform", 1e-3, 1e-1),
    "reg_gamma":    ("log_uniform", 1e-2, 1.0),
    "reg_game_weights": ("log_uniform", 1e-3, 1e-1),
    "grad_clip":    ("choice", [1.0, 3.0, 5.0, 10.0]),
    "rookie_floor": ("uniform", -3.0, -0.1),
    "rookie_tau":   ("uniform", 10.0, 100.0),
}


def _sample_hparams(rng: random.Random) -> dict:
    hparams = {}
    for key, spec in TUNE_SEARCH_SPACE.items():
        if spec[0] == "log_uniform":
            lo, hi = math.log(spec[1]), math.log(spec[2])
            hparams[key] = math.exp(rng.uniform(lo, hi))
        elif spec[0] == "uniform":
            hparams[key] = rng.uniform(spec[1], spec[2])
        elif spec[0] == "choice":
            hparams[key] = rng.choice(spec[1])
    hparams["batch_size"] = int(hparams["batch_size"])
    return hparams


def run_tuning(
    args,
    train_data: PackedData,
    val_data: PackedData,
    n_players: int,
    n_questions: int,
    num_canonical: int,
    maps,
    device: torch.device,
    tournament_dl_tensor,
    tournament_type_tensor,
    canonical_q_idx_tensor,
    weight_module_factory,
    theta_center_mask,
    theta_anchor,
    player_games: np.ndarray,
) -> dict:
    """Run random hyperparameter search and return best config."""
    rng = random.Random(args.seed)
    results: list[dict] = []

    _log_gamma_init: Optional[float] = None
    if args.dl_learn_power:
        _log_gamma_init = math.log(max(args.dl_power_gamma, 1e-6))

    print(f"\n{'='*70}", flush=True)
    print(f"HYPERPARAMETER TUNING: {args.tune_trials} trials, {args.tune_epochs} epochs each", flush=True)
    print(f"{'='*70}\n", flush=True)

    for trial in range(args.tune_trials):
        hp = _sample_hparams(rng)
        set_seed(args.seed + trial)

        theta_reg_weights, question_reg_weights = compute_reg_weights(
            train_data, n_players, n_questions, device,
            canonical_q_idx=getattr(maps, "canonical_q_idx", None),
            num_canonical=num_canonical if num_canonical < n_questions else None,
            question_min_obs=args.question_min_obs,
        )

        model = ChGKModel(
            n_players, n_questions,
            num_canonical_questions=num_canonical if num_canonical < n_questions else None,
            canonical_q_idx=canonical_q_idx_tensor,
            tournament_dl=tournament_dl_tensor,
            tournament_type=tournament_type_tensor,
            dl_transform=args.dl_transform,
            dl_power_gamma=args.dl_power_gamma,
            dl_learn_power=args.dl_learn_power,
        ).to(device)

        wm = weight_module_factory() if weight_module_factory is not None else None
        params = list(model.parameters())
        if wm is not None:
            wm = wm.to(device)
            params += list(wm.parameters())
        optimizer = torch.optim.Adam(params, lr=hp["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )

        # Per-trial rookie anchor from sampled hp
        trial_anchor = None
        if hp["rookie_floor"] < 0:
            anchor_np = hp["rookie_floor"] * np.exp(-player_games.astype(np.float32) / hp["rookie_tau"])
            trial_anchor = torch.from_numpy(anchor_np).float().to(device)

        theta_init, b_init, log_a_init = compute_empirical_init(
            train_data, n_players, n_questions,
        )
        if theta_init is not None and b_init is not None:
            if trial_anchor is not None:
                theta_init = np.minimum(theta_init, anchor_np * 0.5)
            model.theta.data.copy_(torch.from_numpy(theta_init).to(device))
            cq = maps.canonical_q_idx
            if cq is not None and num_canonical < n_questions:
                b_canon = np.zeros(num_canonical, dtype=np.float32)
                cnt = np.zeros(num_canonical, dtype=np.float32)
                np.add.at(b_canon, cq, b_init)
                np.add.at(cnt, cq, 1.0)
                b_canon /= np.maximum(cnt, 1.0)
                model.b.data.copy_(torch.from_numpy(b_canon).to(device))
            else:
                model.b.data.copy_(torch.from_numpy(b_init).to(device))
            apply_identifiability(model, center_theta=True, center_b=False,
                                  theta_center_mask=theta_center_mask)
            if log_a_init is not None:
                if cq is not None and num_canonical < n_questions:
                    a_canon = np.zeros(num_canonical, dtype=np.float32)
                    cnt_a = np.zeros(num_canonical, dtype=np.float32)
                    np.add.at(a_canon, cq, log_a_init)
                    np.add.at(cnt_a, cq, 1.0)
                    a_canon /= np.maximum(cnt_a, 1.0)
                    model.log_a.data.copy_(torch.from_numpy(a_canon).to(device))
                else:
                    model.log_a.data.copy_(torch.from_numpy(log_a_init).to(device))

        best_vloss = float("inf")
        best_epoch = 0
        best_auc = 0.0
        best_brier = 1.0
        patience_cnt = 0

        for epoch in range(args.tune_epochs):
            train_epoch(
                model, train_data, optimizer, device, hp["batch_size"],
                reg_theta=hp["reg_theta"], reg_b=hp["reg_b"], reg_log_a=hp["reg_log_a"],
                reg_type=hp["reg_type"], reg_team_size=hp["reg_team_size"],
                reg_gamma=hp["reg_gamma"], reg_game_weights=hp["reg_game_weights"],
                log_gamma_init=_log_gamma_init,
                center_theta=True, center_b=True,
                theta_reg_weights=theta_reg_weights,
                theta_anchor=trial_anchor,
                question_reg_weights=question_reg_weights,
                theta_center_mask=theta_center_mask,
                weight_module=wm,
                show_progress=False,
                grad_clip=hp["grad_clip"],
            )
            vloss, _, vbrier, vauc = evaluate(model, val_data, device, weight_module=wm)
            scheduler.step(vloss)

            if vloss < best_vloss:
                best_vloss = vloss
                best_epoch = epoch + 1
                best_auc = vauc
                best_brier = vbrier
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 5:
                    break

        hp_short = {k: (f"{v:.2e}" if isinstance(v, float) else v)
                    for k, v in hp.items()}
        result = {
            "trial": trial + 1,
            "val_loss": best_vloss,
            "val_auc": best_auc,
            "val_brier": best_brier,
            "best_epoch": best_epoch,
            "hparams": hp,
        }
        results.append(result)
        print(
            f"Trial {trial+1:2d}/{args.tune_trials} | "
            f"val_loss {best_vloss:.4f} AUC {best_auc:.4f} Brier {best_brier:.4f} "
            f"(ep {best_epoch}) | "
            f"lr={hp['lr']:.1e} bs={hp['batch_size']} "
            f"rθ={hp['reg_theta']:.1e} rb={hp['reg_b']:.1e} ra={hp['reg_log_a']:.1e} "
            f"rookie={hp['rookie_floor']:.1f}/τ={hp['rookie_tau']:.0f}",
            flush=True,
        )

        del model, optimizer, scheduler, wm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results.sort(key=lambda r: r["val_loss"])
    print(f"\n{'='*70}")
    print("TOP 5 CONFIGURATIONS:")
    print(f"{'='*70}")
    for i, r in enumerate(results[:5]):
        hp = r["hparams"]
        print(f"\n#{i+1} val_loss={r['val_loss']:.4f} AUC={r['val_auc']:.4f} Brier={r['val_brier']:.4f} (epoch {r['best_epoch']})")
        print(f"  --lr {hp['lr']:.4e} --batch_size {hp['batch_size']} \\")
        print(f"  --reg_theta {hp['reg_theta']:.4e} --reg_b {hp['reg_b']:.4e} --reg_log_a {hp['reg_log_a']:.4e} \\")
        print(f"  --reg_type {hp['reg_type']:.4e} --reg_team_size {hp['reg_team_size']:.4e} \\")
        print(f"  --reg_gamma {hp['reg_gamma']:.4e} --reg_game_weights {hp['reg_game_weights']:.4e} \\")
        print(f"  --grad_clip {hp['grad_clip']} \\")
        print(f"  --rookie_floor {hp['rookie_floor']:.2f} --rookie_tau {hp['rookie_tau']:.1f}")

    best = results[0]
    print(f"\nBest trial: #{best['trial']} with val_loss={best['val_loss']:.4f}")
    print("\nTo run full training with best config:")
    hp = best["hparams"]
    cmd = (
        f"python train.py --mode {args.mode}"
        + (f" --cache_file {args.cache_file}" if args.cache_file else "")
        + (f" --max_tournaments {args.max_tournaments}" if args.max_tournaments else "")
        + f" --lr {hp['lr']:.4e} --batch_size {hp['batch_size']}"
        + f" --reg_theta {hp['reg_theta']:.4e} --reg_b {hp['reg_b']:.4e} --reg_log_a {hp['reg_log_a']:.4e}"
        + f" --reg_type {hp['reg_type']:.4e} --reg_team_size {hp['reg_team_size']:.4e}"
        + f" --reg_gamma {hp['reg_gamma']:.4e} --reg_game_weights {hp['reg_game_weights']:.4e}"
        + f" --grad_clip {hp['grad_clip']}"
        + f" --rookie_floor {hp['rookie_floor']:.2f} --rookie_tau {hp['rookie_tau']:.1f}"
    )
    print(cmd)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="ChGK model training")
    # -- Data --
    parser.add_argument("--mode", choices=["synthetic", "db"], default="synthetic")
    parser.add_argument("--synthetic_two_pop", action="store_true",
                        help="Synthetic: two weakly connected populations.")
    parser.add_argument("--max_tournaments", type=int, default=None, help="DB: limit tournaments")
    parser.add_argument("--all_tournaments", action="store_true",
                        help="DB: include all tournaments (default: only with points_mask)")
    parser.add_argument("--tournament_dl_filter", choices=["all", "true_dl", "ndcg", "any", "both"],
                        default="all", help="DB: filter by difficulty source. Default all.")
    parser.add_argument("--min_tournament_date", type=str, default="2015-01-01",
                        help="DB: tournaments from this date. Default 2015-01-01.")
    parser.add_argument("--cache_file", type=str, default=None,
                        help="DB: cache file (load if exists, else save after DB load).")
    parser.add_argument("--min_games_in_dataset", type=int, default=0,
                        help="Remove low-game players from rosters. 0=off. Default 0.")
    # -- Training --
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5.1344e-4)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--grad_clip", type=float, default=3.0,
                        help="Max gradient norm (0=off). Default 3.0.")
    parser.add_argument("--reg_theta", type=float, default=4.8621e-5)
    parser.add_argument("--reg_b", type=float, default=5.2031e-2)
    parser.add_argument("--reg_log_a", type=float, default=3.7923e-2)
    parser.add_argument("--reg_type", type=float, default=1.3858e-3,
                        help="L2 on tournament_dl_scale and tournament_type_bias")
    parser.add_argument("--reg_team_size", type=float, default=2.9923e-3,
                        help="L2 on team_size_bias")
    parser.add_argument("--reg_gamma", type=float, default=0.01,
                        help="L2 on log_dl_power_gamma (keeps it near init)")
    parser.add_argument("--reg_game_weights", type=float, default=2.6821e-3,
                        help="L2 on LearnableGameWeights params")
    parser.add_argument("--question_min_obs", type=int, default=50,
                        help="Extra reg boost for questions with fewer observations. 0=off.")
    parser.add_argument("--val_frac", type=float, default=0.15)
    # -- Difficulty transform --
    parser.add_argument("--dl_transform", choices=["linear", "log1p", "power"], default="power",
                        help="Transform for true_dl. Default power.")
    parser.add_argument("--dl_power_gamma", type=float, default=2.0,
                        help="Init gamma for power transform. Default 2.0 (quadratic).")
    parser.add_argument("--no_learn_gamma", action="store_true",
                        help="Fix gamma instead of learning it.")
    # -- Game weights --
    parser.add_argument("--no_learn_weights", action="store_true",
                        help="Use fixed weights instead of learning them.")
    parser.add_argument("--run_ablations", action="store_true",
                        help="Run weighting ablations: type-only, type+size, type+size+mix.")
    # -- Tuning --
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter random search instead of normal training.")
    parser.add_argument("--tune_trials", type=int, default=20,
                        help="Number of random search trials. Default 20.")
    parser.add_argument("--tune_epochs", type=int, default=15,
                        help="Epochs per trial (shorter than full training). Default 15.")
    # -- Output --
    parser.add_argument("--calibration_plot", type=str, default=None, help="Save calibration curve to path")
    parser.add_argument("--save_model", type=str, default=None, help="Save model to .pt file.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Save checkpoints to dir.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path.")
    parser.add_argument("--players_out", type=str, default="results/players.csv",
                        help="Export players CSV. Empty=skip. Default results/players.csv.")
    parser.add_argument("--games_weights_out", type=str, default="results/games_weights.csv",
                        help="Export game weights CSV. Empty=skip. Default results/games_weights.csv.")
    parser.add_argument("--export_min_games", type=int, default=30,
                        help="Min games for player export. 0=all. Default 30.")
    # -- Ranking --
    parser.add_argument("--rating_c", type=float, default=5.0, help="Conservative rating = theta - c*SE")
    # -- Rookie prior --
    parser.add_argument("--rookie_floor", type=float, default=-1.0,
                        help="Negative anchor for theta of inexperienced players. 0=off.")
    parser.add_argument("--rookie_tau", type=float, default=30.0,
                        help="Decay rate (in games) for rookie penalty: anchor = floor*exp(-games/tau)")
    args = parser.parse_args()

    # Derive boolean flags (default: learn both gamma and weights)
    args.learn_weights = not args.no_learn_weights
    args.dl_learn_power = (not args.no_learn_gamma) and (args.dl_transform == "power")

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    packed_data = None
    samples = None
    loaded_from_db_fresh = False
    loaded_from_cache = False
    
    if args.mode == "synthetic":
        if args.synthetic_two_pop:
            samples, maps = generate_synthetic_two_populations(seed=args.seed)
        else:
            samples, maps = generate_synthetic(seed=args.seed)
        packed_data = PackedData.from_arrays(samples_to_arrays(samples))
    else:
        cache_path = args.cache_file
        if cache_path and os.path.isfile(cache_path):
            print(f"Loading from cache: {cache_path}")
            arrays, maps = load_cached(cache_path)
            packed_data = PackedData.from_arrays(arrays)
            loaded_from_cache = True
        else:
            samples, maps = load_from_db(
                max_tournaments=args.max_tournaments,
                only_with_question_data=not args.all_tournaments,
                tournament_dl_filter=args.tournament_dl_filter,
                min_tournament_date=args.min_tournament_date or None,
                min_games=args.min_games_in_dataset,
                seed=args.seed,
            )
            loaded_from_db_fresh = True
            if cache_path:
                for _ in tqdm([1], desc="Saving cache", unit=" file"):
                    save_cached(samples, maps, cache_path, meta={"max_tournaments": args.max_tournaments})
            packed_data = PackedData.from_arrays(samples_to_arrays(samples))

    if packed_data:
        src = "cache" if loaded_from_cache else ("DB (fresh)" if loaded_from_db_fresh else "synthetic")
        print(f"Data loaded from {src}: {len(packed_data)} samples.", flush=True)

    # Optionally remove low-game players from rosters (drop only teams that become empty), then reindex players
    if packed_data and args.min_games_in_dataset > 0 and not loaded_from_db_fresh:
        if loaded_from_cache:
            print("Post-load min-games filtering: cache path -> running vectorized filter.", flush=True)
        else:
            print("Post-load min-games filtering: non-DB-fresh path -> running vectorized filter.", flush=True)
        db_games = get_player_games_from_db(maps.idx_to_player_id)
        if db_games:
            player_games = np.array(
                [db_games.get(pid, 0) for pid in maps.idx_to_player_id],
                dtype=np.int64,
            )
            games_source = "DB"
        else:
            player_games = compute_player_games(
                packed_data, maps.idx_to_question_id, maps.num_players
            )
            games_source = "dataset"
        n_before = len(packed_data)
        packed_data, maps = filter_dataset_by_player_games(
            packed_data, maps, player_games, args.min_games_in_dataset
        )
        n_dropped = n_before - len(packed_data)
        print(
            f"Filtered dataset (games from {games_source}): dropped {n_dropped} samples that became empty after removing players with <{args.min_games_in_dataset} games; "
            f"kept {len(packed_data)} samples, {maps.num_players} players"
        )
    elif packed_data and args.min_games_in_dataset > 0 and loaded_from_db_fresh:
        print(
            "Skipping post-load min-games filtering: it was already applied during DB extraction.",
            flush=True,
        )
    elif packed_data and args.min_games_in_dataset <= 0:
        print("Post-load min-games filtering disabled (min_games_in_dataset <= 0).", flush=True)

    if packed_data:
        train_idx, val_idx = split_indices_by_game_date(
            packed_data,
            getattr(maps, "game_date_ordinal", None),
            args.val_frac,
            args.seed,
        )
        train_data, val_data = packed_data.split(train_idx, val_idx)
    else:
        train_data, val_data = train_val_split(samples, val_frac=args.val_frac, seed=args.seed)
        
    n_players = maps.num_players
    n_questions = maps.num_questions
    print(f"Train {len(train_data)} val {len(val_data)} | players {n_players} questions {n_questions}")

    # Build game-level weights and attach observation weights.
    game_weights = None
    have_game_meta = False
    train_game_mask = np.array([True])  # placeholder
    if isinstance(train_data, PackedData) and isinstance(val_data, PackedData):
        have_game_meta = (
            train_data.game_idx is not None
            and packed_data is not None
            and packed_data.game_idx is not None
            and getattr(maps, "idx_to_game_id", None)
            and getattr(maps, "game_type", None) is not None
            and getattr(maps, "question_game_idx", None) is not None
        )
        if have_game_meta:
            cfg = WeightConfig()
            train_game_mask = np.zeros(len(maps.idx_to_game_id), dtype=bool)
            train_game_mask[np.unique(train_data.game_idx)] = True
            game_weights = compute_game_weights(
                game_idx=packed_data.game_idx,
                q_idx=packed_data.q_idx,
                offsets=packed_data.offsets,
                player_indices_flat=packed_data.player_indices_flat,
                num_players=n_players,
                game_types=[str(x) for x in maps.game_type.tolist()],
                game_date_ordinal=maps.game_date_ordinal if maps.game_date_ordinal is not None else np.full(len(maps.idx_to_game_id), -1, dtype=np.int32),
                train_game_mask=train_game_mask,
                cfg=cfg,
            )
            if not args.learn_weights:
                train_data.obs_weight = game_weights.w_norm[train_data.game_idx]
                val_data.obs_weight = game_weights.w_norm[val_data.game_idx]
                print(
                    f"Game weights (fixed): mean(train)={train_data.obs_weight.mean():.4f}, "
                    f"w_mix mean={game_weights.w_mix.mean():.4f}"
                )
            else:
                print("Game weights: learnable (type, size, mix)")

    weight_module: Optional[LearnableGameWeights] = None
    if have_game_meta and args.learn_weights and game_weights is not None:
        game_types = [str(x) for x in maps.game_type.tolist()]
        type_index = np.array([_type_to_index(t) for t in game_types], dtype=np.int32)
        size_raw = game_weights.w_size.astype(np.float64)
        size_feat = (size_raw / np.maximum(size_raw.mean(), 1e-8)).astype(np.float32)
        weight_module = LearnableGameWeights(
            num_games=len(maps.idx_to_game_id),
            type_index=type_index,
            size_feat=size_feat,
            entropy=game_weights.entropy.astype(np.float32),
            train_game_mask=train_game_mask.astype(np.float32),
        )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt["num_players"] != maps.num_players or ckpt["num_questions"] != maps.num_questions:
            raise ValueError(
                f"Checkpoint shape (players={ckpt['num_players']}, questions={ckpt['num_questions']}) "
                f"does not match data (players={maps.num_players}, questions={maps.num_questions}). "
                "Use the same --cache_file and dataset."
            )
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (checkpoint had epoch {ckpt['epoch']})")
        # Keep dl transform consistent with the checkpoint when available.
        if "dl_transform" in ckpt:
            args.dl_transform = ckpt["dl_transform"]
        if "dl_power_gamma" in ckpt:
            args.dl_power_gamma = float(ckpt["dl_power_gamma"])
        if "dl_learn_power" in ckpt:
            args.dl_learn_power = bool(ckpt["dl_learn_power"])

    tournament_dl_tensor = None
    tournament_type_tensor = None
    if getattr(maps, "tournament_dl", None) is not None:
        tournament_dl_tensor = torch.from_numpy(maps.tournament_dl).float()
        print(f"Using tournament true_dl for school/student correction ({maps.tournament_dl.size} questions)")
    if getattr(maps, "tournament_type", None) is not None:
        tournament_type_tensor = torch.from_numpy(maps.tournament_type).long()
        print("Using tournament type (Очник/Синхрон/Асинхрон) for type-dependent dl scale")
        if args.dl_transform == "power":
            gamma_desc = f"gamma={'learned, init=' if args.dl_learn_power else ''}{args.dl_power_gamma}"
        else:
            gamma_desc = ""
        print(f"Tournament dl transform: {args.dl_transform} {gamma_desc}")

    canonical_q_idx_tensor = None
    num_canonical = n_questions
    if getattr(maps, "canonical_q_idx", None) is not None and maps.num_canonical_questions is not None:
        canonical_q_idx_tensor = torch.from_numpy(maps.canonical_q_idx).long()
        num_canonical = maps.num_canonical_questions
    if args.mode == "db":
        saved = n_questions - num_canonical
        if saved > 0:
            print(f"Paired tournaments: {num_canonical} canonical questions ({saved} shared across paired packages)")
        else:
            print(f"Paired tournaments: no pairs detected, {n_questions} questions")

    # --- Tuning mode: run random search and exit ---
    if args.tune:
        wm_factory = None
        if have_game_meta and args.learn_weights and game_weights is not None:
            _game_types = [str(x) for x in maps.game_type.tolist()]
            _type_index = np.array([_type_to_index(t) for t in _game_types], dtype=np.int32)
            _size_raw = game_weights.w_size.astype(np.float64)
            _size_feat = (_size_raw / np.maximum(_size_raw.mean(), 1e-8)).astype(np.float32)
            _entropy = game_weights.entropy.astype(np.float32)
            _tgm = train_game_mask.astype(np.float32)
            _num_games = len(maps.idx_to_game_id)
            wm_factory = lambda: LearnableGameWeights(
                num_games=_num_games, type_index=_type_index,
                size_feat=_size_feat, entropy=_entropy, train_game_mask=_tgm,
            )

        db_games = get_player_games_from_db(maps.idx_to_player_id)
        if db_games:
            _player_games = np.array([db_games.get(pid, 0) for pid in maps.idx_to_player_id], dtype=np.int64)
        else:
            _player_games = compute_player_games(train_data, maps.idx_to_question_id, n_players)
        MIN_GAMES_FOR_CENTER = 500
        _tcm = torch.from_numpy(_player_games >= MIN_GAMES_FOR_CENTER).to(device) if (_player_games >= MIN_GAMES_FOR_CENTER).any() else None
        if args.rookie_floor < 0:
            _anchor_np = args.rookie_floor * np.exp(-_player_games.astype(np.float32) / args.rookie_tau)
            _theta_anchor = torch.from_numpy(_anchor_np).float().to(device)
        else:
            _theta_anchor = None

        run_tuning(
            args, train_data, val_data, n_players, n_questions, num_canonical,
            maps, device, tournament_dl_tensor, tournament_type_tensor,
            canonical_q_idx_tensor, wm_factory, _tcm, _theta_anchor, _player_games,
        )
        return 0

    model = ChGKModel(
        n_players,
        n_questions,
        num_canonical_questions=num_canonical if num_canonical < n_questions else None,
        canonical_q_idx=canonical_q_idx_tensor,
        tournament_dl=tournament_dl_tensor,
        tournament_type=tournament_type_tensor,
        dl_transform=args.dl_transform,
        dl_power_gamma=args.dl_power_gamma,
        dl_learn_power=args.dl_learn_power,
    ).to(device)
    params = list(model.parameters())
    if weight_module is not None:
        weight_module = weight_module.to(device)
        params = params + list(weight_module.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    if args.resume:
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            print(f"Resume: missing keys in checkpoint (using defaults): {missing}")
        if unexpected:
            print(f"Resume: unexpected keys in checkpoint (ignored): {unexpected}")
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    # Compute observation-normalized regularization weights, theta center mask, and rookie anchor
    theta_reg_weights = None
    question_reg_weights = None
    theta_center_mask = None
    theta_anchor = None
    MIN_GAMES_FOR_CENTER = 500
    if isinstance(train_data, PackedData):
        theta_reg_weights, question_reg_weights = compute_reg_weights(
            train_data, n_players, n_questions, device,
            canonical_q_idx=getattr(maps, "canonical_q_idx", None),
            num_canonical=num_canonical if num_canonical < n_questions else None,
            question_min_obs=args.question_min_obs,
        )
        print(f"Reg weights: theta min={theta_reg_weights.min():.4f} max={theta_reg_weights.max():.4f} | "
              f"question min={question_reg_weights.min():.4f} max={question_reg_weights.max():.4f}")
        if args.question_min_obs > 0:
            print(f"Question low-obs boost: min_obs={args.question_min_obs}")
        reg_extras = []
        if args.reg_type > 0:
            reg_extras.append(f"type={args.reg_type}")
        if args.reg_team_size > 0:
            reg_extras.append(f"team_size={args.reg_team_size}")
        if args.reg_gamma > 0:
            reg_extras.append(f"gamma={args.reg_gamma}")
        if args.reg_game_weights > 0 and weight_module is not None:
            reg_extras.append(f"game_weights={args.reg_game_weights}")
        if reg_extras:
            print(f"Extra L2 reg: {', '.join(reg_extras)}")
        # Games = total in DB (or from dataset if DB unavailable)
        db_games = get_player_games_from_db(maps.idx_to_player_id)
        if db_games:
            player_games = np.array(
                [db_games.get(pid, 0) for pid in maps.idx_to_player_id],
                dtype=np.int64,
            )
            print(f"Player games: from DB (total tournaments)")
        else:
            player_games = compute_player_games(train_data, maps.idx_to_question_id, n_players)
            print(f"Player games: from dataset (DB unavailable)")
        n_experienced = (player_games >= MIN_GAMES_FOR_CENTER).sum()
        if n_experienced > 0:
            theta_center_mask = torch.from_numpy(player_games >= MIN_GAMES_FOR_CENTER).to(device)
            print(f"Theta centering: by {n_experienced} players with >={MIN_GAMES_FOR_CENTER} games (total in DB)")
        else:
            print(f"Theta centering: no players with >={MIN_GAMES_FOR_CENTER} games, using global mean")
        if args.rookie_floor < 0:
            anchor_np = args.rookie_floor * np.exp(-player_games.astype(np.float32) / args.rookie_tau)
            theta_anchor = torch.from_numpy(anchor_np).float().to(device)
            n_affected = (anchor_np < -0.1).sum()
            print(f"Rookie prior: floor={args.rookie_floor}, tau={args.rookie_tau} "
                  f"({n_affected} players with anchor < -0.1)")

    # Data-driven initial values for theta, b and optionally log_a (only when not resuming)
    theta_init = None
    b_init = None
    log_a_init = None
    if not args.resume and isinstance(train_data, PackedData):
        print("Computing empirical init (theta/b and optional log_a) ...")
        theta_init, b_init, log_a_init = compute_empirical_init(
            train_data, n_players, n_questions, eps=1e-4, scale=0.5, scale_log_a=0.8, log_a_clamp=1.5
        )
        if theta_init is not None and b_init is not None:
            if theta_anchor is not None:
                theta_init = np.minimum(theta_init, anchor_np * 0.5)
            model.theta.data.copy_(torch.from_numpy(theta_init).to(device))
            # Aggregate per-question inits to per-canonical-question (average over paired slots)
            cq = maps.canonical_q_idx
            if cq is not None and num_canonical < n_questions:
                b_canon = np.zeros(num_canonical, dtype=np.float32)
                cnt = np.zeros(num_canonical, dtype=np.float32)
                np.add.at(b_canon, cq, b_init)
                np.add.at(cnt, cq, 1.0)
                b_canon /= np.maximum(cnt, 1.0)
                model.b.data.copy_(torch.from_numpy(b_canon).to(device))
            else:
                model.b.data.copy_(torch.from_numpy(b_init).to(device))
            apply_identifiability(
                model, center_theta=True, center_b=False,
                theta_center_mask=theta_center_mask,
            )
            msg = "Initialized theta and b from empirical take rates (scale=0.5), theta centered"
            if log_a_init is not None:
                if cq is not None and num_canonical < n_questions:
                    a_canon = np.zeros(num_canonical, dtype=np.float32)
                    cnt_a = np.zeros(num_canonical, dtype=np.float32)
                    np.add.at(a_canon, cq, log_a_init)
                    np.add.at(cnt_a, cq, 1.0)
                    a_canon /= np.maximum(cnt_a, 1.0)
                    model.log_a.data.copy_(torch.from_numpy(a_canon).to(device))
                else:
                    model.log_a.data.copy_(torch.from_numpy(log_a_init).to(device))
                msg += "; log_a from correlation(taken, team_strength)"
            print(msg)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    _log_gamma_init: Optional[float] = None
    if hasattr(model, "log_dl_power_gamma"):
        _log_gamma_init = math.log(max(args.dl_power_gamma, 1e-6))

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_mean_p = train_epoch(
            model, train_data, optimizer, device, args.batch_size,
            reg_theta=args.reg_theta, reg_b=args.reg_b, reg_log_a=args.reg_log_a,
            reg_type=args.reg_type, reg_team_size=args.reg_team_size,
            reg_gamma=args.reg_gamma, reg_game_weights=args.reg_game_weights,
            log_gamma_init=_log_gamma_init,
            center_theta=True, center_b=True,
            theta_reg_weights=theta_reg_weights,
            theta_anchor=theta_anchor,
            question_reg_weights=question_reg_weights,
            theta_center_mask=theta_center_mask,
            weight_module=weight_module,
            grad_clip=args.grad_clip,
        )
        val_loss, val_mean_p, val_brier, val_auc = evaluate(model, val_data, device, weight_module=weight_module)
        print(
            f"Epoch {epoch+1}/{args.epochs} | loss {train_loss:.4f} mean_p {train_mean_p:.4f} | "
            f"val_loss {val_loss:.4f} val_p {val_mean_p:.4f} Brier {val_brier:.4f} AUC {val_auc:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1

        if args.checkpoint_dir:
            path = os.path.join(args.checkpoint_dir, "latest.pt")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "num_players": n_players,
                "num_questions": n_questions,
                "idx_to_player_id": maps.idx_to_player_id,
                "idx_to_question_id": maps.idx_to_question_id,
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_loss": best_val_loss,
                "dl_transform": args.dl_transform,
                "dl_power_gamma": float(args.dl_power_gamma),
                "dl_learn_power": args.dl_learn_power,
            }
            if getattr(maps, "tournament_dl", None) is not None:
                ckpt_dict["tournament_dl"] = maps.tournament_dl
            if getattr(maps, "tournament_type", None) is not None:
                ckpt_dict["tournament_type"] = maps.tournament_type
            if getattr(maps, "canonical_q_idx", None) is not None:
                ckpt_dict["canonical_q_idx"] = maps.canonical_q_idx
                ckpt_dict["num_canonical_questions"] = maps.num_canonical_questions
            torch.save(ckpt_dict, path)

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Diagnostics + uncertainty-aware ranking
    theta = model.theta.detach().cpu().numpy()
    b = model.b.detach().cpu().numpy()
    a = torch.exp(model.log_a.detach()).cpu().numpy()
    ts_bias = torch.exp(model.team_size_bias.detach()).cpu().numpy()
    idx_to_player = maps.idx_to_player_id
    idx_to_question = maps.idx_to_question_id
    fisher = np.zeros_like(theta, dtype=np.float32)
    if isinstance(train_data, PackedData):
        prior_precision = float(args.reg_theta)
        for start in range(0, len(train_data), args.batch_size):
            end = min(start + args.batch_size, len(train_data))
            batch_idx = np.arange(start, end)
            q, flat_p, sizes, taken = train_data.get_batch(batch_idx)
            q, flat_p, sizes, taken = q.to(device), flat_p.to(device), sizes.to(device), taken.to(device)
            if weight_module is not None and train_data.game_idx is not None:
                w_game = weight_module()
                obs_w = w_game[train_data.game_idx[batch_idx]]
            elif train_data.obs_weight is not None:
                obs_w = torch.from_numpy(train_data.obs_weight[batch_idx]).float().to(device)
            else:
                obs_w = torch.ones(end - start, dtype=torch.float32, device=device)
            fd = fisher_diag_theta(
                model=model,
                question_indices=q,
                player_indices_flat=flat_p,
                team_sizes=sizes,
                taken=taken,
                obs_weights=obs_w,
                num_players=n_players,
            )
            fisher += fd.detach().cpu().numpy().astype(np.float32)
        se, rating = conservative_rating(theta, fisher, prior_precision=prior_precision, c=float(args.rating_c))
    else:
        se = np.full_like(theta, np.nan, dtype=np.float32)
        rating = theta.copy()

    if getattr(model, "dl_learn_power", False):
        gamma = torch.exp(model.log_dl_power_gamma).clamp(min=0.5, max=4.0).item()
        print(f"\nLearned dl power gamma: {gamma:.4f}")

    print("\nTeam size multipliers (exp(bias)):")
    for size in range(1, len(ts_bias)):
        print(f"  size {size}: {ts_bias[size]:.4f}")

    top_players = np.argsort(theta)[::-1][:10]
    print("\nTop 10 strongest players (θ):")
    for i, idx in enumerate(top_players, 1):
        pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
        print(f"  {i}. player_id={pid} θ={theta[idx]:.4f}")

    top_rating = np.argsort(rating)[::-1][:10]
    print("\nTop 10 by conservative rating (theta - c*SE):")
    for i, idx in enumerate(top_rating, 1):
        pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
        print(f"  {i}. player_id={pid} rating={rating[idx]:.4f} θ={theta[idx]:.4f} SE={se[idx]:.4f}")

    top_se = np.argsort(se)[::-1][:10]
    print("\nTop 10 highest uncertainty (SE):")
    for i, idx in enumerate(top_se, 1):
        pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
        print(f"  {i}. player_id={pid} SE={se[idx]:.4f} θ={theta[idx]:.4f}")

    # Top 10 among players with ≥100 games (by distinct tournaments in train data)
    if isinstance(train_data, PackedData) and hasattr(maps, "idx_to_question_id"):
        q_keys = maps.idx_to_question_id
        off = train_data.offsets
        player_tournaments: dict[int, set] = {}
        for i in range(len(train_data.q_idx)):
            qidx = train_data.q_idx[i]
            tid = q_keys[qidx][0] if isinstance(q_keys[qidx], tuple) else q_keys[qidx]
            for pidx in train_data.player_indices_flat[off[i] : off[i + 1]]:
                player_tournaments.setdefault(int(pidx), set()).add(tid)
        player_games = np.array([len(player_tournaments.get(j, set())) for j in range(n_players)])
        eligible = np.where(player_games >= 100)[0]
        if len(eligible) > 0:
            top_eligible = eligible[np.argsort(theta[eligible])[::-1][:10]]
            print("\nTop 10 strongest (θ) with ≥100 games in train:")
            for i, idx in enumerate(top_eligible, 1):
                pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
                print(f"  {i}. player_id={pid} θ={theta[idx]:.4f}  games={player_games[idx]}")

    top_hard = np.argsort(b)[:10]
    print("\nTop 10 hardest questions (b):")
    for i, idx in enumerate(top_hard, 1):
        qid = idx_to_question[idx] if idx < len(idx_to_question) else idx
        print(f"  {i}. question_id={qid} b={b[idx]:.4f}")

    top_selective = np.argsort(a)[::-1][:10]
    print("\nTop 10 most selective questions (a):")
    for i, idx in enumerate(top_selective, 1):
        qid = idx_to_question[idx] if idx < len(idx_to_question) else idx
        print(f"  {i}. question_id={qid} a={a[idx]:.4f}")

    if weight_module is not None:
        ew = weight_module.effective_weights()
        print(
            f"\nLearned game weights (effective):"
            f"\n  type: offline={ew['type_offline']:.3f}  sync={ew['type_sync']:.3f}  async={ew['type_async']:.3f}"
            f"\n  size_coef={ew['size_coef']:.3f}  mix_coef={ew['mix_coef']:.3f}"
        )

    if args.players_out:
        player_ids = [idx_to_player[idx] if idx < len(idx_to_player) else idx for idx in range(len(theta))]
        # Exposure in the current training slice: distinct tournaments and total team-question appearances.
        train_games = np.zeros(len(theta), dtype=np.int64)
        train_samples = np.zeros(len(theta), dtype=np.int64)
        if isinstance(train_data, PackedData):
            q_keys = maps.idx_to_question_id
            off = train_data.offsets
            player_tournaments: dict[int, set] = {}
            for i in range(len(train_data.q_idx)):
                qidx = train_data.q_idx[i]
                tid = q_keys[qidx][0] if isinstance(q_keys[qidx], tuple) else q_keys[qidx]
                for pidx in train_data.player_indices_flat[off[i] : off[i + 1]]:
                    p = int(pidx)
                    train_samples[p] += 1
                    player_tournaments.setdefault(p, set()).add(tid)
            train_games = np.array([len(player_tournaments.get(j, set())) for j in range(len(theta))], dtype=np.int64)
        allowed_idx = set(range(len(theta)))
        if args.export_min_games > 0 and player_ids:
            try:
                import os as _os
                import psycopg2

                url = _os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
                conn = psycopg2.connect(url)
                cur = conn.cursor()
                min_date = (args.min_tournament_date or "").strip()
                if min_date:
                    cur.execute(
                        """
                        SELECT tr.player_id, COUNT(DISTINCT tr.tournament_id)
                        FROM public.tournament_rosters tr
                        JOIN public.tournaments t ON t.id = tr.tournament_id
                        WHERE tr.player_id = ANY(%s)
                          AND (t.start_datetime IS NULL OR t.start_datetime >= %s::timestamp)
                        GROUP BY tr.player_id
                        HAVING COUNT(DISTINCT tr.tournament_id) >= %s
                        """,
                        (player_ids, min_date, args.export_min_games),
                    )
                else:
                    cur.execute(
                        """
                        SELECT player_id, COUNT(DISTINCT tournament_id)
                        FROM public.tournament_rosters
                        WHERE player_id = ANY(%s)
                        GROUP BY player_id
                        HAVING COUNT(DISTINCT tournament_id) >= %s
                        """,
                        (player_ids, args.export_min_games),
                    )
                allowed_ids = {r[0] for r in cur.fetchall()}
                conn.close()
                allowed_idx = {idx for idx in range(len(theta)) if player_ids[idx] in allowed_ids}
                date_note = f" (tournaments >= {min_date})" if min_date else ""
                print(f"\nExport players: min_games {args.export_min_games}{date_note} -> {len(allowed_idx)} of {len(theta)} players")
            except Exception as e:
                print(f"DB unavailable for export_min_games filter ({e}), exporting all players", file=sys.stderr)
        out_path = args.players_out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["player_id", "theta", "SE", "rating", "num_games", "num_obs"])
            for idx in sorted(allowed_idx):
                pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
                w.writerow(
                    [
                        pid,
                        round(float(theta[idx]), 6),
                        round(float(se[idx]), 6) if np.isfinite(se[idx]) else "",
                        round(float(rating[idx]), 6),
                        int(train_games[idx]),
                        int(train_samples[idx]),
                    ]
                )
        print(f"Players saved to {out_path}")

    if args.games_weights_out and game_weights is not None and getattr(maps, "idx_to_game_id", None):
        out_path = args.games_weights_out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        w_game_export = game_weights.w_game
        if weight_module is not None:
            with torch.no_grad():
                w_game_export = weight_module().detach().cpu().numpy()
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["game_id", "w_type", "w_size", "w_mix", "w_game", "N_teams", "entropy", "date"])
            for gi, gid in enumerate(maps.idx_to_game_id):
                d = ""
                if maps.game_date_ordinal is not None and gi < len(maps.game_date_ordinal) and maps.game_date_ordinal[gi] >= 0:
                    import datetime as _dt
                    d = _dt.date.fromordinal(int(maps.game_date_ordinal[gi])).isoformat()
                w.writerow(
                    [
                        gid,
                        round(float(game_weights.w_type[gi]), 6),
                        round(float(game_weights.w_size[gi]), 6),
                        round(float(game_weights.w_mix[gi]), 6),
                        round(float(w_game_export[gi]), 6),
                        round(float(game_weights.n_teams[gi]), 3),
                        round(float(game_weights.entropy[gi]), 6),
                        d,
                    ]
                )
        print(f"Game weights saved to {out_path}")
        top_mix_games = np.argsort(game_weights.w_mix)[::-1][:10]
        print("\nTop 10 mixing games (w_mix):")
        for i, gi in enumerate(top_mix_games, 1):
            gid = maps.idx_to_game_id[gi]
            print(f"  {i}. game_id={gid} w_mix={game_weights.w_mix[gi]:.4f} entropy={game_weights.entropy[gi]:.4f}")

    if args.run_ablations and isinstance(train_data, PackedData) and isinstance(val_data, PackedData) and game_weights is not None:
        print("\nRunning ablations: [type], [type+size], [type+size+mix]")

        def suspicious_count(theta_arr: np.ndarray, se_arr: np.ndarray) -> int:
            t_thr = np.quantile(theta_arr, 0.95)
            s_thr = np.quantile(se_arr[np.isfinite(se_arr)], 0.95)
            return int(np.sum((theta_arr >= t_thr) & (se_arr >= s_thr)))

        def rank_volatility(theta_arr: np.ndarray, rating_arr: np.ndarray, top_n: int = 100) -> float:
            top_theta = set(np.argsort(theta_arr)[::-1][:top_n].tolist())
            top_rating = set(np.argsort(rating_arr)[::-1][:top_n].tolist())
            return 1.0 - (len(top_theta & top_rating) / float(max(1, top_n)))

        # Preserve current full-weight setup.
        train_obs_backup = None if train_data.obs_weight is None else train_data.obs_weight.copy()
        val_obs_backup = None if val_data.obs_weight is None else val_data.obs_weight.copy()
        base_theta = theta.copy()

        ablation_defs = [
            ("type_only", game_weights.w_type),
            ("type_size", game_weights.w_type * game_weights.w_size),
            ("type_size_mix", game_weights.w_type * game_weights.w_size * game_weights.w_mix),
        ]
        ablation_rows = []

        for ab_idx, (name, gw) in enumerate(ablation_defs, 1):
            print(f"  Ablation {ab_idx}/3: {name} ...", flush=True)
            gw = gw.astype(np.float32)
            mean_train = float(np.mean(gw[np.unique(train_data.game_idx)]))
            gw_norm = gw / max(mean_train, 1e-8)
            train_data.obs_weight = gw_norm[train_data.game_idx]
            val_data.obs_weight = gw_norm[val_data.game_idx]

            set_seed(args.seed)
            m = ChGKModel(
                n_players,
                n_questions,
                num_canonical_questions=num_canonical if num_canonical < n_questions else None,
                canonical_q_idx=canonical_q_idx_tensor,
                tournament_dl=tournament_dl_tensor,
                tournament_type=tournament_type_tensor,
                dl_transform=args.dl_transform,
                dl_power_gamma=args.dl_power_gamma,
                dl_learn_power=args.dl_learn_power,
            ).to(device)
            if theta_init is not None and b_init is not None:
                m.theta.data.copy_(torch.from_numpy(theta_init).to(device))
                # Reuse canonical-aggregated init from main model
                m.b.data.copy_(model.b.data.detach().clone())
                if log_a_init is not None:
                    m.log_a.data.copy_(model.log_a.data.detach().clone())
                apply_identifiability(m, center_theta=True, center_b=False, theta_center_mask=theta_center_mask)
            opt = torch.optim.Adam(m.parameters(), lr=args.lr)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
            best = float("inf")
            noimp = 0
            for _epoch in range(args.epochs):
                train_epoch(
                    m, train_data, opt, device, args.batch_size,
                    reg_theta=args.reg_theta, reg_b=args.reg_b, reg_log_a=args.reg_log_a,
                    reg_type=args.reg_type, reg_team_size=args.reg_team_size,
                    reg_gamma=args.reg_gamma, reg_game_weights=args.reg_game_weights,
                    log_gamma_init=_log_gamma_init,
                    center_theta=True, center_b=True,
                    theta_reg_weights=theta_reg_weights,
                    theta_anchor=theta_anchor,
                    question_reg_weights=question_reg_weights,
                    theta_center_mask=theta_center_mask,
                    weight_module=None,
                    show_progress=False,
                    grad_clip=args.grad_clip,
                )
                vloss, _, _, _ = evaluate(m, val_data, device)
                sch.step(vloss)
                if (_epoch + 1) % 10 == 0 or _epoch == 0:
                    print(f"    epoch {_epoch + 1}/{args.epochs} val_loss={vloss:.4f}", flush=True)
                if vloss < best:
                    best = vloss
                    noimp = 0
                else:
                    noimp += 1
                    if noimp >= args.patience:
                        break

            th = m.theta.detach().cpu().numpy()
            fisher_ab = np.zeros_like(th, dtype=np.float32)
            print(f"    computing Fisher ...", flush=True)
            for start in range(0, len(train_data), args.batch_size):
                end = min(start + args.batch_size, len(train_data))
                bidx = np.arange(start, end)
                q, flat_p, sizes, taken = train_data.get_batch(bidx)
                q, flat_p, sizes, taken = q.to(device), flat_p.to(device), sizes.to(device), taken.to(device)
                obs_w = torch.from_numpy(train_data.obs_weight[bidx]).float().to(device)
                fd = fisher_diag_theta(
                    model=m,
                    question_indices=q,
                    player_indices_flat=flat_p,
                    team_sizes=sizes,
                    taken=taken,
                    obs_weights=obs_w,
                    num_players=n_players,
                )
                fisher_ab += fd.detach().cpu().numpy().astype(np.float32)
            se_ab, rating_ab = conservative_rating(
                th,
                fisher_ab,
                prior_precision=float(args.reg_theta),
                c=float(args.rating_c),
            )
            ablation_rows.append(
                (
                    name,
                    best,
                    rank_volatility(base_theta, rating_ab),
                    suspicious_count(th, se_ab),
                )
            )

        print("\nAblation results (holdout logloss / volatility / suspicious):")
        for name, ll, vol, susp in ablation_rows:
            print(f"  {name:14s} logloss={ll:.5f}  volatility={vol:.4f}  suspicious={susp}")

        # Compare full weighting against no-mix variant on the trained model.
        if val_data.game_idx is not None:
            no_mix = (game_weights.w_type * game_weights.w_size).astype(np.float32)
            no_mix /= max(float(np.mean(no_mix[np.unique(train_data.game_idx)])), 1e-8)
            full = game_weights.w_norm.astype(np.float32)
            full_ll = weighted_logloss(
                torch.from_numpy(val_data.taken).float(),
                model.forward_packed(
                    torch.from_numpy(val_data.q_idx).long().to(device),
                    torch.from_numpy(val_data.player_indices_flat).long().to(device),
                    torch.from_numpy(val_data.team_sizes).long().to(device),
                ).detach().cpu(),
                torch.from_numpy(full[val_data.game_idx]).float(),
            ).item()
            no_mix_ll = weighted_logloss(
                torch.from_numpy(val_data.taken).float(),
                model.forward_packed(
                    torch.from_numpy(val_data.q_idx).long().to(device),
                    torch.from_numpy(val_data.player_indices_flat).long().to(device),
                    torch.from_numpy(val_data.team_sizes).long().to(device),
                ).detach().cpu(),
                torch.from_numpy(no_mix[val_data.game_idx]).float(),
            ).item()
            print(f"\nPredictive weighted logloss on val: no_mix={no_mix_ll:.5f}, with_mix={full_ll:.5f}")

        train_data.obs_weight = train_obs_backup
        val_data.obs_weight = val_obs_backup

    if args.calibration_plot:
        # For calibration plot, use val_data
        if isinstance(val_data, PackedData):
            qv, flat_pv, sizes_v, taken_v = val_data.get_batch(np.arange(len(val_data)))
        else:
            qv, flat_pv, sizes_v, taken_v = packed_from_samples(val_data)
        qv, flat_pv, sizes_v, taken_v = qv.to(device), flat_pv.to(device), sizes_v.to(device), taken_v.to(device)
        model.eval()
        with torch.no_grad():
            p_val = model.forward_packed(qv, flat_pv, sizes_v)
        plot_calibration(taken_v, p_val, path=args.calibration_plot)

    if args.save_model:
        path = args.save_model
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {
            "model_state": model.state_dict(),
            "num_players": maps.num_players,
            "num_questions": maps.num_questions,
            "idx_to_player_id": maps.idx_to_player_id,
            "idx_to_question_id": maps.idx_to_question_id,
            "dl_transform": args.dl_transform,
            "dl_power_gamma": float(args.dl_power_gamma),
            "dl_learn_power": args.dl_learn_power,
        }
        if getattr(maps, "canonical_q_idx", None) is not None:
            save_dict["canonical_q_idx"] = maps.canonical_q_idx
            save_dict["num_canonical_questions"] = maps.num_canonical_questions
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
