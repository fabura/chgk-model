"""
Training loop: penalized MLE (binary CE + L2), identifiability (center θ, optional center b).
Optimized for large datasets (NumPy-based cache and packed data).
"""
from __future__ import annotations

import argparse
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
    load_cached,
    load_from_db,
    save_cached,
    samples_to_arrays,
    samples_to_tensors,
    train_val_split,
)
from metrics import auc_roc, brier_score, logloss as metric_logloss, plot_calibration
from model import ChGKModel


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

    kept_indices: list[int] = []
    new_team_sizes_list: list[int] = []
    new_flat_list = []
    for i in range(len(data)):
        start, end = data.offsets[i], data.offsets[i + 1]
        team = data.player_indices_flat[start:end]
        team_active = team[active[team] == 1]
        if team_active.size == 0:
            continue
        kept_indices.append(i)
        new_team_sizes_list.append(int(team_active.size))
        new_flat_list.append(old_to_new[team_active])

    kept_indices_arr = np.array(kept_indices, dtype=np.int64)
    new_q_idx = data.q_idx[kept_indices_arr]
    new_taken = data.taken[kept_indices_arr]
    new_team_sizes = np.array(new_team_sizes_list, dtype=data.team_sizes.dtype)
    if new_flat_list:
        new_player_indices_flat = np.concatenate(new_flat_list).astype(np.int32, copy=False)
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

    new_idx_to_player_id = [maps.idx_to_player_id[j] for j in active_idx]
    new_maps = IndexMaps(
        player_id_to_idx={pid: i for i, pid in enumerate(new_idx_to_player_id)},
        question_id_to_idx=maps.question_id_to_idx,
        idx_to_player_id=new_idx_to_player_id,
        idx_to_question_id=maps.idx_to_question_id,
        tournament_dl=getattr(maps, "tournament_dl", None),
        tournament_type=getattr(maps, "tournament_type", None),
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
    mean_b: Optional[float] = 0.0,
    theta_reg_weights: Optional[torch.Tensor] = None,
    question_reg_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative log-likelihood + L2 regularization.

    When *_reg_weights are provided, the L2 penalty for each parameter is
    scaled by its weight (typically 1/sqrt(obs_count)), so that players or
    questions with many observations are penalized less relative to their
    data signal.
    """
    p = model.forward_packed(question_indices, player_indices_flat, team_sizes)
    nll = metric_logloss(taken, p)

    # --- theta regularization (observation-normalized) ---
    if theta_reg_weights is not None:
        reg_t = reg_theta * (theta_reg_weights * model.theta ** 2).mean()
    else:
        reg_t = reg_theta * (model.theta ** 2).mean()

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
    if mean_b is not None:
        reg = reg + 1e2 * (model.b.mean() - mean_b) ** 2
    return nll + reg


def compute_reg_weights(
    data: PackedData,
    n_players: int,
    n_questions: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-player and per-question regularization weights = 1/sqrt(count).

    Players/questions with more observations get lower weight, so the L2 penalty
    doesn't dominate their data signal.  Unseen items get weight 1.0.
    """
    player_counts = np.zeros(n_players, dtype=np.float64)
    np.add.at(player_counts, data.player_indices_flat, 1.0)
    question_counts = np.bincount(data.q_idx, minlength=n_questions).astype(np.float64)

    # weight = 1 / sqrt(count), with floor of 1 observation to avoid div-by-zero
    player_w = 1.0 / np.sqrt(np.maximum(player_counts, 1.0))
    question_w = 1.0 / np.sqrt(np.maximum(question_counts, 1.0))

    # Normalize so mean weight = 1 (keeps the reg scale comparable to before)
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
    center_theta: bool,
    center_b: bool,
    show_progress: bool = True,
    theta_reg_weights: Optional[torch.Tensor] = None,
    question_reg_weights: Optional[torch.Tensor] = None,
    theta_center_mask: Optional[torch.Tensor] = None,
    theta_freeze_mask: Optional[torch.Tensor] = None,
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
        else:
            batch_samples = [data[i] for i in batch_idx]
            q, flat_p, sizes, taken = packed_from_samples(batch_samples)
            
        q, flat_p, sizes, taken = q.to(device), flat_p.to(device), sizes.to(device), taken.to(device)
        optimizer.zero_grad()
        loss = loss_fn(
            model, q, flat_p, sizes, taken,
            reg_theta=reg_theta, reg_b=reg_b, reg_log_a=reg_log_a,
            mean_b=0.0 if center_b else None,
            theta_reg_weights=theta_reg_weights,
            question_reg_weights=question_reg_weights,
        )
        loss.backward()
        if theta_freeze_mask is not None and model.theta.grad is not None:
            model.theta.grad[theta_freeze_mask] = 0
        optimizer.step()
        apply_identifiability(
            model, center_theta=center_theta, center_b=center_b,
            theta_center_mask=theta_center_mask,
        )
        if theta_freeze_mask is not None:
            with torch.no_grad():
                model.theta[theta_freeze_mask] = 0

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
) -> Tuple[float, float, float, float]:
    """Returns mean_loss, mean_p, brier, auc."""
    model.eval()
    n_samples = len(data)
    if n_samples == 0:
        return 0.0, 0.0, 0.0, float("nan")
        
    all_p = []
    all_taken = []
    total_loss = 0.0
    
    # Evaluate in batches to avoid OOM
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = np.arange(start, end)
        
        if isinstance(data, PackedData):
            q, flat_p, sizes, taken = data.get_batch(batch_idx)
        else:
            batch_samples = [data[i] for i in batch_idx]
            q, flat_p, sizes, taken = packed_from_samples(batch_samples)
            
        q, flat_p, sizes, taken = q.to(device), flat_p.to(device), sizes.to(device), taken.to(device)
        p = model.forward_packed(q, flat_p, sizes)
        
        total_loss += metric_logloss(taken, p).item() * (end - start)
        all_p.append(p.cpu())
        all_taken.append(taken.cpu())
        
    p = torch.cat(all_p)
    taken = torch.cat(all_taken)
    
    mean_loss = total_loss / n_samples
    brier = brier_score(taken, p).item()
    auc = auc_roc(taken, p)
    return mean_loss, p.mean().item(), brier, auc


def main() -> int:
    parser = argparse.ArgumentParser(description="ChGK model training")
    parser.add_argument("--mode", choices=["synthetic", "db"], default="synthetic")
    parser.add_argument("--model_mode", choices=["exp", "sigmoid"], default="exp", help="Team aggregation mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--reg_theta", type=float, default=1e-4)
    parser.add_argument("--reg_b", type=float, default=1e-3)
    parser.add_argument("--reg_log_a", type=float, default=1e-3)
    parser.add_argument(
        "--dl_transform",
        choices=["linear", "log1p", "power"],
        default="linear",
        help="Transform for tournament_dl before scaling: linear/log1p/power. Default linear.",
    )
    parser.add_argument(
        "--dl_log_alpha",
        type=float,
        default=1.0,
        help="Alpha for dl_transform=log1p: f(dl)=log1p(alpha*dl). Default 1.0.",
    )
    parser.add_argument(
        "--dl_power_gamma",
        type=float,
        default=0.5,
        help="Gamma for dl_transform=power: f(dl)=sign(dl)*|dl|^gamma. Default 0.5.",
    )
    parser.add_argument(
        "--dl_normalize_by_type",
        action="store_true",
        default=True,
        help="Use z-score normalization of tournament_dl separately per tournament type before transform/scale. Default on.",
    )
    parser.add_argument(
        "--no_dl_normalize_by_type",
        action="store_false",
        dest="dl_normalize_by_type",
        help="Disable per-type z-score normalization of tournament_dl.",
    )
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--max_tournaments", type=int, default=None, help="DB: limit tournaments")
    parser.add_argument(
        "--all_tournaments",
        action="store_true",
        help="DB: include every tournament with enough questions (default: only those with points_mask data)",
    )
    parser.add_argument(
        "--tournament_dl_filter",
        choices=["all", "true_dl", "ndcg", "any", "both"],
        default="all",
        help="DB: filter tournaments by difficulty-source availability: all/true_dl/ndcg/any/both. Default all.",
    )
    parser.add_argument(
        "--min_tournament_date",
        type=str,
        default="2015-01-01",
        help="DB: only tournaments with start_datetime >= this date (YYYY-MM-DD). Empty string = no filter. Default 2015-01-01.",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        help="DB: path to cache file. If file exists, load from it; else load from DB and save to it. No DB needed on subsequent runs.",
    )
    parser.add_argument(
        "--min_games_in_dataset",
        type=int,
        default=10,
        help="Treat players with fewer games as inactive: remove them from team rosters; drop only teams that become empty. 0=no filter. Default 10.",
    )
    parser.add_argument("--calibration_plot", type=str, default=None, help="Save calibration curve to path")
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Save trained model (theta, b, log_a) and index maps to this path (.pt). Can load later for predictions or analysis.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Save checkpoint after each epoch to this dir (latest.pt). Use with --resume to continue training.",
    )
    parser.add_argument(
        "--players_out",
        type=str,
        default="results/players.csv",
        help="Export player_id,theta to CSV at the end of training (default: results/players.csv). Empty string = do not export.",
    )
    parser.add_argument(
        "--export_min_games",
        type=int,
        default=30,
        help="When exporting players_out: only include players with at least this many games (since min_tournament_date). 0 = export all. Default 30.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path (e.g. checkpoint_dir/latest.pt). Requires same --cache_file and --seed for same train/val split.",
    )
    args = parser.parse_args()

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
    
    if args.mode == "synthetic":
        samples, maps = generate_synthetic(seed=args.seed)
    else:
        cache_path = args.cache_file
        if cache_path and os.path.isfile(cache_path):
            print(f"Loading from cache: {cache_path}")
            arrays, maps = load_cached(cache_path)
            packed_data = PackedData.from_arrays(arrays)
        else:
            samples, maps = load_from_db(
                max_tournaments=args.max_tournaments,
                only_with_question_data=not args.all_tournaments,
                tournament_dl_filter=args.tournament_dl_filter,
                min_tournament_date=args.min_tournament_date or None,
                min_games=args.min_games_in_dataset,
                seed=args.seed,
            )
            if cache_path:
                for _ in tqdm([1], desc="Saving cache", unit=" file"):
                    save_cached(samples, maps, cache_path, meta={"max_tournaments": args.max_tournaments})
            packed_data = PackedData.from_arrays(samples_to_arrays(samples))

    # Optionally remove low-game players from rosters (drop only teams that become empty), then reindex players
    if packed_data and args.min_games_in_dataset > 0:
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

    if packed_data:
        n_samples = len(packed_data)
        indices = np.random.RandomState(args.seed).permutation(n_samples)
        n_val = max(1, int(n_samples * args.val_frac))
        train_data, val_data = packed_data.split(indices[n_val:], indices[:n_val])
    else:
        train_data, val_data = train_val_split(samples, val_frac=args.val_frac, seed=args.seed)
        
    n_players = maps.num_players
    n_questions = maps.num_questions
    print(f"Train {len(train_data)} val {len(val_data)} | players {n_players} questions {n_questions}")

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
        if "dl_log_alpha" in ckpt:
            args.dl_log_alpha = float(ckpt["dl_log_alpha"])
        if "dl_power_gamma" in ckpt:
            args.dl_power_gamma = float(ckpt["dl_power_gamma"])
        if "dl_normalize_by_type" in ckpt:
            args.dl_normalize_by_type = bool(ckpt["dl_normalize_by_type"])

    tournament_dl_tensor = None
    tournament_type_tensor = None
    if getattr(maps, "tournament_dl", None) is not None:
        tournament_dl_tensor = torch.from_numpy(maps.tournament_dl).float()
        print(f"Using tournament true_dl for school/student correction ({maps.tournament_dl.size} questions)")
    if getattr(maps, "tournament_type", None) is not None:
        tournament_type_tensor = torch.from_numpy(maps.tournament_type).long()
        print("Using tournament type (Очник/Синхрон/Асинхрон) for type-dependent dl scale")
    print(
        f"Tournament dl transform: {args.dl_transform}"
        + (
            f" (alpha={args.dl_log_alpha})"
            if args.dl_transform == "log1p"
            else f" (gamma={args.dl_power_gamma})"
            if args.dl_transform == "power"
            else ""
        )
        + (", zscore_by_type=on" if args.dl_normalize_by_type else ", zscore_by_type=off")
    )
    model = ChGKModel(
        n_players,
        n_questions,
        mode=args.model_mode,
        tournament_dl=tournament_dl_tensor,
        tournament_type=tournament_type_tensor,
        dl_transform=args.dl_transform,
        dl_log_alpha=args.dl_log_alpha,
        dl_power_gamma=args.dl_power_gamma,
        dl_normalize_by_type=args.dl_normalize_by_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

    # Compute observation-normalized regularization weights, theta center mask, and freeze mask
    theta_reg_weights = None
    question_reg_weights = None
    theta_center_mask = None
    theta_freeze_mask = None
    MIN_GAMES_FOR_CENTER = 500
    MIN_GAMES_FOR_TRAIN = 10  # Players with fewer games: theta fixed at 0, not updated
    if isinstance(train_data, PackedData):
        theta_reg_weights, question_reg_weights = compute_reg_weights(
            train_data, n_players, n_questions, device,
        )
        print(f"Reg weights: theta min={theta_reg_weights.min():.4f} max={theta_reg_weights.max():.4f} | "
              f"question min={question_reg_weights.min():.4f} max={question_reg_weights.max():.4f}")
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
        n_freeze = (player_games < MIN_GAMES_FOR_TRAIN).sum()
        if n_freeze > 0:
            theta_freeze_mask = torch.from_numpy(player_games < MIN_GAMES_FOR_TRAIN).to(device)
            print(f"Theta frozen at 0 for {n_freeze} players with <{MIN_GAMES_FOR_TRAIN} games (excluded from training)")

    # Data-driven initial values for theta, b and optionally log_a (only when not resuming)
    if not args.resume and isinstance(train_data, PackedData):
        print("Computing empirical init (theta/b and optional log_a) ...")
        theta_init, b_init, log_a_init = compute_empirical_init(
            train_data, n_players, n_questions, eps=1e-4, scale=0.5, scale_log_a=0.8, log_a_clamp=1.5
        )
        if theta_init is not None and b_init is not None:
            model.theta.data.copy_(torch.from_numpy(theta_init).to(device))
            model.b.data.copy_(torch.from_numpy(b_init).to(device))
            apply_identifiability(
                model, center_theta=True, center_b=False,
                theta_center_mask=theta_center_mask,
            )
            msg = "Initialized theta and b from empirical take rates (scale=0.5), theta centered"
            if log_a_init is not None:
                model.log_a.data.copy_(torch.from_numpy(log_a_init).to(device))
                msg += "; log_a from correlation(taken, team_strength)"
            print(msg)

    # Frozen players (e.g. <10 games): theta=0, not updated (after init or after loading checkpoint)
    if theta_freeze_mask is not None:
        with torch.no_grad():
            model.theta[theta_freeze_mask] = 0

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_mean_p = train_epoch(
            model, train_data, optimizer, device, args.batch_size,
            reg_theta=args.reg_theta, reg_b=args.reg_b, reg_log_a=args.reg_log_a,
            center_theta=True, center_b=True,
            theta_reg_weights=theta_reg_weights,
            question_reg_weights=question_reg_weights,
            theta_center_mask=theta_center_mask,
            theta_freeze_mask=theta_freeze_mask,
        )
        val_loss, val_mean_p, val_brier, val_auc = evaluate(model, val_data, device)
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
                "dl_log_alpha": float(args.dl_log_alpha),
                "dl_power_gamma": float(args.dl_power_gamma),
                "dl_normalize_by_type": bool(args.dl_normalize_by_type),
            }
            if getattr(maps, "tournament_dl", None) is not None:
                ckpt_dict["tournament_dl"] = maps.tournament_dl
            if getattr(maps, "tournament_type", None) is not None:
                ckpt_dict["tournament_type"] = maps.tournament_type
            torch.save(ckpt_dict, path)

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Diagnostics
    theta = model.theta.detach().cpu().numpy()
    b = model.b.detach().cpu().numpy()
    a = torch.exp(model.log_a.detach()).cpu().numpy()
    ts_bias = torch.exp(model.team_size_bias.detach()).cpu().numpy()
    idx_to_player = maps.idx_to_player_id
    idx_to_question = maps.idx_to_question_id

    print("\nTeam size multipliers (exp(bias)):")
    for size in range(1, len(ts_bias)):
        print(f"  size {size}: {ts_bias[size]:.4f}")

    top_players = np.argsort(theta)[::-1][:10]
    print("\nTop 10 strongest players (θ):")
    for i, idx in enumerate(top_players, 1):
        pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
        print(f"  {i}. player_id={pid} θ={theta[idx]:.4f}")

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

    if args.players_out:
        import csv

        player_ids = [idx_to_player[idx] if idx < len(idx_to_player) else idx for idx in range(len(theta))]
        # Exposure in the current training slice: distinct tournaments and total team-question appearances.
        train_games = np.zeros(len(theta), dtype=np.int64)
        train_samples = np.zeros(len(theta), dtype=np.int64)
        if isinstance(train_data, PackedData) and hasattr(maps, "idx_to_question_id"):
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
        elif isinstance(train_data, list):
            player_tournaments: dict[int, set] = {}
            for s in train_data:
                qid = s.question_id
                tid = qid[0] if isinstance(qid, tuple) else qid
                for p in s.player_indices:
                    p = int(p)
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
            w.writerow(["player_id", "theta", "train_games", "train_samples"])
            for idx in sorted(allowed_idx):
                pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
                w.writerow([pid, round(float(theta[idx]), 6), int(train_games[idx]), int(train_samples[idx])])
        print(f"Players saved to {out_path}")

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
        torch.save(
            {
                "model_state": model.state_dict(),
                "num_players": maps.num_players,
                "num_questions": maps.num_questions,
                "idx_to_player_id": maps.idx_to_player_id,
                "idx_to_question_id": maps.idx_to_question_id,
                "dl_transform": args.dl_transform,
                "dl_log_alpha": float(args.dl_log_alpha),
                "dl_power_gamma": float(args.dl_power_gamma),
                "dl_normalize_by_type": bool(args.dl_normalize_by_type),
            },
            path,
        )
        print(f"Model saved to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
