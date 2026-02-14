"""
ChGK data loader: index maps, batched (game, question, team) → player indices + taken.
Supports synthetic data and DB load from rating backup (tournament_results + points_mask, tournament_rosters).
Cache: save/load (samples, index maps) to avoid re-querying the DB.
"""
from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class IndexMaps:
    """Maps external ids to contiguous indices for model."""

    player_id_to_idx: dict[int, int] = field(default_factory=dict)
    question_id_to_idx: dict[tuple[int, int] | int, int] = field(default_factory=dict)
    idx_to_player_id: list[int] = field(default_factory=list)
    idx_to_question_id: list[tuple[int, int] | int] = field(default_factory=list)
    # Optional: per-question tournament difficulty (true_dl from DB, fallback ndcg); school/student vs adult
    tournament_dl: Optional[np.ndarray] = None  # shape (num_questions,), float32
    # Optional: per-question tournament type index (0=Очник, 1=Синхрон, 2=Асинхрон) for type-dependent dl scale
    tournament_type: Optional[np.ndarray] = None  # shape (num_questions,), int32

    @property
    def num_players(self) -> int:
        return len(self.idx_to_player_id)

    @property
    def num_questions(self) -> int:
        return len(self.idx_to_question_id)


@dataclass
class Sample:
    """One (game, question, team) observation: question index, list of player indices, binary taken.
    team_strength: optional proxy for team strength (e.g. normalized tournament place, 1=first, 0=last).
    """

    question_idx: int
    player_indices: list[int]
    taken: int  # 0 or 1
    team_strength: Optional[float] = None  # Set when loading from DB (place in tournament)


def build_index_maps(samples: list[Sample]) -> IndexMaps:
    """Build player and question index maps from a list of samples."""
    player_ids: set[int] = set()
    question_ids: set[int | tuple[int, int]] = set()
    for s in samples:
        question_ids.add(s.question_idx)
        for p in s.player_indices:
            player_ids.add(p)
    # Use int keys: we already use indices in Sample when loading from synthetic/DB
    player_sorted = sorted(player_ids)
    question_sorted = sorted(question_ids)
    player_id_to_idx = {p: i for i, p in enumerate(player_sorted)}
    question_id_to_idx = {q: i for i, q in enumerate(question_sorted)}
    return IndexMaps(
        player_id_to_idx=player_id_to_idx,
        question_id_to_idx=question_id_to_idx,
        idx_to_player_id=player_sorted,
        idx_to_question_id=question_sorted,
    )


def samples_to_tensors(
    samples: list[Sample],
    maps: IndexMaps,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    Convert samples to batched tensors for model forward.
    Returns: question_indices [N], list of player_index tensors (ragged), taken [N].
    """
    q_idx = torch.tensor([s.question_idx for s in samples], dtype=torch.long)
    taken = torch.tensor([s.taken for s in samples], dtype=torch.float32)
    player_lists: List[torch.Tensor] = [
        torch.tensor(s.player_indices, dtype=torch.long) for s in samples
    ]
    return q_idx, player_lists, taken


# --- Synthetic data ---


def generate_synthetic(
    num_players: int = 50,
    num_questions: int = 200,
    num_teams_per_game: int = 12,
    players_per_team: int = 6,
    num_games: int = 30,
    seed: int = 42,
) -> Tuple[list[Sample], IndexMaps]:
    """
    Generate synthetic (game, question, team, taken) data from known θ, b, a.
    Each game has num_questions questions and num_teams_per_game teams; each team has players_per_team players.
    """
    rng = np.random.default_rng(seed)
    # True parameters
    theta = rng.standard_normal(num_players).astype(np.float32)
    theta = theta - theta.mean()
    b = rng.standard_normal(num_questions).astype(np.float32)
    b = b - b.mean()
    log_a = np.log(0.5 + rng.exponential(0.5, num_questions).astype(np.float32))

    samples: list[Sample] = []
    # We'll create games: each game has several teams; each team is a random subset of players
    player_indices = list(range(num_players))
    for g in range(num_games):
        # Assign teams: each team = random subset of players (with replacement across games for simplicity)
        team_rosters: List[List[int]] = []
        for _ in range(num_teams_per_game):
            team_rosters.append(rng.choice(player_indices, size=players_per_team, replace=False).tolist())
        for qi in range(num_questions):
            a_i = np.exp(log_a[qi])
            b_i = b[qi]
            for t, roster in enumerate(team_rosters):
                lam_sum = 0.0
                for k in roster:
                    lam_sum += np.exp(-b_i + a_i * theta[k])
                p = 1.0 - np.exp(-lam_sum)
                p = np.clip(p, 1e-6, 1.0 - 1e-6)
                taken = 1 if rng.random() < p else 0
                samples.append(
                    Sample(question_idx=qi, player_indices=roster.copy(), taken=taken)
                )

    maps = IndexMaps(
        player_id_to_idx={i: i for i in range(num_players)},
        question_id_to_idx={i: i for i in range(num_questions)},
        idx_to_player_id=list(range(num_players)),
        idx_to_question_id=list(range(num_questions)),
    )
    return samples, maps


# --- DB loader (rating backup) ---


def load_from_db(
    database_url: Optional[str] = None,
    max_tournaments: Optional[int] = None,
    min_questions: int = 10,
    only_with_question_data: bool = True,
    only_tournaments_with_true_dl: bool = False,
    min_tournament_date: Optional[str] = "2015-01-01",
    min_games: int = 10,
    show_progress: bool = True,
    seed: int = 42,
) -> Tuple[list[Sample], IndexMaps]:
    """
    Load (game_id, question_id, team_id, taken) from rating DB.

    - By default uses all tournaments (or up to max_tournaments) that have both
      questions_count >= min_questions and at least one result row with non-null
      points_mask (i.e. question-level outcomes). Set only_with_question_data=False
      to include every tournament with enough questions (then teams without
      points_mask simply contribute no samples).
    - min_tournament_date: only tournaments with start_datetime >= this date (YYYY-MM-DD).
      None = no date filter. Default 2015-01-01.
    - When only_tournaments_with_true_dl=True, only tournaments that have
      at least one row in true_dls are used. Default False: include all (tournament_dl from true_dl + ndcg fallback).
    - min_games: players with fewer games (total in DB) are excluded entirely; only teams
      where all players have >= min_games are loaded. Default 10.
    - Results: only rows where points_mask IS NOT NULL; teams/tournaments
      without question-level data never contribute samples.
    - Maps: tournament_id -> game, (tournament_id, question_index) -> global question index.
    """
    try:
        import psycopg2
        from psycopg2 import errors as pg_errors
    except ImportError:
        raise ImportError("psycopg2-binary is required for DB load. pip install psycopg2-binary")

    url = database_url or os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
    conn = psycopg2.connect(url)
    cur = conn.cursor()

    # Ensure rating backup has been restored (public.tournaments must exist)
    try:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'tournaments'
        """)
        if cur.fetchone() is None:
            conn.close()
            raise ValueError(
                "Table public.tournaments not found. Restore the rating backup first: "
                "in the rating-db folder run 'docker-compose down -v', then put rating.backup in place and run 'docker-compose up'."
            )
    except Exception as e:
        conn.close()
        raise

    # Tournaments: enough questions, (optionally) points_mask, and start_datetime >= min_tournament_date
    date_cond = ""
    date_param: tuple = ()
    if min_tournament_date:
        date_cond = " AND (t.start_datetime IS NULL OR t.start_datetime >= %s::timestamp)"
        date_param = (min_tournament_date,)
    try:
        if only_with_question_data:
            cur.execute(
                """
                SELECT t.id, COALESCE(t.questions_count, 0) AS qcount
                FROM public.tournaments t
                WHERE COALESCE(t.questions_count, 0) >= %s
                  AND EXISTS (
                    SELECT 1 FROM public.tournament_results r
                    WHERE r.tournament_id = t.id AND r.points_mask IS NOT NULL
                  )
                  """
                + date_cond
                + """
                ORDER BY t.id
                """,
                (min_questions,) + date_param,
            )
        else:
            cur.execute(
                """
                SELECT t.id, COALESCE(t.questions_count, 0) AS qcount
                FROM public.tournaments t
                WHERE COALESCE(t.questions_count, 0) >= %s
                """
                + date_cond
                + """
                ORDER BY t.id
                """,
                (min_questions,) + date_param,
            )
    except pg_errors.UndefinedTable as e:
        conn.close()
        raise ValueError(
            "Rating schema missing (e.g. public.tournaments). Restore the backup: "
            "in rating-db run 'docker-compose down -v' then 'docker-compose up' with rating.backup in the same folder."
        ) from e
    rows = cur.fetchall()
    # Optionally restrict to tournaments that have true_dl (exclude earlier ones without difficulty data)
    if only_tournaments_with_true_dl:
        try:
            cur.execute(
                "SELECT DISTINCT tournament_id FROM public.true_dls WHERE tournament_id IS NOT NULL"
            )
            ids_with_dl = {r[0] for r in cur.fetchall()}
            if ids_with_dl:
                n_before = len(rows)
                rows = [r for r in rows if r[0] in ids_with_dl]
                if rows and n_before > len(rows):
                    min_id = min(r[0] for r in rows)
                    print(
                        f"Restricted to {len(rows)} tournaments with true_dl (earliest id={min_id}, excluded {n_before - len(rows)} without true_dl)"
                    )
        except Exception:
            pass
    if max_tournaments:
        rows = rows[:max_tournaments]
    tournament_ids = [r[0] for r in rows if r[0] is not None]
    t_to_questions: dict[int, int] = {r[0]: r[1] for r in rows if r[0] is not None}

    if not tournament_ids:
        conn.close()
        raise ValueError("No tournaments with enough questions found.")

    # Global question index: (tournament_id, question_index) -> idx
    question_keys: list[tuple[int, int]] = []
    for tid in tournament_ids:
        nq = t_to_questions.get(tid, 0)
        for qi in range(nq):
            question_keys.append((tid, qi))
    question_id_to_idx = {qk: i for i, qk in enumerate(question_keys)}
    num_questions = len(question_keys)

    # Tournament difficulty (true_dl) and optional fallback (ndcg) when true_dl missing
    tournament_dl: Optional[np.ndarray] = None
    try:
        cur.execute("""
            SELECT tournament_id, AVG(true_dl)::double precision
            FROM public.true_dls
            WHERE tournament_id = ANY(%s)
            GROUP BY tournament_id
        """, (tournament_ids,))
        dl_map = {r[0]: float(r[1]) for r in cur.fetchall()}
        missing_tids = [tid for tid, _ in question_keys if tid not in dl_map]
        if missing_tids:
            try:
                cur.execute("""
                    SELECT tournament_id, ndcg::double precision
                    FROM public.ndcg
                    WHERE tournament_id = ANY(%s)
                """, (missing_tids,))
                for r in cur.fetchall():
                    if r[0] not in dl_map:
                        dl_map[r[0]] = float(r[1])
            except Exception:
                pass
        tournament_dl = np.array(
            [dl_map.get(tid, 0.0) for tid, _ in question_keys],
            dtype=np.float32,
        )
    except Exception:
        tournament_dl = None

    # Tournament type (0=Очник, 1=Синхрон, 2=Асинхрон) per tournament for type-dependent dl scale
    tournament_type: Optional[np.ndarray] = None
    try:
        cur.execute(
            "SELECT id, COALESCE(LOWER(type), '') FROM public.tournaments WHERE id = ANY(%s)",
            (tournament_ids,),
        )
        tid_to_type: dict[int, str] = {r[0]: (r[1] or "").strip() for r in cur.fetchall()}

        def type_idx(tid: int) -> int:
            t = tid_to_type.get(tid, "")
            if "асинхрон" in t or "async" in t:
                return 2
            if "синхрон" in t or "sync" in t:
                return 1
            return 0  # очник / очный / иное

        tournament_type = np.array(
            [type_idx(tid) for tid, _ in question_keys],
            dtype=np.int32,
        )
    except Exception:
        tournament_type = None

    # Rosters: (tournament_id, team_id) -> list of player_id (raw)
    cur.execute("""
        SELECT tournament_id, team_id, player_id
        FROM public.tournament_rosters
        WHERE tournament_id = ANY(%s) AND team_id IS NOT NULL AND player_id IS NOT NULL
    """, (tournament_ids,))
    roster_rows = cur.fetchall()
    roster_raw: dict[tuple[int, int], list[int]] = {}
    for tid, team_id, pid in roster_rows:
        key = (tid, team_id)
        if key not in roster_raw:
            roster_raw[key] = []
        roster_raw[key].append(pid)

    # Players with >= min_games (total in DB) — exclude everyone else entirely
    all_pids = set()
    for pids in roster_raw.values():
        all_pids.update(pids)
    if all_pids:
        cur.execute("""
            SELECT player_id, COUNT(DISTINCT tournament_id)
            FROM public.tournament_rosters
            WHERE player_id = ANY(%s)
            GROUP BY player_id
        """, (list(all_pids),))
        player_games = {r[0]: r[1] for r in cur.fetchall()}
    else:
        player_games = {}
    active_players = {pid for pid, g in player_games.items() if g >= min_games}
    if min_games > 0 and active_players != all_pids:
        n_excl = len(all_pids - active_players)
        print(f"Excluded {n_excl} players with <{min_games} games (only loading teams of players with >={min_games})")

    # Keep only teams where all players have >= min_games
    roster_map_filtered: dict[tuple[int, int], list[int]] = {}
    for key, pids in roster_raw.items():
        if not pids or (min_games > 0 and not all(p in active_players for p in pids)):
            continue
        roster_map_filtered[key] = pids

    player_ids = sorted(set().union(*(roster_map_filtered[k] for k in roster_map_filtered)))
    player_id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    num_players = len(player_ids)

    roster_map: dict[tuple[int, int], list[int]] = {}
    for key, pids in roster_map_filtered.items():
        roster_map[key] = [player_id_to_idx[p] for p in pids]

    # Results: tournament_id, team_id, points_mask (only for teams we kept)
    cur.execute("""
        SELECT tournament_id, team_id, points_mask
        FROM public.tournament_results
        WHERE tournament_id = ANY(%s) AND team_id IS NOT NULL AND points_mask IS NOT NULL
    """, (tournament_ids,))
    result_rows = [r for r in cur.fetchall() if (r[0], r[1]) in roster_map]

    # Team strength from tournament place: score = sum(taken), rank by score desc, strength = (n - rank + 1) / n
    team_scores: list[tuple[int, int, int]] = []
    for tid, team_id, points_mask in result_rows:
        key = (tid, team_id)
        if key not in roster_map or not roster_map[key]:
            continue
        mask = points_mask.strip()
        score = sum(1 for c in mask if c == "1")
        team_scores.append((tid, team_id, score))
    from collections import defaultdict
    team_strength_map: dict[tuple[int, int], float] = {}
    by_tournament: dict[int, list[tuple[int, int]]] = defaultdict(list)  # tid -> [(team_id, score), ...]
    for tid, team_id, score in team_scores:
        by_tournament[tid].append((team_id, score))
    for tid, team_list in by_tournament.items():
        team_list.sort(key=lambda x: -x[1])  # desc by score
        n_teams = len(team_list)
        for rank, (team_id, _) in enumerate(team_list, start=1):
            # strength: 1st = 1, last = 1/n
            team_strength_map[(tid, team_id)] = (n_teams - rank + 1) / n_teams

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore
    iterator = tqdm(result_rows, desc="Building samples", unit=" teams") if (show_progress and tqdm) else result_rows

    samples: list[Sample] = []
    for tid, team_id, points_mask in iterator:
        key = (tid, team_id)
        if key not in roster_map or not roster_map[key]:
            continue
        player_indices = roster_map[key]
        strength = team_strength_map.get(key, 0.5)
        mask = points_mask.strip()
        nq = min(len(mask), t_to_questions.get(tid, len(mask)))
        for qi in range(nq):
            if qi >= len(mask):
                break
            taken = 1 if mask[qi] == "1" else 0
            q_global = question_id_to_idx.get((tid, qi))
            if q_global is None:
                continue
            samples.append(
                Sample(question_idx=q_global, player_indices=player_indices.copy(), taken=taken, team_strength=strength)
            )

    conn.close()

    maps = IndexMaps(
        player_id_to_idx=player_id_to_idx,
        question_id_to_idx=question_id_to_idx,
        idx_to_player_id=player_ids,
        idx_to_question_id=question_keys,
        tournament_dl=tournament_dl,
        tournament_type=tournament_type,
    )

    if seed is not None:
        random.seed(seed)
        random.shuffle(samples)

    return samples, maps


def samples_to_arrays(samples: list[Sample]) -> dict[str, np.ndarray]:
    """Build packed arrays from samples (same structure as load_cached returns)."""
    q_idx = np.array([s.question_idx for s in samples], dtype=np.int32)
    taken = np.array([s.taken for s in samples], dtype=np.float32)
    team_sizes = np.array([len(s.player_indices) for s in samples], dtype=np.int32)
    player_indices_flat = np.concatenate([s.player_indices for s in samples]).astype(np.int32)
    out = {
        "q_idx": q_idx,
        "taken": taken,
        "team_sizes": team_sizes,
        "player_indices_flat": player_indices_flat,
    }
    if samples and getattr(samples[0], "team_strength", None) is not None:
        out["team_strength"] = np.array([getattr(s, "team_strength", 0.5) for s in samples], dtype=np.float32)
    return out


def train_val_split(
    samples: list[Sample],
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[list[Sample], list[Sample]]:
    """Random split into train and validation."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(samples) * val_frac))
    val_idx = set(indices[:n_val])
    train = [samples[i] for i in indices[n_val:]]
    val = [samples[i] for i in indices[:n_val]]
    return train, val


# --- Cache (save/load extracted data) ---

CACHE_VERSION = 2


def save_cached(
    samples: list[Sample],
    maps: IndexMaps,
    path: str | Path,
    *,
    meta: Optional[dict] = None,
) -> None:
    """Save (samples, index maps) to a single file using NumPy for speed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to packed NumPy arrays for efficient storage and loading
    q_idx = np.array([s.question_idx for s in samples], dtype=np.int32)
    taken = np.array([s.taken for s in samples], dtype=np.float32)
    team_sizes = np.array([len(s.player_indices) for s in samples], dtype=np.int32)
    player_indices_flat = np.concatenate([s.player_indices for s in samples]).astype(np.int32)

    payload = {
        "version": CACHE_VERSION,
        "q_idx": q_idx,
        "taken": taken,
        "team_sizes": team_sizes,
        "player_indices_flat": player_indices_flat,
        "idx_to_player_id": maps.idx_to_player_id,
        "idx_to_question_id": maps.idx_to_question_id,
        "meta": meta or {},
    }
    if samples and getattr(samples[0], "team_strength", None) is not None:
        payload["team_strength"] = np.array([getattr(s, "team_strength", 0.5) for s in samples], dtype=np.float32)
    if getattr(maps, "tournament_dl", None) is not None:
        payload["tournament_dl"] = maps.tournament_dl
    if getattr(maps, "tournament_type", None) is not None:
        payload["tournament_type"] = maps.tournament_type
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cached(path: str | Path) -> Tuple[dict[str, np.ndarray], IndexMaps]:
    """Load packed arrays and index maps from a cache file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cache file not found: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    version = payload.get("version")
    if version != CACHE_VERSION:
        raise ValueError(
            f"Cache version mismatch: file has {version}, expected {CACHE_VERSION}. "
            "Please delete the cache file and re-run to regenerate in the new fast format."
        )

    idx_to_player_id = payload["idx_to_player_id"]
    idx_to_question_id = payload["idx_to_question_id"]
    tournament_dl = payload.get("tournament_dl")
    tournament_type = payload.get("tournament_type")
    maps = IndexMaps(
        player_id_to_idx={pid: i for i, pid in enumerate(idx_to_player_id)},
        question_id_to_idx={qid: i for i, qid in enumerate(idx_to_question_id)},
        idx_to_player_id=idx_to_player_id,
        idx_to_question_id=idx_to_question_id,
        tournament_dl=tournament_dl,
        tournament_type=tournament_type,
    )

    arrays = {
        "q_idx": payload["q_idx"],
        "taken": payload["taken"],
        "team_sizes": payload["team_sizes"],
        "player_indices_flat": payload["player_indices_flat"],
    }
    if "team_strength" in payload:
        arrays["team_strength"] = payload["team_strength"]
    return arrays, maps
