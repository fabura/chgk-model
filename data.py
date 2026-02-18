"""
ChGK data loader: index maps, batched (game, question, team) → player indices + taken.
Supports synthetic data and DB load from rating backup (tournament_results + points_mask, tournament_rosters).
Cache: save/load (samples, index maps) to avoid re-querying the DB.
"""
from __future__ import annotations

import os
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date
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
    # Optional: per-question game index and game metadata (for weighting/reporting).
    question_game_idx: Optional[np.ndarray] = None  # shape (num_questions,), int32
    idx_to_game_id: list[int] = field(default_factory=list)
    game_type: Optional[np.ndarray] = None  # shape (num_games,), object[str]
    game_date_ordinal: Optional[np.ndarray] = None  # shape (num_games,), int32; -1 = unknown date
    # Canonical question index: questions from paired tournaments share the same canonical idx.
    canonical_q_idx: Optional[np.ndarray] = None  # shape (num_questions,), int32
    num_canonical_questions: Optional[int] = None

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
    game_idx: Optional[int] = None


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
    idx_to_game_id = list(range(num_games))
    question_game_idx = np.zeros(num_questions, dtype=np.int32)
    for g in range(num_games):
        # Assign teams: each team = random subset of players (with replacement across games for simplicity)
        team_rosters: List[List[int]] = []
        for _ in range(num_teams_per_game):
            team_rosters.append(rng.choice(player_indices, size=players_per_team, replace=False).tolist())
        for qi in range(num_questions):
            question_game_idx[qi] = g
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
                    Sample(question_idx=qi, player_indices=roster.copy(), taken=taken, game_idx=g)
                )

    maps = IndexMaps(
        player_id_to_idx={i: i for i in range(num_players)},
        question_id_to_idx={i: i for i in range(num_questions)},
        idx_to_player_id=list(range(num_players)),
        idx_to_question_id=list(range(num_questions)),
        question_game_idx=question_game_idx,
        idx_to_game_id=idx_to_game_id,
        game_type=np.array(["offline"] * num_games, dtype=object),
        game_date_ordinal=np.array([-1] * num_games, dtype=np.int32),
    )
    return samples, maps


def generate_synthetic_two_populations(
    num_players: int = 200,
    questions_per_game: int = 36,
    num_games: int = 40,
    num_teams_per_game: int = 20,
    players_per_team: int = 6,
    mix_games: int = 4,
    seed: int = 42,
) -> Tuple[list[Sample], IndexMaps]:
    """
    Synthetic generator with two weakly connected populations + a few mixing games.
    """
    rng = np.random.default_rng(seed)
    half = num_players // 2
    pop_a = np.arange(0, half, dtype=np.int32)
    pop_b = np.arange(half, num_players, dtype=np.int32)

    theta = rng.normal(0.0, 1.0, size=num_players).astype(np.float32)
    theta = theta - theta.mean()
    # Rookies in pop_b with inflated variance to stress uncertainty ranking.
    rookie_idx = rng.choice(pop_b, size=max(1, len(pop_b) // 8), replace=False)
    theta[rookie_idx] += rng.normal(0.8, 0.4, size=rookie_idx.shape[0]).astype(np.float32)

    n_questions = num_games * questions_per_game
    b = rng.normal(0.0, 1.0, size=n_questions).astype(np.float32)
    b = b - b.mean()
    log_a = np.log(0.5 + rng.exponential(0.5, n_questions).astype(np.float32))

    game_ids = list(range(num_games))
    game_type = np.array(["offline"] * num_games, dtype=object)
    game_date_ordinal = np.array([730120 + g for g in range(num_games)], dtype=np.int32)
    question_game_idx = np.zeros(n_questions, dtype=np.int32)

    # Place mixing games near the end.
    mix_game_ids = set(game_ids[-mix_games:])

    samples: list[Sample] = []
    q_global = 0
    for g in range(num_games):
        team_rosters: List[List[int]] = []
        if g in mix_game_ids:
            game_type[g] = "sync"
            for _ in range(num_teams_per_game):
                n_a = rng.integers(1, players_per_team)
                n_b = players_per_team - n_a
                ta = rng.choice(pop_a, size=n_a, replace=False).tolist()
                tb = rng.choice(pop_b, size=n_b, replace=False).tolist()
                roster = ta + tb
                rng.shuffle(roster)
                team_rosters.append(roster)
        else:
            base = pop_a if g % 2 == 0 else pop_b
            game_type[g] = "async" if g % 5 == 0 else "offline"
            for _ in range(num_teams_per_game):
                team_rosters.append(rng.choice(base, size=players_per_team, replace=False).tolist())

        for _ in range(questions_per_game):
            question_game_idx[q_global] = g
            a_i = float(np.exp(log_a[q_global]))
            b_i = float(b[q_global])
            for roster in team_rosters:
                lam_sum = 0.0
                for k in roster:
                    lam_sum += np.exp(-b_i + a_i * float(theta[k]))
                p = 1.0 - np.exp(-lam_sum)
                p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
                taken = 1 if rng.random() < p else 0
                samples.append(
                    Sample(
                        question_idx=q_global,
                        player_indices=roster.copy(),
                        taken=taken,
                        team_strength=None,
                        game_idx=g,
                    )
                )
            q_global += 1

    maps = IndexMaps(
        player_id_to_idx={i: i for i in range(num_players)},
        question_id_to_idx={i: i for i in range(n_questions)},
        idx_to_player_id=list(range(num_players)),
        idx_to_question_id=list(range(n_questions)),
        question_game_idx=question_game_idx,
        idx_to_game_id=game_ids,
        game_type=game_type,
        game_date_ordinal=game_date_ordinal,
    )
    return samples, maps


# --- Paired tournament detection ---

_FORMAT_RE = re.compile(
    r'\s*\([^)]*(?:синхрон|асинхрон|онлайн|очн|заочн|offline|online)[^)]*\)\s*$',
    re.IGNORECASE,
)
_TRAILING_EDITION_RE = re.compile(r'\s+(?:\d+|[IVXLCDM]+)\.?\s*$')


def _normalize_tournament_title(title: str, strip_edition: bool = False) -> str:
    """Strip format markers, optionally trailing edition numbers, lowercase, collapse whitespace."""
    t = _FORMAT_RE.sub('', title)
    if strip_edition:
        t = _TRAILING_EDITION_RE.sub('', t)
    return ' '.join(t.lower().split())


@dataclass
class TournamentPairGroup:
    tournament_ids: list[int]
    titles: list[str]
    types: list[str]
    questions_count: int
    editors: list[frozenset[int]]
    dates: list[Optional[_date]]
    editors_match: bool
    date_gap_days: Optional[int]


def detect_paired_tournaments(
    cur,
    tournament_ids: list[int],
    *,
    max_date_gap_days: int = 60,
    verbose: bool = True,
) -> list[TournamentPairGroup]:
    """
    Detect tournaments sharing the same question package (sync+async of the same event).
    Returns pairs of linked tournament IDs (each pair = one group).
    Uses pairwise matching within name/qcount buckets to avoid false cross-season merges.
    """
    if not tournament_ids:
        return []

    cur.execute(
        """
        SELECT id, title, COALESCE(LOWER(type), ''), COALESCE(questions_count, 0),
               start_datetime::date, end_datetime::date
        FROM public.tournaments WHERE id = ANY(%s)
        """,
        (tournament_ids,),
    )
    meta = {}
    for row in cur.fetchall():
        tid, title, ttype, qcount, start_dt, end_dt = row
        if tid is None or not title:
            continue
        meta[int(tid)] = {
            "title": title.strip(),
            "type": ttype.strip(),
            "qcount": int(qcount),
            "start": start_dt,
            "end": end_dt,
        }

    cur.execute(
        """
        SELECT tournament_id, array_agg(player_id ORDER BY player_id)
        FROM public.tournament_editors
        WHERE tournament_id = ANY(%s)
        GROUP BY tournament_id
        """,
        (tournament_ids,),
    )
    editors_map: dict[int, frozenset[int]] = {}
    for tid, pids in cur.fetchall():
        if tid is not None and pids:
            editors_map[int(tid)] = frozenset(int(p) for p in pids if p is not None)

    def _normalize(title: str) -> str:
        return _normalize_tournament_title(title, strip_edition=False)

    def _normalize_fuzzy(title: str) -> str:
        return _normalize_tournament_title(title, strip_edition=True)

    _SYNC_TYPES = {"синхрон", "sync", "строго синхронный"}
    _ASYNC_TYPES = {"асинхрон", "async"}

    def _try_pair(candidates: list[int]) -> list[TournamentPairGroup]:
        """Find actual pairs within a bucket of same-name same-qcount tournaments."""
        if len(candidates) < 2:
            return []
        by_type: dict[str, list[int]] = defaultdict(list)
        for t in candidates:
            by_type[meta[t]["type"]].append(t)
        types_present = set(by_type.keys())
        if len(types_present) < 2:
            return []

        sync_tids = [t for typ in _SYNC_TYPES for t in by_type.get(typ, [])]
        async_tids = [t for typ in _ASYNC_TYPES for t in by_type.get(typ, [])]
        offline_tids = [t for t in candidates if meta[t]["type"] not in
                        _SYNC_TYPES | _ASYNC_TYPES]

        source_pool = sync_tids + offline_tids
        target_pool = async_tids
        if not source_pool or not target_pool:
            source_pool = sorted(candidates, key=lambda t: meta[t]["start"] or _date.min)
            target_pool = source_pool[1:]
            source_pool = source_pool[:1]

        used: set[int] = set()
        pairs: list[TournamentPairGroup] = []
        for src in sorted(source_pool, key=lambda t: meta[t]["start"] or _date.min):
            if src in used:
                continue
            best_target = None
            best_score = -1
            src_eds = editors_map.get(src, frozenset())
            src_start = meta[src]["start"]
            for tgt in target_pool:
                if tgt in used or tgt == src:
                    continue
                if meta[tgt]["type"] == meta[src]["type"]:
                    continue
                tgt_eds = editors_map.get(tgt, frozenset())
                score = 0
                if src_eds and tgt_eds and src_eds == tgt_eds:
                    score += 100
                tgt_start = meta[tgt]["start"]
                if src_start and tgt_start:
                    gap = abs((tgt_start - src_start).days)
                    if gap <= max_date_gap_days:
                        score += max(0, max_date_gap_days - gap)
                    else:
                        if score < 100:
                            continue
                if score > best_score:
                    best_score = score
                    best_target = tgt
            if best_target is not None and best_score > 0:
                used.add(src)
                used.add(best_target)
                tids_pair = [src, best_target]
                eds = [editors_map.get(t, frozenset()) for t in tids_pair]
                non_empty_eds = [e for e in eds if e]
                editors_match = len(non_empty_eds) >= 2 and len(set(non_empty_eds)) == 1
                dates_pair = [meta[t]["start"] for t in tids_pair]
                date_gap = None
                if all(d is not None for d in dates_pair):
                    date_gap = abs((dates_pair[1] - dates_pair[0]).days)
                pairs.append(TournamentPairGroup(
                    tournament_ids=tids_pair,
                    titles=[meta[t]["title"] for t in tids_pair],
                    types=[meta[t]["type"] for t in tids_pair],
                    questions_count=meta[tids_pair[0]]["qcount"],
                    editors=eds,
                    dates=dates_pair,
                    editors_match=editors_match,
                    date_gap_days=date_gap,
                ))
        return pairs

    # Pass 1: exact name match (after stripping format markers only)
    buckets_exact: dict[tuple[str, int], list[int]] = defaultdict(list)
    for tid, info in meta.items():
        buckets_exact[(_normalize(info["title"]), info["qcount"])].append(tid)

    groups: list[TournamentPairGroup] = []
    paired_tids: set[int] = set()
    for (norm_title, qcount), tids in sorted(buckets_exact.items()):
        if len(tids) < 2:
            continue
        found = _try_pair(tids)
        groups.extend(found)
        for g in found:
            paired_tids.update(g.tournament_ids)

    # Pass 2: fuzzy match (also strip trailing edition numbers) for unpaired tournaments
    unpaired = [tid for tid in meta if tid not in paired_tids]
    if unpaired:
        buckets_fuzzy: dict[tuple[str, int], list[int]] = defaultdict(list)
        for tid in unpaired:
            info = meta[tid]
            buckets_fuzzy[(_normalize_fuzzy(info["title"]), info["qcount"])].append(tid)
        for (norm_title, qcount), tids in sorted(buckets_fuzzy.items()):
            if len(tids) < 2:
                continue
            found = _try_pair(tids)
            for g in found:
                if g.editors_match:
                    groups.append(g)
                    paired_tids.update(g.tournament_ids)

    if verbose:
        total_linked = sum(len(g.tournament_ids) for g in groups)
        total_questions_shared = sum(g.questions_count * (len(g.tournament_ids) - 1) for g in groups)
        n_with_ed = sum(1 for g in groups if g.editors_match)
        print(f"\nPaired tournament detection: {len(groups)} pairs, "
              f"{total_linked} tournaments linked, "
              f"{total_questions_shared} question slots shared "
              f"({n_with_ed} confirmed by same editors, "
              f"{len(groups) - n_with_ed} by name+date only)")
        n_show = min(5, len(groups))
        for g in groups[:n_show]:
            t1, t2 = g.tournament_ids
            print(f"  {meta[t1]['title']} [{meta[t1]['type']}] <-> {meta[t2]['title']} [{meta[t2]['type']}]")

    return groups


def build_canonical_question_idx(
    question_keys: list[tuple[int, int]],
    groups: list[TournamentPairGroup],
) -> tuple[np.ndarray, int]:
    """
    Build canonical question index array.
    Questions at the same position in linked tournaments map to the same canonical index.
    Returns (canonical_q_idx, num_canonical_questions).
    """
    tid_to_canonical_tid: dict[int, int] = {}
    for g in groups:
        primary = min(g.tournament_ids)
        for tid in g.tournament_ids:
            tid_to_canonical_tid[tid] = primary

    canonical_key_to_idx: dict[tuple[int, int], int] = {}
    canonical_q_idx = np.empty(len(question_keys), dtype=np.int32)
    next_idx = 0

    for i, (tid, qi) in enumerate(question_keys):
        ctid = tid_to_canonical_tid.get(tid, tid)
        ckey = (ctid, qi)
        if ckey not in canonical_key_to_idx:
            canonical_key_to_idx[ckey] = next_idx
            next_idx += 1
        canonical_q_idx[i] = canonical_key_to_idx[ckey]

    return canonical_q_idx, next_idx


# --- DB loader (rating backup) ---


def load_from_db(
    database_url: Optional[str] = None,
    max_tournaments: Optional[int] = None,
    min_questions: int = 10,
    only_with_question_data: bool = True,
    tournament_dl_filter: str = "all",
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
    - tournament_dl_filter controls which tournaments are used by availability of
      difficulty sources:
      * "all" (default): no filter by difficulty source.
      * "true_dl": only tournaments present in true_dls.
      * "ndcg": only tournaments present in ndcg.
      * "any": tournaments present in true_dls or ndcg.
      * "both": tournaments present in both true_dls and ndcg.
    - min_games: players with fewer games (total in DB) are treated as inactive and removed
      from rosters; teams are dropped only if roster becomes empty. Default 10.
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
    # Optional restrict by availability of tournament difficulty sources.
    dl_filter = tournament_dl_filter
    if dl_filter not in {"all", "true_dl", "ndcg", "any", "both"}:
        conn.close()
        raise ValueError(
            f"Unsupported tournament_dl_filter={dl_filter!r}. Use one of: all, true_dl, ndcg, any, both."
        )
    if dl_filter != "all":
        try:
            true_ids: set[int] = set()
            ndcg_ids: set[int] = set()
            if dl_filter in {"true_dl", "any", "both"}:
                cur.execute(
                    "SELECT DISTINCT tournament_id FROM public.true_dls WHERE tournament_id IS NOT NULL"
                )
                true_ids = {int(r[0]) for r in cur.fetchall()}
            if dl_filter in {"ndcg", "any", "both"}:
                cur.execute(
                    "SELECT DISTINCT tournament_id FROM public.ndcg WHERE tournament_id IS NOT NULL"
                )
                ndcg_ids = {int(r[0]) for r in cur.fetchall()}

            if dl_filter == "true_dl":
                allowed_ids = true_ids
            elif dl_filter == "ndcg":
                allowed_ids = ndcg_ids
            elif dl_filter == "any":
                allowed_ids = true_ids | ndcg_ids
            else:  # both
                allowed_ids = true_ids & ndcg_ids

            n_before = len(rows)
            rows = [r for r in rows if int(r[0]) in allowed_ids]
            excluded = n_before - len(rows)
            print(
                f"Applied tournament_dl_filter={dl_filter}: kept {len(rows)} tournaments, excluded {excluded} without required difficulty source(s)"
            )
        except Exception:
            pass
    if max_tournaments and len(rows) > max_tournaments:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_tournaments)
        rows.sort(key=lambda r: r[0])
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

    # Detect paired tournaments (sync+async of same package) and build canonical question index
    pair_groups = detect_paired_tournaments(cur, tournament_ids, verbose=True)
    canonical_q_idx, num_canonical = build_canonical_question_idx(question_keys, pair_groups)
    if num_canonical < num_questions:
        print(f"Canonical questions: {num_canonical} (shared {num_questions - num_canonical} across paired tournaments)")
    else:
        canonical_q_idx = None
        num_canonical = num_questions

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

    # Players with >= min_games (total in DB) are active; low-game players are removed from rosters.
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
    if min_games > 0:
        active_players = {pid for pid in all_pids if player_games.get(pid, 0) >= min_games}
    else:
        active_players = set(all_pids)
    if min_games > 0 and active_players != all_pids:
        n_excl = len(all_pids - active_players)
        print(f"Marked {n_excl} players with <{min_games} games as inactive (they will be removed from rosters)")

    # Keep teams, but remove inactive players from rosters.
    # Drop team only if it becomes empty after filtering.
    roster_map_filtered: dict[tuple[int, int], list[int]] = {}
    n_teams_dropped_empty = 0
    for key, pids in roster_raw.items():
        if not pids:
            continue
        filtered = [p for p in pids if p in active_players] if min_games > 0 else pids
        if not filtered:
            n_teams_dropped_empty += 1
            continue
        roster_map_filtered[key] = filtered
    if min_games > 0:
        print(f"Dropped {n_teams_dropped_empty} teams with empty roster after removing inactive players")

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

    # Tournament metadata used by weighting/reporting.
    cur.execute(
        """
        SELECT id, COALESCE(LOWER(type), ''), start_datetime::date
        FROM public.tournaments
        WHERE id = ANY(%s)
        """,
        (tournament_ids,),
    )
    tmeta = {int(r[0]): ((r[1] or "").strip(), r[2]) for r in cur.fetchall()}
    tid_to_game_idx = {tid: i for i, tid in enumerate(tournament_ids)}

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
                Sample(
                    question_idx=q_global,
                    player_indices=player_indices.copy(),
                    taken=taken,
                    team_strength=strength,
                    game_idx=tid_to_game_idx[tid],
                )
            )

    conn.close()

    def normalize_type(raw: str) -> str:
        if "асинхрон" in raw or "async" in raw:
            return "async"
        if "синхрон" in raw or "sync" in raw:
            return "sync"
        return "offline"

    idx_to_game_id = list(tournament_ids)
    game_type = np.array([normalize_type(tmeta.get(tid, ("", None))[0]) for tid in idx_to_game_id], dtype=object)
    game_date_ordinal = np.array(
        [int(dt.toordinal()) if dt is not None else -1 for _, dt in (tmeta.get(tid, ("", None)) for tid in idx_to_game_id)],
        dtype=np.int32,
    )
    question_game_idx = np.array([tid_to_game_idx[tid] for tid, _ in question_keys], dtype=np.int32)

    maps = IndexMaps(
        player_id_to_idx=player_id_to_idx,
        question_id_to_idx=question_id_to_idx,
        idx_to_player_id=player_ids,
        idx_to_question_id=question_keys,
        tournament_dl=tournament_dl,
        tournament_type=tournament_type,
        question_game_idx=question_game_idx,
        idx_to_game_id=idx_to_game_id,
        game_type=game_type,
        game_date_ordinal=game_date_ordinal,
        canonical_q_idx=canonical_q_idx,
        num_canonical_questions=num_canonical,
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
    if samples and getattr(samples[0], "game_idx", None) is not None:
        out["game_idx"] = np.array([getattr(s, "game_idx", -1) for s in samples], dtype=np.int32)
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

CACHE_VERSION = 4


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
    if samples and getattr(samples[0], "game_idx", None) is not None:
        payload["game_idx"] = np.array([getattr(s, "game_idx", -1) for s in samples], dtype=np.int32)
    if samples and getattr(samples[0], "team_strength", None) is not None:
        payload["team_strength"] = np.array([getattr(s, "team_strength", 0.5) for s in samples], dtype=np.float32)
    if getattr(maps, "tournament_dl", None) is not None:
        payload["tournament_dl"] = maps.tournament_dl
    if getattr(maps, "tournament_type", None) is not None:
        payload["tournament_type"] = maps.tournament_type
    if getattr(maps, "question_game_idx", None) is not None:
        payload["question_game_idx"] = maps.question_game_idx
    if getattr(maps, "idx_to_game_id", None):
        payload["idx_to_game_id"] = maps.idx_to_game_id
    if getattr(maps, "game_type", None) is not None:
        payload["game_type"] = maps.game_type
    if getattr(maps, "game_date_ordinal", None) is not None:
        payload["game_date_ordinal"] = maps.game_date_ordinal
    if getattr(maps, "canonical_q_idx", None) is not None:
        payload["canonical_q_idx"] = maps.canonical_q_idx
        payload["num_canonical_questions"] = maps.num_canonical_questions
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
    question_game_idx = payload.get("question_game_idx")
    idx_to_game_id = payload.get("idx_to_game_id", [])
    game_type = payload.get("game_type")
    game_date_ordinal = payload.get("game_date_ordinal")
    canonical_q_idx = payload.get("canonical_q_idx")
    num_canonical_questions = payload.get("num_canonical_questions")
    maps = IndexMaps(
        player_id_to_idx={pid: i for i, pid in enumerate(idx_to_player_id)},
        question_id_to_idx={qid: i for i, qid in enumerate(idx_to_question_id)},
        idx_to_player_id=idx_to_player_id,
        idx_to_question_id=idx_to_question_id,
        tournament_dl=tournament_dl,
        tournament_type=tournament_type,
        question_game_idx=question_game_idx,
        idx_to_game_id=list(idx_to_game_id),
        game_type=game_type,
        game_date_ordinal=game_date_ordinal,
        canonical_q_idx=canonical_q_idx,
        num_canonical_questions=num_canonical_questions,
    )

    arrays = {
        "q_idx": payload["q_idx"],
        "taken": payload["taken"],
        "team_sizes": payload["team_sizes"],
        "player_indices_flat": payload["player_indices_flat"],
    }
    if "team_strength" in payload:
        arrays["team_strength"] = payload["team_strength"]
    if "game_idx" in payload:
        arrays["game_idx"] = payload["game_idx"]
    return arrays, maps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ChGK data utilities")
    parser.add_argument("--detect_pairs", action="store_true",
                        help="Run paired tournament detection on DB data")
    parser.add_argument("--max_tournaments", type=int, default=None)
    parser.add_argument("--min_tournament_date", type=str, default="2015-01-01")
    args = parser.parse_args()

    if args.detect_pairs:
        import psycopg2
        url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        date_cond = ""
        date_param: tuple = ()
        if args.min_tournament_date:
            date_cond = " AND (t.start_datetime IS NULL OR t.start_datetime >= %s::timestamp)"
            date_param = (args.min_tournament_date,)
        cur.execute(
            """
            SELECT t.id FROM public.tournaments t
            WHERE COALESCE(t.questions_count, 0) >= 10
              AND EXISTS (
                SELECT 1 FROM public.tournament_results r
                WHERE r.tournament_id = t.id AND r.points_mask IS NOT NULL
              )
            """ + date_cond + " ORDER BY t.id",
            date_param or None,
        )
        tids = [r[0] for r in cur.fetchall() if r[0] is not None]
        if args.max_tournaments:
            tids = tids[:args.max_tournaments]
        print(f"Checking {len(tids)} tournaments for paired packages...")
        groups = detect_paired_tournaments(cur, tids, verbose=True)
        conn.close()
