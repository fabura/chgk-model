"""
Build chgk.duckdb for the website.

Inputs:
  --cache       path to data.npz       (observations + IndexMaps)
  --results     path to seq.npz        (theta, b, a, history)
  --questions-db path to questions.db  (texts, authors, editors)
  --out         output duckdb path
  --limit-tournaments N  (optional, for fast iteration)

The script builds a self-contained read-only DuckDB used by the FastAPI app.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running from repo root or website/build/.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached  # noqa: E402
from rating.io import load_results_npz  # noqa: E402


# ---------------------------------------------------------------------------
# Display-time inactivity decay
#
# The model deliberately does not decay θ over calendar time
# (`rho_calendar = 1.0`, see docs/calendar_decay_experiments.md):
# every bit of decay during training cost backtest accuracy.  But for
# a "current strength" board the raw θ leaves long-retired players at
# the very top (their last θ from years ago is still their score).
#
# We fix this at display time only, by precomputing a `theta_display`
# column that shrinks θ toward `DISPLAY_DECAY_PRIOR` once a player has
# been inactive for more than a `DISPLAY_DECAY_GRACE_DAYS` window.
# Decay is exponential with the configured half-life, but it MUST
# NEVER improve a player's standing — so for θ already below the
# prior we just keep θ unchanged (otherwise weak inactive players
# would be silently lifted toward the prior and could overtake weak
# active ones in the ranking):
#
#     factor        = 0.5 ** (max(0, days_inactive - grace) / halflife)
#     decayed       = prior + (θ - prior) * factor       # toward prior
#     theta_display = min(θ, decayed)                    # never make stronger
#
# Defaults: 1-year grace (so a normal off-season doesn't penalise),
# 4-year half-life (very slow), shrink toward 0 (the population centre
# after weekly within-type centering).  Concrete impact:
#
#   θ = +1.0, inactive…    θ = -1.0, inactive…
#     0–1y → +1.00            0–1y → -1.00
#     2y   → +0.84            2y   → -1.00  (clamped; would-be -0.84)
#     3y   → +0.71            3y   → -1.00  (clamped; would-be -0.71)
#     5y   → +0.50            5y   → -1.00  (clamped; would-be -0.50)
#    10y   → +0.21           10y   → -1.00  (clamped; would-be -0.21)
#
# In words: decay only matters for the upper tail (the "current top
# 1000" board); for everyone at or below the population centre it's a
# no-op.  This matches user intuition that "a player who hasn't
# played in years is at most as strong as their last estimate, never
# stronger than that, regardless of where on the scale they sit".
# ---------------------------------------------------------------------------
DISPLAY_DECAY_GRACE_DAYS = 365
DISPLAY_DECAY_HALFLIFE_DAYS = 4 * 365
DISPLAY_DECAY_PRIOR = 0.0


def compute_theta_display(
    theta: float,
    last_game_date: Optional[date],
    today: date,
    *,
    grace_days: int = DISPLAY_DECAY_GRACE_DAYS,
    halflife_days: int = DISPLAY_DECAY_HALFLIFE_DAYS,
    prior: float = DISPLAY_DECAY_PRIOR,
) -> float:
    """Return θ shrunk toward ``prior`` for inactive players, clamped so
    decay never makes anyone stronger (see module docstring)."""
    if last_game_date is None:
        return float(theta)
    days_inactive = (today - last_game_date).days
    if days_inactive <= grace_days:
        return float(theta)
    excess = days_inactive - grace_days
    factor = 0.5 ** (excess / halflife_days)
    decayed = prior + (float(theta) - prior) * factor
    # Clamp: decay must only push players *down* (further from the top
    # of the ranking).  For θ < prior the formula above would inflate
    # them toward the prior, which is the opposite of what we want.
    return float(min(float(theta), decayed))

try:
    import duckdb
except ImportError:
    raise SystemExit("duckdb is required: pip install duckdb")

try:
    import pyarrow as pa
except ImportError:
    raise SystemExit("pyarrow is required: pip install pyarrow")

try:
    import psycopg2
except ImportError:
    raise SystemExit("psycopg2-binary is required: pip install psycopg2-binary")


def _bulk_insert(con, table: str, columns: list[str], data: dict) -> None:
    """Bulk insert via Arrow registration. Much faster than executemany."""
    if not data or len(next(iter(data.values()))) == 0:
        return
    tbl = pa.table({k: data[k] for k in columns})
    name = f"_arrow_{table}"
    con.register(name, tbl)
    con.execute(f"INSERT INTO {table} ({','.join(columns)}) SELECT * FROM {name}")
    con.unregister(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[build_db] {msg}", flush=True)


def _ord_to_date(ord_int: int) -> Optional[date]:
    if ord_int is None or ord_int < 0:
        return None
    try:
        return date.fromordinal(int(ord_int))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 1. Load model artefacts
# ---------------------------------------------------------------------------


def load_model(cache_path: Path, results_path: Path):
    _log(f"Loading cache from {cache_path}…")
    arrays, maps = load_cached(cache_path)
    _log(
        f"  {len(maps.idx_to_player_id):,} players, "
        f"{len(maps.idx_to_question_id):,} question slots, "
        f"{len(maps.idx_to_game_id):,} tournaments, "
        f"{len(arrays['q_idx']):,} observations"
    )
    _log(f"Loading results from {results_path}…")
    res = load_results_npz(results_path)
    _log(
        f"  theta for {len(res.player_id):,} players, "
        f"{len(res.b):,} canonical questions, "
        f"history rows: {0 if res.history_player_id is None else len(res.history_player_id):,}"
    )
    return arrays, maps, res


# ---------------------------------------------------------------------------
# Step 2. Pull metadata from rating DB
# ---------------------------------------------------------------------------


def fetch_rating_db_metadata(
    tournament_ids: list[int],
    player_ids: list[int],
    database_url: Optional[str] = None,
):
    """Fetch player names, tournament metadata, rosters, and team results from rating DB."""
    url = database_url or os.environ.get(
        "DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres"
    )
    _log(f"Connecting to rating DB at {url.split('@')[-1]}…")
    conn = psycopg2.connect(url)
    cur = conn.cursor()

    _log(f"  fetching {len(player_ids):,} player names…")
    cur.execute(
        "SELECT id, first_name, last_name FROM public.players WHERE id = ANY(%s)",
        (player_ids,),
    )
    player_meta = {
        int(r[0]): {"first_name": (r[1] or "").strip(), "last_name": (r[2] or "").strip()}
        for r in cur.fetchall()
    }

    _log(f"  fetching {len(tournament_ids):,} tournament metadata rows…")
    cur.execute(
        """
        SELECT id, title, COALESCE(LOWER(type), ''), start_datetime::date,
               end_datetime::date, COALESCE(questions_count, 0)
        FROM public.tournaments WHERE id = ANY(%s)
        """,
        (tournament_ids,),
    )
    tournament_meta = {}
    for r in cur.fetchall():
        tid = int(r[0])
        raw_type = (r[2] or "").strip()
        if "асинхрон" in raw_type or "async" in raw_type:
            ttype = "async"
        elif "синхрон" in raw_type or "sync" in raw_type:
            ttype = "sync"
        else:
            ttype = "offline"
        tournament_meta[tid] = {
            "title": (r[1] or "").strip(),
            "type": ttype,
            "start_date": r[3],
            "end_date": r[4],
            "n_questions": int(r[5]),
        }

    _log("  fetching tournament_rosters…")
    cur.execute(
        """
        SELECT tournament_id, team_id, player_id
        FROM public.tournament_rosters
        WHERE tournament_id = ANY(%s) AND team_id IS NOT NULL AND player_id IS NOT NULL
        """,
        (tournament_ids,),
    )
    rosters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tid, team_id, pid in cur.fetchall():
        rosters[(int(tid), int(team_id))].append(int(pid))

    # Distinct team ids touched.
    team_ids = list({tid_team[1] for tid_team in rosters.keys()})
    _log(f"  rosters: {len(rosters):,} (tournament, team) pairs, {len(team_ids):,} distinct teams")

    _log("  fetching team names…")
    team_name: dict[int, str] = {}
    if team_ids:
        # Chunk to avoid huge IN.
        chunk = 5000
        for i in range(0, len(team_ids), chunk):
            cur.execute(
                "SELECT id, title FROM public.teams WHERE id = ANY(%s)",
                (team_ids[i : i + chunk],),
            )
            for r in cur.fetchall():
                team_name[int(r[0])] = (r[1] or "").strip()

    _log("  fetching tournament_results…")
    cur.execute(
        """
        SELECT tournament_id, team_id, points_mask, position
        FROM public.tournament_results
        WHERE tournament_id = ANY(%s) AND team_id IS NOT NULL AND points_mask IS NOT NULL
        """,
        (tournament_ids,),
    )
    results_rows = {}
    for tid, team_id, mask, position in cur.fetchall():
        results_rows[(int(tid), int(team_id))] = {
            "points_mask": (mask or "").strip(),
            "position": float(position) if position is not None else None,
        }

    conn.close()
    _log(f"  results: {len(results_rows):,} team rows")
    return player_meta, tournament_meta, rosters, team_name, results_rows


# ---------------------------------------------------------------------------
# Step 3. Question texts from questions.db
# ---------------------------------------------------------------------------


def fetch_question_texts(questions_db_path: Path, tournament_ids: set[int]):
    """
    Returns:
      questions_by_slot: dict (tournament_id, q_in_tournament_0idx) -> question dict
      packs_by_tid: dict tournament_id -> {pack_id, pack_title, pack_editors: [name,...]}
    """
    _log(f"Reading question texts from {questions_db_path}…")
    conn = sqlite3.connect(str(questions_db_path))
    cur = conn.cursor()

    _log("  scanning questions table…")
    questions_by_slot: dict[tuple[int, int], dict] = {}
    pack_ids_seen: dict[int, set[int]] = defaultdict(set)
    n_total = 0
    for row in cur.execute(
        """
        SELECT id, number, text, answer, zachet, nezachet, comment, source,
               pack_id, pack_title, tour_id, tour_number, tour_title,
               authors_json, editors_json, tournaments_json
        FROM questions
        WHERE tournaments_json IS NOT NULL AND tournaments_json != '[]'
        """
    ):
        n_total += 1
        try:
            tournaments = json.loads(row[15])
        except Exception:
            continue
        number = row[1]
        if number is None:
            continue
        slot = int(number) - 1  # questions.number is 1-indexed
        for t in tournaments:
            tid = int(t.get("id"))
            if tid not in tournament_ids:
                continue
            key = (tid, slot)
            if key in questions_by_slot:
                continue  # keep first
            try:
                authors = json.loads(row[13]) if row[13] else []
            except Exception:
                authors = []
            try:
                editors = json.loads(row[14]) if row[14] else []
            except Exception:
                editors = []
            questions_by_slot[key] = {
                "question_id": int(row[0]),
                "text": row[2],
                "answer": row[3],
                "zachet": row[4],
                "nezachet": row[5],
                "comment": row[6],
                "source": row[7],
                "pack_id": int(row[8]) if row[8] is not None else None,
                "pack_title": row[9],
                "tour_id": int(row[10]) if row[10] is not None else None,
                "tour_number": int(row[11]) if row[11] is not None else None,
                "tour_title": row[12],
                "authors": [a.get("name") for a in authors if isinstance(a, dict)],
                "editors": [e.get("name") for e in editors if isinstance(e, dict)],
            }
            if row[8] is not None:
                pack_ids_seen[tid].add(int(row[8]))

    _log(f"  scanned {n_total:,} questions, matched {len(questions_by_slot):,} slots")

    # For each tournament, pick the most-frequent pack_id and fetch its editors.
    primary_pack: dict[int, int] = {}
    for tid, pids in pack_ids_seen.items():
        # Use the smallest pack_id (deterministic). In practice tournaments map to one pack.
        primary_pack[tid] = min(pids)

    pack_meta: dict[int, dict] = {}
    if primary_pack:
        unique_pack_ids = list({pid for pid in primary_pack.values()})
        _log(f"  fetching pack editors for {len(unique_pack_ids)} packs…")
        # SQLite IN
        chunk = 500
        for i in range(0, len(unique_pack_ids), chunk):
            block = unique_pack_ids[i : i + chunk]
            placeholders = ",".join("?" * len(block))
            for r in cur.execute(
                f"SELECT id, title, editors_json FROM packs WHERE id IN ({placeholders})",
                block,
            ):
                try:
                    eds = json.loads(r[2]) if r[2] else []
                except Exception:
                    eds = []
                pack_meta[int(r[0])] = {
                    "title": r[1],
                    "editors": [e.get("name") for e in eds if isinstance(e, dict)],
                }

    packs_by_tid: dict[int, dict] = {}
    for tid, pid in primary_pack.items():
        meta = pack_meta.get(pid, {})
        packs_by_tid[tid] = {
            "pack_id": pid,
            "pack_title": meta.get("title"),
            "pack_editors": meta.get("editors", []),
        }

    conn.close()
    return questions_by_slot, packs_by_tid


# ---------------------------------------------------------------------------
# Step 4. Compute expected takes per team
# ---------------------------------------------------------------------------


def _build_theta_before_map(
    rosters: dict[tuple[int, int], list[int]],
    tid_to_game_idx: dict[int, int],
    history_player_id: Optional[np.ndarray],
    history_game_id: Optional[np.ndarray],
    history_theta: Optional[np.ndarray],
) -> dict[tuple[int, int], float]:
    """Pre-compute θ as it was *before* each (player, tournament) appearance.

    Walks each player's chronological history once, alongside the sorted list
    of tournaments they appear in.  For each (player, tournament) we record
    the θ_after of the player's most recent prior tournament — i.e. their
    rating at the start of the new tournament.  First-time players get 0.0.
    """
    out: dict[tuple[int, int], float] = {}
    if (
        history_player_id is None
        or history_game_id is None
        or history_theta is None
        or len(history_player_id) == 0
    ):
        return out

    # Collect all (player, game_idx) we need (deduplicated per player).
    needed: dict[int, set[int]] = defaultdict(set)
    for (tid, _team), pids in rosters.items():
        g = tid_to_game_idx.get(int(tid))
        if g is None:
            continue
        for p in pids:
            needed[int(p)].add(int(g))
    if not needed:
        return out

    # NOTE: history_game_id stores the **tournament_id** (DB id), not the
    # chronological game_idx.  Convert to game_idx so that ordering is
    # chronological (engine processes tournaments by start_date) and
    # consistent with the keys we look up.
    pid = history_player_id.astype(np.int64)
    raw_tid = history_game_id.astype(np.int64)
    th = history_theta.astype(np.float64)
    gid = np.array(
        [tid_to_game_idx.get(int(t), -1) for t in raw_tid.tolist()],
        dtype=np.int64,
    )
    keep = gid >= 0
    pid = pid[keep]; gid = gid[keep]; th = th[keep]
    order = np.lexsort((gid, pid))
    pid_s = pid[order]
    gid_s = gid[order]
    th_s = th[order]

    # Index of each player's slice.
    unique_pids, starts = np.unique(pid_s, return_index=True)
    starts = np.append(starts, len(pid_s))
    pid_to_slot = {int(p): i for i, p in enumerate(unique_pids.tolist())}

    # Two-pointer sweep per player: theta_before[g] = th_s[j-1] for largest j with gid_s[j] < g.
    for p, gs_set in needed.items():
        slot = pid_to_slot.get(p)
        if slot is None:
            for g in gs_set:
                out[(p, g)] = 0.0
            continue
        lo, hi = int(starts[slot]), int(starts[slot + 1])
        if lo == hi:
            for g in gs_set:
                out[(p, g)] = 0.0
            continue
        gs_sorted = sorted(gs_set)
        # Vectorised binary search of all requested g's into the player slice.
        pos = np.searchsorted(gid_s[lo:hi], gs_sorted, side="left")
        for g, k in zip(gs_sorted, pos.tolist()):
            out[(p, g)] = 0.0 if k == 0 else float(th_s[lo + k - 1])
    return out


def _solve_implied_theta(
    a_arr: np.ndarray,
    b_eff: np.ndarray,
    n: int,
    score_actual: float,
    *,
    lo: float = -8.0,
    hi: float = 8.0,
    iters: int = 40,
) -> float:
    """
    Solve for θ_team such that a hypothetical team of ``n`` identical
    players, each of strength θ_team, would take exactly ``score_actual``
    questions on this pack:

        S_pred(θ) = Σ_q [1 − exp(−n · exp(a_q · θ − b_eff[q]))]

    ``b_eff`` already bakes in δ_size (for the actual team size) and
    δ_pos − δ_pos[anchor].  S_pred is monotone increasing in θ, so a
    plain bisection converges quickly.  Returns ``lo`` / ``hi`` if the
    score is outside the achievable range (e.g. impossible perfect /
    zero score for that pack and team size).
    """
    if n <= 0:
        return float("nan")

    def predicted(theta: float) -> float:
        z = a_arr * theta - b_eff
        # Clip extremely negative z to avoid 0 contribution noise (still O.K.).
        lam = np.exp(z)
        # p_q = 1 − exp(−n·λ_q) — stable formulation via expm1.
        return float(np.sum(-np.expm1(-n * lam)))

    s_lo = predicted(lo)
    s_hi = predicted(hi)
    if score_actual <= s_lo:
        return lo
    if score_actual >= s_hi:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if predicted(mid) < score_actual:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_expected_takes(
    rosters_by_tid: dict[int, list[tuple[int, list[int]]]],
    questions_by_tid: dict[int, list[tuple[int, float, float, int]]],
    player_theta_final: dict[int, float],
    *,
    tid_to_game_idx: dict[int, int],
    delta_size: Optional[np.ndarray],
    team_size_anchor: Optional[int],
    delta_pos: Optional[np.ndarray],
    pos_anchor: Optional[int],
    theta_before: Optional[dict[tuple[int, int], float]] = None,
    score_by_team: Optional[dict[tuple[int, int], int]] = None,
):
    """
    For each (tournament, team) compute the model's expected number of takes:

        z_kq  = −(b_q + δ_size + δ_pos[q]) + a_q · θ_k(at_t)
        λ_kq  = exp(z_kq)
        p_q   = 1 − exp(− Σ_{k∈team} λ_kq)
        E[T]  = Σ_q p_q

    No per-tournament shift (the model no longer has μ_type or ε_t —
    see the 2026-04 cleanup).

    If ``theta_before`` is provided, θ for each player is taken from the
    rating *as it was before the tournament* (eliminating the time-bias
    that final θ introduces for old games).  Otherwise the final θ is used.

    If ``score_by_team`` is provided, also computes ``theta_implied`` per
    (tid, team_id): the per-player θ that a team of identical players
    of the team's actual size would need to match the observed take
    count exactly (after stripping out δ_size and δ_pos).

    Returns:
      expected:        dict (tid, team_id) -> expected_takes (float)
      theta_implied:   dict (tid, team_id) -> θ_team (float)  (empty if
                       ``score_by_team`` is None)
    """
    expected: dict[tuple[int, int], float] = {}
    theta_implied: dict[tuple[int, int], float] = {}

    has_size = delta_size is not None and team_size_anchor is not None
    has_pos = delta_pos is not None and pos_anchor is not None
    size_max = (len(delta_size) - 1) if has_size else 0
    tour_len = len(delta_pos) if has_pos else 0

    for tid, teams in rosters_by_tid.items():
        qs = questions_by_tid.get(tid, [])
        if not qs:
            continue
        b_arr = np.array([b for _, b, _, _ in qs], dtype=np.float64)
        a_arr = np.array([a for _, _, a, _ in qs], dtype=np.float64)
        qi_arr = np.array([qi for qi, _, _, _ in qs], dtype=np.int64)

        g = tid_to_game_idx.get(tid)

        # δ_pos[q] − δ_pos[anchor]; broadcast across teams.
        if has_pos:
            pos = (qi_arr % tour_len).astype(np.int64)
            pos_shift = delta_pos[pos] - delta_pos[pos_anchor]
            pos_shift = pos_shift.astype(np.float64)
        else:
            pos_shift = np.zeros(len(qs), dtype=np.float64)

        for team_id, pids in teams:
            if theta_before is not None and g is not None:
                thetas = np.array(
                    [theta_before.get((int(p), g), 0.0) for p in pids],
                    dtype=np.float64,
                )
            else:
                thetas = np.array(
                    [player_theta_final.get(p, 0.0) for p in pids],
                    dtype=np.float64,
                )
            n = len(thetas)
            if n == 0:
                expected[(tid, team_id)] = 0.0
                continue

            # δ_size depends on actual roster size, anchored at team_size_anchor.
            if has_size:
                ts = max(1, min(n, size_max))
                size_shift = float(delta_size[ts] - delta_size[team_size_anchor])
            else:
                size_shift = 0.0

            # Effective per-question difficulty for this team.
            b_eff = b_arr + size_shift + pos_shift  # (Q,)
            z = -b_eff[None, :] + np.outer(thetas, a_arr)     # (n, Q)
            lam = np.exp(z)
            S = lam.sum(axis=0)
            p = -np.expm1(-S)
            expected[(tid, team_id)] = float(p.sum())

            if score_by_team is not None:
                score = score_by_team.get((tid, team_id))
                if score is not None:
                    theta_implied[(tid, team_id)] = _solve_implied_theta(
                        a_arr, b_eff, n, float(score)
                    )

    return expected, theta_implied


# ---------------------------------------------------------------------------
# Step 5. Write DuckDB
# ---------------------------------------------------------------------------

DDL = """
DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS tournaments;
DROP TABLE IF EXISTS pack_editors;
DROP TABLE IF EXISTS questions;
DROP TABLE IF EXISTS question_aliases;
DROP TABLE IF EXISTS team_games;
DROP TABLE IF EXISTS player_games;
DROP TABLE IF EXISTS player_history;
DROP TABLE IF EXISTS site_meta;

CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    last_name TEXT,
    first_name TEXT,
    theta DOUBLE,           -- raw model output
    theta_display DOUBLE,   -- theta shrunk for inactivity; used for ranking
    games INTEGER,
    last_game_date DATE
);

CREATE TABLE tournaments (
    tournament_id INTEGER PRIMARY KEY,
    game_idx INTEGER,           -- position in idx_to_game_id (used by player_history)
    title TEXT,
    type TEXT,
    start_date DATE,
    end_date DATE,
    n_questions INTEGER,
    n_teams INTEGER,
    pack_id INTEGER,
    pack_title TEXT
);

CREATE INDEX idx_tournaments_game_idx ON tournaments(game_idx);

CREATE TABLE pack_editors (
    tournament_id INTEGER,
    editor_name TEXT
);

CREATE TABLE questions (
    canonical_idx INTEGER PRIMARY KEY,
    primary_tournament_id INTEGER,
    primary_q_in_tournament INTEGER,
    b DOUBLE,
    a DOUBLE,
    n_obs INTEGER,
    n_taken INTEGER,
    text TEXT,
    answer TEXT,
    zachet TEXT,
    comment TEXT,
    source TEXT,
    authors_json TEXT,
    editors_json TEXT
);

CREATE TABLE question_aliases (
    canonical_idx INTEGER,
    tournament_id INTEGER,
    q_in_tournament INTEGER,
    -- per-tournament observation counts; ``questions.n_obs/n_taken`` are
    -- aggregated across paired tournaments via ``canonical_idx`` and
    -- therefore double-count for sync+async pairs.  Use these columns
    -- when displaying the take rate of a question on a specific
    -- tournament page.
    n_obs INTEGER,
    n_taken INTEGER
);

CREATE TABLE team_games (
    tournament_id INTEGER,
    team_id INTEGER,
    team_name TEXT,
    n_players_active INTEGER,
    score_actual INTEGER,
    expected_takes DOUBLE,
    -- "Effective team strength" — per-player θ that a team of identical
    -- players (same actual roster size) would need to match score_actual
    -- exactly, after stripping out δ_t/δ_size/δ_pos.  Comparable to the
    -- per-player θ values shown elsewhere; used by the team page chart.
    team_theta_implied DOUBLE,
    place DOUBLE
);

CREATE TABLE player_games (
    player_id INTEGER,
    tournament_id INTEGER,
    team_id INTEGER,
    game_idx INTEGER,
    theta_after DOUBLE,
    n_takes_team INTEGER,
    expected_takes_team DOUBLE
);

CREATE TABLE player_history (
    player_id INTEGER,
    tournament_id INTEGER,    -- DB tournament_id (from rating engine history)
    theta DOUBLE,             -- θ shifted into the FINAL gauge (post all
                              -- yearly re-centerings) so the displayed
                              -- history graph is continuous and matches
                              -- the current top-players gauge.
    rank_global INTEGER,      -- player's rank among QUALIFIED players
                              -- as of end-of-day on this tournament's date.
                              -- Qualified = last_active within 365 days
                              -- AND played ≥ 30 cumulative tournaments.
                              -- The ≥30 floor matches the player-profile
                              -- "# rank" badge default and strips the
                              -- cold-start rookie wave (e.g. 250-debut
                              -- school sync no longer sinks established
                              -- players' rank for one chart point).
                              -- 0 means the player itself is below the
                              -- threshold on this row (chart skips it).
    n_active INTEGER          -- size of the qualified pool on this date
                              -- (denominator for rank_global).
);

-- Single-row snapshot metadata for the public site footer.
CREATE TABLE site_meta (
    data_as_of DATE,           -- latest tournament start_date in this build
    model_built_at TIMESTAMPTZ -- when DuckDB was baked (UTC)
);

CREATE INDEX idx_player_games_player ON player_games(player_id);
CREATE INDEX idx_player_games_tournament ON player_games(tournament_id);
CREATE INDEX idx_team_games_tournament ON team_games(tournament_id);
CREATE INDEX idx_question_aliases_tid ON question_aliases(tournament_id);
CREATE INDEX idx_player_history_player ON player_history(player_id);
"""


def _apply_recenter_correction(
    history_tid: np.ndarray,
    history_theta: np.ndarray,
    tid_to_ord: dict[int, int],
    recenter_ord: Optional[np.ndarray],
    recenter_delta: Optional[np.ndarray],
) -> np.ndarray:
    """Shift each historical θ row into the FINAL post-all-recenterings gauge.

    The rating engine periodically applies a gauge transform
    (``θ ↑ Δ``, ``b ↑ a·Δ``) to keep the median of active veterans at a
    fixed target.  Predictions are exactly invariant, but raw history
    rows recorded *before* a re-centering event are stored in the gauge
    that was active at that moment, while the final ``b`` we ship to the
    website is in the post-all-recenterings gauge.  This mismatch causes:

    * a visible cliff on the per-player θ chart at each event (≈ −0.94
      on 2017-09 in the current model, the rest are < 0.03);
    * mis-calibrated ``expected_takes`` for old tournaments (since
      ``expected = noisy_or(θ_old_gauge, b_new_gauge)``).

    We fix both by adding the cumulative shift of all *subsequent* events
    to each row's θ.  After the fix, every history row is in the same
    gauge as the final ``b`` and the per-player chart is continuous.
    """
    if (
        recenter_ord is None
        or recenter_delta is None
        or len(recenter_ord) == 0
    ):
        return history_theta.astype(np.float64, copy=False)

    order = np.argsort(np.asarray(recenter_ord))
    rc_ord = np.asarray(recenter_ord)[order].astype(np.int64)
    rc_delta = np.asarray(recenter_delta)[order].astype(np.float64)
    cum_after = np.concatenate([
        np.cumsum(rc_delta[::-1])[::-1],
        np.zeros(1, dtype=np.float64),
    ])
    h_ord = np.fromiter(
        (tid_to_ord.get(int(t), 10**9) for t in history_tid.tolist()),
        dtype=np.int64,
        count=len(history_tid),
    )
    idx = np.searchsorted(rc_ord, h_ord, side="right")
    shift = cum_after[idx]
    return history_theta.astype(np.float64) + shift


def _compute_rank_history(
    history_pid: np.ndarray,
    history_tid: np.ndarray,
    history_theta: np.ndarray,
    tid_to_ord: dict[int, int],
    *,
    active_window_days: int = 365,
    min_games_active: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """For each (player, tournament) row compute the player's global rank.

    Walks history chronologically by **day**: after applying every update
    from every tournament that took place on a given calendar date,
    snapshots ``(latest_theta, last_active_ord, games_count)`` per player
    once and ranks each row using the player's end-of-day θ against the
    *qualified* pool — players that (a) were seen within
    ``active_window_days`` and (b) have played at least
    ``min_games_active`` tournaments cumulatively as of that day.

    Day-level grouping (rather than per-tournament) avoids a snapshot
    artefact: when several tournaments share a date, the per-tournament
    version processed them in ``tournament_id`` order, so a row written
    early in the day did not see debutants from a large school sync that
    sat at the end of the same date.  Those debutants then materialised
    into the active pool by the next snapshot, producing a phantom
    rank drop on the player's *next* tournament.

    The ``min_games_active`` threshold matches the default used by the
    player-profile ``# rank`` badge (``min_games=30``) so the chart and
    the badge are computed against the same population.  It also strips
    out the cold-start "rookie" wave: a single-game player with the
    "rookie boost" (η₁ = 2η₀) often lands at θ ≈ 0 after one strong
    pack, which without this filter would distort everyone else's rank
    on the day of e.g. a 250-debut school sync.

    Rows for players that have not yet reached ``min_games_active`` get
    ``rank = 0`` (filtered out by the frontend chart); ``n_active`` is
    still populated so the page can show the qualified-pool size.

    Returns two int32 arrays aligned with the input arrays:
    ``(rank_per_row, n_active_per_row)``.  Rank is 1-based.  Ties
    broken arbitrarily (by ``count_higher + 1``, so ties share the
    higher rank).
    """
    n = len(history_pid)
    if n == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )

    pid_arr = history_pid.astype(np.int64)
    tid_arr = history_tid.astype(np.int64)
    th_arr = history_theta.astype(np.float64)
    ord_arr = np.fromiter(
        (tid_to_ord.get(int(t), -1) for t in tid_arr.tolist()),
        dtype=np.int64,
        count=n,
    )

    valid = ord_arr >= 0
    if not valid.all():
        # Drop rows for which we have no date (shouldn't happen in practice).
        pass

    # Sort by (ord, tid, pid) so we can group by tournament chronologically.
    order = np.lexsort((pid_arr, tid_arr, ord_arr))
    pid_s = pid_arr[order]
    tid_s = tid_arr[order]
    th_s = th_arr[order]
    ord_s = ord_arr[order]

    # Map player_id → dense index for O(1) per-player state arrays.
    unique_pid = np.unique(pid_s[ord_s >= 0])
    pid_to_idx = {int(p): i for i, p in enumerate(unique_pid.tolist())}
    n_players = len(unique_pid)

    latest_theta = np.full(n_players, np.nan, dtype=np.float64)
    last_active = np.full(n_players, np.iinfo(np.int64).min, dtype=np.int64)
    games_count = np.zeros(n_players, dtype=np.int32)

    rank_s = np.zeros(n, dtype=np.int32)
    nact_s = np.zeros(n, dtype=np.int32)

    # Group rows by ord — chronological day chunks.  All tournaments on
    # the same date share an end-of-day snapshot; see docstring for why.
    boundaries = np.concatenate([
        [0],
        np.where(ord_s[1:] != ord_s[:-1])[0] + 1,
        [n],
    ])

    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        cur_ord = int(ord_s[start])
        if cur_ord < 0:
            continue

        # Apply every update from every tournament on this date first.
        # If a player has multiple rows on the same day, the last one in
        # ``(tid, pid)`` order wins; ranks below use that end-of-day θ
        # for the player so all of their same-date rows share a rank.
        for k in range(start, end):
            p_idx = pid_to_idx[int(pid_s[k])]
            latest_theta[p_idx] = th_s[k]
            last_active[p_idx] = cur_ord
            games_count[p_idx] += 1

        # Snapshot of qualified players: recently active AND past the
        # rookie threshold.  See docstring for why the threshold matters.
        cutoff = cur_ord - active_window_days
        active = (
            (last_active >= cutoff)
            & (games_count >= min_games_active)
            & ~np.isnan(latest_theta)
        )
        active_thetas = latest_theta[active]
        n_active = int(active.sum())

        # Sort once for binary-search ranks.
        sorted_thetas = np.sort(active_thetas)  # ascending

        for k in range(start, end):
            p_idx = pid_to_idx[int(pid_s[k])]
            nact_s[k] = n_active
            if games_count[p_idx] < min_games_active:
                # Player not yet qualified — leave rank=0 so the frontend
                # chart skips this point.
                continue
            th = latest_theta[p_idx]
            higher = n_active - int(
                np.searchsorted(sorted_thetas, th, side="right")
            )
            rank_s[k] = higher + 1

    # Restore original row order.
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    return rank_s[inv], nact_s[inv]


def write_duckdb(
    out_path: Path,
    *,
    arrays,
    maps,
    res,
    player_meta,
    tournament_meta,
    rosters,
    team_name,
    results_rows,
    questions_by_slot,
    packs_by_tid,
):
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Writing DuckDB to {out_path}…")
    con = duckdb.connect(str(out_path))
    con.execute(DDL)

    # ---- gauge-correct historical θ in place ---------------------------
    # Bring every history row into the same gauge as the final ``b`` so
    # the per-player θ chart is continuous and ``expected_takes`` for old
    # tournaments stays calibrated.  Mutating ``res.history_theta`` here
    # propagates automatically to every downstream consumer (theta-before
    # map, ``player_games.theta_after``, ``player_history.theta``).
    gdo_local = maps.game_date_ordinal
    tid_to_ord_local: dict[int, int] = {
        int(tid): int(gdo_local[i])
        for i, tid in enumerate(maps.idx_to_game_id)
        if i < len(gdo_local) and int(gdo_local[i]) >= 0
    }
    if res.history_theta is not None and len(res.history_theta) > 0:
        n_events = (
            0 if res.recenter_ord is None else len(res.recenter_ord)
        )
        if n_events > 0:
            corrected = _apply_recenter_correction(
                res.history_game_id,
                res.history_theta,
                tid_to_ord_local,
                res.recenter_ord,
                res.recenter_delta,
            )
            shift_max = float(np.max(np.abs(corrected - res.history_theta)))
            res.history_theta = corrected.astype(np.float32)
            _log(
                f"Applied recenter correction to {len(corrected):,} history "
                f"rows (max |Δ|={shift_max:.3f}, {n_events} events)."
            )

    # ---- per-canonical aggregation (n_obs, n_taken) ----
    canonical_q_idx = (
        maps.canonical_q_idx
        if maps.canonical_q_idx is not None
        else np.arange(len(maps.idx_to_question_id), dtype=np.int32)
    )
    num_canonical = int(canonical_q_idx.max()) + 1 if len(canonical_q_idx) else 0
    raw_q_idx = arrays["q_idx"]
    taken = arrays["taken"]
    canon_per_obs = canonical_q_idx[raw_q_idx]
    n_obs_per_canon = np.bincount(canon_per_obs, minlength=num_canonical)
    n_taken_per_canon = np.bincount(
        canon_per_obs, weights=taken.astype(np.int64), minlength=num_canonical
    ).astype(np.int64)

    # Pick a "primary" raw slot per canonical = the one with smallest tournament_id.
    primary_tid = np.full(num_canonical, -1, dtype=np.int64)
    primary_qi = np.full(num_canonical, -1, dtype=np.int32)
    for raw_idx, key in enumerate(maps.idx_to_question_id):
        c = int(canonical_q_idx[raw_idx])
        tid, qi = key
        if primary_tid[c] == -1 or int(tid) < primary_tid[c]:
            primary_tid[c] = int(tid)
            primary_qi[c] = int(qi)

    # ---- players ----
    _log("Inserting players…")
    pid_to_theta = {int(pid): float(t) for pid, t in zip(res.player_id, res.theta)}
    pid_to_games = {int(pid): int(g) for pid, g in zip(res.player_id, res.games)}
    # Last game date per DB player_id: derive from history.  Note that the
    # history arrays already store DB IDs (``history_player_id`` ∈ DB
    # player_id, ``history_game_id`` ∈ DB tournament_id).  ``game_date_ordinal``
    # is indexed by the *internal* tournament index (game_idx), so we have to
    # translate tournament_id → game_idx via ``idx_to_game_id`` before
    # looking up the date.  The previous version of this code looked up
    # ``gdo[tid]`` directly, which silently produced NULL or wrong dates
    # because tournament IDs (~13 000) overshoot the gdo array length (~8 800).
    pid_last_game: dict[int, Optional[date]] = {}
    if res.history_player_id is not None:
        gid_to_game_idx = {int(t): i for i, t in enumerate(maps.idx_to_game_id)}
        gdo = maps.game_date_ordinal
        last_tid_per_pid: dict[int, int] = {}
        # We want the *latest* tournament per player; pick the largest
        # game_idx (chronological ordering by start_date is what we
        # actually want, but game_idx is built in chronological order).
        for pid_db, tid_db in zip(
            res.history_player_id.tolist(), res.history_game_id.tolist()
        ):
            g_idx = gid_to_game_idx.get(int(tid_db))
            if g_idx is None:
                continue
            prev = last_tid_per_pid.get(int(pid_db))
            if prev is None or g_idx > prev:
                last_tid_per_pid[int(pid_db)] = g_idx
        for pid_db, g_idx in last_tid_per_pid.items():
            if 0 <= g_idx < len(gdo):
                pid_last_game[pid_db] = _ord_to_date(int(gdo[g_idx]))

    today = date.today()
    pid_arr, last_arr, first_arr, theta_arr, theta_disp_arr, games_arr, last_dt_arr = (
        [], [], [], [], [], [], []
    )
    for pid in maps.idx_to_player_id:
        pid = int(pid)
        meta = player_meta.get(pid, {})
        pid_arr.append(pid)
        last_arr.append(meta.get("last_name") or "")
        first_arr.append(meta.get("first_name") or "")
        theta = pid_to_theta.get(pid, 0.0)
        last_played = pid_last_game.get(pid)
        theta_arr.append(theta)
        theta_disp_arr.append(compute_theta_display(theta, last_played, today))
        games_arr.append(pid_to_games.get(pid, 0))
        last_dt_arr.append(last_played)
    _bulk_insert(
        con,
        "players",
        ["player_id", "last_name", "first_name", "theta", "theta_display", "games", "last_game_date"],
        {
            "player_id": pa.array(pid_arr, type=pa.int32()),
            "last_name": pa.array(last_arr, type=pa.string()),
            "first_name": pa.array(first_arr, type=pa.string()),
            "theta": pa.array(theta_arr, type=pa.float64()),
            "theta_display": pa.array(theta_disp_arr, type=pa.float64()),
            "games": pa.array(games_arr, type=pa.int32()),
            "last_game_date": pa.array(last_dt_arr, type=pa.date32()),
        },
    )

    # ---- tournaments ----
    _log("Inserting tournaments…")
    tid_to_n_teams: dict[int, int] = defaultdict(int)
    for (tid, _team), pids in rosters.items():
        if pids:
            tid_to_n_teams[tid] += 1

    cols: dict[str, list] = {
        "tournament_id": [],
        "game_idx": [],
        "title": [],
        "type": [],
        "start_date": [],
        "end_date": [],
        "n_questions": [],
        "n_teams": [],
        "pack_id": [],
        "pack_title": [],
    }
    for g_idx, tid in enumerate(maps.idx_to_game_id):
        tid = int(tid)
        meta = tournament_meta.get(tid, {})
        pack = packs_by_tid.get(tid, {})
        cols["tournament_id"].append(tid)
        cols["game_idx"].append(g_idx)
        cols["title"].append(meta.get("title") or "")
        cols["type"].append(meta.get("type") or "offline")
        cols["start_date"].append(meta.get("start_date"))
        cols["end_date"].append(meta.get("end_date"))
        cols["n_questions"].append(int(meta.get("n_questions") or 0))
        cols["n_teams"].append(int(tid_to_n_teams.get(tid, 0)))
        cols["pack_id"].append(pack.get("pack_id"))
        cols["pack_title"].append(pack.get("pack_title"))
    _bulk_insert(
        con,
        "tournaments",
        list(cols.keys()),
        {
            "tournament_id": pa.array(cols["tournament_id"], type=pa.int32()),
            "game_idx": pa.array(cols["game_idx"], type=pa.int32()),
            "title": pa.array(cols["title"], type=pa.string()),
            "type": pa.array(cols["type"], type=pa.string()),
            "start_date": pa.array(cols["start_date"], type=pa.date32()),
            "end_date": pa.array(cols["end_date"], type=pa.date32()),
            "n_questions": pa.array(cols["n_questions"], type=pa.int32()),
            "n_teams": pa.array(cols["n_teams"], type=pa.int32()),
            "pack_id": pa.array(cols["pack_id"], type=pa.int32()),
            "pack_title": pa.array(cols["pack_title"], type=pa.string()),
        },
    )

    starts_nonnull = [d for d in cols["start_date"] if d is not None]
    data_as_of_max = max(starts_nonnull) if starts_nonnull else None
    built_at_utc = datetime.now(timezone.utc)
    con.execute(
        "INSERT INTO site_meta (data_as_of, model_built_at) VALUES (?, ?)",
        [data_as_of_max, built_at_utc],
    )
    _log(
        f"site_meta: data_as_of={data_as_of_max}, model_built_at={built_at_utc.isoformat()}"
    )

    # ---- pack_editors ----
    _log("Inserting pack_editors…")
    pe_tid: list[int] = []
    pe_name: list[str] = []
    for tid, pack in packs_by_tid.items():
        for ed in pack.get("pack_editors", []):
            if ed:
                pe_tid.append(int(tid))
                pe_name.append(str(ed))
    if pe_tid:
        _bulk_insert(
            con,
            "pack_editors",
            ["tournament_id", "editor_name"],
            {
                "tournament_id": pa.array(pe_tid, type=pa.int32()),
                "editor_name": pa.array(pe_name, type=pa.string()),
            },
        )

    # ---- questions ----
    _log("Inserting questions…")
    canon_arr = list(range(num_canonical))
    ptid_arr = [int(primary_tid[c]) if primary_tid[c] >= 0 else None for c in canon_arr]
    pqi_arr = [int(primary_qi[c]) if primary_qi[c] >= 0 else None for c in canon_arr]
    b_arr = res.b.astype(np.float64).tolist()
    a_arr = res.a.astype(np.float64).tolist()
    nobs_arr = n_obs_per_canon.astype(np.int64).tolist()
    ntaken_arr = n_taken_per_canon.astype(np.int64).tolist()

    text_arr: list[Optional[str]] = []
    answer_arr: list[Optional[str]] = []
    zachet_arr: list[Optional[str]] = []
    comment_arr: list[Optional[str]] = []
    source_arr: list[Optional[str]] = []
    authors_arr: list[Optional[str]] = []
    editors_arr: list[Optional[str]] = []
    for c in canon_arr:
        ptid = ptid_arr[c]
        pqi = pqi_arr[c]
        q = questions_by_slot.get((ptid, pqi)) if ptid is not None and pqi is not None else None
        if q:
            text_arr.append(q.get("text"))
            answer_arr.append(q.get("answer"))
            zachet_arr.append(q.get("zachet"))
            comment_arr.append(q.get("comment"))
            source_arr.append(q.get("source"))
            authors_arr.append(json.dumps(q.get("authors") or [], ensure_ascii=False))
            editors_arr.append(json.dumps(q.get("editors") or [], ensure_ascii=False))
        else:
            text_arr.append(None)
            answer_arr.append(None)
            zachet_arr.append(None)
            comment_arr.append(None)
            source_arr.append(None)
            authors_arr.append(None)
            editors_arr.append(None)

    _bulk_insert(
        con,
        "questions",
        [
            "canonical_idx", "primary_tournament_id", "primary_q_in_tournament",
            "b", "a", "n_obs", "n_taken",
            "text", "answer", "zachet", "comment", "source",
            "authors_json", "editors_json",
        ],
        {
            "canonical_idx": pa.array(canon_arr, type=pa.int32()),
            "primary_tournament_id": pa.array(ptid_arr, type=pa.int32()),
            "primary_q_in_tournament": pa.array(pqi_arr, type=pa.int32()),
            "b": pa.array(b_arr, type=pa.float64()),
            "a": pa.array(a_arr, type=pa.float64()),
            "n_obs": pa.array(nobs_arr, type=pa.int64()),
            "n_taken": pa.array(ntaken_arr, type=pa.int64()),
            "text": pa.array(text_arr, type=pa.string()),
            "answer": pa.array(answer_arr, type=pa.string()),
            "zachet": pa.array(zachet_arr, type=pa.string()),
            "comment": pa.array(comment_arr, type=pa.string()),
            "source": pa.array(source_arr, type=pa.string()),
            "authors_json": pa.array(authors_arr, type=pa.string()),
            "editors_json": pa.array(editors_arr, type=pa.string()),
        },
    )

    # ---- question_aliases ----
    _log("Inserting question_aliases…")
    # Per-(tournament, q_in_tournament) take counts; raw_q_idx already
    # uniquely identifies a tournament slot, so a plain bincount over
    # observations gives us per-tournament n_obs / n_taken (unlike the
    # canonical bincount above, which collapses sync+async pairs).
    n_obs_per_raw = np.bincount(raw_q_idx, minlength=len(maps.idx_to_question_id)).astype(
        np.int64
    )
    n_taken_per_raw = np.bincount(
        raw_q_idx, weights=taken.astype(np.int64), minlength=len(maps.idx_to_question_id)
    ).astype(np.int64)
    qa_canon = canonical_q_idx.astype(np.int32).tolist()
    qa_tid = [int(k[0]) for k in maps.idx_to_question_id]
    qa_qi = [int(k[1]) for k in maps.idx_to_question_id]
    _bulk_insert(
        con,
        "question_aliases",
        ["canonical_idx", "tournament_id", "q_in_tournament", "n_obs", "n_taken"],
        {
            "canonical_idx": pa.array(qa_canon, type=pa.int32()),
            "tournament_id": pa.array(qa_tid, type=pa.int32()),
            "q_in_tournament": pa.array(qa_qi, type=pa.int32()),
            "n_obs": pa.array(n_obs_per_raw.tolist(), type=pa.int32()),
            "n_taken": pa.array(n_taken_per_raw.tolist(), type=pa.int32()),
        },
    )

    # ---- team_games + player_games ----
    _log("Computing expected takes per team…")

    # group rosters by tournament for vectorised compute
    rosters_by_tid: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for (tid, team_id), pids in rosters.items():
        rosters_by_tid[tid].append((team_id, pids))

    # questions per tournament (use canonical b/a via canonical map)
    questions_by_tid: dict[int, list[tuple[int, float, float, int]]] = defaultdict(list)
    for raw_idx, key in enumerate(maps.idx_to_question_id):
        tid, qi = key
        c = int(canonical_q_idx[raw_idx])
        questions_by_tid[int(tid)].append(
            (int(qi), float(res.b[c]), float(res.a[c]), int(qi))
        )
    for tid in questions_by_tid:
        questions_by_tid[tid].sort(key=lambda x: x[0])

    # tid -> game index in the maps (used to look up ε_t / type)
    tid_to_game_idx: dict[int, int] = {
        int(tid): i for i, tid in enumerate(maps.idx_to_game_id)
    }

    # Build (player, tournament) → θ_before map so expected takes use θ AS IT
    # WAS at the start of each tournament (eliminates time-bias from final θ).
    _log("  building θ-before-tournament map from history…")
    theta_before_map = _build_theta_before_map(
        rosters, tid_to_game_idx,
        res.history_player_id, res.history_game_id, res.history_theta,
    )
    _log(f"    {len(theta_before_map):,} (player, tournament) θ values")

    # Pre-compute observed score per (tid, team) so compute_expected_takes
    # can also solve for the implied team θ.
    score_by_team: dict[tuple[int, int], int] = {}
    for (tid_k, team_k), result in results_rows.items():
        mask = result.get("points_mask") if isinstance(result, dict) else None
        if isinstance(mask, str):
            score_by_team[(int(tid_k), int(team_k))] = sum(
                1 for c in mask if c == "1"
            )

    expected, theta_implied = compute_expected_takes(
        rosters_by_tid,
        questions_by_tid,
        pid_to_theta,
        tid_to_game_idx=tid_to_game_idx,
        delta_size=res.delta_size,
        team_size_anchor=res.team_size_anchor,
        delta_pos=res.delta_pos,
        pos_anchor=res.pos_anchor,
        theta_before=theta_before_map or None,
        score_by_team=score_by_team,
    )
    _log(
        f"  computed expected_takes for {len(expected):,} teams "
        f"(implied θ for {len(theta_implied):,})"
    )

    # Insert team_games + collect player_games
    _log("Inserting team_games…")
    tg_tid: list[int] = []
    tg_team: list[int] = []
    tg_name: list[str] = []
    tg_nact: list[int] = []
    tg_score: list[int] = []
    tg_exp: list[float] = []
    tg_theta: list[Optional[float]] = []
    tg_place: list[Optional[float]] = []
    pg_template: dict[int, dict[int, tuple[int, int, float]]] = {}
    # pg_template: player_id -> {tournament_id: (team_id, n_takes_team, expected_takes_team)}
    for (tid, team_id), pids in rosters.items():
        active_pids = [p for p in pids if p in pid_to_theta]
        if not active_pids:
            continue
        n_active = len(active_pids)
        result = results_rows.get((tid, team_id))
        if result is not None:
            mask = result["points_mask"]
            score_actual = sum(1 for c in mask if c == "1")
            place = result["position"]
        else:
            score_actual = 0
            place = None
        exp = expected.get((tid, team_id), 0.0)
        ti = theta_implied.get((tid, team_id))
        tg_tid.append(int(tid))
        tg_team.append(int(team_id))
        tg_name.append(team_name.get(team_id, ""))
        tg_nact.append(n_active)
        tg_score.append(int(score_actual))
        tg_exp.append(float(exp))
        tg_theta.append(float(ti) if ti is not None else None)
        tg_place.append(place)
        for pid in active_pids:
            pg_template.setdefault(int(pid), {})[int(tid)] = (
                int(team_id),
                int(score_actual),
                float(exp),
            )
    _bulk_insert(
        con,
        "team_games",
        ["tournament_id", "team_id", "team_name", "n_players_active",
         "score_actual", "expected_takes", "team_theta_implied", "place"],
        {
            "tournament_id": pa.array(tg_tid, type=pa.int32()),
            "team_id": pa.array(tg_team, type=pa.int32()),
            "team_name": pa.array(tg_name, type=pa.string()),
            "n_players_active": pa.array(tg_nact, type=pa.int32()),
            "score_actual": pa.array(tg_score, type=pa.int32()),
            "expected_takes": pa.array(tg_exp, type=pa.float64()),
            "team_theta_implied": pa.array(tg_theta, type=pa.float64()),
            "place": pa.array(tg_place, type=pa.float64()),
        },
    )

    # ---- player_games (uses theta history) ----
    _log("Inserting player_games…")
    tid_to_game_idx = {int(tid): i for i, tid in enumerate(maps.idx_to_game_id)}
    # Build per-player history dict for fast lookup.
    history_by_player: dict[int, dict[int, float]] = defaultdict(dict)
    if res.history_player_id is not None:
        for p, g, t in zip(
            res.history_player_id.tolist(),
            res.history_game_id.tolist(),
            res.history_theta.tolist(),
        ):
            history_by_player[int(p)][int(g)] = float(t)

    pg_pid: list[int] = []
    pg_tid: list[int] = []
    pg_team: list[int] = []
    pg_g: list[int] = []
    pg_theta: list[Optional[float]] = []
    pg_takes: list[int] = []
    pg_exp: list[float] = []
    for pid, per_tid in pg_template.items():
        per_g = history_by_player.get(pid, {})
        for tid, (team_id, n_takes_team, exp_team) in per_tid.items():
            g_idx = tid_to_game_idx.get(tid, -1)
            pg_pid.append(pid)
            pg_tid.append(tid)
            pg_team.append(team_id)
            pg_g.append(g_idx)
            pg_theta.append(per_g.get(g_idx))
            pg_takes.append(n_takes_team)
            pg_exp.append(exp_team)
    _bulk_insert(
        con,
        "player_games",
        ["player_id", "tournament_id", "team_id", "game_idx",
         "theta_after", "n_takes_team", "expected_takes_team"],
        {
            "player_id": pa.array(pg_pid, type=pa.int32()),
            "tournament_id": pa.array(pg_tid, type=pa.int32()),
            "team_id": pa.array(pg_team, type=pa.int32()),
            "game_idx": pa.array(pg_g, type=pa.int32()),
            "theta_after": pa.array(pg_theta, type=pa.float64()),
            "n_takes_team": pa.array(pg_takes, type=pa.int32()),
            "expected_takes_team": pa.array(pg_exp, type=pa.float64()),
        },
    )

    # ---- player_history ----
    # NOTE: rating engine writes (player_id, tournament_id, θ_after) into the
    # `history_*` arrays.  We store tournament_id explicitly so JOINs with
    # `tournaments` work without going through the chronological `game_idx`.
    _log("Inserting player_history…")
    if res.history_player_id is not None and len(res.history_player_id) > 0:
        _log("  computing per-(player, tournament) global rank snapshots…")
        rank_arr, nact_arr = _compute_rank_history(
            res.history_player_id,
            res.history_game_id,
            res.history_theta,
            tid_to_ord_local,
        )
        _log(
            f"  rank computed for {len(rank_arr):,} rows "
            f"(max n_active={int(nact_arr.max()) if len(nact_arr) else 0:,})"
        )
        _bulk_insert(
            con,
            "player_history",
            ["player_id", "tournament_id", "theta", "rank_global", "n_active"],
            {
                "player_id": pa.array(res.history_player_id.astype(np.int32)),
                "tournament_id": pa.array(res.history_game_id.astype(np.int32)),
                "theta": pa.array(res.history_theta.astype(np.float64)),
                "rank_global": pa.array(rank_arr.astype(np.int32)),
                "n_active": pa.array(nact_arr.astype(np.int32)),
            },
        )

    # ---- ANALYZE ----
    _log("Finalising…")
    con.execute("PRAGMA force_checkpoint")
    con.close()
    _log(f"Done. Output at {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Build chgk.duckdb for the website")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "data.npz")
    p.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "seq.npz")
    p.add_argument(
        "--questions-db",
        type=Path,
        default=Path("/Users/fbr/Projects/personal/chgk-embedings/data/questions.db"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "website" / "data" / "chgk.duckdb",
    )
    p.add_argument(
        "--limit-tournaments",
        type=int,
        default=None,
        help="If set, restrict to first N tournaments (for fast iteration).",
    )
    p.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Postgres URL for rating DB (overrides DATABASE_URL env var).",
    )
    args = p.parse_args()

    t0 = time.time()
    arrays, maps, res = load_model(args.cache, args.results)

    # Optional restriction for dev iteration.
    if args.limit_tournaments:
        keep_tids = set(int(t) for t in maps.idx_to_game_id[: args.limit_tournaments])
        _log(f"Restricting to first {len(keep_tids):,} tournaments…")
        # filter observations and rebuild maps. Easiest: filter at the end.
        # We'll filter rosters/questions/team rows below by checking membership.
        # For simplicity we keep maps as-is and just filter what we insert.
        tournament_id_filter: Optional[set[int]] = keep_tids
    else:
        tournament_id_filter = None

    tournament_ids = sorted(int(t) for t in maps.idx_to_game_id)
    if tournament_id_filter is not None:
        tournament_ids = [t for t in tournament_ids if t in tournament_id_filter]

    player_ids = [int(p) for p in maps.idx_to_player_id]

    player_meta, tournament_meta, rosters, team_name, results_rows = (
        fetch_rating_db_metadata(
            tournament_ids, player_ids, database_url=args.database_url
        )
    )

    # Filter rosters/results to allowed tournaments (defensive).
    if tournament_id_filter is not None:
        rosters = {
            k: v for k, v in rosters.items() if k[0] in tournament_id_filter
        }
        results_rows = {
            k: v for k, v in results_rows.items() if k[0] in tournament_id_filter
        }

    questions_by_slot, packs_by_tid = fetch_question_texts(
        args.questions_db, set(tournament_ids)
    )

    write_duckdb(
        args.out,
        arrays=arrays,
        maps=maps,
        res=res,
        player_meta=player_meta,
        tournament_meta=tournament_meta,
        rosters=rosters,
        team_name=team_name,
        results_rows=results_rows,
        questions_by_slot=questions_by_slot,
        packs_by_tid=packs_by_tid,
    )

    _log(f"Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
