"""Decisive head-to-head pair stats from packed observation arrays."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from data import IndexMaps

PairOutcome = Literal["win_lo", "win_hi", "tie"]


@dataclass
class PairStat:
    """Aggregated stats for unordered player pair (p_lo < p_hi)."""

    p_lo: int
    p_hi: int
    n_shared: int = 0
    only_lo: int = 0
    only_hi: int = 0
    both_taken: int = 0
    neither: int = 0
    n_same_team: int = 0
    dec_team_theta_sum_lo: float = 0.0
    dec_team_theta_sum_hi: float = 0.0
    dec_teammate_theta_sum_lo: float = 0.0
    dec_teammate_theta_sum_hi: float = 0.0
    dec_context_count: int = 0
    tourney_decisive: dict[int, int] = field(default_factory=dict)

    @property
    def n_decisive(self) -> int:
        return self.only_lo + self.only_hi

    def duel_score_lo(self) -> float:
        nd = self.n_decisive
        return (self.only_lo - self.only_hi) / nd if nd else 0.0

    def avg_team_theta_lo(self) -> float | None:
        return (
            self.dec_team_theta_sum_lo / self.dec_context_count
            if self.dec_context_count
            else None
        )

    def avg_team_theta_hi(self) -> float | None:
        return (
            self.dec_team_theta_sum_hi / self.dec_context_count
            if self.dec_context_count
            else None
        )

    def avg_teammate_theta_lo(self) -> float | None:
        return (
            self.dec_teammate_theta_sum_lo / self.dec_context_count
            if self.dec_context_count
            else None
        )

    def avg_teammate_theta_hi(self) -> float | None:
        return (
            self.dec_teammate_theta_sum_hi / self.dec_context_count
            if self.dec_context_count
            else None
        )


def _pair_key(pa: int, pb: int) -> tuple[int, int]:
    return (pa, pb) if pa < pb else (pb, pa)


def _tri_flat_index(i: int, j: int, e: int) -> int:
    if i > j:
        i, j = j, i
    return i * e - i * (i + 1) // 2 + (j - i - 1)


def _team_context(
    roster: list[int],
    player: int,
    theta: np.ndarray,
) -> tuple[float, float]:
    th = theta[np.asarray(roster, dtype=np.int64)]
    team_mean = float(th.mean())
    if len(roster) <= 1:
        return team_mean, team_mean
    mask = np.array([p != player for p in roster])
    teammate_mean = float(th[mask].mean())
    return team_mean, teammate_mean


def _matrix_to_pair_stats(
    eli_pidx: np.ndarray,
    n_shared: np.ndarray,
    wins_lo: np.ndarray,
    wins_hi: np.ndarray,
    *,
    team_th_lo: np.ndarray | None = None,
    team_th_hi: np.ndarray | None = None,
    tm_th_lo: np.ndarray | None = None,
    tm_th_hi: np.ndarray | None = None,
    ctx_count: np.ndarray | None = None,
) -> dict[tuple[int, int], PairStat]:
    n_eli = len(eli_pidx)
    out: dict[tuple[int, int], PairStat] = {}
    for flat in np.nonzero(n_shared)[0]:
        flat = int(flat)
        # decode flat -> (i, j)
        i = 0
        rem = flat
        while i < n_eli - 1:
            row_len = n_eli - i - 1
            if rem < row_len:
                j = i + 1 + rem
                break
            rem -= row_len
            i += 1
        else:
            continue
        ns = int(n_shared[flat])
        ol = int(wins_lo[flat])
        oh = int(wins_hi[flat])
        p_lo = int(eli_pidx[i])
        p_hi = int(eli_pidx[j])
        st = PairStat(p_lo=p_lo, p_hi=p_hi, n_shared=ns, only_lo=ol, only_hi=oh)
        if ctx_count is not None and team_th_lo is not None:
            cc = int(ctx_count[flat])
            if cc:
                st.dec_context_count = cc
                st.dec_team_theta_sum_lo = float(team_th_lo[flat])
                st.dec_team_theta_sum_hi = float(team_th_hi[flat])
                st.dec_teammate_theta_sum_lo = float(tm_th_lo[flat])
                st.dec_teammate_theta_sum_hi = float(tm_th_hi[flat])
        out[(p_lo, p_hi)] = st
    return out


def build_pair_stats_from_arrays(
    arrays: dict[str, np.ndarray],
    maps: IndexMaps,
    *,
    theta: np.ndarray,
    eligible: np.ndarray | set[int] | None = None,
    exclude_same_team: bool = True,
    collect_context: bool = True,
) -> dict[tuple[int, int], PairStat]:
    """Build decisive H2H pair stats via slot-centric taken×missed cross."""
    q_idx = arrays["q_idx"]
    taken = arrays["taken"]
    team_sizes = arrays["team_sizes"]
    player_flat = arrays["player_indices_flat"]

    if eligible is None:
        eligible_mask = np.ones(maps.num_players, dtype=bool)
    elif isinstance(eligible, set):
        eligible_mask = np.zeros(maps.num_players, dtype=bool)
        for p in eligible:
            eligible_mask[int(p)] = True
    else:
        eligible_mask = np.asarray(eligible, dtype=bool)

    eli_pidx = np.where(eligible_mask)[0].astype(np.int32)
    n_eli = len(eli_pidx)
    if n_eli < 2:
        return {}

    eli_map = np.full(maps.num_players, -1, dtype=np.int32)
    eli_map[eli_pidx] = np.arange(n_eli, dtype=np.int32)
    n_tri = n_eli * (n_eli - 1) // 2
    n_shared = np.zeros(n_tri, dtype=np.int32)
    wins_lo = np.zeros(n_tri, dtype=np.int32)
    wins_hi = np.zeros(n_tri, dtype=np.int32)
    if collect_context:
        ctx_count = np.zeros(n_tri, dtype=np.int32)
        team_th_lo = np.zeros(n_tri, dtype=np.float64)
        team_th_hi = np.zeros(n_tri, dtype=np.float64)
        tm_th_lo = np.zeros(n_tri, dtype=np.float64)
        tm_th_hi = np.zeros(n_tri, dtype=np.float64)
    else:
        ctx_count = team_th_lo = team_th_hi = tm_th_lo = tm_th_hi = None

    n_obs = len(q_idx)
    n_q = len(maps.idx_to_question_id)
    q_tid = np.empty(n_q, dtype=np.int64)
    q_qi = np.empty(n_q, dtype=np.int32)
    for i in range(n_q):
        qid = maps.idx_to_question_id[i]
        if isinstance(qid, tuple):
            q_tid[i] = int(qid[0])
            q_qi[i] = int(qid[1])
        else:
            q_tid[i] = 0
            q_qi[i] = int(qid)
    slot_tid = q_tid[q_idx]
    slot_qi = q_qi[q_idx]
    order = np.lexsort((slot_qi, slot_tid))

    obs_rep = np.repeat(np.arange(n_obs, dtype=np.int64), team_sizes.astype(np.int64))
    has_eligible = eligible_mask[player_flat.astype(np.int64)]
    eligible_obs = np.zeros(n_obs, dtype=bool)
    eligible_obs[np.unique(obs_rep[has_eligible])] = True
    order = order[eligible_obs[order]]

    offsets = np.empty(n_obs + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(team_sizes, out=offsets[1:])

    slot_players: dict[int, tuple[int, float, float, int]] = {}

    if len(order) == 0:
        return _matrix_to_pair_stats(
            eli_pidx, n_shared, wins_lo, wins_hi,
            team_th_lo=team_th_lo, team_th_hi=team_th_hi,
            tm_th_lo=tm_th_lo, tm_th_hi=tm_th_hi, ctx_count=ctx_count,
        )

    slot_keys_sorted = np.column_stack([slot_tid[order], slot_qi[order]])
    diff = np.any(slot_keys_sorted[1:] != slot_keys_sorted[:-1], axis=1)
    seg_starts = np.concatenate([[0], np.flatnonzero(diff) + 1])
    seg_ends = np.concatenate([seg_starts[1:], [len(order)]])

    def _flush_slot() -> None:
        nonlocal slot_players
        if not slot_players:
            return
        taken_p: list[int] = []
        missed_p: list[int] = []
        taken_th: list[float] = []
        missed_th: list[float] = []
        taken_tm: list[float] = []
        missed_tm: list[float] = []
        for p, (t, team_th, tm_th, _obs) in slot_players.items():
            if t:
                taken_p.append(p)
                if collect_context:
                    taken_th.append(team_th)
                    taken_tm.append(tm_th)
            else:
                missed_p.append(p)
                if collect_context:
                    missed_th.append(team_th)
                    missed_tm.append(tm_th)
        slot_players = {}
        if not taken_p or not missed_p:
            return

        pa = np.repeat(np.asarray(taken_p, dtype=np.int32), len(missed_p))
        pb = np.tile(np.asarray(missed_p, dtype=np.int32), len(taken_p))
        ea = eli_map[pa]
        eb = eli_map[pb]
        valid = (ea >= 0) & (eb >= 0)
        if not valid.any():
            return
        pa = pa[valid]
        ea = ea[valid]
        eb = eb[valid]

        lo_is_a = ea < eb
        i = np.where(lo_is_a, ea, eb)
        j = np.where(lo_is_a, eb, ea)
        flats = (i * n_eli) - (i * (i + 1) // 2) + (j - i - 1)

        lo_wins = np.zeros(len(flats), dtype=np.int32)
        hi_wins = np.zeros(len(flats), dtype=np.int32)
        p_lo_arr = eli_pidx[i]
        lo_wins[pa == p_lo_arr] = 1
        hi_wins[pa != p_lo_arr] = 1

        np.add.at(n_shared, flats, 1)
        np.add.at(wins_lo, flats, lo_wins)
        np.add.at(wins_hi, flats, hi_wins)

        if collect_context and ctx_count is not None:
            team_th_a = np.repeat(np.asarray(taken_th, dtype=np.float64), len(missed_p))[valid]
            team_th_b = np.tile(np.asarray(missed_th, dtype=np.float64), len(taken_p))[valid]
            tm_th_a = np.repeat(np.asarray(taken_tm, dtype=np.float64), len(missed_p))[valid]
            tm_th_b = np.tile(np.asarray(missed_tm, dtype=np.float64), len(taken_p))[valid]
            th_lo = np.where(lo_is_a, team_th_a, team_th_b)
            th_hi = np.where(lo_is_a, team_th_b, team_th_a)
            tm_lo = np.where(lo_is_a, tm_th_a, tm_th_b)
            tm_hi = np.where(lo_is_a, tm_th_b, tm_th_a)
            np.add.at(ctx_count, flats, 1)
            np.add.at(team_th_lo, flats, th_lo)
            np.add.at(team_th_hi, flats, th_hi)
            np.add.at(tm_th_lo, flats, tm_lo)
            np.add.at(tm_th_hi, flats, tm_hi)

    for s, seg_end in zip(seg_starts, seg_ends):
        slot_players = {}
        for pos in order[s:seg_end]:
            i = int(pos)
            roster = player_flat[offsets[i] : offsets[i + 1]]
            t = 1 if float(taken[i]) >= 0.5 else 0
            obs_id = int(i)
            rost_elig = roster[eligible_mask[roster]]
            if len(rost_elig) == 0:
                continue
            if collect_context:
                roster_list = [int(p) for p in rost_elig]
                for p in roster_list:
                    team_th, tm_th = _team_context(roster_list, p, theta)
                    prev = slot_players.get(p)
                    if prev is None:
                        slot_players[p] = (t, team_th, tm_th, obs_id)
                    elif exclude_same_team and prev[3] == obs_id:
                        continue
                    elif prev[1] != t:
                        continue
            else:
                for p in rost_elig:
                    p = int(p)
                    prev = slot_players.get(p)
                    if prev is None:
                        slot_players[p] = (t, 0.0, 0.0, obs_id)
                    elif exclude_same_team and prev[3] == obs_id:
                        continue
                    elif prev[1] != t:
                        continue
        _flush_slot()

    return _matrix_to_pair_stats(
        eli_pidx,
        n_shared,
        wins_lo,
        wins_hi,
        team_th_lo=team_th_lo,
        team_th_hi=team_th_hi,
        tm_th_lo=tm_th_lo,
        tm_th_hi=tm_th_hi,
        ctx_count=ctx_count,
    )


def pair_outcome(
    st: PairStat,
    *,
    min_decisive: int = 20,
) -> PairOutcome:
    if st.n_decisive < min_decisive:
        return "tie"
    if st.only_lo > st.only_hi:
        return "win_lo"
    if st.only_hi > st.only_lo:
        return "win_hi"
    return "tie"


def compute_duel_scores(
    pair_stats: dict[tuple[int, int], PairStat],
    player_ids: list[int] | np.ndarray,
    *,
    min_decisive: int = 20,
) -> dict[int, float]:
    players = [int(p) for p in player_ids]
    scores: dict[int, float] = {p: 0.0 for p in players}
    player_set = set(players)
    exposure: dict[int, float] = {p: 0.0 for p in players}

    for (p_lo, p_hi), st in pair_stats.items():
        if st.n_decisive < min_decisive:
            continue
        nd = float(st.n_decisive)
        delta = st.only_lo - st.only_hi
        if p_lo in player_set:
            scores[p_lo] += delta
            exposure[p_lo] += nd
        if p_hi in player_set:
            scores[p_hi] -= delta
            exposure[p_hi] += nd

    return {
        p: scores[p] / exposure[p] if exposure[p] > 0 else 0.0 for p in players
    }


def compute_pairwise_concordance(
    pair_stats: dict[tuple[int, int], PairStat],
    theta: np.ndarray,
    *,
    min_shared: int = 50,
    min_decisive: int = 20,
) -> dict[str, float]:
    correct = 0
    total = 0
    weighted_correct = 0.0
    weight_sum = 0.0

    for st in pair_stats.values():
        if st.n_shared < min_shared or st.n_decisive < min_decisive:
            continue
        outcome = pair_outcome(st, min_decisive=min_decisive)
        if outcome == "tie":
            continue
        p_lo, p_hi = st.p_lo, st.p_hi
        pred_lo = theta[p_lo] > theta[p_hi]
        fact_lo = outcome == "win_lo"
        match = pred_lo == fact_lo
        w = float(st.n_decisive)
        total += 1
        correct += int(match)
        weighted_correct += w * int(match)
        weight_sum += w

    return {
        "n_pairs": float(total),
        "accuracy": correct / total if total else float("nan"),
        "weighted_accuracy": weighted_correct / weight_sum if weight_sum else float("nan"),
    }


def context_explains(
    st: PairStat,
    *,
    delta_theta: float,
) -> bool:
    tm_lo = st.avg_teammate_theta_lo()
    tm_hi = st.avg_teammate_theta_hi()
    if tm_lo is None or tm_hi is None:
        return False
    ctx_delta = tm_lo - tm_hi
    if st.only_lo == st.only_hi:
        return False
    fact_lo_wins = st.only_lo > st.only_hi
    return abs(ctx_delta) >= delta_theta and ((ctx_delta > 0) == fact_lo_wins)


def fit_duel_elo(
    pair_stats: dict[tuple[int, int], PairStat],
    player_ids: list[int] | np.ndarray,
    *,
    min_decisive: int = 20,
    K: float = 32.0,
    n_iter: int = 20,
) -> dict[int, float]:
    players = sorted(int(p) for p in player_ids)
    ratings = {p: 0.0 for p in players}
    edges: list[tuple[int, int, float]] = []
    for (p_lo, p_hi), st in pair_stats.items():
        if st.n_decisive < min_decisive:
            continue
        if p_lo not in ratings or p_hi not in ratings:
            continue
        score_lo = st.only_lo / st.n_decisive
        edges.append((p_lo, p_hi, score_lo))

    for _ in range(n_iter):
        for p_lo, p_hi, score_lo in edges:
            r_lo, r_hi = ratings[p_lo], ratings[p_hi]
            exp_lo = 1.0 / (1.0 + 10.0 ** ((r_hi - r_lo) / 400.0))
            ratings[p_lo] = r_lo + K * (score_lo - exp_lo)
            ratings[p_hi] = r_hi + K * ((1.0 - score_lo) - (1.0 - exp_lo))

    return ratings


def pair_stat_to_dict(
    st: PairStat,
    *,
    theta: np.ndarray,
    maps: IndexMaps,
    delta_theta_min: float = 0.3,
) -> dict[str, Any]:
    p_lo, p_hi = st.p_lo, st.p_hi
    pid_lo = maps.idx_to_player_id[p_lo]
    pid_hi = maps.idx_to_player_id[p_hi]
    d_theta = float(theta[p_lo] - theta[p_hi])
    nd = st.n_decisive
    decisive_rate = abs(st.only_lo - st.only_hi) / nd if nd else 0.0
    tm_lo = st.avg_teammate_theta_lo()
    tm_hi = st.avg_teammate_theta_hi()
    ctx_delta = (tm_lo - tm_hi) if tm_lo is not None and tm_hi is not None else None

    top_tourneys = sorted(
        st.tourney_decisive.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    return {
        "player_id_a": pid_lo,
        "player_id_b": pid_hi,
        "theta_a": float(theta[p_lo]),
        "theta_b": float(theta[p_hi]),
        "n_shared": st.n_shared,
        "n_decisive": nd,
        "only_a": st.only_lo,
        "only_b": st.only_hi,
        "both_taken": st.both_taken,
        "neither": st.neither,
        "duel_score": st.duel_score_lo(),
        "delta_theta": d_theta,
        "decisive_rate_delta": decisive_rate,
        "n_same_team": st.n_same_team,
        "avg_team_theta_a": st.avg_team_theta_lo(),
        "avg_team_theta_b": st.avg_team_theta_hi(),
        "avg_teammate_theta_a": tm_lo,
        "avg_teammate_theta_b": tm_hi,
        "team_context_delta": ctx_delta,
        "context_explains": context_explains(st, delta_theta=delta_theta_min),
        "top_tourney_deltas": top_tourneys,
    }
