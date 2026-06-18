"""Tests for decisive H2H evaluation."""
from __future__ import annotations

import numpy as np

from data import IndexMaps, Sample, samples_to_arrays
from rating.h2h import (
    build_pair_stats_from_arrays,
    compute_duel_scores,
    compute_pairwise_concordance,
    pair_outcome,
)


def _make_maps(
    player_ids: list[int],
    question_ids: list[tuple[int, int]],
    game_ids: list[int] | None = None,
) -> IndexMaps:
    return IndexMaps(
        player_id_to_idx={p: i for i, p in enumerate(player_ids)},
        question_id_to_idx={q: i for i, q in enumerate(question_ids)},
        idx_to_player_id=player_ids,
        idx_to_question_id=question_ids,
        idx_to_game_id=game_ids or [1],
        game_type=np.array(["offline"], dtype=object),
    )


def _build_from_samples(samples: list[Sample], maps: IndexMaps) -> dict:
    arrays = samples_to_arrays(samples)
    return arrays


def test_decisive_only_not_counting_both_taken_as_decisive():
    # Players 10, 20, 30. One slot: team1 takes, team2 misses.
    maps = _make_maps([10, 20, 30], [(1, 0)], [1])
    samples = [
        Sample(question_idx=0, player_indices=[0, 1], taken=1, game_idx=0),
        Sample(question_idx=0, player_indices=[2], taken=0, game_idx=0),
    ]
    arrays = _build_from_samples(samples, maps)
    theta = np.array([1.0, 0.5, -0.5], dtype=np.float64)
    stats = build_pair_stats_from_arrays(arrays, maps, theta=theta, eligible=None)
    st = stats[(0, 2)]
    assert st.only_lo == 1
    assert st.only_hi == 0
    assert st.n_decisive == 1
    assert st.both_taken == 0


def test_same_team_excluded():
    maps = _make_maps([10, 20], [(1, 0)], [1])
    samples = [
        Sample(question_idx=0, player_indices=[0, 1], taken=1, game_idx=0),
    ]
    arrays = _build_from_samples(samples, maps)
    theta = np.array([1.0, 0.5], dtype=np.float64)
    stats = build_pair_stats_from_arrays(arrays, maps, theta=theta, eligible=None)
    assert (0, 1) not in stats


def test_both_taken_not_decisive():
    maps = _make_maps([10, 20], [(1, 0)], [1])
    samples = [
        Sample(question_idx=0, player_indices=[0], taken=1, game_idx=0),
        Sample(question_idx=0, player_indices=[1], taken=1, game_idx=0),
    ]
    arrays = _build_from_samples(samples, maps)
    theta = np.array([1.0, 0.5], dtype=np.float64)
    stats = build_pair_stats_from_arrays(arrays, maps, theta=theta, eligible=None)
    # both took — no taken×missed cross, pair not created
    assert (0, 1) not in stats


def test_pairwise_concordance():
    maps = _make_maps([10, 20, 30], [(1, 0), (1, 1)], [1])
    samples = [
        Sample(question_idx=0, player_indices=[0], taken=1, game_idx=0),
        Sample(question_idx=0, player_indices=[1], taken=0, game_idx=0),
        Sample(question_idx=1, player_indices=[0], taken=1, game_idx=0),
        Sample(question_idx=1, player_indices=[1], taken=0, game_idx=0),
        Sample(question_idx=0, player_indices=[2], taken=0, game_idx=0),
        Sample(question_idx=1, player_indices=[2], taken=1, game_idx=0),
    ]
    arrays = _build_from_samples(samples, maps)
    theta = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    stats = build_pair_stats_from_arrays(arrays, maps, theta=theta, eligible=None)
    st01 = stats[(0, 1)]
    assert st01.only_lo == 2
    assert pair_outcome(st01, min_decisive=2) == "win_lo"
    conc = compute_pairwise_concordance(
        stats, theta, min_shared=1, min_decisive=2
    )
    assert conc["n_pairs"] == 1.0
    assert conc["accuracy"] == 1.0


def test_duel_net_score_ordering():
    maps = _make_maps([10, 20, 30], [(1, q) for q in range(6)], [1])
    samples = []
    for q in range(5):
        samples.append(Sample(question_idx=q, player_indices=[0], taken=1, game_idx=0))
        samples.append(Sample(question_idx=q, player_indices=[1], taken=0, game_idx=0))
        samples.append(Sample(question_idx=q, player_indices=[2], taken=0, game_idx=0))
    # decisive slot: player 1 takes, player 2 misses
    samples.append(Sample(question_idx=5, player_indices=[1], taken=1, game_idx=0))
    samples.append(Sample(question_idx=5, player_indices=[2], taken=0, game_idx=0))
    arrays = _build_from_samples(samples, maps)
    theta = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    stats = build_pair_stats_from_arrays(arrays, maps, theta=theta, eligible=None)
    scores = compute_duel_scores(stats, [0, 1, 2], min_decisive=1)
    assert scores[0] > scores[1] > scores[2]
