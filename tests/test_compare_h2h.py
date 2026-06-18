"""Tests for pairwise compare head-to-head logic."""
from __future__ import annotations

from website.app.compare_h2h import _field_bucket, _pair_compare


def test_field_bucket():
    assert _field_bucket(0.80) == "лёгкие (поле ≥ 75%)"
    assert _field_bucket(0.60) == "средние (55–75%)"
    assert _field_bucket(0.40) == "сложные (35–55%)"
    assert _field_bucket(0.20) == "очень сложные (поле < 35%)"
    assert _field_bucket(None) is None


def test_pair_compare_basic():
    slots_a = {(1, 0): 1, (1, 1): 0, (2, 0): 1}
    slots_b = {(1, 0): 1, (1, 1): 1, (2, 0): 0}
    meta = {
        (1, 0): {"field_rate": 0.8, "mode": "sync", "editors": ["Ed A"]},
        (1, 1): {"field_rate": 0.3, "mode": "sync", "editors": ["Ed A"]},
        (2, 0): {"field_rate": 0.5, "mode": "offline", "editors": []},
    }
    team = {(10, 1): 100, (20, 1): 200, (10, 2): 101, (20, 2): 201}
    out = _pair_compare(10, 20, slots_a, slots_b, meta, team)
    assert out["n_common"] == 3
    assert out["a_taken"] == 2
    assert out["b_taken"] == 2
    assert out["both_taken"] == 1
    assert out["only_a"] == 1
    assert out["only_b"] == 1
    assert out["neither"] == 0
    assert out["tournament_wins_a"] == 1
    assert out["tournament_wins_b"] == 1
