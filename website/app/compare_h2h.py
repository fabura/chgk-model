"""Pairwise head-to-head question stats for the /compare page."""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from typing import Any

from . import db

_FIELD_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("лёгкие (поле ≥ 75%)", 0.75, None),
    ("средние (55–75%)", 0.55, 0.75),
    ("сложные (35–55%)", 0.35, 0.55),
    ("очень сложные (поле < 35%)", 0.0, 0.35),
)

_MIN_EDITOR_N = 15


def _has_points_mask_column() -> bool:
    rows = db.query("PRAGMA table_info('team_games')")
    return any(r["name"] == "points_mask" for r in rows)


def _field_bucket(field_rate: float | None) -> str | None:
    if field_rate is None:
        return None
    for label, lo, hi in _FIELD_BUCKETS:
        if hi is None and field_rate >= lo:
            return label
        if hi is not None and lo <= field_rate < hi:
            return label
    return None


def _parse_editors(blob: str | None) -> list[str]:
    if not blob or blob in ("[]", ""):
        return []
    try:
        items = json.loads(blob)
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            n = (it.get("name") or "").strip()
            if n:
                out.append(n)
        elif isinstance(it, str) and it.strip():
            out.append(it.strip())
    return out


def _load_player_slots(player_ids: list[int]) -> dict[int, dict[tuple[int, int], int]]:
    """Map player_id -> {(tournament_id, q_idx): taken}."""
    if not player_ids or not _has_points_mask_column():
        return {}
    ph = ",".join("?" * len(player_ids))
    rows = db.query(
        f"""
        SELECT pg.player_id, pg.tournament_id, tg.points_mask
        FROM player_games pg
        JOIN team_games tg
          ON tg.tournament_id = pg.tournament_id AND tg.team_id = pg.team_id
        WHERE pg.player_id IN ({ph})
          AND tg.has_breakdown = TRUE
          AND tg.points_mask IS NOT NULL
          AND LENGTH(TRIM(tg.points_mask)) > 0
        """,
        player_ids,
    )
    out: dict[int, dict[tuple[int, int], int]] = {pid: {} for pid in player_ids}
    for r in rows:
        pid = int(r["player_id"])
        tid = int(r["tournament_id"])
        mask = str(r["points_mask"])
        for qi, ch in enumerate(mask):
            out[pid][(tid, qi)] = 1 if ch == "1" else 0
    return out


def _load_slot_meta(tids: set[int]) -> dict[tuple[int, int], dict[str, Any]]:
    if not tids:
        return {}
    ph = ",".join("?" * len(tids))
    rows = db.query(
        f"""
        SELECT
            qa.tournament_id,
            qa.q_in_tournament,
            CASE WHEN qa.n_obs > 0 THEN qa.n_taken::DOUBLE / qa.n_obs ELSE NULL END AS field_rate,
            t.type AS mode,
            q.editors_json
        FROM question_aliases qa
        JOIN tournaments t ON t.tournament_id = qa.tournament_id
        LEFT JOIN questions q ON q.canonical_idx = qa.canonical_idx
        WHERE qa.tournament_id IN ({ph})
          AND qa.n_obs >= 5
        """,
        list(tids),
    )
    meta: dict[tuple[int, int], dict[str, Any]] = {}
    for r in rows:
        key = (int(r["tournament_id"]), int(r["q_in_tournament"]))
        meta[key] = {
            "field_rate": r["field_rate"],
            "mode": str(r["mode"] or "offline"),
            "editors": _parse_editors(r.get("editors_json")),
        }
    return meta


def _bucket_stats() -> dict[str, dict[str, int]]:
    return {"n": 0, "a_takes": 0, "b_takes": 0, "both": 0}


def _pair_compare(
    pid_a: int,
    pid_b: int,
    slots_a: dict[tuple[int, int], int],
    slots_b: dict[tuple[int, int], int],
    meta: dict[tuple[int, int], dict[str, Any]],
    team_by_player_tid: dict[tuple[int, int], int],
) -> dict[str, Any]:
    common = sorted(set(slots_a) & set(slots_b))
    n = len(common)
    if n == 0:
        return {"n_common": 0}

    a_taken = sum(slots_a[k] for k in common)
    b_taken = sum(slots_b[k] for k in common)
    both = only_a = only_b = neither = n_same_team = 0
    by_field: dict[str, dict[str, int]] = defaultdict(_bucket_stats)
    by_mode: dict[str, dict[str, int]] = defaultdict(_bucket_stats)
    by_editor: dict[str, dict[str, int]] = defaultdict(_bucket_stats)

    tourney_scores: dict[int, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "a": 0, "b": 0}
    )

    for key in common:
        tid, _qi = key
        a, b = slots_a[key], slots_b[key]
        if a and b:
            both += 1
        elif a:
            only_a += 1
        elif b:
            only_b += 1
        else:
            neither += 1

        if team_by_player_tid.get((pid_a, tid)) == team_by_player_tid.get((pid_b, tid)):
            n_same_team += 1

        tourney_scores[tid]["n"] += 1
        tourney_scores[tid]["a"] += a
        tourney_scores[tid]["b"] += b

        m = meta.get(key)
        if m:
            fr = m.get("field_rate")
            bk = _field_bucket(fr if fr is None else float(fr))
            if bk:
                d = by_field[bk]
                d["n"] += 1
                d["a_takes"] += a
                d["b_takes"] += b
                d["both"] += int(a and b)
            mode = m.get("mode") or "offline"
            d = by_mode[mode]
            d["n"] += 1
            d["a_takes"] += a
            d["b_takes"] += b
            d["both"] += int(a and b)
            eds = m.get("editors") or []
            for ed in eds:
                d = by_editor[ed]
                d["n"] += 1
                d["a_takes"] += a
                d["b_takes"] += b
                d["both"] += int(a and b)

    n_tournaments = len({k[0] for k in common})
    wins_a = wins_b = ties = 0
    for sc in tourney_scores.values():
        if sc["n"] == 0:
            continue
        if sc["a"] > sc["b"]:
            wins_a += 1
        elif sc["b"] > sc["a"]:
            wins_b += 1
        else:
            ties += 1

    def _fmt_buckets(raw: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
        order = {
            "лёгкие (поле ≥ 75%)": 0,
            "средние (55–75%)": 1,
            "сложные (35–55%)": 2,
            "очень сложные (поле < 35%)": 3,
            "offline": 0,
            "sync": 1,
            "async": 2,
        }
        out = []
        for name, d in raw.items():
            if d["n"] == 0:
                continue
            ar = d["a_takes"] / d["n"]
            br = d["b_takes"] / d["n"]
            out.append(
                {
                    "name": name,
                    "n": d["n"],
                    "a_takes": d["a_takes"],
                    "b_takes": d["b_takes"],
                    "both": d["both"],
                    "a_rate": ar,
                    "b_rate": br,
                    "delta": ar - br,
                }
            )
        out.sort(key=lambda x: order.get(x["name"], 99))
        return out

    editors = [
        {
            "name": name,
            "n": d["n"],
            "a_rate": d["a_takes"] / d["n"],
            "b_rate": d["b_takes"] / d["n"],
            "delta": d["a_takes"] / d["n"] - d["b_takes"] / d["n"],
        }
        for name, d in by_editor.items()
        if d["n"] >= _MIN_EDITOR_N
    ]
    editors_a = sorted(editors, key=lambda x: x["delta"], reverse=True)[:8]
    editors_b = sorted(editors, key=lambda x: x["delta"])[:8]

    return {
        "n_common": n,
        "n_tournaments": n_tournaments,
        "a_taken": a_taken,
        "b_taken": b_taken,
        "a_not_taken": n - a_taken,
        "b_not_taken": n - b_taken,
        "a_rate": a_taken / n,
        "b_rate": b_taken / n,
        "delta_takes": a_taken - b_taken,
        "delta_rate": (a_taken - b_taken) / n,
        "both_taken": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "n_same_team": n_same_team,
        "tournament_wins_a": wins_a,
        "tournament_wins_b": wins_b,
        "tournament_ties": ties,
        "by_field": _fmt_buckets(by_field),
        "by_mode": _fmt_buckets(by_mode),
        "editors_a_better": editors_a,
        "editors_b_better": editors_b,
    }


def compute_pairwise_head_to_head(
    players: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Return (pairwise stats, points_mask_available).

    Each entry has player_a / player_b dicts plus comparison fields.
    """
    if len(players) < 2:
        return [], _has_points_mask_column()

    if not _has_points_mask_column():
        return [], False

    player_ids = [int(p["player_id"]) for p in players]
    slots_by_player = _load_player_slots(player_ids)

    all_tids: set[int] = set()
    for slots in slots_by_player.values():
        all_tids.update(k[0] for k in slots)
    meta = _load_slot_meta(all_tids)

    ph = ",".join("?" * len(player_ids))
    team_rows = db.query(
        f"""
        SELECT player_id, tournament_id, team_id
        FROM player_games
        WHERE player_id IN ({ph})
        """,
        player_ids,
    )
    team_by_player_tid = {
        (int(r["player_id"]), int(r["tournament_id"])): int(r["team_id"])
        for r in team_rows
    }

    name_by_id = {int(p["player_id"]): p for p in players}
    pairs: list[dict[str, Any]] = []
    for pid_a, pid_b in combinations(player_ids, 2):
        stats = _pair_compare(
            pid_a,
            pid_b,
            slots_by_player.get(pid_a, {}),
            slots_by_player.get(pid_b, {}),
            meta,
            team_by_player_tid,
        )
        if stats.get("n_common", 0) == 0:
            continue
        pa, pb = name_by_id[pid_a], name_by_id[pid_b]
        pairs.append(
            {
                "player_a": pa,
                "player_b": pb,
                **stats,
            }
        )
    return pairs, True
