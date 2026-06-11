#!/usr/bin/env python3
"""Offline tilt-resilience diagnostics for teams and players.

The diagnostic asks whether teams underperform *after* a short run of
"avoidable" misses: questions the model thought they were reasonably likely
to take.  It is intentionally read-only: it trains the current sequential
model once with per-observation predictions, reconstructs within-tournament
team question sequences, and writes CSVs under ``results/tilt_resilience/``.

Usage::

    python scripts/diagnostic_tilt_resilience.py --event-id 12826

Main caveat: continuous within-tour streaks require continuous team question
sequences.  The default ``--eval-scope all`` uses all pre-update predictions
for that reason.  ``--eval-scope holdout`` is available as a leakage-free
sanity check, but it sparsifies sequences and is not the default ranking mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached  # noqa: E402
from rating.backtest import backtest  # noqa: E402
from rating.engine import Config  # noqa: E402


@dataclass(frozen=True)
class ObsPoint:
    obs_idx: int
    game_idx: int
    tournament_id: int
    q_in_tournament: int
    p: float
    y: int
    team_theta: float
    roster_local: tuple[int, ...]
    roster_pid: tuple[int, ...]


@dataclass
class Sequence:
    game_idx: int
    tournament_id: int
    roster_local: tuple[int, ...]
    roster_pid: tuple[int, ...]
    q: np.ndarray
    p: np.ndarray
    y: np.ndarray
    team_theta: np.ndarray
    obs_idx: np.ndarray


@dataclass
class TiltEvent:
    tournament_id: int
    game_idx: int
    roster_pid: tuple[int, ...]
    trigger_q: int
    trigger_idx: int
    trigger_p_mean: float
    after_n: int
    after_expected: float
    after_actual: float
    after_residual: float
    next_p: float
    next_y: int
    next_residual: float
    team_theta_mean: float


@dataclass
class MetricAgg:
    n_events: int = 0
    n_after_obs: int = 0
    sum_after_residual: float = 0.0
    sum_after_expected: float = 0.0
    sum_after_actual: float = 0.0
    n_next: int = 0
    sum_next_residual: float = 0.0
    sum_next_y: float = 0.0
    sum_next_p: float = 0.0
    n_sequences: int = 0
    sum_max_streak: float = 0.0

    def add_event(self, ev: TiltEvent) -> None:
        self.n_events += 1
        self.n_after_obs += int(ev.after_n)
        self.sum_after_residual += float(ev.after_residual)
        self.sum_after_expected += float(ev.after_expected)
        self.sum_after_actual += float(ev.after_actual)
        self.n_next += 1
        self.sum_next_residual += float(ev.next_residual)
        self.sum_next_y += float(ev.next_y)
        self.sum_next_p += float(ev.next_p)

    def add_sequence(self, max_streak: int) -> None:
        self.n_sequences += 1
        self.sum_max_streak += float(max_streak)

    @property
    def mean_after_residual(self) -> float:
        if self.n_after_obs <= 0:
            return float("nan")
        return self.sum_after_residual / self.n_after_obs

    @property
    def mean_event_after_residual(self) -> float:
        if self.n_events <= 0:
            return float("nan")
        return self.sum_after_residual / self.n_events

    @property
    def recovery_rate(self) -> float:
        if self.n_next <= 0:
            return float("nan")
        return self.sum_next_y / self.n_next

    @property
    def recovery_expected(self) -> float:
        if self.n_next <= 0:
            return float("nan")
        return self.sum_next_p / self.n_next

    @property
    def recovery_residual(self) -> float:
        if self.n_next <= 0:
            return float("nan")
        return self.sum_next_residual / self.n_next

    @property
    def mean_max_streak(self) -> float:
        if self.n_sequences <= 0:
            return float("nan")
        return self.sum_max_streak / self.n_sequences


def _build_offsets(team_sizes: np.ndarray) -> np.ndarray:
    off = np.empty(len(team_sizes) + 1, dtype=np.int64)
    off[0] = 0
    np.cumsum(team_sizes.astype(np.int64), out=off[1:])
    return off


def _bucket_mode(raw: str) -> str:
    s = str(raw).lower()
    if "async" in s or "асинхрон" in s:
        return "async"
    if "sync" in s or "синхрон" in s:
        return "sync"
    return "offline"


def _q_in_tournament(maps, raw_qi: int) -> int:
    qids = getattr(maps, "idx_to_question_id", None)
    if qids is not None and raw_qi < len(qids):
        qid = qids[raw_qi]
        if isinstance(qid, tuple) and len(qid) >= 2:
            return int(qid[1])
    return int(raw_qi)


def _prediction_scope_mask(
    pred: dict[str, np.ndarray],
    maps,
    cfg: Config,
    *,
    eval_scope: str,
    test_fraction: float,
) -> np.ndarray:
    n = len(pred["pred_p"])
    if eval_scope == "all":
        return np.ones(n, dtype=bool)

    if eval_scope == "holdout":
        holdout = pred.get("is_holdout")
        if holdout is None:
            raise RuntimeError("predictions lack is_holdout; cannot use --eval-scope holdout")
        mask = holdout.astype(bool)
        if not mask.any():
            raise RuntimeError(
                "holdout scope selected, but Config.holdout_obs_fraction is zero "
                "or no held-out predictions were collected"
            )
        return mask

    if eval_scope != "time-tail":
        raise ValueError(f"unknown eval scope: {eval_scope}")

    pred_game = pred["game_idx"]
    gdo = getattr(maps, "game_date_ordinal", None)
    all_games = np.unique(pred_game)
    if gdo is not None:
        known = all_games[
            np.array([int(gdo[g]) >= 0 for g in all_games], dtype=bool)
        ]
    else:
        known = np.array([], dtype=np.int32)
    if len(known) >= 2:
        ordered = known[np.argsort(np.array([int(gdo[g]) for g in known]))]
    else:
        ordered = np.sort(all_games)
    n_test = max(1, int(len(ordered) * float(test_fraction)))
    test_games = set(int(g) for g in ordered[-n_test:])
    return np.fromiter((int(g) in test_games for g in pred_game), count=n, dtype=bool)


def _try_open_duckdb(path: Path):
    if not path.exists():
        print(f"[duckdb] not found: {path}")
        return None
    try:
        import duckdb

        return duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        print(f"[duckdb] unavailable: {exc}")
        return None


def _fetch_tournament_titles(con, tournament_ids: Iterable[int]) -> dict[int, dict]:
    tids = sorted({int(t) for t in tournament_ids})
    if con is None or not tids:
        return {}
    try:
        rows = con.execute(
            "SELECT tournament_id, title, type, start_date "
            "FROM tournaments WHERE tournament_id = ANY(?)",
            [tids],
        ).fetchall()
    except Exception as exc:
        print(f"[duckdb] tournament lookup failed: {exc}")
        return {}
    return {
        int(r[0]): {
            "title": r[1] or "",
            "type": r[2] or "",
            "start_date": str(r[3]) if r[3] is not None else "",
        }
        for r in rows
    }


def _fetch_player_names(con, player_ids: Iterable[int]) -> dict[int, str]:
    pids = sorted({int(p) for p in player_ids})
    if con is None or not pids:
        return {}
    try:
        rows = con.execute(
            "SELECT player_id, COALESCE(last_name,'') || ' ' || "
            "COALESCE(first_name,'') AS name "
            "FROM players WHERE player_id = ANY(?)",
            [pids],
        ).fetchall()
    except Exception as exc:
        print(f"[duckdb] player lookup failed: {exc}")
        return {}
    return {int(r[0]): (r[1] or "").strip() for r in rows}


def _excluded_tournaments_by_title(con, terms: list[str]) -> set[int]:
    if con is None or not terms:
        return set()
    clauses = []
    params: list[str] = []
    for term in terms:
        t = term.strip().lower()
        if not t:
            continue
        clauses.append("LOWER(title) LIKE ?")
        params.append(f"%{t}%")
    if not clauses:
        return set()
    try:
        rows = con.execute(
            "SELECT tournament_id FROM tournaments WHERE " + " OR ".join(clauses),
            params,
        ).fetchall()
    except Exception as exc:
        print(f"[duckdb] title exclusion lookup failed: {exc}")
        return set()
    return {int(r[0]) for r in rows}


def _build_sequences(
    arrays: dict[str, np.ndarray],
    maps,
    pred: dict[str, np.ndarray],
    *,
    scope_mask: np.ndarray,
    mode: str,
    min_questions: int,
    min_roster_size: int,
    exclude_tournament_ids: set[int],
) -> list[Sequence]:
    q_idx = arrays["q_idx"].astype(np.int64)
    team_sizes = arrays["team_sizes"].astype(np.int64)
    pflat = arrays["player_indices_flat"].astype(np.int64)
    offsets = _build_offsets(team_sizes)
    game_types = getattr(maps, "game_type", None)
    idx_to_game_id = getattr(maps, "idx_to_game_id", [])
    pid_by_local = np.asarray(maps.idx_to_player_id, dtype=np.int64)

    pred_p = pred["pred_p"].astype(np.float64)
    pred_y = pred["actual_y"].astype(np.int8)
    pred_g = pred["game_idx"].astype(np.int64)
    pred_obs = pred["obs_idx"].astype(np.int64)
    pred_th = pred.get("team_theta_mean")
    if pred_th is None:
        pred_th = np.zeros(len(pred_p), dtype=np.float64)
    else:
        pred_th = pred_th.astype(np.float64)

    grouped: dict[tuple[int, tuple[int, ...]], list[ObsPoint]] = defaultdict(list)
    n_kept = 0
    for j in np.where(scope_mask)[0]:
        obs = int(pred_obs[j])
        g = int(pred_g[j])
        raw_type = game_types[g] if game_types is not None and g < len(game_types) else "offline"
        mode_bucket = _bucket_mode(str(raw_type))
        if mode != "all" and mode_bucket != mode:
            continue
        s, e = int(offsets[obs]), int(offsets[obs + 1])
        if e <= s:
            continue
        roster_local = tuple(sorted(int(x) for x in pflat[s:e]))
        roster_pid = tuple(int(pid_by_local[x]) for x in roster_local)
        raw_qi = int(q_idx[obs])
        tid = int(idx_to_game_id[g]) if g < len(idx_to_game_id) else g
        if tid in exclude_tournament_ids:
            continue
        if len(roster_local) < min_roster_size:
            continue
        grouped[(g, roster_local)].append(
            ObsPoint(
                obs_idx=obs,
                game_idx=g,
                tournament_id=tid,
                q_in_tournament=_q_in_tournament(maps, raw_qi),
                p=float(pred_p[j]),
                y=int(pred_y[j]),
                team_theta=float(pred_th[j]),
                roster_local=roster_local,
                roster_pid=roster_pid,
            )
        )
        n_kept += 1

    sequences: list[Sequence] = []
    for (g, roster_local), pts in grouped.items():
        if len(pts) < min_questions:
            continue
        pts.sort(key=lambda x: (x.q_in_tournament, x.obs_idx))
        # Duplicate q positions should not happen for one team, but keeping the
        # stable obs_idx tie-breaker makes the diagnostic deterministic.
        sequences.append(
            Sequence(
                game_idx=g,
                tournament_id=pts[0].tournament_id,
                roster_local=roster_local,
                roster_pid=pts[0].roster_pid,
                q=np.asarray([p.q_in_tournament for p in pts], dtype=np.int32),
                p=np.asarray([p.p for p in pts], dtype=np.float64),
                y=np.asarray([p.y for p in pts], dtype=np.int8),
                team_theta=np.asarray([p.team_theta for p in pts], dtype=np.float64),
                obs_idx=np.asarray([p.obs_idx for p in pts], dtype=np.int64),
            )
        )
    print(
        f"[sequences] kept {n_kept:,} obs -> {len(sequences):,} "
        f"team/tournament sequences "
        f"(mode={mode}, min_q={min_questions}, min_roster={min_roster_size})"
    )
    return sequences


def _max_avoidable_streak(y: np.ndarray, p: np.ndarray, threshold: float) -> int:
    cur = 0
    best = 0
    for yi, pi in zip(y, p):
        if int(yi) == 0 and float(pi) >= threshold:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _events_for_sequence(
    seq: Sequence,
    *,
    y_override: Optional[np.ndarray],
    avoidable_p: float,
    streak_n: int,
    after_k: int,
    collect_events: bool,
) -> tuple[MetricAgg, dict[str, MetricAgg], list[TiltEvent]]:
    y = seq.y if y_override is None else y_override.astype(np.int8)
    p = seq.p
    agg = MetricAgg()
    recovery: dict[str, MetricAgg] = defaultdict(MetricAgg)
    events: list[TiltEvent] = []

    max_streak = _max_avoidable_streak(y, p, avoidable_p)
    agg.add_sequence(max_streak)

    cur = 0
    for i in range(len(y) - 1):
        if int(y[i]) == 0 and float(p[i]) >= avoidable_p:
            cur += 1
        else:
            cur = 0

        bucket = "3+" if cur >= 3 else str(cur)
        rec = recovery[bucket]
        rec.n_next += 1
        rec.sum_next_y += float(y[i + 1])
        rec.sum_next_p += float(p[i + 1])
        rec.sum_next_residual += float(y[i + 1]) - float(p[i + 1])

        if cur != streak_n:
            continue
        lo = i + 1
        hi = min(len(y), lo + after_k)
        if hi <= lo:
            continue
        after_y = y[lo:hi].astype(np.float64)
        after_p = p[lo:hi].astype(np.float64)
        ev = TiltEvent(
            tournament_id=seq.tournament_id,
            game_idx=seq.game_idx,
            roster_pid=seq.roster_pid,
            trigger_q=int(seq.q[i]),
            trigger_idx=i,
            trigger_p_mean=float(p[i - streak_n + 1:i + 1].mean()),
            after_n=int(hi - lo),
            after_expected=float(after_p.sum()),
            after_actual=float(after_y.sum()),
            after_residual=float((after_y - after_p).sum()),
            next_p=float(p[lo]),
            next_y=int(y[lo]),
            next_residual=float(y[lo]) - float(p[lo]),
            team_theta_mean=float(seq.team_theta.mean()),
        )
        agg.add_event(ev)
        if collect_events:
            events.append(ev)
    return agg, recovery, events


def _merge_agg(dst: MetricAgg, src: MetricAgg) -> None:
    dst.n_events += src.n_events
    dst.n_after_obs += src.n_after_obs
    dst.sum_after_residual += src.sum_after_residual
    dst.sum_after_expected += src.sum_after_expected
    dst.sum_after_actual += src.sum_after_actual
    dst.n_next += src.n_next
    dst.sum_next_residual += src.sum_next_residual
    dst.sum_next_y += src.sum_next_y
    dst.sum_next_p += src.sum_next_p
    dst.n_sequences += src.n_sequences
    dst.sum_max_streak += src.sum_max_streak


def _compute_observed(
    sequences: list[Sequence],
    *,
    avoidable_p: float,
    streak_n: int,
    after_k: int,
) -> tuple[MetricAgg, dict[str, MetricAgg], list[TiltEvent]]:
    global_agg = MetricAgg()
    recovery: dict[str, MetricAgg] = defaultdict(MetricAgg)
    events: list[TiltEvent] = []
    for seq in sequences:
        seq_agg, seq_rec, seq_events = _events_for_sequence(
            seq,
            y_override=None,
            avoidable_p=avoidable_p,
            streak_n=streak_n,
            after_k=after_k,
            collect_events=True,
        )
        _merge_agg(global_agg, seq_agg)
        for k, v in seq_rec.items():
            _merge_agg(recovery[k], v)
        events.extend(seq_events)
    return global_agg, recovery, events


def _compute_null(
    sequences: list[Sequence],
    *,
    avoidable_p: float,
    streak_n: int,
    after_k: int,
    n_sims: int,
    seed: int,
) -> dict[str, float]:
    if n_sims <= 0:
        return {}
    rng = np.random.default_rng(seed)
    after_means: list[float] = []
    max_streaks: list[float] = []
    n_events: list[int] = []
    for _ in range(n_sims):
        agg = MetricAgg()
        for seq in sequences:
            y_sim = (rng.random(len(seq.p)) < seq.p).astype(np.int8)
            seq_agg, _, _ = _events_for_sequence(
                seq,
                y_override=y_sim,
                avoidable_p=avoidable_p,
                streak_n=streak_n,
                after_k=after_k,
                collect_events=False,
            )
            _merge_agg(agg, seq_agg)
        after_means.append(agg.mean_after_residual)
        max_streaks.append(agg.mean_max_streak)
        n_events.append(agg.n_events)
    return {
        "null_sims": float(n_sims),
        "null_after_residual_mean": float(np.nanmean(after_means)),
        "null_after_residual_sd": float(np.nanstd(after_means, ddof=1))
        if n_sims > 1 else float("nan"),
        "null_max_streak_mean": float(np.nanmean(max_streaks)),
        "null_events_mean": float(np.mean(n_events)),
    }


def _agg_by_team_and_player(
    events: list[TiltEvent],
    *,
    max_player_list: int = 0,
) -> tuple[dict[tuple[int, ...], MetricAgg], dict[int, MetricAgg]]:
    team_aggs: dict[tuple[int, ...], MetricAgg] = defaultdict(MetricAgg)
    player_aggs: dict[int, MetricAgg] = defaultdict(MetricAgg)
    for ev in events:
        team_aggs[ev.roster_pid].add_event(ev)
        for pid in ev.roster_pid:
            player_aggs[int(pid)].add_event(ev)
    return team_aggs, player_aggs


def _fmt_float(x: float, digits: int = 6) -> str:
    if x != x or math.isinf(x):
        return ""
    return f"{x:.{digits}f}"


def _write_summary(
    out_dir: Path,
    observed: MetricAgg,
    recovery: dict[str, MetricAgg],
    null: dict[str, float],
    args: argparse.Namespace,
) -> None:
    null_mean = null.get("null_after_residual_mean", float("nan"))
    null_sd = null.get("null_after_residual_sd", float("nan"))
    z = (
        (observed.mean_after_residual - null_mean) / null_sd
        if null_sd == null_sd and null_sd > 0
        else float("nan")
    )
    rows = [
        {
            "metric": "aftershock",
            "slice": "global",
            "n_sequences": observed.n_sequences,
            "n_events": observed.n_events,
            "n_after_obs": observed.n_after_obs,
            "mean_after_residual": _fmt_float(observed.mean_after_residual),
            "mean_event_after_residual": _fmt_float(observed.mean_event_after_residual),
            "recovery_rate": _fmt_float(observed.recovery_rate),
            "recovery_expected": _fmt_float(observed.recovery_expected),
            "recovery_residual": _fmt_float(observed.recovery_residual),
            "mean_max_streak": _fmt_float(observed.mean_max_streak),
            "null_after_residual_mean": _fmt_float(null_mean),
            "null_after_residual_sd": _fmt_float(null_sd),
            "z_vs_null": _fmt_float(z),
        }
    ]
    for bucket in ("0", "1", "2", "3+"):
        rec = recovery.get(bucket, MetricAgg())
        rows.append(
            {
                "metric": "recovery_next_question",
                "slice": f"streak_{bucket}",
                "n_sequences": "",
                "n_events": "",
                "n_after_obs": "",
                "mean_after_residual": "",
                "mean_event_after_residual": "",
                "recovery_rate": _fmt_float(rec.recovery_rate),
                "recovery_expected": _fmt_float(rec.recovery_expected),
                "recovery_residual": _fmt_float(rec.recovery_residual),
                "mean_max_streak": "",
                "null_after_residual_mean": "",
                "null_after_residual_sd": "",
                "z_vs_null": "",
            }
        )
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "mode": args.mode,
        "eval_scope": args.eval_scope,
        "avoidable_p": args.avoidable_p,
        "streak": args.streak,
        "after": args.after,
        "min_questions": args.min_questions,
        "null": null,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {path}")


def _rows_for_team_aggs(
    team_aggs: dict[tuple[int, ...], MetricAgg],
    *,
    min_events: int,
) -> list[dict]:
    rows: list[dict] = []
    for roster, agg in team_aggs.items():
        if agg.n_events < min_events:
            continue
        rows.append(
            {
                "resilience_score": agg.mean_after_residual,
                "tilt_pp": -100.0 * agg.mean_after_residual,
                "n_events": agg.n_events,
                "n_after_obs": agg.n_after_obs,
                "recovery_rate": agg.recovery_rate,
                "recovery_expected": agg.recovery_expected,
                "recovery_residual": agg.recovery_residual,
                "roster_size": len(roster),
                "roster_player_ids": " ".join(str(p) for p in roster),
                "roster_set": set(roster),
            }
        )
    rows.sort(key=lambda r: (r["resilience_score"], r["n_events"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def _write_team_history(out_dir: Path, rows: list[dict]) -> None:
    path = out_dir / "teams_history.csv"
    fieldnames = [
        "rank",
        "resilience_score",
        "tilt_pp",
        "n_events",
        "n_after_obs",
        "recovery_rate",
        "recovery_expected",
        "recovery_residual",
        "roster_size",
        "roster_player_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({
                k: _fmt_float(row[k]) if isinstance(row.get(k), float) else row.get(k, "")
                for k in fieldnames
            })
    print(f"[write] {path}")


def _write_players(
    out_dir: Path,
    player_aggs: dict[int, MetricAgg],
    names: dict[int, str],
    *,
    min_events: int,
) -> list[dict]:
    rows: list[dict] = []
    for pid, agg in player_aggs.items():
        if agg.n_events < min_events:
            continue
        rows.append(
            {
                "player_id": int(pid),
                "name": names.get(int(pid), ""),
                "resilience_score": agg.mean_after_residual,
                "tilt_pp": -100.0 * agg.mean_after_residual,
                "n_events": agg.n_events,
                "n_after_obs": agg.n_after_obs,
                "recovery_rate": agg.recovery_rate,
                "recovery_expected": agg.recovery_expected,
                "recovery_residual": agg.recovery_residual,
            }
        )
    rows.sort(key=lambda r: (r["resilience_score"], r["n_events"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    path = out_dir / "players.csv"
    fieldnames = [
        "rank",
        "player_id",
        "name",
        "resilience_score",
        "tilt_pp",
        "n_events",
        "n_after_obs",
        "recovery_rate",
        "recovery_expected",
        "recovery_residual",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({
                k: _fmt_float(row[k]) if isinstance(row.get(k), float) else row.get(k, "")
                for k in fieldnames
            })
    print(f"[write] {path}")
    return rows


def _write_events(
    out_dir: Path,
    events: list[TiltEvent],
    titles: dict[int, dict],
    *,
    max_events: int,
) -> None:
    rows: list[dict] = []
    for ev in events:
        meta = titles.get(ev.tournament_id, {})
        rows.append(
            {
                "tournament_id": ev.tournament_id,
                "title": meta.get("title", ""),
                "type": meta.get("type", ""),
                "start_date": meta.get("start_date", ""),
                "trigger_q": ev.trigger_q,
                "trigger_p_mean": ev.trigger_p_mean,
                "after_n": ev.after_n,
                "after_expected": ev.after_expected,
                "after_actual": ev.after_actual,
                "after_residual": ev.after_residual,
                "after_residual_per_q": ev.after_residual / max(ev.after_n, 1),
                "next_p": ev.next_p,
                "next_y": ev.next_y,
                "next_residual": ev.next_residual,
                "team_theta_mean": ev.team_theta_mean,
                "roster_player_ids": " ".join(str(p) for p in ev.roster_pid),
            }
        )
    rows.sort(key=lambda r: r["after_residual_per_q"])
    if max_events > 0:
        rows = rows[:max_events]
    path = out_dir / "events.csv"
    fieldnames = [
        "tournament_id",
        "title",
        "type",
        "start_date",
        "trigger_q",
        "trigger_p_mean",
        "after_n",
        "after_expected",
        "after_actual",
        "after_residual",
        "after_residual_per_q",
        "next_p",
        "next_y",
        "next_residual",
        "team_theta_mean",
        "roster_player_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({
                k: _fmt_float(row[k]) if isinstance(row.get(k), float) else row.get(k, "")
                for k in fieldnames
            })
    print(f"[write] {path}")


def _player_display_name(player: dict) -> str:
    surname = player.get("surname") or player.get("lastName") or ""
    name = player.get("name") or player.get("firstName") or ""
    patronymic = player.get("patronymic") or ""
    full = " ".join(x for x in [surname, name, patronymic] if x).strip()
    return full


def _fetch_event_rosters(event_id: int) -> tuple[dict, list[dict]]:
    from website.app import forecast_api

    meta = forecast_api.get_tournament(int(event_id))
    rosters = forecast_api.get_rosters(int(event_id))
    if not isinstance(rosters, list) or not rosters:
        raise RuntimeError(f"no rosters returned by rating API for event {event_id}")
    return meta, rosters


def _write_event_ranking(
    out_dir: Path,
    *,
    event_id: int,
    player_rows: list[dict],
    team_rows: list[dict],
    min_direct_overlap: int,
    min_direct_events: int,
) -> None:
    try:
        meta, roster_payload = _fetch_event_rosters(event_id)
    except Exception as exc:
        path = out_dir / f"event_{event_id}_teams.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["error"])
            w.writerow([f"Could not fetch event rosters: {exc}"])
        print(f"[event] could not fetch rosters for {event_id}: {exc}")
        print(f"[write] {path}")
        return

    player_by_pid = {int(r["player_id"]): r for r in player_rows}
    history_teams = [
        row for row in team_rows
        if int(row["n_events"]) >= min_direct_events
    ]
    event_rows: list[dict] = []
    for tr in roster_payload:
        team = tr.get("team") or {}
        members = tr.get("teamMembers") or []
        pids: list[int] = []
        member_names: list[str] = []
        unknown = 0
        player_score_sum = 0.0
        player_weight_sum = 0.0
        scored_players = 0
        player_event_sum = 0

        for tm in members:
            pl = tm.get("player") or {}
            pid = pl.get("id")
            if not isinstance(pid, int):
                continue
            pids.append(int(pid))
            member_names.append(_player_display_name(pl) or str(pid))
            prow = player_by_pid.get(int(pid))
            if prow is None:
                unknown += 1
                continue
            n_events = int(prow["n_events"])
            weight = min(50.0, math.sqrt(max(n_events, 1)))
            player_score_sum += float(prow["resilience_score"]) * weight
            player_weight_sum += weight
            scored_players += 1
            player_event_sum += n_events

        player_score = (
            player_score_sum / player_weight_sum
            if player_weight_sum > 0
            else float("nan")
        )

        pset = set(pids)
        direct_score_sum = 0.0
        direct_weight_sum = 0.0
        direct_overlap_events = 0.0
        best_overlap = 0
        for hrow in history_teams:
            hset = hrow.get("roster_set") or set()
            overlap = len(pset & hset)
            if overlap < min_direct_overlap:
                continue
            share = overlap / max(len(pset), 1)
            if share < 0.35:
                continue
            weight = float(hrow["n_events"]) * share
            direct_score_sum += float(hrow["resilience_score"]) * weight
            direct_weight_sum += weight
            direct_overlap_events += weight
            best_overlap = max(best_overlap, overlap)

        direct_score_raw = (
            direct_score_sum / direct_weight_sum
            if direct_weight_sum > 0
            else float("nan")
        )
        direct_score = (
            direct_score_raw
            if direct_weight_sum >= min_direct_events
            else float("nan")
        )
        if direct_score == direct_score and player_score == player_score:
            combined = 0.6 * direct_score + 0.4 * player_score
        elif direct_score == direct_score:
            combined = direct_score
        else:
            combined = player_score

        event_rows.append(
            {
                "team_id": team.get("id", ""),
                "team_name": team.get("name") or "",
                "town": (team.get("town") or {}).get("name") if isinstance(team.get("town"), dict) else "",
                "resilience_score": combined,
                "tilt_pp": -100.0 * combined if combined == combined else float("nan"),
                "player_score": player_score,
                "direct_score": direct_score,
                "scored_players": scored_players,
                "roster_size": len(pids),
                "unknown_players": unknown,
                "player_event_sum": player_event_sum,
                "direct_overlap_events": direct_overlap_events,
                "best_overlap": best_overlap,
                "roster_player_ids": " ".join(str(p) for p in pids),
                "roster_names": "; ".join(member_names),
            }
        )

    event_rows.sort(
        key=lambda r: (
            -1 if r["resilience_score"] != r["resilience_score"] else 0,
            r["resilience_score"] if r["resilience_score"] == r["resilience_score"] else -999.0,
            r["player_event_sum"],
        ),
        reverse=True,
    )
    for i, row in enumerate(event_rows, start=1):
        row["rank"] = i

    path = out_dir / f"event_{event_id}_teams.csv"
    fieldnames = [
        "rank",
        "team_id",
        "team_name",
        "town",
        "resilience_score",
        "tilt_pp",
        "player_score",
        "direct_score",
        "scored_players",
        "roster_size",
        "unknown_players",
        "player_event_sum",
        "direct_overlap_events",
        "best_overlap",
        "roster_player_ids",
        "roster_names",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in event_rows:
            w.writerow({
                k: _fmt_float(row[k]) if isinstance(row.get(k), float) else row.get(k, "")
                for k in fieldnames
            })
    title = meta.get("name") or meta.get("title") or str(event_id)
    print(f"[event] {event_id}: {title}; teams={len(event_rows)}")
    print(f"[write] {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", "--cache", default="data.npz")
    ap.add_argument("--out_dir", default="results/tilt_resilience")
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--mode", choices=["offline", "sync", "async", "all"], default="offline")
    ap.add_argument(
        "--eval-scope",
        choices=["all", "holdout", "time-tail"],
        default="all",
        help="Which collected predictions to use for sequences.",
    )
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--holdout", type=float, default=None)
    ap.add_argument("--holdout-seed", type=int, default=None)
    ap.add_argument("--avoidable-p", type=float, default=0.60)
    ap.add_argument("--streak", type=int, default=2)
    ap.add_argument("--after", type=int, default=3)
    ap.add_argument("--min-questions", type=int, default=18)
    ap.add_argument("--min-roster-size", type=int, default=2)
    ap.add_argument(
        "--exclude-title-terms",
        default="онлайн,online,м-лига,m-лига,m-liga",
        help="Comma-separated lower-case title fragments to exclude as non-offline.",
    )
    ap.add_argument("--min-team-events", type=int, default=3)
    ap.add_argument("--min-player-events", type=int, default=8)
    ap.add_argument("--min-direct-overlap", type=int, default=3)
    ap.add_argument("--min-direct-events", type=int, default=3)
    ap.add_argument("--max-events", type=int, default=5000)
    ap.add_argument("--null-sims", type=int, default=50)
    ap.add_argument("--null-seed", type=int, default=20260608)
    ap.add_argument("--event-id", type=int, default=12826)
    ap.add_argument("--skip-event", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)
    con = _try_open_duckdb(Path(args.duckdb))
    exclude_terms = [x.strip() for x in args.exclude_title_terms.split(",") if x.strip()]
    exclude_tids = _excluded_tournaments_by_title(con, exclude_terms)
    if exclude_tids:
        print(
            f"[filter] excluding {len(exclude_tids):,} tournaments by title terms: "
            f"{', '.join(exclude_terms)}"
        )
    cfg = Config()
    if args.holdout is not None:
        cfg.holdout_obs_fraction = float(args.holdout)
    if args.holdout_seed is not None:
        cfg.holdout_seed = int(args.holdout_seed)

    print("[backtest] running current sequential model with predictions")
    metrics = backtest(arrays, maps, cfg, test_fraction=args.test_fraction, verbose=not args.quiet)
    result = metrics["result"]
    pred = result.predictions
    if pred is None or "obs_idx" not in pred:
        raise RuntimeError("predictions missing obs_idx; cannot align to cache arrays")

    scope = _prediction_scope_mask(
        pred,
        maps,
        cfg,
        eval_scope=args.eval_scope,
        test_fraction=args.test_fraction,
    )
    print(
        f"[scope] {args.eval_scope}: {int(scope.sum()):,}/{len(scope):,} predictions"
    )
    sequences = _build_sequences(
        arrays,
        maps,
        pred,
        scope_mask=scope,
        mode=args.mode,
        min_questions=args.min_questions,
        min_roster_size=args.min_roster_size,
        exclude_tournament_ids=exclude_tids,
    )
    if not sequences:
        raise RuntimeError("no sequences after filtering; relax --mode/--eval-scope/--min-questions")

    observed, recovery, events = _compute_observed(
        sequences,
        avoidable_p=args.avoidable_p,
        streak_n=args.streak,
        after_k=args.after,
    )
    print(
        f"[metrics] events={observed.n_events:,}, "
        f"after_obs={observed.n_after_obs:,}, "
        f"after_res/q={observed.mean_after_residual:+.4f}, "
        f"mean_max_streak={observed.mean_max_streak:.3f}"
    )

    print(f"[null] Bernoulli simulations: {args.null_sims}")
    null = _compute_null(
        sequences,
        avoidable_p=args.avoidable_p,
        streak_n=args.streak,
        after_k=args.after,
        n_sims=args.null_sims,
        seed=args.null_seed,
    )

    team_aggs, player_aggs = _agg_by_team_and_player(events)
    titles = _fetch_tournament_titles(con, [ev.tournament_id for ev in events])
    player_names = _fetch_player_names(con, player_aggs.keys())

    _write_summary(out_dir, observed, recovery, null, args)
    team_rows = _rows_for_team_aggs(team_aggs, min_events=args.min_team_events)
    _write_team_history(out_dir, team_rows)
    player_rows = _write_players(
        out_dir,
        player_aggs,
        player_names,
        min_events=args.min_player_events,
    )
    _write_events(out_dir, events, titles, max_events=args.max_events)

    if not args.skip_event and args.event_id:
        _write_event_ranking(
            out_dir,
            event_id=int(args.event_id),
            player_rows=player_rows,
            team_rows=team_rows,
            min_direct_overlap=int(args.min_direct_overlap),
            min_direct_events=int(args.min_direct_events),
        )
    if con is not None:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
