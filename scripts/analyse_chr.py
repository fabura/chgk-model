#!/usr/bin/env python3
"""Retrospective analysis for ЧР tournaments using chgk-model parameters.

Maps the 2025 hybrid-iterative article metrics to the production noisy-OR
sequential model (θ, b, a).  For tournaments already baked into DuckDB /
seq.npz we use trained ``b`` and ``a``; for very recent events not yet in
the training cache we fall back to model-consistent ``b`` initialisation
from take rates (``QuestionState.init_from_take_rate``) and ``a ≡ 1``
(production ``freeze_log_a=True``).

**Mask sources (take / no-take matrix):**

- Default: ``api.rating.chgk.info`` ``points_mask`` per team.
- ``--xlsx data/КВРМ.xlsx`` (or auto-detect when the file exists): official
  II ЧР КВРМ расплюсовка.  Sheet ``Worksheet`` — header row 2
  (``Team ID``, ``Название``, ``Город``, ``Тур``, questions 1–15); data from
  row 3.  Six tours × 15 questions = 90 binary cells per team (``0``/``1``).
  Team metadata (``team_id``, roster, place) still comes from the API; only
  masks are replaced.  Names are matched to API teams (normalized exact match,
  then unique substring fallback).  Unmatched xlsx rows are reported; when
  ``--xlsx`` is set, API masks are also diffed and discrepancies printed.

Usage:
    python scripts/analyse_chr.py --tournament-id 12826
    python scripts/analyse_chr.py --tournament-id 12826 --compare 11749
    python scripts/analyse_chr.py --tournament-id 12826 --xlsx data/КВРМ.xlsx
    python scripts/analyse_chr.py --tournament-id 12826 --json
    python scripts/analyse_chr.py --tournament-id 12826 --reactions-json data/result.json
    python scripts/analyse_chr.py --tournament-id 12826 --top-comments 10
    python scripts/analyse_chr.py --tournament-id 12826 --plots
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import duckdb

from rating.io import load_results_npz
from rating.pack_calib import init_b_from_take_rate, pack_adjust_b, pack_b_gap
from rating.simulate import simulate_roster_on_pack
from rating_api.client import RatingApiClient

DEFAULT_DUCKDB = REPO_ROOT / "website/data/chgk.duckdb"
DEFAULT_SEQ = REPO_ROOT / "results/seq.npz"
DEFAULT_QUESTIONS_DB = Path(
    "/Users/fbr/Projects/personal/chgk-embedings/data/questions.db"
)
DEFAULT_REACTIONS_JSON = REPO_ROOT / "data/result.json"
DEFAULT_KVRM_XLSX = REPO_ROOT / "data/КВРМ.xlsx"

# Telegram emoji buckets for community sentiment (net = likes − dislikes).
_REACTION_POSITIVE = frozenset(
    "👍 ❤ 🔥 👏 🎉 😍 🥰 💯 👌 🙏 😁 🤩 ⚡ ✍ 💪 🏆 😎 🤝 💋 🫡 🕊 🦄 👀 🍾 ❤‍🔥".split()
)
_REACTION_NEGATIVE = frozenset(
    "👎 💩 🤮 😡 🤬 😤 🤡 🗿 🤢 😠 💀 🙄 😒 🤦 😬 🥴 🥱 😐 🤷‍♂ 🤷‍♀ 😭 💔 😨 😱 🤯".split()
)
_CHANNEL_DISCUSSION = "ЧР–2026. Обсуждение вопросов"
# Lightweight Russian keyword buckets for player comment themes (not ML).
_COMMENT_POSITIVE_KW = (
    "класс",
    "крут",
    "супер",
    "отличн",
    "понрав",
    "красив",
    "хорош",
    "приколь",
    "офиген",
)
_COMMENT_NEGATIVE_KW = (
    "отврат",
    "ужас",
    "неприят",
    "глуп",
    "бред",
    "не хотел",
    "плох",
    "разочар",
    "обидн",
)
_COMMENT_DEBATE_KW = (
    "незач",
    "зачёт",
    "зачет",
    "импорт",
    "несправедлив",
    "почему",
    "связь",
)


@dataclass
class TeamRow:
    team_id: int
    team_name: str
    position: float
    mask: str
    player_ids: list[int]
    score: int = 0
    strength: float = 0.0
    expected: float = 0.0
    theta_implied: float = float("nan")


@dataclass
class QuestionRow:
    q_index: int
    b: float
    a: float
    n_taken: int
    n_teams: int
    take_rate: float
    selectivity: float
    text: str = ""
    editor: str = ""
    sole_team: str = ""
    likes: int = 0
    dislikes: int = 0
    reaction_net: int = 0
    reaction_total: int = 0
    comment_count: int = 0


def _flatten_telegram_text(text: Any) -> str:
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text") or "")
            for part in text
        )
    return str(text or "")


def _score_reactions(reactions: Optional[list[dict[str, Any]]]) -> dict[str, int]:
    likes = dislikes = total = 0
    for row in reactions or []:
        emoji = str(row.get("emoji") or "")
        count = int(row.get("count") or 0)
        total += count
        if emoji in _REACTION_POSITIVE:
            likes += count
        elif emoji in _REACTION_NEGATIVE:
            dislikes += count
    return {
        "likes": likes,
        "dislikes": dislikes,
        "reaction_net": likes - dislikes,
        "reaction_total": total,
    }


def _is_user_message(msg: dict[str, Any]) -> bool:
    from_id = str(msg.get("from_id") or "")
    if from_id.startswith("user"):
        return True
    return str(msg.get("from") or "") != _CHANNEL_DISCUSSION


def _resolve_question_index(
    reply_id: int,
    q_msg_id: dict[int, int],
    by_id: dict[int, dict[str, Any]],
) -> Optional[int]:
    """Walk reply chain until a КВРМ question post is found."""
    seen: set[int] = set()
    while reply_id and reply_id not in seen:
        seen.add(reply_id)
        if reply_id in q_msg_id:
            return q_msg_id[reply_id]
        parent = by_id.get(reply_id)
        if not parent:
            return None
        nxt = parent.get("reply_to_message_id")
        if not isinstance(nxt, int):
            return None
        reply_id = nxt
    return None


def _comment_themes(text: str) -> list[str]:
    low = text.casefold()
    themes: list[str] = []
    if any(k in low for k in _COMMENT_POSITIVE_KW):
        themes.append("positive")
    if any(k in low for k in _COMMENT_NEGATIVE_KW):
        themes.append("negative")
    if any(k in low for k in _COMMENT_DEBATE_KW):
        themes.append("debate")
    return themes


def _load_kvrm_discussion_from_result_json(
    path: Path,
    *,
    tournament_id: int,
) -> tuple[dict[int, dict[str, int]], dict[int, list[dict[str, Any]]]]:
    """Load per-question Telegram reactions and player comments from export.

    Join keys (КВРМ block only — same state machine as reactions):

    - Question posts: text starts with ``Вопрос N.`` → ``q_index = N - 1``.
      Store ``message_id → q_index`` for comment threading.
    - Player comments: user messages with ``reply_to_message_id``; resolve the
      target question by walking the reply chain to a question post (direct
      replies and nested thread replies both count).
    - Reactions: emoji tallies on the question post itself (unchanged).

    Other disciplines (Своя игра, Эрудит-квартет) are skipped.  Export is
    II ЧР-2026 only.
    """
    if not path.exists():
        return {}, {}
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    messages = payload.get("messages") or []
    by_id = {m["id"]: m for m in messages if isinstance(m.get("id"), int)}

    reactions: dict[int, dict[str, int]] = {}
    q_msg_id: dict[int, int] = {}
    comments: dict[int, list[dict[str, Any]]] = defaultdict(list)

    mode: Optional[str] = None
    for msg in messages:
        text = _flatten_telegram_text(msg.get("text"))
        if "Командная викторина с раундами по минуте" in text:
            mode = "kvrm"
            continue
        if mode == "kvrm" and "Эрудит-квартет" in text and text.strip().startswith(
            "II ЧР"
        ):
            mode = "erudit"
            continue
        if mode == "erudit" and re.match(r"^Тур\s+3\.", text.strip()):
            mode = "kvrm2"
            continue
        if mode in ("kvrm", "kvrm2") and text.strip().startswith("Финал эрудит"):
            break
        if mode not in ("kvrm", "kvrm2"):
            continue
        match = re.match(r"^Вопрос\s+(\d+)\.", text)
        if match:
            msg_id = msg.get("id")
            if isinstance(msg_id, int):
                q_index = int(match.group(1)) - 1
                q_msg_id[msg_id] = q_index
                reactions[q_index] = _score_reactions(msg.get("reactions"))
            continue

    for msg in messages:
        reply_id = msg.get("reply_to_message_id")
        if not isinstance(reply_id, int) or not _is_user_message(msg):
            continue
        q_index = _resolve_question_index(reply_id, q_msg_id, by_id)
        if q_index is None:
            continue
        body = _flatten_telegram_text(msg.get("text")).strip()
        if not body:
            continue
        comments[q_index].append(
            {
                "author": str(msg.get("from") or "?"),
                "text": body,
                "direct": reply_id in q_msg_id,
                "themes": _comment_themes(body),
            }
        )

    if reactions and tournament_id not in (12826, 11749):
        # Export is II ЧР-2026 only; callers may pass other ids — data still loads.
        pass
    return reactions, dict(comments)


def _load_reactions_from_result_json(
    path: Path,
    *,
    tournament_id: int,
) -> dict[int, dict[str, int]]:
    reactions, _ = _load_kvrm_discussion_from_result_json(
        path, tournament_id=tournament_id
    )
    return reactions


def _quartile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    return float(np.quantile(np.array(vals, dtype=np.float64), q))


def _analyse_reactions(questions: list[QuestionRow]) -> dict[str, Any]:
    reacted = [q for q in questions if q.reaction_total > 0]
    if len(reacted) < 3:
        return {"n_with_reactions": len(reacted)}

    nets = [float(q.reaction_net) for q in reacted]
    bs = [q.b for q in reacted]
    sels = [q.selectivity for q in reacted]
    takes = [q.take_rate for q in reacted]
    dislikes = [float(q.dislikes) for q in reacted]

    net_q1 = _quartile(nets, 0.25)
    net_q3 = _quartile(nets, 0.75)
    b_q1 = _quartile(bs, 0.25)
    b_q3 = _quartile(bs, 0.75)
    sel_q3 = _quartile(sels, 0.75)

    def _qrow(q: QuestionRow) -> dict[str, Any]:
        return {
            "q": q.q_index + 1,
            "net": q.reaction_net,
            "likes": q.likes,
            "dislikes": q.dislikes,
            "b": round(q.b, 3),
            "selectivity": round(q.selectivity, 3),
            "take_rate": round(q.take_rate, 3),
            "text": q.text[:80],
        }

    loved_easy = [
        _qrow(q)
        for q in sorted(reacted, key=lambda x: -x.reaction_net)
        if q.reaction_net >= net_q3 and q.b <= b_q1
    ]
    hated_selective = [
        _qrow(q)
        for q in sorted(reacted, key=lambda x: x.reaction_net)
        if q.reaction_net <= net_q1 and q.selectivity >= sel_q3
    ]
    hated_hard = [
        _qrow(q)
        for q in sorted(reacted, key=lambda x: x.reaction_net)
        if q.reaction_net <= net_q1 and q.b >= b_q3
    ]
    loved_hard = [
        _qrow(q)
        for q in sorted(reacted, key=lambda x: -x.reaction_net)
        if q.reaction_net >= net_q3 and q.b >= b_q3
    ]

    by_ed: dict[str, list[QuestionRow]] = defaultdict(list)
    for q in reacted:
        by_ed[q.editor or "?"].append(q)

    editor_rows = []
    for ed, qs in sorted(
        by_ed.items(), key=lambda x: -statistics.mean([q.reaction_net for q in x[1]])
    ):
        sels_ed = [q.selectivity for q in qs if not math.isnan(q.selectivity)]
        editor_rows.append(
            {
                "editor": ed,
                "n": len(qs),
                "mean_net": round(statistics.mean([q.reaction_net for q in qs]), 1),
                "mean_b": round(statistics.mean([q.b for q in qs]), 3),
                "mean_selectivity": round(statistics.mean(sels_ed), 3)
                if sels_ed
                else None,
            }
        )

    return {
        "source": "telegram_export",
        "discipline": "КВРМ",
        "join_key": "Вопрос N. → q_index = N-1",
        "n_with_reactions": len(reacted),
        "mean_net": round(statistics.mean(nets), 1),
        "correlations": {
            "net_vs_b": round(_spearman(nets, bs), 3),
            "net_vs_selectivity": round(_spearman(nets, sels), 3),
            "net_vs_take_rate": round(_spearman(nets, takes), 3),
            "dislikes_vs_b": round(_spearman(dislikes, bs), 3),
        },
        "most_loved": [
            _qrow(q) for q in sorted(reacted, key=lambda x: -x.reaction_net)[:8]
        ],
        "most_hated": [
            _qrow(q) for q in sorted(reacted, key=lambda x: x.reaction_net)[:8]
        ],
        "loved_but_easy": loved_easy[:6],
        "hated_but_selective": hated_selective[:6],
        "hated_but_hard": hated_hard[:6],
        "loved_but_hard": loved_hard[:6],
        "editors": editor_rows,
    }


def _sample_comments(
    comments: list[dict[str, Any]],
    *,
    theme: Optional[str] = None,
    limit: int = 3,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(c: dict[str, Any]) -> None:
        key = f"{c['author']}:{c['text'][:40]}"
        if key in seen:
            return
        seen.add(key)
        rows.append({"author": c["author"], "text": c["text"][:160]})

    if theme:
        for c in comments:
            if theme in c.get("themes", []):
                _add(c)
            if len(rows) >= limit:
                return rows
    for c in comments:
        _add(c)
        if len(rows) >= limit:
            break
    return rows


def _analyse_comments(
    questions: list[QuestionRow],
    comments_by_q: dict[int, list[dict[str, Any]]],
    *,
    reaction_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    commented = [q for q in questions if comments_by_q.get(q.q_index)]
    if not commented:
        return {"n_with_comments": 0}

    counts = [float(len(comments_by_q[q.q_index])) for q in commented]
    bs = [q.b for q in commented]
    sels = [q.selectivity for q in commented]
    takes = [q.take_rate for q in commented]
    nets = [float(abs(q.reaction_net)) for q in commented if q.reaction_total > 0]
    dislikes = [float(q.dislikes) for q in commented if q.reaction_total > 0]
    count_for_rx = [
        float(len(comments_by_q[q.q_index]))
        for q in commented
        if q.reaction_total > 0
    ]

    def _qrow(q: QuestionRow) -> dict[str, Any]:
        cs = comments_by_q.get(q.q_index, [])
        theme_totals = defaultdict(int)
        for c in cs:
            for th in c.get("themes", []):
                theme_totals[th] += 1
        return {
            "q": q.q_index + 1,
            "n_comments": len(cs),
            "net": q.reaction_net,
            "b": round(q.b, 3),
            "selectivity": round(q.selectivity, 3),
            "take_rate": round(q.take_rate, 3),
            "themes": dict(theme_totals),
            "text": q.text[:80],
            "samples": _sample_comments(cs, limit=3),
        }

    most_commented = [
        _qrow(q)
        for q in sorted(
            commented, key=lambda x: -len(comments_by_q[x.q_index])
        )[:10]
    ]

    outlier_samples: list[dict[str, Any]] = []
    if reaction_summary:
        loved_qs = {r["q"] for r in reaction_summary.get("most_loved", [])[:6]}
        hated_qs = {r["q"] for r in reaction_summary.get("most_hated", [])[:6]}
        for label, qnums in (("loved", loved_qs), ("hated", hated_qs)):
            for qn in sorted(qnums):
                qi = qn - 1
                cs = comments_by_q.get(qi, [])
                if not cs:
                    continue
                q = next((x for x in questions if x.q_index == qi), None)
                if not q:
                    continue
                outlier_samples.append(
                    {
                        "kind": label,
                        "q": qn,
                        "net": q.reaction_net,
                        "n_comments": len(cs),
                        "samples": _sample_comments(cs, limit=2),
                    }
                )

    return {
        "source": "telegram_export",
        "discipline": "КВРМ",
        "join_key": (
            "reply_to_message_id → walk chain → question post "
            "(Вопрос N. → q_index = N-1)"
        ),
        "n_with_comments": len(commented),
        "total_comments": int(sum(counts)),
        "mean_comments": round(statistics.mean(counts), 1),
        "correlations": {
            "count_vs_b": round(_spearman(counts, bs), 3),
            "count_vs_selectivity": round(_spearman(counts, sels), 3),
            "count_vs_take_rate": round(_spearman(counts, takes), 3),
            "count_vs_abs_net": round(_spearman(count_for_rx, nets), 3)
            if len(count_for_rx) >= 3
            else None,
            "count_vs_dislikes": round(_spearman(count_for_rx, dislikes), 3)
            if len(count_for_rx) >= 3
            else None,
        },
        "most_commented": most_commented,
        "outlier_comment_samples": outlier_samples,
    }


def _load_question_texts(
    questions_db: Path, tournament_id: int
) -> dict[int, dict[str, Any]]:
    if not questions_db.exists():
        return {}
    import sqlite3

    out: dict[int, dict[str, Any]] = {}
    conn = sqlite3.connect(str(questions_db))
    for row in conn.execute(
        """
        SELECT number, text, editors_json, tournaments_json
        FROM questions
        WHERE tournaments_json IS NOT NULL AND tournaments_json != '[]'
        """
    ):
        number, text, editors_json, tournaments_json = row
        try:
            tournaments = json.loads(tournaments_json)
        except Exception:
            continue
        if not any(int(t.get("id", 0)) == tournament_id for t in tournaments):
            continue
        slot = int(number) - 1
        editor = ""
        try:
            eds = json.loads(editors_json or "[]")
            if eds:
                editor = str(eds[0].get("name") or "")
        except Exception:
            pass
        out[slot] = {
            "text": (text or "").replace("\n", " ").strip(),
            "editor": editor,
        }
    conn.close()
    return out


def _normalize_team_name(name: str) -> str:
    s = str(name or "").strip().casefold()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class XlsxTeamRow:
    local_id: int
    team_name: str
    mask: str


def _parse_kvrm_xlsx(path: Path) -> list[XlsxTeamRow]:
    """Parse КВРМ расплюсовка: 6 tours × 15 questions → 90-char mask per team."""
    import pandas as pd

    df = pd.read_excel(path, header=None)
    if len(df) < 3:
        raise ValueError(f"{path}: expected header + data rows")
    data = df.iloc[2:].copy()
    data.columns = ["local_id", "name", "city", "tour"] + list(range(1, 16))
    data = data[pd.to_numeric(data["tour"], errors="coerce").notna()]
    data["tour"] = data["tour"].astype(int)

    def _mask_for_group(group) -> str:
        cells: list[str] = []
        for tour in sorted(group["tour"].unique()):
            row = group[group["tour"] == tour].iloc[0]
            for q in range(1, 16):
                v = row[q]
                cells.append("1" if not pd.isna(v) and int(v) else "0")
        return "".join(cells)

    out: list[XlsxTeamRow] = []
    for local_id, group in data.groupby("local_id"):
        if pd.isna(local_id):
            continue
        name = str(group.iloc[0]["name"] or "").strip()
        mask = _mask_for_group(group)
        out.append(XlsxTeamRow(local_id=int(local_id), team_name=name, mask=mask))
    return out


def _match_xlsx_to_api(
    xlsx_rows: list[XlsxTeamRow],
    api_teams: list[TeamRow],
) -> tuple[dict[int, str], list[str], list[str]]:
    """Map API team_id → xlsx mask.  Returns (masks, unmatched_xlsx, unmatched_api)."""
    api_by_norm = {_normalize_team_name(t.team_name): t for t in api_teams}
    used_api: set[int] = set()
    masks: dict[int, str] = {}
    unmatched_xlsx: list[str] = []

    for xr in xlsx_rows:
        norm = _normalize_team_name(xr.team_name)
        api = api_by_norm.get(norm)
        if api is None:
            candidates = [
                t
                for t in api_teams
                if t.team_id not in used_api
                and (
                    norm in _normalize_team_name(t.team_name)
                    or _normalize_team_name(t.team_name) in norm
                )
            ]
            api = candidates[0] if len(candidates) == 1 else None
        if api is None:
            unmatched_xlsx.append(xr.team_name)
            continue
        used_api.add(api.team_id)
        masks[api.team_id] = xr.mask

    unmatched_api = [t.team_name for t in api_teams if t.team_id not in used_api]
    return masks, unmatched_xlsx, unmatched_api


def _compare_masks(
    api_teams: list[TeamRow], xlsx_masks: dict[int, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team in api_teams:
        xm = xlsx_masks.get(team.team_id)
        if xm is None:
            continue
        am = team.mask
        n_diff = sum(1 for a, b in zip(am, xm) if a != b)
        if n_diff:
            rows.append(
                {
                    "team": team.team_name,
                    "n_diff": n_diff,
                    "api_score": team.score,
                    "xlsx_score": sum(1 for c in xm if c == "1"),
                }
            )
    return rows


def _fetch_teams(
    tournament_id: int,
    *,
    xlsx_path: Optional[Path] = None,
) -> tuple[list[TeamRow], dict[str, Any]]:
    client = RatingApiClient()
    rows = client.get_results(tournament_id, include_team_members=True, include_masks=True)
    teams: list[TeamRow] = []
    for row in rows:
        team = row.get("team") or {}
        tid = team.get("id")
        if not isinstance(tid, int):
            continue
        mask = str(row.get("mask") or "")
        pids = [
            int(tm["player"]["id"])
            for tm in (row.get("teamMembers") or [])
            if isinstance((tm.get("player") or {}).get("id"), int)
        ]
        teams.append(
            TeamRow(
                team_id=int(tid),
                team_name=str(team.get("name") or f"#{tid}"),
                position=float(row.get("position") or 9999),
                mask=mask,
                player_ids=pids,
                score=sum(1 for c in mask if c == "1"),
            )
        )

    meta: dict[str, Any] = {"mask_source": "api"}
    if xlsx_path is not None and xlsx_path.exists():
        xlsx_rows = _parse_kvrm_xlsx(xlsx_path)
        xlsx_masks, unmatched_xlsx, unmatched_api = _match_xlsx_to_api(xlsx_rows, teams)
        discrepancies = _compare_masks(teams, xlsx_masks)
        n_matched = len(xlsx_masks)
        meta = {
            "mask_source": "xlsx",
            "xlsx_path": str(xlsx_path),
            "xlsx_teams": len(xlsx_rows),
            "xlsx_matched": n_matched,
            "xlsx_match_rate": round(n_matched / len(xlsx_rows), 3) if xlsx_rows else 0.0,
            "xlsx_unmatched": unmatched_xlsx,
            "api_unmatched": unmatched_api,
            "mask_discrepancies": discrepancies,
            "n_mask_discrepancies": len(discrepancies),
        }
        for team in teams:
            if team.team_id in xlsx_masks:
                team.mask = xlsx_masks[team.team_id]
                team.score = sum(1 for c in team.mask if c == "1")
    return teams, meta


def _player_thetas(con: duckdb.DuckDBPyConnection, pids: set[int]) -> dict[int, float]:
    if not pids:
        return {}
    ph = ",".join("?" * len(pids))
    return {
        int(r[0]): float(r[1])
        for r in con.execute(
            f"SELECT player_id, theta_display FROM players WHERE player_id IN ({ph})",
            sorted(pids),
        ).fetchall()
        if r[1] is not None
    }


def _team_strength(pids: list[int], pmap: dict[int, float]) -> float:
    vals = sorted((pmap[p] for p in pids if p in pmap), reverse=True)
    return float(sum(vals[:6])) if vals else 0.0


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


# Outliers to annotate on reaction scatter plots (II ЧР article).
_PLOT_LABEL_QS = frozenset({1, 10, 20, 24, 27, 30, 40, 74, 85})


def _questions_for_plot(questions: list[QuestionRow]) -> list[QuestionRow]:
    return [q for q in questions if q.reaction_total > 0]


def _plot_label_candidates(questions: list[QuestionRow]) -> set[int]:
    """Fixed notable Q numbers plus top/bottom net extremes."""
    qs = _questions_for_plot(questions)
    labels = set(_PLOT_LABEL_QS)
    by_net = sorted(qs, key=lambda q: q.reaction_net)
    for q in by_net[:3] + by_net[-3:]:
        labels.add(q.q_index + 1)
    return labels


def _curve_fit_mask(questions: list[QuestionRow]) -> np.ndarray:
    """Questions with ≥2 takes — sole/zero-take outliers distort the trend."""
    return np.array([q.n_taken >= 2 for q in questions], dtype=bool)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _poly_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    deg: int = 2,
    n_pts: int = 200,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    if len(x) < deg + 1:
        return None, None, float("nan")
    coeffs = np.polyfit(x, y, deg)
    x_line = np.linspace(float(x.min()), float(x.max()), n_pts)
    y_line = np.polyval(coeffs, x_line)
    r2 = _r2_score(y, np.polyval(coeffs, x))
    return x_line, y_line, r2


def plot_reaction_scatters(
    questions: list[QuestionRow],
    out_dir: Path,
    *,
    tournament_id: int = 12826,
) -> tuple[dict[str, Path], int]:
    """Scatter plots: community net rating vs model b / empirical selectivity.

    Returns plot paths and count of questions excluded from curve fit (n_taken < 2).
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    reacted = _questions_for_plot(questions)
    if len(reacted) < 3:
        raise ValueError("Need at least 3 questions with reactions to plot")

    nets = np.array([q.reaction_net for q in reacted], dtype=np.float64)
    bs = np.array([q.b for q in reacted], dtype=np.float64)
    sels = np.array([q.selectivity for q in reacted], dtype=np.float64)
    takes = np.array([q.take_rate for q in reacted], dtype=np.float64)
    dislikes = np.array([q.dislikes for q in reacted], dtype=np.float64)
    sizes = np.array(
        [max(18.0, 12.0 + 0.35 * q.reaction_total) for q in reacted],
        dtype=np.float64,
    )
    fit_mask = _curve_fit_mask(reacted)
    n_excluded = int((~fit_mask).sum())
    label_qs = _plot_label_candidates(questions)

    rho_nb = _spearman(nets.tolist(), bs.tolist())
    rho_nsel = _spearman(nets.tolist(), sels.tolist())
    rho_ntake = _spearman(nets.tolist(), takes.tolist())
    rho_db = _spearman(dislikes.tolist(), bs.tolist())
    rho_nb_fit = _spearman(nets[fit_mask].tolist(), bs[fit_mask].tolist())
    rho_nsel_fit = _spearman(nets[fit_mask].tolist(), sels[fit_mask].tolist())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    def _scatter_ax(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        *,
        xlabel: str,
        ylabel: str,
        rho: float,
        color: str = "#2563eb",
        fit_curve: bool = False,
        rho_fit: float = float("nan"),
    ) -> None:
        excl = ~fit_mask
        if excl.any():
            ax.scatter(
                x[excl],
                y[excl],
                s=sizes[excl],
                facecolors="none",
                edgecolors=color,
                alpha=0.4,
                linewidths=1.0,
                zorder=2,
            )
        ax.scatter(
            x[fit_mask],
            y[fit_mask],
            s=sizes[fit_mask],
            c=color,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        if fit_curve and int(fit_mask.sum()) >= 3:
            x_line, y_line, r2 = _poly_curve(x[fit_mask], y[fit_mask])
            if x_line is not None and y_line is not None:
                ax.plot(
                    x_line,
                    y_line,
                    color="#111827",
                    linewidth=2.0,
                    linestyle="-",
                    zorder=4,
                    label="кривая: n взятий ≥ 2",
                )
                stats = f"ρ = {rho_fit:+.2f}\nR² = {r2:.2f}"
            else:
                stats = f"ρ = {rho:+.2f}"
        else:
            stats = f"ρ = {rho:+.2f}"
        ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"II ЧР #{tournament_id}", fontsize=10, color="#6b7280", pad=8)
        ax.text(
            0.03,
            0.97,
            stats,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#e5e7eb"),
        )
        if fit_curve and int(fit_mask.sum()) >= 3:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        for i, q in enumerate(reacted):
            qn = q.q_index + 1
            if qn not in label_qs:
                continue
            ax.annotate(
                f"Q{qn:02d}",
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color="#374151",
            )

    paths: dict[str, Path] = {}

    fig_a, ax_a = plt.subplots(figsize=(6.2, 5.0), dpi=150)
    _scatter_ax(
        ax_a,
        bs,
        nets,
        xlabel="Сложность вопроса (b)",
        ylabel="Рейтинг сообщества (лайки − дизлайки)",
        rho=rho_nb,
        rho_fit=rho_nb_fit,
        fit_curve=True,
    )
    fig_a.tight_layout()
    p_a = out_dir / "ratings_vs_difficulty.png"
    fig_a.savefig(p_a, bbox_inches="tight")
    plt.close(fig_a)
    paths["difficulty"] = p_a

    fig_b, ax_b = plt.subplots(figsize=(6.2, 5.0), dpi=150)
    _scatter_ax(
        ax_b,
        sels,
        nets,
        xlabel="Селективность (ρ Спирмена, взятие ↔ сила)",
        ylabel="Рейтинг сообщества (лайки − дизлайки)",
        rho=rho_nsel,
        rho_fit=rho_nsel_fit,
        color="#7c3aed",
        fit_curve=True,
    )
    fig_b.tight_layout()
    p_b = out_dir / "ratings_vs_selectivity.png"
    fig_b.savefig(p_b, bbox_inches="tight")
    plt.close(fig_b)
    paths["selectivity"] = p_b

    fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)
    _scatter_ax(
        ax_c1,
        bs,
        nets,
        xlabel="Сложность вопроса (b)",
        ylabel="Рейтинг сообщества",
        rho=rho_nb,
        rho_fit=rho_nb_fit,
        fit_curve=True,
    )
    _scatter_ax(
        ax_c2,
        sels,
        nets,
        xlabel="Селективность (ρ Спирмена)",
        ylabel="Рейтинг сообщества",
        rho=rho_nsel,
        rho_fit=rho_nsel_fit,
        color="#7c3aed",
        fit_curve=True,
    )
    fig_c.suptitle(
        f"Реакции Telegram vs модель · II ЧР #{tournament_id}",
        fontsize=12,
        y=1.02,
    )
    fig_c.tight_layout()
    p_c = out_dir / "ratings_vs_model_combined.png"
    fig_c.savefig(p_c, bbox_inches="tight")
    plt.close(fig_c)
    paths["combined"] = p_c

    fig_d, ax_d = plt.subplots(figsize=(6.2, 5.0), dpi=150)
    _scatter_ax(
        ax_d,
        bs,
        dislikes,
        xlabel="Сложность вопроса (b)",
        ylabel="Число дизлайков",
        rho=rho_db,
        color="#dc2626",
    )
    fig_d.tight_layout()
    p_d = out_dir / "dislikes_vs_difficulty.png"
    fig_d.savefig(p_d, bbox_inches="tight")
    plt.close(fig_d)
    paths["dislikes"] = p_d

    fig_e, ax_e = plt.subplots(figsize=(6.2, 5.0), dpi=150)
    _scatter_ax(
        ax_e,
        takes,
        nets,
        xlabel="Доля взятий",
        ylabel="Рейтинг сообщества (лайки − дизлайки)",
        rho=rho_ntake,
        color="#059669",
    )
    fig_e.tight_layout()
    p_e = out_dir / "ratings_vs_take_rate.png"
    fig_e.savefig(p_e, bbox_inches="tight")
    plt.close(fig_e)
    paths["take_rate"] = p_e

    return paths, n_excluded


def _b_from_duckdb(
    con: duckdb.DuckDBPyConnection, tournament_id: int
) -> Optional[list[QuestionRow]]:
    rows = con.execute(
        """
        SELECT primary_q_in_tournament, b, a, n_taken, n_obs,
               COALESCE(text, '')
        FROM questions
        WHERE primary_tournament_id = ?
        ORDER BY primary_q_in_tournament
        """,
        [tournament_id],
    ).fetchall()
    if not rows:
        return None
    return [
        QuestionRow(
            q_index=int(r[0]),
            b=float(r[1]),
            a=float(r[2]),
            n_taken=int(r[3] or 0),
            n_teams=int(r[4] or 0),
            take_rate=(int(r[3] or 0) / int(r[4])) if r[4] else 0.0,
            selectivity=float("nan"),
            text=str(r[5] or "")[:120],
        )
        for r in rows
    ]


def _theta_before_map(
    con: duckdb.DuckDBPyConnection,
    tid: int,
    pids: list[int],
    *,
    cold_init: float,
) -> dict[int, float]:
    gidx = con.execute(
        "SELECT game_idx FROM tournaments WHERE tournament_id = ?", [tid]
    ).fetchone()[0]
    hist = con.execute(
        """
        SELECT ph.player_id, ph.theta, t.game_idx
        FROM player_history ph
        JOIN tournaments t ON t.tournament_id = ph.tournament_id
        WHERE ph.player_id IN (SELECT UNNEST(?))
        ORDER BY ph.player_id, t.game_idx
        """,
        [pids],
    ).fetchall()
    by_player: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for pid, th, gi in hist:
        by_player[int(pid)].append((int(gi), float(th)))
    out: dict[int, float] = {}
    for pid in pids:
        prev = cold_init
        for gi, th in by_player.get(pid, []):
            if gi >= gidx:
                break
            prev = th
        out[pid] = prev
    return out


def _resolve_question_b(
    questions: list[QuestionRow],
    *,
    mode: str,
    teams: list[TeamRow],
    theta_before: dict[int, float],
    mean_expected_trained: float | None = None,
    mean_delta_trained: float | None = None,
) -> tuple[list[QuestionRow], str]:
    """Return questions with ``b`` chosen for expected-takes forecast."""
    if mode == "trained":
        return questions, "seq.npz (trained b)"

    b_tr = np.array([q.b for q in questions], dtype=np.float64)
    b_init = np.array(
        [
            _init_b(
                q.take_rate,
                theta_bar=_theta_bar_for_question(
                    [
                        1 if len(t.mask) > q.q_index and t.mask[q.q_index] == "1" else 0
                        for t in teams
                    ],
                    [
                        statistics.mean(
                            [theta_before.get(p, -1.0) for p in t.player_ids]
                        )
                        if t.player_ids
                        else -1.0
                        for t in teams
                    ],
                ),
            )
            for q in questions
        ],
        dtype=np.float64,
    )
    gap = pack_b_gap(b_tr, b_init)

    if mode == "oracle":
        label = "init b from take rates (oracle)"
        b_use = b_init
    elif mode == "pack-adj":
        b_use, _, applied = pack_adjust_b(b_tr, b_init)
        label = (
            "pack-adj b (oracle, gap≥threshold)"
            if applied
            else "trained b (pack gap below threshold)"
        )
    elif mode == "auto":
        from rating.pack_calib import should_use_pack_adj_retrospective

        retro = False
        if mean_expected_trained is not None and mean_delta_trained is not None:
            retro = should_use_pack_adj_retrospective(
                mean_expected_trained=mean_expected_trained,
                mean_delta_trained=mean_delta_trained,
                b_gap=gap,
            )
        if retro:
            b_use = b_init
            label = "auto: oracle b (elite retrospective gate)"
        else:
            b_use = b_tr
            label = "auto: trained b"
    else:
        raise ValueError(f"unknown expected_b mode: {mode}")

    out = [
        QuestionRow(
            q_index=q.q_index,
            b=float(b_use[i]),
            a=q.a,
            n_taken=q.n_taken,
            n_teams=q.n_teams,
            take_rate=q.take_rate,
            selectivity=q.selectivity,
            text=q.text,
            editor=q.editor,
            sole_team=q.sole_team,
            likes=q.likes,
            dislikes=q.dislikes,
            reaction_net=q.reaction_net,
            reaction_total=q.reaction_total,
            comment_count=q.comment_count,
        )
        for i, q in enumerate(questions)
    ]
    return out, label


def _init_b(
    take_rate: float,
    *,
    team_size_avg: float = 6.0,
    theta_bar: float = 0.0,
) -> float:
    return init_b_from_take_rate(
        take_rate, team_size_avg=team_size_avg, theta_bar=theta_bar
    )


def _theta_bar_for_question(
    takes: list[int],
    strengths: list[float],
    *,
    min_games: int = 3,
    pmap_games: Optional[dict[int, int]] = None,
) -> float:
    """Mean strength of teams that took the question (proxy for θ̄)."""
    vals = [s for t, s in zip(takes, strengths) if t == 1]
    return float(statistics.mean(vals)) if vals else 0.0


def _compute_questions_live(
    teams: list[TeamRow],
    strengths: list[float],
    texts: dict[int, dict[str, Any]],
    *,
    team_size_avg: float = 6.0,
) -> list[QuestionRow]:
    n_q = len(teams[0].mask) if teams else 0
    n_teams = len(teams)
    takes_matrix = np.array(
        [[1 if c == "1" else 0 for c in t.mask] for t in teams], dtype=np.int8
    )
    out: list[QuestionRow] = []
    for qi in range(n_q):
        takes = takes_matrix[:, qi]
        n_taken = int(takes.sum())
        take_rate = n_taken / n_teams
        theta_bar = _theta_bar_for_question(takes.tolist(), strengths)
        b = _init_b(take_rate, team_size_avg=team_size_avg, theta_bar=theta_bar)
        sel = _spearman(strengths, takes.astype(float).tolist())
        sole_team = ""
        if n_taken == 1:
            idx = int(np.argmax(takes))
            sole_team = teams[idx].team_name
        meta = texts.get(qi, {})
        out.append(
            QuestionRow(
                q_index=qi,
                b=b,
                a=1.0,
                n_taken=n_taken,
                n_teams=n_teams,
                take_rate=take_rate,
                selectivity=sel,
                text=meta.get("text", "")[:120],
                editor=meta.get("editor", ""),
                sole_team=sole_team,
            )
        )
    # Recompute selectivity with model b ranking context — already per-q
    return out


def _solve_implied_theta(
    a_arr: np.ndarray,
    b_eff: np.ndarray,
    n: int,
    score: float,
    *,
    lo: float = -8.0,
    hi: float = 8.0,
) -> float:
    if n <= 0:
        return float("nan")
    target = float(score)

    def pred(theta: float) -> float:
        lam = n * np.exp(a_arr * theta - b_eff)
        return float(np.sum(1.0 - np.exp(-lam)))

    if pred(lo) >= target:
        return lo
    if pred(hi) <= target:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if pred(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _expected_and_implied(
    teams: list[TeamRow],
    questions: list[QuestionRow],
    pmap: dict[int, float],
    res,
) -> None:
    b = np.array([q.b for q in questions], dtype=np.float64)
    a = np.array([q.a for q in questions], dtype=np.float64)
    qi = np.arange(len(questions), dtype=np.int64)
    for team in teams:
        thetas = np.array(
            [pmap.get(p, -1.0) for p in team.player_ids], dtype=np.float64
        )
        n_active = len(thetas)
        if n_active == 0:
            continue
        p_q = simulate_roster_on_pack(
            thetas,
            b,
            a,
            q_in_tour=qi,
            delta_size=res.delta_size,
            team_size_anchor=res.team_size_anchor,
            delta_pos=res.delta_pos,
            pos_anchor=res.pos_anchor,
            mode="offline",
            lapse_arr=res.lapse,
            recal_arr=res.recal,
        )
        team.expected = float(p_q.sum())
        # implied θ
        size_shift = 0.0
        if res.delta_size is not None and res.team_size_anchor is not None:
            ds = res.delta_size
            ts = max(1, min(n_active, len(ds) - 1))
            size_shift = float(ds[ts] - ds[int(res.team_size_anchor)])
        pos_shift = np.zeros(len(questions), dtype=np.float64)
        if res.delta_pos is not None and res.pos_anchor is not None:
            tour_len = len(res.delta_pos)
            pos = qi % tour_len
            pos_shift = res.delta_pos[pos] - res.delta_pos[int(res.pos_anchor)]
        b_eff = b + size_shift + pos_shift
        team.theta_implied = _solve_implied_theta(a, b_eff, n_active, team.score)


def _empirical_selectivity(
    teams: list[TeamRow], strengths: list[float]
) -> list[float]:
    n_q = len(teams[0].mask) if teams else 0
    sels: list[float] = []
    for qi in range(n_q):
        takes = [1.0 if t.mask[qi] == "1" else 0.0 for t in teams]
        sels.append(_spearman(strengths, takes))
    return sels


def _editor_stats(questions: list[QuestionRow]) -> list[dict[str, Any]]:
    by_ed: dict[str, list[QuestionRow]] = defaultdict(list)
    for q in questions:
        ed = q.editor or "?"
        by_ed[ed].append(q)
    rows = []
    for ed, qs in sorted(by_ed.items(), key=lambda x: -statistics.mean([q.b for q in x[1]])):
        sels = [q.selectivity for q in qs if not math.isnan(q.selectivity)]
        rows.append(
            {
                "editor": ed,
                "n": len(qs),
                "mean_b": round(statistics.mean([q.b for q in qs]), 3),
                "mean_selectivity": round(statistics.mean(sels), 3) if sels else None,
                "mean_take_rate": round(statistics.mean([q.take_rate for q in qs]), 3),
                "sole_takes": sum(1 for q in qs if q.n_taken == 1),
            }
        )
    return rows


def analyse_tournament(
    tournament_id: int,
    *,
    duckdb_path: Path,
    seq_path: Path,
    questions_db: Path,
    use_duckdb_b: bool = True,
    reactions_json: Optional[Path] = None,
    xlsx_path: Optional[Path] = None,
    expected_b: str = "trained",
) -> dict[str, Any]:
    con = duckdb.connect(str(duckdb_path), read_only=True)
    res = load_results_npz(seq_path)
    teams, mask_meta = _fetch_teams(tournament_id, xlsx_path=xlsx_path)
    if not teams:
        raise SystemExit(f"No teams for tournament {tournament_id}")

    all_pids = {p for t in teams for p in t.player_ids}
    pmap = _player_thetas(con, all_pids)
    strengths = [_team_strength(t.player_ids, pmap) for t in teams]
    for t, s in zip(teams, strengths):
        t.strength = s

    texts = _load_question_texts(questions_db, tournament_id)
    questions = None
    b_source = "init_from_take_rate"
    if use_duckdb_b:
        questions = _b_from_duckdb(con, tournament_id)
        if questions:
            b_source = "seq.npz (DuckDB bake)"
            sels = _empirical_selectivity(teams, strengths)
            for q, sel in zip(questions, sels):
                q.selectivity = sel
                meta = texts.get(q.q_index, {})
                if meta.get("text"):
                    q.text = meta["text"][:120]
                if meta.get("editor"):
                    q.editor = meta["editor"]

    if questions is None:
        questions = _compute_questions_live(teams, strengths, texts)
        b_source = "init_from_take_rate (not yet in seq.npz)"

    if reactions_json is not None:
        reactions, comments_by_q = _load_kvrm_discussion_from_result_json(
            reactions_json, tournament_id=tournament_id
        )
        for q in questions:
            if q.q_index in reactions:
                r = reactions[q.q_index]
                q.likes = r["likes"]
                q.dislikes = r["dislikes"]
                q.reaction_net = r["reaction_net"]
                q.reaction_total = r["reaction_total"]
            if q.q_index in comments_by_q:
                q.comment_count = len(comments_by_q[q.q_index])
    else:
        comments_by_q = {}

    cold_init = float(getattr(res, "cold_init_theta", -1.0))
    theta_before = _theta_before_map(
        con, tournament_id, sorted(all_pids), cold_init=cold_init
    )

    questions_for_exp = questions
    exp_b_label = b_source
    if expected_b != "trained" and questions is not None:
        _expected_and_implied(teams, questions, theta_before, res)
        mean_exp_tr = statistics.mean([t.expected for t in teams])
        mean_act = statistics.mean([t.score for t in teams])
        questions_for_exp, exp_b_label = _resolve_question_b(
            questions,
            mode=expected_b,
            teams=teams,
            theta_before=theta_before,
            mean_expected_trained=mean_exp_tr,
            mean_delta_trained=mean_act - mean_exp_tr,
        )
        for t in teams:
            t.expected = 0.0
        _expected_and_implied(teams, questions_for_exp, theta_before, res)
    else:
        _expected_and_implied(teams, questions, pmap, res)

    # attach sole team names for duckdb path
    if b_source.startswith("seq"):
        for q in questions:
            if q.n_taken == 1:
                for t in teams:
                    if len(t.mask) > q.q_index and t.mask[q.q_index] == "1":
                        q.sole_team = t.team_name
                        break

    teams_sorted_diff = sorted(teams, key=lambda t: t.score - t.expected, reverse=True)
    teams_sorted_impl = sorted(
        teams, key=lambda t: t.theta_implied if not math.isnan(t.theta_implied) else -99,
        reverse=True,
    )

    q_by_b = sorted(questions, key=lambda q: q.b, reverse=True)
    q_by_sel = sorted(questions, key=lambda q: q.selectivity, reverse=True)

    a_vals = [q.a for q in questions]
    report = {
        "tournament_id": tournament_id,
        "n_teams": len(teams),
        "n_questions": len(questions),
        "b_source": b_source,
        "expected_b_mode": expected_b,
        "expected_b_source": exp_b_label if expected_b != "trained" else b_source,
        "masks": mask_meta,
        "a_frozen": bool(np.median(a_vals) == 1.0 and np.std(a_vals) < 0.01),
        "a_mean": round(float(np.mean(a_vals)), 6),
        "a_std": round(float(np.std(a_vals)), 6),
        "b_mean": round(statistics.mean([q.b for q in questions]), 3),
        "b_std": round(statistics.stdev([q.b for q in questions]), 3),
        "mean_score": round(statistics.mean([t.score for t in teams]), 1),
        "mean_expected": round(statistics.mean([t.expected for t in teams]), 1),
        "hardest": [
            {
                "q": q.q_index + 1,
                "b": round(q.b, 3),
                "a": round(q.a, 4),
                "taken": f"{q.n_taken}/{q.n_teams}",
                "selectivity": round(q.selectivity, 3),
                "text": q.text[:80],
                "sole_team": q.sole_team,
            }
            for q in q_by_b[:10]
        ],
        "easiest": [
            {
                "q": q.q_index + 1,
                "b": round(q.b, 3),
                "taken": f"{q.n_taken}/{q.n_teams}",
                "text": q.text[:80],
            }
            for q in q_by_b[-10:]
        ],
        "most_selective": [
            {
                "q": q.q_index + 1,
                "selectivity": round(q.selectivity, 3),
                "b": round(q.b, 3),
                "taken": f"{q.n_taken}/{q.n_teams}",
                "text": q.text[:80],
            }
            for q in q_by_sel[:10]
        ],
        "least_selective": [
            {
                "q": q.q_index + 1,
                "selectivity": round(q.selectivity, 3),
                "b": round(q.b, 3),
                "taken": f"{q.n_taken}/{q.n_teams}",
                "text": q.text[:80],
            }
            for q in q_by_sel[-10:]
        ],
        "sole_takes": [
            {
                "q": q.q_index + 1,
                "b": round(q.b, 3),
                "team": q.sole_team,
                "text": q.text[:80],
            }
            for q in sorted(questions, key=lambda x: -x.b)
            if q.n_taken == 1
        ],
        "overperformers": [
            {
                "team": t.team_name,
                "place": t.position,
                "score": t.score,
                "expected": round(t.expected, 1),
                "diff": round(t.score - t.expected, 1),
                "theta_implied": round(t.theta_implied, 3),
                "strength": round(t.strength, 2),
            }
            for t in teams_sorted_diff[:8]
        ],
        "underperformers": [
            {
                "team": t.team_name,
                "place": t.position,
                "score": t.score,
                "expected": round(t.expected, 1),
                "diff": round(t.score - t.expected, 1),
                "theta_implied": round(t.theta_implied, 3),
                "strength": round(t.strength, 2),
            }
            for t in teams_sorted_diff[-8:]
        ],
        "top_by_implied_theta": [
            {
                "team": t.team_name,
                "place": t.position,
                "score": t.score,
                "theta_implied": round(t.theta_implied, 3),
                "strength": round(t.strength, 2),
            }
            for t in teams_sorted_impl[:10]
        ],
        "editors": _editor_stats(questions),
        "winner": {
            "team": min(teams, key=lambda t: t.position).team_name,
            "score": min(teams, key=lambda t: t.position).score,
            "place": min(teams, key=lambda t: t.position).position,
        },
    }
    if reactions_json is not None:
        report["reactions"] = _analyse_reactions(questions)
        report["comments"] = _analyse_comments(
            questions,
            comments_by_q,
            reaction_summary=report.get("reactions"),
        )
    con.close()
    return report, questions


def _print_report(report: dict[str, Any]) -> None:
    tid = report["tournament_id"]
    print(f"\n{'='*72}\nЧР analysis #{tid}\n{'='*72}")
    print(
        f"Teams: {report['n_teams']}, questions: {report['n_questions']}, "
        f"b from: {report['b_source']}"
    )
    masks = report.get("masks") or {}
    if masks.get("mask_source") == "xlsx":
        print(
            f"Masks: xlsx ({masks.get('xlsx_matched')}/{masks.get('xlsx_teams')} matched, "
            f"rate={masks.get('xlsx_match_rate')})"
        )
        if masks.get("xlsx_unmatched"):
            print(f"  xlsx unmatched: {', '.join(masks['xlsx_unmatched'])}")
        if masks.get("api_unmatched"):
            print(f"  api unmatched: {', '.join(masks['api_unmatched'])}")
        n_disc = masks.get("n_mask_discrepancies", 0)
        if n_disc:
            print(f"  mask discrepancies vs API: {n_disc} teams")
            for row in masks.get("mask_discrepancies", [])[:5]:
                print(
                    f"    {row['team']}: {row['n_diff']} cells "
                    f"(api={row['api_score']} xlsx={row['xlsx_score']})"
                )
        else:
            print("  mask discrepancies vs API: none")
    print(
        f"b: mean={report['b_mean']} std={report['b_std']}; "
        f"a: mean={report['a_mean']} std={report['a_std']} frozen≈1: {report['a_frozen']}"
    )
    print(
        f"Scores: mean actual={report['mean_score']} expected={report['mean_expected']}"
    )
    print(f"Winner: {report['winner']['team']} ({report['winner']['score']} pts)")
    print("\n--- Hardest (by b) ---")
    for q in report["hardest"][:8]:
        print(f"  Q{q['q']:02d} b={q['b']:+.3f} {q['taken']} sel={q['selectivity']:+.3f} | {q['text']}")
    print("\n--- Overperformers (actual - expected) ---")
    for t in report["overperformers"][:6]:
        print(
            f"  {t['team']}: {t['score']} vs {t['expected']} ({t['diff']:+.1f}), "
            f"θ_impl={t['theta_implied']:+.2f}, place={t['place']}"
        )
    print("\n--- Underperformers ---")
    for t in report["underperformers"][:6]:
        print(
            f"  {t['team']}: {t['score']} vs {t['expected']} ({t['diff']:+.1f}), "
            f"place={t['place']}"
        )
    print("\n--- Most selective (empirical Spearman) ---")
    for q in report["most_selective"][:8]:
        print(f"  Q{q['q']:02d} sel={q['selectivity']:+.3f} b={q['b']:+.3f} | {q['text']}")
    print("\n--- Editors ---")
    for e in report["editors"]:
        print(
            f"  {e['editor']:20s} n={e['n']:2d} mean_b={e['mean_b']:+.3f} "
            f"sel={e['mean_selectivity']} sole={e['sole_takes']}"
        )
    rx = report.get("reactions")
    if rx and rx.get("n_with_reactions", 0) >= 3:
        print("\n--- Community reactions (Telegram) ---")
        corr = rx["correlations"]
        print(
            f"  n={rx['n_with_reactions']}, mean net={rx['mean_net']}; "
            f"ρ(net,b)={corr['net_vs_b']}, ρ(net,take)={corr['net_vs_take_rate']}, "
            f"ρ(net,sel)={corr['net_vs_selectivity']}"
        )
        print("  Most loved:")
        for q in rx["most_loved"][:4]:
            print(
                f"    Q{q['q']:02d} net={q['net']:+d} b={q['b']:+.2f} | {q['text'][:60]}"
            )
        print("  Most hated:")
        for q in rx["most_hated"][:4]:
            print(
                f"    Q{q['q']:02d} net={q['net']:+d} b={q['b']:+.2f} | {q['text'][:60]}"
            )
    cm = report.get("comments")
    if cm and cm.get("n_with_comments", 0) >= 3:
        print("\n--- Player comments (Telegram threads) ---")
        corr = cm["correlations"]
        print(
            f"  n={cm['n_with_comments']} questions, {cm['total_comments']} comments "
            f"(mean {cm['mean_comments']}/q); "
            f"ρ(count,b)={corr['count_vs_b']}, "
            f"ρ(count,|net|)={corr['count_vs_abs_net']}, "
            f"ρ(count,dislikes)={corr['count_vs_dislikes']}"
        )
        print("  Most discussed:")
        for q in cm["most_commented"][:6]:
            themes = q.get("themes") or {}
            th = ", ".join(f"{k}:{v}" for k, v in sorted(themes.items()))
            print(
                f"    Q{q['q']:02d} {q['n_comments']} comments net={q['net']:+d} "
                f"b={q['b']:+.2f} [{th}]"
            )
            for s in q.get("samples", [])[:1]:
                print(f"      «{s['text'][:72]}» — {s['author']}")
        for block in cm.get("outlier_comment_samples", [])[:4]:
            print(
                f"  {block['kind']} outlier Q{block['q']:02d} "
                f"({block['n_comments']} comments, net={block['net']:+d}):"
            )
            for s in block.get("samples", []):
                print(f"    «{s['text'][:72]}» — {s['author']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse ЧР with chgk-model b/a/θ")
    ap.add_argument("--tournament-id", type=int, default=12826)
    ap.add_argument("--compare", type=int, default=None, help="YoY tournament id")
    ap.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    ap.add_argument("--seq", type=Path, default=DEFAULT_SEQ)
    ap.add_argument("--questions-db", type=Path, default=DEFAULT_QUESTIONS_DB)
    ap.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="КВРМ расплюсовка (.xlsx); auto-detects data/КВРМ.xlsx when present",
    )
    ap.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Force API masks even if data/КВРМ.xlsx exists",
    )
    ap.add_argument(
        "--reactions-json",
        type=Path,
        default=None,
        help="Telegram export with reactions and player comments (КВРМ block only)",
    )
    ap.add_argument(
        "--top-comments",
        type=int,
        default=0,
        metavar="N",
        help="After the report, list top N questions by comment count",
    )
    ap.add_argument(
        "--expected-b",
        choices=("trained", "oracle", "pack-adj", "auto"),
        default="trained",
        help="b used for expected-takes forecast (oracle/pack-adj are retrospective)",
    )
    ap.add_argument(
        "--forecast-diagnostic",
        action="store_true",
        help="Run forecast bias diagnostic (scripts/forecast_diagnostic.py)",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Save reaction scatter plots (requires reactions JSON)",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=REPO_ROOT / "results/chr2026",
        help="Output directory for --plots PNGs",
    )
    args = ap.parse_args()

    reactions_path = args.reactions_json
    if reactions_path is None and DEFAULT_REACTIONS_JSON.exists():
        reactions_path = DEFAULT_REACTIONS_JSON

    xlsx_path: Optional[Path] = None
    if not args.no_xlsx:
        xlsx_path = args.xlsx
        if xlsx_path is None and DEFAULT_KVRM_XLSX.exists():
            xlsx_path = DEFAULT_KVRM_XLSX

    if args.forecast_diagnostic:
        from scripts.forecast_diagnostic import diagnose as diagnose_forecast

        fd = diagnose_forecast(
            args.tournament_id,
            duckdb_path=args.duckdb,
            seq_path=args.seq,
            xlsx_path=xlsx_path,
            baseline_offline=bool(args.compare),
        )
        if args.compare:
            fd["compare"] = diagnose_forecast(
                args.compare,
                duckdb_path=args.duckdb,
                seq_path=args.seq,
                xlsx_path=None,
                baseline_offline=False,
            )
        if args.json:
            print(json.dumps(fd, ensure_ascii=False, indent=2))
        else:
            s = fd["summary"]
            print(f"\n=== Forecast diagnostic #{fd['tournament_id']} ===")
            print(
                f"Δ={s['mean_delta']:+.2f}  take {s['mean_take_rate_pct']:.1f}% "
                f"vs pred {s['mean_pred_rate_pct']:.1f}%"
            )
            cf = fd["counterfactuals"]
            print(
                f"b gap={cf['b_gap_init_minus_trained']:+.3f}  "
                f"oracle b_init Δ={cf['b_init_oracle_delta']:+.2f}"
            )
        return 0

    report, questions = analyse_tournament(
        args.tournament_id,
        duckdb_path=args.duckdb,
        seq_path=args.seq,
        questions_db=args.questions_db,
        reactions_json=reactions_path,
        xlsx_path=xlsx_path,
        expected_b=args.expected_b,
    )
    if args.compare:
        compare_report, _ = analyse_tournament(
            args.compare,
            duckdb_path=args.duckdb,
            seq_path=args.seq,
            questions_db=args.questions_db,
            reactions_json=None,
            xlsx_path=xlsx_path,
            expected_b=args.expected_b,
        )
        report["compare"] = compare_report

    if args.plots:
        if reactions_path is None:
            raise SystemExit("--plots requires reactions JSON (data/result.json)")
        paths, n_curve_excluded = plot_reaction_scatters(
            questions, args.plots_dir, tournament_id=args.tournament_id
        )
        report["plot_paths"] = {k: str(v) for k, v in paths.items()}
        report["curve_fit_excluded"] = n_curve_excluded
        if not args.json:
            print("\n--- Saved plots ---")
            for k, p in paths.items():
                print(f"  {k}: {p}")
            print(
                f"  curve fit: excluded {n_curve_excluded} questions "
                f"(n_taken < 2); hollow markers on plot"
            )

    if args.forecast_diagnostic:
        from scripts.forecast_diagnostic import diagnose as diagnose_forecast

        fd = diagnose_forecast(
            args.tournament_id,
            duckdb_path=args.duckdb,
            seq_path=args.seq,
            xlsx_path=xlsx_path,
            baseline_offline=bool(args.compare),
        )
        if args.compare:
            fd["compare"] = diagnose_forecast(
                args.compare,
                duckdb_path=args.duckdb,
                seq_path=args.seq,
                xlsx_path=None,
                baseline_offline=False,
            )
        if args.json:
            print(json.dumps(fd, ensure_ascii=False, indent=2))
        else:
            s = fd["summary"]
            print(f"\n=== Forecast diagnostic #{fd['tournament_id']} ===")
            print(
                f"Δ={s['mean_delta']:+.2f}  take {s['mean_take_rate_pct']:.1f}% "
                f"vs pred {s['mean_pred_rate_pct']:.1f}%"
            )
            cf = fd["counterfactuals"]
            print(
                f"b gap={cf['b_gap_init_minus_trained']:+.3f}  "
                f"oracle b_init Δ={cf['b_init_oracle_delta']:+.2f}"
            )
        return 0

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
        if args.top_comments > 0:
            cm = report.get("comments") or {}
            rows = cm.get("most_commented") or []
            print(f"\n--- Top {args.top_comments} by comment count ---")
            for q in rows[: args.top_comments]:
                print(
                    f"Q{q['q']:02d}: {q['n_comments']} comments, net={q['net']:+d}, "
                    f"b={q['b']:+.2f}"
                )
                for s in q.get("samples", [])[:2]:
                    print(f"  «{s['text'][:90]}» — {s['author']}")
        if args.compare and "compare" in report:
            print("\n\n=== YoY comparison ===")
            c = report["compare"]
            print(
                f"#{args.compare}: b_mean={c['b_mean']} sole_takes={len(c['sole_takes'])} "
                f"mean_score={c['mean_score']}"
            )
            print(
                f"#{args.tournament_id}: b_mean={report['b_mean']} "
                f"sole_takes={len(report['sole_takes'])} mean_score={report['mean_score']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
