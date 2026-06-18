"""Personal question-profile analysis: where does a player underperform?

Joins rating outcomes (data.npz) with question texts/metadata (DuckDB +
questions.db) and aggregates residuals vs the field and vs the model's
pre-tournament team prediction (seq.npz + simulate_roster_on_pack).

Usage:
    python scripts/analyse_player_weaknesses.py --player-id 32919
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import duckdb
import numpy as np
import pandas as pd

from data import load_cached
from rating.io import load_results_npz
from rating.simulate import simulate_roster_on_pack

_RE_LATIN = re.compile(r"\bлатин|по-латин|на латын", re.IGNORECASE)
_RE_QUOTES = re.compile(r"[«»\"']")
_RE_PRONOUN = re.compile(r"\b(ОН|ОНА|ОНО|ЕГО|ЕЁ|ИХ|ЭТО)\b")
_RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _detect_techniques(text: str, *, has_audio: bool, has_razdatka: bool) -> list[str]:
    out: list[str] = []
    t = text or ""
    tl = t.lower()
    if _RE_PRONOUN.search(t):
        out.append("подстановка референта (ОН/ЕГО/ЭТО)")
    if _RE_LATIN.search(t):
        out.append("этимология / латынь")
    if ("хлоп" in tl) or _RE_QUOTES.search(t):
        out.append("игра слов / омонимия")
    if _RE_YEAR.search(t) or "сообщ" in tl or "министер" in tl:
        out.append("контекстное расширение (новости/эпоха)")
    if has_audio:
        out.append("аудиовопрос")
    if has_razdatka:
        out.append("раздатка / визуал")
    if "какую" in tl[-40:] or "какое" in tl[-40:] or "какой" in tl[-40:]:
        out.append("«какой/какая/какое» — угадай слово")
    if not out:
        out.append("классическая склейка подсказок")
    return out


def _parse_names(blob: str | None) -> list[str]:
    if not blob or blob in ("[]", ""):
        return []
    try:
        items = json.loads(blob)
    except json.JSONDecodeError:
        return []
    names: list[str] = []
    for it in items:
        if isinstance(it, dict):
            n = (it.get("name") or "").strip()
            if n:
                names.append(n)
        elif isinstance(it, str) and it.strip():
            names.append(it.strip())
    return names


def _parse_tags(blob: str | None) -> list[str]:
    if not blob or blob in ("[]", ""):
        return []
    try:
        items = json.loads(blob)
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            t = (it.get("title") or "").strip()
            if t:
                out.append(t)
        elif isinstance(it, str) and it.strip():
            out.append(it.strip())
    return out


def _complexity_band(raw: str | None) -> str:
    if not raw or raw in ("[]", ""):
        return "unknown"
    try:
        val = float(json.loads(raw)[0])
    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
        return "unknown"
    if val < 8:
        return "лёгкий (<8)"
    if val < 12:
        return "средний (8–12)"
    if val < 16:
        return "сложный (12–16)"
    return "очень сложный (16+)"


@dataclass
class QuestionRow:
    tournament_id: int
    q_in_tournament: int
    taken: int
    field_rate: float
    b: float | None
    mode: str
    title: str
    text: str
    answer: str
    tags: list[str]
    authors: list[str]
    editors: list[str]
    techniques: list[str]
    answer_entity_types: list[str]
    complexity_band: str
    question_db_id: int | None
    p_pred: float | None = None

    @property
    def gap_field(self) -> float:
        return float(self.taken) - self.field_rate

    @property
    def gap_model(self) -> float | None:
        if self.p_pred is None:
            return None
        return float(self.taken) - self.p_pred


def _build_history_by_player(res) -> dict[int, list[tuple[int, float]]]:
    by_player: dict[int, list[tuple[int, float]]] = defaultdict(list)
    if res.history_player_id is None:
        return by_player
    for pid, gid, th in zip(
        res.history_player_id, res.history_game_id, res.history_theta
    ):
        by_player[int(pid)].append((int(gid), float(th)))
    for pid in by_player:
        by_player[pid].sort(key=lambda x: x[0])
    return by_player


def _theta_before(
    by_player: dict[int, list[tuple[int, float]]],
    player_id: int,
    game_idx: int,
    cold_init: float,
) -> float:
    prev = cold_init
    for gi, th in by_player.get(player_id, []):
        if gi >= game_idx:
            break
        prev = th
    return prev


def _mode_from_game_type(game_type: str | None) -> str:
    s = str(game_type or "offline")
    if "async" in s:
        return "async"
    if "sync" in s:
        return "sync"
    return "offline"


def _attach_model_predictions(
    rows: list[QuestionRow],
    slot_context: dict[tuple[int, int], dict],
    maps,
    res,
) -> int:
    """Fill ``p_pred`` on each row; return count of rows with a prediction."""
    by_player = _build_history_by_player(res)
    cold = float(res.cold_init_theta if res.cold_init_theta is not None else -1.0)
    canon = maps.canonical_q_idx
    game_type = maps.game_type
    n_ok = 0
    for row in rows:
        ctx = slot_context.get((row.tournament_id, row.q_in_tournament))
        if ctx is None:
            continue
        q_raw = int(ctx["q_raw"])
        c = int(canon[q_raw]) if canon is not None else q_raw
        thetas = np.array(
            [
                _theta_before(by_player, pid, int(ctx["game_idx"]), cold)
                for pid in ctx["roster_pids"]
            ],
            dtype=np.float64,
        )
        mode = row.mode
        if game_type is not None:
            mode = _mode_from_game_type(str(game_type[int(ctx["game_idx"])]))
        p = simulate_roster_on_pack(
            thetas,
            np.array([float(res.b[c])]),
            np.array([float(res.a[c])]),
            q_in_tour=np.array([row.q_in_tournament], dtype=np.int64),
            delta_size=res.delta_size,
            team_size_anchor=res.team_size_anchor,
            delta_pos=res.delta_pos,
            pos_anchor=res.pos_anchor,
            mode=mode,
            lapse_arr=res.lapse,
            recal_arr=res.recal,
        )
        row.p_pred = float(p[0])
        n_ok += 1
    return n_ok


def _load_player_questions(
    player_id: int,
    cache_file: Path,
    duckdb_path: Path,
    questions_db: Path,
) -> tuple[list[QuestionRow], dict[tuple[int, int], dict]]:
    arrays, maps = load_cached(str(cache_file))
    pidx = maps.player_id_to_idx.get(player_id)
    if pidx is None:
        raise SystemExit(f"player_id {player_id} not in {cache_file}")

    q_idx = arrays["q_idx"]
    taken = arrays["taken"]
    team_sizes = arrays["team_sizes"]
    player_flat = arrays["player_indices_flat"]
    game_idx_arr = arrays.get("game_idx")

    player_slots: dict[tuple[int, int], dict] = {}
    offset = 0
    for i in range(len(q_idx)):
        ts = int(team_sizes[i])
        players = player_flat[offset : offset + ts]
        offset += ts
        if pidx not in players:
            continue
        q = int(q_idx[i])
        tid, qi = maps.idx_to_question_id[q]
        key = (int(tid), int(qi))
        player_slots[key] = {
            "taken": int(taken[i]),
            "game_idx": int(game_idx_arr[i]) if game_idx_arr is not None else -1,
            "roster_pids": [int(maps.idx_to_player_id[int(p)]) for p in players],
            "q_raw": q,
        }

    con = duckdb.connect(str(duckdb_path), read_only=True)
    slot_rows = con.execute(
        """
        SELECT qa.tournament_id, qa.q_in_tournament,
               qa.n_obs, qa.n_taken,
               q.text, q.answer, q.b,
               t.type, t.title,
               q.canonical_idx
        FROM question_aliases qa
        JOIN questions q ON q.canonical_idx = qa.canonical_idx
        JOIN tournaments t ON t.tournament_id = qa.tournament_id
        WHERE qa.n_obs >= 5
          AND q.text IS NOT NULL AND TRIM(q.text) != ''
        """
    ).fetchall()
    con.close()

    duck_by_slot: dict[tuple[int, int], dict] = {}
    for tid, qi, n_obs, n_taken, text, answer, b, mode, title, _canon in slot_rows:
        key = (int(tid), int(qi))
        if key not in player_slots:
            continue
        duck_by_slot[key] = {
            "field_rate": float(n_taken) / float(n_obs),
            "text": text or "",
            "answer": answer or "",
            "b": float(b) if b is not None else None,
            "mode": str(mode or "offline"),
            "title": title or "",
        }

    player_tids = {k[0] for k in player_slots if k in duck_by_slot}
    qdb_meta: dict[tuple[int, int], dict] = {}
    answer_types: dict[int, list[str]] = defaultdict(list)

    conn = sqlite3.connect(str(questions_db))
    for row in conn.execute(
        """
        SELECT id, number, text, answer, tags_json, authors_json, editors_json,
               complexity_json, audio, razdatka_text, razdatka_pic, tournaments_json
        FROM questions
        WHERE tournaments_json IS NOT NULL AND tournaments_json != '[]'
        """
    ):
        (
            qid,
            number,
            _text,
            _answer,
            tags_json,
            authors_json,
            editors_json,
            complexity_json,
            audio,
            razdatka_text,
            razdatka_pic,
            tjson,
        ) = row
        if number is None:
            continue
        slot = int(number) - 1
        try:
            tournaments = json.loads(tjson)
        except json.JSONDecodeError:
            continue
        for t in tournaments:
            tid = int(t.get("id", 0))
            if tid not in player_tids:
                continue
            key = (tid, slot)
            if key in qdb_meta:
                continue
            qdb_meta[key] = {
                "question_db_id": int(qid),
                "tags": _parse_tags(tags_json),
                "authors": _parse_names(authors_json),
                "editors": _parse_names(editors_json),
                "complexity_band": _complexity_band(complexity_json),
                "has_audio": bool((audio or "").strip()),
                "has_razdatka": bool((razdatka_text or "").strip() or (razdatka_pic or "").strip()),
            }

    qids = {m["question_db_id"] for m in qdb_meta.values()}
    if qids:
        ph = ",".join("?" * len(qids))
        for qid, etype in conn.execute(
            f"""
            SELECT qe.question_id, e.type
            FROM question_entities qe
            JOIN entities e ON e.id = qe.entity_id
            WHERE qe.role = 'answer' AND qe.question_id IN ({ph})
            """,
            list(qids),
        ):
            answer_types[int(qid)].append(str(etype))

    conn.close()

    rows: list[QuestionRow] = []
    for key, slot in player_slots.items():
        duck = duck_by_slot.get(key)
        if duck is None:
            continue
        meta = qdb_meta.get(key, {})
        qid = meta.get("question_db_id")
        text = duck["text"]
        rows.append(
            QuestionRow(
                tournament_id=key[0],
                q_in_tournament=key[1],
                taken=slot["taken"],
                field_rate=duck["field_rate"],
                b=duck["b"],
                mode=duck["mode"],
                title=duck["title"],
                text=text,
                answer=duck["answer"],
                tags=meta.get("tags", []),
                authors=meta.get("authors", []),
                editors=meta.get("editors", []),
                techniques=_detect_techniques(
                    text,
                    has_audio=meta.get("has_audio", False),
                    has_razdatka=meta.get("has_razdatka", False),
                ),
                answer_entity_types=answer_types.get(qid, []) if qid else [],
                complexity_band=meta.get("complexity_band", "unknown"),
                question_db_id=qid,
            )
        )
    return rows, player_slots


def _aggregate(
    rows: list[QuestionRow],
    key_fn,
    *,
    min_n: int,
    vs: str = "field",
) -> pd.DataFrame:
    buckets: dict[str, list[QuestionRow]] = defaultdict(list)
    for r in rows:
        keys = key_fn(r)
        if not keys:
            keys = ["(не указано)"]
        for k in keys:
            buckets[str(k)].append(r)

    out = []
    for name, items in buckets.items():
        n = len(items)
        if n < min_n:
            continue
        taken = np.array([x.taken for x in items], dtype=np.float64)
        bs = [x.b for x in items if x.b is not None]
        row: dict = {
            "name": name,
            "n": n,
            "take_rate": float(taken.mean()),
            "b_mean": float(np.mean(bs)) if bs else np.nan,
        }
        if vs == "field":
            field = np.array([x.field_rate for x in items], dtype=np.float64)
            gap = taken - field
            row.update(
                {
                    "field_rate": float(field.mean()),
                    "gap": float(gap.mean()),
                    "gap_se": float(gap.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                }
            )
        else:
            pred_rows = [x for x in items if x.p_pred is not None]
            if len(pred_rows) < min_n:
                continue
            pred = np.array([x.p_pred for x in pred_rows], dtype=np.float64)
            taken_p = np.array([x.taken for x in pred_rows], dtype=np.float64)
            gap = taken_p - pred
            row.update(
                {
                    "n": len(pred_rows),
                    "take_rate": float(taken_p.mean()),
                    "p_pred": float(pred.mean()),
                    "gap": float(gap.mean()),
                    "gap_se": float(gap.std(ddof=1) / np.sqrt(len(pred_rows)))
                    if len(pred_rows) > 1
                    else 0.0,
                    "missed_prob": float(pred[taken_p == 0].sum()),
                    "lucky_prob": float((1.0 - pred[taken_p == 1]).sum()),
                }
            )
        out.append(row)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    return df.sort_values("gap", ascending=True)


def _write_report(
    out_dir: Path,
    player_id: int,
    rows: list[QuestionRow],
    slices_field: dict[str, pd.DataFrame],
    slices_model: dict[str, pd.DataFrame] | None = None,
) -> None:
    n = len(rows)
    taken = np.array([r.taken for r in rows])
    field = np.array([r.field_rate for r in rows])
    gap_field = taken - field
    overall_take = float(taken.mean())
    overall_field = float(field.mean())
    overall_gap_field = float(gap_field.mean())
    gap_label = "перевес над полем" if overall_gap_field >= 0 else "отставание от поля"

    lines = [
        f"# Профиль слабых мест — player_id {player_id}",
        "",
        f"Вопросов с текстом и полевой статистикой: **{n:,}**.",
        f"Личная доля взятий: **{overall_take:.1%}**, поле: **{overall_field:.1%}**, "
        f"{gap_label}: **{overall_gap_field:+.1%}**.",
        "",
        "**gap_field** = взял − доля поля на том же слоте.",
        "",
    ]

    pred_rows = [r for r in rows if r.p_pred is not None]
    if pred_rows:
        pred = np.array([r.p_pred for r in pred_rows], dtype=np.float64)
        taken_p = np.array([r.taken for r in pred_rows], dtype=np.float64)
        gap_model = taken_p - pred
        overall_pred = float(pred.mean())
        overall_gap_model = float(gap_model.mean())
        model_label = "перевес над прогнозом" if overall_gap_model >= 0 else "отставание от прогноза"
        missed = float(pred[taken_p == 0].sum())
        lucky = float((1.0 - pred[taken_p == 1]).sum())
        lines.extend(
            [
                f"Прогноз модели (pre-tournament θ команды): **{overall_pred:.1%}**, "
                f"{model_label}: **{overall_gap_model:+.1%}**.",
                f"Суммарно «упущенная вероятность»: **{missed:,.0f}**; "
                f"«сверх ожидания»: **{lucky:,.0f}**.",
                "",
                "**gap_model** = взял − p_pred для твоей команды.",
                "",
            ]
        )

    section_titles = {
        "tags": "Теги gotquestions (тема/приём)",
        "techniques": "Эвристические приёмы (по тексту вопроса)",
        "answer_entity_types": "Тип ответа (entity extraction)",
        "complexity_band": "Сложность (gotquestions)",
        "authors": "Авторы",
        "editors": "Редакторы",
        "mode": "Формат турнира",
    }

    lines.append("## Относительно поля")
    lines.append("")
    for key, title in section_titles.items():
        df = slices_field.get(key)
        lines.append(f"### {title}")
        lines.append("")
        if df is None or df.empty:
            lines.append("_Недостаточно данных при min_n._")
            lines.append("")
            continue
        worst = df.head(8)
        lines.append("| | n | take | field | gap |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in worst.iterrows():
            lines.append(
                f"| {r['name']} | {int(r['n'])} | {r['take_rate']:.1%} | "
                f"{r['field_rate']:.1%} | {r['gap']:+.1%} |"
            )
        lines.append("")

    if slices_model and pred_rows:
        lines.append("## Относительно прогноза модели")
        lines.append("")
        for key, title in section_titles.items():
            df = slices_model.get(key)
            lines.append(f"### {title}")
            lines.append("")
            if df is None or df.empty:
                lines.append("_Недостаточно данных при min_n._")
                lines.append("")
                continue
            worst = df.head(8)
            lines.append("| | n | take | p_pred | gap | missed Σp |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for _, r in worst.iterrows():
                lines.append(
                    f"| {r['name']} | {int(r['n'])} | {r['take_rate']:.1%} | "
                    f"{r['p_pred']:.1%} | {r['gap']:+.1%} | {r['missed_prob']:.0f} |"
                )
            lines.append("")

        model_misses = sorted(
            [r for r in pred_rows if r.taken == 0 and r.p_pred >= 0.45],
            key=lambda r: r.p_pred,
            reverse=True,
        )[:15]
        lines.append("## Промахи vs модель (p_pred ≥45 %, не взял)")
        lines.append("")
        for r in model_misses:
            snippet = re.sub(r"\s+", " ", r.text)[:160]
            lines.append(
                f"- **p={r.p_pred:.0%}** · {r.mode} · "
                f"[{r.title}](https://rating.chgk.info/tournament/{r.tournament_id}) "
                f"· q{r.q_in_tournament + 1}: _{snippet}…_ → **{r.answer}**"
            )
        lines.append("")

        lucky = sorted(
            [r for r in pred_rows if r.taken == 1 and r.p_pred <= 0.35],
            key=lambda r: r.p_pred,
        )[:15]
        lines.append("## Взял «сверх ожидания» (p_pred ≤35 %, взял)")
        lines.append("")
        for r in lucky:
            snippet = re.sub(r"\s+", " ", r.text)[:160]
            lines.append(
                f"- **p={r.p_pred:.0%}** · {r.mode} · "
                f"[{r.title}](https://rating.chgk.info/tournament/{r.tournament_id}) "
                f"· q{r.q_in_tournament + 1}: _{snippet}…_ → **{r.answer}**"
            )
        lines.append("")

    misses = sorted(
        [r for r in rows if r.taken == 0 and r.field_rate >= 0.55],
        key=lambda r: r.field_rate,
        reverse=True,
    )[:15]
    lines.append("## Промахи vs поле (поле ≥55 %, не взял)")
    lines.append("")
    for r in misses:
        snippet = re.sub(r"\s+", " ", r.text)[:160]
        extra = f", p={r.p_pred:.0%}" if r.p_pred is not None else ""
        lines.append(
            f"- **{r.field_rate:.0%}** поле{extra} · {r.mode} · "
            f"[{r.title}](https://rating.chgk.info/tournament/{r.tournament_id}) "
            f"· q{r.q_in_tournament + 1}: _{snippet}…_ → **{r.answer}**"
        )
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _slice_specs(rows: list[QuestionRow]) -> dict:
    return {
        "tags": (lambda r: r.tags, 15),
        "techniques": (lambda r: r.techniques, 30),
        "answer_entity_types": (
            lambda r: r.answer_entity_types or ["(нет entity)"],
            25,
        ),
        "complexity_band": (lambda r: [r.complexity_band], 30),
        "authors": (lambda r: r.authors, 40),
        "editors": (lambda r: r.editors, 40),
        "mode": (lambda r: [r.mode], 30),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--player-id", type=int, default=32919)
    ap.add_argument("--cache-file", type=Path, default=Path("data.npz"))
    ap.add_argument("--seq-npz", type=Path, default=Path("results/seq.npz"))
    ap.add_argument("--duckdb", type=Path, default=Path("website/data/chgk.duckdb"))
    ap.add_argument(
        "--questions-db",
        type=Path,
        default=Path("/Users/fbr/Projects/personal/chgk-embedings/data/questions.db"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to results/player_<id>_weaknesses/",
    )
    args = ap.parse_args()

    if not args.cache_file.exists():
        print(f"missing cache: {args.cache_file}", file=sys.stderr)
        return 1
    if not args.duckdb.exists():
        print(f"missing duckdb: {args.duckdb}", file=sys.stderr)
        return 1
    if not args.questions_db.exists():
        print(f"missing questions db: {args.questions_db}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or Path(f"results/player_{args.player_id}_weaknesses")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading questions for player {args.player_id}…")
    arrays, maps = load_cached(str(args.cache_file))
    rows, slot_context = _load_player_questions(
        args.player_id, args.cache_file, args.duckdb, args.questions_db
    )
    print(f"  {len(rows):,} questions with text + field stats")

    slices_model: dict[str, pd.DataFrame] | None = None
    if args.seq_npz.exists():
        print(f"Computing model predictions from {args.seq_npz}…")
        res = load_results_npz(args.seq_npz)
        n_pred = _attach_model_predictions(rows, slot_context, maps, res)
        print(f"  {n_pred:,} questions with p_pred")
    else:
        print(f"  skip model predictions: {args.seq_npz} not found", file=sys.stderr)

    per_q = pd.DataFrame(
        [
            {
                "tournament_id": r.tournament_id,
                "q_in_tournament": r.q_in_tournament,
                "taken": r.taken,
                "field_rate": r.field_rate,
                "gap_field": r.gap_field,
                "p_pred": r.p_pred,
                "gap_model": r.gap_model,
                "b": r.b,
                "mode": r.mode,
                "title": r.title,
                "text": r.text,
                "answer": r.answer,
                "tags": "|".join(r.tags),
                "authors": "|".join(r.authors),
                "editors": "|".join(r.editors),
                "techniques": "|".join(r.techniques),
                "answer_entity_types": "|".join(r.answer_entity_types),
                "complexity_band": r.complexity_band,
            }
            for r in rows
        ]
    )
    per_q.to_csv(out_dir / "questions.csv", index=False)

    specs = _slice_specs(rows)
    slices_field = {
        name: _aggregate(rows, fn, min_n=mn, vs="field")
        for name, (fn, mn) in specs.items()
    }
    if any(r.p_pred is not None for r in rows):
        slices_model = {
            name: _aggregate(rows, fn, min_n=mn, vs="model")
            for name, (fn, mn) in specs.items()
        }

    for name, df in slices_field.items():
        if not df.empty:
            df.to_csv(out_dir / f"by_{name}.csv", index=False)
    if slices_model:
        for name, df in slices_model.items():
            if not df.empty:
                df.to_csv(out_dir / f"by_{name}_model.csv", index=False)

    pred_rows = [r for r in rows if r.p_pred is not None]
    summary: dict = {
        "player_id": args.player_id,
        "n_questions": len(rows),
        "take_rate": float(np.mean([r.taken for r in rows])),
        "field_rate": float(np.mean([r.field_rate for r in rows])),
        "gap_field": float(np.mean([r.gap_field for r in rows])),
        "slices_field": {
            k: v.head(10).to_dict(orient="records")
            for k, v in slices_field.items()
            if not v.empty
        },
    }
    if pred_rows:
        summary.update(
            {
                "p_pred": float(np.mean([r.p_pred for r in pred_rows])),
                "gap_model": float(np.mean([r.gap_model for r in pred_rows])),
                "missed_prob_sum": float(
                    sum(r.p_pred for r in pred_rows if r.taken == 0)
                ),
                "lucky_prob_sum": float(
                    sum(1.0 - r.p_pred for r in pred_rows if r.taken == 1)
                ),
                "slices_model": {
                    k: v.head(10).to_dict(orient="records")
                    for k, v in (slices_model or {}).items()
                    if not v.empty
                },
            }
        )
        miss_df = pd.DataFrame(
            [
                {
                    "p_pred": r.p_pred,
                    "field_rate": r.field_rate,
                    "tournament_id": r.tournament_id,
                    "q_in_tournament": r.q_in_tournament,
                    "mode": r.mode,
                    "title": r.title,
                    "text": r.text,
                    "answer": r.answer,
                    "techniques": "|".join(r.techniques),
                }
                for r in sorted(
                    [x for x in pred_rows if x.taken == 0 and x.p_pred >= 0.45],
                    key=lambda x: x.p_pred,
                    reverse=True,
                )[:200]
            ]
        )
        if not miss_df.empty:
            miss_df.to_csv(out_dir / "model_misses.csv", index=False)
        lucky_df = pd.DataFrame(
            [
                {
                    "p_pred": r.p_pred,
                    "field_rate": r.field_rate,
                    "tournament_id": r.tournament_id,
                    "q_in_tournament": r.q_in_tournament,
                    "mode": r.mode,
                    "title": r.title,
                    "text": r.text,
                    "answer": r.answer,
                    "techniques": "|".join(r.techniques),
                }
                for r in sorted(
                    [x for x in pred_rows if x.taken == 1 and x.p_pred <= 0.35],
                    key=lambda x: x.p_pred,
                )[:200]
            ]
        )
        if not lucky_df.empty:
            lucky_df.to_csv(out_dir / "lucky_takes.csv", index=False)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(out_dir, args.player_id, rows, slices_field, slices_model)
    print(f"Wrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
