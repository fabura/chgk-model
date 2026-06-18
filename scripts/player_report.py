"""Full player profile report by rating player_id.

Produces a human-readable overview plus the detailed weakness / model-residual
artefacts from ``analyse_player_weaknesses``.

Usage:
    python scripts/player_report.py --player-id 32919
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import duckdb
import numpy as np
import pandas as pd

from data import load_cached
from rating.io import load_results_npz

# Reuse the weakness-analysis pipeline.
from analyse_player_weaknesses import (  # noqa: E402
    QuestionRow,
    _aggregate,
    _attach_model_predictions,
    _load_player_questions,
    _slice_specs,
)

_MIN_EDITOR_N = 30
_MIN_AUTHOR_N = 30


def _date_only(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "date"):
        value = value.date()
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _count_all_question_slots(player_id: int, cache_file: Path) -> dict:
    """All unique (tournament, question) slots for the player from data.npz."""
    arrays, maps = load_cached(str(cache_file))
    pidx = maps.player_id_to_idx.get(player_id)
    if pidx is None:
        raise SystemExit(f"player_id {player_id} not in {cache_file}")

    q_idx = arrays["q_idx"]
    taken = arrays["taken"]
    team_sizes = arrays["team_sizes"]
    player_flat = arrays["player_indices_flat"]

    slots: dict[tuple[int, int], int] = {}
    offset = 0
    for i in range(len(q_idx)):
        ts = int(team_sizes[i])
        players = player_flat[offset : offset + ts]
        offset += ts
        if pidx not in players:
            continue
        q = int(q_idx[i])
        tid, qi = maps.idx_to_question_id[q]
        slots[(int(tid), int(qi))] = int(taken[i])

    n_taken = sum(slots.values())
    return {
        "n_questions_total": len(slots),
        "n_taken": n_taken,
        "n_not_taken": len(slots) - n_taken,
        "take_rate": n_taken / len(slots) if slots else 0.0,
        "tournament_ids": sorted({k[0] for k in slots}),
    }


def _fetch_tournament_overview(player_id: int, duckdb_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(duckdb_path), read_only=True)
    df = con.execute(
        """
        SELECT
            pg.tournament_id,
            t.title,
            t.type AS mode,
            t.start_date,
            t.n_questions,
            tg.has_breakdown,
            tg.score_actual,
            tg.n_players_active AS team_size,
            pg.n_takes_team,
            pg.expected_takes_team
        FROM player_games pg
        JOIN team_games tg
          ON tg.tournament_id = pg.tournament_id AND tg.team_id = pg.team_id
        JOIN tournaments t ON t.tournament_id = pg.tournament_id
        WHERE pg.player_id = ?
        ORDER BY t.start_date, pg.tournament_id
        """,
        [player_id],
    ).fetchdf()
    con.close()
    return df


def _fetch_player_meta(player_id: int, duckdb_path: Path) -> dict:
    con = duckdb.connect(str(duckdb_path), read_only=True)
    row = con.execute(
        """
        SELECT player_id, first_name, last_name, theta, theta_display, games, last_game_date
        FROM players WHERE player_id = ?
        """,
        [player_id],
    ).fetchone()
    con.close()
    if row is None:
        return {"player_id": player_id}
    return {
        "player_id": int(row[0]),
        "first_name": row[1],
        "last_name": row[2],
        "theta": float(row[3]) if row[3] is not None else None,
        "theta_display": float(row[4]) if row[4] is not None else None,
        "games": int(row[5]) if row[5] is not None else None,
        "last_game_date": row[6].isoformat() if row[6] is not None else None,
    }


def _fetch_geo_overview(player_id: int, duckdb_path: Path) -> dict:
    """Geography from the ChGK map tables baked into the website DuckDB."""
    con = duckdb.connect(str(duckdb_path), read_only=True)
    countries = con.execute(
        """
        SELECT country_name, SUM(n_games) AS n_games
        FROM map_player_regions
        WHERE player_id = ? AND country_name IS NOT NULL
        GROUP BY country_name
        ORDER BY n_games DESC, country_name
        """,
        [player_id],
    ).fetchdf()
    regions = con.execute(
        """
        SELECT COALESCE(region_name, country_name) AS region_name,
               country_name,
               SUM(n_games) AS n_games
        FROM map_player_regions
        WHERE player_id = ?
        GROUP BY region_name, country_name
        ORDER BY n_games DESC, region_name
        """,
        [player_id],
    ).fetchdf()
    towns = con.execute(
        """
        SELECT town_name, region_name, country_name, SUM(n_games) AS n_games
        FROM map_player_towns
        WHERE player_id = ? AND town_name IS NOT NULL AND town_name != 'Онлайн'
        GROUP BY town_name, region_name, country_name
        ORDER BY n_games DESC, town_name
        """,
        [player_id],
    ).fetchdf()
    con.close()
    return {
        "n_countries": int(len(countries)),
        "n_regions": int(len(regions)),
        "n_towns": int(len(towns)),
        "countries": countries.to_dict(orient="records"),
        "regions": regions.to_dict(orient="records"),
        "towns": towns.to_dict(orient="records"),
    }


def _rank_people(
    rows: list[QuestionRow],
    *,
    key_fn,
    min_n: int,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (worst_by_miss_rate, best_by_take_rate) DataFrames."""
    buckets: dict[str, list[QuestionRow]] = defaultdict(list)
    for r in rows:
        keys = key_fn(r)
        if not keys:
            keys = ["(не указано)"]
        for k in keys:
            buckets[str(k)].append(r)

    records = []
    for name, items in buckets.items():
        n = len(items)
        if n < min_n:
            continue
        taken = np.array([x.taken for x in items], dtype=np.float64)
        pred = np.array(
            [x.p_pred for x in items if x.p_pred is not None], dtype=np.float64
        )
        taken_p = np.array(
            [x.taken for x in items if x.p_pred is not None], dtype=np.float64
        )
        field = np.array([x.field_rate for x in items], dtype=np.float64)
        take_rate = float(taken.mean())
        records.append(
            {
                "name": name,
                "n": n,
                "n_taken": int(taken.sum()),
                "n_not_taken": int(n - taken.sum()),
                "take_rate": take_rate,
                "miss_rate": 1.0 - take_rate,
                "field_rate": float(field.mean()),
                "gap_field": take_rate - float(field.mean()),
                "p_pred": float(pred.mean()) if len(pred) else None,
                "gap_model": float((taken_p - pred).mean()) if len(pred) else None,
            }
        )
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(records)
    worst = df.sort_values(["miss_rate", "n"], ascending=[False, False]).head(top_n)
    best = df.sort_values(["take_rate", "n"], ascending=[False, False]).head(top_n)
    return worst, best


def _write_full_report(
    out_dir: Path,
    *,
    meta: dict,
    overview: dict,
    geo: dict,
    rows: list[QuestionRow],
    slices_field: dict[str, pd.DataFrame],
    slices_model: dict[str, pd.DataFrame] | None,
    editors_worst: pd.DataFrame,
    editors_best: pd.DataFrame,
    authors_worst: pd.DataFrame,
    authors_best: pd.DataFrame,
) -> None:
    name = " ".join(
        x for x in (meta.get("first_name"), meta.get("last_name")) if x
    ) or f"player {meta.get('player_id')}"
    theta = meta.get("theta")
    theta_text = f"{theta:+.2f}" if theta is not None else "нет данных"

    lines = [
        f"# Отчёт по игроку — {name}",
        "",
        f"`player_id={meta.get('player_id')}` · рейтинг модели θ: **{theta_text}** · "
        f"турниров в модели: **{meta.get('games', '—')}**",
        "",
        "Этот отчёт отвечает на три практических вопроса: сколько данных есть по игроку, "
        "какие части этих данных можно анализировать по вопросам, и где игрок чаще "
        "всего ошибается или, наоборот, стабильно берёт.",
        "",
        "> Коротко о метриках: **take%** — процент взятых вопросов; **miss%** — процент "
        "невзятых; **поле** — средний результат всех команд на этих же вопросах; "
        "**прогноз модели** — ожидаемый процент взятий команды игрока перед турниром; "
        "**отрыв** — разница между take% игрока и сравнением. Отрицательный отрыв "
        "значит, что игрок/команда взяли меньше, чем ожидалось.",
        "",
        "## 1. Данных достаточно?",
        f"| | |",
        f"|---|---:|",
        f"| Период | {overview['first_date']} — {overview['last_date']} "
        f"({overview['span_years']:.1f} лет) |",
        f"| Турниров (уникальных) | {overview['n_tournaments']:,} |",
        f"| С покомандной **расплюской** (points_mask) | "
        f"{overview['n_tournaments_with_breakdown']:,} "
        f"({overview['pct_tournaments_with_breakdown']:.0f}%) |",
        f"| Без расплюски (только место/сумма) | "
        f"{overview['n_tournaments_without_breakdown']:,} |",
        f"| **Вопросов сыграно** (уник. слоты) | {overview['n_questions_total']:,} |",
        f"| — взято | {overview['n_taken']:,} ({overview['take_rate']:.1%}) |",
        f"| — не взято | {overview['n_not_taken']:,} |",
        f"| С **текстом** вопроса (questions.db) | {overview['n_questions_with_text']:,} "
        f"({overview['pct_questions_with_text']:.0f}%) |",
        f"| Без текста | {overview['n_questions_without_text']:,} |",
        "",
        "Расплюска означает, что в базе есть строка 0/1 по каждому вопросу турнира: "
        "можно узнать, какие именно вопросы команда взяла. Если расплюски нет, турнир "
        "учитывается в числе игр, но не участвует в вопросном анализе. Текст вопроса "
        "нужен для авторов, редакторов, тем и приёмов; поэтому подробные срезы ниже "
        "считаются только на вопросах с текстом.",
        "",
        "### Турниры по формату",
        "",
        "| Формат | турниров | с расплюской |",
        "|---|---:|---:|",
    ]
    for mode, rec in sorted(overview.get("by_mode", {}).items()):
        lines.append(
            f"| {mode} | {rec['n_tournaments']:,} | "
            f"{rec['n_with_breakdown']:,} |"
        )
    lines.append("")
    if geo.get("n_countries"):
        lines.extend(
            [
                "### География игр",
                "",
                f"По данным ЧГК-карты у игрока есть игры в **{geo['n_countries']}** "
                f"странах, **{geo['n_regions']}** регионах и **{geo['n_towns']}** городах.",
                "",
                "| Страна | игр |",
                "|---|---:|",
            ]
        )
        for rec in geo["countries"][:8]:
            lines.append(f"| {rec['country_name']} | {int(rec['n_games'])} |")
        lines.extend(["", "| Город | регион / страна | игр |", "|---|---|---:|"])
        for rec in geo["towns"][:8]:
            place = " / ".join(
                str(x)
                for x in (rec.get("region_name"), rec.get("country_name"))
                if x and str(x) != "nan"
            )
            lines.append(f"| {rec['town_name']} | {place} | {int(rec['n_games'])} |")
        lines.append("")

    # Text subset stats
    if rows:
        taken = np.mean([r.taken for r in rows])
        field = np.mean([r.field_rate for r in rows])
        lines.extend(
            [
                "## 2. Вопросы с текстом — сводка",
                "",
                f"Дальше анализируются **{len(rows):,}** вопросов, где одновременно есть "
                "и результат игрока, и текст вопроса, и статистика поля.",
                "",
                f"| Сравнение | взято игроком | ориентир | отрыв |",
                f"|---|---:|---:|---:|",
                f"| с полем | {taken:.1%} | {field:.1%} | {taken - field:+.1%} |",
            ]
        )
        pred_rows = [r for r in rows if r.p_pred is not None]
        if pred_rows:
            p = np.mean([r.p_pred for r in pred_rows])
            lines.append(
                f"| с прогнозом модели | {taken:.1%} | {p:.1%} | {taken - p:+.1%} |"
            )
            missed = sum(r.p_pred for r in pred_rows if r.taken == 0)
            lucky = sum(1.0 - r.p_pred for r in pred_rows if r.taken == 1)
            lines.extend(
                [
                    "",
                    f"Модель ожидала примерно **{missed:,.0f}** взятий на вопросах, "
                    "которые команда не взяла. Это не «ошибки игрока», а суммарный "
                    "вес промахов: промах на вопросе с прогнозом 90% весит 0.9, "
                    "а с прогнозом 20% — 0.2.",
                    "",
                    f"Наоборот, на взятых вопросах с низкой вероятностью команда набрала "
                    f"примерно **{lucky:,.0f}** взятий сверх ожидания модели.",
                    "",
                ]
            )
        else:
            lines.append("")

    lines.extend(
        [
            "## 3. Редакторы",
            "",
            "Здесь вопрос считается у редактора, если он указан в метаданных пакета. "
            "Если у пакета несколько редакторов, один вопрос попадёт к каждому из них.",
            "",
        ]
    )
    if editors_worst.empty:
        lines.append("_Мало данных (min 30 вопросов на редактора)._")
    else:
        lines.append("### Где чаще всего не брал")
        lines.append("")
        lines.append(
            "| Редактор | вопросов | взял | не взял | не взял, % | взял, % | отрыв от поля | отрыв от модели |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in editors_worst.iterrows():
            gm = f"{r['gap_model']:+.1%}" if r["gap_model"] is not None else "—"
            lines.append(
                f"| {r['name']} | {int(r['n'])} | {int(r['n_taken'])} | "
                f"{int(r['n_not_taken'])} | {r['miss_rate']:.1%} | "
                f"{r['take_rate']:.1%} | {r['gap_field']:+.1%} | {gm} |"
            )
        lines.append("")
    if not editors_best.empty:
        lines.append("### Где чаще всего брал")
        lines.append("")
        lines.append("| Редактор | вопросов | взял, % | отрыв от поля |")
        lines.append("|---|---:|---:|---:|")
        for _, r in editors_best.head(10).iterrows():
            lines.append(
                f"| {r['name']} | {int(r['n'])} | {r['take_rate']:.1%} | "
                f"{r['gap_field']:+.1%} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. Авторы",
            "",
            "Авторы считаются так же: если вопрос подписан несколькими авторами, "
            "он учитывается у каждого. Поэтому суммы по авторам не обязаны совпадать "
            "с общим числом вопросов.",
            "",
        ]
    )
    if authors_best.empty:
        lines.append("_Мало данных (min 30 вопросов на автора)._")
    else:
        lines.append("### Любимые авторы: где чаще всего брал")
        lines.append("")
        lines.append(
            "| Автор | вопросов | взял | не взял | взял, % | отрыв от поля | отрыв от модели |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in authors_best.iterrows():
            gm = f"{r['gap_model']:+.1%}" if r["gap_model"] is not None else "—"
            lines.append(
                f"| {r['name']} | {int(r['n'])} | {int(r['n_taken'])} | "
                f"{int(r['n_not_taken'])} | {r['take_rate']:.1%} | "
                f"{r['gap_field']:+.1%} | {gm} |"
            )
        lines.append("")
    if not authors_worst.empty:
        lines.append("### Авторы с самым высоким процентом невзятий")
        lines.append("")
        lines.append("| Автор | вопросов | не взял, % | взял, % |")
        lines.append("|---|---:|---:|---:|")
        for _, r in authors_worst.iterrows():
            lines.append(
                f"| {r['name']} | {int(r['n'])} | {r['miss_rate']:.1%} | "
                f"{r['take_rate']:.1%} |"
            )
        lines.append("")

    section_titles = {
        "techniques": "Приёмы (эвристика)",
        "tags": "Теги",
        "mode": "Формат",
        "answer_entity_types": "Тип ответа",
    }
    lines.extend(
        [
            "## 5. Темы, приёмы и форматы",
            "",
            "В этих таблицах показаны не самые частые категории, а категории с самым "
            "низким отрывом. Это ответ на вопрос «где результат относительно слабее».",
            "",
        ]
    )
    for key, title in section_titles.items():
        df_f = slices_field.get(key)
        df_m = slices_model.get(key) if slices_model else None
        if (df_f is None or df_f.empty) and (df_m is None or df_m.empty):
            continue
        lines.append(f"### {title}")
        lines.append("")
        if df_f is not None and not df_f.empty:
            lines.append("**Сравнение с полем:**")
            lines.append("")
            lines.append("| Категория | вопросов | взял, % | поле, % | отрыв |")
            lines.append("|---|---:|---:|---:|---:|")
            for _, r in df_f.head(5).iterrows():
                lines.append(
                    f"| {r['name']} | {int(r['n'])} | {r['take_rate']:.1%} | "
                    f"{r['field_rate']:.1%} | {r['gap']:+.1%} |"
                )
            lines.append("")
        if df_m is not None and not df_m.empty:
            lines.append("**Сравнение с прогнозом модели:**")
            lines.append("")
            lines.append("| Категория | вопросов | взял, % | прогноз, % | отрыв | вес промахов |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for _, r in df_m.head(5).iterrows():
                lines.append(
                    f"| {r['name']} | {int(r['n'])} | {r['take_rate']:.1%} | "
                    f"{r['p_pred']:.1%} | {r['gap']:+.1%} | {r['missed_prob']:.0f} |"
                )
            lines.append("")

    lines.extend(
        [
            "## 6. Как читать отчёт",
            "",
            "- Большой процент невзятий сам по себе не всегда проблема: если вопросы "
            "очень сложные, поле тоже их не берёт. Поэтому рядом есть отрыв от поля.",
            "- Отрыв от модели полезен для состава конкретной команды: модель знает силу "
            "игроков до турнира и сложность вопроса. Если отрыв отрицательный, команда "
            "брала меньше, чем ожидалось.",
            "- Разделы по редакторам и авторам считаются только на вопросах с текстами и "
            "метаданными. Для редких авторов/редакторов используется порог минимум "
            "30 вопросов, чтобы не делать выводы по 3-5 случайным вопросам.",
            "- В отчёт намеренно не включён список отдельных невзятых вопросов: он быстро "
            "раздувает markdown и хуже подходит для отправки игроку.",
            "",
            "_Дальше можно добавить тематические кластеры и персональный вклад игрока "
            "внутри команды, но это отдельный слой анализа._",
            "",
        ]
    )

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_player_report(
    player_id: int,
    *,
    cache_file: Path,
    seq_npz: Path,
    duckdb_path: Path,
    questions_db: Path,
    out_dir: Path,
    min_editor_n: int = _MIN_EDITOR_N,
    min_author_n: int = _MIN_AUTHOR_N,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _fetch_player_meta(player_id, duckdb_path)
    geo = _fetch_geo_overview(player_id, duckdb_path)
    tourn_df = _fetch_tournament_overview(player_id, duckdb_path)
    slot_stats = _count_all_question_slots(player_id, cache_file)

    arrays, maps = load_cached(str(cache_file))
    rows, slot_context = _load_player_questions(
        player_id, cache_file, duckdb_path, questions_db
    )

    if seq_npz.exists():
        res = load_results_npz(seq_npz)
        _attach_model_predictions(rows, slot_context, maps, res)

    n_with_text = len(rows)
    n_total = slot_stats["n_questions_total"]
    first_date = tourn_df["start_date"].min() if len(tourn_df) else None
    last_date = tourn_df["start_date"].max() if len(tourn_df) else None
    span_years = (
        (last_date - first_date).days / 365.25
        if first_date and last_date
        else 0.0
    )

    n_with_bd = int(tourn_df["has_breakdown"].sum()) if len(tourn_df) else 0
    n_tourn = len(tourn_df)

    by_mode: dict[str, dict] = {}
    if len(tourn_df):
        for mode, sub in tourn_df.groupby("mode"):
            by_mode[str(mode)] = {
                "n_tournaments": int(len(sub)),
                "n_with_breakdown": int(sub["has_breakdown"].sum()),
            }

    overview = {
        **meta,
        **{k: v for k, v in slot_stats.items() if k != "tournament_ids"},
        "n_tournament_ids": len(slot_stats["tournament_ids"]),
        "first_date": _date_only(first_date),
        "last_date": _date_only(last_date),
        "span_years": span_years,
        "n_tournaments": n_tourn,
        "n_tournaments_with_breakdown": n_with_bd,
        "n_tournaments_without_breakdown": n_tourn - n_with_bd,
        "pct_tournaments_with_breakdown": (
            100.0 * n_with_bd / n_tourn if n_tourn else 0.0
        ),
        "n_questions_with_text": n_with_text,
        "n_questions_without_text": n_total - n_with_text,
        "pct_questions_with_text": (
            100.0 * n_with_text / n_total if n_total else 0.0
        ),
        "geo": {
            "n_countries": geo["n_countries"],
            "n_regions": geo["n_regions"],
            "n_towns": geo["n_towns"],
        },
        "by_mode": by_mode,
    }

    specs = _slice_specs(rows)
    slices_field = {
        name: _aggregate(rows, fn, min_n=mn, vs="field")
        for name, (fn, mn) in specs.items()
    }
    slices_model = None
    if any(r.p_pred is not None for r in rows):
        slices_model = {
            name: _aggregate(rows, fn, min_n=mn, vs="model")
            for name, (fn, mn) in specs.items()
        }

    editors_worst, editors_best = _rank_people(
        rows, key_fn=lambda r: r.editors, min_n=min_editor_n
    )
    authors_worst, authors_best = _rank_people(
        rows, key_fn=lambda r: r.authors, min_n=min_author_n
    )

    pred_rows = [r for r in rows if r.p_pred is not None]

    full_summary = {
        "overview": overview,
        "geo": geo,
        "text_subset": {
            "n": len(rows),
            "take_rate": float(np.mean([r.taken for r in rows])) if rows else None,
            "field_rate": float(np.mean([r.field_rate for r in rows])) if rows else None,
            "gap_field": float(np.mean([r.gap_field for r in rows])) if rows else None,
            "p_pred": float(np.mean([r.p_pred for r in pred_rows])) if pred_rows else None,
            "gap_model": float(np.mean([r.gap_model for r in pred_rows]))
            if pred_rows
            else None,
        },
        "top_editors_worst_miss": editors_worst.to_dict(orient="records"),
        "top_editors_best_take": editors_best.to_dict(orient="records"),
        "top_authors_best_take": authors_best.to_dict(orient="records"),
        "top_authors_worst_miss": authors_worst.to_dict(orient="records"),
    }
    _write_full_report(
        out_dir,
        meta=meta,
        overview=overview,
        geo=geo,
        rows=rows,
        slices_field=slices_field,
        slices_model=slices_model,
        editors_worst=editors_worst,
        editors_best=editors_best,
        authors_worst=authors_worst,
        authors_best=authors_best,
    )
    return full_summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--player-id", type=int, required=True)
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
        help="defaults to results/player_<id>_report/",
    )
    ap.add_argument("--min-editor-n", type=int, default=_MIN_EDITOR_N)
    ap.add_argument("--min-author-n", type=int, default=_MIN_AUTHOR_N)
    args = ap.parse_args()

    for label, path in [
        ("cache", args.cache_file),
        ("duckdb", args.duckdb),
        ("questions-db", args.questions_db),
    ]:
        if not path.exists():
            print(f"missing {label}: {path}", file=sys.stderr)
            return 1

    out_dir = args.out_dir or Path(f"results/player_{args.player_id}_report")
    print(f"Building report for player {args.player_id}…")
    run_player_report(
        args.player_id,
        cache_file=args.cache_file,
        seq_npz=args.seq_npz,
        duckdb_path=args.duckdb,
        questions_db=args.questions_db,
        out_dir=out_dir,
        min_editor_n=args.min_editor_n,
        min_author_n=args.min_author_n,
    )
    print(f"Wrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
