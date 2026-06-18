# Кеш и результаты — `data.npz` / `results/seq.npz`

Два бинарных артефакта между Postgres и сайтом.

## `data.npz` — наблюдения

Пишет: `data.py::save_cached` / `load_from_db` + `--cache_file`.
Читает: `python -m rating --mode cached`, `website/build/build_db.py`.

**Версия**: `CACHE_VERSION_NPZ = 5` (`data.py`). При смене формата — bump version.

### Массивы наблюдений

| Ключ | Shape / тип | Описание |
|------|-------------|----------|
| `q_idx` | (n_obs,) int32 | Индекс вопроса |
| `taken` | (n_obs,) float32 | 0/1 |
| `team_sizes` | (n_obs,) int32 | Размер команды |
| `player_indices_flat` | (sum team_sizes,) int32 | Ragged rosters |
| `game_idx` | (n_obs,) int32 | Индекс турнира |
| `team_strength` | (n_obs,) float32 | Норм. место (optional) |

### Index maps

| Ключ | Описание |
|------|----------|
| `idx_to_player_id` | player_idx → player_id |
| `question_tid`, `question_qi` | question_idx → (tournament_id, q_in_tournament) |
| `question_is_tuple` | 1 если вопросы — пары (tid, qi) |
| `idx_to_game_id` | game_idx → tournament_id |
| `question_game_idx` | question_idx → game_idx |
| `game_type` | str per game: offline/sync/async |
| `game_date_ordinal` | ordinal даты |
| `tournament_dl` | true_dl per question slot |
| `tournament_type` | 0/1/2 per question slot |
| `canonical_q_idx` | raw → canonical question |
| `num_canonical_questions` | scalar |

### Восстановление Sample

Ragged format: для observation `i` ростер =
`player_indices_flat[offset:offset+team_sizes[i]]`, где offset — кумулятивная сумма `team_sizes`.

## `results/seq.npz` — параметры модели

Пишет: `python -m rating --results_npz`.
Читает: `rating/io.py::load_results_npz`, `build_db.py`.

**Версия**: ключ `version` (currently `1`).

| Ключ | Описание |
|------|----------|
| `player_id`, `theta`, `games` | Игроки |
| `question_tid`, `question_qi`, `b`, `a` | Вопросы |
| `canonical_q_idx` | Пары sync+async |
| `delta_size`, `team_size_anchor` | δ_size |
| `delta_pos`, `pos_anchor` | δ_pos |
| `lapse` | shape (3, 2) [mode, is_solo] |
| `recal` | shape (3, 2, 2) [mode, is_solo, αβ] |
| `history_player_id`, `history_game_id`, `history_theta` | История θ |
| `recenter_ord`, `recenter_delta` | Yearly gauge events |
| `cold_init_theta` | Prior для cold-start |

`history_game_id` — это **tournament_id** (не game_idx).

## Legacy `.pkl`

`CACHE_VERSION = 4`. Конвертация: `python data.py --convert_cache old.pkl new.npz`.

## Где менять формат

1. `data.py`: `CACHE_VERSION_NPZ`, `_save_arrays_maps_npz`, `_load_cached_npz`
2. `rating/__main__.py`: `_export_results_npz`
3. `rating/io.py`: `RatingResults`, `load_results_npz`
4. Все потребители + **этот файл**
