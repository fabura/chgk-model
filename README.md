# ChGK probabilistic model

Sequential online rating: estimates **player strength** θ_k, **question difficulty** b_i, and **question discrimination** a_i from binary team answers (taken / not taken) and team compositions. Processes tournaments week by week.

Current default handling of tournament modes:

- tournament shift is decomposed as `delta_t = mu_type[type_t] + eps_t`
- `offline` is the baseline mode
- `async` updates are intentionally weaker for players, question
  difficulty, and especially question discrimination
- tuned defaults currently use the `t6` configuration from
  `docs/async_mode_experiments.md`

## Data mapping from rating DB

- **Games**: `public.tournaments` (one tournament = one game with `questions_count` questions).
- **Question-level outcomes**: `public.tournament_results.points_mask` — string of `'0'`/`'1'` per question (1 = taken).
- **Rosters**: `public.tournament_rosters` (tournament_id, team_id, player_id).

So we have: `(game_id, question_id, team_id, taken)` with `game_id = tournament_id`, `question_id = (tournament_id, question_index)` mapped to a global question index.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # or Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run synthetic demo

```bash
python -m rating --mode synthetic
```

## Run on real DB (after `docker-compose up` in rating-db)

```bash
export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/postgres
python -m rating --mode db --cache_file data.npz
```

### Cache: avoid re-querying the DB

First run: load from DB and save to a cache file. Later runs: load from the file (no DB needed).

```bash
# First run: fetches from DB and writes data.npz
python -m rating --mode db --cache_file data.npz

# Next runs: use cache only (DB can be stopped)
python -m rating --mode cached --cache_file data.npz
```

Use different cache paths for different options (e.g. `data/cache_100.npz` for `--max_tournaments 100`). Delete the file to force a fresh DB export.

### Export results

```bash
# Compact .npz (players, questions, history)
python -m rating --mode cached --cache_file data.npz --results_npz results/seq.npz

# CSV exports
python -m rating --mode cached --cache_file data.npz \
    --players_out results/seq_players.csv \
    --questions_out results/seq_questions.csv
```

### Интерпретация силы θ

Сила игрока θ задаёт его вклад в вероятность взятия вопроса; сама вероятность зависит ещё от сложности вопроса \(b_i\), дискриминации \(a_i\) и состава команды. В **эталонном** случае (один игрок, «средний» вопрос с \(b=0\), \(a=1\)): вероятность взять вопрос = \(1 - \exp(-\exp(\theta))\) — например θ=0 → ~63%, θ=1 → ~93%. Подробно и таблицы: [docs/interpretation.md](docs/interpretation.md). Перевести θ в вероятность: `python scripts/theta_to_prob.py 0.5 1.0` или `python scripts/theta_to_prob.py --table`.

## Layout

- `data.py` — data loader, index maps, synthetic data, DB loader (points_mask → samples).
- `rating/` — sequential online rating: model, engine, players, questions, decay, tournaments.

## Notes

- `docs/async_mode_experiments.md` — checked hypotheses, measured
  backtest results, chosen defaults, and future ideas for mode handling
- `docs/current_model_mechanics.md` — historical note about the previous
  offline/training pipeline; not the source of truth for the current
  `rating/` package
