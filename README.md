# ChGK probabilistic model

Estimates **player strength** θ_k, **question difficulty** b_i, and **question discrimination** a_i from binary team answers (taken / not taken) and team compositions per game.

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

Training uses **GPU** when available: **CUDA** (NVIDIA) or **MPS** (Apple Silicon). You’ll see `Using device: cuda` or `Using device: mps` at startup; otherwise it runs on CPU.

## Run synthetic demo

```bash
python train.py --mode synthetic
```

## Run on real DB (after `docker-compose up` in rating-db)

```bash
export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/postgres
python train.py --mode db
```

### Cache: avoid re-querying the DB

First run: load from DB and save to a cache file. Later runs: load from the file (no DB needed).

```bash
# First run: fetches from DB and writes data/cache.pkl
python train.py --mode db --cache_file data/cache.pkl

# Next runs: use cache only (DB can be stopped)
python train.py --mode db --cache_file data/cache.pkl
```

Use different cache paths for different options (e.g. `data/cache_100.pkl` for `--max_tournaments 100`). Delete the file to force a fresh DB export.

### Save trained model

To write the fitted model (θ, b, a) and index maps to disk for later use or analysis:

```bash
python train.py --mode db --cache_file data/cache.pkl --save_model results/model.pt
```

The `.pt` file contains `model_state`, `num_players`, `num_questions`, `idx_to_player_id`, `idx_to_question_id`. Load with `torch.load(path)` and rehydrate `ChGKModel` from the state dict.

### Checkpoints: save every epoch and resume

To save the current weights after each epoch and be able to restart without losing progress:

```bash
# First run: save checkpoints to a directory (overwrites latest.pt each epoch)
python train.py --mode db --cache_file data/cache.pkl --checkpoint_dir checkpoints

# After interrupt or crash: resume from the last saved epoch
python train.py --mode db --cache_file data/cache.pkl --checkpoint_dir checkpoints --resume checkpoints/latest.pt
```

Use the same `--cache_file` and `--seed` when resuming so the train/val split matches.

### Интерпретация силы θ

Сила игрока θ задаёт его вклад в вероятность взятия вопроса; сама вероятность зависит ещё от сложности вопроса \(b_i\), дискриминации \(a_i\) и состава команды. В **эталонном** случае (один игрок, «средний» вопрос с \(b=0\), \(a=1\)): вероятность взять вопрос = \(1 - \exp(-\exp(\theta))\) — например θ=0 → ~63%, θ=1 → ~93%. Подробно и таблицы: [docs/interpretation.md](docs/interpretation.md). Перевести θ в вероятность: `python scripts/theta_to_prob.py 0.5 1.0` или `python scripts/theta_to_prob.py --table`.

## Layout

- `data.py` — data loader, index maps, synthetic data, DB loader (points_mask → samples).
- `model.py` — `ChGKModel`: θ, b, log_a; forward gives team take probability per question.
- `train.py` — loss (binary CE + L2), identifiability (center θ), training loop.
- `metrics.py` — logloss, Brier, AUC, calibration curve.
