#!/usr/bin/env python3
"""
Загрузить чекпоинт (latest.pt или --save_model) и вывести/сохранить результаты последней эпохи.
Пример:
  python scripts/export_checkpoint_results.py checkpoints/latest.pt
  python scripts/export_checkpoint_results.py checkpoints/latest.pt --players_out players.csv --questions_out questions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import ChGKModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Export results from a training checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint (e.g. checkpoints/latest.pt)")
    parser.add_argument("--players_out", type=str, default=None, help="Save player_id,theta to CSV")
    parser.add_argument("--questions_out", type=str, default=None, help="Save question_id,b,a to CSV")
    parser.add_argument("--min_games", type=int, default=30, help="Only include players with at least this many games in DB (default 30). 0 = include all.")
    parser.add_argument("-n", type=int, default=10, help="Number of top/bottom to print (default 10)")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    td = ckpt.get("tournament_dl")
    tt = ckpt.get("tournament_type")
    tournament_dl = torch.from_numpy(td).float() if isinstance(td, np.ndarray) else td
    tournament_type = torch.from_numpy(tt).long() if isinstance(tt, np.ndarray) else tt
    model = ChGKModel(
        ckpt["num_players"],
        ckpt["num_questions"],
        tournament_dl=tournament_dl,
        tournament_type=tournament_type,
    )
    model.load_state_dict(ckpt["model_state"])

    theta = model.theta.detach().cpu().numpy()
    b = model.b.detach().cpu().numpy()
    a = torch.exp(model.log_a.detach()).cpu().numpy()
    idx_to_player = ckpt["idx_to_player_id"]
    idx_to_question = ckpt["idx_to_question_id"]

    epoch = ckpt.get("epoch", "?")
    print(f"Checkpoint: epoch {epoch}, players {len(theta)}, questions {len(b)}\n")

    n = args.n
    # Top players
    top = np.argsort(theta)[::-1][:n]
    print(f"Top {n} strongest players (θ):")
    for i, idx in enumerate(top, 1):
        pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
        print(f"  {i}. player_id={pid} θ={theta[idx]:.4f}")

    # Top hardest questions (high b)
    top_b = np.argsort(b)[::-1][:n]
    print(f"\nTop {n} hardest questions (b):")
    for i, idx in enumerate(top_b, 1):
        qid = idx_to_question[idx] if idx < len(idx_to_question) else idx
        print(f"  {i}. question_id={qid} b={b[idx]:.4f}")

    # Top most selective (high a)
    top_a = np.argsort(a)[::-1][:n]
    print(f"\nTop {n} most selective questions (a):")
    for i, idx in enumerate(top_a, 1):
        qid = idx_to_question[idx] if idx < len(idx_to_question) else idx
        print(f"  {i}. question_id={qid} a={a[idx]:.4f}")

    # Export CSV
    if args.players_out:
        import csv
        import os
        out = Path(args.players_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        player_ids = [idx_to_player[idx] if idx < len(idx_to_player) else idx for idx in range(len(theta))]
        allowed_idx = set(range(len(theta)))
        if args.min_games > 0 and player_ids:
            try:
                import psycopg2
                url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres")
                conn = psycopg2.connect(url)
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT player_id, COUNT(DISTINCT tournament_id)
                    FROM public.tournament_rosters
                    WHERE player_id = ANY(%s)
                    GROUP BY player_id
                    HAVING COUNT(DISTINCT tournament_id) >= %s
                    """,
                    (player_ids, args.min_games),
                )
                allowed_ids = {r[0] for r in cur.fetchall()}
                conn.close()
                allowed_idx = {idx for idx in range(len(theta)) if player_ids[idx] in allowed_ids}
                print(f"Min games {args.min_games}: exporting {len(allowed_idx)} of {len(theta)} players")
            except Exception as e:
                print(f"DB unavailable for min_games filter ({e}), exporting all players", file=sys.stderr)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["player_id", "theta"])
            for idx in sorted(allowed_idx):
                pid = idx_to_player[idx] if idx < len(idx_to_player) else idx
                w.writerow([pid, round(float(theta[idx]), 6)])
        print(f"\nPlayers saved to {out}")

    if args.questions_out:
        import csv
        out = Path(args.questions_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question_id", "b", "a"])
            for idx in range(len(b)):
                qid = idx_to_question[idx] if idx < len(idx_to_question) else idx
                # question_id can be tuple (tournament_id, question_index)
                qid_str = f"{qid[0]}_{qid[1]}" if isinstance(qid, tuple) else qid
                w.writerow([qid_str, round(float(b[idx]), 6), round(float(a[idx]), 6)])
        print(f"Questions saved to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
