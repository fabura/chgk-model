"""Pretty-print top players from a `--players_out` CSV.

Joins the result against ``results/players_with_names.csv`` (if present)
to attach human-readable names, and shows two rankings:

* by raw θ (highlights peaks but is noisy at low ``games``)
* by ``theta_shrunk`` (Empirical-Bayes shrinkage by ``games``)

Usage:
    python scripts/show_top_players.py results/players_new_defaults.csv [TOP_N]
"""
from __future__ import annotations

import csv
import os
import sys

NAMES_PATH = "results/players_with_names.csv"


def load_names(path: str) -> dict[int, str]:
    if not os.path.exists(path):
        return {}
    out: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pid = int(row["player_id"])
            except (KeyError, ValueError):
                continue
            full = (row.get("first_name") or "").strip() + " " + (row.get("last_name") or "").strip()
            out[pid] = full.strip() or "(no name)"
    return out


def load_players(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "player_id": int(row["player_id"]),
                    "theta": float(row["theta"]),
                    "theta_shrunk": float(row.get("theta_shrunk", row["theta"])),
                    "rank_shrunk": int(row.get("rank_shrunk", 0)),
                    "games": int(row["games"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    names = load_names(NAMES_PATH)
    rows = load_players(path)
    if not rows:
        print(f"No rows in {path}")
        return 2

    print(f"Source: {path}  ({len(rows)} players)\n")

    print(f"=== Top {top_n} by raw theta (no minimum games) ===")
    print(f"{'#':>3}  {'pid':>7}  {'name':<28}  {'theta':>7}  {'shrunk':>7}  {'games':>6}")
    for i, r in enumerate(sorted(rows, key=lambda x: -x["theta"])[:top_n], 1):
        nm = names.get(r["player_id"], "")
        print(
            f"{i:>3}  {r['player_id']:>7}  {nm[:28]:<28}  "
            f"{r['theta']:>7.3f}  {r['theta_shrunk']:>7.3f}  {r['games']:>6}"
        )

    print(f"\n=== Top {top_n} by theta_shrunk (Empirical-Bayes; K=50) ===")
    print(f"{'#':>3}  {'pid':>7}  {'name':<28}  {'theta':>7}  {'shrunk':>7}  {'games':>6}")
    for i, r in enumerate(sorted(rows, key=lambda x: -x["theta_shrunk"])[:top_n], 1):
        nm = names.get(r["player_id"], "")
        print(
            f"{i:>3}  {r['player_id']:>7}  {nm[:28]:<28}  "
            f"{r['theta']:>7.3f}  {r['theta_shrunk']:>7.3f}  {r['games']:>6}"
        )

    print(f"\n=== Top {top_n} among experienced (>= 100 games) by raw theta ===")
    print(f"{'#':>3}  {'pid':>7}  {'name':<28}  {'theta':>7}  {'shrunk':>7}  {'games':>6}")
    exp = [r for r in rows if r["games"] >= 100]
    for i, r in enumerate(sorted(exp, key=lambda x: -x["theta"])[:top_n], 1):
        nm = names.get(r["player_id"], "")
        print(
            f"{i:>3}  {r['player_id']:>7}  {nm[:28]:<28}  "
            f"{r['theta']:>7.3f}  {r['theta_shrunk']:>7.3f}  {r['games']:>6}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
