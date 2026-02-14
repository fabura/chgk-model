#!/usr/bin/env python3
"""
Интерпретация силы θ в вероятности (эталонный сценарий: один игрок, вопрос b=0, a=1).
Пример: python scripts/theta_to_prob.py 0.25 0.5 1.0 1.03
"""
from __future__ import annotations

import argparse
import math
import sys


def p_one_player(theta: float) -> float:
    """Вероятность взять «средний» вопрос (b=0, a=1) одному игроку с силой θ."""
    lam = math.exp(theta)
    return 1.0 - math.exp(-lam)


def p_team_of_6(theta: float) -> float:
    """Вероятность взять «средний» вопрос командой из 6 игроков с одной силой θ."""
    lam_sum = 6.0 * math.exp(theta)
    return 1.0 - math.exp(-lam_sum)


def main() -> int:
    parser = argparse.ArgumentParser(description="θ → вероятность (эталон: b=0, a=1)")
    parser.add_argument("theta", type=float, nargs="*", help="Значения θ (можно несколько)")
    parser.add_argument("--table", action="store_true", help="Вывести таблицу для типичных θ")
    args = parser.parse_args()

    if args.table:
        print("Эталон: один игрок, вопрос с b=0, a=1\n")
        print("  θ     p(взять вопрос одному)")
        print("  " + "-" * 30)
        for t in [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            p = p_one_player(t)
            print(f"  {t:+.2f}   {p:.1%}")
        print("\nЭталон: команда из 6 игроков с одной силой θ, вопрос b=0, a=1\n")
        print("  θ     p(взять вопрос командой)")
        print("  " + "-" * 30)
        for t in [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]:
            p = p_team_of_6(t)
            print(f"  {t:+.2f}   {p:.1%}")
        return 0

    if not args.theta:
        print("Укажите θ: python scripts/theta_to_prob.py 0.5 1.0   или  --table", file=sys.stderr)
        return 1

    print("Эталон: один игрок, вопрос b=0, a=1\n")
    for t in args.theta:
        p = p_one_player(t)
        print(f"  θ = {t:+.4f}  →  p(взять вопрос) = {p:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
