"""A/B test: same table, same seed, with and without opponent modeling.

Demonstrates whether fold-equity-aware bots make better decisions vs the
original equity-only logic. Both runs use identical RNG, so any difference
in winnings is attributable to the decision logic.

Usage:
    uv run python scripts/compare_models.py [num_hands]
"""

import sys

from poker.agents import make_lag, make_maniac, make_nit, make_station, make_tag
from poker.simulator import run_simulation


def make_table() -> list:
    return [
        make_nit("Nit"),
        make_tag("TAG_1"),
        make_tag("TAG_2"),
        make_lag("LAG"),
        make_maniac("Maniac"),
        make_station("Station"),
    ]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    print(f"Running {n} hands twice — same seed (42), same opponents.")
    print("Only difference: opponent modeling on/off.\n")

    print("─── Run A: NO opponent modeling (legacy equity-only) ───")
    stats_a = run_simulation(
        agents=make_table(), num_hands=n, seed=42, use_opponent_models=False,
    )
    print(stats_a)
    print()

    print("─── Run B: WITH opponent modeling (fold-equity aware) ───")
    stats_b = run_simulation(
        agents=make_table(), num_hands=n, seed=42, use_opponent_models=True,
    )
    print(stats_b)
    print()

    print("─── Δ (B − A) won-per-hand by player ───")
    for pid in sorted(stats_a.per_player.keys()):
        a = stats_a.per_player[pid]
        b = stats_b.per_player[pid]
        delta_per_hand = b["avg_won"] - a["avg_won"]
        delta_bb100 = b["bb_per_100"] - a["bb_per_100"]
        sign = "+" if delta_per_hand >= 0 else ""
        print(f"  {pid:<14}  {sign}{delta_per_hand:>7.3f}/hand   {sign}{delta_bb100:>+7.1f} BB/100")


if __name__ == "__main__":
    main()
