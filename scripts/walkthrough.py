"""Walk through a single hand street by street, printing the dashboard at
each decision point. Demonstrates the full Phase 1 engine in action.

Run with:  uv run python scripts/walkthrough.py
"""

import random

from poker.cards import parse_cards
from poker.dashboard import build, render


def main() -> None:
    rng = random.Random(2026)

    hole = parse_cards("Ad Kd")
    flop = parse_cards("7d Qd 2c")
    turn_card = parse_cards("3s")
    river_card = parse_cards("Th")

    print("\n" + "█" * 60)
    print(" Hand walkthrough — Hero defends BB with AdKd vs 1 opponent")
    print("█" * 60)

    # Preflop — facing a 3bb raise into 1.5bb pot (call 2bb)
    print("\n>>> PREFLOP — Villain raises 3bb. Hero faces a 2bb call into 4bb pot.")
    d = build(hole, num_opponents=1, pot=4, call=2, iterations=10_000, rng=rng)
    print(render(d))

    # Flop — flush draw + 2 overcards + nut diamond. Villain bets half pot.
    print("\n>>> FLOP — Villain bets $50 into $100 pot.")
    d = build(hole, flop, num_opponents=1, pot=100, call=50, iterations=10_000, rng=rng)
    print(render(d))

    # Turn — brick. Villain bets again.
    print("\n>>> TURN — Brick (3♠). Villain bets $100 into $200.")
    d = build(hole, flop + turn_card, num_opponents=1, pot=200, call=100,
              iterations=10_000, rng=rng)
    print(render(d))

    # River — also brick. Showdown.
    print("\n>>> RIVER — T♥ (no diamond, no improvement). No further betting.")
    d = build(hole, flop + turn_card + river_card, num_opponents=1,
              iterations=10_000, rng=rng)
    print(render(d))


if __name__ == "__main__":
    main()
