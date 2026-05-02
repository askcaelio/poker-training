"""Demo: the classic flush-draw probability question.

'I have 2 suited diamonds, there are 2 diamonds on the flop. What's the
probability I see a diamond on the next card? And by the river?'

Compares the textbook 9-out flush-draw math to the broader "any improvement"
out count, plus equity vs 1 and 5 random opponents.

Run with:  uv run python scripts/demo_diamonds.py
"""

from poker.cards import Suit, parse_cards
from poker.equity import equity_vs_random
from poker.outs import analyze_outs, prob_specific_suit


def main() -> None:
    hole = parse_cards("Ad Kd")
    board = parse_cards("7d Qd 2c")

    print("=" * 60)
    print(f"Hero:  {' '.join(c.pretty() for c in hole)}")
    print(f"Board: {' '.join(c.pretty() for c in board)}")
    print("=" * 60)

    # ----- Suit-specific probability (the user's exact question) -----
    info = prob_specific_suit(hole, board, Suit.DIAMONDS)
    print("\n--- Diamond probability (suit-specific) ---")
    print(f"Diamonds visible:       {13 - info['suit_remaining']} of 13 (4 here)")
    print(f"Diamonds remaining:     {info['suit_remaining']}")
    print(f"Unseen cards:           {info['unseen']}")
    print(f"P(diamond on turn):     {info['p_next'] * 100:.2f}%   "
          f"({info['suit_remaining']}/{info['unseen']})")
    print(f"P(diamond on river|miss turn): "
          f"{info['p_river_given_missed_turn'] * 100:.2f}%   "
          f"({info['suit_remaining']}/{info['unseen'] - 1})")
    print(f"P(at least one diamond by river): "
          f"{info['p_by_river'] * 100:.2f}%")
    print(f"  → Rule of 4 estimate (9 × 4):    36.00%")
    print(f"  → Rule of 2 single street (9×2): 18.00%")

    # ----- Full outs analysis -----
    draw = analyze_outs(hole, board)
    print("\n--- Full outs analysis (all category-improving cards) ---")
    print(f"Total outs:           {draw.out_count} / {draw.unseen}")
    print(f"Draw types:           {', '.join(draw.labels) or '(no draws)'}")
    print(f"P(any improvement next): {draw.prob_hit_next * 100:.2f}%")
    print(f"P(any improvement by river): {draw.prob_hit_by_river * 100:.2f}%")
    print(f"Rule-of-4/2 heuristic:   {draw.rule_of_4_or_2 * 100:.0f}%")

    # ----- Equity vs a random opponent -----
    eq = equity_vs_random(hole, board, num_opponents=1, iterations=20_000)
    print("\n--- Equity vs 1 random opponent ---")
    print(f"  {eq}")

    # And vs 5 opponents (full table)
    eq6 = equity_vs_random(hole, board, num_opponents=5, iterations=20_000)
    print("\n--- Equity vs 5 random opponents ---")
    print(f"  {eq6}")


if __name__ == "__main__":
    main()
