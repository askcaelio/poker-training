"""Outs counter and street probabilities — the heart of probability intuition.

An "out" is a card that improves your hand. We count them by enumeration: walk
all 47 (flop) or 46 (turn) unseen cards, evaluate hero with each one added to
the board, and any card that bumps the hand to a stronger category counts.

Why enumerate instead of pattern-match? Because pattern detection ("you have a
flush draw → 9 outs") gets fragile fast (combo draws, blockers, the board
already counterfeiting your draw). Enumeration is dead-simple and provably
correct. We add pattern detection on top purely for *labeling* (so the
dashboard can say "9 outs to flush + 4 to straight = 13 unique outs").

The probability math:
  - Cards unseen on the flop: 47 (52 - 2 hole - 3 board)
  - P(hit on turn) = outs / 47
  - P(hit on river | missed turn) = outs / 46
  - P(hit by river) = 1 - (47-outs)/47 * (46-outs)/46
  - Rule of 4 (flop, both streets): outs * 4   (rough — overestimates above ~9 outs)
  - Rule of 2 (single street): outs * 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .cards import Card, Rank, Suit, full_deck
from .evaluator import evaluate


@dataclass(frozen=True)
class DrawAnalysis:
    """Outs and probabilities for hero on flop or turn.

    Attributes:
        outs:               cards that improve hero's hand category
        out_count:          number of outs
        outs_by_destination:  {destination_category: tuple of cards} — splits outs
                              by what category they make ("Flush", "Pair", "Two Pair", etc.)
        unseen:             total unseen cards (47 on flop, 46 on turn)
        prob_hit_next:      exact P(any out on the very next card)
        prob_hit_by_river:  exact P(at least one out by the river)
        rule_of_4_or_2:     heuristic estimate (rule of 4 from flop, rule of 2 from turn)
        rule_reliable:      false when out_count > ~13; rule of 4 noticeably overestimates
        labels:             human-readable draw types ("Flush draw", "OESD", "Gutshot", etc.)
    """

    outs: tuple[Card, ...]
    out_count: int
    outs_by_destination: dict[str, tuple[Card, ...]]
    unseen: int
    prob_hit_next: float
    prob_hit_by_river: float
    rule_of_4_or_2: float
    rule_reliable: bool
    labels: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        label_str = f"  [{', '.join(self.labels)}]" if self.labels else ""
        return (
            f"{self.out_count} outs / {self.unseen} unseen  "
            f"next: {self.prob_hit_next * 100:.1f}%  "
            f"by river: {self.prob_hit_by_river * 100:.1f}%  "
            f"(rule estimate: {self.rule_of_4_or_2 * 100:.0f}%)"
            f"{label_str}"
        )

    def breakdown(self) -> str:
        """Multi-line breakdown of outs by destination category."""
        if not self.outs_by_destination:
            return "(no outs)"
        lines = []
        # Show strongest categories first (Flush before Pair, etc.)
        order = ["Royal Flush", "Straight Flush", "Four of a Kind", "Full House",
                 "Flush", "Straight", "Three of a Kind", "Two Pair", "Pair"]
        for cat in order:
            if cat in self.outs_by_destination:
                cards = self.outs_by_destination[cat]
                lines.append(f"  {len(cards):>2} to {cat}")
        return "\n".join(lines)


def analyze_outs(hole: Iterable[Card], board: Iterable[Card]) -> DrawAnalysis:
    """Enumerate outs that improve hero's hand category on the next card.

    Works on flop (3-card board) or turn (4-card board). On the river there
    are no more cards to come, so calling this with a 5-card board raises.
    """
    hole = list(hole)
    board = list(board)
    if len(hole) != 2:
        raise ValueError(f"hero needs exactly 2 hole cards, got {len(hole)}")
    if len(board) not in (3, 4):
        raise ValueError(
            f"outs analysis requires flop (3) or turn (4) board, got {len(board)}"
        )

    known = set(hole) | set(board)
    unseen_cards = [c for c in full_deck() if c not in known]
    unseen_count = len(unseen_cards)

    current = evaluate(hole, board)
    outs: list[Card] = []
    by_dest: dict[str, list[Card]] = {}
    for c in unseen_cards:
        improved = evaluate(hole, board + [c])
        # An "out" is a category change (e.g., high card → pair, pair → flush).
        # Mere kicker improvements within the same category don't count.
        # Lower category_index = stronger category (treys convention).
        if improved.category_index < current.category_index:
            outs.append(c)
            by_dest.setdefault(improved.category, []).append(c)

    out_count = len(outs)

    # Probabilities
    prob_next = out_count / unseen_count
    if len(board) == 3:
        # Flop: two more cards to come (turn + river)
        miss_turn = (unseen_count - out_count) / unseen_count
        miss_river = (unseen_count - 1 - out_count) / (unseen_count - 1)
        prob_by_river = 1 - miss_turn * miss_river
        rule = out_count * 4 / 100.0
    else:
        # Turn: one card left
        prob_by_river = prob_next
        rule = out_count * 2 / 100.0

    # Rule of 4/2 was designed for ~8-9 outs (flush/straight draws). Above ~13
    # outs the linear approximation deviates noticeably from the exact value.
    rule_reliable = out_count <= 13

    labels = _label_draws(hole, board, outs)

    return DrawAnalysis(
        outs=tuple(outs),
        out_count=out_count,
        outs_by_destination={k: tuple(v) for k, v in by_dest.items()},
        unseen=unseen_count,
        prob_hit_next=prob_next,
        prob_hit_by_river=prob_by_river,
        rule_of_4_or_2=rule,
        rule_reliable=rule_reliable,
        labels=tuple(labels),
    )


def prob_specific_suit(hole: Iterable[Card], board: Iterable[Card], suit: Suit) -> dict:
    """Probability calc for a specific suit appearing on coming streets.

    Useful for the user's example: "I have 2 suited diamonds, 2 diamonds on
    the flop — what's P(diamond on turn)?"

    Returns a dict with: suit_count_remaining, unseen, p_next, p_by_river,
    p_river_given_missed_turn (when on flop).
    """
    hole = list(hole)
    board = list(board)
    if len(board) not in (3, 4):
        raise ValueError(f"need flop or turn board, got {len(board)}")

    visible_of_suit = sum(1 for c in hole + board if c.suit == suit)
    suit_remaining = 13 - visible_of_suit
    known = set(hole) | set(board)
    unseen = 52 - len(known)

    p_next = suit_remaining / unseen
    result = {
        "suit": suit,
        "suit_remaining": suit_remaining,
        "unseen": unseen,
        "p_next": p_next,
    }

    if len(board) == 3:
        miss_turn_p = (unseen - suit_remaining) / unseen
        p_river_given_miss = suit_remaining / (unseen - 1)
        p_by_river = 1 - miss_turn_p * (1 - p_river_given_miss)
        result["p_river_given_missed_turn"] = p_river_given_miss
        result["p_by_river"] = p_by_river
    else:
        result["p_by_river"] = p_next

    return result


def _label_draws(hole: list[Card], board: list[Card], outs: list[Card]) -> list[str]:
    """Identify common draw types for display. Purely cosmetic — counts come
    from enumeration. Multiple labels can apply (e.g. flush draw + OESD).
    """
    labels: list[str] = []
    all_cards = hole + board

    # Flush draw: 4 of one suit visible
    suit_counts: dict[Suit, int] = {}
    for c in all_cards:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
    flush_suits = [s for s, n in suit_counts.items() if n == 4]
    if flush_suits:
        labels.append("Flush draw")

    # Straight draws — _count_straight_outs returns DISTINCT RANKS that
    # complete a straight. 2 ranks = open-ended (8 cards), 1 = gutshot (4),
    # 3+ = "double gutter" or wraparound territory.
    rank_set = {c.rank for c in all_cards}
    straight_rank_count = _count_straight_outs(rank_set, outs, board)
    if straight_rank_count == 2:
        labels.append("OESD")
    elif straight_rank_count == 1:
        labels.append("Gutshot")
    elif straight_rank_count >= 3:
        labels.append(f"Multiple straight draws ({straight_rank_count} ranks)")

    # Overcards: hole card ranks above all board ranks (when no pair)
    if len(board) == 3:
        max_board_rank = max(c.rank for c in board)
        overcards = [h for h in hole if h.rank > max_board_rank]
        # Only label as overcards if hero doesn't already have a pair
        from collections import Counter
        rank_counter = Counter(c.rank for c in all_cards)
        has_pair = any(n >= 2 for n in rank_counter.values())
        if overcards and not has_pair:
            labels.append(f"{len(overcards)} overcard{'s' if len(overcards) > 1 else ''}")

    return labels


def _count_straight_outs(rank_set: set[Rank], outs: list[Card], board: list[Card]) -> int:
    """Count distinct ranks among outs that complete a straight.

    Approximation: a card is a "straight out" if adding its rank to the visible
    rank set creates 5 consecutive ranks somewhere. Doesn't double-count when
    multiple suits of the same rank are outs (e.g. 4 jacks completing the same
    straight count as 1 distinct rank, but as up to 4 cards in the outs list).
    """
    straight_out_ranks: set[Rank] = set()
    for c in outs:
        if c.rank in straight_out_ranks:
            continue
        candidate = rank_set | {c.rank}
        if _has_straight(candidate):
            # Only count if ADDING this rank made the straight (i.e., it didn't
            # already exist without it)
            if not _has_straight(rank_set):
                straight_out_ranks.add(c.rank)
    return len(straight_out_ranks)


def _has_straight(ranks: set[Rank]) -> bool:
    """True if the set contains 5 consecutive ranks (wheel A-2-3-4-5 included)."""
    rank_ints = {int(r) for r in ranks}
    # Treat ace as low for wheel detection
    if Rank.ACE in ranks:
        rank_ints.add(1)
    for low in range(1, 11):  # straights starting at A-low (1) through 10
        if all(low + i in rank_ints for i in range(5)):
            return True
    return False
