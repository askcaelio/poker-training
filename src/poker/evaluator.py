"""Hand evaluator — wraps treys to give a clean Card-based API.

Treys ranks hands 1 (royal flush) through 7462 (worst high-card). We expose
that as `rank` (lower = stronger) but make HandStrength comparisons natural:
`a > b` means "hand a beats hand b". Sorting a list of HandStrength puts the
WORST hands first and the BEST hands last (standard ascending sort by strength).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import total_ordering
from typing import Iterable

from treys import Card as TreysCard
from treys import Evaluator as TreysEvaluator

from .cards import Card


_TREYS_EVAL = TreysEvaluator()


# Treys class indices → human names. Keep our own mapping so we control wording.
_CATEGORY_NAMES: dict[int, str] = {
    0: "Royal Flush",
    1: "Straight Flush",
    2: "Four of a Kind",
    3: "Full House",
    4: "Flush",
    5: "Straight",
    6: "Three of a Kind",
    7: "Two Pair",
    8: "Pair",
    9: "High Card",
}


def _to_treys(card: Card) -> int:
    return TreysCard.new(str(card))


def _to_treys_list(cards: Iterable[Card]) -> list[int]:
    return [_to_treys(c) for c in cards]


@total_ordering
@dataclass(frozen=True)
class HandStrength:
    """The strength of a 5-card poker hand made from 5+ cards.

    Attributes:
        rank: treys rank (1 = best hand possible, 7462 = worst). Lower = stronger.
        category_index: treys class index 0-9 (0 = royal flush, 9 = high card).
        category: human-readable category name.

    Comparison: a > b means a is the stronger hand.
    """

    rank: int
    category_index: int
    category: str = field(compare=False)

    def __lt__(self, other: object) -> bool:
        # Stronger = "greater". treys rank is lower-is-better, so invert.
        if not isinstance(other, HandStrength):
            return NotImplemented
        return self.rank > other.rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandStrength):
            return NotImplemented
        return self.rank == other.rank

    def __hash__(self) -> int:
        return hash(self.rank)

    def __str__(self) -> str:
        return f"{self.category} (rank {self.rank})"


def evaluate(hole_cards: Iterable[Card], board: Iterable[Card]) -> HandStrength:
    """Evaluate the best 5-card hand from hole cards + board.

    Standard Hold'em: 2 hole cards + up to 5 board cards. The evaluator picks
    the best 5-card combination automatically. Works with any total of 5-7 cards.
    """
    hole = list(hole_cards)
    bd = list(board)
    total = len(hole) + len(bd)
    if total < 5:
        raise ValueError(f"need at least 5 cards to evaluate, got {total}")
    if total > 7:
        raise ValueError(f"cannot evaluate more than 7 cards, got {total}")

    rank = _TREYS_EVAL.evaluate(_to_treys_list(bd), _to_treys_list(hole))
    cat_idx = _TREYS_EVAL.get_rank_class(rank)
    return HandStrength(rank=rank, category_index=cat_idx, category=_CATEGORY_NAMES[cat_idx])


def evaluate_5(cards: Iterable[Card]) -> HandStrength:
    """Convenience: evaluate exactly 5 cards as a single hand."""
    cards = list(cards)
    if len(cards) != 5:
        raise ValueError(f"evaluate_5 requires exactly 5 cards, got {len(cards)}")
    return evaluate(cards, [])


def compare(a: HandStrength, b: HandStrength) -> int:
    """Return 1 if a beats b, -1 if b beats a, 0 if tie."""
    if a > b:
        return 1
    if a < b:
        return -1
    return 0
