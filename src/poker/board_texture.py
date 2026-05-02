"""Board texture analysis — classifies a flop/turn/river by structural properties.

Texture matters for bet sizing: wet boards (lots of draws available) deserve
bigger bets to charge those draws; dry boards reward smaller probing bets.

Properties tracked:
  - suit_distribution: max suit count, e.g. (3,1,1,0) on Qs Js 4s = monotone
  - paired:            two cards of same rank present
  - connectedness:     count of straight draws possible (0-7 scale)
  - high_card_count:   broadway cards (T+) on board

`is_wet` is True when there are notable flush draws OR straight draws
available. Wet boards favor big bets (charge draws); dry boards favor small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cards import Card, Rank, Suit


@dataclass(frozen=True)
class BoardTexture:
    suit_max: int                 # most cards of a single suit on the board
    paired: bool                  # two or more cards of same rank
    connected_count: int          # rough straight-availability score (0-7)
    high_card_count: int          # number of T+ ranks
    n_cards: int                  # 3 / 4 / 5

    @property
    def is_monotone(self) -> bool:
        return self.suit_max >= 3 and self.n_cards <= 4 or self.suit_max >= 5

    @property
    def is_two_tone(self) -> bool:
        return self.suit_max == 2

    @property
    def is_rainbow(self) -> bool:
        return self.suit_max == 1

    @property
    def is_wet(self) -> bool:
        """Many draws possible → bet bigger to charge them."""
        return (
            self.suit_max >= 3
            or self.connected_count >= 3
            or (self.suit_max == 2 and self.connected_count >= 2)
        )

    @property
    def is_dry(self) -> bool:
        """Few draws → small bet sufficient. The opposite of wet (and not paired)."""
        return not self.is_wet and not self.paired

    def __str__(self) -> str:
        tags = []
        if self.is_monotone:
            tags.append("monotone")
        elif self.is_two_tone:
            tags.append("two-tone")
        else:
            tags.append("rainbow")
        if self.paired:
            tags.append("paired")
        if self.is_wet:
            tags.append("wet")
        elif self.is_dry:
            tags.append("dry")
        return f"BoardTexture({', '.join(tags)})"


def analyze_texture(board: Iterable[Card]) -> BoardTexture:
    """Compute textural properties of a 3/4/5-card board."""
    board = list(board)
    if len(board) not in (3, 4, 5):
        raise ValueError(f"board must have 3/4/5 cards, got {len(board)}")

    # Suit distribution
    suit_counts = {s: 0 for s in Suit}
    for c in board:
        suit_counts[c.suit] += 1
    suit_max = max(suit_counts.values())

    # Pair detection
    rank_counts: dict[Rank, int] = {}
    for c in board:
        rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1
    paired = any(n >= 2 for n in rank_counts.values())

    # Connectedness: count distinct ranks that contribute to a 5-card straight
    # window. We use a simple metric: how many gaps within the closest 5-rank
    # window touching all board cards.
    rank_ints = sorted({int(r) for r in rank_counts.keys()})
    # Add ace-low if applicable
    if Rank.ACE in rank_counts:
        rank_ints.append(1)
        rank_ints.sort()

    # connected_count: 0-7 score based on straight-completion possibilities
    connected_count = _compute_connectedness(rank_ints)

    # High card count (Broadway = T J Q K A)
    high_card_count = sum(1 for c in board if c.rank >= Rank.TEN)

    return BoardTexture(
        suit_max=suit_max,
        paired=paired,
        connected_count=connected_count,
        high_card_count=high_card_count,
        n_cards=len(board),
    )


def _compute_connectedness(rank_ints: list[int]) -> int:
    """Connectedness score: how many cards within a 5-rank window vs spread.

    Returns 0-7. Higher = more connected = more straight possibilities.
    """
    if len(rank_ints) < 2:
        return 0
    # Score: number of cards within any 5-rank window
    best = 0
    for window_low in range(1, 11):
        window_high = window_low + 4
        in_window = sum(1 for r in rank_ints if window_low <= r <= window_high)
        best = max(best, in_window)
    # Map to 0-7: 3 cards in a window = highly connected
    if best >= 3:
        return 5 + (best - 3) * 2  # 5, 7
    if best == 2:
        # Adjacent or one-gap pair = 2 or 3
        gaps = [b - a for a, b in zip(rank_ints, rank_ints[1:])]
        min_gap = min(gaps) if gaps else 99
        if min_gap == 1:
            return 3
        if min_gap <= 3:
            return 2
        return 1
    return 0
