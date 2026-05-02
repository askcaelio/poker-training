"""Range modeling — weighted distributions over starting hands.

A Range is a weighted dict over the 169 hand classes (e.g. "AKs", "AKo", "QQ").
Weight 1.0 = villain definitely plays this; weight 0.5 = plays it half the time.

Combos per class:
  - Pair  (e.g. "AA"):  6 combos = C(4,2)
  - Suited (e.g. "AKs"): 4 combos
  - Offsuit (e.g. "AKo"): 12 combos
Total: 13 × 6 + 78 × 4 + 78 × 12 = 1326 combos. ✓

Why weighted (vs binary in/out)? Real ranges aren't crisp. A TAG 3-bets AKs
100% but only AJs 30% of the time. Weights capture this.

Range narrowing: each action conditions the range. If a TAG calls an open,
remove their 3-bet hands (they would've raised) and their fold hands (they
would've folded). What's left is their CALLING range — a much narrower set.
This is what bots use for postflop equity calc.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable

from .cards import Card, Rank, Suit, full_deck


# ─── Hand-class enumeration ───────────────────────────────────────────────────

_RANK_CHARS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]


def all_hand_classes() -> list[str]:
    """All 169 starting hand classes, ordered roughly by rank (high pairs first)."""
    classes: list[str] = []
    # Pairs first (strongest first)
    for r in reversed(_RANK_CHARS):
        classes.append(f"{r}{r}")
    # Suited (high card descending, then low card descending)
    for i, h in enumerate(reversed(_RANK_CHARS)):
        for l in reversed(_RANK_CHARS[:len(_RANK_CHARS) - i - 1]):
            classes.append(f"{h}{l}s")
    # Offsuit (same ordering)
    for i, h in enumerate(reversed(_RANK_CHARS)):
        for l in reversed(_RANK_CHARS[:len(_RANK_CHARS) - i - 1]):
            classes.append(f"{h}{l}o")
    return classes


def combos_per_class(hand: str) -> int:
    """Number of card combinations for a 169-class hand string."""
    if len(hand) == 2:
        return 6   # pair (e.g. "AA")
    if hand.endswith("s"):
        return 4   # suited
    if hand.endswith("o"):
        return 12  # offsuit
    raise ValueError(f"invalid hand class: {hand!r}")


def expand_combos(hand: str) -> list[tuple[Card, Card]]:
    """Return all card-pair combos that match the given hand class.

    Excludes nothing — caller must filter by 'known cards' if needed.
    """
    if hand.endswith("s") or hand.endswith("o"):
        high_char, low_char = hand[0], hand[1]
        suited = hand.endswith("s")
    elif len(hand) == 2 and hand[0] == hand[1]:
        high_char = low_char = hand[0]
        suited = False  # not applicable for pairs
    else:
        raise ValueError(f"invalid hand class: {hand!r}")

    high_rank = Rank.from_char(high_char)
    low_rank = Rank.from_char(low_char)

    combos: list[tuple[Card, Card]] = []
    if high_rank == low_rank:
        # pair
        for s1, s2 in itertools.combinations(Suit, 2):
            combos.append((Card(high_rank, s1), Card(low_rank, s2)))
    elif suited:
        for s in Suit:
            combos.append((Card(high_rank, s), Card(low_rank, s)))
    else:
        # offsuit
        for s1 in Suit:
            for s2 in Suit:
                if s1 != s2:
                    combos.append((Card(high_rank, s1), Card(low_rank, s2)))
    return combos


def hand_class_of(c1: Card, c2: Card) -> str:
    """Convert two cards to their 169-class string."""
    if c1.rank == c2.rank:
        return f"{c1.rank.char}{c2.rank.char}"
    high, low = (c1, c2) if c1.rank > c2.rank else (c2, c1)
    suited = c1.suit == c2.suit
    return f"{high.rank.char}{low.rank.char}{'s' if suited else 'o'}"


# ─── Range class ──────────────────────────────────────────────────────────────

@dataclass
class Range:
    """Weighted distribution over the 169 hand classes.

    weights[hand_class] = probability in [0, 1] that villain has this hand
    (within their plausible holdings — not absolute probability).
    """
    weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_set(cls, hands: Iterable[str], weight: float = 1.0) -> "Range":
        """Build a Range from a set of hand classes, all at the same weight."""
        return cls({h: weight for h in hands})

    @classmethod
    def all_hands(cls) -> "Range":
        """Random range — every hand at full weight."""
        return cls({h: 1.0 for h in all_hand_classes()})

    @property
    def total_combos(self) -> float:
        """Sum of weight × combos across all hand classes."""
        return sum(w * combos_per_class(h) for h, w in self.weights.items())

    @property
    def num_hand_classes(self) -> int:
        """How many of the 169 classes have nonzero weight."""
        return sum(1 for w in self.weights.values() if w > 0)

    @property
    def width_pct(self) -> float:
        """Range width as a percentage of all hands (by combos)."""
        return 100.0 * self.total_combos / 1326

    def has(self, hand: str) -> bool:
        return self.weights.get(hand, 0) > 0

    def get(self, hand: str) -> float:
        return self.weights.get(hand, 0.0)

    def restrict_to(self, hands: Iterable[str]) -> "Range":
        """Return new range keeping only the given hand classes."""
        keep = set(hands)
        return Range({h: w for h, w in self.weights.items() if h in keep and w > 0})

    def remove(self, hands: Iterable[str]) -> "Range":
        """Return new range with the given hand classes removed."""
        drop = set(hands)
        return Range({h: w for h, w in self.weights.items() if h not in drop and w > 0})

    def reweight(self, hands: Iterable[str], factor: float) -> "Range":
        """Return new range with weights for given hands multiplied by factor."""
        targets = set(hands)
        new_weights = dict(self.weights)
        for h in targets:
            if h in new_weights:
                new_weights[h] = max(0.0, min(1.0, new_weights[h] * factor))
        return Range(new_weights)

    def union(self, other: "Range") -> "Range":
        """Take the max weight per hand from each range."""
        new_weights = dict(self.weights)
        for h, w in other.weights.items():
            new_weights[h] = max(new_weights.get(h, 0), w)
        return Range(new_weights)

    def __str__(self) -> str:
        return f"Range({self.num_hand_classes} classes, {self.total_combos:.0f} combos, {self.width_pct:.1f}% of hands)"


# ─── Combo expansion that excludes known cards ────────────────────────────────

def sample_combos_from_range(
    rng_range: "Range",
    excluded: set[Card],
) -> list[tuple[tuple[Card, Card], float]]:
    """Enumerate (combo, weight) pairs from a Range, excluding combos that
    use any of the given cards (e.g. hero's hole cards or board cards).

    Returns the full combo list with weights — caller can sample by weight
    for Monte Carlo.
    """
    out: list[tuple[tuple[Card, Card], float]] = []
    for hand, weight in rng_range.weights.items():
        if weight <= 0:
            continue
        for c1, c2 in expand_combos(hand):
            if c1 in excluded or c2 in excluded:
                continue
            out.append(((c1, c2), weight))
    return out
