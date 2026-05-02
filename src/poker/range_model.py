"""Range modeling — weighted distributions over starting hands at COMBO level.

A Range stores a weight per specific 2-card combo (1326 possible combos),
not per 169-class hand. This matters when board texture interacts with suit:
- AKo on a monotone-spade board: combos with a spade in hand have backdoor
  flush draws; combos without a spade have nothing. They should narrow
  differently after a flop bet.

The class-level API is preserved for ergonomics (`Range.from_set({"AA"})`,
`range.has("AKo")`), but the internal storage is per-combo.

Combos per class (for reference):
  - Pair  (e.g. "AA"):  6 combos = C(4,2)
  - Suited (e.g. "AKs"): 4 combos
  - Offsuit (e.g. "AKo"): 12 combos
Total: 13 × 6 + 78 × 4 + 78 × 12 = 1326 combos. ✓
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .cards import Card, Rank, Suit, full_deck


# ─── Hand-class enumeration ───────────────────────────────────────────────────

_RANK_CHARS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]


def all_hand_classes() -> list[str]:
    """All 169 starting hand classes, ordered roughly by rank (high pairs first)."""
    classes: list[str] = []
    for r in reversed(_RANK_CHARS):
        classes.append(f"{r}{r}")
    for i, h in enumerate(reversed(_RANK_CHARS)):
        for l in reversed(_RANK_CHARS[:len(_RANK_CHARS) - i - 1]):
            classes.append(f"{h}{l}s")
    for i, h in enumerate(reversed(_RANK_CHARS)):
        for l in reversed(_RANK_CHARS[:len(_RANK_CHARS) - i - 1]):
            classes.append(f"{h}{l}o")
    return classes


def combos_per_class(hand: str) -> int:
    """Number of card combinations for a 169-class hand string."""
    if len(hand) == 2:
        return 6
    if hand.endswith("s"):
        return 4
    if hand.endswith("o"):
        return 12
    raise ValueError(f"invalid hand class: {hand!r}")


def expand_combos(hand: str) -> list[tuple[Card, Card]]:
    """Return all card-pair combos that match the given hand class.

    Cards within each combo are returned in a canonical order: high card first,
    then low card. For pairs, ordered by suit.
    """
    if hand.endswith("s") or hand.endswith("o"):
        high_char, low_char = hand[0], hand[1]
        suited = hand.endswith("s")
    elif len(hand) == 2 and hand[0] == hand[1]:
        high_char = low_char = hand[0]
        suited = False
    else:
        raise ValueError(f"invalid hand class: {hand!r}")

    high_rank = Rank.from_char(high_char)
    low_rank = Rank.from_char(low_char)

    combos: list[tuple[Card, Card]] = []
    if high_rank == low_rank:
        for s1, s2 in itertools.combinations(Suit, 2):
            combos.append((Card(high_rank, s1), Card(low_rank, s2)))
    elif suited:
        for s in Suit:
            combos.append((Card(high_rank, s), Card(low_rank, s)))
    else:
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


# Canonical combo key: frozenset of two Cards (order-independent)
ComboKey = frozenset


def _make_key(c1: Card, c2: Card) -> ComboKey:
    return frozenset((c1, c2))


# ─── Range class — combo-keyed internally ────────────────────────────────────

@dataclass
class Range:
    """Weighted distribution over the 1326 possible 2-card combos.

    Internal storage: dict[frozenset[Card], float]. Each key is a frozenset of
    the two hole cards (so order-independent), and the value is the weight
    (0-1) representing how often villain holds that specific combo.

    Public API still supports class-level operations (`from_set({"AA"})`,
    `has("AKo")`) — those expand to / aggregate over the relevant combos.
    """
    weights: dict[ComboKey, float] = field(default_factory=dict)

    @classmethod
    def from_set(cls, hands: Iterable[str], weight: float = 1.0) -> "Range":
        """Build a Range from a set of hand classes; every combo of every class
        gets the same weight."""
        w: dict[ComboKey, float] = {}
        for hand in hands:
            for c1, c2 in expand_combos(hand):
                w[_make_key(c1, c2)] = weight
        return cls(w)

    @classmethod
    def all_hands(cls) -> "Range":
        """Random range — every combo at full weight."""
        return cls.from_set(all_hand_classes(), weight=1.0)

    @classmethod
    def from_combos(cls, combos: dict[tuple[Card, Card], float]) -> "Range":
        """Build directly from a {combo: weight} mapping."""
        return cls({_make_key(c1, c2): w for (c1, c2), w in combos.items()})

    @property
    def total_combos(self) -> float:
        """Sum of weights across all combos in the range. Equals 1326 for full range."""
        return sum(self.weights.values())

    @property
    def num_combos(self) -> int:
        """Number of distinct combos with nonzero weight."""
        return sum(1 for w in self.weights.values() if w > 0)

    @property
    def num_hand_classes(self) -> int:
        """How many of the 169 classes have at least one combo with nonzero weight."""
        seen: set[str] = set()
        for combo_key, w in self.weights.items():
            if w <= 0:
                continue
            c1, c2 = tuple(combo_key)
            seen.add(hand_class_of(c1, c2))
        return len(seen)

    @property
    def width_pct(self) -> float:
        """Range width as a percentage of all combos (1326)."""
        return 100.0 * self.total_combos / 1326

    def has(self, hand_class: str) -> bool:
        """True if any combo of this class has nonzero weight."""
        for c1, c2 in expand_combos(hand_class):
            if self.weights.get(_make_key(c1, c2), 0) > 0:
                return True
        return False

    def get(self, hand_class: str) -> float:
        """Average weight across all combos of this class.

        Matches the old class-level `get()` semantics: a fully-weighted class
        returns 1.0, half-weighted returns 0.5, and a partially-narrowed class
        returns the average of its combos' weights.
        """
        combos = expand_combos(hand_class)
        if not combos:
            return 0.0
        total = sum(self.weights.get(_make_key(c1, c2), 0.0) for c1, c2 in combos)
        return total / len(combos)

    def get_combo(self, c1: Card, c2: Card) -> float:
        """Weight for a specific combo (0 if absent)."""
        return self.weights.get(_make_key(c1, c2), 0.0)

    def restrict_to(self, hand_classes: Iterable[str]) -> "Range":
        """Return new range keeping only combos whose class is in `hand_classes`."""
        keep_keys: set[ComboKey] = set()
        for h in hand_classes:
            for c1, c2 in expand_combos(h):
                keep_keys.add(_make_key(c1, c2))
        return Range({k: w for k, w in self.weights.items() if k in keep_keys and w > 0})

    def remove(self, hand_classes: Iterable[str]) -> "Range":
        """Return new range with all combos of the given classes removed."""
        drop_keys: set[ComboKey] = set()
        for h in hand_classes:
            for c1, c2 in expand_combos(h):
                drop_keys.add(_make_key(c1, c2))
        return Range({k: w for k, w in self.weights.items() if k not in drop_keys and w > 0})

    def reweight(self, hand_classes: Iterable[str], factor: float) -> "Range":
        """Return new range with weights for combos of given classes multiplied by factor."""
        targets: set[ComboKey] = set()
        for h in hand_classes:
            for c1, c2 in expand_combos(h):
                targets.add(_make_key(c1, c2))
        new_weights = dict(self.weights)
        for k in targets:
            if k in new_weights:
                new_weights[k] = max(0.0, min(1.0, new_weights[k] * factor))
        return Range(new_weights)

    def union(self, other: "Range") -> "Range":
        """Take the max weight per combo from each range."""
        new_weights = dict(self.weights)
        for k, w in other.weights.items():
            new_weights[k] = max(new_weights.get(k, 0), w)
        return Range(new_weights)

    def __str__(self) -> str:
        return (
            f"Range({self.num_combos} combos, {self.num_hand_classes} classes, "
            f"{self.total_combos:.0f} weighted-combos, {self.width_pct:.1f}% of hands)"
        )


# ─── Combo classification on a board ──────────────────────────────────────────

def classify_combo_on_board(hole: tuple[Card, Card], board: list[Card]) -> str:
    """Classify a 2-card combo on a 3/4/5-card board.

    Returns one of: 'made_strong' (two pair+ or overpair), 'made_pair'
    (single pair, not overpair), 'strong_draw' (flush draw or OESD),
    'weak_draw' (gutshot or overcards), 'air' (high card with no draw).
    """
    from .evaluator import evaluate
    from .outs import analyze_outs

    if len(board) not in (3, 4, 5):
        raise ValueError(f"board must have 3/4/5 cards, got {len(board)}")

    hs = evaluate(hole, board)
    if hs.category_index <= 7:    # Two Pair (7) or stronger
        return "made_strong"
    if hs.category_index == 8:    # Pair
        # Overpair check: pocket pair higher than top board card → monster
        if hole[0].rank == hole[1].rank:
            top_board = max(c.rank for c in board)
            if hole[0].rank > top_board:
                return "made_strong"
        return "made_pair"

    if len(board) == 5:
        return "air"

    try:
        draws = analyze_outs(list(hole), list(board))
    except ValueError:
        return "air"

    has_flush_draw = "Flush draw" in draws.labels
    has_oesd = "OESD" in draws.labels
    has_gutshot = "Gutshot" in draws.labels
    has_overcards = any("overcard" in lbl for lbl in draws.labels)

    if has_flush_draw or has_oesd:
        return "strong_draw"
    if has_gutshot or has_overcards:
        return "weak_draw"
    return "air"


def sample_combos_from_range(
    rng_range: "Range",
    excluded: set[Card],
) -> list[tuple[tuple[Card, Card], float]]:
    """Enumerate (combo, weight) pairs from a Range, excluding combos that
    use any of the given cards (e.g. hero's hole cards or board cards).
    """
    out: list[tuple[tuple[Card, Card], float]] = []
    for combo_key, weight in rng_range.weights.items():
        if weight <= 0:
            continue
        c1, c2 = tuple(combo_key)
        if c1 in excluded or c2 in excluded:
            continue
        # Canonicalize order: high rank first (suit doesn't matter for ordering)
        if c2.rank > c1.rank:
            c1, c2 = c2, c1
        out.append(((c1, c2), weight))
    return out
