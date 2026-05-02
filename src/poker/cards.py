"""Card primitives — Card, Rank, Suit, Deck.

Cards are written in standard poker notation: rank char + suit char.
Ranks: 2 3 4 5 6 7 8 9 T J Q K A
Suits: c (clubs) d (diamonds) h (hearts) s (spades)
Examples: "As" = ace of spades, "Td" = ten of diamonds, "2c" = two of clubs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Iterable


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def char(self) -> str:
        return _RANK_TO_CHAR[self]

    @classmethod
    def from_char(cls, c: str) -> "Rank":
        try:
            return _CHAR_TO_RANK[c.upper()]
        except KeyError as e:
            raise ValueError(f"invalid rank char: {c!r}") from e


class Suit(Enum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"

    @property
    def glyph(self) -> str:
        return _SUIT_TO_GLYPH[self]

    @classmethod
    def from_char(cls, c: str) -> "Suit":
        try:
            return Suit(c.lower())
        except ValueError as e:
            raise ValueError(f"invalid suit char: {c!r}") from e


_RANK_TO_CHAR = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
    Rank.ACE: "A",
}
_CHAR_TO_RANK = {v: k for k, v in _RANK_TO_CHAR.items()}

_SUIT_TO_GLYPH = {
    Suit.CLUBS: "♣",
    Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
}


@dataclass(frozen=True, order=True)
class Card:
    """A playing card. Immutable, hashable, sortable by rank then suit."""

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.char}{self.suit.value}"

    def pretty(self) -> str:
        """Unicode rendering: 'A♠'."""
        return f"{self.rank.char}{self.suit.glyph}"

    @classmethod
    def parse(cls, s: str) -> "Card":
        """Parse a 2-char string like 'As' or 'Td' into a Card."""
        if len(s) != 2:
            raise ValueError(f"card string must be 2 chars, got {s!r}")
        return cls(Rank.from_char(s[0]), Suit.from_char(s[1]))


def parse_cards(s: str) -> list[Card]:
    """Parse a whitespace- or comma-separated string of cards.

    Examples:
        parse_cards("As Kd")       -> [A♠, K♦]
        parse_cards("As, Kd, 2c")  -> [A♠, K♦, 2♣]
    """
    tokens = [t for t in s.replace(",", " ").split() if t]
    return [Card.parse(t) for t in tokens]


def full_deck() -> list[Card]:
    """Return a sorted list of all 52 cards (no shuffling)."""
    return [Card(r, s) for s in Suit for r in Rank]


class Deck:
    """A standard 52-card deck. Shuffles on init unless seed is given a None deck.

    Use deal(n) to take the next n cards. Use remove(cards) to take specific
    cards (useful when setting up a known board / hole cards).
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._cards: list[Card] = full_deck()
        self.shuffle()

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        if n < 0:
            raise ValueError(f"cannot deal {n} cards")
        if n > len(self._cards):
            raise ValueError(f"cannot deal {n} cards, only {len(self._cards)} left")
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def remove(self, cards: Iterable[Card]) -> None:
        """Remove specific cards from the deck. Raises if any aren't present."""
        cards = list(cards)
        missing = [c for c in cards if c not in self._cards]
        if missing:
            raise ValueError(f"cards not in deck: {missing}")
        for c in cards:
            self._cards.remove(c)

    @property
    def remaining(self) -> int:
        return len(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, card: object) -> bool:
        return card in self._cards
