"""Tests for poker.cards — the foundation. If this is wrong, everything is wrong."""

import random

import pytest

from poker.cards import Card, Deck, Rank, Suit, full_deck, parse_cards


class TestRankAndSuit:
    def test_rank_chars_round_trip(self):
        for r in Rank:
            assert Rank.from_char(r.char) is r

    def test_rank_from_char_case_insensitive(self):
        assert Rank.from_char("a") is Rank.ACE
        assert Rank.from_char("A") is Rank.ACE
        assert Rank.from_char("t") is Rank.TEN

    def test_rank_invalid_char_raises(self):
        with pytest.raises(ValueError):
            Rank.from_char("1")
        with pytest.raises(ValueError):
            Rank.from_char("Z")

    def test_rank_ordering(self):
        assert Rank.ACE > Rank.KING > Rank.QUEEN > Rank.TWO

    def test_suit_chars_round_trip(self):
        for s in Suit:
            assert Suit.from_char(s.value) is s

    def test_suit_invalid_char_raises(self):
        with pytest.raises(ValueError):
            Suit.from_char("x")


class TestCard:
    def test_parse_round_trip(self):
        for s in ["As", "Kd", "Th", "2c", "9s"]:
            assert str(Card.parse(s)) == s

    def test_pretty_uses_glyphs(self):
        assert Card.parse("As").pretty() == "A♠"
        assert Card.parse("Td").pretty() == "T♦"

    def test_parse_invalid_length(self):
        with pytest.raises(ValueError):
            Card.parse("Ace")
        with pytest.raises(ValueError):
            Card.parse("A")

    def test_card_equality_and_hashable(self):
        a1 = Card(Rank.ACE, Suit.SPADES)
        a2 = Card.parse("As")
        assert a1 == a2
        assert hash(a1) == hash(a2)
        assert {a1, a2} == {a1}  # dedupes in a set

    def test_card_immutable(self):
        c = Card.parse("As")
        with pytest.raises(Exception):
            c.rank = Rank.TWO  # type: ignore[misc]

    def test_card_sortable(self):
        cards = [Card.parse(s) for s in ["2c", "As", "Tc", "Kh"]]
        sorted_cards = sorted(cards)
        assert [c.rank for c in sorted_cards] == [
            Rank.TWO, Rank.TEN, Rank.KING, Rank.ACE
        ]


class TestParseCards:
    def test_parse_space_separated(self):
        cards = parse_cards("As Kd 2c")
        assert len(cards) == 3
        assert cards[0] == Card.parse("As")

    def test_parse_comma_separated(self):
        cards = parse_cards("As, Kd, 2c")
        assert len(cards) == 3

    def test_parse_empty(self):
        assert parse_cards("") == []
        assert parse_cards("   ") == []


class TestFullDeck:
    def test_has_52_unique_cards(self):
        deck = full_deck()
        assert len(deck) == 52
        assert len(set(deck)) == 52

    def test_has_all_combinations(self):
        deck = full_deck()
        assert all(Card(r, s) in deck for r in Rank for s in Suit)


class TestDeck:
    def test_new_deck_has_52(self):
        d = Deck()
        assert len(d) == 52
        assert d.remaining == 52

    def test_deal_reduces_count(self):
        d = Deck()
        d.deal(5)
        assert d.remaining == 47

    def test_deal_returns_unique_cards(self):
        d = Deck()
        cards = d.deal(52)
        assert len(set(cards)) == 52
        assert d.remaining == 0

    def test_deal_one(self):
        d = Deck()
        c = d.deal_one()
        assert isinstance(c, Card)
        assert d.remaining == 51

    def test_deal_too_many_raises(self):
        d = Deck()
        with pytest.raises(ValueError):
            d.deal(53)

    def test_deal_negative_raises(self):
        d = Deck()
        with pytest.raises(ValueError):
            d.deal(-1)

    def test_seeded_decks_match(self):
        d1 = Deck(rng=random.Random(42))
        d2 = Deck(rng=random.Random(42))
        assert d1.deal(10) == d2.deal(10)

    def test_unseeded_decks_differ(self):
        # Statistically: two random shuffles producing identical first 10 cards
        # is ~1 in 52!/42! ≈ 10^17. Not gonna happen.
        d1 = Deck()
        d2 = Deck()
        assert d1.deal(10) != d2.deal(10)

    def test_remove_specific_cards(self):
        d = Deck()
        d.remove(parse_cards("As Kd"))
        assert d.remaining == 50
        assert Card.parse("As") not in d
        assert Card.parse("Kd") not in d

    def test_remove_missing_raises(self):
        d = Deck()
        d.remove(parse_cards("As"))
        with pytest.raises(ValueError):
            d.remove(parse_cards("As"))  # already gone

    def test_contains(self):
        d = Deck()
        assert Card.parse("As") in d
        d.remove(parse_cards("As"))
        assert Card.parse("As") not in d
