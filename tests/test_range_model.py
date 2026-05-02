"""Tests for range_model — combo counting, expansion, and the Range class."""

import pytest

from poker.cards import Card, Rank, Suit, parse_cards
from poker.range_model import (
    Range, all_hand_classes, combos_per_class, expand_combos,
    hand_class_of, sample_combos_from_range,
)


# ─── Combo counting ──────────────────────────────────────────────────────────

class TestCombos:
    def test_pair_has_six_combos(self):
        assert combos_per_class("AA") == 6

    def test_suited_has_four_combos(self):
        assert combos_per_class("AKs") == 4

    def test_offsuit_has_twelve_combos(self):
        assert combos_per_class("AKo") == 12

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            combos_per_class("AKx")

    def test_total_combos_equals_1326(self):
        # 13 pairs × 6 + 78 suited × 4 + 78 offsuit × 12 = 1326
        total = sum(combos_per_class(h) for h in all_hand_classes())
        assert total == 1326


class TestExpand:
    def test_expand_aces(self):
        combos = expand_combos("AA")
        assert len(combos) == 6
        assert all(c1.rank == Rank.ACE and c2.rank == Rank.ACE for c1, c2 in combos)
        # All distinct cards
        for c1, c2 in combos:
            assert c1 != c2

    def test_expand_aks(self):
        combos = expand_combos("AKs")
        assert len(combos) == 4
        # All same-suit combos
        for c1, c2 in combos:
            assert c1.suit == c2.suit
            assert {c1.rank, c2.rank} == {Rank.ACE, Rank.KING}

    def test_expand_ako(self):
        combos = expand_combos("AKo")
        assert len(combos) == 12
        # All offsuit
        for c1, c2 in combos:
            assert c1.suit != c2.suit


class TestHandClassOf:
    def test_pair(self):
        cards = parse_cards("As Ah")
        assert hand_class_of(cards[0], cards[1]) == "AA"

    def test_suited(self):
        cards = parse_cards("As Ks")
        assert hand_class_of(cards[0], cards[1]) == "AKs"

    def test_offsuit(self):
        cards = parse_cards("As Kh")
        assert hand_class_of(cards[0], cards[1]) == "AKo"


# ─── Range basics ─────────────────────────────────────────────────────────────

class TestRange:
    def test_from_set(self):
        r = Range.from_set({"AA", "KK"})
        assert r.has("AA")
        assert r.has("KK")
        assert not r.has("QQ")
        assert r.total_combos == 12   # 6 + 6

    def test_all_hands(self):
        r = Range.all_hands()
        assert r.total_combos == 1326
        assert r.num_hand_classes == 169
        assert r.width_pct == pytest.approx(100.0)

    def test_restrict_to(self):
        r = Range.from_set({"AA", "KK", "QQ", "AKs"})
        r2 = r.restrict_to({"AA", "AKs"})
        assert r2.has("AA")
        assert r2.has("AKs")
        assert not r2.has("KK")
        assert r2.total_combos == 6 + 4

    def test_remove(self):
        r = Range.from_set({"AA", "KK", "QQ"})
        r2 = r.remove({"KK"})
        assert r2.has("AA")
        assert r2.has("QQ")
        assert not r2.has("KK")

    def test_reweight(self):
        r = Range.from_set({"AA", "KK"})
        r2 = r.reweight({"KK"}, 0.5)
        assert r2.get("AA") == 1.0
        assert r2.get("KK") == 0.5

    def test_width_pct(self):
        # NIT_RANGE has ~6% of hands
        from poker.preflop_ranges import NIT_RANGE
        r = Range.from_set(NIT_RANGE)
        assert 4 < r.width_pct < 10

    def test_str_summary(self):
        r = Range.from_set({"AA", "KK", "QQ"})
        s = str(r)
        assert "3 classes" in s
        assert "18 combos" in s


# ─── Combo sampling with exclusions ──────────────────────────────────────────

class TestSampling:
    def test_excludes_blocker(self):
        # Range = {"AA"}, hero holds As → only 3 combos remain (Ac/Ad, Ac/Ah, Ad/Ah)
        r = Range.from_set({"AA"})
        excluded = {Card.parse("As")}
        combos = sample_combos_from_range(r, excluded)
        assert len(combos) == 3
        for (c1, c2), w in combos:
            assert c1 != Card.parse("As")
            assert c2 != Card.parse("As")
            assert w == 1.0

    def test_no_exclusions(self):
        r = Range.from_set({"AA", "KK"})
        combos = sample_combos_from_range(r, set())
        assert len(combos) == 12   # 6 + 6
