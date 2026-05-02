"""Tests for board texture analysis."""

import pytest

from poker.board_texture import analyze_texture
from poker.cards import parse_cards


class TestSuitDistribution:
    def test_monotone_flop(self):
        t = analyze_texture(parse_cards("Qs Js 4s"))
        assert t.suit_max == 3
        assert t.is_monotone
        assert not t.is_two_tone
        assert not t.is_rainbow

    def test_two_tone_flop(self):
        t = analyze_texture(parse_cards("Qs Js 4d"))
        assert t.suit_max == 2
        assert t.is_two_tone

    def test_rainbow_flop(self):
        t = analyze_texture(parse_cards("Qs Jh 4d"))
        assert t.suit_max == 1
        assert t.is_rainbow


class TestPaired:
    def test_paired_flop(self):
        t = analyze_texture(parse_cards("Kh Kc 7d"))
        assert t.paired

    def test_unpaired_flop(self):
        t = analyze_texture(parse_cards("Kh Qc 7d"))
        assert not t.paired


class TestWetDry:
    def test_dry_rainbow_disconnected(self):
        # K-7-2 rainbow: no flush draw, no straight draw, low connectedness
        t = analyze_texture(parse_cards("Kh 7c 2d"))
        assert t.is_dry
        assert not t.is_wet

    def test_wet_monotone(self):
        t = analyze_texture(parse_cards("Qs Js 4s"))
        assert t.is_wet
        assert not t.is_dry

    def test_wet_connected(self):
        # 8-7-6 with two suits: tons of straights and a flush draw
        t = analyze_texture(parse_cards("8h 7h 6c"))
        assert t.is_wet


class TestHighCards:
    def test_broadway_count(self):
        t = analyze_texture(parse_cards("As Kh Qd"))
        assert t.high_card_count == 3
        t2 = analyze_texture(parse_cards("As 7h 2d"))
        assert t2.high_card_count == 1

    def test_low_board(self):
        t = analyze_texture(parse_cards("8h 7c 4d"))
        assert t.high_card_count == 0


class TestStrFormat:
    def test_str_includes_descriptors(self):
        t = analyze_texture(parse_cards("Qs Js 4s"))
        s = str(t)
        assert "monotone" in s


class TestValidation:
    def test_invalid_board_size(self):
        with pytest.raises(ValueError):
            analyze_texture(parse_cards("Qs Js"))   # 2 cards is invalid
