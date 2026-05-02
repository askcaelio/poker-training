"""Tests for the dashboard assembler. Verifies street detection, conditional
fields (no draws preflop, no pot odds without bet), and that all components
plug together correctly.
"""

import random

import pytest

from poker.cards import Suit, parse_cards
from poker.dashboard import Dashboard, build, render


def rng():
    return random.Random(42)


class TestStreets:
    def test_preflop(self):
        d = build(parse_cards("As Ah"), iterations=2000, rng=rng())
        assert d.street == "Preflop"
        assert d.hand_strength is None    # only 2 cards
        assert d.draw_analysis is None    # not on flop/turn

    def test_flop(self):
        d = build(parse_cards("As Ah"), parse_cards("Kh 7c 2d"),
                  iterations=2000, rng=rng())
        assert d.street == "Flop"
        assert d.hand_strength is not None
        assert d.draw_analysis is not None

    def test_turn(self):
        d = build(parse_cards("As Ah"), parse_cards("Kh 7c 2d 3s"),
                  iterations=2000, rng=rng())
        assert d.street == "Turn"
        assert d.draw_analysis is not None

    def test_river(self):
        d = build(parse_cards("As Ah"), parse_cards("Kh 7c 2d 3s 9h"),
                  iterations=2000, rng=rng())
        assert d.street == "River"
        assert d.hand_strength is not None
        assert d.draw_analysis is None    # no more cards


class TestPotOdds:
    def test_no_pot_no_pot_odds(self):
        d = build(parse_cards("As Ah"), parse_cards("Kh 7c 2d"),
                  iterations=2000, rng=rng())
        assert d.pot_odds is None

    def test_with_pot_and_call(self):
        d = build(parse_cards("As Ah"), parse_cards("Kh 7c 2d"),
                  pot=80, call=20, iterations=2000, rng=rng())
        assert d.pot_odds is not None
        assert d.pot_odds.pot_odds_ratio == "4:1"

    def test_pot_without_call_raises(self):
        with pytest.raises(ValueError):
            build(parse_cards("As Ah"), parse_cards("Kh 7c 2d"),
                  pot=80, iterations=100, rng=rng())

    def test_call_without_pot_raises(self):
        with pytest.raises(ValueError):
            build(parse_cards("As Ah"), parse_cards("Kh 7c 2d"),
                  call=20, iterations=100, rng=rng())


class TestSuitDraws:
    def test_no_flush_draw_no_suit_entry(self):
        d = build(parse_cards("As Ah"), parse_cards("Kh 7c 2d"),
                  iterations=2000, rng=rng())
        assert len(d.suit_draws) == 0

    def test_flush_draw_surfaces_suit(self):
        # Hero AdKd, board has 2 diamonds → 4 diamonds visible
        d = build(parse_cards("Ad Kd"), parse_cards("7d Qd 2c"),
                  iterations=2000, rng=rng())
        assert len(d.suit_draws) == 1
        sd = d.suit_draws[0]
        assert sd.suit == Suit.DIAMONDS
        assert sd.visible == 4
        assert sd.remaining == 9
        # P(diamond on turn) = 9/47 ≈ 0.1915
        assert abs(sd.p_next - 9 / 47) < 1e-6


class TestEquityIntegrated:
    def test_aa_preflop_high_equity(self):
        d = build(parse_cards("As Ah"), iterations=5000, rng=rng())
        assert d.equity.equity_pct > 80


class TestRender:
    def test_render_returns_string_with_key_sections(self):
        d = build(parse_cards("Ad Kd"), parse_cards("7d Qd 2c"),
                  pot=80, call=20, iterations=2000, rng=rng())
        out = render(d)
        assert "Flop" in out
        assert "Hero:" in out
        assert "Board:" in out
        assert "Made hand" in out
        assert "Outs:" in out
        assert "Flush draw" in out
        assert "Equity:" in out
        assert "Decision:" in out
        assert "EV:" in out

    def test_render_preflop_no_decision(self):
        d = build(parse_cards("As Ah"), iterations=2000, rng=rng())
        out = render(d)
        assert "Preflop" in out
        assert "Decision:" not in out
        assert "Outs:" not in out
