"""Tests for pot odds, required equity, and EV calculations."""

import pytest

from poker.pot_odds import (
    PotOddsAnalysis,
    analyze_call,
    ev,
    pot_odds_ratio,
    required_equity,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


class TestRequiredEquity:
    def test_classic_4_to_1(self):
        # Pot $80, call $20 → 4:1, need 20%
        assert approx(required_equity(80, 20), 0.20)

    def test_3_to_1(self):
        assert approx(required_equity(75, 25), 0.25)

    def test_pot_sized_bet_2_to_1(self):
        # Pot $100, call $50 (1/2 pot bet inflates pot to $150) → odds 2:1
        assert approx(required_equity(100, 50), 1 / 3)

    def test_overbet_makes_calls_expensive(self):
        # Pot $50, call $100 (2x pot overbet) → need 2/3 ≈ 66.7%
        assert approx(required_equity(50, 100), 2 / 3)

    def test_check_is_free(self):
        assert required_equity(100, 0) == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            required_equity(-10, 20)
        with pytest.raises(ValueError):
            required_equity(10, -20)


class TestPotOddsRatio:
    def test_clean_4_to_1(self):
        assert pot_odds_ratio(80, 20) == "4:1"

    def test_clean_3_to_1(self):
        assert pot_odds_ratio(75, 25) == "3:1"

    def test_clean_2_to_1(self):
        assert pot_odds_ratio(100, 50) == "2:1"

    def test_check_is_free(self):
        assert pot_odds_ratio(100, 0) == "free"


class TestEv:
    def test_break_even_call(self):
        # 20% equity into 80 pot for 20 call → exactly break-even
        assert approx(ev(0.20, 80, 20), 0.0)

    def test_strong_call(self):
        # 50% equity into 80 pot for 20 call → 0.5*80 - 0.5*20 = 40 - 10 = 30
        assert approx(ev(0.50, 80, 20), 30.0)

    def test_losing_call(self):
        # 10% equity into 80 pot for 20 call → 0.1*80 - 0.9*20 = 8 - 18 = -10
        assert approx(ev(0.10, 80, 20), -10.0)

    def test_zero_equity_loses_call(self):
        assert approx(ev(0.0, 100, 50), -50.0)

    def test_certain_win_gains_pot(self):
        assert approx(ev(1.0, 100, 50), 100.0)

    def test_invalid_equity_raises(self):
        with pytest.raises(ValueError):
            ev(1.5, 100, 50)
        with pytest.raises(ValueError):
            ev(-0.1, 100, 50)


class TestAnalyzeCall:
    def test_strong_call_positive_ev(self):
        # Need 20%, have 50% → big +EV
        a = analyze_call(pot=80, call=20, equity_pct=50)
        assert a.verdict == "+EV call"
        assert a.ev > 0
        assert a.edge_pp == pytest.approx(30.0)
        assert a.pot_odds_ratio == "4:1"
        assert approx(a.required_equity_pct, 20.0)

    def test_break_even(self):
        a = analyze_call(pot=80, call=20, equity_pct=20)
        assert a.verdict == "Break-even"
        assert abs(a.ev) < 0.01
        assert abs(a.edge_pp) <= 0.5

    def test_break_even_band_just_above(self):
        # Equity 0.4pp above required → still considered break-even
        a = analyze_call(pot=80, call=20, equity_pct=20.4)
        assert a.verdict == "Break-even"

    def test_break_even_band_just_below(self):
        a = analyze_call(pot=80, call=20, equity_pct=19.6)
        assert a.verdict == "Break-even"

    def test_losing_call_negative_ev(self):
        # Need 20%, only have 10% → -EV
        a = analyze_call(pot=80, call=20, equity_pct=10)
        assert a.verdict == "-EV call"
        assert a.ev < 0
        assert a.edge_pp == pytest.approx(-10.0)

    def test_overbet_requires_high_equity(self):
        # 2x pot overbet: need 66.7%
        a = analyze_call(pot=50, call=100, equity_pct=50)
        assert a.verdict == "-EV call"
        assert approx(a.required_equity_pct, 66.6667, tol=0.01)

    def test_invalid_equity_pct_raises(self):
        with pytest.raises(ValueError):
            analyze_call(pot=100, call=20, equity_pct=120)

    def test_str_renders_summary(self):
        a = analyze_call(pot=80, call=20, equity_pct=35)
        s = str(a)
        assert "4:1" in s
        assert "20.0%" in s   # required
        assert "35.0%" in s   # actual
        assert "+EV call" in s


class TestEducationalScenarios:
    """Scenarios that surface common training points."""

    def test_flush_draw_with_pot_odds(self):
        # Flush draw on the flop: ~35% to hit by river.
        # Villain bets 1/2 pot → we need 25%. 35 > 25 → +EV.
        a = analyze_call(pot=100, call=50, equity_pct=35)
        assert a.verdict == "+EV call"

    def test_gutshot_versus_pot_sized_bet(self):
        # Gutshot ~16% by river. Pot-sized bet → need 33%. -EV without implied.
        a = analyze_call(pot=100, call=100, equity_pct=16)
        assert a.verdict == "-EV call"

    def test_set_versus_overpair_huge_edge(self):
        # 220 set vs AA on flop ~91%. Any bet is +EV.
        a = analyze_call(pot=100, call=200, equity_pct=91)
        assert a.verdict == "+EV call"
        assert a.ev > 0
