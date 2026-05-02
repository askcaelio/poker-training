"""Tests for equity calculator. Validates against well-known equity values.

These are the classic preflop matchups every poker player has memorized. Our
Monte Carlo should reproduce them within ~1.5% at 30k iterations.
"""

import random

import pytest

from poker.cards import parse_cards
from poker.equity import Equity, equity_vs_hands, equity_vs_random


SEED = 12345  # consistent runs in CI
ITERS = 30_000
TOL = 1.5  # percentage points


def rng():
    return random.Random(SEED)


class TestVsRandom:
    def test_aa_vs_random_about_85pct(self):
        eq = equity_vs_random(parse_cards("As Ah"), iterations=ITERS, rng=rng())
        assert abs(eq.equity_pct - 85.2) < TOL

    def test_72o_vs_random_about_35pct(self):
        # The infamous worst hand
        eq = equity_vs_random(parse_cards("7c 2d"), iterations=ITERS, rng=rng())
        assert abs(eq.equity_pct - 34.6) < TOL

    def test_more_opponents_lowers_equity(self):
        aa_heads_up = equity_vs_random(
            parse_cards("As Ah"), num_opponents=1, iterations=ITERS, rng=rng()
        )
        aa_5way = equity_vs_random(
            parse_cards("As Ah"), num_opponents=5, iterations=ITERS, rng=rng()
        )
        assert aa_heads_up.equity_pct > aa_5way.equity_pct
        # 6-handed AA is roughly ~49% — much lower than heads-up
        assert aa_5way.equity_pct < 60


class TestVsKnownHands:
    def test_aa_vs_kk_about_81pct(self):
        eq = equity_vs_hands(
            parse_cards("As Ah"),
            [parse_cards("Ks Kh")],
            iterations=ITERS, rng=rng(),
        )
        assert abs(eq.equity_pct - 81.7) < TOL

    def test_aks_vs_22_coin_flip(self):
        # Classic "race": almost exactly 50/50. AKs ~50.0%, 22 ~49.0%, ties ~1%.
        eq = equity_vs_hands(
            parse_cards("As Ks"),
            [parse_cards("2c 2d")],
            iterations=ITERS, rng=rng(),
        )
        assert abs(eq.equity_pct - 50.0) < TOL

    def test_ako_vs_qq_underdog(self):
        eq = equity_vs_hands(
            parse_cards("As Kh"),
            [parse_cards("Qs Qh")],
            iterations=ITERS, rng=rng(),
        )
        assert abs(eq.equity_pct - 43.3) < TOL

    def test_dominated_kings(self):
        # KK vs AA: KK is the dominated underdog
        eq = equity_vs_hands(
            parse_cards("Ks Kh"),
            [parse_cards("As Ah")],
            iterations=ITERS, rng=rng(),
        )
        assert abs(eq.equity_pct - 18.3) < TOL


class TestPostflop:
    def test_set_vs_overpair_on_dry_flop(self):
        # 222 has set on 2-7-9 rainbow vs AA: set is huge favorite (~91%)
        eq = equity_vs_hands(
            parse_cards("2c 2d"),
            [parse_cards("As Ah")],
            board=parse_cards("2h 7c 9d"),
            iterations=ITERS, rng=rng(),
        )
        assert eq.equity_pct > 88

    def test_flush_draw_vs_top_pair(self):
        # AsKs (nut flush draw + overcards) vs AdKd top pair on Kh 7s 2s
        # Hero ~45-50% with flush draw + 6 overcard outs
        eq = equity_vs_hands(
            parse_cards("As 5s"),
            [parse_cards("Kd Qd")],
            board=parse_cards("Kh 7s 2s"),
            iterations=ITERS, rng=rng(),
        )
        # Flush draw alone is ~36%; with backdoor straight outs, ~40-45%
        assert 30 < eq.equity_pct < 50

    def test_river_is_deterministic(self):
        # On the river, equity is binary (you won or you lost). Single eval.
        eq = equity_vs_hands(
            parse_cards("As Ah"),
            [parse_cards("Ks Kh")],
            board=parse_cards("Ad 7c 2d 9h 3s"),
            iterations=10_000,  # ignored when board is complete
            rng=rng(),
        )
        assert eq.equity_pct == 100.0
        assert eq.iterations == 1


class TestValidation:
    def test_duplicate_card_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            equity_vs_random(
                parse_cards("As Ah"),
                board=parse_cards("As 7c 2d"),
                iterations=100, rng=rng(),
            )

    def test_wrong_hole_count_raises(self):
        with pytest.raises(ValueError):
            equity_vs_random(parse_cards("As"), iterations=100, rng=rng())

    def test_invalid_board_size_raises(self):
        with pytest.raises(ValueError):
            equity_vs_random(
                parse_cards("As Ah"),
                board=parse_cards("Ks Qs"),  # 2 cards = invalid
                iterations=100, rng=rng(),
            )

    def test_zero_opponents_raises(self):
        with pytest.raises(ValueError):
            equity_vs_random(
                parse_cards("As Ah"),
                num_opponents=0,
                iterations=100, rng=rng(),
            )


class TestDeterminism:
    def test_seeded_runs_match(self):
        a = equity_vs_random(parse_cards("As Ah"), iterations=1000, rng=random.Random(99))
        b = equity_vs_random(parse_cards("As Ah"), iterations=1000, rng=random.Random(99))
        assert a == b

    def test_percentages_sum_to_100(self):
        eq = equity_vs_random(parse_cards("As Ah"), iterations=5000, rng=rng())
        total = eq.win_pct + eq.tie_pct + eq.lose_pct
        assert abs(total - 100.0) < 0.001
