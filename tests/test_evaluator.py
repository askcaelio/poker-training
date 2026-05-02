"""Tests for poker.evaluator — verifies category detection and head-to-head."""

import pytest

from poker.cards import parse_cards
from poker.evaluator import HandStrength, compare, evaluate, evaluate_5


def cat(s: str):
    """Helper: evaluate a 5-card hand from a string and return category."""
    return evaluate_5(parse_cards(s)).category


class TestCategoryDetection:
    def test_royal_flush(self):
        assert cat("As Ks Qs Js Ts") == "Royal Flush"

    def test_straight_flush(self):
        assert cat("9s 8s 7s 6s 5s") == "Straight Flush"

    def test_wheel_straight_flush(self):
        # A-2-3-4-5 of one suit
        assert cat("As 2s 3s 4s 5s") == "Straight Flush"

    def test_four_of_a_kind(self):
        assert cat("As Ah Ad Ac Kd") == "Four of a Kind"

    def test_full_house(self):
        assert cat("As Ah Ad Kc Kd") == "Full House"

    def test_flush(self):
        assert cat("As Ks 9s 4s 2s") == "Flush"

    def test_straight(self):
        assert cat("9s 8h 7d 6c 5s") == "Straight"

    def test_wheel_straight(self):
        assert cat("As 2h 3d 4c 5s") == "Straight"

    def test_broadway_straight(self):
        assert cat("As Kh Qd Jc Ts") == "Straight"

    def test_three_of_a_kind(self):
        assert cat("As Ah Ad Kc 2d") == "Three of a Kind"

    def test_two_pair(self):
        assert cat("As Ah Kd Kc 2d") == "Two Pair"

    def test_pair(self):
        assert cat("As Ah Kd Qc 2d") == "Pair"

    def test_high_card(self):
        assert cat("As Kh Qd Jc 9s") == "High Card"


class TestEvaluateHoleAndBoard:
    def test_hole_plus_flop_minimum(self):
        # 2 hole + 3 board = 5 cards (minimum)
        h = evaluate(parse_cards("As Ks"), parse_cards("Qs Js Ts"))
        assert h.category == "Royal Flush"

    def test_hole_plus_full_board(self):
        # 2 hole + 5 board = 7 cards (full Hold'em)
        h = evaluate(parse_cards("As Ks"), parse_cards("Qs Js Ts 2c 3d"))
        assert h.category == "Royal Flush"

    def test_picks_best_five_from_seven(self):
        # Hole pair + flush draw on board → should pick the flush
        h = evaluate(parse_cards("As 2s"), parse_cards("Ks Qs 9s 7c 7d"))
        assert h.category == "Flush"

    def test_picks_best_five_full_house_over_trips(self):
        # AAA on board + KK in hole → full house, not trips
        h = evaluate(parse_cards("Ks Kh"), parse_cards("As Ah Ad 2c 3d"))
        assert h.category == "Full House"

    def test_too_few_cards_raises(self):
        with pytest.raises(ValueError):
            evaluate(parse_cards("As Ks"), parse_cards("Qs"))

    def test_too_many_cards_raises(self):
        with pytest.raises(ValueError):
            evaluate(parse_cards("As Ks Qs"), parse_cards("Js Ts 9s 8s 7s"))


class TestComparison:
    def test_higher_category_beats_lower(self):
        flush = evaluate_5(parse_cards("As Ks 9s 4s 2s"))
        straight = evaluate_5(parse_cards("9s 8h 7d 6c 5s"))
        assert flush > straight
        assert straight < flush
        assert compare(flush, straight) == 1
        assert compare(straight, flush) == -1

    def test_higher_pair_beats_lower_pair(self):
        aces = evaluate_5(parse_cards("As Ah Kd Qc 2d"))
        kings = evaluate_5(parse_cards("Ks Kh Ad Qc 2d"))
        assert aces > kings

    def test_kicker_breaks_tie(self):
        # Both have pair of aces; AK kicker beats AQ kicker
        ak = evaluate_5(parse_cards("As Ah Kd 5c 2d"))
        aq = evaluate_5(parse_cards("As Ah Qd 5c 2d"))
        assert ak > aq

    def test_identical_hands_tie(self):
        # Same cards, different order → identical strength
        a = evaluate_5(parse_cards("As Ah Kd 5c 2d"))
        b = evaluate_5(parse_cards("Ah As 5c Kd 2d"))
        assert a == b
        assert compare(a, b) == 0

    def test_royal_beats_everything(self):
        royal = evaluate_5(parse_cards("As Ks Qs Js Ts"))
        quads = evaluate_5(parse_cards("As Ah Ad Ac Kd"))
        assert royal > quads


class TestKnownRanks:
    def test_royal_flush_is_rank_one(self):
        assert evaluate_5(parse_cards("As Ks Qs Js Ts")).rank == 1

    def test_worst_high_card_is_max_rank(self):
        # 7-5-4-3-2 of mixed suits is the worst possible 5-card hand
        worst = evaluate_5(parse_cards("7c 5d 4h 3s 2c"))
        assert worst.rank == 7462
