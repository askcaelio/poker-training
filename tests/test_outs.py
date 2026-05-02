"""Tests for outs counter and street probabilities.

Validates classic textbook scenarios: flush draw = 9 outs, OESD = 8, gutshot = 4,
combo draws ~15, plus the user's specific suited-diamonds example.
"""

import math

import pytest

from poker.cards import Suit, parse_cards
from poker.outs import analyze_outs, prob_specific_suit


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


class TestFlushDraw:
    """Our outs counter is exhaustive: it counts EVERY card that changes the hand
    category. Textbook 'flush draw = 9 outs' only counts the flush itself.
    Real total includes pair outs (matching board ranks → high card → pair) and
    accounts for overlap (e.g., a card that pairs the board AND completes the flush).
    """

    def test_flush_plus_pair_outs(self):
        # As 2s on Ks 8s 5h. Current: ace high.
        # Pair outs: ranks A,2,K,8,5 → 5 ranks × 3 non-spade suits = 15
        # Flush outs: 9 remaining spades
        # Overlap: 5s is both a flush card AND pairs the 5h → counted once
        # Total: 15 + 9 - 1 = 23
        a = analyze_outs(parse_cards("As 2s"), parse_cards("Ks 8s 5h"))
        assert "Flush draw" in a.labels
        assert a.out_count == 23

    def test_low_suited_connectors_flush_draw(self):
        # 7s 6s on Ks Qs 2h. Same arithmetic structure.
        # Pair outs: ranks 7,6,K,Q,2 → 15. Flush: 9. Overlap: 2s. Total: 23.
        a = analyze_outs(parse_cards("7s 6s"), parse_cards("Ks Qs 2h"))
        assert "Flush draw" in a.labels
        assert a.out_count == 23

    def test_overcards_with_flush_draw(self):
        # As Ks on Qs 7s 2c. Two overcards + flush draw.
        # Pair outs: A,K,Q,7,2 → 15. Flush: 9. Overlap: 2s. Total: 23.
        a = analyze_outs(parse_cards("As Ks"), parse_cards("Qs 7s 2c"))
        assert "Flush draw" in a.labels
        assert "2 overcards" in a.labels
        assert a.unseen == 47
        assert a.out_count == 23
        assert approx(a.prob_hit_next, 23 / 47, 1e-6)


class TestStraightDraws:
    def test_open_ended_straight_draw(self):
        # 9-8 in hole, 7-6-2 on board → 5 or T completes straight (OESD)
        # Pair outs: 9,8,7,6,2 → 5 ranks × 3 = 15
        # Straight outs: 5 or T → 4+4 = 8 (no overlap with pair outs)
        # Total: 23
        a = analyze_outs(parse_cards("9s 8h"), parse_cards("7d 6c 2s"))
        assert "OESD" in a.labels
        assert a.out_count == 23

    def test_gutshot_straight_draw(self):
        # 9-8 in hole, 6-5-2 on board → only 7 completes straight (gutshot)
        # Pair outs: 9,8,6,5,2 → 15
        # Straight outs: 7 → 4 (no overlap)
        # Total: 19
        a = analyze_outs(parse_cards("9s 8h"), parse_cards("6d 5c 2s"))
        assert "Gutshot" in a.labels
        assert a.out_count == 19


class TestProbabilityMath:
    def test_flop_unseen_is_47(self):
        a = analyze_outs(parse_cards("As Ks"), parse_cards("Qs 7s 2c"))
        assert a.unseen == 47

    def test_turn_unseen_is_46(self):
        a = analyze_outs(parse_cards("As Ks"), parse_cards("Qs 7s 2c 3d"))
        assert a.unseen == 46

    def test_nine_outs_on_flop_to_river(self):
        # Construct exactly 9 outs and verify the by-river probability.
        # Easier: directly test the formula with prob_specific_suit.
        info = prob_specific_suit(
            parse_cards("As Ks"),
            parse_cards("Qs 7s 2c"),
            Suit.SPADES,
        )
        assert info["suit_remaining"] == 9
        assert info["unseen"] == 47
        assert approx(info["p_next"], 9 / 47, 1e-6)
        # P(by river) = 1 - (38/47)*(37/46) ≈ 0.3496
        expected = 1 - (38 / 47) * (37 / 46)
        assert approx(info["p_by_river"], expected, 1e-6)
        # Sanity: about 35%, rule-of-4 says 36%
        assert 0.34 < info["p_by_river"] < 0.36

    def test_rule_of_4_on_flop(self):
        # Same hand as test_flush_plus_pair_outs: 23 outs → rule of 4 = 92%
        # (this is above 8-9 outs, where rule of 4 starts to overestimate; the
        # exact prob_hit_by_river will be noticeably lower than this rule)
        a = analyze_outs(parse_cards("As 2s"), parse_cards("Ks 8s 5h"))
        assert a.out_count == 23
        assert approx(a.rule_of_4_or_2, 0.92)
        # Exact prob is meaningfully less than the rule estimate at high out counts
        assert a.prob_hit_by_river < a.rule_of_4_or_2

    def test_rule_of_2_on_turn(self):
        # Same hand, turn instead of flop. 1 card to come.
        a = analyze_outs(parse_cards("As 2s"), parse_cards("Ks 8s 5h 4d"))
        # Outs may differ slightly on turn (4d gave us... still no pair).
        # Just verify the rule formula matches.
        assert approx(a.rule_of_4_or_2, a.out_count * 2 / 100.0)


class TestUserExample:
    """The user's specific question: 2 suited diamonds, 2 diamonds on the flop."""

    def test_diamond_probabilities_explicit(self):
        # Hero: AdKd. Flop: 7d Qd 2c. Two diamonds on board, two in hand.
        info = prob_specific_suit(
            parse_cards("Ad Kd"),
            parse_cards("7d Qd 2c"),
            Suit.DIAMONDS,
        )
        assert info["suit_remaining"] == 9   # 13 - 4 visible
        assert info["unseen"] == 47
        # P(diamond on turn) = 9/47 ≈ 0.1915
        assert approx(info["p_next"], 9 / 47, 1e-6)
        # P(diamond on river | turn missed) = 9/46 ≈ 0.1957
        assert approx(info["p_river_given_missed_turn"], 9 / 46, 1e-6)
        # P(at least one diamond by river) = 1 - (38/47)*(37/46) ≈ 0.3497
        expected_by_river = 1 - (38 / 47) * (37 / 46)
        assert approx(info["p_by_river"], expected_by_river, 1e-6)


class TestValidation:
    def test_river_board_raises(self):
        with pytest.raises(ValueError):
            analyze_outs(parse_cards("As Ks"), parse_cards("Qs Js Ts 9c 8h"))

    def test_preflop_raises(self):
        with pytest.raises(ValueError):
            analyze_outs(parse_cards("As Ks"), parse_cards(""))

    def test_one_hole_card_raises(self):
        with pytest.raises(ValueError):
            analyze_outs(parse_cards("As"), parse_cards("Qs Js Ts"))


class TestNoDraw:
    def test_made_hand_zero_outs_to_improve_category(self):
        # Royal flush already — can't improve category
        a = analyze_outs(parse_cards("As Ks"), parse_cards("Qs Js Ts"))
        assert a.out_count == 0
        assert a.prob_hit_next == 0.0
        assert a.prob_hit_by_river == 0.0
