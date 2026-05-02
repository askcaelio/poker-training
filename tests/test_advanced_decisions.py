"""Tests for SPR awareness, implied odds, and polarized river logic."""

import random

import pytest

from poker.cards import Deck, parse_cards
from poker.agents import BotAgent, BotConfig, make_tag
from poker.game import (
    Action, ActionType, OtherPlayerView, Player, Position, Street,
    HandView, apply_action, start_hand,
)


def make_view(
    your_hole: str,
    board: str,
    pot: float,
    to_call: float,
    your_stack: float,
    opp_stack: float,
    street: Street,
) -> HandView:
    """Build a synthetic HandView for unit testing decision logic."""
    return HandView(
        your_id="hero",
        your_hole=tuple(parse_cards(your_hole)),
        your_stack=your_stack,
        your_position=Position.BTN,
        your_bet_this_round=0.0,
        board=tuple(parse_cards(board)) if board else (),
        street=street,
        pot=pot,
        current_bet=to_call,
        to_call=to_call,
        min_raise=max(2.0, to_call * 2),
        others=(OtherPlayerView(
            id="opp", position=Position.BB, stack=opp_stack,
            bet_this_round=to_call, total_invested=to_call,
            folded=False, all_in=False,
        ),),
        action_history=(),
    )


# ─── SPR ──────────────────────────────────────────────────────────────────────

class TestSPR:
    def test_low_spr_committed(self):
        # SPR = 5/100 = 0.05 → very committed
        view = make_view("As Ks", "Qh 7d 2c", pot=100, to_call=0,
                         your_stack=5, opp_stack=200, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        spr = bot._spr(view)
        assert spr < 2.0

    def test_standard_spr(self):
        view = make_view("As Ks", "Qh 7d 2c", pot=20, to_call=0,
                         your_stack=80, opp_stack=80, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        spr = bot._spr(view)
        # SPR = min(80, 80) / 20 = 4
        assert 2.0 < spr < 6.0

    def test_deep_spr(self):
        view = make_view("As Ks", "Qh 7d 2c", pot=10, to_call=0,
                         your_stack=300, opp_stack=300, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        spr = bot._spr(view)
        # SPR = min(300,300) / 10 = 30
        assert spr > 6.0


# ─── Implied odds ─────────────────────────────────────────────────────────────

class TestImpliedOdds:
    def test_drawing_hand_gets_implied_discount(self):
        # Drawing hand (eq ~0.30) facing a sub-optimal price; deep stacks
        # should give implied-odds discount, allowing a call that pure pot
        # odds would reject.
        view = make_view("9s 8s", "7s 6h 2c", pot=100, to_call=40,
                         your_stack=200, opp_stack=200, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        # required_eq pure = 40/(100+40) = 0.286
        # eq for OESD + flush draw is ~0.50+ — well above. Test with marginal eq.
        # Use a synthetic equity value to isolate the implied-odds math.
        adjusted = bot._apply_implied_odds(view, required_eq=0.30, eq=0.32)
        # adjusted should be slightly LOWER than 0.30
        assert adjusted < 0.30

    def test_no_implied_for_made_hands(self):
        # Made hand (eq > 0.6) — no implied-odds discount
        view = make_view("Ks Kd", "Kh 7d 2c", pot=100, to_call=40,
                         your_stack=200, opp_stack=200, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        adjusted = bot._apply_implied_odds(view, required_eq=0.30, eq=0.85)
        # No discount — already strong, no draw
        assert adjusted == 0.30

    def test_no_implied_at_low_spr(self):
        # SPR < 2 → committed, no implied odds
        view = make_view("9s 8s", "7s 6h 2c", pot=100, to_call=40,
                         your_stack=10, opp_stack=10, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        adjusted = bot._apply_implied_odds(view, required_eq=0.30, eq=0.32)
        assert adjusted == 0.30   # no discount when committed

    def test_no_implied_on_river(self):
        view = make_view("9s 8s", "7s 6h 2c 3d Kc", pot=100, to_call=40,
                         your_stack=200, opp_stack=200, street=Street.RIVER)
        bot = make_tag(rng=random.Random(0))
        adjusted = bot._apply_implied_odds(view, required_eq=0.30, eq=0.32)
        assert adjusted == 0.30


# ─── Polarized river ──────────────────────────────────────────────────────────

class TestPolarizedRiver:
    def test_strong_value_bets(self):
        # Hero has the nuts (top set) → value bet
        view = make_view("As Ad", "Ah 7d 2c 3s Kc", pot=50, to_call=0,
                         your_stack=200, opp_stack=200, street=Street.RIVER)
        bot = make_tag(rng=random.Random(0))
        action = bot._decide_river(view, eq=0.95)
        assert action.type == ActionType.BET

    def test_middle_strength_checks(self):
        # Bluff catcher — checks for showdown value
        view = make_view("As Ks", "Qh 7d 2c 3s Kc", pot=50, to_call=0,
                         your_stack=200, opp_stack=200, street=Street.RIVER)
        bot = make_tag(rng=random.Random(0))
        action = bot._decide_river(view, eq=0.50)
        assert action.type == ActionType.CHECK

    def test_pure_air_does_not_value_bet(self):
        # Hero has nothing → at minimum, not a value bet
        view = make_view("3c 2d", "Qh 7d 2c 3s Kc", pot=50, to_call=0,
                         your_stack=200, opp_stack=200, street=Street.RIVER)
        bot = make_tag(rng=random.Random(0))
        action = bot._decide_river(view, eq=0.10)
        # Without opponent_table set, bot can't compute fold equity; checks.
        assert action.type == ActionType.CHECK


class TestReverseImpliedOdds:
    def test_one_pair_wet_board_penalty(self):
        # Hero has top pair (KQ on K-9-8 with two spades and connected) — vulnerable
        view = make_view("Ks Kd", "Qs Js Tc", pot=100, to_call=40,
                         your_stack=200, opp_stack=200, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        # Reverse implied: required_eq goes UP for vulnerable made hand on wet board
        # Note: KsKd on QsJsTc is actually overpair... let me reconsider
        # With KK on QJT, hero has overpair which classifies as made_strong
        # Use a true one-pair scenario: KQ on KQT? wait that's two-pair.
        # Just use AK on K-board with wet texture: A♠K♣ on K♠Q♠J♥ (top pair)
        view2 = make_view("Ac Kc", "Ks Qs Jh", pot=100, to_call=40,
                          your_stack=200, opp_stack=200, street=Street.FLOP)
        adjusted = bot._apply_implied_odds(view2, required_eq=0.30, eq=0.65)
        # Top pair on wet (two spades + straight texture) board → penalty applied
        assert adjusted > 0.30

    def test_strong_hand_no_penalty(self):
        # Set on a wet board — no penalty (strong made hand, not vulnerable)
        view = make_view("Ks Kd", "Kh Qs Js", pot=100, to_call=40,
                         your_stack=200, opp_stack=200, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        # Trips (set of K's) is made_strong, not made_pair → no penalty
        adjusted = bot._apply_implied_odds(view, required_eq=0.30, eq=0.85)
        assert adjusted == 0.30


class TestBlockerBluffs:
    def test_nut_flush_blocker_helps_river_bluff(self):
        # Hero has A♠ on a 4-spade board (would-be flush board for villain).
        # Even with low equity, the blocker makes a bluff more credible.
        from poker.cards import Card, Rank, Suit
        view = make_view("As 7d", "Ks 8s 4s 2s 9c", pot=50, to_call=0,
                         your_stack=200, opp_stack=200, street=Street.RIVER)
        bot = make_tag(rng=random.Random(0))
        assert bot._has_blocker_to_villain_value(view) is True

    def test_no_blocker_when_no_flush_card(self):
        # Hero has no spade on a 3-spade board
        view = make_view("Ah 7d", "Ks 8s 4s 2c 9c", pot=50, to_call=0,
                         your_stack=200, opp_stack=200, street=Street.RIVER)
        bot = make_tag(rng=random.Random(0))
        assert bot._has_blocker_to_villain_value(view) is False

    def test_no_blocker_on_dry_board(self):
        # Rainbow board, no flush threat
        view = make_view("As 7d", "Kh 8d 4c", pot=50, to_call=0,
                         your_stack=200, opp_stack=200, street=Street.FLOP)
        bot = make_tag(rng=random.Random(0))
        assert bot._has_blocker_to_villain_value(view) is False
