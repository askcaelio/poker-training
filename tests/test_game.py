"""Tests for the game state machine.

Coverage areas:
  - Position assignment (2-6 handed)
  - Blind posting
  - Hole-card dealing
  - Legal-action enumeration on each street
  - Betting round completion (everyone calls / check-around)
  - Fold-around (single-player wins without showdown)
  - All-in side pots
  - Heads-up edge cases (BTN==SB)
  - HandView is correctly scoped per player
"""

import random

import pytest

from poker.cards import Card, Deck, Rank, Suit, parse_cards
from poker.game import (
    Action, ActionType, HandState, Player, Position, Street,
    apply_action, assign_positions, award_pots, legal_actions, start_hand,
)


def make_players(n: int, stack: float = 100.0) -> list[Player]:
    return [Player(id=f"p{i}", stack=stack) for i in range(n)]


def seeded_deck(seed: int = 42) -> Deck:
    return Deck(rng=random.Random(seed))


# ─── Position assignment ─────────────────────────────────────────────────────

class TestPositions:
    def test_6max_assignment(self):
        players = make_players(6)
        assign_positions(players, button_index=0)
        # button=0 → SB=1, BB=2, UTG=3, MP=4, CO=5, BTN=0
        assert players[0].position == Position.BTN
        assert players[1].position == Position.SB
        assert players[2].position == Position.BB
        assert players[3].position == Position.UTG
        assert players[4].position == Position.MP
        assert players[5].position == Position.CO

    def test_heads_up_button_is_sb(self):
        players = make_players(2)
        assign_positions(players, button_index=0)
        assert players[0].position == Position.SB    # in 2-handed, BTN==SB
        assert players[1].position == Position.BB

    def test_unsupported_count_raises(self):
        players = make_players(7)
        with pytest.raises(ValueError):
            assign_positions(players, button_index=0)


# ─── Hand setup ───────────────────────────────────────────────────────────────

class TestStartHand:
    def test_blinds_posted(self):
        players = make_players(6, stack=100.0)
        s = start_hand(players, button_index=0, deck=seeded_deck(),
                       blinds=(0.5, 1.0))
        sb = next(p for p in players if p.position == Position.SB)
        bb = next(p for p in players if p.position == Position.BB)
        assert sb.stack == 99.5
        assert bb.stack == 99.0
        assert s.pot == 1.5
        assert s.current_bet == 1.0

    def test_hole_cards_dealt(self):
        players = make_players(6)
        start_hand(players, button_index=0, deck=seeded_deck(), blinds=(0.5, 1.0))
        for p in players:
            assert p.hole_cards is not None
            assert len(p.hole_cards) == 2

    def test_first_actor_is_utg_in_6max(self):
        players = make_players(6)
        s = start_hand(players, button_index=0, deck=seeded_deck(), blinds=(0.5, 1.0))
        assert s.actor().position == Position.UTG

    def test_first_actor_is_sb_in_heads_up(self):
        players = make_players(2)
        s = start_hand(players, button_index=0, deck=seeded_deck(), blinds=(0.5, 1.0))
        # In 2-handed, SB acts first preflop (and is the button)
        assert s.actor().position == Position.SB


# ─── Legal actions ────────────────────────────────────────────────────────────

class TestLegalActions:
    def test_facing_bet_can_fold_call_raise(self):
        players = make_players(6)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        legal = legal_actions(s)
        assert ActionType.FOLD in legal
        assert ActionType.CALL in legal
        assert ActionType.RAISE in legal
        assert ActionType.CHECK not in legal

    def test_check_when_bet_matched(self):
        # After preflop ends and everyone reaches flop, first actor can check
        players = make_players(2, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        # SB calls, BB checks → flop
        apply_action(s, Action(ActionType.CALL, 0.5))
        apply_action(s, Action(ActionType.CHECK))
        assert s.street == Street.FLOP
        legal = legal_actions(s)
        assert ActionType.CHECK in legal
        assert ActionType.BET in legal
        assert ActionType.CALL not in legal


# ─── Betting flow ─────────────────────────────────────────────────────────────

class TestBettingFlow:
    def test_fold_around_wins_without_showdown(self):
        # 6-max: everyone folds to BB → BB wins SB+BB
        players = make_players(6, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        # UTG, MP, CO, BTN, SB all fold → BB wins
        for _ in range(5):
            apply_action(s, Action(ActionType.FOLD))
        assert s.is_complete
        awards = award_pots(s)
        bb = next(p for p in players if p.position == Position.BB)
        assert len(awards) == 1
        assert awards[0].winner_ids == (bb.id,)
        # BB started 100, posted 1, won 1.5 → 100.5
        assert bb.stack == 100.5

    def test_heads_up_call_check_advances_to_flop(self):
        players = make_players(2, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        # SB completes (call to 1.0), BB checks
        apply_action(s, Action(ActionType.CALL, 0.5))
        assert s.street == Street.PREFLOP
        apply_action(s, Action(ActionType.CHECK))
        assert s.street == Street.FLOP
        assert len(s.board) == 3
        # First postflop actor is BB (left of button in 2-handed)
        assert s.actor().position == Position.BB

    def test_full_hand_check_through_to_showdown(self):
        # 2-handed: SB limps (calls 0.5 more), BB checks. Then check down to showdown.
        players = make_players(2, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        # In 2-handed, SB acts first preflop (BTN==SB)
        apply_action(s, Action(ActionType.CALL, 0.5))   # SB completes
        apply_action(s, Action(ActionType.CHECK))       # BB checks → flop
        # Postflop: BB acts first (left of button)
        apply_action(s, Action(ActionType.CHECK))       # BB checks flop
        apply_action(s, Action(ActionType.CHECK))       # SB checks flop → turn
        apply_action(s, Action(ActionType.CHECK))       # BB checks turn
        apply_action(s, Action(ActionType.CHECK))       # SB checks turn → river
        apply_action(s, Action(ActionType.CHECK))       # BB checks river
        apply_action(s, Action(ActionType.CHECK))       # SB checks → showdown
        assert s.street == Street.SHOWDOWN
        assert len(s.board) == 5
        awards = award_pots(s)
        # Pot = 1.0 each = 2.0 total
        assert sum(a.amount for a in awards) == 2.0


class TestRaisesAndAllIn:
    def test_simple_raise_increases_current_bet(self):
        players = make_players(6, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        # UTG raises to 3
        apply_action(s, Action(ActionType.RAISE, 3.0))
        assert s.current_bet == 3.0
        # Last raise size = 3.0 - 1.0 = 2.0; min next raise is +2.0 over 3.0 = 5.0

    def test_all_in_short_stack_creates_side_pot(self):
        # Heads-up: short stack goes all-in for less than full call
        # Stacks: p0=10, p1=100. SB=p0 (button in 2-handed).
        players = [Player(id="short", stack=10.0), Player(id="big", stack=100.0)]
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        # Short SB shoves all-in (10 total = +9.5 raise)
        apply_action(s, Action(ActionType.RAISE, 9.5))
        # Big BB calls (needs to add 9 to match 10)
        apply_action(s, Action(ActionType.CALL, 9.0))
        # All-in → run out the board to showdown
        assert s.street == Street.SHOWDOWN or s.is_complete
        awards = award_pots(s)
        # Total pot = 20 (10 each). One winner takes it (or split on tie).
        total_awarded = sum(a.amount for a in awards)
        assert total_awarded == 20.0

    def test_three_way_side_pot(self):
        # Three players, three different stack sizes, all in.
        # In 3-handed: button=0 → SB=1, BB=2, BTN=0. BTN acts first preflop.
        players = [
            Player(id="short", stack=20.0),    # BTN
            Player(id="mid", stack=50.0),       # SB
            Player(id="big", stack=200.0),      # BB
        ]
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        # Short shoves: raise puts in his entire 20 stack
        apply_action(s, Action(ActionType.RAISE, 20.0))
        # Mid (SB) wants to go all-in: raise puts in 49.5 more (total 50)
        apply_action(s, Action(ActionType.RAISE, 49.5))
        # Big (BB) calls the 50: needs to add 49 more to match
        apply_action(s, Action(ActionType.CALL, 49.0))
        # Total invested: short=20, mid=50, big=50, pot=120
        assert s.pot == 120.0
        # Both short and mid are all-in. Big still has chips but no one to bet against
        # → engine should run out the streets to showdown.
        assert s.street == Street.SHOWDOWN
        awards = award_pots(s)
        # Main pot: 20×3=60 (all 3 eligible)
        # Side pot: (50-20)×2=60 (mid and big eligible)
        amounts = sorted(a.amount for a in awards)
        assert amounts == [60.0, 60.0]


class TestHandView:
    def test_view_hides_others_hole_cards(self):
        players = make_players(2, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        view = s.view_for("p0")
        # Has own hole cards
        assert view.your_hole is not None
        # Has only public info about others
        assert all(not hasattr(o, "hole_cards") for o in view.others)

    def test_view_to_call_correct_facing_blind(self):
        players = make_players(6, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        # UTG faces 1.0 to call (BB)
        view = s.view_for(s.actor().id)
        assert view.to_call == 1.0
        assert view.current_bet == 1.0

    def test_view_pot_reflects_state(self):
        players = make_players(6, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        view = s.view_for(s.actor().id)
        assert view.pot == 1.5    # SB + BB

    def test_view_unknown_player_raises(self):
        players = make_players(2, stack=100.0)
        s = start_hand(players, 0, seeded_deck(), (0.5, 1.0))
        with pytest.raises(ValueError):
            s.view_for("nobody")
